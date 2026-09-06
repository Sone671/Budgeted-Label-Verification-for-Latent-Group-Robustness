"""End-to-end image verification runner with public/private manifest isolation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from robust_verify.analysis import query_composition
from robust_verify.config import ensure_output_layout, output_layout
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.oof import fit_oof_probabilities_with_folds
from robust_verify.scoring import ADAPTIVE_METHODS, build_adaptive_ranking, build_rankings
from robust_verify.utils import budget_to_count, class_balanced_accuracy, save_json, seed_everything


_IMAGE_ARRAY_CACHE: dict[tuple[str, ...], np.ndarray] = {}


def _read_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        labels: np.ndarray,
        root: str | Path,
        transform,
        *,
        cache_images: bool = False,
    ):
        self.paths = [Path(root) / str(path) for path in frame["image_relpath"].tolist()]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ids = frame["sample_id"].to_numpy(dtype=np.int64)
        self.transform = transform
        self.images: np.ndarray | None = None
        if cache_images:
            # CIFAR-N stores tiny PNGs; caching decoded uint8 arrays avoids
            # reopening the same 50k files for every epoch and retrain.
            cache_key = tuple(str(path) for path in self.paths)
            cached = _IMAGE_ARRAY_CACHE.get(cache_key)
            if cached is None:
                cached = np.stack(
                    [_read_rgb_uint8(path) for path in self.paths]
                )
                _IMAGE_ARRAY_CACHE[cache_key] = cached
            self.images = cached
        if len(self.paths) != len(self.labels):
            raise ValueError("Image manifest and labels must have equal length.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        if self.images is None:
            with Image.open(self.paths[index]) as image:
                tensor = self.transform(image.convert("RGB"))
        else:
            tensor = self.transform(Image.fromarray(self.images[index], mode="RGB"))
        return tensor, int(self.labels[index]), int(self.ids[index])


class TinyConvNet(nn.Module):
    """Small CPU-friendly network used only by the local smoke test."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def build_image_classifier(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "tiny_cnn":
        return TinyConvNet(num_classes)
    raise ValueError(f"Unsupported end-to-end backbone: {backbone!r}")


def _transform(config: dict[str, Any], *, train: bool):
    size = int(config.get("input_size", 224))
    augment = bool(config.get("augmentation", True)) and train
    transforms_list: list[Any] = [transforms.Resize((size, size))]
    if augment:
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return transforms.Compose(transforms_list)


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(
    frame: pd.DataFrame,
    labels: np.ndarray,
    root: str | Path,
    config: dict[str, Any],
    *,
    train: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed + (17 if train else 23))
    return DataLoader(
        ManifestImageDataset(
            frame,
            labels,
            root,
            _transform(config, train=train),
            cache_images=bool(config.get("cache_images", False)),
        ),
        batch_size=int(config.get("batch_size", 32)),
        shuffle=train,
        generator=generator,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", _device(config).type == "cuda")),
    )


@torch.inference_mode()
def predict_image_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for images, batch_labels, batch_ids in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
        ids.append(batch_ids.numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(ids).astype(np.int64),
    )


def _penultimate_module(model: nn.Module) -> nn.Module:
    """Return the linear classifier whose input is the legal probe embedding."""
    for attribute in ("fc", "classifier"):
        module = getattr(model, attribute, None)
        if isinstance(module, nn.Linear):
            return module
    raise ValueError(
        "TracIn-CP end-to-end acquisition requires a model with an nn.Linear "
        "`fc` or `classifier` head."
    )


@torch.inference_mode()
def predict_image_probabilities_and_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict probabilities and capture penultimate features in loader order.

    The hook is registered on the classification head rather than a named
    backbone block, so it works for both ResNet models and the TinyConvNet used
    by the end-to-end smoke test.  These features, noisy train labels, and
    public validation labels form the legal context for TracIn-CP (val).
    """
    model.eval()
    probabilities: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ids: list[np.ndarray] = []

    def _capture_features(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Could not capture penultimate tensor for TracIn-CP.")
        features.append(inputs[0].detach().flatten(start_dim=1).cpu().numpy())

    handle = _penultimate_module(model).register_forward_pre_hook(_capture_features)
    try:
        for images, batch_labels, batch_ids in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
            ids.append(batch_ids.numpy())
    finally:
        handle.remove()

    if not features:
        raise RuntimeError("TracIn-CP feature extraction produced no batches.")
    return (
        np.concatenate(probabilities),
        np.concatenate(features).astype(np.float32, copy=False),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(ids).astype(np.int64),
    )


@dataclass
class ImageProbeResult:
    best_state: dict[str, torch.Tensor]
    initial_state: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_metric: float
    train_probabilities: np.ndarray
    correctness_history: np.ndarray
    train_features: np.ndarray | None
    val_features: np.ndarray | None
    val_probabilities: np.ndarray | None


def _metric(probabilities: np.ndarray, labels: np.ndarray, selection_metric: str) -> float:
    predictions = probabilities.argmax(axis=1)
    if selection_metric == "accuracy":
        return float((predictions == labels).mean())
    if selection_metric == "balanced_accuracy":
        return class_balanced_accuracy(labels, predictions)
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def train_image_classifier(
    *,
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    val_frame: pd.DataFrame,
    val_labels: np.ndarray,
    data_root: str | Path,
    config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor] | None = None,
    record_dynamics: bool,
    record_features: bool = False,
) -> ImageProbeResult:
    """Train all image-model parameters and retain only legal probe dynamics."""
    seed_everything(seed, deterministic=bool(config.get("deterministic", False)))
    device = _device(config)
    num_classes = int(max(np.max(train_labels), np.max(val_labels)) + 1)
    backbone = str(config.get("backbone", "resnet18"))
    model = build_image_classifier(backbone, num_classes, bool(config.get("pretrained", True)))
    if initial_state is None:
        initial_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(copy.deepcopy(initial_state))
    model.to(device)

    train_loader = _loader(train_frame, train_labels, data_root, config, train=True, seed=seed)
    train_eval_loader = _loader(train_frame, train_labels, data_root, config, train=False, seed=seed)
    val_loader = _loader(val_frame, val_labels, data_root, config, train=False, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config.get("amp", True)))
    epochs = int(config.get("epochs", 10))
    patience = int(config.get("patience", epochs))
    metric_name = str(config.get("selection_metric", "balanced_accuracy"))
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    best_epoch = 0
    remaining_patience = patience
    history: list[np.ndarray] = []

    for epoch in range(epochs):
        model.train()
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and bool(config.get("amp", True))):
                logits = model(images.to(device, non_blocking=True))
                loss = criterion(logits, labels.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if record_dynamics:
            train_probs, train_eval_labels, _ = predict_image_probabilities(
                model, train_eval_loader, device
            )
            history.append((train_probs.argmax(axis=1) == train_eval_labels).astype(np.int8))
        val_probs, val_eval_labels, _ = predict_image_probabilities(model, val_loader, device)
        metric = _metric(val_probs, val_eval_labels, metric_name)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    if record_features:
        train_probs, train_features, _, _ = predict_image_probabilities_and_features(
            model, train_eval_loader, device
        )
        val_probs, val_features, _, _ = predict_image_probabilities_and_features(
            model, val_loader, device
        )
    else:
        train_probs, _, _ = predict_image_probabilities(model, train_eval_loader, device)
        train_features = None
        val_features = None
        val_probs = None
    if record_dynamics:
        correctness = np.stack(history, axis=1)
    else:
        correctness = np.empty((len(train_frame), 0), dtype=np.int8)
    return ImageProbeResult(
        best_state=best_state,
        initial_state=initial_state,
        best_epoch=best_epoch,
        best_validation_metric=float(best_metric),
        train_probabilities=train_probs,
        correctness_history=correctness,
        train_features=train_features,
        val_features=val_features,
        val_probabilities=val_probs,
    )


def _predict_from_state(
    state: dict[str, torch.Tensor],
    frame: pd.DataFrame,
    labels: np.ndarray,
    data_root: str | Path,
    config: dict[str, Any],
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = build_image_classifier(
        str(config.get("backbone", "resnet18")), num_classes, pretrained=False
    )
    model.load_state_dict(state)
    device = _device(config)
    model.to(device)
    loader = _loader(frame, labels, data_root, config, train=False, seed=seed)
    probabilities, _, ids = predict_image_probabilities(model, loader, device)
    return probabilities.argmax(axis=1).astype(np.int64), ids


def _repair_labels(public: pd.DataFrame, private_path: Path, positions: np.ndarray) -> np.ndarray:
    labels = public["noisy_label"].to_numpy(dtype=np.int64).copy()
    sample_ids = public.iloc[positions]["sample_id"].to_numpy(dtype=np.int64)
    labels[positions] = VerificationOracle(private_path).verify(sample_ids)
    return labels


def _cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy a state dict to CPU so replay checkpoints are portable."""
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _load_or_fit_oof_probabilities(
    *,
    e2e_root: Path,
    prefix: str,
    sample_ids: np.ndarray,
    train_features: np.ndarray,
    noisy_labels: np.ndarray,
    seed: int,
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load or create replayable OOF probabilities for Cleanlab.

    OOF Cleanlab is intentionally a linear-probe baseline: the feature
    representation comes from the legal baseline probe, while each example's
    probability is generated by a probe whose fit fold excludes that example.
    The artifact contains only sample IDs, probabilities, fold IDs, and scalar
    run metadata; clean labels, groups, and test outcomes never enter it.
    """
    folds = int(options.get("oof_folds", 5))
    artifact = e2e_root / "oof" / f"{prefix}.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)

    def _validate(
        probabilities: np.ndarray, fold_ids: np.ndarray, stored_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        expected_ids = np.asarray(sample_ids, dtype=np.int64)
        stored_ids = np.asarray(stored_ids, dtype=np.int64)
        if stored_ids.ndim != 1 or not np.array_equal(stored_ids, expected_ids):
            raise ValueError(f"OOF artifact sample IDs do not match {prefix}")
        probabilities = np.asarray(probabilities, dtype=np.float32)
        fold_ids = np.asarray(fold_ids, dtype=np.int64)
        if probabilities.shape != (len(expected_ids), 2):
            raise ValueError(f"OOF artifact probability shape mismatch: {artifact}")
        if fold_ids.shape != (len(expected_ids),):
            raise ValueError(f"OOF artifact fold shape mismatch: {artifact}")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(f"OOF artifact contains non-finite probabilities: {artifact}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"OOF artifact probabilities do not sum to one: {artifact}")
        if folds < 2 or set(np.unique(fold_ids)) != set(range(folds)):
            raise ValueError(f"OOF artifact has incomplete fold IDs: {artifact}")
        if np.any(np.bincount(fold_ids, minlength=folds) == 0):
            raise ValueError(f"OOF artifact has an empty fold: {artifact}")
        return probabilities, fold_ids

    if artifact.is_file():
        with np.load(artifact, allow_pickle=False) as payload:
            required = {"sample_ids", "probabilities", "fold_ids", "seed", "folds"}
            if set(payload.files) != required:
                raise ValueError(f"Unexpected OOF artifact keys: {artifact}")
            if int(np.asarray(payload["seed"]).item()) != int(seed):
                raise ValueError(f"OOF artifact seed mismatch: {artifact}")
            if int(np.asarray(payload["folds"]).item()) != folds:
                raise ValueError(f"OOF artifact fold count mismatch: {artifact}")
            probabilities, fold_ids = _validate(
                payload["probabilities"], payload["fold_ids"], payload["sample_ids"]
            )
            return probabilities, fold_ids, artifact

    probabilities, fold_ids = fit_oof_probabilities_with_folds(
        train_features,
        noisy_labels,
        seed=seed,
        folds=folds,
        options=options,
    )
    probabilities, fold_ids = _validate(probabilities, fold_ids, sample_ids)
    # Write through a sibling temporary file so an interrupted run cannot
    # leave a truncated ``.npz`` that a later resume mistakes for a valid
    # artifact.  Passing an open handle prevents NumPy from appending a
    # second ``.npz`` suffix to the temporary name.
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                sample_ids=np.asarray(sample_ids, dtype=np.int64),
                probabilities=probabilities,
                fold_ids=fold_ids,
                seed=np.asarray(seed, dtype=np.int64),
                folds=np.asarray(folds, dtype=np.int64),
            )
        temporary.replace(artifact)
    finally:
        temporary.unlink(missing_ok=True)
    return probabilities, fold_ids, artifact


def run_end_to_end(config: dict[str, Any]) -> pd.DataFrame:
    """Run a one-shot, full-retraining label-verification matrix on images."""
    layout = ensure_output_layout(config["project"]["output_dir"])
    source_root = Path(config["project"].get("input_dir", config["project"]["output_dir"]))
    source_layout = output_layout(source_root)
    e2e_root = layout["root"] / "end_to_end"
    query_root = e2e_root / "queries"
    checkpoint_root = e2e_root / "checkpoints"
    prediction_root = e2e_root / "predictions"
    query_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)

    val_public = pd.read_csv(source_layout["manifests"] / "val_public.csv").sort_values("sample_id")
    test_public = pd.read_csv(source_layout["manifests"] / "test_public.csv").sort_values("sample_id")
    if "label" not in val_public or "clean_label" in val_public:
        raise ValueError("Validation public manifest must contain label and not clean_label.")
    if "clean_label" in test_public:
        raise ValueError("Test public manifest must not contain clean_label.")
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    test_ids = test_public["sample_id"].to_numpy(dtype=np.int64)
    test_evaluator = PrivateEvaluator(source_layout["manifests"] / "test_private.csv")
    e2e_config = {**config["training"], **config.get("end_to_end", {})}
    methods = list(config["experiment"]["methods"])
    prohibited = sorted(
        method
        for method in methods
        if method.startswith("oracle_") or method.endswith("_oracle")
    )
    if prohibited:
        raise ValueError(
            "End-to-end practical runs cannot use oracle methods: " + ", ".join(prohibited)
        )
    tracin_methods = {"tracin_val_uncertainty", "tracin_noise_weighted"}
    # These practical methods share the same legal probe context: public noisy
    # train labels, public clean-validation labels, final probe probabilities,
    # and penultimate features.  In particular, RepairValue must not fall back
    # to a frozen Stage-1 cache when used in an end-to-end run.
    feature_context_methods = tracin_methods | {
        "expected_repair_value",
        "expected_repair_value_multiclass",
        "expected_repair_value_adaptive_tau",
        "expected_repair_value_capture_tau",
        "noise_gated_repair_value",
        "reliability_gated_repair_value",
        "auto_d3m_query",
        "activeclean_expected_change",
        "active_label_cleaning",
        "oof_cleanlab",
    }
    needs_feature_context = bool(feature_context_methods.intersection(methods))
    budgets = [float(value) for value in config["experiment"]["budgets"]]
    results: list[dict[str, Any]] = []

    for noise_setting in config["noise"]["settings"]:
        noise_name = str(noise_setting["name"])
        for seed_value in config["experiment"]["seeds"]:
            seed = int(seed_value)
            prefix = f"{noise_name}_seed{seed}"
            public = pd.read_csv(source_layout["manifests"] / f"{prefix}_train_public.csv").sort_values("sample_id")
            private_path = source_layout["manifests"] / f"{prefix}_train_private.csv"
            if "clean_label" in public:
                raise RuntimeError("Clean labels leaked into an end-to-end public manifest.")
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
            num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
            probe = train_image_classifier(
                train_frame=public,
                train_labels=noisy_labels,
                val_frame=val_public,
                val_labels=val_labels,
                data_root=config["data"]["root"],
                config=e2e_config,
                seed=seed,
                record_dynamics=True,
                record_features=needs_feature_context,
            )
            baseline_predictions, predicted_test_ids = _predict_from_state(
                probe.best_state, test_public, np.zeros(len(test_public), dtype=np.int64),
                config["data"]["root"], e2e_config, num_classes, seed,
            )
            if not np.array_equal(predicted_test_ids, test_ids):
                raise RuntimeError("Baseline prediction IDs do not match the public test manifest.")
            baseline_metrics = test_evaluator.evaluate(predicted_test_ids, baseline_predictions)
            baseline_prediction_path = prediction_root / prefix / "no_correction.npz"
            baseline_prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                baseline_prediction_path,
                sample_ids=predicted_test_ids,
                predictions=baseline_predictions,
            )
            baseline_checkpoint_path = checkpoint_root / prefix / "no_correction.pt"
            baseline_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state": _cpu_state_dict(probe.best_state),
                    "initial_state": _cpu_state_dict(probe.initial_state),
                    "seed": seed,
                    "noise_name": noise_name,
                    "best_epoch": probe.best_epoch,
                },
                baseline_checkpoint_path,
            )
            legal_frame = public[["sample_id"]].copy()
            selection_context: dict[str, Any] | None = None
            oof_artifact_path: Path | None = None
            if needs_feature_context:
                if (
                    probe.train_features is None
                    or probe.val_features is None
                    or probe.val_probabilities is None
                ):
                    raise RuntimeError(
                        "TracIn-CP context was requested but probe features are unavailable."
                    )
                selection_context = {
                    "train_features": probe.train_features,
                    "val_features": probe.val_features,
                    "val_probs": probe.val_probabilities,
                    "val_labels": val_labels,
                    "baseline_options": config.get("experiment", {}).get(
                        "baseline_options", {}
                    ),
                    "selection_metadata": {},
                }
                if "oof_cleanlab" in methods:
                    oof_probabilities, oof_fold_ids, oof_artifact_path = (
                        _load_or_fit_oof_probabilities(
                            e2e_root=e2e_root,
                            prefix=prefix,
                            sample_ids=public["sample_id"].to_numpy(dtype=np.int64),
                            train_features=probe.train_features,
                            noisy_labels=noisy_labels,
                            seed=seed,
                            options=selection_context["baseline_options"],
                        )
                    )
                    selection_context["oof_train_probs"] = oof_probabilities
                    selection_context["oof_fold_ids"] = oof_fold_ids
            rankings, scores = build_rankings(
                methods=methods,
                probabilities=probe.train_probabilities,
                labels=noisy_labels,
                correctness_history=probe.correctness_history,
                private_frame=legal_frame,
                seed=seed,
                context=selection_context,
            )
            adaptive = sorted(set(methods).intersection(ADAPTIVE_METHODS))
            method_rankings: dict[tuple[str, float], np.ndarray] = {
                (method, budget): ranking for method, ranking in rankings.items() for budget in budgets
            }
            for method in adaptive:
                for budget in budgets:
                    count = budget_to_count(budget, len(public))
                    method_rankings[(method, budget)] = build_adaptive_ranking(
                        method,
                        scores["noise_score"],
                        probe.train_probabilities,
                        legal_frame,
                        count,
                        len(public),
                        losses=scores["loss"],
                        labels=noisy_labels,
                        cluster_path=config.get("experiment", {}).get("cluster_path"),
                        repair_value_scores=scores.get(method),
                        options=config.get("experiment", {}).get("baseline_options", {}),
                    )

            for (method, budget), ranking in method_rankings.items():
                budget_count = budget_to_count(budget, len(public))
                positions = ranking[:budget_count]
                queried_ids = public.iloc[positions]["sample_id"].to_numpy(dtype=np.int64)
                query_path = query_root / prefix / f"{method}_budget_{budget:.4f}.npz"
                query_path.parent.mkdir(parents=True, exist_ok=True)
                # Persist the exact query prefix before the verification oracle can reveal a label.
                np.savez_compressed(query_path, sample_ids=queried_ids)

                repaired_labels = _repair_labels(public, private_path, positions)
                retrained = train_image_classifier(
                    train_frame=public,
                    train_labels=repaired_labels,
                    val_frame=val_public,
                    val_labels=val_labels,
                    data_root=config["data"]["root"],
                    config=e2e_config,
                    seed=seed,
                    initial_state=copy.deepcopy(probe.initial_state),
                    record_dynamics=False,
                )
                predictions, prediction_ids = _predict_from_state(
                    retrained.best_state, test_public, np.zeros(len(test_public), dtype=np.int64),
                    config["data"]["root"], e2e_config, num_classes, seed,
                )
                if not np.array_equal(prediction_ids, test_ids):
                    raise RuntimeError(
                        f"Prediction IDs for {method} do not match the public test manifest."
                    )
                metrics = test_evaluator.evaluate(prediction_ids, predictions)
                prediction_path = (
                    prediction_root / prefix / f"{method}_budget_{budget:.4f}.npz"
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prediction_path,
                    sample_ids=prediction_ids,
                    predictions=predictions,
                )
                checkpoint_path = checkpoint_root / prefix / f"{method}_budget_{budget:.4f}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state": _cpu_state_dict(retrained.best_state),
                        "query_ids": queried_ids,
                        "seed": seed,
                        "noise_name": noise_name,
                        "method": method,
                        "budget_fraction": budget,
                        "best_epoch": retrained.best_epoch,
                    },
                    checkpoint_path,
                )
                private = pd.read_csv(private_path).sort_values("sample_id")
                composition = query_composition(positions, private)
                evaluation_scope = (
                    "worst_class_accuracy"
                    if config["data"].get("dataset") == "cifar10n"
                    else "worst_group_accuracy"
                )
                row = {
                    "dataset": str(config["data"].get("dataset", "unknown")),
                    "noise_name": noise_name,
                    "noise_kind": noise_setting["kind"],
                    "seed": seed,
                    "method": method,
                    "budget_fraction": budget,
                    "budget_count": budget_count,
                    "wga_semantics": evaluation_scope,
                    **baseline_metrics,
                    **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                    **composition,
                    **metrics,
                    "delta_average_accuracy": metrics["average_accuracy"] - baseline_metrics["average_accuracy"],
                    "delta_wga": metrics["wga"] - baseline_metrics["wga"],
                    "probe_validation_metric": probe.best_validation_metric,
                    "retrain_validation_metric": retrained.best_validation_metric,
                    "probe_best_epoch": probe.best_epoch,
                    "retrain_best_epoch": retrained.best_epoch,
                    "query_artifact": str(query_path.relative_to(layout["root"])),
                    "checkpoint": str(checkpoint_path.relative_to(layout["root"])),
                    "prediction_artifact": str(prediction_path.relative_to(layout["root"])),
                    "baseline_checkpoint": str(
                        baseline_checkpoint_path.relative_to(layout["root"])
                    ),
                    "baseline_prediction_artifact": str(
                        baseline_prediction_path.relative_to(layout["root"])
                    ),
                    "oof_probability_artifact": (
                        str(oof_artifact_path.relative_to(layout["root"]))
                        if oof_artifact_path is not None
                        else ""
                    ),
                }
                selection_metadata = selection_context.get("selection_metadata", {}) if selection_context else {}
                method_metadata = selection_metadata.get(method)
                if method_metadata and "selected_tau" in method_metadata:
                    row["selected_tau"] = float(method_metadata["selected_tau"])
                    row["tau_selection_folds"] = int(method_metadata.get("num_folds", 0))
                    row["tau_min_tail_count"] = int(method_metadata.get("min_tail_count", 0))
                if method_metadata and "gate_alpha" in method_metadata:
                    row["gate_alpha"] = float(method_metadata["gate_alpha"])
                    row["gate_reliability"] = float(
                        method_metadata.get("gate_reliability", 0.0)
                    )
                    row["gate_stability_alpha"] = float(
                        method_metadata.get("gate_stability_alpha", 0.0)
                    )
                    row["gate_retention_alpha"] = float(
                        method_metadata.get("gate_retention_alpha", 0.0)
                    )
                    row["gate_noise_retention"] = float(
                        method_metadata.get("gate_noise_retention", 0.0)
                    )
                    row["gate_reference_budget_count"] = int(
                        method_metadata.get("gate_reference_budget_count", 0)
                    )
                    row["gate_fallback"] = bool(
                        method_metadata.get("gate_fallback", False)
                    )
                    row["gate_num_folds"] = int(method_metadata.get("gate_num_folds", 0))
                if evaluation_scope == "worst_class_accuracy":
                    row["delta_worst_class_accuracy"] = row["delta_wga"]
                results.append(row)
                pd.DataFrame(results).to_csv(e2e_root / "results.partial.csv", index=False)

    frame = pd.DataFrame(results)
    frame.to_csv(e2e_root / "results.csv", index=False)
    # Emit a seed-paired comparison table whenever Loss is part of the run.
    # This keeps external-validity and strong-baseline comparisons auditable
    # without treating datasets, budgets, or seeds as interchangeable samples.
    if "loss" in frame["method"].astype(str).unique():
        from robust_verify.e2e_analysis import write_end_to_end_summary

        analysis_config = config.get("analysis", {})
        write_end_to_end_summary(
            frame,
            e2e_root,
            reference_method=str(analysis_config.get("reference_method", "loss")),
            methods=analysis_config.get("methods"),
            bootstrap_replicates=int(analysis_config.get("bootstrap_replicates", 10_000)),
            rng_seed=int(analysis_config.get("rng_seed", 20260827)),
        )
    partial = e2e_root / "results.partial.csv"
    if partial.exists():
        partial.unlink()
    save_json(config, e2e_root / "resolved_config.json")
    return frame
