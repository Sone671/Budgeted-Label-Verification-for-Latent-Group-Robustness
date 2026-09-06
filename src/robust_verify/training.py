from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from robust_verify.utils import class_balanced_accuracy, resolve_device, seed_everything


@dataclass
class ProbeResult:
    best_state: dict[str, torch.Tensor]
    initial_state: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_metric: float
    train_probabilities: np.ndarray
    loss_history: np.ndarray | None
    probability_history: np.ndarray | None
    prediction_history: np.ndarray | None
    margin_history: np.ndarray | None
    correctness_history: np.ndarray | None
    val_probability_history: np.ndarray | None = None


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def make_initial_state(
    input_dim: int,
    num_classes: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    model = LinearHead(input_dim, num_classes)
    return copy.deepcopy(model.state_dict())


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    features: np.ndarray | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if isinstance(features, torch.Tensor) and features.device.type == device.type:
        x_gpu = features
    else:
        x_gpu = torch.from_numpy(np.asarray(features)).float().to(device)
    n = x_gpu.shape[0]
    probabilities = []
    model.eval()
    for start in range(0, n, batch_size):
        batch = x_gpu[start : start + batch_size]
        logits = model(batch)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _validation_metric(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> float:
    predictions = probabilities.argmax(axis=1)
    if metric == "accuracy":
        return float((predictions == labels).mean())
    if metric == "balanced_accuracy":
        return class_balanced_accuracy(labels, predictions)
    raise ValueError(f"Unsupported selection metric: {metric}")


def train_linear_head(
    train_features: np.ndarray | torch.Tensor,
    train_labels: np.ndarray,
    val_features: np.ndarray | torch.Tensor,
    val_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor] | None = None,
    record_dynamics: bool = False,
    sample_weights: np.ndarray | None = None,
) -> ProbeResult:
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    input_dim = int(train_features.shape[1] if isinstance(train_features, np.ndarray) else train_features.shape[1])
    num_classes = int(max(train_labels.max(), val_labels.max()) + 1)

    if initial_state is None:
        initial_state = make_initial_state(input_dim, num_classes, seed)

    model = LinearHead(input_dim, num_classes)
    model.load_state_dict(copy.deepcopy(initial_state))
    model.to(device)

    if isinstance(train_features, torch.Tensor) and train_features.device.type == device.type:
        x_gpu = train_features
    else:
        x_gpu = torch.from_numpy(np.asarray(train_features)).float().to(device)

    train_label_array = np.asarray(train_labels)
    # Pandas/NPZ-backed views can be read-only.  ``from_numpy`` warns that such
    # tensors have undefined behaviour if written, even though the labels are
    # only read here; make a copy only for that uncommon case.
    if not train_label_array.flags.writeable:
        train_label_array = train_label_array.copy()
    y_tensor = torch.from_numpy(train_label_array).long().to(x_gpu.device)
    if sample_weights is None:
        w_tensor = torch.ones(len(train_labels), dtype=torch.float32, device=x_gpu.device)
    else:
        w_tensor = torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)).to(x_gpu.device)

    n_train = x_gpu.shape[0]
    batch_size = int(config.get("batch_size", 256))
    generator = torch.Generator()
    generator.manual_seed(seed + 73)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", epochs))
    selection_metric = str(config.get("selection_metric", "balanced_accuracy"))
    select_last_epoch = bool(config.get("select_last_epoch", False))
    prediction_batch_size = int(config.get("batch_size", 256))

    loss_history: list[np.ndarray] = []
    probability_history: list[np.ndarray] = []
    prediction_history: list[np.ndarray] = []
    margin_history: list[np.ndarray] = []
    correctness_history: list[np.ndarray] = []
    val_probability_history: list[np.ndarray] = []

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    pbar = tqdm(range(epochs), desc="training", leave=False)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            batch_features = x_gpu[idx]
            batch_labels = y_tensor[idx]
            batch_weights = w_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            losses = criterion(logits, batch_labels)
            weighted_loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
            weighted_loss.backward()
            optimizer.step()

        train_probabilities = predict_probabilities(
            model, x_gpu, prediction_batch_size, device
        )
        val_probabilities = predict_probabilities(
            model, val_features, prediction_batch_size, device
        )
        metric = _validation_metric(val_probabilities, val_labels, selection_metric)
        pbar.set_postfix({"val": f"{metric:.4f}", "best": f"{best_metric:.4f}"})

        if record_dynamics:
            label_probability = train_probabilities[
                np.arange(len(train_labels)), train_labels
            ]
            alternative = train_probabilities.copy()
            alternative[np.arange(len(train_labels)), train_labels] = -np.inf
            max_other = alternative.max(axis=1)
            losses = -np.log(np.clip(label_probability, 1e-12, 1.0))
            predictions = train_probabilities.argmax(axis=1)

            loss_history.append(losses.astype(np.float32))
            probability_history.append(train_probabilities.astype(np.float32))
            prediction_history.append(predictions.astype(np.int64))
            margin_history.append((label_probability - max_other).astype(np.float32))
            correctness_history.append((predictions == train_labels).astype(np.int8))
            val_probability_history.append(val_probabilities.astype(np.float32))

        if select_last_epoch:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif metric > best_metric + 1e-12:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    best_train_probabilities = predict_probabilities(
        model, x_gpu, prediction_batch_size, device
    )

    def stack_or_none(items: list[np.ndarray]) -> np.ndarray | None:
        if not items:
            return None
        return np.stack(items, axis=1)

    # probability history has shape [n, epochs, classes].
    probability_array = (
        np.stack(probability_history, axis=1) if probability_history else None
    )

    return ProbeResult(
        best_state=best_state,
        initial_state=copy.deepcopy(initial_state),
        best_epoch=best_epoch,
        best_validation_metric=float(best_metric),
        train_probabilities=best_train_probabilities,
        loss_history=stack_or_none(loss_history),
        probability_history=probability_array,
        prediction_history=stack_or_none(prediction_history),
        margin_history=stack_or_none(margin_history),
        correctness_history=stack_or_none(correctness_history),
        val_probability_history=(
            np.stack(val_probability_history, axis=1)
            if val_probability_history
            else None
        ),
    )


def _worst_group_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    predictions = probabilities.argmax(axis=1)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if len(predictions) != len(labels) or len(groups) != len(labels):
        raise ValueError("Probabilities, labels, and groups must have equal length.")
    scores = [
        float((predictions[groups == group] == labels[groups == group]).mean())
        for group in np.unique(groups)
    ]
    if not scores:
        raise ValueError("At least one validation group is required.")
    return float(min(scores))


def train_group_dro_linear_head(
    train_features: np.ndarray | torch.Tensor,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray | torch.Tensor,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> ProbeResult:
    """Train a linear head with an oracle-group GroupDRO objective.

    The adversarial group weights are updated from mini-batch group losses.
    Group IDs may be non-contiguous. Validation defaults to worst-group
    accuracy, which deliberately makes this an oracle baseline.
    """
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_groups = np.asarray(train_groups, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    val_groups = np.asarray(val_groups, dtype=np.int64)
    if len(train_labels) != len(train_groups):
        raise ValueError("train_labels and train_groups must have equal length.")
    if len(val_labels) != len(val_groups):
        raise ValueError("val_labels and val_groups must have equal length.")
    if len(train_labels) == 0:
        raise ValueError("GroupDRO requires at least one training example.")

    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    input_dim = int(train_features.shape[1])
    num_classes = int(max(train_labels.max(), val_labels.max()) + 1)
    if initial_state is None:
        initial_state = make_initial_state(input_dim, num_classes, seed)

    model = LinearHead(input_dim, num_classes)
    model.load_state_dict(copy.deepcopy(initial_state))
    model.to(device)

    if isinstance(train_features, torch.Tensor):
        x_gpu = train_features.to(device=device, dtype=torch.float32)
    else:
        x_gpu = torch.from_numpy(np.asarray(train_features)).float().to(device)
    y_tensor = torch.from_numpy(train_labels).long().to(device)

    _, encoded_groups = np.unique(train_groups, return_inverse=True)
    num_groups = int(encoded_groups.max()) + 1
    group_tensor = torch.from_numpy(encoded_groups).long().to(device)
    group_counts = torch.bincount(group_tensor, minlength=num_groups).float()
    if bool((group_counts == 0).any()):
        raise ValueError("Every encoded GroupDRO group must be non-empty.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    adversarial_weights = torch.full(
        (num_groups,), 1.0 / num_groups, dtype=torch.float32, device=device
    )
    group_step_size = float(config.get("group_step_size", 0.01))
    group_adjustment = float(config.get("group_adjustment", 0.0))
    if group_step_size < 0:
        raise ValueError("group_step_size must be non-negative.")

    n_train = len(train_labels)
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", epochs))
    selection_metric = str(config.get("selection_metric", "worst_group_accuracy"))
    prediction_batch_size = int(config.get("batch_size", 256))
    generator = torch.Generator()
    generator.manual_seed(seed + 73)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    pbar = tqdm(range(epochs), desc="groupdro", leave=False)
    for epoch in pbar:
        model.train()
        permutation = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, batch_size):
            batch_indices = permutation[start : start + batch_size].to(device)
            batch_features = x_gpu[batch_indices]
            batch_labels = y_tensor[batch_indices]
            batch_groups = group_tensor[batch_indices]

            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(batch_features), batch_labels)
            group_losses = torch.zeros(num_groups, dtype=losses.dtype, device=device)
            present = torch.zeros(num_groups, dtype=torch.bool, device=device)
            for group_index in range(num_groups):
                mask = batch_groups == group_index
                if bool(mask.any()):
                    group_losses[group_index] = losses[mask].mean()
                    present[group_index] = True

            adjusted_losses = group_losses + group_adjustment / group_counts.sqrt()
            with torch.no_grad():
                update = torch.exp(
                    (group_step_size * adjusted_losses.detach()).clamp(max=50.0)
                )
                adversarial_weights[present] *= update[present]
                adversarial_weights /= adversarial_weights.sum().clamp_min(1e-12)

            present_weights = adversarial_weights[present]
            robust_loss = (
                group_losses[present]
                * present_weights / present_weights.sum().clamp_min(1e-12)
            ).sum()
            robust_loss.backward()
            optimizer.step()

        val_probabilities = predict_probabilities(
            model, val_features, prediction_batch_size, device
        )
        if selection_metric == "worst_group_accuracy":
            metric = _worst_group_accuracy(val_probabilities, val_labels, val_groups)
        else:
            metric = _validation_metric(val_probabilities, val_labels, selection_metric)
        pbar.set_postfix({"val": f"{metric:.4f}", "best": f"{best_metric:.4f}"})

        if metric > best_metric + 1e-12:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    best_train_probabilities = predict_probabilities(
        model, x_gpu, prediction_batch_size, device
    )
    return ProbeResult(
        best_state=best_state,
        initial_state=copy.deepcopy(initial_state),
        best_epoch=best_epoch,
        best_validation_metric=float(best_metric),
        train_probabilities=best_train_probabilities,
        loss_history=None,
        probability_history=None,
        prediction_history=None,
        margin_history=None,
        correctness_history=None,
    )


def train_jtt_linear_head(
    train_features: np.ndarray | torch.Tensor,
    train_labels: np.ndarray,
    val_features: np.ndarray | torch.Tensor,
    val_labels: np.ndarray,
    config: dict[str, Any],
    jtt_config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[ProbeResult, np.ndarray]:
    """Train a two-stage, group-blind Just Train Twice (JTT) baseline."""
    identification_epochs = int(jtt_config.get("identification_epochs", 5))
    upweight = float(jtt_config.get("upweight", 10.0))
    if identification_epochs < 1:
        raise ValueError("JTT identification_epochs must be at least 1.")
    if upweight < 1.0:
        raise ValueError("JTT upweight must be at least 1.")

    identification_config = copy.deepcopy(config)
    identification_config.update(jtt_config.get("identification_training", {}))
    identification_config["epochs"] = identification_epochs
    identification_config["patience"] = identification_epochs
    identification_config["select_last_epoch"] = True

    identification_probe = train_linear_head(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        config=identification_config,
        seed=seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
    )
    error_mask = (
        identification_probe.train_probabilities.argmax(axis=1)
        != np.asarray(train_labels, dtype=np.int64)
    )
    sample_weights = np.ones(len(train_labels), dtype=np.float32)
    sample_weights[error_mask] = upweight

    final_config = copy.deepcopy(config)
    final_config.update(jtt_config.get("final_training", {}))
    final_config["select_last_epoch"] = False
    final_probe = train_linear_head(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        config=final_config,
        seed=seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
        sample_weights=sample_weights,
    )
    return final_probe, error_mask


def predict_from_state(
    state: dict[str, torch.Tensor],
    features: np.ndarray | torch.Tensor,
    input_dim: int,
    num_classes: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(config.get("device", "auto"))
    model = LinearHead(input_dim, num_classes)
    model.load_state_dict(state)
    model.to(device)
    probabilities = predict_probabilities(
        model,
        features,
        int(config.get("batch_size", 256)),
        device,
    )
    return probabilities.argmax(axis=1), probabilities
