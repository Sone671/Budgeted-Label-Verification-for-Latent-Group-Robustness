#!/usr/bin/env python
"""Prepare frozen ResNet-50 features and noisy-label probes for CIFAR-100N."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.training import train_linear_head


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def _extract(dataset: CIFAR100, model: nn.Module, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = np.empty((len(dataset), 2048), dtype=np.float32)
    cursor = 0
    for batch_index, (images, _) in enumerate(loader):
        values = model(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
        output[cursor : cursor + len(values)] = values
        cursor += len(values)
        if batch_index % 100 == 0:
            print(f"[features] {cursor}/{len(dataset)}", flush=True)
    return output


def run(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).resolve()
    human_path = Path(args.human_labels).resolve()
    output_root = Path(args.output_root).resolve()
    features_dir = output_root / "features"
    probes_dir = output_root / "probes"
    private_dir = output_root / "private"
    features_dir.mkdir(parents=True, exist_ok=True)
    probes_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(human_path, map_location="cpu", weights_only=False)
    noisy_labels = np.asarray(payload["noisy_label"], dtype=np.int64)
    reference_labels = np.asarray(payload["clean_label"], dtype=np.int64)
    if len(noisy_labels) != 50_000 or len(reference_labels) != 50_000:
        raise ValueError("official CIFAR-100N payload must contain 50,000 labels")

    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    train_dataset = CIFAR100(root=str(raw_root), train=True, download=False, transform=transform)
    validation_dataset = CIFAR100(
        root=str(raw_root), train=False, download=False, transform=transform
    )
    official_train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
    if not np.array_equal(reference_labels, official_train_labels):
        mismatch = int((reference_labels != official_train_labels).sum())
        raise RuntimeError(f"CIFAR-100N/CIFAR-100 training order mismatch: {mismatch}")
    validation_labels = np.asarray(validation_dataset.targets, dtype=np.int64)

    noisy_path = output_root / "noisy_labels.npy"
    private_path = private_dir / "reference_labels.npy"
    np.save(noisy_path, noisy_labels)
    np.save(private_path, reference_labels)

    train_path = features_dir / "train_resnet50_imagenet1k_v2.npy"
    validation_path = features_dir / "validation_resnet50_imagenet1k_v2.npy"
    if train_path.exists() and validation_path.exists() and not args.force_features:
        train_features = np.load(train_path)
        validation_features = np.load(validation_path)
        print("[features] using cached ResNet-50 arrays", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        backbone.to(device).eval()
        print(f"[features] extracting on {device}", flush=True)
        train_features = _extract(train_dataset, backbone, device, int(args.feature_batch_size))
        validation_features = _extract(
            validation_dataset, backbone, device, int(args.feature_batch_size)
        )
        np.save(train_path, train_features)
        np.save(validation_path, validation_features)
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if train_features.shape != (50_000, 2048) or validation_features.shape != (10_000, 2048):
        raise RuntimeError("unexpected CIFAR-100 ResNet-50 feature shapes")

    training_config = {
        "epochs": 5,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": "auto",
    }
    for seed in (0, 1, 2):
        checkpoint_path = probes_dir / f"probe_seed{seed}.pt"
        dynamics_path = probes_dir / f"probe_seed{seed}.dynamics.npz"
        if checkpoint_path.exists() and dynamics_path.exists() and not args.force_probes:
            print(f"[probe] cached seed={seed}", flush=True)
            continue
        print(f"[probe] training seed={seed}", flush=True)
        result = train_linear_head(
            train_features=train_features,
            train_labels=noisy_labels,
            val_features=validation_features,
            val_labels=validation_labels,
            config=training_config,
            seed=seed,
            record_dynamics=True,
        )
        torch.save(
            {
                "best_state": result.best_state,
                "initial_state": result.initial_state,
                "best_epoch": result.best_epoch,
                "best_validation_metric": result.best_validation_metric,
            },
            checkpoint_path,
        )
        np.savez_compressed(
            dynamics_path,
            sample_ids=np.arange(len(noisy_labels), dtype=np.int64),
            train_probabilities=result.train_probabilities.astype(np.float32),
            loss_history=result.loss_history.astype(np.float32),
            probability_history=result.probability_history.astype(np.float32),
            prediction_history=result.prediction_history.astype(np.int64),
            margin_history=result.margin_history.astype(np.float32),
            correctness_history=result.correctness_history.astype(np.int8),
            best_epoch=np.asarray(result.best_epoch, dtype=np.int32),
        )

    manifest = {
        "dataset": "CIFAR-100N",
        "human_label_path": str(human_path),
        "human_label_sha256": _sha256(human_path),
        "cifar_archive_path": str(raw_root / "cifar-100-python.tar.gz"),
        "cifar_archive_sha256": _sha256(raw_root / "cifar-100-python.tar.gz"),
        "feature_extractor": "torchvision ResNet50_Weights.IMAGENET1K_V2; final FC removed",
        "train_feature_path": str(train_path),
        "validation_feature_path": str(validation_path),
        "train_shape": list(train_features.shape),
        "validation_shape": list(validation_features.shape),
        "noise_count": int((noisy_labels != reference_labels).sum()),
        "noise_rate": float((noisy_labels != reference_labels).mean()),
        "probe_seeds": [0, 1, 2],
        "training": training_config,
        "reference_labels_role": "integrity_check_and_private_audit_evaluation_only",
    }
    _atomic_json(manifest, output_root / "preparation_manifest.json")
    print(f"[prepared] {output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=r"G:\eswa\hierarchy_guided_cifar100n_external_2026-07-27\data\cifar100",
    )
    parser.add_argument(
        "--human-labels",
        default=str(ROOT / "data" / "cifar100n_source" / "data" / "CIFAR-100_human.pt"),
    )
    parser.add_argument(
        "--output-root", default=str(ROOT / "outputs" / "audit_budget_advisor_cifar100n")
    )
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
