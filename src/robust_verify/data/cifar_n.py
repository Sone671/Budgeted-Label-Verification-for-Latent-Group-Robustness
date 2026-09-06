"""CIFAR-10N data preparation with an explicit clean-label privacy boundary.

The public CIFAR-N release contains both the human supplied noisy labels and
the original CIFAR-10 labels.  This module deliberately writes the latter only
to private manifests.  Query policies should consume ``*_train_public.csv``
and only reveal labels through :class:`VerificationOracle` after the query set
has been frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


_CIFAR10N_VARIANTS = {
    "worst_label",
    "worse_label",
    "aggre_label",
    "random_label1",
    "random_label2",
    "random_label3",
}


def build_cifar10n_canonical(
    *,
    clean_train_labels: np.ndarray,
    noisy_train_labels: np.ndarray,
    clean_test_labels: np.ndarray,
    validation_count: int,
) -> pd.DataFrame:
    """Build the canonical CIFAR-10N frame without touching image storage.

    The group is the clean semantic class and is strictly evaluation-only.  It
    yields worst-class accuracy rather than a spurious-correlation group claim.
    """
    clean_train = np.asarray(clean_train_labels, dtype=np.int64)
    noisy_train = np.asarray(noisy_train_labels, dtype=np.int64)
    clean_test = np.asarray(clean_test_labels, dtype=np.int64)
    if clean_train.ndim != 1 or noisy_train.ndim != 1 or clean_test.ndim != 1:
        raise ValueError("CIFAR-N labels must be one-dimensional arrays.")
    if len(clean_train) != len(noisy_train):
        raise ValueError("clean_train_labels and noisy_train_labels must align.")
    if not 0 < int(validation_count) < len(clean_test):
        raise ValueError("validation_count must be between 1 and len(clean_test)-1.")
    if (clean_train < 0).any() or (noisy_train < 0).any() or (clean_test < 0).any():
        raise ValueError("CIFAR-N labels must be non-negative.")

    n_train = len(clean_train)
    n_val = int(validation_count)
    n_test = len(clean_test) - n_val
    train = pd.DataFrame(
        {
            "sample_id": np.arange(n_train, dtype=np.int64),
            "image_relpath": [f"images/train/{idx:05d}.png" for idx in range(n_train)],
            "clean_label": clean_train,
            "noisy_label": noisy_train,
            "split": "train",
            "group": clean_train,
            "place": 0,
            "is_minority": 0,
        }
    )
    val = pd.DataFrame(
        {
            "sample_id": np.arange(n_train, n_train + n_val, dtype=np.int64),
            "image_relpath": [f"images/eval/{idx:05d}.png" for idx in range(n_val)],
            "clean_label": clean_test[:n_val],
            "noisy_label": clean_test[:n_val],
            "split": "val",
            "group": clean_test[:n_val],
            "place": 0,
            "is_minority": 0,
        }
    )
    test = pd.DataFrame(
        {
            "sample_id": np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64),
            "image_relpath": [
                f"images/eval/{idx:05d}.png" for idx in range(n_val, len(clean_test))
            ],
            "clean_label": clean_test[n_val:],
            "noisy_label": clean_test[n_val:],
            "split": "test",
            "group": clean_test[n_val:],
            "place": 0,
            "is_minority": 0,
        }
    )
    return pd.concat([train, val, test], ignore_index=True)


def materialize_cifar10n_images(
    root: str | Path,
    train_images: np.ndarray,
    test_images: np.ndarray,
) -> None:
    """Write CIFAR arrays once as PNGs for the manifest-based image pipeline."""
    root = Path(root)
    train_dir = root / "images" / "train"
    eval_dir = root / "images" / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(np.asarray(train_images)):
        path = train_dir / f"{index:05d}.png"
        if not path.exists():
            Image.fromarray(image).save(path)
    for index, image in enumerate(np.asarray(test_images)):
        path = eval_dir / f"{index:05d}.png"
        if not path.exists():
            Image.fromarray(image).save(path)


def _load_cifar10n_payload(root: Path, variant: str, noise_file: str) -> dict[str, Any]:
    if variant not in _CIFAR10N_VARIANTS:
        raise ValueError(
            f"Unknown CIFAR-10N label variant {variant!r}; "
            f"expected one of {sorted(_CIFAR10N_VARIANTS)}"
        )
    path = root / noise_file
    if not path.exists():
        raise FileNotFoundError(
            f"CIFAR-10N annotation file not found: {path}. "
            "Download CIFAR-10_human.pt from the official CIFAR-N release."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("CIFAR-10N annotation file must contain a dictionary.")
    # The official CIFAR-10N release calls this variant ``worse_label``;
    # keep the project's historical ``worst_label`` configuration working.
    if variant == "worst_label" and "worst_label" not in payload and "worse_label" in payload:
        payload = dict(payload)
        payload["worst_label"] = payload["worse_label"]
    missing = {"clean_label", variant}.difference(payload)
    if missing:
        raise KeyError(f"CIFAR-10N annotation file is missing keys: {sorted(missing)}")
    return payload


def load_cifar10n_metadata(data_config: dict[str, Any]) -> pd.DataFrame:
    """Load CIFAR-10/CIFAR-10N, materialize images, and return a canonical frame."""
    root = Path(data_config["root"])
    variant = str(data_config.get("noisy_label_variant", "worst_label"))
    payload = _load_cifar10n_payload(
        root,
        variant,
        str(data_config.get("noise_file", "CIFAR-10_human.pt")),
    )

    from torchvision.datasets import CIFAR10

    download = bool(data_config.get("download", False))
    train_dataset = CIFAR10(root=str(root), train=True, download=download)
    test_dataset = CIFAR10(root=str(root), train=False, download=download)
    clean_train = np.asarray(payload["clean_label"], dtype=np.int64)
    noisy_train = np.asarray(payload[variant], dtype=np.int64)
    original_train = np.asarray(train_dataset.targets, dtype=np.int64)
    if len(clean_train) != len(original_train):
        raise ValueError("CIFAR-10N labels do not align with the CIFAR-10 training set.")
    if not np.array_equal(clean_train, original_train):
        raise ValueError(
            "CIFAR-10N clean_label order differs from torchvision CIFAR-10. "
            "Use the official PyTorch ordering or provide a reordered annotation file."
        )

    canonical = build_cifar10n_canonical(
        clean_train_labels=clean_train,
        noisy_train_labels=noisy_train,
        clean_test_labels=np.asarray(test_dataset.targets, dtype=np.int64),
        validation_count=int(data_config.get("validation_count", 2_000)),
    )
    if bool(data_config.get("materialize_images", True)):
        materialize_cifar10n_images(root, train_dataset.data, test_dataset.data)
    return canonical.sort_values("sample_id").reset_index(drop=True)
