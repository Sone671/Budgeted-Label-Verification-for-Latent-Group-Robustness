"""Out-of-fold probability generation for frozen-feature query baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold


def _validated_inputs(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate OOF inputs and return normalized arrays plus fold count."""
    features = np.asarray(train_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int64)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("train features and labels have incompatible shapes")
    if len(features) == 0:
        raise ValueError("at least one training example is required")
    if labels.min(initial=0) < 0:
        raise ValueError("train labels must be non-negative")
    n_classes = int(labels.max(initial=0) + 1)
    if n_classes < 2:
        raise ValueError("at least two classes are required")
    class_counts = np.bincount(labels, minlength=n_classes)
    folds = int(folds)
    if folds < 2 or folds > int(class_counts.min()):
        raise ValueError(
            f"oof_folds={folds} requires at least {folds} examples per class; "
            f"counts={class_counts.tolist()}"
        )
    return features, labels, folds


def make_oof_fold_ids(
    train_labels: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return a deterministic, complete stratified fold assignment.

    The assignment uses only observed noisy labels.  Persisting it alongside
    OOF probabilities makes the Cleanlab ranking replayable and auditable.
    """
    labels = np.asarray(train_labels, dtype=np.int64)
    # Validate class counts without requiring callers to materialize features.
    _, labels, folds = _validated_inputs(
        np.zeros((len(labels), 1), dtype=np.float32), labels, folds=folds
    )
    opts = options or {}
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(seed) + int(opts.get("oof_split_seed_offset", 17_003)),
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    for fold, (_, held_out_idx) in enumerate(
        splitter.split(np.zeros(len(labels), dtype=np.int8), labels)
    ):
        if (fold_ids[held_out_idx] != -1).any():
            raise RuntimeError("an OOF example was assigned to more than one fold")
        fold_ids[held_out_idx] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("OOF folds do not cover every train example")
    return fold_ids


def fit_oof_probabilities_with_folds(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit deterministic OOF linear probes and return probabilities/folds."""
    features, labels, folds = _validated_inputs(
        train_features, train_labels, folds=folds
    )
    opts = options or {}
    fold_ids = make_oof_fold_ids(labels, seed=seed, folds=folds, options=opts)
    n_classes = int(labels.max(initial=0) + 1)
    probabilities = np.zeros((len(labels), n_classes), dtype=np.float64)
    max_iter = max(1, int(opts.get("oof_max_iter", 40)))
    alpha = float(opts.get("oof_alpha", 1e-4))
    tol = float(opts.get("oof_tol", 1e-3))
    for fold in range(folds):
        held_out_idx = np.flatnonzero(fold_ids == fold)
        fit_idx = np.flatnonzero(fold_ids != fold)
        estimator = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            average=True,
            random_state=int(seed) + fold,
        )
        estimator.fit(features[fit_idx], labels[fit_idx])
        probabilities[held_out_idx] = estimator.predict_proba(features[held_out_idx])
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("OOF classifier produced non-finite probabilities")
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32), fold_ids


def fit_oof_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper returning only OOF probabilities."""
    probabilities, _ = fit_oof_probabilities_with_folds(
        train_features, train_labels, seed=seed, folds=folds, options=options
    )
    return probabilities
