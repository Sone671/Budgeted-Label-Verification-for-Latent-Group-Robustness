#!/usr/bin/env python
"""Replay class-conditional conformal verification on probe artifacts.

The default ``--oof-folds 1`` mode uses the cached *in-sample* train
probabilities produced by Stage 1, so its p-values are useful for ranking
diagnostics but do not have a formal conformal validity guarantee.  With
``--oof-folds K`` for K > 1, the script instead trains K probes from the saved
Stage-1 initialization.  Each train example is scored only by the probe whose
training fold excluded it.

OOF mode also splits the public clean validation set once, stratified by clean
label.  The model-selection part is used for early stopping in every fold; the
calibration part is never used for training or early stopping.  Each fold's
held-out train scores are calibrated against predictions from that same fold's
probe.  Private labels and group annotations are opened only after query
selection is frozen.

The calibration score is cross entropy on the public clean validation split.
For a train example with observed class c and nonconformity A_i, the Mondrian
upper-tail p-value is

    p_i = (1 + #{j in validation: y_j=c and A_j >= A_i}) / (n_c + 1).

Small p-values therefore mean that the observed train label looks unusually
incompatible with the probe relative to clean validation examples of the same
class.  The pilot can rank directly by p-value, or guard that ranking with a
BH/e-BH discovery-count test and fall back to a cached legal score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame, query_composition
from robust_verify.config import load_config
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.features import load_features
from robust_verify.rivet import class_prior_adjusted_noise_score
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count


OOF_CACHE_VERSION = 1


def cross_entropy_nonconformity(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return -log p(observed label); larger values are more nonconforming."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
        raise ValueError("probabilities must have shape [n, num_classes]")
    if ((labels < 0) | (labels >= probabilities.shape[1])).any():
        raise ValueError("labels contain an invalid class index")
    chosen = probabilities[np.arange(len(labels)), labels]
    return -np.log(np.clip(chosen, 1e-12, 1.0))


def mondrian_upper_tail_pvalues(
    candidate_scores: np.ndarray,
    candidate_labels: np.ndarray,
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
) -> np.ndarray:
    """Compute class-conditional conformal p-values with conservative ties.

    A calibration score equal to the candidate score is counted in the upper
    tail.  Every candidate class must occur in the calibration set.
    """

    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    candidate_labels = np.asarray(candidate_labels)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    calibration_labels = np.asarray(calibration_labels)
    if candidate_scores.shape != candidate_labels.shape:
        raise ValueError("candidate scores and labels must have equal shape")
    if calibration_scores.shape != calibration_labels.shape:
        raise ValueError("calibration scores and labels must have equal shape")
    if not np.isfinite(candidate_scores).all() or not np.isfinite(calibration_scores).all():
        raise ValueError("nonconformity scores must be finite")

    pvalues = np.ones(len(candidate_scores), dtype=np.float64)
    for class_value in np.unique(candidate_labels):
        candidate_mask = candidate_labels == class_value
        reference = np.sort(calibration_scores[calibration_labels == class_value])
        if len(reference) == 0:
            raise ValueError(f"class {class_value!r} is absent from calibration labels")
        left = np.searchsorted(
            reference, candidate_scores[candidate_mask], side="left"
        )
        upper_tail_count = len(reference) - left
        pvalues[candidate_mask] = (1.0 + upper_tail_count) / (len(reference) + 1.0)
    return pvalues


def bh_rejections(pvalues: np.ndarray, level: float) -> np.ndarray:
    """Benjamini-Hochberg step-up rejection indices, ordered by p-value."""

    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all():
        raise ValueError("pvalues must be a finite one-dimensional array")
    if ((pvalues < 0.0) | (pvalues > 1.0)).any():
        raise ValueError("pvalues must lie in [0, 1]")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    order = np.argsort(pvalues, kind="stable")
    thresholds = level * np.arange(1, len(pvalues) + 1) / max(1, len(pvalues))
    accepted = np.flatnonzero(pvalues[order] <= thresholds)
    if len(accepted) == 0:
        return np.empty(0, dtype=np.int64)
    return order[: int(accepted[-1]) + 1].astype(np.int64)


def power_evalues(pvalues: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """Convert super-uniform p-values to power e-values.

    For kappa in (0,1), e=(1-kappa)*p**(-kappa) has null expectation at most
    one when p is super-uniform.
    """

    pvalues = np.asarray(pvalues, dtype=np.float64)
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0, 1)")
    if ((pvalues <= 0.0) | (pvalues > 1.0)).any():
        raise ValueError("pvalues must lie in (0, 1]")
    return (1.0 - kappa) * np.power(pvalues, -kappa)


def ebh_rejections(evalues: np.ndarray, level: float) -> np.ndarray:
    """e-BH step-up rejection indices, ordered by decreasing e-value."""

    evalues = np.asarray(evalues, dtype=np.float64)
    if evalues.ndim != 1 or not np.isfinite(evalues).all():
        raise ValueError("evalues must be a finite one-dimensional array")
    if (evalues < 0.0).any():
        raise ValueError("evalues must be nonnegative")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    order = np.argsort(-evalues, kind="stable")
    thresholds = len(evalues) / (level * np.arange(1, len(evalues) + 1))
    accepted = np.flatnonzero(evalues[order] >= thresholds)
    if len(accepted) == 0:
        return np.empty(0, dtype=np.int64)
    return order[: int(accepted[-1]) + 1].astype(np.int64)


def guarded_budget_selection(
    pvalues: np.ndarray,
    fallback_ranking: np.ndarray,
    budget_count: int,
    discovery_indices: np.ndarray,
    *,
    min_discovery_fraction: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Use conformal discoveries only when they cover enough of the budget.

    If the guard activates but the discovery set is slightly smaller than the
    budget, the remaining slots are filled from the frozen fallback ranking.
    """

    pvalues = np.asarray(pvalues, dtype=np.float64)
    fallback_ranking = np.asarray(fallback_ranking, dtype=np.int64)
    discovery_indices = np.asarray(discovery_indices, dtype=np.int64)
    if not 0 <= budget_count <= len(pvalues):
        raise ValueError("budget_count must lie in [0, n]")
    if not 0.0 < min_discovery_fraction <= 1.0:
        raise ValueError("min_discovery_fraction must lie in (0, 1]")
    required = int(np.ceil(min_discovery_fraction * budget_count))
    if len(discovery_indices) >= required:
        discovery_order = discovery_indices[
            np.argsort(pvalues[discovery_indices], kind="stable")
        ]
        conformal = discovery_order[:budget_count]
        if len(conformal) < budget_count:
            chosen = set(conformal.tolist())
            fill = np.asarray(
                [index for index in fallback_ranking if int(index) not in chosen],
                dtype=np.int64,
            )[: budget_count - len(conformal)]
            conformal = np.concatenate([conformal, fill])
        return conformal.astype(np.int64), True
    return fallback_ranking[:budget_count].astype(np.int64), False


def validate_oof_parameters(
    oof_folds: int,
    calibration_fraction: float,
    *,
    noisy_labels: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
) -> None:
    """Validate numeric OOF settings and optional stratification labels."""

    if isinstance(oof_folds, bool) or not isinstance(oof_folds, (int, np.integer)):
        raise ValueError("oof_folds must be an integer")
    if oof_folds < 1:
        raise ValueError("oof_folds must be at least 1")
    if not np.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    if oof_folds == 1:
        return

    if noisy_labels is not None:
        noisy_labels = _validate_stratification_labels(noisy_labels, "noisy_labels")
        _, counts = np.unique(noisy_labels, return_counts=True)
        if int(counts.min()) < oof_folds:
            raise ValueError(
                "every noisy-label class must contain at least oof_folds examples"
            )
    if val_labels is not None:
        val_labels = _validate_stratification_labels(val_labels, "val_labels")
        _, counts = np.unique(val_labels, return_counts=True)
        if int(counts.min()) < 2:
            raise ValueError(
                "every validation class needs at least two examples for disjoint "
                "model-selection and calibration subsets"
            )


def _validate_stratification_labels(labels: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{name} must contain at least two classes")
    return labels


def stratified_validation_split(
    labels: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed, disjoint model-selection and calibration indices."""

    labels = _validate_stratification_labels(labels, "val_labels")
    validate_oof_parameters(
        2,
        calibration_fraction,
        val_labels=labels,
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=calibration_fraction,
        random_state=seed,
    )
    try:
        model_selection, calibration = next(
            splitter.split(np.zeros(len(labels), dtype=np.int8), labels)
        )
    except ValueError as error:
        raise ValueError(
            "calibration_fraction cannot produce stratified model-selection and "
            "calibration subsets containing every validation class"
        ) from error

    model_selection = np.sort(model_selection.astype(np.int64))
    calibration = np.sort(calibration.astype(np.int64))
    if np.intersect1d(model_selection, calibration).size:
        raise RuntimeError("validation split unexpectedly overlaps")
    combined = np.sort(np.concatenate([model_selection, calibration]))
    if not np.array_equal(combined, np.arange(len(labels))):
        raise RuntimeError("validation split does not cover every example exactly once")
    for subset_name, subset in (
        ("model-selection", model_selection),
        ("calibration", calibration),
    ):
        if not np.array_equal(np.unique(labels[subset]), np.unique(labels)):
            raise ValueError(f"{subset_name} subset is missing a validation class")
    return model_selection, calibration


def stratified_oof_fold_ids(
    noisy_labels: np.ndarray,
    oof_folds: int,
    seed: int,
) -> np.ndarray:
    """Assign every train example to exactly one noisy-label-stratified fold."""

    noisy_labels = _validate_stratification_labels(noisy_labels, "noisy_labels")
    validate_oof_parameters(
        oof_folds,
        0.5,
        noisy_labels=noisy_labels,
    )
    if oof_folds == 1:
        return np.zeros(len(noisy_labels), dtype=np.int64)

    splitter = StratifiedKFold(
        n_splits=oof_folds,
        shuffle=True,
        random_state=seed,
    )
    fold_ids = np.full(len(noisy_labels), -1, dtype=np.int64)
    for fold_id, (_, held_out) in enumerate(
        splitter.split(np.zeros(len(noisy_labels), dtype=np.int8), noisy_labels)
    ):
        if (fold_ids[held_out] != -1).any():
            raise RuntimeError("an OOF example was assigned to more than one fold")
        fold_ids[held_out] = fold_id
    if (fold_ids < 0).any():
        raise RuntimeError("OOF folds do not cover every train example")
    return fold_ids


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _oof_cache_directory(
    cache_root: Path,
    *,
    source_root: Path,
    prefix: str,
    oof_folds: int,
    calibration_fraction: float,
    seed: int,
    noisy_labels: np.ndarray,
    val_labels: np.ndarray,
    training_config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
) -> Path:
    payload = {
        "version": OOF_CACHE_VERSION,
        "source_root": str(source_root.resolve()),
        "prefix": prefix,
        "oof_folds": oof_folds,
        "calibration_fraction": calibration_fraction,
        "seed": seed,
        "noisy_labels": _array_digest(noisy_labels),
        "val_labels": _array_digest(val_labels),
        "training_config": training_config,
        "initial_state": _state_digest(initial_state),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:20]
    return cache_root / source_root.name / prefix / cache_key


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_cached_fold(
    path: Path,
    expected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            indices = cached["held_out_indices"].astype(np.int64)
            nonconformity = cached["nonconformity"].astype(np.float64)
            pvalues = cached["pvalues"].astype(np.float64)
    except (OSError, ValueError, KeyError):
        return None
    if not np.array_equal(indices, expected_indices):
        return None
    if nonconformity.shape != indices.shape or pvalues.shape != indices.shape:
        return None
    if not np.isfinite(nonconformity).all() or not np.isfinite(pvalues).all():
        return None
    if ((pvalues <= 0.0) | (pvalues > 1.0)).any():
        return None
    return nonconformity, pvalues


def _load_final_oof_cache(
    path: Path,
    expected_fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            fold_ids = cached["fold_ids"].astype(np.int64)
            nonconformity = cached["nonconformity"].astype(np.float64)
            pvalues = cached["pvalues"].astype(np.float64)
    except (OSError, ValueError, KeyError):
        return None
    if not np.array_equal(fold_ids, expected_fold_ids):
        return None
    if nonconformity.shape != fold_ids.shape or pvalues.shape != fold_ids.shape:
        return None
    if not np.isfinite(nonconformity).all() or not np.isfinite(pvalues).all():
        return None
    if ((pvalues <= 0.0) | (pvalues > 1.0)).any():
        return None
    return nonconformity, pvalues


def compute_oof_conformal_scores(
    *,
    train_features: np.ndarray,
    noisy_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    training_config: dict[str, Any],
    seed: int,
    oof_folds: int,
    calibration_fraction: float,
    initial_state: dict[str, torch.Tensor],
    cache_directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """Train/resume strict OOF probes and return stitched conformal scores.

    Only public features, noisy train labels, clean validation labels, and the
    saved initial state are accepted.  This interface deliberately has no
    private-label or group inputs.
    """

    validate_oof_parameters(
        oof_folds,
        calibration_fraction,
        noisy_labels=noisy_labels,
        val_labels=val_labels,
    )
    if oof_folds <= 1:
        raise ValueError("compute_oof_conformal_scores requires oof_folds > 1")
    if len(train_features) != len(noisy_labels):
        raise ValueError("train_features and noisy_labels must have equal length")
    if len(val_features) != len(val_labels):
        raise ValueError("val_features and val_labels must have equal length")

    model_selection_indices, calibration_indices = stratified_validation_split(
        val_labels,
        calibration_fraction,
        seed,
    )
    fold_ids = stratified_oof_fold_ids(noisy_labels, oof_folds, seed)
    final_cache = cache_directory / "oof_pvalues.npz"
    final_cached = _load_final_oof_cache(final_cache, fold_ids)
    if final_cached is not None:
        nonconformity, pvalues = final_cached
        return (
            nonconformity,
            pvalues,
            fold_ids,
            model_selection_indices,
            calibration_indices,
            True,
            oof_folds,
        )

    cache_directory.mkdir(parents=True, exist_ok=True)
    nonconformity = np.full(len(noisy_labels), np.nan, dtype=np.float64)
    pvalues = np.full(len(noisy_labels), np.nan, dtype=np.float64)
    resumed_fold_count = 0
    input_dim = int(train_features.shape[1])
    for fold_id in range(oof_folds):
        held_out_indices = np.flatnonzero(fold_ids == fold_id).astype(np.int64)
        fold_cache = cache_directory / f"fold_{fold_id:03d}.npz"
        cached = _load_cached_fold(fold_cache, held_out_indices)
        if cached is not None:
            fold_nonconformity, fold_pvalues = cached
            resumed_fold_count += 1
        else:
            train_indices = np.flatnonzero(fold_ids != fold_id).astype(np.int64)
            probe = train_linear_head(
                train_features=train_features[train_indices],
                train_labels=noisy_labels[train_indices],
                val_features=val_features[model_selection_indices],
                val_labels=val_labels[model_selection_indices],
                config=training_config,
                seed=seed + fold_id,
                initial_state=copy.deepcopy(initial_state),
                record_dynamics=False,
            )
            _, held_out_probabilities = predict_from_state(
                probe.best_state,
                train_features[held_out_indices],
                input_dim,
                2,
                training_config,
            )
            _, calibration_probabilities = predict_from_state(
                probe.best_state,
                val_features[calibration_indices],
                input_dim,
                2,
                training_config,
            )
            fold_nonconformity = cross_entropy_nonconformity(
                held_out_probabilities,
                noisy_labels[held_out_indices],
            )
            calibration_nonconformity = cross_entropy_nonconformity(
                calibration_probabilities,
                val_labels[calibration_indices],
            )
            fold_pvalues = mondrian_upper_tail_pvalues(
                fold_nonconformity,
                noisy_labels[held_out_indices],
                calibration_nonconformity,
                val_labels[calibration_indices],
            )
            _atomic_savez(
                fold_cache,
                held_out_indices=held_out_indices,
                nonconformity=fold_nonconformity,
                pvalues=fold_pvalues,
            )
        nonconformity[held_out_indices] = fold_nonconformity
        pvalues[held_out_indices] = fold_pvalues

    if not np.isfinite(nonconformity).all() or not np.isfinite(pvalues).all():
        raise RuntimeError("OOF score stitching left at least one train example unscored")
    _atomic_savez(
        final_cache,
        nonconformity=nonconformity,
        pvalues=pvalues,
        fold_ids=fold_ids,
    )
    return (
        nonconformity,
        pvalues,
        fold_ids,
        model_selection_indices,
        calibration_indices,
        False,
        resumed_fold_count,
    )


def _load_probe(root: Path, prefix: str):
    checkpoint = torch.load(
        root / "stage1" / "probes" / f"{prefix}.pt",
        map_location="cpu",
        weights_only=True,
    )
    dynamics = np.load(root / "stage1" / "probes" / f"{prefix}.dynamics.npz")
    return checkpoint, dynamics


def _evaluate_state(
    state,
    *,
    features: np.ndarray,
    sample_ids: np.ndarray,
    evaluator: PrivateEvaluator,
    input_dim: int,
    training_config: dict,
):
    predictions, probabilities = predict_from_state(
        state, features, input_dim, 2, training_config
    )
    return evaluator.evaluate(sample_ids, predictions), probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", default="uniform20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.02)
    parser.add_argument(
        "--selection",
        choices=["top_budget", "bh_guard", "ebh_guard"],
        default="top_budget",
    )
    parser.add_argument("--fdr-level", type=float, default=0.20)
    parser.add_argument("--evalue-kappa", type=float, default=0.50)
    parser.add_argument(
        "--min-discovery-fraction",
        type=float,
        default=1.0,
        help="Guard activation threshold as a fraction of the query budget.",
    )
    parser.add_argument(
        "--oof-folds",
        type=int,
        default=1,
        help="Number of strict OOF folds; 1 preserves cached Stage-1 scoring.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.50,
        help="Fraction of clean validation reserved exclusively for calibration.",
    )
    parser.add_argument(
        "--oof-cache-root",
        default="outputs/conformal_verification_pilot/oof_cache",
        help="Root for resumable per-fold and final OOF p-value caches.",
    )
    parser.add_argument(
        "--fallback",
        choices=["noise_score", "loss", "cpba"],
        default="cpba",
    )
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--output-root", default="outputs/conformal_verification_pilot"
    )
    args = parser.parse_args()
    validate_oof_parameters(args.oof_folds, args.calibration_fraction)

    config = load_config(args.config)
    source_root = Path(config["project"]["output_dir"])
    prefix = f"{args.noise}_seed{args.seed}"
    run_name = (
        f"{source_root.name}_{prefix}_{args.selection}_"
        f"b{args.budget:g}_q{args.fdr_level:g}_"
        f"k{args.evalue_kappa:g}_fb{args.fallback}"
    )
    if args.selection != "top_budget":
        run_name += f"_cover{args.min_discovery_fraction:g}"
    if args.oof_folds > 1:
        run_name += f"_oof{args.oof_folds}_cal{args.calibration_fraction:g}"
    output_root = ROOT / args.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    train_features, train_ids = load_features(source_root / "features" / "train.npz")
    val_features, val_ids = load_features(source_root / "features" / "val.npz")
    public = align_frame(
        pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
        train_ids,
    )
    val_public = align_frame(
        pd.read_csv(source_root / "manifests" / "val_public.csv"), val_ids
    )
    noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    checkpoint, dynamics = _load_probe(source_root, prefix)
    training_config = dict(config["training"])
    validate_oof_parameters(
        args.oof_folds,
        args.calibration_fraction,
        noisy_labels=noisy_labels,
        val_labels=val_labels,
    )

    oof_fold_ids = np.zeros(len(noisy_labels), dtype=np.int64)
    model_selection_indices = np.arange(len(val_labels), dtype=np.int64)
    calibration_indices = np.arange(len(val_labels), dtype=np.int64)
    oof_cache_hit = False
    resumed_fold_count = 0
    oof_cache_directory: Path | None = None
    if args.oof_folds == 1:
        _, val_probabilities = predict_from_state(
            checkpoint["best_state"],
            val_features,
            train_features.shape[1],
            2,
            training_config,
        )
        train_nonconformity = cross_entropy_nonconformity(
            dynamics["train_probabilities"], noisy_labels
        )
        val_nonconformity = cross_entropy_nonconformity(
            val_probabilities, val_labels
        )
        pvalues = mondrian_upper_tail_pvalues(
            train_nonconformity,
            noisy_labels,
            val_nonconformity,
            val_labels,
        )
    else:
        raw_cache_root = Path(args.oof_cache_root)
        cache_root = raw_cache_root if raw_cache_root.is_absolute() else ROOT / raw_cache_root
        oof_cache_directory = _oof_cache_directory(
            cache_root,
            source_root=source_root,
            prefix=prefix,
            oof_folds=args.oof_folds,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
            noisy_labels=noisy_labels,
            val_labels=val_labels,
            training_config=training_config,
            initial_state=checkpoint["initial_state"],
        )
        (
            train_nonconformity,
            pvalues,
            oof_fold_ids,
            model_selection_indices,
            calibration_indices,
            oof_cache_hit,
            resumed_fold_count,
        ) = compute_oof_conformal_scores(
            train_features=train_features,
            noisy_labels=noisy_labels,
            val_features=val_features,
            val_labels=val_labels,
            training_config=training_config,
            seed=args.seed,
            oof_folds=args.oof_folds,
            calibration_fraction=args.calibration_fraction,
            initial_state=checkpoint["initial_state"],
            cache_directory=oof_cache_directory,
        )
    evalues = power_evalues(pvalues, args.evalue_kappa)
    bh_indices = bh_rejections(pvalues, args.fdr_level)
    ebh_indices = ebh_rejections(evalues, args.fdr_level)

    legal_scores = build_legal_scores(
        dynamics["train_probabilities"], noisy_labels, dynamics["correctness_history"]
    )
    if args.fallback == "cpba":
        fallback_score, _ = class_prior_adjusted_noise_score(
            legal_scores["noise_score"], legal_scores["loss"], noisy_labels
        )
    else:
        fallback_score = legal_scores[args.fallback]
    fallback_ranking = descending_ranking(fallback_score, legal_scores["loss"])
    budget_count = budget_to_count(args.budget, len(train_ids))
    if args.selection == "top_budget":
        query_positions = np.argsort(pvalues, kind="stable")[:budget_count]
        conformal_activated = True
    else:
        discovery_indices = bh_indices if args.selection == "bh_guard" else ebh_indices
        query_positions, conformal_activated = guarded_budget_selection(
            pvalues,
            fallback_ranking,
            budget_count,
            discovery_indices,
            min_discovery_fraction=args.min_discovery_fraction,
        )

    np.save(output_root / "query_sample_ids.npy", train_ids[query_positions])
    np.savez_compressed(
        output_root / "conformal_scores.npz",
        sample_ids=train_ids,
        nonconformity=train_nonconformity,
        pvalue=pvalues,
        fold_id=oof_fold_ids,
        evalue=evalues,
        bh_discovery_sample_ids=train_ids[bh_indices],
        ebh_discovery_sample_ids=train_ids[ebh_indices],
    )
    # Selection is now frozen.  Private diagnostics and verification outcomes
    # are opened only after this point and cannot affect the query set.
    private = align_frame(
        pd.read_csv(source_root / "manifests" / f"{prefix}_train_private.csv"),
        train_ids,
    )
    composition = query_composition(query_positions, private)
    diagnostics = {
        "budget_fraction": args.budget,
        "budget_count": budget_count,
        "selection": args.selection,
        "fdr_level": args.fdr_level,
        "min_discovery_fraction": args.min_discovery_fraction,
        "bh_discovery_count": len(bh_indices),
        "ebh_discovery_count": len(ebh_indices),
        "conformal_activated": conformal_activated,
        "oof_folds": args.oof_folds,
        **composition,
    }
    pd.DataFrame([diagnostics]).to_csv(
        output_root / "query_diagnostics.csv", index=False
    )
    if args.oof_folds == 1:
        score_source = "cached_in_sample_probe"
        validity_assumptions = [
            "under the null, an observed train label is clean and equals its true label",
            "within each observed class, null-clean train examples and clean validation "
            "examples are IID/exchangeable",
            "validation labels are correct",
            "the query rule is fixed without private labels or group annotations",
        ]
        formal_fdr_blockers = [
            "train scores are in-sample rather than out-of-fold",
            "train/validation class-conditional exchangeability is unverified and may fail "
            "under covariate shift",
            "ordinary BH independence or PRDS conditions are unverified",
        ]
    else:
        score_source = "strict_out_of_fold_probe"
        validity_assumptions = [
            "under the null, an observed train label is clean and equals its true label",
            "within each observed class, null-clean train examples and clean calibration "
            "examples are IID/exchangeable",
            "validation calibration labels are correct",
            "the frozen model-selection/calibration split and query rule are fixed without "
            "private labels or group annotations",
        ]
        formal_fdr_blockers = [
            "train/validation class-conditional exchangeability is unverified and may fail "
            "under covariate shift",
            "ordinary BH independence or PRDS conditions are unverified because p-values "
            "share calibration examples and folds have overlapping training sets",
        ]
    metadata = {
        "source_output": str(source_root),
        "noise": args.noise,
        "seed": args.seed,
        "budget": args.budget,
        "selection": args.selection,
        "fallback": args.fallback,
        "fdr_level": args.fdr_level,
        "min_discovery_fraction": args.min_discovery_fraction,
        "evalue_kappa": args.evalue_kappa,
        "score_source": score_source,
        "conformal_score_source": score_source,
        "fallback_score_source": "cached_public_stage1_probe_dynamics",
        "query_score_source": (
            score_source if conformal_activated else "cached_public_stage1_fallback"
        ),
        "oof_folds": args.oof_folds,
        "calibration_fraction": args.calibration_fraction,
        "model_selection_validation_count": len(model_selection_indices),
        "calibration_validation_count": len(calibration_indices),
        "oof_cache_directory": (
            str(oof_cache_directory) if oof_cache_directory is not None else None
        ),
        "oof_final_cache_hit": oof_cache_hit,
        "oof_resumed_fold_count": resumed_fold_count,
        "strict_information_isolation": args.oof_folds > 1,
        "private_or_group_data_used_for_scoring": False,
        "calibration_used_for_training_or_early_stopping": args.oof_folds == 1,
        "saved_initial_state_reused_across_oof_folds": args.oof_folds > 1,
        "validity_scope": "class-conditional null-clean train examples",
        "validity_assumptions": validity_assumptions,
        "formal_fdr_guarantee": False,
        "formal_fdr_blocker": "; ".join(formal_fdr_blockers),
        "formal_fdr_blockers": formal_fdr_blockers,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame([diagnostics]).to_string(index=False))
    if args.score_only:
        return

    test_features, test_ids = load_features(source_root / "features" / "test.npz")
    evaluator = PrivateEvaluator(source_root / "manifests" / "test_private.csv")
    baseline_metrics, _ = _evaluate_state(
        checkpoint["best_state"],
        features=test_features,
        sample_ids=test_ids,
        evaluator=evaluator,
        input_dim=train_features.shape[1],
        training_config=training_config,
    )
    oracle = VerificationOracle(
        source_root / "manifests" / f"{prefix}_train_private.csv"
    )
    repaired_labels = noisy_labels.copy()
    repaired_labels[query_positions] = oracle.verify(train_ids[query_positions])
    retrained = train_linear_head(
        train_features=train_features,
        train_labels=repaired_labels,
        val_features=val_features,
        val_labels=val_labels,
        config=training_config,
        seed=args.seed,
        initial_state=copy.deepcopy(checkpoint["initial_state"]),
        record_dynamics=False,
    )
    metrics, _ = _evaluate_state(
        retrained.best_state,
        features=test_features,
        sample_ids=test_ids,
        evaluator=evaluator,
        input_dim=train_features.shape[1],
        training_config=training_config,
    )
    result = {
        "source_output": source_root.name,
        "noise_name": args.noise,
        "seed": args.seed,
        "method": f"conformal_{args.selection}",
        "budget_fraction": args.budget,
        "budget_count": budget_count,
        "baseline_wga": baseline_metrics["wga"],
        "wga": metrics["wga"],
        "delta_wga": metrics["wga"] - baseline_metrics["wga"],
        "baseline_average_accuracy": baseline_metrics["average_accuracy"],
        "average_accuracy": metrics["average_accuracy"],
        "delta_average_accuracy": (
            metrics["average_accuracy"] - baseline_metrics["average_accuracy"]
        ),
        "bh_discovery_count": len(bh_indices),
        "ebh_discovery_count": len(ebh_indices),
        "conformal_activated": conformal_activated,
        **composition,
    }
    pd.DataFrame([result]).to_csv(output_root / "results.csv", index=False)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
