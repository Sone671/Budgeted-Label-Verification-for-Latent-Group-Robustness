#!/usr/bin/env python3
"""Complete, auditable runners for Experiments 2, 4A, and 4B.

The implementation deliberately separates three questions:

* Exp2 estimates the effect of target-group noisy-label coverage with an exact
  matched-pair intervention.  Budget, oracle precision, clean queries, and
  (clean class, noisy-label loss bin, margin bin) counts are identical within
  each pair.  The only designed difference is that a target-group noisy item
  in Q_high is replaced by a non-target-group noisy item from the same stratum
  in Q_low.
* Exp4A strictly pairs already-produced safe/base results and performs
  seed-level safety inference.
* Exp4B is a real held-out runner.  It locks a validation-only fallback policy,
  re-trains baseline/base/safe models with a common seed, and uses the test set
  only for final evaluation after the fallback decision has been made.

The module imports ``robust_verify`` lazily.  Therefore ``--mode self-test``
can validate matching and statistics without the project being installed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


VALIDATION_SEED = 9_999
DEFAULT_NOISES = ("uniform20", "minority_high_40")
DEFAULT_EXP2_BUDGETS = (0.01, 0.02, 0.05)
DEFAULT_EXP4_BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_DATASETS = {
    "Waterbirds": "outputs/ablation_waterbirds_10seeds",
    "CelebA": "outputs/ablation_celeba_10seeds",
    "CivilComments": "outputs/ablation_civilcomments_10seeds",
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


class MatchingInfeasible(RuntimeError):
    """Raised when no exact matched query pair exists for a locked cell."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _locked_config(path: Path, payload: Mapping[str, Any], resume: bool) -> None:
    """Create a configuration lock or reject a non-identical resumed run."""
    normalized = json.loads(_json(payload))
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"Run already exists at {path}; choose a new output directory or enable resume"
            )
        if previous != normalized:
            raise RuntimeError(
                f"Configuration differs from locked run {path}. "
                "Use a new output directory instead of resuming it."
            )
        return
    _atomic_json(path, normalized)


def _code_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _query_digest(indices: np.ndarray) -> str:
    values = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], where: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{where} is missing columns: {missing}")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _parse_mapping(values: Sequence[str] | None, defaults: Mapping[str, str]) -> dict[str, str]:
    if not values:
        return dict(defaults)
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        result[name] = path
    return result


class CsvJournal:
    """Append-only CSV checkpoint with a fixed schema and resumable keys."""

    def __init__(self, path: Path, fields: Sequence[str], key_fields: Sequence[str]):
        self.path = path
        self.fields = list(fields)
        self.key_fields = list(key_fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: set[tuple[str, ...]] = set()
        if self.path.exists() and self.path.stat().st_size:
            old = pd.read_csv(self.path, dtype=str, keep_default_na=False)
            if list(old.columns) != self.fields:
                raise RuntimeError(f"Checkpoint schema mismatch: {self.path}")
            for _, row in old.iterrows():
                self.completed.add(tuple(str(row[k]) for k in self.key_fields))

    def key(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[k]) for k in self.key_fields)

    def has(self, row: Mapping[str, Any]) -> bool:
        return self.key(row) in self.completed

    def append(self, row: Mapping[str, Any]) -> None:
        key = self.key(row)
        if key in self.completed:
            return
        extras = sorted(set(row) - set(self.fields))
        if extras:
            raise ValueError(f"Unexpected journal fields: {extras}")
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if new_file:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self.fields})
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.add(key)


def _append_gzip_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(_json(payload) + "\n")


# ---------------------------------------------------------------------------
# Data, training, and evaluation adapters
# ---------------------------------------------------------------------------


def default_training_config() -> dict[str, Any]:
    return {
        "epochs": 5,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 3,
        "selection_metric": "balanced_accuracy",
        "device": "auto",
    }


def load_data(output_dir: str, noise_name: str, seed: int) -> dict[str, Any]:
    """Load one condition without falling back across datasets or probes."""
    from robust_verify.analysis import align_frame
    from robust_verify.data.access import PrivateEvaluator
    from robust_verify.features import load_features

    root = Path(output_dir)
    manifests = root / "manifests"
    features = root / "features"
    probes = root / "stage1" / "probes"

    train_feat, train_ids = load_features(features / "train.npz")
    val_feat, val_ids = load_features(features / "val.npz")
    test_feat, test_ids = load_features(features / "test.npz")
    private = align_frame(
        pd.read_csv(manifests / f"{noise_name}_seed{seed}_train_private.csv"), train_ids
    )
    public = align_frame(
        pd.read_csv(manifests / f"{noise_name}_seed{seed}_train_public.csv"), train_ids
    )
    val_public = align_frame(pd.read_csv(manifests / "val_public.csv"), val_ids)
    val_private = align_frame(pd.read_csv(manifests / "val_private.csv"), val_ids)

    _require_columns(private, ["clean_label", "group", "is_noisy"], "train_private")
    _require_columns(public, ["noisy_label"], "train_public")
    _require_columns(val_public, ["label"], "val_public")
    _require_columns(val_private, ["group"], "val_private")

    probe_path = probes / f"{noise_name}_seed{seed}.dynamics.npz"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe missing: {probe_path}")
    with np.load(probe_path) as probe:
        if "train_probabilities" not in probe:
            raise KeyError(f"train_probabilities missing from {probe_path}")
        probabilities = np.asarray(probe["train_probabilities"])
    if probabilities.ndim != 2 or probabilities.shape[0] != len(train_ids):
        raise ValueError(
            f"Invalid probe probabilities {probabilities.shape}; expected ({len(train_ids)}, C)"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"Non-finite probe probabilities: {probe_path}")

    noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    if noisy_labels.min() < 0 or noisy_labels.max() >= probabilities.shape[1]:
        raise ValueError("noisy_label is outside probe probability columns")

    # This is deliberately noisy-label training loss, not clean-label oracle loss.
    noisy_probability = probabilities[np.arange(len(probabilities)), noisy_labels]
    loss = -np.log(np.clip(noisy_probability, 1e-12, 1.0))
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    return {
        "output_dir": str(root),
        "train_feat": train_feat,
        "train_ids": np.asarray(train_ids),
        "val_feat": val_feat,
        "val_ids": np.asarray(val_ids),
        "test_feat": test_feat,
        "test_ids": np.asarray(test_ids),
        "private": private,
        "noisy_labels": noisy_labels,
        "val_labels": val_public["label"].to_numpy(dtype=np.int64),
        "val_groups": val_private["group"].to_numpy(dtype=np.int64),
        "loss": loss,
        "margin": margin,
        "evaluator": PrivateEvaluator(manifests / "test_private.csv"),
    }


def _fit_model(
    train_feat: np.ndarray,
    train_labels: np.ndarray,
    val_feat: np.ndarray,
    val_labels: np.ndarray,
    training_cfg: Mapping[str, Any],
    seed: int,
) -> Any:
    from robust_verify.training import train_linear_head

    return train_linear_head(
        train_feat,
        train_labels,
        val_feat,
        val_labels,
        dict(training_cfg),
        seed=int(seed),
    )


def _predict_model(
    fitted: Any,
    features: np.ndarray,
    feature_dim: int,
    n_classes: int,
    training_cfg: Mapping[str, Any],
) -> np.ndarray:
    from robust_verify.training import predict_from_state

    predictions, _ = predict_from_state(
        fitted.best_state, features, int(feature_dim), int(n_classes), dict(training_cfg)
    )
    return np.asarray(predictions)


def _class_count(*label_arrays: np.ndarray) -> int:
    return int(max(int(np.max(labels)) for labels in label_arrays) + 1)


def _evaluation_metrics(evaluator: Any, test_ids: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    raw = dict(evaluator.evaluate(test_ids, predictions))
    wga = raw.get("wga")
    accuracy = raw.get("average_accuracy", raw.get("accuracy"))
    if wga is None:
        raise KeyError(f"PrivateEvaluator did not return WGA; keys={sorted(raw)}")
    group_accuracy = {}
    for k, v in raw.items():
        if k.startswith("group_") and k.endswith("_accuracy"):
            gid = k.replace("group_", "").replace("_accuracy", "")
            group_accuracy[str(gid)] = float(v)
    return {
        "wga": float(wga),
        "accuracy": float(accuracy) if accuracy is not None else float("nan"),
        "group_accuracy": group_accuracy,
        "raw": raw,
    }


def _train_and_eval(
    train_feat: np.ndarray,
    train_labels: np.ndarray,
    val_feat: np.ndarray,
    val_labels: np.ndarray,
    test_feat: np.ndarray,
    test_ids: np.ndarray,
    evaluator: Any,
    training_cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    fitted = _fit_model(train_feat, train_labels, val_feat, val_labels, training_cfg, seed)
    predictions = _predict_model(
        fitted,
        test_feat,
        train_feat.shape[1],
        _class_count(train_labels, val_labels),
        training_cfg,
    )
    return _evaluation_metrics(evaluator, test_ids, predictions)


def validation_group_diagnostics(
    data: Mapping[str, Any], training_cfg: Mapping[str, Any], seed: int = VALIDATION_SEED
) -> tuple[Any, dict[str, Any]]:
    """Train on noisy training labels, then evaluate groups on clean validation."""
    fitted = _fit_model(
        data["train_feat"],
        data["noisy_labels"],
        data["val_feat"],
        data["val_labels"],
        training_cfg,
        seed,
    )
    predictions = _predict_model(
        fitted,
        data["val_feat"],
        data["train_feat"].shape[1],
        _class_count(data["noisy_labels"], data["val_labels"]),
        training_cfg,
    )
    group_accuracies: dict[int, float] = {}
    group_sizes: dict[int, int] = {}
    for group in sorted(np.unique(data["val_groups"])):
        mask = data["val_groups"] == group
        group_sizes[int(group)] = int(mask.sum())
        group_accuracies[int(group)] = float((predictions[mask] == data["val_labels"][mask]).mean())
    if not group_accuracies:
        raise ValueError("Validation set has no groups")
    target_group = min(group_accuracies, key=lambda g: (group_accuracies[g], g))
    values = np.asarray(list(group_accuracies.values()), dtype=float)
    return fitted, {
        "target_group": int(target_group),
        "group_accuracies": group_accuracies,
        "group_sizes": group_sizes,
        "worst_group_accuracy": float(values.min()),
        "best_group_accuracy": float(values.max()),
        "group_gap": float(values.max() - values.min()),
        "validation_accuracy": float((predictions == data["val_labels"]).mean()),
    }


def identify_target_group(data: Mapping[str, Any], training_cfg: Mapping[str, Any]) -> int:
    """Return the worst validation group after noisy-train model fitting."""
    _, diagnostics = validation_group_diagnostics(data, training_cfg, VALIDATION_SEED)
    return int(diagnostics["target_group"])


def _repair(noisy_labels: np.ndarray, clean_labels: np.ndarray, query: np.ndarray) -> np.ndarray:
    repaired = np.asarray(noisy_labels, dtype=np.int64).copy()
    repaired[np.asarray(query, dtype=np.int64)] = clean_labels[np.asarray(query, dtype=np.int64)]
    return repaired


# ---------------------------------------------------------------------------
# Experiment 2 exact matching
# ---------------------------------------------------------------------------


def _quantile_bins(values: np.ndarray, requested_bins: int) -> np.ndarray:
    if requested_bins <= 1 or len(values) == 0:
        return np.zeros(len(values), dtype=np.int64)
    binned = pd.qcut(pd.Series(values), requested_bins, labels=False, duplicates="drop")
    return binned.fillna(0).astype(np.int64).to_numpy()


def _capped_proportional_quota(
    capacities: Mapping[tuple[Any, ...], int], total: int, rng: np.random.Generator
) -> dict[tuple[Any, ...], int]:
    capacities = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    if total < 0 or total > sum(capacities.values()):
        raise MatchingInfeasible(
            f"quota target {total} exceeds capacity {sum(capacities.values())}"
        )
    if total == 0:
        return {key: 0 for key in capacities}
    keys = sorted(capacities)
    cap = np.asarray([capacities[key] for key in keys], dtype=np.int64)
    raw = total * cap / cap.sum()
    quota = np.floor(raw).astype(np.int64)
    remaining = int(total - quota.sum())
    if remaining:
        tie_break = rng.random(len(keys)) * 1e-9
        order = np.argsort(-(raw - quota + tie_break))
        for position in order:
            if remaining == 0:
                break
            if quota[position] < cap[position]:
                quota[position] += 1
                remaining -= 1
    if remaining:
        raise MatchingInfeasible(f"could not distribute quota remainder {remaining}")
    result = {key: int(value) for key, value in zip(keys, quota)}
    if sum(result.values()) != total:
        raise AssertionError("internal quota total mismatch")
    return result


def _sample_with_quota(
    pools: Mapping[tuple[Any, ...], np.ndarray],
    quotas: Mapping[tuple[Any, ...], int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[tuple[Any, ...], np.ndarray]]:
    selected: list[int] = []
    remaining: dict[tuple[Any, ...], np.ndarray] = {}
    for key in sorted(pools):
        pool = np.asarray(pools[key], dtype=np.int64)
        take = int(quotas.get(key, 0))
        if take > len(pool):
            raise MatchingInfeasible(f"stratum {key} needs {take}, has {len(pool)}")
        if take:
            chosen = np.asarray(rng.choice(pool, size=take, replace=False), dtype=np.int64)
            selected.extend(chosen.tolist())
            remaining[key] = np.setdiff1d(pool, chosen, assume_unique=False)
        else:
            remaining[key] = pool.copy()
    return np.asarray(selected, dtype=np.int64), remaining


def _stratum_counts(
    indices: np.ndarray,
    clean_labels: np.ndarray,
    loss_bins: np.ndarray,
    margin_bins: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    return dict(
        Counter(
            (
                int(clean_labels[index]),
                int(loss_bins[index]),
                int(margin_bins[index]),
            )
            for index in np.asarray(indices, dtype=np.int64)
        )
    )


def _standardized_mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = math.sqrt((float(np.var(x)) + float(np.var(y))) / 2.0)
    difference = float(np.mean(x) - np.mean(y))
    if pooled <= 1e-15:
        return 0.0 if abs(difference) <= 1e-15 else math.copysign(math.inf, difference)
    return difference / pooled


def _categorical_max_smd(x: np.ndarray, y: np.ndarray) -> float:
    maximum = 0.0
    for value in sorted(set(np.asarray(x).tolist()) | set(np.asarray(y).tolist())):
        px = float(np.mean(np.asarray(x) == value))
        py = float(np.mean(np.asarray(y) == value))
        denominator = math.sqrt((px * (1 - px) + py * (1 - py)) / 2.0)
        smd = 0.0 if denominator <= 1e-15 else (px - py) / denominator
        maximum = max(maximum, abs(smd))
    return maximum


def _attempt_exact_pair(
    is_noisy: np.ndarray,
    groups: np.ndarray,
    clean_labels: np.ndarray,
    loss: np.ndarray,
    margin: np.ndarray,
    budget: int,
    precision: float,
    target_group: int,
    high_frac: float,
    low_frac: float,
    n_loss_bins: int,
    n_margin_bins: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(is_noisy)
    arrays = [groups, clean_labels, loss, margin]
    if any(len(values) != n for values in arrays):
        raise ValueError("matching arrays have unequal lengths")
    if budget <= 0 or budget > n:
        raise MatchingInfeasible(f"budget {budget} is outside [1, {n}]")
    if not 0 <= low_frac < high_frac <= 1:
        raise ValueError("require 0 <= low_frac < high_frac <= 1")

    noisy_target = int(round(budget * precision))
    clean_target = budget - noisy_target
    target_high = int(round(noisy_target * high_frac))
    target_low = int(round(noisy_target * low_frac))
    differential_target = target_high - target_low
    if noisy_target <= 0 or differential_target <= 0:
        raise MatchingInfeasible(
            "rounded budget/precision/contrast produces no positive target-group intervention"
        )

    loss_bins = _quantile_bins(loss, n_loss_bins)
    margin_bins = _quantile_bins(margin, n_margin_bins)
    strata = [
        (int(clean_labels[i]), int(loss_bins[i]), int(margin_bins[i])) for i in range(n)
    ]
    clean_pools: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    target_noisy_pools: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    other_noisy_pools: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        if not is_noisy[index]:
            clean_pools[stratum].append(index)
        elif int(groups[index]) == int(target_group):
            target_noisy_pools[stratum].append(index)
        else:
            other_noisy_pools[stratum].append(index)

    clean_np = {key: np.asarray(value, dtype=np.int64) for key, value in clean_pools.items()}
    target_np = {
        key: np.asarray(target_noisy_pools.get(key, []), dtype=np.int64)
        for key in set(target_noisy_pools) | set(other_noisy_pools)
    }
    other_np = {
        key: np.asarray(other_noisy_pools.get(key, []), dtype=np.int64)
        for key in set(target_noisy_pools) | set(other_noisy_pools)
    }

    target_capacity = sum(map(len, target_np.values()))
    other_capacity = sum(map(len, other_np.values()))
    exchange_capacity = sum(min(len(target_np[key]), len(other_np[key])) for key in target_np)
    reasons = []
    if sum(map(len, clean_np.values())) < clean_target:
        reasons.append(f"clean capacity {sum(map(len, clean_np.values()))} < {clean_target}")
    if target_capacity < target_high:
        reasons.append(f"target noisy capacity {target_capacity} < {target_high}")
    if other_capacity < noisy_target - target_low:
        reasons.append(f"non-target noisy capacity {other_capacity} < {noisy_target - target_low}")
    if exchange_capacity < differential_target:
        reasons.append(f"within-stratum exchange capacity {exchange_capacity} < {differential_target}")
    if reasons:
        raise MatchingInfeasible("; ".join(reasons))

    # Each differential slot is TG-noisy in high and NTG-noisy in low, in the
    # same stratum.  Common TG/NTG noisy samples and all clean samples are shared.
    differential_capacity = {
        key: min(len(target_np[key]), len(other_np[key])) for key in target_np
    }
    differential_quota = _capped_proportional_quota(
        differential_capacity, differential_target, rng
    )
    differential_tg, remaining_tg = _sample_with_quota(target_np, differential_quota, rng)
    differential_ntg, remaining_ntg = _sample_with_quota(other_np, differential_quota, rng)

    common_tg_quota = _capped_proportional_quota(
        {key: len(value) for key, value in remaining_tg.items()}, target_low, rng
    )
    common_tg, _ = _sample_with_quota(remaining_tg, common_tg_quota, rng)
    common_ntg_target = noisy_target - target_high
    common_ntg_quota = _capped_proportional_quota(
        {key: len(value) for key, value in remaining_ntg.items()}, common_ntg_target, rng
    )
    common_ntg, _ = _sample_with_quota(remaining_ntg, common_ntg_quota, rng)
    clean_quota = _capped_proportional_quota(
        {key: len(value) for key, value in clean_np.items()}, clean_target, rng
    )
    common_clean, _ = _sample_with_quota(clean_np, clean_quota, rng)

    q_high = np.sort(
        np.concatenate([common_clean, common_tg, common_ntg, differential_tg])
    ).astype(np.int64)
    q_low = np.sort(
        np.concatenate([common_clean, common_tg, common_ntg, differential_ntg])
    ).astype(np.int64)
    high_noisy = q_high[is_noisy[q_high]]
    low_noisy = q_low[is_noisy[q_low]]
    high_counts = _stratum_counts(high_noisy, clean_labels, loss_bins, margin_bins)
    low_counts = _stratum_counts(low_noisy, clean_labels, loss_bins, margin_bins)
    high_target_count = int(np.sum(groups[high_noisy] == target_group))
    low_target_count = int(np.sum(groups[low_noisy] == target_group))

    assert len(q_high) == len(q_low) == budget
    assert len(np.unique(q_high)) == len(np.unique(q_low)) == budget
    assert int(is_noisy[q_high].sum()) == int(is_noisy[q_low].sum()) == noisy_target
    assert set(q_high[~is_noisy[q_high]]) == set(q_low[~is_noisy[q_low]])
    assert high_counts == low_counts
    assert high_target_count == target_high
    assert low_target_count == target_low
    assert high_target_count > low_target_count

    target_noisy_available = int(np.sum(is_noisy & (groups == target_group)))
    loss_smd = _standardized_mean_difference(loss[q_high], loss[q_low])
    margin_smd = _standardized_mean_difference(margin[q_high], margin[q_low])
    class_smd = _categorical_max_smd(clean_labels[q_high], clean_labels[q_low])
    diagnostics = {
        "len_high": int(len(q_high)),
        "len_low": int(len(q_low)),
        "noisy_count_high": int(is_noisy[q_high].sum()),
        "noisy_count_low": int(is_noisy[q_low].sum()),
        "actual_precision_high": float(is_noisy[q_high].mean()),
        "actual_precision_low": float(is_noisy[q_low].mean()),
        "target_noisy_high": high_target_count,
        "target_noisy_low": low_target_count,
        "target_fraction_high": high_target_count / noisy_target,
        "target_fraction_low": low_target_count / noisy_target,
        "target_coverage_high": high_target_count / max(1, target_noisy_available),
        "target_coverage_low": low_target_count / max(1, target_noisy_available),
        "overlap_fraction": len(set(q_high) & set(q_low)) / budget,
        "n_loss_bins_used": int(loss_bins.max() + 1),
        "n_margin_bins_used": int(margin_bins.max() + 1),
        "n_strata": int(len(high_counts)),
        "strata_exact": high_counts == low_counts,
        "loss_smd": loss_smd,
        "margin_smd": margin_smd,
        "class_max_abs_smd": class_smd,
        "max_abs_balance_smd": max(abs(loss_smd), abs(margin_smd), class_smd),
        "class_counts_high": dict(Counter(map(int, clean_labels[q_high]))),
        "class_counts_low": dict(Counter(map(int, clean_labels[q_low]))),
        "stratum_counts_high": {str(key): value for key, value in high_counts.items()},
        "stratum_counts_low": {str(key): value for key, value in low_counts.items()},
    }
    return q_high, q_low, diagnostics


def construct_exact_matched_pair(
    private: pd.DataFrame | None,
    is_noisy: np.ndarray,
    groups: np.ndarray,
    clean_labels: np.ndarray,
    loss: np.ndarray,
    margin: np.ndarray,
    budget: int,
    precision: float,
    target_group: int,
    high_frac: float,
    low_frac: float,
    n_loss_bins: int = 4,
    n_margin_bins: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct an exact pair, coarsening bins only when required.

    ``private`` is accepted for API compatibility, but every private quantity
    used by matching is passed explicitly.  A failed locked contrast raises
    ``MatchingInfeasible``; a short or approximately matched pair is never
    returned.
    """
    del private
    rng = np.random.default_rng() if rng is None else rng
    attempts: list[str] = []
    plans = sorted(
        set(
            [(n_loss_bins, n_margin_bins)]
            + [
                (loss_bins, margin_bins)
                for loss_bins in range(n_loss_bins, 0, -1)
                for margin_bins in range(n_margin_bins, 0, -1)
            ]
        ),
        key=lambda pair: (-(pair[0] + pair[1]), -min(pair), -max(pair)),
    )
    for loss_bins, margin_bins in plans:
        try:
            high, low, diagnostics = _attempt_exact_pair(
                np.asarray(is_noisy, dtype=bool),
                np.asarray(groups),
                np.asarray(clean_labels),
                np.asarray(loss, dtype=float),
                np.asarray(margin, dtype=float),
                int(budget),
                float(precision),
                int(target_group),
                float(high_frac),
                float(low_frac),
                loss_bins,
                margin_bins,
                rng,
            )
            diagnostics["coarsening_attempts"] = len(attempts)
            return high, low, diagnostics
        except MatchingInfeasible as error:
            attempts.append(f"{loss_bins}x{margin_bins}: {error}")
    raise MatchingInfeasible(" | ".join(attempts))


# Backwards-compatible name from the uploaded prototype.
construct_true_exact_pair = construct_exact_matched_pair


EXP2_FIELDS = [
    "dataset", "noise", "budget_fraction", "budget_n", "data_seed", "pair_id",
    "pair_seed", "target_group", "target_group_val_accuracy", "valid", "status", "error",
    "baseline_wga", "high_wga", "low_wga", "delta_high", "delta_low", "ate",
    "query_size_high", "query_size_low", "noisy_count_high", "noisy_count_low",
    "actual_precision_high", "actual_precision_low", "target_noisy_high", "target_noisy_low",
    "target_fraction_high", "target_fraction_low", "target_coverage_high", "target_coverage_low",
    "overlap_fraction", "n_loss_bins_used", "n_margin_bins_used", "n_strata", "strata_exact",
    "loss_smd", "margin_smd", "class_max_abs_smd", "max_abs_balance_smd",
    "class_counts_high", "class_counts_low", "stratum_counts_high", "stratum_counts_low",
    "query_high_sha256", "query_low_sha256", "runtime_seconds",
]


def _exp2_error_row(
    dataset: str,
    noise: str,
    budget_fraction: float,
    budget_n: int,
    data_seed: int,
    pair_id: int,
    pair_seed: int,
    status: str,
    error: BaseException | str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "noise": noise,
        "budget_fraction": budget_fraction,
        "budget_n": budget_n,
        "data_seed": data_seed,
        "pair_id": pair_id,
        "pair_seed": pair_seed,
        "valid": False,
        "status": status,
        "error": str(error)[:2000],
    }


def run_exp2_grid(
    output_dirs: Mapping[str, str] | Sequence[tuple[str, str]],
    output_root: str | Path = "outputs/exp2_complete",
    noises: Sequence[str] = DEFAULT_NOISES,
    budgets: Sequence[float] = DEFAULT_EXP2_BUDGETS,
    data_seeds: Sequence[int] = tuple(range(10)),
    n_pairs: int = 30,
    precision: float = 0.40,
    high_frac: float = 0.40,
    low_frac: float = 0.05,
    n_loss_bins: int = 4,
    n_margin_bins: int = 4,
    training_cfg: Mapping[str, Any] | None = None,
    n_bootstrap: int = 10_000,
    resume: bool = True,
    save_query_indices: bool = True,
) -> pd.DataFrame:
    """Run and checkpoint the full dataset x noise x budget x seed x pair grid."""
    datasets = dict(output_dirs)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(default_training_config() if training_cfg is None else training_cfg)
    run_config = {
        "experiment": "exp2",
        "datasets": datasets,
        "noises": list(noises),
        "budgets": list(map(float, budgets)),
        "data_seeds": list(map(int, data_seeds)),
        "n_pairs": int(n_pairs),
        "precision": float(precision),
        "high_frac": float(high_frac),
        "low_frac": float(low_frac),
        "n_loss_bins": int(n_loss_bins),
        "n_margin_bins": int(n_margin_bins),
        "training": cfg,
        "code_sha256": _code_sha256(),
        "git_revision": _git_revision(),
    }
    _locked_config(root / "exp2_config.json", run_config, resume)
    journal = CsvJournal(
        root / "exp2_pairs.csv",
        EXP2_FIELDS,
        ["dataset", "noise", "budget_fraction", "data_seed", "pair_id"],
    )
    query_path = root / "exp2_queries.jsonl.gz"

    for dataset, output_dir in datasets.items():
        for noise in noises:
            for data_seed in data_seeds:
                data: dict[str, Any] | None = None
                validation_diagnostics: dict[str, Any] | None = None
                load_error: BaseException | None = None
                try:
                    data = load_data(output_dir, noise, int(data_seed))
                    _, validation_diagnostics = validation_group_diagnostics(
                        data, cfg, VALIDATION_SEED
                    )
                except BaseException as error:  # record every expected cell, then continue
                    load_error = error

                for budget_fraction in budgets:
                    budget_n = (
                        max(1, int(round(float(budget_fraction) * len(data["noisy_labels"]))))
                        if data is not None
                        else 0
                    )
                    for pair_id in range(int(n_pairs)):
                        pair_seed = _stable_seed(
                            "exp2", dataset, noise, budget_fraction, data_seed, pair_id
                        )
                        key_row = {
                            "dataset": dataset,
                            "noise": noise,
                            "budget_fraction": float(budget_fraction),
                            "data_seed": int(data_seed),
                            "pair_id": int(pair_id),
                        }
                        if journal.has(key_row):
                            continue
                        if load_error is not None or data is None or validation_diagnostics is None:
                            journal.append(
                                _exp2_error_row(
                                    dataset, noise, float(budget_fraction), budget_n,
                                    int(data_seed), pair_id, pair_seed, "load_error",
                                    load_error or "unknown load error",
                                )
                            )
                            continue

                        started = time.perf_counter()
                        private = data["private"]
                        is_noisy = private["is_noisy"].to_numpy(dtype=bool)
                        groups = private["group"].to_numpy(dtype=np.int64)
                        clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
                        target_group = int(validation_diagnostics["target_group"])
                        try:
                            q_high, q_low, diagnostics = construct_exact_matched_pair(
                                private,
                                is_noisy,
                                groups,
                                clean_labels,
                                data["loss"],
                                data["margin"],
                                budget_n,
                                precision,
                                target_group,
                                high_frac,
                                low_frac,
                                n_loss_bins,
                                n_margin_bins,
                                np.random.default_rng(pair_seed),
                            )
                            # A baseline is trained for every pair using the same seed as
                            # high and low, so both delta-WGA comparisons are paired too.
                            baseline = _train_and_eval(
                                data["train_feat"], data["noisy_labels"], data["val_feat"],
                                data["val_labels"], data["test_feat"], data["test_ids"],
                                data["evaluator"], cfg, pair_seed,
                            )
                            high = _train_and_eval(
                                data["train_feat"],
                                _repair(data["noisy_labels"], clean_labels, q_high),
                                data["val_feat"], data["val_labels"], data["test_feat"],
                                data["test_ids"], data["evaluator"], cfg, pair_seed,
                            )
                            low = _train_and_eval(
                                data["train_feat"],
                                _repair(data["noisy_labels"], clean_labels, q_low),
                                data["val_feat"], data["val_labels"], data["test_feat"],
                                data["test_ids"], data["evaluator"], cfg, pair_seed,
                            )
                            row = {
                                **key_row,
                                "budget_n": budget_n,
                                "pair_seed": pair_seed,
                                "target_group": target_group,
                                "target_group_val_accuracy": validation_diagnostics[
                                    "group_accuracies"
                                ][target_group],
                                "valid": True,
                                "status": "complete",
                                "error": "",
                                "baseline_wga": baseline["wga"],
                                "high_wga": high["wga"],
                                "low_wga": low["wga"],
                                "delta_high": high["wga"] - baseline["wga"],
                                "delta_low": low["wga"] - baseline["wga"],
                                "ate": high["wga"] - low["wga"],
                                "query_size_high": diagnostics["len_high"],
                                "query_size_low": diagnostics["len_low"],
                                "query_high_sha256": _query_digest(q_high),
                                "query_low_sha256": _query_digest(q_low),
                                "runtime_seconds": time.perf_counter() - started,
                            }
                            for name in [
                                "noisy_count_high", "noisy_count_low", "actual_precision_high",
                                "actual_precision_low", "target_noisy_high", "target_noisy_low",
                                "target_fraction_high", "target_fraction_low", "target_coverage_high",
                                "target_coverage_low", "overlap_fraction", "n_loss_bins_used",
                                "n_margin_bins_used", "n_strata", "strata_exact", "loss_smd",
                                "margin_smd", "class_max_abs_smd", "max_abs_balance_smd",
                            ]:
                                row[name] = diagnostics[name]
                            for name in [
                                "class_counts_high", "class_counts_low", "stratum_counts_high",
                                "stratum_counts_low",
                            ]:
                                row[name] = _json(diagnostics[name])
                            journal.append(row)
                            if save_query_indices:
                                _append_gzip_jsonl(
                                    query_path,
                                    {
                                        **key_row,
                                        "pair_seed": pair_seed,
                                        "train_ids_high": data["train_ids"][q_high].tolist(),
                                        "train_ids_low": data["train_ids"][q_low].tolist(),
                                    },
                                )
                        except MatchingInfeasible as error:
                            journal.append(
                                _exp2_error_row(
                                    dataset, noise, float(budget_fraction), budget_n,
                                    int(data_seed), pair_id, pair_seed, "infeasible", error,
                                )
                            )
                        except BaseException as error:
                            detail = f"{error}\n{traceback.format_exc(limit=5)}"
                            journal.append(
                                _exp2_error_row(
                                    dataset, noise, float(budget_fraction), budget_n,
                                    int(data_seed), pair_id, pair_seed, "runtime_error", detail,
                                )
                            )

    pairs = pd.read_csv(root / "exp2_pairs.csv")
    _write_exp2_summaries_and_integrity(
        pairs, root, datasets, noises, budgets, data_seeds, n_pairs, n_bootstrap
    )
    _make_exp2_plots(root)
    return pairs


def _hierarchical_bootstrap(
    values_by_seed: Mapping[int, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seeds = sorted(values_by_seed)
    if len(seeds) < 2:
        return np.asarray([], dtype=float)
    results = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = np.asarray(values_by_seed[int(seed)], dtype=float)
            seed_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        results[iteration] = np.mean(seed_means)
    return results


def _sign_flip_pvalue(values: np.ndarray, null: float = 0.0, alternative: str = "greater") -> float:
    centered = np.asarray(values, dtype=float) - null
    centered = centered[np.isfinite(centered)]
    n = len(centered)
    if n == 0:
        return math.nan
    observed = float(np.mean(centered))
    rng = np.random.default_rng(71_339)
    if n <= 18:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)))
        statistics = np.mean(signs * centered, axis=1)
        if alternative == "greater":
            return float(np.mean(statistics >= observed - 1e-15))
        return float(np.mean(np.abs(statistics) >= abs(observed) - 1e-15))
    samples = 100_000
    signs = rng.choice([-1.0, 1.0], size=(samples, n))
    statistics = np.mean(signs * centered, axis=1)
    count = np.sum(statistics >= observed) if alternative == "greater" else np.sum(
        np.abs(statistics) >= abs(observed)
    )
    return float((count + 1) / (samples + 1))


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _holm_adjust(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.where(np.isfinite(p))[0]
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _write_exp2_summaries_and_integrity(
    pairs: pd.DataFrame,
    root: Path,
    datasets: Mapping[str, str],
    noises: Sequence[str],
    budgets: Sequence[float],
    data_seeds: Sequence[int],
    n_pairs: int,
    n_bootstrap: int,
) -> None:
    valid = pairs[_as_bool(pairs["valid"])].copy()
    numeric = ["baseline_wga", "high_wga", "low_wga", "delta_high", "delta_low", "ate"]
    for column in numeric:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    seed_keys = ["dataset", "noise", "budget_fraction", "data_seed"]
    if valid.empty:
        seed_summary = pd.DataFrame(columns=seed_keys + ["n_pairs", "ate_mean"])
    else:
        seed_summary = (
            valid.groupby(seed_keys, as_index=False)
            .agg(
                n_pairs=("ate", "size"),
                baseline_wga=("baseline_wga", "mean"),
                high_wga=("high_wga", "mean"),
                low_wga=("low_wga", "mean"),
                delta_high=("delta_high", "mean"),
                delta_low=("delta_low", "mean"),
                ate_mean=("ate", "mean"),
                ate_sd=("ate", "std"),
            )
        )
    seed_summary.to_csv(root / "exp2_seed_summary.csv", index=False)

    condition_rows: list[dict[str, Any]] = []
    for condition, frame in valid.groupby(["dataset", "noise", "budget_fraction"]):
        by_seed = {
            int(seed): group["ate"].dropna().to_numpy(dtype=float)
            for seed, group in frame.groupby("data_seed")
            if group["ate"].notna().any()
        }
        seed_ates = np.asarray([values.mean() for values in by_seed.values()])
        bootstrap = _hierarchical_bootstrap(
            by_seed, n_bootstrap, np.random.default_rng(_stable_seed("exp2-bootstrap", *condition))
        )
        ci = np.percentile(bootstrap, [2.5, 97.5]) if len(bootstrap) else [math.nan, math.nan]
        wins = int(np.sum(seed_ates > 0))
        win_ci = _wilson_interval(wins, len(seed_ates))
        nonzero = seed_ates[seed_ates != 0]
        sign_p = (
            float(stats.binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5, alternative="greater").pvalue)
            if len(nonzero)
            else math.nan
        )
        condition_rows.append(
            {
                "dataset": condition[0],
                "noise": condition[1],
                "budget_fraction": condition[2],
                "n_seeds": len(seed_ates),
                "n_pairs": len(frame),
                "ate_mean": float(np.mean(seed_ates)) if len(seed_ates) else math.nan,
                "ate_ci95_low": ci[0],
                "ate_ci95_high": ci[1],
                "paired_permutation_p": _sign_flip_pvalue(seed_ates, 0.0, "greater"),
                "sign_test_p": sign_p,
                "seed_win_rate": wins / len(seed_ates) if len(seed_ates) else math.nan,
                "seed_win_rate_ci95_low": win_ci[0],
                "seed_win_rate_ci95_high": win_ci[1],
                "delta_high_mean": float(frame.groupby("data_seed")["delta_high"].mean().mean()),
                "delta_low_mean": float(frame.groupby("data_seed")["delta_low"].mean().mean()),
            }
        )
    condition_summary = pd.DataFrame(condition_rows)
    if not condition_summary.empty:
        condition_summary["permutation_p_holm"] = _holm_adjust(
            condition_summary["paired_permutation_p"].tolist()
        )
    condition_summary.to_csv(root / "exp2_condition_summary.csv", index=False)

    integrity_rows = []
    key_columns = ["dataset", "noise", "budget_fraction", "data_seed"]
    for dataset, noise, budget, seed in itertools.product(
        datasets, noises, budgets, data_seeds
    ):
        cell = pairs[
            (pairs["dataset"] == dataset)
            & (pairs["noise"] == noise)
            & np.isclose(pd.to_numeric(pairs["budget_fraction"], errors="coerce"), float(budget))
            & (pd.to_numeric(pairs["data_seed"], errors="coerce") == int(seed))
        ]
        valid_count = int(_as_bool(cell["valid"]).sum()) if len(cell) else 0
        statuses = Counter(cell["status"].fillna("missing").astype(str)) if len(cell) else Counter()
        if len(cell) < n_pairs:
            status = "missing"
        elif valid_count == n_pairs:
            status = "complete"
        elif statuses.get("infeasible", 0) == n_pairs:
            status = "infeasible"
        else:
            status = "failed"
        integrity_rows.append(
            {
                **dict(zip(key_columns, [dataset, noise, budget, seed])),
                "expected_pairs": n_pairs,
                "observed_rows": len(cell),
                "valid_pairs": valid_count,
                "status": status,
                "status_counts": _json(statuses),
            }
        )
    pd.DataFrame(integrity_rows).to_csv(root / "exp2_integrity.csv", index=False)


def _make_exp2_plots(root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    summary_path = root / "exp2_condition_summary.csv"
    pairs_path = root / "exp2_pairs.csv"
    if not summary_path.exists() or not pairs_path.exists():
        return
    summary = pd.read_csv(summary_path)
    pairs = pd.read_csv(pairs_path)
    if summary.empty:
        return
    labels = [
        f"{row.dataset} | {row.noise} | {float(row.budget_fraction):.1%}"
        for row in summary.itertuples()
    ]
    y = np.arange(len(summary))
    means = summary["ate_mean"].to_numpy(float)
    errors = np.vstack(
        [means - summary["ate_ci95_low"].to_numpy(float), summary["ate_ci95_high"].to_numpy(float) - means]
    )
    fig, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(summary))))
    axis.errorbar(means, y, xerr=errors, fmt="o", capsize=3)
    axis.axvline(0, color="black", linewidth=1)
    axis.set(yticks=y, yticklabels=labels, xlabel="ATE: WGA(high) - WGA(low)")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(root / "exp2_ate_forest.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(max(9, 0.45 * len(summary)), 5))
    axis.bar(x - 0.2, summary["delta_high_mean"], 0.4, label="High")
    axis.bar(x + 0.2, summary["delta_low_mean"], 0.4, label="Low")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xticks=x, xticklabels=labels, ylabel="Delta WGA")
    axis.tick_params(axis="x", rotation=70)
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "exp2_delta_wga.png", dpi=180)
    plt.close(fig)

    valid = pairs[_as_bool(pairs["valid"])].copy()
    if not valid.empty:
        for column in ["loss_smd", "margin_smd", "class_max_abs_smd"]:
            valid[column] = pd.to_numeric(valid[column], errors="coerce").abs()
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.boxplot(
            [valid["loss_smd"].dropna(), valid["margin_smd"].dropna(), valid["class_max_abs_smd"].dropna()],
            labels=["loss", "margin", "class"],
        )
        axis.set_ylabel("Absolute standardized mean difference")
        axis.axhline(0.1, color="red", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(root / "exp2_matching_balance.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 4A: strict post-hoc safety inference
# ---------------------------------------------------------------------------


def _bootstrap_seed_mean(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.asarray([], dtype=float)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return values[indices].mean(axis=1)


def _lower_tail_cvar(values: np.ndarray, fraction: float = 0.10) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return math.nan
    count = max(1, int(math.ceil(fraction * len(values))))
    return float(np.mean(values[:count]))


def run_exp4a_full(
    results_paths: Mapping[str, str],
    output_root: str | Path = "outputs/exp4_complete",
    base_method: str = "noise_score",
    safe_method: str = "noise_score_cpba_tc_v8_fallback",
    epsilon: float = 0.01,
    n_bootstrap: int = 10_000,
    rng_seed: int = 12_345,
    heldout_seeds: Sequence[int] | None = None,
    max_failure_rate: float = 0.10,
    resume: bool = True,
) -> pd.DataFrame:
    """Strictly pair all safe/base cells and run one-sided safety inference."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _locked_config(
        root / "exp4a_config.json",
        {
            "experiment": "exp4a",
            "results_paths": dict(results_paths),
            "base_method": base_method,
            "safe_method": safe_method,
            "epsilon": float(epsilon),
            "max_failure_rate": float(max_failure_rate),
            "n_bootstrap": int(n_bootstrap),
            "rng_seed": int(rng_seed),
            "heldout_seeds": None if heldout_seeds is None else list(map(int, heldout_seeds)),
            "code_sha256": _code_sha256(),
            "git_revision": _git_revision(),
        },
        resume,
    )
    rows: list[dict[str, Any]] = []
    paired_all: list[pd.DataFrame] = []
    for dataset, path in results_paths.items():
        frame = pd.read_csv(path)
        _require_columns(
            frame,
            ["method", "noise_name", "budget_fraction", "seed", "delta_wga"],
            str(path),
        )
        selected = frame[frame["method"].isin([base_method, safe_method])].copy()
        if heldout_seeds is not None:
            selected = selected[selected["seed"].isin(list(map(int, heldout_seeds)))]
        if selected.empty:
            raise ValueError(f"No selected safe/base rows in {path}")
        condition_keys = selected[["noise_name", "budget_fraction"]].drop_duplicates()
        for condition in condition_keys.itertuples(index=False):
            cell = selected[
                (selected["noise_name"] == condition.noise_name)
                & np.isclose(selected["budget_fraction"].astype(float), float(condition.budget_fraction))
            ]
            base = cell[cell["method"] == base_method]
            safe = cell[cell["method"] == safe_method]
            merge_keys = ["noise_name", "budget_fraction", "seed"]
            if base.duplicated(merge_keys).any() or safe.duplicated(merge_keys).any():
                raise ValueError(
                    f"Duplicate safe/base key in {dataset}/{condition.noise_name}/{condition.budget_fraction}"
                )
            paired = safe.merge(
                base,
                on=merge_keys,
                suffixes=("_safe", "_base"),
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
            if not (paired["_merge"] == "both").all():
                missing = paired.loc[paired["_merge"] != "both", merge_keys + ["_merge"]]
                raise ValueError(
                    f"Unpaired safe/base rows in {dataset}/{condition.noise_name}/"
                    f"{condition.budget_fraction}: {missing.to_dict('records')}"
                )
            if heldout_seeds is not None and set(paired["seed"].astype(int)) != set(
                map(int, heldout_seeds)
            ):
                raise ValueError(
                    f"Held-out seed mismatch in {dataset}/{condition.noise_name}/"
                    f"{condition.budget_fraction}"
                )
            paired["dataset"] = dataset
            paired["difference"] = paired["delta_wga_safe"] - paired["delta_wga_base"]
            paired_all.append(paired)

            # Aggregate by seed first.  This remains correct if a future CSV has
            # several repeated rows per seed within a condition.
            seed_differences = paired.groupby("seed", as_index=False)["difference"].mean()
            differences = seed_differences["difference"].to_numpy(dtype=float)
            bootstrap = _bootstrap_seed_mean(
                differences,
                n_bootstrap,
                np.random.default_rng(_stable_seed(rng_seed, dataset, condition.noise_name, condition.budget_fraction)),
            )
            two_sided = (
                np.percentile(bootstrap, [2.5, 97.5]) if len(bootstrap) else [math.nan, math.nan]
            )
            # A one-sided 95% lower bound uses the 5th bootstrap percentile.
            one_sided_lower = float(np.percentile(bootstrap, 5.0)) if len(bootstrap) else math.nan
            failures = int(np.sum(differences < -epsilon))
            failure_ci = _wilson_interval(failures, len(differences))
            failure_values = differences[differences < -epsilon]
            worst_position = int(np.argmin(differences)) if len(differences) else -1
            rows.append(
                {
                    "dataset": dataset,
                    "noise": condition.noise_name,
                    "budget_fraction": float(condition.budget_fraction),
                    "n_seeds": len(differences),
                    "safe_delta_wga_mean": float(paired.groupby("seed")["delta_wga_safe"].mean().mean()),
                    "base_delta_wga_mean": float(paired.groupby("seed")["delta_wga_base"].mean().mean()),
                    "difference_mean": float(np.mean(differences)),
                    "difference_ci95_low": two_sided[0],
                    "difference_ci95_high": two_sided[1],
                    "difference_one_sided95_low": one_sided_lower,
                    "non_inferior": bool(np.isfinite(one_sided_lower) and one_sided_lower > -epsilon),
                    "noninferiority_permutation_p": _sign_flip_pvalue(differences, -epsilon, "greater"),
                    "failure_rate": failures / len(differences) if len(differences) else math.nan,
                    "failure_rate_ci95_low": failure_ci[0],
                    "failure_rate_ci95_high": failure_ci[1],
                    "max_allowed_failure_rate": max_failure_rate,
                    "failure_rate_acceptable": bool(
                        np.isfinite(failure_ci[1]) and failure_ci[1] < max_failure_rate
                    ),
                    "mean_conditional_harm": float(np.mean(-failure_values)) if len(failure_values) else 0.0,
                    "mean_harm_beyond_epsilon": float(np.mean(-epsilon - failure_values)) if len(failure_values) else 0.0,
                    "cvar10_difference": _lower_tail_cvar(differences),
                    "worst_seed": int(seed_differences.iloc[worst_position]["seed"]) if worst_position >= 0 else "",
                    "worst_seed_difference": float(np.min(differences)) if len(differences) else math.nan,
                    "best_seed_difference": float(np.max(differences)) if len(differences) else math.nan,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["noninferiority_p_holm"] = _holm_adjust(
            result["noninferiority_permutation_p"].tolist()
        )
        result["non_inferior_holm"] = result["non_inferior"] & (
            result["noninferiority_p_holm"] < 0.05
        )
        result["safety_pass"] = (
            result["non_inferior_holm"] & result["failure_rate_acceptable"]
        )
    result.to_csv(root / "exp4a_safety.csv", index=False)

    if paired_all:
        all_pairs = pd.concat(paired_all, ignore_index=True)
        seed_level = (
            all_pairs.groupby(["dataset", "noise_name", "seed"], as_index=False)["difference"].mean()
        )
        overall_rows = []
        for (dataset, noise), frame in seed_level.groupby(["dataset", "noise_name"]):
            worst = frame.loc[frame["difference"].idxmin()]
            conditions = result[(result["dataset"] == dataset) & (result["noise"] == noise)]
            worst_condition = conditions.loc[conditions["difference_mean"].idxmin()]
            failures = all_pairs[
                (all_pairs["dataset"] == dataset)
                & (all_pairs["noise_name"] == noise)
                & (all_pairs["difference"] < -epsilon)
            ]["difference"]
            overall_rows.append(
                {
                    "dataset": dataset,
                    "noise": noise,
                    "n_seed_budget_pairs": int(
                        len(all_pairs[(all_pairs["dataset"] == dataset) & (all_pairs["noise_name"] == noise)])
                    ),
                    "worst_seed": int(worst["seed"]),
                    "worst_seed_mean_difference": float(worst["difference"]),
                    "worst_condition_budget": float(worst_condition["budget_fraction"]),
                    "worst_condition_mean_difference": float(worst_condition["difference_mean"]),
                    "mean_conditional_harm": float((-failures).mean()) if len(failures) else 0.0,
                    "conditions_non_inferior": int(conditions["non_inferior"].sum()),
                    "conditions_total": len(conditions),
                }
            )
        pd.DataFrame(overall_rows).to_csv(root / "exp4a_overall.csv", index=False)
    return result


# ---------------------------------------------------------------------------
# Experiment 4B: held-out safe/fallback re-run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafeFallbackPolicy:
    """Locked, validation-only fallback decision thresholds."""

    min_worst_group_accuracy: float = 0.60
    max_validation_group_gap: float = 0.25
    min_validation_group_size: int = 20
    primary_selector: str = "class_balanced_high_loss"
    fallback_selector: str = "class_loss_margin_balanced_high_loss"
    fallback_loss_bins: int = 4
    fallback_margin_bins: int = 4


def _top_score_query(scores: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0 or budget > len(scores):
        raise ValueError(f"invalid query budget {budget}")
    index = np.arange(len(scores))
    order = np.lexsort((index, -np.asarray(scores, dtype=float)))
    return np.sort(order[:budget].astype(np.int64))


def _equal_capped_quota(capacities: Mapping[Any, int], total: int) -> dict[Any, int]:
    capacities = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    if total > sum(capacities.values()):
        raise ValueError("balanced selector budget exceeds pool")
    quota = {key: 0 for key in capacities}
    remaining = total
    active = sorted(capacities, key=str)
    while remaining and active:
        share = max(1, remaining // len(active))
        progressed = 0
        for key in list(active):
            take = min(share, capacities[key] - quota[key], remaining)
            quota[key] += take
            remaining -= take
            progressed += take
            if remaining == 0:
                break
        active = [key for key in active if quota[key] < capacities[key]]
        if progressed == 0:
            break
    if remaining:
        raise ValueError(f"could not fill balanced selector remainder {remaining}")
    return quota


def _balanced_high_score_query(scores: np.ndarray, strata: Sequence[Any], budget: int) -> np.ndarray:
    pools: defaultdict[Any, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        pools[stratum].append(index)
    quota = _equal_capped_quota({key: len(value) for key, value in pools.items()}, budget)
    selected: list[int] = []
    for key in sorted(pools, key=str):
        pool = np.asarray(pools[key], dtype=np.int64)
        order = np.lexsort((pool, -np.asarray(scores)[pool]))
        selected.extend(pool[order[: quota[key]]].tolist())
    result = np.sort(np.asarray(selected, dtype=np.int64))
    if len(result) != budget or len(np.unique(result)) != budget:
        raise AssertionError("balanced selector did not return exact budget")
    return result


def _fallback_decision(
    validation: Mapping[str, Any], policy: SafeFallbackPolicy
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if validation["worst_group_accuracy"] < policy.min_worst_group_accuracy:
        reasons.append("worst_group_accuracy_below_threshold")
    if validation["group_gap"] > policy.max_validation_group_gap:
        reasons.append("validation_group_gap_above_threshold")
    if min(validation["group_sizes"].values()) < policy.min_validation_group_size:
        reasons.append("validation_group_too_small")
    return bool(reasons), reasons


EXP4B_FIELDS = [
    "dataset", "noise", "budget_fraction", "budget_n", "seed", "run_seed", "valid", "status", "error",
    "fallback_triggered", "fallback_reasons", "policy_sha256", "base_query_sha256", "safe_query_sha256",
    "query_overlap_fraction", "base_query_cost", "safe_query_cost", "baseline_wga", "base_wga", "safe_wga",
    "baseline_accuracy", "base_accuracy", "safe_accuracy", "baseline_group_accuracy", "base_group_accuracy",
    "safe_group_accuracy", "delta_wga_base", "delta_wga_safe", "safe_minus_base", "epsilon",
    "epsilon_failure", "validation_worst_group_accuracy", "validation_group_gap", "validation_group_accuracies",
    "decision_seconds", "training_and_evaluation_seconds", "runtime_seconds",
]


def run_exp4b(
    output_dirs: Mapping[str, str] | Sequence[tuple[str, str]],
    output_root: str | Path = "outputs/exp4_complete",
    noises: Sequence[str] = DEFAULT_NOISES,
    budgets: Sequence[float] = DEFAULT_EXP4_BUDGETS,
    heldout_seeds: Sequence[int] = tuple(range(5, 10)),
    tuning_seeds: Sequence[int] = tuple(range(5)),
    policy: SafeFallbackPolicy | None = None,
    epsilon: float = 0.01,
    training_cfg: Mapping[str, Any] | None = None,
    n_bootstrap: int = 10_000,
    resume: bool = True,
    max_failure_rate: float = 0.10,
) -> pd.DataFrame:
    """Re-run base and locked safe/fallback policies on disjoint held-out seeds."""
    if set(map(int, heldout_seeds)) & set(map(int, tuning_seeds)):
        raise ValueError("heldout_seeds and tuning_seeds must be disjoint")
    datasets = dict(output_dirs)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(default_training_config() if training_cfg is None else training_cfg)
    policy = SafeFallbackPolicy() if policy is None else policy
    policy_payload = asdict(policy)
    policy_sha = hashlib.sha256(_json(policy_payload).encode("utf-8")).hexdigest()
    run_config = {
        "experiment": "exp4b",
        "datasets": datasets,
        "noises": list(noises),
        "budgets": list(map(float, budgets)),
        "heldout_seeds": list(map(int, heldout_seeds)),
        "tuning_seeds_not_evaluated": list(map(int, tuning_seeds)),
        "policy": policy_payload,
        "policy_sha256": policy_sha,
        "epsilon": float(epsilon),
        "max_failure_rate": float(max_failure_rate),
        "training": cfg,
        "code_sha256": _code_sha256(),
        "git_revision": _git_revision(),
    }
    _locked_config(root / "exp4b_config.json", run_config, resume)
    journal = CsvJournal(
        root / "exp4b_runs.csv",
        EXP4B_FIELDS,
        ["dataset", "noise", "budget_fraction", "seed"],
    )
    query_path = root / "exp4b_queries.jsonl.gz"

    for dataset, output_dir in datasets.items():
        for noise in noises:
            for seed in heldout_seeds:
                try:
                    data = load_data(output_dir, noise, int(seed))
                except BaseException as load_error:
                    for budget_fraction in budgets:
                        row = {
                            "dataset": dataset, "noise": noise,
                            "budget_fraction": float(budget_fraction), "budget_n": 0,
                            "seed": int(seed), "run_seed": _stable_seed("exp4b", dataset, noise, budget_fraction, seed),
                            "valid": False, "status": "load_error", "error": str(load_error)[:2000],
                            "policy_sha256": policy_sha, "epsilon": epsilon,
                        }
                        journal.append(row)
                    continue

                private = data["private"]
                clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
                for budget_fraction in budgets:
                    budget_n = max(1, int(round(float(budget_fraction) * len(clean_labels))))
                    run_seed = _stable_seed("exp4b", dataset, noise, budget_fraction, seed)
                    key = {
                        "dataset": dataset,
                        "noise": noise,
                        "budget_fraction": float(budget_fraction),
                        "seed": int(seed),
                    }
                    if journal.has(key):
                        continue
                    started = time.perf_counter()
                    try:
                        # Fit the noisy baseline and make the safety decision using
                        # clean validation only.  No test prediction has occurred yet.
                        baseline_fit, validation = validation_group_diagnostics(
                            data, cfg, run_seed
                        )
                        fallback, reasons = _fallback_decision(validation, policy)
                        base_query = _top_score_query(data["loss"], budget_n)
                        if fallback:
                            loss_bins = _quantile_bins(data["loss"], policy.fallback_loss_bins)
                            margin_bins = _quantile_bins(data["margin"], policy.fallback_margin_bins)
                            strata = [
                                (int(data["noisy_labels"][i]), int(loss_bins[i]), int(margin_bins[i]))
                                for i in range(len(clean_labels))
                            ]
                        else:
                            strata = [int(value) for value in data["noisy_labels"]]
                        safe_query = _balanced_high_score_query(data["loss"], strata, budget_n)
                        decision_seconds = time.perf_counter() - started

                        # The decision is now frozen.  Test is used only for these
                        # final baseline/base/safe evaluations.
                        baseline_predictions = _predict_model(
                            baseline_fit,
                            data["test_feat"],
                            data["train_feat"].shape[1],
                            _class_count(data["noisy_labels"], data["val_labels"]),
                            cfg,
                        )
                        baseline = _evaluation_metrics(
                            data["evaluator"], data["test_ids"], baseline_predictions
                        )
                        base = _train_and_eval(
                            data["train_feat"], _repair(data["noisy_labels"], clean_labels, base_query),
                            data["val_feat"], data["val_labels"], data["test_feat"], data["test_ids"],
                            data["evaluator"], cfg, run_seed,
                        )
                        safe = _train_and_eval(
                            data["train_feat"], _repair(data["noisy_labels"], clean_labels, safe_query),
                            data["val_feat"], data["val_labels"], data["test_feat"], data["test_ids"],
                            data["evaluator"], cfg, run_seed,
                        )
                        safe_minus_base = safe["wga"] - base["wga"]
                        row = {
                            **key,
                            "budget_n": budget_n,
                            "run_seed": run_seed,
                            "valid": True,
                            "status": "complete",
                            "error": "",
                            "fallback_triggered": fallback,
                            "fallback_reasons": _json(reasons),
                            "policy_sha256": policy_sha,
                            "base_query_sha256": _query_digest(base_query),
                            "safe_query_sha256": _query_digest(safe_query),
                            "query_overlap_fraction": len(set(base_query) & set(safe_query)) / budget_n,
                            "base_query_cost": len(base_query),
                            "safe_query_cost": len(safe_query),
                            "baseline_wga": baseline["wga"],
                            "base_wga": base["wga"],
                            "safe_wga": safe["wga"],
                            "baseline_accuracy": baseline["accuracy"],
                            "base_accuracy": base["accuracy"],
                            "safe_accuracy": safe["accuracy"],
                            "baseline_group_accuracy": _json(baseline["group_accuracy"]),
                            "base_group_accuracy": _json(base["group_accuracy"]),
                            "safe_group_accuracy": _json(safe["group_accuracy"]),
                            "delta_wga_base": base["wga"] - baseline["wga"],
                            "delta_wga_safe": safe["wga"] - baseline["wga"],
                            "safe_minus_base": safe_minus_base,
                            "epsilon": epsilon,
                            "epsilon_failure": safe_minus_base < -epsilon,
                            "validation_worst_group_accuracy": validation["worst_group_accuracy"],
                            "validation_group_gap": validation["group_gap"],
                            "validation_group_accuracies": _json(validation["group_accuracies"]),
                            "decision_seconds": decision_seconds,
                            "training_and_evaluation_seconds": (
                                time.perf_counter() - started - decision_seconds
                            ),
                            "runtime_seconds": time.perf_counter() - started,
                        }
                        journal.append(row)
                        _append_gzip_jsonl(
                            query_path,
                            {
                                **key,
                                "run_seed": run_seed,
                                "base_train_ids": data["train_ids"][base_query].tolist(),
                                "safe_train_ids": data["train_ids"][safe_query].tolist(),
                            },
                        )
                    except BaseException as error:
                        journal.append(
                            {
                                **key,
                                "budget_n": budget_n,
                                "run_seed": run_seed,
                                "valid": False,
                                "status": "runtime_error",
                                "error": f"{error}\n{traceback.format_exc(limit=5)}"[:2000],
                                "policy_sha256": policy_sha,
                                "epsilon": epsilon,
                                "runtime_seconds": time.perf_counter() - started,
                            }
                        )

    runs = pd.read_csv(root / "exp4b_runs.csv")
    _write_exp4b_summary(runs, root, epsilon, n_bootstrap, max_failure_rate)
    _write_exp4b_integrity(
        runs, root, datasets, noises, budgets, heldout_seeds
    )
    return runs


def _write_exp4b_summary(
    runs: pd.DataFrame,
    root: Path,
    epsilon: float,
    n_bootstrap: int,
    max_failure_rate: float,
) -> None:
    valid = runs[_as_bool(runs["valid"])].copy()
    rows = []
    for condition, frame in valid.groupby(["dataset", "noise", "budget_fraction"]):
        seed_values = (
            frame.assign(safe_minus_base=pd.to_numeric(frame["safe_minus_base"], errors="coerce"))
            .groupby("seed")["safe_minus_base"]
            .mean()
            .dropna()
        )
        values = seed_values.to_numpy(float)
        bootstrap = _bootstrap_seed_mean(
            values, n_bootstrap, np.random.default_rng(_stable_seed("exp4b-summary", *condition))
        )
        lower = float(np.percentile(bootstrap, 5)) if len(bootstrap) else math.nan
        failures = int(np.sum(values < -epsilon))
        failure_ci = _wilson_interval(failures, len(values))
        rows.append(
            {
                "dataset": condition[0], "noise": condition[1],
                "budget_fraction": condition[2], "n_heldout_seeds": len(values),
                "safe_minus_base_mean": float(np.mean(values)) if len(values) else math.nan,
                "safe_minus_base_one_sided95_low": lower,
                "non_inferior": bool(np.isfinite(lower) and lower > -epsilon),
                "noninferiority_permutation_p": _sign_flip_pvalue(values, -epsilon, "greater"),
                "failure_rate": failures / len(values) if len(values) else math.nan,
                "failure_rate_ci95_low": failure_ci[0],
                "failure_rate_ci95_high": failure_ci[1],
                "max_allowed_failure_rate": max_failure_rate,
                "failure_rate_acceptable": bool(
                    np.isfinite(failure_ci[1]) and failure_ci[1] < max_failure_rate
                ),
                "cvar10_difference": _lower_tail_cvar(values),
                "fallback_rate": float(_as_bool(frame["fallback_triggered"]).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["noninferiority_p_holm"] = _holm_adjust(
            summary["noninferiority_permutation_p"].tolist()
        )
        summary["non_inferior_holm"] = summary["non_inferior"] & (
            summary["noninferiority_p_holm"] < 0.05
        )
        summary["safety_pass"] = (
            summary["non_inferior_holm"] & summary["failure_rate_acceptable"]
        )
    summary.to_csv(root / "exp4b_safety_summary.csv", index=False)


def _write_exp4b_integrity(
    runs: pd.DataFrame,
    root: Path,
    datasets: Mapping[str, str],
    noises: Sequence[str],
    budgets: Sequence[float],
    heldout_seeds: Sequence[int],
) -> None:
    """Materialize the expected held-out matrix, including failed/missing cells."""
    rows = []
    numeric_budget = pd.to_numeric(runs["budget_fraction"], errors="coerce")
    numeric_seed = pd.to_numeric(runs["seed"], errors="coerce")
    for dataset, noise, budget, seed in itertools.product(
        datasets, noises, budgets, heldout_seeds
    ):
        cell = runs[
            (runs["dataset"] == dataset)
            & (runs["noise"] == noise)
            & np.isclose(numeric_budget, float(budget))
            & (numeric_seed == int(seed))
        ]
        if len(cell) == 0:
            status = "missing"
        elif len(cell) > 1:
            status = "duplicate"
        elif bool(_as_bool(cell["valid"]).iloc[0]):
            status = "complete"
        else:
            status = str(cell["status"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "noise": noise,
                "budget_fraction": float(budget),
                "seed": int(seed),
                "observed_rows": len(cell),
                "status": status,
            }
        )
    pd.DataFrame(rows).to_csv(root / "exp4b_integrity.csv", index=False)


# ---------------------------------------------------------------------------
# Synthetic tests and CLI
# ---------------------------------------------------------------------------


def run_self_tests() -> None:
    rng = np.random.default_rng(20260714)
    n = 5_000
    groups = rng.integers(0, 4, size=n)
    labels = rng.integers(0, 2, size=n)
    loss = rng.gamma(2.0, 1.0, size=n)
    margin = rng.beta(2.0, 4.0, size=n)
    is_noisy = rng.random(n) < 0.45
    high, low, diagnostics = construct_exact_matched_pair(
        None,
        is_noisy,
        groups,
        labels,
        loss,
        margin,
        budget=500,
        precision=0.40,
        target_group=0,
        high_frac=0.60,
        low_frac=0.20,
        n_loss_bins=4,
        n_margin_bins=4,
        rng=np.random.default_rng(8),
    )
    assert len(high) == len(low) == 500
    assert int(is_noisy[high].sum()) == int(is_noisy[low].sum()) == 200
    assert diagnostics["target_noisy_high"] == 120
    assert diagnostics["target_noisy_low"] == 40
    assert diagnostics["strata_exact"]
    assert set(high[~is_noisy[high]]) == set(low[~is_noisy[low]])

    try:
        construct_exact_matched_pair(
            None, is_noisy, np.ones(n), labels, loss, margin, 500, 0.4, 0, 0.6, 0.2,
            rng=np.random.default_rng(9),
        )
    except MatchingInfeasible:
        pass
    else:
        raise AssertionError("infeasible match was not rejected")
    print("Self-tests passed: exact budgets, exact noisy counts, exact strata, contrast, infeasibility.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "exp2", "exp4a", "exp4b", "self-test"], default="all")
    parser.add_argument("--dataset", action="append", help="NAME=OUTPUT_DIR; repeat for datasets")
    parser.add_argument("--results", action="append", help="NAME=RESULTS_CSV for Exp4A")
    parser.add_argument("--output-root", default="outputs/exp2_exp4_complete")
    parser.add_argument("--noises", nargs="+", default=list(DEFAULT_NOISES))
    parser.add_argument("--exp2-budgets", nargs="+", type=float, default=list(DEFAULT_EXP2_BUDGETS))
    parser.add_argument("--exp4-budgets", nargs="+", type=float, default=list(DEFAULT_EXP4_BUDGETS))
    parser.add_argument("--exp2-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--exp2-pairs", type=int, default=30)
    parser.add_argument("--precision", type=float, default=0.40)
    parser.add_argument("--high-frac", type=float, default=0.40)
    parser.add_argument("--low-frac", type=float, default=0.05)
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=list(range(5, 10)))
    parser.add_argument("--tuning-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-failure-rate", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--base-method", default="noise_score")
    parser.add_argument("--safe-method", default="noise_score_cpba_tc_v8_fallback")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Explicitly reduce to one dataset/noise/budget/seed and two pairs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "self-test":
        run_self_tests()
        return 0
    datasets = _parse_mapping(args.dataset, DEFAULT_DATASETS)
    default_results = {name: str(Path(path) / "stage1" / "results.csv") for name, path in datasets.items()}
    results = _parse_mapping(args.results, default_results)
    noises = args.noises
    exp2_budgets = args.exp2_budgets
    exp4_budgets = args.exp4_budgets
    exp2_seeds = args.exp2_seeds
    heldout_seeds = args.heldout_seeds
    n_pairs = args.exp2_pairs
    if args.smoke:
        first = next(iter(datasets))
        datasets = {first: datasets[first]}
        results = {first: results[first]}
        noises = noises[:1]
        exp2_budgets = exp2_budgets[:1]
        exp4_budgets = exp4_budgets[:1]
        exp2_seeds = exp2_seeds[:1]
        heldout_seeds = heldout_seeds[:1]
        n_pairs = 2

    root = Path(args.output_root)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv if argv is None else argv),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else "see scipy package",
        "code_sha256": _code_sha256(),
        "git_revision": _git_revision(),
    }
    _atomic_json(root / "run_metadata.json", metadata)
    if args.mode in {"all", "exp2"}:
        run_exp2_grid(
            datasets,
            root / "exp2",
            noises,
            exp2_budgets,
            exp2_seeds,
            n_pairs,
            args.precision,
            args.high_frac,
            args.low_frac,
            n_bootstrap=args.bootstrap,
            resume=not args.no_resume,
        )
    if args.mode in {"all", "exp4a"}:
        run_exp4a_full(
            results,
            root / "exp4",
            args.base_method,
            args.safe_method,
            args.epsilon,
            args.bootstrap,
            heldout_seeds=heldout_seeds,
            max_failure_rate=args.max_failure_rate,
            resume=not args.no_resume,
        )
    if args.mode in {"all", "exp4b"}:
        run_exp4b(
            datasets,
            root / "exp4",
            noises,
            exp4_budgets,
            heldout_seeds,
            args.tuning_seeds,
            epsilon=args.epsilon,
            n_bootstrap=args.bootstrap,
            resume=not args.no_resume,
            max_failure_rate=args.max_failure_rate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
