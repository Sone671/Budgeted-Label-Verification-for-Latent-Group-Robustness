"""Self-contained coverage-selection utilities used by adaptive rankings."""

from __future__ import annotations

from collections import deque
from math import ceil
from typing import Sequence

import numpy as np
import pandas as pd


def _scores(values: Sequence[float] | np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or not np.isfinite(out).all():
        raise ValueError("scores must be a finite one-dimensional array")
    return out


def _buckets(values: Sequence | np.ndarray, n: int) -> np.ndarray:
    out = np.asarray(values)
    if out.ndim != 1 or len(out) != n:
        raise ValueError("buckets must be one-dimensional and align with scores")
    return out


def _stable_order(scores: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    if indices is None:
        indices = np.arange(len(scores), dtype=int)
    indices = np.asarray(indices, dtype=int)
    return indices[np.lexsort((indices, -scores[indices]))]


def coverage_aware_ranking(
    scores: Sequence[float] | np.ndarray,
    buckets: Sequence | np.ndarray,
    *,
    bucket_order: str = "size_asc",
    warm_start_per_bucket: int = 1,
    max_per_bucket: int | None = None,
    fill_remaining_by_score: bool = True,
) -> np.ndarray:
    """Round-robin coverage prefix followed by a stable score-sorted tail."""
    score_array = _scores(scores)
    bucket_array = _buckets(buckets, len(score_array))
    if warm_start_per_bucket < 0:
        raise ValueError("warm_start_per_bucket must be non-negative")
    if max_per_bucket is not None and max_per_bucket <= 0:
        raise ValueError("max_per_bucket must be positive")

    counts = pd.Series(bucket_array).value_counts(sort=False)
    if bucket_order == "size_asc":
        ordered_buckets = sorted(counts.index.tolist(), key=lambda b: (counts[b], str(b)))
    elif bucket_order == "size_desc":
        ordered_buckets = sorted(counts.index.tolist(), key=lambda b: (-counts[b], str(b)))
    elif bucket_order == "appearance":
        ordered_buckets = list(pd.unique(bucket_array))
    else:
        raise ValueError(f"unsupported bucket_order: {bucket_order!r}")

    all_indices = np.arange(len(score_array), dtype=int)
    queues = {
        bucket: deque(_stable_order(score_array, all_indices[bucket_array == bucket]).tolist())
        for bucket in ordered_buckets
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_per_bucket = {bucket: 0 for bucket in ordered_buckets}
    for _ in range(warm_start_per_bucket):
        for bucket in ordered_buckets:
            if not queues[bucket]:
                continue
            if max_per_bucket is not None and selected_per_bucket[bucket] >= max_per_bucket:
                continue
            index = int(queues[bucket].popleft())
            selected.append(index)
            selected_set.add(index)
            selected_per_bucket[bucket] += 1
    if fill_remaining_by_score:
        selected.extend(int(index) for index in _stable_order(score_array) if int(index) not in selected_set)
    return np.asarray(selected, dtype=np.int64)


def _coverage_metrics(buckets: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(buckets[selected], return_counts=True)
    del values
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    total = max(1, len(np.unique(buckets)))
    denominator = max(1, min(total, len(selected)))
    return {
        "largest_bucket_share": float(counts.max() / len(selected)),
        "effective_bucket_ratio": float(np.exp(entropy) / denominator),
    }


def adaptive_coverage_ratio(*, budget_count: int, n_samples: int, schedule: str = "default") -> float:
    if budget_count <= 0 or n_samples <= 0:
        raise ValueError("budget_count and n_samples must be positive")
    fraction = min(float(budget_count) / float(n_samples), 1.0)
    if schedule != "default":
        raise ValueError(f"unsupported coverage schedule: {schedule!r}")
    if fraction <= 0.01:
        return 0.10
    if fraction <= 0.02:
        return 0.35
    if fraction <= 0.05:
        return 0.70
    return 0.90


def budget_adaptive_coverage_ranking(
    scores: Sequence[float] | np.ndarray,
    buckets: Sequence | np.ndarray,
    *,
    budget_count: int,
    coverage_ratio: float | None = None,
    ratio_schedule: str = "default",
    bucket_order: str = "size_asc",
    return_metadata: bool = False,
    **_: object,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int]]:
    score_array = _scores(scores)
    bucket_array = _buckets(buckets, len(score_array))
    clipped_budget = min(max(1, int(budget_count)), len(score_array))
    ratio = (
        adaptive_coverage_ratio(
            budget_count=clipped_budget, n_samples=len(score_array), schedule=ratio_schedule
        )
        if coverage_ratio is None
        else float(coverage_ratio)
    )
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("coverage_ratio must lie in [0, 1]")
    num_buckets = max(1, len(np.unique(bucket_array)))
    warm_start = int(ceil(ratio * clipped_budget / num_buckets))
    ranking = coverage_aware_ranking(
        score_array,
        bucket_array,
        bucket_order=bucket_order,
        warm_start_per_bucket=warm_start,
    )
    if return_metadata:
        return ranking, {
            "budget_count": clipped_budget,
            "num_buckets": num_buckets,
            "coverage_ratio": ratio,
            "warm_start_per_bucket": warm_start,
            "coverage_prefix_target": ratio * clipped_budget,
            "coverage_prefix_capacity": warm_start * num_buckets,
        }
    return ranking


def budget_hybrid_coverage_ranking(
    scores: Sequence[float] | np.ndarray,
    buckets: Sequence | np.ndarray,
    *,
    budget_count: int,
    switch_budget_fraction: float = 0.02,
    coverage_ratio: float | None = None,
    ratio_schedule: str = "default",
    bucket_order: str = "size_asc",
    return_metadata: bool = False,
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    score_array = _scores(scores)
    clipped_budget = min(max(1, int(budget_count)), len(score_array))
    threshold = int(round(float(switch_budget_fraction) * len(score_array)))
    if clipped_budget <= threshold:
        ranking = _stable_order(score_array)
        metadata: dict[str, object] = {"method_decision": "score", "hybrid_reason": "small_budget"}
    else:
        ranking, adaptive = budget_adaptive_coverage_ranking(
            score_array,
            buckets,
            budget_count=clipped_budget,
            coverage_ratio=coverage_ratio,
            ratio_schedule=ratio_schedule,
            bucket_order=bucket_order,
            return_metadata=True,
            **kwargs,
        )
        metadata = {"method_decision": "adaptive", "hybrid_reason": "large_budget", **adaptive}
    if return_metadata:
        return ranking, metadata
    return ranking


def gated_adaptive_coverage_ranking(
    scores: Sequence[float] | np.ndarray,
    buckets: Sequence | np.ndarray,
    *,
    budget_count: int,
    small_budget_fraction: float = 0.02,
    max_largest_bucket_share: float = 0.25,
    min_effective_bucket_ratio: float = 0.35,
    return_metadata: bool = False,
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    score_array = _scores(scores)
    bucket_array = _buckets(buckets, len(score_array))
    clipped_budget = min(max(1, int(budget_count)), len(score_array))
    fraction = clipped_budget / len(score_array)
    pure = _stable_order(score_array)
    metrics = _coverage_metrics(bucket_array, pure[:clipped_budget])
    use_adaptive = fraction > small_budget_fraction and (
        metrics["largest_bucket_share"] >= max_largest_bucket_share
        or metrics["effective_bucket_ratio"] <= min_effective_bucket_ratio
    )
    if use_adaptive:
        ranking, adaptive = budget_adaptive_coverage_ranking(
            score_array, bucket_array, budget_count=clipped_budget, return_metadata=True, **kwargs
        )
        metadata: dict[str, object] = {"method_decision": "adaptive", **metrics, **adaptive}
    else:
        ranking = pure
        metadata = {"method_decision": "score", **metrics}
    if return_metadata:
        return ranking, metadata
    return ranking
