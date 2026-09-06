"""Construct auditable matched query sets for diagnostic interventions.

This module is deliberately separate from acquisition methods.  It uses the
private manifest to construct an *oracle diagnostic control*: high and low
query sets have identical noise counts and identical
``(clean label, loss bin, margin bin)`` counts.  They differ only in the
planned number of noisy examples from a designated target group.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


class MatchConstructionError(RuntimeError):
    """Raised when a frozen matched-pair condition cannot be constructed."""


@dataclass(frozen=True)
class MatchedPair:
    """One high/low pair, expressed as positions in an aligned train frame."""

    high_positions: np.ndarray
    low_positions: np.ndarray
    diagnostics: dict[str, float | int]


def _round_count(value: float) -> int:
    return int(math.floor(value + 0.5))


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return deterministic quantile bins while handling ties and constants."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("matching values must be finite.")
    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, array, side="right").astype(np.int64)


def matching_strata(
    private_frame: pd.DataFrame,
    probabilities: np.ndarray,
    noisy_labels: np.ndarray,
    *,
    n_loss_bins: int = 5,
    n_margin_bins: int = 5,
) -> list[tuple[int, int, int]]:
    """Build the predeclared matching covariates for every train position."""

    required = {"clean_label", "is_noisy", "group"}
    missing = sorted(required.difference(private_frame.columns))
    if missing:
        raise ValueError(f"private manifest lacks matching fields: {missing}")

    probabilities = np.asarray(probabilities, dtype=np.float64)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(private_frame):
        raise ValueError("probabilities must align with the private train manifest.")
    if len(noisy_labels) != len(private_frame):
        raise ValueError("noisy_labels must align with the private train manifest.")
    if noisy_labels.min(initial=0) < 0 or noisy_labels.max(initial=0) >= probabilities.shape[1]:
        raise ValueError("noisy labels are outside the probability columns.")

    own_probability = probabilities[np.arange(len(probabilities)), noisy_labels]
    loss = -np.log(np.clip(own_probability, 1e-12, 1.0))
    alternatives = probabilities.copy()
    alternatives[np.arange(len(probabilities)), noisy_labels] = -np.inf
    margin = own_probability - alternatives.max(axis=1)
    loss_bins = _quantile_bins(loss, n_loss_bins)
    margin_bins = _quantile_bins(margin, n_margin_bins)
    clean_labels = private_frame["clean_label"].to_numpy(dtype=np.int64)
    return [
        (int(clean_labels[index]), int(loss_bins[index]), int(margin_bins[index]))
        for index in range(len(private_frame))
    ]


def _available_choice(
    candidates: Iterable[int],
    count: int,
    used: set[int],
    rng: np.random.Generator,
    *,
    label: str,
) -> np.ndarray:
    available = np.asarray([index for index in candidates if index not in used], dtype=np.int64)
    if len(available) < count:
        raise MatchConstructionError(
            f"insufficient {label}: need {count}, have {len(available)} unused candidates"
        )
    if count == 0:
        return np.empty(0, dtype=np.int64)
    selected = rng.choice(available, size=count, replace=False).astype(np.int64)
    used.update(map(int, selected))
    return selected


def _select_swaps(
    target_by_stratum: dict[tuple[int, int, int], list[int]],
    other_by_stratum: dict[tuple[int, int, int], list[int]],
    count: int,
    used: set[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select target/non-target noisy samples paired within the same stratum."""

    high: list[int] = []
    low: list[int] = []
    for _ in range(count):
        viable = [
            key
            for key in target_by_stratum
            if key in other_by_stratum
            and any(index not in used for index in target_by_stratum[key])
            and any(index not in used for index in other_by_stratum[key])
        ]
        if not viable:
            raise MatchConstructionError(
                f"insufficient within-stratum target/non-target noisy swaps: need {count}, built {len(high)}"
            )
        key = viable[int(rng.integers(len(viable)))]
        high_sample = _available_choice(
            target_by_stratum[key], 1, used, rng, label="target noisy swap"
        )
        low_sample = _available_choice(
            other_by_stratum[key], 1, used, rng, label="non-target noisy swap"
        )
        high.append(int(high_sample[0]))
        low.append(int(low_sample[0]))
    return np.asarray(high, dtype=np.int64), np.asarray(low, dtype=np.int64)


def validate_matched_pair(
    private_frame: pd.DataFrame,
    strata: list[tuple[int, int, int]],
    high_positions: np.ndarray,
    low_positions: np.ndarray,
    *,
    target_group: int,
    expected_noisy_count: int,
    expected_high_target_count: int,
    expected_low_target_count: int,
) -> dict[str, float | int]:
    """Validate the exact covariate balance required by the protocol."""

    high = np.asarray(high_positions, dtype=np.int64)
    low = np.asarray(low_positions, dtype=np.int64)
    if len(high) != len(low) or not len(high):
        raise MatchConstructionError("matched query sets must have the same non-zero size")
    if len(np.unique(high)) != len(high) or len(np.unique(low)) != len(low):
        raise MatchConstructionError("a query set contains duplicate positions")
    if high.min() < 0 or low.min() < 0 or high.max() >= len(private_frame) or low.max() >= len(private_frame):
        raise MatchConstructionError("query positions are outside the aligned train manifest")

    noisy = private_frame["is_noisy"].to_numpy(dtype=bool)
    groups = private_frame["group"].to_numpy(dtype=np.int64)
    high_noisy = int(noisy[high].sum())
    low_noisy = int(noisy[low].sum())
    high_target = int((noisy[high] & (groups[high] == target_group)).sum())
    low_target = int((noisy[low] & (groups[low] == target_group)).sum())
    if (high_noisy, low_noisy) != (expected_noisy_count, expected_noisy_count):
        raise MatchConstructionError("high/low oracle noise counts do not match the frozen target")
    if (high_target, low_target) != (expected_high_target_count, expected_low_target_count):
        raise MatchConstructionError("high/low target-group noisy counts do not match the frozen treatment")

    high_covariates = Counter(strata[int(index)] for index in high)
    low_covariates = Counter(strata[int(index)] for index in low)
    if high_covariates != low_covariates:
        raise MatchConstructionError("matched query sets differ in clean-label/loss/margin counts")

    overlap = len(set(map(int, high)).intersection(map(int, low)))
    return {
        "budget_count": int(len(high)),
        "oracle_noisy_count": high_noisy,
        "oracle_noise_precision": float(high_noisy / len(high)),
        "high_target_noisy_count": high_target,
        "low_target_noisy_count": low_target,
        "high_target_noisy_rate": float(high_target / max(high_noisy, 1)),
        "low_target_noisy_rate": float(low_target / max(low_noisy, 1)),
        "covariate_l1_difference": 0,
        "query_overlap_count": int(overlap),
        "query_overlap_rate": float(overlap / len(high)),
    }


def construct_matched_pairs(
    private_frame: pd.DataFrame,
    probabilities: np.ndarray,
    noisy_labels: np.ndarray,
    *,
    target_group: int,
    budget_count: int,
    oracle_noise_precision: float,
    high_target_noise_fraction: float,
    low_target_noise_fraction: float,
    n_pairs: int,
    seed: int,
    n_loss_bins: int = 5,
    n_margin_bins: int = 5,
) -> list[MatchedPair]:
    """Construct non-reused, exactly balanced high/low matched query pairs.

    Reuse is disallowed across different pairs.  Within a pair, clean examples
    and the non-treatment noisy examples are shared deliberately; this makes
    all measured covariates equal and isolates the target-group-noise dose.
    """

    if budget_count < 1 or n_pairs < 1:
        raise ValueError("budget_count and n_pairs must be positive.")
    for name, value in {
        "oracle_noise_precision": oracle_noise_precision,
        "high_target_noise_fraction": high_target_noise_fraction,
        "low_target_noise_fraction": low_target_noise_fraction,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    noisy_count = _round_count(budget_count * oracle_noise_precision)
    if not 0 < noisy_count < budget_count:
        raise MatchConstructionError(
            "requested precision must leave at least one noisy and one clean query example"
        )
    high_target_count = _round_count(noisy_count * high_target_noise_fraction)
    low_target_count = _round_count(noisy_count * low_target_noise_fraction)
    if not 0 <= low_target_count < high_target_count <= noisy_count:
        raise MatchConstructionError(
            "treatment fractions do not create a positive feasible high/low contrast"
        )

    strata = matching_strata(
        private_frame,
        probabilities,
        noisy_labels,
        n_loss_bins=n_loss_bins,
        n_margin_bins=n_margin_bins,
    )
    noisy = private_frame["is_noisy"].to_numpy(dtype=bool)
    groups = private_frame["group"].to_numpy(dtype=np.int64)
    target_by_stratum: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    other_by_stratum: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    target_noisy: list[int] = []
    other_noisy: list[int] = []
    clean: list[int] = []
    for index, key in enumerate(strata):
        if not noisy[index]:
            clean.append(index)
        elif groups[index] == target_group:
            target_by_stratum[key].append(index)
            target_noisy.append(index)
        else:
            other_by_stratum[key].append(index)
            other_noisy.append(index)

    rng = np.random.default_rng(seed)
    used: set[int] = set()
    pairs: list[MatchedPair] = []
    for pair_index in range(n_pairs):
        try:
            high_swaps, low_swaps = _select_swaps(
                target_by_stratum,
                other_by_stratum,
                high_target_count - low_target_count,
                used,
                rng,
            )
            shared_target = _available_choice(
                target_noisy,
                low_target_count,
                used,
                rng,
                label="shared target noisy examples",
            )
            shared_other = _available_choice(
                other_noisy,
                noisy_count - high_target_count,
                used,
                rng,
                label="shared non-target noisy examples",
            )
            shared_clean = _available_choice(
                clean,
                budget_count - noisy_count,
                used,
                rng,
                label="shared clean examples",
            )
        except MatchConstructionError as error:
            raise MatchConstructionError(f"pair {pair_index}: {error}") from error

        high = np.concatenate((shared_target, high_swaps, shared_other, shared_clean))
        low = np.concatenate((shared_target, low_swaps, shared_other, shared_clean))
        high = rng.permutation(high).astype(np.int64)
        low = rng.permutation(low).astype(np.int64)
        diagnostics = validate_matched_pair(
            private_frame,
            strata,
            high,
            low,
            target_group=target_group,
            expected_noisy_count=noisy_count,
            expected_high_target_count=high_target_count,
            expected_low_target_count=low_target_count,
        )
        diagnostics.update(
            {
                "pair_index": pair_index,
                "target_group": int(target_group),
                "n_loss_bins": int(n_loss_bins),
                "n_margin_bins": int(n_margin_bins),
            }
        )
        pairs.append(MatchedPair(high, low, diagnostics))
    return pairs
