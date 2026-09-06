from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import hypergeom

from robust_verify.utils import budget_to_count


@dataclass(frozen=True)
class AuditStratum:
    band: int
    observed_class: int
    indices: np.ndarray


@dataclass(frozen=True)
class ResidualPosterior:
    mean: float
    upper: float
    upper_width: float


def ranking_band_boundaries(n: int, edges: Sequence[float]) -> np.ndarray:
    if n < 1:
        raise ValueError("population size must be positive")
    values = np.asarray(tuple(float(value) for value in edges), dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two band edges are required")
    if not np.isclose(values[0], 0.0) or not np.isclose(values[-1], 1.0):
        raise ValueError("band edges must start at zero and end at one")
    if (np.diff(values) <= 0.0).any():
        raise ValueError("band edges must be strictly increasing")
    boundaries = np.asarray([budget_to_count(value, n) for value in values], dtype=np.int64)
    boundaries[0] = 0
    boundaries[-1] = n
    if (np.diff(boundaries) <= 0).any():
        raise ValueError("population is too small for the requested score bands")
    return boundaries


def build_band_class_strata(
    ranking: np.ndarray,
    observed_labels: np.ndarray,
    edges: Sequence[float],
) -> tuple[list[AuditStratum], np.ndarray]:
    order = np.asarray(ranking, dtype=np.int64).reshape(-1)
    labels = np.asarray(observed_labels, dtype=np.int64).reshape(-1)
    if len(order) != len(labels) or set(order.tolist()) != set(range(len(labels))):
        raise ValueError("ranking must be a permutation matching observed labels")
    boundaries = ranking_band_boundaries(len(order), edges)
    sample_band = np.empty(len(order), dtype=np.int64)
    for band, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        sample_band[order[start:stop]] = band
    strata: list[AuditStratum] = []
    for band in range(len(boundaries) - 1):
        in_band = sample_band == band
        for observed_class in sorted(np.unique(labels[in_band]).tolist()):
            indices = np.flatnonzero(in_band & (labels == observed_class)).astype(np.int64)
            if len(indices):
                strata.append(
                    AuditStratum(
                        band=int(band),
                        observed_class=int(observed_class),
                        indices=indices,
                    )
                )
    if sum(len(stratum.indices) for stratum in strata) != len(order):
        raise RuntimeError("audit strata must partition the full population")
    return strata, boundaries


def allocate_stratified_scout(stratum_sizes: np.ndarray, total: int) -> np.ndarray:
    sizes = np.asarray(stratum_sizes, dtype=np.int64).reshape(-1)
    if len(sizes) == 0 or (sizes <= 0).any():
        raise ValueError("stratum sizes must be positive")
    if total < len(sizes) or total > int(sizes.sum()):
        raise ValueError("scout total must cover every stratum and not exceed the population")
    allocation = np.ones(len(sizes), dtype=np.int64)
    remaining = int(total - len(sizes))
    capacity = sizes - allocation
    if remaining == 0:
        return allocation
    ideal = remaining * capacity / capacity.sum()
    additions = np.minimum(np.floor(ideal).astype(np.int64), capacity)
    allocation += additions
    remaining -= int(additions.sum())
    fractional = ideal - np.floor(ideal)
    while remaining:
        available = np.flatnonzero(allocation < sizes)
        if len(available) == 0:
            raise RuntimeError("unable to allocate the complete scout")
        order = available[
            np.lexsort((available, -capacity[available], -fractional[available]))
        ]
        take = min(remaining, len(order))
        allocation[order[:take]] += 1
        remaining -= take
    if int(allocation.sum()) != total or (allocation > sizes).any():
        raise RuntimeError("invalid stratified scout allocation")
    return allocation


def draw_stratified_scout(
    strata: Sequence[AuditStratum],
    total: int,
    *,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    sizes = np.asarray([len(stratum.indices) for stratum in strata], dtype=np.int64)
    allocation = allocate_stratified_scout(sizes, total)
    rng = np.random.default_rng(int(seed))
    selected = [
        rng.choice(stratum.indices, size=int(count), replace=False).astype(np.int64)
        for stratum, count in zip(strata, allocation)
    ]
    return selected, allocation


def hypergeometric_upper_error_count(
    *,
    population_size: int,
    sample_size: int,
    observed_errors: int,
    alpha: float,
) -> int:
    """Return an exact one-sided upper confidence bound for a finite error count.

    A simple random sample without replacement contains ``observed_errors``
    errors among ``sample_size`` records from a finite population of size
    ``population_size``.  The returned integer ``upper`` has coverage at least
    ``1 - alpha`` for the unknown total number of errors.  It inverts the
    lower tail of the hypergeometric law and makes no iid, beta-prior, or
    score-calibration assumption.
    """

    if isinstance(population_size, bool) or int(population_size) != population_size:
        raise ValueError("population_size must be an integer")
    if isinstance(sample_size, bool) or int(sample_size) != sample_size:
        raise ValueError("sample_size must be an integer")
    if isinstance(observed_errors, bool) or int(observed_errors) != observed_errors:
        raise ValueError("observed_errors must be an integer")
    population_size = int(population_size)
    sample_size = int(sample_size)
    observed_errors = int(observed_errors)
    if population_size < 1 or not 0 <= sample_size <= population_size:
        raise ValueError("sample_size must lie in [0, population_size]")
    if not 0 <= observed_errors <= sample_size:
        raise ValueError("observed_errors must lie in [0, sample_size]")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if sample_size == 0:
        return population_size
    if sample_size == population_size:
        return observed_errors

    # For a candidate population error count M, P_M[X <= observed_errors]
    # decreases monotonically in M.  Retaining all M whose lower-tail p-value
    # exceeds alpha yields an exact conservative upper confidence limit.
    lower = observed_errors
    upper = population_size - (sample_size - observed_errors)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        tail_probability = float(
            hypergeom.cdf(observed_errors, population_size, midpoint, sample_size)
        )
        if tail_probability > float(alpha):
            lower = midpoint
        else:
            upper = midpoint - 1
    return int(lower)


def beta_residual_posterior(
    *,
    stratum_sizes: np.ndarray,
    scout_counts: np.ndarray,
    scout_errors: np.ndarray,
    stratum_bands: np.ndarray,
    reviewed_bands: np.ndarray,
    prior_band_means: np.ndarray,
    prior_strength: float,
    population_size: int,
    draws: int,
    upper_probability: float,
    seed: int,
) -> ResidualPosterior:
    sizes = np.asarray(stratum_sizes, dtype=np.int64).reshape(-1)
    counts = np.asarray(scout_counts, dtype=np.int64).reshape(-1)
    errors = np.asarray(scout_errors, dtype=np.int64).reshape(-1)
    bands = np.asarray(stratum_bands, dtype=np.int64).reshape(-1)
    reviewed = np.asarray(reviewed_bands, dtype=bool).reshape(-1)
    prior_means = np.asarray(prior_band_means, dtype=np.float64).reshape(-1)
    if not (len(sizes) == len(counts) == len(errors) == len(bands)):
        raise ValueError("stratum vectors must have equal length")
    if (counts < 0).any() or (errors < 0).any() or (errors > counts).any():
        raise ValueError("invalid scout counts or errors")
    if (counts > sizes).any() or population_size < int(sizes.sum()):
        raise ValueError("scout or population sizes are inconsistent")
    if len(reviewed) != len(prior_means) or (bands >= len(prior_means)).any():
        raise ValueError("band metadata are inconsistent")
    if prior_strength < 0.0 or draws < 1 or not 0.0 < upper_probability < 1.0:
        raise ValueError("invalid posterior settings")
    if ((prior_means < 0.0) | (prior_means > 1.0)).any():
        raise ValueError("prior means must lie in [0, 1]")

    alpha = 1.0 + prior_strength * prior_means[bands] + errors
    beta = 1.0 + prior_strength * (1.0 - prior_means[bands]) + counts - errors
    rng = np.random.default_rng(int(seed))
    probabilities = rng.beta(alpha, beta, size=(draws, len(sizes)))
    remaining = sizes - counts
    remaining[reviewed[bands]] = 0
    residual_draws = probabilities @ remaining.astype(np.float64) / float(population_size)
    mean = float(residual_draws.mean())
    upper = float(np.quantile(residual_draws, upper_probability))
    return ResidualPosterior(mean=mean, upper=upper, upper_width=upper - mean)
