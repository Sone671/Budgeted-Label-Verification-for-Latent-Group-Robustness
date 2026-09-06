from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from robust_verify.rivet import spectral_class_cvar


@dataclass(frozen=True)
class ValidationPartition:
    """Row indices for the three legally separated validation roles."""

    early_stop: np.ndarray
    calibration: np.ndarray
    audit: np.ndarray


@dataclass(frozen=True)
class AuditResult:
    """Paired audit result for candidate-minus-reference spectral risk."""

    observed_difference: float
    lower: float
    upper: float
    confidence: float
    bootstrap_replicates: int


def _class_split_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if size < 3:
        raise ValueError("every observed class needs at least three validation rows")
    raw = np.asarray(fractions, dtype=np.float64) * size
    counts = np.floor(raw).astype(np.int64)
    counts = np.maximum(counts, 1)
    while int(counts.sum()) > size:
        candidates = np.flatnonzero(counts > 1)
        if len(candidates) == 0:
            raise ValueError("cannot allocate non-empty validation roles")
        remove = int(candidates[np.argmax(counts[candidates] - raw[candidates])])
        counts[remove] -= 1
    while int(counts.sum()) < size:
        add = int(np.argmax(raw - counts))
        counts[add] += 1
    return tuple(int(value) for value in counts)


def stratified_validation_partition(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    seed: int = 20260810,
) -> ValidationPartition:
    """Split by observed class with assignments stable to input row order."""

    sample_ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(sample_ids) != len(labels):
        raise ValueError("sample_ids and labels must have equal length")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    fractions = tuple(float(value) for value in fractions)
    if len(fractions) != 3 or any(value <= 0.0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")

    roles: list[list[int]] = [[], [], []]
    for class_value in sorted(np.unique(labels).tolist()):
        class_positions = np.flatnonzero(labels == class_value)
        stable = class_positions[np.argsort(sample_ids[class_positions], kind="stable")]
        class_seed = int(seed) + 104729 * (int(class_value) + 1)
        shuffled = stable[np.random.default_rng(class_seed).permutation(len(stable))]
        counts = _class_split_counts(len(shuffled), fractions)
        first, second = counts[0], counts[0] + counts[1]
        roles[0].extend(shuffled[:first].tolist())
        roles[1].extend(shuffled[first:second].tolist())
        roles[2].extend(shuffled[second:].tolist())

    arrays = tuple(np.asarray(sorted(role), dtype=np.int64) for role in roles)
    combined = np.concatenate(arrays)
    if len(combined) != len(sample_ids) or len(np.unique(combined)) != len(sample_ids):
        raise RuntimeError("validation partition is not exhaustive and disjoint")
    return ValidationPartition(*arrays)


def sample_id_hash(sample_ids: np.ndarray, positions: np.ndarray) -> str:
    selected = np.sort(np.asarray(sample_ids, dtype=np.int64)[positions])
    return hashlib.sha256(selected.tobytes()).hexdigest()


def assert_nested_query_prefixes(query_sets: Iterable[np.ndarray]) -> None:
    """Require exact ordered prefixes for a deployable sequential trajectory."""

    arrays = [np.asarray(values, dtype=np.int64).reshape(-1) for values in query_sets]
    if not arrays:
        raise ValueError("at least one query set is required")
    for index, values in enumerate(arrays):
        if len(np.unique(values)) != len(values):
            raise ValueError(f"query set {index} contains duplicate sample IDs")
        if index and not np.array_equal(arrays[index - 1], values[: len(arrays[index - 1])]):
            raise ValueError(
                f"query set {index - 1} is not an ordered prefix of query set {index}"
            )


def nominate_by_calibration(
    candidate_risk: float,
    reference_risk: float,
    *,
    minimum_improvement: float = 0.0,
) -> bool:
    """Nominate only a strict, predeclared calibration-risk improvement."""

    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be non-negative")
    return float(candidate_risk) < float(reference_risk) - minimum_improvement


def paired_stratified_bootstrap_risk_difference(
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    alphas: tuple[float, ...],
    temperature: float,
    confidence: float,
    replicates: int,
    seed: int,
) -> AuditResult:
    """Bootstrap a paired candidate-minus-reference spectral-risk contrast."""

    candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if candidate_probabilities.shape != reference_probabilities.shape:
        raise ValueError("candidate and reference probabilities must have equal shape")
    if candidate_probabilities.ndim != 2 or len(candidate_probabilities) != len(labels):
        raise ValueError("probabilities and labels must have matching rows")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    def risk(probabilities: np.ndarray, selected_labels: np.ndarray) -> float:
        return float(
            spectral_class_cvar(
                probabilities,
                selected_labels,
                alphas=alphas,
                temperature=temperature,
            )[0]
        )

    observed = risk(candidate_probabilities, labels) - risk(reference_probabilities, labels)
    class_positions = [np.flatnonzero(labels == value) for value in sorted(np.unique(labels))]
    rng = np.random.default_rng(int(seed))
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [positions[rng.integers(0, len(positions), size=len(positions))] for positions in class_positions]
        )
        sampled_labels = labels[sampled]
        differences[replicate] = risk(
            candidate_probabilities[sampled], sampled_labels
        ) - risk(reference_probabilities[sampled], sampled_labels)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [tail, 1.0 - tail])
    return AuditResult(
        observed_difference=float(observed),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        bootstrap_replicates=int(replicates),
    )


def audit_accepts(
    result: AuditResult,
    *,
    minimum_improvement: float = 0.0,
) -> bool:
    """Accept only when the simultaneous upper endpoint clears the margin."""

    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be non-negative")
    return float(result.upper) < -minimum_improvement
