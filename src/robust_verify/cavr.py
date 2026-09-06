"""Utilities for exploratory certified audit-versus-repair selection.

The functions in this module deliberately operate on an already purchased
audit panel.  They do not read unaudited group attributes and they do not
claim that the pilot cost can be recovered.  Pilot opportunity cost is handled
by the experiment driver when the selected policy is compared with the strict
label-only action.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AuditedPanelValue:
    """Worst-group accuracy and a simultaneous Wilson lower bound."""

    wga: float
    lower_bound: float
    group_accuracy: np.ndarray
    group_lower_bound: np.ndarray
    group_count: np.ndarray


@dataclass(frozen=True)
class SoftGroupValue:
    """Worst-group accuracy under probabilistic binary attribute membership."""

    wga: float
    group_accuracy: np.ndarray
    group_mass: np.ndarray


def _validate_panel_inputs(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("predictions and labels must be aligned vectors")
    if groups.shape != predictions.shape:
        raise ValueError("groups must align with predictions")
    if len(predictions) == 0:
        raise ValueError("the audited panel cannot be empty")
    if (groups < 0).any():
        raise ValueError("groups must be nonnegative integers")
    return predictions, labels, groups


def wilson_lower_bound(successes: int, count: int, *, alpha: float) -> float:
    """Return the one-sided Wilson lower confidence bound for a proportion."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("successes and count must satisfy 0 <= successes <= count")
    z = NormalDist().inv_cdf(1.0 - alpha)
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    radius = z * sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    )
    return max(0.0, float((center - radius) / denominator))


def audited_panel_value(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    expected_groups: Sequence[int],
    alpha: float = 0.05,
    simultaneous_comparisons: int = 1,
) -> AuditedPanelValue:
    """Evaluate WGA on an audited panel without silently dropping rare groups.

    ``alpha`` is divided across every expected group and candidate comparison.
    A missing expected group makes the lower bound zero and the point WGA
    undefined, represented by ``nan``.  This forces the caller to abstain or
    acquire more audits instead of treating missing groups as harmless.
    """

    predictions, labels, groups = _validate_panel_inputs(predictions, labels, groups)
    expected = np.asarray(list(expected_groups), dtype=np.int64)
    if expected.ndim != 1 or len(expected) == 0 or len(np.unique(expected)) != len(expected):
        raise ValueError("expected_groups must contain unique group identifiers")
    if simultaneous_comparisons <= 0:
        raise ValueError("simultaneous_comparisons must be positive")

    per_cell_alpha = alpha / (len(expected) * simultaneous_comparisons)
    accuracy = np.full(len(expected), np.nan, dtype=np.float64)
    lower = np.zeros(len(expected), dtype=np.float64)
    count = np.zeros(len(expected), dtype=np.int64)
    correct = predictions == labels
    for position, group in enumerate(expected):
        mask = groups == group
        count[position] = int(mask.sum())
        if count[position] == 0:
            continue
        successes = int(correct[mask].sum())
        accuracy[position] = successes / count[position]
        lower[position] = wilson_lower_bound(
            successes, int(count[position]), alpha=per_cell_alpha
        )

    wga = float(np.nan) if (count == 0).any() else float(accuracy.min())
    return AuditedPanelValue(
        wga=wga,
        lower_bound=float(lower.min()),
        group_accuracy=accuracy,
        group_lower_bound=lower,
        group_count=count,
    )


def soft_group_value(
    predictions: np.ndarray,
    labels: np.ndarray,
    attribute_probability: np.ndarray,
    *,
    num_classes: int,
) -> SoftGroupValue:
    """Evaluate WGA using class x probabilistic-attribute group weights.

    Clean validation labels define the class side of each group.  The binary
    attribute side is supplied only by a model fitted from purchased audits.
    This mirrors the legal soft-group semantics of the joint-budget pipeline,
    but evaluates each fully retrained candidate rather than the baseline
    model alone.
    """

    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(attribute_probability, dtype=np.float64)
    if predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("predictions and labels must be aligned vectors")
    if probability.shape != predictions.shape:
        raise ValueError("attribute_probability must align with predictions")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not np.isfinite(probability).all() or (
        (probability < 0.0) | (probability > 1.0)
    ).any():
        raise ValueError("attribute_probability must be finite and lie in [0, 1]")
    if ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError("labels fall outside num_classes")

    correct = (predictions == labels).astype(np.float64)
    accuracy = np.full(num_classes * 2, np.nan, dtype=np.float64)
    mass = np.zeros(num_classes * 2, dtype=np.float64)
    for class_value in range(num_classes):
        class_mask = labels == class_value
        for attribute_value in (0, 1):
            group = 2 * class_value + attribute_value
            attribute_weight = probability if attribute_value else 1.0 - probability
            weight = class_mask.astype(np.float64) * attribute_weight
            mass[group] = float(weight.sum())
            if mass[group] > 0.0:
                accuracy[group] = float(np.dot(weight, correct) / mass[group])
    wga = float(np.nan) if (mass <= 0.0).any() else float(accuracy.min())
    return SoftGroupValue(wga=wga, group_accuracy=accuracy, group_mass=mass)


def select_candidate(
    values: Sequence[float],
    audit_counts: Sequence[int],
) -> int:
    """Select the highest legal value, breaking ties toward fewer audits."""

    score = np.asarray(values, dtype=np.float64)
    cost = np.asarray(audit_counts, dtype=np.int64)
    if score.ndim != 1 or cost.shape != score.shape or len(score) == 0:
        raise ValueError("values and audit_counts must be aligned non-empty vectors")
    if (cost < 0).any():
        raise ValueError("audit_counts must be nonnegative")
    finite = np.isfinite(score)
    if not finite.any():
        raise ValueError("at least one candidate value must be finite")
    candidates = np.flatnonzero(finite)
    order = np.lexsort((candidates, cost[candidates], -score[candidates]))
    return int(candidates[order[0]])


def allocation_regret(candidate_values: Sequence[float], selected: int) -> float:
    """Return ex-post regret within the supplied finite candidate set."""

    values = np.asarray(candidate_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("candidate_values must be a finite non-empty vector")
    if not 0 <= selected < len(values):
        raise ValueError("selected is outside candidate_values")
    return float(values.max() - values[selected])
