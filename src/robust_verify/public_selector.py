"""A public-only R/G/Stop selector for the real-data pilot.

The selector intentionally has no evaluator, clean-label oracle, test-set
metrics, or trajectory-oracle dependency.  It consumes a frozen public audit
design, purchased responses, and public validation predictions.  Its value
proxies are deliberately conservative: strict interval dominance is required
for Continue, while overlapping intervals become Unresolved and use a
minimax-regret fallback.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from robust_verify.public_value import (
    ValueInterval,
    audited_group_accuracy_intervals,
    bernoulli_ipw_group_accuracy_intervals,
    worst_group_accuracy_interval,
)

SelectorStatus = Literal["continue", "stop", "unresolved"]


@dataclass(frozen=True)
class PublicActionInterval:
    kind: Literal["verify_task", "annotate_group"]
    sample_ids: tuple[int, ...]
    value: ValueInterval


@dataclass(frozen=True)
class PublicSelectorDecision:
    status: SelectorStatus
    selected_kind: str
    selected_ids: tuple[int, ...]
    verify_interval: ValueInterval | None
    audit_interval: ValueInterval | None
    group_wga_interval: ValueInterval
    minimax_kind: str
    minimax_regret_upper: float


def _minimax_fallback(
    intervals: Sequence[PublicActionInterval],
) -> tuple[str, float]:
    candidates = [("stop", ValueInterval(0.0, 0.0), ())]
    candidates.extend((item.kind, item.value, item.sample_ids) for item in intervals)
    scored: list[tuple[float, int, str, tuple[int, ...]]] = []
    for kind, value, sample_ids in candidates:
        best_other_upper = max(
            other_value.upper
            for other_kind, other_value, _ in candidates
            if other_kind != kind
        )
        regret = max(0.0, best_other_upper - value.lower)
        scored.append((regret, 0 if kind == "stop" else 1, kind, sample_ids))
    regret, _, kind, _ = min(scored)
    return kind, float(regret)


def _strict_dominant_action(
    intervals: Sequence[PublicActionInterval],
) -> PublicActionInterval | None:
    candidates = list(intervals)
    for candidate in candidates:
        others_upper = max(
            [0.0]
            + [
                other.value.upper
                for other in candidates
                if other.kind != candidate.kind
            ]
        )
        if candidate.value.lower > others_upper:
            return candidate
    return None


def choose_public_action(
    *,
    current_labels: Sequence[int] | np.ndarray,
    train_probabilities: np.ndarray,
    public_validation_correct: Sequence[int] | np.ndarray,
    audited_indices: Sequence[int] | np.ndarray,
    audited_groups: Sequence[int] | np.ndarray,
    inclusion_probabilities: Sequence[float] | np.ndarray,
    candidate_task_ids: Sequence[int] | np.ndarray,
    candidate_audit_ids: Sequence[int] | np.ndarray,
    num_groups: int,
    remaining_budget: float,
    task_batch_cost: float,
    audit_batch_cost: float,
    audit_value_scale: float = 0.75,
    comparisons: int = 1,
    eligible_masks: np.ndarray | None = None,
    allow_uncalibrated_verify: bool = False,
    verify_scores: Sequence[float] | np.ndarray | None = None,
) -> PublicSelectorDecision:
    """Choose one action using only the public state and purchased audits.

    The verify proxy is the average predicted error probability of the proposed
    batch.  The audit proxy is the remaining uncertainty in the public WGA
    interval, scaled by a frozen opportunity-value coefficient.  Both proxies
    are interval-valued and net of normalized action cost.  This is a pilot
    selector, not a claim that predicted label repair equals true retraining
    value; the external evaluator measures that gap.
    """

    labels = np.asarray(current_labels)
    probabilities = np.asarray(train_probabilities, dtype=float)
    validation_correct = np.asarray(public_validation_correct)
    task_ids = np.asarray(candidate_task_ids, dtype=np.int64)
    audit_ids = np.asarray(candidate_audit_ids, dtype=np.int64)
    if labels.ndim != 1 or probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError("current labels and train probabilities must align")
    if len(task_ids) == 0 and len(audit_ids) == 0:
        raise ValueError("at least one continuation candidate is required")
    if remaining_budget < 0 or task_batch_cost < 0 or audit_batch_cost < 0:
        raise ValueError("budget and action costs must be non-negative")
    if not 0.0 < audit_value_scale <= 1.0:
        raise ValueError("audit_value_scale must lie in (0, 1]")

    inclusion = np.asarray(inclusion_probabilities, dtype=float)
    if len(inclusion) and np.allclose(inclusion, inclusion[0]):
        group_intervals = audited_group_accuracy_intervals(
            validation_correct[np.asarray(audited_indices, dtype=np.int64)],
            np.asarray(audited_groups, dtype=np.int64),
            num_groups=num_groups,
            comparisons=comparisons,
        )
    else:
        group_intervals = bernoulli_ipw_group_accuracy_intervals(
            validation_correct,
            audited_indices,
            audited_groups,
            inclusion,
            num_groups=num_groups,
            comparisons=comparisons,
            eligible_masks=eligible_masks,
        )
    group_wga = worst_group_accuracy_interval(group_intervals)

    intervals: list[PublicActionInterval] = []
    if len(task_ids) and task_batch_cost <= remaining_budget:
        if verify_scores is None:
            label_probabilities = probabilities[np.arange(len(labels)), labels]
            action_scores = 1.0 - np.clip(label_probabilities, 0.0, 1.0)
        else:
            action_scores = np.asarray(verify_scores, dtype=float)
            if action_scores.ndim != 1 or len(action_scores) != len(labels):
                raise ValueError("verify_scores must align with current_labels")
            if not np.all(np.isfinite(action_scores)):
                raise ValueError("verify_scores must be finite")
            action_scores = np.clip(action_scores, 0.0, 1.0)
        selected = task_ids[np.argsort(-action_scores[task_ids], kind="stable")]
        center = float(np.mean(action_scores[selected])) - task_batch_cost / max(
            remaining_budget + task_batch_cost, 1e-12
        )
        radius = float(min(1.0, 0.25 + np.std(action_scores[selected]) / np.sqrt(len(selected))))
        intervals.append(
            PublicActionInterval(
                "verify_task",
                tuple(int(item) for item in selected.tolist()),
                ValueInterval(center - radius, center + radius),
            )
        )

    if len(audit_ids) and audit_batch_cost <= remaining_budget:
        center = audit_value_scale * group_wga.radius - audit_batch_cost / max(
            remaining_budget + audit_batch_cost, 1e-12
        )
        radius = float(min(1.0, 0.25 + group_wga.radius))
        intervals.append(
            PublicActionInterval(
                "annotate_group",
                tuple(int(item) for item in audit_ids.tolist()),
                ValueInterval(center - radius, center + radius),
            )
        )

    if not intervals:
        return PublicSelectorDecision(
            "stop",
            "stop",
            (),
            None,
            None,
            group_wga,
            "stop",
            0.0,
        )

    dominant = _strict_dominant_action(intervals)
    if dominant is not None and dominant.kind == "verify_task" and not allow_uncalibrated_verify:
        dominant = None
    minimax_kind, minimax_regret = _minimax_fallback(intervals)
    certified_stop = all(item.value.upper <= 0.0 for item in intervals)
    if dominant is not None:
        selected_kind = dominant.kind
        selected_ids = dominant.sample_ids
        status: SelectorStatus = "continue"
    elif certified_stop:
        selected_kind = "stop"
        selected_ids = ()
        status = "stop"
    else:
        selected_kind = minimax_kind
        selected_ids = (
            ()
            if minimax_kind == "stop"
            else next(item.sample_ids for item in intervals if item.kind == minimax_kind)
        )
        status = "unresolved"
        # A point estimate is not justified by an unresolved certificate.  In
        # particular, the generic minimax rule can prefer a repair batch when
        # its interval overlaps zero; the deployable fallback must abstain
        # from unverified label changes in that case.
        if minimax_kind == "verify_task":
            verify_item = next(item for item in intervals if item.kind == "verify_task")
            if not allow_uncalibrated_verify or verify_item.value.lower <= 0.0:
                selected_kind = "stop"
                selected_ids = ()

    verify_interval = next(
        (item.value for item in intervals if item.kind == "verify_task"), None
    )
    audit_interval = next(
        (item.value for item in intervals if item.kind == "annotate_group"), None
    )
    return PublicSelectorDecision(
        status,
        selected_kind,
        selected_ids,
        verify_interval,
        audit_interval,
        group_wga,
        minimax_kind,
        minimax_regret,
    )
