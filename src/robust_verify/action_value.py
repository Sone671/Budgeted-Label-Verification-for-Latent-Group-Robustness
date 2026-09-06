"""Legal-history action-value proxies and continue/stop certificates.

This module deliberately has no WGA evaluator.  A policy can use the returned
features and proxy intervals, while private child utilities remain confined to
the offline oracle module.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Sequence

import numpy as np

from robust_verify.sequential_audit import AuditAction, SequentialAuditConfig, SequentialAuditState


@dataclass(frozen=True)
class LegalStateFeatures:
    """Features that can be constructed without test groups or clean labels."""

    round_index: int
    spent_cost: Fraction
    remaining_budget: Fraction
    verified_count: int
    audited_attribute_count: int
    remaining_task_ids: tuple[int, ...]
    remaining_attribute_ids: tuple[int, ...]
    transcript_kinds: tuple[str, ...]
    current_label_histogram: tuple[int, ...]
    public_task_features: np.ndarray | None = None
    public_attribute_features: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.public_task_features is not None:
            task_features = np.asarray(self.public_task_features).copy()
            if task_features.ndim != 2:
                raise ValueError("public_task_features must be a matrix")
            task_features.setflags(write=False)
            object.__setattr__(self, "public_task_features", task_features)
        if self.public_attribute_features is not None:
            attribute_features = np.asarray(self.public_attribute_features).copy()
            if attribute_features.ndim != 2:
                raise ValueError("public_attribute_features must be a matrix")
            attribute_features.setflags(write=False)
            object.__setattr__(self, "public_attribute_features", attribute_features)


def build_legal_state_features(
    state: SequentialAuditState,
    config: SequentialAuditConfig,
    *,
    public_task_features: np.ndarray | None = None,
    public_attribute_features: np.ndarray | None = None,
) -> LegalStateFeatures:
    """Build a policy feature snapshot from the legal public transcript only."""

    task_remaining = tuple(
        index for index in range(config.task_pool_size) if index not in state.verified_ids
    )
    attribute_remaining = tuple(
        index for index in range(config.attribute_pool_size) if index not in state.audited_attribute_ids
    )
    task_features = None if public_task_features is None else np.asarray(public_task_features)
    attribute_features = (
        None if public_attribute_features is None else np.asarray(public_attribute_features)
    )
    if task_features is not None and len(task_features) != config.task_pool_size:
        raise ValueError("public_task_features must align with task_pool_size")
    if attribute_features is not None and len(attribute_features) != config.attribute_pool_size:
        raise ValueError("public_attribute_features must align with attribute_pool_size")
    histogram = tuple(np.bincount(state.current_labels, minlength=2).astype(int).tolist())
    return LegalStateFeatures(
        round_index=state.round_index,
        spent_cost=state.spent_cost,
        remaining_budget=state.remaining_budget(config),
        verified_count=len(state.verified_ids),
        audited_attribute_count=len(state.audited_attribute_ids),
        remaining_task_ids=task_remaining,
        remaining_attribute_ids=attribute_remaining,
        transcript_kinds=tuple(event.action.kind for event in state.transcript),
        current_label_histogram=histogram,
        public_task_features=task_features,
        public_attribute_features=attribute_features,
    )


@dataclass(frozen=True)
class ActionValueEstimate:
    """An estimated net continuation value and its simultaneous error radius."""

    action: AuditAction
    estimate: float
    error: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.estimate) or not np.isfinite(self.error):
            raise ValueError("action-value estimates must be finite")
        if self.error < 0:
            raise ValueError("action-value error must be non-negative")

    @property
    def lower(self) -> float:
        return self.estimate - self.error

    @property
    def upper(self) -> float:
        return self.estimate + self.error


CertificateStatus = Literal["continue", "stop", "unresolved"]


@dataclass(frozen=True)
class ActionCertificate:
    """Conservative decision certificate for one legal state."""

    status: CertificateStatus
    selected_action: AuditAction | None
    selected_lower: float
    selected_upper: float
    best_continue_lower: float
    best_continue_upper: float


def certify_action(
    estimates: Sequence[ActionValueEstimate],
) -> ActionCertificate:
    """Certify continue, stop, or unresolved using the document's interval rule.

    Stop has exact value zero.  A continuation is certified only if its lower
    bound is positive; stop is certified only if every continuation upper bound
    is non-positive.  The ambiguous case is explicitly ``unresolved``.
    """

    if any(estimate.action.kind == "stop" for estimate in estimates):
        raise ValueError("continuation estimates must not include Stop")
    if not estimates:
        return ActionCertificate("stop", None, 0.0, 0.0, float("-inf"), float("-inf"))
    ordered = sorted(
        estimates,
        key=lambda item: (-item.estimate, item.action.kind, item.action.sample_ids),
    )
    best = ordered[0]
    best_lower_item = max(
        estimates,
        key=lambda item: (item.lower, item.estimate, item.action.kind, item.action.sample_ids),
    )
    best_lower = max(item.lower for item in estimates)
    best_upper = max(item.upper for item in estimates)
    if best_lower > 0.0:
        return ActionCertificate(
            "continue",
            best_lower_item.action,
            best_lower_item.lower,
            best_lower_item.upper,
            best_lower,
            best_upper,
        )
    if best_upper <= 0.0:
        return ActionCertificate("stop", None, 0.0, 0.0, best_lower, best_upper)
    return ActionCertificate(
        "unresolved",
        None,
        best.lower,
        best.upper,
        best_lower,
        best_upper,
    )


def proxy_action_values(
    *,
    task_action: AuditAction | None,
    group_action: AuditAction | None,
    task_value: float | None,
    group_value: float | None,
    error: float,
) -> tuple[ActionValueEstimate, ...]:
    """Convenience constructor for a legal two-continuation proxy."""

    if error < 0 or not np.isfinite(error):
        raise ValueError("error must be finite and non-negative")
    estimates: list[ActionValueEstimate] = []
    for action, value in ((task_action, task_value), (group_action, group_value)):
        if action is not None and value is not None:
            estimates.append(ActionValueEstimate(action, float(value), float(error)))
    return tuple(estimates)
