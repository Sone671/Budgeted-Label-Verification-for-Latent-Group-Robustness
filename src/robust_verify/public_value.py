"""Public-only value intervals and robust Bellman planning.

The three-world lower bound rules out a universally correct point-valued
R/G/Stop selector.  This module therefore propagates simultaneous intervals
and preserves an explicit unresolved state.  It deliberately has no dependency
on private test evaluators or the offline trajectory oracle.

The unweighted confidence bounds require random or class-stratified random
audits.  The IPW and doubly robust bounds allow unequal, public sampling
probabilities, but require independent Bernoulli inclusion.  None of these
bounds applies to an arbitrary adaptive fixed-size audit design.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, log, sqrt

import numpy as np

from robust_verify.action_value import (
    ActionCertificate,
    ActionValueEstimate,
    certify_action,
)
from robust_verify.sequential_audit import AuditAction


@dataclass(frozen=True)
class ValueInterval:
    """A closed interval known to contain one scalar value."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def radius(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def shifted(self, amount: float) -> ValueInterval:
        if not np.isfinite(amount):
            raise ValueError("interval shift must be finite")
        return ValueInterval(self.lower + amount, self.upper + amount)


@dataclass(frozen=True)
class GroupAccuracyInterval:
    """Simultaneous Hoeffding interval for one audited group."""

    group: int
    successes: int
    count: int
    value: ValueInterval


@dataclass(frozen=True)
class IpwGroupAccuracyInterval:
    """IPW/DR estimates and simultaneous intervals for one latent group.

    ``weighted_successes`` and ``weighted_mass`` are estimated population
    totals.  Dividing them by the public population size gives the point
    estimates underlying ``numerator`` and ``mass`` respectively.
    """

    group: int
    audited_count: int
    weighted_successes: float
    weighted_mass: float
    numerator: ValueInterval
    mass: ValueInterval
    value: ValueInterval


def _validate_bernoulli_audit_inputs(
    prediction_correct: Sequence[int] | np.ndarray,
    audited_indices: Sequence[int] | np.ndarray,
    audited_groups: Sequence[int] | np.ndarray,
    inclusion_probabilities: Sequence[float] | np.ndarray,
    *,
    num_groups: int,
    delta: float,
    comparisons: int,
    eligible_masks: Sequence[Sequence[bool]] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct = np.asarray(prediction_correct)
    indices = np.asarray(audited_indices)
    groups = np.asarray(audited_groups)
    probabilities = np.asarray(inclusion_probabilities, dtype=float)
    if correct.ndim != 1 or len(correct) == 0:
        raise ValueError("prediction_correct must be a non-empty vector")
    if not set(np.unique(correct).tolist()).issubset({0, 1}):
        raise ValueError("prediction_correct must be binary")
    if indices.ndim != 1 or groups.ndim != 1 or len(indices) != len(groups):
        raise ValueError("audited_indices and audited_groups must be aligned vectors")
    if len(indices) and not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("audited_indices must contain integers")
    if len(groups) and not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("audited_groups must contain integer group IDs")
    indices = indices.astype(np.int64, copy=False)
    groups = groups.astype(np.int64, copy=False)
    population_size = len(correct)
    if len(indices) and ((indices < 0).any() or (indices >= population_size).any()):
        raise ValueError("audited_indices contains an index outside the population")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("audited_indices must not contain duplicates")
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if len(groups) and ((groups < 0).any() or (groups >= num_groups).any()):
        raise ValueError("audited_groups contains an ID outside [0, num_groups)")
    if probabilities.ndim != 1 or len(probabilities) != population_size:
        raise ValueError("inclusion_probabilities must align with prediction_correct")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError("inclusion probabilities must be finite and lie in (0, 1]")
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")

    selected = np.zeros(population_size, dtype=bool)
    selected[indices] = True
    if not np.all(selected[probabilities == 1.0]):
        raise ValueError("every item with inclusion probability one must be audited")

    if eligible_masks is None:
        eligible = np.ones((num_groups, population_size), dtype=bool)
    else:
        eligible_array = np.asarray(eligible_masks)
        if eligible_array.shape != (num_groups, population_size):
            raise ValueError("eligible_masks must have shape (num_groups, population_size)")
        if not set(np.unique(eligible_array).tolist()).issubset({0, 1}):
            raise ValueError("eligible_masks must be binary")
        eligible = eligible_array.astype(bool, copy=False)
        if not np.all(eligible.any(axis=0)):
            raise ValueError("every population item must be eligible for at least one group")
        if len(indices) and not np.all(eligible[groups, indices]):
            raise ValueError("an audited group conflicts with eligible_masks")

    return (
        correct.astype(float, copy=False),
        indices,
        groups,
        probabilities,
        eligible,
    )


def _bernstein_radius(
    probabilities: np.ndarray,
    residual_bounds: np.ndarray,
    *,
    log_term: float,
) -> float:
    stochastic = probabilities < 1.0
    active = stochastic & (residual_bounds > 0.0)
    if not np.any(active):
        return 0.0
    active_probabilities = probabilities[active]
    active_bounds = residual_bounds[active]
    variance = np.sum(
        ((1.0 - active_probabilities) / active_probabilities) * active_bounds**2
    )
    centered_weight_bound = np.maximum(
        1.0,
        (1.0 - active_probabilities) / active_probabilities,
    )
    bound = float(np.max(centered_weight_bound * active_bounds))
    linear_term = bound * log_term / 3.0
    return float(linear_term + sqrt(2.0 * variance * log_term + linear_term**2))


def _constrained_ratio_intervals(
    numerator_estimate: float,
    numerator_radius: float,
    mass_estimate: float,
    mass_radius: float,
) -> tuple[ValueInterval, ValueInterval, ValueInterval]:
    def clip_probability(value: float) -> float:
        return min(1.0, max(0.0, value))

    raw_numerator = ValueInterval(
        clip_probability(numerator_estimate - numerator_radius),
        clip_probability(numerator_estimate + numerator_radius),
    )
    raw_mass = ValueInterval(
        clip_probability(mass_estimate - mass_radius),
        clip_probability(mass_estimate + mass_radius),
    )
    numerator_lower = raw_numerator.lower
    numerator_upper = min(raw_numerator.upper, raw_mass.upper)
    mass_lower = max(raw_mass.lower, raw_numerator.lower)
    mass_upper = raw_mass.upper
    if numerator_lower > numerator_upper or mass_lower > mass_upper:
        # The simultaneous confidence box missed the structurally feasible set.
        # A vacuous result is preferable to returning an invalid ratio interval.
        vacuous = ValueInterval(0.0, 1.0)
        return vacuous, vacuous, vacuous
    numerator = ValueInterval(numerator_lower, numerator_upper)
    mass = ValueInterval(mass_lower, mass_upper)
    ratio = ValueInterval(
        0.0 if mass.upper == 0.0 else min(1.0, numerator.lower / mass.upper),
        1.0 if mass.lower == 0.0 else min(1.0, numerator.upper / mass.lower),
    )
    return numerator, mass, ratio


def _bernoulli_group_accuracy_intervals(
    correct: np.ndarray,
    indices: np.ndarray,
    groups: np.ndarray,
    probabilities: np.ndarray,
    eligible: np.ndarray,
    *,
    group_probabilities: np.ndarray | None,
    delta: float,
    comparisons: int,
) -> tuple[IpwGroupAccuracyInterval, ...]:
    population_size = len(correct)
    num_groups = eligible.shape[0]
    # Two estimands (correct mass and total mass) are covered for every group.
    family_size = 2 * num_groups * comparisons
    log_term = log(2.0 * family_size / delta)
    intervals: list[IpwGroupAccuracyInterval] = []
    for group in range(num_groups):
        audited_in_group = groups == group
        audited_count = int(audited_in_group.sum())
        group_indices = indices[audited_in_group]
        inverse_probabilities = 1.0 / probabilities[group_indices]

        if group_probabilities is None:
            weighted_mass = float(inverse_probabilities.sum())
            weighted_successes = float((inverse_probabilities * correct[group_indices]).sum())
            mass_residual_bound = eligible[group].astype(float)
        else:
            predictions = group_probabilities[:, group]
            observed_membership = audited_in_group.astype(float)
            residuals = observed_membership - predictions[indices]
            audited_inverse_probabilities = 1.0 / probabilities[indices]
            weighted_mass = float(
                predictions.sum() + (audited_inverse_probabilities * residuals).sum()
            )
            weighted_successes = float(
                np.dot(correct, predictions)
                + (
                    audited_inverse_probabilities
                    * correct[indices]
                    * residuals
                ).sum()
            )
            mass_residual_bound = eligible[group] * np.maximum(
                predictions,
                1.0 - predictions,
            )

        success_residual_bound = correct * mass_residual_bound
        mass_radius = _bernstein_radius(
            probabilities,
            mass_residual_bound,
            log_term=log_term,
        ) / population_size
        numerator_radius = _bernstein_radius(
            probabilities,
            success_residual_bound,
            log_term=log_term,
        ) / population_size
        numerator, mass, value = _constrained_ratio_intervals(
            weighted_successes / population_size,
            numerator_radius,
            weighted_mass / population_size,
            mass_radius,
        )
        intervals.append(
            IpwGroupAccuracyInterval(
                group=group,
                audited_count=audited_count,
                weighted_successes=weighted_successes,
                weighted_mass=weighted_mass,
                numerator=numerator,
                mass=mass,
                value=value,
            )
        )
    return tuple(intervals)


def bernoulli_ipw_group_accuracy_intervals(
    prediction_correct: Sequence[int] | np.ndarray,
    audited_indices: Sequence[int] | np.ndarray,
    audited_groups: Sequence[int] | np.ndarray,
    inclusion_probabilities: Sequence[float] | np.ndarray,
    *,
    num_groups: int,
    delta: float = 0.05,
    comparisons: int = 1,
    eligible_masks: Sequence[Sequence[bool]] | np.ndarray | None = None,
) -> tuple[IpwGroupAccuracyInterval, ...]:
    """Return simultaneous WGA intervals for a Bernoulli audit design.

    The inclusion indicators must be mutually independent with known,
    pre-specified probabilities.  Only group labels at ``audited_indices`` are
    consumed.  ``eligible_masks`` may encode public, pre-audit impossibility
    constraints and otherwise defaults to allowing every group for every item.
    These guarantees do not cover adaptive fixed-size sampling.
    """

    correct, indices, groups, probabilities, eligible = _validate_bernoulli_audit_inputs(
        prediction_correct,
        audited_indices,
        audited_groups,
        inclusion_probabilities,
        num_groups=num_groups,
        delta=delta,
        comparisons=comparisons,
        eligible_masks=eligible_masks,
    )
    return _bernoulli_group_accuracy_intervals(
        correct,
        indices,
        groups,
        probabilities,
        eligible,
        group_probabilities=None,
        delta=delta,
        comparisons=comparisons,
    )


def bernoulli_dr_group_accuracy_intervals(
    prediction_correct: Sequence[int] | np.ndarray,
    audited_indices: Sequence[int] | np.ndarray,
    audited_groups: Sequence[int] | np.ndarray,
    inclusion_probabilities: Sequence[float] | np.ndarray,
    group_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_groups: int,
    delta: float = 0.05,
    comparisons: int = 1,
    eligible_masks: Sequence[Sequence[bool]] | np.ndarray | None = None,
) -> tuple[IpwGroupAccuracyInterval, ...]:
    """Return cross-fitted AIPW/DR group-accuracy intervals.

    ``group_probabilities`` must have been fixed before this audit draw or
    produced out-of-fold; estimating them on the same audited outcomes voids
    the guarantee.  The sampling design has the same independent-Bernoulli
    restriction as :func:`bernoulli_ipw_group_accuracy_intervals`.
    """

    correct, indices, groups, probabilities, eligible = _validate_bernoulli_audit_inputs(
        prediction_correct,
        audited_indices,
        audited_groups,
        inclusion_probabilities,
        num_groups=num_groups,
        delta=delta,
        comparisons=comparisons,
        eligible_masks=eligible_masks,
    )
    predictions = np.asarray(group_probabilities, dtype=float)
    expected_shape = (len(correct), num_groups)
    if predictions.shape != expected_shape:
        raise ValueError(
            "group_probabilities must have shape (population_size, num_groups)"
        )
    if not np.all(np.isfinite(predictions)) or np.any(predictions < 0.0) or np.any(
        predictions > 1.0
    ):
        raise ValueError("group_probabilities must be finite and lie in [0, 1]")
    if not np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9):
        raise ValueError("each row of group_probabilities must sum to one")
    if np.any(predictions.T[~eligible] != 0.0):
        raise ValueError("group_probabilities must be zero outside eligible_masks")
    return _bernoulli_group_accuracy_intervals(
        correct,
        indices,
        groups,
        probabilities,
        eligible,
        group_probabilities=predictions,
        delta=delta,
        comparisons=comparisons,
    )


def audited_group_accuracy_intervals(
    prediction_correct: Sequence[int] | np.ndarray,
    audited_groups: Sequence[int] | np.ndarray,
    *,
    num_groups: int,
    delta: float = 0.05,
    comparisons: int = 1,
) -> tuple[GroupAccuracyInterval, ...]:
    """Return simultaneous group-accuracy intervals from random audits.

    Hoeffding's inequality and a union bound cover every group and every
    caller-declared comparison with probability at least ``1 - delta``.  A
    missing group receives the honest vacuous interval ``[0, 1]``.
    """

    correct = np.asarray(prediction_correct)
    groups = np.asarray(audited_groups)
    if correct.ndim != 1 or groups.ndim != 1 or len(correct) != len(groups):
        raise ValueError("prediction_correct and audited_groups must be aligned vectors")
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    if not set(np.unique(correct).tolist()).issubset({0, 1}):
        raise ValueError("prediction_correct must be binary")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("audited_groups must contain integer group IDs")
    groups = groups.astype(np.int64, copy=False)
    if len(groups) and ((groups < 0).any() or (groups >= num_groups).any()):
        raise ValueError("audited_groups contains an ID outside [0, num_groups)")

    family_size = num_groups * comparisons
    log_term = log(2.0 * family_size / delta)
    intervals: list[GroupAccuracyInterval] = []
    for group in range(num_groups):
        mask = groups == group
        count = int(mask.sum())
        successes = int(np.asarray(correct[mask], dtype=np.int64).sum())
        if count == 0:
            value = ValueInterval(0.0, 1.0)
        else:
            estimate = successes / count
            radius = sqrt(log_term / (2.0 * count))
            value = ValueInterval(max(0.0, estimate - radius), min(1.0, estimate + radius))
        intervals.append(GroupAccuracyInterval(group, successes, count, value))
    return tuple(intervals)


def worst_group_accuracy_interval(
    intervals: Sequence[GroupAccuracyInterval | IpwGroupAccuracyInterval],
) -> ValueInterval:
    """Propagate simultaneous group intervals through the WGA minimum."""

    if not intervals:
        raise ValueError("at least one group interval is required")
    group_ids = [interval.group for interval in intervals]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("group intervals must have unique group IDs")
    return ValueInterval(
        min(interval.value.lower for interval in intervals),
        min(interval.value.upper for interval in intervals),
    )


def incremental_utility_interval(
    before: ValueInterval,
    after: ValueInterval,
    *,
    normalized_cost: float,
) -> ValueInterval:
    """Bound an after-minus-before utility net of a known normalized cost."""

    if not np.isfinite(normalized_cost) or normalized_cost < 0.0:
        raise ValueError("normalized_cost must be finite and non-negative")
    return ValueInterval(
        after.lower - before.upper - normalized_cost,
        after.upper - before.lower - normalized_cost,
    )


@dataclass(frozen=True)
class PublicOutcomeModel:
    """One plausible public-feedback distribution over child nodes."""

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        probabilities = tuple(float(value) for value in self.probabilities)
        if not probabilities:
            raise ValueError("an outcome model must contain at least one probability")
        if not all(np.isfinite(value) and value >= 0.0 for value in probabilities):
            raise ValueError("outcome probabilities must be finite and non-negative")
        if not isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("outcome probabilities must sum to one")
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True)
class PublicTransition:
    """A legal action with interval reward and plausible feedback laws."""

    action: AuditAction
    child_nodes: tuple[str, ...]
    outcome_models: tuple[PublicOutcomeModel, ...]
    immediate_value: ValueInterval = ValueInterval(0.0, 0.0)

    def __post_init__(self) -> None:
        if self.action.kind == "stop":
            raise ValueError("Stop is represented by the node's exact zero value")
        child_nodes = tuple(str(node) for node in self.child_nodes)
        if not child_nodes or any(not node for node in child_nodes):
            raise ValueError("continuation transitions require non-empty child node IDs")
        if not self.outcome_models:
            raise ValueError("at least one plausible outcome model is required")
        if any(len(model.probabilities) != len(child_nodes) for model in self.outcome_models):
            raise ValueError("each outcome model must align with child_nodes")
        object.__setattr__(self, "child_nodes", child_nodes)


@dataclass(frozen=True)
class PublicDecisionNode:
    """One public history and its finite candidate action menu."""

    node_id: str
    transitions: tuple[PublicTransition, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        actions = [transition.action for transition in self.transitions]
        if len(set(actions)) != len(actions):
            raise ValueError("a public node cannot contain duplicate actions")


@dataclass(frozen=True)
class IntervalActionValue:
    """A legal action paired with its robust Bellman interval."""

    action: AuditAction
    value: ValueInterval

    def as_symmetric_estimate(self) -> ActionValueEstimate:
        return ActionValueEstimate(self.action, self.value.midpoint, self.value.radius)


@dataclass(frozen=True)
class RobustBellmanDecision:
    """Interval decision, certificate, and conservative fallback at one node."""

    node_id: str
    depth_remaining: int
    value: ValueInterval
    action_values: tuple[IntervalActionValue, ...]
    certificate: ActionCertificate
    dominant_action: AuditAction | None
    minimax_action: AuditAction
    minimax_regret_upper: float


@dataclass(frozen=True)
class RobustBellmanSolution:
    """All decisions produced by a finite-horizon robust backup."""

    root: RobustBellmanDecision
    decisions: Mapping[tuple[str, int], RobustBellmanDecision]


def _dominant_action(values: Sequence[IntervalActionValue]) -> AuditAction | None:
    if len(values) == 1:
        return values[0].action
    winners = [
        candidate.action
        for candidate in values
        if candidate.value.lower
        > max(
            other.value.upper
            for other in values
            if other.action != candidate.action
        )
    ]
    return winners[0] if len(winners) == 1 else None


def _minimax_interval_fallback(
    values: Sequence[IntervalActionValue],
) -> tuple[AuditAction, float]:
    """Minimize a conservative regret bound implied only by value intervals."""

    if not values:
        raise ValueError("at least one action value is required")
    if len(values) == 1:
        return values[0].action, 0.0
    scored: list[tuple[float, int, str, tuple[int, ...], AuditAction]] = []
    for candidate in values:
        best_other_upper = max(
            other.value.upper for other in values if other.action != candidate.action
        )
        regret_upper = max(0.0, best_other_upper - candidate.value.lower)
        action = candidate.action
        scored.append(
            (
                regret_upper,
                0 if action.kind == "stop" else 1,
                action.kind,
                action.sample_ids,
                action,
            )
        )
    regret, _, _, _, action = min(scored)
    return action, regret


def solve_robust_bellman(
    nodes: Mapping[str, PublicDecisionNode] | Sequence[PublicDecisionNode],
    root_node: str,
    *,
    max_depth: int,
) -> RobustBellmanSolution:
    """Back up public value intervals over a finite feedback tree.

    Each transition may provide several plausible feedback distributions.  The
    lower backup uses the least favorable distribution and the upper backup the
    most favorable distribution.  Valid simultaneous immediate-value and
    transition confidence sets therefore imply valid Bellman intervals.
    """

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    node_map = (
        {node.node_id: node for node in nodes}
        if not isinstance(nodes, Mapping)
        else dict(nodes)
    )
    if root_node not in node_map:
        raise KeyError(f"unknown root node: {root_node}")
    memo: dict[tuple[str, int], RobustBellmanDecision] = {}

    def visit(node_id: str, depth: int) -> RobustBellmanDecision:
        key = (node_id, depth)
        if key in memo:
            return memo[key]
        node = node_map.get(node_id)
        if node is None:
            raise KeyError(f"unknown child node: {node_id}")

        stop = IntervalActionValue(AuditAction.stop(), ValueInterval(0.0, 0.0))
        action_values: list[IntervalActionValue] = [stop]
        if depth > 0:
            for transition in node.transitions:
                children = [visit(child_node, depth - 1) for child_node in transition.child_nodes]
                lower_expectations = [
                    sum(
                        probability * child.value.lower
                        for probability, child in zip(model.probabilities, children)
                    )
                    for model in transition.outcome_models
                ]
                upper_expectations = [
                    sum(
                        probability * child.value.upper
                        for probability, child in zip(model.probabilities, children)
                    )
                    for model in transition.outcome_models
                ]
                action_values.append(
                    IntervalActionValue(
                        transition.action,
                        ValueInterval(
                            transition.immediate_value.lower + min(lower_expectations),
                            transition.immediate_value.upper + max(upper_expectations),
                        ),
                    )
                )

        value = ValueInterval(
            max(item.value.lower for item in action_values),
            max(item.value.upper for item in action_values),
        )
        continuation = [
            item.as_symmetric_estimate()
            for item in action_values
            if item.action.kind != "stop"
        ]
        minimax_action, minimax_regret = _minimax_interval_fallback(action_values)
        decision = RobustBellmanDecision(
            node_id=node_id,
            depth_remaining=depth,
            value=value,
            action_values=tuple(action_values),
            certificate=certify_action(continuation),
            dominant_action=_dominant_action(action_values),
            minimax_action=minimax_action,
            minimax_regret_upper=minimax_regret,
        )
        memo[key] = decision
        return decision

    root = visit(root_node, max_depth)
    return RobustBellmanSolution(root=root, decisions=dict(memo))
