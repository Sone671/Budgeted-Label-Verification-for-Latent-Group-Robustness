"""Exact cost accounting for the preregistered cost--budget phase map."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ActionAllocation:
    """An integer action mix with exact label-equivalent total cost."""

    budget_capacity: int
    target_group_action_share: float
    group_to_label_cost_ratio: float
    group_audit_count: int
    label_verification_count: int
    realized_group_action_share: float
    total_cost_used: float
    audit_pool_saturated: bool


def allocate_action_mix(
    budget_capacity: int,
    target_group_action_share: float,
    group_to_label_cost_ratio: float,
    *,
    max_group_audits: int,
) -> ActionAllocation:
    """Plan a fixed action-share cell under unequal action costs.

    The budget is measured in label-verification cost units.  Cost ratios are
    parsed from their decimal string, so the preregistered ratios have exact
    rational accounting.  The group count is constrained to a denominator
    multiple of the reduced ratio; the remaining budget is therefore an exact
    integer number of label verifications.
    """

    if budget_capacity < 1:
        raise ValueError("budget_capacity must be positive")
    share = Fraction(str(target_group_action_share))
    ratio = Fraction(str(group_to_label_cost_ratio))
    if not 0 <= share < 1:
        raise ValueError("target_group_action_share must lie in [0, 1)")
    if ratio <= 0:
        raise ValueError("group_to_label_cost_ratio must be positive")
    if max_group_audits < 0:
        raise ValueError("max_group_audits must be nonnegative")

    if share == 0:
        group_count = 0
        saturated = False
    else:
        continuous_actions = Fraction(budget_capacity, 1) / (
            1 - share + ratio * share
        )
        continuous_groups = share * continuous_actions

        # If ratio=p/q, choosing group_count as a multiple of q makes its cost
        # integral.  The rest of the integer budget can then be spent exactly
        # on labels without floating-point tolerances.
        quantum = ratio.denominator
        target_quanta = continuous_groups / quantum
        affordable_quanta = int(Fraction(budget_capacity, 1) / ratio) // quantum
        pool_quanta = max_group_audits // quantum
        maximum_quanta = min(affordable_quanta, pool_quanta)
        lower = target_quanta.numerator // target_quanta.denominator
        candidate_quanta = {
            min(maximum_quanta, max(0, lower)),
            min(maximum_quanta, max(0, lower + 1)),
        }

        def share_error(quanta: int) -> tuple[Fraction, Fraction, int]:
            groups = quanta * quantum
            labels = Fraction(budget_capacity, 1) - ratio * groups
            actions = groups + labels
            realized = Fraction(groups, 1) / actions if actions else Fraction(0)
            return abs(realized - share), abs(Fraction(quanta) - target_quanta), -quanta

        chosen_quanta = min(candidate_quanta, key=share_error)
        group_count = chosen_quanta * quantum
        saturated = (
            pool_quanta < affordable_quanta
            and Fraction(pool_quanta * quantum, 1) < continuous_groups
        )

    remaining = Fraction(budget_capacity, 1) - ratio * group_count
    if remaining.denominator != 1:
        raise AssertionError("the rational action quantum must leave integral cost")
    label_count = remaining.numerator
    if label_count < 0:
        raise AssertionError("planned action cost exceeds the budget")

    action_count = group_count + label_count
    realized_share = group_count / action_count if action_count else 0.0
    total_cost = ratio * group_count + label_count
    if total_cost != budget_capacity:
        raise AssertionError("planned action mix must use the exact budget")

    return ActionAllocation(
        budget_capacity=budget_capacity,
        target_group_action_share=float(share),
        group_to_label_cost_ratio=float(ratio),
        group_audit_count=group_count,
        label_verification_count=label_count,
        realized_group_action_share=float(realized_share),
        total_cost_used=float(total_cost),
        audit_pool_saturated=saturated,
    )
