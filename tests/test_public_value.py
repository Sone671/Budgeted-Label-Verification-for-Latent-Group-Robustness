from fractions import Fraction

import numpy as np
import pytest

from robust_verify.public_value import (
    PublicDecisionNode,
    PublicOutcomeModel,
    PublicTransition,
    ValueInterval,
    audited_group_accuracy_intervals,
    bernoulli_dr_group_accuracy_intervals,
    bernoulli_ipw_group_accuracy_intervals,
    incremental_utility_interval,
    solve_robust_bellman,
    worst_group_accuracy_interval,
)
from robust_verify.sequential_audit import AuditAction


def action(kind: str, sample_id: int) -> AuditAction:
    return AuditAction(kind=kind, sample_ids=(sample_id,), cost=Fraction(1, 1))


def deterministic(
    candidate: AuditAction,
    child: str,
    lower: float,
    upper: float,
) -> PublicTransition:
    return PublicTransition(
        action=candidate,
        child_nodes=(child,),
        outcome_models=(PublicOutcomeModel((1.0,)),),
        immediate_value=ValueInterval(lower, upper),
    )


def test_random_audit_wga_intervals_are_simultaneous_and_vacuous_when_missing() -> None:
    correct = np.array([1, 1, 0, 1, 0, 0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    intervals = audited_group_accuracy_intervals(
        correct,
        groups,
        num_groups=3,
        delta=0.05,
        comparisons=4,
    )
    assert [(item.successes, item.count) for item in intervals] == [(2, 3), (1, 3), (0, 0)]
    assert intervals[2].value == ValueInterval(0.0, 1.0)
    wga = worst_group_accuracy_interval(intervals)
    assert wga.lower == 0.0
    assert 0.0 < wga.upper <= 1.0


def test_more_random_audits_tighten_group_accuracy_interval() -> None:
    small = audited_group_accuracy_intervals(
        [1, 1, 1, 0],
        [0, 0, 0, 0],
        num_groups=1,
    )[0].value
    large = audited_group_accuracy_intervals(
        [1, 1, 1, 0] * 25,
        [0, 0, 0, 0] * 25,
        num_groups=1,
    )[0].value
    assert large.radius < small.radius


def test_bernoulli_ipw_census_is_exact() -> None:
    intervals = bernoulli_ipw_group_accuracy_intervals(
        [1, 0, 1, 1],
        [0, 1, 2, 3],
        [0, 0, 1, 1],
        [1.0, 1.0, 1.0, 1.0],
        num_groups=2,
    )
    assert intervals[0].weighted_successes == 1.0
    assert intervals[0].weighted_mass == 2.0
    assert intervals[0].numerator == ValueInterval(0.25, 0.25)
    assert intervals[0].mass == ValueInterval(0.5, 0.5)
    assert intervals[0].value == ValueInterval(0.5, 0.5)
    assert intervals[1].value == ValueInterval(1.0, 1.0)
    assert worst_group_accuracy_interval(intervals) == ValueInterval(0.5, 0.5)


def test_bernoulli_ipw_only_consumes_purchased_group_labels() -> None:
    intervals = bernoulli_ipw_group_accuracy_intervals(
        [1, 0, 1, 0],
        [1, 3],
        [0, 1],
        [0.5, 0.5, 0.5, 0.5],
        num_groups=2,
    )
    assert [item.audited_count for item in intervals] == [1, 1]
    assert [item.weighted_mass for item in intervals] == [2.0, 2.0]
    assert [item.weighted_successes for item in intervals] == [0.0, 0.0]


def test_ipw_ratio_intervals_apply_success_mass_constraint() -> None:
    intervals = bernoulli_ipw_group_accuracy_intervals(
        [1, 0, 1, 1, 0, 1],
        [0, 2, 4],
        [0, 1, 0],
        [0.5] * 6,
        num_groups=2,
    )
    for interval in intervals:
        assert interval.numerator.upper <= interval.mass.upper
        assert interval.numerator.lower <= interval.mass.lower
        assert 0.0 <= interval.value.lower <= interval.value.upper <= 1.0


def test_dr_point_totals_match_aipw_formula() -> None:
    correct = np.array([1, 0, 1, 1])
    indices = np.array([0, 3])
    groups = np.array([0, 1])
    inclusion = np.array([0.5, 0.5, 0.5, 0.5])
    predictions = np.array(
        [
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
            [0.40, 0.60],
        ]
    )
    intervals = bernoulli_dr_group_accuracy_intervals(
        correct,
        indices,
        groups,
        inclusion,
        predictions,
        num_groups=2,
    )
    observed_group_0 = (groups == 0).astype(float)
    residuals = observed_group_0 - predictions[indices, 0]
    expected_mass_0 = predictions[:, 0].sum() + np.sum(residuals / inclusion[indices])
    expected_success_0 = np.dot(correct, predictions[:, 0]) + np.sum(
        correct[indices] * residuals / inclusion[indices]
    )
    assert intervals[0].weighted_mass == pytest.approx(expected_mass_0)
    assert intervals[0].weighted_successes == pytest.approx(expected_success_0)


def test_bernoulli_dr_census_is_exact_despite_imperfect_predictions() -> None:
    intervals = bernoulli_dr_group_accuracy_intervals(
        [1, 0, 1],
        [0, 1, 2],
        [0, 1, 1],
        [1.0, 1.0, 1.0],
        [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7]],
        num_groups=2,
    )
    assert intervals[0].weighted_mass == pytest.approx(1.0)
    assert intervals[0].weighted_successes == pytest.approx(1.0)
    assert intervals[0].value.radius == 0.0
    assert intervals[0].value.midpoint == pytest.approx(1.0)
    assert intervals[1].weighted_mass == pytest.approx(2.0)
    assert intervals[1].weighted_successes == pytest.approx(1.0)
    assert intervals[1].value.radius == 0.0
    assert intervals[1].value.midpoint == pytest.approx(0.5)


def test_empty_bernoulli_audit_returns_vacuous_accuracy_intervals() -> None:
    intervals = bernoulli_ipw_group_accuracy_intervals(
        [1, 0],
        [],
        [],
        [0.5, 0.5],
        num_groups=2,
    )
    assert all(interval.audited_count == 0 for interval in intervals)
    assert all(interval.value == ValueInterval(0.0, 1.0) for interval in intervals)


def test_more_declared_ipw_comparisons_widen_total_intervals() -> None:
    correct = np.tile([1, 0], 50)
    indices = np.arange(80)
    groups = np.tile([0, 1], 40)
    inclusion = np.full(100, 0.8)
    one = bernoulli_ipw_group_accuracy_intervals(
        correct,
        indices,
        groups,
        inclusion,
        num_groups=2,
        comparisons=1,
    )[0]
    many = bernoulli_ipw_group_accuracy_intervals(
        correct,
        indices,
        groups,
        inclusion,
        num_groups=2,
        comparisons=100,
    )[0]
    assert many.mass.radius > one.mass.radius
    assert many.numerator.radius > one.numerator.radius


@pytest.mark.parametrize(
    ("indices", "groups", "probabilities", "message"),
    [
        ([0, 0], [0, 1], [0.5, 0.5], "duplicates"),
        ([2], [0], [0.5, 0.5], "outside"),
        ([0], [0], [0.0, 0.5], "inclusion probabilities"),
        ([1], [0], [1.0, 0.5], "probability one"),
    ],
)
def test_bernoulli_ipw_rejects_invalid_design_inputs(
    indices, groups, probabilities, message
) -> None:
    with pytest.raises(ValueError, match=message):
        bernoulli_ipw_group_accuracy_intervals(
            [1, 0],
            indices,
            groups,
            probabilities,
            num_groups=2,
        )


def test_dr_rejects_invalid_or_ineligible_group_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        bernoulli_dr_group_accuracy_intervals(
            [1, 0],
            [0],
            [0],
            [0.5, 0.5],
            [[0.2, 0.2], [0.5, 0.5]],
            num_groups=2,
        )
    with pytest.raises(ValueError, match="outside eligible_masks"):
        bernoulli_dr_group_accuracy_intervals(
            [1, 0],
            [0],
            [0],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
            num_groups=2,
            eligible_masks=[[1, 0], [1, 1]],
        )


def test_incremental_utility_uses_worst_case_endpoints_and_cost() -> None:
    value = incremental_utility_interval(
        ValueInterval(0.40, 0.50),
        ValueInterval(0.55, 0.70),
        normalized_cost=0.05,
    )
    assert value.lower == pytest.approx(0.0)
    assert value.upper == pytest.approx(0.25)


def test_robust_bellman_certifies_continue_and_action_dominance() -> None:
    verify = action("verify_task", 0)
    audit = action("annotate_group", 0)
    nodes = {
        "terminal": PublicDecisionNode("terminal"),
        "root": PublicDecisionNode(
            "root",
            transitions=(
                deterministic(verify, "terminal", 0.20, 0.30),
                deterministic(audit, "terminal", 0.05, 0.10),
            ),
        ),
    }
    decision = solve_robust_bellman(nodes, "root", max_depth=1).root
    assert decision.certificate.status == "continue"
    assert decision.certificate.selected_action == verify
    assert decision.dominant_action == verify
    assert decision.minimax_action == verify


def test_robust_bellman_certifies_stop_and_preserves_unresolved() -> None:
    verify = action("verify_task", 0)
    audit = action("annotate_group", 0)
    terminal = PublicDecisionNode("terminal")
    stop_root = PublicDecisionNode(
        "stop",
        transitions=(
            deterministic(verify, "terminal", -0.30, -0.10),
            deterministic(audit, "terminal", -0.20, 0.00),
        ),
    )
    unresolved_root = PublicDecisionNode(
        "unresolved",
        transitions=(
            deterministic(verify, "terminal", -0.10, 0.20),
            deterministic(audit, "terminal", -0.05, 0.10),
        ),
    )
    stop = solve_robust_bellman(
        {"terminal": terminal, "stop": stop_root}, "stop", max_depth=1
    ).root
    unresolved = solve_robust_bellman(
        {"terminal": terminal, "unresolved": unresolved_root},
        "unresolved",
        max_depth=1,
    ).root
    assert stop.certificate.status == "stop"
    assert stop.dominant_action is None
    assert unresolved.certificate.status == "unresolved"
    assert unresolved.dominant_action is None


def test_audit_value_includes_future_feedback_and_probability_ambiguity() -> None:
    verify = action("verify_task", 0)
    audit = action("annotate_group", 0)
    nodes = {
        "terminal": PublicDecisionNode("terminal"),
        "good": PublicDecisionNode(
            "good", transitions=(deterministic(verify, "terminal", 0.80, 0.90),)
        ),
        "bad": PublicDecisionNode(
            "bad", transitions=(deterministic(verify, "terminal", -0.20, -0.10),)
        ),
        "root": PublicDecisionNode(
            "root",
            transitions=(
                PublicTransition(
                    action=audit,
                    child_nodes=("good", "bad"),
                    outcome_models=(
                        PublicOutcomeModel((0.25, 0.75)),
                        PublicOutcomeModel((0.75, 0.25)),
                    ),
                    immediate_value=ValueInterval(-0.10, -0.10),
                ),
            ),
        ),
    }
    decision = solve_robust_bellman(nodes, "root", max_depth=2).root
    audit_value = next(
        item.value for item in decision.action_values if item.action.kind == "annotate_group"
    )
    assert audit_value.lower == pytest.approx(0.10)
    assert audit_value.upper == pytest.approx(0.575)
    assert decision.certificate.status == "continue"
    assert decision.certificate.selected_action == audit


@pytest.mark.parametrize(
    ("correct", "groups", "message"),
    [
        ([1, 2], [0, 0], "binary"),
        ([1, 0], [0, 2], "outside"),
        ([1], [0.0], "integer"),
    ],
)
def test_random_audit_intervals_reject_invalid_inputs(correct, groups, message) -> None:
    with pytest.raises(ValueError, match=message):
        audited_group_accuracy_intervals(correct, groups, num_groups=2)
