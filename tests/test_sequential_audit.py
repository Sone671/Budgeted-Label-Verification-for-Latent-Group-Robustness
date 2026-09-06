from fractions import Fraction

import numpy as np
import pytest

from robust_verify.action_value import (
    ActionValueEstimate,
    certify_action,
    build_legal_state_features,
    proxy_action_values,
)
from robust_verify.sequential_audit import (
    AuditAction,
    SequentialAuditConfig,
    SequentialAuditState,
    apply_action,
    make_batch_action,
    replay,
    state_from_json,
    state_to_json,
)
from robust_verify.trajectory_oracle import (
    TreeNode,
    TreeTransition,
    classify_policy_decision,
    solve_dynamic_program,
)


@pytest.fixture
def config() -> SequentialAuditConfig:
    return SequentialAuditConfig(
        budget=Fraction(3, 1),
        task_label_cost=Fraction(1, 1),
        attribute_cost=Fraction(1, 2),
        task_pool_size=5,
        attribute_pool_size=4,
        max_rounds=4,
    )


def test_stop_is_absorbing_and_zero_cost(config: SequentialAuditConfig) -> None:
    state = SequentialAuditState.initial([0, 1, 0, 1, 0])
    stopped = apply_action(state, AuditAction.stop(), config)
    assert stopped.stopped
    assert stopped.spent_cost == 0
    with pytest.raises(RuntimeError, match="after Stop"):
        apply_action(
            stopped,
            make_batch_action("verify_task", [0], config),
            config,
            responses=[1],
        )


def test_exact_costs_no_repeats_and_budget(config: SequentialAuditConfig) -> None:
    state = SequentialAuditState.initial([0, 0, 1, 1, 0])
    group = make_batch_action("annotate_group", [0, 1], config)
    state = apply_action(state, group, config, responses=[1, 0])
    assert state.spent_cost == Fraction(1, 1)
    assert state.audited_attributes == {0: 1, 1: 0}
    with pytest.raises(ValueError, match="repeats"):
        apply_action(state, make_batch_action("annotate_group", [1], config), config, responses=[1])
    too_expensive = make_batch_action("verify_task", [0, 1, 2], config)
    with pytest.raises(ValueError, match="budget"):
        apply_action(state, too_expensive, config, responses=[1, 1, 0])


def test_json_roundtrip_and_transcript_replay(config: SequentialAuditConfig) -> None:
    initial = np.asarray([0, 0, 1, 1, 0])
    first = apply_action(
        SequentialAuditState.initial(initial),
        make_batch_action("annotate_group", [2], config),
        config,
        responses=[1],
    )
    second = apply_action(
        first,
        make_batch_action("verify_task", [3], config),
        config,
        responses=[0],
    )
    restored = state_from_json(state_to_json(second))
    replayed = replay(initial, config, second.transcript)
    assert restored.current_labels.tolist() == [0, 0, 1, 0, 0]
    assert restored.spent_cost == second.spent_cost == replayed.spent_cost
    assert restored.audited_attributes == replayed.audited_attributes == {2: 1}
    assert restored.verified_ids == replayed.verified_ids == frozenset({3})
    assert tuple(event.action for event in restored.transcript) == tuple(
        event.action for event in replayed.transcript
    )


def test_legal_features_exclude_private_child_utility(config: SequentialAuditConfig) -> None:
    state = apply_action(
        SequentialAuditState.initial([0, 1, 0, 1, 0]),
        make_batch_action("annotate_group", [0], config),
        config,
        responses=[1],
    )
    features = build_legal_state_features(
        state,
        config,
        public_task_features=np.eye(5),
        public_attribute_features=np.eye(4),
    )
    assert features.remaining_budget == Fraction(5, 2)
    assert features.audited_attribute_count == 1
    assert not hasattr(features, "wga")
    assert not hasattr(features, "private_child_utilities")
    assert np.array_equal(features.public_task_features, np.eye(5))


def test_action_value_certificates_continue_stop_and_unresolved(config: SequentialAuditConfig) -> None:
    task = make_batch_action("verify_task", [0], config)
    group = make_batch_action("annotate_group", [0], config)
    assert certify_action(proxy_action_values(
        task_action=task, group_action=group, task_value=0.4, group_value=0.1, error=0.1
    )).status == "continue"
    assert certify_action(proxy_action_values(
        task_action=task, group_action=group, task_value=-0.2, group_value=-0.1, error=0.05
    )).status == "stop"
    assert certify_action(proxy_action_values(
        task_action=task, group_action=group, task_value=0.05, group_value=-0.01, error=0.1
    )).status == "unresolved"


def test_continue_certificate_returns_action_with_positive_lower_bound(
    config: SequentialAuditConfig,
) -> None:
    task = make_batch_action("verify_task", [0], config)
    group = make_batch_action("annotate_group", [0], config)
    certificate = certify_action(
        [
            ActionValueEstimate(task, estimate=0.40, error=0.30),
            ActionValueEstimate(group, estimate=0.20, error=0.00),
        ]
    )
    assert certificate.status == "continue"
    assert certificate.selected_action == group
    assert certificate.selected_lower > 0.0


def test_mvp_rejects_non_binary_task_feedback(config: SequentialAuditConfig) -> None:
    state = SequentialAuditState.initial([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="binary"):
        apply_action(
            state,
            make_batch_action("verify_task", [0], config),
            config,
            responses=[2],
        )


def test_three_action_toy_oracle_has_non_degenerate_regions(config: SequentialAuditConfig) -> None:
    task = make_batch_action("verify_task", [0], config)
    group = make_batch_action("annotate_group", [0], config)
    nodes = {
        "terminal": TreeNode("terminal"),
        "r": TreeNode("r", transitions=(TreeTransition(task, "terminal", 1.0), TreeTransition(group, "terminal", 0.2))),
        "g": TreeNode("g", transitions=(TreeTransition(task, "terminal", 0.1), TreeTransition(group, "terminal", 1.0))),
        "s": TreeNode("s", transitions=(TreeTransition(task, "terminal", -0.1), TreeTransition(group, "terminal", -0.2))),
    }
    r_solution = solve_dynamic_program(nodes, "r", max_depth=1).root
    g_solution = solve_dynamic_program(nodes, "g", max_depth=1).root
    s_solution = solve_dynamic_program(nodes, "s", max_depth=1).root
    assert r_solution.action == task
    assert g_solution.action == group
    assert s_solution.action == AuditAction.stop()
    assert classify_policy_decision(AuditAction.stop(), r_solution) == "premature_stop"
    assert classify_policy_decision(task, s_solution) == "bad_continue"
