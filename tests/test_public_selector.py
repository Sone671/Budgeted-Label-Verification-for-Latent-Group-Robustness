import numpy as np

from robust_verify.public_selector import choose_public_action


def test_public_selector_can_only_use_purchased_group_responses() -> None:
    decision = choose_public_action(
        current_labels=[0, 1, 0, 1],
        train_probabilities=np.array(
            [[0.99, 0.01], [0.10, 0.90], [0.55, 0.45], [0.40, 0.60]]
        ),
        public_validation_correct=[1, 1, 0, 1, 0, 1],
        audited_indices=[1, 4],
        audited_groups=[1, 0],
        inclusion_probabilities=[0.5] * 6,
        candidate_task_ids=[0, 1, 2, 3],
        candidate_audit_ids=[0, 2],
        num_groups=2,
        remaining_budget=4.0,
        task_batch_cost=1.0,
        audit_batch_cost=1.0,
        eligible_masks=np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]),
    )
    assert decision.selected_kind in {"verify_task", "annotate_group", "stop"}
    assert decision.status in {"continue", "stop", "unresolved"}
    assert decision.group_wga_interval.lower <= decision.group_wga_interval.upper


def test_public_selector_stops_when_no_action_is_affordable() -> None:
    decision = choose_public_action(
        current_labels=[0, 1],
        train_probabilities=np.array([[0.5, 0.5], [0.5, 0.5]]),
        public_validation_correct=[1, 0],
        audited_indices=[],
        audited_groups=[],
        inclusion_probabilities=[0.5, 0.5],
        candidate_task_ids=[0, 1],
        candidate_audit_ids=[0],
        num_groups=2,
        remaining_budget=0.0,
        task_batch_cost=1.0,
        audit_batch_cost=1.0,
    )
    assert decision.status == "stop"
    assert decision.selected_kind == "stop"


def test_minimax_stop_fallback_is_not_mislabeled_as_certified_stop() -> None:
    decision = choose_public_action(
        current_labels=[0, 1],
        train_probabilities=np.array([[0.5, 0.5], [0.5, 0.5]]),
        public_validation_correct=[1, 0],
        audited_indices=[],
        audited_groups=[],
        inclusion_probabilities=[0.5, 0.5],
        candidate_task_ids=[0, 1],
        candidate_audit_ids=[0],
        num_groups=2,
        remaining_budget=2.0,
        task_batch_cost=1.0,
        audit_batch_cost=1.0,
    )
    assert decision.status == "unresolved"


def test_uncalibrated_verify_proxy_cannot_certify_repair() -> None:
    decision = choose_public_action(
        current_labels=[0, 0],
        train_probabilities=np.array([[0.01, 0.99], [0.01, 0.99]]),
        public_validation_correct=[1],
        audited_indices=[],
        audited_groups=[],
        inclusion_probabilities=[1.0],
        candidate_task_ids=[0, 1],
        candidate_audit_ids=[],
        num_groups=1,
        remaining_budget=2.0,
        task_batch_cost=1.0,
        audit_batch_cost=1.0,
    )
    assert decision.status == "unresolved"
    assert decision.selected_kind == "stop"


def test_public_verify_scores_control_candidate_order_when_enabled() -> None:
    decision = choose_public_action(
        current_labels=[0, 0],
        train_probabilities=np.full((2, 2), 0.5),
        public_validation_correct=[1],
        audited_indices=[],
        audited_groups=[],
        inclusion_probabilities=[1.0],
        candidate_task_ids=[0, 1],
        candidate_audit_ids=[],
        num_groups=1,
        remaining_budget=2.0,
        task_batch_cost=1.0,
        audit_batch_cost=1.0,
        allow_uncalibrated_verify=True,
        verify_scores=[1.0, 0.8],
    )
    assert decision.status == "continue"
    assert decision.selected_kind == "verify_task"
    assert decision.selected_ids == (0, 1)
