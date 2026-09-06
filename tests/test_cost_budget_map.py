import pytest

from robust_verify.cost_budget_map import allocate_action_mix


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("share", [0.0, 0.25, 0.5, 0.75])
def test_phase_map_allocations_use_exact_cost(ratio: float, share: float) -> None:
    allocation = allocate_action_mix(
        1000,
        share,
        ratio,
        max_group_audits=100_000,
    )
    assert allocation.total_cost_used == 1000.0
    assert allocation.group_audit_count >= 0
    assert allocation.label_verification_count >= 0
    assert allocation.audit_pool_saturated is False
    assert allocation.realized_group_action_share == pytest.approx(share, abs=0.002)


def test_zero_share_is_the_same_label_only_control_for_every_cost() -> None:
    low_cost = allocate_action_mix(137, 0.0, 0.1, max_group_audits=10)
    high_cost = allocate_action_mix(137, 0.0, 2.0, max_group_audits=10)
    assert low_cost.group_audit_count == high_cost.group_audit_count == 0
    assert low_cost.label_verification_count == high_cost.label_verification_count == 137


def test_validation_pool_saturation_is_recorded_and_cost_is_reallocated() -> None:
    allocation = allocate_action_mix(
        1000,
        0.75,
        0.1,
        max_group_audits=205,
    )
    assert allocation.audit_pool_saturated is True
    assert allocation.group_audit_count == 200
    assert allocation.label_verification_count == 980
    assert allocation.total_cost_used == 1000.0


@pytest.mark.parametrize(
    ("budget", "share", "ratio", "pool"),
    [(0, 0.25, 1.0, 10), (10, -0.1, 1.0, 10), (10, 1.0, 1.0, 10), (10, 0.5, 0.0, 10)],
)
def test_invalid_allocation_inputs_are_rejected(
    budget: int, share: float, ratio: float, pool: int
) -> None:
    with pytest.raises(ValueError):
        allocate_action_mix(budget, share, ratio, max_group_audits=pool)
