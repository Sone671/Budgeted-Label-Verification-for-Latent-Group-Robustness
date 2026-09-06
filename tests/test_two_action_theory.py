import pytest

from robust_verify.two_action_theory import (
    SameMarginalWitness,
    minimum_anchor_count,
    same_marginal_test_support,
    verify_same_marginal_lower_bound,
    verify_two_action_lower_bound,
)


def test_two_action_witness_has_tight_half_regret_bound() -> None:
    result = verify_two_action_lower_bound()
    assert result["oracle_world0"] == 1.0
    assert result["oracle_world1"] == 1.0
    assert result["maximum_deterministic_utility_sum"] == 1.0
    assert result["best_deterministic_worst_regret"] == pytest.approx(2.0 / 3.0)
    assert result["tight_randomized_worst_regret"] == 0.5


def test_same_marginal_binary_audit_witness() -> None:
    result = verify_same_marginal_lower_bound()
    assert result["deterministic_policy_count"] == 905
    assert result["anchor_count"] == 47
    assert result["minimum_anchor_count"] == 47
    assert result["same_test_xy_marginal"] is True
    assert result["audit_cardinality"] == 2
    assert result["oracle_world0_exact"] == "47/52"
    assert result["oracle_world1_exact"] == "47/51"
    assert result["maximum_normalized_utility_sum_exact"] == "1"
    assert result["best_deterministic_worst_regret_exact"] == "10/17"
    assert result["tight_randomized_worst_regret_exact"] == "47/103"


def test_same_marginal_support_changes_only_hidden_place() -> None:
    witness = SameMarginalWitness()
    supports = [
        same_marginal_test_support(witness, world, response)
        for world in (0, 1)
        for response in (0, 1)
    ]
    xy = [
        tuple((feature, label, weight) for feature, label, weight, _ in support)
        for support in supports
    ]
    assert all(marginal == xy[0] for marginal in xy[1:])
    assert any(support != supports[0] for support in supports[1:])


def test_general_capacity_anchor_threshold() -> None:
    assert minimum_anchor_count(3, 2) == 47
    witness = SameMarginalWitness(
        label_capacity=4,
        post_audit_capacity=3,
        anchor_count=minimum_anchor_count(4, 3),
    )
    assert witness.randomized_regret_bound > 0


def test_invalid_capacity_or_anchor_is_rejected() -> None:
    with pytest.raises(ValueError):
        SameMarginalWitness(label_capacity=4, post_audit_capacity=2)
    with pytest.raises(ValueError):
        SameMarginalWitness(anchor_count=46)
