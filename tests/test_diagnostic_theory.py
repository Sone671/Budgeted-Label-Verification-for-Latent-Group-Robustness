from __future__ import annotations

from fractions import Fraction

import pytest

from robust_verify.diagnostic_theory import (
    OneActionInstance,
    enumerate_instance,
    pairwise_interaction_remainder_bound,
    same_xy_marginal,
    verify_logistic_embedding,
)


@pytest.mark.parametrize(("budget", "anchor_mass"), [(2, 40), (3, 90), (4, 160)])
def test_all_queries_obey_the_two_world_utility_inequality(
    budget: int, anchor_mass: int
) -> None:
    instance = OneActionInstance(budget, anchor_mass)
    report = enumerate_instance(instance)
    assert report["max_two_world_utility_sum"] == 1
    assert same_xy_marginal(instance)


@pytest.mark.parametrize(("budget", "anchor_mass"), [(2, 40), (3, 90), (4, 160)])
def test_balanced_randomization_attains_the_minimax_bound(
    budget: int, anchor_mass: int
) -> None:
    instance = OneActionInstance(budget, anchor_mass)
    report = enumerate_instance(instance)
    assert report["balanced_randomized_utilities"] == (Fraction(1, 2), Fraction(1, 2))
    assert report["balanced_randomized_worst_regret"] == instance.minimax_regret
    assert instance.minimax_regret == Fraction(
        anchor_mass - budget, 2 * (anchor_mass + budget)
    )


def test_pairwise_stability_instantiates_the_batch_remainder() -> None:
    assert pairwise_interaction_remainder_bound(10, 1000, 4.0) == pytest.approx(0.00018)
    assert pairwise_interaction_remainder_bound(1, 1000, 4.0) == 0.0


def test_invalid_instance_is_rejected() -> None:
    with pytest.raises(ValueError):
        OneActionInstance(budget=1, anchor_mass=100)
    with pytest.raises(ValueError):
        OneActionInstance(budget=4, anchor_mass=4)


@pytest.mark.parametrize(("budget", "anchor_mass"), [(2, 40), (3, 90), (4, 160)])
def test_strongly_convex_logistic_embedding_survives_a_dense_perturbation(
    budget: int, anchor_mass: int
) -> None:
    instance = OneActionInstance(budget, anchor_mass)
    query = instance.balanced_strategy()[0][1]
    assert verify_logistic_embedding(instance, query)
    assert verify_logistic_embedding(instance, query, perturbation=1e-3)
