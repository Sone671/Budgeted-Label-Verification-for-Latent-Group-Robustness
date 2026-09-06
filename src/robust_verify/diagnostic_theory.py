"""Machine checks for the one-action diagnostic-paper lower bound.

This module deliberately excludes the group-audit/label-audit allocation
problem studied by the separate joint-budget manuscript.  The only action is
label verification, with a fixed cardinality budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OneActionInstance:
    """Two observationally equivalent worlds with disjoint repair blocks."""

    budget: int
    anchor_mass: int

    def __post_init__(self) -> None:
        if self.budget < 2:
            raise ValueError("The general construction requires budget >= 2.")
        minimum = max(self.budget + 1, self.budget * (self.budget - 1))
        if self.anchor_mass < minimum:
            raise ValueError(
                "anchor_mass must be at least max(B+1, B(B-1)); "
                f"got B={self.budget}, M={self.anchor_mass}."
            )

    @property
    def candidates(self) -> tuple[int, ...]:
        return tuple(range(2 * self.budget))

    def block(self, world: int) -> frozenset[int]:
        if world not in (0, 1):
            raise ValueError("world must be 0 or 1")
        start = world * self.budget
        return frozenset(range(start, start + self.budget))

    @property
    def oracle_value(self) -> Fraction:
        return Fraction(self.anchor_mass, self.anchor_mass + self.budget)

    @property
    def minimax_regret(self) -> Fraction:
        """Tight randomized regret for the balanced allocation strategy."""
        return self.oracle_value - Fraction(1, 2)

    def utility(self, query: Iterable[int], world: int) -> Fraction:
        selected = frozenset(int(index) for index in query)
        if len(selected) > self.budget:
            raise ValueError("query exceeds the verification budget")
        if not selected.issubset(self.candidates):
            raise ValueError("query contains an unknown candidate")
        active = self.block(world)
        active_repairs = len(selected.intersection(active))
        other_repairs = len(selected.difference(active))
        weak_accuracy = Fraction(active_repairs, self.budget)
        strong_accuracy = Fraction(
            self.anchor_mass + other_repairs,
            self.anchor_mass + self.budget,
        )
        return min(weak_accuracy, strong_accuracy)

    def regret(self, query: Iterable[int], world: int) -> Fraction:
        return self.oracle_value - self.utility(query, world)

    def feasible_queries(self) -> tuple[frozenset[int], ...]:
        result = []
        for size in range(self.budget + 1):
            result.extend(frozenset(items) for items in combinations(self.candidates, size))
        return tuple(result)

    def balanced_strategy(self) -> tuple[tuple[Fraction, frozenset[int]], ...]:
        """Return a strategy attaining expected utility 1/2 in both worlds."""
        left = self.block(0)
        right = self.block(1)
        low = self.budget // 2
        high = self.budget - low
        first = frozenset(sorted(left)[:low] + sorted(right)[:high])
        if low == high:
            return ((Fraction(1), first),)
        second = frozenset(sorted(left)[:high] + sorted(right)[:low])
        return ((Fraction(1, 2), first), (Fraction(1, 2), second))

    def strategy_utility(
        self,
        strategy: Iterable[tuple[Fraction, frozenset[int]]],
        world: int,
    ) -> Fraction:
        strategy = tuple(strategy)
        if sum((weight for weight, _ in strategy), Fraction(0)) != 1:
            raise ValueError("strategy weights must sum to one")
        return sum(
            (weight * self.utility(query, world) for weight, query in strategy),
            Fraction(0),
        )


def enumerate_instance(instance: OneActionInstance) -> dict[str, object]:
    queries = instance.feasible_queries()
    utility_sums = [
        instance.utility(query, 0) + instance.utility(query, 1) for query in queries
    ]
    worst_regrets = [
        max(instance.regret(query, 0), instance.regret(query, 1)) for query in queries
    ]
    strategy = instance.balanced_strategy()
    randomized_utilities = tuple(
        instance.strategy_utility(strategy, world) for world in (0, 1)
    )
    return {
        "budget": instance.budget,
        "anchor_mass": instance.anchor_mass,
        "num_deterministic_queries": len(queries),
        "oracle_value": instance.oracle_value,
        "max_two_world_utility_sum": max(utility_sums),
        "best_deterministic_worst_regret": min(worst_regrets),
        "balanced_randomized_utilities": randomized_utilities,
        "balanced_randomized_worst_regret": max(
            instance.oracle_value - value for value in randomized_utilities
        ),
        "theorem_minimax_regret": instance.minimax_regret,
    }


def same_xy_marginal(instance: OneActionInstance) -> bool:
    """Check that the worlds differ only in hidden group assignment."""
    def atoms(world: int) -> list[tuple[str, int, int, int]]:
        active = instance.block(world)
        rows: list[tuple[str, int, int, int]] = []
        for candidate in instance.candidates:
            group = 1 if candidate in active else 0
            rows.append((f"candidate_{candidate}", 1, 1, group))
        rows.append(("positive_anchor", 1, instance.anchor_mass, 0))
        rows.append(("negative_anchor_0", 0, 1, 2))
        rows.append(("negative_anchor_1", 0, 1, 3))
        return rows

    marginal_0 = sorted((name, label, mass) for name, label, mass, _ in atoms(0))
    marginal_1 = sorted((name, label, mass) for name, label, mass, _ in atoms(1))
    groups_nonempty = all(
        {group for _, _, _, group in atoms(world)} == {0, 1, 2, 3}
        for world in (0, 1)
    )
    return marginal_0 == marginal_1 and groups_nonempty


def pairwise_interaction_remainder_bound(
    budget: int, sample_size: int, interaction_constant: float
) -> float:
    """Bound |Gamma(Q)| from uniformly bounded discrete second differences."""
    if budget < 0 or sample_size <= 0 or interaction_constant < 0:
        raise ValueError("budget and interaction_constant must be nonnegative; N must be positive")
    return interaction_constant * budget * (budget - 1) / (2 * sample_size**2)


def verify_logistic_embedding(
    instance: OneActionInstance,
    query: Iterable[int],
    *,
    perturbation: float = 0.0,
    seed: int = 20260731,
) -> bool:
    """Fit the strongly convex one-hot logistic realization and check signs."""
    from sklearn.linear_model import LogisticRegression

    selected = frozenset(int(index) for index in query)
    if len(selected) > instance.budget or not selected.issubset(instance.candidates):
        raise ValueError("query is infeasible")
    dimension = 2 * instance.budget + 3
    features = np.eye(dimension, dtype=np.float64)
    if perturbation:
        rng = np.random.default_rng(seed)
        dense = rng.normal(size=(dimension, dimension)) / np.sqrt(dimension)
        features = features + float(perturbation) * dense
    labels = np.zeros(dimension, dtype=np.int64)
    labels[2 * instance.budget] = 1  # positive anchor
    for index in selected:
        labels[index] = 1

    model = LogisticRegression(
        fit_intercept=False,
        C=1.0,
        solver="lbfgs",
        max_iter=10_000,
        tol=1e-12,
    )
    model.fit(features, labels)
    predictions = model.predict(features)
    return bool(np.array_equal(predictions, labels))
