"""Finite witnesses for two-action budget-allocation non-identifiability.

The legacy witness uses a three-valued audit signal.  The stronger witness in
this module uses a single binary attribute audit and keeps the complete test
``(X, Y)`` marginal fixed: its two worlds differ only in a hidden binary test
attribute assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import ceil


A_ATOMS = (0, 1, 2)
B_ATOMS = (3, 4)
ALL_ATOMS = A_ATOMS + B_ATOMS


@dataclass(frozen=True)
class SameMarginalWitness:
    """Parameters for the binary-audit, same-test-marginal construction.

    ``label_capacity`` is the maximum number of label verifications without an
    audit.  ``post_audit_capacity`` is the remaining number after buying the
    binary group audit.  These capacities support unequal action costs through
    ``floor(B / c_y)`` and ``floor((B - c_g) / c_y)``.
    """

    label_capacity: int = 3
    post_audit_capacity: int = 2
    anchor_count: int = 47

    def __post_init__(self) -> None:
        n = self.label_capacity
        r = self.post_audit_capacity
        if n < 3:
            raise ValueError("label_capacity must be at least 3")
        if not (n / 2 < r < n):
            raise ValueError(
                "post_audit_capacity must satisfy label_capacity / 2 < "
                "post_audit_capacity < label_capacity"
            )
        if self.anchor_count < minimum_anchor_count(n, r):
            raise ValueError(
                "anchor_count is too small for the normalized utility bound; "
                f"need at least {minimum_anchor_count(n, r)}"
            )

    @property
    def a0_atoms(self) -> tuple[int, ...]:
        return tuple(range(self.post_audit_capacity))

    @property
    def a1_atoms(self) -> tuple[int, ...]:
        start = self.post_audit_capacity
        return tuple(range(start, 2 * start))

    @property
    def c_atoms(self) -> tuple[int, ...]:
        start = 2 * self.post_audit_capacity
        return tuple(range(start, start + self.label_capacity))

    @property
    def all_atoms(self) -> tuple[int, ...]:
        return self.a0_atoms + self.a1_atoms + self.c_atoms

    @property
    def oracle_world0(self) -> Fraction:
        m = self.anchor_count
        return Fraction(m, m + self.post_audit_capacity + self.label_capacity)

    @property
    def oracle_world1(self) -> Fraction:
        m = self.anchor_count
        return Fraction(m, m + 2 * self.post_audit_capacity)

    @property
    def randomized_regret_bound(self) -> Fraction:
        """Tight unnormalised worst-world allocation-regret lower bound."""

        m = self.anchor_count
        return Fraction(
            m,
            2 * m + self.label_capacity + 3 * self.post_audit_capacity,
        )


def minimum_anchor_count(label_capacity: int, post_audit_capacity: int) -> int:
    """Sufficient anchor count for the general normalized utility proof."""

    n = label_capacity
    r = post_audit_capacity
    if n < 3 or not (n / 2 < r < n):
        raise ValueError("capacities must satisfy n >= 3 and n / 2 < r < n")
    no_audit = Fraction(
        n * (r + n) + 4 * r * r * (n - 1),
        2 * r - n,
    )
    after_audit = Fraction(
        n * (r - 1) * (r + n) + 2 * r * r,
        n - r,
    )
    return max(ceil(no_audit), ceil(after_audit))


def _subsets(atoms: tuple[int, ...], maximum_size: int):
    for size in range(maximum_size + 1):
        yield from combinations(atoms, size)


def same_marginal_utilities(
    witness: SameMarginalWitness,
    query: tuple[int, ...],
    audit_response: int,
) -> tuple[Fraction, Fraction]:
    """Return exact WGA improvements in the two hidden-group worlds.

    The test examples and labels are identical in both worlds.  In world 0,
    ``place=1`` selects one of two A blocks according to the audited binary
    response.  In world 1, ``place=1`` selects the C block.  Correct anchors in
    ``place=0`` make the displayed quantities literal worst-group accuracies.
    """

    if audit_response not in (0, 1):
        raise ValueError("the audited place attribute must be binary")
    query_set = set(query)
    if not query_set.issubset(witness.all_atoms):
        raise ValueError("query contains an atom outside the witness")

    active = set(witness.a0_atoms if audit_response == 0 else witness.a1_atoms)
    all_atoms = set(witness.all_atoms)
    c_atoms = set(witness.c_atoms)
    a_atoms = set(witness.a0_atoms + witness.a1_atoms)
    m = witness.anchor_count
    r = witness.post_audit_capacity
    n = witness.label_capacity

    world0_target = Fraction(len(query_set & active), r)
    world0_complement = Fraction(
        m + len(query_set & (all_atoms - active)),
        m + r + n,
    )
    world0 = min(world0_target, world0_complement)

    world1_target = Fraction(len(query_set & c_atoms), n)
    world1_complement = Fraction(m + len(query_set & a_atoms), m + 2 * r)
    world1 = min(world1_target, world1_complement)
    return world0, world1


def same_marginal_deterministic_points(
    witness: SameMarginalWitness,
) -> list[tuple[Fraction, Fraction, str]]:
    """Enumerate a superset of all deterministic sequential-policy vertices."""

    points: list[tuple[Fraction, Fraction, str]] = []
    atoms = witness.all_atoms
    for query in _subsets(atoms, witness.label_capacity):
        u00, u1 = same_marginal_utilities(witness, query, 0)
        u01, _ = same_marginal_utilities(witness, query, 1)
        points.append(((u00 + u01) / 2, u1, f"labels:{query}"))

    post_audit_queries = tuple(_subsets(atoms, witness.post_audit_capacity))
    for query0 in post_audit_queries:
        u00, u10 = same_marginal_utilities(witness, query0, 0)
        for query1 in post_audit_queries:
            u01, u11 = same_marginal_utilities(witness, query1, 1)
            points.append(
                (
                    (u00 + u01) / 2,
                    (u10 + u11) / 2,
                    f"audit:{query0}|{query1}",
                )
            )
    return points


def same_marginal_test_support(
    witness: SameMarginalWitness,
    world: int,
    audit_response: int,
) -> tuple[tuple[str, int, int, int], ...]:
    """Return ``(feature, label, weight, place)`` rows for a test world."""

    if world not in (0, 1) or audit_response not in (0, 1):
        raise ValueError("world and audit_response must both be binary")
    if world == 0:
        place_one = set(
            witness.a0_atoms if audit_response == 0 else witness.a1_atoms
        )
    else:
        place_one = set(witness.c_atoms)

    rows = [
        (f"repairable_{atom}", 1, 1, int(atom in place_one))
        for atom in witness.all_atoms
    ]
    rows.append(("correct_positive_anchor", 1, witness.anchor_count, 0))
    # These fixed anchors make both y x place groups non-empty and always
    # correct; they do not alter the two vulnerable positive-label groups.
    rows.extend(
        [
            ("correct_negative_place0", 0, 1, 0),
            ("correct_negative_place1", 0, 1, 1),
        ]
    )
    return tuple(rows)


def verify_same_marginal_lower_bound(
    witness: SameMarginalWitness | None = None,
) -> dict[str, float | int | str]:
    """Machine-check the binary-audit, same-test-marginal theorem."""

    witness = witness or SameMarginalWitness()
    points = same_marginal_deterministic_points(witness)
    oracle0 = max(point[0] for point in points)
    oracle1 = max(point[1] for point in points)
    maximum_normalized_sum = max(
        point[0] / witness.oracle_world0 + point[1] / witness.oracle_world1
        for point in points
    )
    best_deterministic_regret = min(
        max(oracle0 - point[0], oracle1 - point[1]) for point in points
    )

    supports = [
        same_marginal_test_support(witness, world, response)
        for world in (0, 1)
        for response in (0, 1)
    ]
    xy_marginals = {
        tuple((feature, label, weight) for feature, label, weight, _ in support)
        for support in supports
    }

    assert len(xy_marginals) == 1
    assert oracle0 == witness.oracle_world0
    assert oracle1 == witness.oracle_world1
    assert maximum_normalized_sum <= 1

    return {
        "deterministic_policy_count": len(points),
        "anchor_count": witness.anchor_count,
        "minimum_anchor_count": minimum_anchor_count(
            witness.label_capacity, witness.post_audit_capacity
        ),
        "same_test_xy_marginal": len(xy_marginals) == 1,
        "audit_cardinality": 2,
        "oracle_world0": float(oracle0),
        "oracle_world0_exact": str(oracle0),
        "oracle_world1": float(oracle1),
        "oracle_world1_exact": str(oracle1),
        "maximum_normalized_utility_sum": float(maximum_normalized_sum),
        "maximum_normalized_utility_sum_exact": str(maximum_normalized_sum),
        "best_deterministic_worst_regret": float(best_deterministic_regret),
        "best_deterministic_worst_regret_exact": str(best_deterministic_regret),
        "tight_randomized_worst_regret": float(
            witness.randomized_regret_bound
        ),
        "tight_randomized_worst_regret_exact": str(
            witness.randomized_regret_bound
        ),
    }


@dataclass(frozen=True)
class UtilityPoint:
    policy: str
    world0: float
    world1: float

    @property
    def regret_sum(self) -> float:
        return 2.0 - self.world0 - self.world1

    @property
    def worst_regret(self) -> float:
        return max(1.0 - self.world0, 1.0 - self.world1)


def deterministic_utility_points() -> list[UtilityPoint]:
    """Enumerate deterministic policies for the budget-two witness.

    A no-audit policy verifies at most two atoms.  An audit-first policy sees
    H in {0,1,2} and has budget for one label verification.  Label-first then
    audit is dominated but is already covered by the no-audit subsets of size
    at most one for the utility calculation.
    """

    points: list[UtilityPoint] = []
    for size in range(3):
        for query in combinations(ALL_ATOMS, size):
            world0 = len(set(query) & set(A_ATOMS)) / 3.0
            world1 = len(set(query) & set(B_ATOMS)) / 2.0
            points.append(
                UtilityPoint(f"labels:{','.join(map(str, query))}", world0, world1)
            )

    for response in product(ALL_ATOMS, repeat=3):
        world0 = sum(response[h] == h for h in A_ATOMS) / 3.0
        world1 = sum(atom in B_ATOMS for atom in response) / 6.0
        points.append(
            UtilityPoint(
                f"audit_then:{','.join(map(str, response))}", world0, world1
            )
        )
    return points


def verify_two_action_lower_bound() -> dict[str, float | int]:
    """Return exact diagnostics for the tight 1/2 randomized lower bound."""

    points = deterministic_utility_points()
    oracle0 = max(point.world0 for point in points)
    oracle1 = max(point.world1 for point in points)
    maximum_utility_sum = max(point.world0 + point.world1 for point in points)
    best_deterministic_worst_regret = min(point.worst_regret for point in points)

    # Equal mixing of the two world-optimal extreme policies yields (1/2,1/2).
    tight_mixture_world0 = 0.5
    tight_mixture_world1 = 0.5
    tight_randomized_worst_regret = max(
        oracle0 - tight_mixture_world0, oracle1 - tight_mixture_world1
    )
    return {
        "deterministic_policy_count": len(points),
        "oracle_world0": oracle0,
        "oracle_world1": oracle1,
        "maximum_deterministic_utility_sum": maximum_utility_sum,
        "best_deterministic_worst_regret": best_deterministic_worst_regret,
        "tight_randomized_worst_regret": tight_randomized_worst_regret,
    }
