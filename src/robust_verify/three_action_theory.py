"""Exact three-world R/G/Stop impossibility templates.

The module retains the initial decision-level and asymmetric WGA witnesses for
backward compatibility.  ``SharpThreeWorldWGATemplate`` adds a finite-batch,
state-dependent legal menu that realizes a genuinely three-way lower bound in
same-``(X,Y)`` worst-group accuracy.  Its full-plan net utility table is the
one-third scaling of

    world      R      G      S
    R-world    1      0      0
    G-world    0      1      0
    S-world   -1     -1      0

The sharp construction therefore has exact minimax regret 2/9.  A second
capacity-priced expansion permits every nonempty batch up to the original
three/two-label capacities and has exact minimax regret 1/5.  A separate
seven-atom construction has exact regret 16/81 under actual per-label prices;
a normal-form argument covers arbitrary R/G ordering under total budget three
and at most one audit.  Repeated audits and larger unrestricted budgets remain
open, and none of the results claims that the current experiments realize the
hard instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import ceil, exp, sqrt
from typing import Iterable, Mapping


WORLDS = ("R", "G", "S")
ACTIONS = ("R", "G", "S")


@dataclass(frozen=True)
class FiniteTreeRegretGame:
    """Finite regret game induced by an observationally equivalent policy tree.

    ``utility_table`` stores one row per world and one column per deterministic
    policy of a finite-horizon, finite-branching tree with finite action menus.
    When ``common_transcript`` is true,
    every behavioral randomized
    policy can be represented by a world-independent mixture over these pure
    policies.  The dual certificate

    ``min_p sum_w lambda[w] * regret(w, p)``

    is then a valid lower bound on worst-world regret for every randomized
    policy.  This is the finite-tree lifting step used by the expanded
    three-world construction below; it does not assume a particular WGA model.
    """

    worlds: tuple[str, ...]
    policies: tuple[str, ...]
    utility_table: tuple[tuple[Fraction, ...], ...]
    common_transcript: bool = True
    _world_lookup: dict[str, int] = field(init=False, repr=False, compare=False)
    _policy_lookup: dict[str, int] = field(init=False, repr=False, compare=False)
    _oracle_cache: dict[str, Fraction] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        worlds = tuple(self.worlds)
        policies = tuple(self.policies)
        table = tuple(
            tuple(Fraction(value) for value in row) for row in self.utility_table
        )
        if not worlds or not policies:
            raise ValueError("a regret game needs at least one world and policy")
        if len(set(worlds)) != len(worlds):
            raise ValueError("world names must be unique")
        if len(set(policies)) != len(policies):
            raise ValueError("policy names must be unique")
        if len(table) != len(worlds):
            raise ValueError("utility_table needs one row per world")
        if any(len(row) != len(policies) for row in table):
            raise ValueError("every utility row needs one value per policy")
        if not isinstance(self.common_transcript, bool):
            raise ValueError("common_transcript must be boolean")
        object.__setattr__(self, "worlds", worlds)
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "utility_table", table)
        object.__setattr__(self, "_world_lookup", {
            world: index for index, world in enumerate(worlds)
        })
        object.__setattr__(self, "_policy_lookup", {
            policy: index for index, policy in enumerate(policies)
        })
        object.__setattr__(self, "_oracle_cache", {})

    @classmethod
    def from_mapping(
        cls,
        table: Mapping[str, Mapping[str, Fraction]],
        *,
        common_transcript: bool = True,
    ) -> "FiniteTreeRegretGame":
        """Build a game from ``world -> policy -> utility`` rows."""

        if not table:
            raise ValueError("utility mapping cannot be empty")
        worlds = tuple(table)
        first_row = table[worlds[0]]
        policies = tuple(first_row)
        if not policies:
            raise ValueError("utility mapping needs at least one policy")
        rows: list[tuple[Fraction, ...]] = []
        expected = set(policies)
        for world in worlds:
            row = table[world]
            if set(row) != expected:
                raise ValueError("all worlds must expose the same policy names")
            rows.append(tuple(Fraction(row[policy]) for policy in policies))
        return cls(worlds, policies, tuple(rows), common_transcript)

    def _world_index(self, world: str) -> int:
        if world not in self._world_lookup:
            raise ValueError(f"unknown world: {world}")
        return self._world_lookup[world]

    def _policy_index(self, policy: str) -> int:
        if policy not in self._policy_lookup:
            raise ValueError(f"unknown policy: {policy}")
        return self._policy_lookup[policy]

    def utility(self, world: str, policy: str) -> Fraction:
        return self.utility_table[self._world_index(world)][self._policy_index(policy)]

    def oracle_value(self, world: str) -> Fraction:
        index = self._world_index(world)
        if world not in self._oracle_cache:
            self._oracle_cache[world] = max(self.utility_table[index])
        return self._oracle_cache[world]

    def oracle_values(self) -> dict[str, Fraction]:
        return {world: self.oracle_value(world) for world in self.worlds}

    def oracle_policies(self, world: str) -> tuple[str, ...]:
        value = self.oracle_value(world)
        return tuple(
            policy
            for policy in self.policies
            if self.utility(world, policy) == value
        )

    def regret(self, world: str, policy: str) -> Fraction:
        return self.oracle_value(world) - self.utility(world, policy)

    def regret_table(self) -> dict[str, dict[str, Fraction]]:
        return {
            world: {policy: self.regret(world, policy) for policy in self.policies}
            for world in self.worlds
        }

    def pure_regret_sums(self) -> dict[str, Fraction]:
        return {
            policy: sum(
                (self.regret(world, policy) for world in self.worlds), Fraction(0)
            )
            for policy in self.policies
        }

    def _validate_world_weights(
        self, world_weights: Mapping[str, Fraction]
    ) -> dict[str, Fraction]:
        if set(world_weights) - set(self.worlds):
            raise ValueError("world weights contain an unknown world")
        weights = {
            world: Fraction(world_weights.get(world, Fraction(0)))
            for world in self.worlds
        }
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("world weights must be nonnegative")
        if sum(weights.values(), Fraction(0)) != 1:
            raise ValueError("world weights must sum to one")
        return weights

    def weighted_regret_lower_bound(
        self, world_weights: Mapping[str, Fraction]
    ) -> Fraction:
        """Return the finite-game dual lower-bound certificate."""

        weights = self._validate_world_weights(world_weights)
        return min(
            sum(
                (weights[world] * self.regret(world, policy) for world in self.worlds),
                Fraction(0),
            )
            for policy in self.policies
        )

    def certified_minimax_lower_bound(
        self, world_weights: Mapping[str, Fraction] | None = None
    ) -> Fraction:
        """Return a valid minimax lower bound under common transcripts.

        Uniform world weights are used by default.  Optimizing the weights is
        the usual finite zero-sum LP dual; callers may provide any feasible
        certificate without requiring a floating-point solver.
        """

        if not self.common_transcript:
            raise ValueError(
                "a world-independent mixture requires common transcript laws"
            )
        if world_weights is None:
            weight = Fraction(1, len(self.worlds))
            world_weights = {world: weight for world in self.worlds}
        return self.weighted_regret_lower_bound(world_weights)

    def mixture_regrets(
        self, probabilities: Mapping[str, Fraction]
    ) -> dict[str, Fraction]:
        if set(probabilities) - set(self.policies):
            raise ValueError("mixture contains an unknown policy")
        probs = {
            policy: Fraction(probabilities.get(policy, Fraction(0)))
            for policy in self.policies
        }
        if any(probability < 0 for probability in probs.values()):
            raise ValueError("mixture probabilities must be nonnegative")
        if sum(probs.values(), Fraction(0)) != 1:
            raise ValueError("mixture probabilities must sum to one")
        return {
            world: sum(
                (probs[policy] * self.regret(world, policy) for policy in self.policies),
                Fraction(0),
            )
            for world in self.worlds
        }

    def certify_tight_mixture(
        self,
        probabilities: Mapping[str, Fraction],
        world_weights: Mapping[str, Fraction] | None = None,
    ) -> dict[str, object]:
        """Check matching primal/dual certificates for an exact minimax value."""

        lower = self.certified_minimax_lower_bound(world_weights)
        regrets = self.mixture_regrets(probabilities)
        upper = max(regrets.values())
        if upper != lower:
            raise AssertionError(
                f"mixture upper bound {upper} does not match dual lower bound {lower}"
            )
        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "regrets": regrets,
        }


@dataclass(frozen=True)
class FiniteHorizonBellmanTree:
    """Finite-horizon Bellman model for common-transcript recovery.

    ``nodes_by_depth`` describes a directed acyclic feedback tree.  The
    transition kernel is deliberately world independent: hidden deployment
    worlds may change rewards, but not the legal transcript law.  Rewards are
    one-step *net* utilities, so annotation costs can be included directly.
    Every node must expose ``stop_action``; an empty transition map denotes an
    action that terminates after its immediate reward.

    The class is a machine-checkable form of the conditional recovery result:
    if one common score is within ``E(h)`` of every world's optimal Bellman
    action value, its greedy policy has regret at most twice the expected sum
    of the nodewise errors along its realized trajectory.  This is a recovery
    theorem, not a lower-bound theorem for unrestricted action spaces.
    """

    worlds: tuple[str, ...]
    root: str
    nodes_by_depth: tuple[tuple[str, ...], ...]
    actions: Mapping[str, tuple[str, ...]]
    transitions: Mapping[tuple[str, str], Mapping[str, Fraction]]
    rewards: Mapping[tuple[str, str, str], Fraction]
    stop_action: str = "S"
    _depth_lookup: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        worlds = tuple(self.worlds)
        layers = tuple(tuple(layer) for layer in self.nodes_by_depth)
        if not worlds or len(set(worlds)) != len(worlds):
            raise ValueError("world names must be nonempty and unique")
        if not layers:
            raise ValueError("a Bellman tree needs at least one depth layer")
        if not self.stop_action:
            raise ValueError("stop_action must be nonempty")

        depth_lookup: dict[str, int] = {}
        for depth, layer in enumerate(layers):
            if not layer or len(set(layer)) != len(layer):
                raise ValueError("each tree layer needs unique nonempty nodes")
            for node in layer:
                if node in depth_lookup:
                    raise ValueError("a node cannot occur at two depths")
                depth_lookup[node] = depth
        if self.root not in depth_lookup or depth_lookup[self.root] != 0:
            raise ValueError("root must be a node at depth zero")

        all_nodes = set(depth_lookup)
        raw_actions = dict(self.actions)
        if set(raw_actions) != all_nodes:
            raise ValueError("actions must be supplied for exactly every node")
        normalized_actions: dict[str, tuple[str, ...]] = {}
        for node in depth_lookup:
            node_actions = tuple(raw_actions[node])
            if (
                not node_actions
                or len(set(node_actions)) != len(node_actions)
                or self.stop_action not in node_actions
            ):
                raise ValueError(
                    "every node needs unique actions including the stop action"
                )
            normalized_actions[node] = node_actions

        expected_transition_keys = {
            (node, action)
            for node, node_actions in normalized_actions.items()
            for action in node_actions
        }
        raw_transitions = dict(self.transitions)
        if set(raw_transitions) - expected_transition_keys:
            raise ValueError("transitions contain an unknown node/action pair")
        normalized_transitions: dict[tuple[str, str], dict[str, Fraction]] = {}
        max_depth = len(layers) - 1
        for node, node_actions in normalized_actions.items():
            for action in node_actions:
                key = (node, action)
                transition = raw_transitions.get(key, {})
                normalized = {
                    child: Fraction(probability)
                    for child, probability in dict(transition).items()
                }
                if any(probability < 0 for probability in normalized.values()):
                    raise ValueError("transition probabilities must be nonnegative")
                if normalized and sum(normalized.values(), Fraction(0)) != 1:
                    raise ValueError("nonempty transitions must sum to one")
                if node in depth_lookup and depth_lookup[node] == max_depth and normalized:
                    raise ValueError("deepest-layer actions cannot have children")
                for child in normalized:
                    if child not in depth_lookup:
                        raise ValueError("transition points to an unknown child")
                    if depth_lookup[child] != depth_lookup[node] + 1:
                        raise ValueError("transitions must advance exactly one layer")
                normalized_transitions[key] = normalized

        expected_reward_keys = {
            (world, node, action)
            for world in worlds
            for node, node_actions in normalized_actions.items()
            for action in node_actions
        }
        raw_rewards = dict(self.rewards)
        if set(raw_rewards) != expected_reward_keys:
            missing = expected_reward_keys - set(raw_rewards)
            extra = set(raw_rewards) - expected_reward_keys
            if missing:
                raise ValueError("rewards are missing a world/node/action value")
            raise ValueError(f"rewards contain unknown keys: {sorted(extra)!r}")
        normalized_rewards = {
            key: Fraction(value) for key, value in raw_rewards.items()
        }
        for node in depth_lookup:
            if normalized_transitions[(node, self.stop_action)]:
                raise ValueError("the stop action must be absorbing")
            if any(
                normalized_rewards[(world, node, self.stop_action)] != 0
                for world in worlds
            ):
                raise ValueError("the stop action must have zero remaining value")

        # A node that is listed in the tree must be reachable from the root;
        # otherwise its error certificate could silently be ignored.
        reachable = {self.root}
        for layer in layers[:-1]:
            for node in layer:
                if node not in reachable:
                    continue
                for action in normalized_actions[node]:
                    reachable.update(normalized_transitions[(node, action)])
        if reachable != all_nodes:
            raise ValueError("every declared node must be reachable from root")

        object.__setattr__(self, "worlds", worlds)
        object.__setattr__(self, "nodes_by_depth", layers)
        object.__setattr__(self, "actions", normalized_actions)
        object.__setattr__(self, "transitions", normalized_transitions)
        object.__setattr__(self, "rewards", normalized_rewards)
        object.__setattr__(self, "_depth_lookup", depth_lookup)

    @property
    def max_depth(self) -> int:
        return len(self.nodes_by_depth) - 1

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(node for layer in self.nodes_by_depth for node in layer)

    def _normalize_q_hat(
        self, q_hat: Mapping[tuple[str, str], Fraction]
    ) -> dict[tuple[str, str], Fraction]:
        values = {key: Fraction(value) for key, value in dict(q_hat).items()}
        expected = {
            (node, action)
            for node, node_actions in self.actions.items()
            for action in node_actions
        }
        if set(values) != expected:
            missing = expected - set(values)
            extra = set(values) - expected
            if missing:
                raise ValueError("q_hat is missing a node/action value")
            raise ValueError(f"q_hat contains unknown keys: {sorted(extra)!r}")
        return values

    def _normalize_errors(
        self,
        error_by_node: Mapping[str, Fraction] | Fraction,
    ) -> dict[str, Fraction]:
        if isinstance(error_by_node, Mapping):
            errors = {
                node: Fraction(value) for node, value in dict(error_by_node).items()
            }
            if set(errors) != set(self.nodes):
                raise ValueError("nodewise errors must cover exactly every node")
        else:
            error = Fraction(error_by_node)
            errors = {node: error for node in self.nodes}
        if any(value < 0 for value in errors.values()):
            raise ValueError("action-value errors must be nonnegative")
        return errors

    def optimal_bellman_values(
        self,
    ) -> tuple[
        dict[tuple[str, str, str], Fraction],
        dict[tuple[str, str], Fraction],
    ]:
        """Return exact world-specific ``Q*`` and ``V*`` tables."""

        q_star: dict[tuple[str, str, str], Fraction] = {}
        values: dict[tuple[str, str], Fraction] = {}
        for layer in reversed(self.nodes_by_depth):
            for node in layer:
                for world in self.worlds:
                    row: list[Fraction] = []
                    for action in self.actions[node]:
                        continuation = sum(
                            (
                                probability * values[(world, child)]
                                for child, probability in self.transitions[(node, action)].items()
                            ),
                            Fraction(0),
                        )
                        value = self.rewards[(world, node, action)] + continuation
                        q_star[(world, node, action)] = value
                        row.append(value)
                    values[(world, node)] = max(row)
        return q_star, values

    def greedy_policy(
        self, q_hat: Mapping[tuple[str, str], Fraction]
    ) -> dict[str, str]:
        """Choose the first legal action attaining the largest estimated value."""

        values = self._normalize_q_hat(q_hat)
        policy: dict[str, str] = {}
        for node in self.nodes:
            best_action = self.actions[node][0]
            best_value = values[(node, best_action)]
            for action in self.actions[node][1:]:
                value = values[(node, action)]
                if value > best_value:
                    best_action, best_value = action, value
            policy[node] = best_action
        return policy

    def policy_values(
        self, policy: Mapping[str, str]
    ) -> dict[tuple[str, str], Fraction]:
        """Evaluate a deterministic policy by backward induction."""

        selected = dict(policy)
        if set(selected) != set(self.nodes):
            raise ValueError("a policy must specify one action at every node")
        for node, action in selected.items():
            if action not in self.actions[node]:
                raise ValueError(f"illegal action {action!r} at node {node!r}")
        values: dict[tuple[str, str], Fraction] = {}
        for layer in reversed(self.nodes_by_depth):
            for node in layer:
                action = selected[node]
                continuation_by_world = {
                    world: sum(
                        (
                            probability * values[(world, child)]
                            for child, probability in self.transitions[(node, action)].items()
                        ),
                        Fraction(0),
                    )
                    for world in self.worlds
                }
                for world in self.worlds:
                    values[(world, node)] = (
                        self.rewards[(world, node, action)]
                        + continuation_by_world[world]
                    )
        return values

    def stopping_certificate(
        self,
        node: str,
        q_hat: Mapping[tuple[str, str], Fraction],
        error_by_node: Mapping[str, Fraction] | Fraction,
    ) -> dict[str, object]:
        """Classify a node as certified stop, continuation, or unresolved."""

        if node not in self._depth_lookup:
            raise ValueError(f"unknown node: {node}")
        values = self._normalize_q_hat(q_hat)
        errors = self._normalize_errors(error_by_node)
        continuation = [
            action for action in self.actions[node] if action != self.stop_action
        ]
        if not continuation:
            status = "stop_certified"
            upper = lower = None
        else:
            upper = max(values[(node, action)] + errors[node] for action in continuation)
            lower = max(values[(node, action)] - errors[node] for action in continuation)
            if upper <= 0:
                status = "stop_certified"
            elif lower > 0:
                status = "continue_certified"
            else:
                status = "unresolved"
        return {
            "node": node,
            "status": status,
            "max_continuation_upper": upper,
            "max_continuation_lower": lower,
        }

    def certify_greedy_recovery(
        self,
        q_hat: Mapping[tuple[str, str], Fraction],
        error_by_node: Mapping[str, Fraction] | Fraction,
        *,
        policy: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        """Check the Bellman error certificate and the resulting regret bound.

        The returned ``trajectory_regret_bound`` is the sharper
        ``2 E_{\\pi}[sum_t E(H_t)]`` quantity.  ``depthwise_regret_bound`` is a
        policy-independent, coarser bound obtained by taking the largest node
        error in each layer.
        """

        estimates = self._normalize_q_hat(q_hat)
        errors = self._normalize_errors(error_by_node)
        q_star, oracle_values = self.optimal_bellman_values()
        for world in self.worlds:
            for node in self.nodes:
                for action in self.actions[node]:
                    deviation = abs(
                        estimates[(node, action)] - q_star[(world, node, action)]
                    )
                    if deviation > errors[node]:
                        raise AssertionError(
                            "the supplied simultaneous Bellman error certificate fails "
                            f"at {(world, node, action)!r}: {deviation} > {errors[node]}"
                        )

        selected = (
            self.greedy_policy(estimates) if policy is None else dict(policy)
        )
        if policy is not None:
            greedy = self.greedy_policy(estimates)
            if selected != greedy:
                raise ValueError("the supplied policy is not greedy for q_hat")
        realized_values = self.policy_values(selected)

        rollout_error: dict[str, Fraction] = {}
        for layer in reversed(self.nodes_by_depth):
            for node in layer:
                action = selected[node]
                rollout_error[node] = errors[node] + sum(
                    (
                        probability * rollout_error[child]
                        for child, probability in self.transitions[(node, action)].items()
                    ),
                    Fraction(0),
                )
        expected_error = rollout_error[self.root]
        trajectory_bound = 2 * expected_error
        depthwise_bound = 2 * sum(
            (max(errors[node] for node in layer) for layer in self.nodes_by_depth),
            Fraction(0),
        )
        regrets = {
            world: oracle_values[(world, self.root)]
            - realized_values[(world, self.root)]
            for world in self.worlds
        }
        if any(regret < 0 or regret > trajectory_bound for regret in regrets.values()):
            raise AssertionError("the Bellman recovery bound failed")
        stopping = {
            node: self.stopping_certificate(node, estimates, errors)
            for node in self.nodes
        }
        return {
            "q_star": q_star,
            "oracle_values": {
                world: oracle_values[(world, self.root)] for world in self.worlds
            },
            "greedy_policy": selected,
            "policy_values": {
                world: realized_values[(world, self.root)] for world in self.worlds
            },
            "regrets": regrets,
            "node_errors": errors,
            "expected_error_mass": expected_error,
            "trajectory_regret_bound": trajectory_bound,
            "depthwise_regret_bound": depthwise_bound,
            "stopping_certificates": stopping,
        }


def _bellman_recovery_fixture() -> tuple[
    FiniteHorizonBellmanTree,
    dict[tuple[str, str], Fraction],
    dict[str, Fraction],
]:
    """Build a two-layer interleaved R/G/S tree used by the regression check."""

    layers = (("h0",), ("after_g0", "after_g1", "after_r"))
    node_actions = {
        "h0": ("S", "R", "G"),
        "after_g0": ("S", "R", "G"),
        "after_g1": ("S", "R", "G"),
        "after_r": ("S", "R", "G"),
    }
    transitions: dict[tuple[str, str], dict[str, Fraction]] = {
        ("h0", "R"): {"after_r": Fraction(1)},
        ("h0", "G"): {
            "after_g0": Fraction(1, 2),
            "after_g1": Fraction(1, 2),
        },
    }
    rewards: dict[tuple[str, str, str], Fraction] = {}
    for world in WORLDS:
        rewards[(world, "h0", "S")] = Fraction(0)
        rewards[(world, "h0", "R")] = Fraction(-1, 20)
        rewards[(world, "h0", "G")] = Fraction(-1, 20)
        for node in ("after_g0", "after_g1", "after_r"):
            rewards[(world, node, "S")] = Fraction(0)
            rewards[(world, node, "G")] = Fraction(0)
            rewards[(world, node, "R")] = Fraction(0)
    rewards[("R", "after_r", "G")] = Fraction(1, 5)
    rewards[("G", "after_r", "G")] = Fraction(1, 10)
    rewards[("R", "after_g0", "R")] = Fraction(1, 20)
    rewards[("G", "after_g0", "R")] = Fraction(1, 5)
    rewards[("R", "after_g1", "R")] = Fraction(3, 20)
    rewards[("G", "after_g1", "R")] = Fraction(1, 5)

    tree = FiniteHorizonBellmanTree(
        worlds=WORLDS,
        root="h0",
        nodes_by_depth=layers,
        actions=node_actions,
        transitions=transitions,
        rewards=rewards,
    )
    # The score is intentionally world independent.  Its error is small
    # enough to certify the conditional theorem, but not small enough to
    # identify the hidden root world.
    q_hat = {
        ("h0", "S"): Fraction(0),
        ("h0", "R"): Fraction(1, 20),
        ("h0", "G"): Fraction(1, 20),
        ("after_r", "S"): Fraction(0),
        ("after_r", "R"): Fraction(0),
        ("after_r", "G"): Fraction(1, 10),
        ("after_g0", "S"): Fraction(0),
        ("after_g0", "R"): Fraction(1, 10),
        ("after_g0", "G"): Fraction(0),
        ("after_g1", "S"): Fraction(0),
        ("after_g1", "R"): Fraction(1, 10),
        ("after_g1", "G"): Fraction(0),
    }
    errors = {node: Fraction(1, 10) for node in tree.nodes}
    return tree, q_hat, errors


def verify_finite_horizon_bellman_recovery() -> dict[str, object]:
    """Machine-check the finite-horizon interleaving recovery theorem."""

    tree, q_hat, errors = _bellman_recovery_fixture()
    certificate = tree.certify_greedy_recovery(q_hat, errors)
    assert certificate["oracle_values"] == {
        "R": Fraction(3, 20),
        "G": Fraction(3, 20),
        "S": Fraction(0),
    }
    assert certificate["greedy_policy"] == {
        "h0": "R",
        "after_g0": "R",
        "after_g1": "R",
        "after_r": "G",
    }
    assert certificate["regrets"] == {
        "R": Fraction(0),
        "G": Fraction(1, 10),
        "S": Fraction(1, 20),
    }
    assert certificate["expected_error_mass"] == Fraction(1, 5)
    assert certificate["trajectory_regret_bound"] == Fraction(2, 5)
    assert certificate["depthwise_regret_bound"] == Fraction(2, 5)

    stop_q_hat = dict(q_hat)
    stop_q_hat[("h0", "R")] = Fraction(-1, 10)
    stop_q_hat[("h0", "G")] = Fraction(-1, 5)
    stop_q_hat[("h0", "S")] = Fraction(0)
    tight_errors = {node: Fraction(1, 100) for node in tree.nodes}
    assert tree.stopping_certificate("h0", stop_q_hat, tight_errors)["status"] == (
        "stop_certified"
    )
    continue_q_hat = dict(stop_q_hat)
    continue_q_hat[("h0", "R")] = Fraction(1, 10)
    assert tree.stopping_certificate("h0", continue_q_hat, tight_errors)["status"] == (
        "continue_certified"
    )
    unresolved_q_hat = dict(stop_q_hat)
    unresolved_q_hat[("h0", "R")] = Fraction(1, 100)
    assert tree.stopping_certificate("h0", unresolved_q_hat, tight_errors)["status"] == (
        "unresolved"
    )

    factor_two_tree = FiniteHorizonBellmanTree(
        worlds=("w",),
        root="tight",
        nodes_by_depth=(("tight",),),
        actions={"tight": ("bad", "good", "S")},
        transitions={},
        rewards={
            ("w", "tight", "bad"): Fraction(-1, 10),
            ("w", "tight", "good"): Fraction(1, 10),
            ("w", "tight", "S"): Fraction(0),
        },
    )
    factor_two = factor_two_tree.certify_greedy_recovery(
        {
            ("tight", "bad"): Fraction(0),
            ("tight", "good"): Fraction(0),
            ("tight", "S"): Fraction(-1, 10),
        },
        Fraction(1, 10),
    )
    assert factor_two["greedy_policy"] == {"tight": "bad"}
    assert factor_two["regrets"] == {"w": Fraction(1, 5)}
    assert factor_two["trajectory_regret_bound"] == Fraction(1, 5)

    def stringify(values: Mapping[str, Fraction]) -> dict[str, str]:
        return {key: str(value) for key, value in values.items()}

    return {
        "scope": "finite_horizon_common_transcript_bellman_recovery",
        "node_count": len(tree.nodes),
        "max_depth": tree.max_depth,
        "interleaved_actions": True,
        "multi_branch_feedback": True,
        "oracle_values": stringify(certificate["oracle_values"]),
        "greedy_policy": certificate["greedy_policy"],
        "policy_values": stringify(certificate["policy_values"]),
        "regrets": stringify(certificate["regrets"]),
        "node_error": "1/10",
        "expected_error_mass": str(certificate["expected_error_mass"]),
        "trajectory_regret_bound": str(certificate["trajectory_regret_bound"]),
        "depthwise_regret_bound": str(certificate["depthwise_regret_bound"]),
        "factor_two_tight": True,
        "stopping_statuses": {
            node: details["status"]
            for node, details in certificate["stopping_certificates"].items()
        },
    }


@dataclass(frozen=True)
class ThreeWorldTemplate:
    """Decision-level three-world witness with a common null transcript."""

    positive_value: Fraction = Fraction(1, 1)
    stop_world_harm: Fraction = Fraction(1, 1)

    def __post_init__(self) -> None:
        if self.positive_value <= 0:
            raise ValueError("positive_value must be positive")
        if self.stop_world_harm != 1:
            raise ValueError(
                "the exact constant-sum certificate in this initial template "
                "requires stop_world_harm=1"
            )

    def utility(self, world: str, action: str) -> Fraction:
        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        v = self.positive_value
        if world == "R":
            return {"R": v, "G": Fraction(0), "S": Fraction(0)}[action]
        if world == "G":
            return {"R": Fraction(0), "G": v, "S": Fraction(0)}[action]
        return {"R": -v, "G": -v, "S": Fraction(0)}[action]

    def oracle_value(self, world: str) -> Fraction:
        return max(self.utility(world, action) for action in ACTIONS)

    def expected_utility(
        self, probabilities: Iterable[Fraction], world: str
    ) -> Fraction:
        probs = tuple(probabilities)
        if len(probs) != len(ACTIONS):
            raise ValueError("one probability is required per action")
        if any(p < 0 for p in probs) or sum(probs) != 1:
            raise ValueError("action probabilities must be nonnegative and sum to one")
        return sum(
            p * self.utility(world, action) for p, action in zip(probs, ACTIONS)
        )

    def regrets(self, probabilities: Iterable[Fraction]) -> dict[str, Fraction]:
        return {
            world: self.oracle_value(world)
            - self.expected_utility(probabilities, world)
            for world in WORLDS
        }

    def minimax_regret(self) -> Fraction:
        """Return the exact minimax regret for the initial template.

        For any mixture p, the three regrets sum to 2*positive_value.  Hence
        the maximum is at least 2*positive_value/3, attained by the uniform
        mixture.
        """

        return Fraction(2, 3) * self.positive_value

    def verify(self) -> dict[str, object]:
        uniform = (Fraction(1, 3),) * 3
        regrets = self.regrets(uniform)
        assert all(value == self.minimax_regret() for value in regrets.values())
        return {
            "worlds": WORLDS,
            "actions": ACTIONS,
            "common_transcript": True,
            "oracle_values": {
                world: str(self.oracle_value(world)) for world in WORLDS
            },
            "uniform_policy": [str(value) for value in uniform],
            "uniform_regrets": {world: str(value) for world, value in regrets.items()},
            "tight_minimax_regret": str(self.minimax_regret()),
            "constant_regret_sum": str(sum(regrets.values())),
        }


def verify_three_world_template() -> dict[str, object]:
    """Machine-check the initial decision-level three-world result."""

    return ThreeWorldTemplate().verify()


@dataclass(frozen=True)
class ThreeWorldWGATemplate:
    """Finite same-``(X,Y)`` WGA witness with an unrepairable stop atom.

    This is the concrete, asymmetric extension used in the initial paper
    draft.  ``A_0`` and ``A_1`` have size ``r`` and ``C`` has size ``n``.
    Every candidate label query returns the same clean label, while a hidden
    group assignment determines whether the vulnerable positive cell is an
    audited ``A`` block (``G`` world), ``C`` (``R`` world), or an unrepairable
    test atom (``S`` world).  The class deliberately verifies only the
    inherited pairwise lower bound; it is not the conjectured symmetric
    three-world embedding.
    """

    label_capacity: int = 3
    post_audit_capacity: int = 2
    anchor_mass: Fraction = Fraction(48, 1)
    blocker_mass: Fraction = Fraction(1, 100)

    def __post_init__(self) -> None:
        n = self.label_capacity
        r = self.post_audit_capacity
        if n < 3 or not (n / 2 < r < n):
            raise ValueError("capacities must satisfy n >= 3 and n / 2 < r < n")
        if self.anchor_mass <= 0:
            raise ValueError("anchor_mass must be positive")
        if self.blocker_mass <= 0:
            raise ValueError("blocker_mass must be positive")
        threshold = self.minimum_anchor_mass()
        if self.anchor_mass < threshold:
            raise ValueError(
                "anchor_mass is too small for the normalized utility bound; "
                f"need at least {threshold}"
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
    def blocker_atom(self) -> int:
        return 2 * self.post_audit_capacity + self.label_capacity

    def minimum_anchor_mass(self) -> Fraction:
        """Return sufficient (not necessarily minimal) anchor mass."""

        n = self.label_capacity
        r = self.post_audit_capacity
        e = self.blocker_mass
        m0 = Fraction(
            n * (r + n + e) + 2 * r * (n - 1) * (2 * r + e),
            2 * r - n,
        )
        m1 = Fraction(
            n * (r - 1) * (r + n + e) + r * (2 * r + e),
            n - r,
        )
        # The two displayed thresholds also dominate the coefficient-ordering
        # thresholds for the admissible n,r range.  Keep a direct check in
        # ``verify`` so a future generalization cannot silently rely on this.
        return Fraction(ceil(max(m0, m1)), 1)

    def _coerce_query(self, query: Iterable[int]) -> frozenset[int]:
        q = frozenset(query)
        if not q.issubset(self.all_atoms):
            raise ValueError("query contains an atom outside the legal candidate pool")
        return q

    def utility(
        self, world: str, query: Iterable[int], audit_response: int = 0
    ) -> Fraction:
        """Return the exact raw WGA increment for one realized response."""

        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")
        if audit_response not in (0, 1):
            raise ValueError("audit_response must be binary")
        q = self._coerce_query(query)
        n = self.label_capacity
        r = self.post_audit_capacity
        m = self.anchor_mass
        e = self.blocker_mass
        if world == "S":
            # The unrepairable atom forms a positive cell on which the model
            # remains wrong for every legal query.
            return Fraction(0)
        if world == "G":
            active = set(self.a0_atoms if audit_response == 0 else self.a1_atoms)
            return min(
                Fraction(len(q & active), r),
                Fraction(m + len(q - active), 1) / (m + r + n + e),
            )
        c = set(self.c_atoms)
        a = set(self.a0_atoms + self.a1_atoms)
        return min(
            Fraction(len(q & c), n),
            Fraction(m + len(q & a), 1) / (m + 2 * r + e),
        )

    def oracle_values(self) -> dict[str, Fraction]:
        """World-aware raw values for the initial witness."""

        m = self.anchor_mass
        e = self.blocker_mass
        return {
            "G": m / (m + self.post_audit_capacity + self.label_capacity + e),
            "R": m / (m + 2 * self.post_audit_capacity + e),
            "S": Fraction(0),
        }

    def _subsets(self, maximum_size: int) -> tuple[frozenset[int], ...]:
        atoms = self.all_atoms
        return tuple(
            frozenset(combo)
            for size in range(maximum_size + 1)
            for combo in combinations(atoms, size)
        )

    def deterministic_points(self) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
        """Enumerate no-audit and audit-conditioned deterministic vertices."""

        points: list[tuple[Fraction, Fraction, Fraction]] = []
        for query in self._subsets(self.label_capacity):
            g = (self.utility("G", query, 0) + self.utility("G", query, 1)) / 2
            r = self.utility("R", query, 0)
            points.append((g, r, Fraction(0)))
        post = self._subsets(self.post_audit_capacity)
        for q0 in post:
            for q1 in post:
                g = (self.utility("G", q0, 0) + self.utility("G", q1, 1)) / 2
                r = (self.utility("R", q0, 0) + self.utility("R", q1, 1)) / 2
                points.append((g, r, Fraction(0)))
        return tuple(points)

    def test_support_xy(
        self, world: str, audit_response: int
    ) -> tuple[tuple[str, int, Fraction], ...]:
        """Return the public ``(X,Y)`` support, dropping the hidden group."""

        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")
        if audit_response not in (0, 1):
            raise ValueError("audit_response must be binary")
        rows = [
            (f"repairable_{atom}", 1, Fraction(1)) for atom in self.all_atoms
        ]
        rows.append(("positive_anchor", 1, self.anchor_mass))
        rows.append(("unrepairable_blocker", 1, self.blocker_mass))
        rows.extend(
            [
                ("negative_anchor_0", 0, Fraction(1)),
                ("negative_anchor_1", 0, Fraction(1)),
            ]
        )
        return tuple(rows)

    def hidden_group_support(
        self, world: str, audit_response: int
    ) -> tuple[tuple[str, int], ...]:
        """Return hidden group assignments for invariant-support checks."""

        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")
        if audit_response not in (0, 1):
            raise ValueError("audit_response must be binary")
        if world == "G":
            positive_one = set(self.a0_atoms if audit_response == 0 else self.a1_atoms)
        elif world == "R":
            positive_one = set(self.c_atoms)
        else:
            positive_one = {self.blocker_atom}
        rows = [(f"repairable_{atom}", int(atom in positive_one)) for atom in self.all_atoms]
        rows.append(("positive_anchor", 0))
        rows.append(("unrepairable_blocker", int(self.blocker_atom in positive_one)))
        rows.extend([("negative_anchor_0", 0), ("negative_anchor_1", 1)])
        return tuple(rows)

    def verify(self) -> dict[str, object]:
        """Machine-check the initial asymmetric three-world construction."""

        n = self.label_capacity
        r = self.post_audit_capacity
        m = self.anchor_mass
        e = self.blocker_mass
        a = (m + r + n + e) / (2 * r * m)
        b = (m + 2 * r + e) / (n * m)
        a_prime = (m + r + n + e) / (r * m)
        assert b >= a
        assert a + (n - 1) * b <= 1
        assert a_prime >= b
        assert (r - 1) * a_prime + b <= 1

        points = self.deterministic_points()
        values = self.oracle_values()
        maxima = {
            "G": max(point[0] for point in points),
            "R": max(point[1] for point in points),
            "S": max(point[2] for point in points),
        }
        assert maxima == values
        normalized_sums = [
            point[0] / values["G"] + point[1] / values["R"] for point in points
        ]
        assert max(normalized_sums) <= 1

        supports = [
            self.test_support_xy(world, response)
            for world in WORLDS
            for response in (0, 1)
        ]
        assert len(set(supports)) == 1
        hidden = [
            self.hidden_group_support(world, response)
            for world in WORLDS
            for response in (0, 1)
        ]
        assert len(set(hidden)) > 1

        # The pairwise harmonic lower bound is attained by mixing the two
        # world-aware allocations.  The S-world contributes zero raw regret.
        bound = m / (2 * m + n + 3 * r + 2 * e)
        p_g = values["G"] / (values["G"] + values["R"])
        p_r = 1 - p_g
        mix_g = p_g * values["G"]
        mix_r = p_r * values["R"]
        assert values["G"] - mix_g == bound
        assert values["R"] - mix_r == bound

        return {
            "deterministic_policy_count": len(points),
            "anchor_mass": str(m),
            "blocker_mass": str(e),
            "minimum_anchor_mass": str(self.minimum_anchor_mass()),
            "same_test_xy_marginal": True,
            "hidden_group_assignment_changes": True,
            "oracle_values": {world: str(value) for world, value in values.items()},
            "maximum_normalized_pairwise_sum": str(max(normalized_sums)),
            "inherited_pairwise_regret_bound": str(bound),
            "pairwise_mixture_regrets": {
                "G": str(values["G"] - mix_g),
                "R": str(values["R"] - mix_r),
                "S": "0",
            },
        }


def verify_three_world_wga_template() -> dict[str, object]:
    """Machine-check the concrete initial three-world WGA witness."""

    return ThreeWorldWGATemplate().verify()


SHARP_POLICIES = ("S", "R", "G00", "G10", "G01", "G11")


@dataclass(frozen=True)
class SharpThreeWorldWGATemplate:
    """Sharp same-``(X,Y)`` WGA witness for a finite sequential menu.

    The budget is three unit-cost operations.  At the root, ``R`` purchases the
    fixed three-label batch ``{a,b,x}``; ``G`` purchases a common fair audit bit
    and then permits only the two-label batch ``{a,d_H}``; ``S`` stops.  A pure
    G-policy is named ``Gij``, where ``i`` (respectively ``j``) says whether to
    buy the post-audit batch after ``H=0`` (respectively ``H=1``).

    The construction is deliberately scoped to this legal menu.  It does not
    establish the same bound when arbitrary singleton or batch queries are
    added to the action space.
    """

    anchor_mass: Fraction = Fraction(12, 1)
    blocker_mass: Fraction = Fraction(1, 1)
    cost_multiplier: Fraction = Fraction(1, 3)
    l2_coefficient: Fraction = Fraction(1, 1)

    def __post_init__(self) -> None:
        if self.blocker_mass <= 0:
            raise ValueError("blocker_mass must be positive")
        if self.anchor_mass < 10 + 2 * self.blocker_mass:
            raise ValueError(
                "anchor_mass must be at least 10 + 2 * blocker_mass so the "
                "designated positive cell determines every displayed WGA value"
            )
        if self.cost_multiplier != Fraction(1, 3):
            raise ValueError("the exact sharp table requires cost_multiplier=1/3")
        if self.l2_coefficient <= 0:
            raise ValueError("l2_coefficient must be positive")

    @property
    def repairable_atoms(self) -> tuple[str, ...]:
        return ("a", "b", "c", "x", "d0", "e0", "d1", "e1")

    @property
    def positive_atoms(self) -> tuple[str, ...]:
        return self.repairable_atoms + ("P", "D")

    @property
    def training_atoms(self) -> tuple[str, ...]:
        return self.positive_atoms + ("N0", "N1")

    @property
    def root_r_batch(self) -> frozenset[str]:
        return frozenset(("a", "b", "x"))

    def post_g_batch(self, audit_response: int) -> frozenset[str]:
        self._validate_response(audit_response)
        return frozenset(("a", f"d{audit_response}"))

    @staticmethod
    def _validate_world(world: str) -> None:
        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")

    @staticmethod
    def _validate_response(audit_response: int) -> None:
        if audit_response not in (0, 1):
            raise ValueError("audit_response must be binary")

    @staticmethod
    def _validate_policy(policy: str) -> None:
        if policy not in SHARP_POLICIES:
            raise ValueError(f"unknown sharp-menu policy: {policy}")

    def continues_after_g(self, policy: str, audit_response: int) -> bool:
        self._validate_policy(policy)
        self._validate_response(audit_response)
        if not policy.startswith("G"):
            return False
        return policy[1 + audit_response] == "1"

    def queried_atoms(
        self, policy: str, audit_response: int
    ) -> frozenset[str]:
        self._validate_policy(policy)
        self._validate_response(audit_response)
        if policy == "R":
            return self.root_r_batch
        if policy.startswith("G") and self.continues_after_g(
            policy, audit_response
        ):
            return self.post_g_batch(audit_response)
        return frozenset()

    def normalized_cost(self, policy: str, audit_response: int) -> Fraction:
        """Return branch cost divided by the total budget of three."""

        self._validate_policy(policy)
        self._validate_response(audit_response)
        if policy == "S":
            return Fraction(0)
        if policy == "R":
            return Fraction(1)
        units = 1 + 2 * int(self.continues_after_g(policy, audit_response))
        return Fraction(units, 3)

    def positive_group_one(
        self, world: str, audit_response: int
    ) -> frozenset[str]:
        self._validate_world(world)
        self._validate_response(audit_response)
        if world == "R":
            return frozenset(("a", "b", "c"))
        if world == "G":
            return frozenset(("a", f"d{audit_response}", f"e{audit_response}"))
        return frozenset(("D",))

    def atom_weight(self, atom: str) -> Fraction:
        if atom == "P":
            return self.anchor_mass
        if atom == "D":
            return self.blocker_mass
        if atom in self.training_atoms:
            return Fraction(1)
        raise ValueError(f"unknown atom: {atom}")

    def clean_label_response(self, atom: str) -> int:
        if atom not in self.repairable_atoms:
            raise ValueError("only repairable candidates have a legal label response")
        return 1

    def test_label(self, atom: str) -> int:
        if atom in self.positive_atoms:
            return 1
        if atom in ("N0", "N1"):
            return 0
        raise ValueError(f"unknown atom: {atom}")

    def prediction_is_correct(self, atom: str, queried: Iterable[str]) -> bool:
        q = frozenset(queried)
        if not q.issubset(self.repairable_atoms):
            raise ValueError("query contains an atom outside the legal candidate pool")
        if atom in ("P", "N0", "N1"):
            return True
        if atom == "D":
            return False
        if atom in self.repairable_atoms:
            return atom in q
        raise ValueError(f"unknown atom: {atom}")

    def raw_wga_for_query(
        self, world: str, queried: Iterable[str], audit_response: int = 0
    ) -> Fraction:
        """Compute WGA for an arbitrary legal corrected candidate set."""

        self._validate_world(world)
        self._validate_response(audit_response)
        queried_set = frozenset(queried)
        if not queried_set.issubset(self.repairable_atoms):
            raise ValueError("query contains an atom outside the legal candidate pool")
        group_one = self.positive_group_one(world, audit_response)
        positive_accuracies: list[Fraction] = []
        for attribute in (0, 1):
            atoms = tuple(
                atom
                for atom in self.positive_atoms
                if int(atom in group_one) == attribute
            )
            total = sum((self.atom_weight(atom) for atom in atoms), Fraction(0))
            correct = sum(
                (
                    self.atom_weight(atom)
                    for atom in atoms
                    if self.prediction_is_correct(atom, queried_set)
                ),
                Fraction(0),
            )
            positive_accuracies.append(correct / total)
        # N0 and N1 make the two negative class-attribute cells nonempty and
        # both have accuracy one, so only the positive cells can attain WGA.
        return min(positive_accuracies + [Fraction(1), Fraction(1)])

    def raw_wga(self, world: str, policy: str, audit_response: int) -> Fraction:
        """Compute the exact four-cell WGA after one realized policy branch."""

        self._validate_policy(policy)
        return self.raw_wga_for_query(
            world, self.queried_atoms(policy, audit_response), audit_response
        )

    def expected_raw_wga(self, world: str, policy: str) -> Fraction:
        self._validate_world(world)
        self._validate_policy(policy)
        return sum(
            (self.raw_wga(world, policy, response) for response in (0, 1)),
            Fraction(0),
        ) / 2

    def expected_normalized_cost(self, policy: str) -> Fraction:
        self._validate_policy(policy)
        return sum(
            (self.normalized_cost(policy, response) for response in (0, 1)),
            Fraction(0),
        ) / 2

    def utility(self, world: str, policy: str) -> Fraction:
        return self.expected_raw_wga(world, policy) - (
            self.cost_multiplier * self.expected_normalized_cost(policy)
        )

    def oracle_value(self, world: str) -> Fraction:
        self._validate_world(world)
        return max(self.utility(world, policy) for policy in SHARP_POLICIES)

    def oracle_policy(self, world: str) -> str:
        self._validate_world(world)
        maximizers = [
            policy
            for policy in SHARP_POLICIES
            if self.utility(world, policy) == self.oracle_value(world)
        ]
        if len(maximizers) != 1:
            raise AssertionError(f"world {world} does not have a strict oracle")
        return maximizers[0]

    def regret(self, world: str, policy: str) -> Fraction:
        return self.oracle_value(world) - self.utility(world, policy)

    def expected_mixture_utility(
        self, world: str, probabilities: dict[str, Fraction]
    ) -> Fraction:
        self._validate_world(world)
        if set(probabilities) - set(SHARP_POLICIES):
            raise ValueError("mixture contains an unknown policy")
        if any(probability < 0 for probability in probabilities.values()):
            raise ValueError("mixture probabilities must be nonnegative")
        if sum(probabilities.values(), Fraction(0)) != 1:
            raise ValueError("mixture probabilities must sum to one")
        return sum(
            (
                probability * self.utility(world, policy)
                for policy, probability in probabilities.items()
            ),
            Fraction(0),
        )

    def mixture_regrets(
        self, probabilities: dict[str, Fraction]
    ) -> dict[str, Fraction]:
        return {
            world: self.oracle_value(world)
            - self.expected_mixture_utility(world, probabilities)
            for world in WORLDS
        }

    def test_support_xy(
        self, world: str, audit_response: int
    ) -> tuple[tuple[str, int, Fraction], ...]:
        """Return the complete public test ``(X,Y)`` support."""

        self._validate_world(world)
        self._validate_response(audit_response)
        rows = [
            (atom, 1, self.atom_weight(atom)) for atom in self.positive_atoms
        ]
        rows.extend((("N0", 0, Fraction(1)), ("N1", 0, Fraction(1))))
        return tuple(rows)

    def hidden_group_support(
        self, world: str, audit_response: int
    ) -> tuple[tuple[str, int], ...]:
        """Return the hidden binary attribute attached to each test atom."""

        group_one = self.positive_group_one(world, audit_response)
        rows = [(atom, int(atom in group_one)) for atom in self.positive_atoms]
        rows.extend((("N0", 0), ("N1", 1)))
        return tuple(rows)

    def training_labels(self, queried: Iterable[str]) -> tuple[tuple[str, int], ...]:
        """Return corrupted/corrected labels used by the logistic witness."""

        q = frozenset(queried)
        if not q.issubset(self.repairable_atoms):
            raise ValueError("query contains an atom outside the legal candidate pool")
        rows = [(atom, int(atom in q)) for atom in self.repairable_atoms]
        rows.extend((("P", 1), ("D", 0), ("N0", 0), ("N1", 0)))
        return tuple(rows)

    def logistic_sign_certificate(self, queried: Iterable[str]) -> dict[str, int]:
        """Return optimum coordinate signs implied by strong convexity.

        The coordinate derivative at zero is ``1/2-y`` and is strictly
        increasing.  Its unique root is therefore positive for ``y=1`` and
        negative for ``y=0``.
        """

        return {
            atom: 1 if label == 1 else -1
            for atom, label in self.training_labels(queried)
        }

    def logistic_alpha(self) -> float:
        """Return the one-hot absolute optimum margin for the L2 coefficient."""

        gamma = float(self.l2_coefficient)

        def equation(alpha: float) -> float:
            return 1.0 / (1.0 + exp(-alpha)) + gamma * alpha - 1.0

        lower, upper = 0.0, 1.0
        while equation(upper) < 0.0:
            upper *= 2.0
        for _ in range(80):
            middle = (lower + upper) / 2.0
            if equation(middle) < 0.0:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2.0

    def explicit_stability_radius(self) -> float:
        """Return a sufficient spectral-norm radius for sign preservation.

        Let both the training and test design matrices be within spectral norm
        ``epsilon`` of the 12-dimensional identity.  The returned positive
        root is the largest radius certified by

        ``alpha - K*epsilon*(1+epsilon) - alpha*sqrt(d)*epsilon > 0``,

        where ``alpha`` is the one-hot margin and
        ``K=(sqrt(d)+alpha*sqrt(d)/4)/gamma``.  The guarantee is for strict
        ``epsilon < radius``.
        """

        dimension = len(self.training_atoms)
        gamma = float(self.l2_coefficient)
        alpha = self.logistic_alpha()
        k = (sqrt(dimension) + alpha * sqrt(dimension) / 4.0) / gamma
        linear = k + alpha * sqrt(dimension)
        return (-linear + sqrt(linear * linear + 4.0 * k * alpha)) / (2.0 * k)

    def verify(self) -> dict[str, object]:
        """Machine-check the sharp menu, table, lower bound, and embedding."""

        raw_table = {
            world: {
                "R": self.expected_raw_wga(world, "R"),
                "G": self.expected_raw_wga(world, "G11"),
                "S": self.expected_raw_wga(world, "S"),
            }
            for world in WORLDS
        }
        assert raw_table == {
            "R": {"R": Fraction(2, 3), "G": Fraction(1, 3), "S": Fraction(0)},
            "G": {"R": Fraction(1, 3), "G": Fraction(2, 3), "S": Fraction(0)},
            "S": {"R": Fraction(0), "G": Fraction(0), "S": Fraction(0)},
        }

        net_table = {
            world: {
                "R": self.utility(world, "R"),
                "G": self.utility(world, "G11"),
                "S": self.utility(world, "S"),
            }
            for world in WORLDS
        }
        assert net_table == {
            "R": {"R": Fraction(1, 3), "G": Fraction(0), "S": Fraction(0)},
            "G": {"R": Fraction(0), "G": Fraction(1, 3), "S": Fraction(0)},
            "S": {"R": Fraction(-1, 3), "G": Fraction(-1, 3), "S": Fraction(0)},
        }

        supports = {
            self.test_support_xy(world, response)
            for world in WORLDS
            for response in (0, 1)
        }
        hidden = {
            self.hidden_group_support(world, response)
            for world in WORLDS
            for response in (0, 1)
        }
        assert len(supports) == 1
        assert len(hidden) == 4
        assert {
            self.clean_label_response(atom) for atom in self.repairable_atoms
        } == {1}
        assert all(
            self.prediction_is_correct(atom, (atom,))
            for atom in self.repairable_atoms
        )

        oracles = {world: self.oracle_policy(world) for world in WORLDS}
        values = {world: self.oracle_value(world) for world in WORLDS}
        assert oracles == {"R": "R", "G": "G11", "S": "S"}
        assert values == {"R": Fraction(1, 3), "G": Fraction(1, 3), "S": 0}

        regret_sums = {
            policy: sum(
                (self.regret(world, policy) for world in WORLDS), Fraction(0)
            )
            for policy in SHARP_POLICIES
        }
        assert regret_sums == {
            "S": Fraction(2, 3),
            "R": Fraction(2, 3),
            "G00": Fraction(1),
            "G10": Fraction(5, 6),
            "G01": Fraction(5, 6),
            "G11": Fraction(2, 3),
        }

        tight_mixture = {
            "S": Fraction(1, 3),
            "R": Fraction(1, 3),
            "G11": Fraction(1, 3),
        }
        tight_regrets = self.mixture_regrets(tight_mixture)
        assert tight_regrets == {world: Fraction(2, 9) for world in WORLDS}

        pairwise_mixture = {"R": Fraction(1, 2), "G11": Fraction(1, 2)}
        pairwise_regrets = self.mixture_regrets(pairwise_mixture)
        assert pairwise_regrets["R"] == pairwise_regrets["G"] == Fraction(1, 6)
        assert min(
            self.regret("R", policy) + self.regret("G", policy)
            for policy in SHARP_POLICIES
        ) == Fraction(1, 3)

        # Each atom is one coordinate of a no-intercept logistic model.  At
        # theta_j=0 the coordinate derivative is 1/2-y_j, while the Hessian is
        # at least gamma I.  Hence the unique optimum has positive sign exactly
        # for corrected/P labels and negative sign otherwise at every state.
        legal_queries = {
            self.queried_atoms(policy, response)
            for policy in SHARP_POLICIES
            for response in (0, 1)
        }
        assert len(legal_queries) == 4
        assert all(
            {label for _, label in self.training_labels(query)} <= {0, 1}
            for query in legal_queries
        )
        for query in legal_queries:
            signs = self.logistic_sign_certificate(query)
            assert all(sign != 0 for sign in signs.values())
            for atom, sign in signs.items():
                logistic_is_correct = int(sign > 0) == self.test_label(atom)
                assert logistic_is_correct == self.prediction_is_correct(atom, query)

        alpha = self.logistic_alpha()
        radius = self.explicit_stability_radius()
        dimension = len(self.training_atoms)
        gamma = float(self.l2_coefficient)
        k = (sqrt(dimension) + alpha * sqrt(dimension) / 4.0) / gamma

        def margin_bound(epsilon: float) -> float:
            return alpha - k * epsilon * (1.0 + epsilon) - alpha * sqrt(
                dimension
            ) * epsilon

        assert radius > 0.0
        assert margin_bound(radius * 0.99) > 0.0
        assert margin_bound(radius * 1.01) < 0.0

        def stringify(
            table: dict[str, dict[str, Fraction]],
        ) -> dict[str, dict[str, str]]:
            return {
                world: {action: str(value) for action, value in row.items()}
                for world, row in table.items()
            }
        return {
            "scope": "finite_batch_state_dependent_menu",
            "deterministic_policy_count": len(SHARP_POLICIES),
            "policies": SHARP_POLICIES,
            "same_test_xy_marginal": True,
            "common_audit_law": "Bernoulli(1/2)",
            "common_label_response": 1,
            "hidden_group_assignment_count": len(hidden),
            "raw_plan_table": stringify(raw_table),
            "net_plan_table": stringify(net_table),
            "strict_oracle_policies": oracles,
            "oracle_values": {world: str(value) for world, value in values.items()},
            "pure_policy_regret_sums": {
                policy: str(value) for policy, value in regret_sums.items()
            },
            "regret_sum_lower_bound": "2/3",
            "tight_minimax_regret": "2/9",
            "tight_mixture_regrets": {
                world: str(value) for world, value in tight_regrets.items()
            },
            "two_world_minimax_regret": "1/6",
            "logistic_dimension": len(self.training_atoms),
            "strong_convexity_modulus": str(self.l2_coefficient),
            "one_hot_sign_realization": True,
            "finite_state_local_stability": True,
            "one_hot_margin": self.logistic_alpha(),
            "explicit_spectral_radius": self.explicit_stability_radius(),
            "stability_radius_scope": (
                "strict spectral-norm epsilon < radius for train/test designs"
            ),
            "unrestricted_menu_extension": "open",
        }


PER_LABEL_REPAIRABLE_ATOMS = ("a", "b", "c", "d0", "e0", "d1", "e1")


@dataclass(frozen=True)
class PerLabelBatchThreeWorldWGATemplate:
    """Per-label-priced, single-audit closure with an exact three-way bound.

    A no-audit R plan of size ``s`` costs ``s`` units.  A plan containing the
    fair audit bit costs one audit unit plus the number of labels on its
    realized branch.  Atomic R and G actions may occur in either order, with
    total unit budget three, no repeated label, and at most one audit.  Since
    every purchased label is deterministically one, every such plan has a
    payoff-equivalent normal form: either ``R[Q]`` or ``G[Q0]|[Q1]``.  The
    latter records the two final queried sets, including any common pre-audit
    prefix.  The enumerated G menu is a superset-exact normal-form cover.

    The seven repairable atoms use two bridge atoms: ``a`` joins the R world
    to the ``H=0`` branch and ``b`` joins it to ``H=1``.  Thus a no-audit plan
    cannot reproduce both branch-specific allocations.

    The calibration is intentionally fixed at ``anchor_mass=10``,
    ``blocker_mass=1`` and ``cost_multiplier=1/3``.  It makes every legal
    positive-cell value equal to ``min(k, 2)/3`` and gives the exact ``16/81``
    minimax regret certificate over 905 deterministic terminal-set normal
    forms.  The theorem remains finite and limited to one audit and total
    budget three; repeated audits and larger unrestricted budgets are open.
    """

    anchor_mass: Fraction = Fraction(10, 1)
    blocker_mass: Fraction = Fraction(1, 1)
    cost_multiplier: Fraction = Fraction(1, 3)
    l2_coefficient: Fraction = Fraction(1, 1)

    def __post_init__(self) -> None:
        if self.anchor_mass != Fraction(10, 1):
            raise ValueError(
                "the per-label certificate is calibrated to anchor_mass=10"
            )
        if self.blocker_mass != Fraction(1, 1):
            raise ValueError(
                "the per-label certificate is calibrated to blocker_mass=1"
            )
        if self.cost_multiplier != Fraction(1, 3):
            raise ValueError(
                "the per-label certificate requires cost_multiplier=1/3"
            )
        if self.l2_coefficient <= 0:
            raise ValueError("l2_coefficient must be positive")

    @property
    def repairable_atoms(self) -> tuple[str, ...]:
        return PER_LABEL_REPAIRABLE_ATOMS

    @property
    def positive_atoms(self) -> tuple[str, ...]:
        return self.repairable_atoms + ("P", "D")

    @property
    def training_atoms(self) -> tuple[str, ...]:
        return self.positive_atoms + ("N0", "N1")

    @property
    def r_group(self) -> frozenset[str]:
        return frozenset(("a", "b", "c"))

    @property
    def g0_group(self) -> frozenset[str]:
        return frozenset(("a", "d0", "e0"))

    @property
    def g1_group(self) -> frozenset[str]:
        return frozenset(("b", "d1", "e1"))

    @staticmethod
    def _validate_world(world: str) -> None:
        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")

    @staticmethod
    def _validate_response(audit_response: int) -> None:
        if audit_response not in (0, 1):
            raise ValueError("audit_response must be binary")

    def _coerce_query(
        self,
        query: Iterable[str],
        *,
        maximum_size: int,
        allow_empty: bool = True,
    ) -> frozenset[str]:
        q = frozenset(query)
        if not allow_empty and not q:
            raise ValueError("a legal batch must be nonempty")
        if len(q) > maximum_size:
            raise ValueError("query exceeds the legal batch capacity")
        if not q.issubset(self.repairable_atoms):
            raise ValueError("query contains an atom outside the legal candidate pool")
        return q

    def positive_group_one(
        self, world: str, audit_response: int
    ) -> frozenset[str]:
        self._validate_world(world)
        self._validate_response(audit_response)
        if world == "R":
            return self.r_group
        if world == "G":
            return self.g0_group if audit_response == 0 else self.g1_group
        return frozenset(("D",))

    def atom_weight(self, atom: str) -> Fraction:
        if atom == "P":
            return self.anchor_mass
        if atom == "D":
            return self.blocker_mass
        if atom in self.training_atoms:
            return Fraction(1, 1)
        raise ValueError(f"unknown atom: {atom}")

    def clean_label_response(self, atom: str) -> int:
        if atom not in self.repairable_atoms:
            raise ValueError("only repairable candidates have a legal label response")
        return 1

    def test_label(self, atom: str) -> int:
        if atom in self.positive_atoms:
            return 1
        if atom in ("N0", "N1"):
            return 0
        raise ValueError(f"unknown atom: {atom}")

    def prediction_is_correct(self, atom: str, queried: Iterable[str]) -> bool:
        q = self._coerce_query(queried, maximum_size=len(self.repairable_atoms))
        if atom in ("P", "N0", "N1"):
            return True
        if atom == "D":
            return False
        if atom in self.repairable_atoms:
            return atom in q
        raise ValueError(f"unknown atom: {atom}")

    def raw_wga_for_query(
        self, world: str, queried: Iterable[str], audit_response: int = 0
    ) -> Fraction:
        """Compute WGA for one realized legal query."""

        self._validate_world(world)
        self._validate_response(audit_response)
        q = self._coerce_query(
            queried, maximum_size=len(self.repairable_atoms)
        )
        group_one = self.positive_group_one(world, audit_response)
        positive_accuracies: list[Fraction] = []
        for attribute in (0, 1):
            atoms = tuple(
                atom
                for atom in self.positive_atoms
                if int(atom in group_one) == attribute
            )
            total = sum((self.atom_weight(atom) for atom in atoms), Fraction(0))
            correct = sum(
                (
                    self.atom_weight(atom)
                    for atom in atoms
                    if self.prediction_is_correct(atom, q)
                ),
                Fraction(0),
            )
            positive_accuracies.append(correct / total)
        return min(positive_accuracies + [Fraction(1), Fraction(1)])

    def expected_raw_wga(self, world: str, query: Iterable[str]) -> Fraction:
        self._validate_world(world)
        q = self._coerce_query(query, maximum_size=3)
        return sum(
            (self.raw_wga_for_query(world, q, response) for response in (0, 1)),
            Fraction(0),
        ) / 2

    def test_support_xy(
        self, world: str, audit_response: int
    ) -> tuple[tuple[str, int, Fraction], ...]:
        self._validate_world(world)
        self._validate_response(audit_response)
        rows = [
            (atom, 1, self.atom_weight(atom)) for atom in self.positive_atoms
        ]
        rows.extend((("N0", 0, Fraction(1)), ("N1", 0, Fraction(1))))
        return tuple(rows)

    def hidden_group_support(
        self, world: str, audit_response: int
    ) -> tuple[tuple[str, int], ...]:
        group_one = self.positive_group_one(world, audit_response)
        rows = [(atom, int(atom in group_one)) for atom in self.positive_atoms]
        rows.extend((("N0", 0), ("N1", 1)))
        return tuple(rows)

    def training_labels(self, queried: Iterable[str]) -> tuple[tuple[str, int], ...]:
        q = self._coerce_query(queried, maximum_size=len(self.repairable_atoms))
        rows = [(atom, int(atom in q)) for atom in self.repairable_atoms]
        rows.extend((("P", 1), ("D", 0), ("N0", 0), ("N1", 0)))
        return tuple(rows)

    def logistic_sign_certificate(self, queried: Iterable[str]) -> dict[str, int]:
        return {
            atom: 1 if label == 1 else -1
            for atom, label in self.training_labels(queried)
        }

    def logistic_alpha(self) -> float:
        gamma = float(self.l2_coefficient)

        def equation(alpha: float) -> float:
            return 1.0 / (1.0 + exp(-alpha)) + gamma * alpha - 1.0

        lower, upper = 0.0, 1.0
        while equation(upper) < 0.0:
            upper *= 2.0
        for _ in range(80):
            middle = (lower + upper) / 2.0
            if equation(middle) < 0.0:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2.0

    def explicit_stability_radius(self) -> float:
        dimension = len(self.training_atoms)
        gamma = float(self.l2_coefficient)
        alpha = self.logistic_alpha()
        k = (sqrt(dimension) + alpha * sqrt(dimension) / 4.0) / gamma
        linear = k + alpha * sqrt(dimension)
        return (-linear + sqrt(linear * linear + 4.0 * k * alpha)) / (2.0 * k)

    def _subsets(self, maximum_size: int) -> tuple[frozenset[str], ...]:
        return tuple(
            frozenset(combo)
            for size in range(1, maximum_size + 1)
            for combo in combinations(self.repairable_atoms, size)
        )

    @property
    def root_queries(self) -> tuple[frozenset[str], ...]:
        return self._subsets(3)

    @property
    def post_g_options(self) -> tuple[frozenset[str] | None, ...]:
        return (None,) + self._subsets(2)

    @property
    def policy_specs(self) -> tuple[tuple[object, ...], ...]:
        specs: list[tuple[object, ...]] = [("S",)]
        specs.extend(("R", query) for query in self.root_queries)
        specs.extend(
            ("G", query0, query1)
            for query0 in self.post_g_options
            for query1 in self.post_g_options
        )
        return tuple(specs)

    @property
    def interleaved_trace_skeletons(
        self,
    ) -> tuple[tuple[frozenset[str], frozenset[str], frozenset[str]], ...]:
        """Return every set-level R-before-G/after-G trace skeleton.

        A skeleton contains the deterministic labels bought before the audit
        and the additional labels bought after responses zero and one.  Query
        order within each deterministic segment is payoff irrelevant.
        """

        skeletons: list[
            tuple[frozenset[str], frozenset[str], frozenset[str]]
        ] = []
        for prefix_size in range(3):
            for prefix_tuple in combinations(self.repairable_atoms, prefix_size):
                prefix = frozenset(prefix_tuple)
                remaining = tuple(
                    atom for atom in self.repairable_atoms if atom not in prefix
                )
                remaining_capacity = 2 - prefix_size
                additions = tuple(
                    frozenset(combo)
                    for size in range(remaining_capacity + 1)
                    for combo in combinations(remaining, size)
                )
                skeletons.extend(
                    (prefix, addition0, addition1)
                    for addition0 in additions
                    for addition1 in additions
                )
        return tuple(skeletons)

    @staticmethod
    def normal_form_for_interleaved_trace(
        skeleton: tuple[frozenset[str], frozenset[str], frozenset[str]],
    ) -> tuple[object, ...]:
        """Collapse an interleaved trace to its two final branch query sets."""

        prefix, addition0, addition1 = skeleton
        if prefix & addition0 or prefix & addition1:
            raise ValueError("an interleaved trace cannot query a label twice")
        query0 = prefix | addition0
        query1 = prefix | addition1
        if len(query0) > 2 or len(query1) > 2:
            raise ValueError("an audited trace exceeds the total unit budget")
        return (
            "G",
            query0 if query0 else None,
            query1 if query1 else None,
        )

    def _query_name(self, query: frozenset[str] | None) -> str:
        if query is None:
            return "STOP"
        order = {
            atom: index for index, atom in enumerate(self.repairable_atoms)
        }
        return ",".join(sorted(query, key=order.__getitem__))

    def policy_name(self, spec: tuple[object, ...]) -> str:
        kind = spec[0]
        if kind == "S":
            return "S"
        if kind == "R":
            query = self._coerce_query(spec[1], maximum_size=3, allow_empty=False)
            return f"R[{self._query_name(query)}]"
        if kind == "G":
            query0 = spec[1]
            query1 = spec[2]
            q0 = None if query0 is None else self._coerce_query(
                query0, maximum_size=2, allow_empty=False
            )
            q1 = None if query1 is None else self._coerce_query(
                query1, maximum_size=2, allow_empty=False
            )
            return f"G[{self._query_name(q0)}]|[{self._query_name(q1)}]"
        raise ValueError(f"unknown per-label policy kind: {kind}")

    @property
    def named_policy_specs(self) -> tuple[tuple[str, tuple[object, ...]], ...]:
        return tuple((self.policy_name(spec), spec) for spec in self.policy_specs)

    def utility_for_spec(
        self, world: str, spec: tuple[object, ...]
    ) -> Fraction:
        self._validate_world(world)
        kind = spec[0]
        if kind == "S":
            return Fraction(0)
        if kind == "R":
            query = self._coerce_query(
                spec[1], maximum_size=3, allow_empty=False
            )
            return self.expected_raw_wga(world, query) - (
                self.cost_multiplier * Fraction(len(query), 3)
            )
        if kind == "G":
            query0 = spec[1]
            query1 = spec[2]
            q0 = (
                frozenset()
                if query0 is None
                else self._coerce_query(query0, maximum_size=2, allow_empty=False)
            )
            q1 = (
                frozenset()
                if query1 is None
                else self._coerce_query(query1, maximum_size=2, allow_empty=False)
            )
            raw = (
                self.raw_wga_for_query(world, q0, 0)
                + self.raw_wga_for_query(world, q1, 1)
            ) / 2
            normalized_cost = Fraction(2 + len(q0) + len(q1), 6)
            return raw - self.cost_multiplier * normalized_cost
        raise ValueError(f"unknown per-label policy kind: {kind}")

    def regret_game(self) -> FiniteTreeRegretGame:
        named_specs = self.named_policy_specs
        table = {
            world: {
                name: self.utility_for_spec(world, spec)
                for name, spec in named_specs
            }
            for world in WORLDS
        }
        return FiniteTreeRegretGame.from_mapping(table, common_transcript=True)

    def _tight_mixture(self) -> dict[str, Fraction]:
        root_bridge = self.policy_name(("R", frozenset(("a", "b"))))
        g_bridge = self.policy_name(
            ("G", frozenset(("a", "d0")), frozenset(("b", "d1")))
        )
        return {
            "S": Fraction(2, 9),
            root_bridge: Fraction(5, 9),
            g_bridge: Fraction(2, 9),
        }

    def verify(self) -> dict[str, object]:
        """Machine-check the menu, resource certificate, and embedding."""

        game = self.regret_game()
        assert game.common_transcript
        assert len(self.root_queries) == 63
        assert len(self.post_g_options) == 29
        assert len(game.policies) == 905

        # Normal-form closure for arbitrary atomic R/G ordering.  Before the
        # sole audit, every label response is the same, so the pre-audit set is
        # fixed.  After response h, only the final set Q_h and its cardinality
        # affect WGA and cost.  Conversely, auditing first realizes every pair
        # in the advertised G normal-form menu.
        trace_skeletons = self.interleaved_trace_skeletons
        assert len(trace_skeletons) == 1205
        interleaved_normal_forms = {
            self.normal_form_for_interleaved_trace(skeleton)
            for skeleton in trace_skeletons
        }
        advertised_g_forms = {
            spec for spec in self.policy_specs if spec[0] == "G"
        }
        assert interleaved_normal_forms == advertised_g_forms
        for prefix, addition0, addition1 in trace_skeletons:
            for addition in (addition0, addition1):
                final_query = prefix | addition
                assert 1 + len(prefix) + len(addition) == 1 + len(final_query)

        supports = {
            self.test_support_xy(world, response)
            for world in WORLDS
            for response in (0, 1)
        }
        hidden = {
            self.hidden_group_support(world, response)
            for world in WORLDS
            for response in (0, 1)
        }
        assert len(supports) == 1
        assert len(hidden) == 4

        # At anchor_mass=10 the complement positive cell is at least 2/3,
        # so all legal values collapse to the capped hit-count formula.
        for query in (frozenset(),) + self._subsets(3):
            for world in WORLDS:
                for response in (0, 1):
                    value = self.raw_wga_for_query(world, query, response)
                    if world == "S":
                        assert value == 0
                    else:
                        hits = len(query & self.positive_group_one(world, response))
                        assert value == Fraction(min(hits, 2), 3)

        assert game.oracle_values() == {
            "R": Fraction(4, 9),
            "G": Fraction(1, 3),
            "S": Fraction(0),
        }
        oracle_specs = {
            world: tuple(
                game.policies.index(policy) for policy in game.oracle_policies(world)
            )
            for world in WORLDS
        }
        for indices, expected_kind in zip(
            oracle_specs.values(), ("R", "G", "S")
        ):
            assert all(
                self.named_policy_specs[index][1][0] == expected_kind
                for index in indices
            )
        # Every G-world maximizer has disjoint final branch sets.  An audited
        # realization therefore cannot contain a pre-audit label, proving that
        # the strict optimal root action remains G after allowing interleaving.
        for index in oracle_specs["G"]:
            spec = self.named_policy_specs[index][1]
            assert spec[1] is not None and spec[2] is not None
            assert frozenset(spec[1]).isdisjoint(frozenset(spec[2]))

        # The dual certificate is equivalent to two resource inequalities.
        root_scores: list[Fraction] = []
        for query in self.root_queries:
            raw_r = self.expected_raw_wga("R", query)
            raw_g = self.expected_raw_wga("G", query)
            root_scores.append(raw_r + 4 * raw_g - len(query))
        assert max(root_scores) == 0
        branch_scores: list[Fraction] = []
        for query in (frozenset(),) + self._subsets(2):
            for response in (0, 1):
                raw_r = self.raw_wga_for_query("R", query, response)
                raw_g = self.raw_wga_for_query("G", query, response)
                branch_scores.append(raw_r + 4 * raw_g - 1 - len(query))
        assert max(branch_scores) == 0

        weights = {"R": Fraction(1, 9), "G": Fraction(4, 9), "S": Fraction(4, 9)}
        dual = game.weighted_regret_lower_bound(weights)
        assert dual == Fraction(16, 81)
        certificate = game.certify_tight_mixture(self._tight_mixture(), weights)
        assert certificate["upper_bound"] == Fraction(16, 81)
        assert certificate["regrets"] == {
            world: Fraction(16, 81) for world in WORLDS
        }

        legal_queries = {frozenset()} | set(self._subsets(3))
        assert len(legal_queries) == 64
        for query in legal_queries:
            signs = self.logistic_sign_certificate(query)
            assert all(sign != 0 for sign in signs.values())
            for atom, sign in signs.items():
                assert (int(sign > 0) == self.test_label(atom)) == (
                    self.prediction_is_correct(atom, query)
                )

        alpha = self.logistic_alpha()
        radius = self.explicit_stability_radius()
        dimension = len(self.training_atoms)
        gamma = float(self.l2_coefficient)
        k = (sqrt(dimension) + alpha * sqrt(dimension) / 4.0) / gamma

        def margin_bound(epsilon: float) -> float:
            return alpha - k * epsilon * (1.0 + epsilon) - alpha * sqrt(
                dimension
            ) * epsilon

        assert radius > 0.0
        assert margin_bound(radius * 0.99) > 0.0
        assert margin_bound(radius * 1.01) < 0.0

        def stringify(values: Mapping[str, Fraction]) -> dict[str, str]:
            return {key: str(value) for key, value in values.items()}

        return {
            "scope": "per_label_priced_single_audit_interleaved_menu",
            "repairable_atom_count": len(self.repairable_atoms),
            "root_batch_count": len(self.root_queries),
            "post_g_batch_option_count": len(self.post_g_options),
            "deterministic_policy_count": len(game.policies),
            "deterministic_normal_form_count": len(game.policies),
            "interleaved_trace_skeleton_count": len(trace_skeletons),
            "arbitrary_r_g_ordering": True,
            "maximum_audit_count": 1,
            "total_unit_budget": 3,
            "normal_form_complete": True,
            "same_test_xy_marginal": True,
            "common_transcript": True,
            "oracle_values": stringify(game.oracle_values()),
            "oracle_policy_counts": {
                world: len(game.oracle_policies(world)) for world in WORLDS
            },
            "oracle_action_types": {"R": "R", "G": "G", "S": "S"},
            "dual_world_weights": stringify(weights),
            "dual_lower_bound": str(dual),
            "tight_minimax_regret": str(certificate["upper_bound"]),
            "tight_mixture": stringify(self._tight_mixture()),
            "tight_mixture_regrets": stringify(certificate["regrets"]),
            "max_root_resource_slack": str(max(root_scores)),
            "max_branch_resource_slack": str(max(branch_scores)),
            "pricing_convention": (
                "R costs |Q| units; G costs 1 audit unit plus |Q_h| per branch"
            ),
            "logistic_dimension": dimension,
            "strong_convexity_modulus": str(self.l2_coefficient),
            "one_hot_sign_realization": True,
            "finite_state_local_stability": True,
            "one_hot_margin": alpha,
            "explicit_spectral_radius": radius,
            "repeated_audit_or_larger_budget_extension": "open",
        }


def verify_per_label_batch_three_world_extension() -> dict[str, object]:
    """Machine-check the exact per-label-priced batch theorem."""

    return PerLabelBatchThreeWorldWGATemplate().verify()


@dataclass(frozen=True)
class CapacityBatchThreeWorldWGATemplate:
    """Expanded-menu closure of the sharp three-world witness.

    The root R action may choose any nonempty batch of at most three repairable
    atoms.  After G, each audit branch may stop or choose any nonempty batch of
    at most two atoms.  A batch is *capacity priced*: a root R action consumes
    three units and a continued G branch consumes two units regardless of the
    smaller selected subset.  This is a finite, explicit expansion of the
    original menu, not a per-label-cost or interleaved-action model.

    For the default witness, a finite-game dual certificate with world weights
    ``(1/5, 2/5, 2/5)`` and a matching mixture prove an exact minimax regret of
    ``1/5``.  The lower value relative to ``2/9`` is intentional: adding batch
    choices creates useful cross-world decoy policies.
    """

    witness: SharpThreeWorldWGATemplate = field(
        default_factory=SharpThreeWorldWGATemplate
    )

    def __post_init__(self) -> None:
        if (
            self.witness.anchor_mass != Fraction(12)
            or self.witness.blocker_mass != Fraction(1)
            or self.witness.cost_multiplier != Fraction(1, 3)
        ):
            raise ValueError(
                "the expanded-menu certificate is calibrated to anchor_mass=12, "
                "blocker_mass=1, and cost_multiplier=1/3"
            )

    def _subsets(self, maximum_size: int) -> tuple[frozenset[str], ...]:
        atoms = self.witness.repairable_atoms
        return tuple(
            frozenset(combo)
            for size in range(1, maximum_size + 1)
            for combo in combinations(atoms, size)
        )

    @property
    def root_queries(self) -> tuple[frozenset[str], ...]:
        return self._subsets(3)

    @property
    def post_g_options(self) -> tuple[frozenset[str] | None, ...]:
        return (None,) + self._subsets(2)

    @property
    def policy_specs(self) -> tuple[tuple[object, ...], ...]:
        specs: list[tuple[object, ...]] = [("S",)]
        specs.extend(("R", query) for query in self.root_queries)
        specs.extend(
            ("G", query0, query1)
            for query0 in self.post_g_options
            for query1 in self.post_g_options
        )
        return tuple(specs)

    def _query_name(self, query: frozenset[str] | None) -> str:
        if query is None:
            return "STOP"
        order = {atom: index for index, atom in enumerate(self.witness.repairable_atoms)}
        return ",".join(sorted(query, key=order.__getitem__))

    def policy_name(self, spec: tuple[object, ...]) -> str:
        kind = spec[0]
        if kind == "S":
            return "S"
        if kind == "R":
            query = frozenset(spec[1])
            return f"R[{self._query_name(query)}]"
        if kind == "G":
            query0 = spec[1]
            query1 = spec[2]
            query0 = None if query0 is None else frozenset(query0)
            query1 = None if query1 is None else frozenset(query1)
            return f"G[{self._query_name(query0)}]|[{self._query_name(query1)}]"
        raise ValueError(f"unknown expanded policy kind: {kind}")

    @property
    def named_policy_specs(self) -> tuple[tuple[str, tuple[object, ...]], ...]:
        return tuple(
            (self.policy_name(spec), spec) for spec in self.policy_specs
        )

    def utility_for_spec(
        self, world: str, spec: tuple[object, ...]
    ) -> Fraction:
        if world not in WORLDS:
            raise ValueError(f"unknown world: {world}")
        kind = spec[0]
        if kind == "S":
            return Fraction(0)
        if kind == "R":
            query = frozenset(spec[1])
            raw = sum(
                (
                    self.witness.raw_wga_for_query(world, query, response)
                    for response in (0, 1)
                ),
                Fraction(0),
            ) / 2
            # A root batch is capacity-priced at all three units.
            return raw - self.witness.cost_multiplier
        if kind == "G":
            query0 = spec[1]
            query1 = spec[2]
            query0 = frozenset() if query0 is None else frozenset(query0)
            query1 = frozenset() if query1 is None else frozenset(query1)
            raw = (
                self.witness.raw_wga_for_query(world, query0, 0)
                + self.witness.raw_wga_for_query(world, query1, 1)
            ) / 2
            continues = int(spec[1] is not None) + int(spec[2] is not None)
            normalized_cost = Fraction(1 + continues, 3)
            return raw - self.witness.cost_multiplier * normalized_cost
        raise ValueError(f"unknown expanded policy kind: {kind}")

    def regret_game(self) -> FiniteTreeRegretGame:
        named_specs = self.named_policy_specs
        table = {
            world: {
                name: self.utility_for_spec(world, spec)
                for name, spec in named_specs
            }
            for world in WORLDS
        }
        return FiniteTreeRegretGame.from_mapping(table, common_transcript=True)

    def _tight_mixture(self) -> dict[str, Fraction]:
        decoy_r = self.policy_name(("R", frozenset(("a", "b", "d0"))))
        g_r = self.policy_name(("R", frozenset(("a", "d0", "d1"))))
        return {"S": Fraction(2, 5), decoy_r: Fraction(2, 5), g_r: Fraction(1, 5)}

    def verify(self) -> dict[str, object]:
        """Machine-check the capacity-priced expanded-menu theorem."""

        game = self.regret_game()
        assert game.common_transcript
        assert len(self.root_queries) == 92
        assert len(self.post_g_options) == 37
        assert len(game.policies) == 1462
        assert game.oracle_values() == {
            "R": Fraction(1, 3),
            "G": Fraction(1, 3),
            "S": Fraction(0),
        }

        supports = {
            self.witness.test_support_xy(world, response)
            for world in WORLDS
            for response in (0, 1)
        }
        hidden = {
            self.witness.hidden_group_support(world, response)
            for world in WORLDS
            for response in (0, 1)
        }
        assert len(supports) == 1
        assert len(hidden) == 4

        # The finite cell formulas reduce the weighted raw-value check to a
        # small resource inequality.  The shared atom ``a`` contributes to all
        # three relevant group cells; every other atom contributes to at most
        # one root/branch cell.
        root_scores: list[Fraction] = []
        for query in self.root_queries:
            raw_r = sum(
                (
                    self.witness.raw_wga_for_query("R", query, response)
                    for response in (0, 1)
                ),
                Fraction(0),
            ) / 2
            raw_g = sum(
                (
                    self.witness.raw_wga_for_query("G", query, response)
                    for response in (0, 1)
                ),
                Fraction(0),
            ) / 2
            root_scores.append(raw_r + 2 * raw_g)
        assert max(root_scores) == Fraction(5, 3)

        branch_scores = [
            self.witness.raw_wga_for_query("R", query, response)
            + 2 * self.witness.raw_wga_for_query("G", query, response)
            for query in self._subsets(2)
            for response in (0, 1)
        ]
        assert max(branch_scores) == Fraction(5, 3)

        weights = {"R": Fraction(1, 5), "G": Fraction(2, 5), "S": Fraction(2, 5)}
        dual = game.weighted_regret_lower_bound(weights)
        assert dual == Fraction(1, 5)
        pure_sums = game.pure_regret_sums()
        assert min(pure_sums.values()) == Fraction(1, 2)
        certificate = game.certify_tight_mixture(self._tight_mixture(), weights)
        assert certificate["upper_bound"] == Fraction(1, 5)
        assert certificate["regrets"] == {
            "R": Fraction(1, 5),
            "G": Fraction(1, 5),
            "S": Fraction(1, 5),
        }

        def stringify(values: Mapping[str, Fraction]) -> dict[str, str]:
            return {key: str(value) for key, value in values.items()}
        return {
            "scope": "capacity_priced_arbitrary_batch_menu",
            "root_batch_count": len(self.root_queries),
            "post_g_batch_option_count": len(self.post_g_options),
            "deterministic_policy_count": len(game.policies),
            "same_test_xy_marginal": True,
            "common_transcript": True,
            "oracle_values": stringify(game.oracle_values()),
            "oracle_policy_counts": {
                world: len(game.oracle_policies(world)) for world in WORLDS
            },
            "dual_world_weights": stringify(weights),
            "dual_lower_bound": str(dual),
            "pure_regret_sum_lower_bound": str(min(pure_sums.values())),
            "tight_minimax_regret": str(certificate["upper_bound"]),
            "tight_mixture": stringify(self._tight_mixture()),
            "tight_mixture_regrets": stringify(certificate["regrets"]),
            "max_root_weighted_raw_score": str(max(root_scores)),
            "max_branch_weighted_raw_score": str(max(branch_scores)),
            "pricing_convention": "R=3 units, continued G branch=2 units",
            "same_witness_per_label_or_interleaved_extension": "open",
        }


def verify_sharp_three_world_wga_template() -> dict[str, object]:
    """Machine-check the sharp finite-menu three-world WGA theorem."""

    return SharpThreeWorldWGATemplate().verify()


def verify_capacity_batch_three_world_extension() -> dict[str, object]:
    """Machine-check the expanded capacity-priced batch theorem."""

    return CapacityBatchThreeWorldWGATemplate().verify()
