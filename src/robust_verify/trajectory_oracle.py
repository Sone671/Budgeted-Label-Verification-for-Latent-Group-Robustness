"""Private finite-depth dynamic-programming oracle for the MVP.

This module is for offline analysis only.  It accepts child utilities that a
deployable policy is forbidden to read; :mod:`action_value` has no dependency
on this module, which keeps the legal/private boundary visible in the package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from robust_verify.sequential_audit import AuditAction


@dataclass(frozen=True)
class TreeTransition:
    action: AuditAction
    child_node: str
    immediate_value: float

    def __post_init__(self) -> None:
        if not self.child_node:
            raise ValueError("child_node must be non-empty")
        if not np.isfinite(self.immediate_value):
            raise ValueError("immediate_value must be finite")


@dataclass(frozen=True)
class TreeNode:
    node_id: str
    stop_value: float = 0.0
    transitions: tuple[TreeTransition, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if not np.isfinite(self.stop_value):
            raise ValueError("stop_value must be finite")
        actions = [transition.action for transition in self.transitions]
        if len(set(actions)) != len(actions):
            raise ValueError("a tree node cannot contain duplicate actions")


@dataclass(frozen=True)
class OracleDecision:
    node_id: str
    depth_remaining: int
    value: float
    action: AuditAction
    action_values: tuple[tuple[AuditAction, float], ...]


@dataclass(frozen=True)
class OracleSolution:
    root: OracleDecision
    decisions: Mapping[tuple[str, int], OracleDecision]


def solve_dynamic_program(
    nodes: Mapping[str, TreeNode] | Sequence[TreeNode],
    root_node: str,
    *,
    max_depth: int,
) -> OracleSolution:
    """Solve a finite action tree by backward induction.

    ``immediate_value`` is already net of the action cost.  Stop is assigned
    the node's explicit value, normally zero for a cost-regularized objective.
    Ties prefer Stop, then deterministic action ordering.
    """

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    node_map = {node.node_id: node for node in nodes} if not isinstance(nodes, Mapping) else dict(nodes)
    if root_node not in node_map:
        raise KeyError(f"unknown root node: {root_node}")
    memo: dict[tuple[str, int], OracleDecision] = {}
    visiting: set[tuple[str, int]] = set()

    def visit(node_id: str, depth: int) -> OracleDecision:
        key = (node_id, depth)
        if key in memo:
            return memo[key]
        if key in visiting:
            raise ValueError("action tree contains a cycle at a fixed depth")
        node = node_map.get(node_id)
        if node is None:
            raise KeyError(f"unknown child node: {node_id}")
        visiting.add(key)
        stop = AuditAction.stop()
        values: list[tuple[AuditAction, float]] = [(stop, float(node.stop_value))]
        if depth > 0:
            for transition in node.transitions:
                child = visit(transition.child_node, depth - 1)
                values.append(
                    (transition.action, float(transition.immediate_value + child.value))
                )
        # Stop wins exact ties; otherwise use stable action metadata.
        best_action, best_value = max(
            values,
            key=lambda item: (
                item[1],
                1 if item[0].kind == "stop" else 0,
                "" if item[0].kind == "stop" else item[0].kind,
                () if item[0].kind == "stop" else item[0].sample_ids,
            ),
        )
        decision = OracleDecision(
            node_id=node_id,
            depth_remaining=depth,
            value=float(best_value),
            action=best_action,
            action_values=tuple(values),
        )
        visiting.remove(key)
        memo[key] = decision
        return decision

    root = visit(root_node, max_depth)
    return OracleSolution(root=root, decisions=dict(memo))


def classify_policy_decision(
    chosen_action: AuditAction,
    oracle_decision: OracleDecision,
) -> str:
    """Classify a legal choice against the private oracle at one node."""

    if chosen_action == oracle_decision.action:
        return "oracle_optimal"
    if chosen_action.kind == "stop":
        return "premature_stop"
    if oracle_decision.action.kind == "stop":
        return "bad_continue"
    return "wrong_action_type"
