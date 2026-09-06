"""Run a dependency-light end-to-end smoke check for the three-action MVP.

This intentionally uses a toy tree.  It verifies the protocol machinery and
information boundary before any dataset-specific private WGA outcomes are
opened.
"""
from __future__ import annotations

import json
from fractions import Fraction

from robust_verify.action_value import build_legal_state_features, certify_action, proxy_action_values
from robust_verify.sequential_audit import (
    AuditAction,
    SequentialAuditConfig,
    SequentialAuditState,
    apply_action,
    make_batch_action,
    replay,
    state_to_json,
)
from robust_verify.trajectory_oracle import TreeNode, TreeTransition, solve_dynamic_program


def main() -> None:
    config = SequentialAuditConfig(
        budget=Fraction(3),
        task_label_cost=Fraction(1),
        attribute_cost=Fraction(1, 2),
        task_pool_size=5,
        attribute_pool_size=4,
        max_rounds=4,
    )
    initial = SequentialAuditState.initial([0, 0, 1, 1, 0])
    group = make_batch_action("annotate_group", [0, 1], config)
    task = make_batch_action("verify_task", [3], config)
    after_group = apply_action(initial, group, config, responses=[1, 0])
    after_task = apply_action(after_group, task, config, responses=[0])
    stopped = apply_action(after_task, AuditAction.stop(), config)
    replayed = replay(initial.current_labels, config, stopped.transcript)
    assert replayed.current_labels.tolist() == stopped.current_labels.tolist()
    assert replayed.audited_attributes == stopped.audited_attributes

    legal = build_legal_state_features(after_group, config)
    certificate = certify_action(
        proxy_action_values(
            task_action=task,
            group_action=make_batch_action("annotate_group", [2], config),
            task_value=0.25,
            group_value=-0.05,
            error=0.05,
        )
    )
    terminal = TreeNode("terminal")
    tree = {
        "terminal": terminal,
        "root": TreeNode(
            "root",
            transitions=(
                TreeTransition(task, "terminal", 0.4),
                TreeTransition(group, "terminal", 0.2),
            ),
        ),
    }
    oracle = solve_dynamic_program(tree, "root", max_depth=1).root
    print(
        json.dumps(
            {
                "status": "ok",
                "trajectory_rounds": stopped.round_index,
                "spent_cost": str(stopped.spent_cost),
                "remaining_budget": str(stopped.remaining_budget(config)),
                "legal_verified_count": legal.verified_count,
                "legal_audited_attribute_count": legal.audited_attribute_count,
                "proxy_certificate": certificate.status,
                "oracle_action": oracle.action.kind,
                "oracle_value": oracle.value,
                "state_json_bytes": len(state_to_json(stopped).encode("utf-8")),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
