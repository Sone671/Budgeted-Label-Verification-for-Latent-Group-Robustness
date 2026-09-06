"""Run a synthetic public-only R/G/Stop interval-planning smoke test."""
from __future__ import annotations

import json
from fractions import Fraction

from robust_verify.public_value import (
    PublicDecisionNode,
    PublicOutcomeModel,
    PublicTransition,
    ValueInterval,
    solve_robust_bellman,
)
from robust_verify.sequential_audit import AuditAction


def _action(kind: str, sample_id: int) -> AuditAction:
    return AuditAction(kind=kind, sample_ids=(sample_id,), cost=Fraction(1, 1))


def _transition(
    action: AuditAction,
    child: str,
    value: tuple[float, float],
) -> PublicTransition:
    return PublicTransition(
        action=action,
        child_nodes=(child,),
        outcome_models=(PublicOutcomeModel((1.0,)),),
        immediate_value=ValueInterval(*value),
    )


def main() -> None:
    verify = _action("verify_task", 0)
    audit = _action("annotate_group", 0)
    terminal = PublicDecisionNode("terminal")
    scenarios = {
        "verify": PublicDecisionNode(
            "verify",
            transitions=(
                _transition(verify, "terminal", (0.20, 0.30)),
                _transition(audit, "terminal", (0.05, 0.10)),
            ),
        ),
        "audit": PublicDecisionNode(
            "audit",
            transitions=(
                _transition(verify, "terminal", (0.02, 0.08)),
                _transition(audit, "terminal", (0.15, 0.25)),
            ),
        ),
        "stop": PublicDecisionNode(
            "stop",
            transitions=(
                _transition(verify, "terminal", (-0.20, -0.05)),
                _transition(audit, "terminal", (-0.10, 0.00)),
            ),
        ),
        "unresolved": PublicDecisionNode(
            "unresolved",
            transitions=(
                _transition(verify, "terminal", (-0.05, 0.15)),
                _transition(audit, "terminal", (-0.02, 0.12)),
            ),
        ),
    }
    output: dict[str, object] = {}
    for name, root in scenarios.items():
        decision = solve_robust_bellman(
            {"terminal": terminal, root.node_id: root},
            root.node_id,
            max_depth=1,
        ).root
        output[name] = {
            "certificate": decision.certificate.status,
            "selected_action": (
                None
                if decision.certificate.selected_action is None
                else decision.certificate.selected_action.kind
            ),
            "dominant_action": (
                None if decision.dominant_action is None else decision.dominant_action.kind
            ),
            "minimax_fallback": decision.minimax_action.kind,
            "value_interval": [decision.value.lower, decision.value.upper],
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
