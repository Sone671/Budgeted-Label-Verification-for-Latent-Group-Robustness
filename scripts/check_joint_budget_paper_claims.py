#!/usr/bin/env python
"""Check headline provenance and prohibited claims in joint-budget manuscripts."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT_ROOT = ROOT / "joint_group_label_budget"
FACTS_PATH = JOINT_ROOT / "paper_assets" / "paper_facts.json"
DRAFT_PATH = JOINT_ROOT / "MAIN_CONFERENCE_DRAFT_EN.md"
MANUSCRIPT_PATHS = (
    JOINT_ROOT / "paper" / "main.tex",
    JOINT_ROOT / "paper" / "appendix.tex",
)


def main() -> None:
    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    draft = DRAFT_PATH.read_text(encoding="utf-8")
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS
    )
    lower = facts["theory"]["same_marginal_lower_bound"]
    stability = facts["theory"]["nonorthogonal_stability"]
    phase10 = facts["phase10"]
    phase11 = {row["noise_name"]: row for row in facts["phase11"]["effects"]}

    required = {
        "policy vertices": str(lower["deterministic_policy_count"]),
        "tight regret": str(lower["tight_randomized_worst_regret_exact"]),
        "stability radius": f'{stability["certified_spectral_radius"]:.6f}',
        "Phase-10 cells": f'{phase10["primary_cell_count"]} Phase-10',
        "common raw cells": (
            f'{phase10["common_raw_beneficial_cell_count"]} '
            "noise-budget-cost-share"
        ),
        "common Holm cells": (
            f'leaves {phase10["common_holm_stable_positive_cell_count"]} common cells'
        ),
        "ACS uniform effect": f'{phase11["uniform20"]["mean_pp"]:+.3f}',
        "ACS uniform interval": (
            f'[{phase11["uniform20"]["ci_low_pp"]:.3f}, '
            f'{phase11["uniform20"]["ci_high_pp"]:.3f}]'
        ),
        "ACS minority effect": f'{phase11["minority_high_40"]["mean_pp"]:.3f}',
        "ACS minority interval": (
            f'[{phase11["minority_high_40"]["ci_low_pp"]:.3f}, '
            f'{phase11["minority_high_40"]["ci_high_pp"]:.3f}]'
        ),
        "partial decision": "partial external confirmation",
        "recovery ownership": "cost-aware extension of the one-action",
    }
    missing = {name: value for name, value in required.items() if value not in draft}
    required_manuscript = {
        "round-half-up budget capacity": r"K=\lfloor bN+1/2\rfloor",
        "explicit plug-in value": r"\widehat V_h(m,Q)=\widehat I_h(m)",
        "linear-head retraining scope": "all trainable linear-head parameters",
        "average-accuracy provenance macro": r"\ACSUniformAvgMeanPP",
        "direct data-selection citation": "jain2024improving",
        "direct group-inference citation": "han2024improving",
        "partial decision in canonical paper": "partial external confirmation",
    }
    missing.update(
        {
            f"canonical {name}": value
            for name, value in required_manuscript.items()
            if value not in manuscript
        }
    )

    prohibited = [
        "Sparse group audits universally improve robustness.",
        "A 25% or 50% audit share is optimal.",
        "ACS confirms cross-dataset and cross-noise transfer.",
        "The lower bound applies to arbitrary neural representations.",
    ]
    all_text = f"{draft}\n{manuscript}"
    violations = [phrase for phrase in prohibited if phrase in all_text]
    if "preregistered" in manuscript.lower():
        violations.append("canonical paper overstates the protocol as preregistered")
    if re.search(r"CivilComments.{0,80}held-out|held-out.{0,80}CivilComments", all_text, re.I | re.S):
        violations.append("CivilComments is described as held-out")

    incomplete_secondary = {
        dataset: details["same_label_rows"]
        for dataset, details in phase10["result_row_integrity"].items()
        if not details["same_label_complete"]
    }
    mechanism_complete = phase10.get("mechanism_complete_cell_count", 0)
    if mechanism_complete != phase10["primary_cell_count"]:
        incomplete_secondary["mechanism_complete_cells"] = mechanism_complete
    report = {
        "status": (
            "pass"
            if not missing and not violations and not incomplete_secondary
            else "fail"
        ),
        "missing_required_evidence_strings": missing,
        "prohibited_claim_violations": violations,
        "incomplete_phase10_same_label_rows": incomplete_secondary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
