#!/usr/bin/env python
"""Apply the frozen two-noise Phase-11 ACS confirmation decision."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.cost_budget_analysis import (
    pair_phase10_results,
    phase11_confirmation_decision,
    summarize_phase10_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    joint = results[results["role"].astype(str) == "joint_sparse_attribute"]
    integrity = {
        "joint_rows": len(joint),
        "duplicate_joint_cells": int(
            joint.duplicated(
                [
                    "dataset",
                    "noise_name",
                    "seed",
                    "budget_fraction",
                    "group_to_label_cost_ratio",
                    "target_group_action_share",
                ]
            ).sum()
        ),
        "cost_mismatch_rows": int(
            ((joint["total_cost_used"] - joint["budget_capacity"]).abs() > 1e-9).sum()
        ),
        "saturated_rows": int(joint["audit_pool_saturated"].astype(bool).sum()),
        "unique_seeds": sorted(map(int, joint["seed"].unique())),
        "unique_noises": sorted(map(str, joint["noise_name"].unique())),
    }
    expected_integrity = (
        integrity["joint_rows"] == 20
        and integrity["duplicate_joint_cells"] == 0
        and integrity["cost_mismatch_rows"] == 0
        and integrity["saturated_rows"] == 0
        and integrity["unique_seeds"] == list(range(10))
        and integrity["unique_noises"] == ["minority_high_40", "uniform20"]
    )
    if not expected_integrity:
        raise ValueError(f"Phase-11 integrity failed: {integrity}")

    paired = pair_phase10_results(results)
    summary = summarize_phase10_pairs(paired, expected_seed_count=10)
    decision = phase11_confirmation_decision(summary)
    decision["integrity"] = integrity
    decision["effects"] = [
        {
            "noise_name": str(row.noise_name),
            "mean": float(row.strict_cost_mean),
            "ci_low": float(row.strict_cost_ci_low),
            "ci_high": float(row.strict_cost_ci_high),
            "p_value": float(row.strict_cost_p_value),
            "cell_label": str(row.cell_label),
        }
        for row in summary.itertuples(index=False)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
