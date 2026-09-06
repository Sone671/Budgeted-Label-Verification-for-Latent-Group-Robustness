#!/usr/bin/env python
"""Analyze Phase-7 exploration feedback against Phase-6 and cost controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


KEYS = ["dataset", "noise_name", "seed"]


def _interval(values: pd.Series) -> tuple[float, float, float]:
    value = values.to_numpy(dtype=np.float64)
    mean = float(value.mean())
    if len(value) < 2:
        return mean, math.nan, math.nan
    half = float(t.ppf(0.975, len(value) - 1) * value.std(ddof=1) / np.sqrt(len(value)))
    return mean, mean - half, mean + half


def _formatted(values: pd.Series) -> str:
    mean, low, high = _interval(values)
    return f"{100 * mean:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def analyze(
    phase7_paths: list[Path],
    phase6_paths: list[Path],
    phase4_paths: list[Path],
    output_dir: Path,
    *,
    minimum_improvement_pp: float,
) -> None:
    phase7 = pd.concat([pd.read_csv(path) for path in phase7_paths], ignore_index=True)
    phase6 = pd.concat([pd.read_csv(path) for path in phase6_paths], ignore_index=True)
    phase4 = pd.concat([pd.read_csv(path) for path in phase4_paths], ignore_index=True)
    if phase7.duplicated(KEYS).any():
        raise ValueError("Phase-7 keys are not unique")
    if (phase7["human_action_count"].astype(int) != phase7["total_budget_count"].astype(int)).any():
        raise ValueError("Phase-7 cost mismatch")

    rows: list[dict[str, object]] = []
    for row in phase7.to_dict("records"):
        key_mask6 = (
            phase6["dataset"].eq(row["dataset"])
            & phase6["noise_name"].eq(row["noise_name"])
            & phase6["seed"].astype(int).eq(int(row["seed"]))
        )
        comparator6 = phase6[key_mask6]
        if len(comparator6) != 1:
            raise ValueError(f"missing Phase-6 comparator for {tuple(row[k] for k in KEYS)}")

        key_mask4 = (
            phase4["dataset"].eq(row["dataset"])
            & phase4["noise_name"].eq(row["noise_name"])
            & phase4["seed"].astype(int).eq(int(row["seed"]))
            & np.isclose(phase4["group_share"], 0.25)
        )

        def phase4_one(role: str) -> pd.Series:
            subset = phase4[key_mask4 & phase4["role"].eq(role)]
            if len(subset) != 1:
                raise ValueError(f"missing {role} comparator for {tuple(row[k] for k in KEYS)}")
            return subset.iloc[0]

        one_shot = phase4_one("joint_sparse_attribute")
        total = phase4_one("label_only_total_budget")
        delta = float(row["delta_wga"])
        rows.append({
            **{key: row[key] for key in KEYS},
            "delta_wga": delta,
            "delta_wga_vs_phase6": delta - float(comparator6.iloc[0]["delta_wga"]),
            "delta_wga_vs_one_shot_25": delta - float(one_shot["delta_wga"]),
            "delta_wga_vs_total_budget": delta - float(total["delta_wga"]),
            "exploration_batch_noise_precision": float(
                row["exploration_batch_noise_precision"]
            ),
            "exploration_batch_corrections": int(row["exploration_batch_corrections"]),
        })
    paired = pd.DataFrame(rows).sort_values(KEYS, kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_results.csv", index=False)

    report = [
        "# Phase 7 exploration-feedback report", "",
        "All WGA values are percentage points with paired seed-level 95% t intervals.", "",
        "| Dataset | Noise | Phase-7 dWGA | vs Phase 6 | vs one-shot 25% | vs total-label budget |",
        "|---|---|---:|---:|---:|---:|",
    ]
    summary_rows: list[dict[str, object]] = []
    negative_phase6 = 0
    negative_total = 0
    for (dataset, noise), condition in paired.groupby(["dataset", "noise_name"], sort=True):
        _, _, phase6_high = _interval(condition["delta_wga_vs_phase6"])
        _, _, total_high = _interval(condition["delta_wga_vs_total_budget"])
        negative_phase6 += int(phase6_high < 0.0)
        negative_total += int(total_high < 0.0)
        report.append(
            f"| {dataset} | {noise} | {_formatted(condition['delta_wga'])} | "
            f"{_formatted(condition['delta_wga_vs_phase6'])} | "
            f"{_formatted(condition['delta_wga_vs_one_shot_25'])} | "
            f"{_formatted(condition['delta_wga_vs_total_budget'])} |"
        )
        for measure in (
            "delta_wga", "delta_wga_vs_phase6", "delta_wga_vs_one_shot_25",
            "delta_wga_vs_total_budget",
        ):
            mean, low, high = _interval(condition[measure])
            summary_rows.append({
                "dataset": dataset, "noise_name": noise, "measure": measure,
                "n_seeds": len(condition), "mean": mean,
                "ci95_low": low, "ci95_high": high,
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "condition_summary.csv", index=False)

    improvement_pp = 100 * float(paired["delta_wga_vs_phase6"].mean())
    gate = {
        "n_rows": len(paired),
        "mean_improvement_over_phase6_pp": improvement_pp,
        "minimum_improvement_pp": minimum_improvement_pp,
        "negative_vs_phase6_conditions": negative_phase6,
        "negative_vs_total_budget_conditions": negative_total,
        "pass_improvement": improvement_pp >= minimum_improvement_pp,
        "pass_no_negative_vs_phase6": negative_phase6 == 0,
        "pass_no_negative_vs_total": negative_total == 0,
    }
    gate["gate_passed"] = all(
        gate[name]
        for name in (
            "pass_improvement", "pass_no_negative_vs_phase6",
            "pass_no_negative_vs_total",
        )
    )
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report.extend([
        "", "## Development gate", "",
        f"- Mean improvement over Phase 6: {improvement_pp:+.3f} pp; required: {minimum_improvement_pp:+.3f} pp.",
        f"- Strictly negative conditions versus Phase 6: {negative_phase6}.",
        f"- Strictly negative conditions versus total-label budget: {negative_total}.",
        f"- Gate decision: {'PASS' if gate['gate_passed'] else 'FAIL'}.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-results", nargs="+", required=True)
    parser.add_argument("--phase6-results", nargs="+", required=True)
    parser.add_argument("--phase4-results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-improvement-pp", type=float, default=0.25)
    args = parser.parse_args()
    analyze(
        [Path(path).resolve() for path in args.phase7_results],
        [Path(path).resolve() for path in args.phase6_results],
        [Path(path).resolve() for path in args.phase4_results],
        Path(args.output_dir).resolve(),
        minimum_improvement_pp=args.minimum_improvement_pp,
    )


if __name__ == "__main__":
    main()

