#!/usr/bin/env python
"""Analyze Phase-6 sequential feedback against Phase-4 paired controls."""
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
    array = values.to_numpy(dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, math.nan, math.nan
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return mean, mean - half, mean + half


def _formatted(values: pd.Series) -> str:
    mean, low, high = _interval(values)
    return f"{100 * mean:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def analyze(
    sequential_paths: list[Path],
    phase4_paths: list[Path],
    output_dir: Path,
    *,
    minimum_improvement_pp: float,
) -> None:
    sequential = pd.concat([pd.read_csv(path) for path in sequential_paths], ignore_index=True)
    phase4 = pd.concat([pd.read_csv(path) for path in phase4_paths], ignore_index=True)
    if (sequential["human_action_count"].astype(int) != sequential["total_budget_count"].astype(int)).any():
        raise ValueError("sequential row violates total action cost")
    if sequential.duplicated(KEYS).any():
        raise ValueError("sequential result keys are not unique")

    rows: list[dict[str, object]] = []
    for row in sequential.to_dict("records"):
        mask = (
            phase4["dataset"].eq(row["dataset"])
            & phase4["noise_name"].eq(row["noise_name"])
            & phase4["seed"].astype(int).eq(int(row["seed"]))
            & np.isclose(phase4["group_share"], 0.25)
        )

        def one(role: str) -> pd.Series:
            subset = phase4[mask & phase4["role"].eq(role)]
            if len(subset) != 1:
                raise ValueError(f"missing {role} comparator for {tuple(row[k] for k in KEYS)}")
            return subset.iloc[0]

        one_shot = one("joint_sparse_attribute")
        total = one("label_only_total_budget")
        same = one("label_only_same_label_budget")
        delta = float(row["delta_wga"])
        rows.append({
            **{key: row[key] for key in KEYS},
            "delta_wga": delta,
            "delta_wga_vs_one_shot_25": delta - float(one_shot["delta_wga"]),
            "delta_wga_vs_total_budget": delta - float(total["delta_wga"]),
            "delta_wga_vs_same_label_budget": delta - float(same["delta_wga"]),
            "first_batch_noise_precision": float(row["first_batch_noise_precision"]),
            "first_batch_corrections": int(row["first_batch_corrections"]),
        })
    paired = pd.DataFrame(rows).sort_values(KEYS, kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_results.csv", index=False)

    report = [
        "# Phase 6 sequential feedback report", "",
        "All WGA values are percentage points with paired seed-level 95% t intervals.", "",
        "| Dataset | Noise | Sequential dWGA | vs one-shot 25% | vs same label count | vs total-label budget |",
        "|---|---|---:|---:|---:|---:|",
    ]
    summary_rows: list[dict[str, object]] = []
    negative_one_shot = 0
    negative_total = 0
    for (dataset, noise), condition in paired.groupby(["dataset", "noise_name"], sort=True):
        _, _, one_high = _interval(condition["delta_wga_vs_one_shot_25"])
        _, _, total_high = _interval(condition["delta_wga_vs_total_budget"])
        negative_one_shot += int(one_high < 0.0)
        negative_total += int(total_high < 0.0)
        report.append(
            f"| {dataset} | {noise} | {_formatted(condition['delta_wga'])} | "
            f"{_formatted(condition['delta_wga_vs_one_shot_25'])} | "
            f"{_formatted(condition['delta_wga_vs_same_label_budget'])} | "
            f"{_formatted(condition['delta_wga_vs_total_budget'])} |"
        )
        for measure in (
            "delta_wga", "delta_wga_vs_one_shot_25",
            "delta_wga_vs_same_label_budget", "delta_wga_vs_total_budget",
        ):
            mean, low, high = _interval(condition[measure])
            summary_rows.append({
                "dataset": dataset, "noise_name": noise, "measure": measure,
                "n_seeds": len(condition), "mean": mean,
                "ci95_low": low, "ci95_high": high,
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "condition_summary.csv", index=False)

    improvement_pp = 100 * float(paired["delta_wga_vs_one_shot_25"].mean())
    gate = {
        "n_rows": len(paired),
        "mean_improvement_over_one_shot_pp": improvement_pp,
        "minimum_improvement_pp": minimum_improvement_pp,
        "negative_vs_one_shot_conditions": negative_one_shot,
        "negative_vs_total_budget_conditions": negative_total,
        "pass_improvement": improvement_pp >= minimum_improvement_pp,
        "pass_no_negative_vs_one_shot": negative_one_shot == 0,
        "pass_no_negative_vs_total": negative_total == 0,
    }
    gate["gate_passed"] = all(
        gate[name]
        for name in (
            "pass_improvement", "pass_no_negative_vs_one_shot",
            "pass_no_negative_vs_total",
        )
    )
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report.extend([
        "", "## Development gate", "",
        f"- Mean improvement over one-shot 25%: {improvement_pp:+.3f} pp; required: {minimum_improvement_pp:+.3f} pp.",
        f"- Strictly negative conditions versus one-shot 25%: {negative_one_shot}.",
        f"- Strictly negative conditions versus total-label budget: {negative_total}.",
        f"- Gate decision: {'PASS' if gate['gate_passed'] else 'FAIL'}.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-results", nargs="+", required=True)
    parser.add_argument("--phase4-results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-improvement-pp", type=float, default=0.25)
    args = parser.parse_args()
    analyze(
        [Path(path).resolve() for path in args.sequential_results],
        [Path(path).resolve() for path in args.phase4_results],
        Path(args.output_dir).resolve(),
        minimum_improvement_pp=args.minimum_improvement_pp,
    )


if __name__ == "__main__":
    main()

