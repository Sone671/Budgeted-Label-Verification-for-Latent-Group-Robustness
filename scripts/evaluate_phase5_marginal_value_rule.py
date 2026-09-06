#!/usr/bin/env python
"""Join frozen Phase-5 legal decisions to existing Phase-4 outcomes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


KEYS = ["dataset", "noise_name", "seed"]


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1"}:
            return True
        if value.strip().lower() in {"false", "0"}:
            return False
        raise ValueError(f"cannot parse boolean value {value!r}")
    return bool(value)


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


def evaluate(
    feature_path: Path,
    result_paths: list[Path],
    phase4_selector_path: Path,
    output_dir: Path,
    *,
    scout_share: float,
    expanded_share: float,
    minimum_improvement_pp: float,
    maximum_regret_pp: float,
) -> None:
    features = pd.read_csv(feature_path)
    forbidden = {
        "wga", "delta_wga", "test_wga", "actual_test_weak_group_offline",
        "actual_validation_weak_group_offline", "attribute_balanced_accuracy_offline",
    }
    leaked = forbidden & set(features.columns)
    if leaked:
        raise ValueError(f"legal feature file contains forbidden outcome columns: {sorted(leaked)}")
    if features.duplicated(KEYS).any():
        raise ValueError("legal feature keys are not unique")

    outcomes = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    sparse = outcomes[outcomes["role"].eq("joint_sparse_attribute")]
    rows: list[dict[str, object]] = []
    for feature in features.to_dict("records"):
        key_mask = (
            outcomes["dataset"].eq(feature["dataset"])
            & outcomes["noise_name"].eq(feature["noise_name"])
            & outcomes["seed"].astype(int).eq(int(feature["seed"]))
        )
        expand = _as_bool(feature["expand_by_marginal_value"])
        selected_share = expanded_share if expand else scout_share

        def one(role: str, share: float) -> pd.Series:
            subset = outcomes[
                key_mask
                & outcomes["role"].eq(role)
                & np.isclose(outcomes["group_share"], share)
            ]
            if len(subset) != 1:
                raise ValueError(
                    f"expected one {role} row for "
                    f"{(feature['dataset'], feature['noise_name'], feature['seed'], share)}"
                )
            return subset.iloc[0]

        arm25 = one("joint_sparse_attribute", scout_share)
        arm50 = one("joint_sparse_attribute", expanded_share)
        selected = arm50 if expand else arm25
        total = one("label_only_total_budget", selected_share)
        same = one("label_only_same_label_budget", selected_share)
        selected_delta = float(selected["delta_wga"])
        delta25 = float(arm25["delta_wga"])
        delta50 = float(arm50["delta_wga"])
        rows.append({
            "dataset": feature["dataset"],
            "noise_name": feature["noise_name"],
            "seed": int(feature["seed"]),
            "marginal_net_value": float(feature["marginal_net_value"]),
            "marginal_value_ratio": float(feature["marginal_value_ratio"]),
            "selected_group_share": selected_share,
            "expanded": expand,
            "delta_wga": selected_delta,
            "delta_wga_vs_total_budget": selected_delta - float(total["delta_wga"]),
            "delta_wga_vs_same_label_budget": selected_delta - float(same["delta_wga"]),
            "outcome_delta_50_minus_25": delta50 - delta25,
            "selection_regret": max(delta25, delta50) - selected_delta,
        })

    decision = pd.DataFrame(rows).sort_values(KEYS, kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    decision.to_csv(output_dir / "decision_results.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    report = [
        "# Phase 5-A marginal value screen", "",
        "All WGA values are percentage points with paired seed-level 95% t intervals.", "",
        "| Dataset | Noise | Selected dWGA | vs same label count | vs total-label budget | Selection regret | Expanded |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    negative_conditions = 0
    for (dataset, noise), condition in decision.groupby(["dataset", "noise_name"], sort=True):
        strict_mean, strict_low, strict_high = _interval(condition["delta_wga_vs_total_budget"])
        if strict_high < 0.0:
            negative_conditions += 1
        report.append(
            f"| {dataset} | {noise} | {_formatted(condition['delta_wga'])} | "
            f"{_formatted(condition['delta_wga_vs_same_label_budget'])} | "
            f"{_formatted(condition['delta_wga_vs_total_budget'])} | "
            f"{_formatted(condition['selection_regret'])} | "
            f"{int(condition['expanded'].sum())}/{len(condition)} |"
        )
        for measure in (
            "delta_wga", "delta_wga_vs_same_label_budget",
            "delta_wga_vs_total_budget", "selection_regret",
        ):
            mean, low, high = _interval(condition[measure])
            summary_rows.append({
                "dataset": dataset, "noise_name": noise, "measure": measure,
                "n_seeds": len(condition), "mean": mean,
                "ci95_low": low, "ci95_high": high,
                "expanded_seeds": int(condition["expanded"].sum()),
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "condition_summary.csv", index=False)

    phase4 = pd.read_csv(phase4_selector_path).sort_values(KEYS, kind="stable")
    aligned = decision.merge(
        phase4[KEYS + ["delta_wga_vs_total_budget"]],
        on=KEYS,
        suffixes=("_phase5", "_phase4"),
        validate="one_to_one",
    )
    strict_phase5_pp = 100 * float(decision["delta_wga_vs_total_budget"].mean())
    strict_phase4_pp = 100 * float(aligned["delta_wga_vs_total_budget_phase4"].mean())
    improvement_pp = strict_phase5_pp - strict_phase4_pp
    regret_pp = 100 * float(decision["selection_regret"].mean())
    pass_improvement = improvement_pp >= minimum_improvement_pp
    pass_regret = regret_pp <= maximum_regret_pp
    pass_negative = negative_conditions == 0
    gate_passed = pass_improvement and pass_regret and pass_negative
    gate = {
        "phase5_mean_strict_contrast_pp": strict_phase5_pp,
        "phase4_mean_strict_contrast_pp": strict_phase4_pp,
        "strict_improvement_pp": improvement_pp,
        "required_improvement_pp": minimum_improvement_pp,
        "mean_selection_regret_pp": regret_pp,
        "maximum_regret_pp": maximum_regret_pp,
        "strictly_negative_conditions": negative_conditions,
        "expanded_conditions": int(decision["expanded"].sum()),
        "total_conditions": len(decision),
        "pass_improvement": pass_improvement,
        "pass_regret": pass_regret,
        "pass_no_negative_condition": pass_negative,
        "gate_passed": gate_passed,
    }
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report.extend([
        "", "## Development gate", "",
        f"- Phase 5 mean strict contrast: {strict_phase5_pp:+.3f} pp.",
        f"- Phase 4 mean strict contrast: {strict_phase4_pp:+.3f} pp.",
        f"- Improvement: {improvement_pp:+.3f} pp; required: at least {minimum_improvement_pp:.3f} pp.",
        f"- Mean selection regret: {regret_pp:.3f} pp; maximum: {maximum_regret_pp:.3f} pp.",
        f"- Strictly negative dataset/noise conditions: {negative_conditions}.",
        f"- Marginal-value rule expanded in {int(decision['expanded'].sum())}/{len(decision)} conditions.",
        f"- Gate decision: {'PASS' if gate_passed else 'FAIL'}.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--phase4-selector", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scout-share", type=float, default=0.25)
    parser.add_argument("--expanded-share", type=float, default=0.50)
    parser.add_argument("--minimum-improvement-pp", type=float, default=0.50)
    parser.add_argument("--maximum-regret-pp", type=float, default=0.50)
    args = parser.parse_args()
    evaluate(
        Path(args.features).resolve(),
        [Path(path).resolve() for path in args.results],
        Path(args.phase4_selector).resolve(),
        Path(args.output_dir).resolve(),
        scout_share=args.scout_share,
        expanded_share=args.expanded_share,
        minimum_improvement_pp=args.minimum_improvement_pp,
        maximum_regret_pp=args.maximum_regret_pp,
    )


if __name__ == "__main__":
    main()

