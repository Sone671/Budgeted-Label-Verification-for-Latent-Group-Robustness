#!/usr/bin/env python
"""Create paired statistics and a concise report for the joint-budget pilot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.joint_budget import select_group_audit_share


CORE_ROLES = [
    "joint_sparse_attribute",
    "label_only_same_label_budget",
    "label_only_total_budget",
]
OPTIONAL_ROLES = ["oracle_attribute"]


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(dtype=np.float64)
    if len(array) < 2:
        return float(array.mean()), float("nan"), float("nan")
    mean = float(array.mean())
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return mean, mean - half, mean + half


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def analyze(result_path: Path, output_dir: Path, *, selector_threshold: float) -> None:
    frame = pd.read_csv(result_path)
    if "dataset" not in frame:
        frame["dataset"] = "waterbirds"
    required = {"noise_name", "seed", "group_share", "role", "delta_wga"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"results lack required columns: {missing}")

    wide = frame.pivot_table(
        index=["dataset", "noise_name", "seed", "group_share"],
        columns="role",
        values="delta_wga",
        aggfunc="first",
    ).reset_index()
    missing_roles = sorted(set(CORE_ROLES).difference(wide.columns))
    if missing_roles:
        raise ValueError(f"results lack required roles: {missing_roles}")
    wide["joint_vs_same_label_budget"] = (
        wide["joint_sparse_attribute"] - wide["label_only_same_label_budget"]
    )
    wide["joint_vs_total_budget"] = (
        wide["joint_sparse_attribute"] - wide["label_only_total_budget"]
    )
    if "oracle_attribute" in wide:
        wide["joint_vs_full_attribute"] = (
            wide["joint_sparse_attribute"] - wide["oracle_attribute"]
        )
    wide.to_csv(output_dir / "paired_seed_results.csv", index=False)

    measures = CORE_ROLES + [
        "joint_vs_same_label_budget",
        "joint_vs_total_budget",
    ]
    for role in OPTIONAL_ROLES:
        if role in wide:
            measures.append(role)
    if "joint_vs_full_attribute" in wide:
        measures.append("joint_vs_full_attribute")
    summaries: list[dict[str, float | int | str]] = []
    for (dataset, noise_name, share), condition in wide.groupby(
        ["dataset", "noise_name", "group_share"], sort=True
    ):
        for measure in measures:
            mean, low, high = _mean_ci(condition[measure])
            values = condition[measure].dropna()
            summaries.append(
                {
                    "dataset": dataset,
                    "noise_name": noise_name,
                    "group_share": float(share),
                    "measure": measure,
                    "n_seeds": int(len(values)),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_seeds": int((values > 0.0).sum()),
                    "negative_seeds": int((values < 0.0).sum()),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "condition_summary.csv", index=False)

    sparse = frame[frame["role"] == "joint_sparse_attribute"].copy()
    mechanism = (
        sparse.groupby(["dataset", "noise_name", "group_share"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            attribute_balanced_accuracy=("attribute_balanced_accuracy_offline", "mean"),
            weak_group_identification_rate=("weak_group_identified_offline", "mean"),
            noise_precision=("noise_precision", "mean"),
            corrected_labels=("num_corrected", "mean"),
            weak_group_corrections=("weak_group_corrections", "mean"),
        )
        .sort_values(["dataset", "noise_name", "group_share"])
    )
    mechanism.to_csv(output_dir / "mechanism_summary.csv", index=False)

    report_lines = [
        "# Joint group-audit and label-verification pilot report",
        "",
        "All values below are WGA percentage points. Intervals are paired seed-level "
        "95% t intervals over the condition-specific corruption seeds reported in the CSV.",
        "",
        "## Paired results",
        "",
        "| Dataset | Noise | Group share | Joint dWGA | Joint vs same label budget | Joint vs total label budget |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (dataset, noise_name, share), _ in wide.groupby(
        ["dataset", "noise_name", "group_share"], sort=True
    ):
        condition_summary = summary[
            (summary["dataset"] == dataset)
            & (summary["noise_name"] == noise_name)
            & (summary["group_share"] == share)
        ].set_index("measure")

        def formatted(measure: str) -> str:
            row = condition_summary.loc[measure]
            return (
                f"{_pp(float(row['mean']))} "
                f"[{_pp(float(row['ci95_low']))}, {_pp(float(row['ci95_high']))}]"
            )

        report_lines.append(
            "| {dataset} | {noise} | {share:.0%} | {joint} | {same} | {total} |".format(
                dataset=dataset,
                noise=noise_name,
                share=share,
                joint=formatted("joint_sparse_attribute"),
                same=formatted("joint_vs_same_label_budget"),
                total=formatted("joint_vs_total_budget"),
            )
        )

    selector_rows: list[dict[str, float | int | str]] = []
    for (dataset, noise_name, seed), seed_rows in sparse.groupby(
        ["dataset", "noise_name", "seed"], sort=True
    ):
        base = seed_rows[np.isclose(seed_rows["group_share"], 0.50)]
        if len(base) != 1:
            continue
        selected_share = select_group_audit_share(
            float(base.iloc[0]["mean_expected_group_risk"]),
            threshold=selector_threshold,
        )
        selected = seed_rows[np.isclose(seed_rows["group_share"], selected_share)]
        if len(selected) != 1:
            continue
        total_control = frame[
            (frame["dataset"] == dataset)
            & (frame["noise_name"] == noise_name)
            & (frame["seed"] == seed)
            & np.isclose(frame["group_share"], selected_share)
            & (frame["role"] == "label_only_total_budget")
        ]
        if len(total_control) != 1:
            continue
        selected_delta = float(selected.iloc[0]["delta_wga"])
        total_delta = float(total_control.iloc[0]["delta_wga"])
        selector_rows.append(
            {
                "dataset": dataset,
                "noise_name": noise_name,
                "seed": int(seed),
                "base_mean_expected_group_risk": float(
                    base.iloc[0]["mean_expected_group_risk"]
                ),
                "selected_group_share": selected_share,
                "delta_wga": selected_delta,
                "label_only_total_delta_wga": total_delta,
                "delta_wga_vs_total_budget": selected_delta - total_delta,
            }
        )
    selector = pd.DataFrame(selector_rows)
    selector.to_csv(output_dir / "adaptive_selector_seed_results.csv", index=False)

    selector_summary_rows: list[dict[str, float | int | str]] = []
    if len(selector):
        for (dataset, noise_name), condition in selector.groupby(
            ["dataset", "noise_name"], sort=True
        ):
            for measure in ("delta_wga", "delta_wga_vs_total_budget"):
                mean, low, high = _mean_ci(condition[measure])
                selector_summary_rows.append(
                    {
                        "dataset": dataset,
                        "noise_name": noise_name,
                        "selector_threshold": selector_threshold,
                        "measure": measure,
                        "n_seeds": int(len(condition)),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "expanded_share_seeds": int(
                            np.isclose(condition["selected_group_share"], 0.75).sum()
                        ),
                    }
                )
    selector_summary = pd.DataFrame(selector_summary_rows)
    selector_summary.to_csv(output_dir / "adaptive_selector_summary.csv", index=False)

    report_lines += [
        "",
        f"## Frozen adaptive selector (threshold={selector_threshold:.2f})",
        "",
        "The selector observes mean expected group risk after the 50% audit stage and "
        "expands group auditing to 75% when the signal reaches the frozen threshold.",
        "",
        "| Dataset | Noise | Selected dWGA | Selected vs total label budget | Expanded seeds |",
        "|---|---|---:|---:|---:|",
    ]
    if len(selector_summary):
        for (dataset, noise_name), condition in selector_summary.groupby(
            ["dataset", "noise_name"], sort=True
        ):
            indexed = condition.set_index("measure")

            def selector_formatted(measure: str) -> str:
                row = indexed.loc[measure]
                return (
                    f"{_pp(float(row['mean']))} "
                    f"[{_pp(float(row['ci95_low']))}, {_pp(float(row['ci95_high']))}]"
                )

            expanded = int(indexed.iloc[0]["expanded_share_seeds"])
            n_seeds = int(indexed.iloc[0]["n_seeds"])
            report_lines.append(
                f"| {dataset} | {noise_name} | {selector_formatted('delta_wga')} | "
                f"{selector_formatted('delta_wga_vs_total_budget')} | "
                f"{expanded}/{n_seeds} |"
            )

    absolute_rows = summary[summary["measure"] == "joint_sparse_attribute"]
    primary_rows = summary[summary["measure"] == "joint_vs_same_label_budget"]
    strict_rows = summary[summary["measure"] == "joint_vs_total_budget"]
    absolute_positive = int((absolute_rows["ci95_low"] > 0.0).sum())
    primary_positive = int((primary_rows["ci95_low"] > 0.0).sum())
    strict_positive = int((strict_rows["ci95_low"] > 0.0).sum())
    n_conditions = int(len(absolute_rows))

    report_lines += [
        "",
        "## What the pilot establishes",
        "",
        "- Sparse attribute propagation is feasible: mean offline attribute balanced accuracy is "
        f"{100 * mechanism['attribute_balanced_accuracy'].mean():.1f}%, and the estimated weak "
        f"validation group matches the offline weak group in "
        f"{100 * mechanism['weak_group_identification_rate'].mean():.1f}% of conditions.",
        f"- Positive absolute dWGA has an interval excluding zero in "
        f"{absolute_positive}/{n_conditions} dataset/noise/share conditions.",
        f"- The primary information-value contrast has a positive interval in "
        f"{primary_positive}/{n_conditions} conditions.",
        "",
        "## What the pilot does not establish",
        "",
        f"- The strict total-cost contrast has a positive interval in "
        f"{strict_positive}/{n_conditions} conditions; absent broader held-out evidence, this does "
        "not establish universal cost dominance.",
        *(
            [
                "- The full-attribute comparator is diagnostic only. Any gap diagnoses the fixed "
                "pointwise value model as well as attribute estimation; it is not a global upper bound."
            ]
            if "oracle_attribute" in wide
            else []
        ),
        "- The current evidence uses frozen features, a linear head, synthetic corruption, "
        "equal unit costs for the two audit actions, and two predeclared budget shares.",
        "",
        "## Decision",
        "",
        "The research question passes a feasibility and mechanism screen, but the current "
        "algorithm does not pass the stronger cost-dominance screen. The next confirmatory work "
        "should learn or safely choose the budget split from legal validation/audit signals, "
        "control set concentration, and validate the frozen rule on a held-out dataset/noise family.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="outputs/joint_group_label_budget_pilot/results.csv"
    )
    parser.add_argument(
        "--output-dir", default="outputs/joint_group_label_budget_pilot"
    )
    parser.add_argument("--selector-threshold", type=float, default=0.20)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analyze(
        Path(args.results).resolve(),
        output_dir,
        selector_threshold=args.selector_threshold,
    )


if __name__ == "__main__":
    main()
