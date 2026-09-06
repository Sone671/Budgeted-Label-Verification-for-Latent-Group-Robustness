#!/usr/bin/env python
"""Analyze the frozen Phase-4 staged identifiability selector."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.joint_budget import (
    relative_weak_group_error_margin,
    select_staged_group_audit_share,
)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"cannot parse boolean value: {value!r}")
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


def analyze(
    result_paths: list[Path],
    output_dir: Path,
    *,
    scout_share: float,
    expanded_share: float,
    relative_margin_threshold: float,
) -> None:
    frames = [pd.read_csv(path) for path in result_paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "dataset", "noise_name", "seed", "group_share", "role", "delta_wga",
        "attribute_model_fitted", "human_action_count", "total_budget_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if (frame["human_action_count"].astype(int) != frame["total_budget_count"].astype(int)).any():
        raise ValueError("at least one result row violates the total human-action budget")

    group_columns = [f"estimated_group_{group}_error" for group in range(4)]
    missing_groups = set(group_columns) - set(frame.columns)
    if missing_groups:
        raise ValueError(f"missing group-error columns: {sorted(missing_groups)}")

    rows: list[dict[str, float | int | str | bool]] = []
    keys = ["dataset", "noise_name", "seed"]
    sparse = frame[frame["role"].eq("joint_sparse_attribute")]
    for (dataset, noise, seed), condition in sparse.groupby(keys, sort=True):
        scout = condition[np.isclose(condition["group_share"], scout_share)]
        if len(scout) != 1:
            raise ValueError(f"expected one scout row for {(dataset, noise, seed)}")
        scout_row = scout.iloc[0]
        group_error = scout_row[group_columns].to_numpy(dtype=np.float64)
        selected_share = select_staged_group_audit_share(
            group_error,
            attribute_model_fitted=_as_bool(scout_row["attribute_model_fitted"]),
            relative_margin_threshold=relative_margin_threshold,
            scout_share=scout_share,
            expanded_share=expanded_share,
        )
        selected = condition[np.isclose(condition["group_share"], selected_share)]
        if len(selected) != 1:
            raise ValueError(f"expected one selected row for {(dataset, noise, seed)}")
        selected_row = selected.iloc[0]
        comparator = frame[
            frame["dataset"].eq(dataset)
            & frame["noise_name"].eq(noise)
            & frame["seed"].astype(int).eq(int(seed))
            & np.isclose(frame["group_share"], selected_share)
            & frame["role"].eq("label_only_total_budget")
        ]
        same = frame[
            frame["dataset"].eq(dataset)
            & frame["noise_name"].eq(noise)
            & frame["seed"].astype(int).eq(int(seed))
            & np.isclose(frame["group_share"], selected_share)
            & frame["role"].eq("label_only_same_label_budget")
        ]
        if len(comparator) != 1 or len(same) != 1:
            raise ValueError(f"missing comparator for {(dataset, noise, seed)}")
        total_delta = float(comparator.iloc[0]["delta_wga"])
        same_delta = float(same.iloc[0]["delta_wga"])
        selected_delta = float(selected_row["delta_wga"])
        rows.append({
            "dataset": str(dataset),
            "noise_name": str(noise),
            "seed": int(seed),
            "scout_relative_margin": relative_weak_group_error_margin(group_error),
            "scout_attribute_model_fitted": _as_bool(
                scout_row["attribute_model_fitted"]
            ),
            "selected_group_share": selected_share,
            "expanded": bool(np.isclose(selected_share, expanded_share)),
            "delta_wga": selected_delta,
            "delta_wga_vs_total_budget": selected_delta - total_delta,
            "delta_wga_vs_same_label_budget": selected_delta - same_delta,
            "group_audit_count": int(selected_row["group_audit_count"]),
            "label_query_count": int(selected_row["label_query_count"]),
            "attribute_balanced_accuracy_offline": float(
                selected_row.get("attribute_balanced_accuracy_offline", math.nan)
            ),
            "weak_group_identified_offline": int(
                selected_row.get("weak_group_identified_offline", 0)
            ),
        })

    selected = pd.DataFrame(rows).sort_values(keys, kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selector_seed_results.csv", index=False)

    summary_rows: list[dict[str, float | int | str]] = []
    report = [
        "# Phase 4 staged selector development report", "",
        "All WGA values are percentage points with paired seed-level 95% t intervals.", "",
        "| Dataset | Noise | Selected dWGA | vs same label count | vs total-label budget | Expanded |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (dataset, noise), condition in selected.groupby(["dataset", "noise_name"], sort=True):
        for measure in ("delta_wga", "delta_wga_vs_same_label_budget", "delta_wga_vs_total_budget"):
            mean, low, high = _interval(condition[measure])
            summary_rows.append({
                "dataset": dataset, "noise_name": noise, "measure": measure,
                "n_seeds": len(condition), "mean": mean, "ci95_low": low,
                "ci95_high": high,
                "expanded_seeds": int(condition["expanded"].sum()),
            })
        report.append(
            f"| {dataset} | {noise} | {_formatted(condition['delta_wga'])} | "
            f"{_formatted(condition['delta_wga_vs_same_label_budget'])} | "
            f"{_formatted(condition['delta_wga_vs_total_budget'])} | "
            f"{int(condition['expanded'].sum())}/{len(condition)} |"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "selector_summary.csv", index=False)
    report.extend([
        "", "## Legal scout diagnostics", "",
        f"- Scout share: {scout_share:.2f}; expanded share: {expanded_share:.2f}.",
        f"- Frozen relative weak-group margin threshold: {relative_margin_threshold:.2f}.",
        f"- Two-class attribute model fitted in {int(selected['scout_attribute_model_fitted'].sum())}/{len(selected)} conditions.",
        f"- Selector expanded in {int(selected['expanded'].sum())}/{len(selected)} conditions.",
        "- Offline attribute accuracy and weak-group identity are reported only for mechanism audit; they do not enter selection.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scout-share", type=float, default=0.25)
    parser.add_argument("--expanded-share", type=float, default=0.50)
    parser.add_argument("--relative-margin-threshold", type=float, default=0.10)
    args = parser.parse_args()
    analyze(
        [Path(path).resolve() for path in args.results],
        Path(args.output_dir).resolve(),
        scout_share=args.scout_share,
        expanded_share=args.expanded_share,
        relative_margin_threshold=args.relative_margin_threshold,
    )


if __name__ == "__main__":
    main()
