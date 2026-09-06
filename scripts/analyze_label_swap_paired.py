#!/usr/bin/env python
"""Per-seed paired analysis of label-swap encoding invariance.

Compares results from the label-swapped 10-seed experiment
(outputs/p0_label_swap_10seed) against the original 10-seed ablation
(outputs/ablation_waterbirds_10seeds), matching on (noise_name, seed,
method, budget_fraction).

Reports:
  - per-noise per-method mean diff, 95% paired CI (seed-level)
  - per-seed swap effect for key methods
  - high-risk class verification
  - class-conditional direction reversal analysis

Usage:  python scripts/analyze_label_swap_paired.py

Output: paper/tables/table_label_swap_paired.csv
         paper/tables/table_class_direction_reversal.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EPSILON = 0.005

METHOD_LABELS = {
    "random": "Random",
    "loss": "Loss",
    "entropy": "Entropy",
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Guarded",
    "oracle_noise_group_balanced": "Oracle-GB",
}


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] {path} not found", file=sys.stderr)
        return None
    return pd.read_csv(path)


def _paired_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lower, upper) from paired t-test CI."""
    n = len(diffs)
    if n < 2:
        return float(np.mean(diffs)), float(np.mean(diffs)), float(np.mean(diffs))
    m = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return m, m - t_crit * se, m + t_crit * se


# ── 1. Paired label-swap analysis ───────────────────────────────────────────

def analyze_label_swap_paired(
    swapped_path: Path,
    original_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Per-seed paired comparison of swapped vs original results."""
    swapped = _load(swapped_path)
    original = _load(original_path)
    if swapped is None or original is None:
        return pd.DataFrame()

    rows = []
    methods = sorted(set(swapped["method"].unique()) & set(original["method"].unique()))
    noises = sorted(set(swapped["noise_name"].unique()) & set(original["noise_name"].unique()))

    for method in methods:
        sw = swapped[swapped["method"] == method]
        og = original[original["method"] == method]

        for noise in noises:
            sw_n = sw[sw["noise_name"] == noise]
            og_n = og[og["noise_name"] == noise]

            merged = sw_n.merge(
                og_n,
                on=["seed", "budget_fraction"],
                suffixes=("_swap", "_orig"),
            )
            if len(merged) < 2:
                continue

            diff = (
                merged["delta_wga_swap"].to_numpy(float)
                - merged["delta_wga_orig"].to_numpy(float)
            )
            mean_d, ci_low, ci_high = _paired_ci(diff)
            n_pairs = len(diff)
            n_seeds = merged["seed"].nunique()

            rows.append({
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "noise_name": noise,
                "mean_swapped": float(merged["delta_wga_swap"].mean()),
                "mean_original": float(merged["delta_wga_orig"].mean()),
                "mean_diff": round(mean_d, 6),
                "ci_low": round(ci_low, 6),
                "ci_high": round(ci_high, 6),
                "n_pairs": n_pairs,
                "n_seeds": n_seeds,
                "p_value": round(float(stats.ttest_rel(
                    merged["delta_wga_swap"].to_numpy(float),
                    merged["delta_wga_orig"].to_numpy(float),
                ).pvalue), 6) if n_pairs >= 2 else 1.0,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")
    return result


# ── 2. Per-seed swap effect detail ──────────────────────────────────────────

def analyze_per_seed_swap(
    swapped_path: Path,
    original_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Per-seed, per-noise, per-budget swap effect for key methods."""
    swapped = _load(swapped_path)
    original = _load(original_path)
    if swapped is None or original is None:
        return pd.DataFrame()

    key_methods = [
        "noise_score",
        "noise_score_budget_hybrid_v8",
        "noise_score_cpba_only",
        "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
    ]

    rows = []
    for method in key_methods:
        sw = swapped[swapped["method"] == method]
        og = original[original["method"] == method]
        noises = sorted(set(sw["noise_name"].unique()) & set(og["noise_name"].unique()))

        for noise in noises:
            sw_n = sw[sw["noise_name"] == noise]
            og_n = og[og["noise_name"] == noise]
            merged = sw_n.merge(og_n, on=["seed", "budget_fraction"], suffixes=("_swap", "_orig"))

            for _, row in merged.iterrows():
                rows.append({
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    "noise_name": noise,
                    "seed": int(row["seed"]),
                    "budget_fraction": float(row["budget_fraction"]),
                    "dwga_swapped": float(row["delta_wga_swap"]),
                    "dwga_original": float(row["delta_wga_orig"]),
                    "dwga_diff": float(row["delta_wga_swap"]) - float(row["delta_wga_orig"]),
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")
    return result


# ── 3. Class-conditional direction reversal analysis ────────────────────────

def analyze_class_direction_reversal(
    label_swap_path: Path,
    ablation_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Analyze whether CPBA adapts to reversed risk direction.

    In standard minority_high_40 (class-1-high), CPBA detects class 1 as high-risk.
    In class0_high_40 (class-0-high), CPBA should detect class 0 as high-risk,
    and reweighting behavior should change accordingly.
    """
    swapped = _load(label_swap_path)
    original = _load(ablation_path)
    if swapped is None:
        return pd.DataFrame()

    key_methods = [
        "noise_score",
        "noise_score_budget_hybrid_v8",
        "noise_score_cpba_only",
        "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
    ]

    rows = []
    for method in key_methods:
        sw = swapped[swapped["method"] == method]

        for noise in ["uniform20", "minority_high_40", "class0_high_40"]:
            sw_n = sw[sw["noise_name"] == noise]
            if sw_n.empty:
                continue

            dwga = sw_n["delta_wga"].to_numpy(float)
            rows.append({
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "noise_name": noise,
                "mean_dwga": round(float(np.mean(dwga)), 6),
                "sem_dwga": round(float(np.std(dwga, ddof=1) / np.sqrt(len(dwga))), 6),
                "neg_seed_pct": round(float((dwga < 0).mean()) * 100, 1),
                "n_seeds": len(dwga),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")

    return result


# ── 4. V9 metadata high-risk class verification ────────────────────────────

def verify_high_risk_class(label_swap_path: Path) -> pd.DataFrame:
    """Verify that CPBA correctly identifies the high-risk class under each noise type."""
    v9_path = label_swap_path.parent / "stage1" / "v9_metadata.csv"
    if not v9_path.exists():
        print(f"[SKIP] v9_metadata.csv not found at {v9_path}", file=sys.stderr)
        return pd.DataFrame()

    meta = pd.read_csv(v9_path)
    rows = []
    for (noise, seed, method), group in meta.groupby(["noise_name", "seed", "method"]):
        hr = group["high_risk_class"].mode()
        high_risk = int(hr.iloc[0]) if not hr.empty else -1
        rows.append({
            "noise_name": noise,
            "seed": seed,
            "method": method,
            "high_risk_class": high_risk,
            "mean_alpha": float(group["alpha"].mean()),
        })

    result = pd.DataFrame(rows)
    out_path = label_swap_path.parent / "stage1" / "high_risk_class_verification.csv"
    if not result.empty:
        result.to_csv(out_path, index=False)
        print(f"[OK] {out_path} ({len(result)} rows)")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = Path("outputs")
    paper_tables = Path("paper/tables")

    paper_tables.mkdir(parents=True, exist_ok=True)

    # 1. Paired label-swap analysis
    swapped_path = base / "p0_label_swap_10seed" / "stage1" / "results.csv"
    original_path = base / "ablation_waterbirds_10seeds" / "stage1" / "results.csv"

    analyze_label_swap_paired(
        swapped_path, original_path,
        paper_tables / "table_label_swap_paired.csv",
    )

    analyze_per_seed_swap(
        swapped_path, original_path,
        paper_tables / "table_label_swap_per_seed.csv",
    )

    # 2. Class-conditional direction reversal
    analyze_class_direction_reversal(
        swapped_path, original_path,
        paper_tables / "table_class_direction_reversal.csv",
    )

    # 3. High-risk class verification
    verify_high_risk_class(base / "p0_label_swap_10seed")


if __name__ == "__main__":
    main()
