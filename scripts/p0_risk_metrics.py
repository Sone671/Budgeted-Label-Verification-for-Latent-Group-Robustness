#!/usr/bin/env python
"""P0-5: Full risk and side-effect metrics.

Computes per-condition risk indicators beyond mean ΔWGA:
  - Negative seed ratio      (P(dWGA < 0))
  - CVaR10                    (average of worst 10% seeds)
  - Worst seed                (min dWGA)
  - Max seed                  (max dWGA)
  - Inter-seed std            (stability)
  - Harmful conditions (%)    (mean dWGA < -epsilon)
  - Absolute WGA change per group
  - Average accuracy change
  - Noise precision
  - Minority query coverage

Usage:  python scripts/p0_risk_metrics.py
Output: outputs/*/stage1/risk_metrics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 0.005  # minimum ΔWGA to consider "harmful" vs neutral


def compute_risk_metrics(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    required = {"method", "noise_name", "seed", "budget_fraction", "delta_wga",
                "wga", "baseline_wga", "average_accuracy", "baseline_average_accuracy",
                "noise_precision", "minority_query_rate"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Missing columns: {missing}", file=sys.stderr)

    rows = []
    group_cols = ["method", "noise_name", "budget_fraction"]
    for gkey, group in df.groupby(group_cols):
        dwga = group["delta_wga"].to_numpy(dtype=float)
        wga = group.get("wga", pd.Series(dtype=float)).to_numpy(dtype=float)
        baseline_wga = group.get("baseline_wga", pd.Series(dtype=float)).to_numpy(dtype=float)
        np_val = group.get("noise_precision", pd.Series(dtype=float)).to_numpy(dtype=float)
        mqr = group.get("minority_query_rate", pd.Series(dtype=float)).to_numpy(dtype=float)
        avg_acc = group.get("average_accuracy", pd.Series(dtype=float)).to_numpy(dtype=float)
        bl_acc = group.get("baseline_average_accuracy", pd.Series(dtype=float)).to_numpy(dtype=float)

        n_seeds = len(dwga)
        neg_ratio = float((dwga < 0).mean())
        harmful_ratio = float((dwga < -EPSILON).mean())

        # CVaR10: mean of worst 10% of seeds (min 1 seed)
        k = max(1, int(np.ceil(n_seeds * 0.10)))
        cvar10 = float(np.mean(np.sort(dwga)[:k])) if n_seeds > 0 else 0.0

        row = {
            "method": gkey[0],
            "noise_name": gkey[1],
            "budget_fraction": gkey[2],
            "n_seeds": n_seeds,
            "mean_dwga": float(np.mean(dwga)),
            "sem_dwga": float(np.std(dwga, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
            "std_dwga": float(np.std(dwga, ddof=1)) if n_seeds > 1 else 0.0,
            "min_dwga": float(np.min(dwga)),
            "max_dwga": float(np.max(dwga)),
            "cvar10": cvar10,
            "neg_seed_pct": neg_ratio,
            "harmful_pct": harmful_ratio,
        }

        # Absolute accuracy metrics if available
        if len(wga) > 0:
            row["mean_wga"] = float(np.mean(wga))
        if len(baseline_wga) > 0:
            row["mean_baseline_wga"] = float(np.mean(baseline_wga))
        if len(np_val) > 0:
            row["mean_noise_precision"] = float(np.mean(np_val))
        if len(mqr) > 0:
            row["mean_minority_query_rate"] = float(np.mean(mqr))
        if len(avg_acc) > 0 and len(bl_acc) > 0:
            delta_acc = avg_acc - bl_acc
            row["mean_delta_accuracy"] = float(np.mean(delta_acc))

        # Absolute per-group accuracy means + worst group accuracy change
        group_cols = [c for c in group.columns if c.endswith("_accuracy") and c.startswith("group_")]
        if group_cols:
            baseline_group_cols = [c for c in group.columns if c.startswith("baseline_group_") and c.endswith("_accuracy")]
            for gc in group_cols:
                vals = group[gc].to_numpy(float)
                row[f"mean_{gc}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            # Compute per-group delta accuracy and identify worst group accuracy change
            group_deltas = []
            for gc in group_cols:
                bl_gc = f"baseline_{gc}"
                if bl_gc in group.columns:
                    delta = group[gc].to_numpy(float) - group[bl_gc].to_numpy(float)
                    mean_delta = float(np.mean(delta)) if len(delta) > 0 else 0.0
                    row[f"delta_{gc}"] = mean_delta
                    group_deltas.append(mean_delta)
            if group_deltas:
                row["worst_group_acc_change"] = float(min(group_deltas))
            # Max group accuracy gap vs mean = concentration risk
            means = [row[f"mean_{gc}"] for gc in group_cols if f"mean_{gc}" in row]
            if means:
                row["group_acc_range"] = float(max(means) - min(means))

        rows.append(row)

    return pd.DataFrame(rows)


def _paired_compare(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    noise_name: str,
    budget_fraction: float,
) -> dict:
    """Per-seed paired comparison of dWGA."""
    sub_a = df[(df["method"] == method_a) & (df["noise_name"] == noise_name) &
               (np.isclose(df["budget_fraction"], budget_fraction))]
    sub_b = df[(df["method"] == method_b) & (df["noise_name"] == noise_name) &
               (np.isclose(df["budget_fraction"], budget_fraction))]

    merged = sub_a.merge(sub_b, on="seed", suffixes=("_a", "_b"))
    if len(merged) < 2:
        return {"comparison": f"{method_a}_vs_{method_b}",
                "noise_name": noise_name, "budget_fraction": budget_fraction,
                "n_pairs": len(merged), "mean_diff": 0.0, "wins_a": 0, "wins_b": 0, "ties": 0}

    diff = merged["delta_wga_a"].to_numpy() - merged["delta_wga_b"].to_numpy()
    return {
        "comparison": f"{method_a}_vs_{method_b}",
        "noise_name": noise_name,
        "budget_fraction": budget_fraction,
        "n_pairs": len(diff),
        "mean_diff": float(np.mean(diff)),
        "wins_a": int((diff > EPSILON).sum()),
        "wins_b": int((diff < -EPSILON).sum()),
        "ties": int((np.abs(diff) <= EPSILON).sum()),
    }


def paired_risk_audit(results_path: Path, method_a: str, method_b: str) -> pd.DataFrame:
    """Paired risk audit: for each (noise, budget), compare A vs B."""
    df = pd.read_csv(results_path)
    rows = []
    for noise in df["noise_name"].unique():
        for budget in sorted(df["budget_fraction"].unique()):
            rows.append(_paired_compare(df, method_a, method_b, str(noise), float(budget)))
    return pd.DataFrame(rows)


def main():
    output_dirs = [
        "outputs/ablation_waterbirds_10seeds",
        "outputs/ablation_celeba_10seeds",
        "outputs/ablation_civilcomments_10seeds",
        "outputs/baselines_waterbirds_recommended",
        "outputs/p0_label_swap_10seed",
        "outputs/p0_class_direction_reversal",
        "outputs/p0_unified_baselines_30epoch_waterbirds",
    ]

    for od in output_dirs:
        rp = Path(od) / "stage1" / "results.csv"
        if not rp.exists():
            print(f"[SKIP] {rp} not found")
            continue

        # Infer dataset from path name
        ds = "Waterbirds"
        if "celeba" in str(od).lower():
            ds = "CelebA"
        elif "civilcomments" in str(od).lower():
            ds = "CivilComments"

        risk = compute_risk_metrics(rp)
        risk.insert(0, "dataset", ds)
        out_path = Path(od) / "stage1" / "risk_metrics.csv"
        risk.to_csv(out_path, index=False)
        print(f"[OK] {out_path}  ({len(risk)} rows)")

        # Paired audit: Guarded vs NoiseScore
        guard_name = "noise_score_cpba_tc_v8_fallback"
        if guard_name in risk["method"].unique():
            paired = paired_risk_audit(rp, guard_name, "noise_score")
            paired_path = Path(od) / "stage1" / "paired_audit_guarded_vs_noisescore.csv"
            paired.to_csv(paired_path, index=False)
            print(f"[OK] {paired_path}  ({len(paired)} rows)")

        # Paired audit: noFB vs Guarded
        nofb_name = "noise_score_cpba_tc_no_fallback"
        if nofb_name in risk["method"].unique() and guard_name in risk["method"].unique():
            paired2 = paired_risk_audit(rp, nofb_name, guard_name)
            paired2_path = Path(od) / "stage1" / "paired_audit_nofb_vs_guarded.csv"
            paired2.to_csv(paired2_path, index=False)
            print(f"[OK] {paired2_path}  ({len(paired2)} rows)")


if __name__ == "__main__":
    main()
