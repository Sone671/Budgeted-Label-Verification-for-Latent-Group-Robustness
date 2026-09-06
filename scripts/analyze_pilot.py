#!/usr/bin/env python
"""Comprehensive baseline pilot analysis on Waterbirds.

Writes CSV reports to outputs/baseline_test/stage1/analysis/ and prints the
same tables to stdout. Reports per (noise, budget, method):
  - delta_WGA mean ± SEM
  - negative-run fraction (delta_WGA < 0 across seeds)
  - noise precision
  - minority query rate
  - group query entropy (coverage proxy)
  - regret to best of {v1, v7, v8}
  - gap to oracle_noise_group_balanced

Then assesses whether the v1 -> v8 main line holds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("outputs/baseline_test/stage1/results.csv")
OUT_DIR = Path("outputs/baseline_test/stage1/analysis")

V_LINE = ["noise_score", "noise_score_budget_hybrid", "noise_score_budget_hybrid_v8"]
BASELINES = [
    "random", "loss", "entropy", "margin",
    "aum", "confident_learning",
    "coreset", "uncertainty_diversity", "badge_lite",
]
ORACLE = "oracle_noise_group_balanced"
ORDER = V_LINE + BASELINES + [ORACLE]

LABELS = {
    "noise_score":                       "NoiseScore",
    "noise_score_budget_hybrid":         "BHC",
    "noise_score_budget_hybrid_v8":      "NGC",
    "confident_learning":                "Cleanlab",
    "oracle_noise_group_balanced":       "Oracle-GB",
}


def lbl(m: str) -> str:
    return LABELS.get(m, m)


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main() -> None:
    df = pd.read_csv(RESULTS)
    noises = sorted(df["noise_name"].unique())
    budgets = sorted(df["budget_fraction"].unique())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Per-condition table (numeric) ----
    per_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            for method in ORDER:
                m = sub[sub["method"] == method]
                if m.empty:
                    continue
                dw = m["delta_wga"].to_numpy()
                per_rows.append({
                    "noise_name": noise,
                    "budget_fraction": budget,
                    "method_raw": method,
                    "method": lbl(method),
                    "delta_wga_mean": float(dw.mean()),
                    "delta_wga_sem": sem(dw),
                    "neg_run_fraction": float((dw < 0).mean()),
                    "noise_precision": float(m["noise_precision"].mean()),
                    "minority_query_rate": float(m["minority_query_rate"].mean()),
                    "group_query_entropy": float(m["group_query_entropy"].mean()),
                    "num_corrected": float(m["num_corrected"].mean()),
                    "wga": float(m["wga"].mean()),
                })
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(OUT_DIR / "per_condition.csv", index=False)

    # ---- Print per-condition ----
    for noise in noises:
        print("=" * 120)
        print(f"NOISE: {noise}")
        print("=" * 120)
        for budget in budgets:
            sub = per_df[(per_df["noise_name"] == noise) & (per_df["budget_fraction"] == budget)]
            rows = []
            for _, r in sub.iterrows():
                rows.append({
                    "method": r["method"],
                    "dWGA": f"{r['delta_wga_mean']:+.4f}±{r['delta_wga_sem']:.4f}",
                    "neg%": f"{r['neg_run_fraction'] * 100:.0f}%",
                    "noise_prec": f"{r['noise_precision']:.3f}",
                    "min_q": f"{r['minority_query_rate']:.3f}",
                    "grp_entropy": f"{r['group_query_entropy']:.3f}",
                    "n_corr": f"{r['num_corrected']:.0f}",
                })
            print(f"\n--- budget = {budget:.0%} ---")
            print(pd.DataFrame(rows).to_string(index=False))

    # ---- Regret to best of {v1, v7, v8} ----
    regret_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            v_vals = {v: sub[sub["method"] == v]["delta_wga"].mean() for v in V_LINE}
            v_best = max(v_vals.values())
            v_best_name = max(v_vals, key=v_vals.get)
            for method in BASELINES + [ORACLE]:
                m = sub[sub["method"] == method]
                if m.empty:
                    continue
                mw = m["delta_wga"].mean()
                regret_rows.append({
                    "noise_name": noise,
                    "budget_fraction": budget,
                    "method_raw": method,
                    "method": lbl(method),
                    "delta_wga": float(mw),
                    "best_v_line": lbl(v_best_name),
                    "best_v_line_dWGA": float(v_best),
                    "regret": float(v_best - mw),
                })
    regret_df = pd.DataFrame(regret_rows)
    regret_df.to_csv(OUT_DIR / "regret_to_vline.csv", index=False)

    print("\n" + "=" * 120)
    print("REGRET TO BEST OF {v1, v7, v8}  (positive = method worse than best v-line)")
    print("=" * 120)
    for noise in noises:
        for budget in budgets:
            sub = regret_df[(regret_df["noise_name"] == noise) & (regret_df["budget_fraction"] == budget)]
            best_v = sub["best_v_line"].iloc[0]
            best_v_dw = sub["best_v_line_dWGA"].iloc[0]
            print(f"\n{noise} @ {budget:.0%}: best v-line = {best_v} (dWGA={best_v_dw:+.4f})")
            for _, r in sub.iterrows():
                sign = "+" if r["regret"] > 0 else ""
                print(f"  {r['method']:24s} dWGA={r['delta_wga']:+.4f}  regret={sign}{r['regret']:.4f}")

    # ---- Gap to oracle_noise_group_balanced ----
    gap_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            oracle_dw = sub[sub["method"] == ORACLE]["delta_wga"].mean()
            for method in V_LINE + BASELINES:
                m = sub[sub["method"] == method]
                if m.empty:
                    continue
                mw = m["delta_wga"].mean()
                gap_rows.append({
                    "noise_name": noise,
                    "budget_fraction": budget,
                    "method_raw": method,
                    "method": lbl(method),
                    "delta_wga": float(mw),
                    "oracle_gb_dWGA": float(oracle_dw),
                    "gap_to_oracle": float(oracle_dw - mw),
                })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(OUT_DIR / "gap_to_oracle.csv", index=False)

    print("\n" + "=" * 120)
    print("GAP TO oracle_noise_group_balanced  (positive = room for improvement)")
    print("=" * 120)
    for noise in noises:
        for budget in budgets:
            sub = gap_df[(gap_df["noise_name"] == noise) & (gap_df["budget_fraction"] == budget)]
            oracle_dw = sub["oracle_gb_dWGA"].iloc[0]
            print(f"\n{noise} @ {budget:.0%}: oracle_gb dWGA = {oracle_dw:+.4f}")
            for _, r in sub.iterrows():
                print(f"  {r['method']:24s} dWGA={r['delta_wga']:+.4f}  gap={r['gap_to_oracle']:+.4f}")

    # ---- v1 -> v8 main line validity ----
    v1, v7, v8 = "noise_score", "noise_score_budget_hybrid", "noise_score_budget_hybrid_v8"
    vline_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            d = {v: float(sub[sub["method"] == v]["delta_wga"].mean()) for v in [v1, v7, v8]}
            mx = max(d.values())
            winners = [lbl(k) for k, val in d.items() if abs(val - mx) < 0.0005]
            vline_rows.append({
                "noise_name": noise,
                "budget_fraction": budget,
                "v1_dWGA": d[v1],
                "v7_dWGA": d[v7],
                "v8_dWGA": d[v8],
                "best_v_line": "+".join(winners),
                "v7_minus_v1": d[v7] - d[v1],
                "v8_minus_v1": d[v8] - d[v1],
            })
    vline_df = pd.DataFrame(vline_rows)
    vline_df.to_csv(OUT_DIR / "vline_validity.csv", index=False)

    print("\n" + "=" * 120)
    print("v1 -> v8 MAIN LINE VALIDITY")
    print("=" * 120)
    for _, r in vline_df.iterrows():
        print(f"  {r['noise_name']}@{r['budget_fraction']:.0%}".ljust(28)
              + f"  v1={r['v1_dWGA']:+.4f}  v7={r['v7_dWGA']:+.4f}  v8={r['v8_dWGA']:+.4f}"
              + f"  best={r['best_v_line']}")

    # ---- Overall ranking summary ----
    legal = [m for m in ORDER if m != ORACLE]
    best_per_cond = {}
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget) & (df["method"].isin(legal))]
            best_per_cond[(noise, budget)] = float(sub["delta_wga"].mean())
    total = len(noises) * len(budgets)
    summary_rows = []
    for method in legal:
        m = df[df["method"] == method]
        dw = m["delta_wga"].to_numpy()
        best_count = 0
        for noise in noises:
            for budget in budgets:
                mw = m[(m["noise_name"] == noise) & (m["budget_fraction"] == budget)]["delta_wga"].mean()
                if abs(mw - best_per_cond[(noise, budget)]) < 0.0005:
                    best_count += 1
        summary_rows.append({
            "method_raw": method,
            "method": lbl(method),
            "mean_dWGA": float(dw.mean()),
            "sem": sem(dw),
            "neg_run_fraction": float((dw < 0).mean()),
            "best_count": best_count,
            "best_count_total": total,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_dWGA", ascending=False)
    summary_df.to_csv(OUT_DIR / "overall_ranking.csv", index=False)

    print("\n" + "=" * 120)
    print("OVERALL LEGAL METHOD RANKING by mean dWGA across all conditions")
    print("=" * 120)
    for _, r in summary_df.iterrows():
        print(f"  {r['method']:24s} mean_dWGA={r['mean_dWGA']:+.4f}  sem={r['sem']:.4f}"
              f"  neg%={r['neg_run_fraction'] * 100:.0f}%  best={r['best_count']}/{r['best_count_total']}")

    print(f"\nCSV reports written to: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"  {p.name}")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
