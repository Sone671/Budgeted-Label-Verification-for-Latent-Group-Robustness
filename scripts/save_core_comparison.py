#!/usr/bin/env python
"""Save core 10-seed comparison data to a single summary CSV.

Consolidates: per-condition means, entropy-vs-v8 paired tests, win rates,
v-line internal comparison, and overall verdict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("outputs/entropy_forensic_10seeds/stage1/results.csv")
OUT_DIR = Path("outputs/entropy_forensic_10seeds/stage1/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1, V7, V8 = "noise_score", "noise_score_budget_hybrid", "noise_score_budget_hybrid_v8"
ENT, ORACLE = "entropy", "oracle_noise_group_balanced"
ALL_METHODS = [V1, V7, V8, ENT, ORACLE]

LABELS = {V1: "NoiseScore", V7: "BHC", V8: "NGC", ENT: "Entropy", ORACLE: "Oracle-GB"}


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    df = pd.read_csv(RESULTS)
    noises = sorted(df["noise_name"].unique())
    budgets = sorted(df["budget_fraction"].unique())

    rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            means = {m: float(sub[sub["method"] == m]["delta_wga"].mean()) for m in ALL_METHODS}
            sems = {m: sem(sub[sub["method"] == m]["delta_wga"].to_numpy()) for m in ALL_METHODS}
            neg = {m: float((sub[sub["method"] == m]["delta_wga"] < 0).mean()) for m in ALL_METHODS}
            cvs = {
                m: float(sub[sub["method"] == m]["delta_wga"].std(ddof=1) / abs(means[m]))
                if abs(means[m]) > 1e-8 else float("inf")
                for m in ALL_METHODS
            }
            ent_v = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            v8_v = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            v1_v = sub[sub["method"] == V1]["delta_wga"].to_numpy()
            v7_v = sub[sub["method"] == V7]["delta_wga"].to_numpy()
            diff_ev8 = ent_v - v8_v
            t_ev8, p_ev8 = stats.ttest_rel(ent_v, v8_v)
            try:
                _, p_w_ev8 = stats.wilcoxon(ent_v, v8_v)
            except ValueError:
                p_w_ev8 = float("nan")
            wins_ev8 = int((diff_ev8 > 0.0005).sum())
            losses_ev8 = int((diff_ev8 < -0.0005).sum())
            ties_ev8 = len(diff_ev8) - wins_ev8 - losses_ev8

            t_v8v1, p_v8v1 = stats.ttest_rel(v8_v, v1_v)
            t_v7v1, p_v7v1 = stats.ttest_rel(v7_v, v1_v)

            n_corr = {m: float(sub[sub["method"] == m]["num_corrected"].mean()) for m in ALL_METHODS}
            nprec = {m: float(sub[sub["method"] == m]["noise_precision"].mean()) for m in ALL_METHODS}
            minq = {m: float(sub[sub["method"] == m]["minority_query_rate"].mean()) for m in ALL_METHODS}

            rows.append({
                "noise_name": noise,
                "budget_fraction": budget,
                "n_seeds": len(ent_v),
                "v1_mean": means[V1], "v1_sem": sems[V1], "v1_neg%": neg[V1] * 100, "v1_cv": cvs[V1],
                "v7_mean": means[V7], "v7_sem": sems[V7], "v7_neg%": neg[V7] * 100, "v7_cv": cvs[V7],
                "v8_mean": means[V8], "v8_sem": sems[V8], "v8_neg%": neg[V8] * 100, "v8_cv": cvs[V8],
                "entropy_mean": means[ENT], "entropy_sem": sems[ENT], "entropy_neg%": neg[ENT] * 100,
                "entropy_cv": cvs[ENT],
                "oracle_gb_mean": means[ORACLE], "oracle_gb_sem": sems[ORACLE],
                "entropy_vs_v8_diff": float(diff_ev8.mean()),
                "entropy_vs_v8_t": float(t_ev8), "entropy_vs_v8_p": float(p_ev8),
                "entropy_vs_v8_wilcoxon_p": float(p_w_ev8),
                "entropy_wins": wins_ev8, "entropy_ties": ties_ev8, "entropy_losses": losses_ev8,
                "v8_vs_v1_diff": means[V8] - means[V1], "v8_vs_v1_p": float(p_v8v1),
                "v7_vs_v1_diff": means[V7] - means[V1], "v7_vs_v1_p": float(p_v7v1),
                "gap_v8_to_oracle": means[ORACLE] - means[V8],
                "gap_entropy_to_oracle": means[ORACLE] - means[ENT],
                "v1_num_corrected": n_corr[V1], "v8_num_corrected": n_corr[V8],
                "entropy_num_corrected": n_corr[ENT], "oracle_num_corrected": n_corr[ORACLE],
                "v1_noise_precision": nprec[V1], "v8_noise_precision": nprec[V8],
                "entropy_noise_precision": nprec[ENT],
                "v1_minority_q_rate": minq[V1], "v8_minority_q_rate": minq[V8],
                "entropy_minority_q_rate": minq[ENT],
            })

    summary = pd.DataFrame(rows)
    out = OUT_DIR / "core_comparison_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved: {out}  ({len(summary)} rows, {len(summary.columns)} cols)")
    print(summary[["noise_name", "budget_fraction", "v8_mean", "entropy_mean",
                    "entropy_vs_v8_diff", "entropy_vs_v8_p", "entropy_wins",
                    "entropy_losses", "v8_vs_v1_diff", "v8_vs_v1_p",
                    "gap_v8_to_oracle"]].to_string(index=False))

    # Overall win-rate row
    all_ent = df[df["method"] == ENT]["delta_wga"].to_numpy()
    all_v8 = df[df["method"] == V8]["delta_wga"].to_numpy()
    all_diff = all_ent - all_v8
    overall = pd.DataFrame([{
        "noise_name": "OVERALL",
        "budget_fraction": float("nan"),
        "n_seeds": len(all_diff),
        "v8_mean": float(all_v8.mean()),
        "entropy_mean": float(all_ent.mean()),
        "entropy_vs_v8_diff": float(all_diff.mean()),
        "entropy_vs_v8_p": float(stats.ttest_rel(all_ent, all_v8)[1]),
        "entropy_wins": int((all_diff > 0.0005).sum()),
        "entropy_losses": int((all_diff < -0.0005).sum()),
    }])
    overall_out = OUT_DIR / "overall_winrate.csv"
    overall.to_csv(overall_out, index=False)
    print(f"\nSaved: {overall_out}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
