#!/usr/bin/env python
"""Export CivilComments real-noise disagreement experiment tables."""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("outputs/civilcomments_disagreement_10seeds/analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("outputs/civilcomments_disagreement_10seeds/stage1/results.csv")
ns = pd.read_csv("outputs/civilcomments_disagreement_10seeds/noise_statistics.csv")

LABELS = {
    "random": "Random", "loss": "Loss",
    "noise_score": "NoiseScore", "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "oracle_noise_group_balanced": "Oracle-GB",
}

METHODS = list(LABELS.keys())
NOISES = ["uniform20", "disagreement_natural", "disagreement_matched20"]
BUDGETS = [0.01, 0.05, 0.10]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Full per-condition results
# ═══════════════════════════════════════════════════════════════════════════════
rows_full = []
for noise in NOISES:
    for budget in BUDGETS:
        for method in METHODS:
            sub = df[
                (df["noise_name"] == noise)
                & (df["budget_fraction"] == budget)
                & (df["method"] == method)
            ]
            if sub.empty:
                continue
            vals = sub["delta_wga"].to_numpy()
            rows_full.append({
                "noise": noise,
                "budget": f"{budget:.1%}",
                "budget_fraction": budget,
                "method": method,
                "method_label": LABELS.get(method, method),
                "n_seeds": len(vals),
                "dWGA_mean": round(float(vals.mean()), 4),
                "dWGA_sem": round(sem(vals), 4),
                "dWGA_min": round(float(vals.min()), 4),
                "dWGA_max": round(float(vals.max()), 4),
                "dWGA_neg_pct": round(float((vals < 0).mean()) * 100, 1),
                "noise_precision": round(float(sub["noise_precision"].mean()), 4),
                "minority_query_rate": round(float(sub["minority_query_rate"].mean()), 4),
                "disagreement_lift": round(float(sub["disagreement_lift"].mean()), 4) if "disagreement_lift" in sub.columns else 0.0,
                "queried_disagreement_mean": round(float(sub["queried_disagreement_mean"].mean()), 4) if "queried_disagreement_mean" in sub.columns else 0.0,
                "queried_high_disagreement_rate": round(float(sub["queried_high_disagreement_rate"].mean()), 4) if "queried_high_disagreement_rate" in sub.columns else 0.0,
                "wga_mean": round(float(sub["wga"].mean()), 4),
            })

df_full = pd.DataFrame(rows_full)
df_full.to_csv(OUT / "disagreement_full.csv", index=False)
print(f"TABLE 1: disagreement_full.csv  ({len(df_full)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Noise statistics
# ═══════════════════════════════════════════════════════════════════════════════
rows_noise = []
for noise in NOISES:
    sub = ns[ns["noise_name"] == noise]
    if sub.empty:
        continue
    rows_noise.append({
        "noise": noise,
        "n_seeds": len(sub),
        "overall_rate": round(float(sub["actual_noise_rate"].mean()), 4),
        "majority_rate": round(float(sub["majority_noise_rate"].mean()), 4),
        "minority_rate": round(float(sub["minority_noise_rate"].mean()), 4),
        "minority_majority_ratio": round(float(sub["minority_noise_rate"].mean() / max(sub["majority_noise_rate"].mean(), 1e-8)), 2),
        "num_noisy_mean": round(float(sub["num_noisy"].mean()), 0),
    })

df_noise = pd.DataFrame(rows_noise)
df_noise.to_csv(OUT / "noise_statistics_summary.csv", index=False)
print(f"TABLE 2: noise_statistics_summary.csv  ({len(df_noise)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3: Budget-pooled summary (per noise, avg over budgets)
# ═══════════════════════════════════════════════════════════════════════════════
rows_summary = []
for noise in NOISES:
    for method in METHODS:
        all_vals = []
        all_prec = []
        all_lift = []
        for budget in BUDGETS:
            sub = df[
                (df["noise_name"] == noise)
                & (df["budget_fraction"] == budget)
                & (df["method"] == method)
            ]
            if not sub.empty:
                all_vals.extend(sub["delta_wga"].to_numpy().tolist())
                all_prec.append(sub["noise_precision"].mean())
                all_lift.append(sub["disagreement_lift"].mean() if "disagreement_lift" in sub.columns else 0.0)
        if len(all_vals) == 0:
            continue
        all_vals = np.array(all_vals)
        rows_summary.append({
            "noise": noise,
            "method": method,
            "method_label": LABELS.get(method, method),
            "n_total": len(all_vals),
            "dWGA_mean": round(float(all_vals.mean()), 4),
            "dWGA_sem": round(sem(all_vals), 4),
            "dWGA_neg_pct": round(float((all_vals < 0).mean()) * 100, 1),
            "noise_precision": round(float(np.mean(all_prec)), 4),
            "disagreement_lift": round(float(np.mean(all_lift)), 4),
        })

df_summary = pd.DataFrame(rows_summary)
df_summary.to_csv(OUT / "disagreement_summary.csv", index=False)
print(f"TABLE 3: disagreement_summary.csv  ({len(df_summary)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4: Disagreement lift analysis (b=5%)
# ═══════════════════════════════════════════════════════════════════════════════
rows_lift = []
for noise in NOISES:
    for method in METHODS:
        sub = df[
            (df["noise_name"] == noise)
            & (df["budget_fraction"] == 0.05)
            & (df["method"] == method)
        ]
        if sub.empty:
            continue
        rows_lift.append({
            "noise": noise,
            "method": method,
            "method_label": LABELS.get(method, method),
            "dWGA_mean": round(float(sub["delta_wga"].mean()), 4),
            "noise_precision": round(float(sub["noise_precision"].mean()), 4),
            "disagreement_lift": round(float(sub["disagreement_lift"].mean()), 4),
            "queried_disagreement_mean": round(float(sub["queried_disagreement_mean"].mean()), 4),
            "queried_high_disagreement_rate": round(float(sub["queried_high_disagreement_rate"].mean()), 4),
            "dataset_disagreement_mean": round(float(sub["dataset_disagreement_mean"].mean()), 4),
        })

df_lift = pd.DataFrame(rows_lift)
df_lift.to_csv(OUT / "disagreement_lift_b5.csv", index=False)
print(f"TABLE 4: disagreement_lift_b5.csv  ({len(df_lift)} rows)")

print(f"\nAll CSVs saved to: {OUT}/")
