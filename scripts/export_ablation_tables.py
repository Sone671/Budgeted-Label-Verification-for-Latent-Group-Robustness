#!/usr/bin/env python
"""Extract full ablation tables: one CSV per dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Load ────────────────────────────────────────────────────────────────────
datasets = {
    "waterbirds": pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
    "celeba": pd.read_csv("outputs/ablation_celeba_10seeds/stage1/results.csv"),
    "civilcomments": pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
}

METHODS = [
    "random", "loss", "entropy",
    "noise_score",
    "noise_score_budget_hybrid",
    "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc",
    "noise_score_cpba_only",
    "noise_score_tc_only",
    "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback",
    "oracle_noise_group_balanced",
]

LABELS = {
    "random": "Random",
    "loss": "Loss",
    "entropy": "Entropy",
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc": "CPBA-TC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_tc_only": "TC-only",
    "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-v8FB",
    "oracle_noise_group_balanced": "Oracle-GB",
}

NOISES = ["uniform20", "minority_high_40"]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


for ds_name, df in datasets.items():
    rows = []
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
                wga_vals = sub["wga"].to_numpy()
                rows.append({
                    "noise": noise,
                    "budget": f"{budget:.1%}",
                    "budget_fraction": budget,
                    "method": method,
                    "method_label": LABELS.get(method, method),
                    "n_seeds": len(vals),
                    "dWGA_mean": round(float(vals.mean()), 4),
                    "dWGA_sem": round(sem(vals), 4),
                    "dWGA_std": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0,
                    "dWGA_min": round(float(vals.min()), 4),
                    "dWGA_max": round(float(vals.max()), 4),
                    "dWGA_neg_pct": round(float((vals < 0).mean()) * 100, 1),
                    "WGA_mean": round(float(wga_vals.mean()), 4),
                    "noise_precision": round(float(sub["noise_precision"].mean()), 4),
                    "minority_query_rate": round(float(sub["minority_query_rate"].mean()), 4),
                })

    out = pd.DataFrame(rows)
    out_path = f"outputs/ablation_{ds_name}_table.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(out)} rows)")

print("\nDone.")
