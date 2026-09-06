#!/usr/bin/env python
"""Quick ablation summary: Waterbirds uniform20 high-budget."""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULTS = pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv")

FOCUS = RESULTS[
    (RESULTS["noise_name"] == "uniform20")
    & (RESULTS["budget_fraction"].isin([0.05, 0.10]))
]

METHODS = [
    "noise_score",
    "entropy",
    "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc",
    "noise_score_cpba_only",
    "noise_score_tc_only",
    "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback",
    "noise_score_cpba_tc_entropy_fallback",
    "oracle_noise_group_balanced",
    "oracle_minority",
]

LABELS = {
    "noise_score": "NoiseScore",
    "entropy": "Entropy",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc": "CPBA-TC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_tc_only": "TC-only",
    "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-v8FB",
    "noise_score_cpba_tc_entropy_fallback": "CPBA-TC-entFB",
    "oracle_noise_group_balanced": "Oracle-GB",
    "oracle_minority": "Oracle-min",
}


def sem(x):
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


for budget in [0.05, 0.10]:
    print(f"\n{'='*90}")
    print(f"Waterbirds / uniform20 / budget={budget:.0%}")
    print(f"{'='*90}")
    print(f"{'Method':<20s} {'dWGA_mean':>9s} {'±sem':>9s} {'neg%':>5s} {'min':>9s} {'max':>9s}")
    print("-" * 70)
    for m in METHODS:
        sub = FOCUS[
            (FOCUS["budget_fraction"] == budget) & (FOCUS["method"] == m)
        ]
        if sub.empty:
            continue
        vals = sub["delta_wga"].to_numpy()
        lbl = LABELS.get(m, m)
        print(
            f"{lbl:<20s} {vals.mean():>+9.4f} {sem(vals):>9.4f} "
            f"{(vals < 0).mean() * 100:>4.0f}% {vals.min():>+9.4f} {vals.max():>+9.4f}"
        )

# Also show minority_high_40 summary
print(f"\n{'='*90}")
print("Waterbirds / minority_high_40 / ALL budgets (mean dWGA)")
print(f"{'='*90}")
print(f"{'Method':<20s} {'b=0.5%':>8s} {'b=1%':>8s} {'b=2%':>8s} {'b=5%':>8s} {'b=10%':>8s}")
print("-" * 70)
for m in METHODS:
    lbl = LABELS.get(m, m)
    row = [lbl]
    for b in [0.005, 0.01, 0.02, 0.05, 0.10]:
        sub = RESULTS[
            (RESULTS["noise_name"] == "minority_high_40")
            & (RESULTS["budget_fraction"] == b)
            & (RESULTS["method"] == m)
        ]
        if sub.empty:
            row.append("  N/A")
        else:
            row.append(f"{sub['delta_wga'].mean():+7.4f}")
    print("  ".join(f"{x:<20s}" if i == 0 else f"{x:>8s}" for i, x in enumerate(row)))
