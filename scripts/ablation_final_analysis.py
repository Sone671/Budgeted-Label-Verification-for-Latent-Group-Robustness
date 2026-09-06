#!/usr/bin/env python
"""Cross-dataset ablation summary: Table A + Table B + Table C + Table D."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ── Load ────────────────────────────────────────────────────────────────────
wb = pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv")
ce = pd.read_csv("outputs/ablation_celeba_10seeds/stage1/results.csv")
cc = pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/results.csv")

# ── Methods in display order ────────────────────────────────────────────────
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
BUDGETS = [0.01, 0.02, 0.05, 0.10]

DATASETS = {"waterbirds": wb, "celeba": ce, "civilcomments": cc}


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def get_vals(df, noise, budget, method):
    sub = df[
        (df["noise_name"] == noise)
        & (df["budget_fraction"] == budget)
        & (df["method"] == method)
    ]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE A: Full ablation summary (uniform20, b=5%)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 120)
print("TABLE A: Ablation summary — uniform20, budget=5%")
print("=" * 120)
header = f"{'Method':<20s} {'WB_dWGA':>9s} {'CE_dWGA':>9s} {'CC_dWGA':>9s} {'WB_neg%':>7s} {'CE_neg%':>7s} {'CC_neg%':>7s} {'Mean':>9s}"
print(header)
print("-" * 100)

for m in METHODS:
    vals = []
    negs = []
    lbl = LABELS.get(m, m)
    for ds_name, df in DATASETS.items():
        v = get_vals(df, "uniform20", 0.05, m)
        if len(v) > 0:
            vals.append(f"{v.mean():+9.4f}")
            negs.append(f"{(v < 0).mean() * 100:>6.0f}%")
        else:
            vals.append(f"{'N/A':>9s}")
            negs.append(f"{'N/A':>6s}")
    all_v = np.concatenate(
        [
            get_vals(df, "uniform20", 0.05, m)
            for df in DATASETS.values()
            if len(get_vals(df, "uniform20", 0.05, m)) > 0
        ]
    )
    mean_all = f"{all_v.mean():+9.4f}" if len(all_v) > 0 else f"{'N/A':>9s}"
    print(f"{lbl:<20s} {vals[0]:>9s} {vals[1]:>9s} {vals[2]:>9s} {negs[0]:>7s} {negs[1]:>7s} {negs[2]:>7s} {mean_all:>9s}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE B: Waterbirds failure table (uniform20, b=5%, b=10%)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 120}")
print("TABLE B: Waterbirds failure analysis — uniform20, b=5% + 10%")
print(f"{'=' * 120}")

for budget in [0.05, 0.10]:
    noise = "uniform20"
    print(f"\n  budget={budget:.0%}:")
    print(f"  {'Method':<20s} {'dWGA':>9s} {'±sem':>9s} {'neg%':>6s} {'min':>9s} {'max':>9s} {'noise_prec':>10s}")
    print(f"  {'-' * 65}")
    for m in METHODS:
        sub = wb[
            (wb["noise_name"] == noise)
            & (wb["budget_fraction"] == budget)
            & (wb["method"] == m)
        ]
        if sub.empty:
            continue
        vals = sub["delta_wga"].to_numpy()
        prec = sub["noise_precision"].mean()
        lbl = LABELS.get(m, m)
        print(f"  {lbl:<20s} {vals.mean():>+9.4f} {sem(vals):>9.4f} {(vals<0).mean()*100:>5.0f}% {vals.min():>+9.4f} {vals.max():>+9.4f} {prec:>10.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE C: Component attribution (uniform20, b=5%)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 120}")
print("TABLE C: Component attribution — uniform20, b=5%")
print(f"{'=' * 120}")
print(f"  CPBA effect = CPBA-only - NoiseScore")
print(f"  TC effect   = TC-only - NoiseScore")
print(f"  v8FB effect = CPBA-TC-v8FB - CPBA-TC")
print(f"  noFB effect = CPBA-TC-noFB - CPBA-TC")
print()
print(f"  {'Dataset':<15s} {'CPBA_eff':>9s} {'TC_eff':>9s} {'v8FB_eff':>9s} {'noFB_eff':>9s}")
print(f"  {'-' * 55}")
for ds_name, df in DATASETS.items():
    v1 = get_vals(df, "uniform20", 0.05, "noise_score")
    cpba = get_vals(df, "uniform20", 0.05, "noise_score_cpba_only")
    tc = get_vals(df, "uniform20", 0.05, "noise_score_tc_only")
    cpba_tc = get_vals(df, "uniform20", 0.05, "noise_score_cpba_tc")
    v8fb = get_vals(df, "uniform20", 0.05, "noise_score_cpba_tc_v8_fallback")
    nofb = get_vals(df, "uniform20", 0.05, "noise_score_cpba_tc_no_fallback")
    if len(v1) == 0 or len(cpba) == 0:
        continue
    cpba_eff = f"{(cpba.mean() - v1.mean()):+9.4f}"
    tc_eff = f"{(tc.mean() - v1.mean()):+9.4f}" if len(tc) > 0 else "N/A"
    v8fb_eff = f"{(v8fb.mean() - cpba_tc.mean()):+9.4f}" if len(v8fb) > 0 and len(cpba_tc) > 0 else "N/A"
    nofb_eff = f"{(nofb.mean() - cpba_tc.mean()):+9.4f}" if len(nofb) > 0 and len(cpba_tc) > 0 else "N/A"
    print(f"  {ds_name:<15s} {cpba_eff:>9s} {tc_eff:>9s} {v8fb_eff:>9s} {nofb_eff:>9s}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE D: Paired significance (Waterbirds + CivilComments, uniform20, b=5%)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 120}")
print("TABLE D: Paired significance tests — uniform20, b=5%")
print(f"{'=' * 120}")

PAIRS = [
    ("noise_score_cpba_tc_v8_fallback", "noise_score_cpba_tc", "v8FB vs CPBA-TC"),
    ("noise_score_cpba_tc_no_fallback", "noise_score_cpba_tc", "noFB vs CPBA-TC"),
    ("noise_score_cpba_only", "noise_score", "CPBA-only vs NoiseScore"),
    ("noise_score_tc_only", "noise_score", "TC-only vs NoiseScore"),
    ("noise_score_cpba_tc_v8_fallback", "noise_score_budget_hybrid_v8", "v8FB vs NGC"),
]

for ds_name in ["waterbirds", "celeba", "civilcomments"]:
    df = DATASETS[ds_name]
    print(f"\n  [{ds_name}]")
    for m1, m2, desc in PAIRS:
        v1 = get_vals(df, "uniform20", 0.05, m1)
        v2 = get_vals(df, "uniform20", 0.05, m2)
        if len(v1) < 2 or len(v2) < 2:
            print(f"    {desc:<30s} N/A")
            continue
        try:
            t, p = stats.ttest_rel(v1, v2)
        except:
            p = 1.0
        diff = f"{(v1.mean() - v2.mean()):+.4f}"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {desc:<30s} Δ={diff}  p={p:.4f}  {sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# Minority_high_40 summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 120}")
print("APPENDIX: minority_high_40, b=5%")
print(f"{'=' * 120}")
print(f"  {'Method':<20s} {'WB_dWGA':>9s} {'CE_dWGA':>9s} {'CC_dWGA':>9s}")
for m in METHODS:
    vals = []
    lbl = LABELS.get(m, m)
    for ds_name, df in DATASETS.items():
        v = get_vals(df, "minority_high_40", 0.05, m)
        vals.append(f"{v.mean():+9.4f}" if len(v) > 0 else f"{'N/A':>9s}")
    print(f"  {lbl:<20s} {vals[0]:>9s} {vals[1]:>9s} {vals[2]:>9s}")
