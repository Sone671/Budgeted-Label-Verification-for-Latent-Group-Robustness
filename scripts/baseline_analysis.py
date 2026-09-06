#!/usr/bin/env python
"""Baseline analysis: new baselines vs existing methods across 3 datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ── Load ────────────────────────────────────────────────────────────────────
ABL = {}
for ds, path in [
    ("Waterbirds", "outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
    ("CelebA", "outputs/ablation_celeba_10seeds/stage1/results.csv"),
    ("CivilComments", "outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
]:
    ABL[ds] = pd.read_csv(path)

BAS = {}
for ds, path in [
    ("Waterbirds", "outputs/baselines_waterbirds_recommended/stage1/results.csv"),
    ("CelebA", "outputs/baselines_celeba_recommended/stage1/results.csv"),
    ("CivilComments", "outputs/baselines_civilcomments_10seeds/stage1/results.csv"),
]:
    BAS[ds] = pd.read_csv(path)

MOD = {}
for ds, path in [
    ("Waterbirds", "outputs/baselines_waterbirds_recommended/stage1/model_baselines.csv"),
    ("CelebA", "outputs/baselines_celeba_recommended/stage1/model_baselines.csv"),
    ("CivilComments", "outputs/baselines_civilcomments_10seeds/stage1/model_baselines.csv"),
]:
    MOD[ds] = pd.read_csv(path)

NOISES = ["uniform20", "minority_high_40"]
BUDGETS = [0.005, 0.01, 0.02, 0.05]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def get_vals(df, noise, budget, method):
    sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget) & (df["method"] == method)]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A: Query baselines — uniform20 b=5% (the stress test)
# ═══════════════════════════════════════════════════════════════════════════════

LABELS = {
    "random": "Random", "loss": "Loss", "entropy": "Entropy",
    "forgetting": "Forgetting", "margin": "Margin", "aum": "AUM",
    "confident_learning": "Cleanlab",
    "coreset": "Coreset", "uncertainty_diversity": "Unc+Div",
    "badge_lite": "BADGE", "noise_score": "NoiseScore",
    "noise_score_budget_hybrid": "BHC", "noise_score_budget_hybrid_v8": "NGC",
    "tracin_val_uncertainty": "TracIn(val)",
    "tracin_noise_weighted": "TracIn(noise)",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "oracle_noise_group_balanced": "Oracle-GB",
}

QUERY_METHODS = [
    "random", "loss", "entropy", "forgetting", "margin", "aum",
    "confident_learning", "coreset", "uncertainty_diversity",
    "badge_lite", "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced",
]

print("=" * 110)
print("PANEL A: Query baselines — uniform20, b=5%")
print("=" * 110)
print(f"{'Method':<18s} {'WB_dWGA':>9s} {'CE_dWGA':>9s} {'CC_dWGA':>9s} {'WB_neg%':>7s} {'CE_neg%':>7s} {'CC_neg%':>7s} {'WB_prec':>7s}")
print("-" * 90)

for method in QUERY_METHODS:
    lbl = LABELS.get(method, method)
    row = [lbl]
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        # Try ablation dir first, then baselines dir
        for src in [ABL, BAS]:
            vals = get_vals(src[ds_name], "uniform20", 0.05, method)
            if len(vals) > 0:
                break
        if len(vals) == 0:
            row.append(f"{'N/A':>9s}")
            row.append(f"{'N/A':>7s}")
            row.append(f"{'N/A':>7s}")
            continue
        row.append(f"{vals.mean():+9.4f}")
        row.append(f"{(vals<0).mean()*100:>6.0f}%")
        if ds_name == "Waterbirds":
            s = src[ds_name]
            sub = s[(s["noise_name"]=="uniform20")&(s["budget_fraction"]==0.05)&(s["method"]==method)]
            p = sub["noise_precision"].mean() if not sub.empty else 0.0
            row.append(f"{p:>6.3f}")
    print("  ".join(f"{x:<18s}" if i == 0 else f"{x:>9s}" if i in [1,2,3] else f"{x:>7s}" for i, x in enumerate(row)))


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B: Model baselines (ERM, GroupDRO, JTT)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 110)
print("PANEL B: Model-training baselines (zero-query, no budget)")
print("=" * 110)
print(f"{'Method':<20s} {'WB_WGA':>10s} {'CE_WGA':>10s} {'CC_WGA':>10s}")
print("-" * 55)

MLABELS = {"erm": "ERM", "oracle_groupdro": "Oracle GroupDRO", "jtt": "JTT"}
for method in ["erm", "oracle_groupdro", "jtt"]:
    row = [MLABELS[method]]
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        mod = MOD[ds_name]
        sub = mod[mod["method"] == method]
        if sub.empty:
            row.append(f"{'N/A':>10s}")
        else:
            vals = sub["wga"].to_numpy()
            # Show both uniform20 and minority_high_40 average
            u20 = sub[sub["noise_name"] == "uniform20"]["wga"]
            mh40 = sub[sub["noise_name"] == "minority_high_40"]["wga"]
            row.append(f"{vals.mean():.4f}")
    print("  ".join(f"{x:<20s}" if i == 0 else f"{x:>10s}" for i, x in enumerate(row)))


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C: Key comparisons — CPBA-TC-Safe vs strongest new baselines
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 110)
print("PANEL C: CPBA-TC-Safe vs top new baselines — uniform20 b=5%, paired tests")
print("=" * 110)

SAFE = "noise_score_cpba_tc_v8_fallback"
CHALLENGERS = [
    "tracin_val_uncertainty", "tracin_noise_weighted",
    "coreset", "uncertainty_diversity", "badge_lite",
    "margin", "aum", "confident_learning",
]

for ds_name in ["Waterbirds", "CivilComments"]:
    print(f"\n  [{ds_name}]")
    safe_vals = get_vals(BAS[ds_name], "uniform20", 0.05, SAFE)
    if len(safe_vals) == 0:
        safe_vals = get_vals(ABL[ds_name], "uniform20", 0.05, SAFE)
    for chal in CHALLENGERS:
        chal_vals = get_vals(BAS[ds_name], "uniform20", 0.05, chal)
        if len(chal_vals) < 2 or len(safe_vals) < 2:
            continue
        diff = safe_vals - chal_vals[:len(safe_vals)]
        try:
            t, p = stats.ttest_rel(safe_vals, chal_vals[:len(safe_vals)])  # type: ignore
        except:
            p = 1.0
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        lbl = LABELS.get(chal, chal)
        print(f"    Safe vs {lbl:<18s} D={diff.mean():+7.4f}  p={p:.4f} {sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL D: Noise detection vs WGA repair — the hook reinforcement
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 110)
print("PANEL D: Noise precision vs WGA gain — Waterbirds uniform20")
print("=" * 110)
print(f"{'Method':<18s} {'b=0.5%':>8s} {'b=1%':>8s} {'b=2%':>8s} {'b=5%':>8s} {'prec@5%':>8s} {'WGA@5%':>9s}")
print("-" * 70)

for method in ["loss", "confident_learning", "aum", "noise_score", "tracin_val_uncertainty", "entropy", "coreset", "noise_score_cpba_tc_v8_fallback"]:
    lbl = LABELS.get(method, method)
    row = [lbl]
    ds = BAS["Waterbirds"]
    prec_5 = 0.0
    for b in [0.005, 0.01, 0.02, 0.05]:
        vals = get_vals(ds, "uniform20", b, method)
        row.append(f"{vals.mean():+7.4f}" if len(vals) > 0 else "N/A")
        if b == 0.05 and len(vals) > 0:
            sub = ds[(ds["noise_name"]=="uniform20")&(ds["budget_fraction"]==b)&(ds["method"]==method)]
            prec_5 = sub["noise_precision"].mean()
    row.append(f"{prec_5:.3f}")
    # WGA at 5%
    sub = ds[(ds["noise_name"]=="uniform20")&(ds["budget_fraction"]==0.05)&(ds["method"]==method)]
    if not sub.empty:
        row.append(f"{sub['wga'].mean():.4f}")
    print("  ".join(f"{x:<18s}" if i == 0 else f"{x:>8s}" for i, x in enumerate(row)))


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL E: Overall ranking (pooled over all budgets/noises)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 110)
print("PANEL E: Overall ranking — all budgets, all noises pooled")
print("=" * 110)
print(f"{'Method':<20s} {'WB':>9s} {'CE':>9s} {'CC':>9s} {'Mean':>9s}")

ALL_METHODS_OVERALL = [
    "random", "loss", "entropy", "forgetting", "margin", "aum",
    "confident_learning", "coreset", "uncertainty_diversity",
    "badge_lite", "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced",
]

results_overall = []
for method in ALL_METHODS_OVERALL:
    lbl = LABELS.get(method, method)
    all_vals = []
    per_ds = []
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        for src in [ABL, BAS]:
            ds_vals = []
            for noise in NOISES:
                for b in BUDGETS:
                    v = get_vals(src[ds_name], noise, b, method)
                    ds_vals.extend(v.tolist())
            if len(ds_vals) > 0:
                per_ds.append(np.mean(ds_vals))
                all_vals.extend(ds_vals)
                break
    if len(all_vals) > 0:
        results_overall.append((method, lbl, per_ds, np.mean(all_vals)))

results_overall.sort(key=lambda x: x[3], reverse=True)
for method, lbl, per_ds, mean_val in results_overall:
    pds = " & ".join([f"{v:+9.4f}" for v in per_ds])
    print(f"  {lbl:<20s} {pds}  {mean_val:+9.4f}")
