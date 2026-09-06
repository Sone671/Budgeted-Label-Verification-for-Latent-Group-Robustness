#!/usr/bin/env python
"""Complete comparison: ablation + new baselines + model baselines, 3 datasets."""

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
    sub = df[
        (df["noise_name"] == noise)
        & (df["budget_fraction"] == budget)
        & (df["method"] == method)
    ]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


def get_all_vals(df, method, noise=None):
    sub = df[df["method"] == method]
    if noise is not None:
        sub = sub[sub["noise_name"] == noise]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


def find_vals(ds_name, method, noise, budget):
    for src in [BAS, ABL]:
        if ds_name in src:
            v = get_vals(src[ds_name], noise, budget, method)
            if len(v) > 0:
                return v
    return np.array([])


LABELS = {
    "random": "Random",
    "loss": "Loss",
    "entropy": "Entropy",
    "forgetting": "Forgetting",
    "margin": "Margin",
    "aum": "AUM",
    "confident_learning": "Cleanlab",
    "coreset": "Coreset-Lite",
    "uncertainty_diversity": "Unc+Div-Lite",
    "badge_lite": "BADGE-Lite",
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "tracin_val_uncertainty": "TracIn(val)",
    "tracin_noise_weighted": "TracIn(noise)",
    "tracin_wga_oracle": "TracIn(oracle)",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "noise_score_cpba_tc": "CPBA-TC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_tc_only": "TC-only",
    "oracle_noise_group_balanced": "Oracle-GB",
}

QUERY_ALL = [
    "random", "loss", "entropy",
    "forgetting", "margin", "aum", "confident_learning",
    "coreset", "uncertainty_diversity", "badge_lite",
    "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "noise_score_budget_hybrid", "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc_v8_fallback",
    "oracle_noise_group_balanced",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: 主表 — uniform20 b=5% 压力测试
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 120)
print("TABLE 1: All baselines — uniform20, b=5%")
print("=" * 120)
print(f"{'Method':<18s} {'WB dWGA':>10s} {'WB neg%':>7s} {'WB prec':>7s} {'CE dWGA':>10s} {'CE neg%':>7s} {'CC dWGA':>10s} {'CC neg%':>7s}")
print("-" * 88)

for method in QUERY_ALL:
    lbl = LABELS.get(method, method)
    wb_v = find_vals("Waterbirds", method, "uniform20", 0.05)
    ce_v = find_vals("CelebA", method, "uniform20", 0.05)
    cc_v = find_vals("CivilComments", method, "uniform20", 0.05)

    wb_str = f"{wb_v.mean():+10.4f}" if len(wb_v) > 0 else f"{'  N/A':>10s}"
    wb_neg = f"{(wb_v<0).mean()*100:>6.0f}%" if len(wb_v) > 0 else f"{'  N/A':>6s}"
    ce_str = f"{ce_v.mean():+10.4f}" if len(ce_v) > 0 else f"{'  N/A':>10s}"
    ce_neg = f"{(ce_v<0).mean()*100:>6.0f}%" if len(ce_v) > 0 else f"{'  N/A':>6s}"
    cc_str = f"{cc_v.mean():+10.4f}" if len(cc_v) > 0 else f"{'  N/A':>10s}"
    cc_neg = f"{(cc_v<0).mean()*100:>6.0f}%" if len(cc_v) > 0 else f"{'  N/A':>6s}"

    # Noise precision on WB
    ds = BAS.get("Waterbirds")
    if ds is not None:
        sub = ds[
            (ds["noise_name"] == "uniform20")
            & (ds["budget_fraction"] == 0.05)
            & (ds["method"] == method)
        ]
        wb_prec = f"{sub['noise_precision'].mean():.3f}" if not sub.empty else "  N/A"
    else:
        wb_prec = "  N/A"

    print(f"{lbl:<18s} {wb_str} {wb_neg} {wb_prec:>7s} {ce_str} {ce_neg} {cc_str} {cc_neg}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2: 全局平均 — over all budgets + both noises
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("TABLE 2: Pooled over all budgets + both noises")
print("=" * 120)
print(f"{'Method':<18s} {'WB':>9s} {'CE':>9s} {'CC':>9s} {'Mean':>9s}")

results = []
for method in QUERY_ALL:
    lbl = LABELS.get(method, method)
    per_ds = []
    all_v = []
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        v = find_vals(ds_name, method, "uniform20", 0.05)
        # Try pooled over all conditions
        ds_v = []
        for noise in NOISES:
            for b in BUDGETS:
                ds_v.extend(find_vals(ds_name, method, noise, b).tolist())
        if len(ds_v) > 0:
            per_ds.append(np.mean(ds_v))
            all_v.extend(ds_v)
        else:
            per_ds.append(float("nan"))

    if len(all_v) > 0:
        results.append((lbl, per_ds, np.mean(all_v)))

results.sort(key=lambda x: x[2], reverse=True)
for lbl, per_ds, mean_val in results:
    pds = " & ".join(
        [f"{v:+9.4f}" if not np.isnan(v) else f"{'  N/A':>9s}" for v in per_ds]
    )
    print(f"  {lbl:<18s} {pds}  {mean_val:+9.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3: 模型基线
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("TABLE 3: Model-training baselines (zero-query)")
print("=" * 120)
print(f"{'Method':<20s} {'WB WGA':>10s} {'CE WGA':>10s} {'CC WGA':>10s}")
print("-" * 55)

for method in ["erm", "oracle_groupdro", "jtt"]:
    lbl = {"erm": "ERM", "oracle_groupdro": "Oracle GroupDRO", "jtt": "JTT"}[method]
    row = [lbl]
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        mod = MOD.get(ds_name)
        if mod is None:
            row.append("N/A")
            continue
        sub = mod[mod["method"] == method]
        if sub.empty:
            row.append("N/A")
        else:
            vals = sub["wga"].to_numpy()
            row.append(f"{vals.mean():.4f}±{sem(vals):.4f}")
    print(f"  {row[0]:<20s} {row[1]:>10s} {row[2]:>10s} {row[3]:>10s}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4: 按数据集分类的噪声精度 vs WGA (Hook 验证)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("TABLE 4: Noise precision vs delta-WGA — Waterbirds uniform20, all budgets")
print("=" * 120)
print(f"{'Method':<18s} {'b=0.5%':>8s} {'b=1%':>8s} {'b=2%':>8s} {'b=5%':>8s} {'prec@5%':>8s} {'neg@5%':>7s}")
print("-" * 70)

for method in [
    "loss", "confident_learning", "aum", "margin",
    "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "forgetting", "entropy", "coreset",
    "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced",
]:
    lbl = LABELS.get(method, method)
    row = [lbl]
    prec_5 = 0.0
    neg_5 = 0.0
    for b in [0.005, 0.01, 0.02, 0.05]:
        vals = find_vals("Waterbirds", method, "uniform20", b)
        row.append(f"{vals.mean():+8.4f}" if len(vals) > 0 else f"{'N/A':>8s}")
        if b == 0.05 and len(vals) > 0:
            ds = BAS["Waterbirds"]
            sub = ds[
                (ds["noise_name"] == "uniform20")
                & (ds["budget_fraction"] == b)
                & (ds["method"] == method)
            ]
            if not sub.empty:
                prec_5 = sub["noise_precision"].mean()
                neg_5 = (vals < 0).mean() * 100
    if method not in ["oracle_noise_group_balanced"]:
        row.append(f"{prec_5:.3f}")
        row.append(f"{neg_5:>6.0f}%")
    else:
        row.append("  N/A")
        row.append("  N/A")
    print("  ".join(f"{x:<18s}" if i == 0 else f"{x:>8s}" for i, x in enumerate(row)))


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5: Paired significance — Safe vs key challengers (WB + CC, u20 b=5%)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("TABLE 5: CPBA-TC-Safe vs challengers — uniform20 b=5%, paired t-test")
print("=" * 120)

SAFE = "noise_score_cpba_tc_v8_fallback"
CHALLENGERS = [
    "random", "loss", "entropy", "forgetting",
    "margin", "aum", "confident_learning",
    "coreset", "uncertainty_diversity", "badge_lite",
    "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "noise_score_budget_hybrid_v8",
    "oracle_noise_group_balanced",
]

for ds_name in ["Waterbirds", "CivilComments"]:
    print(f"\n  [{ds_name}]")
    print(f"  {'Comparison':<28s} {'Δ':>9s} {'p':>9s} {'sig':>5s} {'win/loss':>12s}")
    print(f"  {'-'*65}")
    safe_v = find_vals(ds_name, SAFE, "uniform20", 0.05)
    for chal in CHALLENGERS:
        chal_v = find_vals(ds_name, chal, "uniform20", 0.05)
        if len(safe_v) < 2 or len(chal_v) < 2:
            continue
        n = min(len(safe_v), len(chal_v))
        diff = safe_v[:n] - chal_v[:n]
        try:
            t, p = stats.ttest_rel(safe_v[:n], chal_v[:n])  # type: ignore
        except:
            p = 1.0
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        w = int((diff > 0.0005).sum())
        l = int((diff < -0.0005).sum())
        lbl = LABELS.get(chal, chal)
        print(f"  Safe vs {lbl:<22s} {diff.mean():+9.4f} {p:>9.4f} {sig:>5s} {w}/{l:>2}")

print("\nDone. Run: python scripts/baseline_complete_tables.py")
