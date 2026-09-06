#!/usr/bin/env python
"""Export all comparison tables to CSV files."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ── Load ────────────────────────────────────────────────────────────────────
ABL, BAS, MOD = {}, {}, {}
for ds, p in [
    ("Waterbirds", "outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
    ("CelebA", "outputs/ablation_celeba_10seeds/stage1/results.csv"),
    ("CivilComments", "outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
]:
    ABL[ds] = pd.read_csv(p)

for ds, p in [
    ("Waterbirds", "outputs/baselines_waterbirds_recommended/stage1/results.csv"),
    ("CelebA", "outputs/baselines_celeba_recommended/stage1/results.csv"),
    ("CivilComments", "outputs/baselines_civilcomments_10seeds/stage1/results.csv"),
]:
    BAS[ds] = pd.read_csv(p)

for ds, p in [
    ("Waterbirds", "outputs/baselines_waterbirds_recommended/stage1/model_baselines.csv"),
    ("CelebA", "outputs/baselines_celeba_recommended/stage1/model_baselines.csv"),
    ("CivilComments", "outputs/baselines_civilcomments_10seeds/stage1/model_baselines.csv"),
]:
    MOD[ds] = pd.read_csv(p)

OUT = "outputs/paper_figures"
NOISES = ["uniform20", "minority_high_40"]
BUDGETS = [0.005, 0.01, 0.02, 0.05]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def get_vals(df, noise, budget, method):
    sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget) & (df["method"] == method)]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


LABELS = {
    "random": "Random", "loss": "Loss", "entropy": "Entropy",
    "forgetting": "Forgetting", "margin": "Margin", "aum": "AUM",
    "confident_learning": "Cleanlab", "coreset": "Coreset-Lite",
    "uncertainty_diversity": "Unc+Div-Lite", "badge_lite": "BADGE-Lite",
    "noise_score": "NoiseScore", "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "tracin_val_uncertainty": "TracIn(val)", "tracin_noise_weighted": "TracIn(noise)",
    "tracin_wga_oracle": "TracIn(oracle)",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "noise_score_cpba_tc": "CPBA-TC", "noise_score_cpba_only": "CPBA-only",
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


def find_vals(ds_name, method, noise, budget):
    for src in [BAS, ABL]:
        if ds_name in src:
            v = get_vals(src[ds_name], noise, budget, method)
            if len(v) > 0:
                return v
    return np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 1: 主表 — uniform20 b=5%，含列：method, label, WB_dWGA_mean, WB_neg%, WB_prec, CE_dWGA_mean, CE_neg%, CC_dWGA_mean, CC_neg%
# ═══════════════════════════════════════════════════════════════════════════════

rows = []
for method in QUERY_ALL:
    lbl = LABELS.get(method, method)
    wb_v = find_vals("Waterbirds", method, "uniform20", 0.05)
    ce_v = find_vals("CelebA", method, "uniform20", 0.05)
    cc_v = find_vals("CivilComments", method, "uniform20", 0.05)

    row = {
        "method": method,
        "label": lbl,
        "WB_dWGA_mean": round(float(wb_v.mean()), 4) if len(wb_v) > 0 else "",
        "WB_dWGA_sem": round(sem(wb_v), 4) if len(wb_v) > 0 else "",
        "WB_neg_pct": round(float((wb_v < 0).mean()) * 100, 1) if len(wb_v) > 0 else "",
        "CE_dWGA_mean": round(float(ce_v.mean()), 4) if len(ce_v) > 0 else "",
        "CE_dWGA_sem": round(sem(ce_v), 4) if len(ce_v) > 0 else "",
        "CE_neg_pct": round(float((ce_v < 0).mean()) * 100, 1) if len(ce_v) > 0 else "",
        "CC_dWGA_mean": round(float(cc_v.mean()), 4) if len(cc_v) > 0 else "",
        "CC_dWGA_sem": round(sem(cc_v), 4) if len(cc_v) > 0 else "",
        "CC_neg_pct": round(float((cc_v < 0).mean()) * 100, 1) if len(cc_v) > 0 else "",
    }
    # noise precision on WB
    ds = BAS.get("Waterbirds")
    if ds is not None:
        sub = ds[(ds["noise_name"] == "uniform20") & (ds["budget_fraction"] == 0.05) & (ds["method"] == method)]
        if not sub.empty:
            row["WB_noise_prec"] = round(float(sub["noise_precision"].mean()), 4)
    rows.append(row)

df1 = pd.DataFrame(rows)
df1.to_csv(f"{OUT}/table1_all_baselines_u20_5pct.csv", index=False)
print(f"Saved: table1_all_baselines_u20_5pct.csv ({len(df1)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 2: 全局平均 — all budgets + both noises
# ═══════════════════════════════════════════════════════════════════════════════

rows2 = []
for method in QUERY_ALL:
    lbl = LABELS.get(method, method)
    row = {"method": method, "label": lbl}
    all_v = []
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        ds_v = []
        for noise in NOISES:
            for b in BUDGETS:
                ds_v.extend(find_vals(ds_name, method, noise, b).tolist())
        if len(ds_v) > 0:
            row[f"{ds_name}_dWGA_mean"] = round(float(np.mean(ds_v)), 4)
            row[f"{ds_name}_dWGA_sem"] = round(sem(np.array(ds_v)), 4)
            row[f"{ds_name}_n"] = len(ds_v)
            all_v.extend(ds_v)
    if len(all_v) > 0:
        row["overall_mean"] = round(float(np.mean(all_v)), 4)
        row["overall_sem"] = round(sem(np.array(all_v)), 4)
        row["overall_n"] = len(all_v)
    rows2.append(row)

df2 = pd.DataFrame(rows2)
df2.sort_values("overall_mean", ascending=False, inplace=True)
df2.to_csv(f"{OUT}/table2_overall_pooled.csv", index=False)
print(f"Saved: table2_overall_pooled.csv ({len(df2)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 3: 模型基线
# ═══════════════════════════════════════════════════════════════════════════════

rows3 = []
M_LABELS = {"erm": "ERM", "oracle_groupdro": "Oracle GroupDRO", "jtt": "JTT"}
for method in ["erm", "oracle_groupdro", "jtt"]:
    row = {"method": method, "label": M_LABELS[method]}
    for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
        mod = MOD.get(ds_name)
        if mod is None:
            continue
        sub = mod[mod["method"] == method]
        if sub.empty:
            continue
        wga = sub["wga"].to_numpy()
        row[f"{ds_name}_WGA_mean"] = round(float(wga.mean()), 4)
        row[f"{ds_name}_WGA_sem"] = round(sem(wga), 4)
        row[f"{ds_name}_n"] = len(wga)
    rows3.append(row)

df3 = pd.DataFrame(rows3)
df3.to_csv(f"{OUT}/table3_model_baselines.csv", index=False)
print(f"Saved: table3_model_baselines.csv ({len(df3)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 4: Hook — noise precision vs WGA, Waterbirds u20, per-budget
# ═══════════════════════════════════════════════════════════════════════════════

rows4 = []
HOOK_METHODS = [
    "loss", "confident_learning", "aum", "margin",
    "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "forgetting", "entropy", "coreset",
    "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced",
]
for method in HOOK_METHODS:
    lbl = LABELS.get(method, method)
    row = {"method": method, "label": lbl}
    for b in [0.005, 0.01, 0.02, 0.05]:
        vals = find_vals("Waterbirds", method, "uniform20", b)
        if len(vals) > 0:
            row[f"b={b:.1%}"] = round(float(vals.mean()), 4)
    vals_5 = find_vals("Waterbirds", method, "uniform20", 0.05)
    if len(vals_5) > 0:
        ds = BAS["Waterbirds"]
        sub = ds[(ds["noise_name"] == "uniform20") & (ds["budget_fraction"] == 0.05) & (ds["method"] == method)]
        if not sub.empty:
            row["noise_prec_5pct"] = round(float(sub["noise_precision"].mean()), 4)
            row["neg_pct_5pct"] = round(float((vals_5 < 0).mean()) * 100, 1)
            row["wga_5pct"] = round(float(sub["wga"].mean()), 4)
    rows4.append(row)

df4 = pd.DataFrame(rows4)
df4.to_csv(f"{OUT}/table4_hook_precision_vs_wga.csv", index=False)
print(f"Saved: table4_hook_precision_vs_wga.csv ({len(df4)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 5: Paired significance — Safe vs all
# ═══════════════════════════════════════════════════════════════════════════════

rows5 = []
SAFE = "noise_score_cpba_tc_v8_fallback"
for ds_name in ["Waterbirds", "CivilComments"]:
    safe_v = find_vals(ds_name, SAFE, "uniform20", 0.05)
    for chal in QUERY_ALL:
        if chal == SAFE:
            continue
        chal_v = find_vals(ds_name, chal, "uniform20", 0.05)
        if len(safe_v) < 2 or len(chal_v) < 2:
            continue
        n = min(len(safe_v), len(chal_v))
        diff = safe_v[:n] - chal_v[:n]
        try:
            t, p = stats.ttest_rel(safe_v[:n], chal_v[:n])
        except:
            t, p = float("nan"), 1.0
        try:
            _, p_w = stats.wilcoxon(safe_v[:n], chal_v[:n])
        except:
            p_w = float("nan")
        w = int((diff > 0.0005).sum())
        l = int((diff < -0.0005).sum())
        tie = n - w - l
        rows5.append({
            "dataset": ds_name,
            "comparison": f"Safe vs {LABELS.get(chal, chal)}",
            "challenger": chal,
            "n_seeds": n,
            "mean_diff": round(float(diff.mean()), 4),
            "ci95_low": round(float(diff.mean() - 1.96 * sem(diff)), 4),
            "ci95_high": round(float(diff.mean() + 1.96 * sem(diff)), 4),
            "p_ttest": round(p, 4) if np.isfinite(p) else "",
            "p_wilcoxon": round(p_w, 4) if np.isfinite(p_w) else "",
            "sig": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")),
            "win": w, "tie": tie, "loss": l,
        })

df5 = pd.DataFrame(rows5)
df5.to_csv(f"{OUT}/table5_paired_significance.csv", index=False)
print(f"Saved: table5_paired_significance.csv ({len(df5)} rows)")

print(f"\nAll CSVs saved to {OUT}/")
