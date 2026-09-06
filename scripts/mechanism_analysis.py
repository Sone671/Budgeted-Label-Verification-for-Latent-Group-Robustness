#!/usr/bin/env python
"""Mechanism analysis: why does CPBA-TC-Safe work?
Three prongs:
1. noise precision vs dWGA — "finding noise" != WGA repair
2. disagreement_lift vs dWGA — real-noise needs high-disagreement queries
3. minority/group coverage vs dWGA — group structure matters
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

OUT = Path("outputs/mechanism_analysis")
OUT.mkdir(parents=True, exist_ok=True)

MIN_Q_SIZE = 5

# ── Load ────────────────────────────────────────────────────────────────────
SOURCES = {
    "WB_abl": ("Waterbirds", "outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
    "CE_abl": ("CelebA", "outputs/ablation_celeba_10seeds/stage1/results.csv"),
    "CC_abl": ("CivilComments", "outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
    "CC_dis": ("CivilComments", "outputs/civilcomments_disagreement_10seeds/stage1/results.csv"),
    "WB_bas": ("Waterbirds", "outputs/baselines_waterbirds_recommended/stage1/results.csv"),
}

METHODS_FOCUS = [
    "random", "loss", "entropy", "forgetting", "margin", "aum",
    "confident_learning", "coreset", "uncertainty_diversity",
    "badge_lite", "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced",
]

LABELS = {
    "random": "Random", "loss": "Loss", "entropy": "Entropy",
    "noise_score": "NoiseScore", "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "oracle_noise_group_balanced": "Oracle-GB",
    "forgetting": "Forgetting", "confident_learning": "Cleanlab",
    "aum": "AUM", "margin": "Margin", "badge_lite": "BADGE",
    "coreset": "Coreset", "tracin_val_uncertainty": "TracIn(val)",
    "tracin_noise_weighted": "TracIn(noise)",
    "uncertainty_diversity": "Unc+Div",
}

KEY_METHODS = [
    "random", "loss", "entropy", "noise_score",
    "noise_score_budget_hybrid_v8", "noise_score_cpba_tc_v8_fallback",
    "oracle_noise_group_balanced",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PRONG 1: noise_precision vs dWGA — across datasets, uniform20, b=5%
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PRONG 1: Noise precision vs dWGA (uniform20, b=5%)")
print("=" * 70)

rows_p1 = []
for source_key, (ds_name, path) in SOURCES.items():
    df = pd.read_csv(path)
    sub_all = df[(df["noise_name"] == "uniform20") & (df["budget_fraction"] == 0.05)]
    for method in METHODS_FOCUS:
        sub = sub_all[sub_all["method"] == method]
        if len(sub) < MIN_Q_SIZE:
            continue
        vals = sub["delta_wga"].to_numpy()
        prec = sub["noise_precision"].to_numpy()
        rows_p1.append({
            "dataset": ds_name + ("(dis)" if "_dis" in source_key else ""),
            "method": method,
            "method_label": LABELS.get(method, method),
            "n_seeds": len(vals),
            "noise_precision": round(float(prec.mean()), 4),
            "dWGA_mean": round(float(vals.mean()), 4),
            "dWGA_sem": round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 4) if len(vals) > 1 else 0.0,
        })

df_p1 = pd.DataFrame(rows_p1)
df_p1.to_csv(OUT / "mechanism_noise_precision_vs_dWGA.csv", index=False)

# Per-dataset correlation
for ds in sorted(df_p1["dataset"].unique()):
    sub = df_p1[df_p1["dataset"] == ds]
    if len(sub) < 5:
        continue
    r, p = stats.spearmanr(sub["noise_precision"], sub["dWGA_mean"])
    print(f"  {ds:<20s} Spearman rho={r:+.3f}  p={p:.4f}")

# Overall correlation
r_all, p_all = stats.spearmanr(df_p1["noise_precision"], df_p1["dWGA_mean"])
print(f"  {'Overall':<20s} Spearman rho={r_all:+.3f}  p={p_all:.4f}")
print(f"  => {'' if r_all < 0 else 'POSITIVE'} correlation: higher precision ~ higher WGA, but weak/negative desired")


# ═══════════════════════════════════════════════════════════════════════════════
# PRONG 2: disagreement_lift vs dWGA — CC real-noise, b=5%
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PRONG 2: Disagreement lift vs dWGA (CC real-noise, b=5%)")
print("=" * 70)

df_cc = pd.read_csv(SOURCES["CC_dis"][1])
rows_p2 = []
for noise in ["uniform20", "disagreement_natural", "disagreement_matched20"]:
    sub_all = df_cc[(df_cc["noise_name"] == noise) & (df_cc["budget_fraction"] == 0.05)]
    for method in METHODS_FOCUS:
        sub = sub_all[sub_all["method"] == method]
        if len(sub) < MIN_Q_SIZE:
            continue
        vals = sub["delta_wga"].to_numpy()
        lift = sub["disagreement_lift"].to_numpy() if "disagreement_lift" in sub.columns else np.zeros(1)
        rows_p2.append({
            "noise": noise,
            "method": method,
            "method_label": LABELS.get(method, method),
            "n_seeds": len(vals),
            "dWGA_mean": round(float(vals.mean()), 4),
            "disagreement_lift": round(float(lift.mean()), 4),
            "noise_precision": round(float(sub["noise_precision"].mean()), 4),
        })

df_p2 = pd.DataFrame(rows_p2)
df_p2.to_csv(OUT / "mechanism_disagreement_lift_vs_dWGA.csv", index=False)

for noise in sorted(df_p2["noise"].unique()):
    sub = df_p2[df_p2["noise"] == noise]
    if len(sub) < 5:
        continue
    r, p = stats.spearmanr(sub["disagreement_lift"], sub["dWGA_mean"])
    print(f"  {noise:<25s} rho={r:+.3f}  p={p:.4f}")

# Which noise type benefits most from high-disagreement queries?
for noise in sorted(df_p2["noise"].unique()):
    sub = df_p2[df_p2["noise"] == noise]
    top_lift = sub.loc[sub["dWGA_mean"].idxmax()]
    print(f"  {noise:<25s} best method: {top_lift['method_label']} (lift={top_lift['disagreement_lift']:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
# PRONG 3: minority/group coverage vs dWGA — across datasets, uniform20
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PRONG 3: Minority/group coverage vs dWGA")
print("=" * 70)

rows_p3 = []
for source_key, (ds_name, path) in SOURCES.items():
    if "_dis" in source_key:
        continue
    df = pd.read_csv(path)
    for budget in [0.01, 0.05]:
        sub_all = df[(df["noise_name"] == "uniform20") & (df["budget_fraction"] == budget)]
        for method in METHODS_FOCUS:
            sub = sub_all[sub_all["method"] == method]
            if len(sub) < MIN_Q_SIZE:
                continue
            vals = sub["delta_wga"].to_numpy()
            rows_p3.append({
                "dataset": ds_name,
                "budget": f"{budget:.1%}",
                "method": method,
                "method_label": LABELS.get(method, method),
                "n_seeds": len(vals),
                "dWGA_mean": round(float(vals.mean()), 4),
                "noise_precision": round(float(sub["noise_precision"].mean()), 4),
                "minority_query_rate": round(float(sub["minority_query_rate"].mean()), 4),
                "group_query_entropy": round(float(sub["group_query_entropy"].mean()), 4) if "group_query_entropy" in sub.columns else 0.0,
            })

df_p3 = pd.DataFrame(rows_p3)
df_p3.to_csv(OUT / "mechanism_group_coverage_vs_dWGA.csv", index=False)

# Per-dataset: minority_query_rate vs dWGA correlation
for ds in sorted(df_p3["dataset"].unique()):
    for b in ["1%", "5%"]:
        sub = df_p3[(df_p3["dataset"] == ds) & (df_p3["budget"] == b)]
        if len(sub) < 5:
            continue
        r_mq, p_mq = stats.spearmanr(sub["minority_query_rate"], sub["dWGA_mean"])
        r_ge, p_ge = stats.spearmanr(sub["group_query_entropy"].fillna(0), sub["dWGA_mean"])
        print(f"  {ds:<15s} b={b}: minority_q rho={r_mq:+.3f} p={p_mq:.4f} | group_entropy rho={r_ge:+.3f} p={p_ge:.4f}")

# Waterbirds specifically: compare noise_precision vs minority_query_rate
print("\n  --- Waterbirds uniform20 b=5%: feature importance ---")
wb = df_p3[(df_p3["dataset"] == "Waterbirds") & (df_p3["budget"] == "5%")]
for col, label in [("noise_precision", "noise prec"), ("minority_query_rate", "minority q"),
                    ("group_query_entropy", "group entropy")]:
    r, p = stats.spearmanr(wb[col].fillna(0), wb["dWGA_mean"])
    print(f"    {label:<20s} rho={r:+.3f}  p={p:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table: which feature best predicts dWGA?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: Feature importance for dWGA (Spearman rho)")
print("=" * 70)

rows_summ = []
for ds_key, ds_name in [("WB_abl", "Waterbirds"), ("CE_abl", "CelebA"), ("CC_abl", "CivilComments")]:
    df = pd.read_csv(SOURCES[ds_key][1])
    sub = df[(df["noise_name"] == "uniform20") & (df["budget_fraction"] == 0.05)]
    for metric, col in [("noise_precision", "noise_precision"),
                         ("minority_q", "minority_query_rate"),
                         ("group_entropy", "group_query_entropy")]:
        data = sub.groupby("method").agg(
            dWGA=("delta_wga", "mean"),
            metric_val=(col, "mean"),
        ).dropna()
        if len(data) < 5:
            continue
        r, p = stats.spearmanr(data["metric_val"], data["dWGA"])
        rows_summ.append({
            "dataset": ds_name,
            "metric": metric,
            "spearman_rho": round(r, 3),
            "p_value": round(p, 4),
            "n_methods": len(data),
        })

df_summ = pd.DataFrame(rows_summ)
df_summ.to_csv(OUT / "mechanism_summary.csv", index=False)
print(df_summ.to_string(index=False))

print(f"\nAll CSVs saved to: {OUT}/")
