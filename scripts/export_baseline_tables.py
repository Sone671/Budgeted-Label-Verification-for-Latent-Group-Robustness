#!/usr/bin/env python
"""Export complete baseline comparison tables to CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd

OUT = "outputs/baseline_comparison"
from pathlib import Path

Path(OUT).mkdir(parents=True, exist_ok=True)

# ── Load sources ────────────────────────────────────────────────────────────
ABL = {
    "WB": pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
    "CE": pd.read_csv("outputs/ablation_celeba_10seeds/stage1/results.csv"),
    "CC": pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
}
BAS = {
    "WB": pd.read_csv("outputs/baselines_waterbirds_recommended/stage1/results.csv"),
    "CE": pd.read_csv("outputs/baselines_celeba_recommended/stage1/results.csv"),
    "CC": pd.read_csv("outputs/baselines_civilcomments_10seeds/stage1/results.csv"),
}
MOD = {
    "WB": pd.read_csv("outputs/baselines_waterbirds_recommended/stage1/model_baselines.csv"),
    "CE": pd.read_csv("outputs/baselines_celeba_recommended/stage1/model_baselines.csv"),
    "CC": pd.read_csv("outputs/baselines_civilcomments_10seeds/stage1/model_baselines.csv"),
}

DS_NAMES = {"WB": "Waterbirds", "CE": "CelebA", "CC": "CivilComments"}
NOISES = ["uniform20", "minority_high_40"]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]

# ── Method registry ─────────────────────────────────────────────────────────
LABELS = {
    "random": "Random", "loss": "Loss", "entropy": "Entropy",
    "forgetting": "Forgetting", "margin": "Margin", "aum": "AUM",
    "confident_learning": "Cleanlab", "coreset": "Coreset-Lite",
    "uncertainty_diversity": "Unc+Div-Lite", "badge_lite": "BADGE-Lite",
    "tracin_val_uncertainty": "TracIn-CP (val)",
    "tracin_noise_weighted": "TracIn-CP (noise)",
    "noise_score": "NoiseScore", "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc": "CPBA-TC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_tc_only": "TC-only",
    "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "oracle_noise_group_balanced": "Oracle-GB",
}

# All methods in display order
QUERY_ORDER = [
    "random", "loss", "entropy", "forgetting", "margin", "aum",
    "confident_learning", "coreset", "uncertainty_diversity", "badge_lite",
    "tracin_val_uncertainty", "tracin_noise_weighted",
    "noise_score", "noise_score_budget_hybrid", "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc", "noise_score_cpba_only", "noise_score_tc_only",
    "noise_score_cpba_tc_no_fallback", "noise_score_cpba_tc_v8_fallback",
    "oracle_noise_group_balanced",
]

ABLATION_ORDER = [
    "noise_score", "noise_score_cpba_only", "noise_score_tc_only",
    "noise_score_cpba_tc", "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced",
]

MODEL_ORDER = ["erm", "oracle_groupdro", "jtt"]
MODEL_LABELS = {"erm": "ERM", "oracle_groupdro": "Oracle GroupDRO", "jtt": "JTT"}


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def get_vals(ds_key, noise, budget, method):
    for src in [ABL, BAS]:
        sub = src[ds_key]
        sub = sub[
            (sub["noise_name"] == noise)
            & (sub["budget_fraction"] == budget)
            & (sub["method"] == method)
        ]
        if not sub.empty and (
            method not in QUERY_ORDER[:14] or sub["delta_wga"].abs().sum() > 1e-8
        ):
            return sub
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Full query-method comparison (per budget, per noise, per dataset)
# ═══════════════════════════════════════════════════════════════════════════════

rows_full = []
for ds_key in ["WB", "CE", "CC"]:
    for noise in NOISES:
        for budget in BUDGETS:
            for method in QUERY_ORDER:
                sub = get_vals(ds_key, noise, budget, method)
                if sub.empty:
                    continue
                vals = sub["delta_wga"].to_numpy()
                rows_full.append({
                    "dataset": DS_NAMES[ds_key],
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
                    "wga_mean": round(float(sub["wga"].mean()), 4),
                })

df_full = pd.DataFrame(rows_full)
df_full.to_csv(f"{OUT}/query_baselines_full.csv", index=False)
print(f"TABLE 1: query_baselines_full.csv  ({len(df_full)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Budget-pooled summary (avg over budgets, per noise, per dataset)
# ═══════════════════════════════════════════════════════════════════════════════

rows_summary = []
for ds_key in ["WB", "CE", "CC"]:
    for noise in NOISES:
        for method in QUERY_ORDER:
            all_vals = []
            for budget in BUDGETS:
                sub = get_vals(ds_key, noise, budget, method)
                if not sub.empty:
                    all_vals.extend(sub["delta_wga"].to_numpy().tolist())
                    all_vals_prec = sub["noise_precision"].to_numpy()
            if len(all_vals) == 0:
                continue
            all_vals = np.array(all_vals)
            rows_summary.append({
                "dataset": DS_NAMES[ds_key],
                "noise": noise,
                "method": method,
                "method_label": LABELS.get(method, method),
                "n": len(all_vals),
                "dWGA_mean": round(float(all_vals.mean()), 4),
                "dWGA_sem": round(sem(all_vals), 4),
                "dWGA_neg_pct": round(float((all_vals < 0).mean()) * 100, 1),
            })

df_summary = pd.DataFrame(rows_summary)
df_summary.to_csv(f"{OUT}/query_baselines_summary.csv", index=False)
print(f"TABLE 2: query_baselines_summary.csv  ({len(df_summary)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3: Model baselines
# ═══════════════════════════════════════════════════════════════════════════════

rows_model = []
for ds_key in ["WB", "CE", "CC"]:
    mod = MOD[ds_key]
    for method in MODEL_ORDER:
        sub = mod[mod["method"] == method]
        if sub.empty:
            continue
        for noise in NOISES:
            s2 = sub[sub["noise_name"] == noise]
            if s2.empty:
                continue
            wga_vals = s2["wga"].to_numpy()
            rows_model.append({
                "dataset": DS_NAMES[ds_key],
                "noise": noise,
                "method": method,
                "method_label": MODEL_LABELS.get(method, method),
                "n_seeds": len(wga_vals),
                "WGA_mean": round(float(wga_vals.mean()), 4),
                "WGA_sem": round(sem(wga_vals), 4),
                "WGA_min": round(float(wga_vals.min()), 4),
                "WGA_max": round(float(wga_vals.max()), 4),
            })

df_model = pd.DataFrame(rows_model)
df_model.to_csv(f"{OUT}/model_baselines.csv", index=False)
print(f"TABLE 3: model_baselines.csv  ({len(df_model)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4: Overall ranking (all budgets + all noises pooled, per dataset)
# ═══════════════════════════════════════════════════════════════════════════════

rows_overall = []
for method in QUERY_ORDER:
    lbl = LABELS.get(method, method)
    all_vals = []
    per_ds = {}
    for ds_key in ["WB", "CE", "CC"]:
        ds_vals = []
        for noise in NOISES:
            for budget in BUDGETS:
                sub = get_vals(ds_key, noise, budget, method)
                if not sub.empty:
                    dv = sub["delta_wga"].to_numpy()
                    ds_vals.extend(dv.tolist())
        if len(ds_vals) > 0:
            per_ds[DS_NAMES[ds_key]] = round(float(np.mean(ds_vals)), 4)
            all_vals.extend(ds_vals)
    if len(all_vals) == 0:
        continue
    all_vals = np.array(all_vals)
    rows_overall.append({
        "method": method,
        "method_label": lbl,
        **per_ds,
        "Overall": round(float(all_vals.mean()), 4),
        "n_total": len(all_vals),
    })

rows_overall.sort(key=lambda x: x["Overall"], reverse=True)
df_overall = pd.DataFrame(rows_overall)
df_overall.to_csv(f"{OUT}/query_baselines_overall.csv", index=False)
print(f"TABLE 4: query_baselines_overall.csv  ({len(df_overall)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5: Ablation component attribution (uniform20 b=5%, 2 datasets)
# ═══════════════════════════════════════════════════════════════════════════════

rows_ablation = []
for ds_key in ["WB", "CC"]:
    for method in ABLATION_ORDER:
        for src in [ABL, BAS]:
            sub = src[ds_key]
            sub = sub[
                (sub["noise_name"] == "uniform20")
                & (sub["budget_fraction"] == 0.05)
                & (sub["method"] == method)
            ]
            if not sub.empty:
                break
        if sub.empty:
            continue
        vals = sub["delta_wga"].to_numpy()
        rows_ablation.append({
            "dataset": DS_NAMES[ds_key],
            "noise": "uniform20",
            "budget": "5%",
            "method": method,
            "method_label": LABELS.get(method, method),
            "n_seeds": len(vals),
            "dWGA_mean": round(float(vals.mean()), 4),
            "dWGA_sem": round(sem(vals), 4),
            "dWGA_neg_pct": round(float((vals < 0).mean()) * 100, 1),
        })

df_ablation = pd.DataFrame(rows_ablation)
df_ablation.to_csv(f"{OUT}/ablation_attribution.csv", index=False)
print(f"TABLE 5: ablation_attribution.csv  ({len(df_ablation)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 6: Uniform20 b=5% stress test (all methods, 3 datasets)
# ═══════════════════════════════════════════════════════════════════════════════

rows_stress = []
for ds_key in ["WB", "CE", "CC"]:
    for method in QUERY_ORDER:
        sub = get_vals(ds_key, "uniform20", 0.05, method)
        if sub.empty:
            continue
        vals = sub["delta_wga"].to_numpy()
        prec = sub["noise_precision"].mean()
        rows_stress.append({
            "dataset": DS_NAMES[ds_key],
            "noise": "uniform20",
            "budget": "5%",
            "method": method,
            "method_label": LABELS.get(method, method),
            "n_seeds": len(vals),
            "dWGA_mean": round(float(vals.mean()), 4),
            "dWGA_sem": round(sem(vals), 4),
            "dWGA_min": round(float(vals.min()), 4),
            "dWGA_max": round(float(vals.max()), 4),
            "dWGA_neg_pct": round(float((vals < 0).mean()) * 100, 1),
            "noise_precision": round(float(prec), 4),
        })

df_stress = pd.DataFrame(rows_stress)
df_stress.to_csv(f"{OUT}/stress_test_uniform20_b5.csv", index=False)
print(f"TABLE 6: stress_test_uniform20_b5.csv  ({len(df_stress)} rows)")

print(f"\nAll CSVs saved to: {OUT}/")
