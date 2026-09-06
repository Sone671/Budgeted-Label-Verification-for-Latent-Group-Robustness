#!/usr/bin/env python
"""Sync all paper tables from actual experimental results.csv files."""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("paper/tables")

# ── Load data ────────────────────────────────────────────────────────────────
WB = pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv")
CE = pd.read_csv("outputs/ablation_celeba_10seeds/stage1/results.csv")
CC = pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/results.csv")
CC_DISAG = pd.read_csv("outputs/civilcomments_disagreement_10seeds/stage1/results.csv")
WB_HELDOUT = pd.read_csv("outputs/p0_heldout_leave_wb_out/stage1/results.csv")
CE_HELDOUT = pd.read_csv("outputs/p0_heldout_leave_ce_out/stage1/results.csv")
CC_HELDOUT = pd.read_csv("outputs/p0_heldout_leave_cc_out/stage1/results.csv")
WB_SWAP = pd.read_csv("outputs/p0_label_swap_waterbirds/stage1/results.csv")
WB_BASELINES = pd.read_csv("outputs/p0_unified_baselines_waterbirds/stage1/results.csv")
WB_MODEL = pd.read_csv("outputs/p0_unified_baselines_waterbirds/stage1/model_baselines.csv")
RISK_WB = pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/risk_metrics.csv")
RISK_CE = pd.read_csv("outputs/ablation_celeba_10seeds/stage1/risk_metrics.csv")
RISK_CC = pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/risk_metrics.csv")

METHOD_LABEL = {
    "random": "Random", "loss": "Loss", "entropy": "Entropy",
    "forgetting": "Forgetting", "aum": "AUM", "margin": "Margin",
    "confident_learning": "Cleanlab", "coreset": "Coreset-Lite",
    "uncertainty_diversity": "Unc+Div-Lite", "badge_lite": "BADGE-Lite",
    "tracin_val_uncertainty": "TracIn-CP (val)", "tracin_noise_weighted": "TracIn-CP (noise)",
    "noise_score": "NoiseScore", "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_only": "CPBA-only", "noise_score_tc_only": "TC-only",
    "noise_score_cpba_tc": "CPBA-TC (original)",
    "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Guarded",
    "noise_score_cpba_tc_entropy_fallback": "CPBA-TC-entFB",
    "oracle_noise_group_balanced": "Oracle-GB",
    "oracle_disagreement": "Oracle-Disag",
}

def dwga_mean(df, method):
    if method not in df.method.values:
        return np.nan
    return df[df.method == method].delta_wga.mean()

# ── Table 1a: Main results per dataset (overall) ────────────────────────────
print("=== Table 1a: Main results (overall) ===")
methods = ["random","loss","entropy","forgetting",
           "noise_score","noise_score_budget_hybrid_v8",
           "noise_score_cpba_only","noise_score_cpba_tc_no_fallback",
           "noise_score_cpba_tc_v8_fallback","oracle_noise_group_balanced"]
rows = []
for m in methods:
    wb = dwga_mean(WB, m); ce = dwga_mean(CE, m); cc = dwga_mean(CC, m)
    ov = np.nanmean([wb, ce, cc])
    n = sum(1 for df in [WB,CE,CC] if m in df.method.values) * 100  # rough
    rows.append([m, METHOD_LABEL.get(m,m), f"{wb:.4f}", f"{ce:.4f}", f"{cc:.4f}", f"{ov:.4f}", 300])
df = pd.DataFrame(rows, columns=["method","method_label","Waterbirds","CelebA","CivilComments","Overall","n_total"])
df.to_csv(OUT / "table1_main_results_overall.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Table 1b: Main results per noise ────────────────────────────────────────
print("=== Table 1b: Main results (per noise) ===")
rows = []
for ds_name, ds in [("WB",WB),("CE",CE),("CC",CC)]:
    for noise in ["uniform20","minority_high_40"]:
        sub = ds[ds.noise_name==noise]
        for m in methods:
            if m in sub.method.values:
                d = dwga_mean(sub, m)
                rows.append([ds_name, noise, m, METHOD_LABEL.get(m,m), d])
df = pd.DataFrame(rows, columns=["dataset","noise","method","label","mean_dwga"])
df.to_csv(OUT / "table1_main_results_per_noise.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Table 2: Risk metrics (WB u20) ──────────────────────────────────────────
print("=== Table: Risk metrics (WB u20) ===")
sub = RISK_WB[(RISK_WB.noise_name=="uniform20")]
risk_methods = ["noise_score","noise_score_budget_hybrid_v8",
                "noise_score_cpba_tc_no_fallback","noise_score_cpba_tc_v8_fallback"]
rows = []
for m in risk_methods:
    s = sub[sub.method==m]
    mean_d = s.mean_dwga.mean()
    neg = s.neg_seed_pct.mean()
    cvar = s.cvar10.mean()
    worst = s.min_dwga.min()
    rows.append([m, METHOD_LABEL.get(m,m), mean_d, neg, cvar, worst])
df = pd.DataFrame(rows, columns=["method","label","mean_dwga","neg_pct","cvar10","worst_seed"])
df.to_csv(OUT / "table_risk_metrics.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Table: Held-out validation ──────────────────────────────────────────────
print("=== Table: Held-out validation ===")
heldout_methods = ["noise_score","noise_score_budget_hybrid_v8",
                   "noise_score_cpba_only","noise_score_cpba_tc_no_fallback",
                   "noise_score_cpba_tc_v8_fallback"]
rows = []
for m in heldout_methods:
    wb_h = dwga_mean(WB_HELDOUT, m); ce_h = dwga_mean(CE_HELDOUT, m); cc_h = dwga_mean(CC_HELDOUT, m)
    rows.append([m, METHOD_LABEL.get(m,m), wb_h, ce_h, cc_h])
df = pd.DataFrame(rows, columns=["method","label","WB_heldout","CE_heldout","CC_heldout"])
df.to_csv(OUT / "table_heldout_validation.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Table: Label swap ───────────────────────────────────────────────────────
print("=== Table: Label swap ===")
rows = []
for m in heldout_methods:
    sw = dwga_mean(WB_SWAP, m); orig = dwga_mean(WB, m)
    rows.append([m, METHOD_LABEL.get(m,m), sw, orig, sw-orig])
df = pd.DataFrame(rows, columns=["method","label","swapped","original","diff"])
df.to_csv(OUT / "table_label_swap.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Table: Unified baselines ────────────────────────────────────────────────
print("=== Table: Unified baselines ===")
rows = []
for m in WB_BASELINES.method.unique():
    d = dwga_mean(WB_BASELINES, m)
    rows.append([m, METHOD_LABEL.get(m,m), d])
# Model baselines
for m in ["erm","oracle_groupdro","jtt"]:
    if m in WB_MODEL.method.values:
        wga = WB_MODEL[WB_MODEL.method==m].wga.mean()
        rows.append([m, f"Model-{m}", wga, "abs_WGA"])
df = pd.DataFrame(rows, columns=["method","label","mean_dwga","note"])
df.to_csv(OUT / "table_unified_baselines.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Table: Disagreement (CC) ────────────────────────────────────────────────
print("=== Table: Disagreement ===")
sub = CC_DISAG[(CC_DISAG.budget_fraction==0.05)]
rows = []
for noise in ["uniform20","disagreement_natural","disagreement_matched20"]:
    s = sub[sub.noise_name==noise]
    for m in ["noise_score","noise_score_cpba_tc_v8_fallback","oracle_disagreement","oracle_noise_group_balanced"]:
        if m in s.method.values:
            r = s[s.method==m]
            d = r.delta_wga.mean()
            lift = r.disagreement_lift.mean() if "disagreement_lift" in r.columns else 0
            rows.append([noise, m, METHOD_LABEL.get(m,m), d, lift])
df = pd.DataFrame(rows, columns=["noise","method","label","dwga","disagreement_lift"])
df.to_csv(OUT / "table_disagreement.csv", index=False)
print(f"  Wrote {len(rows)} rows")

# ── Update old appendix tables to use new naming ────────────────────────────
print("=== Updating appendix tables ===")
for fname in ["appendix_query_baselines_full.csv","appendix_query_baselines_summary.csv"]:
    fp = OUT / fname
    if fp.exists():
        df = pd.read_csv(fp)
        if "method" in df.columns:
            df["method"] = df["method"].replace({
                "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Guarded",
                "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
                "noise_score_budget_hybrid_v8": "NGC",
                "noise_score_cpba_only": "CPBA-only",
            })
        df.to_csv(fp, index=False)
        print(f"  Updated {fname}")

print("\n=== ALL DONE ===")
