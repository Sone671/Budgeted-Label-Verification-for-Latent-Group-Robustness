#!/usr/bin/env python
"""Real-noise results analysis for CivilComments (with disagreement metrics)."""
import pandas as pd, numpy as np

df = pd.read_csv("outputs/civilcomments_disagreement_10seeds/stage1/results.csv")

labels = {
    "noise_score": "NoiseScore", "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_only": "CPBA-only", "noise_score_cpba_tc_v8_fallback": "Safe",
    "oracle_noise_group_balanced": "Oracle-GB", "random": "Random", "loss": "Loss",
}

methods = ["random", "loss", "noise_score", "noise_score_budget_hybrid_v8",
           "noise_score_cpba_only", "noise_score_cpba_tc_v8_fallback",
           "oracle_noise_group_balanced"]
noises = ["uniform20", "disagreement_natural", "disagreement_matched20"]
budgets = [0.01, 0.05, 0.10]

for noise in noises:
    print("=" * 90)
    print("  " + noise)
    print("=" * 90)
    hdr = "  {:<15s}".format("Method")
    for b in budgets:
        hdr += " | b={:.0%}: {:.0s} {:.0s} {:.0s}".format(b, "", "", "").replace("||", "dWGA prec disL")
    print(hdr)
    print("  {:<15s}".format("") + " |" + " ---- ---- ---- |" * 3)

    for m in methods:
        sub = df[(df["noise_name"] == noise) & (df["method"] == m)]
        if sub.empty: continue
        lbl = labels.get(m, m)
        row_parts = [lbl]
        for b in budgets:
            s = sub[sub["budget_fraction"] == b]
            dw = s["delta_wga"].mean()
            prec = s["noise_precision"].mean()
            dl = s["disagreement_lift"].mean() if "disagreement_lift" in s.columns else 0
            row_parts.append(" {:>+7.4f} {:.3f} {:.2f}".format(dw, prec, dl))
        print("  {:<15s}".format(lbl) + " |".join(row_parts[1:]))

# Noise summary
print()
print("=" * 90)
print("  Noise statistics")
print("=" * 90)
ns = pd.read_csv("outputs/civilcomments_disagreement_10seeds/noise_statistics.csv")
for noise in noises:
    sub = ns[ns["noise_name"] == noise]
    print("  {:25s} overall={:.3f}  minority={:.3f}  majority={:.3f}".format(
        noise, sub["actual_noise_rate"].mean(),
        sub["minority_noise_rate"].mean(), sub["majority_noise_rate"].mean()))

# Disagreement lift analysis
print()
print("=" * 90)
print("  Disagreement lift: does the method query high-disagreement samples?")
print("=" * 90)
for noise in noises:
    print("  --- " + noise + " ---")
    for m in ["loss", "noise_score", "noise_score_cpba_only", "noise_score_cpba_tc_v8_fallback"]:
        sub = df[(df["noise_name"] == noise) & (df["method"] == m)]
        if sub.empty: continue
        for b in [0.05]:
            s = sub[sub["budget_fraction"] == b]
            dl = s["disagreement_lift"].mean()
            high = s["queried_high_disagreement_rate"].mean()
            print("    {:15s} b=5%: lift={:.3f}  high_disag_rate={:.3f}".format(
                labels.get(m, m), dl, high))
