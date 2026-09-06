#!/usr/bin/env python
"""v9 multi-round vs v8 one-shot analysis (5 seeds).

Reports per-condition: dWGA mean±sem, neg%, noise_prec, min_q, num_corrected.
Paired tests: v9 vs v8, v9 vs v1, v9 vs oracle_gb.
Verdict: does v9 close the 5% oracle gap?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("outputs/v9_5seeds/stage1/results.csv")
OUT_DIR = Path("outputs/v9_5seeds/stage1/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1 = "noise_score"
V8 = "noise_score_budget_hybrid_v8"
V9 = "noise_score_multi_round"
ORACLE = "oracle_noise_group_balanced"
ORDER = [V1, V8, V9, ORACLE]
LABELS = {V1: "NoiseScore", V8: "NGC", V9: "CPBA-TC", ORACLE: "Oracle-GB"}


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    df = pd.read_csv(RESULTS)
    noises = sorted(df["noise_name"].unique())
    budgets = sorted(df["budget_fraction"].unique())

    # ---- Per-condition table ----
    print("=" * 115)
    print("v9 MULTI-ROUND vs v8 ONE-SHOT (5 seeds)")
    print("=" * 115)

    rows = []
    for noise in noises:
        print(f"\n--- {noise} ---")
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            print(f"\n  budget={budget:.0%}:")
            print(f"    {'method':>10s}  {'dWGA mean±sem':>18s}  {'neg%':>5s}  {'noise_prec':>10s}  "
                  f"{'min_q':>7s}  {'n_corr':>7s}  {'gap_oracle':>10s}")
            for m in ORDER:
                msub = sub[sub["method"] == m]
                if msub.empty:
                    continue
                dw = msub["delta_wga"].to_numpy()
                oracle_dw = sub[sub["method"] == ORACLE]["delta_wga"].mean()
                gap = oracle_dw - dw.mean()
                print(f"    {LABELS[m]:>10s}  {dw.mean():>+8.4f}±{sem(dw):.4f}  "
                      f"{(dw<0).mean()*100:>4.0f}%  {msub['noise_precision'].mean():>10.3f}  "
                      f"{msub['minority_query_rate'].mean():>7.3f}  "
                      f"{msub['num_corrected'].mean():>7.0f}  {gap:>+10.4f}")
                for seed in sorted(sub["seed"].unique()):
                    rows.append({
                        "noise_name": noise, "budget_fraction": budget,
                        "seed": seed, "method": LABELS[m],
                        "delta_wga": float(msub[msub["seed"] == seed]["delta_wga"].values[0]),
                        "noise_precision": float(msub[msub["seed"] == seed]["noise_precision"].values[0]),
                        "minority_query_rate": float(msub[msub["seed"] == seed]["minority_query_rate"].values[0]),
                        "num_corrected": int(msub[msub["seed"] == seed]["num_corrected"].values[0]),
                        "wga": float(msub[msub["seed"] == seed]["wga"].values[0]),
                    })

    per_seed_df = pd.DataFrame(rows)
    per_seed_df.to_csv(OUT_DIR / "v9_per_seed.csv", index=False)

    # ---- Paired tests ----
    print("\n" + "=" * 115)
    print("PAIRED TESTS: v9 vs v8, v9 vs v1")
    print("=" * 115)

    agg_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            v9_v = sub[sub["method"] == V9]["delta_wga"].to_numpy()
            v8_v = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            v1_v = sub[sub["method"] == V1]["delta_wga"].to_numpy()
            oracle_v = sub[sub["method"] == ORACLE]["delta_wga"].to_numpy()

            print(f"\n  {noise} @ {budget:.0%}:")
            for ref_v, ref_name in [(v8_v, "v8"), (v1_v, "v1"), (oracle_v, "oracle")]:
                diff = v9_v - ref_v
                t_stat, t_p = stats.ttest_rel(v9_v, ref_v)
                wins = int((diff > 0.0005).sum())
                losses = int((diff < -0.0005).sum())
                ties = len(diff) - wins - losses
                tag = ""
                if t_p < 0.05 and diff.mean() > 0:
                    tag = " *** v9 SIGNIFICANTLY BETTER"
                elif t_p < 0.05 and diff.mean() < 0:
                    tag = " *** v9 SIGNIFICANTLY WORSE"
                print(f"    v9 vs {ref_name}: diff={diff.mean():+.4f}±{sem(diff):.4f}  "
                      f"t={t_stat:.2f} p={t_p:.4f}  W/L/T={wins}/{losses}/{ties}{tag}")
                agg_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "comparison": f"v9_vs_{ref_name}",
                    "mean_diff": float(diff.mean()), "sem": sem(diff),
                    "t_stat": float(t_stat), "p_value": float(t_p),
                    "wins": wins, "losses": losses, "ties": ties,
                    "v9_mean": float(v9_v.mean()), "ref_mean": float(ref_v.mean()),
                })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "v9_paired_tests.csv", index=False)

    # ---- 5% budget focus (the key test) ----
    print("\n" + "=" * 115)
    print("5% BUDGET FOCUS: does v9 close the oracle gap?")
    print("=" * 115)
    for noise in noises:
        sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == 0.05)]
        v1_m = sub[sub["method"] == V1]["delta_wga"].mean()
        v8_m = sub[sub["method"] == V8]["delta_wga"].mean()
        v9_m = sub[sub["method"] == V9]["delta_wga"].mean()
        orc_m = sub[sub["method"] == ORACLE]["delta_wga"].mean()
        gap_v8 = orc_m - v8_m
        gap_v9 = orc_m - v9_m
        closed = (gap_v8 - gap_v9) / gap_v8 * 100 if gap_v8 > 0 else 0
        print(f"\n  {noise}:")
        print(f"    v1={v1_m:+.4f}  v8={v8_m:+.4f}  v9={v9_m:+.4f}  oracle={orc_m:+.4f}")
        print(f"    gap v8→oracle: {gap_v8:+.4f}  gap v9→oracle: {gap_v9:+.4f}  closed: {closed:.1f}%")
        print(f"    v9 vs v8: {v9_m - v8_m:+.4f}")

    # ---- Overall win rate ----
    print("\n" + "=" * 115)
    print("OVERALL WIN-RATE: v9 vs v8 across all 30 per-seed comparisons")
    print("=" * 115)
    wins = losses = ties = 0
    all_diffs = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            for seed in sorted(sub["seed"].unique()):
                v9_val = sub[(sub["method"] == V9) & (sub["seed"] == seed)]["delta_wga"].values[0]
                v8_val = sub[(sub["method"] == V8) & (sub["seed"] == seed)]["delta_wga"].values[0]
                d = v9_val - v8_val
                all_diffs.append(d)
                if d > 0.0005:
                    wins += 1
                elif d < -0.0005:
                    losses += 1
                else:
                    ties += 1
    all_diffs = np.array(all_diffs)
    t_stat, t_p = stats.ttest_rel(all_diffs, np.zeros_like(all_diffs))
    print(f"  v9 > v8: {wins}/30  |  v9 < v8: {losses}/30  |  tie: {ties}/30")
    print(f"  mean diff = {all_diffs.mean():+.4f} ± {sem(all_diffs):.4f}  t={t_stat:.2f}  p={t_p:.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 115)
    print("VERDICT")
    print("=" * 115)
    v9_5pct = df[(df["budget_fraction"] == 0.05) & (df["method"] == V9)]["delta_wga"].mean()
    v8_5pct = df[(df["budget_fraction"] == 0.05) & (df["method"] == V8)]["delta_wga"].mean()
    v9_1pct = df[(df["budget_fraction"] == 0.01) & (df["method"] == V9)]["delta_wga"].mean()
    v8_1pct = df[(df["budget_fraction"] == 0.01) & (df["method"] == V8)]["delta_wga"].mean()

    print(f"  v9 @5% mean dWGA = {v9_5pct:+.4f}  (v8 = {v8_5pct:+.4f}, diff = {v9_5pct - v8_5pct:+.4f})")
    print(f"  v9 @1% mean dWGA = {v9_1pct:+.4f}  (v8 = {v8_1pct:+.4f}, diff = {v9_1pct - v8_1pct:+.4f})")

    # Check: v9 @5% > +0.07 and significantly > v8
    target_5pct = 0.07
    passes_5pct = v9_5pct >= target_5pct
    # paired test at 5%
    sub5 = df[df["budget_fraction"] == 0.05]
    v9_5 = sub5[sub5["method"] == V9]["delta_wga"].to_numpy()
    v8_5 = sub5[sub5["method"] == V8]["delta_wga"].to_numpy()
    _, p5 = stats.ttest_rel(v9_5, v8_5)
    sig_5pct = p5 < 0.05 and v9_5pct > v8_5pct
    # Check: v9 @1% not worse than v8
    sub1 = df[df["budget_fraction"] == 0.01]
    v9_1 = sub1[sub1["method"] == V9]["delta_wga"].to_numpy()
    v8_1 = sub1[sub1["method"] == V8]["delta_wga"].to_numpy()
    _, p1 = stats.ttest_rel(v9_1, v8_1)
    no_degrade_1pct = v9_1pct >= v8_1pct - 0.005

    print(f"\n  Criteria:")
    print(f"    v9 @5% >= +0.07: {'PASS' if passes_5pct else 'FAIL'} ({v9_5pct:+.4f})")
    print(f"    v9 @5% sig > v8: {'PASS' if sig_5pct else 'FAIL'} (p={p5:.4f})")
    print(f"    v9 @1% not worse: {'PASS' if no_degrade_1pct else 'FAIL'} (diff={v9_1pct - v8_1pct:+.4f}, p={p1:.4f})")

    if passes_5pct and sig_5pct and no_degrade_1pct:
        print("\n  => GO: v9 multi-round is a viable method contribution.")
    elif sig_5pct and no_degrade_1pct:
        print(f"\n  => PARTIAL: v9 sig > v8 @5% but doesn't reach +0.07 target ({v9_5pct:+.4f}).")
    else:
        print("\n  => NO-GO: v9 does not significantly improve over v8.")

    print(f"\n  CSV reports: {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"    {p.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
