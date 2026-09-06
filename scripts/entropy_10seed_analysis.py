#!/usr/bin/env python
"""10-seed entropy vs v7/v8 statistical analysis.

Paired tests, per-seed breakdown, win-rate, leave-one-out, and verdict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("outputs/entropy_forensic_10seeds/stage1/results.csv")
OUT_DIR = Path("outputs/entropy_forensic_10seeds/stage1/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1 = "noise_score"
V7 = "noise_score_budget_hybrid"
V8 = "noise_score_budget_hybrid_v8"
ENT = "entropy"
ORACLE = "oracle_noise_group_balanced"

LABELS = {V1: "NoiseScore", V7: "BHC", V8: "NGC", ENT: "Entropy", ORACLE: "Oracle-GB"}
ORDER = [V1, V7, V8, ENT, ORACLE]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    df = pd.read_csv(RESULTS)
    noises = sorted(df["noise_name"].unique())
    budgets = sorted(df["budget_fraction"].unique())

    # ============================================================
    # 1. PER-SEED TABLE
    # ============================================================
    print("=" * 110)
    print("1. PER-SEED dWGA (10 seeds)")
    print("=" * 110)

    per_seed_rows = []
    for noise in noises:
        print(f"\n--- {noise} ---")
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            print(f"\n  budget={budget:.0%}:")
            header = f"    {'seed':>4s}"
            for m in ORDER:
                header += f"  {LABELS[m]:>10s}"
            header += f"  {'ent-v8':>8s}"
            print(header)
            for seed in sorted(sub["seed"].unique()):
                line = f"    {seed:>4d}"
                vals = {}
                for m in ORDER:
                    v = sub[(sub["method"] == m) & (sub["seed"] == seed)]["delta_wga"]
                    val = v.values[0] if len(v) else float("nan")
                    vals[m] = val
                    line += f"  {val:>+10.4f}"
                    per_seed_rows.append({
                        "noise_name": noise, "budget_fraction": budget,
                        "seed": seed, "method": LABELS[m], "delta_wga": val,
                    })
                diff = vals.get(ENT, 0) - vals.get(V8, 0)
                line += f"  {diff:>+8.4f}"
                print(line)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUT_DIR / "per_seed_10seeds.csv", index=False)

    # ============================================================
    # 2. AGGREGATE STATISTICS + PAIRED TESTS
    # ============================================================
    print("\n" + "=" * 110)
    print("2. AGGREGATE STATISTICS + PAIRED TESTS (10 seeds)")
    print("=" * 110)

    agg_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            print(f"\n  {noise} @ {budget:.0%}:")
            print(f"    {'method':>10s}  {'mean±sem':>16s}  {'min':>8s}  {'max':>8s}  {'range':>8s}  {'neg%':>5s}  {'CV':>6s}")

            method_vals = {}
            for m in ORDER:
                vals = sub[sub["method"] == m]["delta_wga"].to_numpy()
                method_vals[m] = vals
                cv = float(vals.std(ddof=1) / abs(vals.mean())) if abs(vals.mean()) > 1e-8 else float("inf")
                neg = float((vals < 0).mean())
                print(f"    {LABELS[m]:>10s}  {vals.mean():>+8.4f}±{sem(vals):.4f}  "
                      f"{vals.min():>+8.4f}  {vals.max():>+8.4f}  {vals.max()-vals.min():>8.4f}  "
                      f"{neg*100:>4.0f}%  {cv:>6.2f}")
                agg_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "method": LABELS[m],
                    "mean": float(vals.mean()), "sem": sem(vals),
                    "min": float(vals.min()), "max": float(vals.max()),
                    "range": float(vals.max() - vals.min()),
                    "neg_fraction": neg, "cv": cv,
                })

            # Paired tests: entropy vs v8, entropy vs v1, entropy vs v7
            for ref, ref_name in [(V8, "v8"), (V1, "v1"), (V7, "v7")]:
                ent_v = method_vals[ENT]
                ref_v = method_vals[ref]
                diff = ent_v - ref_v
                t_stat, t_p = stats.ttest_rel(ent_v, ref_v)
                try:
                    w_stat, w_p = stats.wilcoxon(ent_v, ref_v)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                wins = int((diff > 0).sum())
                ties = int((diff == 0).sum())
                losses = int((diff < 0).sum())
                print(f"    entropy vs {ref_name}: mean_diff={diff.mean():+.4f}  "
                      f"t={t_stat:.2f} p={t_p:.4f}  W={w_stat:.1f} p_w={w_p:.4f}  "
                      f"win/tie/loss={wins}/{ties}/{losses}")
                agg_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "method": f"entropy_vs_{ref_name}",
                    "mean": float(diff.mean()), "sem": sem(diff),
                    "min": float(diff.min()), "max": float(diff.max()),
                    "range": float(diff.max() - diff.min()),
                    "neg_fraction": float((diff < 0).mean()),
                    "cv": float(diff.std(ddof=1) / abs(diff.mean())) if abs(diff.mean()) > 1e-8 else float("inf"),
                    "t_stat": float(t_stat), "t_p": float(t_p),
                    "wilcoxon_w": float(w_stat), "wilcoxon_p": float(w_p),
                    "wins": wins, "ties": ties, "losses": losses,
                })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "aggregate_10seeds.csv", index=False)

    # ============================================================
    # 3. LEAVE-ONE-OUT: does any single seed flip the verdict?
    # ============================================================
    print("\n" + "=" * 110)
    print("3. LEAVE-ONE-OUT: entropy mean vs v8 mean (removing each seed)")
    print("=" * 110)

    loo_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            ent_all = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            v8_all = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            seeds = sorted(sub["seed"].unique())

            print(f"\n  {noise} @ {budget:.0%}: full mean ent={ent_all.mean():+.4f} v8={v8_all.mean():+.4f} diff={ent_all.mean()-v8_all.mean():+.4f}")
            for i, seed in enumerate(seeds):
                ent_loo = np.delete(ent_all, i).mean()
                v8_loo = np.delete(v8_all, i).mean()
                diff_loo = ent_loo - v8_loo
                flip = "FLIP!" if (ent_all.mean() > v8_all.mean()) != (ent_loo > v8_loo) else ""
                print(f"    remove seed={seed}: ent_loo={ent_loo:+.4f} v8_loo={v8_loo:+.4f} "
                      f"diff={diff_loo:+.4f} {flip}")
                loo_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "removed_seed": seed,
                    "entropy_loo_mean": float(ent_loo),
                    "v8_loo_mean": float(v8_loo),
                    "diff_loo": float(diff_loo),
                    "full_diff": float(ent_all.mean() - v8_all.mean()),
                    "flips_verdict": flip == "FLIP!",
                })

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(OUT_DIR / "leave_one_out_10seeds.csv", index=False)

    # ============================================================
    # 4. OVERALL WIN-RATE MATRIX
    # ============================================================
    print("\n" + "=" * 110)
    print("4. OVERALL WIN-RATE: entropy vs v8 across all 60 per-seed comparisons")
    print("=" * 110)

    total_wins = total_ties = total_losses = 0
    all_diffs = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            for seed in sorted(sub["seed"].unique()):
                ent_v = sub[(sub["method"] == ENT) & (sub["seed"] == seed)]["delta_wga"].values[0]
                v8_v = sub[(sub["method"] == V8) & (sub["seed"] == seed)]["delta_wga"].values[0]
                d = ent_v - v8_v
                all_diffs.append(d)
                if d > 0.0005:
                    total_wins += 1
                elif d < -0.0005:
                    total_losses += 1
                else:
                    total_ties += 1

    all_diffs = np.array(all_diffs)
    print(f"  entropy > v8: {total_wins}/60")
    print(f"  entropy ≈ v8: {total_ties}/60")
    print(f"  entropy < v8: {total_losses}/60")
    print(f"  mean diff = {all_diffs.mean():+.4f}  sem = {sem(all_diffs):.4f}")
    t_stat, t_p = stats.ttest_rel(all_diffs, np.zeros_like(all_diffs))
    print(f"  one-sample t-test (diff=0): t={t_stat:.2f}  p={t_p:.4f}")

    # ============================================================
    # 5. PER-BUDGET SUMMARY
    # ============================================================
    print("\n" + "=" * 110)
    print("5. PER-BUDGET SUMMARY (averaged across noises and 10 seeds)")
    print("=" * 110)
    print(f"  {'budget':>8s}  {'v1':>12s}  {'v7':>12s}  {'v8':>12s}  {'entropy':>12s}  {'oracle':>12s}  {'ent-v8':>8s}  {'p':>6s}")

    for budget in budgets:
        sub = df[df["budget_fraction"] == budget]
        means = {}
        for m in ORDER:
            means[m] = sub[sub["method"] == m]["delta_wga"].mean()
        ent_vals = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
        v8_vals = sub[sub["method"] == V8]["delta_wga"].to_numpy()
        # Paired across both noises × 10 seeds = 20 pairs
        t_stat, t_p = stats.ttest_rel(ent_vals, v8_vals)
        diff = means[ENT] - means[V8]
        print(f"  {budget:>7.0%}  {means[V1]:>+10.4f}±{sem(sub[sub['method']==V1]['delta_wga'].to_numpy()):.3f}  "
              f"{means[V7]:>+10.4f}±{sem(sub[sub['method']==V7]['delta_wga'].to_numpy()):.3f}  "
              f"{means[V8]:>+10.4f}±{sem(sub[sub['method']==V8]['delta_wga'].to_numpy()):.3f}  "
              f"{means[ENT]:>+10.4f}±{sem(ent_vals):.3f}  "
              f"{means[ORACLE]:>+10.4f}±{sem(sub[sub['method']==ORACLE]['delta_wga'].to_numpy()):.3f}  "
              f"{diff:>+8.4f}  {t_p:>6.4f}")

    # ============================================================
    # 6. VERDICT
    # ============================================================
    print("\n" + "=" * 110)
    print("6. FINAL VERDICT")
    print("=" * 110)

    # Test: is entropy significantly better than v8 at ANY condition?
    any_significant = False
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            ent_v = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            v8_v = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            t_stat, t_p = stats.ttest_rel(ent_v, v8_v)
            if t_p < 0.05 and ent_v.mean() > v8_v.mean():
                any_significant = True
                print(f"  entropy SIGNIFICANTLY > v8 at {noise} @ {budget:.0%} (p={t_p:.4f})")

    # Check: is v8 significantly better than entropy anywhere?
    v8_significant = False
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            ent_v = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            v8_v = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            t_stat, t_p = stats.ttest_rel(ent_v, v8_v)
            if t_p < 0.05 and v8_v.mean() > ent_v.mean():
                v8_significant = True
                print(f"  v8 SIGNIFICANTLY > entropy at {noise} @ {budget:.0%} (p={t_p:.4f})")

    if not any_significant and not v8_significant:
        print("  No significant difference between entropy and v8 at any condition (p > 0.05).")
        print("  -> entropy is statistically TIED with v8, not genuinely stronger.")
    elif any_significant and not v8_significant:
        print("  -> entropy is significantly better than v8 at some conditions.")
    elif not any_significant and v8_significant:
        print("  -> v8 is significantly better than entropy at some conditions.")
    else:
        print("  -> mixed: each wins at different conditions.")

    # v-line internal comparison
    print("\n  v-line internal (v7/v8 vs v1):")
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            d1 = sub[sub["method"] == V1]["delta_wga"].mean()
            d7 = sub[sub["method"] == V7]["delta_wga"].mean()
            d8 = sub[sub["method"] == V8]["delta_wga"].mean()
            v1_vals = sub[sub["method"] == V1]["delta_wga"].to_numpy()
            v8_vals = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            _, p = stats.ttest_rel(v8_vals, v1_vals)
            print(f"    {noise}@{budget:.0%}: v1={d1:+.4f} v7={d7:+.4f} v8={d8:+.4f} "
                  f"v8-v1={d8-d1:+.4f} p={p:.4f}")

    print(f"\n  CSV reports: {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"    {p.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
