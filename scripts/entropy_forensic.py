#!/usr/bin/env python
"""Entropy forensic analysis: is entropy genuinely strong or lucky at 3 seeds?

Dissects:
1. Per-seed dWGA: entropy vs v7/v8 (identify outlier seeds)
2. Query overlap: entropy vs v7/v8 selected sets
3. Score correlation: entropy vs noise_score (Spearman)
4. Query composition: why entropy "works" (mechanism)
5. Variance analysis: CV and outlier contribution
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("outputs/baseline_test/stage1/results.csv")
SCORE_CORR = Path("outputs/baseline_test/stage1/score_correlations.csv")
QUERY_DIR = Path("outputs/baseline_test/stage1/queries")
MANIFEST_DIR = Path("outputs/baseline_test/manifests")
OUT_DIR = Path("outputs/baseline_test/stage1/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1 = "noise_score"
V7 = "noise_score_budget_hybrid"
V8 = "noise_score_budget_hybrid_v8"
ENT = "entropy"
ORACLE = "oracle_noise_group_balanced"

LABELS = {V1: "NoiseScore", V7: "BHC", V8: "NGC", ENT: "Entropy", ORACLE: "Oracle-GB"}


def load_query_indices(noise, seed, method, budget):
    """Load saved query indices (sample_ids) for a method at a budget."""
    prefix = f"{noise}_seed{seed}"
    path = QUERY_DIR / prefix / f"{method}_budget_{budget:.4f}.npy"
    if not path.exists():
        return None
    return np.load(path)


def main():
    df = pd.read_csv(RESULTS)
    noises = sorted(df["noise_name"].unique())
    budgets = sorted(df["budget_fraction"].unique())
    methods_focus = [ENT, V1, V7, V8, ORACLE]

    # ============================================================
    # 1. PER-SEED dWGA BREAKDOWN
    # ============================================================
    print("=" * 100)
    print("1. PER-SEED dWGA: entropy vs v1/v7/v8/oracle_gb")
    print("=" * 100)

    per_seed_rows = []
    for noise in noises:
        print(f"\n--- {noise} ---")
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            print(f"\n  budget={budget:.0%}:")
            header = f"    {'seed':>4s}"
            for m in methods_focus:
                header += f"  {LABELS[m]:>10s}"
            print(header)
            for seed in sorted(sub["seed"].unique()):
                line = f"    {seed:>4d}"
                for m in methods_focus:
                    val = sub[(sub["method"] == m) & (sub["seed"] == seed)]["delta_wga"]
                    v = val.values[0] if len(val) else float("nan")
                    line += f"  {v:>+10.4f}"
                    per_seed_rows.append({
                        "noise_name": noise, "budget_fraction": budget,
                        "seed": seed, "method": LABELS[m],
                        "delta_wga": v,
                    })
                print(line)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUT_DIR / "forensic_per_seed.csv", index=False)

    # ============================================================
    # 2. OUTLIER CONTRIBUTION ANALYSIS
    # ============================================================
    print("\n" + "=" * 100)
    print("2. OUTLIER CONTRIBUTION: how much does each seed pull the mean?")
    print("=" * 100)

    outlier_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            ent_vals = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            v8_vals = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            ent_mean = ent_vals.mean()
            v8_mean = v8_vals.mean()

            # Leave-one-out means
            for i, seed in enumerate(sorted(sub["seed"].unique())):
                ent_loo = np.delete(ent_vals, i).mean()
                v8_loo = np.delete(v8_vals, i).mean()
                outlier_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "seed": seed,
                    "entropy_dWGA": float(ent_vals[i]),
                    "v8_dWGA": float(v8_vals[i]),
                    "entropy_loo_mean": float(ent_loo),
                    "v8_loo_mean": float(v8_loo),
                    "entropy_pull": float(ent_vals[i] - ent_mean),
                    "v8_pull": float(v8_vals[i] - v8_mean),
                    "entropy_vs_v8_diff": float(ent_vals[i] - v8_vals[i]),
                })

            print(f"\n  {noise} @ {budget:.0%}:")
            print(f"    entropy mean={ent_mean:+.4f}  v8 mean={v8_mean:+.4f}  diff={ent_mean - v8_mean:+.4f}")
            for r in outlier_rows[-3:]:
                tag = "***OUTLIER***" if abs(r["entropy_pull"]) > 0.05 else ""
                print(f"    seed={r['seed']}: ent={r['entropy_dWGA']:+.4f} v8={r['v8_dWGA']:+.4f} "
                      f"ent_pull={r['entropy_pull']:+.4f} ent_vs_v8={r['entropy_vs_v8_diff']:+.4f} {tag}")

    outlier_df = pd.DataFrame(outlier_rows)
    outlier_df.to_csv(OUT_DIR / "forensic_outlier.csv", index=False)

    # ============================================================
    # 3. SCORE CORRELATION: entropy vs noise_score
    # ============================================================
    print("\n" + "=" * 100)
    print("3. SCORE CORRELATION: entropy vs noise_score (Spearman)")
    print("=" * 100)
    corr_df = pd.read_csv(SCORE_CORR)
    for _, r in corr_df.iterrows():
        print(f"  {r['noise_name']} seed={r['seed']}: "
              f"entropy~noise_score={r['noise_score']:.4f}  "
              f"entropy~loss={r['loss']:.4f}  "
              f"noise_score~loss={r['noise_score']:.4f}")

    # ============================================================
    # 4. QUERY OVERLAP: entropy vs v7/v8
    # ============================================================
    print("\n" + "=" * 100)
    print("4. QUERY OVERLAP: entropy vs v7/v8 (Jaccard of top-B sample sets)")
    print("=" * 100)

    overlap_rows = []
    for noise in noises:
        for seed in sorted(df[df["noise_name"] == noise]["seed"].unique()):
            for budget in budgets:
                ent_q = load_query_indices(noise, seed, ENT, budget)
                v1_q = load_query_indices(noise, seed, V1, budget)
                v7_q = load_query_indices(noise, seed, V7, budget)
                v8_q = load_query_indices(noise, seed, V8, budget)
                if ent_q is None or v1_q is None:
                    continue
                B = min(len(ent_q), len(v1_q))
                for ref, ref_q, ref_name in [(V1, v1_q, "v1"), (V7, v7_q, "v7"), (V8, v8_q, "v8")]:
                    ent_s = set(ent_q[:B].tolist())
                    ref_s = set(ref_q[:B].tolist())
                    jac = len(ent_s & ref_s) / len(ent_s | ref_s) if ent_s | ref_s else 0
                    only_ent = len(ent_s - ref_s)
                    only_ref = len(ref_s - ent_s)
                    overlap_rows.append({
                        "noise_name": noise, "seed": seed, "budget_fraction": budget,
                        "B": B, "ref": ref_name,
                        "jaccard": float(jac),
                        "overlap_count": len(ent_s & ref_s),
                        "only_entropy": only_ent,
                        "only_ref": only_ref,
                    })

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OUT_DIR / "forensic_query_overlap.csv", index=False)

    for noise in noises:
        print(f"\n  {noise}:")
        for budget in budgets:
            sub = overlap_df[(overlap_df["noise_name"] == noise) & (overlap_df["budget_fraction"] == budget)]
            for ref in ["v1", "v7", "v8"]:
                rsub = sub[sub["ref"] == ref]
                if rsub.empty:
                    continue
                print(f"    {budget:.0%} entropy vs {ref}: "
                      f"jaccard={rsub['jaccard'].mean():.3f} "
                      f"(overlap={rsub['overlap_count'].mean():.0f}/{rsub['B'].mean():.0f})")

    # ============================================================
    # 5. QUERY COMPOSITION: why entropy "works"
    # ============================================================
    print("\n" + "=" * 100)
    print("5. QUERY COMPOSITION: entropy vs v7/v8 mechanism comparison")
    print("=" * 100)

    comp_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            print(f"\n  {noise} @ {budget:.0%}:")
            print(f"    {'method':>10s}  {'dWGA':>8s}  {'noise_prec':>10s}  {'min_q_rate':>10s}  "
                  f"{'mislab_min':>10s}  {'mislab_maj':>10s}  {'grp_entropy':>11s}")
            for m in [ENT, V1, V7, V8, ORACLE]:
                msub = sub[sub["method"] == m]
                if msub.empty:
                    continue
                r = msub.iloc[0]
                print(f"    {LABELS[m]:>10s}  {r['delta_wga']:>+8.4f}  {r['noise_precision']:>10.3f}  "
                      f"{r['minority_query_rate']:>10.3f}  {r['mislabeled_minority']:>10.0f}  "
                      f"{r['mislabeled_majority']:>10.0f}  {r['group_query_entropy']:>11.3f}")
                comp_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "method": LABELS[m], "delta_wga": float(r["delta_wga"]),
                    "noise_precision": float(r["noise_precision"]),
                    "minority_query_rate": float(r["minority_query_rate"]),
                    "mislabeled_minority": int(r["mislabeled_minority"]),
                    "mislabeled_majority": int(r["mislabeled_majority"]),
                    "group_query_entropy": float(r["group_query_entropy"]),
                    "num_corrected": int(r["num_corrected"]),
                })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "forensic_composition.csv", index=False)

    # ============================================================
    # 6. VARIANCE & STATISTICAL SUMMARY
    # ============================================================
    print("\n" + "=" * 100)
    print("6. VARIANCE SUMMARY: CV, range, paired test")
    print("=" * 100)

    var_rows = []
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            ent_vals = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            v8_vals = sub[sub["method"] == V8]["delta_wga"].to_numpy()
            v1_vals = sub[sub["method"] == V1]["delta_wga"].to_numpy()

            for name, vals in [("entropy", ent_vals), ("v8", v8_vals), ("v1", v1_vals)]:
                var_rows.append({
                    "noise_name": noise, "budget_fraction": budget,
                    "method": name,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)),
                    "cv": float(vals.std(ddof=1) / abs(vals.mean())) if abs(vals.mean()) > 1e-8 else float("inf"),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "range": float(vals.max() - vals.min()),
                })

            # Paired test: entropy vs v8
            if len(ent_vals) == len(v8_vals) and len(ent_vals) >= 3:
                diff = ent_vals - v8_vals
                t_stat, p_val = stats.ttest_rel(ent_vals, v8_vals)
                w_stat, w_p = stats.wilcoxon(ent_vals, v8_vals) if len(ent_vals) >= 5 else (None, None)
                print(f"\n  {noise} @ {budget:.0%}: entropy vs v8")
                print(f"    mean diff = {diff.mean():+.4f}  (entropy {'>' if diff.mean() > 0 else '<'} v8)")
                print(f"    paired t: t={t_stat:.3f}  p={p_val:.4f}")
                if w_p is not None:
                    print(f"    Wilcoxon:  W={w_stat:.1f}  p={w_p:.4f}")
                print(f"    per-seed diff: {[f'{d:+.4f}' for d in diff]}")

    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(OUT_DIR / "forensic_variance.csv", index=False)

    # ============================================================
    # 7. VERDICT
    # ============================================================
    print("\n" + "=" * 100)
    print("7. FORENSIC VERDICT")
    print("=" * 100)

    # Count how many seeds entropy beats v8
    entropy_beats_v8 = 0
    total_cmp = 0
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            for seed in sorted(sub["seed"].unique()):
                ent_v = sub[(sub["method"] == ENT) & (sub["seed"] == seed)]["delta_wga"].values
                v8_v = sub[(sub["method"] == V8) & (sub["seed"] == seed)]["delta_wga"].values
                if len(ent_v) and len(v8_v):
                    total_cmp += 1
                    if ent_v[0] > v8_v[0]:
                        entropy_beats_v8 += 1

    print(f"  entropy beats v8: {entropy_beats_v8}/{total_cmp} per-seed comparisons")

    # Identify the outlier seed
    for noise in noises:
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            ent_vals = sub[sub["method"] == ENT]["delta_wga"].to_numpy()
            ent_mean = ent_vals.mean()
            for i, seed in enumerate(sorted(sub["seed"].unique())):
                pull = ent_vals[i] - ent_mean
                if abs(pull) > 0.05:
                    print(f"  OUTLIER: {noise} seed={seed} @ {budget:.0%}: "
                          f"entropy={ent_vals[i]:+.4f} (pull={pull:+.4f})")

    print(f"\n  CSV reports written to: {OUT_DIR}/forensic_*.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
