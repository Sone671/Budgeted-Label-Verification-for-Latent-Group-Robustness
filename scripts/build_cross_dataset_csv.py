#!/usr/bin/env python
"""Build comprehensive cross-dataset comparison CSV: Waterbirds vs CelebA, all methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("outputs/cross_dataset_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load sources ──────────────────────────────────────────────────────────

# Waterbirds 5-seed noise sweep (has v1+v7+v8 with noise names matching CelebA)
wb_v8 = pd.read_csv("outputs/waterbirds_mvp_noise_sweep_v8/stage1/results.csv")
# Waterbirds 5-seed v7 (has entropy, forgetting, etc.)
wb_v7_5seed = pd.read_csv("outputs/waterbirds_mvp_5seeds_v7/stage1/results.csv")
# Waterbirds 5-seed original (has oracle_noise_loss_tiebreak, oracle_noise_random_tiebreak, etc.)
wb_v1_5seed = pd.read_csv("outputs/waterbirds_mvp_5seeds/stage1/results.csv")
# Waterbirds 10-seed v7 (more seeds for statistical power)
wb_v7_10seed = pd.read_csv("outputs/waterbirds_mvp_10seeds_v7/stage1/results.csv")
# Waterbirds noise sweep v1 (10 seeds)
wb_v1_10seed_ns = pd.read_csv("outputs/waterbirds_mvp_10seeds_noise_sweep_v1/stage1/results.csv")
# CelebA 5-seed
celeba = pd.read_csv("outputs/celeba_5seeds/stage1/results.csv")

# All sources now use unified noise names (minority_high_40)
# No rename needed

# Add dataset tag
wb_v8["dataset"] = "waterbirds"
wb_v7_5seed["dataset"] = "waterbirds"
wb_v1_5seed["dataset"] = "waterbirds"
wb_v7_10seed["dataset"] = "waterbirds"
wb_v1_10seed_ns["dataset"] = "waterbirds"
celeba["dataset"] = "celeba"

# ── Methods to keep ───────────────────────────────────────────────────────

METHODS = [
    "random",
    "loss",
    "entropy",
    "noise_score",
    "noise_score_budget_hybrid",
    "noise_score_budget_hybrid_v8",
    "oracle_noise_loss_tiebreak",
    "oracle_noise_group_balanced",
]

METHOD_LABELS = {
    "random":                           "Random",
    "loss":                             "Loss",
    "entropy":                          "Entropy",
    "noise_score":                      "NoiseScore",
    "noise_score_budget_hybrid":        "BHC",
    "noise_score_budget_hybrid_v8":     "NGC",
    "oracle_noise_loss_tiebreak":       "Oracle-noise / loss-tiebreak",
    "oracle_noise_group_balanced":      "Oracle-GB",
}

# Common noise settings between datasets
COMMON_NOISES = ["uniform20", "minority_high_40"]


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def extract_from_source(
    df: pd.DataFrame,
    noise: str,
    budget: float,
    method: str,
    *,
    n_seeds: int | None = None,
) -> dict | None:
    """Extract method stats for a given (noise, budget, method) from a source."""
    sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget) & (df["method"] == method)]
    if sub.empty:
        return None
    vals = sub["delta_wga"].to_numpy()
    if n_seeds is not None and len(vals) != n_seeds:
        return None
    return {
        "delta_wga_mean": float(vals.mean()),
        "delta_wga_sem": sem(vals),
        "delta_wga_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "delta_wga_min": float(vals.min()),
        "delta_wga_max": float(vals.max()),
        "wga_mean": float(sub["wga"].mean()),
        "wga_sem": sem(sub["wga"].to_numpy()),
        "noise_precision": float(sub["noise_precision"].mean()),
        "minority_query_rate": float(sub["minority_query_rate"].mean()),
        "num_corrected_mean": float(sub["num_corrected"].mean()),
        "num_corrected_sem": sem(sub["num_corrected"].to_numpy()),
        "neg_fraction": float((vals < 0).mean()),
        "n_seeds": len(vals),
    }


# ── Build master comparison table ─────────────────────────────────────────

BUDGETS = [0.01, 0.02, 0.05, 0.10]
NOISES = ["uniform20", "minority_high_40"]

# Source priority per method: (source_df, n_seeds_expected)
# We prefer 10-seed sources where available, fall back to 5-seed
SOURCES = {
    "random":      [(wb_v1_5seed, 5), (wb_v7_10seed, 10), (celeba, 5)],
    "loss":        [(wb_v8, 5), (wb_v7_10seed, 10), (celeba, 5)],
    "entropy":     [(wb_v7_10seed, 10), (wb_v1_5seed, 5), (celeba, 5)],
    "noise_score": [(wb_v8, 5), (wb_v7_10seed, 10), (celeba, 5)],
    "noise_score_budget_hybrid":      [(wb_v8, 5), (wb_v7_10seed, 10), (celeba, 5)],
    "noise_score_budget_hybrid_v8":   [(wb_v8, 5), (celeba, 5)],
    "oracle_noise_loss_tiebreak":     [(wb_v1_5seed, 5), (wb_v8, 5)],
    "oracle_noise_group_balanced":    [(wb_v8, 5), (wb_v1_5seed, 5), (celeba, 5)],
}

rows = []
for noise in NOISES:
    for budget in BUDGETS:
        for method in METHODS:
            for source_df, expected_seeds in SOURCES.get(method, []):
                row = extract_from_source(source_df, noise, budget, method, n_seeds=expected_seeds)
                if row is not None:
                    # Also try CelebA
                    celeba_row = extract_from_source(celeba, noise, budget, method, n_seeds=5)
                    wb_row = row

                    combined = {
                        "noise_name": noise,
                        "budget": f"{budget:.0%}",
                        "budget_fraction": budget,
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "wb_n_seeds": wb_row["n_seeds"],
                        "wb_delta_wga_mean": wb_row["delta_wga_mean"],
                        "wb_delta_wga_sem": wb_row["delta_wga_sem"],
                        "wb_delta_wga_min": wb_row["delta_wga_min"],
                        "wb_delta_wga_max": wb_row["delta_wga_max"],
                        "wb_wga_mean": wb_row["wga_mean"],
                        "wb_noise_precision": wb_row["noise_precision"],
                        "wb_minority_query_rate": wb_row["minority_query_rate"],
                        "wb_num_corrected_mean": wb_row["num_corrected_mean"],
                        "wb_neg_fraction": wb_row["neg_fraction"],
                    }
                    if celeba_row is not None:
                        combined.update({
                            "ce_n_seeds": celeba_row["n_seeds"],
                            "ce_delta_wga_mean": celeba_row["delta_wga_mean"],
                            "ce_delta_wga_sem": celeba_row["delta_wga_sem"],
                            "ce_delta_wga_min": celeba_row["delta_wga_min"],
                            "ce_delta_wga_max": celeba_row["delta_wga_max"],
                            "ce_wga_mean": celeba_row["wga_mean"],
                            "ce_noise_precision": celeba_row["noise_precision"],
                            "ce_minority_query_rate": celeba_row["minority_query_rate"],
                            "ce_num_corrected_mean": celeba_row["num_corrected_mean"],
                            "ce_neg_fraction": celeba_row["neg_fraction"],
                            "diff_wb_minus_ce": wb_row["delta_wga_mean"] - celeba_row["delta_wga_mean"],
                        })

                        # Paired test across datasets (unpaired, different samples)
                        # Just report the difference
                        combined["cross_dataset_sign_match"] = (
                            (wb_row["delta_wga_mean"] > 0) == (celeba_row["delta_wga_mean"] > 0)
                        )
                    else:
                        combined["ce_n_seeds"] = 0
                        for k in ["ce_delta_wga_mean", "ce_delta_wga_sem", "ce_delta_wga_min",
                                   "ce_delta_wga_max", "ce_wga_mean", "ce_noise_precision",
                                   "ce_minority_query_rate", "ce_num_corrected_mean",
                                   "ce_neg_fraction", "diff_wb_minus_ce"]:
                            combined[k] = float("nan")
                        combined["cross_dataset_sign_match"] = float("nan")

                    rows.append(combined)
                    break  # Use first available source

master = pd.DataFrame(rows)

# ── Save main CSV ─────────────────────────────────────────────────────────

main_path = OUT_DIR / "cross_dataset_comparison.csv"
master.to_csv(main_path, index=False)
print(f"Saved: {main_path} ({len(master)} rows)")

# ── Also save a pivoted "paper-ready" table ───────────────────────────────

# Pivot: rows = (noise, budget), columns = (method, dataset)
pivot_rows = []
for noise in NOISES:
    for budget in BUDGETS:
        sub = master[master["noise_name"] == noise]
        sub = sub[sub["budget_fraction"] == budget]
        for method in METHODS:
            mrow = sub[sub["method"] == method]
            if mrow.empty:
                continue
            r = mrow.iloc[0]
            pivot_rows.append({
                "noise_name": noise,
                "budget": f"{budget:.0%}",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "waterbirds_dWGA": f"{r['wb_delta_wga_mean']:+.4f} ± {r['wb_delta_wga_sem']:.4f}",
                "celeba_dWGA": f"{r['ce_delta_wga_mean']:+.4f} ± {r['ce_delta_wga_sem']:.4f}" if not np.isnan(r['ce_delta_wga_mean']) else "N/A",
                "waterbirds_WGA": f"{r['wb_wga_mean']:.4f}",
                "celeba_WGA": f"{r['ce_wga_mean']:.4f}" if not np.isnan(r['ce_wga_mean']) else "N/A",
                "waterbirds_minq": f"{r['wb_minority_query_rate']:.3f}",
                "celeba_minq": f"{r['ce_minority_query_rate']:.3f}" if not np.isnan(r['ce_minority_query_rate']) else "N/A",
                "waterbirds_noise_prec": f"{r['wb_noise_precision']:.3f}",
                "celeba_noise_prec": f"{r['ce_noise_precision']:.3f}" if not np.isnan(r['ce_noise_precision']) else "N/A",
                "wb_n_seeds": int(r['wb_n_seeds']),
                "ce_n_seeds": int(r['ce_n_seeds']),
            })

pivot_df = pd.DataFrame(pivot_rows)
pivot_path = OUT_DIR / "cross_dataset_pivot_readable.csv"
pivot_df.to_csv(pivot_path, index=False)
print(f"Saved: {pivot_path} ({len(pivot_df)} rows)")

# ── Per-dataset separate tables ───────────────────────────────────────────

# Waterbirds standalone
wb_rows = []
for noise in NOISES:
    for budget in BUDGETS:
        for method in METHODS:
            sub = master[(master["noise_name"] == noise) & (master["budget_fraction"] == budget) & (master["method"] == method)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            wb_rows.append({
                "noise_name": noise,
                "budget": f"{budget:.0%}",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_seeds": int(r["wb_n_seeds"]),
                "delta_wga_mean": r["wb_delta_wga_mean"],
                "delta_wga_sem": r["wb_delta_wga_sem"],
                "delta_wga_min": r["wb_delta_wga_min"],
                "delta_wga_max": r["wb_delta_wga_max"],
                "wga_mean": r["wb_wga_mean"],
                "noise_precision": r["wb_noise_precision"],
                "minority_query_rate": r["wb_minority_query_rate"],
                "num_corrected_mean": r["wb_num_corrected_mean"],
                "neg_fraction": r["wb_neg_fraction"],
            })

wb_df = pd.DataFrame(wb_rows)
wb_path = OUT_DIR / "waterbirds_summary.csv"
wb_df.to_csv(wb_path, index=False)
print(f"Saved: {wb_path} ({len(wb_df)} rows)")

# CelebA standalone
ce_rows = []
for noise in NOISES:
    for budget in BUDGETS:
        for method in METHODS:
            sub = master[(master["noise_name"] == noise) & (master["budget_fraction"] == budget) & (master["method"] == method)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            if np.isnan(r["ce_delta_wga_mean"]):
                continue
            ce_rows.append({
                "noise_name": noise,
                "budget": f"{budget:.0%}",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_seeds": int(r["ce_n_seeds"]),
                "delta_wga_mean": r["ce_delta_wga_mean"],
                "delta_wga_sem": r["ce_delta_wga_sem"],
                "delta_wga_min": r["ce_delta_wga_min"],
                "delta_wga_max": r["ce_delta_wga_max"],
                "wga_mean": r["ce_wga_mean"],
                "noise_precision": r["ce_noise_precision"],
                "minority_query_rate": r["ce_minority_query_rate"],
                "num_corrected_mean": r["ce_num_corrected_mean"],
                "neg_fraction": r["ce_neg_fraction"],
            })

ce_df = pd.DataFrame(ce_rows)
ce_path = OUT_DIR / "celeba_summary.csv"
ce_df.to_csv(ce_path, index=False)
print(f"Saved: {ce_path} ({len(ce_df)} rows)")

# ── Cross-dataset paired comparison for shared methods ────────────────────

paired_rows = []
for noise in NOISES:
    for budget in BUDGETS:
        for method in METHODS:
            wb = extract_from_source(wb_v8, noise, budget, method, n_seeds=5)
            if wb is None:
                # Fall back to other waterbirds sources
                for src, n in [(wb_v7_5seed, 5), (wb_v1_5seed, 5)]:
                    wb = extract_from_source(src, noise, budget, method, n_seeds=n)
                    if wb is not None:
                        break
            ce = extract_from_source(celeba, noise, budget, method, n_seeds=5)
            if wb is None or ce is None:
                continue
            paired_rows.append({
                "noise_name": noise,
                "budget": f"{budget:.0%}",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "waterbirds_dWGA": wb["delta_wga_mean"],
                "celeba_dWGA": ce["delta_wga_mean"],
                "diff_wb_minus_ce": wb["delta_wga_mean"] - ce["delta_wga_mean"],
                "abs_diff": abs(wb["delta_wga_mean"] - ce["delta_wga_mean"]),
                "waterbirds_minq": wb["minority_query_rate"],
                "celeba_minq": ce["minority_query_rate"],
                "waterbirds_noise_prec": wb["noise_precision"],
                "celeba_noise_prec": ce["noise_precision"],
                "waterbirds_wga": wb["wga_mean"],
                "celeba_wga": ce["wga_mean"],
            })

paired_df = pd.DataFrame(paired_rows)
paired_path = OUT_DIR / "cross_dataset_paired.csv"
paired_df.to_csv(paired_path, index=False)
print(f"Saved: {paired_path} ({len(paired_df)} rows)")

# ── Rank correlation: Waterbirds ranking vs CelebA ranking ────────────────

rank_rows = []
for noise in NOISES:
    for budget in BUDGETS:
        wb_vals = []
        ce_vals = []
        methods_in_pair = []
        for method in METHODS:
            paired = [r for r in paired_rows if r["noise_name"] == noise and r["budget"] == f"{budget:.0%}" and r["method"] == method]
            if paired:
                wb_vals.append(paired[0]["waterbirds_dWGA"])
                ce_vals.append(paired[0]["celeba_dWGA"])
                methods_in_pair.append(method)
        if len(wb_vals) >= 4:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(wb_vals, ce_vals)
            rank_rows.append({
                "noise_name": noise,
                "budget": f"{budget:.0%}",
                "n_methods": len(wb_vals),
                "spearman_rho": rho,
                "spearman_p": pval,
                "methods": ", ".join(methods_in_pair),
            })

rank_df = pd.DataFrame(rank_rows)
rank_path = OUT_DIR / "cross_dataset_rank_correlation.csv"
rank_df.to_csv(rank_path, index=False)
print(f"Saved: {rank_path} ({len(rank_df)} rows)")

# ── Print preview ─────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("CROSS-DATASET COMPARISON: Waterbirds vs CelebA")
print("=" * 100)

for noise in NOISES:
    print(f"\n{'='*100}")
    print(f"NOISE: {noise}")
    print(f"{'='*100}")
    for budget in BUDGETS:
        sub = paired_df[(paired_df["noise_name"] == noise) & (paired_df["budget"] == f"{budget:.0%}")]
        if sub.empty:
            continue
        print(f"\n  BUDGET={budget:.0%}:")
        print(f"    {'method':<32s} {'WB dWGA':>10s} {'CE dWGA':>10s} {'Diff':>10s} {'WB minq':>8s} {'CE minq':>8s} {'WB WGA':>8s} {'CE WGA':>8s}")
        for _, r in sub.sort_values("waterbirds_dWGA", ascending=False, key=abs).iterrows():
            print(f"    {r['method']:<32s} {r['waterbirds_dWGA']:>+9.4f}  {r['celeba_dWGA']:>+9.4f}  {r['diff_wb_minus_ce']:>+9.4f}  "
                  f"{r['waterbirds_minq']:>7.3f}  {r['celeba_minq']:>7.3f}  "
                  f"{r['waterbirds_wga']:>7.4f}  {r['celeba_wga']:>7.4f}")

print(f"\n{'='*100}")
print("RANK CORRELATION: Waterbirds method ranking vs CelebA method ranking")
print(f"{'='*100}")
for _, r in rank_df.iterrows():
    print(f"  {r['noise_name']} @{r['budget']}: ρ={r['spearman_rho']:+.3f}, p={r['spearman_p']:.4f} (n={r['n_methods']} methods)")

print(f"\nAll CSVs saved to: {OUT_DIR}/")
for p in sorted(OUT_DIR.glob("*.csv")):
    print(f"  {p.name}  ({len(pd.read_csv(p))} rows)")
