#!/usr/bin/env python
"""Phase 2-5: Main table + figures + statistics + failure analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("outputs/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ────────────────────────────────────────────────────────────────────
WB = pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv")
CE = pd.read_csv("outputs/ablation_celeba_10seeds/stage1/results.csv")
CC = pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/results.csv")

DATASETS = {"Waterbirds": WB, "CelebA": CE, "CivilComments": CC}
NOISES = ["uniform20", "minority_high_40"]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]

# ── Methods ─────────────────────────────────────────────────────────────────
MAIN_METHODS = [
    "random", "loss", "entropy",
    "noise_score",
    "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc_v8_fallback",
    "oracle_noise_group_balanced",
]

MAIN_LABELS = {
    "random": "Random",
    "loss": "Loss",
    "entropy": "Entropy",
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "oracle_noise_group_balanced": "Oracle-GB",
}

ABLATION_METHODS = [
    "noise_score",
    "noise_score_cpba_only",
    "noise_score_tc_only",
    "noise_score_cpba_tc",
    "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback",
]

ABLATION_LABELS = {
    "noise_score": "NoiseScore",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_tc_only": "TC-only",
    "noise_score_cpba_tc": "CPBA-TC",
    "noise_score_cpba_tc_no_fallback": "noFB",
    "noise_score_cpba_tc_v8_fallback": "Safe(v8FB)",
}


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def get_vals(ds, noise, budget, method):
    sub = ds[
        (ds["noise_name"] == noise)
        & (ds["budget_fraction"] == budget)
        & (ds["method"] == method)
    ]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


def get_vals_pooled(ds, method, noise=None, budget=None):
    sub = ds[ds["method"] == method]
    if noise is not None:
        sub = sub[sub["noise_name"] == noise]
    if budget is not None:
        sub = sub[sub["budget_fraction"] == budget]
    return sub["delta_wga"].to_numpy() if not sub.empty else np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MAIN RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 100)
print("MAIN RESULTS TABLE")
print("=" * 100)

# Per-noise, per-budget, per-dataset
table_rows = []
for ds_name, ds in DATASETS.items():
    for noise in NOISES:
        for budget in BUDGETS:
            for method in MAIN_METHODS:
                vals = get_vals(ds, noise, budget, method)
                if len(vals) == 0:
                    continue
                table_rows.append({
                    "dataset": ds_name,
                    "noise": noise,
                    "budget": f"{budget:.1%}",
                    "budget_fraction": budget,
                    "method": MAIN_LABELS.get(method, method),
                    "n": len(vals),
                    "dWGA_mean": round(float(vals.mean()), 4),
                    "dWGA_sem": round(sem(vals), 4),
                    "dWGA_min": round(float(vals.min()), 4),
                    "dWGA_max": round(float(vals.max()), 4),
                    "neg_pct": round(float((vals < 0).mean()) * 100, 1),
                })

main_df = pd.DataFrame(table_rows)
main_path = OUT_DIR / "main_results_full.csv"
main_df.to_csv(main_path, index=False)
print(f"Saved: {main_path}  ({len(main_df)} rows)")

# Summary: averaged over budgets, per noise
summary_rows = []
for ds_name, ds in DATASETS.items():
    for noise in NOISES:
        for method in MAIN_METHODS:
            vals = get_vals_pooled(ds, method, noise=noise)
            if len(vals) == 0:
                continue
            summary_rows.append({
                "dataset": ds_name,
                "noise": noise,
                "method": MAIN_LABELS.get(method, method),
                "n": len(vals),
                "dWGA_mean": round(float(vals.mean()), 4),
                "dWGA_sem": round(sem(vals), 4),
                "neg_pct": round(float((vals < 0).mean()) * 100, 1),
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "main_results_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}  ({len(summary_df)} rows)")

# Print readable per-dataset summary
for ds_name in ["Waterbirds", "CelebA", "CivilComments"]:
    s = summary_df[summary_df["dataset"] == ds_name]
    n1 = s[s["noise"] == "uniform20"].set_index("method")["dWGA_mean"]
    n2 = s[s["noise"] == "minority_high_40"].set_index("method")["dWGA_mean"]
    print(f"\n  [{ds_name}]")
    print(f"  {'Method':<20s} {'uniform20':>9s} {'min_high40':>11s}")
    print(f"  {'-'*40}")
    for m in MAIN_LABELS.values():
        u = n1.get(m, float("nan"))
        h = n2.get(m, float("nan"))
        print(f"  {m:<20s} {u:>+9.4f} {h:>+11.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})

# ── Fig 1: Noise detection is not WGA repair ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
for ax, (ds_name, ds) in zip(axes, DATASETS.items()):
    points = []
    for method in MAIN_METHODS:
        for noise in NOISES:
            for budget in BUDGETS:
                sub = ds[
                    (ds["noise_name"] == noise)
                    & (ds["budget_fraction"] == budget)
                    & (ds["method"] == method)
                ]
                if sub.empty:
                    continue
                prec = float(sub["noise_precision"].mean())
                dw = float(sub["delta_wga"].mean())
                lbl = MAIN_LABELS.get(method, method)
                pts = "Oracle-GB" if "Oracle" in lbl else ("CPBA-TC-Safe" if "Safe" in lbl else "Other")
                points.append((prec, dw, lbl, pts))

    for prec, dw, lbl, pts in points:
        color = "red" if lbl == "Loss" else ("green" if "Oracle" in lbl else ("blue" if "Safe" in lbl else "gray"))
        marker = "s" if lbl == "Loss" else ("D" if "Oracle" in lbl else ("o" if "Safe" in lbl else "."))
        size = 80 if lbl in ("Loss", "Oracle-GB", "CPBA-TC-Safe") else 20
        alpha_v = 0.9 if lbl in ("Loss", "Oracle-GB", "CPBA-TC-Safe") else 0.35
        ax.scatter(prec, dw, c=color, marker=marker, s=size, alpha=alpha_v, edgecolors="none")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(ds_name)
    ax.set_xlabel("Noise Precision")
    ax.set_ylabel("ΔWGA" if ax is axes[0] else "")

axes[0].legend(
    handles=[
        plt.Line2D([0], [0], marker="s", color="red", linestyle="", markersize=8, label="Loss (high prec, neg WGA)"),
        plt.Line2D([0], [0], marker="o", color="blue", linestyle="", markersize=8, label="CPBA-TC-Safe"),
        plt.Line2D([0], [0], marker="D", color="green", linestyle="", markersize=8, label="Oracle-GB"),
    ],
    loc="lower right", fontsize=8,
)
fig.suptitle("Figure 1: Noise detection precision ≠ WGA repair", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_noise_detection_vs_wga.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_noise_detection_vs_wga.png")


# ── Fig 2: Budget curves ──────────────────────────────────────────────────

BUDGET_METHODS = [
    "noise_score", "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc", "noise_score_cpba_tc_v8_fallback",
    "oracle_noise_group_balanced",
]
BUDGET_LABELS = {
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc": "CPBA-TC",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "oracle_noise_group_balanced": "Oracle-GB",
}
BUDGET_COLORS = {"NoiseScore": "gray", "NGC": "orange", "CPBA-TC": "red", "CPBA-TC-Safe": "blue", "Oracle-GB": "green"}
BUDGET_MARKERS = {"NoiseScore": ".", "NGC": "^", "CPBA-TC": "x", "CPBA-TC-Safe": "o", "Oracle-GB": "D"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for ax, (ds_name, ds) in zip(axes, DATASETS.items()):
    for method in BUDGET_METHODS:
        lbl = BUDGET_LABELS[method]
        xs, ys, es = [], [], []
        for bud in BUDGETS:
            vals = get_vals(ds, "uniform20", bud, method)
            if len(vals) > 0:
                xs.append(bud * 100)
                ys.append(float(vals.mean()))
                es.append(sem(vals))
        if xs:
            ax.errorbar(xs, ys, yerr=es,
                        label=lbl, color=BUDGET_COLORS[lbl],
                        marker=BUDGET_MARKERS[lbl], markersize=6,
                        capsize=3, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"{ds_name} (uniform20)")
    ax.set_xlabel("Budget (%)")
    if ax is axes[0]:
        ax.set_ylabel("ΔWGA")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1%", "2%", "5%", "10%"])

axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("Figure 2: Budget curves (uniform20)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_budget_curves.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_budget_curves.png")


# ── Fig 3: Ablation component attribution ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, ds_name in zip(axes, ["Waterbirds", "CivilComments"]):
    ds = DATASETS[ds_name]
    labels_plot, means_plot = [], []
    for method in ABLATION_METHODS:
        vals = get_vals(ds, "uniform20", 0.05, method)
        if len(vals) == 0:
            continue
        labels_plot.append(ABLATION_LABELS[method])
        means_plot.append(float(vals.mean()))
    colors = ["gray", "orange", "lightblue", "red", "blue", "darkblue"]
    bars = ax.bar(labels_plot, means_plot, color=colors[:len(labels_plot)], edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{ds_name} (uniform20, 5%)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    if ax is axes[0]:
        ax.set_ylabel("ΔWGA")

fig.suptitle("Figure 3: Ablation component attribution", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_ablation_attribution.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_ablation_attribution.png")


# ── Fig 4: Fallback failure analysis (Waterbirds uniform20) ────────────────

FALLBACK_METHODS = [
    "noise_score", "noise_score_cpba_tc",
    "noise_score_cpba_tc_no_fallback", "noise_score_cpba_tc_v8_fallback",
]
FB_LABELS = {
    "noise_score": "NoiseScore",
    "noise_score_cpba_tc": "CPBA-TC\n(v1 FB)",
    "noise_score_cpba_tc_no_fallback": "noFB",
    "noise_score_cpba_tc_v8_fallback": "Safe\n(v8FB)",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
wb_ds = DATASETS["Waterbirds"]
for ax, metric in zip(axes, ["mean", "neg_pct", "worst"]):
    labels_fb, vals_fb = [], []
    for method in FALLBACK_METHODS:
        all_v = []
        for bud in [0.05, 0.10]:
            v = get_vals(wb_ds, "uniform20", bud, method)
            all_v.extend(v.tolist())
        all_v = np.array(all_v)
        if metric == "mean":
            vals_fb.append(float(all_v.mean()))
        elif metric == "neg_pct":
            vals_fb.append(float((all_v < 0).mean()) * 100)
        elif metric == "worst":
            vals_fb.append(float(all_v.min()))
        labels_fb.append(FB_LABELS.get(method, method))

    colors_fb = ["gray", "red", "orange", "blue"]
    ax.bar(labels_fb, vals_fb, color=colors_fb, edgecolor="black", linewidth=0.5)
    ax.set_title({"mean": "Mean ΔWGA", "neg_pct": "Negative Seed %", "worst": "Worst Seed ΔWGA"}[metric])
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--" if metric != "neg_pct" else "-")
    if metric == "mean":
        ax.set_ylabel("ΔWGA")

fig.suptitle("Figure 4: Fallback failure analysis — Waterbirds uniform20 (pooled 5%+10%)",
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_fallback_analysis.png", bbox_inches="tight")
plt.close()
print("Saved: fig4_fallback_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: PAIRED SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("PAIRED SIGNIFICANCE TESTS")
print("=" * 100)

PAIRS_NAMES = {
    ("noise_score_cpba_tc_v8_fallback", "noise_score"): "Safe vs NoiseScore",
    ("noise_score_cpba_tc_v8_fallback", "noise_score_budget_hybrid_v8"): "Safe vs NGC",
    ("noise_score_cpba_tc_v8_fallback", "noise_score_cpba_tc"): "Safe vs CPBA-TC",
    ("noise_score_cpba_only", "noise_score"): "CPBA-only vs NoiseScore",
    ("noise_score_tc_only", "noise_score"): "TC-only vs NoiseScore",
    ("noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced"): "Safe vs Oracle-GB",
}

all_tests = []
for ds_name, ds in DATASETS.items():
    for noise in NOISES:
        for budget in BUDGETS:
            for (m1, m2), desc in PAIRS_NAMES.items():
                v1 = get_vals(ds, noise, budget, m1)
                v2 = get_vals(ds, noise, budget, m2)
                if len(v1) < 2 or len(v2) < 2:
                    continue
                diff = v1 - v2
                try:
                    t_val, p_ttest = stats.ttest_rel(v1, v2)
                except:
                    p_ttest = 1.0
                try:
                    _, p_wilcox = stats.wilcoxon(v1, v2)
                except ValueError:
                    p_wilcox = float("nan")
                wins = int((diff > 0.0005).sum())
                losses = int((diff < -0.0005).sum())
                ties = len(diff) - wins - losses
                all_tests.append({
                    "dataset": ds_name,
                    "noise": noise,
                    "budget": f"{budget:.1%}",
                    "comparison": desc,
                    "n_seeds": len(v1),
                    "mean_diff": round(float(diff.mean()), 4),
                    "ci95_low": round(float(diff.mean() - 1.96 * sem(diff)), 4),
                    "ci95_high": round(float(diff.mean() + 1.96 * sem(diff)), 4),
                    "p_ttest": round(p_ttest, 4),
                    "p_wilcox": round(p_wilcox, 4) if np.isfinite(p_wilcox) else "nan",
                    "win": wins,
                    "tie": ties,
                    "loss": losses,
                    "sig": "***" if p_ttest < 0.001 else ("**" if p_ttest < 0.01 else ("*" if p_ttest < 0.05 else "ns")),
                })

tests_df = pd.DataFrame(all_tests)
tests_path = OUT_DIR / "paired_tests.csv"
tests_df.to_csv(tests_path, index=False)
print(f"Saved: {tests_path}  ({len(tests_df)} rows)")

# Print key tests
for ds_name in ["Waterbirds", "CivilComments"]:
    print(f"\n  [{ds_name} — uniform20, budget=5%]")
    sub = tests_df[
        (tests_df["dataset"] == ds_name)
        & (tests_df["noise"] == "uniform20")
        & (tests_df["budget"] == "5.0%")
    ]
    for _, r in sub.iterrows():
        print(f"    {r['comparison']:<30s} Δ={r['mean_diff']:+7.4f}  p={r['p_ttest']:.4f} {r['sig']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: RESIDUAL FAILURE ANALYSIS (Waterbirds b=10% uniform20)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("RESIDUAL FAILURE ANALYSIS: Waterbirds uniform20 b=10%")
print("=" * 100)

FAIL_METHODS = ["entropy", "noise_score_budget_hybrid_v8",
                "noise_score_cpba_tc_v8_fallback", "noise_score_cpba_tc"]
FAIL_LABELS = {
    "entropy": "Entropy",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Safe",
    "noise_score_cpba_tc": "CPBA-TC",
}

for method in FAIL_METHODS:
    sub = WB[(WB["noise_name"] == "uniform20")
             & (WB["budget_fraction"] == 0.10)
             & (WB["method"] == method)]
    if sub.empty:
        continue
    lbl = FAIL_LABELS.get(method, method)
    dw = sub["delta_wga"]
    mq = sub["minority_query_rate"]
    np_ = sub["noise_precision"]
    nc = sub["num_corrected"]
    print(f"  {lbl:<18s} dWGA={dw.mean():+.4f}±{sem(dw.to_numpy()):.4f}  "
          f"neg%={(dw<0).mean()*100:.0f}%  min={dw.min():+.4f}  "
          f"min_q={mq.mean():.3f}  n_prec={np_.mean():.3f}  n_corr={nc.mean():.1f}")

print("\nAll outputs saved to:", OUT_DIR)
