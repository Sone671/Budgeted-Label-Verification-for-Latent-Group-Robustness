#!/usr/bin/env python
"""Generate figures for the diagnostic-benchmark manuscript.

This script intentionally uses neutral method labels: the guarded v8 fallback
is a diagnostic comparator, not a safety-certified method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANGUAGE = "en"
OUT = Path("paper/figures_diagnostic")

DATASETS = {
    "Waterbirds": pd.read_csv("outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
    "CelebA": pd.read_csv("outputs/ablation_celeba_10seeds/stage1/results.csv"),
    "CivilComments": pd.read_csv("outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
}
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]
NOISES = ["uniform20", "minority_high_40"]

METHODS = {
    "noise_score": ("NoiseScore", "#777777", "o"),
    "loss": ("Loss", "#d62728", "s"),
    "noise_score_cpba_only": ("CPBA-only", "#ff7f0e", "^"),
    "noise_score_cpba_tc_no_fallback": ("CPBA-TC-noFB", "#9467bd", "P"),
    "noise_score_cpba_tc_v8_fallback": ("CPBA-TC-Guarded", "#1f77b4", "o"),
    "oracle_noise_group_balanced": ("Oracle-GB", "#2ca02c", "D"),
}


def text(en: str, zh: str) -> str:
    return zh if LANGUAGE == "zh" else en


def method_label(method: str, label: str) -> str:
    chinese = {"loss": "损失"}
    return chinese.get(method, label) if LANGUAGE == "zh" else label


def subset(df: pd.DataFrame, method: str, noise: str, budget: float) -> pd.DataFrame:
    return df[
        (df.method == method)
        & (df.noise_name == noise)
        & np.isclose(df.budget_fraction, budget)
    ]


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(sem)


def precision_vs_repair() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5.3, 7.0))
    highlighted = {"loss", "noise_score_cpba_tc_v8_fallback", "oracle_noise_group_balanced"}
    for ax, (dataset_name, df) in zip(axes, DATASETS.items()):
        for method, (label, color, marker) in METHODS.items():
            for noise in NOISES:
                for budget in BUDGETS:
                    rows = subset(df, method, noise, budget)
                    if rows.empty:
                        continue
                    size = 46 if method in highlighted else 17
                    alpha = 0.9 if method in highlighted else 0.35
                    ax.scatter(
                        rows.noise_precision.mean(),
                        rows.delta_wga.mean(),
                        s=size,
                        color=color,
                        marker=marker,
                        alpha=alpha,
                        edgecolors="none",
                    )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(dataset_name)
        ax.set_xlabel(text("Noise precision", "噪声精度"))
        ax.set_ylabel(r"$\Delta$WGA")
    handles = [
        plt.Line2D([0], [0], color=color, marker=marker, linestyle="", markersize=7,
                   label=method_label(key, label))
        for key, (label, color, marker) in METHODS.items()
        if key in highlighted
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle(text("Noise detection precision is not a reliable proxy for WGA repair",
                     "噪声检测精度不是 WGA 修复的可靠代理"), y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_precision_vs_repair.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def budget_curves() -> None:
    methods = [
        "noise_score",
        "noise_score_cpba_only",
        "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
        "oracle_noise_group_balanced",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(5.3, 7.0), sharey=False)
    for ax, (dataset_name, df) in zip(axes, DATASETS.items()):
        for method in methods:
            label, color, marker = METHODS[method]
            xs, means, sems = [], [], []
            for budget in BUDGETS:
                rows = subset(df, method, "uniform20", budget)
                mean, sem = mean_and_sem(rows.delta_wga.to_numpy())
                if np.isfinite(mean):
                    xs.append(budget * 100)
                    means.append(mean)
                    sems.append(sem)
            ax.errorbar(xs, means, yerr=sems, color=color, marker=marker, markersize=5,
                        linewidth=1.35, capsize=2, label=method_label(method, label))
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(f"{dataset_name} (uniform20)")
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xticklabels(["1%", "2%", "5%", "10%"])
        ax.set_xlabel(text("Verification budget", "核验预算"))
        ax.set_ylabel(text(r"Mean $\Delta$WGA", r"平均 $\Delta$WGA"))
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.suptitle(text("Budget response is dataset dependent", "预算响应依赖于数据集"), y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_budget_response.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def failure_profile() -> None:
    methods = [
        "noise_score",
        "noise_score_cpba_tc",
        "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
    ]
    labels = {
        "noise_score": "NoiseScore",
        "noise_score_cpba_tc": "CPBA-TC\n(v1 fallback)",
        "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
        "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Guarded\n(v8 fallback)",
    }
    if LANGUAGE == "zh":
        labels["noise_score_cpba_tc"] = "CPBA-TC\n（v1 回退）"
        labels["noise_score_cpba_tc_v8_fallback"] = "CPBA-TC-Guarded\n（v8 回退）"
    colors = ["#777777", "#d62728", "#ff7f0e", "#1f77b4"]
    metrics = [
        (text("Mean $\\Delta$WGA", "平均 $\\Delta$WGA"), lambda values: values.mean()),
        (text("Negative-seed rate (%)", "负增益种子比例（%）"), lambda values: 100 * (values < 0).mean()),
        (text("Worst-seed $\\Delta$WGA", "最差种子 $\\Delta$WGA"), lambda values: values.min()),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(5.3, 7.0))
    df = DATASETS["Waterbirds"]
    values_by_method = {}
    for method in methods:
        values = pd.concat([subset(df, method, "uniform20", b) for b in [0.05, 0.10]]).delta_wga
        values_by_method[method] = values.to_numpy(dtype=float)
    for ax, (title, fn) in zip(axes, metrics):
        values = [fn(values_by_method[method]) for method in methods]
        ax.bar([labels[method] for method in methods], values, color=colors,
               edgecolor="black", linewidth=0.45)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle(text("Fallbacks reduce but do not remove high-budget failure",
                     "回退可缓解、但未消除高预算失败"), y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_failure_profile.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    args = parser.parse_args()
    LANGUAGE = args.language
    OUT = Path("paper/figures_diagnostic" + ("_zh" if LANGUAGE == "zh" else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    style = {"font.size": 10, "axes.titlesize": 11}
    if LANGUAGE == "zh":
        style.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    plt.rcParams.update(style)
    precision_vs_repair()
    budget_curves()
    failure_profile()
    print(f"Wrote diagnostic figures to {OUT}")
