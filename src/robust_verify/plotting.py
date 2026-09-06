from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_wga_budget(results: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    for noise_name, frame in results.groupby("noise_name"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, method_frame in frame.groupby("method"):
            summary = (
                method_frame.groupby("budget_fraction")["wga"]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("budget_fraction")
            )
            x = 100 * summary["budget_fraction"].to_numpy()
            y = summary["mean"].to_numpy()
            sem = summary["sem"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, y - sem, y + sem, alpha=0.15)

        baseline = frame["baseline_wga"].mean()
        ax.axhline(baseline, linestyle="--", label="no_query")
        ax.set_xlabel("Verification budget (%)")
        ax.set_ylabel("Worst-group accuracy")
        ax.set_title(f"WGA–budget curve: {noise_name}")
        ax.legend(fontsize=8)
        _save(fig, output_dir / f"wga_budget_{noise_name}.png")


def plot_corrected_vs_wga(results: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    for noise_name, frame in results.groupby("noise_name"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, method_frame in frame.groupby("method"):
            ax.scatter(
                method_frame["num_corrected"],
                method_frame["delta_wga"],
                label=method,
                alpha=0.8,
            )
        ax.set_xlabel("Number of corrected labels")
        ax.set_ylabel("WGA improvement")
        ax.set_title(f"Corrections versus WGA gain: {noise_name}")
        ax.legend(fontsize=8)
        _save(fig, output_dir / f"corrected_vs_wga_{noise_name}.png")


def plot_query_composition(
    results: pd.DataFrame,
    output_dir: str | Path,
    preferred_budget: float = 0.01,
) -> None:
    output_dir = Path(output_dir)
    components = [
        "mislabeled_majority",
        "mislabeled_minority",
        "clean_majority",
        "clean_minority",
    ]

    for noise_name, frame in results.groupby("noise_name"):
        budgets = sorted(frame["budget_fraction"].unique())
        budget = min(budgets, key=lambda value: abs(value - preferred_budget))
        subset = frame.loc[frame["budget_fraction"] == budget]
        summary = subset.groupby("method")[components].mean().sort_index()

        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(summary))
        x = np.arange(len(summary))
        for component in components:
            values = summary[component].to_numpy()
            ax.bar(x, values, bottom=bottom, label=component)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=35, ha="right")
        ax.set_ylabel("Mean number of queried samples")
        ax.set_title(f"Query composition at {100 * budget:.1f}%: {noise_name}")
        ax.legend(fontsize=8)
        _save(fig, output_dir / f"query_composition_{noise_name}.png")


def create_all_plots(results: pd.DataFrame, output_dir: str | Path) -> None:
    if results.empty:
        return
    plot_wga_budget(results, output_dir)
    plot_corrected_vs_wga(results, output_dir)
    plot_query_composition(results, output_dir)
