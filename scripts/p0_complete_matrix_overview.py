#!/usr/bin/env python
"""Create a non-selective overview of every cell in the frozen S2 matrix.

The plot deliberately treats cells as descriptive observations.  It shows the
absolute no-correction comparison and paired Loss comparison side by side for
all 580 method/noise/budget cells, without promoting selected rows to a new
confirmatory claim.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATASET_ORDER = ("Waterbirds", "CelebA", "CivilComments")
NOISE_ORDER = ("uniform20", "minority_high_40")
NOISE_LABELS = {"uniform20": "Uniform 20%", "minority_high_40": "Minority-high 40%"}
BUDGET_LABELS = {0.005: "0.5%", 0.01: "1%", 0.02: "2%", 0.05: "5%", 0.1: "10%"}
BUDGET_COLORS = {
    0.005: "#440154",
    0.01: "#3b528b",
    0.02: "#21918c",
    0.05: "#5ec962",
    0.1: "#fde725",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def padded_extent(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.min(values), np.max(values)
    padding = max(0.015, 0.07 * (upper - lower))
    return float(lower - padding), float(upper + padding)


def sign_label(absolute: float, paired: float) -> str:
    if absolute > 0 and paired > 0:
        return "absolute-positive / beats-Loss"
    if absolute > 0 and paired <= 0:
        return "absolute-positive / trails-or-ties-Loss"
    if absolute <= 0 and paired > 0:
        return "absolute-nonpositive / beats-Loss"
    return "absolute-nonpositive / trails-or-ties-Loss"


def build_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    display = frame.copy()
    display["absolute_pp"] = 100 * display["absolute_delta_wga"]
    display["paired_pp"] = 100 * display["delta_wga_vs_loss"]
    x_limits = padded_extent(display["absolute_pp"].to_numpy(float))
    y_limits = padded_extent(display["paired_pp"].to_numpy(float))

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharex=True, sharey=True)
    for row, noise in enumerate(NOISE_ORDER):
        for column, dataset in enumerate(DATASET_ORDER):
            axis = axes[row, column]
            subset = display[(display["dataset"] == dataset) & (display["noise"] == noise)]
            practical = subset[~subset["method"].str.startswith("oracle_")]
            oracle = subset[subset["method"].str.startswith("oracle_")]
            for budget in sorted(BUDGET_COLORS):
                for group, marker, size, alpha in ((practical, "o", 27, 0.68), (oracle, "x", 34, 0.86)):
                    cell = group[np.isclose(group["budget_fraction"], budget)]
                    if cell.empty:
                        continue
                    axis.scatter(
                        cell["absolute_pp"],
                        cell["paired_pp"],
                        s=size,
                        marker=marker,
                        color=BUDGET_COLORS[budget],
                        alpha=alpha,
                        linewidths=0.75,
                    )
            axis.axhline(0, color="#606060", linewidth=0.7, zorder=0)
            axis.axvline(0, color="#606060", linewidth=0.7, zorder=0)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_title(f"{dataset} · {NOISE_LABELS[noise]}\n{len(subset)} cells", fontsize=10.5)
            axis.grid(axis="both", color="#d5d5d5", linewidth=0.45, alpha=0.55)
    fig.supxlabel("Absolute ΔWGA vs no correction (pp)", y=0.075)
    fig.supylabel("Paired ΔWGA vs Loss (pp)", x=0.008)

    budget_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, markersize=6, label=BUDGET_LABELS[budget])
        for budget, color in BUDGET_COLORS.items()
    ]
    type_handles = [
        Line2D([0], [0], marker="o", color="#404040", linestyle="None", markersize=6, label="Practical method"),
        Line2D([0], [0], marker="x", color="#404040", linestyle="None", markersize=6, label="Oracle diagnostic"),
    ]
    fig.legend(
        handles=budget_handles + type_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        title="Budget / method type",
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.17, wspace=0.03, hspace=0.18)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_p0_complete_matrix_overview.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(output: Path, source: Path, frame: pd.DataFrame, signs: pd.DataFrame) -> None:
    counts = (
        signs.groupby(["dataset", "noise", "sign_quadrant"], sort=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    lines = [
        "# P0 complete frozen-matrix overview",
        "",
        f"Source: `{source}`",
        f"SHA256: `{sha256_file(source)}`",
        "",
        f"The figure contains all {len(frame)} aggregate method/noise/budget cells. "
        "Cells are descriptive; the figure does not pool budgets or create a new confirmatory test.",
        "",
        "## Cell counts by sign quadrant",
        "",
        "| Dataset | Noise | Quadrant | Cells |",
        "|---|---|---|---:|",
    ]
    for _, row in counts.iterrows():
        lines.append(f"| {row['dataset']} | {row['noise']} | {row['sign_quadrant']} | {row['cell_count']} |")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="appendix_s2_complete_matrix.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = {
        "dataset",
        "noise",
        "budget_fraction",
        "method",
        "absolute_delta_wga",
        "delta_wga_vs_loss",
    }
    frame = pd.read_csv(args.matrix)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required matrix columns: {sorted(missing)}")
    duplicate = frame.duplicated(["dataset", "noise", "budget_fraction", "method"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate complete-matrix keys; refusing to create a pooled overview.")
    expected_pairs = {(dataset, noise) for dataset in DATASET_ORDER for noise in NOISE_ORDER}
    present_pairs = set(frame[["dataset", "noise"]].drop_duplicates().itertuples(index=False, name=None))
    if not expected_pairs.issubset(present_pairs):
        raise ValueError(f"Missing expected dataset/noise panels: {sorted(expected_pairs - present_pairs)}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    signs = frame[["dataset", "noise", "budget_fraction", "method", "absolute_delta_wga", "delta_wga_vs_loss"]].copy()
    signs["sign_quadrant"] = [
        sign_label(absolute, paired)
        for absolute, paired in zip(signs["absolute_delta_wga"], signs["delta_wga_vs_loss"])
    ]
    signs.to_csv(args.out_dir / "complete_matrix_cell_signs.csv", index=False)
    build_figure(frame, args.out_dir)
    write_report(args.out_dir / "P0_COMPLETE_MATRIX_OVERVIEW.md", args.matrix, frame, signs)
    print(f"P0 complete-matrix overview written to {args.out_dir}")


if __name__ == "__main__":
    main()
