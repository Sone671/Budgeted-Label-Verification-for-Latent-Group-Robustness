#!/usr/bin/env python
"""Build locked Phase-10/11 and theory figures for the main-conference draft."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.convex_two_action import logistic_design_stability_radius

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


PHASE10_SUMMARY = (
    ROOT
    / "joint_group_label_budget"
    / "outputs"
    / "phase10_cost_budget_map"
    / "analysis"
    / "cell_summary.csv"
)
PHASE11_SUMMARY = (
    ROOT
    / "joint_group_label_budget"
    / "outputs"
    / "phase11_acs_income"
    / "analysis"
    / "cell_summary.csv"
)

BENEFICIAL = "#4C78A8"
UNRESOLVED = "#D9D9D9"
HARMFUL = "#F28E2B"
NOISE_COLORS = {"uniform20": "#3569A8", "minority_high_40": "#C47A24"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in (".pdf", ".png"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _diagram_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        transform=axis.transAxes,
    )
    axis.add_patch(box)
    return box


def _diagram_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#555555",
            transform=axis.transAxes,
            connectionstyle="arc3,rad=0",
        )
    )


def build_two_world_schematic(output_dir: Path) -> list[Path]:
    """Draw the same-transcript, opposite-oracle construction used in the theorem."""

    fig, axis = plt.subplots(figsize=(7.0, 3.55), constrained_layout=True)
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    _diagram_box(
        axis,
        (0.20, 0.72),
        0.60,
        0.20,
        facecolor="#F3F3F3",
        edgecolor="#777777",
    )
    axis.text(
        0.50,
        0.875,
        "Same observable interaction transcript",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.805,
        r"same test $(X,Y)$  $\bullet$  audit $H\sim\mathrm{Bernoulli}(1/2)$"
        r"  $\bullet$  every checked label returns $0\!\rightarrow\!1$",
        ha="center",
        va="center",
        fontsize=8.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.755,
        "Valid for one-shot, staged, sequential, and randomized policies",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#444444",
        transform=axis.transAxes,
    )

    _diagram_arrow(axis, (0.41, 0.72), (0.27, 0.61))
    _diagram_arrow(axis, (0.59, 0.72), (0.73, 0.61))

    _diagram_box(
        axis,
        (0.04, 0.30),
        0.42,
        0.30,
        facecolor="#EAF2FA",
        edgecolor="#3569A8",
    )
    axis.text(
        0.25,
        0.555,
        r"Hidden world $P_0$",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        transform=axis.transAxes,
    )
    axis.text(
        0.25,
        0.485,
        r"Weak positive group is block $A_H$",
        ha="center",
        va="center",
        fontsize=8.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.25,
        0.425,
        r"Oracle: audit $H$ + verify all $r$ labels in $A_H$",
        ha="center",
        va="center",
        fontsize=8.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.25,
        0.355,
        r"$V_0^\star=M/(M+r+n)$",
        ha="center",
        va="center",
        fontsize=9.2,
        transform=axis.transAxes,
    )

    _diagram_box(
        axis,
        (0.54, 0.30),
        0.42,
        0.30,
        facecolor="#FBF0E4",
        edgecolor="#C47A24",
    )
    axis.text(
        0.75,
        0.555,
        r"Hidden world $P_1$",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        transform=axis.transAxes,
    )
    axis.text(
        0.75,
        0.485,
        r"Weak positive group is block $C$; $H$ is irrelevant",
        ha="center",
        va="center",
        fontsize=8.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.75,
        0.425,
        r"Oracle: skip audit + verify all $n$ labels in $C$",
        ha="center",
        va="center",
        fontsize=8.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.75,
        0.355,
        r"$V_1^\star=M/(M+2r)$",
        ha="center",
        va="center",
        fontsize=9.2,
        transform=axis.transAxes,
    )

    _diagram_arrow(axis, (0.25, 0.30), (0.42, 0.21))
    _diagram_arrow(axis, (0.75, 0.30), (0.58, 0.21))
    _diagram_box(
        axis,
        (0.22, 0.055),
        0.56,
        0.15,
        facecolor="#F3F3F3",
        edgecolor="#777777",
    )
    axis.text(
        0.50,
        0.158,
        "The policy cannot know which opposite allocation is optimal",
        ha="center",
        va="center",
        fontsize=8.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.100,
        r"tight minimax allocation regret $\geq M/(2M+n+3r)\;\longrightarrow\;1/2$",
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        transform=axis.transAxes,
    )
    return _save(fig, output_dir, "two-world-allocation-impossibility")


def build_phase10_map(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    datasets = ["waterbirds", "celeba"]
    noises = ["minority_high_40", "uniform20"]
    budgets = [0.01, 0.02, 0.05]
    ratios = [0.1, 0.25, 0.5, 1.0, 2.0]
    shares = [0.75, 0.50, 0.25]
    label_to_value = {"harmful": -1, "unresolved": 0, "beneficial": 1}
    cmap = ListedColormap([HARMFUL, UNRESOLVED, BENEFICIAL])

    fig, axes = plt.subplots(4, 3, figsize=(7.1, 6.5), constrained_layout=True)
    for row, (dataset, noise) in enumerate(
        (dataset, noise) for dataset in datasets for noise in noises
    ):
        row_data = summary[
            summary["dataset"].astype(str).eq(dataset)
            & summary["noise_name"].astype(str).eq(noise)
        ]
        for column, budget in enumerate(budgets):
            axis = axes[row, column]
            panel = row_data[np.isclose(row_data["budget_fraction"], budget)]
            values = np.zeros((len(shares), len(ratios)), dtype=np.float64)
            effects = np.zeros_like(values)
            adjusted = np.ones_like(values)
            for i, share in enumerate(shares):
                for j, ratio in enumerate(ratios):
                    cell = panel[
                        np.isclose(panel["target_group_action_share"], share)
                        & np.isclose(panel["group_to_label_cost_ratio"], ratio)
                    ]
                    if len(cell) != 1:
                        raise ValueError(
                            f"missing Phase-10 cell {dataset}/{noise}/{budget}/{ratio}/{share}"
                        )
                    record = cell.iloc[0]
                    values[i, j] = label_to_value[str(record["cell_label"])]
                    effects[i, j] = 100.0 * float(record["strict_cost_mean"])
                    adjusted[i, j] = float(record["strict_cost_p_holm"])
            axis.imshow(values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            for i in range(len(shares)):
                for j in range(len(ratios)):
                    suffix = "*" if adjusted[i, j] < 0.05 else ""
                    axis.text(
                        j,
                        i,
                        f"{effects[i, j]:+.1f}{suffix}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="#111111",
                    )
            axis.set_xticks(range(len(ratios)), [str(value) for value in ratios])
            axis.set_yticks(range(len(shares)), [f"{int(100 * value)}%" for value in shares])
            axis.tick_params(labelsize=7.1)
            if row == 0:
                axis.set_title(f"Budget {int(100 * budget)}%", fontsize=8.5, weight="bold")
            if row == 3:
                axis.set_xlabel("Group / label unit cost", fontsize=7.4)
            if column == 0:
                dataset_label = "Waterbirds" if dataset == "waterbirds" else "CelebA"
                noise_label = "minority-high-40" if noise == "minority_high_40" else "uniform20"
                axis.set_ylabel(
                    f"{dataset_label}\n{noise_label}\nAudit action share",
                    fontsize=7.4,
                )
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)

    legend = [
        Patch(facecolor=BENEFICIAL, label="Beneficial (95% CI > 0)"),
        Patch(facecolor=UNRESOLVED, label="Unresolved"),
        Patch(facecolor=HARMFUL, label="Harmful (95% CI < 0)"),
        Line2D([], [], linestyle="none", marker="$*$", color="#111111", label="Holm p < .05"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False, fontsize=7.2)
    fig.suptitle(
        "Strict-cost allocation map (cell text: paired WGA difference, pp)",
        fontsize=9.5,
        weight="bold",
    )
    return _save(fig, output_dir, "phase10-strict-cost-map")


def build_confirmation_forest(
    phase10: pd.DataFrame,
    phase11: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    selected = phase10[
        np.isclose(phase10["budget_fraction"], 0.02)
        & np.isclose(phase10["group_to_label_cost_ratio"], 0.1)
        & np.isclose(phase10["target_group_action_share"], 0.5)
    ].copy()
    acs = phase11.copy()
    acs["dataset"] = "acs_income"
    combined = pd.concat([selected, acs], ignore_index=True, sort=False)
    order = [
        ("waterbirds", "uniform20"),
        ("waterbirds", "minority_high_40"),
        ("celeba", "uniform20"),
        ("celeba", "minority_high_40"),
        ("acs_income", "uniform20"),
        ("acs_income", "minority_high_40"),
    ]
    labels = {
        "waterbirds": "Waterbirds",
        "celeba": "CelebA",
        "acs_income": "ACSIncome",
        "uniform20": "U",
        "minority_high_40": "M",
    }
    rows = []
    for dataset, noise in order:
        cell = combined[
            combined["dataset"].astype(str).eq(dataset)
            & combined["noise_name"].astype(str).eq(noise)
        ]
        if len(cell) != 1:
            raise ValueError(f"missing confirmation forest row {dataset}/{noise}")
        rows.append(cell.iloc[0])

    fig, axis = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for position, record in zip(y, rows):
        noise = str(record["noise_name"])
        mean = 100.0 * float(record["strict_cost_mean"])
        low = 100.0 * float(record["strict_cost_ci_low"])
        high = 100.0 * float(record["strict_cost_ci_high"])
        axis.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o" if noise == "uniform20" else "s",
            color=NOISE_COLORS[noise],
            markersize=6,
            capsize=3,
            linewidth=1.5,
        )
        axis.text(high + 0.10, position, f"{mean:+.2f}", va="center", fontsize=8.5)
    axis.axvline(0.0, color="#444444", linewidth=1.0)
    axis.grid(axis="x", color="#D6D6D6", linewidth=0.6)
    axis.set_yticks(
        y,
        [f"{labels[d]} · {labels[n]}" for d, n in order],
    )
    axis.set_xlabel("Joint − label-only WGA (percentage points)")
    axis.set_title("Frozen 2% / 0.1 / 50% allocation", weight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    return _save(fig, output_dir, "phase11-cross-dataset-confirmation")


def build_stability_plot(output_dir: Path) -> list[Path]:
    dimensions = np.arange(5, 101)
    strengths = [0.5, 1.0, 2.0]
    fig, axis = plt.subplots(figsize=(4.0, 2.8), constrained_layout=True)
    colors = ["#3569A8", "#3A7D44", "#C47A24"]
    for strength, color in zip(strengths, colors):
        radii = [
            logistic_design_stability_radius(int(dimension), strength)
            for dimension in dimensions
        ]
        axis.plot(
            dimensions,
            radii,
            color=color,
            linewidth=1.8,
            label=rf"$\lambda={strength:g}$",
        )
    witness_radius = logistic_design_stability_radius(10, 1.0)
    axis.scatter([10], [witness_radius], color="#111111", marker="D", s=38, zorder=4)
    axis.annotate(
        f"minimal witness\n$\epsilon_\star={witness_radius:.4f}$",
        xy=(10, witness_radius),
        xytext=(22, witness_radius + 0.018),
        arrowprops={"arrowstyle": "-", "color": "#444444", "linewidth": 0.8},
        fontsize=8.5,
    )
    axis.set_xlabel("Feature dimension $d$")
    axis.set_ylabel(r"Certified spectral radius $\epsilon_\star$")
    axis.set_title("Non-orthogonal stability neighborhood", weight="bold")
    axis.grid(color="#D6D6D6", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return _save(fig, output_dir, "nonorthogonal-stability-radius")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "joint_group_label_budget" / "paper_assets",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase10 = pd.read_csv(PHASE10_SUMMARY)
    phase11 = pd.read_csv(PHASE11_SUMMARY)

    generated = []
    generated.extend(build_two_world_schematic(args.output_dir))
    generated.extend(build_phase10_map(phase10, args.output_dir))
    generated.extend(build_confirmation_forest(phase10, phase11, args.output_dir))
    generated.extend(build_stability_plot(args.output_dir))
    manifest = {
        "source_sha256": {
            str(PHASE10_SUMMARY): _sha256(PHASE10_SUMMARY),
            str(PHASE11_SUMMARY): _sha256(PHASE11_SUMMARY),
        },
        "generated": [str(path) for path in generated],
    }
    (args.output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
