"""Generate the Waterbirds full-retraining tail-sensitivity figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "tables" / "table_waterbirds_fulltrain_tau.csv"
OUTPUT = ROOT / "paper" / "figures_diagnostic" / "fig_waterbirds_fulltrain_tau_sensitivity"


def main() -> None:
    frame = pd.read_csv(DATA)
    tau = frame["tau"].to_numpy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.7, 3.05))
    series = (
        ("absolute", "vs no correction", "#1b7f5a", "o"),
        ("vs_loss", "vs Loss", "#c45a2c", "s"),
        ("vs_noise", "vs NoiseScore", "#3568a8", "^"),
    )
    for prefix, label, color, marker in series:
        mean = frame[f"{prefix}_mean"].to_numpy()
        low = frame[f"{prefix}_low"].to_numpy()
        high = frame[f"{prefix}_high"].to_numpy()
        ax.errorbar(
            tau,
            mean,
            yerr=[mean - low, high - mean],
            label=label,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.6,
            capsize=2.5,
            capthick=1.0,
        )

    ax.axhline(0, color="#606060", linewidth=0.9, linestyle="--", zorder=0)
    ax.axvline(0.35, color="#9a9a9a", linewidth=0.8, linestyle=":", zorder=0)
    ax.text(0.347, -6.15, "main setting", ha="right", va="bottom", color="#686868", fontsize=7.5)
    ax.set_xlabel(r"Validation-tail fraction $\tau$")
    ax.set_ylabel(r"Change in WGA (pp)")
    ax.set_xticks(tau)
    ax.set_xlim(0.185, 0.515)
    ax.set_ylim(-6.7, 9.3)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(pad=0.6)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
