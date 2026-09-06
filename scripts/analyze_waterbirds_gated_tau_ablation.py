#!/usr/bin/env python
"""Summarize the Waterbirds RV-Q-Gated tau ablation.

The analysis is seed-level: all intervals are percentile bootstrap intervals
over the 20 fixed seeds. By default it refuses incomplete or mixed-source
blocks, so a partial local run cannot accidentally become a paper table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAUS = (0.20, 0.30, 0.35, 0.40, 0.50)
SEEDS = tuple(range(40, 60))
BOOTSTRAP_REPLICATES = 10_000
RNG_SEED = 20260903


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap input must be a non-empty vector")
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _fmt_effect(mean: float, low: float, high: float) -> str:
    return f"${mean:+.2f}\\;[{low:+.2f},{high:+.2f}]$"


def _positive_count(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    positive = int(np.sum(values > 1e-12))
    negative = int(np.sum(values < -1e-12))
    tied = int(len(values) - positive - negative)
    return f"{positive}/{negative}/{tied}"


def _load(path: Path, *, allow_incomplete: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "seed",
        "tau",
        "source",
        "delta_wga",
        "paired_vs_loss_wga",
        "paired_vs_noise_score_wga",
        "noise_precision",
        "wga",
        "retrain_validation_metric",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    frame["seed"] = frame["seed"].astype(int)
    frame["tau"] = frame["tau"].astype(float).round(2)
    frame = frame[frame["seed"].isin(SEEDS) & frame["tau"].isin(TAUS)].copy()
    if frame.duplicated(["seed", "tau"]).any():
        raise ValueError("Duplicate seed/tau rows in ablation results")
    expected = {(seed, tau) for seed in SEEDS for tau in TAUS}
    actual = set(zip(frame["seed"], frame["tau"]))
    if not allow_incomplete and actual != expected:
        raise ValueError(f"Incomplete ablation: missing {sorted(expected - actual)}")
    if not allow_incomplete:
        sources = {
            tau: set(frame.loc[frame["tau"] == tau, "source"].astype(str))
            for tau in TAUS
        }
        if any(len(values) != 1 for values in sources.values()):
            raise ValueError(f"Each tau must have one source; got {sources}")
        if len({next(iter(values)) for values in sources.values()}) != 1:
            raise ValueError(f"Mixed sources across tau arms: {sources}")
    return frame.sort_values(["tau", "seed"]).reset_index(drop=True)


def _write_outputs(frame: pd.DataFrame, output_dir: Path, figure_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict] = []
    for tau in TAUS:
        block = frame[frame["tau"] == tau].sort_values("seed")
        if block.empty:
            continue
        metrics = {
            "absolute": block["delta_wga"].to_numpy(dtype=float) * 100.0,
            "vs_loss": block["paired_vs_loss_wga"].to_numpy(dtype=float) * 100.0,
            "vs_noise": block["paired_vs_noise_score_wga"].to_numpy(dtype=float) * 100.0,
        }
        row: dict[str, object] = {"tau": tau, "n_seeds": len(block)}
        for prefix, values in metrics.items():
            low, high = _bootstrap_interval(values, rng)
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_low"] = low
            row[f"{prefix}_high"] = high
            row[f"{prefix}_positive"] = _positive_count(values)
        row["precision_mean"] = float(block["noise_precision"].mean() * 100.0)
        row["precision_low"], row["precision_high"] = (
            value * 100.0
            for value in _bootstrap_interval(
                block["noise_precision"].to_numpy(dtype=float), rng
            )
        )
        row["validation_balanced_accuracy_mean"] = float(
            block["retrain_validation_metric"].mean()
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "waterbirds_gated_tau_summary.csv", index=False)
    selected_tau = float(
        summary.sort_values(
            ["validation_balanced_accuracy_mean", "tau"], ascending=[False, True]
        ).iloc[0]["tau"]
    )
    (output_dir / "waterbirds_gated_tau_summary.json").write_text(
        json.dumps(
            {
                "taus": list(TAUS),
                "seeds": list(SEEDS),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "rng_seed": RNG_SEED,
                "source": sorted(frame["source"].astype(str).unique()),
                "selection_rule": "highest mean frozen-feature linear-head validation balanced accuracy; test WGA not used",
                "selected_tau": selected_tau,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tex_lines = [
        r"\begin{table}[H]",
        r"\caption{Waterbirds RV-Q-Gated frozen-feature linear-head tail-fraction sensitivity on seeds 40--59. Effects are percentage-point means with 95\% seed-bootstrap intervals. Positive counts are reported as absolute/Loss/NoiseScore.}",
        r"\label{tab:waterbirds-gated-tau}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$\tau$ & Absolute $\dwga$ & RV-Q-Gated $-$ Loss & RV-Q-Gated $-$ NoiseScore & Positive seeds \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        tex_lines.append(
            f"{row.tau:.2f} & "
            f"{_fmt_effect(row.absolute_mean, row.absolute_low, row.absolute_high)} & "
            f"{_fmt_effect(row.vs_loss_mean, row.vs_loss_low, row.vs_loss_high)} & "
            f"{_fmt_effect(row.vs_noise_mean, row.vs_noise_low, row.vs_noise_high)} & "
            f"{row.absolute_positive.split('/')[0]}/"
            f"{row.vs_loss_positive.split('/')[0]}/"
            f"{row.vs_noise_positive.split('/')[0]} "
            + r"\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output_dir / "appendix_waterbirds_gated_tau.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )
    (output_dir / "selected_tau.txt").write_text(
        f"{selected_tau:.2f}\n", encoding="utf-8"
    )
    selected = summary[summary["tau"] == selected_tau].iloc[0]
    main_row = (
        f"RV-Q-Gated, 40--59 ($\\tau={selected_tau:.2f}$, anchor $=.50$) & "
        f"NoiseScore: {_fmt_effect(selected.vs_noise_mean, selected.vs_noise_low, selected.vs_noise_high)} & "
        f"{_fmt_effect(selected.absolute_mean, selected.absolute_low, selected.absolute_high)} & "
        f"{selected.precision_mean:.2f}\\% & {selected.vs_noise_positive}"
        " \\\\\n"
    )
    (output_dir / "main_waterbirds_gated_tau.tex").write_text(
        main_row, encoding="utf-8"
    )

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
    taus = summary["tau"].to_numpy(dtype=float)
    for prefix, label, color, marker in series:
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        low = summary[f"{prefix}_low"].to_numpy(dtype=float)
        high = summary[f"{prefix}_high"].to_numpy(dtype=float)
        ax.errorbar(
            taus,
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
    ax.set_xlabel(r"Validation-tail fraction $\tau$")
    ax.set_ylabel(r"Change in WGA (pp)")
    ax.set_xticks(taus)
    ax.set_xlim(0.185, 0.515)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(pad=0.6)
    fig.savefig(figure_dir / "fig_waterbirds_gated_tau_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_waterbirds_gated_tau_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(
            ROOT
            / "outputs"
            / "waterbirds_gated_tau_ablation_seed40_59_frozen"
            / "ablation_results.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "paper" / "tables" / "waterbirds_gated_tau_ablation"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(ROOT / "paper" / "figures_diagnostic"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    frame = _load(Path(args.results).resolve(), allow_incomplete=args.allow_incomplete)
    _write_outputs(
        frame,
        Path(args.output_dir).resolve(),
        Path(args.figure_dir).resolve(),
    )
    print(f"wrote outputs for {len(frame)} rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
