#!/usr/bin/env python
"""Audit the locked local CelebA modern-baseline diagnostic.

This analysis is intentionally descriptive/exploratory.  It validates the
complete seed-by-method grid, keeps seeds as the resampling unit, and reports
both absolute WGA change versus no correction and paired change versus Loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.absolute_utility import (
    audit_absolute_utility,
    bootstrap_mean_ci,
    exact_two_sided_sign_test,
    prepare_results,
)
from robust_verify.scoring import METHOD_LABELS


EXPECTED_METHODS = (
    "loss",
    "tracin_val_uncertainty",
    "tracin_multicheckpoint_val",
    "expected_repair_value",
    "auto_d3m_query",
)
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_NOISE = "uniform20"
EXPECTED_BUDGET = 0.10
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20260731


def validate_grid(frame: pd.DataFrame) -> None:
    expected = {
        (EXPECTED_NOISE, seed, method, EXPECTED_BUDGET)
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
    }
    observed = {
        (str(row.noise_name), int(row.seed), str(row.method), round(float(row.budget_fraction), 12))
        for row in frame.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or len(frame) != len(expected):
        raise ValueError(
            "Incomplete or contaminated local diagnostic grid: "
            f"rows={len(frame)}, expected={len(expected)}, missing={missing[:5]}, extra={extra[:5]}"
        )


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (method, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[method] = running
    return adjusted


def build_method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    loss = frame[frame["method"] == "loss"].set_index("seed")["delta_wga"]
    rows: list[dict[str, object]] = []
    raw_relative_p: dict[str, float] = {}
    relative_values: dict[str, np.ndarray] = {}

    for method in EXPECTED_METHODS:
        group = frame[frame["method"] == method].sort_values("seed")
        absolute = group["delta_wga"].to_numpy(float)
        absolute_ci = bootstrap_mean_ci(absolute, rng, BOOTSTRAP_REPLICATES)
        if method == "loss":
            relative = np.zeros(len(group), dtype=float)
            relative_ci = (0.0, 0.0)
            raw_relative_p[method] = 1.0
        else:
            indexed = group.set_index("seed")["delta_wga"]
            if not indexed.index.equals(loss.index):
                raise ValueError(f"Seed pairing against Loss failed for {method}")
            relative = (indexed - loss).to_numpy(float)
            relative_ci = bootstrap_mean_ci(relative, rng, BOOTSTRAP_REPLICATES)
            raw_relative_p[method] = exact_two_sided_sign_test(relative)
        relative_values[method] = relative
        rows.append(
            {
                "method": method,
                "display_name": METHOD_LABELS.get(method, method),
                "n_seeds": len(group),
                "mean_delta_wga": float(absolute.mean()),
                "absolute_ci_low": absolute_ci[0],
                "absolute_ci_high": absolute_ci[1],
                "absolute_negative_seeds": int((absolute < 0).sum()),
                "absolute_positive_seeds": int((absolute > 0).sum()),
                "mean_delta_wga_vs_loss": float(relative.mean()),
                "relative_ci_low": relative_ci[0],
                "relative_ci_high": relative_ci[1],
                "relative_negative_seeds": int((relative < 0).sum()),
                "relative_positive_seeds": int((relative > 0).sum()),
                "relative_sign_p_raw": raw_relative_p[method],
                "mean_noise_precision": float(group["noise_precision"].mean()),
                "mean_num_corrected": float(group["num_corrected"].mean()),
                "mean_minority_query_rate": float(group["minority_query_rate"].mean()),
                "mean_wga": float(group["wga"].mean()),
                "mean_baseline_wga": float(group["baseline_wga"].mean()),
            }
        )

    adjusted = _holm_adjust(
        {method: value for method, value in raw_relative_p.items() if method != "loss"}
    )
    summary = pd.DataFrame(rows)
    summary["relative_sign_p_holm"] = summary["method"].map(adjusted)
    summary.loc[summary["method"] == "loss", "relative_sign_p_holm"] = 1.0
    return summary


def _pct(value: float, *, signed: bool = True) -> str:
    pattern = "+.2f" if signed else ".2f"
    return format(100.0 * value, pattern)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# CelebA 本地现代基线审计（探索性）",
        "",
        "条件：ResNet-50 冻结特征、uniform-20% 标签噪声、10% 查询预算、seed 0--9。",
        "区间为以外层 seed 为重采样单位的 50,000 次 percentile bootstrap 95% CI。",
        "本结果不属于 seeds 10--19 的端到端确认性检验。AUTO-D3M-Q 是标签查询动作空间适配，",
        "不是原论文 100-trial 端到端 AUTO-D3M 的精确复现。",
        "",
        "| 方法 | ΔWGA vs 不修正 (pp) | 配对 vs Loss (pp) | 噪声精度 | 少数组查询率 | 负收益 seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        absolute = (
            f"{_pct(row.mean_delta_wga)} "
            f"[{_pct(row.absolute_ci_low)}, {_pct(row.absolute_ci_high)}]"
        )
        relative = "--" if row.method == "loss" else (
            f"{_pct(row.mean_delta_wga_vs_loss)} "
            f"[{_pct(row.relative_ci_low)}, {_pct(row.relative_ci_high)}]"
        )
        lines.append(
            f"| {row.display_name} | {absolute} | {relative} | "
            f"{_pct(row.mean_noise_precision, signed=False)}% | "
            f"{_pct(row.mean_minority_query_rate, signed=False)}% | "
            f"{row.absolute_negative_seeds}/{row.n_seeds} |"
        )
    lines.extend(
        [
            "",
            "解释边界：这是固定特征上的机制筛查。只有跨 seed 一致、且端到端确认实验复现的方向，",
            "才应升级为主文结论。显著性数值为描述性证据；四个对 Loss 的比较另给 exact sign-test",
            "及 Holm 校正结果（见 `method_summary.csv`）。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "loss": "Loss",
        "tracin_val_uncertainty": r"TracIn-CP (single)",
        "tracin_multicheckpoint_val": r"TracIn-CP (multi)",
        "expected_repair_value": r"RepairValue (tail)",
        "auto_d3m_query": r"AUTO-D3M-Q (adapt.)",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & $\Delta$WGA (pp) & vs. Loss (pp) & Noise prec. (\%) & Neg. seeds \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        absolute = f"{_pct(row.mean_delta_wga)} [{_pct(row.absolute_ci_low)}, {_pct(row.absolute_ci_high)}]"
        relative = "--" if row.method == "loss" else (
            f"{_pct(row.mean_delta_wga_vs_loss)} [{_pct(row.relative_ci_low)}, {_pct(row.relative_ci_high)}]"
        )
        lines.append(
            f"{labels[row.method]} & {absolute} & {relative} & "
            f"{_pct(row.mean_noise_precision, signed=False)} & "
            f"{row.absolute_negative_seeds}/{row.n_seeds} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/results.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/local_celeba_modern_baselines_uniform20_10pct/audit"
        ),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=Path("paper/tables/table_local_modern_baselines.tex"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.results)
    validate_grid(raw)
    prepared = prepare_results(raw, dataset="CelebA")
    audit = audit_absolute_utility(
        prepared,
        include_methods=list(EXPECTED_METHODS),
        n_bootstrap=BOOTSTRAP_REPLICATES,
        rng_seed=BOOTSTRAP_SEED,
    )
    summary = build_method_summary(prepared)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    audit.absolute_vs_no_correction.to_csv(
        args.output_dir / "absolute_vs_no_correction.csv", index=False
    )
    audit.paired_vs_loss.to_csv(args.output_dir / "paired_vs_loss.csv", index=False)
    audit.per_seed_absolute.to_csv(args.output_dir / "per_seed_absolute.csv", index=False)
    audit.per_seed_vs_loss.to_csv(args.output_dir / "per_seed_vs_loss.csv", index=False)
    audit.coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    metadata = {
        **audit.metadata,
        "analysis_status": "exploratory_frozen_feature_diagnostic",
        "expected_noise": EXPECTED_NOISE,
        "expected_budget_fraction": EXPECTED_BUDGET,
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_methods": list(EXPECTED_METHODS),
        "auto_d3m_scope": "query-space adaptation; not exact 100-trial end-to-end reproduction",
    }
    (args.output_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, args.output_dir / "REPORT_ZH.md")
    write_latex(summary, args.latex_output)
    print(summary.to_string(index=False))
    print(f"Audit written to {args.output_dir}")


if __name__ == "__main__":
    main()
