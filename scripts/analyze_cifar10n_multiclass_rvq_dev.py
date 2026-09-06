#!/usr/bin/env python
"""Audit the exploratory CIFAR-10N multiclass RV-Q matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.absolute_utility import (
    bootstrap_mean_ci,
    exact_two_sided_sign_test,
    prepare_results,
)


EXPECTED_METHODS = (
    "random",
    "loss",
    "noise_score",
    "expected_repair_value_multiclass",
)
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_BUDGETS = (0.01, 0.02)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260829


def validate_grid(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_results(frame, dataset="CIFAR-10N")
    expected = {
        (seed, method, round(budget, 12))
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
        for budget in EXPECTED_BUDGETS
    }
    observed = {
        (int(row.seed), str(row.method), round(float(row.budget_fraction), 12))
        for row in prepared.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or len(prepared) != len(expected):
        raise ValueError(
            "Incomplete or contaminated CIFAR-10N multiclass grid: "
            f"rows={len(prepared)}, expected={len(expected)}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return prepared


def summarize(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    low, high = bootstrap_mean_ci(values, rng, BOOTSTRAP_REPLICATES)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
        "sign_test_p": exact_two_sided_sign_test(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def build_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    absolute_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []
    for budget, condition in frame.groupby("budget_fraction", sort=True):
        by_method = {
            method: condition[condition["method"] == method].set_index("seed")
            for method in EXPECTED_METHODS
        }
        for method in EXPECTED_METHODS:
            values = by_method[method]["delta_wga"].to_numpy(float)
            row = {
                "dataset": "CIFAR-10N",
                "noise_name": "cifar10n_worst",
                "budget_fraction": float(budget),
                "method": method,
                "estimand": "delta_wca_vs_no_correction",
            }
            row.update(summarize(values, rng))
            absolute_rows.append(row)
        for reference in ("loss", "noise_score"):
            joined = by_method["expected_repair_value_multiclass"][["delta_wga"]].join(
                by_method[reference][["delta_wga"]],
                lsuffix="_method",
                rsuffix="_reference",
                how="inner",
            )
            values = (
                joined["delta_wga_method"] - joined["delta_wga_reference"]
            ).to_numpy(float)
            row = {
                "dataset": "CIFAR-10N",
                "noise_name": "cifar10n_worst",
                "budget_fraction": float(budget),
                "method": "expected_repair_value_multiclass",
                "reference_method": reference,
                "estimand": "paired_delta_wca_method_minus_reference",
            }
            row.update(summarize(values, rng))
            paired_rows.append(row)
            for seed, value in zip(joined.index, values, strict=True):
                per_seed_rows.append(
                    {
                        "dataset": "CIFAR-10N",
                        "noise_name": "cifar10n_worst",
                        "seed": int(seed),
                        "budget_fraction": float(budget),
                        "method": "expected_repair_value_multiclass",
                        "reference_method": reference,
                        "paired_effect": float(value),
                    }
                )
    return (
        pd.DataFrame(absolute_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(per_seed_rows),
    )


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def write_report(
    absolute: pd.DataFrame,
    paired: pd.DataFrame,
    path: Path,
) -> None:
    labels = {
        "random": "Random",
        "loss": "Loss",
        "noise_score": "NoiseScore",
        "expected_repair_value_multiclass": "RV-Q (multiclass)",
    }
    lines = [
        "# CIFAR-10N 多分类 RV-Q 补充实验",
        "",
        "条件：CIFAR-10N `worst_label` 真实人类标注噪声，10 类；ResNet-50 冻结特征、线性头重训，"
        "10 个 training seeds（0--9），查询预算 1%/2%。",
        "这是 exploratory frozen-feature extension，不覆盖原有 full-e2e CIFAR-10N block，"
        "也不与 Waterbirds/CelebA 的 latent-group WGA 结果合并。此处的 `delta_wga` 实际表示 WCA（worst-class accuracy）变化。",
        "bootstrap 区间以 training seed 为重采样单位；sign-test 仅为描述性证据。",
        "",
        "## 绝对效果",
        "",
        "| 预算 | 方法 | ΔWCA vs no-correction (pp) | 正/负 seeds | 最差 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in absolute.itertuples(index=False):
        lines.append(
            f"| {row.budget_fraction:.0%} | {labels[row.method]} | "
            f"{_pp(row.mean)} [{_pp(row.ci_low)}, {_pp(row.ci_high)}] | "
            f"{row.positive}/{row.negative} | {_pp(row.minimum)} |"
        )
    lines.extend(
        [
            "",
            "## 多分类 RV-Q 配对比较",
            "",
            "正值表示多分类 RV-Q 优于 reference；所有比较按相同 training seed 配对。",
            "",
            "| 预算 | reference | 配对 ΔWCA (pp) | 正/负 | p |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.budget_fraction:.0%} | {labels[row.reference_method]} | "
            f"{_pp(row.mean)} [{_pp(row.ci_low)}, {_pp(row.ci_high)}] | "
            f"{row.positive}/{row.negative} | {row.sign_test_p:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "多分类 RV-Q 在本 block 中相对 Loss 为正，但相对 NoiseScore 接近持平或略低。"
            "因此该实验支持的是 C>2 形式可运行、并能在真实十分类噪声上完成合法查询与重训，"
            "不是多分类性能领先或 latent-group WGA 转移的证据。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/cifar10n_multiclass_rvq_frozen_dev/stage1/results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cifar10n_multiclass_rvq_frozen_dev/audit"),
    )
    args = parser.parse_args()
    prepared = validate_grid(pd.read_csv(args.results))
    absolute, paired, per_seed = build_summaries(prepared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    absolute.to_csv(args.output_dir / "absolute_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_summary.csv", index=False)
    per_seed.to_csv(args.output_dir / "paired_per_seed.csv", index=False)
    metadata = {
        "analysis_status": "exploratory_frozen_feature_multiclass_extension",
        "dataset": "CIFAR-10N",
        "noise_variant": "worst_label",
        "num_classes": 10,
        "metric": "worst_class_accuracy",
        "n_rows": int(len(prepared)),
        "seeds": list(EXPECTED_SEEDS),
        "budgets": list(EXPECTED_BUDGETS),
        "methods": list(EXPECTED_METHODS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_results": str(args.results),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    write_report(absolute, paired, args.output_dir / "REPORT_ZH.md")
    print(absolute.to_string(index=False))
    print("\nPaired summary")
    print(paired.to_string(index=False))
    print(f"\nAudit written to {args.output_dir}")


if __name__ == "__main__":
    main()
