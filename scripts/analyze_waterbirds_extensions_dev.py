#!/usr/bin/env python
"""Summarize the exploratory Waterbirds extension matrix.

This report keeps the outer corruption seed as the resampling unit.  It is
deliberately separate from the registered Waterbirds confirmation blocks.
"""

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


EXPECTED_NOISE = ("uniform20", "minority_high_40")
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_BUDGETS = (0.02, 0.05)
EXPECTED_METHODS = (
    "random",
    "loss",
    "noise_score",
    "expected_repair_value",
    "expected_repair_value_multiclass",
    "reliability_gated_repair_value",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260829


def validate_grid(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_results(frame, dataset="Waterbirds")
    expected = {
        (noise, seed, method, round(budget, 12))
        for noise in EXPECTED_NOISE
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
        for budget in EXPECTED_BUDGETS
    }
    observed = {
        (
            str(row.noise_name),
            int(row.seed),
            str(row.method),
            round(float(row.budget_fraction), 12),
        )
        for row in prepared.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or len(prepared) != len(expected):
        raise ValueError(
            "Incomplete or contaminated Waterbirds extension grid: "
            f"rows={len(prepared)}, expected={len(expected)}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return prepared


def _summary(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
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
    method_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    for (noise_name, budget), condition in frame.groupby(
        ["noise_name", "budget_fraction"], sort=True
    ):
        condition = condition.sort_values("seed")
        by_method = {
            method: condition[condition["method"] == method].set_index("seed")
            for method in EXPECTED_METHODS
        }
        for method in EXPECTED_METHODS:
            values = by_method[method]["delta_wga"].to_numpy(float)
            row = {
                "noise_name": noise_name,
                "budget_fraction": float(budget),
                "method": method,
                "estimand": "delta_wga_vs_no_correction",
            }
            row.update(_summary(values, rng))
            method_rows.append(row)

        for method, reference in (
            ("expected_repair_value_multiclass", "expected_repair_value"),
            ("reliability_gated_repair_value", "noise_score"),
            ("reliability_gated_repair_value", "expected_repair_value"),
        ):
            joined = by_method[method][["delta_wga"]].join(
                by_method[reference][["delta_wga"]],
                lsuffix="_method",
                rsuffix="_reference",
                how="inner",
            )
            values = (
                joined["delta_wga_method"] - joined["delta_wga_reference"]
            ).to_numpy(float)
            row = {
                "noise_name": noise_name,
                "budget_fraction": float(budget),
                "method": method,
                "reference_method": reference,
                "estimand": "paired_delta_wga_method_minus_reference",
            }
            row.update(_summary(values, rng))
            paired_rows.append(row)

        gate = by_method["reliability_gated_repair_value"]
        for column in (
            "gate_alpha",
            "gate_reliability",
            "gate_stability_alpha",
            "gate_retention_alpha",
            "gate_noise_retention",
            "gate_reference_budget_count",
            "gate_num_folds",
        ):
            gate[column] = pd.to_numeric(gate[column], errors="coerce")
        fallback = gate["gate_fallback"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        gate_rows.append(
            {
                "noise_name": noise_name,
                "budget_fraction": float(budget),
                "n": int(len(gate)),
                "mean_alpha": float(gate["gate_alpha"].mean()),
                "median_alpha": float(gate["gate_alpha"].median()),
                "min_alpha": float(gate["gate_alpha"].min()),
                "max_alpha": float(gate["gate_alpha"].max()),
                "mean_reliability": float(gate["gate_reliability"].mean()),
                "mean_stability_alpha": float(gate["gate_stability_alpha"].mean()),
                "mean_retention_alpha": float(gate["gate_retention_alpha"].mean()),
                "mean_noise_retention": float(gate["gate_noise_retention"].mean()),
                "fallback_count": int(fallback.sum()),
                "zero_alpha_count": int((gate["gate_alpha"] <= 1e-12).sum()),
            }
        )
    return pd.DataFrame(method_rows), pd.DataFrame(paired_rows), pd.DataFrame(gate_rows)


def query_equivalence(results_path: Path) -> pd.DataFrame:
    query_root = results_path.parent / "queries"
    rows: list[dict[str, object]] = []
    for noise in EXPECTED_NOISE:
        for seed in EXPECTED_SEEDS:
            for budget in EXPECTED_BUDGETS:
                folder = query_root / f"{noise}_seed{seed}"
                suffix = f"budget_{budget:.4f}.json"
                binary_path = folder / f"expected_repair_value_{suffix}"
                multiclass_path = folder / f"expected_repair_value_multiclass_{suffix}"
                if not binary_path.is_file() or not multiclass_path.is_file():
                    raise FileNotFoundError(f"Missing query artifacts for {noise}, seed {seed}, budget {budget}")
                binary = json.loads(binary_path.read_text(encoding="utf-8"))
                multiclass = json.loads(multiclass_path.read_text(encoding="utf-8"))
                binary_ids = [int(value) for value in binary["sample_ids"]]
                multiclass_ids = [int(value) for value in multiclass["sample_ids"]]
                rows.append(
                    {
                        "noise_name": noise,
                        "seed": seed,
                        "budget_fraction": budget,
                        "binary_count": len(binary_ids),
                        "multiclass_count": len(multiclass_ids),
                        "query_equal": binary_ids == multiclass_ids,
                        "binary_sha256": _ids_digest(binary_ids),
                        "multiclass_sha256": _ids_digest(multiclass_ids),
                    }
                )
    return pd.DataFrame(rows)


def _ids_digest(values: list[int]) -> str:
    import hashlib

    payload = ",".join(str(value) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def write_report(
    method_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    equivalence: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Waterbirds 扩展开发性评估",
        "",
        "条件：固定特征、5 epoch、10 个 outer seeds（0--9）、uniform20 与 minority_high_40，查询预算 2%/5%。",
        "该矩阵是开发/机制筛查，不属于已注册的 Waterbirds confirmation block，也不应与正式结果合并。",
        "bootstrap 区间以 outer seed 为重采样单位；p 值仅作描述性 sign-test。",
        "",
        "## 绝对效果（相对 no-correction 的 ΔWGA，百分点）",
        "",
        "| 噪声 | 预算 | 方法 | 均值 [95% CI] | 正/负 seeds | 最差 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    labels = {
        "random": "Random",
        "loss": "Loss",
        "noise_score": "NoiseScore",
        "expected_repair_value": "RV-Q",
        "expected_repair_value_multiclass": "RV-Q (multiclass)",
        "reliability_gated_repair_value": "RV-Q (gated)",
    }
    for row in method_summary.itertuples(index=False):
        lines.append(
            f"| {row.noise_name} | {row.budget_fraction:.0%} | {labels[row.method]} | "
            f"{_pp(row.mean)} [{_pp(row.ci_low)}, {_pp(row.ci_high)}] | "
            f"{row.positive}/{row.negative} | {_pp(row.minimum)} |"
        )
    lines.extend(
        [
            "",
            "## 配对效果",
            "",
            "正值表示方法优于 reference；区间和 sign-test 都以 seed 为配对单位。",
            "",
            "| 噪声 | 预算 | 方法 | reference | 均值 [95% CI] | 正/负 | p |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in paired_summary.itertuples(index=False):
        lines.append(
            f"| {row.noise_name} | {row.budget_fraction:.0%} | {labels[row.method]} | "
            f"{labels[row.reference_method]} | {_pp(row.mean)} [{_pp(row.ci_low)}, {_pp(row.ci_high)}] | "
            f"{row.positive}/{row.negative} | {row.sign_test_p:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 门控行为",
            "",
            "| 噪声 | 预算 | mean α | α 范围 | mean reliability | mean retention | fallback | α=0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_summary.itertuples(index=False):
        lines.append(
            f"| {row.noise_name} | {row.budget_fraction:.0%} | {row.mean_alpha:.3f} | "
            f"[{row.min_alpha:.3f}, {row.max_alpha:.3f}] | {row.mean_reliability:.3f} | "
            f"{row.mean_noise_retention:.3f} | {row.fallback_count}/{row.n} | "
            f"{row.zero_alpha_count}/{row.n} |"
        )
    equal_count = int(equivalence["query_equal"].sum())
    lines.extend(
        [
            "",
            "## 多分类退化校验",
            "",
            f"RV-Q (multiclass) 与二分类 RV-Q 的 query artifact 在 {equal_count}/{len(equivalence)} 个 "
            "noise×seed×budget 条件下完全一致。该结果证明实现的二分类退化关系，不等同于多分类任务上的性能证据。",
            "",
            "## 结论边界",
            "",
            "门控是否值得进入 ResNet 端到端确认实验，应看配对 ΔWGA 是否跨噪声和预算保持非负，尤其是 minority_high_40 的 5% 条件。",
            "本报告不把开发性提升转写为主文结论；后续若升级实验，需锁定 gate 超参并使用新的 confirmation seeds。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/waterbirds_extensions_dev/stage1/results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/waterbirds_extensions_dev/audit"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.results)
    prepared = validate_grid(raw)
    method_summary, paired_summary, gate_summary = build_summaries(prepared)
    equivalence = query_equivalence(args.results)
    if not bool(equivalence["query_equal"].all()):
        raise AssertionError("Multiclass and binary RV-Q query artifacts differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    paired_summary.to_csv(args.output_dir / "paired_summary.csv", index=False)
    gate_summary.to_csv(args.output_dir / "gate_summary.csv", index=False)
    equivalence.to_csv(args.output_dir / "query_equivalence.csv", index=False)
    metadata = {
        "analysis_status": "exploratory_waterbirds_extension",
        "dataset": "Waterbirds",
        "n_rows": int(len(prepared)),
        "noise_settings": list(EXPECTED_NOISE),
        "seeds": list(EXPECTED_SEEDS),
        "budgets": list(EXPECTED_BUDGETS),
        "methods": list(EXPECTED_METHODS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "query_equivalence_count": int(equivalence["query_equal"].sum()),
        "query_equivalence_total": int(len(equivalence)),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    write_report(method_summary, paired_summary, gate_summary, equivalence, args.output_dir / "REPORT_ZH.md")
    print(method_summary.to_string(index=False))
    print("\nPaired summary")
    print(paired_summary.to_string(index=False))
    print("\nGate summary")
    print(gate_summary.to_string(index=False))
    print(f"\nQuery equality: {int(equivalence['query_equal'].sum())}/{len(equivalence)}")
    print(f"Audit written to {args.output_dir}")


if __name__ == "__main__":
    main()
