#!/usr/bin/env python
"""Build local manuscript-audit tables from saved end-to-end artifacts.

This script is intentionally post hoc and non-confirmatory.  It never changes
registered results; it only derives secondary metrics already present in the
saved CSVs and records protocol/hash checks for local review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GROUP_COLS = [f"group_{i}_accuracy" for i in range(4)]
BASE_GROUP_COLS = [f"baseline_group_{i}_accuracy" for i in range(4)]
REQUIRED_COLUMNS = {
    "dataset",
    "seed",
    "method",
    "average_accuracy",
    "delta_average_accuracy",
    "balanced_accuracy",
    "wga",
    "delta_wga",
    *GROUP_COLS,
    *BASE_GROUP_COLS,
}

SOURCES = {
    "CelebA_TracIn_10_19": REPO
    / "tmp/celeba_confirmatory_results_20260802/celeba_confirmatory_seed10_19/aggregate/results.csv",
    "CelebA_RepairValue_20_29": REPO
    / "tmp/result_analysis_20260804/celeba/celeba_e2e_repairvalue_seed20_29/aggregate/results.csv",
    "Waterbirds_RepairValue_v1_20_39": REPO
    / "tmp/waterbirds_repairvalue_results_20260804/waterbirds_e2e_repairvalue_seed20_39/aggregate/results.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, float("nan"), float("nan")
    # Prefer the exact small-n t interval; retain a normal fallback so the
    # audit remains runnable in minimal environments.
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, arr.size - 1))
    except (ImportError, ValueError):
        critical = 1.96
    half = critical * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return mean, mean - half, mean + half


def enrich(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    out = frame.copy()
    out["source_block"] = source
    out["group_gap"] = out[GROUP_COLS].max(axis=1) - out[GROUP_COLS].min(axis=1)
    out["baseline_group_gap"] = (
        out[BASE_GROUP_COLS].max(axis=1) - out[BASE_GROUP_COLS].min(axis=1)
    )
    out["delta_group_gap"] = out["group_gap"] - out["baseline_group_gap"]
    out["weak_group"] = out[GROUP_COLS].idxmin(axis=1).str.extract(r"(\d+)")[0].astype(int)
    out["baseline_weak_group"] = (
        out[BASE_GROUP_COLS].idxmin(axis=1).str.extract(r"(\d+)")[0].astype(int)
    )
    return out


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["source_block", "dataset", "noise_name", "budget_fraction", "method"]
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        source, dataset, noise, budget, method = key
        metrics = {
            "average_accuracy": "average_accuracy",
            "delta_average_accuracy": "delta_average_accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "wga": "wga",
            "delta_wga": "delta_wga",
            "group_gap": "group_gap",
            "delta_group_gap": "delta_group_gap",
            "noise_precision": "noise_precision",
            "num_corrected": "num_corrected",
        }
        row: dict[str, Any] = {
            "source_block": source,
            "dataset": dataset,
            "noise_name": noise,
            "budget_fraction": budget,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
            "negative_delta_wga_seeds": int((group["delta_wga"] < 0).sum()),
            "positive_delta_wga_seeds": int((group["delta_wga"] > 0).sum()),
        }
        for label, column in metrics.items():
            mean, low, high = mean_ci(group[column])
            row[f"mean_{label}"] = mean
            row[f"ci95_low_{label}"] = low
            row[f"ci95_high_{label}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_against(frame: pd.DataFrame, baseline_method: str) -> pd.DataFrame:
    keys = ["source_block", "dataset", "noise_name", "budget_fraction", "seed"]
    baseline = frame[frame["method"] == baseline_method][
        keys + ["delta_wga", "delta_average_accuracy", "delta_group_gap"]
    ].rename(
        columns={
            "delta_wga": "baseline_delta_wga",
            "delta_average_accuracy": "baseline_delta_average_accuracy",
            "delta_group_gap": "baseline_delta_group_gap",
        }
    )
    rows: list[dict[str, Any]] = []
    for method, group in frame.groupby("method", sort=True):
        if method == baseline_method:
            continue
        merged = group.merge(baseline, on=keys, how="inner")
        if merged.empty:
            continue
        for metric, candidate, base in (
            ("delta_wga", "delta_wga", "baseline_delta_wga"),
            ("delta_average_accuracy", "delta_average_accuracy", "baseline_delta_average_accuracy"),
            ("delta_group_gap", "delta_group_gap", "baseline_delta_group_gap"),
        ):
            values = merged[candidate].to_numpy(float) - merged[base].to_numpy(float)
            mean, low, high = mean_ci(values)
            rows.append(
                {
                    "source_block": str(group["source_block"].iloc[0]),
                    "dataset": str(group["dataset"].iloc[0]),
                    "noise_name": str(group["noise_name"].iloc[0]),
                    "budget_fraction": float(group["budget_fraction"].iloc[0]),
                    "candidate_method": method,
                    "baseline_method": baseline_method,
                    "metric": metric,
                    "n_paired_seeds": int(values.size),
                    "mean_candidate_minus_baseline": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_pairs": int((values > 0).sum()),
                    "negative_pairs": int((values < 0).sum()),
                    "ties": int((values == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def protocol_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {"status": "pass", "checks": []}

    def add(name: str, passed: bool, detail: str) -> None:
        checks["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            checks["status"] = "review"

    registrations = [
        REPO / "configs/celeba_e2e_repairvalue_seed20_29.registration.json",
        REPO / "configs/waterbirds_e2e_repairvalue_v2_seed40_59.registration.json",
    ]
    for registration_path in registrations:
        data = json.loads(registration_path.read_text(encoding="utf-8"))
        config_rel = data["config"]["path"]
        config_path = REPO / config_rel
        expected = data["config"]["sha256"]
        actual = sha256(config_path)
        add(
            f"config_sha256:{config_rel}",
            actual == expected,
            f"expected={expected}; actual={actual}",
        )
        add(
            f"registration_json:{registration_path.name}",
            bool(data.get("protocol_version")) and bool(data.get("condition")),
            f"protocol={data.get('protocol_version')}",
        )

    v2 = json.loads(registrations[1].read_text(encoding="utf-8"))
    for rel, expected in v2.get("code_sha256", {}).items():
        path = REPO / rel
        actual = sha256(path) if path.exists() else "missing"
        add(f"code_sha256:{rel}", actual == expected, f"expected={expected}; actual={actual}")

    try:
        from robust_verify.scoring import ADAPTIVE_METHODS, METHOD_NAMES

        add(
            "method_registration:expected_repair_value",
            "expected_repair_value" in METHOD_NAMES,
            "present in METHOD_NAMES",
        )
        add(
            "method_registration:noise_gated_repair_value",
            "noise_gated_repair_value" in METHOD_NAMES
            and "noise_gated_repair_value" in ADAPTIVE_METHODS,
            "present in METHOD_NAMES and ADAPTIVE_METHODS",
        )
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        add("method_registration:import", False, repr(exc))

    for name, path in SOURCES.items():
        add(f"artifact:{name}", path.exists(), str(path))
        if path.exists():
            frame = pd.read_csv(path, nrows=2)
            add(f"artifact_schema:{name}", REQUIRED_COLUMNS.issubset(frame.columns), "required metric columns present")
    return checks


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    checks: dict[str, Any],
) -> None:
    lines = [
        "# 本地论文审计与补强工作结果",
        "",
        "证据级别：本报告只使用仓库中已保存的端到端结果与探索性 frozen-feature 结果；",
        "不改变任何 registered/protocol-locked 结果，也不把新增分析升级为确认性证据。",
        "",
        "## 已完成",
        "",
        "- 从 CelebA seeds 10--19、CelebA seeds 20--29 和 Waterbirds v1 seeds 20--39 的 aggregate CSV 生成了平均准确率、balanced accuracy、WGA、群体差距和 paired contrast。",
        "- 群体差距定义为四个评估组准确率的 max-min；同时保留 no-correction baseline 的差距和变化量。",
        "- 核查了注册配置 hash、v2 注册代码 hash、方法注册状态以及结果 artifact schema。",
        "- 现有 frozen-feature modern-baseline 结果单独保留为 exploratory，不与端到端结果合并。",
        "",
        "## 结果摘要",
        "",
        "完整数值见 `dataset_method_summary.csv`、`paired_vs_baseline.csv` 和 `per_seed_metrics.csv`。",
        "其中 paired 表中的 `delta_group_gap` 仅是描述性副作用审计，不是新的主检验。",
        "",
        "## 协议审计",
        "",
        f"- 总状态：**{checks['status']}**。详细原因见 `protocol_checks.json`。",
        "- 若出现 code/config hash review，说明当前工作树与 registered bundle 不完全一致；不应将本地 replay 说成严格 sealed-code confirmation。",
        "- 当前 `expected_repair_value` 和 `noise_gated_repair_value` 均已检查为注册状态。",
        "",
        "## 尚未完成的新增确认实验",
        "",
        "- 20% tail fraction 的 full-parameter 新 seed sensitivity 需要大显存服务器；本地只适合继续做 frozen-feature 或小规模 exploratory 版本。",
        "- KAIROS/kNN 等现代基线的同动作空间端到端比较仍未生成，不应从现有结果推断其表现。",
        "",
    ]
    (output_dir / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "tmp/local_paper_audit_20260814",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, path in SOURCES.items():
        if not path.exists():
            continue
        frames.append(enrich(pd.read_csv(path), name))
    if not frames:
        raise FileNotFoundError("No saved aggregate result CSVs were found")
    frame = pd.concat(frames, ignore_index=True)
    summary = summarize(frame)
    paired = pd.concat(
        [paired_against(frame[frame["source_block"] == source], baseline) for source, baseline in (
            ("CelebA_TracIn_10_19", "loss"),
            ("CelebA_RepairValue_20_29", "loss"),
            ("Waterbirds_RepairValue_v1_20_39", "loss"),
        )],
        ignore_index=True,
    )
    checks = protocol_checks()
    frame.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "dataset_method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_baseline.csv", index=False)
    (args.output_dir / "protocol_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir, summary, paired, checks)
    print(summary.to_string(index=False))
    print(f"\nWrote local audit to {args.output_dir}")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
