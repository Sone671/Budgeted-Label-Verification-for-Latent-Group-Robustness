#!/usr/bin/env python
"""Run and summarize an exploratory frozen-feature tail-fraction screen.

The screen reuses the already prepared CelebA seed 0--9 features/manifests and
does not touch any registered end-to-end output.  Its results are explicitly
exploratory and are intended to answer the reviewer question about sensitivity
to the 20% validation tail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from robust_verify.stage1 import run_stage1

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs/celeba_local_modern_baselines_uniform20_10pct.yaml"
EXISTING_020 = REPO / "outputs/local_celeba_modern_baselines_uniform20_10pct"


def config_for_fraction(base: dict, fraction: float) -> dict:
    config = yaml.safe_load(yaml.safe_dump(base))
    config["project"]["output_dir"] = str(
        REPO / f"outputs/local_celeba_tail_sensitivity_f{round(fraction * 100):02d}"
    )
    config["experiment"]["baseline_options"]["repair_value_tail_fraction"] = float(fraction)
    config["experiment"]["methods"] = ["loss", "expected_repair_value"]
    config["experiment"]["analysis_status"] = "exploratory_frozen_feature_tail_sensitivity"
    return config


def run_fraction(base: dict, fraction: float) -> Path:
    config = config_for_fraction(base, fraction)
    output = Path(config["project"]["output_dir"])
    result_path = output / "stage1/results.csv"
    if not result_path.exists():
        run_stage1(config)
    return result_path


def load_result(path: Path, fraction: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["method"].isin(["loss", "expected_repair_value"])].copy()
    frame["tail_fraction"] = float(fraction)
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fraction, method), group in frame.groupby(["tail_fraction", "method"], sort=True):
        values = group["delta_wga"].to_numpy(float)
        rows.append(
            {
                "tail_fraction": fraction,
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
                "mean_delta_wga_pp": float(values.mean() * 100.0),
                "std_delta_wga_pp": float(values.std(ddof=1) * 100.0),
                "positive_delta_wga_seeds": int((values > 0).sum()),
                "negative_delta_wga_seeds": int((values < 0).sum()),
                "mean_delta_average_accuracy_pp": float(
                    group["delta_average_accuracy"].mean() * 100.0
                ),
                "mean_noise_precision_pct": float(group["noise_precision"].mean() * 100.0),
                "mean_group_gap_pp": float(
                    (
                        group[[f"group_{i}_accuracy" for i in range(4)]].max(axis=1)
                        - group[[f"group_{i}_accuracy" for i in range(4)]].min(axis=1)
                    ).mean()
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    loss = frame[frame["method"] == "loss"][
        ["tail_fraction", "seed", "delta_wga", "delta_average_accuracy"]
    ].rename(
        columns={
            "delta_wga": "loss_delta_wga",
            "delta_average_accuracy": "loss_delta_average_accuracy",
        }
    )
    rv = frame[frame["method"] == "expected_repair_value"].merge(
        loss, on=["tail_fraction", "seed"], how="inner"
    )
    rv["rv_minus_loss_wga_pp"] = (rv["delta_wga"] - rv["loss_delta_wga"]) * 100.0
    rv["rv_minus_loss_avg_acc_pp"] = (
        rv["delta_average_accuracy"] - rv["loss_delta_average_accuracy"]
    ) * 100.0
    return rv[
        [
            "tail_fraction",
            "seed",
            "rv_minus_loss_wga_pp",
            "rv_minus_loss_avg_acc_pp",
            "noise_precision",
            "delta_wga",
            "delta_average_accuracy",
        ]
    ]


def query_overlap(fractions: list[float]) -> pd.DataFrame:
    rows = []
    roots = {
        0.20: EXISTING_020,
        **{
            f: REPO / f"outputs/local_celeba_tail_sensitivity_f{round(f * 100):02d}"
            for f in fractions
        },
    }
    seeds = range(10)
    for left in sorted(roots):
        for right in sorted(roots):
            if left >= right:
                continue
            values = []
            for seed in seeds:
                left_path = (
                    roots[left]
                    / "stage1/queries"
                    / f"uniform20_seed{seed}"
                    / "expected_repair_value_budget_0.1000.npy"
                )
                right_path = (
                    roots[right]
                    / "stage1/queries"
                    / f"uniform20_seed{seed}"
                    / "expected_repair_value_budget_0.1000.npy"
                )
                if not left_path.exists() or not right_path.exists():
                    continue
                left_ids = set(np.load(left_path).tolist())
                right_ids = set(np.load(right_path).tolist())
                values.append(len(left_ids & right_ids) / max(1, len(left_ids | right_ids)))
            rows.append(
                {
                    "left_tail_fraction": left,
                    "right_tail_fraction": right,
                    "paired_seed_count": len(values),
                    "mean_query_jaccard": float(np.mean(values)) if values else float("nan"),
                    "min_query_jaccard": float(np.min(values)) if values else float("nan"),
                    "max_query_jaccard": float(np.max(values)) if values else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.30, 0.40])
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "tmp/local_paper_audit_20260814/tail_sensitivity"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))

    frames = [
        load_result(EXISTING_020 / "stage1/results.csv", 0.20),
    ]
    for fraction in args.fractions:
        frames.append(load_result(run_fraction(base, fraction), fraction))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output_dir / "per_seed_results.csv", index=False)
    summarize(frame).to_csv(args.output_dir / "tail_summary.csv", index=False)
    paired(frame).to_csv(args.output_dir / "paired_vs_loss.csv", index=False)
    query_overlap(args.fractions).to_csv(args.output_dir / "query_overlap.csv", index=False)
    (args.output_dir / "STATUS.md").write_text(
        "# Frozen-feature tail sensitivity\n\n"
        "这是复用 CelebA seed 0--9 的 exploratory 分析，不属于注册端到端确认。\n"
        "它比较 tail fraction 0.10/0.20/0.30/0.40 下的线性头重训结果和查询集合稳定性；\n"
        "不得与 seeds 20--29 的 RepairValue-Q registered paired gate 合并。\n",
        encoding="utf-8",
    )
    print(summarize(frame).to_string(index=False))
    print(f"Wrote tail sensitivity to {args.output_dir}")


if __name__ == "__main__":
    main()
