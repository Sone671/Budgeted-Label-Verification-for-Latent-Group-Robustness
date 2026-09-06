#!/usr/bin/env python
"""Analyze frozen Phase-10 results and apply the preregistered gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.cost_budget_analysis import (
    break_even_table,
    pair_phase10_results,
    phase10_gate,
    summarize_phase10_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.results if path.exists()]
    if not frames:
        raise FileNotFoundError("no Phase-10 result file exists")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    paired = pair_phase10_results(combined)
    summary = summarize_phase10_pairs(
        paired, expected_seed_count=args.expected_seeds
    )
    break_even = break_even_table(summary)
    decision = phase10_gate(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
    mechanism_columns = [
        *[
            "dataset",
            "noise_name",
            "budget_fraction",
            "group_to_label_cost_ratio",
            "target_group_action_share",
        ],
        "n_seeds",
        "same_label_n_seeds",
        "same_label_mean",
        "same_label_ci_low",
        "same_label_ci_high",
        "label_opportunity_cost_mean",
        "label_opportunity_cost_n_seeds",
        "label_opportunity_cost_ci_low",
        "label_opportunity_cost_ci_high",
        "strict_cost_mean",
        "strict_cost_ci_low",
        "strict_cost_ci_high",
        "decomposition_residual_max_abs",
    ]
    summary[mechanism_columns].to_csv(
        args.output_dir / "mechanism_summary.csv", index=False
    )
    break_even.to_csv(args.output_dir / "break_even.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
