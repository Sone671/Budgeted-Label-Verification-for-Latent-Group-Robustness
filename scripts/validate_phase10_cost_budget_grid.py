#!/usr/bin/env python
"""Print and validate the locked Phase-10 cost--budget allocation grid."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.cost_budget_map import allocate_action_mix
from robust_verify.utils import budget_to_count


DEFAULT_CONFIG = (
    ROOT
    / "joint_group_label_budget"
    / "configs"
    / "phase10_cost_budget_map_locked.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    grid = config["grid"]
    rows = []
    for budget_fraction in grid["total_budget_fractions"]:
        capacity = budget_to_count(float(budget_fraction), args.train_size)
        for ratio in grid["group_to_label_cost_ratios"]:
            for share in grid["group_action_shares"]:
                allocation = allocate_action_mix(
                    capacity,
                    float(share),
                    float(ratio),
                    max_group_audits=args.validation_size,
                )
                rows.append(
                    {
                        "budget_fraction": float(budget_fraction),
                        **asdict(allocation),
                    }
                )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
