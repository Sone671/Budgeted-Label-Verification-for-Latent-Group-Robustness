#!/usr/bin/env python
"""Validate the P0 method vocabulary and export its one canonical registry."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from robust_verify.config import load_config
from robust_verify.scoring import ADAPTIVE_METHODS, METHOD_LABELS, METHOD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export frozen P0 method names.")
    parser.add_argument("--plan", default="configs/p0_wave_s1.yaml")
    parser.add_argument("--out", default="outputs/p0_control/method_name_registry.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    used: set[str] = set()
    for item in plan["runs"]:
        config = load_config(plan_path.parent.parent / item["config"])
        methods = set(config["experiment"].get("methods", []))
        unknown = sorted(methods.difference(METHOD_NAMES))
        missing_label = sorted(methods.difference(METHOD_LABELS))
        if unknown or missing_label:
            raise ValueError(
                f"{item['config']}: unknown={unknown}; missing canonical display label={missing_label}"
            )
        used.update(methods)
    rows = [
        {
            "method_id": method,
            "display_name": METHOD_LABELS[method],
            "family": "oracle" if method.startswith("oracle_") else "adaptive" if method in ADAPTIVE_METHODS else "legal",
        }
        for method in sorted(used)
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Validated {len(rows)} canonical method names; wrote {output}")


if __name__ == "__main__":
    main()
