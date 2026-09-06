#!/usr/bin/env python
"""Re-run one frozen seed in an isolated output directory and compare it."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.config import load_config
from robust_verify.stage1 import run_stage1


def _filter_config(config: dict, *, output_dir: Path, seed: int, noises: list[str] | None) -> dict:
    result = copy.deepcopy(config)
    original_output = result["project"]["output_dir"]
    if seed not in result["experiment"]["seeds"]:
        raise ValueError(f"seed {seed} is not in the frozen config")
    result["project"]["input_dir"] = original_output
    result["project"]["output_dir"] = str(output_dir.resolve())
    result["experiment"]["seeds"] = [seed]
    if noises:
        available = {item["name"] for item in result["noise"]["settings"]}
        unknown = sorted(set(noises).difference(available))
        if unknown:
            raise ValueError(f"unknown frozen noise names: {unknown}")
        result["noise"]["settings"] = [item for item in result["noise"]["settings"] if item["name"] in noises]
    return result


def _compare(reference: pd.DataFrame, reproduced: pd.DataFrame, *, tolerance: float) -> dict:
    keys = ["noise_name", "seed", "method", "budget_fraction"]
    reference = reference[keys + ["wga", "delta_wga"]].copy()
    reproduced = reproduced[keys + ["wga", "delta_wga"]].copy()
    joined = reference.merge(reproduced, on=keys, suffixes=("_reference", "_reproduced"), how="outer", indicator=True)
    common = joined[joined["_merge"] == "both"].copy()
    for metric in ("wga", "delta_wga"):
        common[f"abs_error_{metric}"] = np.abs(common[f"{metric}_reference"] - common[f"{metric}_reproduced"])
    max_error = float(common[["abs_error_wga", "abs_error_delta_wga"]].to_numpy().max()) if len(common) else float("inf")
    return {
        "reference_rows": len(reference),
        "reproduced_rows": len(reproduced),
        "matched_rows": len(common),
        "missing_or_extra_rows": int((joined["_merge"] != "both").sum()),
        "max_absolute_error": max_error,
        "tolerance": tolerance,
        "passed": bool(len(common) == len(reference) == len(reproduced) and max_error <= tolerance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated local seed reproduction for Stage-1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", action="append")
    parser.add_argument("--reference-results", help="Defaults to the frozen config's results.csv")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = load_config(args.config)
    output_root = Path(args.output_dir)
    if not args.report_only:
        run_stage1(_filter_config(frozen, output_dir=output_root, seed=args.seed, noises=args.noise))
    reference_path = Path(args.reference_results) if args.reference_results else Path(frozen["project"]["output_dir"]) / "stage1" / "results.csv"
    reproduced_path = output_root / "stage1" / "results.csv"
    if not reference_path.exists() or not reproduced_path.exists():
        raise FileNotFoundError("Both reference and reproduced results.csv are required")
    reference = pd.read_csv(reference_path)
    reproduced = pd.read_csv(reproduced_path)
    reference = reference[reference["seed"] == args.seed]
    if args.noise:
        reference = reference[reference["noise_name"].isin(args.noise)]
    report = _compare(reference, reproduced, tolerance=args.tolerance)
    report.update({"reference_results": str(reference_path), "reproduced_results": str(reproduced_path)})
    report_path = output_root / "stage1" / "seed0_reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
