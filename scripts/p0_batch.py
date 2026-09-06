#!/usr/bin/env python
"""Run a frozen P0 matrix, with resumable isolated pilot subsets.

The normal invocation keeps every config exactly as locked in the plan.  A
seed/budget/noise selector creates a separate output tree and reads the frozen
Stage-0 artifacts from the original output directory, so a pilot can never
silently contaminate a confirmatory run.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from robust_verify.config import load_config
from robust_verify.p0 import canonical_json, config_fingerprint, sha256_file
from robust_verify.stage0 import run_stage0
from robust_verify.stage1 import run_stage1


def _csv_values(value: str | None, cast):
    return None if value is None else [cast(item.strip()) for item in value.split(",") if item.strip()]


def _load_plan(path: Path) -> dict[str, Any]:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list):
        raise ValueError("P0 plan must be a mapping with a non-empty runs list")
    return plan


def _apply_plan_overrides(config: dict[str, Any], item: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    """Apply explicit output/input roots without changing the source YAML."""
    result = copy.deepcopy(config)
    for key in ("output_dir", "input_dir"):
        if key not in item:
            continue
        value = Path(item[key]).expanduser()
        if not value.is_absolute():
            value = (plan_path.parent.parent / value).resolve()
        result["project"][key] = str(value)
    return result


def _locked_payload(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    records = []
    for item in plan["runs"]:
        config_path = (plan_path.parent.parent / item["config"]).resolve()
        config = _apply_plan_overrides(load_config(config_path), item, plan_path)
        records.append(
            {
                "dataset": item["dataset"],
                "role": item.get("role", "confirmatory"),
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "resolved_config_sha256": config_fingerprint(config),
            }
        )
    return {"schema_version": "p0-batch-lock-v1", "plan": str(plan_path.resolve()), "runs": records}


def _write_or_validate_lock(control_dir: Path, plan_path: Path, plan: dict[str, Any]) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    path = control_dir / "p0_wave_s1.lock.json"
    payload = _locked_payload(plan_path, plan)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"P0 plan differs from locked protocol: {path}. Create a new plan version instead."
            )
        return
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _pilot_config(
    config: dict[str, Any],
    *,
    seeds: list[int] | None,
    budgets: list[float] | None,
    noises: list[str] | None,
) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    experiment = selected["experiment"]
    if seeds is not None:
        unknown = sorted(set(seeds).difference(experiment["seeds"]))
        if unknown:
            raise ValueError(f"Requested seeds not in frozen config: {unknown}")
        experiment["seeds"] = seeds
    if budgets is not None:
        unknown = sorted(set(budgets).difference(experiment["budgets"]))
        if unknown:
            raise ValueError(f"Requested budgets not in frozen config: {unknown}")
        experiment["budgets"] = budgets
    if noises is not None:
        available = {item["name"] for item in selected["noise"]["settings"]}
        unknown = sorted(set(noises).difference(available))
        if unknown:
            raise ValueError(f"Requested noise settings not in frozen config: {unknown}")
        selected["noise"]["settings"] = [
            item for item in selected["noise"]["settings"] if item["name"] in noises
        ]

    tag = "_".join(
        [
            "seeds-" + "-".join(map(str, experiment["seeds"])),
            "budgets-" + "-".join(f"{float(value):.4f}" for value in experiment["budgets"]),
            "noise-" + "-".join(item["name"] for item in selected["noise"]["settings"]),
        ]
    )
    original_output = selected["project"]["output_dir"]
    source_output = selected["project"].get("input_dir", original_output)
    selected["project"]["input_dir"] = source_output
    selected["project"]["output_dir"] = str(Path(original_output) / "p0_pilots" / tag)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked P0 experiment matrix.")
    parser.add_argument("--plan", default="configs/p0_wave_s1.yaml")
    parser.add_argument("--control-dir", default="outputs/p0_control")
    parser.add_argument("--dataset", action="append", help="Dataset(s) to run; repeatable.")
    parser.add_argument(
        "--role",
        action="append",
        help="Protocol role(s) to run; repeatable (for example: confirmatory).",
    )
    parser.add_argument("--stage", choices=("stage0", "stage1", "all"), default="stage1")
    parser.add_argument("--seeds", help="Pilot-only comma-separated subset of frozen seeds.")
    parser.add_argument("--budgets", help="Pilot-only comma-separated subset of frozen budgets.")
    parser.add_argument("--noises", help="Pilot-only comma-separated noise names.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    plan = _load_plan(plan_path)
    _write_or_validate_lock(Path(args.control_dir), plan_path, plan)
    seeds = _csv_values(args.seeds, int)
    budgets = _csv_values(args.budgets, float)
    noises = _csv_values(args.noises, str)
    is_pilot = any(value is not None for value in (seeds, budgets, noises))

    selected_runs = [
        run
        for run in plan["runs"]
        if (not args.dataset or run["dataset"] in args.dataset)
        and (not args.role or run.get("role", "confirmatory") in args.role)
    ]
    if not selected_runs:
        raise ValueError("No runs match --dataset")
    for run in selected_runs:
        config_path = (plan_path.parent.parent / run["config"]).resolve()
        config = _apply_plan_overrides(load_config(config_path), run, plan_path)
        if is_pilot:
            config = _pilot_config(config, seeds=seeds, budgets=budgets, noises=noises)
        print(
            f"[{run['dataset']}] role={run.get('role', 'confirmatory')} "
            f"output={config['project']['output_dir']}"
        )
        if args.dry_run:
            continue
        if args.stage in {"stage0", "all"}:
            if is_pilot:
                raise ValueError("Pilot subsets reuse frozen Stage-0 artifacts; stage0 is not permitted")
            run_stage0(config)
        if args.stage in {"stage1", "all"}:
            results = run_stage1(config)
            print(f"  completed/resumed: {len(results)} rows")


if __name__ == "__main__":
    main()
