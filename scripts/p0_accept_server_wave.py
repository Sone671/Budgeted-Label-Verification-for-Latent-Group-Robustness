#!/usr/bin/env python
"""Validate server Wave-1 raw outputs before they enter P0 analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_verify.config import load_config
from robust_verify.p0 import sha256_file


RESULT_COLUMNS = {
    "noise_name", "seed", "method", "budget_fraction", "budget_count",
    "baseline_wga", "wga", "delta_wga", "noise_precision",
}


def _result_spec(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--result must be DATASET=/path/to/stage1/results.csv")
        dataset, raw_path = value.split("=", 1)
        if dataset in result:
            raise ValueError(f"duplicate --result dataset: {dataset}")
        result[dataset] = Path(raw_path).resolve()
    return result


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def _expected_runs(plan_path: Path) -> dict[str, dict[str, Any]]:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    runs = {}
    for item in plan.get("runs", []):
        if item.get("role", "confirmatory") != "confirmatory":
            continue
        dataset = item["dataset"]
        if dataset in runs:
            raise ValueError(f"confirmatory plan contains duplicate dataset {dataset}")
        config = load_config(plan_path.parent.parent / item["config"])
        runs[dataset] = config
    return runs


def _validate_run(dataset: str, path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    root = path.parent.parent
    if not path.exists():
        _check(checks, "results.csv present", False, str(path))
        return checks
    frame = pd.read_csv(path)
    missing = sorted(RESULT_COLUMNS.difference(frame.columns))
    _check(checks, "result schema", not missing, "missing=" + ",".join(missing) if missing else "required columns present")
    if missing:
        return checks

    expected_seeds = {int(value) for value in config["experiment"]["seeds"]}
    expected_noises = {item["name"] for item in config["noise"]["settings"]}
    expected_budgets = {float(value) for value in config["experiment"]["budgets"]}
    expected_methods = set(config["experiment"]["methods"])
    expected_rows = len(expected_seeds) * len(expected_noises) * len(expected_budgets) * len(expected_methods)
    duplicate = frame.duplicated(["noise_name", "seed", "method", "budget_fraction"], keep=False)
    _check(checks, "no duplicate result keys", not duplicate.any(), f"duplicate rows={int(duplicate.sum())}")
    _check(checks, "complete row count", len(frame) == expected_rows, f"actual={len(frame)}, expected={expected_rows}")
    _check(checks, "seed whitelist and completeness", set(frame["seed"].astype(int)) == expected_seeds, f"actual={sorted(frame['seed'].unique())}")
    _check(checks, "noise whitelist and completeness", set(frame["noise_name"]) == expected_noises, f"actual={sorted(frame['noise_name'].unique())}")
    actual_budgets = {round(float(value), 10) for value in frame["budget_fraction"]}
    _check(checks, "budget whitelist and completeness", actual_budgets == {round(value, 10) for value in expected_budgets}, f"actual={sorted(actual_budgets)}")
    _check(checks, "method whitelist and completeness", set(frame["method"]) == expected_methods, f"actual={sorted(frame['method'].unique())}")
    _check(checks, "finite primary metrics", bool(np.isfinite(frame[["baseline_wga", "wga", "delta_wga"]].to_numpy(float)).all()), "baseline_wga,wga,delta_wga")

    run_manifest = root / "stage1" / "run_manifest.json"
    artifact_manifest = root / "stage1" / "artifact_manifest.json"
    _check(checks, "frozen run manifest", run_manifest.exists(), str(run_manifest))
    _check(checks, "artifact hash manifest", artifact_manifest.exists(), str(artifact_manifest))
    if artifact_manifest.exists():
        manifest = json.loads(artifact_manifest.read_text(encoding="utf-8"))
        records = {record["path"]: record["sha256"] for record in manifest.get("files", [])}
        expected_hash = records.get("stage1/results.csv")
        _check(checks, "results hash matches artifact manifest", expected_hash == sha256_file(path), "results.csv")

    query_missing = checkpoint_missing = artifact_missing = bad_query_count = 0
    for _, row in frame.iterrows():
        base = root / "stage1" / "queries" / f"{row['noise_name']}_seed{int(row['seed'])}"
        query = base / f"{row['method']}_budget_{float(row['budget_fraction']):.4f}.npy"
        artifact = base / f"{row['method']}_budget_{float(row['budget_fraction']):.4f}.json"
        if not query.exists():
            query_missing += 1
        else:
            ids = np.load(query, allow_pickle=False)
            if len(ids) != int(row["budget_count"]) or len(np.unique(ids)) != len(ids):
                bad_query_count += 1
        if not artifact.exists():
            artifact_missing += 1
        checkpoint = root / str(row.get("retrain_checkpoint", ""))
        if not str(row.get("retrain_checkpoint", "")) or not checkpoint.exists():
            checkpoint_missing += 1
    _check(checks, "query IDs saved and valid", query_missing == bad_query_count == 0, f"missing={query_missing}, invalid={bad_query_count}")
    _check(checks, "query proxy artifacts saved", artifact_missing == 0, f"missing={artifact_missing}")
    _check(checks, "retraining checkpoints saved", checkpoint_missing == 0, f"missing={checkpoint_missing}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accept or reject a server P0 Wave-1 return bundle.")
    parser.add_argument("--plan", default="configs/p0_wave_s1.yaml")
    parser.add_argument("--result", action="append", required=True, help="DATASET=/path/to/stage1/results.csv; repeatable")
    parser.add_argument(
        "--skip-dataset",
        action="append",
        default=[],
        help="Predeclared conditional dataset(s) excluded from this acceptance run.",
    )
    parser.add_argument("--out", required=True, help="Output JSON report")
    parser.add_argument("--allow-fail", action="store_true", help="Write a report without nonzero exit on rejection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    expected = _expected_runs(plan_path)
    supplied = _result_spec(args.result)
    report: dict[str, Any] = {"plan": str(plan_path), "runs": {}}
    for dataset, config in expected.items():
        if dataset in args.skip_dataset:
            report["runs"][dataset] = {
                "passed": True,
                "skipped": True,
                "checks": [{"check": "conditional dataset exclusion", "passed": True, "detail": "predeclared skip"}],
            }
            continue
        path = supplied.get(dataset)
        checks = (
            _validate_run(dataset, path, config)
            if path is not None
            else [{"check": "server result supplied", "passed": False, "detail": "not supplied"}]
        )
        report["runs"][dataset] = {"passed": all(check["passed"] for check in checks), "checks": checks}
    unknown_skips = sorted(set(args.skip_dataset).difference(expected))
    unexpected = sorted(set(supplied).difference(expected))
    report["unexpected_datasets"] = unexpected
    report["unknown_skipped_datasets"] = unknown_skips
    report["passed"] = not unexpected and not unknown_skips and all(run["passed"] for run in report["runs"].values())
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = ["# P0 Wave-1 acceptance report", "", f"Overall: **{'PASS' if report['passed'] else 'REJECT'}**", ""]
    for dataset, run in report["runs"].items():
        markdown.append(f"## {dataset}: {'PASS' if run['passed'] else 'REJECT'}")
        for check in run["checks"]:
            markdown.append(f"- [{'x' if check['passed'] else ' '}] {check['check']}: {check['detail']}")
        markdown.append("")
    output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {output}")
    if not report["passed"] and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
