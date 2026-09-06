#!/usr/bin/env python
"""Audit protocol fidelity and effective sample support for P0 baselines."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.config import load_config
from robust_verify.scoring import METHOD_LABELS, forgetting_counts


def _probe_dynamics(output_root: Path) -> dict[str, float | int | str]:
    paths = sorted((output_root / "stage1" / "probes").glob("*.dynamics.npz"))
    if not paths:
        return {"status": "missing", "probe_count": 0}
    epochs: list[int] = []
    nonzero_rates: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            correctness = data["correctness_history"]
            margin = data["margin_history"]
            if correctness.ndim != 2 or margin.ndim != 2 or correctness.shape != margin.shape:
                raise ValueError(f"Invalid dynamics layout: {path}")
            epochs.append(correctness.shape[1])
            nonzero_rates.append(float((forgetting_counts(correctness) > 0).mean()))
    return {
        "status": "available",
        "probe_count": len(paths),
        "min_epochs": min(epochs),
        "max_epochs": max(epochs),
        "mean_nonzero_forgetting_rate": float(np.mean(nonzero_rates)),
    }


def _coverage(results_path: Path, method: str) -> tuple[int, int]:
    if not results_path.exists():
        return 0, 0
    frame = pd.read_csv(results_path)
    method_rows = frame[frame["method"] == method]
    conditions = frame[["noise_name", "seed", "budget_fraction"]].drop_duplicates()
    return len(method_rows), len(conditions)


def audit(config_path: Path, output_root: Path) -> pd.DataFrame:
    config = load_config(config_path)
    experiment = config["experiment"]
    requested = set(experiment.get("methods", []))
    training_baselines = set(experiment.get("training_baselines", []))
    results_path = output_root / "stage1" / "results.csv"
    dynamics = _probe_dynamics(output_root)
    expected_query_conditions = (
        len(experiment["seeds"])
        * len(config["noise"]["settings"])
        * len(experiment["budgets"])
    )
    rows = []

    for method in ("forgetting", "aum", "confident_learning"):
        requested_method = method in requested
        result_rows, observed_conditions = _coverage(results_path, method)
        row = {
            "baseline": METHOD_LABELS.get(method, method),
            "method_id": method,
            "requested_in_config": requested_method,
            "result_rows": result_rows,
            "observed_condition_rows": observed_conditions,
            "expected_condition_rows": expected_query_conditions,
            "result_coverage_complete": (
                result_rows == expected_query_conditions == observed_conditions
                if requested_method
                else False
            ),
            "dynamics_status": dynamics["status"],
            "protocol": "",
            "effective_support": "",
            "caveat": "",
        }
        if method == "forgetting":
            row.update(
                {
                    "protocol": "correct-to-incorrect transitions across saved probe epochs",
                    "effective_support": f"mean nonzero rate={dynamics.get('mean_nonzero_forgetting_rate', 'NA')}",
                    "caveat": "Not interpretable as a strong dynamics baseline when the nonzero rate is near zero.",
                }
            )
        elif method == "aum":
            row.update(
                {
                    "protocol": "mean label-margin over saved probe epochs",
                    "effective_support": f"epoch range={dynamics.get('min_epochs', 'NA')}..{dynamics.get('max_epochs', 'NA')}",
                    "caveat": "Short trajectories are diagnostic only; report the actual epoch support.",
                }
            )
        else:
            installed = importlib.util.find_spec("cleanlab") is not None
            row.update(
                {
                    "protocol": "cleanlab.get_label_quality_scores using in-sample probe probabilities",
                    "effective_support": "one probability vector per training example",
                    "caveat": "In-sample probabilities are not OOF; this must be disclosed.",
                    "cleanlab_installed": installed,
                }
            )
        rows.append(row)

    model_path = output_root / "stage1" / "model_baselines.csv"
    model_rows = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    for method in ("erm", "jtt"):
        subset = model_rows[model_rows.get("method", pd.Series(dtype=str)) == method]
        rows.append(
            {
                "baseline": method.upper() if method == "erm" else "JTT",
                "method_id": method,
                "requested_in_config": method in training_baselines,
                "result_rows": len(subset),
                "observed_condition_rows": len(
                    subset[["noise_name", "seed"]].drop_duplicates()
                ) if not subset.empty else 0,
                "expected_condition_rows": len(experiment["seeds"]) * len(config["noise"]["settings"]),
                "result_coverage_complete": len(subset)
                == len(experiment["seeds"]) * len(config["noise"]["settings"]),
                "dynamics_status": "not_applicable",
                "protocol": (
                    "identification run followed by upweighted final training"
                    if method == "jtt"
                    else "shared linear-probe ERM checkpoint"
                ),
                "effective_support": (
                    f"identification_epochs={experiment.get('training_baseline_options', {}).get('jtt', {}).get('identification_epochs', 'NA')}"
                    if method == "jtt"
                    else "one model per noise/seed"
                ),
                "caveat": "Inspect identification_error_count before comparing JTT." if method == "jtt" else "",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write P0 baseline integrity reports.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir) if args.out_dir else output_root / "stage1" / "baseline_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = audit(Path(args.config), output_root)
    report.to_csv(out_dir / "baseline_integrity.csv", index=False)
    lines = ["# Baseline integrity audit", "", report.to_markdown(index=False), ""]
    (out_dir / "baseline_integrity.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "baseline_integrity.json").write_text(
        json.dumps(report.to_dict("records"), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'baseline_integrity.csv'}")


if __name__ == "__main__":
    main()
