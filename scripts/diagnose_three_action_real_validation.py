"""Diagnose the real-data three-action MVP before investing in a policy.

This script is intentionally descriptive: it never tunes thresholds on the
private oracle outcomes and never reports oracle agreement as learnability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_dataset(root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(root / dataset / "summary.csv")
    states = pd.read_csv(root / dataset / "states.csv")
    required_summary = {
        "dataset",
        "noise_name",
        "seed",
        "cost_ratio",
        "lambda",
        "oracle_action",
        "oracle_root_action_values",
        "headroom_vs_best_fixed",
        "baseline_wga",
        "always_label_wga",
        "audit_then_label_wga",
        "attribute_bacc",
    }
    required_states = {
        "state_hash",
        "remaining_budget",
        "classification",
        "oracle_action",
        "transcript_kinds",
    }
    missing = (required_summary - set(summary.columns)) | (required_states - set(states.columns))
    if missing:
        raise ValueError(f"{dataset}: missing diagnostic fields: {sorted(missing)}")
    return summary, states


def _margin(value_json: str) -> tuple[float, str, float]:
    values = json.loads(value_json)
    ordered = sorted(values.items(), key=lambda item: float(item[1]), reverse=True)
    if len(ordered) < 2:
        raise ValueError("oracle action values must contain at least two actions")
    return float(ordered[0][1] - ordered[1][1]), str(ordered[0][0]), float(ordered[0][1])


def diagnose_dataset(root: Path, dataset: str) -> dict[str, Any]:
    summary, states = _load_dataset(root, dataset)
    parsed = summary["oracle_root_action_values"].map(_margin)
    summary = summary.copy()
    summary["root_action_margin"] = parsed.map(lambda value: value[0])
    summary["root_best_value"] = parsed.map(lambda value: value[2])
    summary["root_best_from_values"] = parsed.map(lambda value: value[1])
    root_zero = summary[np.isclose(summary["lambda"], 0.0)]
    root_actions = root_zero["oracle_action"].value_counts().to_dict()
    state_action_counts = states["oracle_action"].value_counts().to_dict()
    root_margin = root_zero["root_action_margin"]
    r_delta = root_zero["always_label_wga"] - root_zero["baseline_wga"]
    g_delta = root_zero["audit_then_label_wga"] - root_zero["baseline_wga"]
    g_minus_r = root_zero["audit_then_label_wga"] - root_zero["always_label_wga"]
    integrity = {
        "summary_rows": int(len(summary)),
        "state_rows": int(len(states)),
        "unique_state_hashes": int(states["state_hash"].nunique()),
        "negative_remaining_budget_rows": int((states["remaining_budget"] < -1e-9).sum()),
        "classification_counts": states["classification"].value_counts().to_dict(),
        "transcript_kinds": sorted(states["transcript_kinds"].dropna().unique().tolist()),
    }
    return {
        "dataset": dataset,
        "root_cases_lambda0": int(len(root_zero)),
        "root_action_counts_lambda0": root_actions,
        "all_three_root_actions_present": all(
            action in root_actions for action in ("verify_task", "annotate_group", "stop")
        ),
        "all_three_nonterminal_actions_present": all(
            action in state_action_counts
            for action in ("verify_task", "annotate_group", "stop")
        ),
        "stop_root_present": "stop" in root_actions,
        "state_action_counts": state_action_counts,
        "root_action_margin_pp": {
            "min": float(root_margin.min() * 100),
            "median": float(root_margin.median() * 100),
            "max": float(root_margin.max() * 100),
        },
        "headroom_pp_lambda0": {
            "mean": float(root_zero["headroom_vs_best_fixed"].mean() * 100),
            "median": float(root_zero["headroom_vs_best_fixed"].median() * 100),
            "max": float(root_zero["headroom_vs_best_fixed"].max() * 100),
            "positive_fraction": float((root_zero["headroom_vs_best_fixed"] > 1e-9).mean()),
        },
        "wga_delta_pp_lambda0": {
            "R_mean": float(r_delta.mean() * 100),
            "R_min": float(r_delta.min() * 100),
            "R_max": float(r_delta.max() * 100),
            "G_mean": float(g_delta.mean() * 100),
            "G_min": float(g_delta.min() * 100),
            "G_max": float(g_delta.max() * 100),
            "G_minus_R_mean": float(g_minus_r.mean() * 100),
        },
        "attribute_bacc": {
            "mean": float(root_zero["attribute_bacc"].mean()),
            "min": float(root_zero["attribute_bacc"].min()),
            "max": float(root_zero["attribute_bacc"].max()),
        },
        "integrity": integrity,
        "learnability_evidence": {
            "evaluated": False,
            "reason": "states contain a fixed candidate path and private oracle labels, not a held-out legal policy",
        },
    }


def _recommendation(diagnostics: dict[str, Any]) -> dict[str, Any]:
    datasets = diagnostics["datasets"]
    all_three = all(
        item["all_three_nonterminal_actions_present"] for item in datasets.values()
    )
    any_headroom = any(item["headroom_pp_lambda0"]["positive_fraction"] > 0 for item in datasets.values())
    integrity_ok = all(
        item["integrity"]["negative_remaining_budget_rows"] == 0 for item in datasets.values()
    )
    if all_three and any_headroom and integrity_ok:
        verdict = "conditional_go"
        next_step = "freeze a legal action-value estimator and test it on held-out seeds"
    elif any_headroom and integrity_ok:
        verdict = "benchmark_or_theory_only"
        next_step = "do not claim a universal policy; complete expected-feedback oracle and learnability audit"
    else:
        verdict = "stop_method_development"
        next_step = "retain only impossibility/diagnostic results and stop adding policy variants"
    return {
        "verdict": verdict,
        "next_step": next_step,
        "reasons": {
            "all_datasets_have_R_G_S_nonterminal_regions": all_three,
            "some_headroom_exists": any_headroom,
            "engineering_integrity": integrity_ok,
            "legal_policy_learnability_tested": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "three_action_real_validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "three_action_real_validation" / "diagnostics.json",
    )
    args = parser.parse_args()
    diagnostics: dict[str, Any] = {
        "protocol": "three_action_real_validation_diagnostics_v1",
        "datasets": {
            dataset: diagnose_dataset(args.input_dir, dataset)
            for dataset in ("waterbirds", "celeba")
        },
    }
    diagnostics["recommendation"] = _recommendation(diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics["recommendation"], sort_keys=True))


if __name__ == "__main__":
    main()
