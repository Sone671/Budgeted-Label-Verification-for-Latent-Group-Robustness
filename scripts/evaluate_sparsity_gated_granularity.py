#!/usr/bin/env python
"""Evaluate a predeclared sparsity-gated granularity policy.

This post-processing experiment uses already generated, frozen curves.  If a
fine band-by-class scout has at least four labels per non-empty cell on
average, it keeps the fixed fine 95% advisor.  Otherwise it uses the
simultaneous global/band/band-by-class ladder.  The gate depends only on
population/scout geometry, not returned clean labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_budget_advisor_cifar100n import _load_public_conditions  # noqa: E402
from audit_budget_advisor_pilot import DEFAULT_CONFIGS, _load_conditions  # noqa: E402


QUALITY_TARGETS_OLD = (0.10, 0.15, 0.20)
QUALITY_TARGETS_CIFAR = (0.30, 0.35, 0.40)
ADAPTIVE_LEVELS = ("global", "band", "band_class")
FINE_LEVEL = "fine_baseline_95"
MIN_SCOUTS_PER_FINE_CELL = 4.0


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _old_cell_counts() -> dict[tuple[str, str, int], int]:
    import argparse as _argparse

    conditions, _ = _load_conditions(
        _argparse.Namespace(
            configs=",".join(str(path) for path in DEFAULT_CONFIGS),
            seeds=",".join(str(value) for value in range(10)),
            noise="uniform20,minority_high_40",
        )
    )
    return {
        (condition.dataset, condition.noise_name, int(condition.seed)): len(condition.strata)
        for condition in conditions
    }


def _cifar_cell_counts(source_root: Path) -> dict[tuple[str, str, int], int]:
    conditions, _ = _load_public_conditions(source_root, (0, 1, 2))
    return {
        ("CIFAR-100N", "human", int(condition.seed)): len(condition.strata)
        for condition in conditions
    }


def _recommend(
    curves: pd.DataFrame,
    *,
    cell_counts: dict[tuple[str, str, int], int],
    quality_targets: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = curves.copy()
    enriched["fine_stratum_count"] = [
        int(cell_counts[(str(row.dataset), str(row.noise_name), int(row.seed))])
        for row in enriched.itertuples()
    ]
    enriched["sparsity_ratio"] = enriched["fine_stratum_count"] / enriched["scout_count"]
    rows: list[dict[str, Any]] = []
    selected_curve_rows: list[dict[str, Any]] = []
    keys = ["dataset", "noise_name", "seed", "scout_replicate"]
    for key, episode in enriched.groupby(keys, sort=True):
        ratio = float(episode["sparsity_ratio"].iloc[0])
        sparse = ratio > 1.0 / MIN_SCOUTS_PER_FINE_CELL
        if sparse:
            candidates = episode.loc[episode["level"].isin(ADAPTIVE_LEVELS)].copy()
            selected_policy = "adaptive_simultaneous"
        else:
            candidates = episode.loc[episode["level"] == FINE_LEVEL].copy()
            selected_policy = "fine_baseline_95"
        selected_curve_rows.extend(candidates.to_dict("records"))
        for target in quality_targets:
            eligible = candidates.loc[candidates["upper"] <= target].sort_values(
                ["prefix_budget_fraction", "upper", "level"], kind="stable"
            )
            oracle_rows = episode.loc[
                episode["actual_remaining_error_rate"] <= target
            ].sort_values("prefix_budget_fraction", kind="stable")
            chosen = None if eligible.empty else eligible.iloc[0]
            oracle = None if oracle_rows.empty else oracle_rows.iloc[0]
            issued = chosen is not None
            feasible = oracle is not None
            safe = bool(issued and float(chosen["actual_remaining_error_rate"]) <= target)
            rows.append(
                {
                    "dataset": key[0],
                    "noise_name": key[1],
                    "seed": int(key[2]),
                    "scout_replicate": int(key[3]),
                    "quality_target": target,
                    "fine_stratum_count": int(episode["fine_stratum_count"].iloc[0]),
                    "sparsity_ratio": ratio,
                    "sparse_gate_triggered": sparse,
                    "selected_policy": selected_policy,
                    "recommendation_issued": issued,
                    "recommended_level": str(chosen["level"]) if issued else "infeasible",
                    "recommended_prefix_budget": float(chosen["prefix_budget_fraction"]) if issued else np.nan,
                    "recommended_union_review_fraction": float(chosen["union_review_fraction"]) if issued else np.nan,
                    "actual_at_recommendation": float(chosen["actual_remaining_error_rate"]) if issued else np.nan,
                    "recommendation_safe": safe,
                    "oracle_feasible": feasible,
                    "oracle_prefix_budget": float(oracle["prefix_budget_fraction"]) if feasible else np.nan,
                    "excess_prefix_budget": (
                        float(chosen["prefix_budget_fraction"]) - float(oracle["prefix_budget_fraction"])
                        if issued and safe and feasible else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selected_curve_rows)


def _summary(curves: pd.DataFrame, recommendations: pd.DataFrame) -> dict[str, Any]:
    issued = recommendations["recommendation_issued"].astype(bool)
    safe = recommendations.loc[issued, "recommendation_safe"].astype(bool)
    metrics = {
        "curve_rows": int(len(curves)),
        "recommendation_rows": int(len(recommendations)),
        "upper_coverage": float(curves["covers"].mean()),
        "recommendation_coverage": float(issued.mean()),
        "unsafe_recommendation_rate": float((~safe).mean()) if len(safe) else 1.0,
        "mean_excess_prefix_budget": float(recommendations["excess_prefix_budget"].dropna().mean())
        if recommendations["excess_prefix_budget"].notna().any() else 1.0,
        "sparse_gate_fraction": float(recommendations["sparse_gate_triggered"].mean()),
    }
    gates = {
        "upper_coverage_at_least_0_90": metrics["upper_coverage"] >= 0.90,
        "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
        "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
        "mean_excess_budget_at_most_0_02": metrics["mean_excess_prefix_budget"] <= 0.02,
    }
    return {"protocol": "sparsity_gated_granularity_v1", "metrics": metrics, "gates": gates, "passed": bool(all(gates.values()))}


def run(args: argparse.Namespace) -> Path:
    old_curves = pd.read_csv(Path(args.old_curves).resolve())
    cifar_curves = pd.read_csv(Path(args.cifar_curves).resolve())
    old_counts = _old_cell_counts()
    cifar_counts = _cifar_cell_counts(Path(args.cifar_source_root).resolve())
    old_recommendations, old_selected = _recommend(
        old_curves,
        cell_counts=old_counts,
        quality_targets=QUALITY_TARGETS_OLD,
    )
    cifar_recommendations, cifar_selected = _recommend(
        cifar_curves,
        cell_counts=cifar_counts,
        quality_targets=QUALITY_TARGETS_CIFAR,
    )
    old_recommendations["scope"] = "old_seed_development"
    cifar_recommendations["scope"] = "seen_cifar_diagnostic"
    recommendations = pd.concat([old_recommendations, cifar_recommendations], ignore_index=True)
    selected_curves = pd.concat([old_selected, cifar_selected], ignore_index=True)
    summary = {
        "protocol": "sparsity_gated_granularity_v1",
        "old_seed_development": _summary(old_selected, old_recommendations),
        "seen_cifar_diagnostic": _summary(cifar_selected, cifar_recommendations),
        "scope": "old_seed_development_plus_seen_cifar_diagnostic",
        "not_confirmatory": True,
        "gate_rule": "trigger adaptive simultaneous ladder when average scout labels per fine cell < 4",
    }
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_curves.to_csv(output / "selected_curves.csv", index=False)
    recommendations.to_csv(output / "recommendations.csv", index=False)
    _atomic_json(summary, output / "summary.json")
    design = {
        "protocol": "sparsity_gated_granularity_v1",
        "scope": summary["scope"],
        "fine_cell_scout_threshold": MIN_SCOUTS_PER_FINE_CELL,
        "quality_targets": {
            "old_seed_development": list(QUALITY_TARGETS_OLD),
            "seen_cifar_diagnostic": list(QUALITY_TARGETS_CIFAR),
        },
        "inputs": [str(Path(args.old_curves).resolve()), str(Path(args.cifar_curves).resolve())],
        "not_confirmatory": True,
    }
    _atomic_json(design, output / "design.json")
    print(f"[sparsity-gated-complete] {output / 'summary.json'}", flush=True)
    return output / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-curves",
        default=str(ROOT / "outputs" / "granularity_adaptive_audit_dev" / "residual_curves.csv"),
    )
    parser.add_argument(
        "--cifar-curves",
        default=str(ROOT / "outputs" / "granularity_adaptive_cifar100n_diagnostic" / "residual_curves.csv"),
    )
    parser.add_argument("--cifar-source-root", default=str(ROOT / "outputs" / "audit_budget_advisor_cifar100n"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "sparsity_gated_granularity_dev"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
