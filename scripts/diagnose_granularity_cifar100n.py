#!/usr/bin/env python
"""Post-failure CIFAR-100N diagnostic for adaptive audit granularity.

CIFAR-100N is already observed and this script is explicitly development-only.
It asks whether global/band pooling can explain and reduce the frozen fine
band-by-class advisor's efficiency failure.  Its result is not confirmatory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_budget_advisor_cifar100n import _load_public_conditions  # noqa: E402
from granularity_adaptive_audit_advisor import (  # noqa: E402
    LEVELS,
    _recommend,
    _run_episode,
    _summarize,
)


DEFAULT_SOURCE = ROOT / "outputs" / "audit_budget_advisor_cifar100n"
DEFAULT_OUTPUT = ROOT / "outputs" / "granularity_adaptive_cifar100n_diagnostic"
CANDIDATE_BUDGETS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
QUALITY_TARGETS = (0.30, 0.35, 0.40)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    public_conditions, public_manifest = _load_public_conditions(source, (0, 1, 2))
    design = {
        "protocol": "risk_controlled_adaptive_audit_granularity_v1",
        "stage": "post_failure_cifar100n_diagnostic_only",
        "not_confirmatory": True,
        "source_ranking_manifest": str(source / "ranking_manifest_pre_private.json"),
        "ranking_hashes": [record["ranking_sha256"] for record in public_manifest["records"]],
        "granularity_levels": list(LEVELS),
        "simultaneous_alpha": float(args.alpha),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": 1_000,
        "scout_replicates": int(args.scout_replicates),
        "posterior_draws": int(args.posterior_draws),
        "frozen_baseline_result": {
            "upper_coverage": 1.0,
            "recommendation_coverage": 0.6677777777777778,
            "unsafe_recommendation_rate": 0.0,
            "mean_excess_prefix_budget": 0.0978369384359401,
        },
        "forbidden_outcomes": ["WGA", "test metrics", "groups", "retraining utility", "source prior"],
    }
    _atomic_json(design, output / "design_pre_private.json")

    reference = np.load(source / "private" / "reference_labels.npy", allow_pickle=False)
    is_error = reference.astype(np.int64) != public_conditions[0].observed_labels
    conditions = [
        SimpleNamespace(
            dataset="CIFAR-100N",
            noise_name="human",
            seed=condition.seed,
            observed_labels=condition.observed_labels,
            is_error=is_error,
            strata=condition.strata,
            population_size=condition.population_size,
        )
        for condition in public_conditions
    ]
    run_args = SimpleNamespace(
        seed=int(args.seed),
        scout_count=1_000,
        posterior_draws=int(args.posterior_draws),
        alpha=float(args.alpha),
        candidate_budgets=CANDIDATE_BUDGETS,
    )
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        print(f"[cifar-granularity] seed={condition.seed}", flush=True)
        for replicate in range(int(args.scout_replicates)):
            rows.extend(_run_episode(condition, replicate=replicate, args=run_args))
    curves = pd.DataFrame(rows)
    recommendations = _recommend(
        curves.loc[curves["level"].isin(LEVELS)],
        "upper",
        quality_targets=QUALITY_TARGETS,
    )
    summary = _summarize(curves, recommendations, quality_targets=QUALITY_TARGETS)
    summary["stage"] = "post_failure_cifar100n_diagnostic_only"
    summary["not_confirmatory"] = True
    summary["relative_to_frozen_fine_baseline"] = {
        "frozen_mean_excess_prefix_budget": 0.0978369384359401,
        "diagnostic_adaptive_mean_excess_prefix_budget": summary["adaptive_simultaneous"]["metrics"]["mean_excess_prefix_budget"],
        "absolute_reduction": 0.0978369384359401
        - summary["adaptive_simultaneous"]["metrics"]["mean_excess_prefix_budget"],
    }
    curves.to_csv(output / "residual_curves.csv", index=False)
    recommendations.to_csv(output / "recommendations.csv", index=False)
    _atomic_json(summary, output / "summary.json")
    print(f"[cifar-granularity-complete] {output / 'summary.json'}", flush=True)
    return output / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scout-replicates", type=int, default=30)
    parser.add_argument("--posterior-draws", type=int, default=1_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
