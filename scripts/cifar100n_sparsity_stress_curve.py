#!/usr/bin/env python
"""Stress-test the sparsity gate by varying scout density on seen CIFAR-100N."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_budget_advisor_cifar100n import _load_public_conditions  # noqa: E402
from evaluate_sparsity_gated_granularity import _recommend, _summary  # noqa: E402
from granularity_adaptive_audit_advisor import LEVELS, _run_episode  # noqa: E402


SOURCE = ROOT / "outputs" / "audit_budget_advisor_cifar100n"
OUTPUT = ROOT / "outputs" / "cifar100n_sparsity_stress_curve"
DEFAULT_SCOUT_COUNTS = (700, 1_000, 2_000, 4_000)
CANDIDATE_BUDGETS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
QUALITY_TARGETS = (0.30, 0.35, 0.40)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    public_conditions, _manifest = _load_public_conditions(source, (0, 1, 2))
    reference = np.load(source / "private" / "reference_labels.npy", allow_pickle=False).astype(np.int64)
    is_error = reference != public_conditions[0].observed_labels
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
    scout_counts = tuple(int(value.strip()) for value in args.scout_counts.split(",") if value.strip())
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for scout_count in scout_counts:
        args_episode = SimpleNamespace(
            seed=int(args.seed),
            scout_count=scout_count,
            posterior_draws=int(args.posterior_draws),
            alpha=float(args.alpha),
            candidate_budgets=CANDIDATE_BUDGETS,
        )
        rows: list[dict] = []
        for condition in conditions:
            print(f"[stress] scout={scout_count} seed={condition.seed}", flush=True)
            for replicate in range(int(args.scout_replicates)):
                rows.extend(_run_episode(condition, replicate=replicate, args=args_episode))
        curves = pd.DataFrame(rows)
        curves.to_csv(output / f"all_curves_scout{scout_count}.csv", index=False)
        recommendations, selected = _recommend(
            curves,
            cell_counts={("CIFAR-100N", "human", int(condition.seed)): len(condition.strata) for condition in conditions},
            quality_targets=QUALITY_TARGETS,
        )
        selected["scout_density"] = scout_count / len(conditions[0].strata)
        recommendations["scout_density"] = scout_count / len(conditions[0].strata)
        summary = _summary(selected, recommendations)
        summary["scout_count"] = scout_count
        summary["average_scout_labels_per_fine_cell"] = scout_count / len(conditions[0].strata)
        summaries[str(scout_count)] = summary
        all_rows.extend(selected.to_dict("records"))
        recommendations.to_csv(output / f"recommendations_scout{scout_count}.csv", index=False)
    selected_curves = pd.DataFrame(all_rows)
    selected_curves.to_csv(output / "selected_curves.csv", index=False)
    _atomic_json(
        {
            "protocol": "sparsity_gated_granularity_v1",
            "stage": "seen_cifar100n_sparsity_stress_diagnostic",
            "not_confirmatory": True,
            "scout_counts": list(scout_counts),
            "min_scouts_per_fine_cell": 4.0,
            "summaries": summaries,
        },
        output / "summary.json",
    )
    print(f"[stress-complete] {output / 'summary.json'}", flush=True)
    return output / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE))
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--scout-replicates", type=int, default=20)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--scout-counts", default=",".join(str(value) for value in DEFAULT_SCOUT_COUNTS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
