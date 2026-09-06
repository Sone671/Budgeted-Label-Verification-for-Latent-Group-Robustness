#!/usr/bin/env python
"""Development screen for a risk-controlled adaptive audit granularity.

This is a new method variant, separate from the frozen target-only baseline.
It uses the same public NoiseScore ranking and scout, but compares three
target-only representations of the audit population:

* global;
* NoiseScore band;
* NoiseScore band x observed noisy class.

Each representation receives a simultaneous one-sided posterior bound
(Bonferroni over the three representations).  The advisor chooses the least
prefix budget that satisfies the quality target over any representation.  The
fine band-by-class baseline is also evaluated at an unadjusted 95% endpoint.
No source-dataset prior, downstream utility, group metric, or retraining is
used.
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

from audit_budget_advisor_pilot import (  # noqa: E402
    BAND_EDGES,
    CANDIDATE_BUDGETS,
    Condition,
    DEFAULT_CONFIGS,
    _atomic_csv,
    _atomic_json,
    _load_conditions,
)


QUALITY_TARGETS = (0.10, 0.15, 0.20)
LEVELS = ("global", "band", "band_class")
DEFAULT_OUTPUT = ROOT / "outputs" / "granularity_adaptive_audit_dev"


def _groups(condition: Condition, level: str) -> list[np.ndarray]:
    if level == "band_class":
        return [stratum.indices for stratum in condition.strata]
    if level == "band":
        output: list[np.ndarray] = []
        for band in range(len(BAND_EDGES) - 1):
            parts = [stratum.indices for stratum in condition.strata if stratum.band == band]
            if parts:
                output.append(np.concatenate(parts).astype(np.int64, copy=False))
        return output
    if level == "global":
        return [np.arange(condition.population_size, dtype=np.int64)]
    raise ValueError(f"unknown audit granularity: {level}")


def _prefix_mask(condition: Condition, budget: float) -> np.ndarray:
    mask = np.zeros(condition.population_size, dtype=bool)
    for stratum in condition.strata:
        if BAND_EDGES[stratum.band + 1] <= float(budget) + 1e-12:
            mask[stratum.indices] = True
    return mask


def _posterior_upper(
    groups: list[np.ndarray],
    scout_mask: np.ndarray,
    is_error: np.ndarray,
    prefix_mask: np.ndarray,
    *,
    population_size: int,
    draws: int,
    upper_probability: float,
    seed: int,
) -> tuple[float, float, int, int]:
    sizes = np.asarray([len(indices) for indices in groups], dtype=np.int64)
    counts = np.asarray([int(scout_mask[indices].sum()) for indices in groups], dtype=np.int64)
    errors = np.asarray(
        [int(is_error[indices][scout_mask[indices]].sum()) for indices in groups],
        dtype=np.int64,
    )
    remaining_sizes = np.asarray(
        [int((~scout_mask[indices] & ~prefix_mask[indices]).sum()) for indices in groups],
        dtype=np.int64,
    )
    if (errors > counts).any() or (counts > sizes).any() or (remaining_sizes < 0).any():
        raise RuntimeError("invalid scout/group accounting")
    alpha = 1.0 + errors.astype(np.float64)
    beta = 1.0 + counts.astype(np.float64) - errors.astype(np.float64)
    rng = np.random.default_rng(int(seed))
    probabilities = rng.beta(alpha, beta, size=(int(draws), len(groups)))
    residual_draws = probabilities @ remaining_sizes.astype(np.float64) / float(population_size)
    mean = float(residual_draws.mean())
    upper = float(np.quantile(residual_draws, upper_probability))
    remaining_error = int(
        is_error[~scout_mask & ~prefix_mask].sum()
    )
    remaining_count = int((~scout_mask & ~prefix_mask).sum())
    return mean, upper, remaining_error, remaining_count


def _run_episode(
    condition: Condition,
    *,
    replicate: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidate_budgets = tuple(
        float(value) for value in getattr(args, "candidate_budgets", CANDIDATE_BUDGETS)
    )
    scout_seed = int(args.seed) + 1_000_003 * condition.seed + 10_007 * replicate
    selected, _allocation = __import__("robust_verify.audit_budget", fromlist=["draw_stratified_scout"]).draw_stratified_scout(
        condition.strata, int(args.scout_count), seed=scout_seed
    )
    scout_mask = np.zeros(condition.population_size, dtype=bool)
    for indices in selected:
        scout_mask[indices] = True
    rows: list[dict[str, Any]] = []
    simultaneous_upper = 1.0 - float(args.alpha) / len(LEVELS)
    for budget_index, budget in enumerate(candidate_budgets):
        prefix_mask = _prefix_mask(condition, budget)
        actual_error = int(condition.is_error[~scout_mask & ~prefix_mask].sum())
        actual_count = int((~scout_mask & ~prefix_mask).sum())
        actual_rate = actual_error / condition.population_size
        union_fraction = 1.0 - actual_count / condition.population_size
        for level_index, level in enumerate(LEVELS):
            mean, upper, _, _ = _posterior_upper(
                _groups(condition, level),
                scout_mask,
                condition.is_error,
                prefix_mask,
                population_size=condition.population_size,
                draws=int(args.posterior_draws),
                upper_probability=simultaneous_upper,
                seed=scout_seed + 7_919 * (budget_index + 1) + 997 * (level_index + 1),
            )
            rows.append(
                {
                    "dataset": condition.dataset,
                    "noise_name": condition.noise_name,
                    "seed": condition.seed,
                    "scout_replicate": replicate,
                    "scout_count": int(args.scout_count),
                    "prefix_budget_fraction": budget,
                    "union_review_fraction": union_fraction,
                    "level": level,
                    "posterior_mean": mean,
                    "upper": upper,
                    "upper_probability": simultaneous_upper,
                    "actual_remaining_error_rate": actual_rate,
                    "covers": bool(upper + 1e-12 >= actual_rate),
                }
            )
        # The unadjusted fine-granularity baseline is evaluated separately so
        # the paired efficiency comparison is not confounded by alpha splitting.
        mean, upper, _, _ = _posterior_upper(
            _groups(condition, "band_class"),
            scout_mask,
            condition.is_error,
            prefix_mask,
            population_size=condition.population_size,
            draws=int(args.posterior_draws),
            upper_probability=1.0 - float(args.alpha),
            seed=scout_seed + 31_337 * (budget_index + 1),
        )
        rows.append(
            {
                "dataset": condition.dataset,
                "noise_name": condition.noise_name,
                "seed": condition.seed,
                "scout_replicate": replicate,
                "scout_count": int(args.scout_count),
                "prefix_budget_fraction": budget,
                "union_review_fraction": union_fraction,
                "level": "fine_baseline_95",
                "posterior_mean": mean,
                "upper": upper,
                "upper_probability": 1.0 - float(args.alpha),
                "actual_remaining_error_rate": actual_rate,
                "covers": bool(upper + 1e-12 >= actual_rate),
            }
        )
    return rows


def _recommend(
    curves: pd.DataFrame,
    upper_column: str,
    *,
    allowed_levels: tuple[str, ...] | None = LEVELS,
    quality_targets: tuple[float, ...] = QUALITY_TARGETS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "noise_name", "seed", "scout_replicate"]
    for key, episode in curves.groupby(keys, sort=True):
        for target in quality_targets:
            eligible_mask = episode[upper_column] <= target
            if allowed_levels is not None:
                eligible_mask &= episode["level"].isin(allowed_levels)
            eligible = episode.loc[eligible_mask].sort_values(
                ["prefix_budget_fraction", upper_column, "level"],
                kind="stable",
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
    return pd.DataFrame(rows)


def _summarize(
    curves: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    quality_targets: tuple[float, ...] = QUALITY_TARGETS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for advisor, subset in {
        "adaptive_simultaneous": curves.loc[curves["level"].isin(LEVELS)],
        "fine_baseline_95": curves.loc[curves["level"] == "fine_baseline_95"],
    }.items():
        rec = recommendations if advisor == "adaptive_simultaneous" else _recommend(
            subset, "upper", allowed_levels=None, quality_targets=quality_targets
        )
        issued = rec["recommendation_issued"].astype(bool)
        safe = rec.loc[issued, "recommendation_safe"].astype(bool)
        metrics = {
            "curve_rows": int(len(subset)),
            "upper_coverage": float(subset["covers"].mean()),
            "recommendation_coverage": float(issued.mean()),
            "unsafe_recommendation_rate": float((~safe).mean()) if len(safe) else 1.0,
            "mean_excess_prefix_budget": float(rec["excess_prefix_budget"].dropna().mean())
            if rec["excess_prefix_budget"].notna().any() else 1.0,
        }
        gates = {
            "upper_coverage_at_least_0_90": metrics["upper_coverage"] >= 0.90,
            "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
            "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
            "mean_excess_budget_at_most_0_02": metrics["mean_excess_prefix_budget"] <= 0.02,
        }
        result[advisor] = {"metrics": metrics, "gates": gates, "passed": bool(all(gates.values()))}
    adaptive_excess = result["adaptive_simultaneous"]["metrics"]["mean_excess_prefix_budget"]
    baseline_excess = result["fine_baseline_95"]["metrics"]["mean_excess_prefix_budget"]
    result["paired_efficiency_improvement"] = baseline_excess - adaptive_excess
    result["decision"] = {
        "adaptive_core_gates_pass": result["adaptive_simultaneous"]["passed"],
        "at_least_half_excess_reduction": bool(adaptive_excess <= 0.5 * baseline_excess),
    }
    return result


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_root)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    design = {
        "protocol": "risk_controlled_adaptive_audit_granularity_v1",
        "stage": "old_seed_development",
        "granularity_levels": list(LEVELS),
        "simultaneous_alpha": float(args.alpha),
        "band_edges": list(BAND_EDGES),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": int(args.scout_count),
        "scout_replicates": int(args.scout_replicates),
        "posterior_draws": int(args.posterior_draws),
        "primary_outcomes": ["coverage", "safe recommendation", "excess prefix budget"],
        "forbidden_outcomes": ["WGA", "test metrics", "group metrics", "retraining utility", "source prior"],
        "configs": [str(path) for path in DEFAULT_CONFIGS],
    }
    _atomic_json(design, output / "design_pre_private.json")
    conditions, config_record = _load_conditions(
        argparse.Namespace(
            configs=",".join(str(path) for path in DEFAULT_CONFIGS),
            seeds=args.seeds,
            noise=args.noise,
        )
    )
    design["configs"] = config_record
    _atomic_json(design, output / "design.json")
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        print(
            f"[granularity-condition] {condition_index + 1}/{len(conditions)} "
            f"{condition.dataset}/{condition.noise_name}/seed{condition.seed}",
            flush=True,
        )
        for replicate in range(int(args.scout_replicates)):
            rows.extend(_run_episode(condition, replicate=replicate, args=args))
    curves = pd.DataFrame(rows)
    adaptive_recs = _recommend(curves.loc[curves["level"].isin(LEVELS)], "upper")
    summary = _summarize(curves, adaptive_recs)
    design["curve_row_count"] = int(len(curves))
    design["recommendation_row_count"] = int(len(adaptive_recs))
    _atomic_json(design, output / "design.json")
    _atomic_csv(curves, output / "residual_curves.csv")
    _atomic_csv(adaptive_recs, output / "recommendations.csv")
    _atomic_json(summary, output / "summary.json")
    print(f"[granularity-complete] {output / 'summary.json'}", flush=True)
    return output / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise", default="uniform20,minority_high_40")
    parser.add_argument("--scout-count", type=int, default=200)
    parser.add_argument("--scout-replicates", type=int, default=30)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
