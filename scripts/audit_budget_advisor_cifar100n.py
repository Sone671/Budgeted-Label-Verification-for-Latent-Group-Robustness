#!/usr/bin/env python
"""Run the frozen target-only audit-budget advisor on CIFAR-100N.

The script deliberately separates the public ranking/design phase from the
private clean-label evaluation phase.  It first freezes hashes for the noisy
labels, probe dynamics, NoiseScore rankings, and strata; only then does it
read the reference labels used to simulate scout returns and evaluate the
advisor.  The reference labels never enter ranking construction or a
recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.audit_budget import (  # noqa: E402
    AuditStratum,
    beta_residual_posterior,
    build_band_class_strata,
    draw_stratified_scout,
)
from robust_verify.scoring import build_legal_scores, descending_ranking  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "audit_budget_advisor_cifar100n"
BAND_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
CANDIDATE_BUDGETS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
QUALITY_TARGETS = (0.30, 0.35, 0.40)
PROBE_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Condition:
    seed: int
    observed_labels: np.ndarray
    ranking: np.ndarray
    strata: list[AuditStratum]
    population_size: int
    ranking_sha256: str
    score_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _freeze_json(payload: dict[str, Any], path: Path) -> None:
    """Write an immutable design artifact, refusing silent protocol changes."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"frozen artifact differs from existing {path}")
        return
    _atomic_json(payload, path)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("--seeds must contain at least one integer")
    return values


def _validate_sample_ids(sample_ids: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if len(values) != n or not np.array_equal(np.sort(values), np.arange(n, dtype=np.int64)):
        raise RuntimeError("probe sample_ids must be a permutation of training order")
    # Dynamics rows are keyed by sample_id; all downstream arrays use the
    # official CIFAR-100 training order.
    return np.argsort(values, kind="stable")


def _load_public_conditions(output_root: Path, seeds: tuple[int, ...]) -> tuple[list[Condition], dict[str, Any]]:
    noisy_path = output_root / "noisy_labels.npy"
    noisy_labels = np.load(noisy_path, allow_pickle=False).astype(np.int64)
    if noisy_labels.shape != (50_000,) or not ((0 <= noisy_labels).all() & (noisy_labels < 100).all()):
        raise RuntimeError("unexpected CIFAR-100N noisy label array")

    conditions: list[Condition] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        dynamics_path = output_root / "probes" / f"probe_seed{seed}.dynamics.npz"
        if not dynamics_path.exists():
            raise FileNotFoundError(dynamics_path)
        with np.load(dynamics_path, allow_pickle=False) as dynamics:
            sample_ids = dynamics["sample_ids"].astype(np.int64)
            row_order = _validate_sample_ids(sample_ids, len(noisy_labels))
            probabilities = dynamics["train_probabilities"].astype(np.float64)
            correctness = dynamics["correctness_history"].astype(np.int8)
            if probabilities.shape[0] != len(noisy_labels) or probabilities.shape[1] != 100:
                raise RuntimeError(f"unexpected probability shape for seed {seed}: {probabilities.shape}")
            if correctness.shape[0] != len(noisy_labels) or correctness.ndim != 2:
                raise RuntimeError(f"unexpected correctness shape for seed {seed}: {correctness.shape}")
            probabilities = probabilities[row_order]
            correctness = correctness[row_order]
            scores = build_legal_scores(probabilities, noisy_labels, correctness)
            ranking = descending_ranking(scores["noise_score"], scores["loss"])
            strata, boundaries = build_band_class_strata(ranking, noisy_labels, BAND_EDGES)
            score_hash = _sha256_array(scores["noise_score"].astype(np.float64))
            ranking_hash = _sha256_array(ranking.astype(np.int64))
            dynamics_hash = _sha256_file(dynamics_path)
        records.append(
            {
                "seed": int(seed),
                "dynamics_path": str(dynamics_path),
                "dynamics_sha256": dynamics_hash,
                "score_sha256": score_hash,
                "ranking_sha256": ranking_hash,
                "population_size": int(len(noisy_labels)),
                "stratum_count": int(len(strata)),
                "band_boundaries": boundaries.tolist(),
                "nonempty_band_class_cells": int(len(strata)),
            }
        )
        conditions.append(
            Condition(
                seed=int(seed),
                observed_labels=noisy_labels.copy(),
                ranking=ranking,
                strata=strata,
                population_size=len(noisy_labels),
                ranking_sha256=ranking_hash,
                score_sha256=score_hash,
            )
        )
    public_manifest = {
        "protocol": "target_only_stratified_audit_advisor_cifar100n_v1",
        "dataset": "CIFAR-100N",
        "noisy_labels_path": str(noisy_path),
        "noisy_labels_sha256": _sha256_file(noisy_path),
        "probe_seeds": [int(value) for value in seeds],
        "band_edges": list(BAND_EDGES),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": 1_000,
        "posterior_draws": 2_000,
        "upper_probability": 0.95,
        "prior": "target_only_Beta(1,1)",
        "records": records,
        "private_reference_labels_loaded": False,
    }
    return conditions, public_manifest


def _reviewed_bands(budget: float) -> np.ndarray:
    return np.asarray(BAND_EDGES[1:], dtype=np.float64) <= float(budget) + 1e-12


def _run_curve(
    condition: Condition,
    is_error: np.ndarray,
    *,
    scout_replicate: int,
    base_seed: int,
    posterior_draws: int,
) -> list[dict[str, Any]]:
    scout_seed = base_seed + 1_000_003 * condition.seed + 101 * scout_replicate
    selected, allocation = draw_stratified_scout(
        condition.strata, 1_000, seed=scout_seed
    )
    sizes = np.asarray([len(stratum.indices) for stratum in condition.strata], dtype=np.int64)
    bands = np.asarray([stratum.band for stratum in condition.strata], dtype=np.int64)
    errors = np.asarray(
        [int(is_error[indices].sum()) for indices in selected], dtype=np.int64
    )
    # Remove scout examples once.  Prefix review then zeroes whole score bands.
    scout_error_by_stratum = errors.astype(np.int64)
    residual_stratum_errors = np.asarray(
        [int(is_error[stratum.indices].sum()) for stratum in condition.strata], dtype=np.int64
    ) - scout_error_by_stratum
    residual_stratum_sizes = sizes - allocation
    rows: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(CANDIDATE_BUDGETS):
        reviewed = _reviewed_bands(budget)
        posterior = beta_residual_posterior(
            stratum_sizes=sizes,
            scout_counts=allocation,
            scout_errors=errors,
            stratum_bands=bands,
            reviewed_bands=reviewed,
            prior_band_means=np.zeros(len(BAND_EDGES) - 1, dtype=np.float64),
            prior_strength=0.0,
            population_size=condition.population_size,
            draws=posterior_draws,
            upper_probability=0.95,
            seed=scout_seed + 7_919 * (budget_index + 1),
        )
        residual_error = int(residual_stratum_errors[~reviewed[bands]].sum())
        residual_count = int(residual_stratum_sizes[~reviewed[bands]].sum())
        actual = residual_error / condition.population_size
        rows.append(
            {
                "dataset": "CIFAR-100N",
                "seed": condition.seed,
                "scout_replicate": scout_replicate,
                "scout_count": 1_000,
                "scout_fraction": 1_000 / condition.population_size,
                "prefix_budget_fraction": budget,
                "union_review_fraction": 1.0 - residual_count / condition.population_size,
                "actual_remaining_error_rate": actual,
                "target_only_posterior_mean": posterior.mean,
                "target_only_upper": posterior.upper,
                "target_only_upper_width": posterior.upper_width,
                "target_only_covers": bool(posterior.upper + 1e-12 >= actual),
                "ranking_sha256": condition.ranking_sha256,
                "score_sha256": condition.score_sha256,
            }
        )
    return rows


def _recommendations(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, episode in curves.groupby(["seed", "scout_replicate"], sort=True):
        episode = episode.sort_values("prefix_budget_fraction", kind="stable")
        for target in QUALITY_TARGETS:
            recommended_rows = episode.loc[episode["target_only_upper"] <= target]
            oracle_rows = episode.loc[episode["actual_remaining_error_rate"] <= target]
            recommended = None if recommended_rows.empty else recommended_rows.iloc[0]
            oracle = None if oracle_rows.empty else oracle_rows.iloc[0]
            issued = recommended is not None
            feasible = oracle is not None
            safe = bool(issued and float(recommended["actual_remaining_error_rate"]) <= target)
            excess = np.nan
            if issued and feasible:
                excess = float(recommended["prefix_budget_fraction"]) - float(oracle["prefix_budget_fraction"])
            rows.append(
                {
                    "dataset": "CIFAR-100N",
                    "seed": int(key[0]),
                    "scout_replicate": int(key[1]),
                    "quality_target": target,
                    "recommendation_issued": issued,
                    "recommended_prefix_budget": float(recommended["prefix_budget_fraction"]) if issued else np.nan,
                    "recommended_union_review_fraction": float(recommended["union_review_fraction"]) if issued else np.nan,
                    "actual_at_recommendation": float(recommended["actual_remaining_error_rate"]) if issued else np.nan,
                    "recommendation_safe": safe,
                    "oracle_feasible": feasible,
                    "oracle_prefix_budget": float(oracle["prefix_budget_fraction"]) if feasible else np.nan,
                    "excess_prefix_budget": excess,
                }
            )
    return pd.DataFrame(rows)


def _summarize(curves: pd.DataFrame, recommendations: pd.DataFrame) -> dict[str, Any]:
    issued = recommendations["recommendation_issued"].astype(bool)
    safe_issued = recommendations.loc[issued, "recommendation_safe"].astype(bool)
    metrics = {
        "curve_rows": int(len(curves)),
        "condition_count": int(curves[["seed", "scout_replicate"]].drop_duplicates().shape[0]),
        "upper_coverage": float(curves["target_only_covers"].mean()),
        "recommendation_coverage": float(issued.mean()),
        "unsafe_recommendation_rate": float((~safe_issued).mean()) if len(safe_issued) else 1.0,
        "mean_excess_prefix_budget": float(recommendations["excess_prefix_budget"].dropna().mean())
        if recommendations["excess_prefix_budget"].notna().any()
        else 1.0,
        "mean_union_review_fraction_issued": float(
            recommendations.loc[issued, "recommended_union_review_fraction"].mean()
        )
        if issued.any()
        else float("nan"),
    }
    gates = {
        "upper_coverage_at_least_0_90": metrics["upper_coverage"] >= 0.90,
        "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
        "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
        "mean_excess_budget_at_most_0_02": metrics["mean_excess_prefix_budget"] <= 0.02,
    }
    return {
        "protocol": "target_only_stratified_audit_advisor_cifar100n_v1",
        "primary": "target_only",
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "decision": {
            "all_cifar100n_gates_pass": bool(all(gates.values())),
            "second_real_noise_dataset_allowed": bool(all(gates.values())),
        },
    }


def _write_report(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        "# CIFAR-100N Target-Only Audit Advisor Report",
        "",
        "The ranking and audit design were frozen before private reference labels were loaded.",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|:---:|",
        f"| 95% upper-bound coverage | {metrics['upper_coverage']:.4f} | {'PASS' if summary['gates']['upper_coverage_at_least_0_90'] else 'FAIL'} |",
        f"| Recommendation coverage | {metrics['recommendation_coverage']:.4f} | {'PASS' if summary['gates']['recommendation_coverage_at_least_0_30'] else 'FAIL'} |",
        f"| Unsafe recommendation rate | {metrics['unsafe_recommendation_rate']:.4f} | {'PASS' if summary['gates']['unsafe_rate_at_most_0_10'] else 'FAIL'} |",
        f"| Mean excess prefix budget | {100 * metrics['mean_excess_prefix_budget']:.2f} pp | {'PASS' if summary['gates']['mean_excess_budget_at_most_0_02'] else 'FAIL'} |",
        "",
        f"CIFAR-100N configuration: **{'PASS' if summary['passed'] else 'FAIL'}**.",
        "",
        f"Second real-noise dataset allowed: **{'YES' if summary['decision']['second_real_noise_dataset_allowed'] else 'NO'}**.",
        "",
        "This is an audit-budget result only; no WGA, retraining utility, hidden-group, or test metric is evaluated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    conditions, ranking_manifest = _load_public_conditions(output_root, seeds)
    design = {
        "protocol": "target_only_stratified_audit_advisor_cifar100n_v1",
        "evidence_role": "new_public_migration",
        "dataset": "CIFAR-100N",
        "band_edges": list(BAND_EDGES),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": 1_000,
        "scout_repetitions_per_probe_seed": int(args.scout_replicates),
        "posterior_draws": int(args.posterior_draws),
        "upper_probability": 0.95,
        "prior": "independent target-only Beta(1,1) per band-class cell",
        "probe_seeds": [int(value) for value in seeds],
        "forbidden_before_recommendation": [
            "reference clean labels outside the scout",
            "test metrics",
            "downstream retraining utility",
            "hidden groups and worst-group metrics",
        ],
        "ranking_manifest": "ranking_manifest_pre_private.json",
    }
    # These artifacts are intentionally created before the private load below.
    _freeze_json(design, output_root / "design_pre_private.json")
    _freeze_json(ranking_manifest, output_root / "ranking_manifest_pre_private.json")

    private_path = output_root / "private" / "reference_labels.npy"
    reference_labels = np.load(private_path, allow_pickle=False).astype(np.int64)
    if reference_labels.shape != (50_000,) or not ((0 <= reference_labels).all() & (reference_labels < 100).all()):
        raise RuntimeError("unexpected CIFAR-100N reference label array")
    is_error = reference_labels != conditions[0].observed_labels
    if int(is_error.sum()) != 20_100:
        raise RuntimeError(f"unexpected CIFAR-100N noise count: {int(is_error.sum())}")

    curve_rows: list[dict[str, Any]] = []
    for condition in conditions:
        print(f"[audit-seed] seed={condition.seed} ranking={condition.ranking_sha256[:12]}", flush=True)
        for replicate in range(int(args.scout_replicates)):
            curve_rows.extend(
                _run_curve(
                    condition,
                    is_error,
                    scout_replicate=replicate,
                    base_seed=int(args.seed),
                    posterior_draws=int(args.posterior_draws),
                )
            )
            if (replicate + 1) % 10 == 0:
                print(f"[audit-seed] seed={condition.seed} repetitions={replicate + 1}", flush=True)
    curves = pd.DataFrame(curve_rows)
    recommendations = _recommendations(curves)
    summary = _summarize(curves, recommendations)
    design["private_reference_sha256"] = _sha256_file(private_path)
    design["private_reference_role"] = "audit_simulation_and_post_freeze_evaluation_only"
    design["curve_row_count"] = int(len(curves))
    design["recommendation_row_count"] = int(len(recommendations))
    _atomic_json(design, output_root / "design.json")
    curves.to_csv(output_root / "residual_curves.csv", index=False)
    recommendations.to_csv(output_root / "recommendations.csv", index=False)
    _atomic_json(summary, output_root / "summary.json")
    _write_report(summary, output_root / "CIFAR100N_AUDIT_REPORT.md")
    print(f"[audit-complete] passed={summary['passed']} output={output_root}", flush=True)
    return output_root / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default=",".join(str(value) for value in PROBE_SEEDS))
    parser.add_argument("--scout-replicates", type=int, default=100)
    parser.add_argument("--posterior-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
