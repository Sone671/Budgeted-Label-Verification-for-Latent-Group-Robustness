#!/usr/bin/env python
"""Confirm the frozen sparsity-gated audit advisor on Food-101N.

The public phase constructs a target-population NoiseScore ranking from the
frozen public Food-101N probe outputs, then freezes all ranking and design
artifacts.  Only after that checkpoint does the script open official human
verification values to simulate scout answers and evaluate the prescribed
advisor.  Verification values never affect a score, ranking, stratum, gate,
or choice between audit representations.
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from granularity_adaptive_audit_advisor import LEVELS, _run_episode  # noqa: E402
from robust_verify.audit_budget import AuditStratum, build_band_class_strata  # noqa: E402
from robust_verify.scoring import build_legal_scores, descending_ranking  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "sparsity_gated_food101n"
BAND_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
CANDIDATE_BUDGETS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
QUALITY_TARGETS = (0.10, 0.15, 0.20)
PROBE_SEEDS = (0, 1, 2)
ALPHA = 0.05
SCOUT_COUNT = 1_000
MIN_SCOUTS_PER_FINE_CELL = 4.0


@dataclass(frozen=True)
class Condition:
    dataset: str
    noise_name: str
    seed: int
    observed_labels: np.ndarray
    ranking: np.ndarray
    strata: list[AuditStratum]
    population_size: int
    is_error: np.ndarray
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
    """Write an immutable pre-private artifact and reject silent rewrites."""
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
    if any(value not in PROBE_SEEDS for value in values):
        raise ValueError(f"Food-101N protocol fixes probe seeds to {PROBE_SEEDS}")
    return values


def _load_public_conditions(output_root: Path, seeds: tuple[int, ...]) -> tuple[list[Condition], dict[str, Any]]:
    """Build target-population conditions without opening verification values."""
    public_root = output_root / "public"
    manifest_path = public_root / "imagelist_public.csv"
    train_ids_path = public_root / "train_sample_ids.npy"
    audit_ids_path = public_root / "audit_population_sample_ids.npy"
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "path", "noisy_label", "public_probe_split"}
    if set(manifest.columns) != required or len(manifest) != 310_009:
        raise RuntimeError("unexpected Food-101N public manifest")
    sample_ids = manifest["sample_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, np.arange(len(manifest), dtype=np.int64)):
        raise RuntimeError("public manifest sample IDs must be contiguous and ordered")
    noisy_all = manifest["noisy_label"].to_numpy(dtype=np.int64)
    if not ((0 <= noisy_all).all() & (noisy_all < 101).all()):
        raise RuntimeError("unexpected Food-101N noisy labels")
    train_ids = np.load(train_ids_path, allow_pickle=False).astype(np.int64)
    audit_ids = np.load(audit_ids_path, allow_pickle=False).astype(np.int64)
    if train_ids.ndim != 1 or audit_ids.ndim != 1 or len(audit_ids) != 52_867:
        raise RuntimeError("unexpected Food-101N public sample index arrays")
    if not np.array_equal(train_ids, np.sort(train_ids, kind="stable")):
        raise RuntimeError("training sample IDs must be sorted")
    audit_positions = np.searchsorted(train_ids, audit_ids)
    if (audit_positions >= len(train_ids)).any() or not np.array_equal(train_ids[audit_positions], audit_ids):
        raise RuntimeError("all audit samples must be in the public probe-training split")
    observed_labels = noisy_all[audit_ids].copy()
    if np.unique(observed_labels).size != 101:
        raise RuntimeError("audit population must include every observed class")

    conditions: list[Condition] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        dynamics_path = output_root / "probes" / f"probe_seed{seed}.dynamics.npz"
        if not dynamics_path.exists():
            raise FileNotFoundError(dynamics_path)
        with np.load(dynamics_path, allow_pickle=False) as dynamics:
            dynamics_ids = dynamics["sample_ids"].astype(np.int64)
            probabilities = dynamics["train_probabilities"].astype(np.float64)
            correctness = dynamics["correctness_history"].astype(np.int8)
        if not np.array_equal(dynamics_ids, train_ids):
            raise RuntimeError(f"probe sample IDs differ from public train IDs for seed {seed}")
        if probabilities.shape != (len(train_ids), 101) or correctness.shape[0] != len(train_ids) or correctness.ndim != 2:
            raise RuntimeError(f"unexpected probe dynamics shape for seed {seed}")
        local_probabilities = probabilities[audit_positions]
        local_correctness = correctness[audit_positions]
        scores = build_legal_scores(local_probabilities, observed_labels, local_correctness)
        ranking = descending_ranking(scores["noise_score"], scores["loss"])
        strata, boundaries = build_band_class_strata(ranking, observed_labels, BAND_EDGES)
        if not strata or sum(len(stratum.indices) for stratum in strata) != len(audit_ids):
            raise RuntimeError(f"invalid audit strata for seed {seed}")
        score_hash = _sha256_array(scores["noise_score"].astype(np.float64))
        ranking_hash = _sha256_array(ranking.astype(np.int64))
        records.append(
            {
                "seed": int(seed),
                "dynamics_path": str(dynamics_path),
                "dynamics_sha256": _sha256_file(dynamics_path),
                "score_sha256": score_hash,
                "ranking_sha256": ranking_hash,
                "population_size": int(len(audit_ids)),
                "stratum_count": int(len(strata)),
                "average_scout_labels_per_fine_cell": SCOUT_COUNT / len(strata),
                "sparsity_gate_triggered": bool(SCOUT_COUNT / len(strata) < MIN_SCOUTS_PER_FINE_CELL),
                "band_boundaries": boundaries.tolist(),
            }
        )
        # The array is intentionally an all-false placeholder until private
        # values are loaded after the freeze checkpoint below.
        conditions.append(
            Condition(
                dataset="Food-101N",
                noise_name="human_verified",
                seed=int(seed),
                observed_labels=observed_labels.copy(),
                ranking=ranking,
                strata=strata,
                population_size=len(audit_ids),
                is_error=np.zeros(len(audit_ids), dtype=bool),
                ranking_sha256=ranking_hash,
                score_sha256=score_hash,
            )
        )
    public_manifest = {
        "protocol": "sparsity_gated_granularity_food101n_v1",
        "dataset": "Food-101N",
        "source_public_manifest": str(manifest_path),
        "source_public_manifest_sha256": _sha256_file(manifest_path),
        "train_sample_ids_sha256": _sha256_file(train_ids_path),
        "audit_population_sample_ids_sha256": _sha256_file(audit_ids_path),
        "audit_observed_labels_sha256": _sha256_array(observed_labels),
        "audit_population_size": int(len(audit_ids)),
        "probe_seeds": [int(value) for value in seeds],
        "band_edges": list(BAND_EDGES),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": SCOUT_COUNT,
        "posterior_draws": 2_000,
        "fine_cell_scout_threshold": MIN_SCOUTS_PER_FINE_CELL,
        "private_reference_labels_loaded": False,
        "records": records,
    }
    return conditions, public_manifest


def _with_private_errors(conditions: list[Condition], private_path: Path) -> list[Condition]:
    is_error = np.load(private_path, allow_pickle=False).astype(bool)
    if is_error.shape != (conditions[0].population_size,):
        raise RuntimeError("unexpected Food-101N private audit vector shape")
    return [
        Condition(
            dataset=condition.dataset,
            noise_name=condition.noise_name,
            seed=condition.seed,
            observed_labels=condition.observed_labels,
            ranking=condition.ranking,
            strata=condition.strata,
            population_size=condition.population_size,
            is_error=is_error,
            ranking_sha256=condition.ranking_sha256,
            score_sha256=condition.score_sha256,
        )
        for condition in conditions
    ]


def _select_policy(
    curves: pd.DataFrame,
    *,
    fine_cell_counts: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pre-frozen gate; no private result chooses the policy."""
    selected_curve_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    keys = ["dataset", "noise_name", "seed", "scout_replicate"]
    for key, episode in curves.groupby(keys, sort=True):
        fine_count = fine_cell_counts[int(key[2])]
        sparse = SCOUT_COUNT / fine_count < MIN_SCOUTS_PER_FINE_CELL
        if sparse:
            candidates = episode.loc[episode["level"].isin(LEVELS)].copy()
            policy = "adaptive_simultaneous"
        else:
            candidates = episode.loc[episode["level"] == "fine_baseline_95"].copy()
            policy = "fine_baseline_95"
        candidates["fine_stratum_count"] = fine_count
        candidates["average_scout_labels_per_fine_cell"] = SCOUT_COUNT / fine_count
        candidates["sparse_gate_triggered"] = sparse
        candidates["selected_policy"] = policy
        selected_curve_rows.extend(candidates.to_dict("records"))
        for target in QUALITY_TARGETS:
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
            recommendation_rows.append(
                {
                    "dataset": key[0],
                    "noise_name": key[1],
                    "seed": int(key[2]),
                    "scout_replicate": int(key[3]),
                    "quality_target": target,
                    "fine_stratum_count": fine_count,
                    "average_scout_labels_per_fine_cell": SCOUT_COUNT / fine_count,
                    "sparse_gate_triggered": sparse,
                    "selected_policy": policy,
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
    return pd.DataFrame(selected_curve_rows), pd.DataFrame(recommendation_rows)


def _summarize(curves: pd.DataFrame, recommendations: pd.DataFrame) -> dict[str, Any]:
    issued = recommendations["recommendation_issued"].astype(bool)
    safe_issued = recommendations.loc[issued, "recommendation_safe"].astype(bool)
    metrics = {
        "curve_rows": int(len(curves)),
        "condition_count": int(curves[["seed", "scout_replicate"]].drop_duplicates().shape[0]),
        "upper_coverage": float(curves["covers"].mean()),
        "recommendation_coverage": float(issued.mean()),
        "unsafe_recommendation_rate": float((~safe_issued).mean()) if len(safe_issued) else 1.0,
        "mean_excess_prefix_budget": float(recommendations["excess_prefix_budget"].dropna().mean())
        if recommendations["excess_prefix_budget"].notna().any() else 1.0,
        "mean_union_review_fraction_issued": float(
            recommendations.loc[issued, "recommended_union_review_fraction"].mean()
        ) if issued.any() else float("nan"),
        "sparse_gate_fraction": float(recommendations["sparse_gate_triggered"].mean()),
    }
    gates = {
        "upper_coverage_at_least_0_90": metrics["upper_coverage"] >= 0.90,
        "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
        "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
        "mean_excess_budget_at_most_0_02": metrics["mean_excess_prefix_budget"] <= 0.02,
    }
    return {
        "protocol": "sparsity_gated_granularity_food101n_v1",
        "primary": "target_only_sparsity_gated_granularity",
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "decision": {
            "all_food101n_gates_pass": bool(all(gates.values())),
            "frozen_variant_advances": bool(all(gates.values())),
        },
    }


def _write_report(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Food-101N Sparsity-Gated Audit Advisor Confirmation",
        "",
        "The public ranking, strata, and sparsity-gate decision were frozen before official verification values were loaded.",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|:---:|",
        f"| Upper-bound coverage | {metrics['upper_coverage']:.4f} | {'PASS' if summary['gates']['upper_coverage_at_least_0_90'] else 'FAIL'} |",
        f"| Recommendation coverage | {metrics['recommendation_coverage']:.4f} | {'PASS' if summary['gates']['recommendation_coverage_at_least_0_30'] else 'FAIL'} |",
        f"| Unsafe recommendation rate | {metrics['unsafe_recommendation_rate']:.4f} | {'PASS' if summary['gates']['unsafe_rate_at_most_0_10'] else 'FAIL'} |",
        f"| Mean excess prefix budget | {100 * metrics['mean_excess_prefix_budget']:.2f} pp | {'PASS' if summary['gates']['mean_excess_budget_at_most_0_02'] else 'FAIL'} |",
        "",
        f"Frozen Food-101N configuration: **{'PASS' if summary['passed'] else 'FAIL'}**.",
        "",
        "The result is limited to target-only label-audit budgeting; no WGA, test metric, hidden-group analysis, retraining utility, source prior, or new NoiseScore claim is included.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    # Public ranking/design stage: this function never opens private values.
    conditions, ranking_manifest = _load_public_conditions(output_root, seeds)
    design = {
        "protocol": "sparsity_gated_granularity_food101n_v1",
        "evidence_role": "frozen_independent_confirmation",
        "dataset": "Food-101N",
        "protocol_document": str(ROOT / "audit_budget_advisor" / "FOOD101N_PROTOCOL.md"),
        "band_edges": list(BAND_EDGES),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": SCOUT_COUNT,
        "scout_repetitions_per_probe_seed": int(args.scout_replicates),
        "posterior_draws": int(args.posterior_draws),
        "target_only_prior": "independent_Beta(1,1)",
        "fine_advisor_upper_probability": 1.0 - ALPHA,
        "adaptive_simultaneous_upper_probability": 1.0 - ALPHA / len(LEVELS),
        "fine_cell_scout_threshold": MIN_SCOUTS_PER_FINE_CELL,
        "gate_rule": "use fine 95% only when scout_count / fine_stratum_count >= 4; otherwise use the simultaneous global/band/band_class ladder",
        "probe_seeds": [int(value) for value in seeds],
        "forbidden_before_recommendation": [
            "official verification values outside the sampled scout",
            "test metrics",
            "downstream retraining utility",
            "hidden groups and worst-group metrics",
            "source-dataset priors",
        ],
        "ranking_manifest": "ranking_manifest_pre_private.json",
    }
    # Immutable checkpoints.  Private values are not even named until after
    # these two writes successfully return.
    _freeze_json(design, output_root / "design_pre_private.json")
    _freeze_json(ranking_manifest, output_root / "ranking_manifest_pre_private.json")

    private_path = output_root / "private" / "audit_is_error.npy"
    private_conditions = _with_private_errors(conditions, private_path)
    curve_rows: list[dict[str, Any]] = []
    episode_args = argparse.Namespace(
        seed=int(args.seed),
        scout_count=SCOUT_COUNT,
        posterior_draws=int(args.posterior_draws),
        alpha=ALPHA,
        candidate_budgets=CANDIDATE_BUDGETS,
    )
    for condition in private_conditions:
        print(f"[food101n-audit] seed={condition.seed} ranking={condition.ranking_sha256[:12]}", flush=True)
        for replicate in range(int(args.scout_replicates)):
            curve_rows.extend(_run_episode(condition, replicate=replicate, args=episode_args))
            if (replicate + 1) % 10 == 0:
                print(f"[food101n-audit] seed={condition.seed} repetitions={replicate + 1}", flush=True)
    all_curves = pd.DataFrame(curve_rows)
    cell_counts = {condition.seed: len(condition.strata) for condition in private_conditions}
    selected_curves, recommendations = _select_policy(all_curves, fine_cell_counts=cell_counts)
    summary = _summarize(selected_curves, recommendations)
    design["private_reference_sha256"] = _sha256_file(private_path)
    design["private_reference_role"] = "scout_simulation_and_post_freeze_evaluation_only"
    design["all_curve_row_count"] = int(len(all_curves))
    design["selected_curve_row_count"] = int(len(selected_curves))
    design["recommendation_row_count"] = int(len(recommendations))
    _atomic_json(design, output_root / "design.json")
    all_curves.to_csv(output_root / "all_residual_curves.csv", index=False)
    selected_curves.to_csv(output_root / "residual_curves.csv", index=False)
    recommendations.to_csv(output_root / "recommendations.csv", index=False)
    _atomic_json(summary, output_root / "summary.json")
    _write_report(summary, output_root / "FOOD101N_AUDIT_REPORT.md")
    print(f"[food101n-audit-complete] passed={summary['passed']} output={output_root}", flush=True)
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
