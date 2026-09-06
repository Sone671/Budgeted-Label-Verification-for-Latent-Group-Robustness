#!/usr/bin/env python
"""Development screen for finite-population prefix-error certification.

This is deliberately not an acquisition or repair-utility method.  It treats
a frozen public ordering as an input, draws one global uniform sentinel panel,
and gives exact hypergeometric upper bounds on the error count left after each
predeclared review prefix.  The public ranking is frozen before simulated
verification labels are opened.
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

from audit_budget_advisor_pilot import DEFAULT_CONFIGS, _dataset_name, _parse_csv, _sha256
from robust_verify.analysis import align_frame
from robust_verify.audit_budget import hypergeometric_upper_error_count
from robust_verify.config import load_config
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.utils import budget_to_count


DEFAULT_OUTPUT = ROOT / "outputs" / "finite_population_prefix_certification_dev"
QUALITY_TARGETS = (0.10, 0.15, 0.20)
PREFIX_BUDGETS = tuple(float(value) / 200.0 for value in range(41))
ALPHA = 0.05
MAX_MEAN_EXCESS_PREFIX_BUDGET = 0.05


@dataclass(frozen=True)
class PublicCondition:
    dataset: str
    noise_name: str
    seed: int
    observed_labels: np.ndarray
    ranking: np.ndarray
    population_size: int
    public_path: Path
    private_path: Path
    dynamics_path: Path
    ranking_sha256: str
    score_sha256: str


@dataclass(frozen=True)
class Condition:
    public: PublicCondition
    is_error: np.ndarray


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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _freeze_json(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"frozen artifact differs from existing {path}")
        return
    _atomic_json(payload, path)


def _sentinel_count(population_size: int) -> int:
    return min(500, max(1, population_size // 10))


def _load_public_conditions(args: argparse.Namespace) -> tuple[list[PublicCondition], dict[str, Any]]:
    conditions: list[PublicCondition] = []
    records: list[dict[str, Any]] = []
    configs: dict[str, dict[str, str]] = {}
    for raw_config in _parse_csv(args.configs, str):
        config_path = Path(raw_config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_config(config_path)
        source_root = Path(config["project"]["output_dir"])
        if not source_root.is_absolute():
            source_root = ROOT / source_root
        dataset = _dataset_name(config, source_root)
        configs[dataset] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "source_output": str(source_root),
        }
        for noise in _parse_csv(args.noise, str):
            for seed in _parse_csv(args.seeds, int):
                prefix = f"{noise}_seed{seed}"
                dynamics_path = source_root / "stage1" / "probes" / f"{prefix}.dynamics.npz"
                public_path = source_root / "manifests" / f"{prefix}_train_public.csv"
                private_path = source_root / "manifests" / f"{prefix}_train_private.csv"
                with np.load(dynamics_path, allow_pickle=False) as dynamics:
                    sample_ids = dynamics["sample_ids"].astype(np.int64)
                    public = align_frame(pd.read_csv(public_path), sample_ids)
                    observed = public["noisy_label"].to_numpy(dtype=np.int64)
                    legal_scores = build_legal_scores(
                        dynamics["train_probabilities"], observed, dynamics["correctness_history"]
                    )
                ranking = descending_ranking(legal_scores["noise_score"], legal_scores["loss"])
                if len(ranking) != len(observed) or not np.array_equal(np.sort(ranking), np.arange(len(observed))):
                    raise RuntimeError(f"invalid public ranking for {prefix}")
                ranking_hash = _sha256_array(ranking.astype(np.int64))
                score_hash = _sha256_array(legal_scores["noise_score"].astype(np.float64))
                public_condition = PublicCondition(
                    dataset=dataset,
                    noise_name=noise,
                    seed=int(seed),
                    observed_labels=observed,
                    ranking=ranking,
                    population_size=len(observed),
                    public_path=public_path,
                    private_path=private_path,
                    dynamics_path=dynamics_path,
                    ranking_sha256=ranking_hash,
                    score_sha256=score_hash,
                )
                conditions.append(public_condition)
                records.append(
                    {
                        "dataset": dataset,
                        "noise_name": noise,
                        "seed": int(seed),
                        "population_size": len(observed),
                        "sentinel_count": _sentinel_count(len(observed)),
                        "public_manifest_path": str(public_path),
                        "public_manifest_sha256": _sha256(public_path),
                        "dynamics_path": str(dynamics_path),
                        "dynamics_sha256": _sha256(dynamics_path),
                        "ranking_sha256": ranking_hash,
                        "score_sha256": score_hash,
                    }
                )
    return conditions, {
        "protocol": "finite_population_prefix_certification_v1",
        "stage": "old_seed_development_only",
        "configs": configs,
        "prefix_budgets": list(PREFIX_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "familywise_alpha": ALPHA,
        "alpha_per_prefix": ALPHA / len(PREFIX_BUDGETS),
        "sentinel_rule": "min(500, floor(0.10 * population_size))",
        "records": records,
        "private_verification_loaded": False,
    }


def _with_private_errors(conditions: list[PublicCondition]) -> list[Condition]:
    output: list[Condition] = []
    for condition in conditions:
        # The private manifest is intentionally opened only after the design
        # and ranking manifest have been frozen by run().
        with np.load(condition.dynamics_path, allow_pickle=False) as dynamics:
            sample_ids = dynamics["sample_ids"].astype(np.int64)
        private = align_frame(pd.read_csv(condition.private_path), sample_ids)
        is_error = private["is_noisy"].to_numpy(dtype=bool)
        if is_error.shape != (condition.population_size,):
            raise RuntimeError(f"invalid private verification vector for {condition.private_path}")
        output.append(Condition(public=condition, is_error=is_error))
    return output


def _run_episode(
    condition: Condition,
    *,
    replicate: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    public = condition.public
    n = public.population_size
    sentinel_count = _sentinel_count(n)
    seed = (
        int(base_seed)
        + 1_000_003 * public.seed
        + 10_007 * replicate
        + (0 if public.noise_name == "uniform20" else 503)
        + {"celeba": 0, "civilcomments": 31, "waterbirds": 67}[public.dataset]
    )
    sentinel = np.random.default_rng(seed).choice(n, size=sentinel_count, replace=False)
    rank_position = np.empty(n, dtype=np.int64)
    rank_position[public.ranking] = np.arange(n, dtype=np.int64)
    sentinel_position = rank_position[sentinel]
    per_prefix_alpha = ALPHA / len(PREFIX_BUDGETS)
    rows: list[dict[str, Any]] = []
    for budget in PREFIX_BUDGETS:
        prefix_count = budget_to_count(budget, n)
        suffix_size = n - prefix_count
        sentinel_in_suffix = sentinel_position >= prefix_count
        suffix_sentinel = sentinel[sentinel_in_suffix]
        sampled_suffix_count = len(suffix_sentinel)
        observed_suffix_errors = int(condition.is_error[suffix_sentinel].sum())
        upper_suffix_errors = hypergeometric_upper_error_count(
            population_size=suffix_size,
            sample_size=sampled_suffix_count,
            observed_errors=observed_suffix_errors,
            alpha=per_prefix_alpha,
        )
        # Both prefix items and sentinel items are reviewed/corrected.  The
        # unreviewed error count therefore removes observed sentinel errors.
        upper_remaining_errors = upper_suffix_errors - observed_suffix_errors
        suffix_indices = public.ranking[prefix_count:]
        actual_suffix_errors = int(condition.is_error[suffix_indices].sum())
        actual_remaining_errors = actual_suffix_errors - observed_suffix_errors
        remaining_count = suffix_size - sampled_suffix_count
        if min(upper_remaining_errors, actual_remaining_errors, remaining_count) < 0:
            raise RuntimeError("invalid sentinel residual accounting")
        upper_rate = upper_remaining_errors / n
        actual_rate = actual_remaining_errors / n
        rows.append(
            {
                "dataset": public.dataset,
                "noise_name": public.noise_name,
                "seed": public.seed,
                "sentinel_replicate": replicate,
                "population_size": n,
                "sentinel_count": sentinel_count,
                "prefix_budget_fraction": budget,
                "prefix_count": prefix_count,
                "sentinel_suffix_count": sampled_suffix_count,
                "sentinel_suffix_errors": observed_suffix_errors,
                "union_review_fraction": 1.0 - remaining_count / n,
                "actual_remaining_error_rate": actual_rate,
                "finite_population_upper": upper_rate,
                "covers": bool(upper_rate + 1e-12 >= actual_rate),
                "ranking_sha256": public.ranking_sha256,
                "score_sha256": public.score_sha256,
            }
        )
    return rows


def _recommendations(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "noise_name", "seed", "sentinel_replicate"]
    for key, episode in curves.groupby(keys, sort=True):
        episode = episode.sort_values("prefix_budget_fraction", kind="stable")
        for target in QUALITY_TARGETS:
            eligible = episode.loc[episode["finite_population_upper"] <= target]
            oracle_rows = episode.loc[episode["actual_remaining_error_rate"] <= target]
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
                    "sentinel_replicate": int(key[3]),
                    "quality_target": target,
                    "recommendation_issued": issued,
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


def _summarize(curves: pd.DataFrame, recommendations: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"datasets": {}}
    for dataset, curve in curves.groupby("dataset", sort=True):
        rec = recommendations.loc[recommendations["dataset"] == dataset]
        episode_coverage = curve.groupby(
            ["noise_name", "seed", "sentinel_replicate"], sort=True
        )["covers"].all()
        issued = rec["recommendation_issued"].astype(bool)
        safe_issued = rec.loc[issued, "recommendation_safe"].astype(bool)
        metrics = {
            "individual_upper_coverage": float(curve["covers"].mean()),
            "familywise_upper_coverage": float(episode_coverage.mean()),
            "recommendation_coverage": float(issued.mean()),
            "unsafe_recommendation_rate": float((~safe_issued).mean()) if len(safe_issued) else 1.0,
            "mean_excess_prefix_budget": float(rec["excess_prefix_budget"].dropna().mean())
            if rec["excess_prefix_budget"].notna().any() else 1.0,
        }
        gates = {
            "familywise_coverage_at_least_0_90": metrics["familywise_upper_coverage"] >= 0.90,
            "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
            "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
            "mean_excess_budget_at_most_0_05": (
                metrics["mean_excess_prefix_budget"] <= MAX_MEAN_EXCESS_PREFIX_BUDGET
            ),
        }
        result["datasets"][str(dataset)] = {
            "metrics": metrics,
            "gates": gates,
            "strong_efficiency_milestone_2pp": (
                metrics["mean_excess_prefix_budget"] <= 0.02
            ),
            "passed": bool(all(gates.values())),
        }
    pass_count = sum(int(value["passed"]) for value in result["datasets"].values())
    result["decision"] = {
        "datasets_passed": pass_count,
        "development_go": pass_count >= 2,
        "independent_protocol_allowed": pass_count >= 2,
    }
    return result


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Finite-Population Prefix Certification: Development Report",
        "",
        "All results are on previously inspected old seeds and are development-only.",
        "",
        "| Dataset | Familywise coverage | Certificate coverage | Unsafe issued | Excess prefix budget | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for dataset, result in summary["datasets"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {dataset} | {metrics['familywise_upper_coverage']:.3f} | "
            f"{metrics['recommendation_coverage']:.3f} | "
            f"{metrics['unsafe_recommendation_rate']:.3f} | "
            f"{100 * metrics['mean_excess_prefix_budget']:.2f} pp | "
            f"{'YES' if result['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            f"Datasets passed: {summary['decision']['datasets_passed']}/3.",
            "",
            "This screen does not evaluate WGA, test metrics, group composition, or retraining utility.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    design = {
        "protocol": "finite_population_prefix_certification_v1",
        "stage": "old_seed_development_only",
        "prefix_budgets": list(PREFIX_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "familywise_alpha": ALPHA,
        "alpha_per_prefix": ALPHA / len(PREFIX_BUDGETS),
        "max_mean_excess_prefix_budget": MAX_MEAN_EXCESS_PREFIX_BUDGET,
        "sentinel_rule": "min(500, floor(0.10 * population_size))",
        "sentinel_repetitions": int(args.sentinel_replicates),
        "ordering_role": "fixed public input; no new detector claim",
        "forbidden_outcomes": ["WGA", "test metrics", "group metrics", "retraining utility", "source prior"],
    }
    _freeze_json(design, output / "design_pre_private.json")
    public_conditions, ranking_manifest = _load_public_conditions(args)
    _freeze_json(ranking_manifest, output / "ranking_manifest_pre_private.json")
    conditions = _with_private_errors(public_conditions)
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        public = condition.public
        print(
            f"[prefix-cert] {public.dataset}/{public.noise_name}/seed{public.seed} "
            f"sentinel={_sentinel_count(public.population_size)}",
            flush=True,
        )
        for replicate in range(int(args.sentinel_replicates)):
            rows.extend(_run_episode(condition, replicate=replicate, base_seed=int(args.seed)))
    curves = pd.DataFrame(rows)
    recommendations = _recommendations(curves)
    summary = _summarize(curves, recommendations)
    design["curve_row_count"] = int(len(curves))
    design["recommendation_row_count"] = int(len(recommendations))
    design["private_manifest_role"] = "simulated sentinel responses and post-freeze evaluation only"
    _atomic_json(design, output / "design.json")
    _atomic_csv(curves, output / "residual_curves.csv")
    _atomic_csv(recommendations, output / "recommendations.csv")
    _atomic_json(summary, output / "summary.json")
    _write_report(summary, output / "DEVELOPMENT_REPORT.md")
    print(f"[prefix-cert-complete] {output / 'summary.json'}", flush=True)
    return output / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(str(path) for path in DEFAULT_CONFIGS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise", default="uniform20,minority_high_40")
    parser.add_argument("--sentinel-replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
