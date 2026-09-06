#!/usr/bin/env python
"""Run frozen v2 finite-population prefix certification on dopanim.

The script loads public human labels and probe trajectories to construct and
hash every ranking before opening the separate iNaturalist truth array.  It
then simulates the frozen random sentinel audit and reports data-quality
certificates only; no downstream model utility is evaluated.
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

from robust_verify.audit_budget import hypergeometric_upper_error_count
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.utils import budget_to_count


PUBLIC_ROOT = ROOT / "outputs" / "dopanim_public"
PROBE_ROOT = ROOT / "outputs" / "dopanim_audit_public"
DEFAULT_OUTPUT = ROOT / "outputs" / "finite_population_prefix_certification_dopanim_v2"
PROBE_SEEDS = (0, 1, 2)
PREFIX_BUDGETS = tuple(float(value) / 200.0 for value in range(41))
QUALITY_TARGETS = (0.20, 0.25, 0.30)
ALPHA = 0.05
SENTINEL_COUNT = 500
MAX_MEAN_EXCESS = 0.05


@dataclass(frozen=True)
class PublicCondition:
    seed: int
    observed_labels: np.ndarray
    ranking: np.ndarray
    score_sha256: str
    ranking_sha256: str


@dataclass(frozen=True)
class Condition:
    public: PublicCondition
    is_error: np.ndarray


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
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"frozen artifact differs from existing {path}")
        return
    _atomic_json(payload, path)


def _load_public_conditions(public_root: Path, probe_root: Path, seeds: tuple[int, ...]) -> tuple[list[PublicCondition], dict[str, Any]]:
    labels_path = public_root / "public" / "observed_labels.npy"
    manifest_path = public_root / "public" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
    if len(manifest) != len(labels) or not np.array_equal(manifest["sample_id"], np.arange(len(labels))):
        raise RuntimeError("unexpected dopanim public manifest")
    if labels.ndim != 1 or not ((0 <= labels).all() & (labels < 15).all()):
        raise RuntimeError("unexpected dopanim public labels")
    conditions: list[PublicCondition] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        dynamics_path = probe_root / "probes" / f"probe_seed{seed}.dynamics.npz"
        with np.load(dynamics_path, allow_pickle=False) as dynamics:
            sample_ids = dynamics["sample_ids"].astype(np.int64)
            probabilities = dynamics["train_probabilities"].astype(np.float64)
            history = dynamics["correctness_history"].astype(np.int8)
        if not np.array_equal(sample_ids, np.arange(len(labels))):
            raise RuntimeError(f"unexpected sample IDs for probe seed {seed}")
        if probabilities.shape != (len(labels), 15) or history.shape[0] != len(labels):
            raise RuntimeError(f"unexpected probe dynamics for seed {seed}")
        scores = build_legal_scores(probabilities, labels, history)
        ranking = descending_ranking(scores["noise_score"], scores["loss"])
        score_sha = _sha256_array(scores["noise_score"].astype(np.float64))
        ranking_sha = _sha256_array(ranking.astype(np.int64))
        conditions.append(
            PublicCondition(
                seed=seed,
                observed_labels=labels.copy(),
                ranking=ranking,
                score_sha256=score_sha,
                ranking_sha256=ranking_sha,
            )
        )
        records.append(
            {
                "seed": seed,
                "dynamics_path": str(dynamics_path),
                "dynamics_sha256": _sha256_file(dynamics_path),
                "score_sha256": score_sha,
                "ranking_sha256": ranking_sha,
                "population_size": len(labels),
            }
        )
    return conditions, {
        "protocol": "finite_population_prefix_certification_dopanim_v2",
        "dataset": "dopanim",
        "public_manifest_sha256": _sha256_file(manifest_path),
        "observed_labels_sha256": _sha256_file(labels_path),
        "probe_seeds": list(seeds),
        "prefix_budgets": list(PREFIX_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "sentinel_count": SENTINEL_COUNT,
        "familywise_alpha": ALPHA,
        "alpha_per_prefix": ALPHA / len(PREFIX_BUDGETS),
        "private_ground_truth_loaded": False,
        "records": records,
    }


def _attach_private_errors(conditions: list[PublicCondition], public_root: Path) -> list[Condition]:
    path = public_root / "private" / "audit_is_error.npy"
    is_error = np.load(path, allow_pickle=False).astype(bool)
    if is_error.shape != (len(conditions[0].observed_labels),):
        raise RuntimeError("unexpected dopanim private audit vector")
    return [Condition(public=condition, is_error=is_error) for condition in conditions]


def _run_episode(condition: Condition, *, replicate: int, base_seed: int) -> list[dict[str, Any]]:
    public = condition.public
    n = len(public.observed_labels)
    sentinel_seed = int(base_seed) + 1_000_003 * public.seed + 10_007 * replicate
    sentinel = np.random.default_rng(sentinel_seed).choice(n, size=SENTINEL_COUNT, replace=False)
    positions = np.empty(n, dtype=np.int64)
    positions[public.ranking] = np.arange(n, dtype=np.int64)
    sentinel_positions = positions[sentinel]
    alpha_per_prefix = ALPHA / len(PREFIX_BUDGETS)
    rows: list[dict[str, Any]] = []
    for budget in PREFIX_BUDGETS:
        prefix_count = budget_to_count(budget, n)
        suffix_size = n - prefix_count
        in_suffix = sentinel_positions >= prefix_count
        suffix_sentinel = sentinel[in_suffix]
        sampled_count = len(suffix_sentinel)
        sampled_errors = int(condition.is_error[suffix_sentinel].sum())
        upper_suffix_errors = hypergeometric_upper_error_count(
            population_size=suffix_size,
            sample_size=sampled_count,
            observed_errors=sampled_errors,
            alpha=alpha_per_prefix,
        )
        suffix = public.ranking[prefix_count:]
        actual_suffix_errors = int(condition.is_error[suffix].sum())
        actual_remaining = actual_suffix_errors - sampled_errors
        upper_remaining = upper_suffix_errors - sampled_errors
        remaining_count = suffix_size - sampled_count
        if min(actual_remaining, upper_remaining, remaining_count) < 0:
            raise RuntimeError("invalid dopanim sentinel accounting")
        rows.append(
            {
                "dataset": "dopanim",
                "seed": public.seed,
                "sentinel_replicate": replicate,
                "population_size": n,
                "sentinel_count": SENTINEL_COUNT,
                "prefix_budget_fraction": budget,
                "prefix_count": prefix_count,
                "sentinel_suffix_count": sampled_count,
                "sentinel_suffix_errors": sampled_errors,
                "union_review_fraction": 1.0 - remaining_count / n,
                "actual_remaining_error_rate": actual_remaining / n,
                "finite_population_upper": upper_remaining / n,
                "covers": bool(upper_remaining + 1e-12 >= actual_remaining),
                "score_sha256": public.score_sha256,
                "ranking_sha256": public.ranking_sha256,
            }
        )
    return rows


def _recommendations(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, episode in curves.groupby(["seed", "sentinel_replicate"], sort=True):
        episode = episode.sort_values("prefix_budget_fraction", kind="stable")
        for target in QUALITY_TARGETS:
            eligible = episode.loc[episode["finite_population_upper"] <= target]
            oracle_rows = episode.loc[episode["actual_remaining_error_rate"] <= target]
            chosen = None if eligible.empty else eligible.iloc[0]
            oracle = None if oracle_rows.empty else oracle_rows.iloc[0]
            issued = chosen is not None
            feasible = oracle is not None
            safe = bool(issued and float(chosen["actual_remaining_error_rate"]) <= target)
            excess = float(chosen["prefix_budget_fraction"] - oracle["prefix_budget_fraction"]) if issued and safe and feasible else np.nan
            cost_ratio = float(chosen["union_review_fraction"] / oracle["union_review_fraction"]) if issued and safe and feasible else np.nan
            rows.append(
                {
                    "dataset": "dopanim",
                    "seed": int(key[0]),
                    "sentinel_replicate": int(key[1]),
                    "quality_target": target,
                    "recommendation_issued": issued,
                    "recommended_prefix_budget": float(chosen["prefix_budget_fraction"]) if issued else np.nan,
                    "recommended_union_review_fraction": float(chosen["union_review_fraction"]) if issued else np.nan,
                    "actual_at_recommendation": float(chosen["actual_remaining_error_rate"]) if issued else np.nan,
                    "recommendation_safe": safe,
                    "oracle_feasible": feasible,
                    "oracle_prefix_budget": float(oracle["prefix_budget_fraction"]) if feasible else np.nan,
                    "oracle_union_review_fraction": float(oracle["union_review_fraction"]) if feasible else np.nan,
                    "excess_prefix_budget": excess,
                    "total_review_cost_ratio": cost_ratio,
                }
            )
    return pd.DataFrame(rows)


def _summarize(curves: pd.DataFrame, recommendations: pd.DataFrame) -> dict[str, Any]:
    episode_coverage = curves.groupby(["seed", "sentinel_replicate"], sort=True)["covers"].all()
    issued = recommendations["recommendation_issued"].astype(bool)
    safe_issued = recommendations.loc[issued, "recommendation_safe"].astype(bool)
    safe_excess = recommendations["excess_prefix_budget"].dropna()
    metrics = {
        "familywise_upper_coverage": float(episode_coverage.mean()),
        "individual_upper_coverage": float(curves["covers"].mean()),
        "recommendation_coverage": float(issued.mean()),
        "unsafe_recommendation_rate": float((~safe_issued).mean()) if len(safe_issued) else 1.0,
        "mean_excess_prefix_budget": float(safe_excess.mean()) if len(safe_excess) else 1.0,
        "median_excess_prefix_budget": float(safe_excess.median()) if len(safe_excess) else 1.0,
        "mean_issued_union_review_fraction": float(recommendations.loc[issued, "recommended_union_review_fraction"].mean()) if issued.any() else float("nan"),
        "mean_total_review_cost_ratio": float(recommendations["total_review_cost_ratio"].dropna().mean()) if recommendations["total_review_cost_ratio"].notna().any() else float("inf"),
    }
    gates = {
        "familywise_coverage_at_least_0_90": metrics["familywise_upper_coverage"] >= 0.90,
        "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
        "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
        "mean_excess_budget_at_most_0_05": metrics["mean_excess_prefix_budget"] <= MAX_MEAN_EXCESS,
    }
    return {
        "protocol": "finite_population_prefix_certification_dopanim_v2",
        "metrics": metrics,
        "gates": gates,
        "strong_efficiency_milestone_2pp": metrics["mean_excess_prefix_budget"] <= 0.02,
        "passed": bool(all(gates.values())),
    }


def _write_report(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["metrics"]
    gates = summary["gates"]
    lines = [
        "# dopanim Finite-Population Prefix Certification (v2)",
        "",
        "Public rankings and design hashes were frozen before the private iNaturalist truth was loaded.",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|:---:|",
        f"| Familywise upper-bound coverage | {metrics['familywise_upper_coverage']:.4f} | {'PASS' if gates['familywise_coverage_at_least_0_90'] else 'FAIL'} |",
        f"| Certificate availability | {metrics['recommendation_coverage']:.4f} | {'PASS' if gates['recommendation_coverage_at_least_0_30'] else 'FAIL'} |",
        f"| Unsafe certificate rate | {metrics['unsafe_recommendation_rate']:.4f} | {'PASS' if gates['unsafe_rate_at_most_0_10'] else 'FAIL'} |",
        f"| Mean excess prefix budget | {100 * metrics['mean_excess_prefix_budget']:.2f} pp | {'PASS' if gates['mean_excess_budget_at_most_0_05'] else 'FAIL'} |",
        f"| Mean total review-cost ratio | {metrics['mean_total_review_cost_ratio']:.3f} | report only |",
        "",
        f"Practical v2 result: **{'PASS' if summary['passed'] else 'FAIL'}**.",
        f"Strong 2pp efficiency milestone: **{'PASS' if summary['strong_efficiency_milestone_2pp'] else 'FAIL'}**.",
        "",
        "No WGA, test metric, group variable, retraining utility, source prior, or new NoiseScore claim is evaluated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or any(seed not in PROBE_SEEDS for seed in seeds):
        raise ValueError(f"seeds must be a non-empty subset of {PROBE_SEEDS}")
    public_root = Path(args.public_root).resolve()
    probe_root = Path(args.probe_root).resolve()
    conditions, ranking_manifest = _load_public_conditions(public_root, probe_root, seeds)
    design = {
        "protocol": "finite_population_prefix_certification_dopanim_v2",
        "protocol_document": str(ROOT / "finite_population_certification" / "DOPANIM_PROTOCOL_V2.md"),
        "dataset": "dopanim",
        "probe_seeds": list(seeds),
        "sentinel_count": SENTINEL_COUNT,
        "sentinel_repetitions_per_seed": int(args.sentinel_replicates),
        "prefix_budgets": list(PREFIX_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "familywise_alpha": ALPHA,
        "alpha_per_prefix": ALPHA / len(PREFIX_BUDGETS),
        "max_mean_excess_prefix_budget": MAX_MEAN_EXCESS,
        "strong_efficiency_milestone": 0.02,
        "forbidden_before_certificate": ["private ground truth outside sentinel", "WGA", "test metrics", "groups", "retraining utility", "source priors"],
        "ranking_manifest": "ranking_manifest_pre_private.json",
    }
    _freeze_json(design, output / "design_pre_private.json")
    _freeze_json(ranking_manifest, output / "ranking_manifest_pre_private.json")

    private_conditions = _attach_private_errors(conditions, public_root)
    rows: list[dict[str, Any]] = []
    for condition in private_conditions:
        print(f"[dopanim-cert] seed={condition.public.seed} ranking={condition.public.ranking_sha256[:12]}", flush=True)
        for replicate in range(int(args.sentinel_replicates)):
            rows.extend(_run_episode(condition, replicate=replicate, base_seed=int(args.seed)))
            if (replicate + 1) % 20 == 0:
                print(f"[dopanim-cert] seed={condition.public.seed} repetitions={replicate + 1}", flush=True)
    curves = pd.DataFrame(rows)
    recommendations = _recommendations(curves)
    summary = _summarize(curves, recommendations)
    design["curve_row_count"] = int(len(curves))
    design["recommendation_row_count"] = int(len(recommendations))
    design["private_truth_role"] = "sentinel simulation and post-freeze evaluation only"
    _atomic_json(design, output / "design.json")
    curves.to_csv(output / "residual_curves.csv", index=False)
    recommendations.to_csv(output / "recommendations.csv", index=False)
    _atomic_json(summary, output / "summary.json")
    _write_report(summary, output / "DOPANIM_CERTIFICATION_REPORT.md")
    print(f"[dopanim-cert-complete] passed={summary['passed']} output={output}", flush=True)
    return output / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", default=str(PUBLIC_ROOT))
    parser.add_argument("--probe-root", default=str(PROBE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sentinel-replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
