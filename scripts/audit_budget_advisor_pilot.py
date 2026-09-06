#!/usr/bin/env python
"""Old-seed pilot for a transfer-calibrated label-audit budget advisor."""

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

from robust_verify.analysis import align_frame
from robust_verify.audit_budget import (
    AuditStratum,
    beta_residual_posterior,
    build_band_class_strata,
    draw_stratified_scout,
)
from robust_verify.config import load_config
from robust_verify.scoring import build_legal_scores, descending_ranking


DEFAULT_CONFIGS = (
    ROOT / "configs" / "ablation_waterbirds_10seeds.yaml",
    ROOT / "configs" / "ablation_celeba_10seeds.yaml",
    ROOT / "configs" / "ablation_civilcomments_10seeds.yaml",
)
DEFAULT_OUTPUT = ROOT / "outputs" / "audit_budget_advisor_pilot"
BAND_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
CANDIDATE_BUDGETS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
QUALITY_TARGETS = (0.10, 0.15, 0.20)


@dataclass
class Condition:
    dataset: str
    noise_name: str
    seed: int
    observed_labels: np.ndarray
    is_error: np.ndarray
    strata: list[AuditStratum]
    population_size: int


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_name(config: dict[str, Any], source_root: Path) -> str:
    explicit = str(config.get("data", {}).get("dataset", "")).strip().lower()
    if explicit:
        return explicit
    for name in ("waterbirds", "celeba", "civilcomments"):
        if name in source_root.name.lower():
            return name
    raise ValueError(f"cannot infer dataset from {source_root}")


def _parse_csv(raw: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated input must contain values")
    return values


def _load_conditions(args: argparse.Namespace) -> tuple[list[Condition], dict[str, Any]]:
    seeds = _parse_csv(args.seeds, int)
    noises = _parse_csv(args.noise, str)
    conditions: list[Condition] = []
    config_record: dict[str, Any] = {}
    for raw_config in _parse_csv(args.configs, str):
        config_path = Path(raw_config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_config(config_path)
        source_root = Path(config["project"]["output_dir"])
        if not source_root.is_absolute():
            source_root = ROOT / source_root
        dataset = _dataset_name(config, source_root)
        config_record[dataset] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "source_output": str(source_root),
        }
        print(f"[load-audit] dataset={dataset}", flush=True)
        for noise in noises:
            for seed in seeds:
                prefix = f"{noise}_seed{seed}"
                dynamics_path = source_root / "stage1" / "probes" / f"{prefix}.dynamics.npz"
                dynamics = np.load(dynamics_path)
                sample_ids = dynamics["sample_ids"].astype(np.int64)
                public = align_frame(
                    pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
                    sample_ids,
                )
                private = align_frame(
                    pd.read_csv(source_root / "manifests" / f"{prefix}_train_private.csv"),
                    sample_ids,
                )
                observed = public["noisy_label"].to_numpy(dtype=np.int64)
                legal_scores = build_legal_scores(
                    dynamics["train_probabilities"],
                    observed,
                    dynamics["correctness_history"],
                )
                ranking = descending_ranking(legal_scores["noise_score"], legal_scores["loss"])
                strata, _ = build_band_class_strata(ranking, observed, BAND_EDGES)
                conditions.append(
                    Condition(
                        dataset=dataset,
                        noise_name=noise,
                        seed=seed,
                        observed_labels=observed,
                        is_error=private["is_noisy"].to_numpy(dtype=bool),
                        strata=strata,
                        population_size=len(observed),
                    )
                )
                dynamics.close()
    return conditions, config_record


def _source_prior(conditions: list[Condition], held_out: str) -> np.ndarray:
    rates: list[list[float]] = [[] for _ in range(len(BAND_EDGES) - 1)]
    for condition in conditions:
        if condition.dataset == held_out:
            continue
        for stratum in condition.strata:
            rates[stratum.band].append(float(condition.is_error[stratum.indices].mean()))
    means = np.asarray([np.mean(values) for values in rates], dtype=np.float64)
    if not np.isfinite(means).all():
        raise RuntimeError(f"missing source prior band for held-out {held_out}")
    return means


def _reviewed_bands(budget: float) -> np.ndarray:
    upper_edges = np.asarray(BAND_EDGES[1:], dtype=np.float64)
    return upper_edges <= float(budget) + 1e-12


def _actual_residual(
    condition: Condition,
    selected_by_stratum: list[np.ndarray],
    reviewed_bands: np.ndarray,
) -> tuple[float, float]:
    residual_errors = 0
    remaining_count = 0
    for stratum, scout_indices in zip(condition.strata, selected_by_stratum):
        if reviewed_bands[stratum.band]:
            continue
        scout_mask = np.isin(stratum.indices, scout_indices, assume_unique=False)
        remaining = stratum.indices[~scout_mask]
        residual_errors += int(condition.is_error[remaining].sum())
        remaining_count += len(remaining)
    return (
        residual_errors / condition.population_size,
        1.0 - remaining_count / condition.population_size,
    )


def _run_curve(
    condition: Condition,
    prior_means: np.ndarray,
    *,
    scout_count: int,
    scout_replicate: int,
    args: argparse.Namespace,
    fold_index: int,
) -> list[dict[str, Any]]:
    scout_seed = (
        int(args.seed)
        + 1_000_003 * fold_index
        + 10_007 * condition.seed
        + 101 * scout_replicate
        + (0 if condition.noise_name == "uniform20" else 53)
    )
    selected, allocation = draw_stratified_scout(
        condition.strata, scout_count, seed=scout_seed
    )
    sizes = np.asarray([len(stratum.indices) for stratum in condition.strata], dtype=np.int64)
    errors = np.asarray(
        [int(condition.is_error[indices].sum()) for indices in selected], dtype=np.int64
    )
    bands = np.asarray([stratum.band for stratum in condition.strata], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(CANDIDATE_BUDGETS):
        reviewed = _reviewed_bands(budget)
        posterior_seed = scout_seed + 7919 * (budget_index + 1)
        transferred = beta_residual_posterior(
            stratum_sizes=sizes,
            scout_counts=allocation,
            scout_errors=errors,
            stratum_bands=bands,
            reviewed_bands=reviewed,
            prior_band_means=prior_means,
            prior_strength=float(args.prior_strength),
            population_size=condition.population_size,
            draws=int(args.posterior_draws),
            upper_probability=float(args.upper_probability),
            seed=posterior_seed,
        )
        target_only = beta_residual_posterior(
            stratum_sizes=sizes,
            scout_counts=allocation,
            scout_errors=errors,
            stratum_bands=bands,
            reviewed_bands=reviewed,
            prior_band_means=np.zeros_like(prior_means),
            prior_strength=0.0,
            population_size=condition.population_size,
            draws=int(args.posterior_draws),
            upper_probability=float(args.upper_probability),
            seed=posterior_seed + 3571,
        )
        actual, union_review_fraction = _actual_residual(condition, selected, reviewed)
        rows.append(
            {
                "dataset": condition.dataset,
                "noise_name": condition.noise_name,
                "seed": condition.seed,
                "scout_replicate": scout_replicate,
                "scout_count": scout_count,
                "scout_fraction": scout_count / condition.population_size,
                "prefix_budget_fraction": budget,
                "union_review_fraction": union_review_fraction,
                "actual_remaining_error_rate": actual,
                "transferred_posterior_mean": transferred.mean,
                "transferred_upper": transferred.upper,
                "transferred_upper_width": transferred.upper_width,
                "target_only_posterior_mean": target_only.mean,
                "target_only_upper": target_only.upper,
                "target_only_upper_width": target_only.upper_width,
                "transferred_covers": bool(transferred.upper + 1e-12 >= actual),
                "target_only_covers": bool(target_only.upper + 1e-12 >= actual),
            }
        )
    return rows


def _recommendations(curves: pd.DataFrame, *, upper_column: str) -> pd.DataFrame:
    if upper_column not in {"transferred_upper", "target_only_upper"}:
        raise ValueError("unsupported recommendation upper-bound column")
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "noise_name", "seed", "scout_replicate"]
    for key, episode in curves.groupby(keys, sort=True):
        episode = episode.sort_values("prefix_budget_fraction", kind="stable")
        for target in QUALITY_TARGETS:
            recommended_rows = episode.loc[episode[upper_column] <= target]
            oracle_rows = episode.loc[episode["actual_remaining_error_rate"] <= target]
            recommended = None if not len(recommended_rows) else recommended_rows.iloc[0]
            oracle = None if not len(oracle_rows) else oracle_rows.iloc[0]
            issued = recommended is not None
            oracle_feasible = oracle is not None
            safe = bool(
                issued and float(recommended["actual_remaining_error_rate"]) <= target
            )
            excess = np.nan
            if safe and oracle_feasible:
                excess = float(recommended["prefix_budget_fraction"]) - float(
                    oracle["prefix_budget_fraction"]
                )
            rows.append(
                {
                    "dataset": key[0],
                    "noise_name": key[1],
                    "seed": int(key[2]),
                    "scout_replicate": int(key[3]),
                    "quality_target": target,
                    "recommendation_issued": issued,
                    "recommended_prefix_budget": (
                        float(recommended["prefix_budget_fraction"]) if issued else np.nan
                    ),
                    "recommended_union_review_fraction": (
                        float(recommended["union_review_fraction"]) if issued else np.nan
                    ),
                    "actual_at_recommendation": (
                        float(recommended["actual_remaining_error_rate"]) if issued else np.nan
                    ),
                    "recommendation_safe": safe,
                    "oracle_feasible": oracle_feasible,
                    "oracle_prefix_budget": (
                        float(oracle["prefix_budget_fraction"]) if oracle_feasible else np.nan
                    ),
                    "excess_prefix_budget": excess,
                }
            )
    return pd.DataFrame(rows)


def _summarize(
    curves: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    primary: str,
    stage: str,
) -> dict[str, Any]:
    if primary not in {"transferred", "target_only"}:
        raise ValueError("unsupported primary advisor")
    summary: dict[str, Any] = {"primary": primary, "stage": stage, "held_out": {}}
    for dataset in sorted(curves["dataset"].unique()):
        curve = curves.loc[curves["dataset"] == dataset]
        rec = recommendations.loc[recommendations["dataset"] == dataset]
        issued = rec["recommendation_issued"].astype(bool)
        safe_issued = rec.loc[issued, "recommendation_safe"].astype(bool)
        width_ratio = curve["transferred_upper_width"] / curve[
            "target_only_upper_width"
        ].clip(lower=1e-12)
        primary_coverage = (
            float(curve["transferred_covers"].mean())
            if primary == "transferred"
            else float(curve["target_only_covers"].mean())
        )
        metrics = {
            "curve_rows": int(len(curve)),
            "primary_upper_coverage": primary_coverage,
            "transferred_upper_coverage": float(curve["transferred_covers"].mean()),
            "target_only_upper_coverage": float(curve["target_only_covers"].mean()),
            "mean_relative_width_reduction": float((1.0 - width_ratio).mean()),
            "recommendation_coverage": float(issued.mean()),
            "unsafe_recommendation_rate": (
                float((~safe_issued).mean()) if len(safe_issued) else 1.0
            ),
            "mean_excess_prefix_budget": (
                float(rec["excess_prefix_budget"].dropna().mean())
                if rec["excess_prefix_budget"].notna().any()
                else 1.0
            ),
        }
        gates = {
            "upper_coverage_at_least_0_90": metrics["primary_upper_coverage"] >= 0.90,
            "recommendation_coverage_at_least_0_30": metrics["recommendation_coverage"] >= 0.30,
            "unsafe_rate_at_most_0_10": metrics["unsafe_recommendation_rate"] <= 0.10,
            "mean_excess_budget_at_most_0_02": metrics["mean_excess_prefix_budget"] <= 0.02,
        }
        if primary == "transferred":
            gates["width_reduction_at_least_0_05"] = (
                metrics["mean_relative_width_reduction"] >= 0.05
            )
        summary["held_out"][dataset] = {
            "metrics": metrics,
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
    pass_count = sum(int(result["passed"]) for result in summary["held_out"].values())
    summary["decision"] = {
        "held_out_datasets_passed": pass_count,
        "old_seed_expansion_allowed": bool(stage == "pilot" and pass_count >= 2),
        "new_public_protocol_allowed": bool(
            stage == "old_seed_validation" and pass_count >= 2
        ),
        "new_seed_or_dataset_migration_allowed": False,
    }
    return summary


def _write_report(summary: dict[str, Any], path: Path) -> None:
    primary = str(summary["primary"])
    lines = [
        "# Audit Budget Advisor Pilot Report",
        "",
        f"Primary advisor: `{primary}`.",
        "",
        "| Dataset | Upper coverage | Recommendation coverage | Unsafe issued | Excess prefix budget | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for dataset, result in summary["held_out"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {dataset} | {metrics['primary_upper_coverage']:.3f} | "
            f"{metrics['recommendation_coverage']:.3f} | "
            f"{metrics['unsafe_recommendation_rate']:.3f} | "
            f"{100 * metrics['mean_excess_prefix_budget']:.2f} pp | "
            f"{'YES' if result['passed'] else 'NO'} |"
        )
    decision = summary["decision"]
    if summary["stage"] == "pilot":
        next_line = (
            f"Proceed to old seeds 3--9: "
            f"**{'YES' if decision['old_seed_expansion_allowed'] else 'NO'}**."
        )
    else:
        next_line = (
            f"Freeze a new-public-data protocol: "
            f"**{'YES' if decision['new_public_protocol_allowed'] else 'NO'}**."
        )
    lines.extend(
        [
            "",
            f"Held-out datasets passed: {decision['held_out_datasets_passed']}/3.",
            "",
            next_line,
            "",
            "New-seed/new-dataset migration remains closed in this pilot.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    design = {
        "protocol": (
            "target_only_stratified_audit_advisor_v1"
            if args.primary == "target_only"
            else "transfer_calibrated_label_audit_budget_advisor_v1"
        ),
        "evidence_role": f"development_only_{args.stage}_seeds_{args.seeds.replace(',', '_')}",
        "band_edges": list(BAND_EDGES),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "quality_targets": list(QUALITY_TARGETS),
        "scout_count": int(args.scout_count),
        "scout_replicates": int(args.scout_replicates),
        "prior_strength": float(args.prior_strength),
        "posterior_draws": int(args.posterior_draws),
        "upper_probability": float(args.upper_probability),
        "seed": int(args.seed),
        "primary": str(args.primary),
        "stage": str(args.stage),
        "forbidden_outcomes": ["WGA", "test metrics", "group metrics", "retraining utility"],
    }
    _atomic_json(design, output_root / "design_pre_private.json")
    conditions, config_record = _load_conditions(args)
    design["configs"] = config_record
    curve_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    datasets = sorted({condition.dataset for condition in conditions})
    for fold_index, held_out in enumerate(datasets):
        prior = _source_prior(conditions, held_out)
        for band, mean in enumerate(prior):
            prior_rows.append(
                {"held_out_dataset": held_out, "band": band, "source_prior_mean": mean}
            )
        targets = [condition for condition in conditions if condition.dataset == held_out]
        for condition in targets:
            scout_count = min(int(args.scout_count), condition.population_size)
            if scout_count < len(condition.strata):
                raise ValueError("scout count is too small to cover all target strata")
            print(
                f"[audit-fold] held_out={held_out} noise={condition.noise_name} seed={condition.seed}",
                flush=True,
            )
            for scout_replicate in range(int(args.scout_replicates)):
                curve_rows.extend(
                    _run_curve(
                        condition,
                        prior,
                        scout_count=scout_count,
                        scout_replicate=scout_replicate,
                        args=args,
                        fold_index=fold_index,
                    )
                )
    curves = pd.DataFrame(curve_rows)
    recommendation_upper = (
        "transferred_upper" if args.primary == "transferred" else "target_only_upper"
    )
    recommendations = _recommendations(curves, upper_column=recommendation_upper)
    summary = _summarize(
        curves,
        recommendations,
        primary=args.primary,
        stage=args.stage,
    )
    design["condition_count"] = len(conditions)
    design["curve_row_count"] = len(curves)
    design["recommendation_row_count"] = len(recommendations)
    _atomic_csv(pd.DataFrame(prior_rows), output_root / "source_band_priors.csv")
    _atomic_csv(curves, output_root / "residual_curves.csv")
    _atomic_csv(recommendations, output_root / "recommendations.csv")
    _atomic_json(design, output_root / "design.json")
    _atomic_json(summary, output_root / "summary.json")
    _write_report(summary, output_root / "PILOT_REPORT.md")
    print(f"[audit-complete] {output_root / 'summary.json'}", flush=True)
    return output_root / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        default=",".join(str(path) for path in DEFAULT_CONFIGS),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise", default="uniform20,minority_high_40")
    parser.add_argument("--scout-count", type=int, default=200)
    parser.add_argument("--scout-replicates", type=int, default=100)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--upper-probability", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--primary",
        choices=("transferred", "target_only"),
        default="transferred",
    )
    parser.add_argument(
        "--stage",
        choices=("pilot", "old_seed_validation"),
        default="pilot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
