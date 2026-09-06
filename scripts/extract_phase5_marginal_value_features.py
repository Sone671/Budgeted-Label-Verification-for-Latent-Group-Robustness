#!/usr/bin/env python
"""Extract Phase-5 legal marginal-value features without loading WGA outcomes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame
from robust_verify.config import load_config, output_layout
from robust_verify.features import load_features
from robust_verify.joint_budget import (
    estimate_soft_group_error,
    estimate_staged_marginal_value,
    fit_attribute_probability_model,
    joint_repair_scores,
    relative_weak_group_error_margin,
    select_class_balanced_audits,
    select_marginal_value_audit_share,
)
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import predict_from_state
from robust_verify.utils import budget_to_count


def _binary_entropy(probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))))


def _binary_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _budget_counts(
    train_count: int,
    *,
    total_fraction: float,
    subscout_share: float,
    scout_share: float,
    expanded_share: float,
) -> dict[str, int]:
    total = budget_to_count(total_fraction, train_count)

    def label_count(share: float) -> int:
        return budget_to_count(total_fraction * (1.0 - share), train_count)

    labels_subscout = label_count(subscout_share)
    labels_scout = label_count(scout_share)
    labels_expanded = label_count(expanded_share)
    return {
        "total": total,
        "audit_subscout": total - labels_subscout,
        "audit_scout": total - labels_scout,
        "audit_expanded": total - labels_expanded,
        "labels_scout": labels_scout,
        "labels_expanded": labels_expanded,
    }


def extract(
    config_paths: list[Path],
    output_path: Path,
    *,
    seeds: list[int],
    noises: list[str],
    total_fraction: float,
    subscout_share: float,
    scout_share: float,
    expanded_share: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        source_root = Path(config["project"].get("input_dir", config["project"]["output_dir"]))
        if not source_root.is_absolute():
            source_root = ROOT / source_root
        layout = output_layout(source_root)
        dataset = str(config.get("data", {}).get("dataset", "waterbirds"))

        train_np, train_ids = load_features(layout["features"] / "train.npz")
        val_np, val_ids = load_features(layout["features"] / "val.npz")
        val_public = align_frame(pd.read_csv(layout["manifests"] / "val_public.csv"), val_ids)
        val_private = align_frame(pd.read_csv(layout["manifests"] / "val_private.csv"), val_ids)
        val_labels = val_public["label"].to_numpy(dtype=np.int64)
        # The complete vector is loaded by the simulator, but every downstream
        # feature indexes it only at the paid scout positions.
        simulated_private_attribute = val_private["place"].to_numpy(dtype=np.int64)
        counts = _budget_counts(
            len(train_ids),
            total_fraction=total_fraction,
            subscout_share=subscout_share,
            scout_share=scout_share,
            expanded_share=expanded_share,
        )
        if counts["audit_expanded"] > len(val_ids):
            raise ValueError(f"expanded audit is infeasible for {dataset}")

        attribute_cache: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            sub_positions = select_class_balanced_audits(
                val_labels, counts["audit_subscout"], seed=100_000 + seed
            )
            scout_positions = select_class_balanced_audits(
                val_labels, counts["audit_scout"], seed=100_000 + seed
            )
            if not np.array_equal(sub_positions, scout_positions[: len(sub_positions)]):
                raise ValueError(f"nested audit prefix failed for {dataset} seed={seed}")
            revealed_positions = scout_positions
            revealed_attribute = simulated_private_attribute[revealed_positions]
            sub_attribute = revealed_attribute[: len(sub_positions)]
            sub_model = fit_attribute_probability_model(
                val_np[sub_positions], sub_attribute, seed=seed
            )
            scout_model = fit_attribute_probability_model(
                val_np[scout_positions], revealed_attribute, seed=seed
            )
            train_sub_probability = sub_model.predict_probability(train_np)
            train_scout_probability = scout_model.predict_probability(train_np)
            val_sub_probability = sub_model.predict_probability(val_np)
            val_scout_probability = scout_model.predict_probability(val_np)
            holdout_positions = scout_positions[len(sub_positions) :]
            holdout_attribute = simulated_private_attribute[holdout_positions]
            holdout_probability = val_sub_probability[holdout_positions]
            attribute_cache[seed] = {
                "sub_model_fitted": sub_model.fitted,
                "scout_model_fitted": scout_model.fitted,
                "train_sub_probability": train_sub_probability,
                "train_scout_probability": train_scout_probability,
                "val_sub_probability": val_sub_probability,
                "val_scout_probability": val_scout_probability,
                "sub_positive_count": int(sub_attribute.sum()),
                "scout_positive_count": int(revealed_attribute.sum()),
                "holdout_log_loss": _binary_log_loss(holdout_attribute, holdout_probability),
                "holdout_brier": float(np.mean((holdout_attribute - holdout_probability) ** 2)),
            }

        val_features = torch.from_numpy(val_np).float()
        for noise_name in noises:
            for seed in seeds:
                prefix = f"{noise_name}_seed{seed}"
                public = align_frame(
                    pd.read_csv(layout["manifests"] / f"{prefix}_train_public.csv"),
                    train_ids,
                )
                noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
                checkpoint = torch.load(
                    layout["probes"] / f"{prefix}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                dynamics = np.load(layout["probes"] / f"{prefix}.dynamics.npz")
                train_probabilities = dynamics["train_probabilities"].astype(np.float64)
                scores = build_legal_scores(
                    train_probabilities,
                    noisy_labels,
                    dynamics["correctness_history"],
                )
                _, val_probabilities = predict_from_state(
                    checkpoint["best_state"],
                    val_features,
                    int(train_np.shape[1]),
                    int(train_probabilities.shape[1]),
                    config["training"],
                )
                cached = attribute_cache[seed]
                error_sub, _ = estimate_soft_group_error(
                    val_probabilities,
                    val_labels,
                    cached["val_sub_probability"],
                )
                error_scout, mass_scout = estimate_soft_group_error(
                    val_probabilities,
                    val_labels,
                    cached["val_scout_probability"],
                )
                score_sub, risk_sub = joint_repair_scores(
                    scores["noise_score"],
                    train_probabilities,
                    cached["train_sub_probability"],
                    error_sub,
                )
                score_scout, risk_scout = joint_repair_scores(
                    scores["noise_score"],
                    train_probabilities,
                    cached["train_scout_probability"],
                    error_scout,
                )
                ranking_sub = descending_ranking(score_sub, scores["loss"])
                ranking_scout = descending_ranking(score_scout, scores["loss"])
                estimate = estimate_staged_marginal_value(
                    ranking_sub,
                    ranking_scout,
                    score_scout,
                    expanded_label_count=counts["labels_expanded"],
                    scout_label_count=counts["labels_scout"],
                    observed_audit_increment=(
                        counts["audit_scout"] - counts["audit_subscout"]
                    ),
                    next_audit_increment=(
                        counts["audit_expanded"] - counts["audit_scout"]
                    ),
                )
                selected_share = select_marginal_value_audit_share(
                    estimate,
                    attribute_model_fitted=bool(cached["scout_model_fitted"]),
                    scout_share=scout_share,
                    expanded_share=expanded_share,
                )
                k = counts["labels_expanded"]
                overlap = len(
                    np.intersect1d(ranking_sub[:k], ranking_scout[:k], assume_unique=True)
                )
                scout_score_top = float(score_scout[ranking_scout[:k]].sum())
                opportunity_noise = float(
                    scores["noise_score"][
                        ranking_scout[counts["labels_expanded"] : counts["labels_scout"]]
                    ].sum()
                )
                rows.append({
                    "feature_schema": "phase5a_v1",
                    "dataset": dataset,
                    "noise_name": noise_name,
                    "seed": seed,
                    "total_budget_count": counts["total"],
                    "subscout_audit_count": counts["audit_subscout"],
                    "scout_audit_count": counts["audit_scout"],
                    "expanded_audit_count": counts["audit_expanded"],
                    "scout_label_count": counts["labels_scout"],
                    "expanded_label_count": counts["labels_expanded"],
                    "subscout_attribute_model_fitted": bool(cached["sub_model_fitted"]),
                    "scout_attribute_model_fitted": bool(cached["scout_model_fitted"]),
                    "subscout_attribute_positive_count": cached["sub_positive_count"],
                    "scout_attribute_positive_count": cached["scout_positive_count"],
                    "subscout_holdout_attribute_log_loss": cached["holdout_log_loss"],
                    "subscout_holdout_attribute_brier": cached["holdout_brier"],
                    "train_attribute_posterior_shift": float(
                        np.mean(
                            np.abs(
                                cached["train_scout_probability"]
                                - cached["train_sub_probability"]
                            )
                        )
                    ),
                    "validation_attribute_posterior_shift": float(
                        np.mean(
                            np.abs(
                                cached["val_scout_probability"]
                                - cached["val_sub_probability"]
                            )
                        )
                    ),
                    "scout_attribute_entropy": _binary_entropy(
                        cached["val_scout_probability"]
                    ),
                    "soft_group_error_shift_l1": float(np.abs(error_scout - error_sub).sum()),
                    "soft_weak_group_stable": bool(np.argmax(error_sub) == np.argmax(error_scout)),
                    "scout_relative_weak_group_margin": relative_weak_group_error_margin(
                        error_scout
                    ),
                    "scout_min_soft_group_mass_fraction": float(
                        mass_scout.min() / max(mass_scout.sum(), 1e-12)
                    ),
                    "mean_expected_group_risk_shift": float(
                        np.mean(np.abs(risk_scout - risk_sub))
                    ),
                    "expanded_topk_turnover": float(1.0 - overlap / k),
                    "revealed_information_gain": estimate.revealed_information_gain,
                    "extrapolation_scale": estimate.extrapolation_scale,
                    "extrapolated_information_gain": estimate.extrapolated_information_gain,
                    "label_opportunity_cost": estimate.label_opportunity_cost,
                    "marginal_net_value": estimate.net_value,
                    "marginal_value_ratio": estimate.value_ratio,
                    "revealed_gain_fraction_of_topk_value": float(
                        estimate.revealed_information_gain / max(scout_score_top, 1e-12)
                    ),
                    "opportunity_cost_fraction_of_topk_value": float(
                        estimate.label_opportunity_cost / max(scout_score_top, 1e-12)
                    ),
                    "forgone_noise_score_mass": opportunity_noise,
                    "selected_group_share": selected_share,
                    "expand_by_marginal_value": bool(np.isclose(selected_share, expanded_share)),
                })

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "noise_name", "seed"], kind="stable"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metadata = {
        "feature_schema": "phase5a_v1",
        "rows": len(result),
        "configs": [str(path) for path in config_paths],
        "contains_outcome_columns": False,
        "forbidden_inputs_not_loaded": [
            "phase4 WGA outcomes",
            "test labels/groups/WGA",
            "clean training labels",
            "validation place outside paid scout positions in feature computations",
        ],
    }
    output_path.with_suffix(".design.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result


def _csv_values(raw: str, cast: Any) -> list[Any]:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--noises", default="uniform20,minority_high_40")
    parser.add_argument("--total-budget", type=float, default=0.02)
    parser.add_argument("--subscout-share", type=float, default=0.125)
    parser.add_argument("--scout-share", type=float, default=0.25)
    parser.add_argument("--expanded-share", type=float, default=0.50)
    args = parser.parse_args()
    extract(
        [Path(path).resolve() for path in args.configs],
        Path(args.output).resolve(),
        seeds=_csv_values(args.seeds, int),
        noises=_csv_values(args.noises, str),
        total_fraction=args.total_budget,
        subscout_share=args.subscout_share,
        scout_share=args.scout_share,
        expanded_share=args.expanded_share,
    )


if __name__ == "__main__":
    main()

