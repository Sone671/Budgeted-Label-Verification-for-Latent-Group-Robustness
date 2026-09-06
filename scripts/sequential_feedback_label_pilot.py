#!/usr/bin/env python
"""Run the Phase-6 sequential label-feedback ranking pilot."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame, query_composition
from robust_verify.config import load_config, output_layout
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.features import load_features
from robust_verify.joint_budget import (
    estimate_feedback_group_noise_rates,
    estimate_ipw_feedback_group_noise_rates,
    estimate_soft_group_error,
    feedback_updated_joint_scores,
    fit_attribute_probability_model,
    joint_repair_scores,
    select_class_balanced_audits,
    select_soft_group_stratified_exploration,
    soft_group_probabilities,
)
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count


def _csv_values(raw: str, cast: Any) -> list[Any]:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated argument must contain values")
    return values


def _query_diagnostics(
    positions: np.ndarray, private: pd.DataFrame, *, weak_group: int
) -> dict[str, int]:
    groups = private["group"].to_numpy(dtype=np.int64)
    noisy = private["is_noisy"].to_numpy(dtype=bool)
    selected_groups = groups[positions]
    selected_noisy = noisy[positions]
    result = {
        "weak_group_queries": int((selected_groups == weak_group).sum()),
        "weak_group_corrections": int(
            ((selected_groups == weak_group) & selected_noisy).sum()
        ),
    }
    for group in sorted(np.unique(groups).tolist()):
        result[f"query_group_{group}_count"] = int((selected_groups == group).sum())
        result[f"corrected_group_{group}_count"] = int(
            ((selected_groups == group) & selected_noisy).sum()
        )
    return result


def run(args: argparse.Namespace) -> pd.DataFrame:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    source_root = Path(config["project"].get("input_dir", config["project"]["output_dir"]))
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    layout = output_layout(source_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.csv"

    train_np, train_ids = load_features(layout["features"] / "train.npz")
    val_np, val_ids = load_features(layout["features"] / "val.npz")
    test_np, test_ids = load_features(layout["features"] / "test.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = torch.from_numpy(train_np).float().to(device)
    val_features = torch.from_numpy(val_np).float().to(device)
    test_features = torch.from_numpy(test_np).float().to(device)

    val_public = align_frame(pd.read_csv(layout["manifests"] / "val_public.csv"), val_ids)
    val_private = align_frame(pd.read_csv(layout["manifests"] / "val_private.csv"), val_ids)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    simulated_val_attribute = val_private["place"].to_numpy(dtype=np.int64)
    evaluator = PrivateEvaluator(layout["manifests"] / "test_private.csv")
    dataset = str(config.get("data", {}).get("dataset", "waterbirds"))

    total_count = budget_to_count(args.total_budget, len(train_ids))
    total_label_count = budget_to_count(
        args.total_budget * (1.0 - args.group_share), len(train_ids)
    )
    group_audit_count = total_count - total_label_count
    first_label_count = budget_to_count(
        args.total_budget * args.first_label_batch_share, len(train_ids)
    )
    second_label_count = total_label_count - first_label_count
    exploration_label_count = int(
        round(first_label_count * args.first_batch_exploration_fraction)
    )
    exploitation_label_count = first_label_count - exploration_label_count
    if min(group_audit_count, first_label_count, second_label_count) <= 0:
        raise ValueError("all sequential action batches must be positive")
    if min(exploitation_label_count, exploration_label_count) < 0:
        raise ValueError("invalid first-batch exploration fraction")
    if group_audit_count > len(val_ids):
        raise ValueError("group audit count exceeds validation pool")

    prior_rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if result_path.exists():
        prior = pd.read_csv(result_path)
        prior_rows = prior.to_dict("records")
        completed = {
            (str(row["dataset"]), str(row["noise_name"]), int(row["seed"]))
            for row in prior_rows
        }

    rows = prior_rows
    for noise_name in args.noises:
        for seed in args.seeds:
            key = (dataset, noise_name, seed)
            if key in completed:
                continue
            prefix = f"{noise_name}_seed{seed}"
            public_path = layout["manifests"] / f"{prefix}_train_public.csv"
            private_path = layout["manifests"] / f"{prefix}_train_private.csv"
            public = align_frame(pd.read_csv(public_path), train_ids)
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)

            checkpoint = torch.load(
                layout["probes"] / f"{prefix}.pt",
                map_location="cpu",
                weights_only=False,
            )
            dynamics = np.load(layout["probes"] / f"{prefix}.dynamics.npz")
            train_probabilities = dynamics["train_probabilities"].astype(np.float64)
            scores = build_legal_scores(
                train_probabilities, noisy_labels, dynamics["correctness_history"]
            )
            val_predictions, val_probabilities = predict_from_state(
                checkpoint["best_state"],
                val_features,
                int(train_np.shape[1]),
                int(train_probabilities.shape[1]),
                config["training"],
            )
            test_predictions, _ = predict_from_state(
                checkpoint["best_state"],
                test_features,
                int(train_np.shape[1]),
                int(train_probabilities.shape[1]),
                config["training"],
            )
            baseline = evaluator.evaluate(test_ids, test_predictions)
            test_group_accuracy = {
                int(name.split("_")[1]): float(value)
                for name, value in baseline.items()
                if name.startswith("group_") and name.endswith("_accuracy")
            }
            actual_test_weak_group = min(test_group_accuracy, key=test_group_accuracy.get)

            audit_positions = select_class_balanced_audits(
                val_labels, group_audit_count, seed=100_000 + seed
            )
            audit_ids = val_ids[audit_positions]
            audit_path = output_dir / "audits" / dataset / prefix / "place_audit.npy"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(audit_path, audit_ids)
            audited_attribute = simulated_val_attribute[audit_positions]
            attribute_model = fit_attribute_probability_model(
                val_np[audit_positions], audited_attribute, seed=seed
            )
            train_attribute_probability = attribute_model.predict_probability(train_np)
            val_attribute_probability = attribute_model.predict_probability(val_np)
            group_error, group_mass = estimate_soft_group_error(
                val_probabilities, val_labels, val_attribute_probability
            )
            initial_score, _ = joint_repair_scores(
                scores["noise_score"],
                train_probabilities,
                train_attribute_probability,
                group_error,
            )
            initial_ranking = descending_ranking(initial_score, scores["loss"])
            exploitation_positions = initial_ranking[:exploitation_label_count]
            train_group_probability = soft_group_probabilities(
                train_probabilities, train_attribute_probability
            )
            if exploration_label_count:
                exploration_positions, exploration_inclusion_probability = (
                    select_soft_group_stratified_exploration(
                        train_group_probability,
                        exploration_label_count,
                        seed=200_000 + seed,
                        exclude=exploitation_positions,
                    )
                )
            else:
                exploration_positions = np.empty(0, dtype=np.int64)
                exploration_inclusion_probability = np.empty(0, dtype=np.float64)
            first_positions = np.concatenate(
                [exploitation_positions, exploration_positions]
            )
            first_ids = train_ids[first_positions]
            oracle = VerificationOracle(private_path)
            first_clean_labels = oracle.verify(first_ids)
            first_error = (first_clean_labels != noisy_labels[first_positions]).astype(
                np.float64
            )

            if exploration_label_count:
                exploration_error = first_error[exploitation_label_count:]
                feedback_rate = estimate_ipw_feedback_group_noise_rates(
                    train_group_probability[exploration_positions],
                    exploration_error,
                    exploration_inclusion_probability,
                )
            else:
                exploration_error = np.empty(0, dtype=np.float64)
                feedback_rate = estimate_feedback_group_noise_rates(
                    train_group_probability[first_positions], first_error
                )
            updated_score, _ = feedback_updated_joint_scores(
                scores["noise_score"],
                train_group_probability,
                group_error,
                feedback_rate,
            )
            updated_ranking = descending_ranking(updated_score, scores["loss"])
            available = updated_ranking[
                ~np.isin(updated_ranking, first_positions, assume_unique=True)
            ]
            second_positions = available[:second_label_count]
            final_positions = np.concatenate([first_positions, second_positions])
            if len(np.unique(final_positions)) != total_label_count:
                raise ValueError("sequential label query union has duplicates or wrong size")

            query_dir = output_dir / "queries" / dataset / prefix
            query_dir.mkdir(parents=True, exist_ok=True)
            first_query_path = query_dir / "first_label_batch.npy"
            exploitation_query_path = query_dir / "first_exploitation_batch.npy"
            exploration_query_path = query_dir / "first_exploration_batch.npy"
            exploration_probability_path = query_dir / "exploration_inclusion_probability.npy"
            second_query_path = query_dir / "second_label_batch.npy"
            final_query_path = query_dir / "final_label_union.npy"
            np.save(first_query_path, first_ids)
            np.save(exploitation_query_path, train_ids[exploitation_positions])
            np.save(exploration_query_path, train_ids[exploration_positions])
            np.save(exploration_probability_path, exploration_inclusion_probability)
            np.save(second_query_path, train_ids[second_positions])
            np.save(final_query_path, train_ids[final_positions])

            repaired = noisy_labels.copy()
            repaired[first_positions] = first_clean_labels
            repaired[second_positions] = oracle.verify(train_ids[second_positions])
            retrained = train_linear_head(
                train_features=train_features,
                train_labels=repaired,
                val_features=val_features,
                val_labels=val_labels,
                config=config["training"],
                seed=seed,
                initial_state=copy.deepcopy(checkpoint["initial_state"]),
                record_dynamics=False,
            )
            predictions, _ = predict_from_state(
                retrained.best_state,
                test_features,
                int(train_np.shape[1]),
                int(train_probabilities.shape[1]),
                config["training"],
            )
            metrics = evaluator.evaluate(test_ids, predictions)

            # Offline diagnostics are opened only after both query batches are frozen.
            private = align_frame(pd.read_csv(private_path), train_ids)
            composition = query_composition(final_positions, private)
            row: dict[str, Any] = {
                "role": (
                    "sequential_exploration_feedback_25pct_group"
                    if exploration_label_count
                    else "sequential_feedback_25pct_group"
                ),
                "dataset": dataset,
                "noise_name": noise_name,
                "seed": seed,
                "total_budget_fraction": args.total_budget,
                "total_budget_count": total_count,
                "group_share": args.group_share,
                "group_audit_count": group_audit_count,
                "label_query_count": total_label_count,
                "first_label_query_count": first_label_count,
                "first_exploitation_query_count": exploitation_label_count,
                "first_exploration_query_count": exploration_label_count,
                "second_label_query_count": second_label_count,
                "human_action_count": group_audit_count + total_label_count,
                "first_batch_corrections": int(first_error.sum()),
                "first_batch_noise_precision": float(first_error.mean()),
                "exploration_batch_corrections": int(exploration_error.sum()),
                "exploration_batch_noise_precision": (
                    float(exploration_error.mean())
                    if len(exploration_error)
                    else float("nan")
                ),
                "exploration_inclusion_probability_min": (
                    float(exploration_inclusion_probability.min())
                    if len(exploration_inclusion_probability)
                    else float("nan")
                ),
                "exploration_inclusion_probability_max": (
                    float(exploration_inclusion_probability.max())
                    if len(exploration_inclusion_probability)
                    else float("nan")
                ),
                "wga": float(metrics["wga"]),
                "baseline_wga": float(baseline["wga"]),
                "delta_wga": float(metrics["wga"] - baseline["wga"]),
                "average_accuracy": float(metrics["average_accuracy"]),
                "delta_average_accuracy": float(
                    metrics["average_accuracy"] - baseline["average_accuracy"]
                ),
                "num_corrected": int(composition["num_corrected"]),
                "noise_precision": float(composition["noise_precision"]),
                "attribute_model_fitted": bool(attribute_model.fitted),
                "audited_attribute_positive_count": int(audited_attribute.sum()),
                "attribute_balanced_accuracy_offline": float(
                    balanced_accuracy_score(
                        simulated_val_attribute,
                        (val_attribute_probability >= 0.5).astype(np.int64),
                    )
                ),
                "audit_path": str(audit_path),
                "first_query_path": str(first_query_path),
                "exploitation_query_path": str(exploitation_query_path),
                "exploration_query_path": str(exploration_query_path),
                "exploration_probability_path": str(exploration_probability_path),
                "second_query_path": str(second_query_path),
                "query_path": str(final_query_path),
                **_query_diagnostics(
                    final_positions, private, weak_group=actual_test_weak_group
                ),
            }
            for group, value in enumerate(feedback_rate):
                row[f"feedback_group_{group}_noise_rate"] = float(value)
                row[f"estimated_group_{group}_error"] = float(group_error[group])
                row[f"estimated_group_{group}_mass"] = float(group_mass[group])
            rows.append(row)
            completed.add(key)
            pd.DataFrame(rows).sort_values(
                ["dataset", "noise_name", "seed"], kind="stable"
            ).to_csv(result_path, index=False)
            print(
                f"{dataset}/{noise_name} seed={seed} feedback="
                f"{int(first_error.sum())}/{first_label_count} "
                f"dWGA={row['delta_wga']:+.4f}"
            )
            sys.stdout.flush()

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "noise_name", "seed"], kind="stable"
    )
    result.to_csv(result_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noises", default="uniform20,minority_high_40")
    parser.add_argument("--total-budget", type=float, default=0.02)
    parser.add_argument("--group-share", type=float, default=0.25)
    parser.add_argument("--first-label-batch-share", type=float, default=0.25)
    parser.add_argument(
        "--first-batch-exploration-fraction", type=float, default=0.0
    )
    args = parser.parse_args()
    args.seeds = _csv_values(args.seeds, int)
    args.noises = _csv_values(args.noises, str)
    run(args)


if __name__ == "__main__":
    main()
