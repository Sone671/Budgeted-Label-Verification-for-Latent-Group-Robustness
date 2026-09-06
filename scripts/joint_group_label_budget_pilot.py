#!/usr/bin/env python
"""Pilot joint allocation of group-attribute audits and label verification.

The total human-action budget is fixed.  A declared share audits only the
binary ``place`` attribute on the public clean validation pool; the remaining
actions verify training labels.  Sparse audited attributes train a propagation
model, which estimates validation group risks and reweights NoiseScore by the
expected risk of the groups to which each training example may belong.

Private validation attributes and training labels are opened only through the
two simulated audit actions.  Full private information is used after query
selection for diagnostics and for a clearly labelled full-attribute
comparator.  That comparator fixes the same acquisition formula and is not a
global performance upper bound.
"""
from __future__ import annotations

import argparse
import copy
import json
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
    estimate_soft_group_error,
    fit_attribute_probability_model,
    joint_repair_scores,
    relative_weak_group_error_margin,
    select_class_balanced_audits,
)
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count


DEFAULT_CONFIG = ROOT / "configs" / "ablation_waterbirds_10seeds.yaml"


def _csv_values(raw: str, cast: Any) -> list[Any]:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def _actual_group_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[int, float]:
    return {
        int(group): float((predictions[groups == group] == labels[groups == group]).mean())
        for group in sorted(np.unique(groups).tolist())
    }


def _query_diagnostics(
    query_positions: np.ndarray,
    private: pd.DataFrame,
    *,
    weak_group: int,
) -> dict[str, float | int]:
    query_positions = np.asarray(query_positions, dtype=np.int64)
    groups = private["group"].to_numpy(dtype=np.int64)
    noisy = private["is_noisy"].to_numpy(dtype=bool)
    selected_groups = groups[query_positions]
    selected_noisy = noisy[query_positions]
    result: dict[str, float | int] = {
        "weak_group_queries": int((selected_groups == weak_group).sum()),
        "weak_group_corrections": int(
            ((selected_groups == weak_group) & selected_noisy).sum()
        ),
    }
    for group in sorted(np.unique(groups).tolist()):
        result[f"query_group_{int(group)}_count"] = int(
            (selected_groups == group).sum()
        )
        result[f"corrected_group_{int(group)}_count"] = int(
            ((selected_groups == group) & selected_noisy).sum()
        )
    return result


def _existing_noise_score_reference(
    existing: pd.DataFrame,
    *,
    noise_name: str,
    seed: int,
    label_count: int,
) -> pd.Series:
    subset = existing[
        (existing["noise_name"].astype(str) == noise_name)
        & (existing["seed"].astype(int) == seed)
        & (existing["method"].astype(str) == "noise_score")
        & (existing["budget_count"].astype(int) == label_count)
    ]
    if len(subset) != 1:
        raise ValueError(
            f"expected one cached NoiseScore row for {noise_name} seed={seed} "
            f"label_count={label_count}, found {len(subset)}"
        )
    return subset.iloc[0]


def _cached_query_positions(
    source_layout: dict[str, Path],
    train_ids: np.ndarray,
    *,
    noise_name: str,
    seed: int,
    budget_fraction: float,
) -> np.ndarray:
    path = (
        source_layout["queries"]
        / f"{noise_name}_seed{seed}"
        / f"noise_score_budget_{budget_fraction:.4f}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"cached NoiseScore query IDs are missing: {path}")
    query_ids = np.load(path).astype(np.int64)
    lookup = {int(sample_id): index for index, sample_id in enumerate(train_ids)}
    return np.asarray([lookup[int(sample_id)] for sample_id in query_ids], dtype=np.int64)


def _control_result(
    reference: pd.Series,
    query_positions: np.ndarray,
    private: pd.DataFrame,
    *,
    role: str,
    total_budget_fraction: float,
    total_budget_count: int,
    group_share: float,
    group_audit_count: int,
    weak_group: int,
    attribute_metadata: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "role": role,
        "noise_name": str(reference["noise_name"]),
        "seed": int(reference["seed"]),
        "total_budget_fraction": total_budget_fraction,
        "total_budget_count": total_budget_count,
        "group_share": group_share,
        "group_audit_count": group_audit_count,
        "label_query_count": int(reference["budget_count"]),
        "human_action_count": group_audit_count + int(reference["budget_count"]),
        "wga": float(reference["wga"]),
        "baseline_wga": float(reference["baseline_wga"]),
        "delta_wga": float(reference["delta_wga"]),
        "average_accuracy": float(reference["average_accuracy"]),
        "delta_average_accuracy": float(reference["delta_average_accuracy"]),
        "num_corrected": int(reference["num_corrected"]),
        "noise_precision": float(reference["noise_precision"]),
        **attribute_metadata,
        **_query_diagnostics(query_positions, private, weak_group=weak_group),
    }
    return row


def _retrain_result(
    *,
    role: str,
    score: np.ndarray,
    secondary_score: np.ndarray,
    label_query_count: int,
    train_ids: np.ndarray,
    train_features: torch.Tensor,
    noisy_labels: np.ndarray,
    private_path: Path,
    private: pd.DataFrame,
    val_features: torch.Tensor,
    val_labels: np.ndarray,
    test_features: torch.Tensor,
    test_ids: np.ndarray,
    evaluator: PrivateEvaluator,
    initial_state: dict[str, torch.Tensor],
    training_config: dict[str, Any],
    seed: int,
    baseline_wga: float,
    baseline_average_accuracy: float,
    weak_group: int,
    total_budget_fraction: float,
    total_budget_count: int,
    group_share: float,
    group_audit_count: int,
    attribute_metadata: dict[str, Any],
    query_path: Path,
) -> dict[str, Any]:
    ranking = descending_ranking(score, secondary_score)
    query_positions = ranking[:label_query_count]
    query_ids = train_ids[query_positions]
    query_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(query_path, query_ids)

    repaired = noisy_labels.copy()
    repaired[query_positions] = VerificationOracle(private_path).verify(query_ids)
    retrained = train_linear_head(
        train_features=train_features,
        train_labels=repaired,
        val_features=val_features,
        val_labels=val_labels,
        config=training_config,
        seed=seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
    )
    predictions, _ = predict_from_state(
        retrained.best_state,
        test_features,
        int(train_features.shape[1]),
        int(max(noisy_labels.max(), val_labels.max()) + 1),
        training_config,
    )
    metrics = evaluator.evaluate(test_ids, predictions)
    composition = query_composition(query_positions, private)
    return {
        "role": role,
        "noise_name": private_path.name.split("_seed", 1)[0],
        "seed": seed,
        "total_budget_fraction": total_budget_fraction,
        "total_budget_count": total_budget_count,
        "group_share": group_share,
        "group_audit_count": group_audit_count,
        "label_query_count": label_query_count,
        "human_action_count": group_audit_count + label_query_count,
        "wga": float(metrics["wga"]),
        "baseline_wga": baseline_wga,
        "delta_wga": float(metrics["wga"] - baseline_wga),
        "average_accuracy": float(metrics["average_accuracy"]),
        "delta_average_accuracy": float(
            metrics["average_accuracy"] - baseline_average_accuracy
        ),
        "num_corrected": int(composition["num_corrected"]),
        "noise_precision": float(composition["noise_precision"]),
        "query_path": str(query_path),
        **attribute_metadata,
        **_query_diagnostics(query_positions, private, weak_group=weak_group),
    }


def run_pilot(args: argparse.Namespace) -> pd.DataFrame:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    source_root = Path(
        config["project"].get("input_dir", config["project"]["output_dir"])
    )
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
    val_attributes = val_private["place"].to_numpy(dtype=np.int64)
    val_groups = val_private["group"].to_numpy(dtype=np.int64)
    evaluator = PrivateEvaluator(layout["manifests"] / "test_private.csv")
    existing = pd.read_csv(layout["stage1"] / "results.csv")

    dataset_name = str(config.get("data", {}).get("dataset", "waterbirds"))
    completed: set[tuple[str, str, int, float, str]] = set()
    rows: list[dict[str, Any]] = []
    if result_path.exists():
        prior = pd.read_csv(result_path)
        if "dataset" not in prior:
            prior["dataset"] = "waterbirds"
        rows = prior.to_dict("records")
        completed = {
            (
                str(row["dataset"]),
                str(row["noise_name"]),
                int(row["seed"]),
                float(row["group_share"]),
                str(row["role"]),
            )
            for row in rows
        }

    design = {
        "config": str(config_path),
        "dataset": dataset_name,
        "total_budget_fraction": args.total_budget,
        "group_shares": args.group_shares,
        "seeds": args.seeds,
        "noises": args.noises,
        "group_audit_pool": "public clean validation features; audit reveals place only",
        "label_verification_pool": "noisy training set; verification reveals clean label only",
        "selection": "class-balanced random validation audit, fixed before place is revealed",
        "joint_score": "NoiseScore times expected soft-group validation error",
        "primary_comparison": "joint_sparse_attribute vs label_only_same_label_budget",
        "cost_comparison": "joint_sparse_attribute vs label_only_total_budget",
        "oracle_attribute_interpretation": (
            "full-attribute information under the same score formula; not a global upper bound"
        ),
    }
    (output_dir / "design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_budget_count = budget_to_count(args.total_budget, len(train_ids))
    for noise_name in args.noises:
        for seed in args.seeds:
            prefix = f"{noise_name}_seed{seed}"
            public_path = layout["manifests"] / f"{prefix}_train_public.csv"
            private_path = layout["manifests"] / f"{prefix}_train_private.csv"
            public = align_frame(pd.read_csv(public_path), train_ids)
            private = align_frame(pd.read_csv(private_path), train_ids)
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)

            probe_path = layout["probes"] / f"{prefix}.pt"
            dynamics_path = layout["probes"] / f"{prefix}.dynamics.npz"
            checkpoint = torch.load(probe_path, map_location="cpu", weights_only=False)
            dynamics = np.load(dynamics_path)
            train_probabilities = dynamics["train_probabilities"].astype(np.float64)
            correctness_history = dynamics["correctness_history"]
            scores = build_legal_scores(
                train_probabilities, noisy_labels, correctness_history
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
            baseline_metrics = evaluator.evaluate(test_ids, test_predictions)
            val_group_accuracy = _actual_group_accuracy(
                val_predictions, val_labels, val_groups
            )
            actual_val_weak_group = min(val_group_accuracy, key=val_group_accuracy.get)
            test_group_accuracy = {
                int(key.split("_")[1]): float(value)
                for key, value in baseline_metrics.items()
                if key.startswith("group_") and key.endswith("_accuracy")
            }
            actual_test_weak_group = min(test_group_accuracy, key=test_group_accuracy.get)

            for group_share in args.group_shares:
                if not 0.0 < group_share < 1.0:
                    raise ValueError("group shares must lie strictly between zero and one")
                # Preserve the repository's canonical fractional label budgets
                # when the total count is odd (for example CelebA).  Any
                # one-action rounding remainder is assigned to group auditing.
                label_query_count = budget_to_count(
                    args.total_budget * (1.0 - group_share), len(train_ids)
                )
                group_audit_count = total_budget_count - label_query_count
                if group_audit_count > len(val_ids) or label_query_count < 1:
                    raise ValueError("budget split is infeasible")

                audit_positions = select_class_balanced_audits(
                    val_labels, group_audit_count, seed=100_000 + seed
                )
                audit_path = (
                    output_dir
                    / "audits"
                    / dataset_name
                    / prefix
                    / f"place_audit_share_{group_share:.3f}.npy"
                )
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(audit_path, val_ids[audit_positions])
                audited_attributes = val_attributes[audit_positions]
                attribute_model = fit_attribute_probability_model(
                    val_np[audit_positions], audited_attributes, seed=seed
                )
                train_attribute_probability = attribute_model.predict_probability(train_np)
                val_attribute_probability = attribute_model.predict_probability(val_np)
                group_error, group_mass = estimate_soft_group_error(
                    val_probabilities, val_labels, val_attribute_probability
                )
                joint_score, expected_group_risk = joint_repair_scores(
                    scores["noise_score"],
                    train_probabilities,
                    train_attribute_probability,
                    group_error,
                )
                estimated_weak_group = int(np.argmax(group_error))
                sparse_attribute_ba = float(
                    balanced_accuracy_score(
                        val_attributes,
                        (val_attribute_probability >= 0.5).astype(np.int64),
                    )
                )
                common_metadata: dict[str, Any] = {
                    "dataset": dataset_name,
                    "attribute_model": "sparse_audit_logistic",
                    "attribute_model_fitted": bool(attribute_model.fitted),
                    "audited_attribute_positive_count": int(audited_attributes.sum()),
                    "audited_attribute_positive_rate": float(audited_attributes.mean()),
                    "audit_path": str(audit_path),
                    "attribute_balanced_accuracy_offline": sparse_attribute_ba,
                    "estimated_weak_group": estimated_weak_group,
                    "actual_validation_weak_group_offline": actual_val_weak_group,
                    "actual_test_weak_group_offline": actual_test_weak_group,
                    "weak_group_identified_offline": int(
                        estimated_weak_group == actual_val_weak_group
                    ),
                    "mean_expected_group_risk": float(expected_group_risk.mean()),
                    "relative_weak_group_error_margin": (
                        relative_weak_group_error_margin(group_error)
                    ),
                }
                for group, value in enumerate(group_error):
                    common_metadata[f"estimated_group_{group}_error"] = float(value)
                    common_metadata[f"estimated_group_{group}_mass"] = float(group_mass[group])

                label_fraction = label_query_count / len(train_ids)
                full_reference = _existing_noise_score_reference(
                    existing,
                    noise_name=noise_name,
                    seed=seed,
                    label_count=total_budget_count,
                )
                try:
                    same_reference = _existing_noise_score_reference(
                        existing,
                        noise_name=noise_name,
                        seed=seed,
                        label_count=label_query_count,
                    )
                except ValueError:
                    # Phase-4 staged shares can leave a label count outside the
                    # canonical Stage-1 budget grid.  In that case, evaluate
                    # the exact-count NoiseScore control below instead of
                    # silently rounding to a cached budget.
                    same_reference = None
                full_positions = _cached_query_positions(
                    layout,
                    train_ids,
                    noise_name=noise_name,
                    seed=seed,
                    budget_fraction=float(full_reference["budget_fraction"]),
                )
                controls = [
                    (
                        "label_only_total_budget",
                        full_reference,
                        full_positions,
                        {**common_metadata, "attribute_model": "none"},
                    ),
                ]
                if same_reference is not None:
                    same_positions = _cached_query_positions(
                        layout,
                        train_ids,
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=float(same_reference["budget_fraction"]),
                    )
                    controls.append(
                        (
                            "label_only_same_label_budget",
                            same_reference,
                            same_positions,
                            {**common_metadata, "attribute_model": "none"},
                        )
                    )
                for role, reference, positions, metadata in controls:
                    key = (dataset_name, noise_name, seed, group_share, role)
                    if key not in completed:
                        rows.append(
                            _control_result(
                                reference,
                                positions,
                                private,
                                role=role,
                                total_budget_fraction=args.total_budget,
                                total_budget_count=total_budget_count,
                                group_share=group_share,
                                group_audit_count=(
                                    0 if role == "label_only_total_budget" else group_audit_count
                                ),
                                weak_group=actual_test_weak_group,
                                attribute_metadata=metadata,
                            )
                        )
                        completed.add(key)

                same_key = (
                    dataset_name,
                    noise_name,
                    seed,
                    group_share,
                    "label_only_same_label_budget",
                )
                if same_reference is None and same_key not in completed:
                    same_query_path = (
                        output_dir
                        / "queries"
                        / dataset_name
                        / prefix
                        / f"label_only_same_label_budget_share_{group_share:.3f}.npy"
                    )
                    rows.append(
                        _retrain_result(
                            role="label_only_same_label_budget",
                            score=scores["noise_score"],
                            secondary_score=scores["loss"],
                            label_query_count=label_query_count,
                            train_ids=train_ids,
                            train_features=train_features,
                            noisy_labels=noisy_labels,
                            private_path=private_path,
                            private=private,
                            val_features=val_features,
                            val_labels=val_labels,
                            test_features=test_features,
                            test_ids=test_ids,
                            evaluator=evaluator,
                            initial_state=checkpoint["initial_state"],
                            training_config=config["training"],
                            seed=seed,
                            baseline_wga=float(baseline_metrics["wga"]),
                            baseline_average_accuracy=float(
                                baseline_metrics["average_accuracy"]
                            ),
                            weak_group=actual_test_weak_group,
                            total_budget_fraction=args.total_budget,
                            total_budget_count=total_budget_count,
                            group_share=group_share,
                            group_audit_count=group_audit_count,
                            attribute_metadata={
                                **common_metadata,
                                "attribute_model": "none",
                            },
                            query_path=same_query_path,
                        )
                    )
                    completed.add(same_key)
                    pd.DataFrame(rows).to_csv(result_path, index=False)

                joint_key = (
                    dataset_name,
                    noise_name,
                    seed,
                    group_share,
                    "joint_sparse_attribute",
                )
                if joint_key not in completed:
                    query_path = (
                        output_dir
                        / "queries"
                        / dataset_name
                        / prefix
                        / f"joint_sparse_attribute_share_{group_share:.3f}.npy"
                    )
                    rows.append(
                        _retrain_result(
                            role="joint_sparse_attribute",
                            score=joint_score,
                            secondary_score=scores["loss"],
                            label_query_count=label_query_count,
                            train_ids=train_ids,
                            train_features=train_features,
                            noisy_labels=noisy_labels,
                            private_path=private_path,
                            private=private,
                            val_features=val_features,
                            val_labels=val_labels,
                            test_features=test_features,
                            test_ids=test_ids,
                            evaluator=evaluator,
                            initial_state=checkpoint["initial_state"],
                            training_config=config["training"],
                            seed=seed,
                            baseline_wga=float(baseline_metrics["wga"]),
                            baseline_average_accuracy=float(
                                baseline_metrics["average_accuracy"]
                            ),
                            weak_group=actual_test_weak_group,
                            total_budget_fraction=args.total_budget,
                            total_budget_count=total_budget_count,
                            group_share=group_share,
                            group_audit_count=group_audit_count,
                            attribute_metadata=common_metadata,
                            query_path=query_path,
                        )
                    )
                    completed.add(joint_key)
                    pd.DataFrame(rows).to_csv(result_path, index=False)

                oracle_key = (
                    dataset_name,
                    noise_name,
                    seed,
                    group_share,
                    "oracle_attribute",
                )
                if args.include_oracle_attribute and oracle_key not in completed:
                    oracle_error, oracle_mass = estimate_soft_group_error(
                        val_probabilities,
                        val_labels,
                        val_attributes.astype(np.float64),
                    )
                    oracle_score, oracle_expected_risk = joint_repair_scores(
                        scores["noise_score"],
                        train_probabilities,
                        private["place"].to_numpy(dtype=np.float64),
                        oracle_error,
                    )
                    oracle_metadata = {
                        **common_metadata,
                        "attribute_model": "oracle_attribute_offline",
                        "attribute_model_fitted": True,
                        "attribute_balanced_accuracy_offline": 1.0,
                        "estimated_weak_group": int(np.argmax(oracle_error)),
                        "weak_group_identified_offline": int(
                            int(np.argmax(oracle_error)) == actual_val_weak_group
                        ),
                        "mean_expected_group_risk": float(
                            oracle_expected_risk.mean()
                        ),
                    }
                    for group, value in enumerate(oracle_error):
                        oracle_metadata[f"estimated_group_{group}_error"] = float(value)
                        oracle_metadata[f"estimated_group_{group}_mass"] = float(
                            oracle_mass[group]
                        )
                    query_path = (
                        output_dir
                        / "queries"
                        / dataset_name
                        / prefix
                        / f"oracle_attribute_share_{group_share:.3f}.npy"
                    )
                    rows.append(
                        _retrain_result(
                            role="oracle_attribute",
                            score=oracle_score,
                            secondary_score=scores["loss"],
                            label_query_count=label_query_count,
                            train_ids=train_ids,
                            train_features=train_features,
                            noisy_labels=noisy_labels,
                            private_path=private_path,
                            private=private,
                            val_features=val_features,
                            val_labels=val_labels,
                            test_features=test_features,
                            test_ids=test_ids,
                            evaluator=evaluator,
                            initial_state=checkpoint["initial_state"],
                            training_config=config["training"],
                            seed=seed,
                            baseline_wga=float(baseline_metrics["wga"]),
                            baseline_average_accuracy=float(
                                baseline_metrics["average_accuracy"]
                            ),
                            weak_group=actual_test_weak_group,
                            total_budget_fraction=args.total_budget,
                            total_budget_count=total_budget_count,
                            group_share=group_share,
                            group_audit_count=group_audit_count,
                            attribute_metadata=oracle_metadata,
                            query_path=query_path,
                        )
                    )
                    completed.add(oracle_key)

                pd.DataFrame(rows).sort_values(
                    ["dataset", "noise_name", "seed", "group_share", "role"],
                    kind="stable",
                ).to_csv(result_path, index=False)
                joint_row = next(
                    row
                    for row in reversed(rows)
                    if row.get("dataset", "waterbirds") == dataset_name
                    and row["noise_name"] == noise_name
                    and int(row["seed"]) == seed
                    and float(row["group_share"]) == group_share
                    and row["role"] == "joint_sparse_attribute"
                )
                print(
                    f"{dataset_name}/{noise_name} seed={seed} share={group_share:.2f} "
                    f"audit={group_audit_count} labels={label_query_count} "
                    f"attrBA={sparse_attribute_ba:.3f} dWGA={joint_row['delta_wga']:+.4f}"
                )
                sys.stdout.flush()

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "noise_name", "seed", "group_share", "role"], kind="stable"
    )
    result.to_csv(result_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "joint_group_label_budget_pilot")
    )
    parser.add_argument("--noises", default="uniform20,minority_high_40")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--total-budget", type=float, default=0.02)
    parser.add_argument("--group-shares", default="0.50,0.75")
    parser.add_argument("--include-oracle-attribute", action="store_true")
    parsed = parser.parse_args()
    parsed.noises = _csv_values(parsed.noises, str)
    parsed.seeds = _csv_values(parsed.seeds, int)
    parsed.group_shares = _csv_values(parsed.group_shares, float)
    run_pilot(parsed)


if __name__ == "__main__":
    main()
