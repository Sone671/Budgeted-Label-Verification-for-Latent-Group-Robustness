#!/usr/bin/env python
"""Run the frozen Phase-10 fixed cost--budget grid with resumable cells."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame
from robust_verify.config import load_config, output_layout
from robust_verify.cost_budget_map import ActionAllocation, allocate_action_mix
from robust_verify.data.access import PrivateEvaluator
from robust_verify.features import load_features
from robust_verify.joint_budget import (
    estimate_soft_group_error,
    fit_attribute_probability_model,
    joint_repair_scores,
    relative_weak_group_error_margin,
    select_class_balanced_audits,
)
from robust_verify.scoring import build_legal_scores
from robust_verify.training import predict_from_state
from robust_verify.utils import budget_to_count
from scripts.joint_group_label_budget_pilot import (
    _actual_group_accuracy,
    _cached_query_positions,
    _control_result,
    _existing_noise_score_reference,
    _retrain_result,
)


DEFAULT_PHASE_CONFIG = (
    ROOT
    / "joint_group_label_budget"
    / "configs"
    / "phase10_cost_budget_map_locked.yaml"
)

CELL_FIELDS = (
    "dataset",
    "noise_name",
    "seed",
    "budget_fraction",
    "group_to_label_cost_ratio",
    "target_group_action_share",
    "role",
)


def _csv_values(raw: str | None, default: list[Any], cast: Any) -> list[Any]:
    if raw is None:
        return [cast(value) for value in default]
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated override cannot be empty")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    if len(frame):
        existing_sort = [field for field in CELL_FIELDS if field in frame.columns]
        frame = frame.sort_values(existing_sort, kind="stable")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            # Windows readers and antivirus scanners can briefly retain a
            # handle after inspecting the previous checkpoint.
            time.sleep(0.1)


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int, str, str, str, str]:
    return (
        str(row["dataset"]),
        str(row["noise_name"]),
        int(row["seed"]),
        f"{float(row['budget_fraction']):.6f}",
        f"{float(row['group_to_label_cost_ratio']):.6f}",
        f"{float(row['target_group_action_share']):.6f}",
        str(row["role"]),
    )


def _decorate(
    row: dict[str, Any],
    *,
    dataset: str,
    budget_fraction: float,
    allocation: ActionAllocation,
    role: str,
) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if role == "joint_sparse_attribute":
        actual_group_count = allocation.group_audit_count
        actual_label_count = allocation.label_verification_count
        actual_cost = allocation.total_cost_used
    elif role == "label_only_total_budget":
        actual_group_count = 0
        actual_label_count = allocation.budget_capacity
        actual_cost = float(allocation.budget_capacity)
    elif role == "label_only_same_label_count":
        actual_group_count = 0
        actual_label_count = allocation.label_verification_count
        actual_cost = float(allocation.label_verification_count)
    else:
        raise ValueError(f"unsupported Phase-10 role: {role}")
    actual_action_count = actual_group_count + actual_label_count
    result.update(
        {
            "dataset": dataset,
            "role": role,
            "budget_fraction": budget_fraction,
            "budget_capacity": allocation.budget_capacity,
            "group_to_label_cost_ratio": allocation.group_to_label_cost_ratio,
            "target_group_action_share": allocation.target_group_action_share,
            "planned_group_audit_count": allocation.group_audit_count,
            "planned_label_verification_count": (
                allocation.label_verification_count
            ),
            "planned_realized_group_action_share": (
                allocation.realized_group_action_share
            ),
            "realized_group_action_share": (
                actual_group_count / actual_action_count
                if actual_action_count
                else 0.0
            ),
            "group_audit_count": actual_group_count,
            "label_query_count": actual_label_count,
            "total_cost_used": actual_cost,
            "audit_pool_saturated": allocation.audit_pool_saturated,
            "human_action_count": actual_action_count,
        }
    )
    return result


def _load_phase_grid(args: argparse.Namespace) -> dict[str, Any]:
    phase_path = Path(args.phase_config).resolve()
    with phase_path.open("r", encoding="utf-8") as handle:
        phase = yaml.safe_load(handle)
    if phase["project"]["status"] not in {
        "protocol_frozen_before_phase10_outcomes",
        "frozen_before_acs_wga",
    }:
        raise ValueError("cost-budget config is not marked frozen")
    grid = phase["grid"]
    return {
        "path": phase_path,
        "phase": phase,
        "budgets": _csv_values(
            args.budget_fractions, grid["total_budget_fractions"], float
        ),
        "ratios": _csv_values(
            args.cost_ratios, grid["group_to_label_cost_ratios"], float
        ),
        "shares": _csv_values(
            args.group_shares, grid["group_action_shares"], float
        ),
        "seeds": _csv_values(args.seeds, phase["scope"]["seeds"], int),
        "noises": _csv_values(args.noises, phase["scope"]["noises"], str),
    }


def _preflight(
    *,
    args: argparse.Namespace,
    source_config_path: Path,
    source_config: dict[str, Any],
    data_layout: dict[str, Path],
    run_layout: dict[str, Path],
    phase_grid: dict[str, Any],
    train_count: int,
    validation_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    missing: list[str] = []
    required = [
        data_layout["features"] / "train.npz",
        data_layout["features"] / "val.npz",
        data_layout["features"] / "test.npz",
        data_layout["manifests"] / "val_public.csv",
        data_layout["manifests"] / "val_private.csv",
        data_layout["manifests"] / "test_private.csv",
        run_layout["stage1"] / "results.csv",
    ]
    for noise in phase_grid["noises"]:
        for seed in phase_grid["seeds"]:
            prefix = f"{noise}_seed{seed}"
            required.extend(
                [
                    data_layout["manifests"] / f"{prefix}_train_public.csv",
                    data_layout["manifests"] / f"{prefix}_train_private.csv",
                    run_layout["probes"] / f"{prefix}.pt",
                    run_layout["probes"] / f"{prefix}.dynamics.npz",
                ]
            )
    missing.extend(str(path) for path in required if not path.exists())

    allocations = []
    for budget in phase_grid["budgets"]:
        capacity = budget_to_count(budget, train_count)
        for ratio in phase_grid["ratios"]:
            for share in phase_grid["shares"]:
                allocation = allocate_action_mix(
                    capacity,
                    share,
                    ratio,
                    max_group_audits=validation_count,
                )
                allocations.append(
                    {"budget_fraction": budget, **asdict(allocation)}
                )

    report = {
        "status": "pass" if not missing else "fail",
        "source_config": str(source_config_path),
        "source_config_sha256": _sha256(source_config_path),
        "phase_config": str(phase_grid["path"]),
        "phase_config_sha256": _sha256(phase_grid["path"]),
        "dataset": str(source_config.get("data", {}).get("dataset", "waterbirds")),
        "train_count": train_count,
        "validation_count": validation_count,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "selected_seeds": phase_grid["seeds"],
        "selected_noises": phase_grid["noises"],
        "selected_budget_fractions": phase_grid["budgets"],
        "selected_cost_ratios": phase_grid["ratios"],
        "selected_group_shares": phase_grid["shares"],
        "allocation_cell_count": len(allocations),
        "saturated_cell_count": sum(
            bool(row["audit_pool_saturated"]) for row in allocations
        ),
        "missing": missing,
        "allocations": allocations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    has_override = any(
        value is not None
        for value in (
            args.seeds,
            args.noises,
            args.budget_fractions,
            args.cost_ratios,
            args.group_shares,
        )
    )
    if has_override:
        selection = json.dumps(
            {
                "seeds": phase_grid["seeds"],
                "noises": phase_grid["noises"],
                "budgets": phase_grid["budgets"],
                "ratios": phase_grid["ratios"],
                "shares": phase_grid["shares"],
            },
            sort_keys=True,
        ).encode("utf-8")
        selection_id = hashlib.sha256(selection).hexdigest()[:12]
        preflight_path = (
            output_dir / "preflight_shards" / f"preflight_{selection_id}.json"
        )
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        preflight_path = output_dir / "preflight.json"
    report["preflight_path"] = str(preflight_path)
    preflight_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if missing:
        raise FileNotFoundError(f"Phase-10 preflight found {len(missing)} missing files")
    return report


def run(args: argparse.Namespace) -> pd.DataFrame:
    source_config_path = Path(args.source_config).resolve()
    source_config = load_config(source_config_path)
    run_root = Path(source_config["project"]["output_dir"])
    data_root = Path(source_config["project"].get("input_dir", run_root))
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    run_layout = output_layout(run_root)
    data_layout = output_layout(data_root)
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "results.csv"
    phase_grid = _load_phase_grid(args)

    train_np, train_ids = load_features(data_layout["features"] / "train.npz")
    val_np, val_ids = load_features(data_layout["features"] / "val.npz")
    test_np, test_ids = load_features(data_layout["features"] / "test.npz")
    preflight = _preflight(
        args=args,
        source_config_path=source_config_path,
        source_config=source_config,
        data_layout=data_layout,
        run_layout=run_layout,
        phase_grid=phase_grid,
        train_count=len(train_ids),
        validation_count=len(val_ids),
        output_dir=output_dir,
    )
    if args.preflight_only:
        print(json.dumps({key: value for key, value in preflight.items() if key != "allocations"}, indent=2))
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = torch.from_numpy(train_np).float().to(device)
    val_features = torch.from_numpy(val_np).float().to(device)
    test_features = torch.from_numpy(test_np).float().to(device)

    val_public = align_frame(
        pd.read_csv(data_layout["manifests"] / "val_public.csv"), val_ids
    )
    val_private = align_frame(
        pd.read_csv(data_layout["manifests"] / "val_private.csv"), val_ids
    )
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    val_attributes = val_private["place"].to_numpy(dtype=np.int64)
    val_groups = val_private["group"].to_numpy(dtype=np.int64)
    evaluator = PrivateEvaluator(data_layout["manifests"] / "test_private.csv")
    existing = pd.read_csv(run_layout["stage1"] / "results.csv")
    dataset = str(source_config.get("data", {}).get("dataset", "waterbirds"))

    rows: list[dict[str, Any]] = []
    if result_path.exists():
        rows = pd.read_csv(result_path).to_dict("records")
    completed = {_cell_key(row) for row in rows}

    for noise_name in phase_grid["noises"]:
        for seed in phase_grid["seeds"]:
            prefix = f"{noise_name}_seed{seed}"
            public_path = data_layout["manifests"] / f"{prefix}_train_public.csv"
            private_path = data_layout["manifests"] / f"{prefix}_train_private.csv"
            public = align_frame(pd.read_csv(public_path), train_ids)
            private = align_frame(pd.read_csv(private_path), train_ids)
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)

            checkpoint = torch.load(
                run_layout["probes"] / f"{prefix}.pt",
                map_location="cpu",
                weights_only=False,
            )
            dynamics = np.load(run_layout["probes"] / f"{prefix}.dynamics.npz")
            train_probabilities = dynamics["train_probabilities"].astype(np.float64)
            correctness_history = dynamics["correctness_history"]
            legal_scores = build_legal_scores(
                train_probabilities, noisy_labels, correctness_history
            )
            val_predictions, val_probabilities = predict_from_state(
                checkpoint["best_state"],
                val_features,
                int(train_np.shape[1]),
                int(train_probabilities.shape[1]),
                source_config["training"],
            )
            test_predictions, _ = predict_from_state(
                checkpoint["best_state"],
                test_features,
                int(train_np.shape[1]),
                int(train_probabilities.shape[1]),
                source_config["training"],
            )
            baseline_metrics = evaluator.evaluate(test_ids, test_predictions)
            val_group_accuracy = _actual_group_accuracy(
                val_predictions, val_labels, val_groups
            )
            actual_val_weak_group = min(
                val_group_accuracy, key=val_group_accuracy.get
            )
            test_group_accuracy = {
                int(key.split("_")[1]): float(value)
                for key, value in baseline_metrics.items()
                if key.startswith("group_") and key.endswith("_accuracy")
            }
            actual_test_weak_group = min(
                test_group_accuracy, key=test_group_accuracy.get
            )

            audit_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
            same_label_cache: dict[int, dict[str, Any]] = {}

            for budget_fraction in phase_grid["budgets"]:
                capacity = budget_to_count(budget_fraction, len(train_ids))
                full_reference = _existing_noise_score_reference(
                    existing,
                    noise_name=noise_name,
                    seed=seed,
                    label_count=capacity,
                )
                full_positions = _cached_query_positions(
                    run_layout,
                    train_ids,
                    noise_name=noise_name,
                    seed=seed,
                    budget_fraction=float(full_reference["budget_fraction"]),
                )
                for ratio in phase_grid["ratios"]:
                    for target_share in phase_grid["shares"]:
                        allocation = allocate_action_mix(
                            capacity,
                            target_share,
                            ratio,
                            max_group_audits=len(val_ids),
                        )
                        total_control_key = _cell_key(
                            {
                                "dataset": dataset,
                                "noise_name": noise_name,
                                "seed": seed,
                                "budget_fraction": budget_fraction,
                                "group_to_label_cost_ratio": ratio,
                                "target_group_action_share": target_share,
                                "role": "label_only_total_budget",
                            }
                        )
                        if total_control_key not in completed:
                            control = _control_result(
                                full_reference,
                                full_positions,
                                private,
                                role="label_only_total_budget",
                                total_budget_fraction=budget_fraction,
                                total_budget_count=capacity,
                                group_share=0.0,
                                group_audit_count=0,
                                weak_group=actual_test_weak_group,
                                attribute_metadata={"attribute_model": "none"},
                            )
                            rows.append(
                                _decorate(
                                    control,
                                    dataset=dataset,
                                    budget_fraction=budget_fraction,
                                    allocation=allocation,
                                    role="label_only_total_budget",
                                )
                            )
                            completed.add(total_control_key)
                            _safe_write_csv(rows, result_path)

                        if target_share == 0.0:
                            continue

                        label_count = allocation.label_verification_count
                        secondary_key = _cell_key(
                            {
                                "dataset": dataset,
                                "noise_name": noise_name,
                                "seed": seed,
                                "budget_fraction": budget_fraction,
                                "group_to_label_cost_ratio": ratio,
                                "target_group_action_share": target_share,
                                "role": "label_only_same_label_count",
                            }
                        )
                        if secondary_key not in completed and not args.skip_secondary:
                            if label_count not in same_label_cache:
                                query_path = (
                                    output_dir
                                    / "queries"
                                    / dataset
                                    / prefix
                                    / f"noise_score_count_{label_count}.npy"
                                )
                                same_label_cache[label_count] = _retrain_result(
                                    role="label_only_same_label_count",
                                    score=legal_scores["noise_score"],
                                    secondary_score=legal_scores["loss"],
                                    label_query_count=label_count,
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
                                    training_config=source_config["training"],
                                    seed=seed,
                                    baseline_wga=float(baseline_metrics["wga"]),
                                    baseline_average_accuracy=float(
                                        baseline_metrics["average_accuracy"]
                                    ),
                                    weak_group=actual_test_weak_group,
                                    total_budget_fraction=budget_fraction,
                                    total_budget_count=capacity,
                                    group_share=allocation.realized_group_action_share,
                                    group_audit_count=allocation.group_audit_count,
                                    attribute_metadata={"attribute_model": "none"},
                                    query_path=query_path,
                                )
                            rows.append(
                                _decorate(
                                    same_label_cache[label_count],
                                    dataset=dataset,
                                    budget_fraction=budget_fraction,
                                    allocation=allocation,
                                    role="label_only_same_label_count",
                                )
                            )
                            completed.add(secondary_key)
                            _safe_write_csv(rows, result_path)

                        joint_key = _cell_key(
                            {
                                "dataset": dataset,
                                "noise_name": noise_name,
                                "seed": seed,
                                "budget_fraction": budget_fraction,
                                "group_to_label_cost_ratio": ratio,
                                "target_group_action_share": target_share,
                                "role": "joint_sparse_attribute",
                            }
                        )
                        if joint_key in completed:
                            continue

                        group_count = allocation.group_audit_count
                        if group_count not in audit_cache:
                            audit_positions = select_class_balanced_audits(
                                val_labels, group_count, seed=100_000 + seed
                            )
                            audit_path = (
                                output_dir
                                / "audits"
                                / dataset
                                / prefix
                                / f"place_audit_count_{group_count}.npy"
                            )
                            audit_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(audit_path, val_ids[audit_positions])
                            audited_attributes = val_attributes[audit_positions]
                            attribute_model = fit_attribute_probability_model(
                                val_np[audit_positions],
                                audited_attributes,
                                seed=seed,
                            )
                            train_attribute_probability = (
                                attribute_model.predict_probability(train_np)
                            )
                            val_attribute_probability = (
                                attribute_model.predict_probability(val_np)
                            )
                            group_error, group_mass = estimate_soft_group_error(
                                val_probabilities,
                                val_labels,
                                val_attribute_probability,
                            )
                            joint_score, expected_group_risk = joint_repair_scores(
                                legal_scores["noise_score"],
                                train_probabilities,
                                train_attribute_probability,
                                group_error,
                            )
                            estimated_weak_group = int(np.argmax(group_error))
                            metadata: dict[str, Any] = {
                                "attribute_model": "sparse_audit_logistic",
                                "attribute_model_fitted": bool(attribute_model.fitted),
                                "audited_attribute_positive_count": int(
                                    audited_attributes.sum()
                                ),
                                "audited_attribute_positive_rate": float(
                                    audited_attributes.mean()
                                ),
                                "audit_path": str(audit_path),
                                "attribute_balanced_accuracy_offline": float(
                                    balanced_accuracy_score(
                                        val_attributes,
                                        (
                                            val_attribute_probability >= 0.5
                                        ).astype(np.int64),
                                    )
                                ),
                                "estimated_weak_group": estimated_weak_group,
                                "actual_validation_weak_group_offline": (
                                    actual_val_weak_group
                                ),
                                "actual_test_weak_group_offline": (
                                    actual_test_weak_group
                                ),
                                "weak_group_identified_offline": int(
                                    estimated_weak_group == actual_val_weak_group
                                ),
                                "mean_expected_group_risk": float(
                                    expected_group_risk.mean()
                                ),
                                "relative_weak_group_error_margin": (
                                    relative_weak_group_error_margin(group_error)
                                ),
                            }
                            for group, value in enumerate(group_error):
                                metadata[f"estimated_group_{group}_error"] = float(value)
                                metadata[f"estimated_group_{group}_mass"] = float(
                                    group_mass[group]
                                )
                            audit_cache[group_count] = (joint_score, metadata)

                        joint_score, metadata = audit_cache[group_count]
                        query_path = (
                            output_dir
                            / "queries"
                            / dataset
                            / prefix
                            / (
                                f"joint_b{budget_fraction:.3f}_r{ratio:.3f}_"
                                f"s{target_share:.3f}_g{group_count}_y{label_count}.npy"
                            )
                        )
                        joint = _retrain_result(
                            role="joint_sparse_attribute",
                            score=joint_score,
                            secondary_score=legal_scores["loss"],
                            label_query_count=label_count,
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
                            training_config=source_config["training"],
                            seed=seed,
                            baseline_wga=float(baseline_metrics["wga"]),
                            baseline_average_accuracy=float(
                                baseline_metrics["average_accuracy"]
                            ),
                            weak_group=actual_test_weak_group,
                            total_budget_fraction=budget_fraction,
                            total_budget_count=capacity,
                            group_share=allocation.realized_group_action_share,
                            group_audit_count=group_count,
                            attribute_metadata=metadata,
                            query_path=query_path,
                        )
                        rows.append(
                            _decorate(
                                joint,
                                dataset=dataset,
                                budget_fraction=budget_fraction,
                                allocation=allocation,
                                role="joint_sparse_attribute",
                            )
                        )
                        completed.add(joint_key)
                        _safe_write_csv(rows, result_path)
                        print(
                            f"{dataset}/{prefix} b={budget_fraction:.3f} "
                            f"rho={ratio:.2f} share={target_share:.2f} "
                            f"g={group_count} y={label_count} "
                            f"dWGA={joint['delta_wga']:+.4f}"
                        )
                        sys.stdout.flush()

    result = pd.DataFrame(rows)
    _safe_write_csv(rows, result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--phase-config", default=str(DEFAULT_PHASE_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--noises")
    parser.add_argument("--budget-fractions")
    parser.add_argument("--cost-ratios")
    parser.add_argument("--group-shares")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Defer same-label-count controls; primary strict-cost cells are unchanged.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
