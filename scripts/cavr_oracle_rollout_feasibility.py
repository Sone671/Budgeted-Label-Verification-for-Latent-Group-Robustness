#!/usr/bin/env python
"""Stage-A upper-bound audit for certified audit-versus-repair rollouts.

This driver intentionally uses the clean outcomes of already executed Phase-10
query sets.  It is therefore an oracle-rollout feasibility check, not a legal
deployment algorithm.  Candidate selection sees only predictions on the
smallest positive-share audit panel; full validation groups and test WGA are
used only after the selection is frozen for offline evaluation.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame
from robust_verify.cavr import (
    allocation_regret,
    audited_panel_value,
    select_candidate,
    soft_group_value,
)
from robust_verify.config import load_config, output_layout
from robust_verify.data.access import PrivateEvaluator
from robust_verify.features import load_features
from robust_verify.joint_budget import fit_attribute_probability_model
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count
from scripts.joint_group_label_budget_pilot import (
    _cached_query_positions,
    _existing_noise_score_reference,
)


def _csv_values(raw: str, cast: Any) -> list[Any]:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated input cannot be empty")
    return values


def _single_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def _query_positions_from_ids(
    query_ids: np.ndarray,
    train_ids: np.ndarray,
) -> np.ndarray:
    mapping = {int(sample_id): position for position, sample_id in enumerate(train_ids)}
    try:
        positions = np.asarray([mapping[int(sample_id)] for sample_id in query_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"query id {error.args[0]} is absent from train features") from error
    if len(np.unique(positions)) != len(positions):
        raise ValueError("query artifact contains duplicate sample ids")
    return positions


def _candidate_rows(
    phase_results: pd.DataFrame,
    *,
    noise_name: str,
    seed: int,
    budget_fraction: float,
    cost_ratio: float,
    shares: list[float],
) -> dict[float, pd.Series]:
    base = phase_results[
        (phase_results["noise_name"].astype(str) == noise_name)
        & (phase_results["seed"].astype(int) == seed)
        & np.isclose(phase_results["budget_fraction"].astype(float), budget_fraction)
        & np.isclose(phase_results["group_to_label_cost_ratio"].astype(float), cost_ratio)
    ]
    rows: dict[float, pd.Series] = {}
    for share in shares:
        if np.isclose(share, 0.0):
            candidates = base[base["role"].astype(str) == "label_only_total_budget"]
            # Phase 10 duplicates the same strict control for each positive
            # target share.  Their outcome and label count must be identical.
            if candidates.empty:
                raise ValueError("strict label-only row is missing")
            if candidates["wga"].astype(float).nunique() != 1:
                raise ValueError("duplicated strict label-only rows disagree")
            rows[share] = candidates.iloc[0]
        else:
            candidates = base[
                (base["role"].astype(str) == "joint_sparse_attribute")
                & np.isclose(base["target_group_action_share"].astype(float), share)
            ]
            rows[share] = _single_row(candidates, f"joint share={share}")
    return rows


def _load_or_train_predictions(
    *,
    cache_path: Path,
    query_positions: np.ndarray,
    clean_labels: np.ndarray,
    noisy_labels: np.ndarray,
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: np.ndarray,
    test_features: torch.Tensor,
    test_ids: np.ndarray,
    evaluator: PrivateEvaluator,
    initial_state: dict[str, torch.Tensor],
    training_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, float]:
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["val_predictions"].astype(np.int64), float(cached["test_wga"])

    repaired = noisy_labels.copy()
    repaired[query_positions] = clean_labels[query_positions]
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
    val_predictions, _ = predict_from_state(
        retrained.best_state,
        val_features,
        int(train_features.shape[1]),
        int(max(noisy_labels.max(), val_labels.max()) + 1),
        training_config,
    )
    test_predictions, _ = predict_from_state(
        retrained.best_state,
        test_features,
        int(train_features.shape[1]),
        int(max(noisy_labels.max(), val_labels.max()) + 1),
        training_config,
    )
    test_wga = float(evaluator.evaluate(test_ids, test_predictions)["wga"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        val_predictions=val_predictions.astype(np.int64),
        test_wga=np.asarray(test_wga, dtype=np.float64),
        query_positions=query_positions.astype(np.int64),
    )
    return val_predictions.astype(np.int64), test_wga


def _run_dataset(
    *,
    source_config_path: Path,
    phase_result_path: Path,
    output_dir: Path,
    seeds: list[int],
    noises: list[str],
    budget_fraction: float,
    cost_ratio: float,
    shares: list[float],
    alpha: float,
) -> list[dict[str, Any]]:
    source_config = load_config(source_config_path)
    dataset = str(source_config.get("data", {}).get("dataset", "waterbirds"))
    run_root = Path(source_config["project"]["output_dir"])
    data_root = Path(source_config["project"].get("input_dir", run_root))
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    run_layout = output_layout(run_root)
    data_layout = output_layout(data_root)

    train_np, train_ids = load_features(data_layout["features"] / "train.npz")
    val_np, val_ids = load_features(data_layout["features"] / "val.npz")
    test_np, test_ids = load_features(data_layout["features"] / "test.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = torch.from_numpy(train_np).float().to(device)
    val_features = torch.from_numpy(val_np).float().to(device)
    test_features = torch.from_numpy(test_np).float().to(device)

    val_public = align_frame(pd.read_csv(data_layout["manifests"] / "val_public.csv"), val_ids)
    val_private = align_frame(pd.read_csv(data_layout["manifests"] / "val_private.csv"), val_ids)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    val_attributes = val_private["place"].to_numpy(dtype=np.int64)
    val_groups = val_private["group"].to_numpy(dtype=np.int64)
    expected_groups = sorted(np.unique(val_groups).astype(int).tolist())
    evaluator = PrivateEvaluator(data_layout["manifests"] / "test_private.csv")
    existing = pd.read_csv(run_layout["stage1"] / "results.csv")
    phase_results = pd.read_csv(phase_result_path)
    capacity = budget_to_count(budget_fraction, len(train_ids))
    output_rows: list[dict[str, Any]] = []

    for noise_name in noises:
        for seed in seeds:
            prefix = f"{noise_name}_seed{seed}"
            public = align_frame(
                pd.read_csv(data_layout["manifests"] / f"{prefix}_train_public.csv"),
                train_ids,
            )
            private = align_frame(
                pd.read_csv(data_layout["manifests"] / f"{prefix}_train_private.csv"),
                train_ids,
            )
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
            clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
            checkpoint = torch.load(
                run_layout["probes"] / f"{prefix}.pt",
                map_location="cpu",
                weights_only=False,
            )
            rows = _candidate_rows(
                phase_results,
                noise_name=noise_name,
                seed=seed,
                budget_fraction=budget_fraction,
                cost_ratio=cost_ratio,
                shares=shares,
            )
            positive_shares = [share for share in shares if share > 0.0]
            scout_share = min(positive_shares)
            scout_ids = np.load(Path(str(rows[scout_share]["audit_path"])))
            scout_positions = _query_positions_from_ids(scout_ids, val_ids)
            scout_labels = val_labels[scout_positions]
            scout_groups = val_groups[scout_positions]
            attribute_model = fit_attribute_probability_model(
                val_np[scout_positions],
                val_attributes[scout_positions],
                seed=seed,
            )
            val_attribute_probability = attribute_model.predict_probability(val_np)

            full_reference = _existing_noise_score_reference(
                existing,
                noise_name=noise_name,
                seed=seed,
                label_count=capacity,
            )
            label_only_positions = _cached_query_positions(
                run_layout,
                train_ids,
                noise_name=noise_name,
                seed=seed,
                budget_fraction=float(full_reference["budget_fraction"]),
            )

            for share in shares:
                row = rows[share]
                if np.isclose(share, 0.0):
                    query_positions = label_only_positions
                else:
                    query_ids = np.load(Path(str(row["query_path"])))
                    query_positions = _query_positions_from_ids(query_ids, train_ids)
                expected_label_count = int(row["label_query_count"])
                if len(query_positions) != expected_label_count:
                    raise ValueError(
                        f"{dataset}/{prefix}/share={share}: query count mismatch"
                    )

                cache_path = (
                    output_dir
                    / "prediction_cache"
                    / dataset
                    / prefix
                    / (
                        f"b{budget_fraction:.3f}_r{cost_ratio:.3f}_"
                        f"s{share:.3f}.npz"
                    )
                )
                val_predictions, test_wga = _load_or_train_predictions(
                    cache_path=cache_path,
                    query_positions=query_positions,
                    clean_labels=clean_labels,
                    noisy_labels=noisy_labels,
                    train_features=train_features,
                    val_features=val_features,
                    val_labels=val_labels,
                    test_features=test_features,
                    test_ids=test_ids,
                    evaluator=evaluator,
                    initial_state=checkpoint["initial_state"],
                    training_config=source_config["training"],
                    seed=seed,
                )
                panel = audited_panel_value(
                    val_predictions[scout_positions],
                    scout_labels,
                    scout_groups,
                    expected_groups=expected_groups,
                    alpha=alpha,
                    simultaneous_comparisons=len(shares),
                )
                full = audited_panel_value(
                    val_predictions,
                    val_labels,
                    val_groups,
                    expected_groups=expected_groups,
                    alpha=alpha,
                    simultaneous_comparisons=len(shares),
                )
                soft = soft_group_value(
                    val_predictions,
                    val_labels,
                    val_attribute_probability,
                    num_classes=int(val_labels.max()) + 1,
                )
                saved_wga = float(row["wga"])
                output_rows.append(
                    {
                        "dataset": dataset,
                        "noise_name": noise_name,
                        "seed": seed,
                        "budget_fraction": budget_fraction,
                        "cost_ratio": cost_ratio,
                        "target_group_action_share": share,
                        "group_audit_count": int(row["group_audit_count"]),
                        "label_query_count": expected_label_count,
                        "scout_share": scout_share,
                        "scout_audit_count": len(scout_positions),
                        "scout_min_group_count": int(panel.group_count.min()),
                        "scout_group_counts": json.dumps(panel.group_count.tolist()),
                        "panel_wga": panel.wga,
                        "panel_wilson_lcb": panel.lower_bound,
                        "soft_group_wga": soft.wga,
                        "soft_group_masses": json.dumps(soft.group_mass.tolist()),
                        "attribute_model_fitted": bool(attribute_model.fitted),
                        "full_validation_wga": full.wga,
                        "retrained_test_wga": test_wga,
                        "saved_phase10_test_wga": saved_wga,
                        "test_wga_integrity_abs_diff": abs(test_wga - saved_wga),
                    }
                )
                pd.DataFrame(output_rows).to_csv(output_dir / "candidate_values.csv", index=False)
                print(
                    f"{dataset}/{prefix} share={share:.2f} "
                    f"panel={panel.wga:.4f} lcb={panel.lower_bound:.4f} "
                    f"soft={soft.wga:.4f} test={test_wga:.4f}"
                )
                sys.stdout.flush()
    return output_rows


def _selector_rows(candidate_frame: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    keys = ["dataset", "noise_name", "seed", "budget_fraction", "cost_ratio"]
    for condition, group in candidate_frame.groupby(keys, sort=True):
        group = group.sort_values("target_group_action_share", kind="stable").reset_index(drop=True)
        shares = group["target_group_action_share"].to_numpy(dtype=float)
        audits = group["group_audit_count"].to_numpy(dtype=int)
        test = group["retrained_test_wga"].to_numpy(dtype=float)
        selectors = {
            "panel_point": select_candidate(group["panel_wga"], audits),
            "panel_wilson_lcb": select_candidate(group["panel_wilson_lcb"], audits),
            "soft_group_point": select_candidate(group["soft_group_wga"], audits),
            "full_validation_oracle": select_candidate(group["full_validation_wga"], audits),
        }
        label_index = int(np.flatnonzero(np.isclose(shares, 0.0))[0])
        for name, selected in selectors.items():
            output.append(
                {
                    **dict(zip(keys, condition, strict=True)),
                    "selector": name,
                    "selected_share": shares[selected],
                    "selected_test_wga": test[selected],
                    "best_test_wga": float(test.max()),
                    "label_only_test_wga": test[label_index],
                    "allocation_regret": allocation_regret(test, selected),
                    "strict_label_contrast": float(test[selected] - test[label_index]),
                    "within_0_25pp_of_oracle": bool(test.max() - test[selected] <= 0.0025 + 1e-12),
                    "scout_min_group_count": int(group["scout_min_group_count"].min()),
                }
            )
    return pd.DataFrame(output)


def _decision(selector_frame: pd.DataFrame, *, primary_selector: str) -> dict[str, Any]:
    primary = selector_frame[selector_frame["selector"] == primary_selector].copy()
    mean_regret = float(primary["allocation_regret"].mean())
    mean_contrast = float(primary["strict_label_contrast"].mean())
    severe_harm_count = int((primary["strict_label_contrast"] < -0.01 - 1e-12).sum())
    near_oracle_count = int(primary["within_0_25pp_of_oracle"].sum())
    gates = {
        "mean_regret_at_most_0_5pp": mean_regret <= 0.005 + 1e-12,
        "mean_label_contrast_at_least_0_5pp": mean_contrast >= 0.005 - 1e-12,
        "severe_harm_count_at_most_one": severe_harm_count <= 1,
        "near_oracle_count_at_least_eight": near_oracle_count >= 8,
    }
    return {
        "stage": "A_oracle_rollout_upper_bound",
        "primary_selector": primary_selector,
        "cell_count": len(primary),
        "mean_allocation_regret": mean_regret,
        "mean_allocation_regret_pp": 100.0 * mean_regret,
        "mean_strict_label_contrast": mean_contrast,
        "mean_strict_label_contrast_pp": 100.0 * mean_contrast,
        "severe_harm_count": severe_harm_count,
        "near_oracle_count": near_oracle_count,
        "all_scout_panels_cover_expected_groups": bool(
            (primary["scout_min_group_count"] > 0).all()
        ),
        "gates": gates,
        "decision": "advance_to_stage_b" if all(gates.values()) else "stop_cavr",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", action="append", required=True)
    parser.add_argument("--phase-results", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noises", default="uniform20,minority_high_40")
    parser.add_argument("--budget-fraction", type=float, default=0.02)
    parser.add_argument("--cost-ratio", type=float, default=0.1)
    parser.add_argument("--shares", default="0,0.25,0.5,0.75")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--primary-selector", default="soft_group_point")
    args = parser.parse_args()

    if len(args.source_config) != len(args.phase_results):
        raise ValueError("--source-config and --phase-results counts must match")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _csv_values(args.seeds, int)
    noises = _csv_values(args.noises, str)
    shares = _csv_values(args.shares, float)
    if not any(np.isclose(share, 0.0) for share in shares):
        raise ValueError("candidate shares must contain the strict label-only action")
    if not any(share > 0.0 for share in shares):
        raise ValueError("candidate shares must contain a positive audit action")

    rows: list[dict[str, Any]] = []
    for source, phase in zip(args.source_config, args.phase_results, strict=True):
        rows.extend(
            _run_dataset(
                source_config_path=Path(source).resolve(),
                phase_result_path=Path(phase).resolve(),
                output_dir=output_dir,
                seeds=seeds,
                noises=noises,
                budget_fraction=args.budget_fraction,
                cost_ratio=args.cost_ratio,
                shares=shares,
                alpha=args.alpha,
            )
        )

    candidates = pd.DataFrame(rows)
    candidates.to_csv(output_dir / "candidate_values.csv", index=False)
    selectors = _selector_rows(candidates)
    selectors.to_csv(output_dir / "selector_results.csv", index=False)
    decision = _decision(selectors, primary_selector=args.primary_selector)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
