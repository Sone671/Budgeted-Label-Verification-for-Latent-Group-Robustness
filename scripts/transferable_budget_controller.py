#!/usr/bin/env python
"""Old-seed development for cross-task marginal-utility budget control.

The command has three explicit phases:

1. ``extract`` retrains nested-prefix heads and writes legal trajectory features
   without loading test features or the private test manifest;
2. ``label`` opens old development test outcomes only after the legal artifact
   is frozen and constructs meta-labels;
3. ``evaluate`` performs leave-one-dataset-out selective control.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from robust_verify.config import load_config
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.features import load_features
from robust_verify.risk_stopping import (
    assert_nested_query_prefixes,
    sample_id_hash,
    stratified_validation_partition,
)
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.transfer_budget import (
    assert_legal_feature_names,
    balanced_accuracy_present_classes,
    build_legal_transition_features,
    feedback_summary,
    fit_predict_seed_block_bootstrap,
    selective_decisions,
)
from robust_verify.utils import budget_to_count, resolve_device


DEFAULT_CONFIGS = (
    ROOT / "configs" / "ablation_waterbirds_10seeds.yaml",
    ROOT / "configs" / "ablation_celeba_10seeds.yaml",
    ROOT / "configs" / "ablation_civilcomments_10seeds.yaml",
)
DEFAULT_OUTPUT = ROOT / "outputs" / "transferable_budget_control_pilot"
BOOKKEEPING_COLUMNS = (
    "dataset",
    "source_output",
    "noise_name",
    "seed",
    "current_budget_fraction_key",
    "next_budget_fraction_key",
    "previous_state_path",
    "current_state_path",
    "next_state_path",
)


def _parse_csv(raw: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _dataset_name(config: dict[str, Any], source_root: Path) -> str:
    explicit = str(config.get("data", {}).get("dataset", "")).strip().lower()
    if explicit:
        return explicit
    lowered = source_root.name.lower()
    for name in ("waterbirds", "celeba", "civilcomments"):
        if name in lowered:
            return name
    raise ValueError(f"cannot infer dataset name from {source_root}")


def _query_path(
    source_root: Path, noise: str, seed: int, method: str, budget: float
) -> Path:
    return (
        source_root
        / "stage1"
        / "queries"
        / f"{noise}_seed{seed}"
        / f"{method}_budget_{budget:.4f}.npy"
    )


def _state_path(
    output_root: Path,
    dataset: str,
    noise: str,
    seed: int,
    budget: float,
) -> Path:
    return output_root / "states" / dataset / f"{noise}_seed{seed}" / f"budget_{budget:.4f}.pt"


def _load_probe(source_root: Path, noise: str, seed: int) -> dict[str, Any]:
    path = source_root / "stage1" / "probes" / f"{noise}_seed{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def _save_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"best_state": state}, path)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["best_state"]


def _train_or_load_state(
    *,
    path: Path,
    force: bool,
    train_features: np.ndarray | torch.Tensor,
    noisy_labels: np.ndarray,
    repaired_positions: np.ndarray,
    repaired_values: np.ndarray,
    early_features: np.ndarray | torch.Tensor,
    early_labels: np.ndarray,
    training_config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if path.exists() and not force:
        return _load_state(path)
    labels = noisy_labels.copy()
    labels[repaired_positions] = repaired_values
    result = train_linear_head(
        train_features=train_features,
        train_labels=labels,
        val_features=early_features,
        val_labels=early_labels,
        config=training_config,
        seed=seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
    )
    _save_state(result.best_state, path)
    return result.best_state


def _probabilities(
    state: dict[str, torch.Tensor],
    features: np.ndarray | torch.Tensor,
    *,
    training_config: dict[str, Any],
) -> np.ndarray:
    weight = state["classifier.weight"]
    return predict_from_state(
        state,
        features,
        int(weight.shape[1]),
        int(weight.shape[0]),
        training_config,
    )[1]


def _device_copy(values: np.ndarray, device: torch.device) -> np.ndarray | torch.Tensor:
    if device.type != "cuda":
        return values
    return torch.from_numpy(values).float().to(device)


def _config_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.configs:
        return tuple(_resolve(value) for value in _parse_csv(args.configs, str))
    return DEFAULT_CONFIGS


def extract(args: argparse.Namespace) -> Path:
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    legal_path = output_root / "legal_trajectories.csv"
    budgets = tuple(sorted(set(_parse_csv(args.budgets, float))))
    seeds = _parse_csv(args.seeds, int)
    noises = _parse_csv(args.noise, str)
    fractions = tuple(_parse_csv(args.validation_fractions, float))
    if len(budgets) < 2 or any(value <= 0.0 or value > 1.0 for value in budgets):
        raise ValueError("at least two positive budgets in (0, 1] are required")
    if len(fractions) != 3:
        raise ValueError("validation-fractions must contain three entries")

    design: dict[str, Any] = {
        "protocol": "transferable_marginal_utility_old_seed_v1",
        "evidence_role": "development_only_previously_opened_seed_block",
        "method": args.method,
        "budgets": list(budgets),
        "mandatory_scout_budget": budgets[0],
        "seeds": list(seeds),
        "noise_settings": list(noises),
        "validation_fractions": list(fractions),
        "split_seed": int(args.split_seed),
        "cost_wga_pp_per_budget_percent": float(args.cost),
        "bootstrap_replicates": int(args.bootstrap_reps),
        "bootstrap_quantiles": [0.10, 0.90],
        "continue_threshold": 0.60,
        "stop_threshold": 0.40,
        "config_sha256": {},
        "validation_split_hashes": {},
    }
    rows: list[dict[str, Any]] = []
    feature_names: tuple[str, ...] | None = None

    for config_path in _config_paths(args):
        config = load_config(config_path)
        source_root = _resolve(config["project"]["output_dir"])
        dataset = _dataset_name(config, source_root)
        design["config_sha256"][dataset] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        }
        print(f"[extract] dataset={dataset} source={source_root}", flush=True)

        train_features_np, train_ids = load_features(source_root / "features" / "train.npz")
        val_features_np, val_ids = load_features(source_root / "features" / "val.npz")
        val_public = align_frame(
            pd.read_csv(source_root / "manifests" / "val_public.csv"), val_ids
        )
        val_labels = val_public["label"].to_numpy(dtype=np.int64)
        partition = stratified_validation_partition(
            val_ids,
            val_labels,
            fractions=fractions,
            seed=args.split_seed,
        )
        design["validation_split_hashes"][dataset] = {
            "early_stop": sample_id_hash(val_ids, partition.early_stop),
            "calibration": sample_id_hash(val_ids, partition.calibration),
            "audit": sample_id_hash(val_ids, partition.audit),
            "counts": {
                "early_stop": int(len(partition.early_stop)),
                "calibration": int(len(partition.calibration)),
                "audit": int(len(partition.audit)),
            },
        }
        training_config = dict(config["training"])
        device = resolve_device(training_config.get("device", "auto"))
        train_features = _device_copy(train_features_np, device)
        early_features = _device_copy(val_features_np[partition.early_stop], device)
        calibration_features = _device_copy(val_features_np[partition.calibration], device)
        audit_features = _device_copy(val_features_np[partition.audit], device)
        del train_features_np, val_features_np

        train_lookup = {int(sample_id): index for index, sample_id in enumerate(train_ids)}
        for noise in noises:
            for seed in seeds:
                prefix = f"{noise}_seed{seed}"
                print(f"[trajectory] {dataset}/{prefix}", flush=True)
                public = align_frame(
                    pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
                    train_ids,
                )
                noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
                probe = _load_probe(source_root, noise, seed)
                oracle = VerificationOracle(
                    source_root / "manifests" / f"{prefix}_train_private.csv"
                )
                query_paths = [
                    _query_path(source_root, noise, seed, args.method, budget)
                    for budget in budgets
                ]
                query_sets = [np.load(path, allow_pickle=False) for path in query_paths]
                assert_nested_query_prefixes(query_sets)
                for budget, query_ids in zip(budgets, query_sets):
                    expected = budget_to_count(budget, len(train_ids))
                    if len(query_ids) != expected:
                        raise ValueError(
                            f"{dataset}/{prefix}/{budget:g}: expected {expected}, got {len(query_ids)}"
                        )

                states: dict[float, dict[str, torch.Tensor]] = {}
                probabilities: dict[float, tuple[np.ndarray, np.ndarray]] = {}
                baseline_path = _state_path(output_root, dataset, noise, seed, 0.0)
                states[0.0] = _train_or_load_state(
                    path=baseline_path,
                    force=args.force_states,
                    train_features=train_features,
                    noisy_labels=noisy_labels,
                    repaired_positions=np.asarray([], dtype=np.int64),
                    repaired_values=np.asarray([], dtype=np.int64),
                    early_features=early_features,
                    early_labels=val_labels[partition.early_stop],
                    training_config=training_config,
                    seed=seed,
                    initial_state=probe["initial_state"],
                )
                probabilities[0.0] = (
                    _probabilities(
                        states[0.0], calibration_features, training_config=training_config
                    ),
                    _probabilities(states[0.0], audit_features, training_config=training_config),
                )

                returned_by_budget: dict[float, np.ndarray] = {}
                for budget, query_ids in zip(budgets, query_sets):
                    positions = np.asarray(
                        [train_lookup[int(sample_id)] for sample_id in query_ids],
                        dtype=np.int64,
                    )
                    returned = oracle.verify(query_ids)
                    returned_by_budget[budget] = returned
                    path = _state_path(output_root, dataset, noise, seed, budget)
                    states[budget] = _train_or_load_state(
                        path=path,
                        force=args.force_states,
                        train_features=train_features,
                        noisy_labels=noisy_labels,
                        repaired_positions=positions,
                        repaired_values=returned,
                        early_features=early_features,
                        early_labels=val_labels[partition.early_stop],
                        training_config=training_config,
                        seed=seed,
                        initial_state=probe["initial_state"],
                    )
                    probabilities[budget] = (
                        _probabilities(
                            states[budget], calibration_features, training_config=training_config
                        ),
                        _probabilities(
                            states[budget], audit_features, training_config=training_config
                        ),
                    )

                previous_budget = 0.0
                previous_count = 0
                for current_budget, next_budget, query_ids in zip(
                    budgets[:-1], budgets[1:], query_sets[:-1]
                ):
                    positions = np.asarray(
                        [train_lookup[int(sample_id)] for sample_id in query_ids],
                        dtype=np.int64,
                    )
                    feedback = feedback_summary(
                        noisy_labels[positions],
                        returned_by_budget[current_budget],
                        previous_count=previous_count,
                    )
                    cal_current, audit_current = probabilities[current_budget]
                    cal_previous, audit_previous = probabilities[previous_budget]
                    features = build_legal_transition_features(
                        current_budget=current_budget,
                        next_budget=next_budget,
                        calibration_current=cal_current,
                        calibration_previous=cal_previous,
                        calibration_labels=val_labels[partition.calibration],
                        audit_current=audit_current,
                        audit_previous=audit_previous,
                        audit_labels=val_labels[partition.audit],
                        feedback=feedback,
                    )
                    names = tuple(features)
                    if feature_names is None:
                        feature_names = names
                        assert_legal_feature_names(feature_names)
                    elif names != feature_names:
                        raise RuntimeError("legal feature order changed across trajectory rows")
                    rows.append(
                        {
                            "dataset": dataset,
                            "source_output": source_root.name,
                            "noise_name": noise,
                            "seed": int(seed),
                            "current_budget_fraction_key": float(current_budget),
                            "next_budget_fraction_key": float(next_budget),
                            "previous_state_path": str(
                                _state_path(output_root, dataset, noise, seed, previous_budget)
                            ),
                            "current_state_path": str(
                                _state_path(output_root, dataset, noise, seed, current_budget)
                            ),
                            "next_state_path": str(
                                _state_path(output_root, dataset, noise, seed, next_budget)
                            ),
                            **features,
                        }
                    )
                    previous_budget = current_budget
                    previous_count = len(query_ids)

        del train_features, early_features, calibration_features, audit_features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if feature_names is None:
        raise RuntimeError("no trajectory rows were extracted")
    design["feature_names"] = list(feature_names)
    design["legal_row_count"] = len(rows)
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "noise_name", "seed", "current_budget_fraction_key"],
        kind="stable",
    )
    _atomic_csv(frame, legal_path)
    design["legal_trajectory_sha256"] = _sha256(legal_path)
    _atomic_json(design, output_root / "design.json")
    print(f"[frozen-legal] rows={len(frame)} path={legal_path}", flush=True)
    return legal_path


def label(args: argparse.Namespace) -> Path:
    output_root = _resolve(args.output_root)
    legal_path = output_root / "legal_trajectories.csv"
    design_path = output_root / "design.json"
    if not legal_path.exists() or not design_path.exists():
        raise FileNotFoundError("extract must freeze legal trajectories before label")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if _sha256(legal_path) != design["legal_trajectory_sha256"]:
        raise RuntimeError("legal trajectory hash does not match frozen design")
    legal = pd.read_csv(legal_path)
    outcomes: list[dict[str, Any]] = []

    for config_path in _config_paths(args):
        config = load_config(config_path)
        source_root = _resolve(config["project"]["output_dir"])
        dataset = _dataset_name(config, source_root)
        subset = legal.loc[legal["dataset"] == dataset]
        if not len(subset):
            continue
        print(f"[label] opening old development test artifacts for {dataset}", flush=True)
        test_features, test_ids = load_features(source_root / "features" / "test.npz")
        evaluator = PrivateEvaluator(source_root / "manifests" / "test_private.csv")
        training_config = dict(config["training"])
        state_records: dict[tuple[str, int, float], str] = {}
        for row in subset.itertuples(index=False):
            if np.isclose(
                float(row.current_budget_fraction_key),
                float(design["mandatory_scout_budget"]),
            ):
                state_records[(row.noise_name, int(row.seed), 0.0)] = row.previous_state_path
            state_records[
                (row.noise_name, int(row.seed), float(row.current_budget_fraction_key))
            ] = row.current_state_path
            state_records[
                (row.noise_name, int(row.seed), float(row.next_budget_fraction_key))
            ] = row.next_state_path
        for (noise, seed, budget), raw_path in state_records.items():
            expected_path = _state_path(output_root, dataset, noise, seed, budget)
            if Path(raw_path).resolve() != expected_path.resolve():
                raise RuntimeError(
                    f"state/budget path mismatch for {dataset}/{noise}/seed{seed}/{budget:g}: "
                    f"{raw_path}"
                )
        for (noise, seed, budget), raw_path in sorted(state_records.items()):
            path = Path(raw_path)
            state = _load_state(path)
            weight = state["classifier.weight"]
            predictions = predict_from_state(
                state,
                test_features,
                int(weight.shape[1]),
                int(weight.shape[0]),
                training_config,
            )[0]
            metrics = evaluator.evaluate(test_ids, predictions)
            outcomes.append(
                {
                    "dataset": dataset,
                    "noise_name": noise,
                    "seed": seed,
                    "budget_fraction": budget,
                    "average_accuracy": metrics["average_accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "wga": metrics["wga"],
                    "state_path": str(path),
                    "state_sha256": _sha256(path),
                }
            )

    outcome_frame = pd.DataFrame(outcomes).sort_values(
        ["dataset", "noise_name", "seed", "budget_fraction"], kind="stable"
    )
    outcome_path = output_root / "private_development_outcomes.csv"
    _atomic_csv(outcome_frame, outcome_path)
    lookup = {
        (row.dataset, row.noise_name, int(row.seed), float(row.budget_fraction)): float(row.wga)
        for row in outcome_frame.itertuples(index=False)
    }
    meta = legal.copy()
    current_wga: list[float] = []
    next_wga: list[float] = []
    baseline_wga: list[float] = []
    fixed_wga: list[float] = []
    best_small_wga: list[float] = []
    for row in meta.itertuples(index=False):
        key = (row.dataset, row.noise_name, int(row.seed))
        current_wga.append(lookup[(*key, float(row.current_budget_fraction_key))])
        next_wga.append(lookup[(*key, float(row.next_budget_fraction_key))])
        baseline_wga.append(lookup[(*key, 0.0)])
        fixed_wga.append(lookup[(*key, max(float(value) for value in design["budgets"]))])
        best_small_wga.append(
            max(lookup[(*key, budget)] for budget in (0.005, 0.01, 0.02))
        )
    meta["target_current_wga"] = current_wga
    meta["target_next_wga"] = next_wga
    meta["target_baseline_wga"] = baseline_wga
    meta["target_fixed_cap_wga"] = fixed_wga
    meta["target_best_small_wga"] = best_small_wga
    meta["target_delta_wga_pp"] = 100.0 * (
        meta["target_next_wga"] - meta["target_current_wga"]
    )
    incremental_percent = 100.0 * (
        meta["next_budget_fraction_key"] - meta["current_budget_fraction_key"]
    )
    meta["target_net_utility_pp"] = (
        meta["target_delta_wga_pp"]
        - float(design["cost_wga_pp_per_budget_percent"]) * incremental_percent
    )
    meta["target_continue"] = (meta["target_net_utility_pp"] > 0.0).astype(int)
    meta_path = output_root / "development_meta_dataset.csv"
    _atomic_csv(meta, meta_path)
    private_design = {
        "private_outcome_sha256": _sha256(outcome_path),
        "meta_dataset_sha256": _sha256(meta_path),
        "private_outcomes_opened_after_legal_freeze": True,
        "meta_rows": int(len(meta)),
    }
    _atomic_json(private_design, output_root / "private_labeling_record.json")
    print(f"[labeled] rows={len(meta)} positives={int(meta['target_continue'].sum())}", flush=True)
    return meta_path


def _policy_metrics(dataset_frame: pd.DataFrame) -> tuple[dict[str, float], list[dict[str, Any]]]:
    condition_rows: list[dict[str, Any]] = []
    for (noise, seed), condition in dataset_frame.groupby(["noise_name", "seed"], sort=True):
        condition = condition.sort_values("current_budget_fraction_key", kind="stable")
        first = condition.iloc[0]
        selected_budget = float(first["current_budget_fraction_key"])
        selected_wga = float(first["target_current_wga"])
        reached = 0
        trace: list[str] = []
        for row in condition.itertuples(index=False):
            if not np.isclose(float(row.current_budget_fraction_key), selected_budget):
                break
            reached += 1
            action = str(row.controller_action)
            trace.append(f"{row.current_budget_fraction_key:g}:{action}")
            if action != "continue":
                break
            selected_budget = float(row.next_budget_fraction_key)
            selected_wga = float(row.target_next_wga)
        baseline = float(first["target_baseline_wga"])
        fixed = float(first["target_fixed_cap_wga"])
        best_small = float(first["target_best_small_wga"])
        condition_rows.append(
            {
                "dataset": first["dataset"],
                "noise_name": noise,
                "seed": int(seed),
                "selected_budget_fraction": selected_budget,
                "selected_wga": selected_wga,
                "baseline_wga": baseline,
                "fixed_cap_wga": fixed,
                "best_small_wga": best_small,
                "delta_wga_pp": 100.0 * (selected_wga - baseline),
                "delta_vs_fixed_cap_pp": 100.0 * (selected_wga - fixed),
                "best_small_regret_pp": 100.0 * max(0.0, best_small - selected_wga),
                "negative_utility": bool(selected_wga < baseline),
                "fixed_cap_negative_utility": bool(fixed < baseline),
                "advanced_beyond_scout": bool(selected_budget > float(first["current_budget_fraction_key"])),
                "reached_decisions": reached,
                "trace": "|".join(trace),
            }
        )
    conditions = pd.DataFrame(condition_rows)
    fixed_negative = float(conditions["fixed_cap_negative_utility"].mean())
    selected_negative = float(conditions["negative_utility"].mean())
    metrics = {
        "condition_count": int(len(conditions)),
        "mean_selected_budget_fraction": float(conditions["selected_budget_fraction"].mean()),
        "mean_delta_wga_pp": float(conditions["delta_wga_pp"].mean()),
        "mean_delta_vs_fixed_cap_pp": float(conditions["delta_vs_fixed_cap_pp"].mean()),
        "selected_negative_rate": selected_negative,
        "fixed_cap_negative_rate": fixed_negative,
        "negative_rate_reduction": fixed_negative - selected_negative,
        "mean_best_small_regret_pp": float(conditions["best_small_regret_pp"].mean()),
        "advance_condition_rate": float(conditions["advanced_beyond_scout"].mean()),
    }
    return metrics, condition_rows


def evaluate(args: argparse.Namespace) -> Path:
    output_root = _resolve(args.output_root)
    design = json.loads((output_root / "design.json").read_text(encoding="utf-8"))
    meta_path = output_root / "development_meta_dataset.csv"
    if not meta_path.exists():
        raise FileNotFoundError("label must create development_meta_dataset.csv first")
    meta = pd.read_csv(meta_path)
    feature_names = tuple(str(value) for value in design["feature_names"])
    assert_legal_feature_names(feature_names)
    missing = sorted(set(feature_names).difference(meta.columns))
    if missing:
        raise ValueError(f"meta dataset is missing legal features: {missing}")

    decision_frames: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "protocol": design["protocol"],
        "outer_split": "leave_one_dataset_out",
        "feature_names": list(feature_names),
        "held_out": {},
    }
    all_condition_rows: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(sorted(meta["dataset"].unique())):
        train = meta.loc[meta["dataset"] != held_out].copy()
        test = meta.loc[meta["dataset"] == held_out].copy()
        train_groups = (
            train["dataset"].astype(str)
            + ":"
            + train["noise_name"].astype(str)
            + ":"
            + train["seed"].astype(int).astype(str)
        )
        probability_samples = fit_predict_seed_block_bootstrap(
            train.loc[:, feature_names].to_numpy(dtype=np.float64),
            train["target_continue"].to_numpy(dtype=np.int64),
            train_groups.to_numpy(),
            test.loc[:, feature_names].to_numpy(dtype=np.float64),
            replicates=int(design["bootstrap_replicates"]),
            seed=int(design["split_seed"]) + 10007 * fold_index,
        )
        decisions = selective_decisions(
            probability_samples,
            lower_quantile=float(design["bootstrap_quantiles"][0]),
            upper_quantile=float(design["bootstrap_quantiles"][1]),
            continue_threshold=float(design["continue_threshold"]),
            stop_threshold=float(design["stop_threshold"]),
        )
        test["controller_action"] = [decision.action for decision in decisions]
        test["controller_probability_median"] = [
            decision.median_probability for decision in decisions
        ]
        test["controller_probability_lower"] = [
            decision.lower_probability for decision in decisions
        ]
        test["controller_probability_upper"] = [
            decision.upper_probability for decision in decisions
        ]
        decision_frames.append(test)

        truth = test["target_continue"].to_numpy(dtype=np.int64)
        all_action_prediction = np.asarray(
            [1 if decision.action == "continue" else 0 for decision in decisions],
            dtype=np.int64,
        )
        covered = np.asarray([decision.action != "abstain" for decision in decisions])
        continue_mask = np.asarray([decision.action == "continue" for decision in decisions])
        static = {
            "row_count": int(len(test)),
            "positive_rate": float(truth.mean()),
            "balanced_accuracy": balanced_accuracy_present_classes(truth, all_action_prediction),
            "selective_coverage": float(covered.mean()),
            "selective_accuracy": (
                float((truth[covered] == all_action_prediction[covered]).mean())
                if covered.any()
                else 0.0
            ),
            "continue_action_rate": float(continue_mask.mean()),
            "erroneous_continue_rate": (
                float((truth[continue_mask] == 0).mean()) if continue_mask.any() else 1.0
            ),
            "abstention_rate": float((~covered).mean()),
        }
        policy, condition_rows = _policy_metrics(test)
        all_condition_rows.extend(condition_rows)
        if policy["fixed_cap_negative_rate"] >= 0.30:
            downside_gate = policy["negative_rate_reduction"] >= 0.30
            downside_gate_type = "reduce_high_fixed_cap_downside"
        else:
            downside_gate = (
                policy["selected_negative_rate"]
                <= policy["fixed_cap_negative_rate"] + 0.10
            )
            downside_gate_type = "do_not_create_material_downside"
        gates = {
            "balanced_accuracy_at_least_0_65": static["balanced_accuracy"] >= 0.65,
            "selective_coverage_at_least_0_30": static["selective_coverage"] >= 0.30,
            "erroneous_continue_rate_at_most_0_10": static["erroneous_continue_rate"] <= 0.10,
            "advance_condition_rate_at_least_0_20": policy["advance_condition_rate"] >= 0.20,
            "best_small_regret_at_most_1pp": policy["mean_best_small_regret_pp"] <= 1.0,
            downside_gate_type: bool(downside_gate),
        }
        summary["held_out"][held_out] = {
            "training_datasets": sorted(train["dataset"].unique().tolist()),
            "static_decisions": static,
            "sequential_policy": policy,
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
        print(
            f"[fold] held_out={held_out} passed={all(gates.values())} "
            f"bacc={static['balanced_accuracy']:.3f} coverage={static['selective_coverage']:.3f} "
            f"bad_continue={static['erroneous_continue_rate']:.3f}",
            flush=True,
        )

    pass_count = sum(bool(value["passed"]) for value in summary["held_out"].values())
    seed_counts = meta.groupby("dataset")["seed"].nunique().to_dict()
    pilot = min(seed_counts.values()) < 10
    waterbirds = summary["held_out"].get("waterbirds", {})
    waterbirds_policy = waterbirds.get("sequential_policy", {})
    waterbirds_safety = (
        float(waterbirds_policy.get("negative_rate_reduction", -1.0)) >= 0.30
    )
    summary["decision"] = {
        "seed_counts": {str(key): int(value) for key, value in seed_counts.items()},
        "stage": "three_seed_pilot" if pilot else "full_old_seed_development",
        "held_out_datasets_passed": int(pass_count),
        "old_seed_expansion_allowed": bool(pilot and pass_count >= 2),
        "new_seed_or_dataset_migration_allowed": bool(
            (not pilot) and pass_count >= 2 and waterbirds_safety
        ),
        "waterbirds_safety_improvement": bool(waterbirds_safety),
    }
    decisions_path = output_root / "lodo_decisions.csv"
    conditions_path = output_root / "lodo_policy_conditions.csv"
    summary_path = output_root / "lodo_summary.json"
    _atomic_csv(pd.concat(decision_frames, ignore_index=True), decisions_path)
    _atomic_csv(pd.DataFrame(all_condition_rows), conditions_path)
    _atomic_json(summary, summary_path)
    _write_report(summary, output_root / "PILOT_REPORT.md")
    return summary_path


def _write_report(summary: dict[str, Any], path: Path) -> None:
    decision = summary["decision"]
    lines = [
        "# Transferable Budget Controller: Old-Seed Development Report",
        "",
        f"Stage: `{decision['stage']}`.",
        "",
        "| Held-out dataset | BAcc | Coverage | Bad continue | Mean budget | ΔWGA | Fixed-10 downside | Selected downside | Best-small regret | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for dataset, result in summary["held_out"].items():
        static = result["static_decisions"]
        policy = result["sequential_policy"]
        lines.append(
            f"| {dataset} | {static['balanced_accuracy']:.3f} | "
            f"{static['selective_coverage']:.3f} | {static['erroneous_continue_rate']:.3f} | "
            f"{100 * policy['mean_selected_budget_fraction']:.2f}% | "
            f"{policy['mean_delta_wga_pp']:+.3f} pp | "
            f"{policy['fixed_cap_negative_rate']:.3f} | {policy['selected_negative_rate']:.3f} | "
            f"{policy['mean_best_small_regret_pp']:.3f} pp | "
            f"{'YES' if result['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            f"Held-out datasets passed: {decision['held_out_datasets_passed']}/3.",
            "",
            f"Expand to all old seeds: **{'YES' if decision['old_seed_expansion_allowed'] else 'NO'}**.",
            "",
            f"Migrate to new seeds/data: **{'YES' if decision['new_seed_or_dataset_migration_allowed'] else 'NO'}**.",
            "",
            "An abstention deploys the current checkpoint. Dataset and corruption identity are excluded from the controller matrix.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "label", "evaluate", "all"))
    parser.add_argument("--configs", default=None, help="Comma-separated config paths")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise", default="uniform20,minority_high_40")
    parser.add_argument("--budgets", default="0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--method", default="noise_score_cpba_only")
    parser.add_argument("--validation-fractions", default="0.4,0.3,0.3")
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--cost", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--force-states", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"extract", "all"}:
        extract(args)
    if args.stage in {"label", "all"}:
        label(args)
    if args.stage in {"evaluate", "all"}:
        summary_path = evaluate(args)
        print(f"[complete] {summary_path}")


if __name__ == "__main__":
    main()
