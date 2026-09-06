#!/usr/bin/env python
"""Sequential, three-way-split budget stopping for label verification.

`prepare` freezes decisions without opening test features or private test data.
`evaluate` first reconstructs the fixed-cap comparator, then opens the test
artifacts.  Existing Stage-1 query IDs are treated as a public acquisition
trajectory; only exact ordered prefixes are legal for sequential stopping.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
    audit_accepts,
    nominate_by_calibration,
    paired_stratified_bootstrap_risk_difference,
    sample_id_hash,
    stratified_validation_partition,
)
from robust_verify.rivet import spectral_class_cvar
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count


DEFAULT_CONFIG = ROOT / "configs" / "ablation_waterbirds_10seeds.yaml"
RESULT_FIELDS = [
    "source_output",
    "noise_name",
    "seed",
    "method",
    "selected_budget_fraction",
    "selected_query_count",
    "spent_budget_fraction",
    "spent_query_count",
    "stop_reason",
    "accepted_checkpoints",
    "baseline_wga",
    "selected_wga",
    "fixed_cap_budget_fraction",
    "fixed_cap_wga",
    "delta_wga_vs_baseline",
    "delta_wga_vs_fixed_cap",
    "negative_utility",
    "average_accuracy",
    "balanced_accuracy",
    "decision_path",
]


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
    frame.reindex(columns=RESULT_FIELDS).to_csv(
        temporary, index=False, quoting=csv.QUOTE_MINIMAL
    )
    temporary.replace(path)


def _state_path(
    output_root: Path,
    source_name: str,
    noise: str,
    seed: int,
    budget: float,
    *,
    comparator: bool = False,
) -> Path:
    folder = "comparators" if comparator else "policy_states"
    return (
        output_root
        / folder
        / source_name
        / f"{noise}_seed{seed}"
        / f"budget_{budget:.4f}.pt"
    )


def _decision_path(output_root: Path, source_name: str, noise: str, seed: int) -> Path:
    return output_root / "decisions" / source_name / f"{noise}_seed{seed}.json"


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


def _load_probe(source_root: Path, noise: str, seed: int) -> dict[str, Any]:
    path = source_root / "stage1" / "probes" / f"{noise}_seed{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def _fit_state(
    *,
    train_features: np.ndarray,
    noisy_labels: np.ndarray,
    repaired_positions: np.ndarray,
    repaired_values: np.ndarray,
    early_features: np.ndarray,
    early_labels: np.ndarray,
    training_config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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
    return result.best_state


def _probabilities(
    state: dict[str, torch.Tensor],
    features: np.ndarray,
    *,
    input_dim: int,
    num_classes: int,
    training_config: dict[str, Any],
) -> np.ndarray:
    return predict_from_state(
        state, features, input_dim, num_classes, training_config
    )[1]


def _risk(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    alphas: tuple[float, ...],
    temperature: float,
) -> float:
    return float(
        spectral_class_cvar(
            probabilities, labels, alphas=alphas, temperature=temperature
        )[0]
    )


def _save_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"best_state": state}, path)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["best_state"]


def _upsert_result(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = pd.DataFrame(columns=RESULT_FIELDS)
    key = (
        str(row["source_output"]),
        str(row["noise_name"]),
        int(row["seed"]),
        str(row["method"]),
    )
    if len(frame):
        keep = [
            (
                str(existing["source_output"]),
                str(existing["noise_name"]),
                int(existing["seed"]),
                str(existing["method"]),
            )
            != key
            for _, existing in frame.iterrows()
        ]
        frame = frame.loc[keep]
    if len(frame):
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row], columns=RESULT_FIELDS)
    frame = frame.sort_values(
        ["source_output", "noise_name", "seed", "method"], kind="stable"
    ).reset_index(drop=True)
    _atomic_csv(frame, path)


def prepare(args: argparse.Namespace) -> list[Path]:
    config_path = _resolve(args.config)
    config = load_config(config_path)
    source_root = _resolve(config["project"]["output_dir"])
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = (
        _parse_csv(args.seeds, int)
        if args.seeds is not None
        else tuple(int(value) for value in config["experiment"]["seeds"])
    )
    noises = _parse_csv(args.noise, str)
    budgets = tuple(sorted(set(_parse_csv(args.budgets, float))))
    alphas = tuple(_parse_csv(args.alphas, float))
    fractions = tuple(_parse_csv(args.validation_fractions, float))
    if len(fractions) != 3:
        raise ValueError("validation-fractions must contain three values")
    if any(budget <= 0.0 or budget > 1.0 for budget in budgets):
        raise ValueError("budgets must lie in (0, 1]")
    if args.scout_budget != 0.0 and not np.isclose(args.scout_budget, budgets[0]):
        raise ValueError("scout-budget must be zero or the first positive checkpoint")
    if args.familywise_alpha <= 0.0 or args.familywise_alpha >= 1.0:
        raise ValueError("familywise-alpha must lie in (0, 1)")

    train_features, train_ids = load_features(source_root / "features" / "train.npz")
    val_features, val_ids = load_features(source_root / "features" / "val.npz")
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
    split_hashes = {
        "early_stop": sample_id_hash(val_ids, partition.early_stop),
        "calibration": sample_id_hash(val_ids, partition.calibration),
        "audit": sample_id_hash(val_ids, partition.audit),
    }
    split_counts = {
        "early_stop": int(len(partition.early_stop)),
        "calibration": int(len(partition.calibration)),
        "audit": int(len(partition.audit)),
    }
    training_config = dict(config["training"])
    source_name = source_root.name
    input_dim = int(train_features.shape[1])
    audited_comparisons = len(budgets) - int(args.scout_budget > 0.0)
    per_comparison_confidence = 1.0 - args.familywise_alpha / max(
        1, audited_comparisons
    )

    design = {
        "protocol": (
            "mandatory_scout_sequential_risk_stopping_v2"
            if args.scout_budget > 0.0
            else "sequential_three_way_risk_stopping_v1"
        ),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_output": source_name,
        "method": args.method,
        "budgets": list(budgets),
        "alphas": list(alphas),
        "temperature": args.temperature,
        "validation_fractions": list(fractions),
        "split_seed": args.split_seed,
        "split_hashes": split_hashes,
        "split_counts": split_counts,
        "familywise_alpha": args.familywise_alpha,
        "per_comparison_confidence": per_comparison_confidence,
        "bootstrap_replicates": args.bootstrap_reps,
        "scout_budget": args.scout_budget,
        "minimum_calibration_improvement": args.minimum_calibration_improvement,
        "minimum_audit_improvement": args.minimum_audit_improvement,
        "evidence_role": "development_only_existing_test_outcomes_previously_seen",
    }
    _atomic_json(design, output_root / "design.json")

    decision_paths: list[Path] = []
    for noise in noises:
        for seed in seeds:
            decision_path = _decision_path(output_root, source_name, noise, seed)
            decision_paths.append(decision_path)
            if decision_path.exists() and not args.force:
                print(f"[resume] {decision_path}")
                continue

            query_paths = [
                _query_path(source_root, noise, seed, args.method, budget)
                for budget in budgets
            ]
            query_sets = [np.load(path, allow_pickle=False) for path in query_paths]
            assert_nested_query_prefixes(query_sets)
            query_hashes = {
                f"{budget:g}": _sha256(path)
                for budget, path in zip(budgets, query_paths)
            }
            for budget, query_ids in zip(budgets, query_sets):
                expected = budget_to_count(budget, len(train_ids))
                if len(query_ids) != expected:
                    raise ValueError(
                        f"{noise} seed={seed} b={budget:g}: "
                        f"expected {expected} query IDs, found {len(query_ids)}"
                    )

            prefix = f"{noise}_seed{seed}"
            public = align_frame(
                pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
                train_ids,
            )
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
            num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
            probe = _load_probe(source_root, noise, seed)
            oracle = VerificationOracle(
                source_root / "manifests" / f"{prefix}_train_private.csv"
            )

            baseline_path = _state_path(
                output_root, source_name, noise, seed, 0.0
            )
            baseline_state = _fit_state(
                train_features=train_features,
                noisy_labels=noisy_labels,
                repaired_positions=np.asarray([], dtype=np.int64),
                repaired_values=np.asarray([], dtype=np.int64),
                early_features=val_features[partition.early_stop],
                early_labels=val_labels[partition.early_stop],
                training_config=training_config,
                seed=seed,
                initial_state=probe["initial_state"],
            )
            _save_state(baseline_state, baseline_path)
            best_state = baseline_state
            best_state_path = baseline_path
            best_budget = 0.0
            best_query_count = 0
            best_calibration_probabilities = _probabilities(
                best_state,
                val_features[partition.calibration],
                input_dim=input_dim,
                num_classes=num_classes,
                training_config=training_config,
            )
            best_calibration_risk = _risk(
                best_calibration_probabilities,
                val_labels[partition.calibration],
                alphas=alphas,
                temperature=args.temperature,
            )
            best_audit_probabilities: np.ndarray | None = None
            train_id_lookup = {
                int(value): index for index, value in enumerate(train_ids)
            }
            trace: list[dict[str, Any]] = []
            spent_budget = 0.0
            spent_query_count = 0
            stop_reason = "fixed_cap_reached"

            for checkpoint_index, (budget, query_path, query_ids) in enumerate(
                zip(budgets, query_paths, query_sets)
            ):
                spent_budget = float(budget)
                spent_query_count = int(len(query_ids))
                positions = np.asarray(
                    [train_id_lookup[int(sample_id)] for sample_id in query_ids],
                    dtype=np.int64,
                )
                candidate_state = _fit_state(
                    train_features=train_features,
                    noisy_labels=noisy_labels,
                    repaired_positions=positions,
                    repaired_values=oracle.verify(query_ids),
                    early_features=val_features[partition.early_stop],
                    early_labels=val_labels[partition.early_stop],
                    training_config=training_config,
                    seed=seed,
                    initial_state=probe["initial_state"],
                )
                candidate_path = _state_path(
                    output_root, source_name, noise, seed, budget
                )
                _save_state(candidate_state, candidate_path)
                candidate_calibration_probabilities = _probabilities(
                    candidate_state,
                    val_features[partition.calibration],
                    input_dim=input_dim,
                    num_classes=num_classes,
                    training_config=training_config,
                )
                candidate_calibration_risk = _risk(
                    candidate_calibration_probabilities,
                    val_labels[partition.calibration],
                    alphas=alphas,
                    temperature=args.temperature,
                )
                mandatory_scout = bool(
                    args.scout_budget > 0.0
                    and np.isclose(budget, args.scout_budget)
                )
                nominated = nominate_by_calibration(
                    candidate_calibration_risk,
                    best_calibration_risk,
                    minimum_improvement=args.minimum_calibration_improvement,
                )
                step: dict[str, Any] = {
                    "budget_fraction": float(budget),
                    "query_count": int(len(query_ids)),
                    "incremental_query_count": int(len(query_ids) - best_query_count),
                    "reference_budget_fraction": float(best_budget),
                    "candidate_calibration_risk": candidate_calibration_risk,
                    "reference_calibration_risk": best_calibration_risk,
                    "nominated": bool(nominated),
                    "accepted": False,
                    "acceptance_role": (
                        "mandatory_scout" if mandatory_scout else "audited_increment"
                    ),
                    "state_path": str(candidate_path),
                    "query_path": str(query_path),
                }
                if mandatory_scout:
                    step["nominated"] = True
                    step["accepted"] = True
                    trace.append(step)
                    best_state = candidate_state
                    best_state_path = candidate_path
                    best_budget = float(budget)
                    best_query_count = int(len(query_ids))
                    best_calibration_probabilities = (
                        candidate_calibration_probabilities
                    )
                    best_calibration_risk = candidate_calibration_risk
                    best_audit_probabilities = None
                    continue
                if not nominated:
                    stop_reason = "calibration_no_improvement"
                    trace.append(step)
                    break

                candidate_audit_probabilities = _probabilities(
                    candidate_state,
                    val_features[partition.audit],
                    input_dim=input_dim,
                    num_classes=num_classes,
                    training_config=training_config,
                )
                if best_audit_probabilities is None:
                    best_audit_probabilities = _probabilities(
                        best_state,
                        val_features[partition.audit],
                        input_dim=input_dim,
                        num_classes=num_classes,
                        training_config=training_config,
                    )
                audit = paired_stratified_bootstrap_risk_difference(
                    candidate_audit_probabilities,
                    best_audit_probabilities,
                    val_labels[partition.audit],
                    alphas=alphas,
                    temperature=args.temperature,
                    confidence=per_comparison_confidence,
                    replicates=args.bootstrap_reps,
                    seed=args.split_seed + 1009 * seed + checkpoint_index,
                )
                accepted = audit_accepts(
                    audit, minimum_improvement=args.minimum_audit_improvement
                )
                step["audit"] = {
                    "observed_difference": audit.observed_difference,
                    "lower": audit.lower,
                    "upper": audit.upper,
                    "confidence": audit.confidence,
                    "bootstrap_replicates": audit.bootstrap_replicates,
                }
                step["accepted"] = bool(accepted)
                trace.append(step)
                if not accepted:
                    stop_reason = "audit_rejected"
                    break

                best_state = candidate_state
                best_state_path = candidate_path
                best_budget = float(budget)
                best_query_count = int(len(query_ids))
                best_calibration_probabilities = candidate_calibration_probabilities
                best_calibration_risk = candidate_calibration_risk
                best_audit_probabilities = candidate_audit_probabilities

            decision = {
                **design,
                "noise_name": noise,
                "seed": int(seed),
                "query_hashes": query_hashes,
                "selected_budget_fraction": best_budget,
                "selected_query_count": best_query_count,
                "selected_state_path": str(best_state_path),
                "spent_budget_fraction": spent_budget,
                "spent_query_count": spent_query_count,
                "stop_reason": stop_reason,
                "accepted_checkpoints": int(sum(bool(item["accepted"]) for item in trace)),
                "trace": trace,
                "test_artifacts_opened": False,
            }
            _atomic_json(decision, decision_path)
            print(
                f"[frozen] {prefix} selected={best_budget:g} spent={spent_budget:g} "
                f"reason={stop_reason}"
            )
    return decision_paths


def evaluate(args: argparse.Namespace, decision_paths: list[Path] | None = None) -> None:
    config_path = _resolve(args.config)
    config = load_config(config_path)
    source_root = _resolve(config["project"]["output_dir"])
    output_root = _resolve(args.output_root)
    source_name = source_root.name
    if decision_paths is None:
        seeds = (
            _parse_csv(args.seeds, int)
            if args.seeds is not None
            else tuple(int(value) for value in config["experiment"]["seeds"])
        )
        noises = _parse_csv(args.noise, str)
        decision_paths = [
            _decision_path(output_root, source_name, noise, seed)
            for noise in noises
            for seed in seeds
        ]
    missing = [path for path in decision_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen decisions: {missing[:3]}")

    train_features, train_ids = load_features(source_root / "features" / "train.npz")
    val_features, val_ids = load_features(source_root / "features" / "val.npz")
    val_public = align_frame(
        pd.read_csv(source_root / "manifests" / "val_public.csv"), val_ids
    )
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    training_config = dict(config["training"])
    input_dim = int(train_features.shape[1])
    prepared: list[tuple[dict[str, Any], dict[str, torch.Tensor]]] = []

    # Comparator reconstruction is completed before test artifacts are opened.
    for decision_path in decision_paths:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        partition = stratified_validation_partition(
            val_ids,
            val_labels,
            fractions=tuple(float(value) for value in decision["validation_fractions"]),
            seed=int(decision["split_seed"]),
        )
        observed_hashes = {
            "early_stop": sample_id_hash(val_ids, partition.early_stop),
            "calibration": sample_id_hash(val_ids, partition.calibration),
            "audit": sample_id_hash(val_ids, partition.audit),
        }
        if observed_hashes != decision["split_hashes"]:
            raise RuntimeError(f"validation split hash mismatch: {decision_path}")
        noise = str(decision["noise_name"])
        seed = int(decision["seed"])
        prefix = f"{noise}_seed{seed}"
        fixed_budget = max(float(value) for value in decision["budgets"])
        fixed_path = _state_path(
            output_root,
            source_name,
            noise,
            seed,
            fixed_budget,
            comparator=True,
        )
        if not fixed_path.exists():
            public = align_frame(
                pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
                train_ids,
            )
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
            probe = _load_probe(source_root, noise, seed)
            query_path = _query_path(
                source_root, noise, seed, str(decision["method"]), fixed_budget
            )
            query_ids = np.load(query_path, allow_pickle=False)
            lookup = {int(value): index for index, value in enumerate(train_ids)}
            positions = np.asarray([lookup[int(value)] for value in query_ids], dtype=np.int64)
            oracle = VerificationOracle(
                source_root / "manifests" / f"{prefix}_train_private.csv"
            )
            fixed_state = _fit_state(
                train_features=train_features,
                noisy_labels=noisy_labels,
                repaired_positions=positions,
                repaired_values=oracle.verify(query_ids),
                early_features=val_features[partition.early_stop],
                early_labels=val_labels[partition.early_stop],
                training_config=training_config,
                seed=seed,
                initial_state=probe["initial_state"],
            )
            _save_state(fixed_state, fixed_path)
        prepared.append((decision, _load_state(fixed_path)))

    test_features, test_ids = load_features(source_root / "features" / "test.npz")
    evaluator = PrivateEvaluator(source_root / "manifests" / "test_private.csv")
    results_path = output_root / "selected_results.csv"
    for decision, fixed_state in prepared:
        noise = str(decision["noise_name"])
        seed = int(decision["seed"])
        prefix = f"{noise}_seed{seed}"
        public = align_frame(
            pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
            train_ids,
        )
        noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
        num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
        baseline_path = _state_path(output_root, source_name, noise, seed, 0.0)
        selected_path = Path(decision["selected_state_path"])
        states = {
            "baseline": _load_state(baseline_path),
            "selected": _load_state(selected_path),
            "fixed": fixed_state,
        }
        metrics: dict[str, dict[str, float]] = {}
        for name, state in states.items():
            predictions = predict_from_state(
                state, test_features, input_dim, num_classes, training_config
            )[0]
            metrics[name] = evaluator.evaluate(test_ids, predictions)
        baseline_wga = float(metrics["baseline"]["wga"])
        selected_wga = float(metrics["selected"]["wga"])
        fixed_wga = float(metrics["fixed"]["wga"])
        row = {
            "source_output": source_name,
            "noise_name": noise,
            "seed": seed,
            "method": str(decision["method"]),
            "selected_budget_fraction": float(decision["selected_budget_fraction"]),
            "selected_query_count": int(decision["selected_query_count"]),
            "spent_budget_fraction": float(decision["spent_budget_fraction"]),
            "spent_query_count": int(decision["spent_query_count"]),
            "stop_reason": str(decision["stop_reason"]),
            "accepted_checkpoints": int(decision["accepted_checkpoints"]),
            "baseline_wga": baseline_wga,
            "selected_wga": selected_wga,
            "fixed_cap_budget_fraction": max(float(value) for value in decision["budgets"]),
            "fixed_cap_wga": fixed_wga,
            "delta_wga_vs_baseline": selected_wga - baseline_wga,
            "delta_wga_vs_fixed_cap": selected_wga - fixed_wga,
            "negative_utility": int(selected_wga < baseline_wga),
            "average_accuracy": float(metrics["selected"]["average_accuracy"]),
            "balanced_accuracy": float(metrics["selected"]["balanced_accuracy"]),
            "decision_path": str(
                _decision_path(output_root, source_name, noise, seed)
            ),
        }
        _upsert_result(results_path, row)
        print(
            f"[test] {prefix} selected={row['selected_budget_fraction']:g} "
            f"spent={row['spent_budget_fraction']:g} "
            f"dWGA={row['delta_wga_vs_baseline']:+.4f} "
            f"vs_cap={row['delta_wga_vs_fixed_cap']:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential risk-aware stopping with independent validation roles."
    )
    parser.add_argument("--phase", choices=("prepare", "evaluate", "all"), default="prepare")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--noise", default="uniform20")
    parser.add_argument("--method", default="noise_score_cpba_only")
    parser.add_argument("--budgets", default="0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--alphas", default="0.01,0.02,0.05")
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--validation-fractions", default="0.4,0.3,0.3")
    parser.add_argument("--split-seed", type=int, default=20260810)
    parser.add_argument("--familywise-alpha", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--minimum-calibration-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-audit-improvement", type=float, default=0.0)
    parser.add_argument(
        "--scout-budget",
        type=float,
        default=0.0,
        help="Unconditionally accept the first checkpoint as a paid scout.",
    )
    parser.add_argument(
        "--output-root", default="outputs/risk_aware_budget_stopping"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "evaluate":
        evaluate(args)
    else:
        decisions = prepare(args)
        evaluate(args, decisions)


if __name__ == "__main__":
    main()
