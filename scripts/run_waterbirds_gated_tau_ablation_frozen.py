#!/usr/bin/env python
"""Waterbirds RV-Q-Gated tau ablation on fixed ResNet-50 features.

This is a local, exploratory ablation for the disjoint Waterbirds seeds 40--59.
It does not run full-network end-to-end training. The image feature cache is
reused, a deterministic linear probe is fit once per seed, and every tau arm
uses the same frozen-feature linear-head retraining protocol after querying.
The gated ranking keeps a NoiseScore anchor prefix and varies only the
validation-tail fraction used by the repair-value fill.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robust_verify.analysis import query_composition
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.scoring import build_adaptive_ranking, build_legal_scores, build_rankings
from robust_verify.training import LinearHead, make_initial_state, predict_probabilities, train_linear_head
from robust_verify.utils import budget_to_count, resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_ROOT = (
    ROOT
    / "tmp"
    / "waterbirds_gated_tau_ablation_source_20260903"
    / "outputs"
    / "waterbirds_e2e_repairvalue_v2_seed40_59"
)
DEFAULT_FEATURE_ROOT = ROOT / "outputs" / "ablation_waterbirds_20seeds" / "features"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "waterbirds_gated_tau_ablation_seed40_59_frozen"
SEEDS = tuple(range(40, 60))
TAUS = (0.20, 0.30, 0.35, 0.40, 0.50)
METHOD = "noise_gated_repair_value"
NOISE_NAME = "uniform20"
BUDGET_FRACTION = 0.10
CANDIDATE_MULTIPLIER = 2.0
NOISE_ANCHOR_FRACTION = 0.50


@dataclass(frozen=True)
class FeatureBundle:
    train_features: np.ndarray
    train_ids: np.ndarray
    val_features: np.ndarray
    val_ids: np.ndarray
    test_features: np.ndarray
    test_ids: np.ndarray


@dataclass(frozen=True)
class SeedData:
    seed: int
    public: pd.DataFrame
    private: pd.DataFrame
    val_public: pd.DataFrame
    test_public: pd.DataFrame
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    noisy_labels: np.ndarray
    val_labels: np.ndarray
    clean_labels: np.ndarray
    test_ids: np.ndarray


def _aligned_manifest(path: Path, sample_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"manifest lacks sample_id: {path}")
    indexed = frame.set_index("sample_id", drop=False)
    if not indexed.index.is_unique:
        raise ValueError(f"manifest has duplicate sample IDs: {path}")
    wanted = np.asarray(sample_ids, dtype=np.int64)
    missing = sorted(set(wanted.tolist()).difference(indexed.index.tolist()))
    if missing:
        raise ValueError(f"manifest is missing sample IDs: {path}")
    return indexed.loc[wanted].reset_index(drop=True)


def _load_features(feature_root: Path) -> FeatureBundle:
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        path = feature_root / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"feature cache is missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if not {"features", "sample_ids"}.issubset(payload.files):
                raise ValueError(f"feature cache schema mismatch: {path}")
            arrays[f"{split}_features"] = np.asarray(payload["features"], dtype=np.float32)
            arrays[f"{split}_ids"] = np.asarray(payload["sample_ids"], dtype=np.int64)
    if arrays["train_features"].ndim != 2:
        raise ValueError("train feature cache must be a matrix")
    if arrays["train_features"].shape[1] != arrays["val_features"].shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    return FeatureBundle(**arrays)


def _load_seed(
    *,
    seed: int,
    manifests: Path,
    features: FeatureBundle,
) -> SeedData:
    prefix = f"{NOISE_NAME}_seed{seed}"
    public = _aligned_manifest(manifests / f"{prefix}_train_public.csv", features.train_ids)
    private = _aligned_manifest(manifests / f"{prefix}_train_private.csv", features.train_ids)
    val_public = _aligned_manifest(manifests / "val_public.csv", features.val_ids)
    test_public = _aligned_manifest(manifests / "test_public.csv", features.test_ids)
    if not {"sample_id", "noisy_label"}.issubset(public.columns):
        raise ValueError(f"public training manifest lacks noisy labels: {prefix}")
    if not {"sample_id", "clean_label", "is_noisy", "group"}.issubset(private.columns):
        raise ValueError(f"private training manifest lacks audit fields: {prefix}")
    if "label" not in val_public.columns:
        raise ValueError("validation manifest must expose clean labels")
    return SeedData(
        seed=seed,
        public=public,
        private=private,
        val_public=val_public,
        test_public=test_public,
        train_features=features.train_features,
        val_features=features.val_features,
        test_features=features.test_features,
        noisy_labels=public["noisy_label"].to_numpy(dtype=np.int64),
        val_labels=val_public["label"].to_numpy(dtype=np.int64),
        clean_labels=private["clean_label"].to_numpy(dtype=np.int64),
        test_ids=features.test_ids,
    )


def _save_probe(path: Path, probe, *, seed: int, config: dict[str, object]) -> None:
    if probe.correctness_history is None or probe.val_probability_history is None:
        raise RuntimeError("linear probe did not record the dynamics required for scoring")
    path.parent.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "cache_version": 1,
        "seed": int(seed),
        "config": dict(config),
        "best_state": {key: value.detach().cpu() for key, value in probe.best_state.items()},
        "initial_state": {
            key: value.detach().cpu() for key, value in probe.initial_state.items()
        },
        "best_epoch": int(probe.best_epoch),
        "best_validation_metric": float(probe.best_validation_metric),
    }
    state_tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_payload, state_tmp)
    state_tmp.replace(path)
    dynamics_path = path.with_suffix(".dynamics.npz")
    dynamics_tmp = dynamics_path.with_suffix(".tmp.npz")
    val_history = np.asarray(probe.val_probability_history)
    np.savez_compressed(
        dynamics_tmp,
        train_probabilities=np.asarray(probe.train_probabilities, dtype=np.float64),
        correctness_history=np.asarray(probe.correctness_history, dtype=np.int8),
        val_probability_history=val_history.astype(np.float32),
        best_epoch=np.asarray(probe.best_epoch, dtype=np.int64),
    )
    dynamics_tmp.replace(dynamics_path)


def _load_probe(path: Path, *, seed: int) -> dict[str, object] | None:
    dynamics_path = path.with_suffix(".dynamics.npz")
    if not path.is_file() or not dynamics_path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    with np.load(dynamics_path, allow_pickle=False) as arrays:
        required = {"train_probabilities", "correctness_history", "val_probability_history", "best_epoch"}
        if not required.issubset(arrays.files):
            return None
        result = {
            "best_state": payload["best_state"],
            "initial_state": payload["initial_state"],
            "best_epoch": int(payload["best_epoch"]),
            "best_validation_metric": float(payload["best_validation_metric"]),
            "train_probabilities": np.asarray(arrays["train_probabilities"], dtype=np.float64),
            "correctness_history": np.asarray(arrays["correctness_history"], dtype=np.int8),
            "val_probability_history": np.asarray(arrays["val_probability_history"], dtype=np.float64),
        }
    if int(payload.get("seed", -1)) != int(seed):
        return None
    return result


def _fit_and_evaluate(
    *,
    data: SeedData,
    labels: np.ndarray,
    initial_state: dict[str, torch.Tensor],
    config: dict[str, object],
    evaluator: PrivateEvaluator,
) -> tuple[dict[str, float], object]:
    result = train_linear_head(
        train_features=data.train_features,
        train_labels=np.asarray(labels, dtype=np.int64),
        val_features=data.val_features,
        val_labels=data.val_labels,
        config=config,
        seed=data.seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
    )
    model = LinearHead(data.train_features.shape[1], 2)
    model.load_state_dict(result.best_state)
    device = resolve_device(config.get("device", "auto"))
    model.to(device)
    probabilities = predict_probabilities(
        model,
        data.test_features,
        int(config.get("batch_size", 4096)),
        device,
    )
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    metrics = evaluator.evaluate(data.test_ids, predictions)
    return {key: float(value) for key, value in metrics.items()}, result


def _probe_for_seed(
    *,
    data: SeedData,
    probe_path: Path,
    config: dict[str, object],
) -> dict[str, object]:
    cached = _load_probe(probe_path, seed=data.seed)
    if cached is not None:
        return cached
    initial_state = make_initial_state(data.train_features.shape[1], 2, data.seed)
    probe = train_linear_head(
        train_features=data.train_features,
        train_labels=data.noisy_labels,
        val_features=data.val_features,
        val_labels=data.val_labels,
        config=config,
        seed=data.seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=True,
    )
    _save_probe(probe_path, probe, seed=data.seed, config=config)
    return {
        "best_state": probe.best_state,
        "initial_state": probe.initial_state,
        "best_epoch": int(probe.best_epoch),
        "best_validation_metric": float(probe.best_validation_metric),
        "train_probabilities": np.asarray(probe.train_probabilities, dtype=np.float64),
        "correctness_history": np.asarray(probe.correctness_history, dtype=np.int8),
        "val_probability_history": np.asarray(probe.val_probability_history, dtype=np.float64),
    }


def _row_key(seed: int, tau: float) -> tuple[int, float]:
    return int(seed), round(float(tau), 2)


def run(args: argparse.Namespace) -> pd.DataFrame:
    protocol_root = Path(args.protocol_root).resolve()
    feature_root = Path(args.feature_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifests = protocol_root / "source" / "manifests"
    if not manifests.is_dir():
        raise FileNotFoundError(f"protocol manifests are missing: {manifests}")
    features = _load_features(feature_root)
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else list(SEEDS)
    )
    unknown = sorted(set(seeds).difference(SEEDS))
    if unknown:
        raise ValueError(f"seeds must lie in 40--59: {unknown}")
    taus = [round(float(value), 2) for value in args.taus.split(",") if value.strip()]
    invalid = sorted(set(taus).difference(TAUS))
    if invalid:
        raise ValueError(f"tau values must be from {TAUS}: {invalid}")
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "ablation_results.csv"
    if results_path.is_file() and results_path.stat().st_size:
        existing = pd.read_csv(results_path)
        if not set(existing.get("source", [])) <= {"local_frozen_feature_ablation"}:
            raise ValueError("existing output mixes a non-frozen ablation source")
        rows = existing.to_dict("records")
    else:
        rows = []
    completed = {
        _row_key(row["seed"], row["tau"])
        for row in rows
        if str(row.get("source", "")) == "local_frozen_feature_ablation"
    }
    requested = {_row_key(seed, tau) for seed in seeds for tau in taus}
    if args.dry_run:
        print(
            f"[dry-run] frozen features; seeds={seeds}; taus={taus}; "
            f"completed={len(completed)}; pending={len(requested - completed)}"
        )
        return pd.DataFrame(rows)

    probe_config = {
        "epochs": int(args.probe_epochs),
        "batch_size": int(args.probe_batch_size),
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 3,
        "selection_metric": "balanced_accuracy",
        "device": args.device,
    }
    retrain_config = {
        "epochs": int(args.retrain_epochs),
        "batch_size": int(args.retrain_batch_size),
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": args.device,
    }
    evaluator = PrivateEvaluator(manifests / "test_private.csv")
    probe_dir = output_root / "probes"
    query_dir = output_root / "queries"
    prediction_dir = output_root / "predictions"

    for seed in seeds:
        data = _load_seed(seed=seed, manifests=manifests, features=features)
        probe_path = probe_dir / f"{NOISE_NAME}_seed{seed}.pt"
        probe = _probe_for_seed(data=data, probe_path=probe_path, config=probe_config)
        train_probs = np.asarray(probe["train_probabilities"], dtype=np.float64)
        correctness = np.asarray(probe["correctness_history"], dtype=np.int8)
        val_probs = np.asarray(probe["val_probability_history"], dtype=np.float64)[
            :, int(probe["best_epoch"]), :
        ]
        legal = build_legal_scores(train_probs, data.noisy_labels, correctness)
        losses = np.asarray(legal["loss"], dtype=np.float64)
        noise_score = np.asarray(legal["noise_score"], dtype=np.float64)
        budget_count = budget_to_count(BUDGET_FRACTION, len(data.public))
        initial_state = probe["initial_state"]

        baseline_labels = {
            "no_correction": data.noisy_labels,
        }
        loss_positions = np.argsort(-losses, kind="stable")[:budget_count]
        noise_positions = np.lexsort((-losses, -noise_score))[:budget_count]
        baseline_labels["loss"] = data.noisy_labels.copy()
        baseline_labels["loss"][loss_positions] = data.clean_labels[loss_positions]
        baseline_labels["noise_score"] = data.noisy_labels.copy()
        baseline_labels["noise_score"][noise_positions] = data.clean_labels[noise_positions]
        baseline_metrics: dict[str, dict[str, float]] = {}
        for name, labels in baseline_labels.items():
            baseline_metrics[name], _ = _fit_and_evaluate(
                data=data,
                labels=labels,
                initial_state=initial_state,
                config=retrain_config,
                evaluator=evaluator,
            )
        seed_started = time.perf_counter()

        for tau in taus:
            key = _row_key(seed, tau)
            if key in completed:
                continue
            options = {
                "repair_value_tail_fraction": tau,
                "repair_value_candidate_multiplier": CANDIDATE_MULTIPLIER,
                "repair_value_noise_anchor_fraction": NOISE_ANCHOR_FRACTION,
            }
            context = {
                "train_features": data.train_features,
                "val_features": data.val_features,
                "val_probs": val_probs,
                "val_labels": data.val_labels,
                "baseline_options": options,
                "max_budget_count": budget_count,
            }
            _, score_map = build_rankings(
                methods=[METHOD],
                probabilities=train_probs,
                labels=data.noisy_labels,
                correctness_history=correctness,
                private_frame=data.public[["sample_id"]].copy(),
                seed=seed,
                context=context,
            )
            ranking = build_adaptive_ranking(
                METHOD,
                noise_score,
                train_probs,
                data.public[["sample_id"]].copy(),
                budget_count,
                len(data.public),
                losses=losses,
                labels=data.noisy_labels,
                repair_value_scores=score_map[METHOD],
                options=options,
            )
            positions = np.asarray(ranking[:budget_count], dtype=np.int64)
            queried_ids = data.public.iloc[positions]["sample_id"].to_numpy(dtype=np.int64)
            query_path = query_dir / f"seed_{seed}" / f"tau_{tau:.2f}.npz"
            query_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(query_path, sample_ids=queried_ids)
            repaired_labels = data.noisy_labels.copy()
            repaired_labels[positions] = VerificationOracle(
                manifests / f"{NOISE_NAME}_seed{seed}_train_private.csv"
            ).verify(queried_ids)
            metrics, retrained = _fit_and_evaluate(
                data=data,
                labels=repaired_labels,
                initial_state=initial_state,
                config=retrain_config,
                evaluator=evaluator,
            )
            prediction_path = prediction_dir / f"seed_{seed}" / f"tau_{tau:.2f}.npz"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            model = LinearHead(data.train_features.shape[1], 2)
            model.load_state_dict(retrained.best_state)
            device = resolve_device(retrain_config.get("device", "auto"))
            model.to(device)
            test_predictions = predict_probabilities(
                model,
                data.test_features,
                int(retrain_config["batch_size"]),
                device,
            ).argmax(axis=1).astype(np.int64)
            np.savez_compressed(
                prediction_path,
                sample_ids=data.test_ids,
                predictions=test_predictions,
            )
            composition = query_composition(positions, data.private)
            row = {
                "dataset": "waterbirds",
                "protocol_block": "waterbirds_gated_tau_ablation_seed40_59_frozen",
                "seed": seed,
                "tau": tau,
                "method": METHOD,
                "source": "local_frozen_feature_ablation",
                "retraining_scope": "frozen_feature_linear_head",
                "full_network_training_executed": False,
                "budget_fraction": BUDGET_FRACTION,
                "budget_count": budget_count,
                "candidate_multiplier": CANDIDATE_MULTIPLIER,
                "noise_anchor_fraction": NOISE_ANCHOR_FRACTION,
                "baseline_wga": baseline_metrics["no_correction"]["wga"],
                "baseline_average_accuracy": baseline_metrics["no_correction"]["average_accuracy"],
                "loss_wga": baseline_metrics["loss"]["wga"],
                "noise_score_wga": baseline_metrics["noise_score"]["wga"],
                "wga": metrics["wga"],
                "average_accuracy": metrics["average_accuracy"],
                "delta_wga": metrics["wga"] - baseline_metrics["no_correction"]["wga"],
                "delta_average_accuracy": metrics["average_accuracy"]
                - baseline_metrics["no_correction"]["average_accuracy"],
                "paired_vs_loss_wga": metrics["wga"] - baseline_metrics["loss"]["wga"],
                "paired_vs_noise_score_wga": metrics["wga"]
                - baseline_metrics["noise_score"]["wga"],
                "queried_count": int(composition["num_queried"]),
                "corrected_count": int(composition["num_corrected"]),
                "noise_precision": float(composition["noise_precision"]),
                "minority_query_rate": float(composition["minority_query_rate"]),
                "probe_best_epoch": int(probe["best_epoch"]),
                "probe_validation_metric": float(probe["best_validation_metric"]),
                "retrain_best_epoch": int(retrained.best_epoch),
                "retrain_validation_metric": float(retrained.best_validation_metric),
                "query_artifact": str(query_path.relative_to(output_root)),
                "prediction_artifact": str(prediction_path.relative_to(output_root)),
                "elapsed_seconds": time.perf_counter() - seed_started,
            }
            rows = [
                prior
                for prior in rows
                if _row_key(prior["seed"], prior["tau"]) != key
            ]
            rows.append(row)
            pd.DataFrame(rows).sort_values(["seed", "tau"]).to_csv(results_path, index=False)
            completed.add(key)
            print(
                f"seed {seed} tau={tau:.2f}: "
                f"delta_wga={100 * row['delta_wga']:+.3f} pp, "
                f"vs NoiseScore={100 * row['paired_vs_noise_score_wga']:+.3f} pp, "
                f"precision={100 * row['noise_precision']:.2f}%",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["seed", "tau"]).reset_index(drop=True)
        frame.to_csv(results_path, index=False)
    expected = {_row_key(seed, tau) for seed in seeds for tau in TAUS}
    actual = {
        _row_key(row.seed, row.tau)
        for row in frame.itertuples(index=False)
        if int(row.seed) in seeds
    }
    metadata = {
        "status": "complete" if expected.issubset(actual) else "incomplete",
        "dataset": "waterbirds",
        "seeds": seeds,
        "taus": list(TAUS),
        "budget_fraction": BUDGET_FRACTION,
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "noise_anchor_fraction": NOISE_ANCHOR_FRACTION,
        "retraining_scope": "frozen_feature_linear_head",
        "full_network_training_executed": False,
        "feature_root": str(feature_root),
        "protocol_root": str(protocol_root),
        "probe_config": probe_config,
        "retrain_config": retrain_config,
        "selection_rule": "mean frozen-feature linear-head validation balanced accuracy; test WGA not used",
    }
    (output_root / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if expected.issubset(actual):
        (output_root / "ABLATION_COMPLETE.json").write_text(
            json.dumps({**metadata, "rows": len(expected)}, indent=2), encoding="utf-8"
        )
    print(f"wrote {len(frame)} rows to {results_path}", flush=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(DEFAULT_PROTOCOL_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", default=None, help="comma-separated subset of 40--59")
    parser.add_argument("--taus", default=",".join(f"{tau:.2f}" for tau in TAUS))
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--retrain-epochs", type=int, default=20)
    parser.add_argument("--retrain-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="retained for launcher compatibility")
    parser.add_argument("--allow-replay-mismatch", action="store_true", help="retained for launcher compatibility")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
