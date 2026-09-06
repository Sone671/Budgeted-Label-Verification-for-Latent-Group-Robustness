#!/usr/bin/env python
"""Run frozen-feature RepairValue-Q retraining sensitivity conditions.

The initial image probe is reused from cached Stage-1 dynamics. Each condition
changes only the validation pool used to score RepairValue-Q, then performs the
actual label repair and linear-head retraining on frozen features. Clean full
validation labels remain the checkpoint-selection set during retraining, so
validation-quality effects are isolated to acquisition scoring.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from robust_verify.config import output_layout
from robust_verify.data.access import PrivateEvaluator
from robust_verify.end_to_end import _device as image_device
from robust_verify.training import (
    LinearHead,
    make_initial_state,
    predict_probabilities,
    train_linear_head,
)
from robust_verify.utils import budget_to_count

try:
    from scripts.analyze_repairvalue_validation_sensitivity import (
        condition_grid,
        load_seed_data,
        score_condition,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from analyze_repairvalue_validation_sensitivity import (
        condition_grid,
        load_seed_data,
        score_condition,
    )


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO / "outputs/ablation_celeba_10seeds"
DEFAULT_PROBE_DIR = (
    REPO / "outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/probes"
)
DEFAULT_BASELINE_RESULTS = (
    REPO / "outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/results.csv"
)
DEFAULT_OUTPUT_DIR = REPO / "outputs/local_repairvalue_validation_retraining"


def _load_private_labels(input_dir: str | Path, seed: int, noise_name: str) -> np.ndarray:
    root = output_layout(input_dir)
    path = root["manifests"] / f"{noise_name}_seed{int(seed)}_train_private.csv"
    frame = pd.read_csv(path).sort_values("sample_id")
    return frame["clean_label"].to_numpy(dtype=np.int64)


def _load_baseline_row(
    path: str | Path,
    *,
    seed: int,
    method: str = "loss",
    budget_fraction: float,
    noise_name: str | None = None,
) -> dict[str, float]:
    frame = pd.read_csv(path)
    rows = frame[
        frame["seed"].eq(int(seed))
        & frame["method"].eq(method)
        & np.isclose(frame["budget_fraction"].astype(float), float(budget_fraction))
    ]
    if noise_name is not None and "noise_name" in frame.columns:
        rows = rows[rows["noise_name"].eq(noise_name)]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one cached {method} baseline for seed={seed}, "
            f"budget={budget_fraction}; found {len(rows)}"
        )
    row = rows.iloc[0]
    return {
        "baseline_wga": float(row["baseline_wga"]),
        "baseline_average_accuracy": float(row["baseline_average_accuracy"]),
        "loss_delta_wga": float(row["delta_wga"]),
        "loss_delta_average_accuracy": float(row["delta_average_accuracy"]),
    }


def _predict_test(
    state: dict[str, torch.Tensor],
    test_features: np.ndarray,
    *,
    input_dim: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = LinearHead(input_dim, 2)
    model.load_state_dict({key: value.detach().cpu() for key, value in state.items()})
    model.to(device)
    probabilities = predict_probabilities(model, test_features, batch_size, device)
    return probabilities.argmax(axis=1).astype(np.int64)


def retrain_condition(
    *,
    input_dir: str | Path,
    probe_dir: str | Path,
    baseline_results: str | Path,
    seed: int,
    tau: float,
    validation_fraction: float,
    validation_noise_rate: float,
    validation_noise_mode: str,
    subset_seed: int,
    corruption_seed: int,
    budget_fraction: float,
    noise_name: str,
    training_config: dict[str, object],
    validation_checkpoint_fallback: bool = False,
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    data = load_seed_data(
        input_dir,
        probe_dir,
        seed=seed,
        noise_name=noise_name,
        validation_checkpoint_fallback=validation_checkpoint_fallback,
    )
    private_root = output_layout(input_dir)["manifests"]
    private_train = pd.read_csv(
        private_root / f"{noise_name}_seed{int(seed)}_train_private.csv"
    ).sort_values("sample_id")
    if not np.array_equal(
        private_train["sample_id"].to_numpy(dtype=np.int64), data.train_sample_ids
    ):
        raise ValueError(f"Private train IDs do not align for seed {seed}")
    clean_train_labels = private_train["clean_label"].to_numpy(dtype=np.int64)
    score_row, ranking = score_condition(
        data,
        tau=tau,
        validation_fraction=validation_fraction,
        validation_noise_rate=validation_noise_rate,
        validation_noise_mode=validation_noise_mode,
        subset_seed=subset_seed,
        corruption_seed=corruption_seed,
        budget_fraction=budget_fraction,
    )
    budget_count = budget_to_count(budget_fraction, len(data.train_labels))
    query_positions = ranking[:budget_count]
    repaired_labels = data.train_labels.copy()
    repaired_labels[query_positions] = clean_train_labels[query_positions]

    initial_state = make_initial_state(data.train_features.shape[1], 2, seed)
    retrained = train_linear_head(
        train_features=data.train_features,
        train_labels=repaired_labels,
        val_features=data.validation_features,
        val_labels=data.validation_labels,
        config=training_config,
        seed=seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
    )
    root = output_layout(input_dir)
    test_npz = np.load(root["features"] / "test.npz")
    test_features = np.asarray(test_npz["features"], dtype=np.float32)
    test_ids = np.asarray(test_npz["sample_ids"], dtype=np.int64)
    device = image_device(training_config)
    predictions = _predict_test(
        retrained.best_state,
        test_features,
        input_dim=data.train_features.shape[1],
        device=device,
        batch_size=int(training_config.get("batch_size", 4096)),
    )
    evaluator = PrivateEvaluator(root["manifests"] / "test_private.csv")
    metrics = evaluator.evaluate(test_ids, predictions)
    baseline = baseline_metrics or _load_baseline_row(
        baseline_results,
        seed=seed,
        budget_fraction=budget_fraction,
        noise_name=noise_name,
    )
    row = {
        **score_row,
        **metrics,
        **baseline,
        "delta_wga": metrics["wga"] - baseline["baseline_wga"],
        "delta_average_accuracy": (
            metrics["average_accuracy"] - baseline["baseline_average_accuracy"]
        ),
        "rv_minus_loss_wga_pp": (
            metrics["wga"] - baseline["baseline_wga"] - baseline["loss_delta_wga"]
        )
        * 100.0,
        "rv_minus_loss_average_accuracy_pp": (
            metrics["average_accuracy"]
            - baseline["baseline_average_accuracy"]
            - baseline["loss_delta_average_accuracy"]
        )
        * 100.0,
        "retraining_scope": "frozen_feature_linear_head",
        "retraining_best_epoch": int(retrained.best_epoch),
        "retraining_best_validation_metric": float(retrained.best_validation_metric),
    }
    return row


def run_retraining_analysis(
    *,
    input_dir: str | Path,
    probe_dir: str | Path,
    baseline_results: str | Path,
    output_dir: str | Path,
    seeds: Iterable[int],
    taus: Iterable[float],
    validation_fractions: Iterable[float],
    validation_noise_rates: Iterable[float],
    validation_noise_modes: Iterable[str],
    budget_fraction: float,
    noise_name: str,
    design: str,
    training_config: dict[str, object],
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions = condition_grid(
        taus,
        validation_fractions,
        validation_noise_rates,
        validation_noise_modes,
        design=design,
    )
    partial_path = output / "per_seed_retraining.partial.csv"
    existing = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
    completed = {
        (int(row.seed), int(row.condition_index))
        for row in existing.itertuples()
    } if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    for seed in [int(value) for value in seeds]:
        for condition_index, tau, fraction, rate, mode in conditions:
            key = (seed, int(condition_index))
            if key in completed:
                continue
            row = retrain_condition(
                input_dir=input_dir,
                probe_dir=probe_dir,
                baseline_results=baseline_results,
                seed=seed,
                tau=tau,
                validation_fraction=fraction,
                validation_noise_rate=rate,
                validation_noise_mode=mode,
                subset_seed=seed * 1_000_003 + condition_index * 10_007,
                corruption_seed=seed * 2_000_003 + condition_index * 10_009,
                budget_fraction=budget_fraction,
                noise_name=noise_name,
                training_config=training_config,
            )
            row["condition_index"] = int(condition_index)
            rows.append(row)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(
                f"seed={seed} condition={condition_index} tau={tau:.2f} "
                f"val_fraction={fraction:.2f} val_noise={rate:.2f}/{mode} complete"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_retraining.csv", index=False)
    if partial_path.exists():
        partial_path.unlink()
    if not frame.empty:
        keys = [
            "tail_fraction",
            "validation_fraction",
            "validation_noise_rate",
            "validation_noise_mode",
        ]
        summary = frame.groupby(keys, sort=True).agg(
            n_seeds=("seed", "nunique"),
            mean_delta_wga_pp=("delta_wga", lambda values: 100.0 * values.mean()),
            mean_rv_minus_loss_wga_pp=("rv_minus_loss_wga_pp", "mean"),
            mean_query_noise_precision_pct=(
                "query_noise_precision", lambda values: 100.0 * values.mean()
            ),
            mean_validation_label_accuracy_pct=(
                "validation_label_accuracy", lambda values: 100.0 * values.mean()
            ),
            positive_rv_minus_loss_seeds=(
                "rv_minus_loss_wga_pp", lambda values: int((values > 0).sum())
            ),
        ).reset_index()
        summary.to_csv(output / "retraining_sensitivity_summary.csv", index=False)
    metadata = {
        "analysis_status": "exploratory_frozen_feature_retraining",
        "full_network_training_attempted": True,
        "full_network_training_completed": False,
        "full_network_training_executed": False,
        "linear_head_retraining_executed": True,
        "initial_probe_reused_from_cache": True,
        "validation_quality_affects": "RepairValue-Q scoring only",
        "retraining_checkpoint_selection_labels": "full_clean_validation",
        "design": design,
        "condition_count_per_seed": len(conditions),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "STATUS.md").write_text(
        "# RepairValue-Q validation sensitivity retraining\n\n"
        "This is exploratory frozen-feature linear-head retraining. The initial image probe "
        "is cached;\n"
        "each row performs actual query-label repair, linear-head retraining, and clean test "
        "evaluation.\n"
        "It is not a full-network ResNet-50 confirmation. A local full-network attempt was "
        "started but did not complete on the available 6 GiB GPU; no endpoint result from "
        "that attempt is included here.\n",
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--taus", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument(
        "--validation-fractions", nargs="+", type=float,
        default=[0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument(
        "--validation-noise-rates", nargs="+", type=float,
        default=[0.0, 0.05, 0.10, 0.20],
    )
    parser.add_argument(
        "--validation-noise-modes", nargs="+",
        choices=["symmetric", "class_conditional"], default=["symmetric"],
    )
    parser.add_argument("--design", choices=["one_factor", "full_factorial"], default="one_factor")
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--noise-name", default="uniform20")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": args.device,
    }
    frame = run_retraining_analysis(
        input_dir=args.input_dir,
        probe_dir=args.probe_dir,
        baseline_results=args.baseline_results,
        output_dir=args.output_dir,
        seeds=args.seeds,
        taus=args.taus,
        validation_fractions=args.validation_fractions,
        validation_noise_rates=args.validation_noise_rates,
        validation_noise_modes=args.validation_noise_modes,
        budget_fraction=args.budget_fraction,
        noise_name=args.noise_name,
        design=args.design,
        training_config=config,
    )
    if not frame.empty:
        print(frame.tail().to_string(index=False))
    print(f"Wrote frozen-feature retraining artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
