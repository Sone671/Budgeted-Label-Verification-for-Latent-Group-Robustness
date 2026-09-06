#!/usr/bin/env python
"""Explore the Waterbirds feasibility region for RepairValue-Q.

The default ``score`` mode reuses cached Waterbirds features and probe
dynamics. It varies noise type, query budget, tail fraction, validation-pool
size, and validation-label quality without running image training. The
optional ``retrain`` mode performs frozen-feature linear-head retraining for
the same conditions and evaluates the clean private test set.

Group and clean-label fields are read only after a query ranking is frozen;
they are emitted as post-hoc diagnostics and never enter the score itself.
Full-network end-to-end training is deliberately outside this entry point.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scripts.analyze_repairvalue_validation_sensitivity import (
        condition_grid,
        load_seed_data,
        score_condition,
    )
    from scripts.run_repairvalue_validation_retraining import (
        _predict_test,
        retrain_condition,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from analyze_repairvalue_validation_sensitivity import (
        condition_grid,
        load_seed_data,
        score_condition,
    )
    from run_repairvalue_validation_retraining import _predict_test, retrain_condition
from robust_verify.config import output_layout
from robust_verify.data.access import PrivateEvaluator
from robust_verify.end_to_end import _device as image_device
from robust_verify.scoring import descending_ranking
from robust_verify.training import make_initial_state, train_linear_head
from robust_verify.utils import budget_to_count


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO / "outputs/ablation_waterbirds_10seeds"
DEFAULT_PROBE_DIR = DEFAULT_INPUT_DIR / "stage1" / "probes"
DEFAULT_BASELINE_RESULTS = DEFAULT_INPUT_DIR / "stage1" / "results.csv"
DEFAULT_OUTPUT_DIR = REPO / "outputs/local_waterbirds_repairvalue_exploration"


def _private_frame(input_dir: str | Path, noise_name: str, seed: int) -> pd.DataFrame:
    manifest = (
        output_layout(input_dir)["manifests"]
        / f"{noise_name}_seed{int(seed)}_train_private.csv"
    )
    frame = pd.read_csv(manifest).sort_values("sample_id").reset_index(drop=True)
    required = {"sample_id", "is_noisy", "is_minority", "group", "clean_label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Waterbirds private train manifest lacks columns: {missing}")
    return frame


def _entropy(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def posthoc_query_diagnostics(
    private: pd.DataFrame,
    ranking: np.ndarray,
    budget_fraction: float,
) -> dict[str, float]:
    """Compute group/noise composition after the query set is frozen."""
    budget_count = budget_to_count(budget_fraction, len(private))
    positions = np.asarray(ranking[:budget_count], dtype=np.int64)
    noisy = private["is_noisy"].to_numpy(dtype=bool)
    minority = private["is_minority"].to_numpy(dtype=bool)
    groups = private["group"].to_numpy(dtype=np.int64)
    query_noisy_minority = int(np.count_nonzero(noisy[positions] & minority[positions]))
    total_noisy_minority = int(np.count_nonzero(noisy & minority))
    group_counts = np.bincount(groups[positions], minlength=int(groups.max()) + 1)
    return {
        "posthoc_minority_query_rate": float(minority[positions].mean())
        if len(positions)
        else float("nan"),
        "posthoc_noisy_minority_recall": (
            query_noisy_minority / total_noisy_minority
            if total_noisy_minority
            else float("nan")
        ),
        "posthoc_query_group_entropy": _entropy(group_counts),
        "posthoc_query_group_count": int(len(np.unique(groups[positions]))),
    }


def _same_protocol_references(
    *,
    data,
    private: pd.DataFrame,
    input_dir: str | Path,
    seed: int,
    budget_fraction: float,
    training_config: dict[str, object],
) -> dict[str, float]:
    """Train no-correction and Loss references under the retraining protocol."""
    root = output_layout(input_dir)
    test_npz = np.load(root["features"] / "test.npz")
    test_features = np.asarray(test_npz["features"], dtype=np.float32)
    test_ids = np.asarray(test_npz["sample_ids"], dtype=np.int64)
    evaluator = PrivateEvaluator(root["manifests"] / "test_private.csv")
    device = image_device(training_config)
    initial_state = make_initial_state(data.train_features.shape[1], 2, seed)

    def fit_and_evaluate(labels: np.ndarray) -> dict[str, float]:
        result = train_linear_head(
            train_features=data.train_features,
            train_labels=labels,
            val_features=data.validation_features,
            val_labels=data.validation_labels,
            config=training_config,
            seed=seed,
            initial_state=copy.deepcopy(initial_state),
            record_dynamics=False,
        )
        predictions = _predict_test(
            result.best_state,
            test_features,
            input_dim=data.train_features.shape[1],
            device=device,
            batch_size=int(training_config.get("batch_size", 4096)),
        )
        return evaluator.evaluate(test_ids, predictions)

    baseline_metrics = fit_and_evaluate(data.train_labels)
    loss_positions = descending_ranking(data.losses, data.losses)[
        : budget_to_count(budget_fraction, len(data.train_labels))
    ]
    loss_labels = data.train_labels.copy()
    clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
    loss_labels[loss_positions] = clean_labels[loss_positions]
    loss_metrics = fit_and_evaluate(loss_labels)
    return {
        "baseline_wga": float(baseline_metrics["wga"]),
        "baseline_average_accuracy": float(baseline_metrics["average_accuracy"]),
        "loss_delta_wga": float(loss_metrics["wga"] - baseline_metrics["wga"]),
        "loss_delta_average_accuracy": float(
            loss_metrics["average_accuracy"] - baseline_metrics["average_accuracy"]
        ),
    }


def _score_rows(
    *,
    input_dir: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    seeds: Iterable[int],
    noise_names: Iterable[str],
    budgets: Iterable[float],
    taus: Iterable[float],
    validation_fractions: Iterable[float],
    validation_noise_rates: Iterable[float],
    validation_noise_modes: Iterable[str],
    design: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    conditions = condition_grid(
        taus,
        validation_fractions,
        validation_noise_rates,
        validation_noise_modes,
        design=design,
    )
    for noise_name in noise_names:
        for seed in [int(value) for value in seeds]:
            data = load_seed_data(input_dir, probe_dir, seed=seed, noise_name=noise_name)
            private = _private_frame(input_dir, noise_name, seed)
            if not np.array_equal(
                private["sample_id"].to_numpy(dtype=np.int64), data.train_sample_ids
            ):
                raise ValueError(f"Private train IDs do not align for {noise_name}, seed {seed}")
            for budget in [float(value) for value in budgets]:
                reference_row, reference_ranking = score_condition(
                    data,
                    tau=0.20,
                    validation_fraction=1.0,
                    validation_noise_rate=0.0,
                    validation_noise_mode="symmetric",
                    subset_seed=seed + 100_000,
                    corruption_seed=seed + 200_000,
                    budget_fraction=budget,
                )
                reference_count = int(reference_row["budget_count"])
                reference_positions = reference_ranking[:reference_count]
                for condition_index, tau, fraction, rate, mode in conditions:
                    subset_seed = seed * 1_000_003 + condition_index * 10_007
                    corruption_seed = seed * 2_000_003 + condition_index * 10_009
                    row, ranking = score_condition(
                        data,
                        tau=tau,
                        validation_fraction=fraction,
                        validation_noise_rate=rate,
                        validation_noise_mode=mode,
                        subset_seed=subset_seed,
                        corruption_seed=corruption_seed,
                        budget_fraction=budget,
                        reference_query_positions=reference_positions,
                    )
                    row.update(
                        {
                            "dataset": "waterbirds",
                            "noise_name": noise_name,
                            "condition_index": int(condition_index),
                            **posthoc_query_diagnostics(private, ranking, budget),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def _retrain_rows(
    *,
    input_dir: str | Path,
    probe_dir: str | Path,
    baseline_results: str | Path,
    seeds: Iterable[int],
    noise_names: Iterable[str],
    budgets: Iterable[float],
    taus: Iterable[float],
    validation_fractions: Iterable[float],
    validation_noise_rates: Iterable[float],
    validation_noise_modes: Iterable[str],
    design: str,
    epochs: int,
    batch_size: int,
    device: str,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Run selected frozen-feature retraining conditions with resumable outputs."""
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
        (int(row.seed), str(row.noise_name), float(row.budget_fraction), int(row.condition_index))
        for row in existing.itertuples()
    } if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": device,
    }
    for noise_name in noise_names:
        for seed in [int(value) for value in seeds]:
            data = load_seed_data(
                input_dir,
                probe_dir,
                seed=seed,
                noise_name=noise_name,
                validation_checkpoint_fallback=True,
            )
            private = _private_frame(input_dir, noise_name, seed)
            reference_cache: dict[float, dict[str, float]] = {}
            for budget in [float(value) for value in budgets]:
                if budget not in reference_cache:
                    reference_cache[budget] = _same_protocol_references(
                        data=data,
                        private=private,
                        input_dir=input_dir,
                        seed=seed,
                        budget_fraction=budget,
                        training_config=config,
                    )
                for condition_index, tau, fraction, rate, mode in conditions:
                    key = (seed, noise_name, budget, int(condition_index))
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
                        budget_fraction=budget,
                        noise_name=noise_name,
                        training_config=config,
                        validation_checkpoint_fallback=True,
                        baseline_metrics=reference_cache[budget],
                    )
                    _, ranking = score_condition(
                        data,
                        tau=tau,
                        validation_fraction=fraction,
                        validation_noise_rate=rate,
                        validation_noise_mode=mode,
                        subset_seed=seed * 1_000_003 + condition_index * 10_007,
                        corruption_seed=seed * 2_000_003 + condition_index * 10_009,
                        budget_fraction=budget,
                    )
                    row.update(
                        {
                            "dataset": "waterbirds",
                            "noise_name": noise_name,
                            "condition_index": int(condition_index),
                            **posthoc_query_diagnostics(private, ranking, budget),
                        }
                    )
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(partial_path, index=False)
                    print(
                        f"noise={noise_name} seed={seed} budget={budget:.3f} "
                        f"condition={condition_index} complete"
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_retraining.csv", index=False)
    if partial_path.exists():
        partial_path.unlink()
    if not frame.empty:
        summary_keys = [
            "noise_name",
            "budget_fraction",
            "tail_fraction",
            "validation_fraction",
            "validation_noise_rate",
            "validation_noise_mode",
        ]
        summary = frame.groupby(summary_keys, sort=True).agg(
            n_seeds=("seed", "nunique"),
            mean_delta_wga_pp=("delta_wga", lambda values: 100.0 * values.mean()),
            mean_rv_minus_loss_wga_pp=("rv_minus_loss_wga_pp", "mean"),
            mean_query_noise_precision_pct=(
                "query_noise_precision", lambda values: 100.0 * values.mean()
            ),
            mean_posthoc_minority_query_rate_pct=(
                "posthoc_minority_query_rate", lambda values: 100.0 * values.mean()
            ),
            mean_posthoc_noisy_minority_recall_pct=(
                "posthoc_noisy_minority_recall", lambda values: 100.0 * values.mean()
            ),
            positive_rv_minus_loss_seeds=(
                "rv_minus_loss_wga_pp", lambda values: int((values > 0).sum())
            ),
        ).reset_index()
        summary.to_csv(output / "retraining_sensitivity_summary.csv", index=False)
    return frame


def _write_score_artifacts(frame: pd.DataFrame, output_dir: str | Path, design: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_scores.csv", index=False)
    if not frame.empty:
        keys = [
            "noise_name",
            "budget_fraction",
            "tail_fraction",
            "validation_fraction",
            "validation_noise_rate",
            "validation_noise_mode",
        ]
        summary = frame.groupby(keys, sort=True).agg(
            n_seeds=("seed", "nunique"),
            mean_validation_size=("validation_size", "mean"),
            mean_validation_label_accuracy_pct=(
                "validation_label_accuracy", lambda values: 100.0 * values.mean()
            ),
            mean_query_noise_precision_pct=(
                "query_noise_precision", lambda values: 100.0 * values.mean()
            ),
            mean_posthoc_minority_query_rate_pct=(
                "posthoc_minority_query_rate", lambda values: 100.0 * values.mean()
            ),
            mean_posthoc_noisy_minority_recall_pct=(
                "posthoc_noisy_minority_recall", lambda values: 100.0 * values.mean()
            ),
            mean_query_group_entropy=("posthoc_query_group_entropy", "mean"),
        ).reset_index()
        summary.to_csv(output / "score_sensitivity_summary.csv", index=False)
    metadata = {
        "analysis_status": "exploratory_waterbirds_repairvalue",
        "dataset": "waterbirds",
        "design": design,
        "training_executed": False,
        "full_network_training_executed": False,
        "group_labels_used_for": "posthoc_query_diagnostics_only",
        "clean_training_labels_used_for": "offline_query_precision_only",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "STATUS.md").write_text(
        "# Waterbirds RepairValue-Q feasibility exploration\n\n"
        "Score-only scan over cached frozen features and probe dynamics. Training and "
        "full-network end-to-end fitting are not executed in this output. Group and "
        "clean-label fields are post-hoc diagnostics after query freezing.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["score", "retrain"], default="score")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--noise-names", nargs="+", default=["uniform20", "minority_high_40"]
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--validation-fractions", nargs="+", type=float, default=[0.25, 0.50, 1.0])
    parser.add_argument(
        "--validation-noise-rates", nargs="+", type=float,
        default=[0.0, 0.05, 0.10],
    )
    parser.add_argument(
        "--validation-noise-modes", nargs="+",
        choices=["symmetric", "class_conditional"], default=["symmetric"],
    )
    parser.add_argument("--design", choices=["one_factor", "full_factorial"], default="one_factor")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.mode == "score":
        frame = _score_rows(
            input_dir=args.input_dir,
            probe_dir=args.probe_dir,
            output_dir=args.output_dir,
            seeds=args.seeds,
            noise_names=args.noise_names,
            budgets=args.budgets,
            taus=args.taus,
            validation_fractions=args.validation_fractions,
            validation_noise_rates=args.validation_noise_rates,
            validation_noise_modes=args.validation_noise_modes,
            design=args.design,
        )
        _write_score_artifacts(frame, args.output_dir, args.design)
    else:
        frame = _retrain_rows(
            input_dir=args.input_dir,
            probe_dir=args.probe_dir,
            baseline_results=args.baseline_results,
            seeds=args.seeds,
            noise_names=args.noise_names,
            budgets=args.budgets,
            taus=args.taus,
            validation_fractions=args.validation_fractions,
            validation_noise_rates=args.validation_noise_rates,
            validation_noise_modes=args.validation_noise_modes,
            design=args.design,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )
        (Path(args.output_dir) / "metadata.json").write_text(
            json.dumps(
                {
                    "analysis_status": "exploratory_waterbirds_frozen_feature_retraining",
                    "dataset": "waterbirds",
                    "design": args.design,
                    "training_executed": True,
                    "full_network_training_executed": False,
                    "linear_head_retraining_executed": True,
                    "reference_protocol": "same_frozen_feature_linear_head",
                    "initial_probe_reused_from_cache": True,
                    "group_labels_used_for": "posthoc_query_diagnostics_only",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (Path(args.output_dir) / "STATUS.md").write_text(
            "# Waterbirds RepairValue-Q feasibility retraining\n\n"
            "Exploratory frozen-feature linear-head retraining. The initial image probe "
            "is cached; full-network end-to-end training is not executed. Group and "
            "clean-label fields are post-hoc diagnostics after query freezing.\n",
            encoding="utf-8",
        )
    print(f"Wrote Waterbirds exploration artifacts to {args.output_dir}")
    if not frame.empty:
        print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
