#!/usr/bin/env python
"""Compare RepairValue-Q with legal baselines under one frozen-head protocol.

This is an exploratory feasibility tool.  It reuses cached Waterbirds
features/probe dynamics, freezes every query ranking before reading private
clean labels, and retrains the same frozen linear head for each method.  The
baseline methods are retrained under exactly the same protocol as
RepairValue-Q, so their paired differences are not mixed across epoch counts
or cached references.

The tool deliberately does not modify any registered end-to-end protocol.  It
writes a standalone comparison package containing per-seed rows, deterministic
bootstrap summaries, and a protocol/input fingerprint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from robust_verify.absolute_utility import bootstrap_mean_ci, exact_two_sided_sign_test
from robust_verify.config import output_layout
from robust_verify.data.access import PrivateEvaluator
from robust_verify.end_to_end import _device as image_device
from robust_verify.scoring import descending_ranking
from robust_verify.training import LinearHead, make_initial_state, predict_probabilities, train_linear_head
from robust_verify.utils import budget_to_count, entropy_from_probabilities

try:
    from scripts.analyze_repairvalue_validation_sensitivity import (
        SeedData,
        condition_grid,
        load_seed_data,
        score_condition,
    )
    from scripts.run_waterbirds_repairvalue_exploration import (
        _private_frame,
        posthoc_query_diagnostics,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from analyze_repairvalue_validation_sensitivity import (
        SeedData,
        condition_grid,
        load_seed_data,
        score_condition,
    )
    from run_waterbirds_repairvalue_exploration import (
        _private_frame,
        posthoc_query_diagnostics,
    )


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO / "outputs/ablation_waterbirds_10seeds"
DEFAULT_PROBE_DIR = DEFAULT_INPUT_DIR / "stage1" / "probes"
DEFAULT_OUTPUT_DIR = REPO / "outputs/local_waterbirds_repairvalue_comparison"
DEFAULT_METHODS = ("loss", "noise_score", "entropy", "expected_repair_value")
REFERENCE_METHODS = ("loss", "noise_score", "entropy")
BOOTSTRAP_REPLICATES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_fingerprint(input_dir: Path, probe_dir: Path, seeds: Sequence[int], noise_names: Sequence[str]) -> dict[str, object]:
    root = output_layout(input_dir)
    paths: list[Path] = [root["features"] / "train.npz", root["features"] / "val.npz"]
    for noise_name in noise_names:
        for seed in seeds:
            paths.extend(
                [
                    root["manifests"] / f"{noise_name}_seed{seed}_train_public.csv",
                    root["manifests"] / f"{noise_name}_seed{seed}_train_private.csv",
                    probe_dir / f"{noise_name}_seed{seed}.dynamics.npz",
                ]
            )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(missing))
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO))
        except ValueError:
            return str(path)

    return {
        "files": {display_path(path): _sha256(path) for path in paths},
        "file_count": len(paths),
    }


def ranking_for_method(data: SeedData, method: str) -> np.ndarray:
    """Return a legal ranking using only the public SeedData fields."""
    if method == "loss":
        return descending_ranking(data.losses, data.losses)
    if method == "noise_score":
        return descending_ranking(data.noise_score, data.losses)
    if method == "entropy":
        entropy = entropy_from_probabilities(data.train_probabilities)
        return descending_ranking(entropy, data.losses)
    raise ValueError(f"ranking_for_method does not build adaptive method {method!r}")


def _predict_test(
    state: Mapping[str, torch.Tensor],
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


def _fit_and_evaluate(
    *,
    data: SeedData,
    input_dir: Path,
    labels: np.ndarray,
    seed: int,
    training_config: Mapping[str, object],
    initial_state: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    retrained = train_linear_head(
        train_features=data.train_features,
        train_labels=np.asarray(labels, dtype=np.int64),
        val_features=data.validation_features,
        val_labels=data.validation_labels,
        config=dict(training_config),
        seed=seed,
        initial_state=copy.deepcopy(dict(initial_state)),
        record_dynamics=False,
    )
    root = output_layout(input_dir)
    test_npz = np.load(root["features"] / "test.npz")
    test_features = np.asarray(test_npz["features"], dtype=np.float32)
    test_ids = np.asarray(test_npz["sample_ids"], dtype=np.int64)
    predictions = _predict_test(
        retrained.best_state,
        test_features,
        input_dim=data.train_features.shape[1],
        device=image_device(training_config),
        batch_size=int(training_config.get("batch_size", 4096)),
    )
    evaluator = PrivateEvaluator(root["manifests"] / "test_private.csv")
    metrics = evaluator.evaluate(test_ids, predictions)
    return {
        "average_accuracy": float(metrics["average_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "wga": float(metrics["wga"]),
        "best_epoch": int(retrained.best_epoch),
        "best_validation_metric": float(retrained.best_validation_metric),
    }


def _base_row(
    *,
    data: SeedData,
    private: pd.DataFrame,
    method: str,
    ranking: np.ndarray,
    budget_fraction: float,
    baseline_metrics: Mapping[str, float],
    metrics: Mapping[str, float],
) -> dict[str, object]:
    budget_count = budget_to_count(budget_fraction, len(data.train_labels))
    positions = np.asarray(ranking[:budget_count], dtype=np.int64)
    noisy = private["is_noisy"].to_numpy(dtype=bool)
    return {
        "dataset": "waterbirds",
        "noise_name": private.attrs.get("noise_name", ""),
        "seed": int(data.seed),
        "method": method,
        "condition_index": -1,
        "tail_fraction": np.nan,
        "validation_fraction": np.nan,
        "validation_noise_rate": np.nan,
        "validation_noise_mode": "reference",
        "budget_fraction": float(budget_fraction),
        "budget_count": int(budget_count),
        "query_noise_precision": float(noisy[positions].mean()) if len(positions) else np.nan,
        "query_noisy_count": int(noisy[positions].sum()),
        "query_jaccard_to_reference": np.nan,
        "average_accuracy": float(metrics["average_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "wga": float(metrics["wga"]),
        "baseline_wga": float(baseline_metrics["wga"]),
        "baseline_average_accuracy": float(baseline_metrics["average_accuracy"]),
        "delta_wga": float(metrics["wga"] - baseline_metrics["wga"]),
        "delta_average_accuracy": float(
            metrics["average_accuracy"] - baseline_metrics["average_accuracy"]
        ),
        "retraining_best_epoch": int(metrics["best_epoch"]),
        "retraining_best_validation_metric": float(metrics["best_validation_metric"]),
        "retraining_scope": "frozen_feature_linear_head",
    }


def _rv_row(
    *,
    data: SeedData,
    private: pd.DataFrame,
    score_row: Mapping[str, object],
    ranking: np.ndarray,
    budget_fraction: float,
    baseline_metrics: Mapping[str, float],
    metrics: Mapping[str, float],
    condition_index: int,
    tau: float,
    validation_fraction: float,
    validation_noise_rate: float,
    validation_noise_mode: str,
) -> dict[str, object]:
    row = _base_row(
        data=data,
        private=private,
        method="expected_repair_value",
        ranking=ranking,
        budget_fraction=budget_fraction,
        baseline_metrics=baseline_metrics,
        metrics=metrics,
    )
    row.update(
        {
            "condition_index": int(condition_index),
            "tail_fraction": float(tau),
            "validation_fraction": float(validation_fraction),
            "validation_noise_rate": float(validation_noise_rate),
            "validation_noise_mode": validation_noise_mode,
            "validation_size": int(score_row["validation_size"]),
            "validation_label_accuracy": float(score_row["validation_label_accuracy"]),
            "score_rank_correlation_to_noise": float(score_row["score_rank_correlation_to_noise"]),
            "active_tail_count": int(score_row["active_tail_count"]),
            "active_tail_count_min_by_class": int(score_row["active_tail_count_min_by_class"]),
            **posthoc_query_diagnostics(private, ranking, budget_fraction),
        }
    )
    return row


def _bootstrap_summary(values: Sequence[float], seed: int) -> tuple[float, float]:
    return bootstrap_mean_ci(
        np.asarray(values, dtype=np.float64),
        np.random.default_rng(int(seed)),
        BOOTSTRAP_REPLICATES,
    )


def summarize_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize RV-Q and its paired legal baseline contrasts."""
    rv = frame[frame["method"].eq("expected_repair_value")].copy()
    refs = frame[frame["method"].isin(REFERENCE_METHODS)].copy()
    if rv.empty:
        return pd.DataFrame()
    keys = [
        "noise_name",
        "budget_fraction",
        "tail_fraction",
        "validation_fraction",
        "validation_noise_rate",
        "validation_noise_mode",
    ]
    rows: list[dict[str, object]] = []
    for group_key, group in rv.groupby(keys, sort=True, dropna=False):
        group = group.sort_values("seed")
        row = dict(zip(keys, group_key))
        absolute = group["delta_wga"].to_numpy(dtype=float) * 100.0
        abs_ci = _bootstrap_summary(absolute, 101)
        row.update(
            {
                "n_seeds": int(group["seed"].nunique()),
                "mean_delta_wga_pp": float(absolute.mean()),
                "delta_wga_ci_low_pp": float(abs_ci[0]),
                "delta_wga_ci_high_pp": float(abs_ci[1]),
                "positive_absolute_seeds": int((absolute > 0).sum()),
                "mean_query_noise_precision_pct": float(
                    group["query_noise_precision"].mean() * 100.0
                ),
                "mean_posthoc_minority_query_rate_pct": float(
                    group["posthoc_minority_query_rate"].mean() * 100.0
                ),
            }
        )
        rv_by_seed = group.set_index("seed")
        for method_index, method in enumerate(REFERENCE_METHODS):
            ref = refs[
                refs["method"].eq(method)
                & refs["noise_name"].eq(group_key[0])
                & np.isclose(refs["budget_fraction"].astype(float), float(group_key[1]))
            ].set_index("seed")
            missing = sorted(set(rv_by_seed.index) - set(ref.index))
            if missing:
                raise ValueError(f"Missing {method} reference seeds: {missing}")
            paired = np.asarray(
                [
                    (float(rv_by_seed.loc[seed, "delta_wga"]) - float(ref.loc[seed, "delta_wga"]))
                    * 100.0
                    for seed in rv_by_seed.index
                ],
                dtype=np.float64,
            )
            ci = _bootstrap_summary(paired, 1001 + method_index)
            prefix = method.replace("_", "_")
            row.update(
                {
                    f"mean_vs_{prefix}_pp": float(paired.mean()),
                    f"vs_{prefix}_ci_low_pp": float(ci[0]),
                    f"vs_{prefix}_ci_high_pp": float(ci[1]),
                    f"positive_vs_{prefix}_seeds": int((paired > 0).sum()),
                    f"sign_test_vs_{prefix}_p": float(exact_two_sided_sign_test(paired)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_comparison(
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
    methods: Iterable[str],
    epochs: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    input_path = Path(input_dir).resolve()
    probe_path = Path(probe_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_values = [int(value) for value in seeds]
    noise_values = [str(value) for value in noise_names]
    method_values = tuple(dict.fromkeys(str(value) for value in methods))
    unknown = set(method_values) - set(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if "expected_repair_value" not in method_values:
        raise ValueError("The comparison package requires expected_repair_value")
    missing_references = sorted(set(REFERENCE_METHODS) - set(method_values))
    if missing_references:
        raise ValueError(
            "The comparison package requires all legal references: "
            + ", ".join(missing_references)
        )
    conditions = condition_grid(
        taus,
        validation_fractions,
        validation_noise_rates,
        validation_noise_modes,
        design=design,
    )
    config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": device,
    }
    private_root = output_layout(input_path)["manifests"]
    test_rows: list[dict[str, object]] = []
    partial_path = output / "per_seed_comparison.partial.csv"
    existing = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        (
            int(row.seed),
            str(row.noise_name),
            float(row.budget_fraction),
            str(row.method),
            int(row.condition_index),
        )
        for row in existing.itertuples()
    } if not existing.empty else set()

    for noise_name in noise_values:
        for seed in seed_values:
            data = load_seed_data(
                input_path,
                probe_path,
                seed=seed,
                noise_name=noise_name,
                validation_checkpoint_fallback=True,
            )
            private = _private_frame(input_path, noise_name, seed)
            private.attrs["noise_name"] = noise_name
            clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
            initial_state = make_initial_state(data.train_features.shape[1], 2, seed)
            for budget_fraction in [float(value) for value in budgets]:
                root = output_layout(input_path)
                baseline_metrics = _fit_and_evaluate(
                    data=data,
                    input_dir=input_path,
                    labels=data.train_labels,
                    seed=seed,
                    training_config=config,
                    initial_state=initial_state,
                )
                method_metrics: dict[str, dict[str, float]] = {}
                method_rankings: dict[str, np.ndarray] = {}
                for method in method_values:
                    if method == "expected_repair_value":
                        continue
                    ranking = ranking_for_method(data, method)
                    positions = ranking[: budget_to_count(budget_fraction, len(data.train_labels))]
                    repaired = data.train_labels.copy()
                    repaired[positions] = clean_labels[positions]
                    method_rankings[method] = ranking
                    method_metrics[method] = _fit_and_evaluate(
                        data=data,
                        input_dir=input_path,
                        labels=repaired,
                        seed=seed,
                        training_config=config,
                        initial_state=initial_state,
                    )
                    key = (seed, noise_name, budget_fraction, method, -1)
                    if key not in completed:
                        row = _base_row(
                            data=data,
                            private=private,
                            method=method,
                            ranking=ranking,
                            budget_fraction=budget_fraction,
                            baseline_metrics=baseline_metrics,
                            metrics=method_metrics[method],
                        )
                        rows.append(row)
                        completed.add(key)

                for condition_index, tau, fraction, rate, mode in conditions:
                    key = (seed, noise_name, budget_fraction, "expected_repair_value", int(condition_index))
                    if key in completed:
                        continue
                    score_row, ranking = score_condition(
                        data,
                        tau=tau,
                        validation_fraction=fraction,
                        validation_noise_rate=rate,
                        validation_noise_mode=mode,
                        subset_seed=seed * 1_000_003 + condition_index * 10_007,
                        corruption_seed=seed * 2_000_003 + condition_index * 10_009,
                        budget_fraction=budget_fraction,
                    )
                    positions = ranking[: budget_to_count(budget_fraction, len(data.train_labels))]
                    repaired = data.train_labels.copy()
                    repaired[positions] = clean_labels[positions]
                    metrics = _fit_and_evaluate(
                        data=data,
                        input_dir=input_path,
                        labels=repaired,
                        seed=seed,
                        training_config=config,
                        initial_state=initial_state,
                    )
                    rows.append(
                        _rv_row(
                            data=data,
                            private=private,
                            score_row=score_row,
                            ranking=ranking,
                            budget_fraction=budget_fraction,
                            baseline_metrics=baseline_metrics,
                            metrics=metrics,
                            condition_index=int(condition_index),
                            tau=tau,
                            validation_fraction=fraction,
                            validation_noise_rate=rate,
                            validation_noise_mode=mode,
                        )
                    )
                    completed.add(key)
                    pd.DataFrame(rows).to_csv(partial_path, index=False)
                    print(
                        f"noise={noise_name} seed={seed} budget={budget_fraction:.3f} "
                        f"condition={condition_index} RV-Q complete"
                    )

                pd.DataFrame(rows).to_csv(partial_path, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_comparison.csv", index=False)
    if partial_path.exists():
        partial_path.unlink()
    summary = summarize_comparison(frame)
    summary.to_csv(output / "comparison_summary.csv", index=False)
    fingerprint = _input_fingerprint(input_path, probe_path, seed_values, noise_values)
    metadata = {
        "analysis_status": "exploratory_waterbirds_rvq_frozen_feature_comparison",
        "dataset": "waterbirds",
        "methods": list(method_values),
        "reference_methods": list(REFERENCE_METHODS),
        "seeds": seed_values,
        "noise_names": noise_values,
        "budgets": [float(value) for value in budgets],
        "conditions": [
            {
                "condition_index": int(index),
                "tail_fraction": float(tau),
                "validation_fraction": float(fraction),
                "validation_noise_rate": float(rate),
                "validation_noise_mode": mode,
            }
            for index, tau, fraction, rate, mode in conditions
        ],
        "training": config,
        "retraining_scope": "frozen_feature_linear_head",
        "initial_probe_reused_from_cache": True,
        "full_network_training_executed": False,
        "private_fields_used_for": "post_query_label_repair_and_posthoc_diagnostics",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "input_fingerprint": fingerprint,
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "STATUS.md").write_text(
        "# Waterbirds RepairValue-Q frozen-feature comparison\n\n"
        "Exploratory same-protocol comparison of RepairValue-Q, Loss, NoiseScore, "
        "and optional Entropy. Query rankings are frozen before private clean labels "
        "are read. This package does not establish full-network transfer or replace "
        "registered end-to-end protocols.\n",
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--noise-names", nargs="+", default=["uniform20", "minority_high_40"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.20, 0.50])
    parser.add_argument("--validation-fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument("--validation-noise-rates", nargs="+", type=float, default=[0.0, 0.10])
    parser.add_argument("--validation-noise-modes", nargs="+", choices=["symmetric", "class_conditional"], default=["symmetric"])
    parser.add_argument("--design", choices=["one_factor", "full_factorial"], default="one_factor")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_comparison(
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
        methods=args.methods,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
