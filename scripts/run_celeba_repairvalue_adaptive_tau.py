#!/usr/bin/env python
"""Evaluate validation-only adaptive-tau RepairValue-Q on CelebA.

This exploratory runner reuses the cached CelebA features and probe dynamics,
selects tau before private labels are read, and retrains a frozen linear head
under the same protocol used by the Waterbirds capture-tau experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from robust_verify.absolute_utility import bootstrap_mean_ci, exact_two_sided_sign_test
from robust_verify.config import output_layout
from robust_verify.data.access import PrivateEvaluator
from robust_verify.end_to_end import _device as image_device
from robust_verify.modern_baselines import (
    expected_repair_value_scores,
    select_tail_fraction_by_capture,
)
from robust_verify.scoring import descending_ranking
from robust_verify.training import LinearHead, make_initial_state, predict_probabilities, train_linear_head
from robust_verify.utils import budget_to_count

try:
    from scripts.analyze_repairvalue_validation_sensitivity import SeedData, load_seed_data
    from scripts.run_waterbirds_repairvalue_exploration import posthoc_query_diagnostics
except ModuleNotFoundError:
    from analyze_repairvalue_validation_sensitivity import SeedData, load_seed_data
    from run_waterbirds_repairvalue_exploration import posthoc_query_diagnostics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO / "outputs/ablation_celeba_10seeds"
DEFAULT_PROBE_DIR = DEFAULT_INPUT_DIR / "stage1" / "probes"
DEFAULT_OUTPUT_DIR = REPO / "outputs/local_celeba_repairvalue_capture_tau"
CANDIDATE_TAUS = (0.05, 0.10, 0.20, 0.30, 0.50)
BOOTSTRAP_REPLICATES = 10_000


def _private_frame(input_dir: Path, noise_name: str, seed: int) -> pd.DataFrame:
    path = output_layout(input_dir)["manifests"] / f"{noise_name}_seed{seed}_train_private.csv"
    frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
    required = {"sample_id", "clean_label", "is_noisy", "is_minority", "group"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"CelebA private train manifest lacks columns: {missing}")
    return frame


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
    return predict_probabilities(model, test_features, batch_size, device).argmax(axis=1).astype(np.int64)


def _fit_and_evaluate(
    *,
    data: SeedData,
    input_dir: Path,
    labels: np.ndarray,
    seed: int,
    training_config: Mapping[str, object],
    initial_state: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    result = train_linear_head(
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
        result.best_state,
        test_features,
        input_dim=data.train_features.shape[1],
        device=image_device(training_config),
        batch_size=int(training_config.get("batch_size", 4096)),
    )
    metrics = PrivateEvaluator(root["manifests"] / "test_private.csv").evaluate(test_ids, predictions)
    return {
        "average_accuracy": float(metrics["average_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "wga": float(metrics["wga"]),
        "best_epoch": int(result.best_epoch),
        "best_validation_metric": float(result.best_validation_metric),
    }


def _ranking(data: SeedData, tau: float | None = None, method: str = "loss") -> np.ndarray:
    if method == "loss":
        return descending_ranking(data.losses, data.losses)
    if method == "noise_score":
        return descending_ranking(data.noise_score, data.losses)
    if method == "rvq":
        scores = expected_repair_value_scores(
            train_features=data.train_features,
            train_labels=data.train_labels,
            val_features=data.validation_features,
            val_probabilities=data.validation_probabilities,
            val_labels=data.validation_labels,
            noise_posterior_proxy=data.noise_score,
            tail_fraction=float(tau),
        )
        return descending_ranking(scores, data.losses)
    raise ValueError(f"Unknown ranking method: {method}")


def _row(
    *,
    data: SeedData,
    private: pd.DataFrame,
    method: str,
    ranking: np.ndarray,
    budget_fraction: float,
    baseline: Mapping[str, float],
    metrics: Mapping[str, float],
    selected_tau: float | None = None,
    selection_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    count = budget_to_count(budget_fraction, len(data.train_labels))
    positions = np.asarray(ranking[:count], dtype=np.int64)
    noisy = private["is_noisy"].to_numpy(dtype=bool)
    row: dict[str, object] = {
        "dataset": "celeba",
        "noise_name": private.attrs["noise_name"],
        "seed": int(data.seed),
        "method": method,
        "budget_fraction": float(budget_fraction),
        "budget_count": int(count),
        "selected_tau": float(selected_tau) if selected_tau is not None else np.nan,
        "query_noise_precision": float(noisy[positions].mean()),
        "wga": float(metrics["wga"]),
        "baseline_wga": float(baseline["wga"]),
        "delta_wga": float(metrics["wga"] - baseline["wga"]),
        "average_accuracy": float(metrics["average_accuracy"]),
        "baseline_average_accuracy": float(baseline["average_accuracy"]),
        "delta_average_accuracy": float(metrics["average_accuracy"] - baseline["average_accuracy"]),
        "best_epoch": int(metrics["best_epoch"]),
        "best_validation_metric": float(metrics["best_validation_metric"]),
        "retraining_scope": "frozen_feature_linear_head",
        "selection_metadata_json": json.dumps(selection_metadata, sort_keys=True) if selection_metadata else "",
    }
    row.update(posthoc_query_diagnostics(private, ranking, budget_fraction))
    return row


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (noise, budget), group in frame.groupby(["noise_name", "budget_fraction"], sort=True):
        group = group.sort_values("seed")
        adaptive = group[group["method"].eq("adaptive_capture")].set_index("seed")
        baseline = group[group["method"].eq("baseline")].set_index("seed")
        if adaptive.empty:
            continue
        row: dict[str, object] = {
            "noise_name": noise,
            "budget_fraction": float(budget),
            "seeds": int(adaptive.index.nunique()),
            "selected_tau_mean": float(adaptive["selected_tau"].mean()),
            "selected_tau_min": float(adaptive["selected_tau"].min()),
            "selected_tau_max": float(adaptive["selected_tau"].max()),
            "query_precision_mean": float(adaptive["query_noise_precision"].mean() * 100.0),
            "absolute_delta_wga_pp": float(adaptive["delta_wga"].mean() * 100.0),
        }
        rng = np.random.default_rng(20260827)
        absolute = adaptive["delta_wga"].to_numpy(dtype=float) * 100.0
        row["absolute_ci_low_pp"], row["absolute_ci_high_pp"] = bootstrap_mean_ci(
            absolute, rng, BOOTSTRAP_REPLICATES
        )
        for method, label in (
            ("loss", "loss"),
            ("noise_score", "noise_score"),
            ("rvq_tau02", "rvq_tau02"),
            ("rvq_tau05", "rvq_tau05"),
        ):
            ref = group[group["method"].eq(method)].set_index("seed")
            if ref.empty:
                continue
            values = (adaptive["delta_wga"] - ref["delta_wga"]).to_numpy(dtype=float) * 100.0
            low, high = bootstrap_mean_ci(values, rng, BOOTSTRAP_REPLICATES)
            row[f"paired_vs_{label}_pp"] = float(values.mean())
            row[f"paired_vs_{label}_ci_low_pp"] = float(low)
            row[f"paired_vs_{label}_ci_high_pp"] = float(high)
            row[f"paired_vs_{label}_positive_seeds"] = int((values > 0).sum())
            row[f"paired_vs_{label}_sign_test_p"] = float(exact_two_sided_sign_test(values))
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    *,
    input_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    noise_names: Sequence[str],
    budgets: Sequence[float],
    epochs: int,
    batch_size: int,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": device,
    }
    rows: list[dict[str, object]] = []
    for noise_name in noise_names:
        for seed in seeds:
            print(f"[{noise_name}] seed={seed}: loading", flush=True)
            data = load_seed_data(
                input_dir,
                probe_dir,
                seed=int(seed),
                noise_name=noise_name,
                validation_checkpoint_fallback=True,
            )
            private = _private_frame(input_dir, noise_name, int(seed))
            private.attrs["noise_name"] = noise_name
            clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
            if len(clean_labels) != len(data.train_labels):
                raise ValueError("private train manifest and feature cache have different lengths")
            initial_state = make_initial_state(data.train_features.shape[1], 2, int(seed))
            for budget in budgets:
                budget = float(budget)
                count = budget_to_count(budget, len(data.train_labels))
                adaptive_tau, adaptive_meta = select_tail_fraction_by_capture(
                    train_features=data.train_features,
                    train_labels=data.train_labels,
                    val_features=data.validation_features,
                    val_probabilities=data.validation_probabilities,
                    val_labels=data.validation_labels,
                    noise_posterior_proxy=data.noise_score,
                    budget_count=count,
                    candidate_tail_fractions=CANDIDATE_TAUS,
                    num_folds=5,
                    min_tail_count=4,
                    stability_sample_size=4096,
                    seed=int(seed),
                )
                rankings = {
                    "loss": _ranking(data, method="loss"),
                    "noise_score": _ranking(data, method="noise_score"),
                    "rvq_tau02": _ranking(data, tau=0.20, method="rvq"),
                    "rvq_tau05": _ranking(data, tau=0.50, method="rvq"),
                    "adaptive_capture": _ranking(data, tau=adaptive_tau, method="rvq"),
                }
                print(
                    f"[{noise_name}] seed={seed} budget={budget:.2%}: selected tau={adaptive_tau:.2f}",
                    flush=True,
                )
                baseline = _fit_and_evaluate(
                    data=data,
                    input_dir=input_dir,
                    labels=data.train_labels,
                    seed=int(seed),
                    training_config=config,
                    initial_state=initial_state,
                )
                rows.append(
                    _row(
                        data=data,
                        private=private,
                        method="baseline",
                        ranking=np.arange(len(data.train_labels)),
                        budget_fraction=budget,
                        baseline=baseline,
                        metrics=baseline,
                    )
                )
                for method in ("loss", "noise_score", "rvq_tau02", "rvq_tau05", "adaptive_capture"):
                    labels = data.train_labels.copy()
                    positions = rankings[method][:count]
                    labels[positions] = clean_labels[positions]
                    metrics = _fit_and_evaluate(
                        data=data,
                        input_dir=input_dir,
                        labels=labels,
                        seed=int(seed),
                        training_config=config,
                        initial_state=initial_state,
                    )
                    rows.append(
                        _row(
                            data=data,
                            private=private,
                            method=method,
                            ranking=rankings[method],
                            budget_fraction=budget,
                            baseline=baseline,
                            metrics=metrics,
                            selected_tau=adaptive_tau if method == "adaptive_capture" else None,
                            selection_metadata=adaptive_meta if method == "adaptive_capture" else None,
                        )
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_seed_comparison.csv", index=False)
    _summary(frame).to_csv(output_dir / "comparison_summary.csv", index=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": "celeba",
                "input_dir": str(input_dir.resolve()),
                "probe_dir": str(probe_dir.resolve()),
                "seeds": [int(value) for value in seeds],
                "noise_names": list(noise_names),
                "budgets": [float(value) for value in budgets],
                "candidate_taus": list(CANDIDATE_TAUS),
                "selection_rule": "out_of_sample_capture",
                "training": config,
                "retraining_scope": "frozen_feature_linear_head",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'per_seed_comparison.csv'}", flush=True)
    print(f"Wrote {output_dir / 'comparison_summary.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise-names", nargs="+", default=["uniform20", "minority_high_40"])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(
        input_dir=args.input_dir,
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        noise_names=args.noise_names,
        budgets=args.budgets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
