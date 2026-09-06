#!/usr/bin/env python
"""Evaluate RepairValue-Q with a validation-only adaptive tail fraction.

Standalone exploratory block, separate from the registered 20-seed comparison
package.  It reuses that package's cached Waterbirds features and probe
dynamics, selects the tail fraction tau per seed using only legal pre-query
observables (train features and noisy labels, validation features,
probabilities, and clean labels), retrains the same frozen linear head, and
pairs the result against the stored reference rows (Loss, NoiseScore, and
RepairValue-Q at fixed tau) from the existing comparison package.

The adaptive rule is ``select_adaptive_tail_fraction``: five stratified
validation folds, per-candidate ranking stability on a sampled training
subset, and a class-conditional tail-loss contrast, combined as
0.8 * stability + 0.2 * contrast rank.  No clean training labels, group
labels, or test outcomes enter the selection.

Protocol notes:
- tau is selected once per (noise, seed) and reused across budgets, mirroring
  a practitioner who fixes tau before querying;
- retraining shares the comparison package's per-seed initial state, epochs,
  batch size, learning rate, weight decay, patience, and balanced-accuracy
  checkpoint selection, so paired differences are not mixed across protocols;
- every recomputed no-correction baseline WGA is checked against the stored
  package value, and one fixed-tau cell is replayed end to end to verify
  protocol equivalence before any adaptive row is trusted.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from robust_verify.absolute_utility import (
    bootstrap_mean_ci,
    exact_two_sided_sign_test,
)
from robust_verify.modern_baselines import (
    expected_repair_value_scores,
    robust_set_repair_value_ranking,
    select_adaptive_tail_fraction,
    select_tail_fraction_by_calibrated_capture,
    select_tail_fraction_by_capture,
)
from robust_verify.scoring import descending_ranking
from robust_verify.training import make_initial_state
from robust_verify.utils import budget_to_count

try:
    from scripts.analyze_repairvalue_validation_sensitivity import (
        SeedData,
        load_seed_data,
    )
    from scripts.run_waterbirds_repairvalue_comparison import (
        _fit_and_evaluate,
        _private_frame,
    )
    from scripts.run_waterbirds_repairvalue_exploration import (
        posthoc_query_diagnostics,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from analyze_repairvalue_validation_sensitivity import (
        SeedData,
        load_seed_data,
    )
    from run_waterbirds_repairvalue_comparison import (
        _fit_and_evaluate,
        _private_frame,
    )
    from run_waterbirds_repairvalue_exploration import (
        posthoc_query_diagnostics,
    )

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO / "outputs/ablation_waterbirds_10seeds"
DEFAULT_PROBE_DIR = DEFAULT_INPUT_DIR / "stage1" / "probes"
DEFAULT_REFERENCE = (
    REPO / "outputs/local_waterbirds_repairvalue_comparison_20seeds"
    / "per_seed_comparison.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO / "outputs/local_waterbirds_repairvalue_adaptive_tau"
)
BOOTSTRAP_REPLICATES = 10_000
CANDIDATE_TAIL_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
BASELINE_TOLERANCE = 1e-9


def _reference_lookup(
    frame: pd.DataFrame,
) -> dict[tuple[str, int, float, str, int], dict[str, float]]:
    """Index stored per-seed rows by (noise, seed, budget, method, condition)."""
    lookup: dict[tuple[str, int, float, str, int], dict[str, float]] = {}
    for row in frame.itertuples():
        key = (
            str(row.noise_name),
            int(row.seed),
            float(row.budget_fraction),
            str(row.method),
            int(row.condition_index),
        )
        record = {
            "delta_wga": float(row.delta_wga),
            "baseline_wga": float(row.baseline_wga),
            "wga": float(row.wga),
            "query_noise_precision": float(row.query_noise_precision),
        }
        if key in lookup and lookup[key] != record:
            raise ValueError(f"reference package has duplicate rows for {key}")
        lookup[key] = record
    return lookup


def _rvq_ranking(data: SeedData, tau: float) -> np.ndarray:
    """Rank by the fixed-tau RepairValue-Q product score, ties by loss."""
    scores = expected_repair_value_scores(
        train_features=data.train_features,
        train_labels=data.train_labels,
        val_features=data.validation_features,
        val_probabilities=data.validation_probabilities,
        val_labels=data.validation_labels,
        noise_posterior_proxy=data.noise_score,
        tail_fraction=tau,
    )
    return descending_ranking(scores, data.losses)


def _summarize(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Mean effects with seed-level bootstrap intervals and sign tests."""
    loss = reference[reference["method"].eq("loss")][
        ["noise_name", "seed", "budget_fraction", "delta_wga"]
    ].rename(columns={"delta_wga": "loss_delta_wga"})
    noise = reference[reference["method"].eq("noise_score")][
        ["noise_name", "seed", "budget_fraction", "delta_wga"]
    ].rename(columns={"delta_wga": "noise_score_delta_wga"})
    tau02 = reference[
        reference["method"].eq("expected_repair_value")
        & reference["condition_index"].eq(0)
    ][["noise_name", "seed", "budget_fraction", "delta_wga"]].rename(
        columns={"delta_wga": "rvq_tau02_delta_wga"}
    )
    tau05 = reference[
        reference["method"].eq("expected_repair_value")
        & reference["condition_index"].eq(1)
    ][["noise_name", "seed", "budget_fraction", "delta_wga"]].rename(
        columns={"delta_wga": "rvq_tau05_delta_wga"}
    )
    merged = (
        frame.merge(loss, on=["noise_name", "seed", "budget_fraction"], how="left")
        .merge(noise, on=["noise_name", "seed", "budget_fraction"], how="left")
        .merge(tau02, on=["noise_name", "seed", "budget_fraction"], how="left")
        .merge(tau05, on=["noise_name", "seed", "budget_fraction"], how="left")
    )
    rng = np.random.default_rng(20260827)
    rows: list[dict[str, object]] = []
    for (noise_name, budget), group in merged.groupby(
        ["noise_name", "budget_fraction"], sort=True
    ):
        seeds = len(group)
        absolute = group["delta_wga"].to_numpy(dtype=np.float64) * 100.0
        def paired(column: str) -> np.ndarray:
            return (
                group["delta_wga"].to_numpy(dtype=np.float64)
                - group[column].to_numpy(dtype=np.float64)
            ) * 100.0

        entry: dict[str, object] = {
            "noise_name": noise_name,
            "budget_fraction": float(budget),
            "seeds": int(seeds),
            "selected_tau_mean": float(group["selected_tau"].mean()),
            "selected_tau_min": float(group["selected_tau"].min()),
            "selected_tau_max": float(group["selected_tau"].max()),
            "query_precision_mean": float(
                group["query_noise_precision"].mean() * 100.0
            ),
            "absolute_delta_wga_pp": float(absolute.mean()),
            "absolute_ci_low_pp": bootstrap_mean_ci(absolute, rng, BOOTSTRAP_REPLICATES)[0],
            "absolute_ci_high_pp": bootstrap_mean_ci(absolute, rng, BOOTSTRAP_REPLICATES)[1],
        }
        for label, column in (
            ("loss", "loss_delta_wga"),
            ("noise_score", "noise_score_delta_wga"),
            ("rvq_tau02", "rvq_tau02_delta_wga"),
            ("rvq_tau05", "rvq_tau05_delta_wga"),
        ):
            values = paired(column)
            low, high = bootstrap_mean_ci(values, rng, BOOTSTRAP_REPLICATES)
            positives = int(np.sum(values > 0))
            ties = int(np.sum(values == 0))
            entry[f"paired_vs_{label}_pp"] = float(values.mean())
            entry[f"paired_vs_{label}_ci_low_pp"] = low
            entry[f"paired_vs_{label}_ci_high_pp"] = high
            entry[f"paired_vs_{label}_positive_seeds"] = positives
            entry[f"paired_vs_{label}_tie_seeds"] = ties
            entry[f"paired_vs_{label}_sign_test_p"] = exact_two_sided_sign_test(values)
        rows.append(entry)
    return pd.DataFrame(rows)


def run_evaluation(
    *,
    input_dir: Path,
    probe_dir: Path,
    reference_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    noise_names: Sequence[str],
    budgets: Sequence[float],
    epochs: int,
    batch_size: int,
    device: str,
    verify_cell: bool,
    rule: str,
) -> None:
    reference = pd.read_csv(reference_path)
    lookup = _reference_lookup(reference)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": device,
    }

    rows: list[dict[str, object]] = []
    baseline_deviation = 0.0
    for noise_name in noise_names:
        for seed in seeds:
            data = load_seed_data(
                input_dir,
                probe_dir,
                seed=seed,
                noise_name=noise_name,
                validation_checkpoint_fallback=True,
            )
            private = _private_frame(input_dir, noise_name, seed)
            private.attrs["noise_name"] = noise_name
            clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
            initial_state = make_initial_state(data.train_features.shape[1], 2, seed)

            budget_values = [float(value) for value in budgets]
            adaptive_rankings: dict[float, np.ndarray] = {}
            selection_records: dict[float, tuple[float, dict[str, object]]] = {}
            if rule == "set_robust":
                # Joint set selection: each budget gets its own prefix because
                # the worst-case accumulation and diversity penalty are
                # explicitly budget-aware.
                for budget_fraction in budget_values:
                    budget_count = budget_to_count(
                        budget_fraction, len(data.train_labels)
                    )
                    ranking, metadata = robust_set_repair_value_ranking(
                        train_features=data.train_features,
                        train_labels=data.train_labels,
                        val_features=data.validation_features,
                        val_probabilities=data.validation_probabilities,
                        val_labels=data.validation_labels,
                        noise_posterior_proxy=data.noise_score,
                        budget_count=int(budget_count),
                        candidate_tail_fractions=(0.10, 0.20, 0.50),
                        num_folds=5,
                        min_tail_count=4,
                        uncertainty_beta=0.50,
                        redundancy_lambda=0.05,
                        candidate_multiplier=3.0,
                        projection_dim=32,
                        seed=seed,
                    )
                    selection_records[budget_fraction] = (float("nan"), metadata)
                    adaptive_rankings[budget_fraction] = ranking
            elif rule == "stability_contrast":
                # tau selection uses only legal pre-query observables and is
                # budget-independent, so it runs once per (noise, seed).
                tau, metadata = select_adaptive_tail_fraction(
                    train_features=data.train_features,
                    train_labels=data.train_labels,
                    val_features=data.validation_features,
                    val_probabilities=data.validation_probabilities,
                    val_labels=data.validation_labels,
                    candidate_tail_fractions=CANDIDATE_TAIL_FRACTIONS,
                    num_folds=5,
                    min_tail_count=4,
                    stability_sample_size=4096,
                    seed=seed,
                )
                for budget_fraction in budget_values:
                    selection_records[budget_fraction] = (tau, metadata)
                    adaptive_rankings[budget_fraction] = _rvq_ranking(data, tau)
            else:
                # The capture rule is budget-aware: the selection region
                # mirrors the actual query budget, so tau is chosen per
                # (noise, seed, budget) from legal observables only.
                for budget_fraction in budget_values:
                    budget_count = budget_to_count(
                        budget_fraction, len(data.train_labels)
                    )
                    selector = (
                        select_tail_fraction_by_calibrated_capture
                        if rule == "calibrated_capture"
                        else select_tail_fraction_by_capture
                    )
                    tau_b, metadata_b = selector(
                        train_features=data.train_features,
                        train_labels=data.train_labels,
                        val_features=data.validation_features,
                        val_probabilities=data.validation_probabilities,
                        val_labels=data.validation_labels,
                        noise_posterior_proxy=data.noise_score,
                        budget_count=int(budget_count),
                        candidate_tail_fractions=CANDIDATE_TAIL_FRACTIONS,
                        num_folds=5,
                        min_tail_count=4,
                        stability_sample_size=4096,
                        seed=seed,
                    )
                    selection_records[budget_fraction] = (tau_b, metadata_b)
                    adaptive_rankings[budget_fraction] = _rvq_ranking(data, tau_b)

            for budget_fraction in budget_values:
                baseline_metrics = _fit_and_evaluate(
                    data=data,
                    input_dir=input_dir,
                    labels=data.train_labels,
                    seed=seed,
                    training_config=training_config,
                    initial_state=initial_state,
                )
                stored_baseline = lookup[
                    (noise_name, seed, budget_fraction, "loss", -1)
                ]["baseline_wga"]
                deviation = abs(baseline_metrics["wga"] - stored_baseline)
                baseline_deviation = max(baseline_deviation, deviation)
                if deviation > BASELINE_TOLERANCE:
                    raise RuntimeError(
                        f"baseline WGA drift for {noise_name} seed {seed} "
                        f"budget {budget_fraction}: {deviation}"
                    )

                if verify_cell and seed == int(seeds[0]) and noise_name == noise_names[0]:
                    replay_ranking = _rvq_ranking(data, 0.50)
                    stored_delta = lookup[
                        (noise_name, seed, budget_fraction, "expected_repair_value", 1)
                    ]["delta_wga"]
                    replay_positions = replay_ranking[
                        : budget_to_count(budget_fraction, len(data.train_labels))
                    ]
                    replayed = data.train_labels.copy()
                    replayed[replay_positions] = clean_labels[replay_positions]
                    replay_metrics = _fit_and_evaluate(
                        data=data,
                        input_dir=input_dir,
                        labels=replayed,
                        seed=seed,
                        training_config=training_config,
                        initial_state=initial_state,
                    )
                    replay_delta = (
                        replay_metrics["wga"] - baseline_metrics["wga"]
                    )
                    if abs(replay_delta - stored_delta) > BASELINE_TOLERANCE:
                        raise RuntimeError(
                            "fixed-tau replay drift: "
                            f"{replay_delta} vs stored {stored_delta}"
                        )

                positions = adaptive_rankings[budget_fraction][
                    : budget_to_count(budget_fraction, len(data.train_labels))
                ]
                repaired = data.train_labels.copy()
                repaired[positions] = clean_labels[positions]
                metrics = _fit_and_evaluate(
                    data=data,
                    input_dir=input_dir,
                    labels=repaired,
                    seed=seed,
                    training_config=training_config,
                    initial_state=initial_state,
                )
                tau, metadata = selection_records[budget_fraction]
                if rule == "set_robust":
                    diagnostic = float(metadata["selected_prefix_worst_case"])
                else:
                    selected_candidate = next(
                        candidate
                        for candidate in metadata["candidates"]
                        if candidate["tau"] == tau
                    )
                if rule == "set_robust":
                    diagnostic = float(metadata["selected_prefix_worst_case"])
                elif rule == "stability_contrast":
                    diagnostic = float(selected_candidate["rank_stability"])
                elif rule == "capture":
                    diagnostic = float(selected_candidate["mean_capture"])
                else:
                    diagnostic = float(selected_candidate["combined_score"])
                row: dict[str, object] = {
                    "dataset": "waterbirds",
                    "noise_name": noise_name,
                    "seed": int(seed),
                    "method": (
                        "robust_set_repair_value"
                        if rule == "set_robust"
                        else f"expected_repair_value_{rule}_tau"
                    ),
                    "budget_fraction": float(budget_fraction),
                    "budget_count": int(
                        budget_to_count(budget_fraction, len(data.train_labels))
                    ),
                    "selected_tau": float(tau),
                    "wga": float(metrics["wga"]),
                    "baseline_wga": float(baseline_metrics["wga"]),
                    "delta_wga": float(metrics["wga"] - baseline_metrics["wga"]),
                    "delta_average_accuracy": float(
                        metrics["average_accuracy"]
                        - baseline_metrics["average_accuracy"]
                    ),
                    "query_noise_precision": float(
                        private["is_noisy"].to_numpy(dtype=bool)[positions].mean()
                    ),
                    "retraining_best_epoch": int(metrics["best_epoch"]),
                    "retraining_scope": "frozen_feature_linear_head",
                    "selection_rule": rule,
                    "selection_diagnostic": diagnostic,
                    "selection_metadata_json": json.dumps(metadata),
                    **posthoc_query_diagnostics(
                        private, adaptive_rankings[budget_fraction], budget_fraction
                    ),
                }
                rows.append(row)
                print(
                    f"{noise_name} seed {seed} budget {budget_fraction}: "
                    f"tau={tau:.2f} delta_wga={row['delta_wga']:+.4f} "
                    f"precision={row['query_noise_precision']:.3f}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_seed_adaptive.csv", index=False)
    summary = _summarize(frame, reference)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    if rule == "set_robust":
        rule_description = (
            "robust_set_repair_value_ranking (cross-fitted scenarios over "
            "tau={0.10,0.20,0.50}, mean-minus-0.5*std score, worst-case "
            "prefix accumulation with cosine redundancy penalty)"
        )
    elif rule == "stability_contrast":
        rule_description = (
            "select_adaptive_tail_fraction "
            "(0.8*rank_stability + 0.2*contrast_rank, folds=5, "
            "min_tail_count=4, stability_sample=4096)"
        )
    elif rule == "capture":
        rule_description = (
            "select_tail_fraction_by_capture "
            "(out-of-sample first-order capture ratio, folds=5, "
            "min_tail_count=4, stability_sample=4096, budget-aware top-k)"
        )
    else:
        rule_description = (
            "select_tail_fraction_by_calibrated_capture "
            "(equal-weight calibrated held-out effect rank + capture rank, "
            "folds=5, min_tail_count=4, stability_sample=4096, budget-aware top-k)"
        )
    protocol = {
        "analysis_status": f"exploratory_waterbirds_rvq_{rule}_tau",
        "dataset": "waterbirds",
        "method": (
            "robust_set_repair_value"
            if rule == "set_robust"
            else f"expected_repair_value_{rule}_tau"
        ),
        "selection_rule": rule_description,
        "candidate_tail_fractions": list(CANDIDATE_TAIL_FRACTIONS),
        "tau_scope": (
            "selected jointly per (noise, seed, budget)"
            if rule == "set_robust"
            else "selected once per (noise, seed), budget-independent"
            if rule == "stability_contrast"
            else "selected per (noise, seed, budget); budget-aware capture region"
        ),
        "seeds": [int(seed) for seed in seeds],
        "noise_names": list(noise_names),
        "budgets": [float(value) for value in budgets],
        "training": dict(training_config),
        "retraining_scope": "frozen_feature_linear_head",
        "reference_package": str(reference_path),
        "baseline_max_deviation": float(baseline_deviation),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "private_fields_used_for": "post_query_label_repair_and_posthoc_diagnostics",
    }
    with open(output_dir / "protocol.json", "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    print(f"wrote {output_dir / 'comparison_summary.csv'}", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument(
        "--noise-names", nargs="+", default=["uniform20", "minority_high_40"]
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-verify-cell", action="store_true")
    parser.add_argument(
        "--rule",
        choices=["stability_contrast", "capture", "calibrated_capture", "set_robust"],
        default="stability_contrast",
        help="tau-selection rule evaluated by this package",
    )
    args = parser.parse_args()
    run_evaluation(
        input_dir=args.input_dir,
        probe_dir=args.probe_dir,
        reference_path=args.reference,
        output_dir=args.output_dir,
        seeds=args.seeds,
        noise_names=args.noise_names,
        budgets=args.budgets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        verify_cell=not args.no_verify_cell,
        rule=args.rule,
    )


if __name__ == "__main__":
    main()
