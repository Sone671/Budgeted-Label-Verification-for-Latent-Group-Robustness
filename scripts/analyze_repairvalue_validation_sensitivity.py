#!/usr/bin/env python
"""Score-only RepairValue-Q sensitivity analysis.

This script deliberately reuses frozen features and cached probe dynamics. It
does not import or call a training entry point, verify labels, or retrain a
head. Validation-size and validation-label-quality perturbations are applied
only while recomputing the legal query scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from robust_verify.config import output_layout
from robust_verify.modern_baselines import (
    class_conditional_tail_weights,
    expected_repair_value_scores,
)
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import LinearHead, predict_probabilities
from robust_verify.utils import budget_to_count, normalized_rank


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO / "outputs/ablation_celeba_10seeds"
DEFAULT_PROBE_DIR = (
    REPO / "outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/probes"
)
DEFAULT_OUTPUT_DIR = REPO / "tmp/local_paper_audit_20260824/repairvalue_validation_sensitivity"


@dataclass(frozen=True)
class SeedData:
    """All cached arrays needed for score-only analysis for one seed."""

    seed: int
    train_features: np.ndarray
    train_labels: np.ndarray
    train_probabilities: np.ndarray
    correctness_history: np.ndarray
    train_sample_ids: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    validation_probabilities: np.ndarray
    validation_sample_ids: np.ndarray
    noisy_mask: np.ndarray
    noise_score: np.ndarray
    losses: np.ndarray


def _aligned_manifest(path: Path, sample_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"manifest lacks sample_id: {path}")
    indexed = frame.set_index("sample_id", drop=False)
    wanted = np.asarray(sample_ids, dtype=np.int64)
    if len(indexed.index) != len(set(indexed.index.tolist())):
        raise ValueError(f"manifest has duplicate sample IDs: {path}")
    missing = sorted(set(wanted.tolist()).difference(indexed.index.tolist()))
    if missing:
        raise ValueError(f"manifest is missing {len(missing)} sample IDs: {path}")
    return indexed.loc[wanted].reset_index(drop=True)


def load_seed_data(
    input_dir: str | Path,
    probe_dir: str | Path,
    *,
    seed: int,
    noise_name: str = "uniform20",
    validation_checkpoint_fallback: bool = False,
) -> SeedData:
    """Load one cached seed and fail if the required score-only artifacts are absent."""
    root = output_layout(input_dir)
    probe_root = Path(probe_dir)
    train_npz = np.load(root["features"] / "train.npz")
    val_npz = np.load(root["features"] / "val.npz")
    train_features = np.asarray(train_npz["features"], dtype=np.float32)
    train_ids = np.asarray(train_npz["sample_ids"], dtype=np.int64)
    val_features = np.asarray(val_npz["features"], dtype=np.float32)
    val_ids = np.asarray(val_npz["sample_ids"], dtype=np.int64)

    prefix = f"{noise_name}_seed{int(seed)}"
    public = _aligned_manifest(root["manifests"] / f"{prefix}_train_public.csv", train_ids)
    private = _aligned_manifest(root["manifests"] / f"{prefix}_train_private.csv", train_ids)
    val_public = _aligned_manifest(root["manifests"] / "val_public.csv", val_ids)

    dynamics_path = probe_root / f"{prefix}.dynamics.npz"
    if not dynamics_path.exists():
        raise FileNotFoundError(
            "Cached probe dynamics are required; this score-only script never "
            f"trains missing probes: {dynamics_path}"
        )
    dynamics = np.load(dynamics_path)
    required = {"train_probabilities", "correctness_history"}
    missing = sorted(required.difference(dynamics.files))
    if missing:
        raise ValueError(f"probe dynamics lacks required arrays {missing}: {dynamics_path}")
    best_epoch = (
        int(np.asarray(dynamics["best_epoch"]).item())
        if "best_epoch" in dynamics.files
        else -1
    )
    train_probabilities = np.asarray(dynamics["train_probabilities"], dtype=np.float64)
    correctness = np.asarray(dynamics["correctness_history"], dtype=np.int8)
    if "val_probability_history" in dynamics.files:
        val_history = np.asarray(dynamics["val_probability_history"])
        if best_epoch < 0 or best_epoch >= val_history.shape[1]:
            raise ValueError(f"best_epoch is outside validation history: {dynamics_path}")
        val_probabilities = np.asarray(val_history[:, best_epoch, :], dtype=np.float64)
    elif validation_checkpoint_fallback:
        # Older probe archives retained the best checkpoint but not validation
        # probabilities. Recompute them from that checkpoint on frozen features.
        checkpoint_path = probe_root / f"{prefix}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Probe archive lacks val_probability_history and its checkpoint: "
                f"{checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("best_state") if isinstance(checkpoint, dict) else None
        if state is None:
            raise ValueError(f"Probe checkpoint lacks best_state: {checkpoint_path}")
        model = LinearHead(train_features.shape[1], 2)
        model.load_state_dict(state)
        val_probabilities = np.asarray(
            predict_probabilities(
                model,
                val_features,
                batch_size=4096,
                device=torch.device("cpu"),
            ),
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"probe dynamics lacks val_probability_history: {dynamics_path}; "
            "set validation_checkpoint_fallback=True only when a cached probe "
            "checkpoint is available"
        )
    train_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    if train_probabilities.shape[0] != len(train_labels):
        raise ValueError(f"train probability/manifest mismatch: {dynamics_path}")
    if val_probabilities.shape[0] != len(val_labels):
        raise ValueError(f"validation probability/manifest mismatch: {dynamics_path}")
    if not np.array_equal(np.asarray(dynamics["sample_ids"], dtype=np.int64), train_ids):
        raise ValueError(f"probe sample IDs do not match train features: {dynamics_path}")

    legal = build_legal_scores(train_probabilities, train_labels, correctness)
    noisy_column = "is_noisy"
    if noisy_column not in private.columns:
        raise ValueError(f"private manifest lacks {noisy_column}: {root['manifests']}")
    return SeedData(
        seed=int(seed),
        train_features=train_features,
        train_labels=train_labels,
        train_probabilities=train_probabilities,
        correctness_history=correctness,
        train_sample_ids=train_ids,
        validation_features=val_features,
        validation_labels=val_labels,
        validation_probabilities=val_probabilities,
        validation_sample_ids=val_ids,
        noisy_mask=private[noisy_column].to_numpy(dtype=bool),
        noise_score=legal["noise_score"],
        losses=legal["loss"],
    )


def stratified_validation_indices(
    labels: np.ndarray,
    fraction: float,
    *,
    seed: int,
) -> np.ndarray:
    """Return a deterministic, per-class validation subset."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("validation labels must be a non-empty vector")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("validation fraction must lie in (0, 1]")
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        count = max(1, int(np.ceil(float(fraction) * len(positions))))
        if count > len(positions):
            count = len(positions)
        selected.append(np.sort(rng.choice(positions, size=count, replace=False)))
    return np.sort(np.concatenate(selected).astype(np.int64))


def corrupt_validation_labels(
    labels: np.ndarray,
    rate: float,
    *,
    seed: int,
    mode: str = "symmetric",
) -> np.ndarray:
    """Create controlled binary validation-label corruption for score analysis.

    ``symmetric`` flips a fraction of all rows. ``class_conditional`` flips a
    fraction of class-1 rows only, which is useful for a targeted quality check.
    The clean input array is never modified.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("validation labels must be a non-empty vector")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("validation corruption rate must lie in [0, 1]")
    if mode not in {"symmetric", "class_conditional"}:
        raise ValueError("validation corruption mode must be symmetric or class_conditional")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("validation-label corruption currently supports binary labels")
    corrupted = labels.copy()
    if rate == 0.0:
        return corrupted
    eligible = np.arange(len(labels)) if mode == "symmetric" else np.flatnonzero(labels == 1)
    count = max(1, int(np.round(float(rate) * len(eligible))))
    count = min(count, len(eligible))
    chosen = np.random.default_rng(int(seed)).choice(eligible, size=count, replace=False)
    corrupted[chosen] = 1 - corrupted[chosen]
    return corrupted


def _score_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = normalized_rank(np.asarray(first, dtype=np.float64))
    second_rank = normalized_rank(np.asarray(second, dtype=np.float64))
    if len(first_rank) <= 1 or np.std(first_rank) == 0.0 or np.std(second_rank) == 0.0:
        return 1.0 if np.array_equal(first_rank, second_rank) else 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def score_condition(
    data: SeedData,
    *,
    tau: float,
    validation_fraction: float,
    validation_noise_rate: float,
    validation_noise_mode: str,
    subset_seed: int,
    corruption_seed: int,
    budget_fraction: float,
    reference_query_positions: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Recompute one RepairValue-Q ranking without any training."""
    val_indices = stratified_validation_indices(
        data.validation_labels, validation_fraction, seed=subset_seed
    )
    clean_labels = data.validation_labels[val_indices]
    scored_labels = corrupt_validation_labels(
        clean_labels,
        validation_noise_rate,
        seed=corruption_seed,
        mode=validation_noise_mode,
    )
    scores = expected_repair_value_scores(
        train_features=data.train_features,
        train_labels=data.train_labels,
        val_features=data.validation_features[val_indices],
        val_probabilities=data.validation_probabilities[val_indices],
        val_labels=scored_labels,
        noise_posterior_proxy=data.noise_score,
        tail_fraction=tau,
    )
    ranking = descending_ranking(scores, data.losses)
    budget_count = budget_to_count(budget_fraction, len(data.train_sample_ids))
    query_positions = ranking[:budget_count]
    query_mask = data.noisy_mask[query_positions]
    tail_weights = class_conditional_tail_weights(
        data.validation_probabilities[val_indices], scored_labels, tail_fraction=tau
    )
    row: dict[str, object] = {
        "seed": data.seed,
        "tail_fraction": float(tau),
        "validation_fraction": float(validation_fraction),
        "validation_size": int(len(val_indices)),
        "validation_noise_rate": float(validation_noise_rate),
        "validation_noise_mode": validation_noise_mode,
        "validation_label_accuracy": float(np.mean(clean_labels == scored_labels)),
        "budget_fraction": float(budget_fraction),
        "budget_count": int(budget_count),
        "query_noise_precision": float(np.mean(query_mask)) if len(query_mask) else float("nan"),
        "query_noisy_count": int(query_mask.sum()),
        "query_jaccard_to_reference": float("nan"),
        "score_rank_correlation_to_noise": _score_correlation(scores, data.noise_score),
        "active_tail_count": int(np.count_nonzero(tail_weights)),
        "active_tail_count_min_by_class": int(
            min(
                np.count_nonzero(tail_weights[scored_labels == label])
                for label in np.unique(scored_labels)
            )
        ),
    }
    if reference_query_positions is not None:
        left = set(query_positions.tolist())
        right = set(np.asarray(reference_query_positions, dtype=np.int64).tolist())
        row["query_jaccard_to_reference"] = len(left & right) / max(1, len(left | right))
    return row, ranking


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score-only rows without pretending they are retraining effects."""
    if frame.empty:
        return pd.DataFrame()
    keys = [
        "tail_fraction",
        "validation_fraction",
        "validation_noise_rate",
        "validation_noise_mode",
    ]
    grouped = frame.groupby(keys, sort=True, dropna=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        mean_validation_size=("validation_size", "mean"),
        mean_validation_label_accuracy=("validation_label_accuracy", "mean"),
        mean_query_noise_precision=("query_noise_precision", "mean"),
        mean_query_jaccard_to_reference=("query_jaccard_to_reference", "mean"),
        mean_score_rank_correlation_to_noise=("score_rank_correlation_to_noise", "mean"),
        mean_active_tail_count_min_by_class=("active_tail_count_min_by_class", "mean"),
    ).reset_index()
    for column in (
        "mean_validation_label_accuracy",
        "mean_query_noise_precision",
    ):
        summary[column] *= 100.0
    return summary


def condition_grid(
    taus: Iterable[float],
    validation_fractions: Iterable[float],
    validation_noise_rates: Iterable[float],
    validation_noise_modes: Iterable[str],
    *,
    design: str = "one_factor",
) -> list[tuple[int, float, float, float, str]]:
    """Build either a compact one-factor grid or an explicit full factorial."""
    tau_values = [float(value) for value in taus]
    fraction_values = [float(value) for value in validation_fractions]
    rate_values = [float(value) for value in validation_noise_rates]
    mode_values = list(validation_noise_modes)
    if design not in {"one_factor", "full_factorial"}:
        raise ValueError("design must be one_factor or full_factorial")
    conditions: list[tuple[int, float, float, float, str]] = []
    if design == "full_factorial":
        for fraction_index, fraction in enumerate(fraction_values):
            for mode_index, mode in enumerate(mode_values):
                for rate_index, rate in enumerate(rate_values):
                    for tau_index, tau in enumerate(tau_values):
                        index = (
                            fraction_index * len(mode_values) * len(rate_values) * len(tau_values)
                            + mode_index * len(rate_values) * len(tau_values)
                            + rate_index * len(tau_values)
                            + tau_index
                        )
                        conditions.append((index, tau, fraction, rate, mode))
        return conditions

    # The compact default keeps each sensitivity axis interpretable and avoids
    # multiplying all dimensions before the user explicitly requests it.
    index = 0
    for tau in tau_values:
        conditions.append((index, tau, 1.0, 0.0, "symmetric"))
        index += 1
    for fraction in fraction_values:
        conditions.append((index, 0.20, fraction, 0.0, "symmetric"))
        index += 1
    for mode in mode_values:
        for rate in rate_values:
            conditions.append((index, 0.20, 1.0, rate, mode))
            index += 1
    # Remove duplicate default rows while retaining the first occurrence.
    unique: list[tuple[int, float, float, float, str]] = []
    seen: set[tuple[float, float, float, str]] = set()
    for condition in conditions:
        key = condition[1:]
        if key not in seen:
            seen.add(key)
            unique.append(condition)
    return unique


def run_analysis(
    *,
    input_dir: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    seeds: Iterable[int],
    taus: Iterable[float],
    validation_fractions: Iterable[float],
    validation_noise_rates: Iterable[float],
    validation_noise_modes: Iterable[str],
    budget_fraction: float,
    noise_name: str = "uniform20",
    design: str = "one_factor",
) -> pd.DataFrame:
    """Run score-only conditions and write per-seed and aggregate artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions = condition_grid(
        taus,
        validation_fractions,
        validation_noise_rates,
        validation_noise_modes,
        design=design,
    )
    rows: list[dict[str, object]] = []

    for seed in [int(value) for value in seeds]:
        data = load_seed_data(input_dir, probe_dir, seed=seed, noise_name=noise_name)
        reference_row, reference_ranking = score_condition(
            data,
            tau=0.20,
            validation_fraction=1.0,
            validation_noise_rate=0.0,
            validation_noise_mode="symmetric",
            subset_seed=seed + 100_000,
            corruption_seed=seed + 200_000,
            budget_fraction=budget_fraction,
        )
        reference_count = int(reference_row["budget_count"])
        reference_positions = reference_ranking[:reference_count]
        for condition_index, tau, fraction, rate, mode in conditions:
            subset_seed = seed * 1_000_003 + condition_index * 10_007
            corruption_seed = seed * 2_000_003 + condition_index * 10_009
            row, _ = score_condition(
                data,
                tau=tau,
                validation_fraction=fraction,
                validation_noise_rate=rate,
                validation_noise_mode=mode,
                subset_seed=subset_seed,
                corruption_seed=corruption_seed,
                budget_fraction=budget_fraction,
                reference_query_positions=reference_positions,
            )
            row["condition_index"] = condition_index
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_scores.csv", index=False)
    summarize_conditions(frame).to_csv(output / "score_sensitivity_summary.csv", index=False)
    metadata = {
        "analysis_status": "exploratory_frozen_feature_score_only",
        "training_executed": False,
        "label_verification_executed": False,
        "retraining_executed": False,
        "reference": {
            "tail_fraction": 0.20,
            "validation_fraction": 1.0,
            "validation_noise_rate": 0.0,
        },
        "noise_name": noise_name,
        "design": design,
        "condition_count_per_seed": len(conditions),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "STATUS.md").write_text(
        "# RepairValue-Q validation sensitivity\n\n"
        "This is an exploratory frozen-feature score-only analysis. It reuses cached probe "
        "dynamics and\n"
        "does not train, verify labels, retrain, or alter any registered confirmatory output.\n",
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
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
        choices=["symmetric", "class_conditional"],
        default=["symmetric"],
    )
    parser.add_argument("--design", choices=["one_factor", "full_factorial"], default="one_factor")
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--noise-name", default="uniform20")
    args = parser.parse_args()
    frame = run_analysis(
        input_dir=args.input_dir,
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        taus=args.taus,
        validation_fractions=args.validation_fractions,
        validation_noise_rates=args.validation_noise_rates,
        validation_noise_modes=args.validation_noise_modes,
        budget_fraction=args.budget_fraction,
        noise_name=args.noise_name,
        design=args.design,
    )
    print(summarize_conditions(frame).to_string(index=False))
    print(f"Wrote score-only sensitivity artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
