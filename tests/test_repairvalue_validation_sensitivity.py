from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_verify.scoring import build_legal_scores
from scripts.analyze_repairvalue_validation_sensitivity import (
    SeedData,
    corrupt_validation_labels,
    condition_grid,
    score_condition,
    stratified_validation_indices,
    summarize_conditions,
)


def _seed_data() -> SeedData:
    rng = np.random.default_rng(17)
    train_features = rng.normal(size=(12, 4)).astype(np.float32)
    val_features = rng.normal(size=(10, 4)).astype(np.float32)
    train_labels = np.asarray([0, 1] * 6)
    val_labels = np.asarray([0] * 5 + [1] * 5)
    train_logits = rng.normal(size=(12, 2))
    train_probs = np.exp(train_logits - train_logits.max(axis=1, keepdims=True))
    train_probs /= train_probs.sum(axis=1, keepdims=True)
    val_logits = rng.normal(size=(10, 2))
    val_probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    val_probs /= val_probs.sum(axis=1, keepdims=True)
    correctness = (train_probs.argmax(axis=1)[:, None] == train_labels[:, None]).astype(np.int8)
    legal = build_legal_scores(train_probs, train_labels, correctness)
    return SeedData(
        seed=3,
        train_features=train_features,
        train_labels=train_labels,
        train_probabilities=train_probs,
        correctness_history=correctness,
        train_sample_ids=np.arange(12),
        validation_features=val_features,
        validation_labels=val_labels,
        validation_probabilities=val_probs,
        validation_sample_ids=np.arange(100, 110),
        noisy_mask=np.asarray([True, False] * 6),
        noise_score=legal["noise_score"],
        losses=legal["loss"],
    )


def test_stratified_validation_indices_are_deterministic_and_class_balanced() -> None:
    labels = np.asarray([0] * 8 + [1] * 2)
    first = stratified_validation_indices(labels, 0.25, seed=9)
    second = stratified_validation_indices(labels, 0.25, seed=9)

    assert np.array_equal(first, second)
    assert (labels[first] == 0).sum() == 2
    assert (labels[first] == 1).sum() == 1
    assert np.array_equal(first, np.sort(first))


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_stratified_validation_indices_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        stratified_validation_indices(np.asarray([0, 1]), fraction, seed=1)


def test_validation_label_corruption_preserves_clean_input_and_mode_semantics() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    clean_copy = labels.copy()
    symmetric = corrupt_validation_labels(labels, 0.5, seed=4, mode="symmetric")
    targeted = corrupt_validation_labels(labels, 0.5, seed=4, mode="class_conditional")

    assert np.array_equal(labels, clean_copy)
    assert np.sum(labels != symmetric) == 3
    assert np.all(targeted[labels == 0] == 0)
    assert np.sum(labels != targeted) == 2
    assert np.array_equal(corrupt_validation_labels(labels, 0.0, seed=4), labels)


def test_score_condition_is_score_only_and_returns_valid_ranking() -> None:
    data = _seed_data()
    reference_row, reference_ranking = score_condition(
        data,
        tau=0.20,
        validation_fraction=1.0,
        validation_noise_rate=0.0,
        validation_noise_mode="symmetric",
        subset_seed=1,
        corruption_seed=2,
        budget_fraction=0.25,
    )
    row, ranking = score_condition(
        data,
        tau=0.50,
        validation_fraction=0.50,
        validation_noise_rate=0.20,
        validation_noise_mode="symmetric",
        subset_seed=3,
        corruption_seed=4,
        budget_fraction=0.25,
        reference_query_positions=reference_ranking[: int(reference_row["budget_count"])],
    )

    assert sorted(ranking.tolist()) == list(range(len(data.train_labels)))
    assert row["validation_size"] == 6
    assert row["active_tail_count_min_by_class"] >= 1
    assert row["active_tail_count_min_by_class"] <= row["active_tail_count"]
    assert row["query_noise_precision"] >= 0.0
    assert 0.0 <= row["query_jaccard_to_reference"] <= 1.0
    assert np.isfinite(float(row["score_rank_correlation_to_noise"]))


def test_summarize_conditions_reports_percentages_without_retraining_claims() -> None:
    data = _seed_data()
    rows = []
    for tau in (0.2, 0.4):
        row, _ = score_condition(
            data,
            tau=tau,
            validation_fraction=1.0,
            validation_noise_rate=0.0,
            validation_noise_mode="symmetric",
            subset_seed=1,
            corruption_seed=2,
            budget_fraction=0.25,
        )
        rows.append(row)
    summary = summarize_conditions(pd.DataFrame(rows))

    assert len(summary) == 2
    assert set(summary["n_seeds"]) == {1}
    assert np.all(
        (summary["mean_query_noise_precision"] >= 0.0)
        & (summary["mean_query_noise_precision"] <= 100.0)
    )
    assert "mean_query_jaccard_to_reference" in summary.columns


def test_default_condition_grid_is_one_factor_and_full_factorial_is_explicit() -> None:
    compact = condition_grid(
        [0.1, 0.2],
        [0.5, 1.0],
        [0.0, 0.1],
        ["symmetric"],
    )
    factorial = condition_grid(
        [0.1, 0.2],
        [0.5, 1.0],
        [0.0, 0.1],
        ["symmetric"],
        design="full_factorial",
    )

    assert len(compact) < len(factorial)
    assert len({condition[1:] for condition in compact}) == len(compact)
    assert any(condition[1:] == (0.2, 1.0, 0.1, "symmetric") for condition in compact)
    assert len(factorial) == 8
