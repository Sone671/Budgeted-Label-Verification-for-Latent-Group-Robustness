from __future__ import annotations

import numpy as np
import pytest

from scripts.conformal_verification_pilot import (
    bh_rejections,
    ebh_rejections,
    guarded_budget_selection,
    mondrian_upper_tail_pvalues,
    power_evalues,
    stratified_oof_fold_ids,
    stratified_validation_split,
    validate_oof_parameters,
)


def test_mondrian_pvalues_use_class_conditional_upper_tail_with_ties() -> None:
    candidate_scores = np.array([0.5, 0.0, 0.4, 0.2])
    candidate_labels = np.array([0, 0, 1, 1])
    calibration_scores = np.array([0.1, 0.3, 0.2, 0.4])
    calibration_labels = np.array([0, 0, 1, 1])

    actual = mondrian_upper_tail_pvalues(
        candidate_scores,
        candidate_labels,
        calibration_scores,
        calibration_labels,
    )

    # Large nonconformity has a small p-value.  The class-1 score tied at 0.4
    # counts that calibration item in the upper tail.
    np.testing.assert_allclose(actual, [1 / 3, 1.0, 2 / 3, 1.0])


def test_mondrian_pvalues_are_invariant_to_label_permutation() -> None:
    scores = np.array([0.5, 0.0, 0.4, 0.2])
    labels = np.array([0, 0, 1, 1])
    calibration_scores = np.array([0.1, 0.3, 0.2, 0.4])
    calibration_labels = np.array([0, 0, 1, 1])
    original = mondrian_upper_tail_pvalues(
        scores, labels, calibration_scores, calibration_labels
    )
    swapped = mondrian_upper_tail_pvalues(
        scores, 1 - labels, calibration_scores, 1 - calibration_labels
    )
    np.testing.assert_allclose(original, swapped)


def test_bh_and_ebh_step_up_rules() -> None:
    pvalues = np.array([0.001, 0.01, 0.20, 0.80])
    np.testing.assert_array_equal(bh_rejections(pvalues, 0.10), [0, 1])

    evalues = np.array([100.0, 50.0, 1.0, 1.0])
    np.testing.assert_array_equal(ebh_rejections(evalues, 0.10), [0, 1])

    converted = power_evalues(np.array([0.25, 1.0]), kappa=0.5)
    np.testing.assert_allclose(converted, [1.0, 0.5])


def test_guard_requires_enough_discoveries_to_fill_budget() -> None:
    pvalues = np.array([0.4, 0.1, 0.3, 0.2])
    fallback = np.array([3, 2, 1, 0])

    selected, activated = guarded_budget_selection(
        pvalues, fallback, budget_count=2, discovery_indices=np.array([1])
    )
    np.testing.assert_array_equal(selected, [3, 2])
    assert not activated

    selected, activated = guarded_budget_selection(
        pvalues, fallback, budget_count=2, discovery_indices=np.array([1, 3])
    )
    np.testing.assert_array_equal(selected, [1, 3])
    assert activated


def test_guard_can_fill_a_small_discovery_shortfall_from_fallback() -> None:
    pvalues = np.array([0.04, 0.01, 0.03, 0.02, 0.9])
    fallback = np.array([4, 3, 2, 1, 0])
    selected, activated = guarded_budget_selection(
        pvalues,
        fallback,
        budget_count=4,
        discovery_indices=np.array([0, 1, 3]),
        min_discovery_fraction=0.75,
    )
    assert activated
    np.testing.assert_array_equal(selected, [1, 3, 0, 4])


def test_validation_split_is_stratified_disjoint_and_exhaustive() -> None:
    labels = np.repeat(np.arange(3), 8)
    model_selection, calibration = stratified_validation_split(
        labels,
        calibration_fraction=0.5,
        seed=17,
    )

    assert not np.intersect1d(model_selection, calibration).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate([model_selection, calibration])),
        np.arange(len(labels)),
    )
    np.testing.assert_array_equal(np.unique(labels[model_selection]), [0, 1, 2])
    np.testing.assert_array_equal(np.unique(labels[calibration]), [0, 1, 2])
    repeated = stratified_validation_split(labels, 0.5, seed=17)
    np.testing.assert_array_equal(repeated[0], model_selection)
    np.testing.assert_array_equal(repeated[1], calibration)


def test_stratified_oof_folds_cover_every_example_exactly_once() -> None:
    labels = np.tile(np.array([0, 1]), 12)
    fold_ids = stratified_oof_fold_ids(labels, oof_folds=4, seed=3)

    held_out = [np.flatnonzero(fold_ids == fold_id) for fold_id in range(4)]
    np.testing.assert_array_equal(
        np.sort(np.concatenate(held_out)),
        np.arange(len(labels)),
    )
    assert sum(len(indices) for indices in held_out) == len(labels)
    for indices in held_out:
        np.testing.assert_array_equal(np.unique(labels[indices]), [0, 1])


@pytest.mark.parametrize(
    ("oof_folds", "calibration_fraction"),
    [(0, 0.5), (-2, 0.5), (2, 0.0), (2, 1.0), (2, float("nan"))],
)
def test_oof_parameter_validation_rejects_invalid_numeric_settings(
    oof_folds: int,
    calibration_fraction: float,
) -> None:
    with pytest.raises(ValueError):
        validate_oof_parameters(oof_folds, calibration_fraction)


def test_oof_parameter_validation_rejects_too_few_examples_per_class() -> None:
    with pytest.raises(ValueError, match="noisy-label class"):
        validate_oof_parameters(
            3,
            0.5,
            noisy_labels=np.array([0, 0, 1, 1, 1]),
            val_labels=np.array([0, 0, 1, 1]),
        )
