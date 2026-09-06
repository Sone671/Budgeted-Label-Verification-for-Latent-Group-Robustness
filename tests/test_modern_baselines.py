from __future__ import annotations

import numpy as np
import pandas as pd

from robust_verify.influence import compute_tracin_influence, uncertainty_weights
from robust_verify.modern_baselines import (
    active_label_cleaning_scores,
    activeclean_expected_model_change_scores,
    auto_d3m_query_adaptation_scores,
    class_conditional_tail_weights,
    expected_repair_value_adaptive_scores,
    expected_repair_value_calibrated_tau_scores,
    expected_repair_value_capture_scores,
    expected_repair_value_scores,
    robust_set_repair_value_ranking,
    multicheckpoint_tracin_influence,
    oof_cleanlab_scores,
)
from robust_verify.oof import fit_oof_probabilities
from robust_verify.scoring import build_rankings
from robust_verify.training import train_linear_head


def _probabilities(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    logits = rng.normal(size=shape)
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def test_multicheckpoint_tracin_matches_checkpoint_average() -> None:
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(13, 6)).astype(np.float32)
    val_x = rng.normal(size=(10, 6)).astype(np.float32)
    train_y = np.asarray([0, 1] * 6 + [0])
    val_y = np.asarray([0, 1] * 5)
    train_history = _probabilities(rng, (13, 4, 2)).astype(np.float32)
    val_history = _probabilities(rng, (10, 4, 2)).astype(np.float32)

    actual = multicheckpoint_tracin_influence(
        train_features=train_x,
        train_probability_history=train_history,
        train_labels=train_y,
        val_features=val_x,
        val_probability_history=val_history,
        val_labels=val_y,
    )
    expected = np.mean(
        [
            compute_tracin_influence(
                train_features=train_x,
                train_probs=train_history[:, checkpoint],
                train_labels=train_y,
                val_features=val_x,
                val_probs=val_history[:, checkpoint],
                val_labels=val_y,
                val_weights=uncertainty_weights(val_history[:, checkpoint]),
            )
            for checkpoint in range(train_history.shape[1])
        ],
        axis=0,
    )
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_class_conditional_tail_weights_are_class_balanced() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    weights = class_conditional_tail_weights(probabilities, labels, tail_fraction=0.5)

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[labels == 0].sum(), 0.5)
    assert np.isclose(weights[labels == 1].sum(), 0.5)
    assert np.count_nonzero(weights) == 4


def test_expected_repair_value_prefers_helpful_binary_flip() -> None:
    train_x = np.asarray([[1.0], [-1.0], [0.0]], dtype=np.float32)
    train_y = np.asarray([0, 0, 0])
    val_x = np.ones((4, 1), dtype=np.float32)
    val_y = np.ones(4, dtype=np.int64)
    val_probs = np.tile(np.asarray([[0.8, 0.2]]), (4, 1))

    scores = expected_repair_value_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=np.ones(3),
        tail_fraction=1.0,
    )

    assert scores[0] > scores[2] > scores[1]


def test_robust_set_repair_value_returns_deterministic_permutation() -> None:
    rng = np.random.default_rng(101)
    train_x = rng.normal(size=(30, 6)).astype(np.float32)
    val_x = rng.normal(size=(24, 6)).astype(np.float32)
    train_y = np.asarray([0, 1] * 15)
    val_y = np.asarray([0, 1] * 12)
    val_probs = _probabilities(rng, (24, 2))
    proxy = np.linspace(0.1, 1.0, len(train_y))
    kwargs = dict(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        budget_count=7,
        candidate_tail_fractions=(0.2, 0.5),
        num_folds=3,
        min_tail_count=2,
        uncertainty_beta=0.4,
        redundancy_lambda=0.1,
        seed=17,
    )
    first, first_meta = robust_set_repair_value_ranking(**kwargs)
    second, second_meta = robust_set_repair_value_ranking(**kwargs)
    assert np.array_equal(first, second)
    assert first_meta == second_meta
    assert np.array_equal(np.sort(first), np.arange(len(train_y)))
    assert len(first_meta["selected_indices"]) == 7
    assert first_meta["num_scenarios"] == 6
    assert np.isfinite(first_meta["selected_prefix_worst_case"])


def test_adaptive_tail_selection_is_deterministic_and_validation_only() -> None:
    rng = np.random.default_rng(13)
    train_x = rng.normal(size=(24, 5)).astype(np.float32)
    val_x = rng.normal(size=(20, 5)).astype(np.float32)
    train_y = np.asarray([0, 1] * 12)
    val_y = np.asarray([0] * 10 + [1] * 10)
    val_probs = _probabilities(rng, (20, 2))
    proxy = np.linspace(0.1, 1.0, len(train_y))

    first_scores, first_meta = expected_repair_value_adaptive_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        candidate_tail_fractions=(0.2, 0.5),
        num_folds=3,
        min_tail_count=2,
        stability_sample_size=12,
        seed=7,
    )
    second_scores, second_meta = expected_repair_value_adaptive_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        candidate_tail_fractions=(0.2, 0.5),
        num_folds=3,
        min_tail_count=2,
        stability_sample_size=12,
        seed=7,
    )

    assert np.array_equal(first_scores, second_scores)
    assert first_meta == second_meta
    assert first_meta["selected_tau"] in (0.2, 0.5)
    assert len(first_meta["candidates"]) == 2
    assert np.all(np.isfinite(first_scores))


def test_capture_tail_selection_is_deterministic_and_budget_aware() -> None:
    rng = np.random.default_rng(23)
    train_x = rng.normal(size=(24, 5)).astype(np.float32)
    val_x = rng.normal(size=(20, 5)).astype(np.float32)
    train_y = np.asarray([0, 1] * 12)
    val_y = np.asarray([0] * 10 + [1] * 10)
    val_probs = _probabilities(rng, (20, 2))
    proxy = np.linspace(0.1, 1.0, len(train_y))

    first_scores, first_meta = expected_repair_value_capture_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        budget_count=4,
        candidate_tail_fractions=(0.2, 0.5),
        num_folds=3,
        min_tail_count=2,
        stability_sample_size=12,
        seed=7,
    )
    second_scores, second_meta = expected_repair_value_capture_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        budget_count=4,
        candidate_tail_fractions=(0.2, 0.5),
        num_folds=3,
        min_tail_count=2,
        stability_sample_size=12,
        seed=7,
    )

    assert np.array_equal(first_scores, second_scores)
    assert first_meta == second_meta
    assert first_meta["selected_tau"] in (0.2, 0.5)
    assert first_meta["budget_count"] == 4
    assert len(first_meta["candidates"]) == 2
    assert np.all(np.isfinite(first_scores))
    # The returned scores must equal the fixed-tau rule at the selected tau.
    reference = expected_repair_value_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        tail_fraction=float(first_meta["selected_tau"]),
    )
    np.testing.assert_allclose(first_scores, reference)


def test_capture_tail_selection_falls_back_when_folds_are_invalid() -> None:
    rng = np.random.default_rng(29)
    train_x = rng.normal(size=(12, 4)).astype(np.float32)
    val_x = rng.normal(size=(8, 4)).astype(np.float32)
    train_y = np.asarray([0, 1] * 6)
    val_y = np.asarray([0] * 4 + [1] * 4)
    val_probs = _probabilities(rng, (8, 2))
    proxy = np.linspace(0.1, 1.0, len(train_y))

    _, metadata = expected_repair_value_capture_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        budget_count=2,
        candidate_tail_fractions=(0.05, 0.20, 0.50),
        num_folds=3,
        min_tail_count=100,
        stability_sample_size=12,
        seed=3,
    )
    assert metadata["fallback"] is True
    assert metadata["selected_tau"] == 0.20


def test_calibrated_capture_tau_is_deterministic_and_matches_selected_rule() -> None:
    rng = np.random.default_rng(37)
    train_x = rng.normal(size=(24, 5)).astype(np.float32)
    val_x = rng.normal(size=(20, 5)).astype(np.float32)
    train_y = np.asarray([0, 1] * 12)
    val_y = np.asarray([0] * 10 + [1] * 10)
    val_probs = _probabilities(rng, (20, 2))
    proxy = np.linspace(0.1, 1.0, len(train_y))
    kwargs = dict(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        budget_count=4,
        candidate_tail_fractions=(0.2, 0.5),
        num_folds=3,
        min_tail_count=2,
        stability_sample_size=12,
        seed=11,
    )

    first_scores, first_meta = expected_repair_value_calibrated_tau_scores(**kwargs)
    second_scores, second_meta = expected_repair_value_calibrated_tau_scores(**kwargs)

    assert np.array_equal(first_scores, second_scores)
    assert first_meta == second_meta
    assert first_meta["selected_tau"] in (0.2, 0.5)
    assert first_meta["effect_rank_weight"] == 0.5
    assert first_meta["capture_rank_weight"] == 0.5
    assert len(first_meta["candidates"]) == 2
    assert np.all(np.isfinite(first_scores))
    reference = expected_repair_value_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        tail_fraction=float(first_meta["selected_tau"]),
    )
    np.testing.assert_allclose(first_scores, reference)


def test_auto_d3m_query_adaptation_is_deterministic_and_group_blind() -> None:
    rng = np.random.default_rng(19)
    train_x = rng.normal(size=(18, 7)).astype(np.float32)
    val_x = rng.normal(size=(12, 7)).astype(np.float32)
    train_y = np.asarray([0, 1] * 9)
    val_y = np.asarray([0] * 6 + [1] * 6)
    val_probs = _probabilities(rng, (12, 2))
    proxy = np.linspace(0.1, 1.0, len(train_y))
    options = {
        "auto_d3m_projection_dim": 5,
        "auto_d3m_projection_seed": 23,
        "auto_d3m_tail_fraction": 0.34,
    }
    cache: dict[tuple, object] = {}

    first = auto_d3m_query_adaptation_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        options=options,
        cache=cache,
    )
    second = auto_d3m_query_adaptation_scores(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=proxy,
        options=options,
        cache=cache,
    )

    assert np.all(np.isfinite(first))
    assert np.all((0.0 <= first) & (first <= 1.0))
    assert np.array_equal(first, second)
    assert len(cache) == 1


def test_active_cleaning_adaptations_are_finite_and_ranked() -> None:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(16, 5)).astype(np.float32)
    labels = np.asarray([0, 1] * 8)
    probabilities = np.tile(np.asarray([[0.9, 0.1], [0.2, 0.8]]), (8, 1))
    probabilities[3] = [0.98, 0.02]
    probabilities[4] = [0.02, 0.98]

    emc = activeclean_expected_model_change_scores(
        train_features=features,
        train_probabilities=probabilities,
        train_labels=labels,
    )
    alc = active_label_cleaning_scores(
        train_probabilities=probabilities,
        train_labels=labels,
    )
    assert emc.shape == alc.shape == (16,)
    assert np.all(np.isfinite(emc)) and np.all(np.isfinite(alc))
    assert np.all((0.0 <= emc) & (emc <= 1.0))


def test_oof_cleanlab_is_deterministic_and_uses_held_out_rows() -> None:
    rng = np.random.default_rng(43)
    features = rng.normal(size=(30, 6)).astype(np.float32)
    labels = np.asarray([0, 1, 2] * 10)
    first = fit_oof_probabilities(features, labels, seed=9, folds=5, options={"oof_max_iter": 10})
    second = fit_oof_probabilities(features, labels, seed=9, folds=5, options={"oof_max_iter": 10})
    assert first.shape == (30, 3)
    assert np.allclose(first.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(first, second)
    scores = oof_cleanlab_scores(oof_train_probabilities=first, train_labels=labels)
    assert np.all(np.isfinite(scores))
    assert np.all((0.0 <= scores) & (scores <= 1.0))


def test_build_rankings_registers_all_modern_baselines() -> None:
    rng = np.random.default_rng(29)
    n_train, n_val, dim, checkpoints = 18, 12, 7, 3
    train_y = np.asarray([0, 1] * 9)
    val_y = np.asarray([0] * 6 + [1] * 6)
    train_probs = _probabilities(rng, (n_train, 2))
    train_history = _probabilities(rng, (n_train, checkpoints, 2)).astype(np.float32)
    val_history = _probabilities(rng, (n_val, checkpoints, 2)).astype(np.float32)
    val_probs = _probabilities(rng, (n_val, 2))
    correctness = (
        train_history.argmax(axis=2) == train_y[:, None]
    ).astype(np.int8)
    methods = [
        "tracin_multicheckpoint_val",
        "expected_repair_value",
        "expected_repair_value_adaptive_tau",
        "expected_repair_value_capture_tau",
        "expected_repair_value_calibrated_tau",
        "auto_d3m_query",
        "activeclean_expected_change",
        "active_label_cleaning",
        "oof_cleanlab",
    ]
    context = {
        "train_features": rng.normal(size=(n_train, dim)).astype(np.float32),
        "val_features": rng.normal(size=(n_val, dim)).astype(np.float32),
        "val_probs": val_probs,
        "val_labels": val_y,
        "probability_history": train_history,
        "val_probability_history": val_history,
        "modern_baseline_cache": {},
        "baseline_options": {
            "auto_d3m_projection_dim": 5,
            "auto_d3m_tail_fraction": 0.34,
            "repair_value_adaptive_tau_candidates": [0.2, 0.5],
            "repair_value_adaptive_tau_folds": 3,
            "repair_value_adaptive_tau_min_count": 2,
            "repair_value_adaptive_tau_sample_size": 12,
        },
        "oof_train_probs": fit_oof_probabilities(
            rng.normal(size=(n_train, dim)).astype(np.float32),
            train_y,
            seed=29,
            folds=3,
            options={"oof_max_iter": 10},
        ),
    }

    rankings, scores = build_rankings(
        methods=methods,
        probabilities=train_probs,
        labels=train_y,
        correctness_history=correctness,
        private_frame=pd.DataFrame({"sample_id": np.arange(n_train)}),
        seed=29,
        context=context,
    )

    for method in methods:
        assert sorted(rankings[method].tolist()) == list(range(n_train))
        assert scores[method].shape == (n_train,)
        assert np.all(np.isfinite(scores[method]))
    assert context["selection_metadata"]["expected_repair_value_adaptive_tau"]["selected_tau"] in (0.2, 0.5)


def test_training_records_validation_probabilities_at_matching_checkpoints() -> None:
    rng = np.random.default_rng(31)
    train_x = rng.normal(size=(20, 4)).astype(np.float32)
    val_x = rng.normal(size=(8, 4)).astype(np.float32)
    train_y = np.asarray([0, 1] * 10)
    val_y = np.asarray([0, 1] * 4)
    result = train_linear_head(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        config={
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "patience": 3,
            "selection_metric": "balanced_accuracy",
            "select_last_epoch": True,
            "device": "cpu",
        },
        seed=31,
        record_dynamics=True,
    )

    assert result.probability_history is not None
    assert result.val_probability_history is not None
    assert result.probability_history.shape == (20, 3, 2)
    assert result.val_probability_history.shape == (8, 3, 2)
