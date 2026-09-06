import numpy as np
import pytest

from robust_verify.transfer_budget import (
    assert_legal_feature_names,
    balanced_accuracy_present_classes,
    build_legal_transition_features,
    feedback_summary,
    fit_predict_seed_block_bootstrap,
    selective_decisions,
    validation_summary,
)


def _probs(values):
    return np.asarray(values, dtype=np.float64)


def test_validation_summary_is_group_free_and_finite():
    probabilities = _probs([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    summary = validation_summary(probabilities, np.array([0, 1, 1, 0]))
    assert set(summary) == {
        "balanced_accuracy",
        "min_class_accuracy",
        "class_accuracy_gap",
        "ce_mean",
        "ce_q90",
        "class_cvar10_max",
        "entropy_mean",
        "margin_q10",
    }
    assert all(np.isfinite(value) for value in summary.values())


def test_feedback_uses_only_current_prefix_and_recent_batch():
    noisy = np.array([0, 0, 1, 1, 0, 1])
    clean = np.array([0, 1, 1, 0, 0, 1])
    result = feedback_summary(noisy, clean, previous_count=4)
    assert result["feedback_cumulative_correction_rate"] == pytest.approx(2 / 6)
    assert result["feedback_recent_correction_rate"] == pytest.approx(0.0)
    assert 0.0 <= result["feedback_returned_label_entropy"] <= 1.0


def test_transition_feature_names_are_stable_and_legal():
    current = _probs([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    previous = _probs([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.2, 0.8]])
    feedback = feedback_summary(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]), previous_count=2
    )
    features = build_legal_transition_features(
        current_budget=0.005,
        next_budget=0.01,
        calibration_current=current,
        calibration_previous=previous,
        calibration_labels=np.array([0, 1, 1, 0]),
        audit_current=current,
        audit_previous=previous,
        audit_labels=np.array([0, 1, 1, 0]),
        feedback=feedback,
    )
    assert_legal_feature_names(tuple(features))
    assert len(features) == 47
    assert all(np.isfinite(value) for value in features.values())


def test_forbidden_feature_names_are_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        assert_legal_feature_names(("validation_loss", "target_wga"))


def test_bootstrap_is_trajectory_blocked_and_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 3))
    y = np.array([0, 0, 1, 1] * 3)
    groups = np.repeat(["a", "b", "c"], 4)
    first = fit_predict_seed_block_bootstrap(x, y, groups, x[:4], replicates=20, seed=7)
    second = fit_predict_seed_block_bootstrap(x, y, groups, x[:4], replicates=20, seed=7)
    assert first.shape == (20, 4)
    assert np.allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_selective_controller_abstains_between_quantiles():
    samples = np.array(
        [
            [0.9, 0.5, 0.1],
            [0.8, 0.5, 0.2],
            [0.7, 0.5, 0.3],
            [0.9, 0.5, 0.1],
        ]
    )
    decisions = selective_decisions(samples)
    assert [decision.action for decision in decisions] == ["continue", "abstain", "stop"]


def test_present_class_balanced_accuracy_handles_single_class():
    assert balanced_accuracy_present_classes(np.array([1, 1]), np.array([1, 0])) == pytest.approx(0.5)
