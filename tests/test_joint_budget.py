from __future__ import annotations

import numpy as np

from robust_verify.joint_budget import (
    estimate_soft_group_error,
    estimate_staged_marginal_value,
    estimate_feedback_group_noise_rates,
    estimate_ipw_feedback_group_noise_rates,
    feedback_updated_joint_scores,
    fit_attribute_probability_model,
    joint_repair_scores,
    relative_weak_group_error_margin,
    select_class_balanced_audits,
    select_group_audit_share,
    select_marginal_value_audit_share,
    select_soft_group_stratified_exploration,
    select_staged_group_audit_share,
    soft_group_probabilities,
)


def test_class_balanced_audits_have_exact_count_and_use_both_classes() -> None:
    labels = np.array([0] * 8 + [1] * 2)
    selected = select_class_balanced_audits(labels, 4, seed=7)
    assert len(selected) == 4
    assert len(np.unique(selected)) == 4
    assert set(labels[selected].tolist()) == {0, 1}


def test_one_class_attribute_audit_has_smoothed_constant_fallback() -> None:
    features = np.eye(3, dtype=np.float32)
    model = fit_attribute_probability_model(features, np.zeros(3), seed=0)
    probability = model.predict_probability(np.ones((2, 3), dtype=np.float32))
    assert not model.fitted
    assert np.allclose(probability, 0.2)


def test_soft_group_probabilities_sum_to_one() -> None:
    class_probability = np.array([[0.8, 0.2], [0.3, 0.7]])
    attribute_probability = np.array([0.25, 0.6])
    groups = soft_group_probabilities(class_probability, attribute_probability)
    assert groups.shape == (2, 4)
    assert np.allclose(groups.sum(axis=1), 1.0)
    assert np.allclose(groups[0], [0.6, 0.2, 0.15, 0.05])


def test_soft_group_error_uses_clean_validation_class_side() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    attribute_probability = np.array([0.0, 1.0, 0.0, 1.0])
    error, mass = estimate_soft_group_error(
        probabilities, labels, attribute_probability
    )
    assert np.allclose(mass, [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(error, [0.0, 1.0, 1.0, 0.0])


def test_joint_scores_prioritize_noise_in_high_risk_group() -> None:
    noise_score = np.array([0.8, 0.8])
    class_probability = np.array([[0.95, 0.05], [0.05, 0.95]])
    attribute_probability = np.array([0.0, 1.0])
    group_error = np.array([0.1, 0.1, 0.1, 0.9])
    score, value = joint_repair_scores(
        noise_score, class_probability, attribute_probability, group_error
    )
    assert value[1] > value[0]
    assert score[1] > score[0]


def test_group_share_selector_uses_frozen_risk_threshold() -> None:
    assert select_group_audit_share(0.199) == 0.50
    assert select_group_audit_share(0.200) == 0.75


def test_relative_weak_group_error_margin_uses_top_two_groups() -> None:
    assert np.isclose(
        relative_weak_group_error_margin(np.array([0.1, 0.8, 0.2, 0.4])),
        0.5,
    )
    assert relative_weak_group_error_margin(np.zeros(4)) == 0.0


def test_staged_selector_expands_only_with_identifiable_attribute_and_margin() -> None:
    separated = np.array([0.1, 0.8, 0.2, 0.4])
    ambiguous = np.array([0.1, 0.8, 0.2, 0.75])
    assert (
        select_staged_group_audit_share(
            separated, attribute_model_fitted=True
        )
        == 0.50
    )
    assert (
        select_staged_group_audit_share(
            ambiguous, attribute_model_fitted=True
        )
        == 0.25
    )
    assert (
        select_staged_group_audit_share(
            separated, attribute_model_fitted=False
        )
        == 0.25
    )


def test_marginal_value_estimate_compares_information_gain_to_label_cost() -> None:
    score = np.array([10.0, 9.0, 8.0, 1.0])
    scout = np.array([0, 1, 2, 3])
    low_gain = estimate_staged_marginal_value(
        np.array([1, 0, 2, 3]),
        scout,
        score,
        expanded_label_count=1,
        scout_label_count=3,
        observed_audit_increment=1,
        next_audit_increment=2,
    )
    assert low_gain.revealed_information_gain == 1.0
    assert low_gain.extrapolated_information_gain == 2.0
    assert low_gain.label_opportunity_cost == 17.0
    assert select_marginal_value_audit_share(
        low_gain, attribute_model_fitted=True
    ) == 0.25

    high_gain = estimate_staged_marginal_value(
        np.array([3, 1, 2, 0]),
        scout,
        score,
        expanded_label_count=1,
        scout_label_count=3,
        observed_audit_increment=1,
        next_audit_increment=2,
    )
    assert high_gain.revealed_information_gain == 9.0
    assert high_gain.net_value == 1.0
    assert select_marginal_value_audit_share(
        high_gain, attribute_model_fitted=True
    ) == 0.50
    assert select_marginal_value_audit_share(
        high_gain, attribute_model_fitted=False
    ) == 0.25


def test_feedback_noise_rates_use_soft_membership_and_beta_smoothing() -> None:
    probability = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    error = np.array([1.0, 1.0, 0.0])
    rate = estimate_feedback_group_noise_rates(probability, error)
    assert np.allclose(rate, [2.5 / 3.5, 1.5 / 3.5])


def test_feedback_updated_scores_prioritize_high_yield_group() -> None:
    score, value = feedback_updated_joint_scores(
        np.array([0.8, 0.8]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0.5, 0.5]),
        np.array([0.2, 0.8]),
    )
    assert value[1] > value[0]
    assert score[1] > score[0]


def test_soft_group_exploration_is_stratified_disjoint_and_auditable() -> None:
    probability = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]]
    )
    selected, inclusion = select_soft_group_stratified_exploration(
        probability, 3, seed=4, exclude=np.array([0])
    )
    assert len(selected) == 3
    assert 0 not in selected
    assert len(np.unique(selected)) == 3
    assert set(probability[selected].argmax(axis=1).tolist()) == {0, 1}
    assert ((inclusion > 0.0) & (inclusion <= 1.0)).all()


def test_ipw_feedback_rates_recover_weighted_group_yields() -> None:
    probability = np.array([[1.0, 0.0], [0.0, 1.0]])
    error = np.array([1.0, 0.0])
    inclusion = np.array([0.5, 0.25])
    rate = estimate_ipw_feedback_group_noise_rates(
        probability, error, inclusion
    )
    assert np.allclose(rate, [3.0 / 4.0, 1.0 / 6.0])
