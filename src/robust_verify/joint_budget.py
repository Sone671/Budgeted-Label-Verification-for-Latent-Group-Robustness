"""Utilities for jointly budgeting group-attribute audits and label verification.

The group audit in this module reveals only a binary nuisance/protected
attribute (the manifest's ``place`` column).  It never reveals a clean training
label or a full ``clean_label x place`` group ID.  Clean validation labels are
assumed public, matching the main Stage-1 protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class AttributeProbabilityModel:
    """A sparse-audit binary attribute model with a one-class fallback."""

    estimator: LogisticRegression | None
    constant_probability: float
    fitted: bool

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("features must have shape [n, d]")
        if self.estimator is None:
            return np.full(len(features), self.constant_probability, dtype=np.float64)
        normalized = normalize(features, norm="l2", axis=1, copy=True)
        return self.estimator.predict_proba(normalized)[:, 1].astype(np.float64)


@dataclass(frozen=True)
class MarginalValueEstimate:
    """Score-unit comparison of more auditing versus more label queries."""

    revealed_information_gain: float
    extrapolation_scale: float
    extrapolated_information_gain: float
    label_opportunity_cost: float
    net_value: float
    value_ratio: float


def select_class_balanced_audits(
    clean_validation_labels: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select an audit set using only public clean validation labels.

    Sampling is round-robin across class-specific shuffled queues.  No group or
    attribute value is read while constructing the audit set.
    """

    labels = np.asarray(clean_validation_labels)
    if labels.ndim != 1:
        raise ValueError("clean_validation_labels must be one-dimensional")
    if not 0 <= count <= len(labels):
        raise ValueError("count must lie in [0, n_validation]")
    if count == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    queues: list[list[int]] = []
    for value in sorted(np.unique(labels).tolist()):
        members = np.flatnonzero(labels == value).astype(np.int64)
        rng.shuffle(members)
        queues.append(members.tolist())

    selected: list[int] = []
    while len(selected) < count and queues:
        remaining: list[list[int]] = []
        for queue in queues:
            if len(selected) >= count:
                break
            if queue:
                selected.append(int(queue.pop()))
            if queue:
                remaining.append(queue)
        queues = remaining
    return np.asarray(selected, dtype=np.int64)


def fit_attribute_probability_model(
    audited_features: np.ndarray,
    audited_attributes: np.ndarray,
    *,
    seed: int,
) -> AttributeProbabilityModel:
    """Fit a group-free feature-to-attribute model from audited examples.

    A Beta(1, 1) smoothed prevalence is retained as a deterministic fallback
    when the small audit contains only one attribute value.
    """

    features = np.asarray(audited_features, dtype=np.float32)
    attributes = np.asarray(audited_attributes, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(attributes):
        raise ValueError("audited_features and audited_attributes must align")
    if len(attributes) == 0:
        raise ValueError("at least one attribute audit is required")
    if not set(np.unique(attributes).tolist()).issubset({0, 1}):
        raise ValueError("the pilot currently supports a binary audited attribute")

    smoothed_prevalence = float((attributes.sum() + 1.0) / (len(attributes) + 2.0))
    if len(np.unique(attributes)) < 2:
        return AttributeProbabilityModel(
            estimator=None,
            constant_probability=smoothed_prevalence,
            fitted=False,
        )

    normalized = normalize(features, norm="l2", axis=1, copy=True)
    estimator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    )
    estimator.fit(normalized, attributes)
    return AttributeProbabilityModel(
        estimator=estimator,
        constant_probability=smoothed_prevalence,
        fitted=True,
    )


def soft_group_probabilities(
    class_probabilities: np.ndarray,
    attribute_probability: np.ndarray,
) -> np.ndarray:
    """Return probabilities for groups ``group = 2 * class + attribute``."""

    class_probabilities = np.asarray(class_probabilities, dtype=np.float64)
    attribute_probability = np.asarray(attribute_probability, dtype=np.float64)
    if class_probabilities.ndim != 2:
        raise ValueError("class_probabilities must have shape [n, num_classes]")
    if attribute_probability.shape != (len(class_probabilities),):
        raise ValueError("attribute_probability must have shape [n]")
    if not np.isfinite(class_probabilities).all() or not np.isfinite(
        attribute_probability
    ).all():
        raise ValueError("probabilities must be finite")
    if ((attribute_probability < 0.0) | (attribute_probability > 1.0)).any():
        raise ValueError("attribute probabilities must lie in [0, 1]")

    class_sums = class_probabilities.sum(axis=1)
    if not np.allclose(class_sums, 1.0, atol=1e-5):
        raise ValueError("class probabilities must sum to one")

    attribute = np.column_stack([1.0 - attribute_probability, attribute_probability])
    groups = class_probabilities[:, :, None] * attribute[:, None, :]
    return groups.reshape(len(class_probabilities), -1)


def estimate_soft_group_error(
    validation_class_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    validation_attribute_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate validation error and mass for each soft latent group.

    Clean validation labels are used to define the class side of the group.
    Only the audited-attribute model supplies the other side.
    """

    probabilities = np.asarray(validation_class_probabilities, dtype=np.float64)
    labels = np.asarray(validation_labels, dtype=np.int64)
    attribute_probability = np.asarray(
        validation_attribute_probability, dtype=np.float64
    )
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("validation probabilities and labels must align")
    if attribute_probability.shape != (len(labels),):
        raise ValueError("validation attribute probabilities must align")

    predictions = probabilities.argmax(axis=1)
    errors = (predictions != labels).astype(np.float64)
    num_classes = probabilities.shape[1]
    group_error = np.zeros(num_classes * 2, dtype=np.float64)
    group_mass = np.zeros(num_classes * 2, dtype=np.float64)
    for class_value in range(num_classes):
        class_mask = labels == class_value
        for attribute_value in (0, 1):
            group = 2 * class_value + attribute_value
            attribute_weight = (
                attribute_probability
                if attribute_value == 1
                else 1.0 - attribute_probability
            )
            weight = class_mask.astype(np.float64) * attribute_weight
            mass = float(weight.sum())
            group_mass[group] = mass
            group_error[group] = float(np.dot(weight, errors) / max(mass, 1e-12))
    return group_error, group_mass


def joint_repair_scores(
    noise_score: np.ndarray,
    train_class_probabilities: np.ndarray,
    train_attribute_probability: np.ndarray,
    group_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score examples by estimated label-error probability times group risk."""

    noise_score = np.asarray(noise_score, dtype=np.float64)
    group_error = np.asarray(group_error, dtype=np.float64)
    group_probability = soft_group_probabilities(
        train_class_probabilities, train_attribute_probability
    )
    if noise_score.shape != (len(group_probability),):
        raise ValueError("noise_score must align with train probabilities")
    if group_error.shape != (group_probability.shape[1],):
        raise ValueError("group_error must contain one value per soft group")
    if (group_error < 0.0).any() or not np.isfinite(group_error).all():
        raise ValueError("group errors must be finite and nonnegative")

    expected_group_risk = group_probability @ group_error
    score = noise_score * np.maximum(expected_group_risk, 1e-12)
    return score, expected_group_risk


def select_group_audit_share(
    mean_expected_group_risk: float,
    *,
    threshold: float = 0.20,
    base_share: float = 0.50,
    expanded_share: float = 0.75,
) -> float:
    """Choose whether to expand group auditing after the base audit stage.

    The signal is available after fitting the base sparse-attribute model and
    computing validation group risk.  The default 0.20 threshold is frozen by
    the Phase-2 protocol before cross-dataset evaluation.
    """

    values = (mean_expected_group_risk, threshold, base_share, expanded_share)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("selector inputs must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not 0.0 < base_share < expanded_share < 1.0:
        raise ValueError("shares must satisfy 0 < base < expanded < 1")
    return expanded_share if mean_expected_group_risk >= threshold else base_share


def relative_weak_group_error_margin(group_error: np.ndarray) -> float:
    """Return the relative separation between the two highest group errors.

    The statistic is available from the soft groups inferred after an audit.
    A value near zero means that the identity of the estimated weak group is
    not separated from its nearest competitor.
    """

    error = np.asarray(group_error, dtype=np.float64)
    if error.ndim != 1 or len(error) < 2:
        raise ValueError("group_error must contain at least two values")
    if not np.isfinite(error).all() or (error < 0.0).any():
        raise ValueError("group errors must be finite and nonnegative")
    top_two = np.partition(error, -2)[-2:]
    highest = float(top_two.max())
    second = float(top_two.min())
    if highest <= 0.0:
        return 0.0
    return (highest - second) / highest


def select_staged_group_audit_share(
    group_error: np.ndarray,
    *,
    attribute_model_fitted: bool,
    relative_margin_threshold: float = 0.10,
    scout_share: float = 0.25,
    expanded_share: float = 0.50,
) -> float:
    """Choose whether to expand a staged group-attribute audit.

    The scout audit has already been paid for, so the conservative action is
    to stop at ``scout_share`` rather than pretend that its cost can be
    reclaimed for an all-label policy.  Expansion requires both a genuine
    two-class attribute fit and a separated soft weak-group estimate.
    """

    values = (relative_margin_threshold, scout_share, expanded_share)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("selector inputs must be finite")
    if not 0.0 <= relative_margin_threshold <= 1.0:
        raise ValueError("relative_margin_threshold must lie in [0, 1]")
    if not 0.0 < scout_share < expanded_share < 1.0:
        raise ValueError("shares must satisfy 0 < scout < expanded < 1")
    margin = relative_weak_group_error_margin(group_error)
    if bool(attribute_model_fitted) and margin >= relative_margin_threshold:
        return expanded_share
    return scout_share


def estimate_staged_marginal_value(
    subscout_ranking: np.ndarray,
    scout_ranking: np.ndarray,
    scout_score: np.ndarray,
    *,
    expanded_label_count: int,
    scout_label_count: int,
    observed_audit_increment: int,
    next_audit_increment: int,
) -> MarginalValueEstimate:
    """Compare extrapolated ranking information with forgone label value.

    Both rankings are evaluated using the later scout score.  The information
    term is therefore the score recovered by the scout ranking relative to the
    nested subscout ranking at the expanded policy's label count.  The cost is
    the scout-score mass of labels between the expanded and scout counts.
    """

    score = np.asarray(scout_score, dtype=np.float64)
    early = np.asarray(subscout_ranking, dtype=np.int64)
    later = np.asarray(scout_ranking, dtype=np.int64)
    if score.ndim != 1 or not np.isfinite(score).all() or (score < 0.0).any():
        raise ValueError("scout_score must be finite, nonnegative, and one-dimensional")
    if early.ndim != 1 or later.ndim != 1:
        raise ValueError("rankings must be one-dimensional")
    if len(early) < scout_label_count or len(later) < scout_label_count:
        raise ValueError("rankings are shorter than scout_label_count")
    if not 0 < expanded_label_count < scout_label_count <= len(score):
        raise ValueError("label counts must satisfy 0 < expanded < scout <= n")
    if observed_audit_increment <= 0 or next_audit_increment <= 0:
        raise ValueError("audit increments must be positive")
    for name, ranking in (("subscout", early), ("scout", later)):
        prefix = ranking[:scout_label_count]
        if ((prefix < 0) | (prefix >= len(score))).any():
            raise ValueError(f"{name} ranking contains an out-of-range index")
        if len(np.unique(prefix)) != len(prefix):
            raise ValueError(f"{name} ranking prefix contains duplicate indices")

    later_value = float(score[later[:expanded_label_count]].sum())
    early_value = float(score[early[:expanded_label_count]].sum())
    revealed_gain = max(later_value - early_value, 0.0)
    scale = float(next_audit_increment / observed_audit_increment)
    extrapolated_gain = scale * revealed_gain
    opportunity_cost = float(
        score[later[expanded_label_count:scout_label_count]].sum()
    )
    net_value = extrapolated_gain - opportunity_cost
    if opportunity_cost > 0.0:
        value_ratio = extrapolated_gain / opportunity_cost
    elif extrapolated_gain > 0.0:
        value_ratio = float("inf")
    else:
        value_ratio = 0.0
    return MarginalValueEstimate(
        revealed_information_gain=revealed_gain,
        extrapolation_scale=scale,
        extrapolated_information_gain=extrapolated_gain,
        label_opportunity_cost=opportunity_cost,
        net_value=net_value,
        value_ratio=value_ratio,
    )


def select_marginal_value_audit_share(
    estimate: MarginalValueEstimate,
    *,
    attribute_model_fitted: bool,
    scout_share: float = 0.25,
    expanded_share: float = 0.50,
) -> float:
    """Expand only for strictly positive estimated net marginal value."""

    if not 0.0 < scout_share < expanded_share < 1.0:
        raise ValueError("shares must satisfy 0 < scout < expanded < 1")
    if bool(attribute_model_fitted) and estimate.net_value > 0.0:
        return expanded_share
    return scout_share


def estimate_feedback_group_noise_rates(
    queried_group_probability: np.ndarray,
    queried_label_error: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Estimate soft-group correction rates from paid label feedback."""

    probability = np.asarray(queried_group_probability, dtype=np.float64)
    error = np.asarray(queried_label_error, dtype=np.float64)
    if probability.ndim != 2 or error.shape != (len(probability),):
        raise ValueError("queried group probabilities and errors must align")
    if len(probability) == 0:
        raise ValueError("at least one queried label is required")
    if not np.isfinite(probability).all() or (probability < 0.0).any():
        raise ValueError("group probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("group probabilities must sum to one")
    if not np.isfinite(error).all() or ((error < 0.0) | (error > 1.0)).any():
        raise ValueError("queried_label_error must lie in [0, 1]")
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be finite and positive")

    soft_mass = probability.sum(axis=0)
    corrected_mass = probability.T @ error
    return (alpha + corrected_mass) / (alpha + beta + soft_mass)


def feedback_updated_joint_scores(
    noise_score: np.ndarray,
    train_group_probability: np.ndarray,
    group_error: np.ndarray,
    feedback_noise_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update joint repair scores using first-batch correction feedback."""

    noise = np.asarray(noise_score, dtype=np.float64)
    probability = np.asarray(train_group_probability, dtype=np.float64)
    error = np.asarray(group_error, dtype=np.float64)
    rate = np.asarray(feedback_noise_rate, dtype=np.float64)
    if probability.ndim != 2 or noise.shape != (len(probability),):
        raise ValueError("noise score and group probabilities must align")
    if error.shape != (probability.shape[1],) or rate.shape != error.shape:
        raise ValueError("group error and feedback rate must match group columns")
    if not np.isfinite(noise).all() or (noise < 0.0).any():
        raise ValueError("noise scores must be finite and nonnegative")
    if not np.isfinite(probability).all() or (probability < 0.0).any():
        raise ValueError("group probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("group probabilities must sum to one")
    if not np.isfinite(error).all() or (error < 0.0).any():
        raise ValueError("group errors must be finite and nonnegative")
    if not np.isfinite(rate).all() or ((rate < 0.0) | (rate > 1.0)).any():
        raise ValueError("feedback noise rates must lie in [0, 1]")

    expected_feedback_value = probability @ (error * rate)
    return noise * np.maximum(expected_feedback_value, 1e-12), expected_feedback_value


def select_soft_group_stratified_exploration(
    group_probability: np.ndarray,
    count: int,
    *,
    seed: int,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample uniformly within argmax-soft-group strata with known support."""

    probability = np.asarray(group_probability, dtype=np.float64)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("group_probability must be a finite matrix")
    if (probability < 0.0).any() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("group probabilities must be nonnegative and sum to one")
    if not 0 <= count <= len(probability):
        raise ValueError("count must lie in [0, n]")
    excluded = np.zeros(len(probability), dtype=bool)
    if exclude is not None:
        indices = np.asarray(exclude, dtype=np.int64)
        if indices.ndim != 1 or ((indices < 0) | (indices >= len(probability))).any():
            raise ValueError("exclude contains an invalid index")
        excluded[indices] = True
    if count > int((~excluded).sum()):
        raise ValueError("count exceeds the eligible population")
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    hard_group = probability.argmax(axis=1)
    rng = np.random.default_rng(seed)
    queues: dict[int, list[int]] = {}
    eligible_size: dict[int, int] = {}
    for group in range(probability.shape[1]):
        members = np.flatnonzero((hard_group == group) & ~excluded).astype(np.int64)
        if len(members):
            rng.shuffle(members)
            queues[group] = members.tolist()
            eligible_size[group] = len(members)

    selected: list[int] = []
    selected_group: list[int] = []
    while len(selected) < count and queues:
        remaining: dict[int, list[int]] = {}
        for group in sorted(queues):
            queue = queues[group]
            if len(selected) >= count:
                remaining[group] = queue
                continue
            if queue:
                selected.append(int(queue.pop()))
                selected_group.append(group)
            if queue:
                remaining[group] = queue
        queues = remaining
    if len(selected) != count:
        raise RuntimeError("failed to fill exploration sample")

    quota = {
        group: selected_group.count(group) for group in set(selected_group)
    }
    inclusion = np.asarray(
        [quota[group] / eligible_size[group] for group in selected_group],
        dtype=np.float64,
    )
    return np.asarray(selected, dtype=np.int64), inclusion


def estimate_ipw_feedback_group_noise_rates(
    queried_group_probability: np.ndarray,
    queried_label_error: np.ndarray,
    inclusion_probability: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Estimate population soft-group correction rates by inverse weighting."""

    probability = np.asarray(queried_group_probability, dtype=np.float64)
    error = np.asarray(queried_label_error, dtype=np.float64)
    inclusion = np.asarray(inclusion_probability, dtype=np.float64)
    if probability.ndim != 2 or error.shape != (len(probability),):
        raise ValueError("queried probabilities and errors must align")
    if inclusion.shape != (len(probability),):
        raise ValueError("inclusion_probability must align with queries")
    if len(probability) == 0:
        raise ValueError("at least one exploration query is required")
    if not np.isfinite(inclusion).all() or ((inclusion <= 0.0) | (inclusion > 1.0)).any():
        raise ValueError("inclusion probabilities must lie in (0, 1]")
    if not np.isfinite(probability).all() or (probability < 0.0).any():
        raise ValueError("group probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("group probabilities must sum to one")
    if not np.isfinite(error).all() or ((error < 0.0) | (error > 1.0)).any():
        raise ValueError("queried_label_error must lie in [0, 1]")
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be finite and positive")

    weighted_probability = probability / inclusion[:, None]
    soft_mass = weighted_probability.sum(axis=0)
    corrected_mass = weighted_probability.T @ error
    return (alpha + corrected_mass) / (alpha + beta + soft_mass)
