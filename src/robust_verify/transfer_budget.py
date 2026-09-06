from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VALIDATION_METRICS = (
    "balanced_accuracy",
    "min_class_accuracy",
    "class_accuracy_gap",
    "ce_mean",
    "ce_q90",
    "class_cvar10_max",
    "entropy_mean",
    "margin_q10",
)

FORBIDDEN_FEATURE_TOKENS = (
    "dataset",
    "noise_name",
    "corruption_name",
    "group",
    "test",
    "wga",
    "unqueried_clean",
)


@dataclass(frozen=True)
class SelectiveDecision:
    action: str
    median_probability: float
    lower_probability: float
    upper_probability: float


def _as_probability_matrix(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("probabilities and labels must have matching rows")
    if len(values) == 0 or values.shape[1] < 2:
        raise ValueError("at least one row and two classes are required")
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("probabilities and labels must be finite")
    if (targets < 0).any() or (targets >= values.shape[1]).any():
        raise ValueError("labels must index probability columns")
    row_sums = values.sum(axis=1)
    if (values < 0.0).any() or not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("rows must contain non-negative probabilities summing to one")
    return values, targets


def validation_summary(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return group-free, scale-stable descriptors of a validation prediction set."""

    values, targets = _as_probability_matrix(probabilities, labels)
    clipped = np.clip(values, 1e-12, 1.0)
    predictions = values.argmax(axis=1)
    losses = -np.log(clipped[np.arange(len(targets)), targets])
    entropies = -(clipped * np.log(clipped)).sum(axis=1) / np.log(values.shape[1])
    ordered = np.sort(values, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]

    class_accuracies: list[float] = []
    class_cvars: list[float] = []
    for class_value in sorted(np.unique(targets).tolist()):
        mask = targets == class_value
        class_accuracies.append(float((predictions[mask] == targets[mask]).mean()))
        class_losses = np.sort(losses[mask])
        tail_count = max(1, int(np.ceil(0.10 * len(class_losses))))
        class_cvars.append(float(class_losses[-tail_count:].mean()))

    return {
        "balanced_accuracy": float(np.mean(class_accuracies)),
        "min_class_accuracy": float(np.min(class_accuracies)),
        "class_accuracy_gap": float(np.max(class_accuracies) - np.min(class_accuracies)),
        "ce_mean": float(losses.mean()),
        "ce_q90": float(np.quantile(losses, 0.90)),
        "class_cvar10_max": float(np.max(class_cvars)),
        "entropy_mean": float(entropies.mean()),
        "margin_q10": float(np.quantile(margins, 0.10)),
    }


def feedback_summary(
    noisy_labels: np.ndarray,
    returned_labels: np.ndarray,
    *,
    previous_count: int,
) -> dict[str, float]:
    """Summarize only labels returned in the purchased ordered prefix."""

    noisy = np.asarray(noisy_labels, dtype=np.int64).reshape(-1)
    returned = np.asarray(returned_labels, dtype=np.int64).reshape(-1)
    if len(noisy) != len(returned) or len(noisy) == 0:
        raise ValueError("queried noisy and returned labels must have equal non-zero length")
    if previous_count < 0 or previous_count >= len(noisy):
        raise ValueError("previous_count must identify a strict preceding prefix")

    changed = returned != noisy
    recent_noisy = noisy[previous_count:]
    recent_changed = changed[previous_count:]

    def correction_gap(observed: np.ndarray, flags: np.ndarray) -> float:
        rates = [float(flags[observed == value].mean()) for value in np.unique(observed)]
        return float(max(rates) - min(rates)) if len(rates) > 1 else 0.0

    def normalized_entropy(values: np.ndarray) -> float:
        _, counts = np.unique(values, return_counts=True)
        if len(counts) <= 1:
            return 0.0
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))

    previous_yield = float(changed[:previous_count].mean()) if previous_count else float(changed.mean())
    recent_yield = float(recent_changed.mean())
    return {
        "feedback_cumulative_correction_rate": float(changed.mean()),
        "feedback_recent_correction_rate": recent_yield,
        "feedback_yield_change": float(recent_yield - previous_yield),
        "feedback_cumulative_class_gap": correction_gap(noisy, changed),
        "feedback_recent_class_gap": correction_gap(recent_noisy, recent_changed),
        "feedback_returned_label_entropy": normalized_entropy(returned),
        "feedback_observed_label_entropy": normalized_entropy(noisy),
    }


def _prediction_drift(current: np.ndarray, previous: np.ndarray) -> tuple[float, float]:
    current_values = np.asarray(current, dtype=np.float64)
    previous_values = np.asarray(previous, dtype=np.float64)
    if current_values.shape != previous_values.shape:
        raise ValueError("current and previous probabilities must have equal shape")
    changed = current_values.argmax(axis=1) != previous_values.argmax(axis=1)
    l1 = np.abs(current_values - previous_values).sum(axis=1) / 2.0
    return float(changed.mean()), float(l1.mean())


def build_legal_transition_features(
    *,
    current_budget: float,
    next_budget: float,
    calibration_current: np.ndarray,
    calibration_previous: np.ndarray,
    calibration_labels: np.ndarray,
    audit_current: np.ndarray,
    audit_previous: np.ndarray,
    audit_labels: np.ndarray,
    feedback: Mapping[str, float],
) -> dict[str, float]:
    """Build the frozen legal feature vector available before buying `next_budget`."""

    current_budget = float(current_budget)
    next_budget = float(next_budget)
    if current_budget <= 0.0 or next_budget <= current_budget or next_budget > 1.0:
        raise ValueError("budgets must satisfy 0 < current_budget < next_budget <= 1")

    cal_current = validation_summary(calibration_current, calibration_labels)
    cal_previous = validation_summary(calibration_previous, calibration_labels)
    audit_current_summary = validation_summary(audit_current, audit_labels)
    audit_previous_summary = validation_summary(audit_previous, audit_labels)

    features: dict[str, float] = {
        "current_budget_fraction": current_budget,
        "next_budget_fraction": next_budget,
        "incremental_budget_fraction": next_budget - current_budget,
        "budget_ratio": next_budget / current_budget,
    }
    for name in VALIDATION_METRICS:
        current_mean = 0.5 * (cal_current[name] + audit_current_summary[name])
        previous_mean = 0.5 * (cal_previous[name] + audit_previous_summary[name])
        cal_delta = cal_current[name] - cal_previous[name]
        audit_delta = audit_current_summary[name] - audit_previous_summary[name]
        features[f"validation_{name}"] = float(current_mean)
        features[f"validation_delta_{name}"] = float(current_mean - previous_mean)
        features[f"validation_role_gap_{name}"] = float(
            abs(cal_current[name] - audit_current_summary[name])
        )
        features[f"validation_delta_role_gap_{name}"] = float(abs(cal_delta - audit_delta))

    cal_flip, cal_l1 = _prediction_drift(calibration_current, calibration_previous)
    audit_flip, audit_l1 = _prediction_drift(audit_current, audit_previous)
    features.update(
        {
            "validation_prediction_flip_rate": 0.5 * (cal_flip + audit_flip),
            "validation_probability_l1_drift": 0.5 * (cal_l1 + audit_l1),
            "validation_prediction_flip_role_gap": abs(cal_flip - audit_flip),
            "validation_probability_l1_role_gap": abs(cal_l1 - audit_l1),
        }
    )
    features.update({str(name): float(value) for name, value in feedback.items()})
    assert_legal_feature_names(tuple(features))
    if not np.isfinite(np.fromiter(features.values(), dtype=np.float64)).all():
        raise ValueError("legal transition features must be finite")
    return features


def assert_legal_feature_names(names: Sequence[str]) -> None:
    lowered = [str(name).lower() for name in names]
    collisions = sorted(
        name
        for name in lowered
        if any(token in name for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if collisions:
        raise ValueError(f"forbidden controller feature names: {collisions}")
    if len(set(lowered)) != len(lowered):
        raise ValueError("controller feature names must be unique")


def fit_predict_seed_block_bootstrap(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: Sequence[str],
    test_features: np.ndarray,
    *,
    replicates: int = 200,
    seed: int = 20260811,
) -> np.ndarray:
    """Fit deterministic L2-logistic bootstrap models at whole-trajectory granularity."""

    x_train = np.asarray(train_features, dtype=np.float64)
    y_train = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    x_test = np.asarray(test_features, dtype=np.float64)
    groups = np.asarray(train_groups, dtype=str).reshape(-1)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test features must be two-dimensional with equal columns")
    if len(x_train) != len(y_train) or len(groups) != len(y_train):
        raise ValueError("train rows, labels, and groups must have equal length")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("train_labels must be binary")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("model features must be finite")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two trajectory groups are required")
    row_lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(int(seed))
    predictions = np.empty((replicates, len(x_test)), dtype=np.float64)

    for replicate in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_rows = np.concatenate([row_lookup[group] for group in sampled_groups])
        sampled_y = y_train[sampled_rows]
        if len(np.unique(sampled_y)) == 1:
            predictions[replicate] = (float(sampled_y.sum()) + 1.0) / (len(sampled_y) + 2.0)
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=int(seed) + replicate,
                    ),
                ),
            ]
        )
        model.fit(x_train[sampled_rows], sampled_y)
        predictions[replicate] = model.predict_proba(x_test)[:, 1]
    return predictions


def selective_decisions(
    probability_samples: np.ndarray,
    *,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    continue_threshold: float = 0.60,
    stop_threshold: float = 0.40,
) -> list[SelectiveDecision]:
    samples = np.asarray(probability_samples, dtype=np.float64)
    if samples.ndim != 2 or len(samples) == 0:
        raise ValueError("probability_samples must have shape [replicates, rows]")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if not 0.0 <= stop_threshold < continue_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= stop < continue <= 1")
    if (samples < 0.0).any() or (samples > 1.0).any():
        raise ValueError("probability samples must lie in [0, 1]")

    lower = np.quantile(samples, lower_quantile, axis=0)
    upper = np.quantile(samples, upper_quantile, axis=0)
    median = np.median(samples, axis=0)
    decisions: list[SelectiveDecision] = []
    for center, low, high in zip(median, lower, upper):
        if low >= continue_threshold:
            action = "continue"
        elif high <= stop_threshold:
            action = "stop"
        else:
            action = "abstain"
        decisions.append(
            SelectiveDecision(
                action=action,
                median_probability=float(center),
                lower_probability=float(low),
                upper_probability=float(high),
            )
        )
    return decisions


def balanced_accuracy_present_classes(labels: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if len(truth) != len(predicted) or len(truth) == 0:
        raise ValueError("labels and predictions must have equal non-zero length")
    recalls = [float((predicted[truth == value] == value).mean()) for value in np.unique(truth)]
    return float(np.mean(recalls))
