"""Stronger legal baselines for frozen-feature label-verification audits.

The AUTO-D3M query method below is an explicit action-space adaptation: the
published method removes negatively aligned training examples, whereas this
project can only query and correct labels.  We therefore multiply its legal
attribution-alignment rank by a label-error proxy.  It must be reported as an
adaptation, not as a reproduction of the original 100-trial end-to-end method.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robust_verify.influence import compute_tracin_influence, uncertainty_weights
from robust_verify.utils import entropy_from_probabilities, normalized_rank


def activeclean_expected_model_change_scores(
    *,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    """Score examples by a frozen-linear-head expected model change proxy.

    ActiveClean is adapted here to the one-shot correction setting.  The
    expected correction probability is ``1 - p(noisy_label)`` and the norm of
    the label-flip gradient is proportional to the frozen feature norm.  The
    score is legal: it uses only the noisy label, probe probabilities, and
    public frozen features.  It is deliberately reported as an adaptation,
    not as a reproduction of the original interactive ActiveClean system.
    """
    features = np.asarray(train_features, dtype=np.float64)
    probabilities = np.asarray(train_probabilities, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if features.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("features and probabilities must be two-dimensional")
    if len(features) != len(probabilities) or len(labels) != len(features):
        raise ValueError("train arrays have incompatible lengths")
    if probabilities.shape[1] < 2:
        raise ValueError("at least two classes are required")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= probabilities.shape[1]:
        raise ValueError("train labels are outside the probability columns")

    observed_probability = probabilities[np.arange(len(labels)), labels]
    expected_flip_probability = 1.0 - np.clip(observed_probability, 0.0, 1.0)
    feature_norm = np.linalg.norm(features, axis=1)
    # For a softmax head, ||e_y - e_c|| is sqrt(2) for any label flip.
    score = expected_flip_probability * feature_norm * np.sqrt(2.0)
    return normalized_rank(score)


def active_label_cleaning_scores(
    *,
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    """Return a one-shot Active Label Cleaning (CE - entropy) score.

    The original active-cleaning workflow is adapted to this correction-only,
    one-shot audit.  Higher cross-entropy and lower predictive entropy receive
    higher priority.  Both terms are standardized by percentile rank before
    subtraction so their relative scale cannot depend on the feature model.
    """
    probabilities = np.asarray(train_probabilities, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels have incompatible shapes")
    if probabilities.shape[1] < 2:
        raise ValueError("at least two classes are required")
    losses = -np.log(
        np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    entropy = entropy_from_probabilities(probabilities)
    # Higher CE and lower entropy are more suspicious.
    return normalized_rank(losses) - normalized_rank(entropy)


def oof_cleanlab_scores(
    *,
    oof_train_probabilities: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    """Return OOF Cleanlab label-quality scores as query priorities.

    Cleanlab's confident-learning score is computed from probabilities that
    were generated without fitting on the corresponding example.  The caller
    must provide a complete, finite OOF matrix; this prevents accidental use of
    the in-sample probe under the ``oof_cleanlab`` method ID.
    """
    probabilities = np.asarray(oof_train_probabilities, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("OOF probabilities and labels have incompatible shapes")
    if probabilities.shape[1] < 2 or not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities must be finite with at least two classes")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= probabilities.shape[1]:
        raise ValueError("train labels are outside the OOF probability columns")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("OOF probabilities must sum to one per row")

    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as exc:
        raise ImportError(
            "oof_cleanlab requires cleanlab>=2.7; install server requirements"
        ) from exc
    quality = get_label_quality_scores(labels, pred_probs=probabilities)
    # Cleanlab quality is higher for likely-correct labels.  Query priority is
    # therefore the reverse rank, with CE as a deterministic tie-breaker later.
    return normalized_rank(-np.asarray(quality, dtype=np.float64))


def multicheckpoint_tracin_influence(
    *,
    train_features: np.ndarray,
    train_probability_history: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probability_history: np.ndarray,
    val_labels: np.ndarray,
) -> np.ndarray:
    """Average legal TracIn-CP (val) influence across all epoch checkpoints."""
    train_history = np.asarray(train_probability_history)
    val_history = np.asarray(val_probability_history)
    if train_history.ndim != 3 or val_history.ndim != 3:
        raise ValueError("probability histories must have shape [n, checkpoints, classes]")
    if train_history.shape[1:] != val_history.shape[1:]:
        raise ValueError("train and validation histories must share checkpoints and classes")
    if train_history.shape[0] != len(train_labels):
        raise ValueError("train history and labels have different lengths")
    if val_history.shape[0] != len(val_labels):
        raise ValueError("validation history and labels have different lengths")
    checkpoints = train_history.shape[1]
    if checkpoints == 0:
        raise ValueError("at least one checkpoint is required")

    # The binary case admits an exact batched form.  For softmax with two
    # classes, residual[:, 0] == -residual[:, 1], hence every checkpoint only
    # needs one feature-space projection.  This avoids repeatedly scanning the
    # full CelebA feature matrix in a Python loop.
    if train_history.shape[2] == 2:
        val_x = np.asarray(val_features)
        val_y = np.asarray(val_labels, dtype=np.int64)
        gradient_differences = []
        for checkpoint in range(checkpoints):
            val_probs = val_history[:, checkpoint, :]
            val_residual = val_probs.copy()
            val_residual[np.arange(len(val_y)), val_y] -= 1.0
            weights = uncertainty_weights(val_probs)
            validation_gradient = (val_residual * weights[:, None]).T @ val_x
            gradient_differences.append(
                (validation_gradient[1] - validation_gradient[0]).astype(np.float32)
            )
        gradient_matrix = np.stack(gradient_differences, axis=0)
        projections = np.asarray(train_features, dtype=np.float32) @ gradient_matrix.T
        train_y = np.asarray(train_labels, dtype=np.int64)
        residual_one = train_history[:, :, 1] - (train_y == 1)[:, None]
        return -np.mean(residual_one * projections, axis=1, dtype=np.float64)

    scores = np.zeros(len(train_labels), dtype=np.float64)
    for checkpoint in range(checkpoints):
        val_probs = val_history[:, checkpoint, :]
        scores += compute_tracin_influence(
            train_features=train_features,
            train_probs=train_history[:, checkpoint, :],
            train_labels=train_labels,
            val_features=val_features,
            val_probs=val_probs,
            val_labels=val_labels,
            val_weights=uncertainty_weights(val_probs),
        )
    return scores / checkpoints


def class_conditional_tail_weights(
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    *,
    tail_fraction: float = 0.20,
) -> np.ndarray:
    """Equal-class weights on the highest clean-validation CE losses."""
    probabilities = np.asarray(val_probabilities, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("validation probabilities and labels have incompatible shapes")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1]")
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    classes = np.unique(labels)
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in classes:
        positions = np.flatnonzero(labels == label)
        count = max(1, int(np.ceil(tail_fraction * len(positions))))
        tail = positions[np.argsort(losses[positions], kind="mergesort")[-count:]]
        weights[tail] = 1.0 / (len(classes) * count)
    return weights


def repair_value_rank_scores(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    tail_fraction: float = 0.20,
) -> np.ndarray:
    """Return the normalized first-order tail-loss repair-value component."""
    train_x = np.asarray(train_features)
    val_x = np.asarray(val_features)
    labels = np.asarray(train_labels, dtype=np.int64)
    probabilities = np.asarray(val_probabilities, dtype=np.float64)
    if probabilities.shape[1] != 2 or set(np.unique(labels)).difference({0, 1}):
        raise ValueError("repair value currently supports binary labels only")
    if len(train_x) != len(labels):
        raise ValueError("train arrays have different lengths")
    weights = class_conditional_tail_weights(
        probabilities, val_labels, tail_fraction=tail_fraction
    )
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(val_labels)), np.asarray(val_labels, dtype=np.int64)] = 1.0
    validation_gradient = ((probabilities - onehot) * weights[:, None]).T @ val_x

    # gradient(corrected label) - gradient(observed label) equals
    # onehot(observed) - onehot(corrected), independent of train probabilities.
    # In the binary case a single matrix-vector product is sufficient and,
    # unlike indexing validation_gradient by every label, does not materialise
    # an [n_train, feature_dim] float64 temporary (several GB on CelebA).
    class_zero_value = train_x @ (
        validation_gradient[0] - validation_gradient[1]
    ).astype(np.float32)
    correction_value = np.where(labels == 0, class_zero_value, -class_zero_value)
    return normalized_rank(correction_value)


def multiclass_repair_value_rank_scores(
    *,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    tail_fraction: float = 0.20,
) -> np.ndarray:
    """Return a soft-posterior repair-value rank for C-class labels.

    For an observed label ``t``, the candidate correction to class ``c`` has
    first-order value ``(g_t - g_c)^T phi(x)``.  The unknown clean class is
    marginalised using the probe posterior after removing the observed class.
    With two classes this reduces exactly to ``repair_value_rank_scores``.
    """
    train_x = np.asarray(train_features, dtype=np.float32)
    train_probs = np.asarray(train_probabilities, dtype=np.float64)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.asarray(val_features, dtype=np.float32)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.int64)

    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features must be compatible matrices")
    if train_probs.ndim != 2 or val_probs.ndim != 2:
        raise ValueError("train and validation probabilities must be matrices")
    if train_probs.shape[1] != val_probs.shape[1] or train_probs.shape[1] < 2:
        raise ValueError("train and validation probabilities must share at least two classes")
    if len(train_x) != len(train_y) or len(train_x) != len(train_probs):
        raise ValueError("training arrays have incompatible lengths")
    if len(val_x) != len(val_y) or len(val_x) != len(val_probs):
        raise ValueError("validation arrays have incompatible lengths")
    num_classes = int(train_probs.shape[1])
    if train_y.min(initial=0) < 0 or train_y.max(initial=0) >= num_classes:
        raise ValueError("train labels are outside the probability columns")
    if val_y.min(initial=0) < 0 or val_y.max(initial=0) >= num_classes:
        raise ValueError("validation labels are outside the probability columns")
    if not np.all(np.isfinite(train_probs)) or not np.all(np.isfinite(val_probs)):
        raise ValueError("probabilities must be finite")

    weights = class_conditional_tail_weights(
        val_probs, val_y, tail_fraction=tail_fraction
    )
    onehot = np.zeros_like(val_probs)
    onehot[np.arange(len(val_y)), val_y] = 1.0
    validation_gradient = ((val_probs - onehot) * weights[:, None]).T @ val_x

    # Use the model posterior as a legal clean-label posterior proxy.  The
    # observed noisy class is excluded, then the remaining mass is renormalized.
    posterior = np.clip(train_probs, 0.0, None).copy()
    posterior[np.arange(len(train_y)), train_y] = 0.0
    normalizer = posterior.sum(axis=1, keepdims=True)
    posterior = np.divide(
        posterior,
        normalizer,
        out=np.full_like(posterior, 1.0 / (num_classes - 1)),
        where=normalizer > 1e-12,
    )

    # ``candidate_values[:, c]`` is the value if the noisy label is repaired
    # to class c.  The matrix product keeps memory linear in N*d plus N*C.
    candidate_values = np.empty((len(train_x), num_classes), dtype=np.float64)
    for cls in range(num_classes):
        candidate_direction = validation_gradient[train_y] - validation_gradient[cls]
        candidate_values[:, cls] = np.einsum("nd,nd->n", train_x, candidate_direction)
    candidate_values[np.arange(len(train_y)), train_y] = 0.0
    expected_value = np.sum(posterior * candidate_values, axis=1)
    return normalized_rank(expected_value)


def expected_repair_value_multiclass_scores(
    *,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    tail_fraction: float = 0.20,
) -> np.ndarray:
    """Combine a legal noise proxy with the C-class repair-value rank."""
    proxy = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if len(proxy) != len(train_labels):
        raise ValueError("train arrays and noise proxy have different lengths")
    value_rank = multiclass_repair_value_rank_scores(
        train_features=train_features,
        train_probabilities=train_probabilities,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        tail_fraction=tail_fraction,
    )
    return np.clip(proxy, 0.0, 1.0) * value_rank


def reliability_gated_repair_value_scores(
    *,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    tail_fraction: float = 0.20,
    num_folds: int = 5,
    min_stability: float = 0.50,
    full_stability: float = 0.85,
    reference_budget_count: int | None = None,
    min_noise_retention: float = 0.75,
    full_noise_retention: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Blend RV-Q with NoiseScore using validation-only signal reliability.

    Reliability combines rank agreement of the repair-value signal across
    stratified validation folds with retention of the detector signal in the
    proposed candidate prefix.  The gate is deliberately independent of test
    WGA, hidden groups, and clean training labels.  Low reliability falls back
    continuously to the detector, while a stable, detector-compatible value
    signal receives full weight.
    """
    if not 0.0 <= min_stability <= full_stability <= 1.0:
        raise ValueError("stability thresholds must satisfy 0 <= min <= full <= 1")
    if not 0.0 <= min_noise_retention <= full_noise_retention <= 1.0:
        raise ValueError("noise-retention thresholds must satisfy 0 <= min <= full <= 1")
    train_x = np.asarray(train_features)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.asarray(val_features)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    noise = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if len(noise) != len(train_y):
        raise ValueError("noise proxy and training labels have different lengths")

    folds = _stratified_validation_folds(val_y, num_folds=num_folds, seed=seed)
    fold_scores: list[np.ndarray] = []
    for held_out in folds:
        fit_mask = np.ones(len(val_y), dtype=bool)
        fit_mask[held_out] = False
        if not fit_mask.any():
            continue
        fold_scores.append(
            multiclass_repair_value_rank_scores(
                train_features=train_x,
                train_probabilities=np.asarray(train_probabilities),
                train_labels=train_y,
                val_features=val_x[fit_mask],
                val_probabilities=val_probs[fit_mask],
                val_labels=val_y[fit_mask],
                tail_fraction=tail_fraction,
            )
        )
    if len(fold_scores) < 2:
        stability = 0.0
    else:
        correlations = [
            _spearman_correlation(fold_scores[i], fold_scores[j])
            for i in range(len(fold_scores))
            for j in range(i + 1, len(fold_scores))
        ]
        stability = float(np.mean(correlations)) if correlations else 0.0

    stability_alpha = float(np.clip(
        (stability - min_stability) / max(full_stability - min_stability, 1e-12),
        0.0,
        1.0,
    ))
    rv_score = expected_repair_value_multiclass_scores(
        train_features=train_x,
        train_probabilities=np.asarray(train_probabilities),
        train_labels=train_y,
        val_features=val_x,
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=noise,
        tail_fraction=tail_fraction,
    )
    noise_clip = np.clip(noise, 0.0, 1.0)
    rv_order = np.lexsort((np.arange(len(rv_score)), -rv_score))
    noise_order = np.lexsort((np.arange(len(noise_clip)), -noise_clip))
    if reference_budget_count is None:
        reference_budget_count = max(1, int(np.ceil(0.05 * len(noise_clip))))
    reference_budget_count = int(np.clip(reference_budget_count, 1, len(noise_clip)))
    rv_noise_mean = float(np.mean(noise_clip[rv_order[:reference_budget_count]]))
    noise_top_mean = float(np.mean(noise_clip[noise_order[:reference_budget_count]]))
    noise_retention = rv_noise_mean / max(noise_top_mean, 1e-12)
    retention_alpha = float(np.clip(
        (noise_retention - min_noise_retention)
        / max(full_noise_retention - min_noise_retention, 1e-12),
        0.0,
        1.0,
    ))
    alpha = min(stability_alpha, retention_alpha)
    blended = alpha * rv_score + (1.0 - alpha) * noise_clip
    return blended, {
        "gate_reliability": stability,
        "gate_alpha": alpha,
        "gate_stability_alpha": stability_alpha,
        "gate_retention_alpha": retention_alpha,
        "gate_noise_retention": noise_retention,
        "gate_reference_budget_count": reference_budget_count,
        "gate_fallback": bool(alpha == 0.0),
        "gate_num_folds": len(fold_scores),
        "gate_min_stability": float(min_stability),
        "gate_full_stability": float(full_stability),
        "gate_min_noise_retention": float(min_noise_retention),
        "gate_full_noise_retention": float(full_noise_retention),
        "gate_tail_fraction": float(tail_fraction),
    }


def expected_repair_value_scores(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    tail_fraction: float = 0.20,
) -> np.ndarray:
    """Rank v1 expected repair value by an unconstrained product score."""
    proxy = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if len(proxy) != len(train_labels):
        raise ValueError("train arrays have different lengths")
    value_rank = repair_value_rank_scores(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        tail_fraction=tail_fraction,
    )
    return np.clip(proxy, 0.0, 1.0) * value_rank


def _spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return a deterministic rank correlation without an optional scipy dependency."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("correlation inputs must be one-dimensional with equal shape")
    first = normalized_rank(first)
    second = normalized_rank(second)
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def _stratified_validation_folds(
    labels: np.ndarray,
    *,
    num_folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Create deterministic, approximately class-balanced validation folds."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("validation labels must contain at least two rows")
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    num_folds = min(int(num_folds), len(labels))
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(num_folds)]
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        shuffled = positions[rng.permutation(len(positions))]
        for offset, position in enumerate(shuffled):
            buckets[offset % num_folds].append(int(position))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets if bucket]


def select_adaptive_tail_fraction(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    candidate_tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    stability_sample_size: int = 4096,
    seed: int = 0,
) -> tuple[float, dict[str, object]]:
    """Select tau from validation-only tail and ranking stability diagnostics.

    The candidate is chosen without clean training labels, groups, or test outcomes.
    A small deterministic training subset is sufficient for the stability diagnostic;
    the selected candidate is later used to score the full training set.
    """
    train_x = np.asarray(train_features)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.asarray(val_features)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features must be compatible matrices")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("feature and label arrays have incompatible lengths")
    if val_probs.ndim != 2 or len(val_probs) != len(val_y) or val_probs.shape[1] != 2:
        raise ValueError("adaptive tail selection currently supports binary validation probabilities")
    candidates = tuple(sorted(set(float(value) for value in candidate_tail_fractions)))
    if not candidates or any(not 0.0 < value <= 1.0 for value in candidates):
        raise ValueError("candidate_tail_fractions must lie in (0, 1]")
    if min_tail_count < 1 or stability_sample_size < 1:
        raise ValueError("min_tail_count and stability_sample_size must be positive")

    folds = _stratified_validation_folds(val_y, num_folds=num_folds, seed=seed)
    rng = np.random.default_rng(seed + 7919)
    if len(train_x) > stability_sample_size:
        sample_indices = np.sort(rng.choice(len(train_x), size=stability_sample_size, replace=False))
    else:
        sample_indices = np.arange(len(train_x), dtype=np.int64)
    sample_x = train_x[sample_indices]
    sample_y = train_y[sample_indices]
    val_losses = -np.log(np.clip(val_probs[np.arange(len(val_y)), val_y], 1e-12, 1.0))

    diagnostics: list[dict[str, object]] = []
    for tau in candidates:
        fold_scores: list[np.ndarray] = []
        contrast_values: list[float] = []
        min_fold_tail = np.iinfo(np.int64).max
        valid = True
        for held_out in folds:
            fit_mask = np.ones(len(val_y), dtype=bool)
            fit_mask[held_out] = False
            fit_labels = val_y[fit_mask]
            if len(fit_labels) == 0:
                valid = False
                break
            counts = [int(np.sum(fit_labels == label)) for label in np.unique(val_y)]
            fold_tail_count = min(max(1, int(np.ceil(tau * count))) for count in counts)
            min_fold_tail = min(min_fold_tail, fold_tail_count)
            if fold_tail_count < min_tail_count:
                valid = False
            fold_scores.append(
                repair_value_rank_scores(
                    train_features=sample_x,
                    train_labels=sample_y,
                    val_features=val_x[fit_mask],
                    val_probabilities=val_probs[fit_mask],
                    val_labels=fit_labels,
                    tail_fraction=tau,
                )
            )
            fit_losses = val_losses[fit_mask]
            tail_means = []
            for label in np.unique(fit_labels):
                positions = np.flatnonzero(fit_labels == label)
                count = max(1, int(np.ceil(tau * len(positions))))
                tail_means.append(float(np.mean(fit_losses[positions[np.argsort(fit_losses[positions])[-count:]]])))
            contrast_values.append(float(np.mean(tail_means) - np.mean(fit_losses)))

        if not fold_scores:
            valid = False
        if valid and len(fold_scores) >= 2:
            correlations = [
                _spearman_correlation(fold_scores[i], fold_scores[j])
                for i in range(len(fold_scores))
                for j in range(i + 1, len(fold_scores))
            ]
            stability = float(np.mean(correlations))
        else:
            stability = -1.0
        diagnostics.append(
            {
                "tau": tau,
                "valid": bool(valid),
                "min_fold_tail_count": int(min_fold_tail if min_fold_tail != np.iinfo(np.int64).max else 0),
                "rank_stability": stability,
                "tail_loss_contrast": float(np.mean(contrast_values)) if contrast_values else float("nan"),
            }
        )

    valid_rows = [row for row in diagnostics if bool(row["valid"])]
    if not valid_rows:
        # Keep the method defined for very small validation sets while exposing the fallback.
        valid_rows = diagnostics
    contrasts = np.asarray([float(row["tail_loss_contrast"]) for row in valid_rows], dtype=np.float64)
    finite_contrasts = np.where(np.isfinite(contrasts), contrasts, np.nanmin(contrasts[np.isfinite(contrasts)]) if np.isfinite(contrasts).any() else 0.0)
    if len(valid_rows) == 1 or np.ptp(finite_contrasts) == 0.0:
        contrast_ranks = np.zeros(len(valid_rows), dtype=np.float64)
    else:
        contrast_ranks = normalized_rank(finite_contrasts)
    for row, contrast_rank in zip(valid_rows, contrast_ranks):
        row["contrast_rank"] = float(contrast_rank)
        row["selection_score"] = float(0.8 * float(row["rank_stability"]) + 0.2 * contrast_rank)
    selected = max(valid_rows, key=lambda row: (float(row["selection_score"]), -float(row["tau"])))
    return float(selected["tau"]), {
        "selected_tau": float(selected["tau"]),
        "num_folds": len(folds),
        "min_tail_count": int(min_tail_count),
        "stability_sample_size": int(len(sample_indices)),
        "candidates": diagnostics,
    }


def expected_repair_value_adaptive_scores(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    candidate_tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    stability_sample_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return RV-Q scores using a validation-only adaptive tail fraction."""
    tau, metadata = select_adaptive_tail_fraction(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        candidate_tail_fractions=candidate_tail_fractions,
        num_folds=num_folds,
        min_tail_count=min_tail_count,
        stability_sample_size=stability_sample_size,
        seed=seed,
    )
    scores = expected_repair_value_scores(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        noise_posterior_proxy=noise_posterior_proxy,
        tail_fraction=tau,
    )
    return scores, metadata


def _binary_tail_gradient_contrast(
    *,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    tail_fraction: float,
) -> np.ndarray:
    """Return g_0 - g_1 for the class-balanced tail objective (binary labels)."""
    weights = class_conditional_tail_weights(
        val_probabilities, val_labels, tail_fraction=tail_fraction
    )
    onehot = np.zeros_like(val_probabilities)
    onehot[np.arange(len(val_labels)), np.asarray(val_labels, dtype=np.int64)] = 1.0
    gradient = ((val_probabilities - onehot) * weights[:, None]).T @ val_features
    return gradient[0] - gradient[1]


def select_tail_fraction_by_capture(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    budget_count: int,
    candidate_tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    stability_sample_size: int = 4096,
    seed: int = 0,
) -> tuple[float, dict[str, object]]:
    """Select tau by out-of-sample capture of the first-order repair effect.

    For each candidate tau and each stratified validation fold, the repair-value
    direction is fit on the remaining folds, a top-k query set is selected by
    the noise-weighted first-order score, and its predicted effect is measured
    under the held-out fold's own value signal.  The candidate score is the
    mean capture ratio ``E_hold / O_hold``, where ``O_hold`` is the effect of
    the held-out fold's own top-k set.  The criterion uses only legal pre-query
    observables, is not mechanically monotone in tau, and is budget-aware: the
    capture region k mirrors the actual query budget.
    """
    train_x = np.asarray(train_features)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.asarray(val_features)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    proxy_all = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features must be compatible matrices")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("feature and label arrays have incompatible lengths")
    if len(proxy_all) != len(train_x):
        raise ValueError("noise posterior proxy must align with the training set")
    if val_probs.ndim != 2 or len(val_probs) != len(val_y) or val_probs.shape[1] != 2:
        raise ValueError("capture selection currently supports binary validation probabilities")
    if set(np.unique(train_y)).difference({0, 1}):
        raise ValueError("capture selection currently supports binary training labels")
    if budget_count < 1:
        raise ValueError("budget_count must be positive")
    candidates = tuple(sorted(set(float(value) for value in candidate_tail_fractions)))
    if not candidates or any(not 0.0 < value <= 1.0 for value in candidates):
        raise ValueError("candidate_tail_fractions must lie in (0, 1]")
    if min_tail_count < 1 or stability_sample_size < 1:
        raise ValueError("min_tail_count and stability_sample_size must be positive")

    folds = _stratified_validation_folds(val_y, num_folds=num_folds, seed=seed)
    rng = np.random.default_rng(seed + 7919)
    num_train = len(train_x)
    if num_train > stability_sample_size:
        sample_indices = np.sort(
            rng.choice(num_train, size=stability_sample_size, replace=False)
        )
    else:
        sample_indices = np.arange(num_train, dtype=np.int64)
    sample_x = train_x[sample_indices]
    sample_sign = 1.0 - 2.0 * train_y[sample_indices]
    sample_proxy = np.clip(proxy_all[sample_indices], 0.0, 1.0)
    sample_count = len(sample_indices)
    capture_count = max(1, int(np.ceil(budget_count * sample_count / num_train)))

    diagnostics: list[dict[str, object]] = []
    for tau in candidates:
        fold_captures: list[float] = []
        fold_oracles: list[float] = []
        valid = True
        for held_out in folds:
            fit_mask = np.ones(len(val_y), dtype=bool)
            fit_mask[held_out] = False
            fit_labels = val_y[fit_mask]
            if len(fit_labels) == 0:
                valid = False
                break
            counts = [int(np.sum(fit_labels == label)) for label in np.unique(val_y)]
            fold_tail_count = min(max(1, int(np.ceil(tau * count))) for count in counts)
            if fold_tail_count < min_tail_count:
                valid = False
                break
            fit_direction = _binary_tail_gradient_contrast(
                val_features=val_x[fit_mask],
                val_probabilities=val_probs[fit_mask],
                val_labels=fit_labels,
                tail_fraction=tau,
            )
            held_direction = _binary_tail_gradient_contrast(
                val_features=val_x[held_out],
                val_probabilities=val_probs[held_out],
                val_labels=val_y[held_out],
                tail_fraction=tau,
            )
            fit_effect = sample_proxy * sample_sign * (sample_x @ fit_direction)
            held_effect = sample_proxy * sample_sign * (sample_x @ held_direction)
            fit_top = np.argpartition(fit_effect, -capture_count)[-capture_count:]
            held_top = np.argpartition(held_effect, -capture_count)[-capture_count:]
            oracle_effect = float(held_effect[held_top].sum())
            predicted_effect = float(held_effect[fit_top].sum())
            if oracle_effect > 1e-12:
                fold_captures.append(predicted_effect / oracle_effect)
                fold_oracles.append(oracle_effect)
        mean_capture = float(np.mean(fold_captures)) if fold_captures else float("nan")
        diagnostics.append(
            {
                "tau": tau,
                "valid": bool(valid and fold_captures),
                "mean_capture": mean_capture,
                "mean_oracle_effect": (
                    float(np.mean(fold_oracles)) if fold_oracles else float("nan")
                ),
                "num_capture_folds": len(fold_captures),
            }
        )

    valid_rows = [row for row in diagnostics if bool(row["valid"])]
    if not valid_rows:
        # Deterministic fallback for degenerate validation pools: the middle
        # candidate keeps the method defined while exposing the fallback.
        selected = min(candidates, key=lambda value: abs(value - 0.20))
    else:
        finite = [row for row in valid_rows if np.isfinite(float(row["mean_capture"]))]
        pool = finite if finite else valid_rows
        selected = max(
            pool, key=lambda row: (float(row["mean_capture"]), -float(row["tau"]))
        )["tau"]
    return float(selected), {
        "selected_tau": float(selected),
        "rule": "out_of_sample_capture",
        "budget_count": int(budget_count),
        "capture_count": int(capture_count),
        "num_folds": len(folds),
        "min_tail_count": int(min_tail_count),
        "stability_sample_size": int(sample_count),
        "fallback": not valid_rows,
        "candidates": diagnostics,
    }


def expected_repair_value_capture_scores(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    budget_count: int,
    candidate_tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    stability_sample_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return RV-Q scores whose tail fraction maximizes out-of-sample capture."""
    tau, metadata = select_tail_fraction_by_capture(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        noise_posterior_proxy=noise_posterior_proxy,
        budget_count=budget_count,
        candidate_tail_fractions=candidate_tail_fractions,
        num_folds=num_folds,
        min_tail_count=min_tail_count,
        stability_sample_size=stability_sample_size,
        seed=seed,
    )
    scores = expected_repair_value_scores(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        noise_posterior_proxy=noise_posterior_proxy,
        tail_fraction=tau,
    )
    return scores, metadata


def select_tail_fraction_by_calibrated_capture(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    budget_count: int,
    candidate_tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    stability_sample_size: int = 4096,
    seed: int = 0,
) -> tuple[float, dict[str, object]]:
    """Select tau using calibrated effect magnitude and held-out capture.

    The v2 capture ratio removes the candidate-dependent scale of the tail
    gradient.  This variant restores that information by normalizing the
    held-out predicted effect by the held-out direction norm, then combines
    its candidate rank equally with the capture-ratio rank.  All statistics
    remain cross-fitted and use only legal pre-query observables.
    """
    train_x = np.asarray(train_features)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.asarray(val_features)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    proxy_all = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features must be compatible matrices")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("feature and label arrays have incompatible lengths")
    if len(proxy_all) != len(train_x):
        raise ValueError("noise posterior proxy must align with the training set")
    if val_probs.ndim != 2 or len(val_probs) != len(val_y) or val_probs.shape[1] != 2:
        raise ValueError("calibrated capture selection currently supports binary validation probabilities")
    if set(np.unique(train_y)).difference({0, 1}):
        raise ValueError("calibrated capture selection currently supports binary training labels")
    if budget_count < 1 or min_tail_count < 1 or stability_sample_size < 1:
        raise ValueError("budget_count, min_tail_count, and stability_sample_size must be positive")
    candidates = tuple(sorted(set(float(value) for value in candidate_tail_fractions)))
    if not candidates or any(not 0.0 < value <= 1.0 for value in candidates):
        raise ValueError("candidate_tail_fractions must lie in (0, 1]")

    folds = _stratified_validation_folds(val_y, num_folds=num_folds, seed=seed)
    rng = np.random.default_rng(seed + 7919)
    num_train = len(train_x)
    sample_indices = (
        np.sort(rng.choice(num_train, size=stability_sample_size, replace=False))
        if num_train > stability_sample_size
        else np.arange(num_train, dtype=np.int64)
    )
    sample_x = train_x[sample_indices]
    sample_sign = 1.0 - 2.0 * train_y[sample_indices]
    sample_proxy = np.clip(proxy_all[sample_indices], 0.0, 1.0)
    capture_count = max(1, int(np.ceil(budget_count * len(sample_indices) / num_train)))

    diagnostics: list[dict[str, object]] = []
    for tau in candidates:
        fold_captures: list[float] = []
        fold_effects: list[float] = []
        fold_oracles: list[float] = []
        valid = True
        for held_out in folds:
            fit_mask = np.ones(len(val_y), dtype=bool)
            fit_mask[held_out] = False
            fit_labels = val_y[fit_mask]
            counts = [int(np.sum(fit_labels == label)) for label in np.unique(val_y)]
            fold_tail_count = min(max(1, int(np.ceil(tau * count))) for count in counts)
            if not len(fit_labels) or fold_tail_count < min_tail_count:
                valid = False
                break
            fit_direction = _binary_tail_gradient_contrast(
                val_features=val_x[fit_mask],
                val_probabilities=val_probs[fit_mask],
                val_labels=fit_labels,
                tail_fraction=tau,
            )
            held_direction = _binary_tail_gradient_contrast(
                val_features=val_x[held_out],
                val_probabilities=val_probs[held_out],
                val_labels=val_y[held_out],
                tail_fraction=tau,
            )
            fit_effect = sample_proxy * sample_sign * (sample_x @ fit_direction)
            held_effect = sample_proxy * sample_sign * (sample_x @ held_direction)
            fit_top = np.argpartition(fit_effect, -capture_count)[-capture_count:]
            held_top = np.argpartition(held_effect, -capture_count)[-capture_count:]
            oracle_effect = float(held_effect[held_top].sum())
            predicted_effect = float(held_effect[fit_top].sum())
            if oracle_effect > 1e-12:
                fold_captures.append(predicted_effect / oracle_effect)
                fold_effects.append(
                    predicted_effect / max(float(np.linalg.norm(held_direction)), 1e-12)
                )
                fold_oracles.append(oracle_effect)
        diagnostics.append(
            {
                "tau": tau,
                "valid": bool(valid and fold_captures),
                "mean_capture": float(np.mean(fold_captures)) if fold_captures else float("nan"),
                "mean_calibrated_effect": float(np.mean(fold_effects)) if fold_effects else float("nan"),
                "mean_oracle_effect": float(np.mean(fold_oracles)) if fold_oracles else float("nan"),
                "num_capture_folds": len(fold_captures),
            }
        )

    valid_rows = [row for row in diagnostics if bool(row["valid"])]
    finite = [
        row for row in valid_rows
        if np.isfinite(float(row["mean_capture"]))
        and np.isfinite(float(row["mean_calibrated_effect"]))
    ]
    if not finite:
        selected = min(candidates, key=lambda value: abs(value - 0.20))
    else:
        capture_values = np.asarray([float(row["mean_capture"]) for row in finite])
        effect_values = np.asarray([float(row["mean_calibrated_effect"]) for row in finite])
        capture_ranks = normalized_rank(capture_values)
        effect_ranks = normalized_rank(effect_values)
        for row, capture_rank, effect_rank in zip(finite, capture_ranks, effect_ranks):
            row["capture_rank"] = float(capture_rank)
            row["calibrated_effect_rank"] = float(effect_rank)
            row["combined_score"] = float(0.5 * (capture_rank + effect_rank))
        selected = max(
            finite,
            key=lambda row: (float(row["combined_score"]), -float(row["tau"])),
        )["tau"]
    return float(selected), {
        "selected_tau": float(selected),
        "rule": "calibrated_effect_capture",
        "budget_count": int(budget_count),
        "capture_count": int(capture_count),
        "num_folds": len(folds),
        "min_tail_count": int(min_tail_count),
        "stability_sample_size": int(len(sample_indices)),
        "fallback": not finite,
        "effect_rank_weight": 0.5,
        "capture_rank_weight": 0.5,
        "candidates": diagnostics,
    }


def expected_repair_value_calibrated_tau_scores(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    budget_count: int,
    candidate_tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    stability_sample_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return RV-Q scores using the calibrated-effect adaptive tau rule."""
    tau, metadata = select_tail_fraction_by_calibrated_capture(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        noise_posterior_proxy=noise_posterior_proxy,
        budget_count=budget_count,
        candidate_tail_fractions=candidate_tail_fractions,
        num_folds=num_folds,
        min_tail_count=min_tail_count,
        stability_sample_size=stability_sample_size,
        seed=seed,
    )
    scores = expected_repair_value_scores(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        noise_posterior_proxy=noise_posterior_proxy,
        tail_fraction=tau,
    )
    return scores, metadata


def robust_set_repair_value_ranking(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    budget_count: int,
    candidate_tail_fractions: tuple[float, ...] = (0.10, 0.20, 0.50),
    num_folds: int = 5,
    min_tail_count: int = 4,
    uncertainty_beta: float = 0.50,
    redundancy_lambda: float = 0.05,
    candidate_multiplier: float = 3.0,
    projection_dim: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a set-aware robust RepairValue-Q query ranking.

    Unlike pointwise RV-Q (or a reliability gate), this routine chooses the
    queried prefix jointly.  It builds cross-fitted validation scenarios over
    several tail fractions, scores each candidate by ``mean - beta * std``
    across scenarios, and greedily maximizes the worst-case accumulated value.
    A small cosine-similarity penalty discourages repeatedly selecting nearly
    duplicate frozen features.  All quantities are validation/train-observable
    and the returned permutation is deterministic for a fixed ``seed``.

    This is intentionally exposed as an exploratory method: the scenario
    values are normalized per scenario, so the resulting objective is robust
    to tail-gradient scale but should be validated with the same retraining
    protocol as the registered methods before promotion.
    """
    train_x = np.asarray(train_features, dtype=np.float32)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.asarray(val_features, dtype=np.float32)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    noise = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features must be compatible matrices")
    if len(train_x) != len(train_y) or len(train_x) != len(noise):
        raise ValueError("training arrays and noise proxy have incompatible lengths")
    if len(val_x) != len(val_y) or val_probs.ndim != 2 or len(val_probs) != len(val_y):
        raise ValueError("validation arrays have incompatible lengths")
    if val_probs.shape[1] != 2 or set(np.unique(train_y)).difference({0, 1}):
        raise ValueError("robust set RepairValue-Q currently supports binary labels only")
    if len(train_x) == 0 or not 1 <= int(budget_count) <= len(train_x):
        raise ValueError("budget_count must lie in [1, n_train]")
    if num_folds < 2 or min_tail_count < 1:
        raise ValueError("num_folds must be at least two and min_tail_count positive")
    if not np.isfinite(uncertainty_beta) or uncertainty_beta < 0:
        raise ValueError("uncertainty_beta must be finite and non-negative")
    if not np.isfinite(redundancy_lambda) or redundancy_lambda < 0:
        raise ValueError("redundancy_lambda must be finite and non-negative")
    if not np.isfinite(candidate_multiplier) or candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be finite and at least one")
    if projection_dim < 2:
        raise ValueError("projection_dim must be at least two")
    candidates = tuple(sorted(set(float(value) for value in candidate_tail_fractions)))
    if not candidates or any(not 0.0 < value <= 1.0 for value in candidates):
        raise ValueError("candidate_tail_fractions must lie in (0, 1]")

    folds = _stratified_validation_folds(val_y, num_folds=num_folds, seed=seed)
    scenario_values: list[np.ndarray] = []
    scenario_names: list[str] = []
    skipped: list[dict[str, object]] = []
    train_sign = 1.0 - 2.0 * train_y.astype(np.float64)
    noise_clip = np.clip(noise, 0.0, 1.0)
    for fold_index, held_out in enumerate(folds):
        fit_mask = np.ones(len(val_y), dtype=bool)
        fit_mask[held_out] = False
        fit_y = val_y[fit_mask]
        if len(fit_y) == 0:
            continue
        class_counts = [int(np.sum(fit_y == label)) for label in np.unique(val_y)]
        for tau in candidates:
            tail_count = min(max(1, int(np.ceil(tau * count))) for count in class_counts)
            if tail_count < min_tail_count:
                skipped.append({"fold": fold_index, "tau": tau, "tail_count": tail_count})
                continue
            direction = _binary_tail_gradient_contrast(
                val_features=val_x[fit_mask],
                val_probabilities=val_probs[fit_mask],
                val_labels=fit_y,
                tail_fraction=tau,
            )
            # Keep the same legal detector/value factorization as RV-Q:
            # rank the signed repair effect within each scenario, then gate it
            # by the detector posterior.  This prevents arbitrary embedding
            # scale from overwhelming the noise signal in the set objective.
            raw = train_sign * (train_x @ direction)
            value_rank = normalized_rank(raw)
            scenario = noise_clip * value_rank
            scale = float(np.std(scenario))
            if not np.isfinite(scale) or scale < 1e-12:
                skipped.append({"fold": fold_index, "tau": tau, "tail_count": tail_count, "reason": "zero_scale"})
                continue
            scenario_values.append(scenario / scale)
            scenario_names.append(f"fold{fold_index}:tau{tau:g}")
    if not scenario_values:
        raise ValueError("no valid validation scenarios; lower min_tail_count or add validation data")
    values = np.stack(scenario_values, axis=0).astype(np.float64, copy=False)
    robust_score = values.mean(axis=0) - float(uncertainty_beta) * values.std(axis=0)

    # A deterministic random projection makes the diversity penalty cheap even
    # for the high-dimensional cached embeddings used by Waterbirds/CelebA.
    rng = np.random.default_rng(int(seed) + 104729)
    dim = min(int(projection_dim), train_x.shape[1])
    projection = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(train_x.shape[1], dim))
    projected = train_x.astype(np.float64) @ projection
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)

    n_train = len(train_x)
    candidate_count = min(n_train, max(int(budget_count), int(np.ceil(candidate_multiplier * budget_count))))
    base_order = np.lexsort((np.arange(n_train), -noise_clip, -robust_score))
    pool = base_order[:candidate_count]
    current = np.zeros(values.shape[0], dtype=np.float64)
    selected: list[int] = []
    remaining = pool.tolist()
    for _ in range(int(budget_count)):
        candidate_array = np.asarray(remaining, dtype=np.int64)
        new_current = current[:, None] + values[:, candidate_array]
        # Robust set utility is the aggregate mean across scenarios minus an
        # uncertainty penalty on the aggregate.  Using the minimum scenario
        # directly is needlessly brittle when one fold has a noisy tail.
        current_objective = float(np.mean(current) - uncertainty_beta * np.std(current))
        new_objective = np.mean(new_current, axis=0) - uncertainty_beta * np.std(new_current, axis=0)
        marginal = new_objective - current_objective
        if selected:
            similarity = np.max(
                np.maximum(0.0, projected[candidate_array] @ projected[np.asarray(selected)].T),
                axis=1,
            )
        else:
            similarity = np.zeros(len(candidate_array), dtype=np.float64)
        objective = marginal - float(redundancy_lambda) * similarity
        order = np.lexsort((candidate_array, -noise_clip[candidate_array], -robust_score[candidate_array], -objective))
        pick = int(candidate_array[order[0]])
        selected.append(pick)
        current += values[:, pick]
        remaining.remove(pick)

    leftovers = np.asarray(remaining, dtype=np.int64)
    outside = base_order[candidate_count:]
    if len(leftovers):
        leftovers = leftovers[np.lexsort((leftovers, -noise_clip[leftovers], -robust_score[leftovers]))]
    ranking = np.concatenate((np.asarray(selected, dtype=np.int64), leftovers, outside))
    return ranking, {
        "method": "robust_set_repair_value",
        "num_scenarios": int(values.shape[0]),
        "scenario_names": scenario_names,
        "skipped_scenarios": skipped,
        "num_folds": int(len(folds)),
        "candidate_tail_fractions": list(candidates),
        "uncertainty_beta": float(uncertainty_beta),
        "redundancy_lambda": float(redundancy_lambda),
        "candidate_multiplier": float(candidate_multiplier),
        "candidate_count": int(candidate_count),
        "projection_dim": int(dim),
        "budget_count": int(budget_count),
        "robust_score_mean": float(np.mean(robust_score)),
        "robust_score_std": float(np.std(robust_score)),
        "selected_prefix_worst_case": float(np.min(current)),
        "selected_prefix_mean": float(np.mean(current)),
        "selected_prefix_robust_objective": float(np.mean(current) - uncertainty_beta * np.std(current)),
        "selected_indices": selected,
    }


def _project_augmented_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    *,
    projection_dim: int,
    projection_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.random_projection import SparseRandomProjection
    from sklearn.utils.extmath import safe_sparse_dot

    train_x = np.asarray(train_features, dtype=np.float32)
    val_x = np.asarray(val_features, dtype=np.float32)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation features have incompatible shapes")
    if projection_dim < 2:
        raise ValueError("projection_dim must be at least two")
    projector = SparseRandomProjection(
        n_components=projection_dim,
        density="auto",
        dense_output=True,
        random_state=projection_seed,
    )
    projector.fit(np.zeros((1, train_x.shape[1] + 1), dtype=np.float32))
    components = projector.components_
    train_projected = safe_sparse_dot(
        train_x, components[:, :-1].T, dense_output=True
    )
    val_projected = safe_sparse_dot(val_x, components[:, :-1].T, dense_output=True)
    bias = np.asarray(components[:, -1].toarray()).reshape(1, -1)
    train_projected = np.asarray(train_projected, dtype=np.float32) + bias
    val_projected = np.asarray(val_projected, dtype=np.float32) + bias
    return train_projected.astype(np.float32), val_projected.astype(np.float32)


def auto_d3m_query_adaptation_scores(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    noise_posterior_proxy: np.ndarray,
    options: dict[str, Any] | None = None,
    cache: dict[tuple[Any, ...], Any] | None = None,
) -> np.ndarray:
    """Compute a group-blind AUTO-D3M-style label-query adaptation.

    The implementation uses one frozen linear-head trial and a deterministic
    sparse projection. It follows AUTO-D3M's attribution-PC pseudo-group and
    loss-weighted group-alignment construction, then adapts removal to label
    verification by multiplying harmful-alignment rank by NoiseScore.
    """
    settings = dict(options or {})
    projection_dim = int(settings.get("auto_d3m_projection_dim", 64))
    projection_seed = int(settings.get("auto_d3m_projection_seed", 20260731))
    tail_fraction = float(settings.get("auto_d3m_tail_fraction", 0.35))
    beta = float(settings.get("auto_d3m_beta", 1.0))
    ridge = float(settings.get("auto_d3m_ridge", 1e-4))
    if not 0 < tail_fraction < 0.5:
        raise ValueError("auto_d3m_tail_fraction must lie in (0, 0.5)")
    labels = np.asarray(train_labels, dtype=np.int64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    val_probs = np.asarray(val_probabilities, dtype=np.float64)
    proxy = np.asarray(noise_posterior_proxy, dtype=np.float64)
    if val_probs.shape[1] != 2 or set(np.unique(labels)).difference({0, 1}):
        raise ValueError("AUTO-D3M query adaptation currently supports binary labels only")
    if len(labels) != len(proxy):
        raise ValueError("train labels and posterior proxy have different lengths")

    shared = cache if cache is not None else {}
    projection_key = (
        "auto_d3m_projection",
        id(train_features),
        id(val_features),
        projection_dim,
        projection_seed,
    )
    projected = shared.get(projection_key)
    if projected is None:
        projected = _project_augmented_features(
            train_features,
            val_features,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
        )
        shared[projection_key] = projected
    train_projected, val_projected = projected

    train_sign = np.where(labels == 1, 1.0, -1.0).astype(np.float32)
    val_sign = np.where(val_y == 1, 1.0, -1.0).astype(np.float32)
    train_gradients = train_projected * train_sign[:, None]
    val_gradients = val_projected * val_sign[:, None]
    gram = train_gradients.T.astype(np.float64) @ train_gradients.astype(np.float64)
    scale = max(float(np.trace(gram) / projection_dim), 1e-12)
    regularized = gram + ridge * scale * np.eye(projection_dim)
    inverse = np.linalg.inv(regularized)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    gram_sqrt = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T

    correct_probabilities = val_probs[np.arange(len(val_y)), val_y]
    val_losses = -np.log(np.clip(correct_probabilities, 1e-12, 1.0))
    after_kernel = (
        (1.0 - correct_probabilities)[:, None]
        * (val_gradients.astype(np.float64) @ inverse)
    )

    class_alignments = []
    for label in np.unique(val_y):
        positions = np.flatnonzero(val_y == label)
        if len(positions) < 4:
            raise ValueError("AUTO-D3M query adaptation needs at least four validation examples per class")
        class_kernel = after_kernel[positions]
        centered = class_kernel - class_kernel.mean(axis=0, keepdims=True)
        compressed = centered @ gram_sqrt
        covariance = compressed.T @ compressed
        _, vectors = np.linalg.eigh(covariance)
        pc_scores = compressed @ vectors[:, -1]
        count = max(1, int(np.ceil(tail_fraction * len(positions))))
        order = np.argsort(pc_scores, kind="mergesort")
        low_local = order[:count]
        high_local = order[-count:]
        losses = val_losses[positions]
        weak_local = (
            high_local if losses[high_local].mean() >= losses[low_local].mean() else low_local
        )
        weak_mask = np.zeros(len(positions), dtype=bool)
        weak_mask[weak_local] = True
        group_masks = (weak_mask, ~weak_mask)
        group_losses = np.asarray([losses[mask].mean() for mask in group_masks])
        logits = beta * (group_losses - group_losses.max())
        group_weights = np.exp(logits)
        group_weights /= group_weights.sum()
        alignment = np.zeros(len(labels), dtype=np.float64)
        for weight, mask in zip(group_weights, group_masks, strict=True):
            mean_factor = class_kernel[mask].mean(axis=0)
            alignment += weight * (mean_factor @ train_gradients.T)
        class_alignments.append(alignment)

    # Equal observed-class weighting avoids allowing the majority class to
    # dominate the no-group pseudo-group construction.
    alignment = np.mean(np.stack(class_alignments), axis=0)
    harmful_alignment_rank = normalized_rank(-alignment)
    return np.clip(proxy, 0.0, 1.0) * harmful_alignment_rank
