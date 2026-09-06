from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from robust_verify.scoring import build_legal_scores
from robust_verify.utils import normalized_rank, rank_within_class


@dataclass(frozen=True)
class RivetScores:
    """First-order verification values for a binary linear head."""

    utility: np.ndarray
    leverage: np.ndarray
    noise_probability: np.ndarray
    tail_indices: np.ndarray
    validation_cvar: float
    alpha: float
    damping: float


@dataclass(frozen=True)
class FeedbackPosterior:
    """Noise posterior learned only from outcomes inside the query budget."""

    probability: np.ndarray
    exploration_indices: np.ndarray
    exploration_outcomes: np.ndarray


@dataclass(frozen=True)
class SpectralRivetScores:
    """Influence values for a smooth maximum of class-conditional CVaRs."""

    utility: np.ndarray
    leverage: np.ndarray
    noise_probability: np.ndarray
    cell_classes: np.ndarray
    cell_alphas: np.ndarray
    cell_risks: np.ndarray
    cell_weights: np.ndarray
    spectral_risk: float
    temperature: float
    damping: float


def class_prior_adjusted_noise_score(
    noise_score: np.ndarray,
    losses: np.ndarray,
    noisy_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the paper's legal CPBA class-risk prior to a noise score.

    The prior uses only observed labels and probe losses.  It is returned as a
    score rather than a ranking so it can serve as the p_i factor in RIVET.
    """

    from scipy.stats import spearmanr

    noise_score = np.asarray(noise_score, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    noisy_labels = np.asarray(noisy_labels)
    if losses.shape != noise_score.shape or noisy_labels.shape != noise_score.shape:
        raise ValueError("noise_score, losses, and noisy_labels must have equal shape")
    classes = np.unique(noisy_labels)
    if len(classes) == 0:
        raise ValueError("at least one observed class is required")

    preservation = float(
        spearmanr(rank_within_class(losses, noisy_labels), normalized_rank(losses)).correlation
    )
    if not np.isfinite(preservation):
        preservation = 1.0
    preservation = float(np.clip(preservation, 0.0, 1.0))
    mean_loss = {
        value: float(losses[noisy_labels == value].mean()) for value in classes
    }
    total_mean_loss = sum(mean_loss.values()) + 1e-12
    adjusted = noise_score.copy()
    metadata: dict[str, float] = {"rank_preservation": preservation}
    for value in classes:
        risk_share = mean_loss[value] / total_mean_loss
        risk_ratio = risk_share / (1.0 / len(classes))
        weight = float(np.clip(1.0 + (1.0 - preservation) * (risk_ratio - 1.0), 0.5, 2.0))
        adjusted[noisy_labels == value] *= weight
        metadata[f"weight_class_{value}"] = weight
        metadata[f"mean_loss_class_{value}"] = mean_loss[value]
    return adjusted, metadata


def legal_noise_features(
    probabilities: np.ndarray,
    noisy_labels: np.ndarray,
    correctness_history: np.ndarray,
    leverage: np.ndarray,
) -> np.ndarray:
    """Build group-free covariates for feedback calibration of P(label error).

    All columns are available before verification: probe probabilities,
    training dynamics, observed labels, and the robust influence term.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    correctness_history = np.asarray(correctness_history)
    leverage = np.asarray(leverage, dtype=np.float64)
    if probabilities.shape != (len(noisy_labels), 2):
        raise ValueError("probabilities must have shape [n, 2]")
    if correctness_history.ndim != 2 or len(correctness_history) != len(noisy_labels):
        raise ValueError("correctness_history must have shape [n, epochs]")
    if leverage.shape != (len(noisy_labels),):
        raise ValueError("leverage must have shape [n]")

    scores = build_legal_scores(probabilities, noisy_labels, correctness_history)
    observed_probability = probabilities[np.arange(len(noisy_labels)), noisy_labels]
    disagreement = (probabilities.argmax(axis=1) != noisy_labels).astype(np.float64)
    error_frequency = 1.0 - correctness_history.mean(axis=1)
    label = noisy_labels.astype(np.float64)
    core = np.column_stack(
        [
            normalized_rank(scores["loss"]),
            normalized_rank(scores["entropy"]),
            normalized_rank(scores["forgetting"]),
            normalized_rank(scores["noise_score"]),
            normalized_rank(1.0 - observed_probability),
            normalized_rank(error_frequency),
            disagreement,
            normalized_rank(leverage),
            normalized_rank(np.abs(leverage)),
            (leverage > 0.0).astype(np.float64),
        ]
    )
    # Explicit class interactions let a small calibrator undo class-conditional
    # score distortion without observing protected attributes or true groups.
    return np.column_stack([core, label, core * label[:, None]])


def feedback_exploration_indices(
    base_priority: np.ndarray,
    noisy_labels: np.ndarray,
    count: int,
    *,
    seed: int,
    targeted_fraction: float = 0.5,
    num_bins: int = 10,
) -> np.ndarray:
    """Mix high-utility queries with label/score-stratified exploration."""

    base_priority = np.asarray(base_priority, dtype=np.float64)
    noisy_labels = np.asarray(noisy_labels)
    n = len(base_priority)
    if noisy_labels.shape != (n,):
        raise ValueError("noisy_labels must have shape [n]")
    if not 0 <= count <= n:
        raise ValueError("count must lie in [0, n]")
    if not 0.0 <= targeted_fraction <= 1.0:
        raise ValueError("targeted_fraction must lie in [0, 1]")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    if count == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    target_count = min(count, int(round(targeted_fraction * count)))
    targeted = np.lexsort((np.arange(n), -base_priority))[:target_count]
    chosen = [int(index) for index in targeted]
    chosen_set = set(chosen)

    bins = np.minimum(
        num_bins - 1,
        np.floor(normalized_rank(base_priority) * num_bins).astype(int),
    )
    queues: list[list[int]] = []
    for label in np.unique(noisy_labels):
        for bin_index in range(num_bins):
            members = np.where((noisy_labels == label) & (bins == bin_index))[0]
            members = np.asarray(
                [index for index in members if int(index) not in chosen_set],
                dtype=np.int64,
            )
            rng.shuffle(members)
            if len(members):
                queues.append(members.tolist())

    while len(chosen) < count and queues:
        next_queues: list[list[int]] = []
        for queue in queues:
            if len(chosen) >= count:
                break
            chosen.append(int(queue.pop()))
            if queue:
                next_queues.append(queue)
        queues = next_queues

    if len(chosen) < count:
        remaining = np.asarray(
            [index for index in range(n) if index not in set(chosen)], dtype=np.int64
        )
        rng.shuffle(remaining)
        chosen.extend(remaining[: count - len(chosen)].tolist())
    return np.asarray(chosen, dtype=np.int64)


def trust_region_candidate_selection(
    prior_order: np.ndarray,
    prior_score: np.ndarray,
    leverage: np.ndarray,
    count: int,
    *,
    candidate_multiplier: float = 1.5,
    core_fraction: float = 0.75,
) -> np.ndarray:
    """Keep a baseline core and let influence replace only the query fringe.

    This is a Hamming trust region around a high-precision baseline set.  The
    number of samples exposed to a misspecified value proxy is at most
    ``count - floor(core_fraction * count)`` instead of the full budget.
    """

    prior_order = np.asarray(prior_order, dtype=np.int64)
    prior_score = np.asarray(prior_score, dtype=np.float64)
    leverage = np.asarray(leverage, dtype=np.float64)
    n = len(prior_order)
    if prior_score.shape != (n,) or leverage.shape != (n,):
        raise ValueError("prior_score and leverage must match prior_order length")
    if len(np.unique(prior_order)) != n or set(prior_order.tolist()) != set(range(n)):
        raise ValueError("prior_order must be a permutation of [0, n)")
    if not 0 <= count <= n:
        raise ValueError("count must lie in [0, n]")
    if candidate_multiplier < 1.0:
        raise ValueError("candidate_multiplier must be at least one")
    if not 0.0 <= core_fraction <= 1.0:
        raise ValueError("core_fraction must lie in [0, 1]")
    if count == 0:
        return np.empty(0, dtype=np.int64)

    core_count = min(count, int(np.floor(core_fraction * count)))
    core = prior_order[:core_count]
    candidate_count = min(n, int(np.ceil(candidate_multiplier * count)))
    candidates = prior_order[core_count:candidate_count]
    local_order = np.lexsort(
        (
            candidates,
            -prior_score[candidates],
            -leverage[candidates],
        )
    )
    fringe = candidates[local_order[: count - core_count]]
    return np.concatenate([core, fringe])


def fit_feedback_noise_posterior(
    feature_matrix: np.ndarray,
    exploration_indices: np.ndarray,
    exploration_outcomes: np.ndarray,
    *,
    seed: int,
    fallback_score: np.ndarray,
) -> FeedbackPosterior:
    """Fit a compact nonlinear posterior using verified exploration labels."""

    from sklearn.ensemble import HistGradientBoostingClassifier

    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    exploration_indices = np.asarray(exploration_indices, dtype=np.int64)
    exploration_outcomes = np.asarray(exploration_outcomes, dtype=np.int64)
    fallback_score = np.asarray(fallback_score, dtype=np.float64)
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional")
    if exploration_outcomes.shape != (len(exploration_indices),):
        raise ValueError("one exploration outcome is required per queried index")
    if fallback_score.shape != (len(feature_matrix),):
        raise ValueError("fallback_score must have shape [n]")
    if len(exploration_indices) == 0 or len(np.unique(exploration_outcomes)) < 2:
        probability = normalized_rank(fallback_score)
    else:
        counts = np.bincount(exploration_outcomes, minlength=2).astype(np.float64)
        weights = len(exploration_outcomes) / (2.0 * counts[exploration_outcomes])
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=max(10, len(exploration_indices) // 100),
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(
            feature_matrix[exploration_indices],
            exploration_outcomes,
            sample_weight=weights,
        )
        probability = model.predict_proba(feature_matrix)[:, 1]
    return FeedbackPosterior(
        probability=np.clip(probability, 0.0, 1.0),
        exploration_indices=exploration_indices,
        exploration_outcomes=exploration_outcomes,
    )


def binary_cross_entropy_losses(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities must have shape [n, 2]")
    if labels.shape != (len(probabilities),):
        raise ValueError("labels must have shape [n]")
    chosen = probabilities[np.arange(len(labels)), labels]
    return -np.log(np.clip(chosen, 1e-12, 1.0))


def empirical_cvar(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> float:
    """Mean cross-entropy in the worst alpha fraction."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    losses = binary_cross_entropy_losses(probabilities, labels)
    tail_count = min(len(losses), max(1, int(np.ceil(alpha * len(losses)))))
    tail = np.argpartition(losses, len(losses) - tail_count)[-tail_count:]
    return float(losses[tail].mean())


def spectral_class_cvar(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.01, 0.02, 0.05),
    temperature: float = 5.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth maximum over observed-class conditional CVaRs.

    The log-sum-exp risk is group-free and label-permutation invariant.  Its
    cells cover several tail masses so the acquisition rule does not depend on
    one guessed latent-group prevalence.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
        raise ValueError("probabilities and labels must have matching rows")
    if not alphas or any(not 0.0 < alpha <= 1.0 for alpha in alphas):
        raise ValueError("alphas must be a non-empty sequence in (0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    cell_classes: list[int] = []
    cell_alphas: list[float] = []
    cell_risks: list[float] = []
    for class_value in sorted(np.unique(labels).tolist()):
        mask = labels == class_value
        if not mask.any():
            continue
        for alpha in alphas:
            cell_classes.append(int(class_value))
            cell_alphas.append(float(alpha))
            cell_risks.append(
                empirical_cvar(probabilities[mask], labels[mask], float(alpha))
            )
    risks = np.asarray(cell_risks, dtype=np.float64)
    if len(risks) == 0:
        raise ValueError("at least one validation class is required")
    logits = temperature * risks
    offset = float(logits.max())
    exp_logits = np.exp(logits - offset)
    weights = exp_logits / exp_logits.sum()
    spectral_risk = (offset + np.log(exp_logits.sum())) / temperature
    return (
        float(spectral_risk),
        np.asarray(cell_classes, dtype=np.int64),
        np.asarray(cell_alphas, dtype=np.float64),
        risks,
        weights,
    )


def _tail_indices(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    losses = binary_cross_entropy_losses(probabilities, labels)
    tail_count = min(len(losses), max(1, int(np.ceil(alpha * len(losses)))))
    order = np.lexsort((np.arange(len(losses)), -losses))
    return order[:tail_count].astype(np.int64), losses


def _binary_logistic_hessian(
    features: np.ndarray,
    probability_class1: np.ndarray,
    *,
    damping: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the damped Hessian for beta = w_1 - w_0, including bias."""

    if damping <= 0:
        raise ValueError("damping must be positive")
    features = np.asarray(features, dtype=np.float32)
    probability_class1 = np.asarray(probability_class1, dtype=np.float32)
    if probability_class1.shape != (len(features),):
        raise ValueError("probability_class1 must have shape [n]")

    n, dimension = features.shape
    hessian = torch.zeros(
        (dimension + 1, dimension + 1),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        x = torch.from_numpy(features[start:stop]).to(device)
        ones = torch.ones((len(x), 1), dtype=x.dtype, device=device)
        augmented = torch.cat([x, ones], dim=1)
        p = torch.from_numpy(probability_class1[start:stop]).to(device)
        curvature = (p * (1.0 - p)).clamp_min(1e-7)
        weighted = augmented * curvature.sqrt().unsqueeze(1)
        hessian.addmm_(weighted.T, weighted)

    hessian /= float(n)
    hessian.diagonal().add_(float(damping))
    return hessian


def _tail_gradient(
    features: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    tail_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Gradient of empirical CVaR for a binary logistic linear head."""

    features = np.asarray(features, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    dimension = features.shape[1]
    gradient = torch.zeros(dimension + 1, dtype=torch.float32, device=device)

    for start in range(0, len(tail_indices), batch_size):
        idx = tail_indices[start : start + batch_size]
        x = torch.from_numpy(features[idx]).to(device)
        p = torch.from_numpy(probabilities[idx, 1]).to(device)
        y = torch.from_numpy(labels[idx].astype(np.float32)).to(device)
        residual = p - y
        gradient[:-1].add_(x.T @ residual)
        gradient[-1].add_(residual.sum())

    gradient /= float(len(tail_indices))
    return gradient


def binary_rivet_scores(
    *,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    noisy_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    noise_score: np.ndarray,
    rank_noise_score: bool = True,
    alpha: float = 0.05,
    damping: float = 1e-3,
    hessian_batch_size: int = 8192,
    device: str | torch.device = "cpu",
) -> RivetScores:
    """Compute group-free expected tail-risk improvement scores.

    By default the noise posterior is provisionally represented by the
    percentile rank of ``noise_score``.  Set ``rank_noise_score=False`` only
    when the input is already a probability (for example, the diagnostic
    oracle corruption indicator).
    """

    train_features = np.asarray(train_features, dtype=np.float32)
    validation_features = np.asarray(validation_features, dtype=np.float32)
    train_probabilities = np.asarray(train_probabilities, dtype=np.float32)
    validation_probabilities = np.asarray(validation_probabilities, dtype=np.float32)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    validation_labels = np.asarray(validation_labels, dtype=np.int64)
    noise_score = np.asarray(noise_score, dtype=np.float64)

    if train_probabilities.shape != (len(train_features), 2):
        raise ValueError("train_probabilities must have shape [n_train, 2]")
    if validation_probabilities.shape != (len(validation_features), 2):
        raise ValueError("validation_probabilities must have shape [n_val, 2]")
    if noisy_labels.shape != (len(train_features),):
        raise ValueError("noisy_labels must have shape [n_train]")
    if validation_labels.shape != (len(validation_features),):
        raise ValueError("validation_labels must have shape [n_val]")
    if noise_score.shape != (len(train_features),):
        raise ValueError("noise_score must have shape [n_train]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    resolved_device = torch.device(device)
    tail_indices, validation_losses = _tail_indices(
        validation_probabilities,
        validation_labels,
        alpha,
    )
    hessian = _binary_logistic_hessian(
        train_features,
        train_probabilities[:, 1],
        damping=damping,
        batch_size=hessian_batch_size,
        device=resolved_device,
    )
    gradient = _tail_gradient(
        validation_features,
        validation_probabilities,
        validation_labels,
        tail_indices,
        batch_size=hessian_batch_size,
        device=resolved_device,
    )

    # The damped Hessian is positive definite; Cholesky is faster and more
    # stable than a generic solve for the last-layer dimensions used here.
    factor = torch.linalg.cholesky(hessian)
    inverse_hessian_gradient = torch.cholesky_solve(
        gradient[:, None], factor
    )[:, 0]
    vector = inverse_hessian_gradient.detach().cpu().numpy().astype(np.float64)

    # For binary cross-entropy, changing label t to 1-t changes the gradient
    # by (2t-1) * [phi(x), 1].  The 1/n factor does not affect ranking but is
    # retained so the score has the scale of predicted CVaR improvement.
    projection = train_features.astype(np.float64) @ vector[:-1] + vector[-1]
    flip_direction = 2.0 * noisy_labels.astype(np.float64) - 1.0
    leverage = flip_direction * projection / float(len(train_features))

    if rank_noise_score:
        noise_probability = normalized_rank(noise_score)
    else:
        if not np.isfinite(noise_score).all():
            raise ValueError("pre-calibrated noise probabilities must be finite")
        if ((noise_score < 0.0) | (noise_score > 1.0)).any():
            raise ValueError("pre-calibrated noise probabilities must lie in [0, 1]")
        noise_probability = noise_score.copy()
    utility = noise_probability * leverage
    return RivetScores(
        utility=utility,
        leverage=leverage,
        noise_probability=noise_probability,
        tail_indices=tail_indices,
        validation_cvar=float(validation_losses[tail_indices].mean()),
        alpha=float(alpha),
        damping=float(damping),
    )


def spectral_binary_rivet_scores(
    *,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    noisy_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    noise_score: np.ndarray,
    rank_noise_score: bool = True,
    alphas: tuple[float, ...] = (0.01, 0.02, 0.05),
    temperature: float = 5.0,
    damping: float = 1e-3,
    hessian_batch_size: int = 8192,
    device: str | torch.device = "cpu",
) -> SpectralRivetScores:
    """Expected verification value for hierarchical spectral tail risk.

    Unlike :func:`binary_rivet_scores`, this computes the Hessian once and
    aggregates influence directions from every observed-class/tail-mass cell.
    No group annotation or private corruption indicator is used.
    """

    train_features = np.asarray(train_features, dtype=np.float32)
    validation_features = np.asarray(validation_features, dtype=np.float32)
    train_probabilities = np.asarray(train_probabilities, dtype=np.float32)
    validation_probabilities = np.asarray(validation_probabilities, dtype=np.float32)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    validation_labels = np.asarray(validation_labels, dtype=np.int64)
    noise_score = np.asarray(noise_score, dtype=np.float64)
    if train_probabilities.shape != (len(train_features), 2):
        raise ValueError("train_probabilities must have shape [n_train, 2]")
    if validation_probabilities.shape != (len(validation_features), 2):
        raise ValueError("validation_probabilities must have shape [n_val, 2]")
    if noisy_labels.shape != (len(train_features),):
        raise ValueError("noisy_labels must have shape [n_train]")
    if validation_labels.shape != (len(validation_features),):
        raise ValueError("validation_labels must have shape [n_val]")
    if noise_score.shape != (len(train_features),):
        raise ValueError("noise_score must have shape [n_train]")

    (
        spectral_risk,
        cell_classes,
        cell_alphas,
        cell_risks,
        cell_weights,
    ) = spectral_class_cvar(
        validation_probabilities,
        validation_labels,
        alphas=alphas,
        temperature=temperature,
    )

    resolved_device = torch.device(device)
    hessian = _binary_logistic_hessian(
        train_features,
        train_probabilities[:, 1],
        damping=damping,
        batch_size=hessian_batch_size,
        device=resolved_device,
    )
    factor = torch.linalg.cholesky(hessian)
    combined_leverage = np.zeros(len(train_features), dtype=np.float64)

    for class_value, alpha, weight in zip(
        cell_classes, cell_alphas, cell_weights, strict=True
    ):
        class_indices = np.flatnonzero(validation_labels == class_value)
        local_tail, _ = _tail_indices(
            validation_probabilities[class_indices],
            validation_labels[class_indices],
            float(alpha),
        )
        tail_indices = class_indices[local_tail]
        gradient = _tail_gradient(
            validation_features,
            validation_probabilities,
            validation_labels,
            tail_indices,
            batch_size=hessian_batch_size,
            device=resolved_device,
        )
        inverse_hessian_gradient = torch.cholesky_solve(
            gradient[:, None], factor
        )[:, 0]
        vector = inverse_hessian_gradient.detach().cpu().numpy().astype(np.float64)
        projection = train_features.astype(np.float64) @ vector[:-1] + vector[-1]
        flip_direction = 2.0 * noisy_labels.astype(np.float64) - 1.0
        combined_leverage += (
            float(weight)
            * flip_direction
            * projection
            / float(len(train_features))
        )

    if rank_noise_score:
        noise_probability = normalized_rank(noise_score)
    else:
        if not np.isfinite(noise_score).all():
            raise ValueError("pre-calibrated noise probabilities must be finite")
        if ((noise_score < 0.0) | (noise_score > 1.0)).any():
            raise ValueError("pre-calibrated noise probabilities must lie in [0, 1]")
        noise_probability = noise_score.copy()
    return SpectralRivetScores(
        utility=noise_probability * combined_leverage,
        leverage=combined_leverage,
        noise_probability=noise_probability,
        cell_classes=cell_classes,
        cell_alphas=cell_alphas,
        cell_risks=cell_risks,
        cell_weights=cell_weights,
        spectral_risk=spectral_risk,
        temperature=float(temperature),
        damping=float(damping),
    )
