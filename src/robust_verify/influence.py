from __future__ import annotations

import numpy as np


def _onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), num_classes), dtype=np.float64)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def compute_tracin_influence(
    train_features: np.ndarray,
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    *,
    val_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form first-order TracIn influence of each train sample on a weighted
    validation loss, for a linear head with cross-entropy.

    influence[i] = -sum_j w_j * (p_i - y_i) . (q_j - t_j) * (x_i . x_j)

    The minus sign follows the first-order SGD update: the validation-loss change
    caused by training on example ``i`` is ``-eta * <g_val, g_i>``. Therefore a
    higher value means the example is more harmful and gets higher query priority.

    Single checkpoint (best probe state). The learning-rate scale is constant across
    samples and does not affect the ranking, so it is omitted.
    """
    num_classes = int(train_probs.shape[1])
    train_resid = train_probs.astype(np.float64) - _onehot(train_labels, num_classes)
    val_resid = val_probs.astype(np.float64) - _onehot(val_labels, num_classes)

    if val_weights is None:
        val_weights = np.ones(len(val_labels), dtype=np.float64)
    val_weights = np.asarray(val_weights, dtype=np.float64)

    # For a linear softmax head, the per-example gradient is the outer product
    # ``residual (K) x feature (D)``.  Summing validation gradients first avoids
    # materialising two [n_train, n_val] matrices, which is prohibitive for
    # CelebA and CivilComments.
    # Keep feature matrices in their stored dtype (normally float32) to avoid a
    # second full-size copy. Mixed operations with float64 residuals still
    # accumulate the final influence scores in float64.
    train_x = np.asarray(train_features)
    val_x = np.asarray(val_features)
    weighted_val_resid = val_resid * val_weights[:, None]
    validation_gradient = weighted_val_resid.T @ val_x  # [K, D]
    return -np.einsum(
        "nk,nd,kd->n", train_resid, train_x, validation_gradient, optimize=True
    )


def worst_group_mask(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask over validation samples selecting the worst-group
    (lowest accuracy) slice. Uses true group labels -> oracle only.
    """
    val_preds = val_probs.argmax(axis=1)
    val_groups = np.asarray(val_groups, dtype=np.int64)
    group_accuracy = {}
    for group in np.unique(val_groups):
        mask = val_groups == group
        group_accuracy[int(group)] = float((val_preds[mask] == val_labels[mask]).mean())
    worst_group = min(group_accuracy, key=group_accuracy.get)
    return val_groups == worst_group


def uncertainty_weights(val_probs: np.ndarray) -> np.ndarray:
    """Legal proxy for hard / worst-group-like validation samples: (1 - max prob)."""
    return 1.0 - np.max(val_probs, axis=1)
