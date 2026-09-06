from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from robust_verify.utils import entropy_from_probabilities, normalized_rank, rank_within_class

GroupOrder = Literal["size_asc", "size_desc", "sorted", "appearance"]

NOISE_RISK_LOSS_RATIO_THRESHOLD = 4.0

METHOD_NAMES = {
    "random",
    "loss",
    "entropy",
    "forgetting",
    "noise_score",
    "noise_score_coverage",
    "noise_score_cluster_coverage",
    "noise_score_cluster_coverage_adaptive",
    "noise_score_gated_adaptive",
    "noise_score_budget_hybrid",
    "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc",
    "noise_score_cpba_only",
    "noise_score_tc_only",
    "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback",
    "noise_score_cpba_tc_entropy_fallback",
    "noise_score_multi_round",
    "aum",
    "confident_learning",
    "margin",
    "coreset",
    "uncertainty_diversity",
    "badge_lite",
    "nn_agreement",
    "nn_label_spreading",
    "stratified_random",
    "tracin_val_uncertainty",
    "tracin_noise_weighted",
    "tracin_multicheckpoint_val",
    "expected_repair_value",
    "expected_repair_value_multiclass",
    "expected_repair_value_adaptive_tau",
    "expected_repair_value_capture_tau",
    "expected_repair_value_calibrated_tau",
    "noise_gated_repair_value",
    "reliability_gated_repair_value",
    "auto_d3m_query",
    "activeclean_expected_change",
    "active_label_cleaning",
    "oof_cleanlab",
    "tracin_wga_oracle",
    "oracle_noise",
    "oracle_noise_loss_tiebreak",
    "oracle_noise_random_tiebreak",
    "oracle_noise_group_balanced",
    "oracle_disagreement",
    "oracle_minority",
    "oracle_group_balanced",
}

ADAPTIVE_METHODS = {
    "noise_score_cluster_coverage_adaptive",
    "noise_score_gated_adaptive",
    "noise_score_budget_hybrid",
    "noise_score_budget_hybrid_v8",
    "noise_score_cpba_tc",
    "noise_score_cpba_only",
    "noise_score_tc_only",
    "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback",
    "noise_score_cpba_tc_entropy_fallback",
    "noise_gated_repair_value",
}

# Unified paper display names (v→short label)
METHOD_LABELS = {
    "noise_score":                       "NoiseScore",
    "noise_score_coverage":              "NoiseScore+Cov",
    "noise_score_cluster_coverage":      "NoiseScore+CluCov",
    "noise_score_cluster_coverage_adaptive": "NoiseScore+AdaCov",
    "noise_score_gated_adaptive":        "NoiseScore+GatedCov",
    "noise_score_budget_hybrid":         "BHC",
    "noise_score_budget_hybrid_v8":      "NGC",
    "noise_score_cpba_tc":               "CPBA-TC",
    "noise_score_cpba_only":             "CPBA-only",
    "noise_score_tc_only":               "TC-only",
    "noise_score_cpba_tc_no_fallback":   "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback":   "CPBA-TC-v8FB",
    "noise_score_cpba_tc_entropy_fallback": "CPBA-TC-entFB",
    "noise_score_multi_round":           "NoiseScore+MR",
    "oracle_noise":                      "Oracle-noise",
    "oracle_noise_loss_tiebreak":        "Oracle-noise / loss-tiebreak",
    "oracle_noise_random_tiebreak":      "Oracle-noise / random-tiebreak",
    "oracle_noise_group_balanced":       "Oracle-GB",
    "oracle_disagreement":               "Oracle-Disag",
    "oracle_minority":                   "Oracle-minority",
    "oracle_group_balanced":             "Oracle-group-balanced",
    "aum":                               "AUM",
    "confident_learning":     "Cleanlab",
    "margin":                 "Margin",
    "nn_agreement":           "NN-Agree",
    "nn_label_spreading":     "NN-LabelSpread",
    "stratified_random":      "StratifiedRandom",
    "coreset":                "Coreset-Lite",
    "uncertainty_diversity":             "Unc+Div-Lite",
    "badge_lite":                        "BADGE-Lite",
    "tracin_val_uncertainty":            "TracIn-CP (val)",
    "tracin_noise_weighted":             "TracIn-CP (noise)",
    "tracin_multicheckpoint_val":        "TracIn-CP (multi)",
    "expected_repair_value":             "RepairValue (tail)",
    "expected_repair_value_adaptive_tau": "RepairValue-Q (adaptive tau)",
    "expected_repair_value_capture_tau": "RepairValue-Q (capture tau)",
    "expected_repair_value_calibrated_tau": "RepairValue-Q (calibrated tau)",
    "noise_gated_repair_value":          "RepairValue-Q v2 (noise-gated)",
    "expected_repair_value_multiclass": "RV-Q (multiclass)",
    "reliability_gated_repair_value": "RV-Q (reliability-gated)",
    "auto_d3m_query":                    "AUTO-D3M-Q (adapt.)",
    "activeclean_expected_change":       "ActiveClean (EMC adapt.)",
    "active_label_cleaning":             "Active Label Cleaning",
    "oof_cleanlab":                      "Cleanlab (OOF)",
    "tracin_wga_oracle":                 "TracIn-CP (oracle)",
    "forgetting":                        "Forgetting",
    "random":                            "Random",
    "loss":                              "Loss",
    "entropy":                           "Entropy",
}


def _as_1d_float_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_frame_and_losses(
    private_frame: pd.DataFrame,
    losses: Sequence[float] | np.ndarray,
    *,
    required_cols: Iterable[str] = (),
) -> np.ndarray:
    losses_arr = _as_1d_float_array(losses, name="losses")
    if len(private_frame) != len(losses_arr):
        raise ValueError(
            f"private_frame and losses length mismatch: "
            f"{len(private_frame)} != {len(losses_arr)}"
        )
    missing = [col for col in required_cols if col not in private_frame.columns]
    if missing:
        raise KeyError(f"private_frame is missing required columns: {missing}")
    return losses_arr


def _stable_descending(values: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    """Return indices sorted by values descending, breaking ties by original index."""
    if indices is None:
        indices = np.arange(len(values), dtype=int)
    indices = np.asarray(indices, dtype=int)
    local_values = values[indices]
    order = np.lexsort((indices, -local_values))
    return indices[order]


def ranking_to_scores(
    ranked_indices: Sequence[int] | np.ndarray, n: int | None = None
) -> np.ndarray:
    """Convert a ranking into descending scores so existing top-B score code can reuse it."""
    ranked = np.asarray(ranked_indices, dtype=int)
    if ranked.ndim != 1:
        raise ValueError(f"ranked_indices must be 1D, got shape={ranked.shape}")
    if n is None:
        n = len(ranked)
    if len(ranked) != n:
        raise ValueError(f"ranking length must equal n: {len(ranked)} != {n}")
    if set(ranked.tolist()) != set(range(n)):
        raise ValueError("ranked_indices must be a permutation of 0..n-1")
    scores = np.empty(n, dtype=float)
    scores[ranked] = np.arange(n, 0, -1, dtype=float)
    return scores


def final_loss(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_probability = probabilities[np.arange(len(labels)), labels]
    return -np.log(np.clip(label_probability, 1e-12, 1.0))


def forgetting_counts(correctness_history: np.ndarray) -> np.ndarray:
    if correctness_history.ndim != 2:
        raise ValueError("correctness_history must have shape [n, epochs].")
    if correctness_history.shape[1] <= 1:
        return np.zeros(correctness_history.shape[0], dtype=np.int64)
    previous = correctness_history[:, :-1]
    current = correctness_history[:, 1:]
    return ((previous == 1) & (current == 0)).sum(axis=1)


def build_legal_scores(
    probabilities: np.ndarray,
    labels: np.ndarray,
    correctness_history: np.ndarray,
) -> dict[str, np.ndarray]:
    losses = final_loss(probabilities, labels)
    entropy = entropy_from_probabilities(probabilities)
    forgetting = forgetting_counts(correctness_history)
    disagreement = (probabilities.argmax(axis=1) != labels).astype(np.float64)
    noisy_label_confidence = probabilities[np.arange(len(labels)), labels]

    noise_score = (
        rank_within_class(losses, labels)
        + rank_within_class(1.0 - noisy_label_confidence, labels)
        + rank_within_class(entropy, labels)
        + rank_within_class(disagreement, labels)
    ) / 4.0

    return {
        "loss": losses,
        "entropy": entropy,
        "forgetting": forgetting.astype(np.float64),
        "noise_score": noise_score,
    }


def descending_ranking(
    primary: np.ndarray,
    secondary: np.ndarray | None = None,
) -> np.ndarray:
    primary = np.asarray(primary)
    if secondary is None:
        secondary = np.zeros(len(primary))
    secondary = np.asarray(secondary)
    # np.lexsort uses the last key as primary.
    return np.lexsort((-secondary, -primary)).astype(np.int64)


def random_ranking(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n).astype(np.int64)


def oracle_noise_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
) -> np.ndarray:
    """Preserved for backward compatibility. Alias for oracle_noise_loss_tiebreak."""
    return oracle_noise_loss_tiebreak_ranking(private_frame, losses)


def oracle_noise_loss_tiebreak_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
    *,
    noisy_col: str = "is_noisy",
) -> np.ndarray:
    """Noisy-first oracle with loss tie-break (preserves the old oracle_noise behavior).

    Priority:
    1. noisy samples first;
    2. within noisy samples, loss descending;
    3. clean samples after noisy samples, loss descending.
    """
    losses_arr = _validate_frame_and_losses(
        private_frame,
        losses,
        required_cols=(noisy_col,),
    )
    noisy = private_frame[noisy_col].astype(bool).to_numpy()
    all_indices = np.arange(len(losses_arr), dtype=int)
    noisy_ranked = _stable_descending(losses_arr, all_indices[noisy])
    clean_ranked = _stable_descending(losses_arr, all_indices[~noisy])
    return np.concatenate([noisy_ranked, clean_ranked]).astype(np.int64)


def oracle_noise_random_tiebreak_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
    *,
    noisy_col: str = "is_noisy",
    seed: int = 0,
) -> np.ndarray:
    """Noisy-first oracle with random order inside the noisy set.

    The clean tail remains sorted by loss descending.
    """
    losses_arr = _validate_frame_and_losses(
        private_frame,
        losses,
        required_cols=(noisy_col,),
    )
    noisy = private_frame[noisy_col].astype(bool).to_numpy()
    all_indices = np.arange(len(losses_arr), dtype=int)
    rng = np.random.default_rng(seed)
    noisy_indices = all_indices[noisy].copy()
    rng.shuffle(noisy_indices)
    clean_ranked = _stable_descending(losses_arr, all_indices[~noisy])
    return np.concatenate([noisy_indices.astype(np.int64), clean_ranked])


def _ordered_groups(
    noisy_frame: pd.DataFrame,
    *,
    group_col: str,
    order: str = "size_asc",
) -> list:
    if order == "appearance":
        return list(pd.unique(noisy_frame[group_col]))
    counts = noisy_frame[group_col].value_counts(sort=False)
    if order == "size_asc":
        return sorted(counts.index.tolist(), key=lambda g: (counts[g], str(g)))
    if order == "size_desc":
        return sorted(counts.index.tolist(), key=lambda g: (-counts[g], str(g)))
    if order == "sorted":
        return sorted(counts.index.tolist(), key=str)
    raise ValueError(f"unknown group order: {order}")


def oracle_noise_group_balanced_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
    *,
    group_col: str = "group",
    noisy_col: str = "is_noisy",
    group_order: str = "size_asc",
) -> np.ndarray:
    """Group-balanced oracle-noise ranking (diagnostic upper bound only).

    Priority:
    1. noisy samples first;
    2. within noisy samples, round-robin across true groups;
    3. within each group queue, loss descending;
    4. after all noisy samples, append clean samples by loss descending.

    Uses true group labels — only for diagnostic upper-bound analysis.
    """
    losses_arr = _validate_frame_and_losses(
        private_frame,
        losses,
        required_cols=(group_col, noisy_col),
    )
    noisy = private_frame[noisy_col].astype(bool).to_numpy()
    all_indices = np.arange(len(losses_arr), dtype=int)
    noisy_frame = private_frame.loc[noisy, [group_col]].copy()
    noisy_frame["_idx"] = all_indices[noisy]
    if noisy_frame.empty:
        return _stable_descending(losses_arr).astype(np.int64)
    queues = {}
    for group, sub in noisy_frame.groupby(group_col, sort=False):
        group_indices = sub["_idx"].to_numpy(dtype=int)
        queues[group] = deque(_stable_descending(losses_arr, group_indices).tolist())
    ranked_noisy: list[int] = []
    groups = _ordered_groups(noisy_frame, group_col=group_col, order=group_order)
    while any(queues[group] for group in groups):
        for group in groups:
            if queues[group]:
                ranked_noisy.append(queues[group].popleft())
    clean_ranked = _stable_descending(losses_arr, all_indices[~noisy])
    return np.concatenate([np.asarray(ranked_noisy, dtype=np.int64), clean_ranked])


def oracle_group_balanced_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
) -> np.ndarray:
    groups = private_frame["group"].to_numpy(dtype=int)
    queues: dict[int, list[int]] = {}
    for group in sorted(np.unique(groups)):
        members = np.where(groups == group)[0]
        ordered = members[np.argsort(-losses[members], kind="mergesort")]
        queues[int(group)] = list(int(value) for value in ordered)

    ranking: list[int] = []
    active_groups = sorted(queues)
    from collections import deque as _deque
    for group in active_groups:
        queues[group] = _deque(queues[group])
    while active_groups:
        next_active = []
        for group in active_groups:
            if queues[group]:
                ranking.append(queues[group].popleft())
            if queues[group]:
                next_active.append(group)
        active_groups = next_active

    return np.asarray(ranking, dtype=np.int64)


def oracle_disagreement_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
) -> np.ndarray:
    """Sort by annotator disagreement score descending (diagnostic baseline).

    Uses private annotator disagreement scores (2*min(toxicity, 1-toxicity))
    to rank samples by label ambiguity. Only available when the dataset
    provides multi-annotator scores (e.g. CivilComments).
    """
    if "disagreement_score" not in private_frame.columns:
        return _stable_descending(losses)
    disagreement = private_frame["disagreement_score"].to_numpy(dtype=float)
    order = np.lexsort((-np.asarray(losses), -disagreement))
    return order.astype(np.int64)


def score_by_method(
    method: str,
    private_frame: pd.DataFrame,
    losses: Sequence[float] | np.ndarray,
    *,
    group_col: str = "group",
    noisy_col: str = "is_noisy",
    seed: int = 0,
    group_order: str = "size_asc",
) -> np.ndarray:
    """Return scores for a query method. Higher score = higher query priority."""
    canonical = "oracle_noise_loss_tiebreak" if method == "oracle_noise" else method
    if canonical == "loss":
        losses_arr = _as_1d_float_array(losses, name="losses")
        ranking = _stable_descending(losses_arr)
    elif canonical == "oracle_noise_loss_tiebreak":
        ranking = oracle_noise_loss_tiebreak_ranking(
            private_frame, losses, noisy_col=noisy_col
        )
    elif canonical == "oracle_noise_random_tiebreak":
        ranking = oracle_noise_random_tiebreak_ranking(
            private_frame, losses, noisy_col=noisy_col, seed=seed
        )
    elif canonical == "oracle_noise_group_balanced":
        ranking = oracle_noise_group_balanced_ranking(
            private_frame, losses, group_col=group_col, noisy_col=noisy_col,
            group_order=group_order,
        )
    else:
        raise ValueError(
            f"unknown method={method!r}; expected one of {sorted(METHOD_NAMES)}"
        )
    return ranking_to_scores(ranking, len(private_frame))


def oracle_minority_ranking(
    private_frame: pd.DataFrame,
    losses: np.ndarray,
) -> np.ndarray:
    minority = private_frame["is_minority"].to_numpy(dtype=int)
    noisy = private_frame["is_noisy"].to_numpy(dtype=int)

    # Highest priority: noisy minority, clean minority, noisy majority, clean majority.
    category = np.where(
        (minority == 1) & (noisy == 1),
        3,
        np.where(
            (minority == 1) & (noisy == 0),
            2,
            np.where(noisy == 1, 1, 0),
        ),
    )
    return descending_ranking(category.astype(float), losses)


def aum_ranking(margin_history: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """Area Under the Margin (Pleiss et al. 2020).

    AUM[i] = mean over epochs of (label_prob - max_other_prob). Lower (more
    negative) AUM means the given label was consistently less likely than a
    competitor -> more likely mislabeled -> higher query priority.
    """
    if margin_history is None:
        raise ValueError("aum requires margin_history in context")
    if margin_history.ndim != 2:
        raise ValueError("margin_history must have shape [n, epochs]")
    aum = margin_history.mean(axis=1)
    order = np.lexsort((-np.asarray(losses), aum))
    return order.astype(np.int64)


def confident_learning_ranking(
    probabilities: np.ndarray,
    labels: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    """Confident Learning (Northcutt et al. 2021) via cleanlab label-quality scores.

    lower score = dirtier label -> higher query priority. In-sample probabilities
    are used (not cross-validated); this is a documented caveat of the baseline.
    """
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as exc:
        raise ImportError(
            "confident_learning requires the optional baseline dependency; "
            "install it with `pip install cleanlab>=2.7`."
        ) from exc

    scores = get_label_quality_scores(labels, pred_probs=probabilities)
    order = np.lexsort((-np.asarray(losses), scores))
    return order.astype(np.int64)


def _tracin_influence(mode: str, context: dict, noise_score: np.ndarray) -> np.ndarray:
    from robust_verify.influence import (
        compute_tracin_influence,
        uncertainty_weights,
        worst_group_mask,
    )

    val_probs = context["val_probs"]
    val_labels = context["val_labels"]
    common = dict(
        train_features=context["train_features"],
        train_probs=context["train_probs"],
        train_labels=context["train_labels"],
        val_features=context["val_features"],
        val_probs=val_probs,
        val_labels=val_labels,
    )
    if mode == "val_uncertainty":
        return compute_tracin_influence(**common, val_weights=uncertainty_weights(val_probs))
    if mode == "noise_weighted":
        influence = compute_tracin_influence(
            **common, val_weights=uncertainty_weights(val_probs)
        )
        # Larger positive influence means more harmful. normalized_rank with the
        # default direction maps larger values toward 1, matching NoiseScore.
        return normalized_rank(influence) * np.asarray(noise_score)
    if mode == "wga_oracle":
        mask = worst_group_mask(val_probs, val_labels, context["val_groups"])
        return compute_tracin_influence(**common, val_weights=mask.astype(np.float64))
    raise ValueError(f"unknown tracin mode: {mode}")


def _modern_baseline_score(
    method: str,
    context: dict,
    noise_score: np.ndarray,
    options: dict,
) -> np.ndarray:
    from robust_verify.modern_baselines import (
        active_label_cleaning_scores,
        activeclean_expected_model_change_scores,
        auto_d3m_query_adaptation_scores,
        expected_repair_value_adaptive_scores,
        expected_repair_value_capture_scores,
        expected_repair_value_calibrated_tau_scores,
        expected_repair_value_scores,
        expected_repair_value_multiclass_scores,
        reliability_gated_repair_value_scores,
        multicheckpoint_tracin_influence,
        oof_cleanlab_scores,
        repair_value_rank_scores,
    )

    common = {
        "train_features": context["train_features"],
        "train_labels": context["train_labels"],
        "val_features": context["val_features"],
        "val_labels": context["val_labels"],
    }
    if method == "tracin_multicheckpoint_val":
        train_history = context.get("probability_history")
        val_history = context.get("val_probability_history")
        if train_history is None or val_history is None:
            raise ValueError(
                "tracin_multicheckpoint_val requires train and validation "
                "probability histories from the same checkpoints"
            )
        return multicheckpoint_tracin_influence(
            train_features=common["train_features"],
            train_probability_history=train_history,
            train_labels=common["train_labels"],
            val_features=common["val_features"],
            val_probability_history=val_history,
            val_labels=common["val_labels"],
        )
    if method == "expected_repair_value":
        return expected_repair_value_scores(
            **common,
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            tail_fraction=float(options.get("repair_value_tail_fraction", 0.20)),
        )
    if method == "expected_repair_value_multiclass":
        return expected_repair_value_multiclass_scores(
            **common,
            train_probabilities=context["train_probs"],
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            tail_fraction=float(options.get("repair_value_tail_fraction", 0.20)),
        )
    if method == "expected_repair_value_adaptive_tau":
        candidates = tuple(
            float(value)
            for value in options.get(
                "repair_value_adaptive_tau_candidates",
                (0.05, 0.10, 0.20, 0.30, 0.50),
            )
        )
        method_scores, metadata = expected_repair_value_adaptive_scores(
            **common,
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            candidate_tail_fractions=candidates,
            num_folds=int(options.get("repair_value_adaptive_tau_folds", 5)),
            min_tail_count=int(options.get("repair_value_adaptive_tau_min_count", 4)),
            stability_sample_size=int(
                options.get("repair_value_adaptive_tau_sample_size", 4096)
            ),
            seed=int(context.get("seed", 0)),
        )
        context.setdefault("selection_metadata", {})[method] = metadata
        return method_scores
    if method == "expected_repair_value_capture_tau":
        candidates = tuple(
            float(value)
            for value in options.get(
                "repair_value_adaptive_tau_candidates",
                (0.05, 0.10, 0.20, 0.30, 0.50),
            )
        )
        budget_count = int(
            np.ceil(
                float(options.get("repair_value_capture_budget_fraction", 0.02))
                * len(context["train_labels"])
            )
        )
        method_scores, metadata = expected_repair_value_capture_scores(
            **common,
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            budget_count=budget_count,
            candidate_tail_fractions=candidates,
            num_folds=int(options.get("repair_value_adaptive_tau_folds", 5)),
            min_tail_count=int(options.get("repair_value_adaptive_tau_min_count", 4)),
            stability_sample_size=int(
                options.get("repair_value_adaptive_tau_sample_size", 4096)
            ),
            seed=int(context.get("seed", 0)),
        )
        context.setdefault("selection_metadata", {})[method] = metadata
        return method_scores
    if method == "expected_repair_value_calibrated_tau":
        candidates = tuple(
            float(value)
            for value in options.get(
                "repair_value_adaptive_tau_candidates",
                (0.05, 0.10, 0.20, 0.30, 0.50),
            )
        )
        budget_count = int(
            np.ceil(
                float(options.get("repair_value_capture_budget_fraction", 0.02))
                * len(context["train_labels"])
            )
        )
        method_scores, metadata = expected_repair_value_calibrated_tau_scores(
            **common,
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            budget_count=budget_count,
            candidate_tail_fractions=candidates,
            num_folds=int(options.get("repair_value_adaptive_tau_folds", 5)),
            min_tail_count=int(options.get("repair_value_adaptive_tau_min_count", 4)),
            stability_sample_size=int(
                options.get("repair_value_adaptive_tau_sample_size", 4096)
            ),
            seed=int(context.get("seed", 0)),
        )
        context.setdefault("selection_metadata", {})[method] = metadata
        return method_scores
    if method == "noise_gated_repair_value":
        return repair_value_rank_scores(
            **common,
            val_probabilities=context["val_probs"],
            tail_fraction=float(options.get("repair_value_tail_fraction", 0.20)),
        )
    if method == "reliability_gated_repair_value":
        default_reference_budget = max(
            1, int(np.ceil(0.05 * len(context["train_labels"])))
        )
        method_scores, metadata = reliability_gated_repair_value_scores(
            **common,
            train_probabilities=context["train_probs"],
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            tail_fraction=float(options.get("repair_value_tail_fraction", 0.20)),
            num_folds=int(options.get("repair_value_gate_folds", 5)),
            min_stability=float(options.get("repair_value_gate_min_stability", 0.50)),
            full_stability=float(options.get("repair_value_gate_full_stability", 0.85)),
            reference_budget_count=int(
                context.get("max_budget_count", default_reference_budget)
            ),
            min_noise_retention=float(
                options.get("repair_value_gate_min_noise_retention", 0.75)
            ),
            full_noise_retention=float(
                options.get("repair_value_gate_full_noise_retention", 0.95)
            ),
            seed=int(context.get("seed", 0)),
        )
        context.setdefault("selection_metadata", {})[method] = metadata
        return method_scores
    if method == "auto_d3m_query":
        return auto_d3m_query_adaptation_scores(
            **common,
            val_probabilities=context["val_probs"],
            noise_posterior_proxy=noise_score,
            options=options,
            cache=context.get("modern_baseline_cache"),
        )
    if method == "activeclean_expected_change":
        return activeclean_expected_model_change_scores(
            train_features=common["train_features"],
            train_probabilities=context["train_probs"],
            train_labels=common["train_labels"],
        )
    if method == "active_label_cleaning":
        return active_label_cleaning_scores(
            train_probabilities=context["train_probs"],
            train_labels=common["train_labels"],
        )
    if method == "oof_cleanlab":
        oof_probs = context.get("oof_train_probs")
        if oof_probs is None:
            raise ValueError("oof_cleanlab requires cached out-of-fold probabilities")
        return oof_cleanlab_scores(
            oof_train_probabilities=oof_probs,
            train_labels=common["train_labels"],
        )
    raise ValueError(f"unknown modern baseline method: {method}")


def margin_ranking(probabilities: np.ndarray, labels: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """Final-epoch margin uncertainty sampling.

    margin = p[label] - max_{k != label} p[k]. Smaller margin = more uncertain =
    higher query priority. Tie-break: loss descending.
    """
    n = len(labels)
    label_prob = probabilities[np.arange(n), labels]
    alt = probabilities.copy()
    alt[np.arange(n), labels] = -np.inf
    max_other = alt.max(axis=1)
    margin = label_prob - max_other
    order = np.lexsort((-np.asarray(losses), margin))
    return order.astype(np.int64)


_CORESET_CACHE: dict[tuple, np.ndarray] = {}


def _stratified_random_ranking(
    labels: np.ndarray,
    losses: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Stratified random query: random sample within each noisy-label class.

    Queries are allocated proportional to class size, then randomized within
    each class. Loss tie-break ensures deterministic ordering within class.
    This is a stronger random baseline than simple permutation random.
    """
    rng = np.random.default_rng(seed + 20_000)
    n = len(labels)
    ranking = np.empty(n, dtype=np.int64)
    pos = 0
    for c in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        ranking[pos : pos + len(idx)] = idx
        pos += len(idx)
    return ranking


def nn_label_spreading_ranking(
    train_features: np.ndarray,
    labels: np.ndarray,
    losses: np.ndarray,
    k: int = 20,
    temperature: float = 0.5,
) -> np.ndarray:
    """Nearest-neighbor label spreading baseline.

    Inspired by label propagation and spreading methods. For each sample,
    compute a pseudo-label via weighted k-NN voting in frozen feature space:
        pred_i = softmax(sum_j sim(x_i, x_j) * onehot(y_j)) / temperature
    The disagreement between the noisy label and the pseudo-label serves as
    a noise indicator. Higher disagreement → more likely mislabeled → higher
    query priority.

    This is more principled than simple NN-Agree: it uses soft cosine
    similarity weights and entropy of the NN prediction distribution to
    identify label-ambiguous regions.
    """
    from sklearn.metrics import pairwise_distances

    n = int(train_features.shape[0])
    k_eff = min(k, n - 1)
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_classes = int(labels_arr.max() + 1)

    distances = pairwise_distances(train_features, metric="cosine", n_jobs=1)
    np.fill_diagonal(distances, np.inf)
    nn_idx = np.argpartition(distances, k_eff, axis=1)[:, :k_eff]

    similarities = 1.0 - distances[np.arange(n)[:, None], nn_idx]
    similarities = np.exp(similarities / max(temperature, 1e-8))

    nn_labels = labels_arr[nn_idx]
    pseudo_probs = np.zeros((n, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = (nn_labels == c).astype(np.float64)
        pseudo_probs[:, c] = np.sum(similarities * mask, axis=1) / (
            similarities.sum(axis=1) + 1e-12
        )

    noisy_label_prob = pseudo_probs[np.arange(n), labels_arr]
    pseudo_entropy = -np.sum(
        pseudo_probs * np.log(np.clip(pseudo_probs, 1e-12, 1.0)), axis=1
    )

    score = (1.0 - noisy_label_prob) + 0.3 * normalized_rank(pseudo_entropy)
    order = np.lexsort((-np.asarray(losses), -score))
    return order.astype(np.int64)


def nn_agreement_ranking(
    train_features: np.ndarray,
    labels: np.ndarray,
    losses: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Nearest-neighbor label agreement baseline.

    For each sample, compute the fraction of its k-nearest neighbors (in frozen
    feature space) that share the same noisy label.  Low agreement signals that
    the sample is an outlier relative to its class → more likely mislabeled →
    higher query priority.

    This is a fully legal method: it uses only frozen features and noisy
    labels (no clean labels, no group annotations).  Distance computation uses
    ``sklearn.metrics.pairwise_distances``.
    """
    from sklearn.metrics import pairwise_distances

    n = int(train_features.shape[0])
    k = min(k, n - 1)
    distances = pairwise_distances(train_features, metric="cosine", n_jobs=1)
    # Set self-distance to inf so that the sample itself is not counted
    np.fill_diagonal(distances, np.inf)
    nn_indices = np.argpartition(distances, k, axis=1)[:, :k]

    labels = np.asarray(labels, dtype=np.int64)
    agreement = (labels[nn_indices] == labels[:, None]).mean(axis=1)
    # Lower agreement = more likely mislabeled → higher priority
    order = np.lexsort((-np.asarray(losses), agreement))
    return order.astype(np.int64)


def _complete_ranking(prefix: np.ndarray, fallback_scores: np.ndarray) -> np.ndarray:
    """Append all unselected examples in stable descending fallback order."""
    prefix = np.asarray(prefix, dtype=np.int64)
    fallback_scores = np.asarray(fallback_scores, dtype=np.float64)
    n = len(fallback_scores)
    if len(prefix) != len(np.unique(prefix)) or np.any((prefix < 0) | (prefix >= n)):
        raise ValueError("prefix must contain unique indices in [0, n)")
    used = np.zeros(n, dtype=bool)
    used[prefix] = True
    remaining = np.flatnonzero(~used)
    tail = _stable_descending(fallback_scores, remaining)
    return np.concatenate([prefix, tail]).astype(np.int64)


def _project_embeddings(
    embeddings: np.ndarray, *, projection_dim: int, seed: int
) -> np.ndarray:
    """Deterministic Gaussian projection used only by scalable diversity baselines."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={x.shape}")
    if projection_dim <= 0 or x.shape[1] <= projection_dim:
        return x
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0.0, 1.0 / np.sqrt(projection_dim), size=(x.shape[1], projection_dim)
    ).astype(np.float32)
    return x @ projection


def _farthest_first_ranking(
    embeddings: np.ndarray,
    *,
    seed_weights: np.ndarray | None = None,
    beta: float = 0.0,
    max_select: int | None = None,
) -> np.ndarray:
    """Farthest-first prefix, optionally truncated after ``max_select`` points.

    If ``seed_weights`` is given and ``beta > 0``, each step picks
    argmax( (1-beta)*norm(w) + beta*norm(min_dist) ) instead of pure farthest-first,
    implementing an uncertainty-diversity hybrid.
    """
    n = len(embeddings)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if max_select is None:
        max_select = n
    max_select = max(1, min(int(max_select), n))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = embeddings / np.clip(norms, 1e-12, None)
    self_norm = (X * X).sum(1)
    cen = X.mean(0)
    d2_cen = self_norm + float((cen * cen).sum()) - 2.0 * (X @ cen)
    if seed_weights is not None:
        first = int(np.argmax(seed_weights * (1.0 + d2_cen / (d2_cen.max() + 1e-12))))
    else:
        first = int(np.argmax(d2_cen))
    order = [first]
    min_d2 = (self_norm + self_norm[first] - 2.0 * (X @ X[first])).astype(np.float64)
    min_d2[first] = -np.inf
    if beta > 0 and seed_weights is not None:
        w = seed_weights / (seed_weights.max() + 1e-12)
        for _ in range(max_select - 1):
            finite = np.isfinite(min_d2)
            if not finite.any():
                break
            dmax = float(min_d2[finite].max()) + 1e-12
            score = (1.0 - beta) * w + beta * (min_d2 / dmax)
            score[~finite] = -np.inf
            pick = int(np.argmax(score))
            order.append(pick)
            new_d2 = self_norm + self_norm[pick] - 2.0 * (X @ X[pick])
            min_d2 = np.minimum(min_d2, new_d2)
            min_d2[pick] = -np.inf
    else:
        for _ in range(max_select - 1):
            pick = int(np.argmax(min_d2))
            order.append(pick)
            new_d2 = self_norm + self_norm[pick] - 2.0 * (X @ X[pick])
            min_d2 = np.minimum(min_d2, new_d2)
            min_d2[pick] = -np.inf
    return np.asarray(order, dtype=np.int64)


def _scalable_diversity_ranking(
    embeddings: np.ndarray,
    *,
    fallback_scores: np.ndarray,
    seed_weights: np.ndarray | None,
    beta: float,
    seed: int,
    max_budget_count: int,
    options: dict | None,
) -> np.ndarray:
    """Build a useful diverse prefix without an O(n^2) full-data traversal.

    The candidate pool and farthest-first prefix are capped.  After that prefix,
    the ranking falls back to the method's uncertainty score (or a seeded random
    order for pure coreset).  This keeps every ranking a full permutation while
    making the baseline practical on the two large datasets.
    """
    opts = options or {}
    n = len(embeddings)
    max_candidates = max(1, int(opts.get("max_candidates", 4096)))
    max_steps = max(1, int(opts.get("max_diversity_steps", 1024)))
    projection_dim = int(opts.get("projection_dim", 32))
    candidate_multiplier = max(1, int(opts.get("candidate_multiplier", 4)))
    select_count = min(n, max(1, int(max_budget_count)), max_steps)
    pool_count = min(n, max_candidates, max(select_count, candidate_multiplier * select_count))

    fallback_scores = np.asarray(fallback_scores, dtype=np.float64)
    if seed_weights is None:
        rng = np.random.default_rng(seed)
        random_priority = rng.random(n)
        pool = _stable_descending(random_priority)[:pool_count]
        completion_scores = random_priority
        local_weights = None
    else:
        seed_weights = np.asarray(seed_weights, dtype=np.float64)
        pool = _stable_descending(seed_weights)[:pool_count]
        completion_scores = fallback_scores
        local_weights = seed_weights[pool]

    projected = _project_embeddings(
        np.asarray(embeddings)[pool], projection_dim=projection_dim, seed=seed + 991
    )
    local_prefix = _farthest_first_ranking(
        projected,
        seed_weights=local_weights,
        beta=beta,
        max_select=select_count,
    )
    return _complete_ranking(pool[local_prefix], completion_scores)


def coreset_ranking(
    train_features: np.ndarray,
    *,
    seed: int = 0,
    max_budget_count: int | None = None,
    options: dict | None = None,
) -> np.ndarray:
    """Scalable k-center-greedy prefix on frozen features."""
    if max_budget_count is None:
        max_budget_count = len(train_features)
    opts_key = tuple(sorted((options or {}).items()))
    key = (id(train_features), seed, int(max_budget_count), opts_key)
    if key in _CORESET_CACHE:
        return _CORESET_CACHE[key]
    ranking = _scalable_diversity_ranking(
        train_features,
        fallback_scores=np.zeros(len(train_features), dtype=np.float64),
        seed_weights=None,
        beta=0.0,
        seed=seed,
        max_budget_count=max_budget_count,
        options=options,
    )
    _CORESET_CACHE[key] = ranking
    return ranking


def uncertainty_diversity_ranking(
    probabilities: np.ndarray,
    train_features: np.ndarray,
    losses: np.ndarray,
    *,
    seed: int = 0,
    max_budget_count: int | None = None,
    options: dict | None = None,
) -> np.ndarray:
    """Uncertainty + diversity hybrid: farthest-first on features, each step balances
    normalized entropy against distance-to-selected (beta=0.5). Seeded by entropy."""
    entropy = entropy_from_probabilities(probabilities)
    if max_budget_count is None:
        max_budget_count = len(train_features)
    return _scalable_diversity_ranking(
        train_features,
        fallback_scores=entropy,
        seed_weights=entropy,
        beta=0.5,
        seed=seed,
        max_budget_count=max_budget_count,
        options=options,
    )


def badge_lite_ranking(
    probabilities: np.ndarray,
    labels: np.ndarray,
    train_features: np.ndarray,
    losses: np.ndarray,
    *,
    seed: int = 0,
    max_budget_count: int | None = None,
    options: dict | None = None,
) -> np.ndarray:
    """BADGE-lite (Ash et al. 2020, simplified): farthest-first on gradient embeddings
    g_i = (p_i - y_onehot) kron x_i, seeded by loss (uncertainty proxy)."""
    if max_budget_count is None:
        max_budget_count = len(train_features)
    opts = options or {}
    n = len(labels)
    max_candidates = max(1, int(opts.get("max_candidates", 4096)))
    max_steps = max(1, int(opts.get("max_diversity_steps", 1024)))
    candidate_multiplier = max(1, int(opts.get("candidate_multiplier", 4)))
    select_count = min(n, max(1, int(max_budget_count)), max_steps)
    pool_count = min(n, max_candidates, max(select_count, candidate_multiplier * select_count))
    loss_scores = np.asarray(losses, dtype=np.float64)
    pool = _stable_descending(loss_scores)[:pool_count]

    # Construct gradient embeddings only for the candidate pool.  Materialising
    # them for all CivilComments examples can otherwise require several GB.
    resid = probabilities[pool].astype(np.float32).copy()
    resid[np.arange(pool_count), labels[pool]] -= 1.0
    grad_emb = np.einsum(
        "nk,nd->nkd", resid, np.asarray(train_features)[pool].astype(np.float32)
    ).reshape(pool_count, -1)
    projected = _project_embeddings(
        grad_emb,
        projection_dim=int(opts.get("projection_dim", 32)),
        seed=seed + 991,
    )
    local_prefix = _farthest_first_ranking(
        projected,
        seed_weights=loss_scores[pool],
        beta=0.5,
        max_select=select_count,
    )
    return _complete_ranking(pool[local_prefix], loss_scores)


def noise_score_coverage_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    *,
    warm_start_per_bucket: int = 4,
) -> np.ndarray:
    """Coverage-aware NoiseScore v2 ranking.

    Uses predicted class as the coverage bucket to prevent any single class
    from monopolizing the top-B prefix. The bucket is a LEGAL proxy — it only
    uses the model's own predictions, not private group labels.

    Internally delegates to the bundled coverage-selection utility.
    """
    from robust_verify.selection import coverage_aware_ranking

    buckets = probabilities.argmax(axis=1).astype(str)
    return coverage_aware_ranking(
        noise_score,
        buckets,
        bucket_order="size_asc",
        warm_start_per_bucket=warm_start_per_bucket,
        fill_remaining_by_score=True,
    )


def noise_score_cluster_coverage_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    *,
    cluster_path: str = "outputs/cluster_assignments/clusters_k10.csv",
    warm_start_per_bucket: int = 4,
) -> np.ndarray:
    """Coverage-aware NoiseScore v3 ranking using representation clusters.

    Uses k-means clusters on ResNet features as coverage buckets. This provides
    finer-grained coverage than predicted_class (only 2 classes), helping to
    diversify query composition at larger budgets.

    The cluster file must exist at ``cluster_path`` and contain columns
    ``sample_id`` and ``cluster``. Clusters are generated from frozen ResNet
    features via k-means and do NOT use private group labels.

    Defaults to k=10, warm_start=4 based on cross-budget sweep across
    uniform20 and minority_high_40 (best composite score across 2%/5%/10%).
    """
    from pathlib import Path
    from robust_verify.selection import coverage_aware_ranking

    cluster_df = pd.read_csv(Path(cluster_path))
    buckets = _align_clusters(private_frame, cluster_df)

    return coverage_aware_ranking(
        noise_score,
        buckets,
        bucket_order="size_asc",
        warm_start_per_bucket=warm_start_per_bucket,
        fill_remaining_by_score=True,
    )


def _align_clusters(private_frame, cluster_df):
    """Map sample_ids to cluster labels with validation."""
    cluster_map = dict(zip(cluster_df["sample_id"], cluster_df["cluster"]))
    sample_ids = private_frame["sample_id"].to_numpy(dtype=int)
    buckets = []
    unk = 0
    for sid in sample_ids:
        c = cluster_map.get(int(sid))
        if c is None:
            unk += 1
            c = 0
        buckets.append(str(c))
    if unk > 0:
        print(f"[WARNING] _align_clusters: {unk}/{len(sample_ids)} sample_ids not in cluster file. "
              f"Cluster file may be for a different dataset. Coverage diversification may be impaired.")
    return np.array(buckets)


def _build_v8_ngc_ranking(
    noise_score: np.ndarray,
    buckets: np.ndarray,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
) -> np.ndarray:
    """NGC (v8) ranking: budget gate + noise-risk loss-ratio gate.

    Used as a standalone method and as a safe fallback policy for ablations.
    """
    from robust_verify.selection import budget_hybrid_coverage_ranking

    switch_budget_count = max(1, int(round(n_total * 0.02)))
    if budget_count <= switch_budget_count:
        return descending_ranking(noise_score, losses)

    loss_sorted = np.argsort(-losses)
    topb = loss_sorted[:min(budget_count, len(losses))]
    loss_ratio = float(losses[topb].mean() / losses.mean())

    if loss_ratio > NOISE_RISK_LOSS_RATIO_THRESHOLD:
        return descending_ranking(noise_score, losses)

    ranking, _ = budget_hybrid_coverage_ranking(
        noise_score,
        buckets,
        budget_count=budget_count,
        switch_budget_fraction=0.02,
        coverage_ratio=None,
        ratio_schedule="default",
        bucket_order="size_asc",
        return_metadata=True,
    )
    return ranking


def _rank_preservation(losses: np.ndarray, labels: np.ndarray) -> float:
    """Spearman correlation between class-internal and global rank of losses.

    High value -> class-internal ranking preserves the global signal -> safe.
    Low value  -> class-internal ranking distorts the signal -> need correction.
    """
    class_rank = rank_within_class(losses, labels)
    global_rank = normalized_rank(losses)
    class_centered = class_rank - class_rank.mean()
    global_centered = global_rank - global_rank.mean()
    denominator = np.linalg.norm(class_centered) * np.linalg.norm(global_centered)
    rho = float(class_centered @ global_centered / denominator) if denominator > 0 else 1.0
    if not np.isfinite(rho):
        rho = 1.0  # conservative: avoid over-correction
    return max(0.0, min(1.0, rho))


def _class_mean_loss(losses: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Mean loss per noisy-label class (legal noise-propensity proxy)."""
    result = {}
    for c in np.unique(labels):
        mask = labels == c
        result[int(c)] = float(losses[mask].mean()) if mask.any() else 0.0
    return result


def _build_cpba_ablation_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
    use_cpba: bool = True,
    use_tc: bool = True,
    fallback_policy: str = "v1",
    alpha_threshold: float = 0.90,
    ablation_name: str = "cpba_tc",
) -> tuple[np.ndarray, dict]:
    """Parameterized CPBA-TC ablation engine.

    Combines two optional mechanisms:
    1. CPBA — class-prior balanced allocation via loss-based reweighting.
    2. TC   — tail coverage via cluster-coverage diversification.

    Fallback policies (when CPBA signal is weak, i.e. alpha >= threshold):
      - ``"v1"``      : pure NoiseScore ranking
      - ``"none"``    : no hard fallback; use CPBA-reweighted ranking
      - ``"v8"``      : NGC (v8) noise-risk-gated coverage
      - ``"entropy"`` : entropy-only ranking

    Returns (ranking, metadata_dict).
    """
    import os
    from pathlib import Path
    from robust_verify.selection import budget_hybrid_coverage_ranking

    if cluster_path is None:
        k = os.environ.get("NOISE_SCORE_CLUSTER_K", "20")
        cluster_path = f"outputs/cluster_assignments/clusters_k{k}.csv"

    n = len(labels)
    classes = sorted(np.unique(labels).tolist())
    num_classes = len(classes)
    cluster_path_obj = Path(cluster_path)

    # --- Precompute base rankings for fallbacks ---
    v1_ranking = descending_ranking(noise_score, losses)
    entropy_ranking: np.ndarray | None = None
    if fallback_policy == "entropy":
        entropy_score = entropy_from_probabilities(probabilities)
        entropy_ranking = descending_ranking(entropy_score, losses)

    def _topb_class_counts(ranking: np.ndarray, bc: int) -> dict[int, int]:
        top = ranking[:min(bc, len(ranking))]
        top_labels = labels[top] if len(top) > 0 else np.array([])
        return {int(c): int((top_labels == c).sum()) for c in classes}

    # --- Risk estimation (always computed for metadata regardless of use_cpba) ---
    alpha = _rank_preservation(losses, labels)
    ml = _class_mean_loss(losses, labels)
    ml0 = ml.get(classes[0], 0.0)
    ml1 = ml.get(classes[1], 0.0) if num_classes > 1 else 0.0
    class_loss_ratio = round(ml1 / (ml0 + 1e-8), 4) if num_classes > 1 else 0.0

    risk_order = sorted(classes, key=lambda c: ml.get(c, 0.0), reverse=True)
    high_risk_class = risk_order[0] if len(risk_order) > 0 else classes[0]
    total_ml = sum(ml.values()) + 1e-12

    fallback_triggered = False
    gate_decision = "cpba_activated"
    weight_class0 = 1.0
    weight_class1 = 1.0

    # Pre-compute "wouldbe" weights (what CPBA would apply if not gated)
    # Useful for ablation diagnostics — shows how much fallback deviates
    wouldbe_weight_class0 = float(max(0.5, min(2.0,
        1.0 + (1.0 - alpha) * ((ml0 / total_ml) / (1.0 / num_classes) - 1.0)
    )))
    wouldbe_weight_class1 = wouldbe_weight_class0
    if num_classes > 1:
        wouldbe_weight_class1 = float(max(0.5, min(2.0,
            1.0 + (1.0 - alpha) * ((ml1 / total_ml) / (1.0 / num_classes) - 1.0)
        )))

    # --- Fallback gate ---
    if use_cpba and fallback_policy != "none" and alpha >= alpha_threshold:
        fallback_triggered = True
        gate_decision = f"fallback_{fallback_policy}"

        if fallback_policy == "v1":
            ranking = v1_ranking
        elif fallback_policy == "v8":
            # Build NGC-style ranking for fallback
            if cluster_path_obj.exists():
                cluster_df_fb = pd.read_csv(cluster_path_obj)
                cluster_map_fb = dict(zip(cluster_df_fb["sample_id"], cluster_df_fb["cluster"]))
                sample_ids_fb = private_frame["sample_id"].to_numpy(dtype=int)
                buckets_fb = np.array([str(cluster_map_fb.get(int(sid), 0)) for sid in sample_ids_fb])
            else:
                buckets_fb = np.array([str(0) for _ in range(n)])
            ranking = _build_v8_ngc_ranking(
                noise_score, buckets_fb, budget_count, n_total, losses=losses,
            )
        elif fallback_policy == "entropy" and entropy_ranking is not None:
            ranking = entropy_ranking
        else:
            ranking = v1_ranking

        topb_counts_fb = _topb_class_counts(ranking, budget_count)
        metadata = {
            "method": ablation_name,
            "ablation_name": ablation_name,
            "use_cpba": use_cpba,
            "use_tc": use_tc,
            "fallback_policy": fallback_policy,
            "alpha_threshold": alpha_threshold,
            "alpha": round(alpha, 4),
            "mean_loss_class0": round(ml0, 4),
            "mean_loss_class1": round(ml1, 4),
            "class_loss_ratio": class_loss_ratio,
            "high_risk_class": int(high_risk_class),
            "gate_decision": gate_decision,
            "fallback_triggered": fallback_triggered,
            "weight_class0": round(weight_class0, 4),
            "weight_class1": round(weight_class1, 4),
            "wouldbe_weight_class0": round(wouldbe_weight_class0, 4),
            "wouldbe_weight_class1": round(wouldbe_weight_class1, 4),
            "tc_activated": False,
            "num_classes": num_classes,
            "cluster_entropy_before": 0.0,
            "cluster_entropy_after": 0.0,
            "num_replacements": 0,
            "alloc_class_0": topb_counts_fb.get(classes[0], 0),
            "alloc_class_1": topb_counts_fb.get(classes[1], 0) if num_classes > 1 else 0,
        }
        return (ranking, metadata)

    # --- CPBA reweighting (if enabled) ---
    if use_cpba:
        reweighted = noise_score.copy().astype(np.float64)
        for c in classes:
            mask = labels == c
            if not mask.any():
                continue
            risk_share = ml.get(c, 0.0) / total_ml
            avg_share = 1.0 / num_classes
            risk_ratio = risk_share / (avg_share + 1e-12)
            weight = 1.0 + (1.0 - alpha) * (risk_ratio - 1.0)
            weight = float(max(0.5, min(2.0, weight)))
            reweighted[mask] *= weight
        base_score = reweighted
        # Record class-level weights
        weight_class0 = float(max(0.5, min(2.0, 1.0 + (1.0 - alpha) * (
            (ml.get(classes[0], 0.0) / total_ml) / (1.0 / num_classes) - 1.0
        ))))
        if num_classes > 1:
            weight_class1 = float(max(0.5, min(2.0, 1.0 + (1.0 - alpha) * (
                (ml1 / total_ml) / (1.0 / num_classes) - 1.0
            ))))
    else:
        base_score = noise_score

    base_ranking = descending_ranking(base_score, losses)

    # --- Tail Coverage (if enabled) ---
    switch_budget = max(1, int(round(n_total * 0.02)))
    tc_activated = False
    cluster_entropy_before = 0.0
    cluster_entropy_after = 0.0
    num_replacements = 0

    if not use_tc or budget_count <= switch_budget or not cluster_path_obj.exists():
        full_ranking = list(base_ranking[:budget_count])
        full_set = set(full_ranking)
        for idx in descending_ranking(base_score, losses):
            if int(idx) not in full_set:
                full_ranking.append(int(idx))
                full_set.add(int(idx))
        ranking = np.array(full_ranking, dtype=np.int64)
    else:
        cluster_df = pd.read_csv(cluster_path_obj)
        buckets = _align_clusters(private_frame, cluster_df)

        base_top = base_ranking[:budget_count]
        base_clusters = buckets[base_top]
        _, base_counts = np.unique(base_clusters, return_counts=True)
        base_probs = base_counts / base_counts.sum()
        cluster_entropy_before = float(
            -np.sum(base_probs * np.log(base_probs + 1e-12))
        )

        coverage_ranking, _ = budget_hybrid_coverage_ranking(
            base_score,
            buckets,
            budget_count=budget_count,
            switch_budget_fraction=0.02,
            coverage_ratio=None,
            ratio_schedule="default",
            bucket_order="size_asc",
            return_metadata=True,
        )

        tc_activated = True
        base_count = budget_count // 2
        merged: list[int] = [int(i) for i in base_ranking[:base_count]]
        merged_set: set[int] = set(merged)
        for idx in coverage_ranking:
            idx_i = int(idx)
            if idx_i not in merged_set:
                merged.append(idx_i)
                merged_set.add(idx_i)
                num_replacements += 1
                if len(merged) >= budget_count:
                    break

        ranking = np.array(merged[:budget_count], dtype=np.int64)

        after_clusters = buckets[ranking]
        _, after_counts = np.unique(after_clusters, return_counts=True)
        after_probs = after_counts / after_counts.sum()
        cluster_entropy_after = float(
            -np.sum(after_probs * np.log(after_probs + 1e-12))
        )

    topb_counts = _topb_class_counts(ranking, budget_count)

    metadata = {
        "method": ablation_name,
        "ablation_name": ablation_name,
        "use_cpba": use_cpba,
        "use_tc": use_tc,
        "fallback_policy": fallback_policy,
        "alpha_threshold": alpha_threshold,
        "alpha": round(alpha, 4),
        "mean_loss_class0": round(ml0, 4),
        "mean_loss_class1": round(ml1, 4),
        "class_loss_ratio": class_loss_ratio,
        "high_risk_class": int(high_risk_class),
        "gate_decision": gate_decision,
        "fallback_triggered": fallback_triggered,
        "weight_class0": round(weight_class0, 4),
        "weight_class1": round(weight_class1, 4),
        "wouldbe_weight_class0": round(wouldbe_weight_class0, 4),
        "wouldbe_weight_class1": round(wouldbe_weight_class1, 4),
        "tc_activated": tc_activated,
        "num_classes": num_classes,
        "cluster_entropy_before": round(cluster_entropy_before, 4),
        "cluster_entropy_after": round(cluster_entropy_after, 4),
        "num_replacements": num_replacements,
        "alloc_class_0": topb_counts.get(classes[0], 0),
        "alloc_class_1": topb_counts.get(classes[1], 0) if num_classes > 1 else 0,
    }

    return (ranking, metadata)


# ── Ablation wrapper functions ────────────────────────────────────────────

def build_v9_cpba_tc_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """v9 CPBA-TC: full method with v1 fallback (original)."""
    return _build_cpba_ablation_ranking(
        noise_score, probabilities, private_frame, budget_count, n_total,
        losses=losses, labels=labels, cluster_path=cluster_path,
        use_cpba=True, use_tc=True, fallback_policy="v1",
        ablation_name="cpba_tc",
    )


def build_cpba_only_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """CPBA-only: class-prior reweighting, no tail coverage, no fallback."""
    return _build_cpba_ablation_ranking(
        noise_score, probabilities, private_frame, budget_count, n_total,
        losses=losses, labels=labels, cluster_path=cluster_path,
        use_cpba=True, use_tc=False, fallback_policy="none",
        ablation_name="cpba_only",
    )


def build_tc_only_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """TC-only: tail coverage only, no CPBA reweighting, no fallback."""
    return _build_cpba_ablation_ranking(
        noise_score, probabilities, private_frame, budget_count, n_total,
        losses=losses, labels=labels, cluster_path=cluster_path,
        use_cpba=False, use_tc=True, fallback_policy="none",
        ablation_name="tc_only",
    )


def build_cpba_tc_no_fallback_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """CPBA-TC without hard fallback: alpha modulates weight continuously."""
    return _build_cpba_ablation_ranking(
        noise_score, probabilities, private_frame, budget_count, n_total,
        losses=losses, labels=labels, cluster_path=cluster_path,
        use_cpba=True, use_tc=True, fallback_policy="none",
        ablation_name="cpba_tc_no_fallback",
    )


def build_cpba_tc_v8_fallback_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """CPBA-TC with NGC (v8) fallback when alpha signal is weak."""
    return _build_cpba_ablation_ranking(
        noise_score, probabilities, private_frame, budget_count, n_total,
        losses=losses, labels=labels, cluster_path=cluster_path,
        use_cpba=True, use_tc=True, fallback_policy="v8",
        ablation_name="cpba_tc_v8_fallback",
    )


def build_cpba_tc_entropy_fallback_ranking(
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray,
    labels: np.ndarray,
    cluster_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """CPBA-TC with entropy fallback when alpha signal is weak."""
    return _build_cpba_ablation_ranking(
        noise_score, probabilities, private_frame, budget_count, n_total,
        losses=losses, labels=labels, cluster_path=cluster_path,
        use_cpba=True, use_tc=True, fallback_policy="entropy",
        ablation_name="cpba_tc_entropy_fallback",
    )


# ── Module-level dispatch tables for CPBA ablations ────────────────────────

CPBA_METHODS = {
    "noise_score_cpba_tc",
    "noise_score_cpba_only",
    "noise_score_tc_only",
    "noise_score_cpba_tc_no_fallback",
    "noise_score_cpba_tc_v8_fallback",
    "noise_score_cpba_tc_entropy_fallback",
}

_CPBA_DISPATCH = {
    "noise_score_cpba_tc":                  build_v9_cpba_tc_ranking,
    "noise_score_cpba_only":                build_cpba_only_ranking,
    "noise_score_tc_only":                  build_tc_only_ranking,
    "noise_score_cpba_tc_no_fallback":      build_cpba_tc_no_fallback_ranking,
    "noise_score_cpba_tc_v8_fallback":      build_cpba_tc_v8_fallback_ranking,
    "noise_score_cpba_tc_entropy_fallback": build_cpba_tc_entropy_fallback_ranking,
}


def build_noise_gated_repair_value_ranking(
    noise_score: np.ndarray,
    repair_value_score: np.ndarray,
    losses: np.ndarray,
    budget_count: int,
    n_total: int,
    *,
    candidate_multiplier: float = 2.0,
    noise_anchor_fraction: float = 0.5,
) -> np.ndarray:
    """Build the hard-gated RepairValue-Q v2 ranking for one budget.

    The first ``noise_anchor_fraction * B`` selections are detector anchors.
    Remaining selections maximize the repair-value component inside the top
    ``candidate_multiplier * B`` NoiseScore candidates.  NoiseScore and loss
    break repair-value ties.  The returned array is a complete permutation,
    although only its first ``budget_count`` entries are queried.
    """
    noise = _as_1d_float_array(noise_score, name="noise_score")
    repair = _as_1d_float_array(repair_value_score, name="repair_value_score")
    tie_losses = _as_1d_float_array(losses, name="losses")
    if n_total != len(noise) or len(repair) != n_total or len(tie_losses) != n_total:
        raise ValueError("noise-gated RepairValue arrays must match n_total")
    if not 1 <= budget_count <= n_total:
        raise ValueError("budget_count must lie in [1, n_total]")
    if not math.isfinite(float(candidate_multiplier)) or candidate_multiplier < 1.0:
        raise ValueError("candidate_multiplier must be finite and at least 1")
    if not math.isfinite(float(noise_anchor_fraction)) or not 0 <= noise_anchor_fraction <= 1:
        raise ValueError("noise_anchor_fraction must lie in [0, 1]")

    noise_order = descending_ranking(noise, tie_losses)
    candidate_count = min(
        n_total,
        max(budget_count, math.ceil(candidate_multiplier * budget_count)),
    )
    anchor_count = min(budget_count, math.ceil(noise_anchor_fraction * budget_count))
    anchors = noise_order[:anchor_count]
    candidates = noise_order[anchor_count:candidate_count]
    candidate_order = np.lexsort(
        (-tie_losses[candidates], -noise[candidates], -repair[candidates])
    )
    reranked_candidates = candidates[candidate_order]
    selected = np.concatenate(
        [anchors, reranked_candidates[: budget_count - anchor_count]]
    ).astype(np.int64, copy=False)

    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[selected] = True
    candidate_remainder = reranked_candidates[budget_count - anchor_count :]
    outside = noise_order[candidate_count:]
    full = np.concatenate([selected, candidate_remainder, outside]).astype(np.int64, copy=False)
    if (
        len(full) != n_total
        or len(np.unique(full)) != n_total
        or not selected_mask[full[:budget_count]].all()
    ):
        raise RuntimeError("noise-gated RepairValue did not produce a valid permutation")
    return full


def build_adaptive_ranking(
    method: str,
    noise_score: np.ndarray,
    probabilities: np.ndarray,
    private_frame: pd.DataFrame,
    budget_count: int,
    n_total: int,
    *,
    losses: np.ndarray | None = None,
    cluster_path: str | None = None,
    labels: np.ndarray | None = None,
    repair_value_scores: np.ndarray | None = None,
    options: dict | None = None,
) -> np.ndarray:
    """Build a budget-adaptive or hybrid ranking for a single budget."""
    import os
    from pathlib import Path
    from robust_verify.selection import (
        budget_adaptive_coverage_ranking,
        budget_hybrid_coverage_ranking,
        gated_adaptive_coverage_ranking,
    )

    if method == "noise_gated_repair_value":
        if losses is None:
            raise ValueError("noise_gated_repair_value requires losses for deterministic ties")
        if repair_value_scores is None:
            raise ValueError("noise_gated_repair_value requires repair_value_scores")
        settings = dict(options or {})
        return build_noise_gated_repair_value_ranking(
            noise_score,
            repair_value_scores,
            losses,
            budget_count,
            n_total,
            candidate_multiplier=float(settings.get("repair_value_candidate_multiplier", 2.0)),
            noise_anchor_fraction=float(settings.get("repair_value_noise_anchor_fraction", 0.5)),
        )

    # CPBA ablation dispatch
    if method in CPBA_METHODS:
        if losses is None:
            raise ValueError(f"{method} requires losses for CPBA allocation")
        if labels is None:
            raise ValueError(f"{method} requires labels (noisy labels) for CPBA allocation")
        ranking_fn = _CPBA_DISPATCH[method]
        ranking, v9_meta = ranking_fn(
            noise_score=noise_score,
            probabilities=probabilities,
            private_frame=private_frame,
            budget_count=budget_count,
            n_total=n_total,
            losses=losses,
            labels=labels,
            cluster_path=cluster_path,
        )
        object.__setattr__(private_frame, "_v9_meta", v9_meta)
        return ranking

    if cluster_path is None:
        k = os.environ.get("NOISE_SCORE_CLUSTER_K", "20")
        cluster_path = f"outputs/cluster_assignments/clusters_k{k}.csv"

    # For budget_hybrid: if the budget is small, skip the cluster/adaptive
    # pipeline entirely and use v1's exact tie-breaking for strict identity.
    if method in {
        "noise_score_budget_hybrid",
        "noise_score_budget_hybrid_v8",
    } and losses is not None:
        switch_budget_count = max(1, int(round(n_total * 0.02)))
        if budget_count <= switch_budget_count:
            return descending_ranking(noise_score, losses)

    cluster_df = pd.read_csv(Path(cluster_path))
    buckets = _align_clusters(private_frame, cluster_df)

    if method == "noise_score_gated_adaptive":
        ranking, meta = gated_adaptive_coverage_ranking(
            noise_score,
            buckets,
            budget_count=budget_count,
            small_budget_fraction=0.02,
            max_largest_bucket_share=0.25,
            min_effective_bucket_ratio=0.35,
            return_metadata=True,
        )
    elif method == "noise_score_budget_hybrid_v8":
        if losses is None:
            raise ValueError("v8 requires losses for noise-risk gate")
        ranking = _build_v8_ngc_ranking(
            noise_score, buckets, budget_count, n_total, losses=losses,
        )
    elif method == "noise_score_budget_hybrid":
        ranking, meta = budget_hybrid_coverage_ranking(
            noise_score,
            buckets,
            budget_count=budget_count,
            switch_budget_fraction=0.02,
            coverage_ratio=None,
            ratio_schedule="default",
            bucket_order="size_asc",
            return_metadata=True,
        )
    else:
        ranking, meta = budget_adaptive_coverage_ranking(
            noise_score,
            buckets,
            budget_count=budget_count,
            coverage_ratio=None,
            ratio_schedule="default",
            bucket_order="size_asc",
            return_metadata=True,
        )
    return ranking


def build_rankings(
    methods: list[str],
    probabilities: np.ndarray,
    labels: np.ndarray,
    correctness_history: np.ndarray,
    private_frame: pd.DataFrame,
    seed: int,
    context: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = build_legal_scores(probabilities, labels, correctness_history)
    losses = scores["loss"]
    rankings: dict[str, np.ndarray] = {}
    ctx = dict(context or {})
    ctx["seed"] = int(seed)
    if context is not None:
        context.setdefault("selection_metadata", {})
        ctx["selection_metadata"] = context["selection_metadata"]

    if "random" in methods:
        rankings["random"] = random_ranking(len(labels), seed + 10_000)
    for method in ("loss", "entropy", "forgetting", "noise_score"):
        if method in methods:
            rankings[method] = descending_ranking(scores[method], losses)

    if "noise_score_coverage" in methods:
        rankings["noise_score_coverage"] = noise_score_coverage_ranking(
            scores["noise_score"], probabilities
        )
    if "noise_score_cluster_coverage" in methods:
        rankings["noise_score_cluster_coverage"] = noise_score_cluster_coverage_ranking(
            scores["noise_score"], probabilities, private_frame
        )
    if "oracle_noise" in methods:
        rankings["oracle_noise"] = oracle_noise_loss_tiebreak_ranking(private_frame, losses)
    if "oracle_noise_loss_tiebreak" in methods:
        rankings["oracle_noise_loss_tiebreak"] = oracle_noise_loss_tiebreak_ranking(
            private_frame, losses
        )
    if "oracle_noise_random_tiebreak" in methods:
        rankings["oracle_noise_random_tiebreak"] = oracle_noise_random_tiebreak_ranking(
            private_frame, losses, seed=seed
        )
    if "oracle_noise_group_balanced" in methods:
        rankings["oracle_noise_group_balanced"] = oracle_noise_group_balanced_ranking(
            private_frame, losses
        )
    if "oracle_minority" in methods:
        rankings["oracle_minority"] = oracle_minority_ranking(private_frame, losses)
    if "oracle_group_balanced" in methods:
        rankings["oracle_group_balanced"] = oracle_group_balanced_ranking(
            private_frame, losses
        )
    if "oracle_disagreement" in methods:
        rankings["oracle_disagreement"] = oracle_disagreement_ranking(
            private_frame, losses
        )

    # --- External / influence-based baselines ---
    tracin_methods = {
        "tracin_val_uncertainty",
        "tracin_noise_weighted",
        "tracin_wga_oracle",
    }
    modern_methods = {
        "tracin_multicheckpoint_val",
        "expected_repair_value",
        "expected_repair_value_multiclass",
        "expected_repair_value_adaptive_tau",
        "expected_repair_value_capture_tau",
        "expected_repair_value_calibrated_tau",
        "noise_gated_repair_value",
        "reliability_gated_repair_value",
        "auto_d3m_query",
        "activeclean_expected_change",
        "active_label_cleaning",
        "oof_cleanlab",
    }
    needs_context = (
        {"aum", "coreset", "uncertainty_diversity", "badge_lite", "nn_agreement", "nn_label_spreading"}
        | tracin_methods
        | modern_methods
    )
    if needs_context.intersection(methods):
        if context is None:
            raise ValueError(
                "methods {needs_context} require a context dict".format(
                    needs_context=sorted(needs_context.intersection(methods))
                )
            )
        ctx["train_probs"] = probabilities
        ctx["train_labels"] = labels

    max_budget_count = int(ctx.get("max_budget_count", len(labels)))
    baseline_options = ctx.get("baseline_options", {})

    if "aum" in methods:
        rankings["aum"] = aum_ranking(ctx["margin_history"], losses)
    if "confident_learning" in methods:
        rankings["confident_learning"] = confident_learning_ranking(
            probabilities, labels, losses
        )
    if "margin" in methods:
        rankings["margin"] = margin_ranking(probabilities, labels, losses)
    if "coreset" in methods:
        rankings["coreset"] = coreset_ranking(
            ctx["train_features"], seed=seed, max_budget_count=max_budget_count,
            options=baseline_options,
        )
    if "uncertainty_diversity" in methods:
        rankings["uncertainty_diversity"] = uncertainty_diversity_ranking(
            probabilities, ctx["train_features"], losses, seed=seed,
            max_budget_count=max_budget_count, options=baseline_options,
        )
    if "badge_lite" in methods:
        rankings["badge_lite"] = badge_lite_ranking(
            probabilities, labels, ctx["train_features"], losses, seed=seed,
            max_budget_count=max_budget_count, options=baseline_options,
        )
    for method in ("tracin_val_uncertainty", "tracin_noise_weighted", "tracin_wga_oracle"):
        if method in methods:
            mode = method.split("tracin_", 1)[1]
            influence_scores = _tracin_influence(mode, ctx, scores["noise_score"])
            rankings[method] = descending_ranking(influence_scores, losses)

    for method in (
        "tracin_multicheckpoint_val",
        "expected_repair_value",
        "expected_repair_value_multiclass",
        "expected_repair_value_adaptive_tau",
        "expected_repair_value_capture_tau",
        "expected_repair_value_calibrated_tau",
        "noise_gated_repair_value",
        "reliability_gated_repair_value",
        "auto_d3m_query",
        "activeclean_expected_change",
        "active_label_cleaning",
        "oof_cleanlab",
    ):
        if method in methods:
            method_scores = _modern_baseline_score(
                method, ctx, scores["noise_score"], baseline_options
            )
            scores[method] = method_scores
            if method not in ADAPTIVE_METHODS:
                rankings[method] = descending_ranking(method_scores, losses)

    if "nn_agreement" in methods:
        rankings["nn_agreement"] = nn_agreement_ranking(
            ctx["train_features"], labels, losses, k=20,
        )
    if "nn_label_spreading" in methods:
        rankings["nn_label_spreading"] = nn_label_spreading_ranking(
            ctx["train_features"], labels, losses, k=20,
        )
    if "stratified_random" in methods:
        rankings["stratified_random"] = _stratified_random_ranking(
            labels, losses, seed,
        )

    # Adaptive / multi-round methods are handled per-budget by the caller; skip here.
    skip_methods = ADAPTIVE_METHODS | {"noise_score_multi_round"}
    unknown = sorted(
        set(methods).difference(rankings).difference(skip_methods)
    )
    if unknown:
        raise ValueError(f"Unknown or unbuilt methods: {unknown}")

    return rankings, scores
