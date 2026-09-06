from __future__ import annotations

import numpy as np
import pandas as pd

from robust_verify.analysis import query_composition
from robust_verify.scoring import (
    build_adaptive_ranking,
    forgetting_counts,
    oracle_noise_group_balanced_ranking,
    oracle_noise_loss_tiebreak_ranking,
    oracle_noise_random_tiebreak_ranking,
    ranking_to_scores,
    score_by_method,
)
from robust_verify.utils import budget_to_count, normalized_rank, rank_within_class


def toy_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "group": ["majority"] * 8 + ["minority"] * 4,
            "is_noisy": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    losses = np.array([2.6, 2.1, 1.8, 1.4, 0.7, 0.6, 0.4, 0.2, 0.9, 0.8, 0.5, 0.3])
    return frame, losses


def test_budget_to_count() -> None:
    assert budget_to_count(0.0, 100) == 0
    assert budget_to_count(0.01, 100) == 1
    assert budget_to_count(0.005, 4795) == 24


def test_normalized_rank() -> None:
    values = np.array([30.0, 10.0, 20.0])
    ranks = normalized_rank(values)
    assert np.allclose(ranks, [1.0, 0.0, 0.5])


def test_rank_within_class() -> None:
    values = np.array([1.0, 3.0, 10.0, 20.0])
    labels = np.array([0, 0, 1, 1])
    ranks = rank_within_class(values, labels)
    assert np.allclose(ranks, [0.0, 1.0, 0.0, 1.0])


def test_forgetting_counts() -> None:
    history = np.array(
        [
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ]
    )
    assert np.array_equal(forgetting_counts(history), [1, 0, 1])


def test_query_composition() -> None:
    private = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "is_noisy": [1, 1, 0, 0],
            "is_minority": [0, 1, 0, 1],
            "group": [0, 1, 2, 3],
        }
    )
    result = query_composition(np.array([0, 1, 3]), private)
    assert result["num_corrected"] == 2
    assert result["mislabeled_majority"] == 1
    assert result["mislabeled_minority"] == 1
    assert result["clean_minority"] == 1


def test_normalized_rank_ties() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0])
    ranks = normalized_rank(values)
    assert np.allclose(ranks, [1 / 6, 1 / 6, 5 / 6, 5 / 6])


# ---- New oracle-noise tests ----


def test_loss_tiebreak_preserves_old_oracle_behavior() -> None:
    frame, losses = toy_frame()
    ranking = oracle_noise_loss_tiebreak_ranking(frame, losses)
    # Noisy indices 0,1,2,3,8,9 first, sorted by loss descending
    assert ranking[:6].tolist() == [0, 1, 2, 3, 8, 9]


def test_group_balanced_hits_minority_noisy_early() -> None:
    frame, losses = toy_frame()
    ranking = oracle_noise_group_balanced_ranking(frame, losses)
    selected_groups = frame.iloc[ranking[:4]]["group"].tolist()
    # Round-robin alternates minority (group "minority") and majority
    assert selected_groups == ["minority", "majority", "minority", "majority"]
    assert ranking[:4].tolist() == [8, 0, 9, 1]


def test_random_tiebreak_is_permutation_and_noisy_first() -> None:
    frame, losses = toy_frame()
    ranking = oracle_noise_random_tiebreak_ranking(frame, losses, seed=7)
    assert sorted(ranking.tolist()) == list(range(len(frame)))
    # First 6 should all be noisy
    assert frame.iloc[ranking[:6]]["is_noisy"].sum() == 6


def test_ranking_to_scores_selects_prefix() -> None:
    ranking = np.array([3, 1, 0, 2])
    scores = ranking_to_scores(ranking)
    assert np.argsort(-scores).tolist() == [3, 1, 0, 2]


def test_score_by_method_loss_tiebreak() -> None:
    frame, losses = toy_frame()
    scores = score_by_method("oracle_noise_loss_tiebreak", frame, losses)
    # Top-4 should all be noisy
    top4 = np.argsort(-scores)[:4]
    assert frame.iloc[top4]["is_noisy"].sum() == 4


def test_oracle_noise_backward_compat() -> None:
    """oracle_noise should be an alias for oracle_noise_loss_tiebreak."""
    frame, losses = toy_frame()
    legacy = score_by_method("oracle_noise", frame, losses)
    explicit = score_by_method("oracle_noise_loss_tiebreak", frame, losses)
    assert np.array_equal(legacy, explicit)


def test_cpba_only_accepts_a_legal_sample_id_frame_and_returns_a_permutation() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.65, 0.15],
            [0.10, 0.25, 0.65],
            [0.60, 0.30, 0.10],
            [0.35, 0.45, 0.20],
            [0.10, 0.50, 0.40],
            [0.45, 0.35, 0.20],
            [0.25, 0.50, 0.25],
            [0.20, 0.35, 0.45],
        ]
    )
    ranking = build_adaptive_ranking(
        "noise_score_cpba_only",
        noise_score=np.linspace(0.1, 0.9, len(labels)),
        probabilities=probabilities,
        private_frame=pd.DataFrame({"sample_id": np.arange(len(labels))}),
        budget_count=3,
        n_total=len(labels),
        losses=np.linspace(1.0, 2.0, len(labels)),
        labels=labels,
    )

    assert sorted(ranking.tolist()) == list(range(len(labels)))
