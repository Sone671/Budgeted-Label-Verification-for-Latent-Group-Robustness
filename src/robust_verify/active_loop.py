from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from robust_verify.data.access import VerificationOracle
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import train_linear_head


@dataclass
class MultiRoundResult:
    query_positions: np.ndarray
    round_queries: list[np.ndarray]
    round_scores: list[np.ndarray]
    final_labels: np.ndarray


def run_multi_round(
    *,
    train_features: np.ndarray | torch.Tensor,
    train_ids: np.ndarray,
    noisy_labels: np.ndarray,
    val_features: np.ndarray | torch.Tensor,
    val_labels: np.ndarray,
    public_frame: pd.DataFrame,
    private_manifest_path,
    initial_state: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    budget_count: int,
    num_rounds: int = 3,
    input_dim: int,
    num_classes: int,
) -> MultiRoundResult:
    """Run multi-round active querying with noise_score re-scoring.

    Each round:
      1. Retrain probe from initial_state on current labels
      2. Score with noise_score (re-computed on the fresh probe)
      3. Query the top budget/num_rounds samples not yet queried
      4. Verify and correct those samples

    Returns the union of all queried positions and the final corrected labels.
    """
    oracle = VerificationOracle(private_manifest_path)
    current_labels = noisy_labels.copy()
    queried_positions: list[int] = []
    queried_set: set[int] = set()
    round_queries: list[np.ndarray] = []
    round_scores: list[np.ndarray] = []

    per_round = max(1, budget_count // num_rounds)
    remaining = budget_count

    for round_idx in range(num_rounds):
        is_last = round_idx == num_rounds - 1
        this_round_count = remaining if is_last else per_round
        if this_round_count <= 0:
            break

        probe = train_linear_head(
            train_features=train_features,
            train_labels=current_labels,
            val_features=val_features,
            val_labels=val_labels,
            config=training_config,
            seed=seed,
            initial_state=copy.deepcopy(initial_state),
            record_dynamics=True,
        )

        scores = build_legal_scores(
            probabilities=probe.train_probabilities,
            labels=current_labels,
            correctness_history=probe.correctness_history,
        )
        noise_score = scores["noise_score"]
        losses = scores["loss"]

        masked_score = noise_score.copy()
        for pos in queried_set:
            masked_score[pos] = -np.inf

        ranking = descending_ranking(masked_score, losses)
        round_positions = ranking[:this_round_count]

        round_queries.append(round_positions)
        round_scores.append(noise_score)

        round_sample_ids = train_ids[round_positions]
        verified = oracle.verify(round_sample_ids)
        current_labels[round_positions] = verified

        queried_positions.extend(round_positions.tolist())
        queried_set.update(round_positions.tolist())
        remaining -= this_round_count

    query_positions = np.array(sorted(queried_positions), dtype=np.int64)
    return MultiRoundResult(
        query_positions=query_positions,
        round_queries=round_queries,
        round_scores=round_scores,
        final_labels=current_labels,
    )
