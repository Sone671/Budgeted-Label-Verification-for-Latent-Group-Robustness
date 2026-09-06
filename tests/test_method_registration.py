from __future__ import annotations

import numpy as np
import pandas as pd

from robust_verify.scoring import (
    ADAPTIVE_METHODS,
    METHOD_NAMES,
    NOISE_RISK_LOSS_RATIO_THRESHOLD,
    build_adaptive_ranking,
    descending_ranking,
)


def test_noise_score_versions_are_registered_and_v8_dispatches() -> None:
    assert {
        "noise_score",
        "noise_score_budget_hybrid",
        "noise_score_budget_hybrid_v8",
    } <= METHOD_NAMES
    assert {"noise_score_budget_hybrid", "noise_score_budget_hybrid_v8"} <= ADAPTIVE_METHODS
    assert NOISE_RISK_LOSS_RATIO_THRESHOLD == 4.0

    noise_score = np.asarray([0.2, 0.9, 0.4, 0.7])
    losses = np.asarray([0.1, 0.6, 0.3, 0.2])
    labels = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray(
        [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]],
        dtype=float,
    )
    private_frame = pd.DataFrame({"sample_id": np.arange(len(labels))})

    ranking = build_adaptive_ranking(
        "noise_score_budget_hybrid_v8",
        noise_score,
        probabilities,
        private_frame,
        budget_count=1,
        n_total=len(labels),
        losses=losses,
        labels=labels,
    )

    np.testing.assert_array_equal(ranking, descending_ranking(noise_score, losses))
