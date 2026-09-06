from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_verify.matching import MatchConstructionError, construct_matched_pairs, matching_strata


def _synthetic_condition() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    probabilities = []
    labels = []
    # Every clean-label/loss/margin stratum has target and non-target noisy
    # candidates, plus a clean candidate.  This supports two non-reused pairs.
    for clean_label in (0, 1):
        for repeat in range(8):
            for group in (0, 1):
                rows.append({"clean_label": clean_label, "is_noisy": True, "group": group})
                labels.append(clean_label)
                probabilities.append([0.8 - 0.01 * repeat, 0.2 + 0.01 * repeat])
            rows.append({"clean_label": clean_label, "is_noisy": False, "group": 0})
            labels.append(clean_label)
            probabilities.append([0.75 - 0.01 * repeat, 0.25 + 0.01 * repeat])
    return pd.DataFrame(rows), np.asarray(probabilities), np.asarray(labels)


def test_constructed_pairs_have_exact_predeclared_balance() -> None:
    private, probabilities, labels = _synthetic_condition()
    pairs = construct_matched_pairs(
        private,
        probabilities,
        labels,
        target_group=1,
        budget_count=10,
        oracle_noise_precision=0.6,
        high_target_noise_fraction=0.5,
        low_target_noise_fraction=1 / 6,
        n_pairs=2,
        seed=7,
    )
    strata = matching_strata(private, probabilities, labels)
    seen = set()
    for pair in pairs:
        assert pair.diagnostics["oracle_noisy_count"] == 6
        assert pair.diagnostics["high_target_noisy_count"] == 3
        assert pair.diagnostics["low_target_noisy_count"] == 1
        assert pair.diagnostics["covariate_l1_difference"] == 0
        assert len(pair.high_positions) == len(pair.low_positions) == 10
        assert sorted(strata[index] for index in pair.high_positions) == sorted(
            strata[index] for index in pair.low_positions
        )
        # The constructor does not recycle candidates between distinct pairs.
        pair_members = set(pair.high_positions).union(pair.low_positions)
        assert not seen.intersection(pair_members)
        seen.update(pair_members)


def test_pair_construction_rejects_insufficient_stratum_swaps() -> None:
    private, probabilities, labels = _synthetic_condition()
    private.loc[private["group"].eq(1), "group"] = 0
    with pytest.raises(MatchConstructionError, match="swaps"):
        construct_matched_pairs(
            private,
            probabilities,
            labels,
            target_group=1,
            budget_count=10,
            oracle_noise_precision=0.6,
            high_target_noise_fraction=0.5,
            low_target_noise_fraction=1 / 6,
            n_pairs=1,
            seed=7,
        )
