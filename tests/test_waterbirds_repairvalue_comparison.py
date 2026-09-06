from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_waterbirds_repairvalue_comparison import (
    REFERENCE_METHODS,
    ranking_for_method,
    summarize_comparison,
)


class _ToyData:
    losses = np.asarray([0.1, 0.8, 0.3, 0.7])
    noise_score = np.asarray([0.2, 0.4, 0.9, 0.1])
    train_probabilities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]],
        dtype=float,
    )


def test_legal_rankings_are_descending_and_deterministic() -> None:
    data = _ToyData()
    for method in REFERENCE_METHODS:
        ranking = ranking_for_method(data, method)
        assert ranking.tolist() == sorted(
            range(len(ranking)),
            key=lambda index: (
                -(data.losses[index] if method == "loss" else
                  data.noise_score[index] if method == "noise_score" else
                  -(data.train_probabilities[index] * np.log(data.train_probabilities[index])).sum()),
                -data.losses[index],
            ),
        )
        assert sorted(ranking.tolist()) == list(range(len(ranking)))


def test_comparison_summary_pairs_each_reference_by_seed_and_condition() -> None:
    rows = []
    for seed in range(3):
        common = {
            "dataset": "waterbirds",
            "noise_name": "uniform20",
            "seed": seed,
            "budget_fraction": 0.05,
            "tail_fraction": np.nan,
            "validation_fraction": np.nan,
            "validation_noise_rate": np.nan,
            "validation_noise_mode": "reference",
            "delta_wga": 0.01 + seed * 0.001,
            "query_noise_precision": 0.2,
            "posthoc_minority_query_rate": 0.1,
        }
        for method, delta in (("loss", 0.0), ("noise_score", -0.002), ("entropy", 0.001)):
            row = dict(common)
            row.update(method=method, delta_wga=delta)
            rows.append(row)
        row = dict(common)
        row.update(
            method="expected_repair_value",
            condition_index=7,
            tail_fraction=0.5,
            validation_fraction=1.0,
            validation_noise_rate=0.0,
            validation_noise_mode="symmetric",
        )
        rows.append(row)

    summary = summarize_comparison(pd.DataFrame(rows))
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_seeds"] == 3
    assert row["positive_vs_loss_seeds"] == 3
    assert row["positive_vs_noise_score_seeds"] == 3
    assert row["positive_vs_entropy_seeds"] == 3
    assert row["mean_vs_loss_pp"] == pytest.approx(1.1)
    assert row["mean_vs_noise_score_pp"] == pytest.approx(1.3)
    assert row["mean_vs_entropy_pp"] == pytest.approx(1.0)


def test_summary_rejects_missing_reference_seed() -> None:
    rows = [
        {
            "noise_name": "uniform20",
            "seed": 0,
            "budget_fraction": 0.02,
            "method": "loss",
            "delta_wga": 0.0,
            "query_noise_precision": 0.1,
            "posthoc_minority_query_rate": 0.1,
        },
        {
            "noise_name": "uniform20",
            "seed": 0,
            "budget_fraction": 0.02,
            "method": "noise_score",
            "delta_wga": 0.0,
            "query_noise_precision": 0.1,
            "posthoc_minority_query_rate": 0.1,
        },
        {
            "noise_name": "uniform20",
            "seed": 0,
            "budget_fraction": 0.02,
            "method": "entropy",
            "delta_wga": 0.0,
            "query_noise_precision": 0.1,
            "posthoc_minority_query_rate": 0.1,
        },
        {
            "noise_name": "uniform20",
            "seed": 1,
            "budget_fraction": 0.02,
            "method": "expected_repair_value",
            "tail_fraction": 0.5,
            "validation_fraction": 1.0,
            "validation_noise_rate": 0.0,
            "validation_noise_mode": "symmetric",
            "delta_wga": 0.01,
            "query_noise_precision": 0.1,
            "posthoc_minority_query_rate": 0.1,
        },
    ]
    with pytest.raises(ValueError, match="Missing loss reference seeds"):
        summarize_comparison(pd.DataFrame(rows))
