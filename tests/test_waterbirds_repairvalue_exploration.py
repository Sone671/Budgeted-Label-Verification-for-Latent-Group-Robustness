from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_waterbirds_repairvalue_exploration import (
    posthoc_query_diagnostics,
)
from scripts.run_repairvalue_validation_retraining import _load_baseline_row


def _private_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": np.arange(8),
            "is_noisy": [1, 1, 0, 0, 1, 0, 1, 0],
            "is_minority": [1, 0, 1, 0, 1, 0, 0, 1],
            "group": [0, 0, 3, 1, 2, 0, 1, 3],
            "clean_label": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_posthoc_diagnostics_only_use_frozen_query_positions() -> None:
    private = _private_frame()
    ranking = np.asarray([4, 0, 7, 2, 1, 3, 5, 6])

    diagnostics = posthoc_query_diagnostics(private, ranking, 0.25)

    # budget_to_count(0.25, 8) = 2: positions 4 and 0.
    assert diagnostics["posthoc_minority_query_rate"] == 1.0
    assert diagnostics["posthoc_noisy_minority_recall"] == 1.0
    assert diagnostics["posthoc_query_group_count"] == 2
    assert diagnostics["posthoc_query_group_entropy"] > 0.0


def test_posthoc_diagnostics_handle_noisy_minority_denominator_zero() -> None:
    private = _private_frame()
    private["is_noisy"] = 0
    ranking = np.arange(len(private))

    diagnostics = posthoc_query_diagnostics(private, ranking, 0.25)

    assert np.isnan(diagnostics["posthoc_noisy_minority_recall"])
    assert diagnostics["posthoc_minority_query_rate"] >= 0.0


def test_baseline_lookup_filters_noise_name(tmp_path) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"seed": 0, "method": "loss", "budget_fraction": 0.02,
             "noise_name": "uniform20", "baseline_wga": 0.5,
             "baseline_average_accuracy": 0.7, "delta_wga": 0.01,
             "delta_average_accuracy": 0.02},
            {"seed": 0, "method": "loss", "budget_fraction": 0.02,
             "noise_name": "minority_high_40", "baseline_wga": 0.4,
             "baseline_average_accuracy": 0.6, "delta_wga": -0.01,
             "delta_average_accuracy": -0.02},
        ]
    ).to_csv(path, index=False)

    row = _load_baseline_row(
        path, seed=0, budget_fraction=0.02, noise_name="minority_high_40"
    )

    assert row["baseline_wga"] == 0.4
    assert row["loss_delta_wga"] == -0.01
