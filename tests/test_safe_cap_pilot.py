from __future__ import annotations

import numpy as np
import pytest

from scripts.safe_cap_pilot import query_ids_to_positions, select_candidates


def _row(budget: float, balanced_accuracy: float, risk: float) -> dict:
    return {
        "budget_fraction": budget,
        "val_balanced_accuracy": balanced_accuracy,
        "spectral_class_cvar": risk,
    }


def test_query_ids_are_mapped_as_sample_ids() -> None:
    train_ids = np.asarray([40, 10, 30, 20])
    assert query_ids_to_positions(np.asarray([20, 40]), train_ids).tolist() == [3, 0]


def test_query_id_validation_rejects_duplicates_and_unknowns() -> None:
    train_ids = np.asarray([10, 20, 30])
    with pytest.raises(ValueError, match="duplicate"):
        query_ids_to_positions(np.asarray([10, 10]), train_ids)
    with pytest.raises(KeyError, match="unknown"):
        query_ids_to_positions(np.asarray([40]), train_ids)


def test_predeclared_selection_includes_baseline_and_favors_smaller_ties() -> None:
    rows = [
        _row(0.0, 0.80, 1.2),
        _row(0.01, 0.82, 1.1),
        _row(0.02, 0.82, 1.0),
        _row(0.05, 0.79, 1.0),
    ]
    chosen = select_candidates(rows)
    assert chosen["max_val_balanced_accuracy"]["budget_fraction"] == 0.01
    assert chosen["min_spectral_risk"]["budget_fraction"] == 0.02
    assert chosen["baseline_no_query"]["budget_fraction"] == 0.0


def test_baseline_can_win_a_safe_stop_rule() -> None:
    rows = [_row(0.0, 0.90, 0.8), _row(0.02, 0.85, 1.0)]
    chosen = select_candidates(rows)
    assert chosen["max_val_balanced_accuracy"]["budget_fraction"] == 0.0
    assert chosen["min_spectral_risk"]["budget_fraction"] == 0.0
