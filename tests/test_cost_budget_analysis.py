import numpy as np
import pandas as pd
import pytest

from robust_verify.cost_budget_analysis import (
    holm_adjust,
    pair_phase10_results,
    phase10_gate,
    phase11_confirmation_decision,
    summarize_phase10_pairs,
)


def _result_rows(dataset: str, delta: float, seeds: int = 10) -> list[dict]:
    rows = []
    for seed in range(seeds):
        common = {
            "dataset": dataset,
            "noise_name": "uniform20",
            "seed": seed,
            "budget_fraction": 0.02,
            "group_to_label_cost_ratio": 0.5,
            "target_group_action_share": 0.25,
            "audit_pool_saturated": False,
            "planned_group_audit_count": 10,
            "planned_label_verification_count": 30,
            "planned_realized_group_action_share": 0.25,
        }
        rows.extend(
            [
                {**common, "role": "joint_sparse_attribute", "wga": 0.5 + delta},
                {**common, "role": "label_only_total_budget", "wga": 0.5},
                {**common, "role": "label_only_same_label_count", "wga": 0.49},
            ]
        )
    return rows


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust(np.array([0.03, 0.01, np.nan, 0.04]))
    assert np.isnan(adjusted[2])
    assert adjusted[1] <= adjusted[0] <= adjusted[3]
    assert np.all(adjusted[np.isfinite(adjusted)] <= 1.0)


def test_complete_positive_cells_pass_cross_dataset_gate() -> None:
    frame = pd.DataFrame(
        _result_rows("waterbirds", 0.02) + _result_rows("celeba", 0.01)
    )
    paired = pair_phase10_results(frame)
    summary = summarize_phase10_pairs(paired)
    assert set(summary["cell_label"]) == {"beneficial"}
    decision = phase10_gate(summary)
    assert decision["decision"] == "advance_to_new_dataset_preregistration"
    assert decision["common_beneficial_cell_count"] == 1
    assert np.allclose(paired["label_opportunity_cost"], 0.01)
    assert np.allclose(paired["decomposition_residual"], 0.0)
    assert summary.iloc[0]["decomposition_residual_max_abs"] == pytest.approx(0.0)


def test_incomplete_cells_are_not_interpreted() -> None:
    frame = pd.DataFrame(_result_rows("waterbirds", 0.02, seeds=3))
    summary = summarize_phase10_pairs(pair_phase10_results(frame))
    assert summary.iloc[0]["cell_label"] == "incomplete"
    assert phase10_gate(summary)["decision"] == "incomplete"


def test_missing_required_dataset_is_incomplete() -> None:
    frame = pd.DataFrame(_result_rows("waterbirds", 0.02))
    summary = summarize_phase10_pairs(pair_phase10_results(frame))
    decision = phase10_gate(summary)
    assert decision["decision"] == "incomplete"
    assert decision["missing_required_datasets"] == ["celeba"]


def test_duplicate_role_rows_are_rejected() -> None:
    rows = _result_rows("waterbirds", 0.02, seeds=1)
    rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        pair_phase10_results(pd.DataFrame(rows))


def test_phase11_requires_both_preregistered_noises() -> None:
    summary = pd.DataFrame(
        [
            {"noise_name": "uniform20", "cell_label": "beneficial"},
            {"noise_name": "minority_high_40", "cell_label": "harmful"},
        ]
    )
    assert (
        phase11_confirmation_decision(summary)["decision"]
        == "partial_external_confirmation"
    )
    summary.loc[1, "cell_label"] = "beneficial"
    assert (
        phase11_confirmation_decision(summary)["decision"]
        == "strong_external_confirmation"
    )
