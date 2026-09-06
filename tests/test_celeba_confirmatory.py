from __future__ import annotations

import copy

import pandas as pd
import pytest

from robust_verify.celeba_confirmatory import (
    FROZEN_METHODS,
    FROZEN_SEEDS,
    protocol_sha256,
    summarize_confirmatory_results,
    validate_frozen_config,
    validate_result_frame,
)
from robust_verify.config import load_config


CONFIG_PATH = "configs/celeba_e2e_confirmatory_seed10_19.yaml"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _result_frame(candidate_delta: float = -0.01) -> pd.DataFrame:
    rows = []
    for seed in FROZEN_SEEDS:
        for method in FROZEN_METHODS:
            delta = -0.004 if method == "loss" else candidate_delta
            rows.append(
                {
                    "noise_name": "uniform20",
                    "noise_kind": "uniform",
                    "seed": seed,
                    "method": method,
                    "budget_fraction": 0.10,
                    "budget_count": 100,
                    "num_queried": 100,
                    "num_corrected": 99,
                    "noise_precision": 0.995,
                    "minority_query_rate": 0.01,
                    "average_accuracy": 0.80 + delta,
                    "balanced_accuracy": 0.75 + delta,
                    "wga": 0.50 + delta,
                    "baseline_average_accuracy": 0.80,
                    "baseline_balanced_accuracy": 0.75,
                    "baseline_wga": 0.50,
                    "delta_average_accuracy": delta,
                    "delta_wga": delta,
                    "query_artifact": "query.npz",
                    "checkpoint": "checkpoint.pt",
                    "prediction_artifact": "predictions.npz",
                    "baseline_checkpoint": "baseline.pt",
                    "baseline_prediction_artifact": "baseline_predictions.npz",
                }
            )
    return pd.DataFrame(rows)


def test_frozen_config_validates_and_path_overrides_do_not_change_hash() -> None:
    config = _config()
    validate_frozen_config(config)
    changed_paths = copy.deepcopy(config)
    changed_paths["data"]["root"] = "/server/datasets/celeba"
    changed_paths["project"]["output_dir"] = "/server/results/run"
    assert protocol_sha256(changed_paths) == protocol_sha256(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", list(range(0, 10))),
        ("methods", ["loss", "noise_score"]),
        ("budgets", [0.02]),
    ],
)
def test_frozen_config_rejects_scientific_changes(field: str, value: object) -> None:
    config = _config()
    config["experiment"][field] = value
    with pytest.raises(ValueError, match="Frozen CelebA protocol mismatch"):
        validate_frozen_config(config)


def test_confirmatory_summary_applies_all_three_registered_gates() -> None:
    frame = _result_frame()
    summary = summarize_confirmatory_results(frame)
    assert summary["primary_gate_passed"] is True
    assert summary["negative_seed_count"] == 10
    assert summary["mean_noise_precision"] == pytest.approx(0.995)
    assert summary["bootstrap_95_ci"] == pytest.approx([-0.01, -0.01])


def test_result_validation_rejects_an_inconsistent_absolute_baseline() -> None:
    frame = _result_frame()
    frame.loc[0, "delta_wga"] = 0.2
    with pytest.raises(ValueError, match="delta_wga is inconsistent"):
        validate_result_frame(frame, expected_seeds=FROZEN_SEEDS)

