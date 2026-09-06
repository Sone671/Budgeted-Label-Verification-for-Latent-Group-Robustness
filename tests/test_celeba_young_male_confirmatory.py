from __future__ import annotations

import copy

import pandas as pd
import pytest

from robust_verify.celeba_young_male_confirmatory import (
    FROZEN_METHODS,
    FROZEN_SEEDS,
    protocol_sha256,
    summarize_confirmatory_results,
    validate_frozen_config,
    validate_result_frame,
)
from robust_verify.config import load_config

CONFIG_PATH = "configs/celeba_e2e_young_male_rvq_seed60_79.yaml"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _result_frame(rv_delta: float = 0.04) -> pd.DataFrame:
    deltas = {
        "random": 0.0,
        "loss": 0.01,
        "noise_score": 0.02,
        "oof_cleanlab": 0.03,
        "expected_repair_value": rv_delta,
    }
    rows: list[dict[str, object]] = []
    for seed in FROZEN_SEEDS:
        for method in FROZEN_METHODS:
            delta = deltas[method]
            rows.append(
                {
                    "noise_name": "uniform20",
                    "noise_kind": "uniform",
                    "seed": seed,
                    "method": method,
                    "budget_fraction": 0.10,
                    "budget_count": 100,
                    "num_queried": 100,
                    "num_corrected": 20,
                    "noise_precision": 0.20,
                    "minority_query_rate": 0.10,
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
                    "oof_probability_artifact": "oof.npz",
                }
            )
    return pd.DataFrame(rows)


def test_frozen_config_validates_and_paths_do_not_change_hash() -> None:
    config = _config()
    validate_frozen_config(config)
    changed = copy.deepcopy(config)
    changed["data"]["root"] = "/server/celeba"
    changed["project"]["output_dir"] = "/server/results/young_male"
    assert protocol_sha256(changed) == protocol_sha256(config)


def test_frozen_config_rejects_method_or_seed_changes() -> None:
    config = _config()
    config["experiment"]["methods"] = ["random", "loss"]
    with pytest.raises(ValueError, match="Frozen CelebA Young/Male protocol mismatch"):
        validate_frozen_config(config)

    config = _config()
    config["experiment"]["seeds"] = list(range(40, 60))
    with pytest.raises(ValueError, match="Frozen CelebA Young/Male protocol mismatch"):
        validate_frozen_config(config)


def test_summary_requires_paired_and_absolute_positive_effects() -> None:
    frame = _result_frame()
    summary = summarize_confirmatory_results(frame)

    assert summary["primary_gate_passed"] is True
    assert summary["paired_vs_loss"]["mean_delta_wga_difference"] == pytest.approx(0.03)
    assert summary["paired_vs_loss"]["positive_seed_count"] == 20
    assert summary["absolute_utility"]["mean_delta_wga"] == pytest.approx(0.04)

    failed = _result_frame(rv_delta=0.0)
    assert summarize_confirmatory_results(failed)["primary_gate_passed"] is False


def test_result_validation_rejects_inconsistent_delta() -> None:
    frame = _result_frame()
    frame.loc[0, "delta_wga"] = 0.5
    with pytest.raises(ValueError, match="delta_wga is inconsistent"):
        validate_result_frame(frame, expected_seeds=FROZEN_SEEDS)
