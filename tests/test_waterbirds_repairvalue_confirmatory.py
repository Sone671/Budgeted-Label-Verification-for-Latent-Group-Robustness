from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from robust_verify.config import load_config
from robust_verify.waterbirds_repairvalue_confirmatory import (
    FROZEN_METHODS,
    FROZEN_SEEDS,
    build_group_mechanism_audit,
    protocol_sha256,
    summarize_confirmatory_results,
    validate_frozen_config,
    validate_result_frame,
)

CONFIG_PATH = "configs/waterbirds_e2e_repairvalue_seed20_39.yaml"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _result_frame(repair_delta: float = 0.02) -> pd.DataFrame:
    deltas = {
        "random": 0.0,
        "loss": 0.005,
        "noise_score": 0.004,
        "tracin_val_uncertainty": -0.001,
        "expected_repair_value": repair_delta,
    }
    rows = []
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
    changed_paths["data"]["root"] = "/server/datasets/waterbirds"
    changed_paths["project"]["output_dir"] = "/server/results/repairvalue"
    assert protocol_sha256(changed_paths) == protocol_sha256(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", list(range(10, 30))),
        ("methods", ["loss", "expected_repair_value"]),
        ("budgets", [0.02]),
        ("baseline_options", {"repair_value_tail_fraction": 0.35}),
    ],
)
def test_frozen_config_rejects_scientific_changes(field: str, value: object) -> None:
    config = _config()
    config["experiment"][field] = value
    with pytest.raises(ValueError, match="Frozen Waterbirds RepairValue protocol mismatch"):
        validate_frozen_config(config)


def test_confirmatory_summary_applies_registered_paired_gate() -> None:
    summary = summarize_confirmatory_results(_result_frame())

    assert summary["primary_gate_passed"] is True
    assert summary["paired_vs_loss"]["mean_delta_wga_difference"] == pytest.approx(0.015)
    assert summary["paired_vs_loss"]["bootstrap_95_ci"] == pytest.approx([0.015, 0.015])
    assert summary["paired_vs_loss"]["positive_seed_count"] == 20
    assert summary["absolute_secondary"]["mean_delta_wga"] == pytest.approx(0.02)


def test_confirmatory_summary_retains_a_failed_null_result() -> None:
    frame = _result_frame()
    for seed in (20, 21, 22, 23, 24, 25):
        frame.loc[
            frame["seed"].eq(seed) & frame["method"].eq("expected_repair_value"),
            ["average_accuracy", "balanced_accuracy", "wga"],
        ] = [0.80, 0.75, 0.50]
        frame.loc[
            frame["seed"].eq(seed) & frame["method"].eq("expected_repair_value"),
            ["delta_average_accuracy", "delta_wga"],
        ] = [0.0, 0.0]

    summary = summarize_confirmatory_results(frame)

    assert summary["paired_vs_loss"]["positive_seed_count"] == 14
    assert summary["primary_gate"]["positive_paired_seed_count_at_least_15"] is False
    assert summary["primary_gate_passed"] is False


def test_confirmatory_summary_requires_positive_absolute_utility() -> None:
    frame = _result_frame()
    for method, delta in (("loss", -0.02), ("expected_repair_value", -0.005)):
        selected = frame["method"].eq(method)
        frame.loc[selected, "average_accuracy"] = 0.80 + delta
        frame.loc[selected, "balanced_accuracy"] = 0.75 + delta
        frame.loc[selected, "wga"] = 0.50 + delta
        frame.loc[selected, "delta_average_accuracy"] = delta
        frame.loc[selected, "delta_wga"] = delta

    summary = summarize_confirmatory_results(frame)

    assert summary["paired_vs_loss"]["mean_delta_wga_difference"] == pytest.approx(0.015)
    assert summary["paired_vs_loss"]["positive_seed_count"] == 20
    assert summary["primary_gate"]["paired_bootstrap_ci_lower_above_zero"] is True
    assert summary["primary_gate"]["positive_paired_seed_count_at_least_15"] is True
    assert summary["primary_gate"]["absolute_bootstrap_ci_lower_above_zero"] is False
    assert summary["primary_gate_passed"] is False


def test_result_validation_rejects_an_inconsistent_absolute_baseline() -> None:
    frame = _result_frame()
    frame.loc[0, "delta_wga"] = 0.2
    with pytest.raises(ValueError, match="delta_wga is inconsistent"):
        validate_result_frame(frame, expected_seeds=FROZEN_SEEDS)


def test_group_mechanism_audit_uses_each_seed_baseline_worst_group(tmp_path) -> None:
    frame = _result_frame()
    for group in range(4):
        frame[f"baseline_group_{group}_accuracy"] = 0.70 + 0.02 * group
        frame[f"group_{group}_accuracy"] = 0.71 + 0.02 * group
    frame["baseline_group_3_accuracy"] = 0.40
    frame["group_3_accuracy"] = 0.45

    source = tmp_path / "source"
    query_root = tmp_path / "queries"
    (source / "manifests").mkdir(parents=True)
    query_root.mkdir()
    for seed in FROZEN_SEEDS:
        private = pd.DataFrame(
            {
                "sample_id": [0, 1, 2, 3],
                "group": [0, 3, 3, 2],
                "is_noisy": [0, 1, 0, 1],
            }
        )
        private.to_csv(
            source / "manifests" / f"uniform20_seed{seed}_train_private.csv",
            index=False,
        )
        for method in FROZEN_METHODS:
            query_path = query_root / f"seed{seed}_{method}.npz"
            np.savez_compressed(query_path, sample_ids=np.array([0, 1]))
            frame.loc[frame["seed"].eq(seed) & frame["method"].eq(method), "query_artifact"] = (
                query_path.relative_to(tmp_path).as_posix()
            )

    audit = build_group_mechanism_audit(frame, output_root=tmp_path, source_root=source)

    assert len(audit) == 100
    assert set(audit["baseline_worst_group"]) == {3}
    assert set(audit["weak_group_queries"]) == {1}
    assert set(audit["weak_group_corrections"]) == {1}
    assert set(audit["weak_group_noise_recall"]) == {1.0}
    assert audit["weak_group_accuracy_change"].to_numpy() == pytest.approx(np.full(100, 0.05))
