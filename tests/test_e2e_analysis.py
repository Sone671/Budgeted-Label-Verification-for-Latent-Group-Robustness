from __future__ import annotations

import pandas as pd
import pytest

from robust_verify.e2e_analysis import (
    load_end_to_end_results,
    summarize_end_to_end,
    write_end_to_end_summary,
)


def _results() -> pd.DataFrame:
    rows = []
    deltas = {
        "loss": [0.01, 0.00, -0.01],
        "expected_repair_value": [0.03, 0.02, 0.01],
        "auto_d3m_query": [0.00, 0.01, -0.02],
    }
    for seed in range(3):
        baseline = 0.50 + 0.01 * seed
        rows.append(
            {
                "dataset": "waterbirds",
                "noise_name": "uniform20",
                "seed": seed,
                "method": "no_correction",
                "budget_fraction": 0.0,
                "baseline_wga": baseline,
                "wga": baseline,
                "delta_wga": 0.0,
            }
        )
        for method, values in deltas.items():
            rows.append(
                {
                    "dataset": "waterbirds",
                    "noise_name": "uniform20",
                    "seed": seed,
                    "method": method,
                    "budget_fraction": 0.1,
                    "baseline_wga": baseline,
                    "wga": baseline + values[seed],
                    "delta_wga": values[seed],
                }
            )
    return pd.DataFrame(rows)


def test_summarize_end_to_end_pairs_by_seed_and_keeps_absolute_effect() -> None:
    summary, per_seed, metadata = summarize_end_to_end(
        _results(), bootstrap_replicates=200, rng_seed=7
    )
    rv = summary.query("method == 'expected_repair_value'").iloc[0]
    assert rv["paired_n_paired_seeds"] == 3
    assert rv["paired_mean_effect"] == pytest.approx(0.02)
    assert rv["absolute_mean_effect"] == pytest.approx(0.02)
    assert rv["paired_positive_seed_count"] == 3
    assert len(per_seed.query("method == 'auto_d3m_query'")) == 3
    assert metadata["paired_estimand"] == "delta_wga(method) - delta_wga(reference_method)"


def test_load_legacy_csv_requires_or_accepts_dataset_name(tmp_path) -> None:
    frame = _results().drop(columns=["dataset"])
    path = tmp_path / "legacy.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="dataset column"):
        load_end_to_end_results([path])
    loaded = load_end_to_end_results([path], dataset_names=["civilcomments"])
    assert set(loaded["dataset"]) == {"civilcomments"}


def test_write_end_to_end_summary_creates_replayable_artifacts(tmp_path) -> None:
    paths = write_end_to_end_summary(_results(), tmp_path, bootstrap_replicates=100)
    assert all(path.is_file() for path in paths.values())
    output = pd.read_csv(paths["summary"])
    assert set(output["method"]) == {"auto_d3m_query", "expected_repair_value"}
    manifest = paths["manifest"].read_text(encoding="utf-8")
    assert "reference_method" in manifest


def test_oracle_is_diagnostic_only_unless_explicitly_requested() -> None:
    oracle = _results().query("method == 'loss'").copy()
    oracle["method"] = "oracle_group_balanced"
    oracle["delta_wga"] = 0.10
    oracle["wga"] = oracle["baseline_wga"] + oracle["delta_wga"]
    frame = pd.concat(
        [
            _results(),
            oracle,
        ],
        ignore_index=True,
    )
    summary, _, metadata = summarize_end_to_end(frame, bootstrap_replicates=50)
    assert "oracle_group_balanced" not in set(summary["method"])
    assert "oracle_group_balanced" not in metadata["methods"]

    explicit, _, _ = summarize_end_to_end(
        frame, methods=["oracle_group_balanced"], bootstrap_replicates=50
    )
    assert set(explicit["method"]) == {"oracle_group_balanced"}
