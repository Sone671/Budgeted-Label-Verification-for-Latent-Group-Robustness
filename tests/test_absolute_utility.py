from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_verify.absolute_utility import (
    audit_absolute_utility,
    exact_two_sided_sign_test,
    prepare_results,
)


def _results() -> pd.DataFrame:
    rows = []
    values = {
        "loss": [0.01, 0.00, -0.01],
        "practical_bad": [-0.02, -0.03, -0.02],
        "practical_good": [0.03, 0.02, 0.01],
        "oracle_noise": [0.10, 0.10, 0.10],
    }
    for method, deltas in values.items():
        for seed, delta in enumerate(deltas):
            baseline = 0.50 + seed * 0.01
            rows.append(
                {
                    "dataset": "toy",
                    "noise_name": "uniform20",
                    "seed": seed,
                    "method": method,
                    "budget_fraction": 0.02,
                    "baseline_wga": baseline,
                    "wga": baseline + delta,
                    "delta_wga": delta,
                }
            )
    return pd.DataFrame(rows)


def test_absolute_and_loss_comparisons_stay_separate() -> None:
    audit = audit_absolute_utility(prepare_results(_results()), n_bootstrap=400, rng_seed=7)

    assert set(audit.absolute_vs_no_correction["method"]) == {"practical_bad", "practical_good"}
    bad_absolute = audit.absolute_vs_no_correction.query("method == 'practical_bad'").iloc[0]
    assert bad_absolute["comparator"] == "no_correction"
    assert bad_absolute["mean_effect"] == pytest.approx(-0.07 / 3.0)
    assert bad_absolute["evidence"] == "absolute_harm_vs_no_correction"

    bad_loss = audit.paired_vs_loss.query("method == 'practical_bad'").iloc[0]
    assert bad_loss["comparator"] == "loss"
    assert bad_loss["mean_method_delta_wga"] == pytest.approx(-0.07 / 3.0)
    assert bad_loss["mean_loss_delta_wga"] == pytest.approx(0.0)
    assert bad_loss["mean_effect"] == pytest.approx(-0.07 / 3.0)
    assert bad_loss["evidence"] == "relative_harm_vs_loss"


def test_oracles_are_never_reported_as_practical_methods() -> None:
    audit = audit_absolute_utility(prepare_results(_results()), n_bootstrap=100)
    assert "oracle_noise" not in audit.absolute_vs_no_correction["method"].tolist()
    assert "oracle_noise" not in audit.paired_vs_loss["method"].tolist()


def test_per_seed_pairing_uses_only_common_seeds() -> None:
    frame = _results()
    frame = frame[~((frame["method"] == "loss") & (frame["seed"] == 2))]
    audit = audit_absolute_utility(prepare_results(frame), n_bootstrap=100)
    row = audit.coverage.query("method == 'practical_bad'").iloc[0]
    assert row["candidate_seed_count"] == 3
    assert row["loss_seed_count"] == 2
    assert row["paired_seed_count"] == 2
    assert row["unpaired_candidate_seed_count"] == 1
    assert len(audit.per_seed_vs_loss.query("method == 'practical_bad'")) == 2


def test_prepare_results_rejects_inconsistent_no_correction_effect() -> None:
    frame = _results()
    frame.loc[0, "delta_wga"] = 0.99
    with pytest.raises(ValueError, match="delta_wga does not equal"):
        prepare_results(frame)


def test_exact_sign_test_drops_ties() -> None:
    assert exact_two_sided_sign_test(np.array([-1.0, -2.0, -3.0])) == pytest.approx(0.25)
    assert exact_two_sided_sign_test(np.array([-1.0, 0.0, 0.0])) == pytest.approx(1.0)
