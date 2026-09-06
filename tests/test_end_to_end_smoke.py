from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from robust_verify.end_to_end import run_end_to_end


def _write_image(root, relative_path: str, label: int) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (235, 35, 35) if label == 0 else (35, 35, 235)
    Image.new("RGB", (24, 24), color=color).save(path)


def _make_manifests(root) -> None:
    manifests = root / "source" / "manifests"
    manifests.mkdir(parents=True)

    clean_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    noisy_train = clean_train.copy()
    noisy_train[[0, 5]] = 1 - noisy_train[[0, 5]]
    train_rows = []
    for sample_id, (clean, noisy) in enumerate(zip(clean_train, noisy_train, strict=True)):
        relative = f"images/train_{sample_id}.png"
        _write_image(root, relative, int(clean))
        train_rows.append(
            {
                "sample_id": sample_id,
                "image_relpath": relative,
                "split": "train",
                "noisy_label": int(noisy),
                "current_label": int(noisy),
                "verified_flag": 0,
                "clean_label": int(clean),
                "place": int(sample_id % 2),
                "group": int(2 * clean + (sample_id % 2)),
                "is_minority": int(clean != sample_id % 2),
                "is_noisy": int(clean != noisy),
            }
        )
    train = pd.DataFrame(train_rows)
    train.drop(columns=["clean_label", "place", "group", "is_minority", "is_noisy"]).to_csv(
        manifests / "toy_seed0_train_public.csv", index=False
    )
    train.drop(columns=["image_relpath", "noisy_label", "current_label", "verified_flag"]).to_csv(
        manifests / "toy_seed0_train_private.csv", index=False
    )

    def write_eval(split: str, ids: np.ndarray) -> None:
        labels = np.array([0, 1, 0, 1])
        rows = []
        for sample_id, label in zip(ids, labels, strict=True):
            relative = f"images/{split}_{sample_id}.png"
            _write_image(root, relative, int(label))
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "image_relpath": relative,
                    "split": split,
                    "label": int(label),
                    "clean_label": int(label),
                    "place": int(sample_id % 2),
                    "group": int(2 * label + (sample_id % 2)),
                    "is_minority": int(label != sample_id % 2),
                }
            )
        frame = pd.DataFrame(rows)
        public_columns = ["sample_id", "image_relpath", "split"]
        if split == "val":
            public_columns.append("label")
        frame[public_columns].to_csv(manifests / f"{split}_public.csv", index=False)
        frame.drop(columns=["label"], errors="ignore").to_csv(
            manifests / f"{split}_private.csv", index=False
        )

    write_eval("val", np.arange(100, 104))
    write_eval("test", np.arange(200, 204))


def _config(root) -> dict:
    return {
        "project": {
            "output_dir": str(root / "output"),
            "input_dir": str(root / "source"),
        },
        "data": {"dataset": "waterbirds", "root": str(root)},
        "features": {"skip_extraction": True},
        "training": {"selection_metric": "balanced_accuracy"},
        "end_to_end": {
            "backbone": "tiny_cnn",
            "pretrained": False,
            "input_size": 16,
            "augmentation": False,
            "batch_size": 8,
            "num_workers": 0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "epochs": 1,
            "patience": 1,
            "amp": False,
            "device": "cpu",
        },
        "noise": {"settings": [{"name": "toy", "kind": "uniform", "rate": 0.25}]},
        "experiment": {
            "seeds": [0],
            "budgets": [0.25],
            "methods": ["random", "loss"],
        },
    }


def test_end_to_end_runs_on_tiny_images_and_writes_replay_artifacts(tmp_path) -> None:
    _make_manifests(tmp_path)
    config = _config(tmp_path)

    results = run_end_to_end(config)

    assert set(results["method"]) == {"random", "loss"}
    assert set(results["budget_count"]) == {2}
    assert set(results["wga_semantics"]) == {"worst_group_accuracy"}
    assert "delta_wga" in results.columns
    result_root = tmp_path / "output" / "end_to_end"
    assert (result_root / "results.csv").is_file()
    for artifact in results["query_artifact"]:
        assert (tmp_path / "output" / artifact).is_file()
    for checkpoint in results["checkpoint"]:
        assert (tmp_path / "output" / checkpoint).is_file()
    for artifact in results["prediction_artifact"]:
        assert (tmp_path / "output" / artifact).is_file()
    assert results["baseline_checkpoint"].nunique() == 1
    assert results["baseline_prediction_artifact"].nunique() == 1
    assert (tmp_path / "output" / results.iloc[0]["baseline_checkpoint"]).is_file()
    assert (
        tmp_path / "output" / results.iloc[0]["baseline_prediction_artifact"]
    ).is_file()


def test_end_to_end_runs_practical_tracin_with_feature_context(tmp_path) -> None:
    _make_manifests(tmp_path)
    config = _config(tmp_path)
    config = copy.deepcopy(config)
    config["experiment"]["methods"] = ["loss", "tracin_val_uncertainty"]

    results = run_end_to_end(config)

    assert set(results["method"]) == {"loss", "tracin_val_uncertainty"}
    assert set(results["budget_count"]) == {2}
    assert results["delta_wga"].notna().all()
    for artifact in results["query_artifact"]:
        assert (tmp_path / "output" / artifact).is_file()


def test_end_to_end_runs_repair_value_with_legal_feature_context(tmp_path) -> None:
    _make_manifests(tmp_path)
    config = copy.deepcopy(_config(tmp_path))
    config["experiment"]["methods"] = ["loss", "expected_repair_value"]
    config["experiment"]["baseline_options"] = {
        "repair_value_tail_fraction": 0.20,
    }

    results = run_end_to_end(config)

    assert set(results["method"]) == {"loss", "expected_repair_value"}
    assert set(results["budget_count"]) == {2}
    assert results["delta_wga"].notna().all()
    repair = results.loc[results["method"].eq("expected_repair_value")].iloc[0]
    assert (tmp_path / "output" / repair["query_artifact"]).is_file()
    assert (tmp_path / "output" / repair["checkpoint"]).is_file()


def test_end_to_end_runs_oof_cleanlab_and_repair_value_together(tmp_path) -> None:
    """OOF Cleanlab is generated from the legal probe context and replayable."""
    _make_manifests(tmp_path)
    config = copy.deepcopy(_config(tmp_path))
    config["experiment"]["methods"] = [
        "random",
        "loss",
        "noise_score",
        "oof_cleanlab",
        "expected_repair_value",
    ]
    config["experiment"]["baseline_options"] = {
        "repair_value_tail_fraction": 0.20,
        "oof_folds": 2,
        "oof_alpha": 0.0001,
        "oof_max_iter": 10,
        "oof_tol": 0.001,
        "oof_split_seed_offset": 17003,
    }

    results = run_end_to_end(config)

    assert set(results["method"]) == {
        "random",
        "loss",
        "noise_score",
        "oof_cleanlab",
        "expected_repair_value",
    }
    assert results["oof_probability_artifact"].nunique() == 1
    oof_path = tmp_path / "output" / results.iloc[0]["oof_probability_artifact"]
    assert oof_path.is_file()
    with np.load(oof_path, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "sample_ids",
            "probabilities",
            "fold_ids",
            "seed",
            "folds",
        }
        assert payload["probabilities"].shape == (8, 2)
        assert np.allclose(payload["probabilities"].sum(axis=1), 1.0, atol=1e-5)
        assert set(np.unique(payload["fold_ids"])) == {0, 1}


def test_end_to_end_runs_noise_gated_repair_value(tmp_path) -> None:
    _make_manifests(tmp_path)
    config = copy.deepcopy(_config(tmp_path))
    config["experiment"]["methods"] = ["noise_score", "noise_gated_repair_value"]
    config["experiment"]["baseline_options"] = {
        "repair_value_tail_fraction": 0.20,
        "repair_value_candidate_multiplier": 2.0,
        "repair_value_noise_anchor_fraction": 0.5,
    }

    results = run_end_to_end(config)

    assert set(results["method"]) == {"noise_score", "noise_gated_repair_value"}
    assert set(results["budget_count"]) == {2}
    assert results["delta_wga"].notna().all()
    v2 = results.loc[results["method"].eq("noise_gated_repair_value")].iloc[0]
    assert (tmp_path / "output" / v2["query_artifact"]).is_file()
    assert (tmp_path / "output" / v2["checkpoint"]).is_file()


@pytest.mark.parametrize("method", ["oracle_noise", "tracin_wga_oracle"])
def test_end_to_end_rejects_oracle_acquisition_methods(tmp_path, method: str) -> None:
    _make_manifests(tmp_path)
    config = _config(tmp_path)
    config = copy.deepcopy(config)
    config["experiment"]["methods"] = [method]

    with pytest.raises(ValueError, match="cannot use oracle methods"):
        run_end_to_end(config)
