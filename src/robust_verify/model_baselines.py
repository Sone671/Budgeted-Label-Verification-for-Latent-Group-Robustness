from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robust_verify.analysis import align_frame
from robust_verify.data.access import PrivateEvaluator
from robust_verify.training import (
    make_initial_state,
    predict_from_state,
    train_group_dro_linear_head,
    train_jtt_linear_head,
)


SUPPORTED_MODEL_BASELINES = ("erm", "oracle_groupdro", "jtt")


def _load_existing_rows(
    final_path: Path,
    partial_path: Path,
) -> dict[tuple[str, int, str], dict]:
    frames = []
    for path in (final_path, partial_path):
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["noise_name", "seed", "method"], keep="last"
    )
    return {
        (str(row["noise_name"]), int(row["seed"]), str(row["method"])): row
        for row in combined.to_dict("records")
    }


def _save_checkpoint(path: Path, probe, **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "best_state": probe.best_state,
            "initial_state": probe.initial_state,
            "best_epoch": probe.best_epoch,
            "best_validation_metric": probe.best_validation_metric,
            **metadata,
        },
        path,
    )


def _evaluate_state(
    state: dict[str, torch.Tensor],
    test_features: np.ndarray | torch.Tensor,
    test_ids: np.ndarray,
    input_dim: int,
    num_classes: int,
    training_config: dict[str, Any],
    evaluator: PrivateEvaluator,
) -> dict[str, float]:
    predictions, _ = predict_from_state(
        state, test_features, input_dim, num_classes, training_config
    )
    return evaluator.evaluate(test_ids, predictions)


def run_model_baselines(
    *,
    config: dict[str, Any],
    layout: dict[str, Path],
    source_layout: dict[str, Path] | None = None,
    train_features: np.ndarray | torch.Tensor,
    train_ids: np.ndarray,
    val_features: np.ndarray | torch.Tensor,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    test_features: np.ndarray | torch.Tensor,
    test_ids: np.ndarray,
    test_evaluator: PrivateEvaluator,
) -> pd.DataFrame:
    """Run ERM, oracle GroupDRO, and JTT without altering query-method rows."""
    experiment = config.get("experiment", {})
    requested = [str(name).lower() for name in experiment.get("training_baselines", [])]
    if not requested:
        return pd.DataFrame()
    if len(requested) != len(set(requested)):
        raise ValueError("experiment.training_baselines contains duplicate names.")
    unsupported = sorted(set(requested).difference(SUPPORTED_MODEL_BASELINES))
    if unsupported:
        raise ValueError(
            f"Unsupported training baselines: {unsupported}. "
            f"Available: {list(SUPPORTED_MODEL_BASELINES)}"
        )

    options = experiment.get("training_baseline_options", {})
    source_layout = source_layout or layout
    training_config = config["training"]
    final_path = layout["stage1"] / "model_baselines.csv"
    partial_path = layout["stage1"] / "model_baselines_partial.csv"
    rows_by_key = _load_existing_rows(final_path, partial_path)
    expected_keys: list[tuple[str, int, str]] = []

    for noise_setting in config["noise"]["settings"]:
        noise_name = str(noise_setting["name"])
        noise_kind = str(noise_setting["kind"])
        for seed_value in experiment["seeds"]:
            seed = int(seed_value)
            prefix = f"{noise_name}_seed{seed}"
            public = align_frame(
                pd.read_csv(source_layout["manifests"] / f"{prefix}_train_public.csv"),
                train_ids,
            )
            private = align_frame(
                pd.read_csv(source_layout["manifests"] / f"{prefix}_train_private.csv"),
                train_ids,
            )
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
            train_groups = private["group"].to_numpy(dtype=np.int64)
            input_dim = int(train_features.shape[1])
            num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
            initial_state = make_initial_state(input_dim, num_classes, seed)

            for method in requested:
                key = (noise_name, seed, method)
                expected_keys.append(key)
                if key in rows_by_key:
                    continue

                metadata: dict[str, Any] = {}
                if method == "erm":
                    checkpoint_path = layout["probes"] / f"{prefix}.pt"
                    if not checkpoint_path.exists():
                        raise FileNotFoundError(
                            f"ERM probe is missing: {checkpoint_path}. Run Stage 1 first."
                        )
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    state = checkpoint["best_state"]
                    best_epoch = int(checkpoint["best_epoch"])
                    best_validation_metric = float(
                        checkpoint["best_validation_metric"]
                    )
                    selection_metric = str(
                        training_config.get("selection_metric", "balanced_accuracy")
                    )
                    uses_oracle_groups = False
                elif method == "oracle_groupdro":
                    groupdro_config = copy.deepcopy(training_config)
                    groupdro_config.update(options.get("oracle_groupdro", {}))
                    groupdro_config.setdefault(
                        "selection_metric", "worst_group_accuracy"
                    )
                    # The generic training config usually selects balanced
                    # accuracy; the oracle baseline explicitly selects WGA.
                    if "selection_metric" not in options.get("oracle_groupdro", {}):
                        groupdro_config["selection_metric"] = "worst_group_accuracy"
                    probe = train_group_dro_linear_head(
                        train_features=train_features,
                        train_labels=noisy_labels,
                        train_groups=train_groups,
                        val_features=val_features,
                        val_labels=val_labels,
                        val_groups=val_groups,
                        config=groupdro_config,
                        seed=seed,
                        initial_state=copy.deepcopy(initial_state),
                    )
                    checkpoint_path = layout["probes"] / f"{prefix}_oracle_groupdro.pt"
                    _save_checkpoint(checkpoint_path, probe)
                    state = probe.best_state
                    best_epoch = int(probe.best_epoch)
                    best_validation_metric = float(probe.best_validation_metric)
                    selection_metric = str(groupdro_config["selection_metric"])
                    uses_oracle_groups = True
                    metadata.update(
                        {
                            "group_step_size": float(
                                groupdro_config.get("group_step_size", 0.01)
                            ),
                            "group_adjustment": float(
                                groupdro_config.get("group_adjustment", 0.0)
                            ),
                        }
                    )
                else:
                    jtt_options = options.get("jtt", {})
                    probe, error_mask = train_jtt_linear_head(
                        train_features=train_features,
                        train_labels=noisy_labels,
                        val_features=val_features,
                        val_labels=val_labels,
                        config=training_config,
                        jtt_config=jtt_options,
                        seed=seed,
                        initial_state=copy.deepcopy(initial_state),
                    )
                    checkpoint_path = layout["probes"] / f"{prefix}_jtt.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        probe,
                        identification_error_count=int(error_mask.sum()),
                    )
                    state = probe.best_state
                    best_epoch = int(probe.best_epoch)
                    best_validation_metric = float(probe.best_validation_metric)
                    selection_metric = str(
                        jtt_options.get("final_training", {}).get(
                            "selection_metric",
                            training_config.get(
                                "selection_metric", "balanced_accuracy"
                            ),
                        )
                    )
                    uses_oracle_groups = False
                    metadata.update(
                        {
                            "jtt_identification_epochs": int(
                                jtt_options.get("identification_epochs", 5)
                            ),
                            "jtt_upweight": float(jtt_options.get("upweight", 10.0)),
                            "jtt_identification_error_count": int(error_mask.sum()),
                        }
                    )

                metrics = _evaluate_state(
                    state=state,
                    test_features=test_features,
                    test_ids=test_ids,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    training_config=training_config,
                    evaluator=test_evaluator,
                )
                row = {
                    "noise_name": noise_name,
                    "noise_kind": noise_kind,
                    "seed": seed,
                    "method": method,
                    "budget_fraction": 0.0,
                    "budget_count": 0,
                    "label_source": "noisy_train_labels",
                    "uses_oracle_groups": uses_oracle_groups,
                    "best_epoch": best_epoch,
                    "selection_metric": selection_metric,
                    "best_validation_metric": best_validation_metric,
                    **metadata,
                    **metrics,
                }
                rows_by_key[key] = row
                pd.DataFrame(
                    [rows_by_key[item] for item in expected_keys if item in rows_by_key]
                ).to_csv(partial_path, index=False)
                print(
                    f"[model baseline {noise_name} s={seed}] "
                    f"{method:16s} WGA={metrics['wga']:.4f}"
                )

    result = pd.DataFrame([rows_by_key[key] for key in expected_keys])
    result.to_csv(final_path, index=False)
    if partial_path.exists():
        partial_path.unlink()
    return result
