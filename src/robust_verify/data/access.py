from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.utils import class_balanced_accuracy


class VerificationOracle:
    """Returns clean class labels only; group attributes are never returned."""

    def __init__(self, private_manifest: str | Path):
        private = pd.read_csv(private_manifest)
        self._labels = dict(
            zip(
                private["sample_id"].astype(int),
                private["clean_label"].astype(int),
                strict=True,
            )
        )

    def verify(self, sample_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(sample_ids, dtype=np.int64)
        missing = [int(sample_id) for sample_id in ids if int(sample_id) not in self._labels]
        if missing:
            raise KeyError(f"Unknown sample IDs requested: {missing[:5]}")
        return np.asarray([self._labels[int(sample_id)] for sample_id in ids], dtype=np.int64)


class PrivateEvaluator:
    """Evaluation boundary that internally reads clean labels and groups."""

    def __init__(self, private_manifest: str | Path):
        self.private = pd.read_csv(private_manifest).sort_values("sample_id")

    def evaluate(
        self,
        sample_ids: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        frame = self.private.set_index("sample_id").loc[np.asarray(sample_ids)].reset_index()
        labels = frame["clean_label"].to_numpy(dtype=np.int64)
        groups = frame["group"].to_numpy(dtype=np.int64)
        predictions = np.asarray(predictions, dtype=np.int64)

        if len(predictions) != len(labels):
            raise ValueError("Prediction count does not match evaluator manifest.")

        result: dict[str, float] = {
            "average_accuracy": float((predictions == labels).mean()),
            "balanced_accuracy": class_balanced_accuracy(labels, predictions),
        }

        group_scores = []
        for group in sorted(np.unique(groups)):
            mask = groups == group
            accuracy = float((predictions[mask] == labels[mask]).mean())
            result[f"group_{int(group)}_accuracy"] = accuracy
            result[f"group_{int(group)}_count"] = int(mask.sum())
            group_scores.append(accuracy)

        result["wga"] = float(min(group_scores))
        return result
