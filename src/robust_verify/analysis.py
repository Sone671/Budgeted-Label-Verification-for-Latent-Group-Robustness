from __future__ import annotations

import math

import numpy as np
import pandas as pd


def align_frame(frame: pd.DataFrame, sample_ids: np.ndarray) -> pd.DataFrame:
    indexed = frame.set_index("sample_id")
    missing = sorted(set(map(int, sample_ids)).difference(indexed.index.astype(int)))
    if missing:
        raise KeyError(f"Manifest missing sample IDs: {missing[:5]}")
    return indexed.loc[np.asarray(sample_ids, dtype=int)].reset_index()


def query_composition(
    query_positions: np.ndarray,
    private_frame: pd.DataFrame,
) -> dict[str, float | int]:
    selected = private_frame.iloc[np.asarray(query_positions, dtype=int)]
    n = len(selected)
    if n == 0:
        return {
            "num_queried": 0,
            "num_corrected": 0,
            "noise_precision": 0.0,
            "minority_query_rate": 0.0,
            "clean_minority_rate": 0.0,
            "minority_recall": 0.0,
            "group_query_entropy": 0.0,
            "mislabeled_majority": 0,
            "mislabeled_minority": 0,
            "clean_majority": 0,
            "clean_minority": 0,
        }

    noisy = selected["is_noisy"].to_numpy(dtype=bool)
    minority = selected["is_minority"].to_numpy(dtype=bool)
    all_minority = int(private_frame["is_minority"].sum())

    group_counts = selected["group"].value_counts(normalize=True)
    entropy = -float(sum(p * math.log(p) for p in group_counts if p > 0))

    result = {
        "num_queried": int(n),
        "num_corrected": int(noisy.sum()),
        "noise_precision": float(noisy.mean()),
        "minority_query_rate": float(minority.mean()),
        "clean_minority_rate": float((~noisy & minority).mean()),
        "minority_recall": float(minority.sum() / max(all_minority, 1)),
        "group_query_entropy": entropy,
        "mislabeled_majority": int((noisy & ~minority).sum()),
        "mislabeled_minority": int((noisy & minority).sum()),
        "clean_majority": int((~noisy & ~minority).sum()),
        "clean_minority": int((~noisy & minority).sum()),
    }

    # Disagreement metrics (CivilComments / datasets with annotator scores)
    if "disagreement_score" in private_frame.columns:
        queried_disag = private_frame.iloc[
            query_positions
        ]["disagreement_score"].to_numpy(dtype=float)
        dataset_disag = private_frame["disagreement_score"].to_numpy(dtype=float)
        result["queried_disagreement_mean"] = float(queried_disag.mean())
        result["queried_high_disagreement_rate"] = float((queried_disag >= 0.8).mean())
        result["dataset_disagreement_mean"] = float(dataset_disag.mean())
        result["disagreement_lift"] = (
            float(queried_disag.mean() / dataset_disag.mean())
            if dataset_disag.mean() > 0
            else 0.0
        )

    return result


def score_correlations(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    legal = pd.DataFrame(scores)
    correlation = legal.corr(method="spearman")
    return correlation.rename_axis("score").reset_index()
