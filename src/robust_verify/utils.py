from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def budget_to_count(fraction: float, n: int) -> int:
    if not 0 <= fraction <= 1:
        raise ValueError(f"Budget fraction must be in [0, 1], got {fraction}")
    if fraction == 0:
        return 0
    return min(n, max(1, int(math.floor(fraction * n + 0.5))))


def normalized_rank(values: np.ndarray, descending: bool = False) -> np.ndarray:
    """Return tie-aware percentile ranks in [0, 1].

    Equal values receive the same average rank, avoiding sample-order artifacts
    for discrete signals such as prediction disagreement.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("normalized_rank expects a one-dimensional array.")
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(n, dtype=np.float64)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        average_position = 0.5 * (start + end - 1)
        sorted_ranks[start:end] = average_position
        start = end

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    ranks /= n - 1
    return 1.0 - ranks if descending else ranks


def rank_within_class(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    labels = np.asarray(labels)
    if len(values) != len(labels):
        raise ValueError("values and labels must have equal length.")

    result = np.zeros(len(values), dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == label
        result[mask] = normalized_rank(values[mask])
    return result


def entropy_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def class_balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    class_scores = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.any():
            class_scores.append(float((predictions[mask] == labels[mask]).mean()))
    return float(np.mean(class_scores)) if class_scores else float("nan")


def safe_sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))
