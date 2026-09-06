from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def calibrate_flip_scale(
    weights: np.ndarray,
    target_rate: float,
    max_probability: float = 0.5,
) -> float:
    """Find a scalar *scale* such that clip(scale * weights, max_probability).mean() ≈ target_rate."""
    weights = np.asarray(weights, dtype=np.float64)

    low = 0.0
    high = 1.0
    while np.minimum(high * weights, max_probability).mean() < target_rate:
        high *= 2.0
        if high > 1e8:
            raise ValueError("Target noise rate is not achievable.")

    for _ in range(80):
        middle = (low + high) / 2.0
        achieved = np.minimum(middle * weights, max_probability).mean()
        if achieved < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def inject_noise(
    train_frame: pd.DataFrame,
    noise_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required = {"sample_id", "image_relpath", "clean_label", "group", "is_minority"}
    missing = required.difference(train_frame.columns)
    if missing:
        raise ValueError(f"Training frame missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    clean = train_frame["clean_label"].to_numpy(dtype=np.int64)
    minority = train_frame["is_minority"].to_numpy(dtype=bool)

    kind = noise_config["kind"]
    if kind == "existing_real":
        if "noisy_label" not in train_frame.columns:
            raise ValueError("existing_real noise requires a train-frame noisy_label column.")
        noisy = train_frame["noisy_label"].to_numpy(dtype=np.int64)
        corrupt = noisy != clean
        public = train_frame[["sample_id", "image_relpath", "split"]].copy()
        public["noisy_label"] = noisy
        public["current_label"] = noisy
        public["verified_flag"] = 0
        private = train_frame[
            ["sample_id", "clean_label", "place", "group", "is_minority", "split"]
        ].copy()
        private["is_noisy"] = corrupt.astype(int)
        private["noise_probability"] = np.nan
        stats = {
            "seed": int(seed),
            "n": int(len(train_frame)),
            "actual_noise_rate": float(corrupt.mean()),
            "majority_noise_rate": float(corrupt.mean()),
            "minority_noise_rate": 0.0,
            "num_noisy": int(corrupt.sum()),
        }
        return public, private, stats
    if kind == "uniform":
        probability = np.full(len(train_frame), float(noise_config["rate"]))
    elif kind == "group_dependent":
        majority_rate = float(noise_config["majority_rate"])
        minority_rate = float(noise_config["minority_rate"])
        probability = np.where(minority, minority_rate, majority_rate)
    elif kind == "disagreement_natural":
        toxicity = train_frame["toxicity_score"].to_numpy(dtype=float)
        toxicity = np.clip(toxicity, 0.0, 1.0)
        noisy = (rng.random(len(train_frame)) < toxicity).astype(np.int64)
        corrupt = noisy != clean
        probability = np.where(clean == 1, 1.0 - toxicity, toxicity)
        private = train_frame[
            ["sample_id", "clean_label", "place", "group", "is_minority", "split"]
        ].copy()
        private["is_noisy"] = corrupt.astype(int)
        private["noise_probability"] = probability
        private["toxicity_score"] = train_frame["toxicity_score"].to_numpy(dtype=float)
        private["disagreement_score"] = 2.0 * np.minimum(
            private["toxicity_score"], 1.0 - private["toxicity_score"],
        )
        public = train_frame[["sample_id", "image_relpath", "split"]].copy()
        public["noisy_label"] = noisy
        public["current_label"] = noisy
        public["verified_flag"] = 0
        stats = {
            "seed": int(seed), "n": int(len(train_frame)),
            "actual_noise_rate": float(corrupt.mean()),
            "majority_noise_rate": float(corrupt[~minority].mean()) if (~minority).any() else 0.0,
            "minority_noise_rate": float(corrupt[minority].mean()) if minority.any() else 0.0,
            "num_noisy": int(corrupt.sum()),
        }
        return public, private, stats
    elif kind == "disagreement_matched":
        toxicity = train_frame["toxicity_score"].to_numpy(dtype=float)
        disagreement = 2.0 * np.minimum(toxicity, 1.0 - toxicity)
        epsilon = float(noise_config.get("ambiguity_epsilon", 0.02))
        target_rate = float(noise_config["rate"])
        max_flip = float(noise_config.get("max_flip_probability", 0.50))
        weights = epsilon + disagreement
        scale = calibrate_flip_scale(weights, target_rate, max_flip)
        probability = np.minimum(scale * weights, max_flip)
    else:
        raise ValueError(f"Unsupported noise kind: {kind}")

    if kind not in ("disagreement_natural",):
        corrupt = rng.random(len(train_frame)) < probability
        noisy = clean.copy()
        noisy[corrupt] = 1 - noisy[corrupt]

        public = train_frame[["sample_id", "image_relpath", "split"]].copy()
        public["noisy_label"] = noisy
        public["current_label"] = noisy
        public["verified_flag"] = 0

        private = train_frame[
            ["sample_id", "clean_label", "place", "group", "is_minority", "split"]
        ].copy()
        private["is_noisy"] = corrupt.astype(int)
        private["noise_probability"] = probability

        if "toxicity_score" in train_frame.columns:
            private["toxicity_score"] = train_frame["toxicity_score"].to_numpy(dtype=float)
            private["disagreement_score"] = 2.0 * np.minimum(
                private["toxicity_score"], 1.0 - private["toxicity_score"],
            )

        stats = {
            "seed": int(seed), "n": int(len(train_frame)),
            "actual_noise_rate": float(corrupt.mean()),
            "majority_noise_rate": float(corrupt[~minority].mean()) if (~minority).any() else 0.0,
            "minority_noise_rate": float(corrupt[minority].mean()) if minority.any() else 0.0,
            "num_noisy": int(corrupt.sum()),
        }
        return public, private, stats


def write_validation_and_test_manifests(
    canonical: pd.DataFrame,
    manifest_root: str | Path,
) -> None:
    root = Path(manifest_root)
    root.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        frame = canonical.loc[canonical["split"] == split].copy()
        public = frame[["sample_id", "image_relpath", "split"]].copy()
        if split == "val":
            public["label"] = frame["clean_label"].to_numpy()
        public.to_csv(root / f"{split}_public.csv", index=False)

        private = frame[
            ["sample_id", "clean_label", "place", "group", "is_minority", "split"]
        ].copy()
        private.to_csv(root / f"{split}_private.csv", index=False)
