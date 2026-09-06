#!/usr/bin/env python
"""Experiment 2: Matched Query-Set Causal Intervention.
Proves that group coverage explains WGA gains beyond noise detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Tuple

# ── Load data for one condition ────────────────────────────────────────────
def load_condition(
    manifest_dir: str, features_path: str,
    noise_name: str, seed: int,
) -> dict:
    """Load private manifest and features for one (noise, seed) pair."""
    from robust_verify.analysis import align_frame
    from robust_verify.features import load_features
    
    train_feat, train_ids = load_features(features_path)
    
    private = pd.read_csv(
        Path(manifest_dir) / f"{noise_name}_seed{seed}_train_private.csv"
    )
    private = align_frame(private, train_ids)
    
    return {
        "features": train_feat,
        "private": private,
        "sample_ids": train_ids,
    }


# ── Stratification helpers ─────────────────────────────────────────────────
def stratify_samples(
    private: pd.DataFrame,
    n_strata_loss: int = 5,
    n_strata_margin: int = 5,
) -> dict:
    """Build stratified index pools for matching."""
    groups = private["group"].to_numpy(dtype=int)
    is_noisy = private["is_noisy"].to_numpy(dtype=bool)
    clean_labels = private["clean_label"].to_numpy(dtype=int)
    
    # Loss strata (use noise_probability as proxy if loss not available)
    loss_proxy = private.get("noise_probability", pd.Series(np.ones(len(private)))).to_numpy()
    
    # Build pool of noisy samples per (group, loss_decile)
    noisy_pools: dict[tuple, np.ndarray] = defaultdict(lambda: np.array([], dtype=int))
    clean_pools: dict[tuple, np.ndarray] = defaultdict(lambda: np.array([], dtype=int))
    
    loss_deciles = np.digitize(loss_proxy, np.percentile(loss_proxy, np.linspace(0, 100, n_strata_loss + 1)[1:-1]))
    
    all_idx = np.arange(len(private))
    for g in sorted(np.unique(groups)):
        for ld in range(n_strata_loss):
            mask = (groups == g) & (loss_deciles == ld)
            if not mask.any():
                continue
            indices = all_idx[mask]
            noisy = indices[is_noisy[mask]]
            clean = indices[~is_noisy[mask]]
            if len(noisy) > 0:
                noisy_pools[(int(g), int(ld))] = noisy
            if len(clean) > 0:
                clean_pools[(int(g), int(ld))] = clean
    
    worst_group = _get_worst_group(private)
    
    return {
        "noisy_pools": noisy_pools,
        "clean_pools": clean_pools,
        "groups": groups,
        "worst_group": worst_group,
        "is_noisy": is_noisy,
    }


def _get_worst_group(private: pd.DataFrame) -> int:
    groups = private["group"].to_numpy(dtype=int)
    counts = pd.Series(groups).value_counts()
    return int(counts.idxmin())  # smallest group = worst (heuristic)


# ── Query set construction ─────────────────────────────────────────────────
def construct_matched_pair(
    strata: dict,
    budget: int,
    precision: float,
    high_cover_frac: float = 0.7,
    low_cover_frac: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a matched pair (Q_high_cover, Q_low_cover).
    
    Both sets have:
    - Exactly budget samples
    - Exactly precision * budget noisy samples
    - Matched class and loss distributions
    
    Q_high_cover has high_cover_frac of noisy samples from worst group.
    Q_low_cover has low_cover_frac of noisy samples from worst group.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_noisy = int(budget * precision)
    n_clean = budget - n_noisy
    worst_group = strata["worst_group"]
    
    noisy_pools = strata["noisy_pools"]
    clean_pools = strata["clean_pools"]
    
    # Collect all available noisy indices, tagged by whether they're in worst group
    worst_noisy = []
    other_noisy = []
    for (g, ld), pool in noisy_pools.items():
        if g == worst_group:
            worst_noisy.extend(pool.tolist())
        else:
            other_noisy.extend(pool.tolist())
    
    worst_noisy = np.array(worst_noisy, dtype=int)
    other_noisy = np.array(other_noisy, dtype=int)
    
    # Collect all clean indices
    all_clean = np.concatenate([pool for pool in clean_pools.values()]) if clean_pools else np.array([], dtype=int)
    
    def _build_query(n_worst_noisy: int, n_other_noisy: int, n_clean: int):
        q = []
        if n_worst_noisy > 0 and len(worst_noisy) >= n_worst_noisy:
            q.extend(rng.choice(worst_noisy, size=n_worst_noisy, replace=False))
        if n_other_noisy > 0 and len(other_noisy) >= n_other_noisy:
            q.extend(rng.choice(other_noisy, size=n_other_noisy, replace=False))
        if n_clean > 0 and len(all_clean) >= n_clean:
            q.extend(rng.choice(all_clean, size=n_clean, replace=False))
        return np.array(q, dtype=int)
    
    n_worst_high = int(n_noisy * high_cover_frac)
    n_other_high = n_noisy - n_worst_high
    Q_high = _build_query(n_worst_high, n_other_high, n_clean)
    
    n_worst_low = int(n_noisy * low_cover_frac)
    n_other_low = n_noisy - n_worst_low
    Q_low = _build_query(n_worst_low, n_other_low, n_clean)
    
    return Q_high, Q_low


# ── Intervention: relabel + retrain ─────────────────────────────────────────
def run_intervention(
    features: np.ndarray,
    private: pd.DataFrame,
    query_set: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_ids: np.ndarray,
    evaluator,
    training_config: dict,
    seed: int,
) -> float:
    """Relabel query set with clean labels, retrain, return ΔWGA."""
    from robust_verify.training import train_linear_head, predict_from_state
    from robust_verify.data.access import PrivateEvaluator
    
    # Relabel
    repaired = train_labels.copy()
    clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
    for idx in query_set:
        repaired[idx] = clean_labels[idx]
    
    # Retrain
    result = train_linear_head(
        train_features=features,
        train_labels=repaired,
        val_features=val_features,
        val_labels=val_labels,
        config=training_config,
        seed=seed + 10000,
    )
    
    predictions, _ = predict_from_state(
        result.best_state, test_features,
        int(features.shape[1]),
        int(max(train_labels.max(), val_labels.max()) + 1),
        training_config,
    )
    metrics = evaluator.evaluate(test_ids, predictions)
    
    return float(metrics["wga"])


def compute_baseline_wga(
    features, train_labels, val_features, val_labels,
    test_features, test_ids, evaluator, training_config, seed,
):
    """Train on noisy labels, return baseline WGA."""
    from robust_verify.training import train_linear_head, predict_from_state
    result = train_linear_head(
        train_features=features, train_labels=train_labels,
        val_features=val_features, val_labels=val_labels,
        config=training_config, seed=seed,
    )
    predictions, _ = predict_from_state(
        result.best_state, test_features,
        int(features.shape[1]),
        int(max(train_labels.max(), val_labels.max()) + 1),
        training_config,
    )
    metrics = evaluator.evaluate(test_ids, predictions)
    return float(metrics["wga"])


# ── Main experiment ─────────────────────────────────────────────────────────
def run_causal_experiment(
    output_dir: str,
    noise_name: str,
    budget_fraction: float,
    precision: float = 0.4,
    n_pairs: int = 20,
    seed: int = 0,
) -> dict:
    """Run full causal intervention experiment for one condition."""
    from robust_verify.features import load_features
    from robust_verify.data.access import PrivateEvaluator
    from robust_verify.analysis import align_frame
    
    layout = {
        "manifests": Path(output_dir) / "manifests",
        "features": Path(output_dir) / "features",
    }
    
    # Load data
    train_feat, train_ids = load_features(layout["features"] / "train.npz")
    val_feat, val_ids = load_features(layout["features"] / "val.npz")
    test_feat, test_ids = load_features(layout["features"] / "test.npz")
    
    private = align_frame(
        pd.read_csv(layout["manifests"] / f"{noise_name}_seed{seed}_train_private.csv"),
        train_ids,
    )
    public = align_frame(
        pd.read_csv(layout["manifests"] / f"{noise_name}_seed{seed}_train_public.csv"),
        train_ids,
    )
    val_public = align_frame(
        pd.read_csv(layout["manifests"] / "val_public.csv"), val_ids,
    )
    
    noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    evaluator = PrivateEvaluator(layout["manifests"] / "test_private.csv")
    
    training_config = {
        "epochs": 5, "batch_size": 256, "learning_rate": 0.001,
        "weight_decay": 0.0001, "patience": 3,
        "selection_metric": "balanced_accuracy", "device": "auto",
    }
    
    # Compute baseline WGA
    baseline_wga = compute_baseline_wga(
        train_feat, noisy_labels, val_feat, val_labels,
        test_feat, test_ids, evaluator, training_config, seed,
    )
    print(f"  Baseline WGA: {baseline_wga:.4f}")
    
    # Build stratification
    budget = int(budget_fraction * len(train_ids))
    strata = stratify_samples(private)
    
    rng = np.random.default_rng(seed + 42)
    
    high_results = []
    low_results = []
    
    for pair_idx in range(n_pairs):
        Q_high, Q_low = construct_matched_pair(
            strata, budget, precision, rng=rng,
        )
        
        if len(Q_high) < budget * 0.5 or len(Q_low) < budget * 0.5:
            continue
        
        wga_high = run_intervention(
            train_feat, private, Q_high, noisy_labels,
            val_feat, val_labels, test_feat, test_ids,
            evaluator, training_config, seed + 10000 + pair_idx * 2,
        )
        wga_low = run_intervention(
            train_feat, private, Q_low, noisy_labels,
            val_feat, val_labels, test_feat, test_ids,
            evaluator, training_config, seed + 10001 + pair_idx * 2,
        )
        
        high_results.append(wga_high - baseline_wga)
        low_results.append(wga_low - baseline_wga)
        
        if (pair_idx + 1) % 5 == 0:
            d_high = np.mean(high_results)
            d_low = np.mean(low_results)
            print(f"  pair {pair_idx+1}/{n_pairs}: dWGA_high={d_high:+.4f}  dWGA_low={d_low:+.4f}  ATE={d_high-d_low:+.4f}")
    
    high_arr = np.array(high_results)
    low_arr = np.array(low_results)
    ate = high_arr - low_arr
    
    return {
        "baseline_wga": baseline_wga,
        "n_pairs": len(high_arr),
        "dWGA_high_mean": float(high_arr.mean()),
        "dWGA_low_mean": float(low_arr.mean()),
        "ATE_mean": float(ate.mean()),
        "ATE_sem": float(ate.std(ddof=1) / np.sqrt(len(ate))) if len(ate) > 1 else 0.0,
        "win_rate": float((ate > 0).mean()),
        "high_values": high_arr.tolist(),
        "low_values": low_arr.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    result = run_causal_experiment(
        output_dir="outputs/ablation_waterbirds_10seeds",
        noise_name="uniform20",
        budget_fraction=0.05,
        n_pairs=5,  # small test
        seed=0,
    )
    print(f"\n=== Results ===")
    for k, v in result.items():
        if isinstance(v, list):
            continue
        print(f"  {k}: {v}")
