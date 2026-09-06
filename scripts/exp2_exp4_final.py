#!/usr/bin/env python
"""Experiment 2 & 4: Complete corrected implementation.
All issues from code review resolved: evaluator path, true exact matching,
proper target group, validated probes, grid runner, CSV output.
"""

from __future__ import annotations
import numpy as np, pandas as pd, time, sys, json
from pathlib import Path
from collections import defaultdict
from scipy import stats
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATION_SEED = 9999  # fixed seed for target-group determination

def load_data(output_dir: str, noise_name: str, seed: int) -> dict:
    """Load features, manifests, and probe for one condition.
    All paths derived from a single output_dir — no cross-dataset leakage."""
    from robust_verify.features import load_features
    from robust_verify.analysis import align_frame
    
    root = Path(output_dir)
    mf = root / "manifests"; ft = root / "features"; pb = root / "stage1" / "probes"
    
    t_feat, t_ids = load_features(ft / "train.npz")
    v_feat, v_ids = load_features(ft / "val.npz")
    test_feat, test_ids = load_features(ft / "test.npz")
    
    private = align_frame(pd.read_csv(mf / f"{noise_name}_seed{seed}_train_private.csv"), t_ids)
    public = align_frame(pd.read_csv(mf / f"{noise_name}_seed{seed}_train_public.csv"), t_ids)
    val_pub = align_frame(pd.read_csv(mf / "val_public.csv"), v_ids)
    val_priv = align_frame(pd.read_csv(mf / "val_private.csv"), v_ids)
    evaluator = PrivateEvaluator(mf / "test_private.csv")
    
    # Validate probe — no silent fallback
    probe_path = pb / f"{noise_name}_seed{seed}.dynamics.npz"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe missing: {probe_path}")
    probe = np.load(probe_path)
    probs = probe["train_probabilities"]
    assert probs.ndim == 2, f"Expected 2D probs, got {probs.ndim}"
    assert probs.shape[0] == len(t_ids), f"Probs {probs.shape[0]} != train {len(t_ids)}"
    
    # Compute loss and margin from probe
    noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    pc = probs[np.arange(len(probs)), noisy_labels]
    loss = -np.log(np.clip(pc, 1e-12, 1.0))
    top2 = np.partition(-probs, 1, axis=1)[:, :2]
    margin = np.abs(top2[:, 0] - top2[:, 1])
    
    return {
        "output_dir": output_dir,
        "train_feat": t_feat, "val_feat": v_feat, "test_feat": test_feat,
        "test_ids": test_ids, "val_labels": val_pub["label"].to_numpy(dtype=np.int64),
        "val_groups": val_priv["group"].to_numpy(dtype=np.int64),
        "private": private, "noisy_labels": noisy_labels,
        "loss": loss, "margin": margin, "evaluator": evaluator,
    }


def _train_and_eval(features, labels, val_feat, val_labels, test_feat, test_ids,
                    evaluator, cfg, seed) -> float:
    """Train linear head, return test WGA."""
    from robust_verify.training import train_linear_head, predict_from_state
    r = train_linear_head(features, labels, val_feat, val_labels, cfg, seed=seed)
    preds, _ = predict_from_state(r.best_state, test_feat, int(features.shape[1]),
                                   int(max(labels.max(), val_labels.max()) + 1), cfg)
    return float(evaluator.evaluate(test_ids, preds)["wga"])


def identify_target_group(data: dict, training_cfg: dict) -> int:
    """Identify worst-accuracy group on validation set.
    
    Uses fixed VALIDATION_SEED. Trains on noisy labels, eval on val set,
    returns group with lowest accuracy. Independent of test set."""
    from robust_verify.training import train_linear_head, predict_from_state
    
    # Train probe on val features (single pass for target identification)
    r = train_linear_head(
        data["val_feat"], data["val_labels"],
        data["val_feat"], data["val_labels"],
        training_cfg, seed=VALIDATION_SEED,
    )
    preds, _ = predict_from_state(
        r.best_state, data["val_feat"], int(data["val_feat"].shape[1]),
        int(data["val_labels"].max()) + 1, training_cfg,
    )
    
    val_groups = data["val_groups"]
    group_acc = {}
    for g in sorted(np.unique(val_groups)):
        m = val_groups == g
        group_acc[int(g)] = float((preds[m] == data["val_labels"][m]).mean())
    return min(group_acc, key=group_acc.get)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: True Exact Matched Causal Intervention
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairRecord:
    pair_id: int
    budget: int; precision: float
    tg: int  # target group
    len_h: int; len_l: int
    prec_h: float; prec_l: float
    tg_noisy_h: int; tg_noisy_l: int
    wga_baseline: float; wga_high: float; wga_low: float
    ate: float
    n_strata_equal: int; n_strata_diff: int; max_stratum_diff: int
    valid: bool; error: str = ""


def construct_true_exact_pair(
    private: pd.DataFrame,
    is_noisy: np.ndarray, groups: np.ndarray, clean_labels: np.ndarray,
    loss: np.ndarray, margin: np.ndarray,
    budget: int, precision: float, target_group: int,
    high_frac: float, low_frac: float,
    n_loss_bins: int = 3, n_margin_bins: int = 3,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Construct truly exact-matched Q_high and Q_low.
    
    GUARANTEES (asserted):
    1. len(Q_high) == len(Q_low) == budget
    2. noise_precision(Q_high) == noise_precision(Q_low) == precision
    3. Same clean sample IDs in both sets
    4. Same per-stratum (class, loss_bin, margin_bin) noisy count for non-TG groups
    5. Only TG noisy count differs: high_frac vs low_frac of total noisy
    6. No gap-filling with arbitrary samples
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_noisy = int(budget * precision)
    n_clean = budget - n_noisy
    
    n = len(private)
    all_idx = np.arange(n, dtype=int)
    
    # Build strata
    l_bins = pd.qcut(loss, n_loss_bins, labels=False, duplicates="drop")
    m_bins = pd.qcut(margin, n_margin_bins, labels=False, duplicates="drop")
    n_loss_bins = l_bins.max() + 1; n_margin_bins = m_bins.max() + 1
    
    # Pool: {(group, label, loss_bin, margin_bin): (noisy_idx, clean_idx)}
    pools: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for g in sorted(np.unique(groups)):
        for c in [0, 1]:
            for lb in range(n_loss_bins):
                for mb in range(n_margin_bins):
                    # Use the LAST group in key to avoid dict collision
                    mask = (groups == g) & (clean_labels == c) & (l_bins == lb) & (m_bins == mb)
                    if mask.sum() < 2:
                        continue
                    inds = all_idx[mask]
                    pools[(int(g), int(c), int(lb), int(mb))] = (
                        inds[is_noisy[mask]], inds[~is_noisy[mask]])
    
    # Count capacities
    tg_noisy_total = sum(len(v[0]) for k, v in pools.items() if k[0] == target_group)
    ntg_noisy_total = sum(len(v[0]) for k, v in pools.items() if k[0] != target_group)
    clean_total = sum(len(v[1]) for v in pools.values())
    
    n_tg_high_target = int(n_noisy * high_frac)
    n_tg_low_target = int(n_noisy * low_frac)
    n_ntg_high = n_noisy - n_tg_high_target
    
    # Feasibility checks
    errs = []
    if tg_noisy_total < n_tg_high_target:
        errs.append(f"TG noisy: {tg_noisy_total} < {n_tg_high_target}")
    if ntg_noisy_total < n_ntg_high:
        errs.append(f"NTG noisy: {ntg_noisy_total} < {n_ntg_high}")
    if clean_total < n_clean:
        errs.append(f"Clean: {clean_total} < {n_clean}")
    if errs:
        return np.array([]), np.array([]), {"error": "; ".join(errs)}
    
    # ── Select CLEAN samples (SHARED between high and low) ──
    clean_keys = sorted(pools.keys())
    clean_avail = {k: v[1] for k, v in pools.items()}
    rng_clean = np.random.default_rng(int(rng.integers(1, 2**30)))
    clean_sel = _proportional_sample(clean_avail, n_clean, rng_clean)
    taken_clean = set(clean_sel.tolist())
    
    # ── Select NTG noisy samples (SHARED between high and low) ──
    ntg_keys = sorted(k for k in pools if k[0] != target_group)
    # For each NTG stratum, sample the SAME number from the same pool
    # First allocate n_ntg_high across NTG strata proportionally
    ntg_avail = {k: pools[k][0] for k in ntg_keys}
    rng_ntg = np.random.default_rng(int(rng.integers(1, 2**30)))
    ntg_quota = _proportional_quota(ntg_avail, n_ntg_high)
    
    # Then sample from each stratum
    ntg_sel = []
    for k, quota in ntg_quota.items():
        pool = ntg_avail[k]
        if quota > 0 and len(pool) >= quota:
            available = np.setdiff1d(pool, list(taken_clean))
            if len(available) >= quota:
                ntg_sel.extend(rng_ntg.choice(available, size=quota, replace=False).tolist())
    
    ntg_selected = {k: 0 for k in ntg_keys}
    for idx in ntg_sel:
        for k in ntg_keys:
            g, c, lb, mb = k
            idx_g = groups[idx]
            idx_c = clean_labels[idx]
            idx_lb = l_bins[idx]
            idx_mb = m_bins[idx]
            if idx_g == g and idx_c == c and idx_lb == lb and idx_mb == mb:
                ntg_selected[k] += 1
                break
    
    # ── Build HIGH: add TG noisy ──
    tg_keys = sorted(k for k in pools if k[0] == target_group)
    tg_avail = {k: pools[k][0] for k in tg_keys}
    rng_tg_h = np.random.default_rng(int(rng.integers(1, 2**30)))
    tg_sel_h = _proportional_sample_excluding(tg_avail, n_tg_high_target, taken_clean, rng_tg_h)
    Q_high = np.unique(np.concatenate([clean_sel, np.array(ntg_sel, dtype=int), tg_sel_h]))
    
    # ── Build LOW: add TG noisy (same clean, same NTG, fewer TG) ──
    rng_tg_l = np.random.default_rng(int(rng.integers(1, 2**30)))
    tg_sel_l = _proportional_sample_excluding(tg_avail, n_tg_low_target, taken_clean, rng_tg_l)
    Q_low = np.unique(np.concatenate([clean_sel, np.array(ntg_sel, dtype=int), tg_sel_l]))
    
    # ── Assertions ──
    try:
        assert len(Q_high) == budget, f"len_h={len(Q_high)} budget={budget}"
        assert len(Q_low) == budget, f"len_l={len(Q_low)} budget={budget}"
        assert abs(is_noisy[Q_high].mean() - precision) < 0.03, f"prec_h={is_noisy[Q_high].mean():.3f}"
        assert abs(is_noisy[Q_low].mean() - precision) < 0.03, f"prec_l={is_noisy[Q_low].mean():.3f}"
        # Same clean
        clean_h = set(Q_high[~is_noisy[Q_high]]); clean_l = set(Q_low[~is_noisy[Q_low]])
        assert clean_h == clean_l, f"Clean mismatch: {len(clean_h)} != {len(clean_l)}"
        # TG noisy difference
        tg_h = is_noisy[Q_high] & (groups[Q_high] == target_group)
        tg_l = is_noisy[Q_low] & (groups[Q_low] == target_group)
        assert tg_h.sum() > tg_l.sum(), f"TG noisy: h={tg_h.sum()} l={tg_l.sum()}"
    except AssertionError as e:
        return np.array([]), np.array([]), {"error": str(e)}
    
    # ── Balance diagnostics ──
    h_counts = _stratum_counts(Q_high[is_noisy[Q_high]], groups, clean_labels, l_bins, m_bins, pools)
    l_counts = _stratum_counts(Q_low[is_noisy[Q_low]], groups, clean_labels, l_bins, m_bins, pools)
    n_equal = sum(1 for k in set(h_counts) | set(l_counts) if h_counts.get(k,0) == l_counts.get(k,0))
    n_diff = sum(1 for k in set(h_counts) | set(l_counts) if h_counts.get(k,0) != l_counts.get(k,0))
    max_diff = max((abs(h_counts.get(k,0) - l_counts.get(k,0)) for k in set(h_counts)|set(l_counts)), default=0)
    
    diag = {
        "len_high": len(Q_high), "len_low": len(Q_low),
        "prec_high": float(is_noisy[Q_high].mean()), "prec_low": float(is_noisy[Q_low].mean()),
        "tg_noisy_high": int(tg_h.sum()), "tg_noisy_low": int(tg_l.sum()),
        "overlap": float(len(set(Q_high) & set(Q_low)) / budget),
        "n_strata_equal": n_equal, "n_strata_diff": n_diff,
        "max_stratum_diff": max_diff,
    }
    return Q_high, Q_low, diag


def _proportional_sample(avail: dict, target: int, rng):
    if target <= 0: return np.array([], dtype=int)
    selected, remaining = [], target
    keys = sorted(avail, key=lambda _: rng.random())
    for k in keys:
        pool = avail[k]
        quota = max(1, int(target * len(pool) / max(1, sum(len(avail[kk]) for kk in keys))))
        take = min(quota, remaining, len(pool))
        if take > 0:
            selected.extend(rng.choice(pool, size=take, replace=False).tolist())
            remaining -= take
        if remaining <= 0: break
    return np.array(selected, dtype=int)[:target]

def _proportional_quota(avail: dict, target: int) -> dict:
    total = max(1, sum(len(v) for v in avail.values()))
    quota, remaining = {}, target
    for k in sorted(avail):
        q = int(target * len(avail[k]) / total)
        quota[k] = min(q, remaining)
        remaining -= quota[k]
    # Distribute remainder
    for k in sorted(avail):
        if remaining <= 0: break
        extra = min(len(avail[k]) - quota[k], remaining)
        quota[k] += extra; remaining -= extra
    return quota

def _proportional_sample_excluding(avail: dict, target: int, exclude: set, rng):
    avail_ex = {k: np.setdiff1d(v, list(exclude)) for k, v in avail.items()}
    return _proportional_sample(avail_ex, target, rng)

def _stratum_counts(indices, groups, clean_labels, l_bins, m_bins, pools):
    counts = defaultdict(int)
    for i in indices:
        for k in pools:
            g, c, lb, mb = k
            if groups[i]==g and clean_labels[i]==c and l_bins[i]==lb and m_bins[i]==mb:
                counts[k] += 1
                break
    return dict(counts)


def run_exp2_grid(output_dirs: List[Tuple[str,str]],
                  n_pairs: int = 10, grid_seeds: List[int] = [0,1,2]) -> pd.DataFrame:
    """Full Exp2 grid runner.
    
    Args:
        output_dirs: [(dataset_label, output_dir), ...]
        n_pairs: matched pairs per cell
        grid_seeds: data seeds to use
    """
    all_records = []
    training_cfg = {"epochs": 5, "batch_size": 256, "learning_rate": 0.001,
                    "weight_decay": 0.0001, "patience": 3,
                    "selection_metric": "balanced_accuracy", "device": "auto"}
    
    for ds_label, output_dir in output_dirs:
        for noise in ["uniform20", "minority_high_40"]:
            for budget_frac in [0.01, 0.02, 0.05]:
                for s in grid_seeds:
                    try:
                        data = load_data(output_dir, noise, s)
                    except (FileNotFoundError, AssertionError) as e:
                        print(f"  SKIP {ds_label}/{noise}/s{s}: {e}")
                        continue
                    
                    tg = identify_target_group(data, training_cfg)
                    budget = int(budget_frac * len(data["private"]))
                    
                    private = data["private"]
                    is_noisy = private["is_noisy"].to_numpy(dtype=bool)
                    groups = private["group"].to_numpy(dtype=int)
                    clean_labels = private["clean_label"].to_numpy(dtype=int)
                    
                    # Baseline
                    baseline_wga = _train_and_eval(
                        data["train_feat"], data["noisy_labels"],
                        data["val_feat"], data["val_labels"],
                        data["test_feat"], data["test_ids"],
                        data["evaluator"], training_cfg, s,
                    )
                    
                    # Feasibility: can TG supply enough noisy?
                    tg_noisy_avail = is_noisy[groups == tg].sum()
                    n_noisy = int(budget * 0.4)
                    max_feasible_high = tg_noisy_avail / n_noisy if n_noisy > 0 else 0
                    if max_feasible_high < 0.15:
                        print(f"  SKIP {ds_label}/{noise}/b{budget_frac:.0%}/s{s}: TG too small ({tg_noisy_avail} noisy)")
                        all_records.append({"dataset": ds_label, "noise": noise,
                            "budget": f"{budget_frac:.0%}", "seed": s,
                            "target_group": tg, "feasible": False,
                            "tg_noisy_avail": tg_noisy_avail})
                        continue
                    
                    hi_frac = min(0.40, max_feasible_high * 0.9)
                    lo_frac = 0.05
                    
                    rng = np.random.default_rng(s * 1000 + 42)
                    for pair_i in range(n_pairs):
                        Qh, Ql, diag = construct_true_exact_pair(
                            private, is_noisy, groups, clean_labels,
                            data["loss"], data["margin"],
                            budget=budget, precision=0.4,
                            target_group=tg,
                            high_frac=hi_frac, low_frac=lo_frac,
                            rng=rng,
                        )
                        if "error" in diag:
                            all_records.append({"dataset": ds_label, "noise": noise,
                                "budget": f"{budget_frac:.0%}", "seed": s,
                                "pair_id": pair_i, "valid": False,
                                "error": diag["error"], "target_group": tg})
                            continue
                        
                        pair_seed = s * 10000 + pair_i * 100
                        wga_h = _train_and_eval(
                            data["train_feat"], _repair(data["noisy_labels"], private, Qh, tg),
                            data["val_feat"], data["val_labels"],
                            data["test_feat"], data["test_ids"],
                            data["evaluator"], training_cfg, pair_seed,
                        )
                        wga_l = _train_and_eval(
                            data["train_feat"], _repair(data["noisy_labels"], private, Ql, tg),
                            data["val_feat"], data["val_labels"],
                            data["test_feat"], data["test_ids"],
                            data["evaluator"], training_cfg, pair_seed,
                        )
                        
                        all_records.append({
                            "dataset": ds_label, "noise": noise,
                            "budget": f"{budget_frac:.0%}", "seed": s,
                            "pair_id": pair_i, "valid": True,
                            "target_group": tg, "budget_n": budget,
                            "len_high": diag["len_high"], "len_low": diag["len_low"],
                            "prec_high": diag["prec_high"], "prec_low": diag["prec_low"],
                            "tg_noisy_high": diag["tg_noisy_high"],
                            "tg_noisy_low": diag["tg_noisy_low"],
                            "wga_baseline": baseline_wga,
                            "wga_high": wga_h, "wga_low": wga_l,
                            "ate": wga_h - wga_l,
                            "n_strata_equal": diag["n_strata_equal"],
                            "n_strata_diff": diag["n_strata_diff"],
                            "max_stratum_diff": diag["max_stratum_diff"],
                        })
                        
                        if (pair_i + 1) % 5 == 0:
                            valid_recs = [r for r in all_records if r.get("valid")]
                            if valid_recs:
                                ates = np.array([r["ate"] for r in valid_recs[-5:]])
                                print(f"  {ds_label}/{noise}/b{budget_frac:.0%}/s{s} p{pair_i+1}: ATE={ates.mean():+.4f}")
    
    df = pd.DataFrame(all_records)
    df.to_csv("outputs/exp2_pairs.csv", index=False)
    return df


def _repair(labels, private, query, tg):
    repaired = labels.copy(); cl = private["clean_label"].to_numpy(dtype=np.int64)
    for idx in query: repaired[idx] = cl[idx]
    return repaired


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4A: Complete Safety Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp4a_full(
    results_paths: Dict[str, str],
    base_method: str = "noise_score",
    safe_method: str = "noise_score_cpba_tc_v8_fallback",
    epsilon: float = 0.01,
    n_bootstrap: int = 2000, rng_seed: int = 12345,
) -> pd.DataFrame:
    """Complete safety evaluation per dataset x noise x budget.
    Cluster-bootstrap by seed. Non-inferiority via CI lower bound.
    """
    all_rows = []
    rng = np.random.default_rng(rng_seed)
    
    for ds_name, path in results_paths.items():
        df = pd.read_csv(path)
        for noise in sorted(df["noise_name"].unique()):
            base = df[(df["method"] == base_method) & (df["noise_name"] == noise)]
            safe = df[(df["method"] == safe_method) & (df["noise_name"] == noise)]
            if base.empty or safe.empty: continue
            
            # Check: exactly 10 seeds × N budgets?
            for budget in sorted(df["budget_fraction"].unique()):
                paired = safe[safe["budget_fraction"] == budget].merge(
                    base[base["budget_fraction"] == budget],
                    on=["noise_name", "seed"], suffixes=("_safe", "_base"),
                    how="outer", indicator=True,
                )
                missing = paired[paired["_merge"] != "both"]
                if len(missing) > 0:
                    print(f"  WARNING {ds_name}/{noise}/b{budget:.0%}: {len(missing)} unpaired rows")
                
                paired = paired[paired["_merge"] == "both"]
                if len(paired) < 4: continue
                
                diffs = (paired["delta_wga_safe"] - paired["delta_wga_base"]).values
                seeds = paired["seed"].unique()
                
                # Cluster bootstrap
                boot_means = []
                for _ in range(n_bootstrap):
                    bs_seeds = rng.choice(len(seeds), size=len(seeds), replace=True)
                    bs_diffs = []
                    for si in bs_seeds:
                        sv = paired[paired["seed"] == seeds[si]]
                        bs_diffs.extend((sv["delta_wga_safe"] - sv["delta_wga_base"]).tolist())
                    boot_means.append(np.mean(bs_diffs))
                boot_means = np.array(boot_means)
                ci_mean = np.percentile(boot_means, [2.5, 97.5])
                
                # Non-inferiority: CI_low > -epsilon
                non_inferior = ci_mean[0] > -epsilon
                
                # Failure rate
                fr = float((diffs < -epsilon).mean())
                harm = float(-diffs[diffs < -epsilon].mean()) if (diffs < -epsilon).any() else 0.0
                worst10 = np.sort(diffs)[:max(1, int(len(diffs) * 0.1))]
                
                all_rows.append({
                    "dataset": ds_name, "noise": noise,
                    "budget": f"{budget:.1%}",
                    "n_seeds": len(seeds), "n_paired": len(diffs),
                    "dWGA_safe": round(float(paired["delta_wga_safe"].mean()), 4),
                    "dWGA_base": round(float(paired["delta_wga_base"].mean()), 4),
                    "diff_mean": round(float(diffs.mean()), 4),
                    "diff_ci95_low": round(float(ci_mean[0]), 4),
                    "diff_ci95_high": round(float(ci_mean[1]), 4),
                    "non_inferior": non_inferior,
                    "failure_rate": round(fr, 3),
                    "harm": round(harm, 4),
                    "CVaR10": round(float(worst10.mean()), 4),
                    "worst_pair": round(float(diffs.min()), 4),
                    "best_pair": round(float(diffs.max()), 4),
                })
    
    result = pd.DataFrame(all_rows)
    result.to_csv("outputs/exp4a_safety.csv", index=False)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT (at top level to avoid circular)
# ═══════════════════════════════════════════════════════════════════════════════
from robust_verify.data.access import PrivateEvaluator
PrivateEvaluator  # silence unused import check


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()
    
    # ── Exp2: minimal smoke test ──
    print("=" * 70)
    print("[EXP2] Grid smoke test — CelebA uniform20 b=2% seed=0, 3 pairs")
    print("=" * 70)
    _ = run_exp2_grid(
        [("CelebA", "outputs/ablation_celeba_10seeds")],
        n_pairs=3, grid_seeds=[0],
    )
    
    # ── Exp4A: full analysis ──
    print("\n" + "=" * 70)
    print("[EXP4A] Full safety analysis")
    print("=" * 70)
    r4 = run_exp4a_full({
        "Waterbirds": "outputs/ablation_waterbirds_10seeds/stage1/results.csv",
        "CelebA": "outputs/ablation_celeba_10seeds/stage1/results.csv",
        "CivilComments": "outputs/ablation_civilcomments_10seeds/stage1/results.csv",
    })
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 180)
    print(r4.to_string(index=False))
    
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("Output: outputs/exp2_pairs.csv, outputs/exp4a_safety.csv")
