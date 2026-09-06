#!/usr/bin/env python
"""Historical P0-4 design note for the ten-seed matched-pair intervention.

Design specification for upgrading the existing 3-seed matching intervention
(described in exp2_causal_intervention.py) to 10 independent data seeds.

Key design elements:
  - 10 independent data seeds (noise generation + initial state)
  - Per seed: 5-10 matched query pairs (Q_high, Q_low)
  - Budget: 1% and 2% (feasible; 5% structurally infeasible for minority noise)
  - Matching variables (fixed per pair):
      * oracle noise precision
      * clean-label sample composition
      * (class, loss_bin, margin_bin) count vector
  - Treatment: target-group noise sample ratio
      * Q_high: ~40% target-group noise samples
      * Q_low: ~5% target-group noise samples
  - Evaluated with: paired seed-level bootstrap, randomization test
  - Report: ATE, 95% CI, per-dataset per-noise breakdown

Dose-response extension:
  - Instead of binary (high/low), use 5 dose levels: 5%, 15%, 25%, 35%, 45%
  - Test for monotonic relationship between target-group coverage and WGA

Runnable implementation:
  python server/scripts/matched_pair_intervention.py --dry-run
  python server/scripts/matched_pair_intervention.py

The server runner supersedes the old prototype in
``scripts/exp2_causal_intervention.py``.  It enforces exact matching on clean
label/loss-bin/margin-bin counts, saves query IDs and checkpoints, supports
resume, and reports seed-level bootstrap/randomization statistics.

Output structure:
  outputs/server/matched_pair_10seed/
    waterbirds_minority_high_40/
      budget_0.01/
        seed_0/
          pair_0_high_query.npy, pair_0_low_query.npy
          pair_0_high_wga.csv, pair_0_low_wga.csv
          ...
    results.csv  (pooled across seeds)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS = ["waterbirds", "celeba", "civilcomments"]
NOISES = ["minority_high_40", "uniform20"]
BUDGETS = [0.01, 0.02]  # 1% and 2% (5% infeasible for matching)
N_DATA_SEEDS = 10       # independent data seeds (up from 3)
N_PAIRS_PER_SEED = 8    # matched pairs per seed (up from 10 in 3-seed pilot)
DOSE_LEVELS = [0.05, 0.15, 0.25, 0.35, 0.45]  # for dose-response extension

# Matching tolerance
NOISE_PRECISION_TOL = 0.02   # oracle noise precision difference < 2pp
CLEAN_COUNT_TOL = 1          # clean sample count difference ≤ 1
BIN_COUNT_TOL = 2            # per-bin count difference ≤ 2

N_LOSS_BINS = 5
N_MARGIN_BINS = 5


def build_matching_spec() -> dict:
    """Return the experiment specification as a dict for documentation."""
    return {
        "design": "matched-pair causal intervention",
        "independence_unit": "data_seed",
        "n_data_seeds": N_DATA_SEEDS,
        "n_pairs_per_seed": N_PAIRS_PER_SEED,
        "total_pairs": N_DATA_SEEDS * N_PAIRS_PER_SEED,
        "budgets": BUDGETS,
        "noises": NOISES,
        "datasets": DATASETS,
        "treatment": "target_group_noise_sample_ratio",
        "treatment_levels": {"high": 0.40, "low": 0.05},
        "matching_covariates": [
            "oracle_noise_precision",
            "clean_sample_count",
            "(class × loss_bin × margin_bin) count vector",
        ],
        "evaluation": [
            "ATE: mean(WGA_high - WGA_low)",
            "95% CI: seed-level bootstrap (n=10)",
            "Randomization test: p-value for ATE=0",
            "Per-dataset per-noise breakdown",
            "Dose-response: Spearman rho across 5 dose levels",
        ],
        "upgrade_from_pilot": {
            "n_seeds": "3 → 10 (enables meaningful statistical inference)",
            "n_pairs_per_seed": "10 → 8 (balanced design)",
            "analysis_unit": "method-means → seed-level means",
            "dose_response": "binary → 5-level continuous (optional)",
        },
    }


def print_spec():
    """Print the experimental specification for documentation."""
    import json
    spec = build_matching_spec()
    print(json.dumps(spec, indent=2, ensure_ascii=False))


def create_output_structure(base_dir: str = "outputs/p0_matching_10seed"):
    """Create output directory structure."""
    base = Path(base_dir)
    for ds in DATASETS:
        for noise in NOISES:
            for budget in BUDGETS:
                for seed in range(N_DATA_SEEDS):
                    path = base / ds / noise / f"budget_{budget:.3f}" / f"seed_{seed}"
                    path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created output structure under {base}")


if __name__ == "__main__":
    print_spec()
    print()
    print("Runnable command: python server/scripts/matched_pair_intervention.py --dry-run")
