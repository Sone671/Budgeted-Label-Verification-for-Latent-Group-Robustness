#!/usr/bin/env python
"""Generate unified baseline comparison tables for items 1-3.

Produces:
  - table_unified_baselines_3datasets.csv : 3-dataset comparison, method groups
  - table_30epoch_sensitivity_10seed.csv  : 5-epoch vs 30-epoch, 10 seeds
  - table_baseline_groups.csv             : Query baselines by category

Reads from existing results CSVs when available; otherwise creates
template structure for post-execution fill-in.

Usage:  python scripts/generate_unified_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ───────────────────────────────────────────────────────────────

NOISES = ["uniform20", "minority_high_40"]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]
DATASETS = ["Waterbirds", "CelebA", "CivilComments"]
EPSILON = 0.005

METHOD_LABELS = {
    "random": "Random",
    "loss": "Loss",
    "entropy": "Entropy",
    "stratified_random": "StratifiedRandom",
    "forgetting": "Forgetting",
    "margin": "Margin",
    "aum": "AUM",
    "confident_learning": "Cleanlab",
    "nn_agreement": "NN-Agree",
    "nn_label_spreading": "NN-LabelSpread",
    "coreset": "Coreset-Lite",
    "uncertainty_diversity": "Unc+Div-Lite",
    "badge_lite": "BADGE-Lite",
    "tracin_val_uncertainty": "TracIn(val)",
    "tracin_noise_weighted": "TracIn(noise)",
    "tracin_wga_oracle": "TracIn(oracle)",
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "noise_score_cpba_only": "CPBA-only",
    "noise_score_tc_only": "TC-only",
    "noise_score_cpba_tc_no_fallback": "CPBA-TC-noFB",
    "noise_score_cpba_tc_v8_fallback": "CPBA-TC-Guarded",
    "noise_score_cpba_tc_entropy_fallback": "CPBA-TC-entFB",
    "oracle_noise_group_balanced": "Oracle-GB",
    "oracle_minority": "Oracle-minority",
}

# Method grouping for organized table display
METHOD_GROUPS = {
    "Basic queries": [
        "random", "loss", "entropy", "stratified_random",
    ],
    "Noise detection": [
        "noise_score", "confident_learning", "margin",
    ],
    "Training dynamics": [
        "forgetting", "aum",
    ],
    "Diversity + coverage": [
        "coreset", "uncertainty_diversity", "badge_lite",
    ],
    "Feature-space agreement": [
        "nn_agreement", "nn_label_spreading",
    ],
    "Our methods": [
        "noise_score_budget_hybrid_v8", "noise_score_cpba_only",
        "noise_score_tc_only", "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
    ],
    "Oracle diagnostic": [
        "oracle_noise_group_balanced", "oracle_minority",
    ],
}

MODEL_BASELINE_LABELS = {
    "erm": "ERM (noisy labels)",
    "oracle_groupdro": "Oracle GroupDRO",
    "jtt": "JTT",
}


def safe_load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def macro_mean(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), sem(values)


# ── TABLE 1: Unified baselines across 3 datasets ────────────────────────────

def build_unified_baselines_table(
    ablation_paths: dict[str, Path],
    baseline_paths: dict[str, Path],
    output_path: Path,
) -> pd.DataFrame:
    """Build unified baseline comparison across 3 datasets, pooled over budgets + noises."""
    rows = []
    all_methods = [
        "random", "loss", "entropy", "stratified_random",
        "forgetting", "margin", "aum", "confident_learning",
        "nn_agreement", "nn_label_spreading",
        "coreset", "uncertainty_diversity", "badge_lite",
        "noise_score", "noise_score_budget_hybrid_v8",
        "noise_score_cpba_only", "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
        "oracle_noise_group_balanced",
    ]

    for method in all_methods:
        lbl = METHOD_LABELS.get(method, method)
        row = {"method": method, "label": lbl}
        per_ds_vals = []
        for ds in DATASETS:
            vals = []
            for src_path in [ablation_paths.get(ds), baseline_paths.get(ds)]:
                if src_path is None:
                    continue
                df = safe_load(src_path)
                if df is None or df.empty:
                    continue
                sub = df[df["method"] == method]
                if sub.empty:
                    continue
                vals.extend(sub["delta_wga"].to_numpy(float).tolist())

            if vals:
                arr = np.array(vals)
                row[f"{ds}_mean"] = round(float(arr.mean()), 6)
                row[f"{ds}_sem"] = round(sem(arr), 6)
                row[f"{ds}_n"] = len(arr)
                row[f"{ds}_neg_pct"] = round(float((arr < 0).mean()) * 100, 1)
                per_ds_vals.extend(arr)
            else:
                row[f"{ds}_mean"] = None
                row[f"{ds}_sem"] = None
                row[f"{ds}_n"] = 0
                row[f"{ds}_neg_pct"] = None

        if per_ds_vals:
            all_arr = np.array(per_ds_vals)
            row["macro_mean"] = round(float(np.mean(per_ds_vals)), 6)
            row["macro_sem"] = round(sem(np.array(per_ds_vals)), 6)
        else:
            row["macro_mean"] = None
            row["macro_sem"] = None

        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")
    return result


# ── TABLE 2: 30-epoch sensitivity comparison ────────────────────────────────

def build_30epoch_sensitivity_table(
    epoch5_paths: dict[str, Path],
    epoch30_paths: dict[str, Path],
    output_path: Path,
) -> pd.DataFrame:
    """Compare 5-epoch vs 30-epoch results across datasets."""
    methods = [
        "noise_score", "noise_score_budget_hybrid_v8",
        "noise_score_cpba_only", "noise_score_cpba_tc_no_fallback",
        "noise_score_cpba_tc_v8_fallback",
    ]

    rows = []
    for ds in DATASETS:
        ep5_df = safe_load(epoch5_paths.get(ds)) if ds in epoch5_paths else None
        ep30_df = safe_load(epoch30_paths.get(ds)) if ds in epoch30_paths else None
        if ep5_df is None or ep30_df is None:
            continue

        for method in methods:
            lbl = METHOD_LABELS.get(method, method)

            v5 = ep5_df[ep5_df["method"] == method]["delta_wga"].to_numpy(float)
            v30 = ep30_df[ep30_df["method"] == method]["delta_wga"].to_numpy(float)

            if len(v5) < 2 or len(v30) < 2:
                continue

            # Merge on seed+budget for paired comparison
            merged = ep5_df[ep5_df["method"] == method].merge(
                ep30_df[ep30_df["method"] == method],
                on=["seed", "budget_fraction", "noise_name"],
                suffixes=("_5", "_30"),
            )
            if len(merged) < 2:
                # Fallback: report unpaired summary
                diff_mean = float(v30.mean() - v5.mean())
                diff_ci_low = diff_mean
                diff_ci_high = diff_mean
                n_seeds = min(len(v5), len(v30))
            else:
                diff = (
                    merged["delta_wga_30"].to_numpy(float)
                    - merged["delta_wga_5"].to_numpy(float)
                )
                diff_mean = float(np.mean(diff))
                n_seeds = int(merged["seed"].nunique())
                if n_seeds >= 2:
                    se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                    t_val = stats.t.ppf(0.975, len(diff) - 1)
                    diff_ci_low = round(diff_mean - t_val * se_diff, 6)
                    diff_ci_high = round(diff_mean + t_val * se_diff, 6)
                else:
                    diff_ci_low = diff_mean
                    diff_ci_high = diff_mean

            rows.append({
                "dataset": ds,
                "method": method,
                "label": lbl,
                "mean_5epoch": round(float(v5.mean()), 6),
                "sem_5epoch": round(sem(v5), 6),
                "mean_30epoch": round(float(v30.mean()), 6),
                "sem_30epoch": round(sem(v30), 6),
                "diff_30vs5": round(diff_mean, 6),
                "diff_ci_low": round(diff_ci_low, 6),
                "diff_ci_high": round(diff_ci_high, 6),
                "n_seeds": n_seeds,
                "n_conditions_5": len(v5),
                "n_conditions_30": len(v30),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")
    return result


# ── TABLE 3: Baselines by category (organized display) ──────────────────────

def build_baseline_category_table(
    ablation_paths: dict[str, Path],
    baseline_paths: dict[str, Path],
    output_path: Path,
) -> pd.DataFrame:
    """Build category-organized baseline comparison with per-dataset breakdown."""
    rows = []
    for group_name, group_methods in METHOD_GROUPS.items():
        for method in group_methods:
            lbl = METHOD_LABELS.get(method, method)
            row = {"group": group_name, "method": method, "label": lbl}
            for ds in DATASETS:
                vals = []
                for src_path in [ablation_paths.get(ds), baseline_paths.get(ds)]:
                    if src_path is None:
                        continue
                    df = safe_load(src_path)
                    if df is None or df.empty:
                        continue
                    sub = df[df["method"] == method]
                    if sub.empty:
                        continue
                    vals.extend(sub["delta_wga"].to_numpy(float).tolist())
                if vals:
                    arr = np.array(vals)
                    row[f"{ds}_dwga"] = f"{np.mean(arr):+.4f}"
                    row[f"{ds}_neg_pct"] = f"{(arr<0).mean()*100:.0f}%"
                else:
                    row[f"{ds}_dwga"] = "N/A"
                    row[f"{ds}_neg_pct"] = "N/A"
            rows.append(row)
        # Blank separator row between groups
        rows.append(dict.fromkeys(rows[0].keys(), "") if rows else {})

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")
    return result


# ── TABLE 4: Model training baselines ───────────────────────────────────────

def build_model_baselines_table(
    baseline_paths: dict[str, Path],
    output_path: Path,
) -> pd.DataFrame:
    """Extract model training baselines (ERM, GroupDRO, JTT) from model_baselines.csv."""
    rows = []
    for method in ["erm", "oracle_groupdro", "jtt"]:
        lbl = MODEL_BASELINE_LABELS.get(method, method)
        row = {"method": method, "label": lbl}
        for ds in DATASETS:
            src_path = baseline_paths.get(ds)
            if src_path is None:
                row[f"{ds}_wga"] = None
                row[f"{ds}_sem"] = None
                continue
            mb_path = src_path.parent / "model_baselines.csv" if src_path else None
            if mb_path is None or not (mb_path).exists():
                # Try alternate locations
                alt_path = Path(str(src_path).replace("stage1/results.csv", "stage1/model_baselines.csv"))
                if not alt_path.exists():
                    row[f"{ds}_wga"] = None
                    continue
                mb_path = alt_path
            mb = safe_load(mb_path)
            if mb is None or mb.empty:
                row[f"{ds}_wga"] = None
                continue
            sub = mb[mb["method"] == method]
            if sub.empty:
                row[f"{ds}_wga"] = None
                continue
            vals = sub["wga"].to_numpy(float)
            row[f"{ds}_wga"] = round(float(vals.mean()), 4)
            row[f"{ds}_sem"] = round(sem(vals), 4)
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output_path, index=False)
        print(f"[OK] {output_path} ({len(result)} rows)")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    paper_tables = Path("paper/tables")
    paper_tables.mkdir(parents=True, exist_ok=True)

    # Paths to existing results
    ablation_paths = {
        "Waterbirds": Path("outputs/ablation_waterbirds_10seeds/stage1/results.csv"),
        "CelebA": Path("outputs/ablation_celeba_10seeds/stage1/results.csv"),
        "CivilComments": Path("outputs/ablation_civilcomments_10seeds/stage1/results.csv"),
    }

    baseline_paths = {
        "Waterbirds": Path("outputs/baselines_waterbirds_recommended/stage1/results.csv"),
        "CelebA": Path("outputs/baselines_celeba_recommended/stage1/results.csv"),
        "CivilComments": Path("outputs/baselines_civilcomments_10seeds/stage1/results.csv"),
    }

    epoch30_paths = {
        "Waterbirds": Path("outputs/p0_30epoch_waterbirds_10seed/stage1/results.csv"),
        "CelebA": Path("outputs/p0_30epoch_celeba_10seed/stage1/results.csv"),
        "CivilComments": Path("outputs/p0_30epoch_civilcomments_10seed/stage1/results.csv"),
    }

    # Generate tables
    build_unified_baselines_table(
        ablation_paths, baseline_paths,
        paper_tables / "table_unified_baselines_3datasets.csv",
    )

    build_30epoch_sensitivity_table(
        ablation_paths, epoch30_paths,
        paper_tables / "table_30epoch_sensitivity_10seed.csv",
    )

    build_baseline_category_table(
        ablation_paths, baseline_paths,
        paper_tables / "table_baseline_groups.csv",
    )

    build_model_baselines_table(
        baseline_paths,
        paper_tables / "table_model_baselines_3datasets.csv",
    )


if __name__ == "__main__":
    main()
