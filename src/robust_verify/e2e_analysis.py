"""Auditable summaries for end-to-end full-retraining experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from robust_verify.absolute_utility import bootstrap_mean_ci, exact_two_sided_sign_test, prepare_results

CONDITION_COLUMNS = ["dataset", "noise_name", "budget_fraction"]
REQUIRED_COLUMNS = {"dataset", "noise_name", "seed", "method", "budget_fraction", "delta_wga"}


def _summary(values: Sequence[float], *, rng: np.random.Generator, replicates: int) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("Cannot summarize an empty or non-vector effect array")
    low, high = bootstrap_mean_ci(array, rng, replicates)
    return {
        "n_paired_seeds": int(len(array)),
        "mean_effect": float(array.mean()),
        "median_effect": float(np.median(array)),
        "std_effect": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "positive_seed_count": int((array > 0).sum()),
        "negative_seed_count": int((array < 0).sum()),
        "tie_seed_count": int((array == 0).sum()),
        "sign_test_two_sided_p": exact_two_sided_sign_test(array),
    }


def load_end_to_end_results(
    paths: Iterable[str | Path], *, dataset_names: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load and validate runner CSVs, adding dataset names for legacy files."""
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("At least one end-to-end result CSV is required")
    names = list(dataset_names or [])
    if names and len(names) != len(resolved):
        raise ValueError("dataset_names must have one entry per input CSV")
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(resolved):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns:
            if not names:
                raise ValueError(f"{path} has no dataset column; pass dataset_names")
            frame["dataset"] = names[index]
        frame["dataset"] = frame["dataset"].astype(str)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS.difference(combined.columns)
    if missing:
        raise ValueError(f"End-to-end results are missing required columns: {sorted(missing)}")
    prepared = prepare_results(combined)
    if "wga_semantics" not in prepared.columns:
        prepared["wga_semantics"] = "worst_group_accuracy"
    return prepared


def summarize_end_to_end(
    results: pd.DataFrame,
    *,
    reference_method: str = "loss",
    methods: Sequence[str] | None = None,
    bootstrap_replicates: int = 10_000,
    rng_seed: int = 20260827,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Summarize absolute and seed-matched effects by condition."""
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    prepared = prepare_results(results)
    if "wga_semantics" not in prepared.columns:
        prepared["wga_semantics"] = "worst_group_accuracy"
    available = sorted(prepared["method"].astype(str).unique())
    if reference_method not in available:
        raise ValueError(f"Reference method {reference_method!r} is absent; available={available}")
    if methods is None:
        requested = [
            method
            for method in available
            if not method.startswith("oracle_") and not method.endswith("_oracle")
        ]
    else:
        requested = list(dict.fromkeys(methods))
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Requested methods are absent from results: {unknown}")
    candidates = [m for m in requested if m not in {reference_method, "no_correction"}]
    rng = np.random.default_rng(rng_seed)
    summary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for condition, group in prepared.groupby(CONDITION_COLUMNS, sort=True, dropna=False):
        reference = group[group["method"].astype(str).eq(reference_method)][["seed", "delta_wga"]]
        if reference.empty:
            continue
        reference = reference.rename(columns={"delta_wga": "reference_delta_wga"})
        for method in candidates:
            candidate = group[group["method"].astype(str).eq(method)][["seed", "delta_wga"]]
            joined = candidate.merge(reference, on="seed", how="inner", validate="one_to_one")
            if joined.empty:
                continue
            joined["paired_effect"] = joined["delta_wga"] - joined["reference_delta_wga"]
            values = joined["paired_effect"].to_numpy(float)
            absolute = candidate["delta_wga"].to_numpy(float)
            row = dict(zip(CONDITION_COLUMNS, condition, strict=True))
            row.update({"method": method, "reference_method": reference_method})
            row.update({f"paired_{k}": v for k, v in _summary(values, rng=rng, replicates=bootstrap_replicates).items()})
            row.update({f"absolute_{k}": v for k, v in _summary(absolute, rng=rng, replicates=bootstrap_replicates).items()})
            row["wga_semantics"] = str(group["wga_semantics"].iloc[0])
            summary_rows.append(row)
            for record in joined.itertuples(index=False):
                seed_rows.append({
                    **dict(zip(CONDITION_COLUMNS, condition, strict=True)),
                    "seed": int(record.seed), "method": method, "reference_method": reference_method,
                    "method_delta_wga": float(record.delta_wga),
                    "reference_delta_wga": float(record.reference_delta_wga),
                    "paired_effect": float(record.paired_effect),
                })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(CONDITION_COLUMNS + ["method"], kind="stable").reset_index(drop=True)
    per_seed = pd.DataFrame(seed_rows)
    if not per_seed.empty:
        per_seed = per_seed.sort_values(CONDITION_COLUMNS + ["method", "seed"], kind="stable").reset_index(drop=True)
    metadata = {
        "reference_method": reference_method, "methods": candidates,
        "datasets": sorted(prepared["dataset"].astype(str).unique().tolist()),
        "n_input_rows": int(len(prepared)), "bootstrap_replicates": int(bootstrap_replicates),
        "rng_seed": int(rng_seed), "condition_columns": CONDITION_COLUMNS,
        "paired_estimand": "delta_wga(method) - delta_wga(reference_method)",
    }
    return summary, per_seed, metadata


def write_end_to_end_summary(
    results: pd.DataFrame,
    output_dir: str | Path,
    *,
    reference_method: str = "loss",
    methods: Sequence[str] | None = None,
    bootstrap_replicates: int = 10_000,
    rng_seed: int = 20260827,
) -> dict[str, Path]:
    """Write normalized input, condition summary, seed summary and manifest."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    prepared = prepare_results(results)
    summary, per_seed, metadata = summarize_end_to_end(
        prepared, reference_method=reference_method, methods=methods,
        bootstrap_replicates=bootstrap_replicates, rng_seed=rng_seed,
    )
    paths = {
        "raw": target / "results_normalized.csv",
        "summary": target / "summary_paired.csv",
        "per_seed": target / "summary_per_seed.csv",
        "manifest": target / "summary_manifest.json",
    }
    prepared.drop(columns=[c for c in prepared.columns if c.startswith("_")], errors="ignore").to_csv(paths["raw"], index=False)
    summary.to_csv(paths["summary"], index=False)
    per_seed.to_csv(paths["per_seed"], index=False)
    paths["manifest"].write_text(json.dumps(metadata, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    return paths
