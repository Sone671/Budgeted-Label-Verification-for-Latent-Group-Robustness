#!/usr/bin/env python
"""Produce the P0 seed-level confirmatory statistics from raw results.csv.

Budgets are retained as condition labels: this script never treats multiple
budgets from the same seed as independent observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["dataset", "noise_name", "seed", "method", "budget_fraction"]
REQUIRED_COLUMNS = {"noise_name", "seed", "method", "budget_fraction", "delta_wga"}


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-a with ties omitted, avoiding an optional scipy dependency."""
    concordant = discordant = 0
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            dx = np.sign(x[left] - x[right])
            dy = np.sign(y[left] - y[right])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _read_results(paths: Iterable[Path], dataset_names: list[str] | None) -> pd.DataFrame:
    frames = []
    for index, path in enumerate(paths):
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        dataset = (
            dataset_names[index]
            if dataset_names is not None
            else (frame["dataset"].iloc[0] if "dataset" in frame.columns else path.parent.parent.name)
        )
        frame["dataset"] = dataset
        frame["_query_root"] = str(path.parent / "queries")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    duplicate = result.duplicated(KEY_COLUMNS, keep=False)
    if duplicate.any():
        examples = result.loc[duplicate, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"Duplicate result keys; refusing to pool them: {examples}")
    return result


def _summary(results: pd.DataFrame, *, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    group_columns = ["dataset", "noise_name", "budget_fraction", "method"]
    for key, group in results.groupby(group_columns, sort=True):
        values = group["delta_wga"].to_numpy(dtype=float)
        low, high = _bootstrap_ci(values, rng, n_bootstrap)
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            {
                "n_seeds": len(values),
                "mean_delta_wga": float(values.mean()),
                "paired_bootstrap_ci_low": low,
                "paired_bootstrap_ci_high": high,
                "negative_seed_rate": float((values < 0).mean()),
                "worst_seed_delta_wga": float(values.min()),
                "mean_noise_precision": float(group["noise_precision"].mean())
                if "noise_precision" in group
                else float("nan"),
                "mean_delta_average_accuracy": float(group["delta_average_accuracy"].mean())
                if "delta_average_accuracy" in group
                else float("nan"),
            }
        )
        for column in sorted(
            col for col in group if col.startswith("group_") and col.endswith("_accuracy")
        ):
            baseline_column = f"baseline_{column}"
            if baseline_column in group:
                row[f"mean_delta_{column}"] = float(
                    (group[column] - group[baseline_column]).mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_comparisons(
    results: pd.DataFrame,
    comparator: str,
    *,
    label: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    condition_columns = ["dataset", "noise_name", "budget_fraction"]
    for condition, group in results.groupby(condition_columns, sort=True):
        reference = group[group["method"] == comparator][["seed", "delta_wga"]]
        if reference.empty:
            continue
        for method, candidate in group.groupby("method", sort=True):
            if method == comparator:
                continue
            joined = candidate[["seed", "delta_wga"]].merge(
                reference, on="seed", suffixes=("_method", "_reference"), validate="one_to_one"
            )
            values = (joined["delta_wga_method"] - joined["delta_wga_reference"]).to_numpy(float)
            low, high = _bootstrap_ci(values, rng, n_bootstrap)
            row = dict(zip(condition_columns, condition, strict=True))
            row.update(
                {
                    "method": method,
                    "comparator": comparator,
                    "comparison": label,
                    "n_paired_seeds": len(values),
                    "mean_paired_delta_wga": float(values.mean()),
                    "paired_bootstrap_ci_low": low,
                    "paired_bootstrap_ci_high": high,
                    "method_wins": int((values > 0).sum()),
                    "comparator_wins": int((values < 0).sum()),
                    "ties": int((values == 0).sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _rank_and_reversal_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows = []
    reversal_rows = []
    condition_columns = ["dataset", "noise_name", "budget_fraction", "seed"]
    for condition, group in results.groupby(condition_columns, sort=True):
        if "noise_precision" in group and len(group) >= 2:
            rank_rows.append(
                {
                    **dict(zip(condition_columns, condition, strict=True)),
                    "proxy": "noise_precision",
                    "n_methods": len(group),
                    "spearman": float(group["noise_precision"].rank().corr(group["delta_wga"].rank())),
                    "kendall": _kendall_tau(
                        group["noise_precision"].to_numpy(float), group["delta_wga"].to_numpy(float)
                    ),
                }
            )
        if "proxy_change" in group:
            eligible = group.dropna(subset=["proxy_change", "delta_wga"])
            if len(eligible):
                reversal_rows.append(
                    {
                        **dict(zip(condition_columns, condition, strict=True)),
                        "n_methods": len(eligible),
                        "proxy_improved_wga_declined_count": int(
                            ((eligible["proxy_change"] > 0) & (eligible["delta_wga"] < 0)).sum()
                        ),
                        "proxy_improved_wga_declined_rate": float(
                            ((eligible["proxy_change"] > 0) & (eligible["delta_wga"] < 0)).mean()
                        ),
                    }
                )
    return pd.DataFrame(rank_rows), pd.DataFrame(reversal_rows)


def _seed_regret(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["dataset", "noise_name", "budget_fraction", "seed"]
    for condition, group in results.groupby(group_columns, sort=True):
        legal = group[~group["method"].str.startswith("oracle_")]
        if legal.empty:
            continue
        best = float(legal["delta_wga"].max())
        for _, record in group.iterrows():
            rows.append(
                {
                    **dict(zip(group_columns, condition, strict=True)),
                    "method": record["method"],
                    "delta_wga": record["delta_wga"],
                    "best_legal_delta_wga": best,
                    "regret_to_best_legal": best - float(record["delta_wga"]),
                }
            )
    return pd.DataFrame(rows)


def _query_overlap(results: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    for _, record in results.iterrows():
        if record["method"] == reference:
            continue
        root = Path(record["_query_root"])
        prefix = f"{record['noise_name']}_seed{int(record['seed'])}"
        filename = f"{record['method']}_budget_{float(record['budget_fraction']):.4f}.npy"
        ref_file = root / prefix / f"{reference}_budget_{float(record['budget_fraction']):.4f}.npy"
        path = root / prefix / filename
        row = {key: record[key] for key in KEY_COLUMNS}
        row["reference_method"] = reference
        if not path.exists() or not ref_file.exists():
            row.update({"query_artifacts_available": False, "query_jaccard": float("nan")})
        else:
            selected = set(np.load(path, allow_pickle=False).astype(int).tolist())
            comparator = set(np.load(ref_file, allow_pickle=False).astype(int).tolist())
            row.update(
                {
                    "query_artifacts_available": True,
                    "query_jaccard": len(selected & comparator) / max(len(selected | comparator), 1),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 seed-level statistics and integrity tables.")
    parser.add_argument("--results", action="append", required=True, help="Path to a raw stage1/results.csv; repeatable.")
    parser.add_argument("--dataset", action="append", help="Dataset name matching each --results path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-method", default="noise_score")
    parser.add_argument("--oracle-method", default="oracle_noise_group_balanced")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(value).resolve() for value in args.results]
    if args.dataset is not None and len(args.dataset) != len(paths):
        raise ValueError("Provide exactly one --dataset value per --results path")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = _read_results(paths, args.dataset)
    rng = np.random.default_rng(args.rng_seed)

    summary = _summary(results, n_bootstrap=args.bootstrap, rng=rng)
    versus_reference = _paired_comparisons(
        results, args.reference_method, label="versus_reference", n_bootstrap=args.bootstrap, rng=rng
    )
    versus_oracle = _paired_comparisons(
        results, args.oracle_method, label="versus_oracle", n_bootstrap=args.bootstrap, rng=rng
    )
    correlations, reversals = _rank_and_reversal_metrics(results)
    regret = _seed_regret(results)
    overlap = _query_overlap(results, args.reference_method)
    outputs = {
        "per_seed_raw.csv": results.drop(columns=["_query_root"]),
        "summary.csv": summary,
        "paired_vs_reference.csv": versus_reference,
        "paired_vs_oracle.csv": versus_oracle,
        "rank_correlations.csv": correlations,
        "direction_reversals.csv": reversals,
        "seed_regret.csv": regret,
        "query_overlap.csv": overlap,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False)
    report = {
        "n_raw_rows": len(results),
        "datasets": sorted(results["dataset"].unique().tolist()),
        "reference_method": args.reference_method,
        "oracle_method": args.oracle_method,
        "bootstrap_replicates": args.bootstrap,
        "proxy_reversal_rows": len(reversals),
        "query_overlap_available_rate": float(overlap["query_artifacts_available"].mean()) if len(overlap) else 0.0,
    }
    (out_dir / "statistics_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote P0 statistics to {out_dir}")


if __name__ == "__main__":
    main()
