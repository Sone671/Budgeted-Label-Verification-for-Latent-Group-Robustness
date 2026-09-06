#!/usr/bin/env python
"""Validate budget-as-an-upper-cap stopping on cached Stage 1 queries.

Every candidate is trained and selected using only the public clean validation
split.  Test features and the private evaluator are not opened until all
stopping decisions have been frozen.  The script is intentionally independent
from the Stage 1 runner and never changes its caches.
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame
from robust_verify.config import load_config
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.features import load_features
from robust_verify.rivet import spectral_class_cvar
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count, class_balanced_accuracy


DEFAULT_CONFIG = ROOT / "configs" / "ablation_waterbirds_10seeds.yaml"
RESULTS_NAME = "results_per_budget.csv"
SELECTED_NAME = "selected_results.csv"

RESULT_FIELDS = [
    "source_output",
    "noise_name",
    "seed",
    "method",
    "budget_fraction",
    "budget_cap_count",
    "actual_query_count",
    "candidate_kind",
    "val_balanced_accuracy",
    "spectral_class_cvar",
    "alphas",
    "temperature",
    "query_path",
]

SELECTED_FIELDS = [
    "source_output",
    "noise_name",
    "seed",
    "method",
    "selection_rule",
    "selection_role",
    "selected_budget_fraction",
    "selected_query_count",
    "selected_val_balanced_accuracy",
    "selected_spectral_class_cvar",
    "wga",
    "average_accuracy",
    "balanced_accuracy",
    "baseline_wga",
    "delta_wga_vs_baseline",
    "fixed_cap_budget_fraction",
    "fixed_cap_wga",
    "delta_wga_vs_fixed_cap",
    "alphas",
    "temperature",
]


def _parse_csv(raw: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def _alpha_key(alphas: Iterable[float]) -> str:
    return ",".join(f"{float(alpha):g}" for alpha in alphas)


def _resolve_from_root(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _read_csv(path: Path, fields: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=fields)
    frame = pd.read_csv(path)
    for field in fields:
        if field not in frame:
            frame[field] = np.nan
    return frame[fields]


def _atomic_write_csv(frame: pd.DataFrame, path: Path, fields: list[str]) -> None:
    """Replace a CSV only after its complete successor is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.reindex(columns=fields).to_csv(
        temporary,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    temporary.replace(path)


def _row_key(row: pd.Series | dict[str, Any], *, selected: bool) -> tuple[Any, ...]:
    common = (
        str(row["source_output"]),
        str(row["noise_name"]),
        int(row["seed"]),
        str(row["method"]),
        str(row["alphas"]),
        float(row["temperature"]),
    )
    if selected:
        return common + (str(row["selection_rule"]),)
    return common + (round(float(row["budget_fraction"]), 12),)


def _upsert(
    frame: pd.DataFrame,
    row: dict[str, Any],
    *,
    path: Path,
    fields: list[str],
    selected: bool,
) -> pd.DataFrame:
    key = _row_key(row, selected=selected)
    if len(frame):
        keep = [
            _row_key(existing, selected=selected) != key
            for _, existing in frame.iterrows()
        ]
        frame = frame.loc[keep]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    sort_fields = ["source_output", "noise_name", "seed", "method"]
    sort_fields += ["selection_rule" if selected else "budget_fraction"]
    frame = frame.sort_values(sort_fields, kind="stable").reset_index(drop=True)
    _atomic_write_csv(frame, path, fields)
    return frame


def query_ids_to_positions(query_ids: np.ndarray, train_ids: np.ndarray) -> np.ndarray:
    """Map cached sample IDs to feature rows with strict cache validation."""

    query_ids = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    train_ids = np.asarray(train_ids, dtype=np.int64).reshape(-1)
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("query cache contains duplicate sample IDs")
    lookup = {int(sample_id): position for position, sample_id in enumerate(train_ids)}
    missing = [int(sample_id) for sample_id in query_ids if int(sample_id) not in lookup]
    if missing:
        raise KeyError(f"query cache contains unknown sample IDs: {missing[:5]}")
    return np.asarray([lookup[int(sample_id)] for sample_id in query_ids], dtype=np.int64)


def select_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Freeze the three predeclared policies; ties favor fewer queries."""

    if not rows:
        raise ValueError("at least one candidate is required")
    baseline = [row for row in rows if float(row["budget_fraction"]) == 0.0]
    if len(baseline) != 1:
        raise ValueError("exactly one no-query baseline candidate is required")
    by_ba = min(
        rows,
        key=lambda row: (
            -float(row["val_balanced_accuracy"]),
            float(row["budget_fraction"]),
        ),
    )
    by_risk = min(
        rows,
        key=lambda row: (
            float(row["spectral_class_cvar"]),
            float(row["budget_fraction"]),
        ),
    )
    return {
        "max_val_balanced_accuracy": by_ba,
        "min_spectral_risk": by_risk,
        "baseline_no_query": baseline[0],
    }


def _candidate_key(
    source_name: str,
    noise: str,
    seed: int,
    method: str,
    budget: float,
    alphas: tuple[float, ...],
    temperature: float,
) -> tuple[Any, ...]:
    return (
        source_name,
        noise,
        seed,
        method,
        _alpha_key(alphas),
        float(temperature),
        round(float(budget), 12),
    )


def _checkpoint_path(
    output_root: Path,
    source_name: str,
    noise: str,
    seed: int,
    method: str,
    budget: float,
) -> Path:
    safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
    return (
        output_root
        / "candidate_states"
        / source_name
        / f"{noise}_seed{seed}"
        / f"{safe_method}_budget_{budget:.4f}.pt"
    )


def _validation_metrics(
    state: dict[str, torch.Tensor],
    val_features: np.ndarray,
    val_labels: np.ndarray,
    input_dim: int,
    num_classes: int,
    training_config: dict[str, Any],
    alphas: tuple[float, ...],
    temperature: float,
) -> tuple[float, float]:
    predictions, probabilities = predict_from_state(
        state,
        val_features,
        input_dim,
        num_classes,
        training_config,
    )
    balanced_accuracy = class_balanced_accuracy(val_labels, predictions)
    risk, _, _, _, _ = spectral_class_cvar(
        probabilities,
        val_labels,
        alphas=alphas,
        temperature=temperature,
    )
    return balanced_accuracy, risk


def _load_probe(source_root: Path, prefix: str) -> dict[str, Any]:
    path = source_root / "stage1" / "probes" / f"{prefix}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def _fit_candidate(
    *,
    train_features: np.ndarray,
    noisy_labels: np.ndarray,
    repaired_positions: np.ndarray,
    repaired_values: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    training_config: dict[str, Any],
    seed: int,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    repaired_labels = noisy_labels.copy()
    repaired_labels[repaired_positions] = repaired_values
    result = train_linear_head(
        train_features=train_features,
        train_labels=repaired_labels,
        val_features=val_features,
        val_labels=val_labels,
        config=training_config,
        seed=seed,
        initial_state=copy.deepcopy(initial_state),
        record_dynamics=False,
    )
    return result.best_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group-free validation stopping with cached query-budget caps."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--noise", default="uniform20")
    parser.add_argument("--method", default="noise_score_budget_hybrid_v8")
    parser.add_argument("--budgets", default=None, help="Comma-separated budget caps.")
    parser.add_argument("--alphas", default="0.01,0.02,0.05")
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--output-root", default="outputs/safe_cap_pilot")
    args = parser.parse_args()

    config_path = _resolve_from_root(args.config)
    config = load_config(config_path)
    source_root = _resolve_from_root(config["project"]["output_dir"])
    output_root = _resolve_from_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = (
        _parse_csv(args.seeds, int)
        if args.seeds is not None
        else tuple(int(seed) for seed in config["experiment"]["seeds"])
    )
    budgets = (
        _parse_csv(args.budgets, float)
        if args.budgets is not None
        else tuple(float(value) for value in config["experiment"]["budgets"])
    )
    budgets = tuple(sorted(set(budgets)))
    alphas = tuple(float(value) for value in _parse_csv(args.alphas, float))
    if any(budget <= 0.0 or budget > 1.0 for budget in budgets):
        raise ValueError("all candidate budgets must lie in (0, 1]")
    if any(alpha <= 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("all alphas must lie in (0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")

    noise_names = _parse_csv(args.noise, str)
    configured_noise = {
        str(setting["name"]) for setting in config["noise"]["settings"]
    }
    unknown_noise = sorted(set(noise_names) - configured_noise)
    if unknown_noise:
        raise ValueError(f"noise settings absent from config: {unknown_noise}")

    results_path = output_root / RESULTS_NAME
    selected_path = output_root / SELECTED_NAME
    results = _read_csv(results_path, RESULT_FIELDS)
    selected_results = _read_csv(selected_path, SELECTED_FIELDS)
    completed_candidates = {
        _row_key(row, selected=False) for _, row in results.iterrows()
    }

    train_features, train_ids = load_features(source_root / "features" / "train.npz")
    val_features, val_ids = load_features(source_root / "features" / "val.npz")
    val_public = align_frame(
        pd.read_csv(source_root / "manifests" / "val_public.csv"), val_ids
    )
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    training_config = dict(config["training"])
    input_dim = int(train_features.shape[1])
    source_name = source_root.name
    alpha_key = _alpha_key(alphas)

    # Phase 1: compute every validation-only candidate and checkpoint each row.
    for noise in noise_names:
        for seed in seeds:
            prefix = f"{noise}_seed{seed}"
            public_path = source_root / "manifests" / f"{prefix}_train_public.csv"
            private_path = source_root / "manifests" / f"{prefix}_train_private.csv"
            public = align_frame(pd.read_csv(public_path), train_ids)
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
            num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
            checkpoint = _load_probe(source_root, prefix)

            baseline_key = _candidate_key(
                source_name,
                noise,
                seed,
                args.method,
                0.0,
                alphas,
                args.temperature,
            )
            if baseline_key not in completed_candidates:
                val_ba, risk = _validation_metrics(
                    checkpoint["best_state"],
                    val_features,
                    val_labels,
                    input_dim,
                    num_classes,
                    training_config,
                    alphas,
                    args.temperature,
                )
                row = {
                    "source_output": source_name,
                    "noise_name": noise,
                    "seed": seed,
                    "method": args.method,
                    "budget_fraction": 0.0,
                    "budget_cap_count": 0,
                    "actual_query_count": 0,
                    "candidate_kind": "baseline_no_query",
                    "val_balanced_accuracy": val_ba,
                    "spectral_class_cvar": risk,
                    "alphas": alpha_key,
                    "temperature": args.temperature,
                    "query_path": "",
                }
                results = _upsert(
                    results,
                    row,
                    path=results_path,
                    fields=RESULT_FIELDS,
                    selected=False,
                )
                completed_candidates.add(baseline_key)
                print(f"[val] {prefix} baseline BA={val_ba:.4f} risk={risk:.4f}")

            oracle = VerificationOracle(private_path)
            for budget in budgets:
                candidate_key = _candidate_key(
                    source_name,
                    noise,
                    seed,
                    args.method,
                    budget,
                    alphas,
                    args.temperature,
                )
                if candidate_key in completed_candidates:
                    print(f"[resume] {prefix} {args.method} b={budget:.4f}")
                    continue
                query_path = (
                    source_root
                    / "stage1"
                    / "queries"
                    / prefix
                    / f"{args.method}_budget_{budget:.4f}.npy"
                )
                query_ids = np.load(query_path, allow_pickle=False)
                positions = query_ids_to_positions(query_ids, train_ids)
                cap_count = budget_to_count(budget, len(train_ids))
                if len(positions) > cap_count:
                    raise ValueError(
                        f"{query_path} contains {len(positions)} IDs, above cap {cap_count}"
                    )
                repaired_values = oracle.verify(np.asarray(query_ids, dtype=np.int64))
                state = _fit_candidate(
                    train_features=train_features,
                    noisy_labels=noisy_labels,
                    repaired_positions=positions,
                    repaired_values=repaired_values,
                    val_features=val_features,
                    val_labels=val_labels,
                    training_config=training_config,
                    seed=seed,
                    initial_state=checkpoint["initial_state"],
                )
                state_path = _checkpoint_path(
                    output_root,
                    source_name,
                    noise,
                    seed,
                    args.method,
                    budget,
                )
                state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"best_state": state}, state_path)
                val_ba, risk = _validation_metrics(
                    state,
                    val_features,
                    val_labels,
                    input_dim,
                    num_classes,
                    training_config,
                    alphas,
                    args.temperature,
                )
                row = {
                    "source_output": source_name,
                    "noise_name": noise,
                    "seed": seed,
                    "method": args.method,
                    "budget_fraction": budget,
                    "budget_cap_count": cap_count,
                    "actual_query_count": len(positions),
                    "candidate_kind": "queried_cap",
                    "val_balanced_accuracy": val_ba,
                    "spectral_class_cvar": risk,
                    "alphas": alpha_key,
                    "temperature": args.temperature,
                    "query_path": str(query_path),
                }
                results = _upsert(
                    results,
                    row,
                    path=results_path,
                    fields=RESULT_FIELDS,
                    selected=False,
                )
                completed_candidates.add(candidate_key)
                print(
                    f"[val] {prefix} b={budget:.4f} q={len(positions)} "
                    f"BA={val_ba:.4f} risk={risk:.4f}"
                )

    # Freeze all choices before test features or private test data are opened.
    frozen: list[dict[str, Any]] = []
    for noise in noise_names:
        for seed in seeds:
            mask = (
                (results["source_output"].astype(str) == source_name)
                & (results["noise_name"].astype(str) == noise)
                & (results["seed"].astype(int) == seed)
                & (results["method"].astype(str) == args.method)
                & (results["alphas"].astype(str) == alpha_key)
                & np.isclose(results["temperature"].astype(float), args.temperature)
                & results["budget_fraction"].astype(float).isin((0.0, *budgets))
            )
            candidate_rows = results.loc[mask].to_dict("records")
            expected_budgets = {0.0, *budgets}
            observed_budgets = {
                round(float(row["budget_fraction"]), 12) for row in candidate_rows
            }
            if observed_budgets != {round(value, 12) for value in expected_budgets}:
                raise RuntimeError(
                    f"incomplete candidates for {noise} seed={seed}: {observed_budgets}"
                )
            rules = select_candidates(candidate_rows)
            fixed_cap = max(
                candidate_rows, key=lambda row: float(row["budget_fraction"])
            )
            frozen.append(
                {
                    "noise": noise,
                    "seed": seed,
                    "rules": rules,
                    "fixed_cap": fixed_cap,
                }
            )
            print(
                f"[frozen] {noise}_seed{seed} "
                + " ".join(
                    f"{rule}=b{float(row['budget_fraction']):g}"
                    for rule, row in rules.items()
                )
            )

    # Phase 2: only now is private test evaluation permitted.
    expected_selected_keys = {
        (
            source_name,
            item["noise"],
            item["seed"],
            args.method,
            alpha_key,
            float(args.temperature),
            rule,
        )
        for item in frozen
        for rule in (*item["rules"].keys(), "fixed_cap")
    }
    completed_selected = {
        _row_key(row, selected=True) for _, row in selected_results.iterrows()
    }
    if expected_selected_keys.issubset(completed_selected):
        print(f"[done] {selected_path} already contains all requested selections")
        return

    test_features, test_ids = load_features(source_root / "features" / "test.npz")
    evaluator = PrivateEvaluator(source_root / "manifests" / "test_private.csv")
    for item in frozen:
        noise = item["noise"]
        seed = int(item["seed"])
        prefix = f"{noise}_seed{seed}"
        public = align_frame(
            pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"),
            train_ids,
        )
        noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
        num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
        probe = _load_probe(source_root, prefix)
        oracle = VerificationOracle(
            source_root / "manifests" / f"{prefix}_train_private.csv"
        )

        candidate_rows = dict(item["rules"])
        candidate_rows["fixed_cap"] = item["fixed_cap"]
        unique_budgets = {
            float(row["budget_fraction"]) for row in candidate_rows.values()
        }
        test_metrics: dict[float, dict[str, float]] = {}
        for budget in sorted(unique_budgets):
            if budget == 0.0:
                state = probe["best_state"]
            else:
                state_path = _checkpoint_path(
                    output_root,
                    source_name,
                    noise,
                    seed,
                    args.method,
                    budget,
                )
                if state_path.exists():
                    state = torch.load(
                        state_path, map_location="cpu", weights_only=True
                    )["best_state"]
                else:
                    query_path = (
                        source_root
                        / "stage1"
                        / "queries"
                        / prefix
                        / f"{args.method}_budget_{budget:.4f}.npy"
                    )
                    query_ids = np.load(query_path, allow_pickle=False)
                    positions = query_ids_to_positions(query_ids, train_ids)
                    state = _fit_candidate(
                        train_features=train_features,
                        noisy_labels=noisy_labels,
                        repaired_positions=positions,
                        repaired_values=oracle.verify(query_ids),
                        val_features=val_features,
                        val_labels=val_labels,
                        training_config=training_config,
                        seed=seed,
                        initial_state=probe["initial_state"],
                    )
            predictions, _ = predict_from_state(
                state,
                test_features,
                input_dim,
                num_classes,
                training_config,
            )
            test_metrics[budget] = evaluator.evaluate(test_ids, predictions)

        baseline_wga = test_metrics[0.0]["wga"]
        fixed_budget = float(item["fixed_cap"]["budget_fraction"])
        fixed_wga = test_metrics[fixed_budget]["wga"]
        for rule, candidate in candidate_rows.items():
            selected_key = (
                source_name,
                noise,
                seed,
                args.method,
                alpha_key,
                float(args.temperature),
                rule,
            )
            if selected_key in completed_selected:
                continue
            budget = float(candidate["budget_fraction"])
            metrics = test_metrics[budget]
            row = {
                "source_output": source_name,
                "noise_name": noise,
                "seed": seed,
                "method": args.method,
                "selection_rule": rule,
                "selection_role": (
                    "fixed_cap_comparator" if rule == "fixed_cap" else "safe_stop"
                ),
                "selected_budget_fraction": budget,
                "selected_query_count": int(candidate["actual_query_count"]),
                "selected_val_balanced_accuracy": float(
                    candidate["val_balanced_accuracy"]
                ),
                "selected_spectral_class_cvar": float(
                    candidate["spectral_class_cvar"]
                ),
                "wga": metrics["wga"],
                "average_accuracy": metrics["average_accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "baseline_wga": baseline_wga,
                "delta_wga_vs_baseline": metrics["wga"] - baseline_wga,
                "fixed_cap_budget_fraction": fixed_budget,
                "fixed_cap_wga": fixed_wga,
                "delta_wga_vs_fixed_cap": metrics["wga"] - fixed_wga,
                "alphas": alpha_key,
                "temperature": args.temperature,
            }
            selected_results = _upsert(
                selected_results,
                row,
                path=selected_path,
                fields=SELECTED_FIELDS,
                selected=True,
            )
            completed_selected.add(selected_key)
            print(
                f"[test] {prefix} {rule} b={budget:g} "
                f"WGA={metrics['wga']:.4f} vs_cap={row['delta_wga_vs_fixed_cap']:+.4f}"
            )


if __name__ == "__main__":
    main()
