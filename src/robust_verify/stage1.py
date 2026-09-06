from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robust_verify.analysis import align_frame, query_composition, score_correlations
from robust_verify.config import ensure_output_layout, output_layout
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.features import load_features
from robust_verify.plotting import create_all_plots
from robust_verify.scoring import (
    ADAPTIVE_METHODS,
    METHOD_NAMES,
    build_adaptive_ranking,
    build_rankings,
)
from robust_verify.active_loop import run_multi_round
from robust_verify.model_baselines import run_model_baselines
from robust_verify.oof import fit_oof_probabilities
from robust_verify.training import (
    make_initial_state,
    predict_from_state,
    train_linear_head,
)
from robust_verify.utils import budget_to_count
from robust_verify.p0 import (
    write_artifact_manifest,
    write_or_validate_run_manifest,
    write_query_artifact,
)


def _append_csv_row(path: Path, row: dict, *, fixed_fields: list[str] | None = None) -> None:
    """Append a dict row to a CSV file, creating it with a header if new.

    If ``fixed_fields`` is given, the CSV always uses that column set
    (missing keys default to ""), guaranteeing a stable schema.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fields = fixed_fields if fixed_fields else sorted(row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


V9_METADATA_FIELDS = [
    "method",
    "ablation_name",
    "noise_name",
    "seed",
    "budget_fraction",
    "budget_count",
    "use_cpba",
    "use_tc",
    "fallback_policy",
    "alpha_threshold",
    "alpha",
    "mean_loss_class0",
    "mean_loss_class1",
    "class_loss_ratio",
    "high_risk_class",
    "gate_decision",
    "fallback_triggered",
    "weight_class0",
    "weight_class1",
    "wouldbe_weight_class0",
    "wouldbe_weight_class1",
    "tc_activated",
    "num_classes",
    "cluster_entropy_before",
    "cluster_entropy_after",
    "num_replacements",
    "alloc_class_0",
    "alloc_class_1",
]


def _normalize_v9_meta(raw: dict) -> dict:
    """Ensure all V9_METADATA_FIELDS keys exist (default to empty string)."""
    out = {k: "" for k in V9_METADATA_FIELDS}
    for k, v in raw.items():
        if k in out:
            out[k] = v
    return out


def _save_probe(
    path: Path,
    probe,
    train_sample_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "best_state": probe.best_state,
            "initial_state": probe.initial_state,
            "best_epoch": probe.best_epoch,
            "best_validation_metric": probe.best_validation_metric,
        },
        path,
    )
    np.savez_compressed(
        path.with_suffix(".dynamics.npz"),
        sample_ids=train_sample_ids,
        train_probabilities=probe.train_probabilities,
        loss_history=probe.loss_history,
        probability_history=probe.probability_history,
        prediction_history=probe.prediction_history,
        margin_history=probe.margin_history,
        correctness_history=probe.correctness_history,
        val_probability_history=probe.val_probability_history,
        best_epoch=np.asarray(probe.best_epoch),
    )


def _save_retrain_checkpoint(
    path: Path,
    retrained,
    *,
    method: str,
    noise_name: str,
    seed: int,
    budget_fraction: float,
    queried_sample_ids: np.ndarray,
) -> None:
    """Save the exact model state used for one verified-retraining result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "best_state": retrained.best_state,
            "initial_state": retrained.initial_state,
            "best_epoch": retrained.best_epoch,
            "best_validation_metric": retrained.best_validation_metric,
            "method": method,
            "noise_name": noise_name,
            "seed": int(seed),
            "budget_fraction": float(budget_fraction),
            "queried_sample_ids": np.asarray(queried_sample_ids, dtype=np.int64),
        },
        path,
    )


def _result_row(
    *,
    noise_name: str,
    noise_kind: str,
    seed: int,
    method: str,
    budget_fraction: float,
    budget_count: int,
    baseline_metrics: dict[str, float],
    composition: dict,
    metrics: dict[str, float],
    proxy_name: str,
    proxy_before: float,
    proxy_after: float,
    checkpoint_path: Path,
    query_artifact_path: Path,
    output_root: Path,
    selection_metadata: dict | None = None,
) -> dict:
    """Build one stable, self-describing result row."""
    row = {
        "noise_name": noise_name,
        "noise_kind": noise_kind,
        "seed": seed,
        "method": method,
        "budget_fraction": budget_fraction,
        "budget_count": budget_count,
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        **composition,
        **metrics,
        "proxy_name": proxy_name,
        "proxy_before": float(proxy_before),
        "proxy_after": float(proxy_after),
        "proxy_change": float(proxy_after) - float(proxy_before),
        "proxy_higher_is_better": True,
        "retrain_checkpoint": str(checkpoint_path.relative_to(output_root)).replace("\\", "/"),
        "query_artifact": str(query_artifact_path.relative_to(output_root)).replace("\\", "/"),
    }
    row["delta_average_accuracy"] = row["average_accuracy"] - row["baseline_average_accuracy"]
    row["delta_wga"] = row["wga"] - row["baseline_wga"]
    if selection_metadata and "selected_tau" in selection_metadata:
        row["selected_tau"] = float(selection_metadata["selected_tau"])
        row["tau_selection_folds"] = int(selection_metadata.get("num_folds", 0))
        row["tau_min_tail_count"] = int(selection_metadata.get("min_tail_count", 0))
    if selection_metadata and "gate_alpha" in selection_metadata:
        row["gate_alpha"] = float(selection_metadata["gate_alpha"])
        row["gate_reliability"] = float(selection_metadata.get("gate_reliability", 0.0))
        row["gate_stability_alpha"] = float(
            selection_metadata.get("gate_stability_alpha", 0.0)
        )
        row["gate_retention_alpha"] = float(
            selection_metadata.get("gate_retention_alpha", 0.0)
        )
        row["gate_noise_retention"] = float(
            selection_metadata.get("gate_noise_retention", 0.0)
        )
        row["gate_reference_budget_count"] = int(
            selection_metadata.get("gate_reference_budget_count", 0)
        )
        row["gate_fallback"] = bool(selection_metadata.get("gate_fallback", False))
        row["gate_num_folds"] = int(selection_metadata.get("gate_num_folds", 0))
    return row


def _repair_labels(
    public_frame: pd.DataFrame,
    private_manifest_path: Path,
    query_positions: np.ndarray,
) -> np.ndarray:
    labels = public_frame["noisy_label"].to_numpy(dtype=np.int64).copy()
    sample_ids = public_frame.iloc[query_positions]["sample_id"].to_numpy(dtype=np.int64)
    oracle = VerificationOracle(private_manifest_path)
    labels[query_positions] = oracle.verify(sample_ids)
    return labels


def run_stage1(config: dict) -> pd.DataFrame:
    layout = ensure_output_layout(config["project"]["output_dir"])
    source_root = Path(config["project"].get("input_dir", config["project"]["output_dir"]))
    source_layout = output_layout(source_root)
    required = [
        source_layout["features"] / "train.npz",
        source_layout["features"] / "val.npz",
        source_layout["features"] / "test.npz",
        source_layout["manifests"] / "val_public.csv",
        source_layout["manifests"] / "test_private.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 0 outputs are missing. Run rv-stage0 first.\n" + "\n".join(missing)
        )
    write_or_validate_run_manifest(layout["root"], config, source_root=source_root)

    train_features, train_ids = load_features(source_layout["features"] / "train.npz")
    val_features, val_ids = load_features(source_layout["features"] / "val.npz")
    test_features, test_ids = load_features(source_layout["features"] / "test.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features_gpu = torch.from_numpy(train_features).float().to(device)
    val_features_gpu = torch.from_numpy(val_features).float().to(device)
    test_features_gpu = torch.from_numpy(test_features).float().to(device)

    val_public = align_frame(
        pd.read_csv(source_layout["manifests"] / "val_public.csv"),
        val_ids,
    )
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    val_private = align_frame(
        pd.read_csv(source_layout["manifests"] / "val_private.csv"),
        val_ids,
    )
    val_groups = val_private["group"].to_numpy(dtype=np.int64)
    test_evaluator = PrivateEvaluator(source_layout["manifests"] / "test_private.csv")

    correlation_frames: list[pd.DataFrame] = []
    modern_baseline_cache: dict[tuple, object] = {}

    methods = list(config["experiment"]["methods"])
    unknown_methods = sorted(set(methods).difference(METHOD_NAMES))
    if unknown_methods:
        raise ValueError(f"Unknown method IDs in experiment config: {unknown_methods}")
    budgets = [float(value) for value in config["experiment"]["budgets"]]
    training_config = config["training"]

    # --- Checkpoint: load partial results and detect completed combos ---
    partial_path = layout["stage1"] / "results_partial.csv"
    results_path = layout["stage1"] / "results.csv"
    all_methods = set(methods)
    all_budgets = set(budgets)

    # Safety check: warn if existing results have MORE methods than current config
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing_methods = set(existing["method"].astype(str))
        if existing_methods - all_methods:
            missing = sorted(existing_methods - all_methods)
            print(f"[WARNING] results.csv has {len(existing_methods)} methods, "
                  f"but config only requests {len(all_methods)}. "
                  f"Missing from config: {missing}. "
                  f"Running with fewer methods will drop existing results on save. "
                  f"Backup results.csv or use full method list.")
        del existing_methods
    results: list[dict] = []
    completed_combos: set[tuple[str, int]] = set()
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        if not partial.empty:
            for (noise_name, seed), group in partial.groupby(["noise_name", "seed"]):
                seed_val = int(seed)
                noise_str = str(noise_name)
                present = set(zip(group["method"].astype(str), group["budget_fraction"]))
                expected = {(m, b) for m in all_methods for b in all_budgets}
                if present >= expected:
                    completed_combos.add((noise_str, seed_val))
                    results.extend(group.to_dict("records"))
                else:
                    print(f"[checkpoint] {noise_str} seed={seed_val}: incomplete ({len(present)}/{len(expected)} rows), will re-run")
            if completed_combos:
                print(f"[checkpoint] Resuming: {len(completed_combos)} combos complete, {len(results)} rows loaded")
    else:
        results = []

    # Pre-compute total work for progress display
    noise_settings = config["noise"]["settings"]
    seeds = config["experiment"]["seeds"]
    total_combos = len(noise_settings) * len(seeds)
    combo_idx = 0

    for noise_setting in noise_settings:
        noise_name = noise_setting["name"]

        for seed_value in config["experiment"]["seeds"]:
            seed = int(seed_value)
            if (noise_name, seed) in completed_combos:
                combo_idx += 1
                print(f"[checkpoint] [{combo_idx}/{total_combos}] Skipping {noise_name} seed={seed} (already done)")
                continue
            combo_idx += 1
            prefix = f"{noise_name}_seed{seed}"
            public_path = source_layout["manifests"] / f"{prefix}_train_public.csv"
            private_path = source_layout["manifests"] / f"{prefix}_train_private.csv"
            public = align_frame(pd.read_csv(public_path), train_ids)
            private = align_frame(pd.read_csv(private_path), train_ids)
            noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)

            input_dim = int(train_features.shape[1])
            num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
            initial_state = make_initial_state(input_dim, num_classes, seed)

            probe_path = layout["probes"] / f"{prefix}.pt"
            dynamics_path = layout["probes"] / f"{prefix}.dynamics.npz"
            if probe_path.exists() and dynamics_path.exists():
                print(f"[checkpoint] Loading probe {prefix}")
                ckpt = torch.load(probe_path, map_location="cpu")
                dyn = np.load(dynamics_path)
                from robust_verify.training import ProbeResult
                probe = ProbeResult(
                    best_state=ckpt["best_state"],
                    initial_state=ckpt["initial_state"],
                    best_epoch=int(ckpt["best_epoch"]),
                    best_validation_metric=float(ckpt["best_validation_metric"]),
                    train_probabilities=dyn["train_probabilities"],
                    loss_history=dyn["loss_history"],
                    probability_history=dyn["probability_history"],
                    prediction_history=dyn["prediction_history"],
                    margin_history=dyn["margin_history"],
                    correctness_history=dyn["correctness_history"],
                    val_probability_history=(
                        dyn["val_probability_history"]
                        if "val_probability_history" in dyn.files
                        else None
                    ),
                )
            else:
                probe = train_linear_head(
                    train_features=train_features_gpu,
                    train_labels=noisy_labels,
                    val_features=val_features_gpu,
                    val_labels=val_labels,
                    config=training_config,
                    seed=seed,
                    initial_state=initial_state,
                    record_dynamics=True,
                )
                _save_probe(
                    probe_path,
                    probe,
                    train_ids,
                )

            if (
                "tracin_multicheckpoint_val" in methods
                and probe.val_probability_history is None
            ):
                print(
                    f"[checkpoint] Rebuilding {prefix}: cached probe lacks "
                    "validation checkpoint history"
                )
                probe = train_linear_head(
                    train_features=train_features_gpu,
                    train_labels=noisy_labels,
                    val_features=val_features_gpu,
                    val_labels=val_labels,
                    config=training_config,
                    seed=seed,
                    initial_state=initial_state,
                    record_dynamics=True,
                )
                _save_probe(probe_path, probe, train_ids)

            baseline_predictions, _ = predict_from_state(
                probe.best_state,
                test_features_gpu,
                input_dim,
                num_classes,
                training_config,
            )
            baseline_metrics = test_evaluator.evaluate(test_ids, baseline_predictions)

            if probe.correctness_history is None:
                raise RuntimeError("Probe dynamics were not recorded.")

            _, val_probs = predict_from_state(
                probe.best_state,
                val_features_gpu,
                input_dim,
                num_classes,
                training_config,
            )
            context = {
                "train_features": train_features,
                "val_features": val_features,
                "val_probs": val_probs,
                "val_labels": val_labels,
                "val_groups": val_groups,
                "margin_history": probe.margin_history,
                "probability_history": probe.probability_history,
                "val_probability_history": probe.val_probability_history,
                "max_budget_count": max(
                    budget_to_count(value, len(train_ids)) for value in budgets
                ),
                "baseline_options": config.get("experiment", {}).get(
                    "baseline_options", {}
                ),
                "modern_baseline_cache": modern_baseline_cache,
                "selection_metadata": {},
            }

            if "oof_cleanlab" in methods:
                oof_path = layout["probes"] / f"{prefix}.oof_train_probs.npz"
                if oof_path.exists():
                    cached_oof = np.load(oof_path)
                    oof_ids = cached_oof["sample_ids"] if "sample_ids" in cached_oof.files else None
                    if oof_ids is None or not np.array_equal(oof_ids, train_ids):
                        raise ValueError(f"OOF cache sample IDs do not match {prefix}")
                    oof_train_probs = cached_oof["probabilities"]
                else:
                    oof_train_probs = fit_oof_probabilities(
                        train_features,
                        noisy_labels,
                        seed=seed,
                        folds=int(config.get("experiment", {}).get("baseline_options", {}).get("oof_folds", 5)),
                        options=config.get("experiment", {}).get("baseline_options", {}),
                    )
                    np.savez_compressed(
                        oof_path,
                        sample_ids=train_ids,
                        probabilities=oof_train_probs,
                    )
                context["oof_train_probs"] = oof_train_probs

            rankings, scores = build_rankings(
                methods=methods,
                probabilities=probe.train_probabilities,
                labels=noisy_labels,
                correctness_history=probe.correctness_history,
                private_frame=private,
                seed=seed,
                context=context,
            )

            corr = score_correlations(scores)
            corr["noise_name"] = noise_name
            corr["seed"] = seed
            correlation_frames.append(corr)

            run_query_dir = layout["queries"] / prefix
            run_query_dir.mkdir(parents=True, exist_ok=True)

            for method, ranking in rankings.items():
                np.save(run_query_dir / f"{method}_ranking.npy", train_ids[ranking])

                for budget_fraction in budgets:
                    budget_count = budget_to_count(budget_fraction, len(train_ids))
                    query_positions = ranking[:budget_count]
                    queried_sample_ids = train_ids[query_positions]
                    np.save(
                        run_query_dir / f"{method}_budget_{budget_fraction:.4f}.npy",
                        queried_sample_ids,
                    )

                    composition = query_composition(query_positions, private)
                    repaired_labels = _repair_labels(public, private_path, query_positions)

                    retrained = train_linear_head(
                        train_features=train_features_gpu,
                        train_labels=repaired_labels,
                        val_features=val_features_gpu,
                        val_labels=val_labels,
                        config=training_config,
                        seed=seed,
                        initial_state=copy.deepcopy(initial_state),
                        record_dynamics=False,
                    )
                    predictions, _ = predict_from_state(
                        retrained.best_state,
                        test_features_gpu,
                        input_dim,
                        num_classes,
                        training_config,
                    )
                    metrics = test_evaluator.evaluate(test_ids, predictions)
                    checkpoint_path = (
                        layout["checkpoints"] / prefix / f"{method}_budget_{budget_fraction:.4f}.pt"
                    )
                    _save_retrain_checkpoint(
                        checkpoint_path,
                        retrained,
                        method=method,
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=budget_fraction,
                        queried_sample_ids=queried_sample_ids,
                    )
                    query_artifact_path = run_query_dir / f"{method}_budget_{budget_fraction:.4f}.json"
                    write_query_artifact(
                        query_artifact_path,
                        method=method,
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=budget_fraction,
                        sample_ids=queried_sample_ids,
                        proxy_name=str(training_config.get("selection_metric", "balanced_accuracy")),
                        proxy_before=probe.best_validation_metric,
                        proxy_after=retrained.best_validation_metric,
                        checkpoint_path=checkpoint_path.relative_to(layout["root"]),
                    )
                    row = _result_row(
                        noise_name=noise_name,
                        noise_kind=noise_setting["kind"],
                        seed=seed,
                        method=method,
                        budget_fraction=budget_fraction,
                        budget_count=budget_count,
                        baseline_metrics=baseline_metrics,
                        composition=composition,
                        metrics=metrics,
                        proxy_name=str(training_config.get("selection_metric", "balanced_accuracy")),
                        proxy_before=probe.best_validation_metric,
                        proxy_after=retrained.best_validation_metric,
                        checkpoint_path=checkpoint_path,
                        query_artifact_path=query_artifact_path,
                        output_root=layout["root"],
                        selection_metadata=context.get("selection_metadata", {}).get(method),
                    )
                    results.append(row)
                    pd.DataFrame(results).to_csv(partial_path, index=False)
                    print(f"[{combo_idx}/{total_combos} {noise_name} s={seed}] {method:30s} b={budget_fraction:.4f} WGA={metrics['wga']:.4f} dWGA={row['delta_wga']:+.4f}")
                    sys.stdout.flush()

            # --- Adaptive methods: ranking depends on budget ---
            cluster_path = config.get("experiment", {}).get("cluster_path")
            adaptive_methods_in_use = sorted(set(methods).intersection(ADAPTIVE_METHODS))
            for method in adaptive_methods_in_use:
                n_total = len(train_ids)
                for budget_fraction in budgets:
                    budget_count = budget_to_count(budget_fraction, n_total)
                    # Build budget-adaptive ranking
                    ranking = build_adaptive_ranking(
                        method=method,
                        noise_score=scores["noise_score"],
                        probabilities=probe.train_probabilities,
                        private_frame=private,
                        budget_count=budget_count,
                        n_total=n_total,
                        losses=scores.get("loss"),
                        cluster_path=cluster_path,
                        labels=noisy_labels,
                        repair_value_scores=scores.get(method),
                        options=config.get("experiment", {}).get("baseline_options", {}),
                    )
                    # Save v9 metadata if available
                    v9_meta = getattr(private, "_v9_meta", None)
                    if v9_meta is not None:
                        v9_meta["noise_name"] = noise_name
                        v9_meta["seed"] = seed
                        v9_meta["budget_fraction"] = budget_fraction
                        v9_meta["budget_count"] = budget_count
                        meta_path = layout["stage1"] / "v9_metadata.csv"
                        _append_csv_row(meta_path, _normalize_v9_meta(v9_meta),
                                        fixed_fields=V9_METADATA_FIELDS)
                        del private._v9_meta
                    query_positions = ranking[:budget_count]
                    queried_sample_ids = train_ids[query_positions]
                    np.save(
                        run_query_dir / f"{method}_budget_{budget_fraction:.4f}.npy",
                        queried_sample_ids,
                    )

                    composition = query_composition(query_positions, private)
                    repaired_labels = _repair_labels(public, private_path, query_positions)

                    retrained = train_linear_head(
                        train_features=train_features_gpu,
                        train_labels=repaired_labels,
                        val_features=val_features_gpu,
                        val_labels=val_labels,
                        config=training_config,
                        seed=seed,
                        initial_state=copy.deepcopy(initial_state),
                        record_dynamics=False,
                    )
                    predictions, _ = predict_from_state(
                        retrained.best_state,
                        test_features_gpu,
                        input_dim,
                        num_classes,
                        training_config,
                    )
                    metrics = test_evaluator.evaluate(test_ids, predictions)
                    checkpoint_path = (
                        layout["checkpoints"] / prefix / f"{method}_budget_{budget_fraction:.4f}.pt"
                    )
                    _save_retrain_checkpoint(
                        checkpoint_path,
                        retrained,
                        method=method,
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=budget_fraction,
                        queried_sample_ids=queried_sample_ids,
                    )
                    query_artifact_path = run_query_dir / f"{method}_budget_{budget_fraction:.4f}.json"
                    write_query_artifact(
                        query_artifact_path,
                        method=method,
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=budget_fraction,
                        sample_ids=queried_sample_ids,
                        proxy_name=str(training_config.get("selection_metric", "balanced_accuracy")),
                        proxy_before=probe.best_validation_metric,
                        proxy_after=retrained.best_validation_metric,
                        checkpoint_path=checkpoint_path.relative_to(layout["root"]),
                    )
                    row = _result_row(
                        noise_name=noise_name,
                        noise_kind=noise_setting["kind"],
                        seed=seed,
                        method=method,
                        budget_fraction=budget_fraction,
                        budget_count=budget_count,
                        baseline_metrics=baseline_metrics,
                        composition=composition,
                        metrics=metrics,
                        proxy_name=str(training_config.get("selection_metric", "balanced_accuracy")),
                        proxy_before=probe.best_validation_metric,
                        proxy_after=retrained.best_validation_metric,
                        checkpoint_path=checkpoint_path,
                        query_artifact_path=query_artifact_path,
                        output_root=layout["root"],
                        selection_metadata=context.get("selection_metadata", {}).get(method),
                    )
                    results.append(row)
                    pd.DataFrame(results).to_csv(partial_path, index=False)
                    print(f"[{combo_idx}/{total_combos} {noise_name} s={seed}] {method:30s} b={budget_fraction:.4f} WGA={metrics['wga']:.4f} dWGA={row['delta_wga']:+.4f}")
                    sys.stdout.flush()

            # --- v9: multi-round active querying ---
            if "noise_score_multi_round" in methods:
                n_total = len(train_ids)
                for budget_fraction in budgets:
                    budget_count = budget_to_count(budget_fraction, n_total)
                    mr = run_multi_round(
                        train_features=train_features_gpu,
                        train_ids=train_ids,
                        noisy_labels=noisy_labels,
                        val_features=val_features_gpu,
                        val_labels=val_labels,
                        public_frame=public,
                        private_manifest_path=private_path,
                        initial_state=initial_state,
                        training_config=training_config,
                        seed=seed,
                        budget_count=budget_count,
                        num_rounds=3,
                        input_dim=input_dim,
                        num_classes=num_classes,
                    )
                    query_positions = mr.query_positions
                    queried_sample_ids = train_ids[query_positions]
                    np.save(
                        run_query_dir / f"noise_score_multi_round_budget_{budget_fraction:.4f}.npy",
                        queried_sample_ids,
                    )

                    composition = query_composition(query_positions, private)
                    repaired_labels = mr.final_labels

                    retrained = train_linear_head(
                        train_features=train_features_gpu,
                        train_labels=repaired_labels,
                        val_features=val_features_gpu,
                        val_labels=val_labels,
                        config=training_config,
                        seed=seed,
                        initial_state=copy.deepcopy(initial_state),
                        record_dynamics=False,
                    )
                    predictions, _ = predict_from_state(
                        retrained.best_state,
                        test_features_gpu,
                        input_dim,
                        num_classes,
                        training_config,
                    )
                    metrics = test_evaluator.evaluate(test_ids, predictions)
                    checkpoint_path = (
                        layout["checkpoints"] / prefix
                        / f"noise_score_multi_round_budget_{budget_fraction:.4f}.pt"
                    )
                    _save_retrain_checkpoint(
                        checkpoint_path,
                        retrained,
                        method="noise_score_multi_round",
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=budget_fraction,
                        queried_sample_ids=queried_sample_ids,
                    )
                    query_artifact_path = (
                        run_query_dir / f"noise_score_multi_round_budget_{budget_fraction:.4f}.json"
                    )
                    write_query_artifact(
                        query_artifact_path,
                        method="noise_score_multi_round",
                        noise_name=noise_name,
                        seed=seed,
                        budget_fraction=budget_fraction,
                        sample_ids=queried_sample_ids,
                        proxy_name=str(training_config.get("selection_metric", "balanced_accuracy")),
                        proxy_before=probe.best_validation_metric,
                        proxy_after=retrained.best_validation_metric,
                        checkpoint_path=checkpoint_path.relative_to(layout["root"]),
                    )
                    row = _result_row(
                        noise_name=noise_name,
                        noise_kind=noise_setting["kind"],
                        seed=seed,
                        method="noise_score_multi_round",
                        budget_fraction=budget_fraction,
                        budget_count=budget_count,
                        baseline_metrics=baseline_metrics,
                        composition=composition,
                        metrics=metrics,
                        proxy_name=str(training_config.get("selection_metric", "balanced_accuracy")),
                        proxy_before=probe.best_validation_metric,
                        proxy_after=retrained.best_validation_metric,
                        checkpoint_path=checkpoint_path,
                        query_artifact_path=query_artifact_path,
                        output_root=layout["root"],
                    )
                    results.append(row)
                    pd.DataFrame(results).to_csv(partial_path, index=False)
                    print(f"[{combo_idx}/{total_combos} {noise_name} s={seed}] noise_score_multi_round b={budget_fraction:.4f} WGA={metrics['wga']:.4f} dWGA={row['delta_wga']:+.4f}")
                    sys.stdout.flush()

    result_frame = pd.DataFrame(results)
    result_path = layout["stage1"] / "results.csv"
    result_frame.to_csv(result_path, index=False)

    # Clean up checkpoint file after successful completion
    if partial_path.exists():
        partial_path.unlink()

    if correlation_frames:
        correlations = pd.concat(correlation_frames, ignore_index=True)
        correlations.to_csv(layout["stage1"] / "score_correlations.csv", index=False)

    create_all_plots(result_frame, layout["plots"])

    # Model-training baselines (ERM, oracle GroupDRO, JTT) — separate table
    run_model_baselines(
        config=config,
        layout=layout,
        source_layout=source_layout,
        train_features=train_features_gpu,
        train_ids=train_ids,
        val_features=val_features_gpu,
        val_labels=val_labels,
        val_groups=val_groups,
        test_features=test_features_gpu,
        test_ids=test_ids,
        test_evaluator=test_evaluator,
    )
    write_artifact_manifest(layout["root"])
    return result_frame
