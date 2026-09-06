#!/usr/bin/env python
"""Pilot group-free hierarchical spectral-tail acquisition on cached features.

This is a development/triage driver, not a paper-result generator.  It keeps
the paper's probe, initialization, validation labels, retraining protocol and
private evaluation boundary fixed.  Selection never reads private group data.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.analysis import align_frame, query_composition
from robust_verify.config import load_config
from robust_verify.data.access import PrivateEvaluator
from robust_verify.features import load_features
from robust_verify.rivet import (
    class_prior_adjusted_noise_score,
    spectral_binary_rivet_scores,
    spectral_class_cvar,
)
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _load_probe(root: Path, prefix: str):
    checkpoint = torch.load(
        root / "stage1" / "probes" / f"{prefix}.pt",
        map_location="cpu",
        weights_only=True,
    )
    dynamics = np.load(root / "stage1" / "probes" / f"{prefix}.dynamics.npz")
    return checkpoint, dynamics


def _evaluate_state(
    state: dict[str, torch.Tensor],
    *,
    features: np.ndarray,
    sample_ids: np.ndarray,
    evaluator: PrivateEvaluator,
    input_dim: int,
    training_config: dict,
) -> tuple[dict[str, float], np.ndarray]:
    predictions, probabilities = predict_from_state(
        state, features, input_dim, 2, training_config
    )
    return evaluator.evaluate(sample_ids, predictions), probabilities


def _gate_selection(
    prior_order: np.ndarray,
    prior_score: np.ndarray,
    leverage: np.ndarray,
    count: int,
    multiplier: float,
    *,
    positive_only: bool,
) -> np.ndarray:
    candidate_count = min(len(prior_order), int(np.ceil(multiplier * count)))
    candidates = prior_order[:candidate_count]
    local_order = descending_ranking(leverage[candidates], prior_score[candidates])
    ordered = candidates[local_order]
    if positive_only:
        ordered = ordered[leverage[ordered] > 0.0]
    return ordered[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", default="uniform20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", default="0.02")
    parser.add_argument("--alphas", default="0.01,0.02,0.05")
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--candidate-multipliers", default="1.25,1.5")
    parser.add_argument("--include-positive-cap", action="store_true")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--output-root", default="outputs/spectral_gate_pilot")
    args = parser.parse_args()

    alphas = _parse_floats(args.alphas)
    budgets = _parse_floats(args.budgets)
    multipliers = _parse_floats(args.candidate_multipliers)
    if any(multiplier < 1.0 for multiplier in multipliers):
        raise ValueError("candidate multipliers must be at least one")

    config = load_config(args.config)
    source_root = Path(config["project"]["output_dir"])
    prefix = f"{args.noise}_seed{args.seed}"
    run_name = (
        f"{source_root.name}_{prefix}_a{'-'.join(f'{a:g}' for a in alphas)}_"
        f"t{args.temperature:g}_d{args.damping:g}"
    )
    output_root = ROOT / args.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    train_features, train_ids = load_features(source_root / "features" / "train.npz")
    val_features, val_ids = load_features(source_root / "features" / "val.npz")
    test_features, test_ids = load_features(source_root / "features" / "test.npz")
    public = align_frame(
        pd.read_csv(source_root / "manifests" / f"{prefix}_train_public.csv"), train_ids
    )
    private = align_frame(
        pd.read_csv(source_root / "manifests" / f"{prefix}_train_private.csv"), train_ids
    )
    val_public = align_frame(
        pd.read_csv(source_root / "manifests" / "val_public.csv"), val_ids
    )
    noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    clean_labels = private["clean_label"].to_numpy(dtype=np.int64)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)

    checkpoint, dynamics = _load_probe(source_root, prefix)
    train_probabilities = dynamics["train_probabilities"]
    correctness_history = dynamics["correctness_history"]
    training_config = dict(config["training"])
    input_dim = int(train_features.shape[1])
    _, val_probabilities = predict_from_state(
        checkpoint["best_state"], val_features, input_dim, 2, training_config
    )
    legal_scores = build_legal_scores(
        train_probabilities, noisy_labels, correctness_history
    )
    cpba_score, cpba_metadata = class_prior_adjusted_noise_score(
        legal_scores["noise_score"], legal_scores["loss"], noisy_labels
    )
    prior_order = descending_ranking(cpba_score, legal_scores["loss"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spectral = spectral_binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities,
        noisy_labels=noisy_labels,
        validation_features=val_features,
        validation_probabilities=val_probabilities,
        validation_labels=val_labels,
        noise_score=cpba_score,
        alphas=alphas,
        temperature=args.temperature,
        damping=args.damping,
        device=device,
    )

    selections: dict[tuple[str, float], np.ndarray] = {}
    diagnostic_rows: list[dict] = []
    for budget in budgets:
        count = budget_to_count(budget, len(train_ids))
        methods: dict[str, np.ndarray] = {"cpba": prior_order[:count]}
        for multiplier in multipliers:
            methods[f"spectral_gate_k{multiplier:g}"] = _gate_selection(
                prior_order,
                cpba_score,
                spectral.leverage,
                count,
                multiplier,
                positive_only=False,
            )
            if args.include_positive_cap:
                methods[f"spectral_cap_k{multiplier:g}"] = _gate_selection(
                    prior_order,
                    cpba_score,
                    spectral.leverage,
                    count,
                    multiplier,
                    positive_only=True,
                )
        if args.include_oracle:
            is_noisy = private["is_noisy"].to_numpy(dtype=np.float64)
            oracle_order = descending_ranking(
                is_noisy * spectral.leverage, legal_scores["loss"]
            )
            methods["spectral_oracle"] = oracle_order[:count]

        for method, selected in methods.items():
            selections[(method, budget)] = selected
            composition = query_composition(selected, private)
            diagnostic_rows.append(
                {
                    "method": method,
                    "budget_fraction": budget,
                    "budget_cap": count,
                    "actual_query_count": len(selected),
                    "positive_leverage_count": int((spectral.leverage[selected] > 0).sum()),
                    "predicted_leverage_sum": float(spectral.leverage[selected].sum()),
                    **composition,
                }
            )
            np.save(
                output_root / f"{method}_budget_{budget:.4f}_sample_ids.npy",
                train_ids[selected],
            )

    pd.DataFrame(diagnostic_rows).to_csv(
        output_root / "query_diagnostics.csv", index=False
    )
    np.savez_compressed(
        output_root / "spectral_scores.npz",
        sample_ids=train_ids,
        leverage=spectral.leverage,
        utility=spectral.utility,
        cell_classes=spectral.cell_classes,
        cell_alphas=spectral.cell_alphas,
        cell_risks=spectral.cell_risks,
        cell_weights=spectral.cell_weights,
    )
    metadata = {
        "source_output": str(source_root),
        "noise": args.noise,
        "seed": args.seed,
        "alphas": alphas,
        "temperature": args.temperature,
        "damping": args.damping,
        "candidate_multipliers": multipliers,
        "device": device,
        "spectral_risk_before": spectral.spectral_risk,
        "positive_leverage_fraction": float((spectral.leverage > 0.0).mean()),
        "cell_classes": spectral.cell_classes.tolist(),
        "cell_alphas": spectral.cell_alphas.tolist(),
        "cell_risks": spectral.cell_risks.tolist(),
        "cell_weights": spectral.cell_weights.tolist(),
        "cpba_prior": cpba_metadata,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))
    if args.score_only:
        return

    evaluator = PrivateEvaluator(source_root / "manifests" / "test_private.csv")
    baseline_metrics, _ = _evaluate_state(
        checkpoint["best_state"],
        features=test_features,
        sample_ids=test_ids,
        evaluator=evaluator,
        input_dim=input_dim,
        training_config=training_config,
    )
    result_rows: list[dict] = []
    for (method, budget), selected in selections.items():
        repaired_labels = noisy_labels.copy()
        repaired_labels[selected] = clean_labels[selected]
        retrained = train_linear_head(
            train_features=train_features,
            train_labels=repaired_labels,
            val_features=val_features,
            val_labels=val_labels,
            config=training_config,
            seed=args.seed,
            initial_state=copy.deepcopy(checkpoint["initial_state"]),
            record_dynamics=False,
        )
        metrics, _ = _evaluate_state(
            retrained.best_state,
            features=test_features,
            sample_ids=test_ids,
            evaluator=evaluator,
            input_dim=input_dim,
            training_config=training_config,
        )
        _, repaired_val_probabilities = predict_from_state(
            retrained.best_state, val_features, input_dim, 2, training_config
        )
        risk_after, _, _, _, _ = spectral_class_cvar(
            repaired_val_probabilities,
            val_labels,
            alphas=alphas,
            temperature=args.temperature,
        )
        composition = query_composition(selected, private)
        row = {
            "source_output": source_root.name,
            "noise_name": args.noise,
            "seed": args.seed,
            "method": method,
            "budget_fraction": budget,
            "budget_cap": budget_to_count(budget, len(train_ids)),
            "actual_query_count": len(selected),
            "baseline_wga": baseline_metrics["wga"],
            "wga": metrics["wga"],
            "delta_wga": metrics["wga"] - baseline_metrics["wga"],
            "baseline_average_accuracy": baseline_metrics["average_accuracy"],
            "average_accuracy": metrics["average_accuracy"],
            "delta_average_accuracy": (
                metrics["average_accuracy"] - baseline_metrics["average_accuracy"]
            ),
            "spectral_risk_before": spectral.spectral_risk,
            "spectral_risk_after": risk_after,
            "delta_spectral_risk": risk_after - spectral.spectral_risk,
            **composition,
        }
        result_rows.append(row)
        pd.DataFrame(result_rows).to_csv(output_root / "results.csv", index=False)
        print(
            f"{method} b={budget:g} q={len(selected)} "
            f"precision={composition['noise_precision']:.3f} "
            f"dRisk={row['delta_spectral_risk']:+.4f} "
            f"dWGA={row['delta_wga']:+.4f}"
        )


if __name__ == "__main__":
    main()
