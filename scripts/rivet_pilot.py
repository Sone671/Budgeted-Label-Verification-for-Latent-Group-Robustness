#!/usr/bin/env python
"""Run a small RIVET-point feasibility experiment on cached Stage 0/1 artifacts.

The pilot intentionally tests only the core theoretical claim: whether a
group-free CVaR influence term adds useful information beyond NoiseScore.
It does not yet implement the proposed submodular batch objective.
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
    binary_rivet_scores,
    class_prior_adjusted_noise_score,
    empirical_cvar,
    feedback_exploration_indices,
    fit_feedback_noise_posterior,
    legal_noise_features,
    trust_region_candidate_selection,
)
from robust_verify.scoring import build_legal_scores, descending_ranking
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count


def _parse_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


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
    num_classes: int,
    training_config: dict,
) -> tuple[dict[str, float], np.ndarray]:
    predictions, probabilities = predict_from_state(
        state,
        features,
        input_dim,
        num_classes,
        training_config,
    )
    return evaluator.evaluate(sample_ids, predictions), probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", default="uniform20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument(
        "--noise-proxy",
        choices=[
            "noise_score",
            "loss",
            "cpba",
            "cpba_gate",
            "trust_region",
            "feedback",
            "oracle",
        ],
        default="noise_score",
        help="Noise posterior proxy. 'feedback' is deployable; 'oracle' is diagnostic-only.",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.25,
        help="For feedback, fraction of each budget used to learn the posterior.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.5,
        help="For cpba_gate, high-confidence candidate pool size divided by budget.",
    )
    parser.add_argument(
        "--core-fraction",
        type=float,
        default=0.75,
        help="For trust_region, fraction of CPBA's top-B set that cannot be replaced.",
    )
    parser.add_argument("--budgets", default="0.01,0.02,0.05")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--output-root", default="outputs/rivet_pilot")
    args = parser.parse_args()

    config = load_config(args.config)
    source_root = Path(config["project"]["output_dir"])
    prefix = f"{args.noise}_seed{args.seed}"
    run_name = (
        f"{source_root.name}_{prefix}_{args.noise_proxy}_"
        f"a{args.alpha:g}_d{args.damping:g}"
    )
    if args.noise_proxy in {"cpba_gate", "trust_region"}:
        run_name += f"_k{args.candidate_multiplier:g}"
    if args.noise_proxy == "trust_region":
        run_name += f"_c{args.core_fraction:g}"
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
    num_classes = 2

    _, val_probabilities = predict_from_state(
        checkpoint["best_state"],
        val_features,
        input_dim,
        num_classes,
        training_config,
    )
    legal_scores = build_legal_scores(
        train_probabilities,
        noisy_labels,
        correctness_history,
    )
    cpba_metadata = {}
    if args.noise_proxy == "noise_score":
        noise_proxy = legal_scores["noise_score"]
    elif args.noise_proxy == "loss":
        noise_proxy = legal_scores["loss"]
    elif args.noise_proxy in {"cpba", "cpba_gate", "trust_region"}:
        noise_proxy, cpba_metadata = class_prior_adjusted_noise_score(
            legal_scores["noise_score"], legal_scores["loss"], noisy_labels
        )
    elif args.noise_proxy == "oracle":
        # Diagnostic upper bound only.  This must never be presented as a
        # deployable method because it reads private corruption status.
        noise_proxy = private["is_noisy"].to_numpy(dtype=np.float64)
    else:
        # The initial score selects only the exploration portion.  The final
        # posterior is fitted separately at each budget from verified outcomes.
        noise_proxy = legal_scores["noise_score"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rivet = binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities,
        noisy_labels=noisy_labels,
        validation_features=val_features,
        validation_probabilities=val_probabilities,
        validation_labels=val_labels,
        noise_score=noise_proxy,
        rank_noise_score=args.noise_proxy != "oracle",
        alpha=args.alpha,
        damping=args.damping,
        device=device,
    )
    ranking = descending_ranking(rivet.utility, legal_scores["loss"])
    if args.noise_proxy not in {"feedback", "cpba_gate", "trust_region"}:
        np.save(output_root / "rivet_ranking_sample_ids.npy", train_ids[ranking])
    np.savez_compressed(
        output_root / "rivet_scores.npz",
        sample_ids=train_ids,
        utility=rivet.utility,
        leverage=rivet.leverage,
        noise_probability=rivet.noise_probability,
        tail_indices=rivet.tail_indices,
    )

    budgets = _parse_floats(args.budgets)
    feedback_features = None
    if args.noise_proxy == "feedback":
        feedback_features = legal_noise_features(
            train_probabilities,
            noisy_labels,
            correctness_history,
            rivet.leverage,
        )
    selections: dict[float, tuple[np.ndarray, np.ndarray, int]] = {}

    def selection_for_budget(budget: float) -> tuple[np.ndarray, np.ndarray, int]:
        if budget in selections:
            return selections[budget]
        count = budget_to_count(budget, len(train_ids))
        if args.noise_proxy == "trust_region":
            prior_order = descending_ranking(noise_proxy, legal_scores["loss"])
            selected = trust_region_candidate_selection(
                prior_order,
                noise_proxy,
                rivet.leverage,
                count,
                candidate_multiplier=args.candidate_multiplier,
                core_fraction=args.core_fraction,
            )
            trust_utility = np.full(len(train_ids), -np.inf, dtype=np.float64)
            trust_utility[prior_order[: int(np.ceil(args.candidate_multiplier * count))]] = (
                rivet.leverage[
                    prior_order[: int(np.ceil(args.candidate_multiplier * count))]
                ]
            )
            result = (selected, trust_utility, 0)
            np.save(
                output_root / f"rivet_ranking_budget_{budget:.4f}_sample_ids.npy",
                train_ids[selected],
            )
        elif args.noise_proxy == "cpba_gate":
            if args.candidate_multiplier < 1.0:
                raise ValueError("candidate_multiplier must be at least 1")
            candidate_count = min(
                len(train_ids), int(np.ceil(args.candidate_multiplier * count))
            )
            prior_order = descending_ranking(noise_proxy, legal_scores["loss"])
            candidates = prior_order[:candidate_count]
            local_order = descending_ranking(
                rivet.leverage[candidates], noise_proxy[candidates]
            )
            selected = candidates[local_order[:count]]
            gate_utility = np.full(len(train_ids), -np.inf, dtype=np.float64)
            gate_utility[candidates] = rivet.leverage[candidates]
            result = (selected, gate_utility, 0)
            remaining_candidates = candidates[local_order[count:]]
            candidate_set = set(candidates.tolist())
            outside = np.asarray(
                [index for index in prior_order if int(index) not in candidate_set],
                dtype=np.int64,
            )
            np.save(
                output_root / f"rivet_ranking_budget_{budget:.4f}_sample_ids.npy",
                train_ids[np.concatenate([selected, remaining_candidates, outside])],
            )
        elif args.noise_proxy != "feedback":
            result = (ranking[:count], rivet.utility, 0)
        else:
            if not 0.0 < args.exploration_fraction <= 1.0:
                raise ValueError("exploration_fraction must lie in (0, 1]")
            exploration_count = min(
                count,
                max(1, int(round(args.exploration_fraction * count))),
            )
            exploration = feedback_exploration_indices(
                rivet.utility,
                noisy_labels,
                exploration_count,
                seed=args.seed + int(round(1_000_000 * budget)),
            )
            # These outcomes are revealed by the verification oracle and only
            # for samples already charged to the current query budget.
            outcomes = (clean_labels[exploration] != noisy_labels[exploration]).astype(
                np.int64
            )
            posterior = fit_feedback_noise_posterior(
                feedback_features,
                exploration,
                outcomes,
                seed=args.seed,
                fallback_score=legal_scores["noise_score"],
            )
            feedback_utility = posterior.probability * rivet.leverage
            remaining_order = descending_ranking(feedback_utility, legal_scores["loss"])
            exploration_set = set(exploration.tolist())
            exploitation = np.asarray(
                [index for index in remaining_order if int(index) not in exploration_set],
                dtype=np.int64,
            )
            selected = np.concatenate(
                [exploration, exploitation[: count - exploration_count]]
            )
            result = (selected, feedback_utility, exploration_count)
            np.save(
                output_root / f"rivet_ranking_budget_{budget:.4f}_sample_ids.npy",
                train_ids[np.concatenate([exploration, exploitation])],
            )
            np.savez_compressed(
                output_root / f"feedback_posterior_budget_{budget:.4f}.npz",
                sample_ids=train_ids,
                noise_probability=posterior.probability,
                utility=feedback_utility,
                exploration_sample_ids=train_ids[exploration],
                exploration_outcomes=outcomes,
            )
        selections[budget] = result
        return result

    diagnostics = []
    for budget in budgets:
        count = budget_to_count(budget, len(train_ids))
        selected, active_utility, exploration_count = selection_for_budget(budget)
        composition = query_composition(selected, private)
        diagnostics.append(
            {
                "budget_fraction": budget,
                "budget_count": count,
                "exploration_count": exploration_count,
                "predicted_utility_sum": float(active_utility[selected].sum()),
                "predicted_utility_min": float(active_utility[selected].min()),
                "positive_utility_available": int((active_utility > 0).sum()),
                **composition,
            }
        )
    pd.DataFrame(diagnostics).to_csv(output_root / "query_diagnostics.csv", index=False)

    metadata = {
        "source_output": str(source_root),
        "noise": args.noise,
        "seed": args.seed,
        "alpha": args.alpha,
        "damping": args.damping,
        "noise_proxy": args.noise_proxy,
        "exploration_fraction": (
            args.exploration_fraction if args.noise_proxy == "feedback" else 0.0
        ),
        "candidate_multiplier": (
            args.candidate_multiplier
            if args.noise_proxy in {"cpba_gate", "trust_region"}
            else 0.0
        ),
        "core_fraction": (
            args.core_fraction if args.noise_proxy == "trust_region" else 0.0
        ),
        "device": device,
        "validation_cvar_before": rivet.validation_cvar,
        "tail_count": int(len(rivet.tail_indices)),
        "positive_leverage_fraction": float((rivet.leverage > 0).mean()),
        "positive_utility_count": int((rivet.utility > 0).sum()),
        "cpba_prior": cpba_metadata,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(diagnostics).to_string(index=False))

    if args.score_only:
        return

    test_evaluator = PrivateEvaluator(source_root / "manifests" / "test_private.csv")
    baseline_metrics, _ = _evaluate_state(
        checkpoint["best_state"],
        features=test_features,
        sample_ids=test_ids,
        evaluator=test_evaluator,
        input_dim=input_dim,
        num_classes=num_classes,
        training_config=training_config,
    )

    rows: list[dict] = []
    for budget in budgets:
        count = budget_to_count(budget, len(train_ids))
        selected, _, _ = selection_for_budget(budget)
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
            evaluator=test_evaluator,
            input_dim=input_dim,
            num_classes=num_classes,
            training_config=training_config,
        )
        _, repaired_val_probabilities = predict_from_state(
            retrained.best_state,
            val_features,
            input_dim,
            num_classes,
            training_config,
        )
        composition = query_composition(selected, private)
        row = {
            "source_output": source_root.name,
            "noise_name": args.noise,
            "seed": args.seed,
            "method": (
                f"rivet_{args.noise_proxy}_a{args.alpha:g}"
                + (
                    f"_k{args.candidate_multiplier:g}"
                    if args.noise_proxy in {"cpba_gate", "trust_region"}
                    else ""
                )
                + (
                    f"_c{args.core_fraction:g}"
                    if args.noise_proxy == "trust_region"
                    else ""
                )
            ),
            "budget_fraction": budget,
            "budget_count": count,
            "baseline_wga": baseline_metrics["wga"],
            "wga": metrics["wga"],
            "delta_wga": metrics["wga"] - baseline_metrics["wga"],
            "baseline_average_accuracy": baseline_metrics["average_accuracy"],
            "average_accuracy": metrics["average_accuracy"],
            "delta_average_accuracy": (
                metrics["average_accuracy"] - baseline_metrics["average_accuracy"]
            ),
            "validation_cvar_before": rivet.validation_cvar,
            "validation_cvar_after": empirical_cvar(
                repaired_val_probabilities, val_labels, args.alpha
            ),
            **composition,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_root / "results.csv", index=False)
        print(
            f"RIVET b={budget:.3f} corrected={composition['num_corrected']} "
            f"dCVaR={row['validation_cvar_after'] - row['validation_cvar_before']:+.5f} "
            f"dWGA={row['delta_wga']:+.5f}"
        )


if __name__ == "__main__":
    main()
