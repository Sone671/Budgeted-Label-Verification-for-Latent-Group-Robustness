"""Run a public-only R/G/Stop pilot with blind private evaluation.

The selector is imported from ``robust_verify.public_selector`` and has no
private-data dependency.  Private train/validation/test manifests are used by
this driver only to return already-purchased responses and to score the final
episode, never to choose an action or fit a policy parameter.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robust_verify.data.access import PrivateEvaluator
from robust_verify.features import load_features
from robust_verify.modern_baselines import reliability_gated_repair_value_scores
from robust_verify.public_selector import PublicSelectorDecision, choose_public_action
from robust_verify.public_value import (
    audited_group_accuracy_intervals,
    bernoulli_ipw_group_accuracy_intervals,
)
from robust_verify.training import predict_from_state, train_linear_head

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "ablation_celeba_10seeds"
DEFAULT_OUTPUT = ROOT / "outputs" / "public_only_three_action_celeba_pilot"


@dataclass
class PublicEpisode:
    seed: int
    noise_name: str
    baseline_wga: float
    final_wga: float
    net_utility: float
    verify_only_wga: float
    verify_only_net_utility: float
    selector_delta_vs_verify: float
    rounds: int
    spent_cost: float
    statuses: tuple[str, ...]
    actions: tuple[str, ...]
    unresolved_count: int
    final_public_wga_interval: tuple[float, float]
    final_certificate: str
    selected_verify: bool
    false_continue: bool
    stop_action: bool


def _load_case(root: Path, noise_name: str, seed: int) -> dict[str, object]:
    features = root / "features"
    manifests = root / "manifests"
    train_features, train_ids = load_features(features / "train.npz")
    val_features, val_ids = load_features(features / "val.npz")
    test_features, test_ids = load_features(features / "test.npz")
    public_train = pd.read_csv(manifests / f"{noise_name}_seed{seed}_train_public.csv")
    private_train = pd.read_csv(manifests / f"{noise_name}_seed{seed}_train_private.csv")
    public_val = pd.read_csv(manifests / "val_public.csv")
    private_val = pd.read_csv(manifests / "val_private.csv")
    private_test = manifests / "test_private.csv"
    checkpoint = torch.load(
        root / "stage1" / "probes" / f"{noise_name}_seed{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    dynamics = np.load(root / "stage1" / "probes" / f"{noise_name}_seed{seed}.dynamics.npz")
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))["training"]
    noisy_labels = public_train.set_index("sample_id").loc[train_ids, "noisy_label"].to_numpy(np.int64)
    val_labels = public_val.set_index("sample_id").loc[val_ids, "label"].to_numpy(np.int64)
    clean_train = private_train.set_index("sample_id").loc[train_ids, "clean_label"].to_numpy(np.int64)
    val_groups = private_val.set_index("sample_id").loc[val_ids, "group"].to_numpy(np.int64)
    return {
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "noisy_labels": noisy_labels,
        "val_labels": val_labels,
        "clean_train": clean_train,
        "val_groups": val_groups,
        "private_test": private_test,
        "checkpoint": checkpoint,
        "dynamics": {key: dynamics[key] for key in dynamics.files},
        "training_config": config,
    }


def _predict(case: dict[str, object], labels: np.ndarray) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    train_features = case["train_features"]
    val_features = case["val_features"]
    config = case["training_config"]
    checkpoint = case["checkpoint"]
    result = train_linear_head(
        train_features=train_features,
        train_labels=labels,
        val_features=val_features,
        val_labels=case["val_labels"],
        config=config,
        seed=0,
        initial_state=checkpoint["initial_state"],
        record_dynamics=False,
    )
    _, train_probabilities = predict_from_state(
        result.best_state,
        train_features,
        int(train_features.shape[1]),
        2,
        config,
    )
    val_predictions, _ = predict_from_state(
        result.best_state,
        val_features,
        int(train_features.shape[1]),
        2,
        config,
    )
    return result.best_state, train_probabilities, (val_predictions == case["val_labels"]).astype(np.int64)


def _baseline(case: dict[str, object]) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    checkpoint = case["checkpoint"]
    config = case["training_config"]
    train_features = case["train_features"]
    val_features = case["val_features"]
    state = checkpoint["best_state"]
    _, train_probabilities = predict_from_state(
        state, train_features, int(train_features.shape[1]), 2, config
    )
    val_predictions, _ = predict_from_state(
        state, val_features, int(train_features.shape[1]), 2, config
    )
    return state, train_probabilities, (val_predictions == case["val_labels"]).astype(np.int64)


def _run_episode(
    root: Path,
    noise_name: str,
    seed: int,
    *,
    budget_fraction: float,
    audit_probability: float,
    task_batch_size: int,
    audit_batch_size: int,
    audit_cost_ratio: float,
    audit_design: str,
    verify_score_mode: str,
    allow_uncalibrated_verify: bool,
) -> PublicEpisode:
    case = _load_case(root, noise_name, seed)
    train_features = case["train_features"]
    val_ids = case["val_ids"]
    labels = np.array(case["noisy_labels"], copy=True)
    eligible_masks = np.vstack(
        [case["val_labels"] == (group // 2) for group in range(4)]
    )
    baseline_state, train_probabilities, val_correct = _baseline(case)
    _, val_probabilities = predict_from_state(
        baseline_state,
        case["val_features"],
        int(train_features.shape[1]),
        2,
        case["training_config"],
    )
    baseline_noise = 1.0 - train_probabilities[np.arange(len(labels)), labels]
    verify_scores = baseline_noise
    if verify_score_mode == "public_repair_value":
        verify_scores, _ = reliability_gated_repair_value_scores(
            train_features=train_features,
            train_probabilities=train_probabilities,
            train_labels=labels,
            val_features=case["val_features"],
            val_probabilities=val_probabilities,
            val_labels=case["val_labels"],
            noise_posterior_proxy=baseline_noise,
            reference_budget_count=task_batch_size,
            num_folds=5,
            seed=seed,
        )
    elif verify_score_mode != "noise":
        raise ValueError(f"unsupported verify score mode: {verify_score_mode}")
    evaluator = PrivateEvaluator(case["private_test"])
    baseline_predictions, _ = predict_from_state(
        case["checkpoint"]["best_state"],
        case["test_features"],
        int(train_features.shape[1]),
        2,
        case["training_config"],
    )
    baseline_metrics = evaluator.evaluate(case["test_ids"], baseline_predictions)
    budget = max(float(len(labels)) * budget_fraction, float(task_batch_size))
    verify_ids = np.argsort(-verify_scores, kind="stable")[:task_batch_size]
    verify_labels = labels.copy()
    verify_labels[verify_ids] = case["clean_train"][verify_ids]
    verify_state, _, _ = _predict(case, verify_labels)
    verify_predictions, _ = predict_from_state(
        verify_state,
        case["test_features"],
        int(train_features.shape[1]),
        2,
        case["training_config"],
    )
    verify_metrics = evaluator.evaluate(case["test_ids"], verify_predictions)
    verify_only_net_utility = float(
        verify_metrics["wga"] - baseline_metrics["wga"] - task_batch_size / budget
    )
    rng = np.random.default_rng(seed + 1000)
    if audit_design == "uniform":
        inclusion = np.full(len(val_ids), audit_probability, dtype=float)
    elif audit_design == "public_uncertainty":
        uncertainty = 1.0 - np.max(val_probabilities, axis=1)
        weights = 0.5 + np.clip(uncertainty, 0.0, 1.0)
        lower = min(1e-4, audit_probability)
        upper = 1.0
        # Calibrate a public, fixed propensity vector to the requested mean.
        for _ in range(60):
            scale = 0.5 * (lower + upper)
            if float(np.clip(scale * weights, 1e-4, 1.0).mean()) < audit_probability:
                lower = scale
            else:
                upper = scale
        inclusion = np.clip(0.5 * (lower + upper) * weights, 1e-4, 1.0)
    else:
        raise ValueError(f"unsupported audit design: {audit_design}")
    # Each inclusion indicator is independent and its probability is frozen
    # using public state before any private group response is revealed.
    included = rng.random(len(val_ids)) < inclusion
    available_audit = np.flatnonzero(included).astype(np.int64)
    audited_indices: list[int] = []
    audited_groups: list[int] = []
    private_val_by_id = pd.read_csv(root / "manifests" / "val_private.csv").set_index("sample_id")
    private_train = case["clean_train"]
    verified: set[int] = set()
    spent = 0.0
    actions: list[str] = []
    statuses: list[str] = []
    final_interval = (0.0, 1.0)
    certificate = "stop"

    # The full Bernoulli sample is the purchased group-audit action.  Keeping
    # every included unit is required for the stated IPW guarantee; selecting
    # a prefix after seeing inclusion indicators would change the design.
    if len(available_audit):
        scout_values = private_val_by_id.loc[val_ids[available_audit], "group"].to_numpy(np.int64)
        audited_indices.extend(int(i) for i in available_audit.tolist())
        audited_groups.extend(int(v) for v in scout_values.tolist())
        spent += float(len(available_audit)) * audit_cost_ratio
        actions.append("annotate_group")
        statuses.append("audit_purchased")

    for _ in range(1):
        task_candidates = np.asarray(
            [i for i in np.argsort(-verify_scores, kind="stable") if i not in verified],
            dtype=np.int64,
        )[:task_batch_size]
        audit_candidates = np.asarray([], dtype=np.int64)
        decision: PublicSelectorDecision = choose_public_action(
            current_labels=labels,
            train_probabilities=train_probabilities,
            public_validation_correct=val_correct,
            audited_indices=audited_indices,
            audited_groups=audited_groups,
            inclusion_probabilities=inclusion,
            candidate_task_ids=task_candidates,
            candidate_audit_ids=audit_candidates,
            num_groups=4,
            remaining_budget=budget - spent,
            task_batch_cost=float(task_batch_size),
            audit_batch_cost=float(audit_batch_size) * audit_cost_ratio,
            comparisons=2,
            eligible_masks=eligible_masks,
            allow_uncalibrated_verify=allow_uncalibrated_verify,
            verify_scores=verify_scores,
        )
        statuses.append(decision.status)
        certificate = decision.status
        final_interval = (decision.group_wga_interval.lower, decision.group_wga_interval.upper)
        if decision.selected_kind == "stop":
            actions.append("stop")
            break
        actions.append(decision.selected_kind)
        if decision.selected_kind == "verify_task":
            selected = np.asarray(decision.selected_ids, dtype=np.int64)
            labels[selected] = private_train[selected]
            verified.update(int(i) for i in selected.tolist())
            spent += float(task_batch_size)
            _, train_probabilities, val_correct = _predict(case, labels)
        else:
            selected = np.asarray(decision.selected_ids, dtype=np.int64)
            values = private_val_by_id.loc[val_ids[selected], "group"].to_numpy(np.int64)
            audited_indices.extend(int(i) for i in selected.tolist())
            audited_groups.extend(int(v) for v in values.tolist())
            spent += float(audit_batch_size) * audit_cost_ratio

    if verified:
        final_state, _, _ = _predict(case, labels)
    else:
        final_state = case["checkpoint"]["best_state"]
    final_predictions, _ = predict_from_state(
        final_state,
        case["test_features"],
        int(train_features.shape[1]),
        2,
        case["training_config"],
    )
    final_metrics = evaluator.evaluate(case["test_ids"], final_predictions)
    net_utility = float(final_metrics["wga"] - baseline_metrics["wga"] - spent / max(budget, 1e-12))
    return PublicEpisode(
        seed=seed,
        noise_name=noise_name,
        baseline_wga=float(baseline_metrics["wga"]),
        final_wga=float(final_metrics["wga"]),
        net_utility=net_utility,
        verify_only_wga=float(verify_metrics["wga"]),
        verify_only_net_utility=verify_only_net_utility,
        selector_delta_vs_verify=net_utility - verify_only_net_utility,
        rounds=len(actions),
        spent_cost=spent,
        statuses=tuple(statuses),
        actions=tuple(actions),
        unresolved_count=sum(status == "unresolved" for status in statuses),
        final_public_wga_interval=final_interval,
        final_certificate=certificate,
        selected_verify="verify_task" in actions,
        false_continue=("verify_task" in actions and final_metrics["wga"] < baseline_metrics["wga"]),
        stop_action=bool(actions and actions[-1] == "stop"),
    )


def run_coverage(
    case_root: Path,
    *,
    seed: int,
    draws: int,
    audit_probability: float,
    audit_design: str,
) -> dict[str, float]:
    case = _load_case(case_root, "uniform20", seed)
    baseline_state, _, val_correct = _baseline(case)
    _, val_probabilities = predict_from_state(
        baseline_state,
        case["val_features"],
        int(case["train_features"].shape[1]),
        2,
        case["training_config"],
    )
    groups = case["val_groups"]
    true_values = []
    for group in range(4):
        mask = groups == group
        true_values.append(float(val_correct[mask].mean()) if mask.any() else 0.0)
    rng = np.random.default_rng(seed + 5000)
    covered = 0
    widths: list[float] = []
    for _ in range(draws):
        if audit_design == "uniform":
            inclusion = np.full(len(groups), audit_probability, dtype=float)
        else:
            uncertainty = 1.0 - np.max(val_probabilities, axis=1)
            weights = 0.5 + np.clip(uncertainty, 0.0, 1.0)
            lower, upper = min(1e-4, audit_probability), 1.0
            for _ in range(60):
                scale = 0.5 * (lower + upper)
                if float(np.clip(scale * weights, 1e-4, 1.0).mean()) < audit_probability:
                    lower = scale
                else:
                    upper = scale
            inclusion = np.clip(0.5 * (lower + upper) * weights, 1e-4, 1.0)
        sampled = np.flatnonzero(rng.random(len(groups)) < inclusion)
        if audit_design == "uniform":
            intervals = audited_group_accuracy_intervals(
                val_correct[sampled],
                groups[sampled],
                num_groups=4,
                delta=0.05,
                comparisons=2,
            )
        else:
            intervals = bernoulli_ipw_group_accuracy_intervals(
                val_correct,
                sampled,
                groups[sampled],
                inclusion_probabilities=inclusion,
                num_groups=4,
                delta=0.05,
                comparisons=2,
            )
        covered += int(all(interval.value.lower <= true_values[interval.group] <= interval.value.upper for interval in intervals))
        widths.append(float(max(interval.value.upper - interval.value.lower for interval in intervals)))
    return {"draws": float(draws), "simultaneous_coverage": covered / draws, "mean_max_width": float(np.mean(widths))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--noise", nargs="+", default=["uniform20", "minority_high_40"])
    parser.add_argument("--coverage-draws", type=int, default=1000)
    parser.add_argument("--budget-fraction", type=float, default=0.02)
    parser.add_argument("--audit-probability", type=float, default=0.10)
    parser.add_argument(
        "--audit-design",
        choices=["uniform", "public_uncertainty"],
        default="uniform",
    )
    parser.add_argument("--task-batch-size", type=int, default=256)
    parser.add_argument("--audit-batch-size", type=int, default=512)
    parser.add_argument("--audit-cost-ratio", type=float, default=1.0)
    parser.add_argument(
        "--verify-score-mode",
        choices=["noise", "public_repair_value"],
        default="noise",
    )
    parser.add_argument("--allow-uncalibrated-verify", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for noise in args.noise:
        for seed in args.seeds:
            print(f"[public-only] noise={noise} seed={seed}", flush=True)
            episode = _run_episode(
                args.root,
                noise,
                seed,
                budget_fraction=args.budget_fraction,
                audit_probability=args.audit_probability,
                task_batch_size=args.task_batch_size,
                audit_batch_size=args.audit_batch_size,
                audit_cost_ratio=args.audit_cost_ratio,
                audit_design=args.audit_design,
                verify_score_mode=args.verify_score_mode,
                allow_uncalibrated_verify=args.allow_uncalibrated_verify,
            )
            rows.append(episode.__dict__)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "episodes.csv", index=False)
    coverage = run_coverage(
        args.root,
        seed=args.seeds[0],
        draws=args.coverage_draws,
        audit_probability=args.audit_probability,
        audit_design=args.audit_design,
    )
    certificate_decisions = max(len(rows), 1)
    summary = {
        "protocol": (
            "public_only_three_action_v3_full_bernoulli_exploratory"
            if args.allow_uncalibrated_verify
            else "public_only_three_action_v3_full_bernoulli_safe"
        ),
        "audit_design": args.audit_design,
        "audit_probability_target": args.audit_probability,
        "budget_fraction": args.budget_fraction,
        "verify_score_mode": args.verify_score_mode,
        "allow_uncalibrated_verify": args.allow_uncalibrated_verify,
        "selector_private_imports": False,
        "episodes": len(frame),
        "mean_net_utility": float(frame["net_utility"].mean()),
        "unresolved_rate": float(frame["unresolved_count"].sum() / certificate_decisions),
        "stop_action_rate": float(frame["stop_action"].mean()),
        "false_continue_rate": float(frame["false_continue"].mean()),
        "mean_spent_cost": float(frame["spent_cost"].mean()),
        "coverage": coverage,
        "private_use": "purchased-response simulator and final blind evaluator only",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
