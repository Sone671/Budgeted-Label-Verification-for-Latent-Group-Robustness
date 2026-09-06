"""Run the frozen three-action MVP protocol on cached real datasets.

The policy-side computations use only public manifests, frozen features, and
already purchased feedback.  Private manifests are accessed in this driver
only to simulate purchased responses and to compute offline WGA/oracle
utilities.  This is a development oracle audit, not a deployable selector.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robust_verify.action_value import build_legal_state_features
from robust_verify.analysis import align_frame
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.features import load_features
from robust_verify.joint_budget import (
    estimate_soft_group_error,
    fit_attribute_probability_model,
    joint_repair_scores,
    select_class_balanced_audits,
)
from robust_verify.scoring import build_legal_scores
from robust_verify.sequential_audit import (
    AuditAction,
    SequentialAuditConfig,
    SequentialAuditState,
    apply_action,
    make_batch_action,
    replay,
    state_to_json,
)
from robust_verify.trajectory_oracle import (
    TreeNode,
    TreeTransition,
    classify_policy_decision,
    solve_dynamic_program,
)
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count, class_balanced_accuracy


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = {
    "waterbirds": ROOT / "outputs" / "v9_waterbirds_3seeds",
    "celeba": ROOT / "outputs" / "v9_celeba_3seeds",
    # The original ``celeba`` cache is the historical Eyeglasses task.  Keep
    # that entry stable and expose the requested Blond_Hair task explicitly.
    "celeba_blond_hair": ROOT / "outputs" / "celeba_5seeds",
    "civilcomments": ROOT / "outputs" / "v9_civilcomments_3seeds",
    # ACSIncome is the frozen external-confirmation cache (10 seeds).  Keep
    # it as an explicit alias so the three-action driver can use exactly the
    # same legal R/G/stop protocol as the image/text tasks.
    "acs_income": ROOT / "outputs" / "phase11_acs_income_10seeds",
}
NOISE_NAMES = {
    "waterbirds": ("uniform20", "minority_high_40"),
    "celeba": ("uniform20", "minority_high_40"),
    "celeba_blond_hair": ("uniform20", "minority_high_40"),
    "civilcomments": ("uniform20", "minority_high_40"),
    "acs_income": ("uniform20", "minority_high_40"),
}
LAMBDA_GRID = (0.0, 0.01, 0.02)
COST_RATIOS = (0.1, 1.0)
BUDGET_FRACTION = 0.02
ACTION_BUDGET_FRACTION = 0.25
MAX_ROUNDS = 4
TREE_DEPTH = 2
_R_BRANCH_CACHE: dict[
    tuple[str, str, int],
    tuple[AuditAction, SequentialAuditState, dict[str, float], np.ndarray, AuditAction, SequentialAuditState, dict[str, float]],
] = {}


@dataclass
class CaseData:
    dataset: str
    root: Path
    noise_name: str
    seed: int
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    public: pd.DataFrame
    private: pd.DataFrame
    val_public: pd.DataFrame
    val_private: pd.DataFrame
    noisy_labels: np.ndarray
    val_labels: np.ndarray
    baseline_train_probabilities: np.ndarray
    baseline_val_probabilities: np.ndarray
    baseline_state: dict[str, torch.Tensor]
    initial_state: dict[str, torch.Tensor]
    training_config: dict[str, Any]
    baseline_metrics: dict[str, float]
    evaluator: PrivateEvaluator
    label_oracle: VerificationOracle


@dataclass
class CaseBranches:
    config: SequentialAuditConfig
    initial_state: SequentialAuditState
    r1_action: AuditAction
    r1_state: SequentialAuditState
    r1_metrics: dict[str, float]
    r1_probabilities: np.ndarray
    r2_action: AuditAction
    r2_state: SequentialAuditState
    r2_metrics: dict[str, float]
    g1_action: AuditAction
    g1_state: SequentialAuditState
    gr_action: AuditAction
    gr_state: SequentialAuditState
    gr_metrics: dict[str, float]
    attribute_bacc: float
    group_error: np.ndarray
    audit_sequence: np.ndarray


def _load_probe(root: Path, prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt_path = root / "stage1" / "probes" / f"{prefix}.pt"
    dynamics_path = root / "stage1" / "probes" / f"{prefix}.dynamics.npz"
    if not ckpt_path.exists() or not dynamics_path.exists():
        raise FileNotFoundError(f"missing cached probe/dynamics for {prefix} under {root}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dynamics = np.load(dynamics_path)
    return checkpoint, {key: dynamics[key] for key in dynamics.files}


def _load_case(dataset: str, root: Path, noise_name: str, seed: int) -> CaseData:
    features_root = root / "features"
    manifests_root = root / "manifests"
    train_features, train_ids = load_features(features_root / "train.npz")
    val_features, val_ids = load_features(features_root / "val.npz")
    test_features, test_ids = load_features(features_root / "test.npz")
    prefix = f"{noise_name}_seed{seed}"
    public = align_frame(
        pd.read_csv(manifests_root / f"{prefix}_train_public.csv"), train_ids
    )
    private = align_frame(
        pd.read_csv(manifests_root / f"{prefix}_train_private.csv"), train_ids
    )
    val_public = align_frame(pd.read_csv(manifests_root / "val_public.csv"), val_ids)
    val_private = align_frame(pd.read_csv(manifests_root / "val_private.csv"), val_ids)
    checkpoint, dynamics = _load_probe(root, prefix)
    noisy_labels = public["noisy_label"].to_numpy(dtype=np.int64)
    val_labels = val_public["label"].to_numpy(dtype=np.int64)
    evaluator = PrivateEvaluator(manifests_root / "test_private.csv")
    label_oracle = VerificationOracle(manifests_root / f"{prefix}_train_private.csv")
    input_dim = int(train_features.shape[1])
    num_classes = int(max(noisy_labels.max(), val_labels.max()) + 1)
    training_config = json.loads((root / "resolved_config.json").read_text())[
        "training"
    ]
    baseline_state = checkpoint["best_state"]
    initial_state = checkpoint["initial_state"]
    _, baseline_val_probabilities = predict_from_state(
        baseline_state,
        val_features,
        input_dim,
        num_classes,
        training_config,
    )
    baseline_train_probabilities = dynamics["train_probabilities"]
    baseline_predictions, _ = predict_from_state(
        baseline_state,
        test_features,
        input_dim,
        num_classes,
        training_config,
    )
    baseline_metrics = evaluator.evaluate(test_ids, baseline_predictions)
    return CaseData(
        dataset=dataset,
        root=root,
        noise_name=noise_name,
        seed=seed,
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        public=public,
        private=private,
        val_public=val_public,
        val_private=val_private,
        noisy_labels=noisy_labels,
        val_labels=val_labels,
        baseline_train_probabilities=baseline_train_probabilities,
        baseline_val_probabilities=baseline_val_probabilities,
        baseline_state=baseline_state,
        initial_state=initial_state,
        training_config=training_config,
        baseline_metrics=baseline_metrics,
        evaluator=evaluator,
        label_oracle=label_oracle,
    )


def _repair_labels(case: CaseData, positions: tuple[int, ...]) -> np.ndarray:
    labels = case.noisy_labels.copy()
    ids = case.train_ids[np.asarray(positions, dtype=np.int64)]
    labels[np.asarray(positions, dtype=np.int64)] = case.label_oracle.verify(ids)
    return labels


def _train_branch(
    case: CaseData,
    labels: np.ndarray,
) -> tuple[dict[str, torch.Tensor], np.ndarray, dict[str, float]]:
    input_dim = int(case.train_features.shape[1])
    num_classes = int(max(labels.max(), case.val_labels.max()) + 1)
    probe = train_linear_head(
        train_features=case.train_features,
        train_labels=labels,
        val_features=case.val_features,
        val_labels=case.val_labels,
        config=case.training_config,
        seed=case.seed,
        initial_state=copy.deepcopy(case.initial_state),
        record_dynamics=False,
    )
    predictions, probabilities = predict_from_state(
        probe.best_state,
        case.train_features,
        input_dim,
        num_classes,
        case.training_config,
    )
    test_predictions, _ = predict_from_state(
        probe.best_state,
        case.test_features,
        input_dim,
        num_classes,
        case.training_config,
    )
    del predictions
    metrics = case.evaluator.evaluate(case.test_ids, test_predictions)
    return probe.best_state, probabilities, metrics


def _legal_noise_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(np.int8)[:, None]
    return build_legal_scores(probabilities, labels, correctness)["noise_score"]


def _task_action(
    scores: np.ndarray,
    state: SequentialAuditState,
    count: int,
    config: SequentialAuditConfig,
) -> AuditAction:
    available = np.asarray(
        [index for index in range(config.task_pool_size) if index not in state.verified_ids],
        dtype=np.int64,
    )
    if count <= 0 or count > len(available):
        raise ValueError("requested task batch is not available")
    order = np.lexsort((available, -np.asarray(scores)[available]))
    return make_batch_action("verify_task", available[order[:count]], config)


def _group_action(
    audit_sequence: np.ndarray,
    state: SequentialAuditState,
    count: int,
    config: SequentialAuditConfig,
) -> AuditAction:
    available = [
        int(index)
        for index in audit_sequence.tolist()
        if int(index) not in state.audited_attribute_ids
    ]
    if count <= 0 or count > len(available):
        raise ValueError("requested group batch is not available")
    return make_batch_action("annotate_group", available[:count], config)


def _apply_task(
    case: CaseData,
    state: SequentialAuditState,
    action: AuditAction,
    config: SequentialAuditConfig,
) -> SequentialAuditState:
    responses = case.label_oracle.verify(case.train_ids[np.asarray(action.sample_ids)])
    return apply_action(state, action, config, responses=responses)


def _apply_group(
    case: CaseData,
    state: SequentialAuditState,
    action: AuditAction,
    config: SequentialAuditConfig,
) -> SequentialAuditState:
    val_private = case.val_private.set_index("sample_id")
    responses = val_private.loc[case.val_ids[np.asarray(action.sample_ids)], "place"].to_numpy(
        dtype=np.int64
    )
    return apply_action(state, action, config, responses=responses)


def _state_hash(state: SequentialAuditState) -> str:
    return hashlib.sha256(state_to_json(state).encode("utf-8")).hexdigest()


def _state_record(
    state: SequentialAuditState,
    config: SequentialAuditConfig,
    *,
    state_id: str,
    policy_action: str,
    oracle_action: str,
    classification: str,
    oracle_value: float,
    task_value: float,
    group_value: float,
    current_wga: float,
    final_wga: float,
    baseline_wga: float,
    lambda_value: float,
    cost_ratio: float,
    attribute_bacc: float,
    before_worst_group: str,
    after_worst_group: str,
) -> dict[str, Any]:
    legal = build_legal_state_features(state, config)
    return {
        "state_id": state_id,
        "state_hash": _state_hash(state),
        "policy_action": policy_action,
        "oracle_action": oracle_action,
        "classification": classification,
        "oracle_value": float(oracle_value),
        "task_value": float(task_value),
        "group_value": float(group_value),
        "lambda": lambda_value,
        "cost_ratio": cost_ratio,
        "spent_cost": float(state.spent_cost),
        "remaining_budget": float(state.remaining_budget(config)),
        "verified_count": legal.verified_count,
        "audited_attribute_count": legal.audited_attribute_count,
        "current_label_class0": legal.current_label_histogram[0],
        "current_label_class1": legal.current_label_histogram[1],
        "transcript_kinds": ",".join(legal.transcript_kinds),
        "current_wga": current_wga,
        "final_wga": final_wga,
        "baseline_wga": baseline_wga,
        "attribute_bacc": attribute_bacc,
        "before_worst_group": before_worst_group,
        "after_worst_group": after_worst_group,
    }


def _worst_group(metrics: dict[str, float]) -> str:
    values = {
        key: value
        for key, value in metrics.items()
        if key.startswith("group_") and key.endswith("_accuracy")
    }
    return min(values, key=lambda key: (values[key], key)) if values else ""


def _build_branches(case: CaseData, cost_ratio: float) -> CaseBranches:
    n_train = len(case.train_ids)
    budget_count = budget_to_count(BUDGET_FRACTION, n_train)
    first_count = max(1, int(budget_count * ACTION_BUDGET_FRACTION))
    group_count = min(len(case.val_ids), max(1, int(first_count / cost_ratio)))
    config = SequentialAuditConfig(
        budget=Fraction(budget_count),
        task_label_cost=Fraction(1),
        attribute_cost=Fraction(str(cost_ratio)),
        task_pool_size=n_train,
        attribute_pool_size=len(case.val_ids),
        max_rounds=MAX_ROUNDS,
    )
    initial = SequentialAuditState.initial(case.noisy_labels)
    baseline_scores = _legal_noise_scores(
        case.baseline_train_probabilities, case.noisy_labels
    )
    cache_key = (case.dataset, case.noise_name, case.seed)
    cached = _R_BRANCH_CACHE.get(cache_key)
    if cached is None:
        r1_action = _task_action(baseline_scores, initial, first_count, config)
        r1_state = _apply_task(case, initial, r1_action, config)
        _, r1_probabilities, r1_metrics = _train_branch(case, r1_state.current_labels)
        r1_scores = _legal_noise_scores(r1_probabilities, r1_state.current_labels)
        r2_action = _task_action(r1_scores, r1_state, first_count, config)
        r2_state = _apply_task(case, r1_state, r2_action, config)
        _, _, r2_metrics = _train_branch(case, r2_state.current_labels)
        _R_BRANCH_CACHE[cache_key] = (
            r1_action,
            r1_state,
            r1_metrics,
            r1_probabilities,
            r2_action,
            r2_state,
            r2_metrics,
        )
    else:
        (
            r1_action,
            r1_state,
            r1_metrics,
            r1_probabilities,
            r2_action,
            r2_state,
            r2_metrics,
        ) = cached

    audit_sequence = select_class_balanced_audits(
        case.val_labels,
        min(len(case.val_ids), 2 * group_count),
        seed=case.seed,
    )
    g1_action = _group_action(audit_sequence, initial, group_count, config)
    g1_state = _apply_group(case, initial, g1_action, config)
    audited_positions = np.asarray(g1_action.sample_ids, dtype=np.int64)
    audited_attributes = g1_state.audited_attributes
    attr_values = np.asarray(
        [audited_attributes[int(index)] for index in audited_positions], dtype=np.int64
    )
    attribute_model = fit_attribute_probability_model(
        case.val_features[audited_positions], attr_values, seed=case.seed
    )
    train_attr_probability = attribute_model.predict_probability(case.train_features)
    val_attr_probability = attribute_model.predict_probability(case.val_features)
    group_error, _ = estimate_soft_group_error(
        case.baseline_val_probabilities,
        case.val_labels,
        val_attr_probability,
    )
    joint_scores, _ = joint_repair_scores(
        baseline_scores,
        case.baseline_train_probabilities,
        train_attr_probability,
        group_error,
    )
    gr_action = _task_action(joint_scores, g1_state, first_count, config)
    gr_state = _apply_task(case, g1_state, gr_action, config)
    _, _, gr_metrics = _train_branch(case, gr_state.current_labels)
    attr_bacc = class_balanced_accuracy(
        case.val_private["place"].to_numpy(dtype=np.int64),
        (val_attr_probability >= 0.5).astype(np.int64),
    )
    return CaseBranches(
        config=config,
        initial_state=initial,
        r1_action=r1_action,
        r1_state=r1_state,
        r1_metrics=r1_metrics,
        r1_probabilities=r1_probabilities,
        r2_action=r2_action,
        r2_state=r2_state,
        r2_metrics=r2_metrics,
        g1_action=g1_action,
        g1_state=g1_state,
        gr_action=gr_action,
        gr_state=gr_state,
        gr_metrics=gr_metrics,
        attribute_bacc=attr_bacc,
        group_error=group_error,
        audit_sequence=audit_sequence,
    )


def _action_value(decision: Any, action: AuditAction) -> float:
    for candidate, value in decision.action_values:
        if candidate == action:
            return float(value)
    raise KeyError("action is absent from oracle decision")


def _run_case(case: CaseData, cost_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    branches = _build_branches(case, cost_ratio)
    config = branches.config
    baseline_wga = case.baseline_metrics["wga"]
    prefix = f"{case.dataset}/{case.noise_name}/seed{case.seed}/r{cost_ratio:g}"
    first_cost = float(branches.r1_action.cost)
    group_cost = float(branches.g1_action.cost)
    r1_delta = branches.r1_metrics["wga"] - baseline_wga
    r2_delta = branches.r2_metrics["wga"] - branches.r1_metrics["wga"]
    gr_delta = branches.gr_metrics["wga"] - baseline_wga
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    # The same realized private feedback is used for all lambda sensitivities;
    # lambda is declared before opening child WGA and only changes the oracle
    # objective, never the action ranking or retraining data.
    for lambda_value in LAMBDA_GRID:
        def net(delta: float, cost: float) -> float:
            return float(delta - lambda_value * cost / float(config.budget))

        r1_net = net(r1_delta, first_cost)
        r2_net = net(r2_delta, first_cost)
        g_cost_net = net(0.0, group_cost)
        gr_net = net(gr_delta, first_cost)
        terminal = TreeNode(f"{prefix}/terminal/{lambda_value}")
        def optional_group_action(state: SequentialAuditState) -> AuditAction | None:
            remaining = [
                int(index)
                for index in branches.audit_sequence.tolist()
                if int(index) not in state.audited_attribute_ids
            ]
            count = min(len(branches.g1_action.sample_ids), len(remaining))
            if count == 0:
                return None
            return _group_action(branches.audit_sequence, state, count, config)

        r_after_group = optional_group_action(branches.r1_state)
        g_after_group = optional_group_action(branches.g1_state)
        r_after_transitions = [TreeTransition(branches.r2_action, terminal.node_id, r2_net)]
        if r_after_group is not None:
            r_after_transitions.append(TreeTransition(r_after_group, terminal.node_id, net(0.0, float(r_after_group.cost))))
        g_after_transitions = [TreeTransition(branches.gr_action, terminal.node_id, gr_net)]
        if g_after_group is not None:
            g_after_transitions.append(TreeTransition(g_after_group, terminal.node_id, net(0.0, float(g_after_group.cost))))
        r_after = TreeNode(
            f"{prefix}/after_r/{lambda_value}",
            transitions=tuple(r_after_transitions),
        )
        g_after = TreeNode(
            f"{prefix}/after_g/{lambda_value}",
            transitions=tuple(g_after_transitions),
        )
        root = TreeNode(
            f"{prefix}/root/{lambda_value}",
            transitions=(
                TreeTransition(branches.r1_action, r_after.node_id, r1_net),
                TreeTransition(branches.g1_action, g_after.node_id, g_cost_net),
            ),
        )
        solution = solve_dynamic_program(
            {
                terminal.node_id: terminal,
                r_after.node_id: r_after,
                g_after.node_id: g_after,
                root.node_id: root,
            },
            root.node_id,
            max_depth=TREE_DEPTH,
        )
        root_decision = solution.root
        r_decision = solution.decisions[(r_after.node_id, 1)]
        g_decision = solution.decisions[(g_after.node_id, 1)]
        label_first_value = r1_net + r2_net
        audit_first_value = g_cost_net + gr_net
        fixed_values = {
            "always_label": label_first_value,
            "audit_then_label": audit_first_value,
        }
        best_fixed_name = max(fixed_values, key=lambda name: (fixed_values[name], name))
        best_fixed_value = fixed_values[best_fixed_name]
        summaries.append(
            {
                "dataset": case.dataset,
                "noise_name": case.noise_name,
                "seed": case.seed,
                "cost_ratio": cost_ratio,
                "lambda": lambda_value,
                "budget": float(config.budget),
                "r_batch_count": len(branches.r1_action.sample_ids),
                "g_batch_count": len(branches.g1_action.sample_ids),
                "spent_always_label": float(branches.r2_state.spent_cost),
                "spent_audit_then_label": float(branches.gr_state.spent_cost),
                "baseline_wga": baseline_wga,
                "always_label_wga": branches.r2_metrics["wga"],
                "audit_then_label_wga": branches.gr_metrics["wga"],
                "oracle_value": root_decision.value,
                "oracle_action": root_decision.action.kind,
                "always_label_value": label_first_value,
                "audit_then_label_value": audit_first_value,
                "best_fixed_name": best_fixed_name,
                "best_fixed_value": best_fixed_value,
                "stop_now_value": 0.0,
                "headroom_vs_best_fixed": root_decision.value - best_fixed_value,
                "label_first_regret": root_decision.value - label_first_value,
                "audit_first_regret": root_decision.value - audit_first_value,
                "attribute_bacc": branches.attribute_bacc,
                "r1_noise_precision": float(case.private.iloc[list(branches.r1_action.sample_ids)]["is_noisy"].mean()),
                "r2_noise_precision": float(case.private.iloc[list(branches.r2_action.sample_ids)]["is_noisy"].mean()),
                "gr_noise_precision": float(case.private.iloc[list(branches.gr_action.sample_ids)]["is_noisy"].mean()),
                "baseline_worst_group": _worst_group(case.baseline_metrics),
                "always_label_worst_group": _worst_group(branches.r2_metrics),
                "audit_then_label_worst_group": _worst_group(branches.gr_metrics),
                "oracle_root_action_values": json.dumps(
                    {action.kind: value for action, value in root_decision.action_values},
                    sort_keys=True,
                ),
            }
        )
        common = dict(
            dataset=case.dataset,
            noise_name=case.noise_name,
            seed=case.seed,
            cost_ratio=cost_ratio,
            lambda_value=lambda_value,
        )
        rows.extend(
            [
                {
                    **common,
                    **_state_record(
                        branches.initial_state,
                        config,
                        state_id=f"{prefix}/root",
                        policy_action=branches.r1_action.kind,
                        oracle_action=root_decision.action.kind,
                        classification=classify_policy_decision(branches.r1_action, root_decision),
                        oracle_value=root_decision.value,
                        task_value=_action_value(root_decision, branches.r1_action),
                        group_value=_action_value(root_decision, branches.g1_action),
                        current_wga=baseline_wga,
                        final_wga=branches.r2_metrics["wga"],
                        baseline_wga=baseline_wga,
                        lambda_value=lambda_value,
                        cost_ratio=cost_ratio,
                        attribute_bacc=branches.attribute_bacc,
                        before_worst_group=_worst_group(case.baseline_metrics),
                        after_worst_group=_worst_group(branches.r2_metrics),
                    ),
                },
                {
                    **common,
                    **_state_record(
                        branches.r1_state,
                        config,
                        state_id=f"{prefix}/after_r",
                        policy_action=branches.r2_action.kind,
                        oracle_action=r_decision.action.kind,
                        classification=classify_policy_decision(branches.r2_action, r_decision),
                        oracle_value=r_decision.value,
                        task_value=_action_value(r_decision, branches.r2_action),
                        group_value=(
                            _action_value(r_decision, r_after_group)
                            if r_after_group is not None
                            else 0.0
                        ),
                        current_wga=branches.r1_metrics["wga"],
                        final_wga=branches.r2_metrics["wga"],
                        baseline_wga=baseline_wga,
                        lambda_value=lambda_value,
                        cost_ratio=cost_ratio,
                        attribute_bacc=branches.attribute_bacc,
                        before_worst_group=_worst_group(branches.r1_metrics),
                        after_worst_group=_worst_group(branches.r2_metrics),
                    ),
                },
                {
                    **common,
                    **_state_record(
                        branches.g1_state,
                        config,
                        state_id=f"{prefix}/after_g",
                        policy_action=branches.gr_action.kind,
                        oracle_action=g_decision.action.kind,
                        classification=classify_policy_decision(branches.gr_action, g_decision),
                        oracle_value=g_decision.value,
                        task_value=_action_value(g_decision, branches.gr_action),
                        group_value=(
                            _action_value(g_decision, g_after_group)
                            if g_after_group is not None
                            else 0.0
                        ),
                        current_wga=baseline_wga,
                        final_wga=branches.gr_metrics["wga"],
                        baseline_wga=baseline_wga,
                        lambda_value=lambda_value,
                        cost_ratio=cost_ratio,
                        attribute_bacc=branches.attribute_bacc,
                        before_worst_group=_worst_group(case.baseline_metrics),
                        after_worst_group=_worst_group(branches.gr_metrics),
                    ),
                },
            ]
        )
    # Verify one transcript with the deterministic replay boundary before any
    # result is written.  Stop is exercised as the absorbing terminal action.
    replayed = replay(
        case.noisy_labels,
        branches.config,
        apply_action(branches.r2_state, AuditAction.stop(), branches.config).transcript,
    )
    if replayed.current_labels.tolist() != branches.r2_state.current_labels.tolist():
        raise AssertionError("real-data transcript replay mismatch")
    group_replayed = replay(
        case.noisy_labels,
        branches.config,
        apply_action(branches.gr_state, AuditAction.stop(), branches.config).transcript,
    )
    if group_replayed.audited_attributes != branches.gr_state.audited_attributes:
        raise AssertionError("real-data group transcript replay mismatch")
    return rows, summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DIRS),
        default=["waterbirds"],
        help="datasets to validate; default is the faster Waterbirds development block",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "three_action_real_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "three_action_real_data_mvp_v1",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "budget_fraction": BUDGET_FRACTION,
        "per_action_budget_fraction": ACTION_BUDGET_FRACTION,
        "max_rounds": MAX_ROUNDS,
        "tree_depth": TREE_DEPTH,
        "cost_ratios_group_to_label": list(COST_RATIOS),
        "lambda_grid": list(LAMBDA_GRID),
        "task_ranking": "legal NoiseScore recomputed after each label batch",
        "group_audit": "class-balanced public validation sequence",
        "attribute_model": "sparse logistic regression with one-class prevalence fallback",
        "utility": "WGA delta - lambda * normalized accumulated cost",
        "private_boundary": "private manifests only for purchased responses, test WGA, offline oracle",
        "warning": "realized-feedback finite tree; not a deployable policy or confirmation test",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        root = DATASET_DIRS[dataset]
        if not root.exists():
            raise FileNotFoundError(f"dataset cache does not exist: {root}")
        for noise_name in NOISE_NAMES[dataset]:
            for seed in args.seeds:
                print(f"[real-mvp] {dataset} {noise_name} seed={seed}", flush=True)
                case = _load_case(dataset, root, noise_name, seed)
                for cost_ratio in COST_RATIOS:
                    print(f"  cost ratio={cost_ratio:g}", flush=True)
                    rows, summaries = _run_case(case, cost_ratio)
                    all_rows.extend(
                        [{"dataset": dataset, "noise_name": noise_name, "seed": seed, **row} for row in rows]
                    )
                    all_summaries.extend(summaries)
    states = pd.DataFrame(all_rows)
    summaries = pd.DataFrame(all_summaries)
    states.to_csv(args.output_dir / "states.csv", index=False)
    summaries.to_csv(args.output_dir / "summary.csv", index=False)
    summary = {
        "status": "ok",
        "state_rows": int(len(states)),
        "summary_rows": int(len(summaries)),
        "oracle_action_frequency": summaries["oracle_action"].value_counts().to_dict(),
        "headroom_mean": float(summaries["headroom_vs_best_fixed"].mean()),
        "headroom_median": float(summaries["headroom_vs_best_fixed"].median()),
        "premature_stop_rate": float((states["classification"] == "premature_stop").mean()),
        "bad_continue_rate": float((states["classification"] == "bad_continue").mean()),
        "unresolved_rate": None,
        "certificate_status": "not_evaluated_in_private_oracle_audit",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
