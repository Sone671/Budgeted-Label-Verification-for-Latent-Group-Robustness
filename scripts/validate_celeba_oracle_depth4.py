"""Run a depth-four private Oracle audit on cached real-data tasks.

This is an offline diagnostic.  At each state the driver exposes one legal
NoiseScore label batch (R) and one legal class-balanced attribute batch (G),
applies the *realized* private responses, retrains after R, and then solves the
resulting finite tree by backward induction.  It is not a deployable policy:
response-outcome branches are not enumerated and private test WGA is used only
to score the Oracle tree.
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

from robust_verify.action_value import build_legal_state_features
from robust_verify.joint_budget import (
    estimate_soft_group_error,
    fit_attribute_probability_model,
    joint_repair_scores,
    select_class_balanced_audits,
)
from robust_verify.sequential_audit import (
    AuditAction,
    SequentialAuditConfig,
    SequentialAuditState,
    apply_action,
    make_batch_action,
    replay,
    state_to_json,
)
from robust_verify.trajectory_oracle import TreeNode, TreeTransition, solve_dynamic_program
from robust_verify.training import predict_from_state, train_linear_head
from robust_verify.utils import budget_to_count, class_balanced_accuracy

# Import the existing real-data loader and legal/private-boundary helpers.  The
# imported module has no side effects because its CLI is guarded by __main__.
from validate_three_action_real_data import (  # type: ignore
    BUDGET_FRACTION,
    DATASET_DIRS,
    NOISE_NAMES,
    _apply_group,
    _apply_task,
    _legal_noise_scores,
    _load_case,
    _worst_group,
)


ROOT = Path(__file__).resolve().parents[1]
TREE_DEPTH = 4
ACTION_BUDGET_FRACTION = 0.25
DEFAULT_OUTPUT = ROOT / "outputs" / "three_action_real_validation" / "real_oracle_depth4"


@dataclass
class NodeData:
    node_id: str
    state: SequentialAuditState
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray
    metrics: dict[str, float]
    depth: int
    attribute_bacc: float


def _state_hash(state: SequentialAuditState) -> str:
    return hashlib.sha256(state_to_json(state).encode("utf-8")).hexdigest()[:16]


def _batch_count(
    state: SequentialAuditState,
    config: SequentialAuditConfig,
    kind: str,
    target: int,
) -> int:
    used = state.verified_ids if kind == "verify_task" else state.audited_attribute_ids
    pool_size = config.task_pool_size if kind == "verify_task" else config.attribute_pool_size
    available = pool_size - len(used)
    unit_cost = config.task_label_cost if kind == "verify_task" else config.attribute_cost
    affordable = int(state.remaining_budget(config) // unit_cost)
    return max(0, min(int(target), int(available), affordable))


def _make_group_action(
    audit_sequence: np.ndarray,
    state: SequentialAuditState,
    count: int,
    config: SequentialAuditConfig,
) -> AuditAction | None:
    available = [
        int(index)
        for index in audit_sequence.tolist()
        if int(index) not in state.audited_attribute_ids
    ]
    if count <= 0 or not available:
        return None
    return make_batch_action("annotate_group", available[:count], config)


def _fit_state(case: Any, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Train from the shared initial state and return train/validation scores."""
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
    _, train_probabilities = predict_from_state(
        probe.best_state, case.train_features, input_dim, num_classes, case.training_config
    )
    test_predictions, _ = predict_from_state(
        probe.best_state, case.test_features, input_dim, num_classes, case.training_config
    )
    metrics = case.evaluator.evaluate(case.test_ids, test_predictions)
    val_probabilities = predict_from_state(
        probe.best_state, case.val_features, input_dim, num_classes, case.training_config
    )[1]
    return train_probabilities, val_probabilities, metrics


def _attribute_signal(case: Any, node: NodeData) -> tuple[np.ndarray, float] | None:
    if not node.state.audited_attributes:
        return None
    positions = np.asarray(sorted(node.state.audited_attributes), dtype=np.int64)
    values = np.asarray([node.state.audited_attributes[int(i)] for i in positions], dtype=np.int64)
    model = fit_attribute_probability_model(case.val_features[positions], values, seed=case.seed)
    val_attribute_probability = model.predict_probability(case.val_features)
    train_attribute_probability = model.predict_probability(case.train_features)
    group_error, _ = estimate_soft_group_error(
        node.val_probabilities, case.val_labels, val_attribute_probability
    )
    return train_attribute_probability, class_balanced_accuracy(
        case.val_private["place"].to_numpy(dtype=np.int64),
        (val_attribute_probability >= 0.5).astype(np.int64),
    )


def _candidate_actions(
    case: Any,
    node: NodeData,
    config: SequentialAuditConfig,
    audit_sequence: np.ndarray,
    task_target: int,
    group_target: int,
) -> list[AuditAction]:
    actions: list[AuditAction] = []
    task_count = _batch_count(node.state, config, "verify_task", task_target)
    if task_count:
        scores = _legal_noise_scores(node.train_probabilities, node.state.current_labels)
        signal = _attribute_signal(case, node)
        if signal is not None:
            train_attribute_probability, _ = signal
            positions = np.asarray(sorted(node.state.audited_attributes), dtype=np.int64)
            values = np.asarray([node.state.audited_attributes[int(i)] for i in positions], dtype=np.int64)
            model = fit_attribute_probability_model(case.val_features[positions], values, seed=case.seed)
            val_attribute_probability = model.predict_probability(case.val_features)
            group_error, _ = estimate_soft_group_error(
                node.val_probabilities, case.val_labels, val_attribute_probability
            )
            scores, _ = joint_repair_scores(
                scores, node.train_probabilities, train_attribute_probability, group_error
            )
        available = np.asarray(
            [i for i in range(config.task_pool_size) if i not in node.state.verified_ids],
            dtype=np.int64,
        )
        order = np.lexsort((available, -np.asarray(scores)[available]))
        actions.append(make_batch_action("verify_task", available[order[:task_count]], config))
    group_count = _batch_count(node.state, config, "annotate_group", group_target)
    group_action = _make_group_action(audit_sequence, node.state, group_count, config)
    if group_action is not None:
        actions.append(group_action)
    return actions


def _run_case(
    case: Any,
    cost_ratio: float,
    *,
    tree_depth: int = TREE_DEPTH,
    action_budget_fraction: float = ACTION_BUDGET_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tree_depth < 1:
        raise ValueError("tree_depth must be positive")
    if not 0.0 < action_budget_fraction <= 1.0:
        raise ValueError("action_budget_fraction must lie in (0, 1]")
    n_train = len(case.train_ids)
    budget_count = budget_to_count(BUDGET_FRACTION, n_train)
    task_target = max(1, int(budget_count * action_budget_fraction))
    group_target = max(1, int(task_target / cost_ratio))
    config = SequentialAuditConfig(
        budget=Fraction(budget_count),
        task_label_cost=Fraction(1),
        attribute_cost=Fraction(str(cost_ratio)),
        task_pool_size=n_train,
        attribute_pool_size=len(case.val_ids),
        max_rounds=tree_depth,
    )
    audit_sequence = select_class_balanced_audits(case.val_labels, len(case.val_ids), seed=case.seed)
    initial = SequentialAuditState.initial(case.noisy_labels)
    baseline = NodeData(
        node_id="root",
        state=initial,
        train_probabilities=case.baseline_train_probabilities,
        val_probabilities=case.baseline_val_probabilities,
        metrics=case.baseline_metrics,
        depth=0,
        attribute_bacc=float("nan"),
    )
    prefix = f"{case.dataset}/{case.noise_name}/seed{case.seed}/r{cost_ratio:g}"
    nodes: dict[str, NodeData] = {"root": baseline}
    edges: dict[str, list[tuple[AuditAction, str]]] = {}
    fit_cache: dict[bytes, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {
        case.noisy_labels.tobytes(): (
            case.baseline_train_probabilities,
            case.baseline_val_probabilities,
            case.baseline_metrics,
        )
    }

    def child_node(parent: NodeData, action: AuditAction) -> NodeData:
        if action.kind == "verify_task":
            state = _apply_task(case, parent.state, action, config)
            key = state.current_labels.tobytes()
            fitted = fit_cache.get(key)
            if fitted is None:
                fitted = _fit_state(case, state.current_labels)
                fit_cache[key] = fitted
            train_probabilities, val_probabilities, metrics = fitted
        else:
            state = _apply_group(case, parent.state, action, config)
            train_probabilities, val_probabilities, metrics = (
                parent.train_probabilities,
                parent.val_probabilities,
                parent.metrics,
            )
        attr_signal = _attribute_signal(
            case,
            NodeData("tmp", state, train_probabilities, val_probabilities, metrics, parent.depth + 1, float("nan")),
        )
        attr_bacc = float(attr_signal[1]) if attr_signal is not None else float("nan")
        node_id = f"{prefix}/d{parent.depth + 1}/{action.kind}/{_state_hash(state)}"
        existing = nodes.get(node_id)
        if existing is not None:
            return existing
        child = NodeData(
            node_id=node_id,
            state=state,
            train_probabilities=train_probabilities,
            val_probabilities=val_probabilities,
            metrics=metrics,
            depth=parent.depth + 1,
            attribute_bacc=attr_bacc,
        )
        nodes[node_id] = child
        return child

    def expand(node: NodeData) -> None:
        if node.depth >= tree_depth:
            return
        actions = _candidate_actions(case, node, config, audit_sequence, task_target, group_target)
        transitions: list[tuple[AuditAction, str]] = []
        for action in actions:
            child = child_node(node, action)
            transitions.append((action, child.node_id))
        edges[node.node_id] = transitions
        for _, child_id in transitions:
            if nodes[child_id].depth == node.depth + 1:
                expand(nodes[child_id])

    expand(baseline)
    # Every action choice is scored against the parent's current WGA.  The
    # lambda-dependent cost is added when constructing the DP tree below.
    summaries: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    baseline_wga = float(case.baseline_metrics["wga"])
    for lambda_value in (0.0, 0.01, 0.02):
        tree: dict[str, TreeNode] = {}
        for node_id, node in nodes.items():
            transitions = []
            for action, child_id in edges.get(node_id, []):
                child = nodes[child_id]
                delta = float(child.metrics["wga"] - node.metrics["wga"])
                immediate = delta - lambda_value * float(action.cost) / float(config.budget)
                transitions.append(TreeTransition(action, child_id, immediate))
            tree[node_id] = TreeNode(node_id, transitions=tuple(transitions))
        depth2 = solve_dynamic_program(tree, "root", max_depth=2)
        depth4 = solve_dynamic_program(tree, "root", max_depth=tree_depth)
        root2, root4 = depth2.root, depth4.root
        root_values = {a.kind: float(v) for a, v in root4.action_values}
        summaries.append(
            {
                "dataset": case.dataset,
                "noise_name": case.noise_name,
                "seed": case.seed,
                "cost_ratio": cost_ratio,
                "lambda": lambda_value,
                "budget": float(config.budget),
                "task_batch_target": task_target,
                "group_batch_target": group_target,
                "node_count": len(nodes),
                "transition_count": sum(len(v) for v in edges.values()),
                "oracle_depth2_value": float(root2.value),
                "oracle_depth2_action": root2.action.kind,
                "oracle_depth4_value": float(root4.value),
                "oracle_depth4_action": root4.action.kind,
                "depth4_minus_depth2": float(root4.value - root2.value),
                "baseline_wga": baseline_wga,
                "oracle_root_action_values": json.dumps(root_values, sort_keys=True),
                "replay_checks": len(nodes),
                "private_boundary": "private responses/test WGA only; not deployable",
            }
        )
        for node_id, node in nodes.items():
            decision = (
                depth4.decisions[(node_id, tree_depth - node.depth)]
                if node.depth <= tree_depth
                else None
            )
            state_rows.append(
                {
                    "dataset": case.dataset,
                    "noise_name": case.noise_name,
                    "seed": case.seed,
                    "cost_ratio": cost_ratio,
                    "lambda": lambda_value,
                    "node_id": node_id,
                    "depth": node.depth,
                    "state_hash": _state_hash(node.state),
                    "verified_count": len(node.state.verified_ids),
                    "audited_attribute_count": len(node.state.audited_attributes),
                    "spent_cost": float(node.state.spent_cost),
                    "remaining_budget": float(node.state.remaining_budget(config)),
                    "current_wga": float(node.metrics["wga"]),
                    "wga_delta_from_baseline": float(node.metrics["wga"] - baseline_wga),
                    "oracle_action": decision.action.kind if decision is not None else "",
                    "oracle_value": float(decision.value) if decision is not None else 0.0,
                    "attribute_bacc": node.attribute_bacc,
                    "worst_group": _worst_group(node.metrics),
                    "transcript_kinds": ",".join(event.action.kind for event in node.state.transcript),
                }
            )

    # Replay all generated transcripts at the public state-machine boundary.
    for node in nodes.values():
        replayed = replay(case.noisy_labels, config, node.state.transcript)
        if replayed.current_labels.tolist() != node.state.current_labels.tolist():
            raise AssertionError(f"label replay mismatch at {node.node_id}")
        if replayed.audited_attributes != node.state.audited_attributes:
            raise AssertionError(f"attribute replay mismatch at {node.node_id}")
    return state_rows, summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DIRS),
        default=["celeba"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--noise-names",
        nargs="+",
        default=list(NOISE_NAMES["celeba"]),
        choices=sorted({name for names in NOISE_NAMES.values() for name in names}),
    )
    parser.add_argument("--cost-ratios", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=TREE_DEPTH,
        help="maximum number of sequential action rounds (default: 4)",
    )
    parser.add_argument(
        "--per-action-budget-fraction",
        type=float,
        default=ACTION_BUDGET_FRACTION,
        help="fraction of the total budget targeted by each action batch (default: 0.25)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": f"three_action_real_oracle_depth{args.tree_depth}_v1",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "noise_names": args.noise_names,
        "cost_ratios_group_to_label": args.cost_ratios,
        "budget_fraction": BUDGET_FRACTION,
        "per_action_budget_fraction": args.per_action_budget_fraction,
        "tree_depth": args.tree_depth,
        "candidate_actions_per_state": "one legal R (NoiseScore or joint-repair after G) plus one legal G (class-balanced public validation sequence)",
        "feedback_branching": "single realized private response per action; response-outcome enumeration is intentionally absent",
        "retraining": "from shared initial state after every R action; G leaves model unchanged",
        "utility": "incremental test WGA - lambda * normalized action cost",
        "private_boundary": "private manifests only for purchased responses, test WGA and offline Oracle scoring",
        "warning": "diagnostic finite realized-feedback Oracle; not a deployable policy or confirmation test",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        root = DATASET_DIRS[dataset]
        for noise_name in args.noise_names:
            if noise_name not in NOISE_NAMES[dataset]:
                raise ValueError(f"noise {noise_name!r} is not configured for {dataset}")
            for seed in args.seeds:
                print(f"[oracle-depth4] {dataset} {noise_name} seed={seed}", flush=True)
                case = _load_case(dataset, root, noise_name, seed)
                for cost_ratio in args.cost_ratios:
                    print(f"  cost ratio={cost_ratio:g}", flush=True)
                    state_rows, case_summaries = _run_case(
                        case,
                        float(cost_ratio),
                        tree_depth=args.tree_depth,
                        action_budget_fraction=args.per_action_budget_fraction,
                    )
                    rows.extend(state_rows)
                    summaries.extend(case_summaries)
    pd.DataFrame(rows).to_csv(args.output_dir / "states.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    summary = {
        "status": "ok",
        "state_rows": len(rows),
        "summary_rows": len(summaries),
        "node_count_mean": float(pd.DataFrame(summaries)["node_count"].mean()),
        "transition_count_mean": float(pd.DataFrame(summaries)["transition_count"].mean()),
        "oracle_depth4_action_frequency": pd.DataFrame(summaries)["oracle_depth4_action"].value_counts().to_dict(),
        "oracle_depth2_action_frequency": pd.DataFrame(summaries)["oracle_depth2_action"].value_counts().to_dict(),
        "depth4_minus_depth2_mean": float(pd.DataFrame(summaries)["depth4_minus_depth2"].mean()),
        "depth4_minus_depth2_median": float(pd.DataFrame(summaries)["depth4_minus_depth2"].median()),
        "replay_status": "passed",
        "certificate_status": "not_evaluated_in_private_oracle_audit",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
