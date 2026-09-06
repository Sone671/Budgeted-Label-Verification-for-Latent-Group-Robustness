#!/usr/bin/env python
"""Phase-B last-accepted rollback screen on the existing pilot artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.transfer_budget import (
    assert_legal_feature_names,
    balanced_accuracy_present_classes,
    fit_predict_seed_block_bootstrap,
    selective_decisions,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "transferable_budget_control_pilot"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _key(value: float) -> float:
    return round(float(value), 8)


def _augment_targets(meta: pd.DataFrame, design: dict[str, Any], outcomes: pd.DataFrame) -> pd.DataFrame:
    budgets = tuple(float(value) for value in design["budgets"])
    budget_keys = {_key(value): index for index, value in enumerate(budgets)}
    outcome_lookup = {
        (
            str(row.dataset),
            str(row.noise_name),
            int(row.seed),
            _key(row.budget_fraction),
        ): float(row.wga)
        for row in outcomes.itertuples(index=False)
    }
    result = meta.copy()
    previous_wga: list[float] = []
    previous_budget: list[float] = []
    accept_utility: list[float] = []
    accept_label: list[int] = []
    for row in result.itertuples(index=False):
        current = float(row.current_budget_fraction_key)
        index = budget_keys[_key(current)]
        prior = 0.0 if index == 0 else budgets[index - 1]
        key = (str(row.dataset), str(row.noise_name), int(row.seed))
        current_value = outcome_lookup[(*key, _key(current))]
        prior_value = outcome_lookup[(*key, _key(prior))]
        delta_pp = 100.0 * (current_value - prior_value)
        cost_pp = float(design["cost_wga_pp_per_budget_percent"]) * 100.0 * (current - prior)
        utility = delta_pp - cost_pp
        previous_budget.append(prior)
        previous_wga.append(prior_value)
        accept_utility.append(utility)
        accept_label.append(int(utility > 0.0))
    result["previous_budget_fraction_key"] = previous_budget
    result["target_previous_wga"] = previous_wga
    result["target_accept_net_utility_pp"] = accept_utility
    result["target_accept_current"] = accept_label
    return result


def _policy_for_dataset(frame: pd.DataFrame, scout_budget: float) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for (noise, seed), condition in frame.groupby(["noise_name", "seed"], sort=True):
        condition = condition.sort_values("current_budget_fraction_key", kind="stable")
        first = condition.iloc[0]
        accepted_budget = 0.0
        accepted_wga = float(first["target_baseline_wga"])
        spent_budget = 0.0
        trace: list[str] = []
        for row in condition.itertuples(index=False):
            current = float(row.current_budget_fraction_key)
            spent_budget = current
            accept_action = str(row.acceptance_action)
            trace.append(f"{current:g}:accept={accept_action}")
            if accept_action != "accept":
                break
            accepted_budget = current
            accepted_wga = float(row.target_current_wga)
            continuation_action = str(row.continuation_action)
            trace[-1] += f",next={continuation_action}"
            if continuation_action != "continue":
                break
            accepted_budget = float(row.next_budget_fraction_key)
            accepted_wga = float(row.target_next_wga)
            spent_budget = accepted_budget

        baseline = float(first["target_baseline_wga"])
        fixed_cap = float(first["target_fixed_cap_wga"])
        best_small = float(first["target_best_small_wga"])
        scout_wga = float(first["target_current_wga"])
        rows.append(
            {
                "dataset": first["dataset"],
                "noise_name": noise,
                "seed": int(seed),
                "selected_budget_fraction": accepted_budget,
                "spent_budget_fraction": spent_budget,
                "selected_wga": accepted_wga,
                "baseline_wga": baseline,
                "scout_wga": scout_wga,
                "fixed_cap_wga": fixed_cap,
                "best_small_wga": best_small,
                "delta_wga_pp": 100.0 * (accepted_wga - baseline),
                "delta_vs_fixed_cap_pp": 100.0 * (accepted_wga - fixed_cap),
                "best_small_regret_pp": 100.0 * max(0.0, best_small - accepted_wga),
                "negative_utility": bool(accepted_wga < baseline),
                "scout_negative_utility": bool(scout_wga < baseline),
                "fixed_cap_negative_utility": bool(fixed_cap < baseline),
                "advanced_beyond_scout": bool(accepted_budget > scout_budget),
                "trace": "|".join(trace),
            }
        )
    condition_frame = pd.DataFrame(rows)
    fixed_scout_negative = float(condition_frame["scout_negative_utility"].mean())
    selected_negative = float(condition_frame["negative_utility"].mean())
    metrics = {
        "condition_count": int(len(condition_frame)),
        "mean_selected_budget_fraction": float(condition_frame["selected_budget_fraction"].mean()),
        "mean_spent_budget_fraction": float(condition_frame["spent_budget_fraction"].mean()),
        "mean_delta_wga_pp": float(condition_frame["delta_wga_pp"].mean()),
        "mean_delta_vs_fixed_cap_pp": float(condition_frame["delta_vs_fixed_cap_pp"].mean()),
        "selected_negative_rate": selected_negative,
        "fixed_cap_negative_rate": float(condition_frame["fixed_cap_negative_utility"].mean()),
        "scout_negative_rate": fixed_scout_negative,
        "scout_negative_rate_reduction": fixed_scout_negative - selected_negative,
        "mean_best_small_regret_pp": float(condition_frame["best_small_regret_pp"].mean()),
        "advance_condition_rate": float(condition_frame["advanced_beyond_scout"].mean()),
    }
    return metrics, rows


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase-B Rollback Budget Controller Report",
        "",
        "This screen reuses only the already opened three-seed pilot artifacts.",
        "",
        "| Held-out | Accept BAcc | Accept coverage | Bad accept | Bad continue | Selected negative | Scout negative | Scout reduction | Best-small regret | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for dataset, result in summary["held_out"].items():
        accept = result["acceptance"]
        continuation = result["continuation"]
        policy = result["policy"]
        lines.append(
            f"| {dataset} | {accept['balanced_accuracy']:.3f} | {accept['selective_coverage']:.3f} | "
            f"{accept['erroneous_acceptance_rate']:.3f} | {continuation['erroneous_continue_rate']:.3f} | "
            f"{policy['selected_negative_rate']:.3f} | {policy['scout_negative_rate']:.3f} | "
            f"{policy['scout_negative_rate_reduction']:.3f} | {policy['mean_best_small_regret_pp']:.3f} pp | "
            f"{'YES' if result['passed'] else 'NO'} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            f"Held-out datasets passed: {decision['held_out_datasets_passed']}/3.",
            "",
            f"Proceed to old seeds 3--9: **{'YES' if decision['old_seed_expansion_allowed'] else 'NO'}**.",
            "",
            "New seed/dataset migration: **NO** for this screen.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: Any) -> Path:
    output_root = Path(args.output_root).resolve()
    design = json.loads((output_root / "design.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(output_root / "development_meta_dataset.csv")
    outcomes = pd.read_csv(output_root / "private_development_outcomes.csv")
    feature_names = tuple(str(value) for value in design["feature_names"])
    assert_legal_feature_names(feature_names)
    meta = _augment_targets(meta, design, outcomes)
    summary: dict[str, Any] = {
        "protocol": "transferable_marginal_utility_rollback_screen_v1",
        "feature_names": list(feature_names),
        "held_out": {},
    }
    all_decisions: list[pd.DataFrame] = []
    all_conditions: list[dict[str, Any]] = []
    datasets = sorted(meta["dataset"].unique().tolist())
    for fold_index, held_out in enumerate(datasets):
        train = meta.loc[meta["dataset"] != held_out].copy()
        test = meta.loc[meta["dataset"] == held_out].copy()
        train_groups = (
            train["dataset"].astype(str)
            + ":"
            + train["noise_name"].astype(str)
            + ":"
            + train["seed"].astype(int).astype(str)
        )
        x_train = train.loc[:, feature_names].to_numpy(dtype=np.float64)
        x_test = test.loc[:, feature_names].to_numpy(dtype=np.float64)
        accept_samples = fit_predict_seed_block_bootstrap(
            x_train,
            train["target_accept_current"].to_numpy(dtype=np.int64),
            train_groups.to_numpy(),
            x_test,
            replicates=int(design["bootstrap_replicates"]),
            seed=int(design["split_seed"]) + 30011 * fold_index,
        )
        continue_samples = fit_predict_seed_block_bootstrap(
            x_train,
            train["target_continue"].to_numpy(dtype=np.int64),
            train_groups.to_numpy(),
            x_test,
            replicates=int(design["bootstrap_replicates"]),
            seed=int(design["split_seed"]) + 50021 * fold_index,
        )
        accept_decisions = selective_decisions(
            accept_samples,
            lower_quantile=float(design["bootstrap_quantiles"][0]),
            upper_quantile=float(design["bootstrap_quantiles"][1]),
            continue_threshold=float(design["continue_threshold"]),
            stop_threshold=float(design["stop_threshold"]),
        )
        continue_decisions = selective_decisions(
            continue_samples,
            lower_quantile=float(design["bootstrap_quantiles"][0]),
            upper_quantile=float(design["bootstrap_quantiles"][1]),
            continue_threshold=float(design["continue_threshold"]),
            stop_threshold=float(design["stop_threshold"]),
        )
        # The shared selective primitive calls the positive action "continue";
        # in this first-stage model that same action means "accept current".
        acceptance_actions = [
            {"continue": "accept", "stop": "reject", "abstain": "abstain"}[d.action]
            for d in accept_decisions
        ]
        test["acceptance_action"] = acceptance_actions
        test["acceptance_probability_median"] = [d.median_probability for d in accept_decisions]
        test["acceptance_probability_lower"] = [d.lower_probability for d in accept_decisions]
        test["acceptance_probability_upper"] = [d.upper_probability for d in accept_decisions]
        test["continuation_action"] = [d.action for d in continue_decisions]
        test["continuation_probability_median"] = [d.median_probability for d in continue_decisions]
        test["continuation_probability_lower"] = [d.lower_probability for d in continue_decisions]
        test["continuation_probability_upper"] = [d.upper_probability for d in continue_decisions]
        all_decisions.append(test)

        accept_truth = test["target_accept_current"].to_numpy(dtype=np.int64)
        accept_prediction = np.asarray(
            [int(action == "accept") for action in acceptance_actions], dtype=np.int64
        )
        accept_covered = np.asarray([action != "abstain" for action in acceptance_actions])
        accept_mask = accept_prediction == 1
        continuation_truth = test["target_continue"].to_numpy(dtype=np.int64)
        continue_mask = np.asarray([d.action == "continue" for d in continue_decisions])
        acceptance_stats = {
            "row_count": int(len(test)),
            "positive_rate": float(accept_truth.mean()),
            "balanced_accuracy": balanced_accuracy_present_classes(accept_truth, accept_prediction),
            "selective_coverage": float(accept_covered.mean()),
            "accept_action_rate": float(accept_mask.mean()),
            "erroneous_acceptance_rate": (
                float((accept_truth[accept_mask] == 0).mean()) if accept_mask.any() else 1.0
            ),
            "abstention_rate": float((~accept_covered).mean()),
        }
        continuation_stats = {
            "row_count": int(len(test)),
            "positive_rate": float(continuation_truth.mean()),
            "erroneous_continue_rate": (
                float((continuation_truth[continue_mask] == 0).mean())
                if continue_mask.any()
                else 1.0
            ),
            "continue_action_rate": float(continue_mask.mean()),
        }
        policy, condition_rows = _policy_for_dataset(
            test, float(design["mandatory_scout_budget"])
        )
        all_conditions.extend(condition_rows)
        gates = {
            "accept_balanced_accuracy_at_least_0_65": acceptance_stats["balanced_accuracy"] >= 0.65,
            "accept_selective_coverage_at_least_0_30": acceptance_stats["selective_coverage"] >= 0.30,
            "erroneous_acceptance_at_most_0_10": acceptance_stats["erroneous_acceptance_rate"] <= 0.10,
            "erroneous_continue_at_most_0_10": continuation_stats["erroneous_continue_rate"] <= 0.10,
            "advance_condition_rate_at_least_0_20": policy["advance_condition_rate"] >= 0.20,
            "best_small_regret_at_most_1pp": policy["mean_best_small_regret_pp"] <= 1.0,
            "scout_negative_rate_reduction_at_least_0_30": (
                policy["scout_negative_rate_reduction"] >= 0.30
                if held_out == "waterbirds"
                else True
            ),
        }
        summary["held_out"][held_out] = {
            "training_datasets": sorted(train["dataset"].unique().tolist()),
            "acceptance": acceptance_stats,
            "continuation": continuation_stats,
            "policy": policy,
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
        print(
            f"[rollback-fold] held_out={held_out} passed={all(gates.values())} "
            f"accept_bacc={acceptance_stats['balanced_accuracy']:.3f} "
            f"bad_accept={acceptance_stats['erroneous_acceptance_rate']:.3f} "
            f"bad_continue={continuation_stats['erroneous_continue_rate']:.3f}",
            flush=True,
        )

    pass_count = sum(bool(value["passed"]) for value in summary["held_out"].values())
    summary["decision"] = {
        "stage": "phase_b_existing_three_seed_pilot",
        "held_out_datasets_passed": int(pass_count),
        "old_seed_expansion_allowed": bool(pass_count >= 2),
        "new_seed_or_dataset_migration_allowed": False,
    }
    decision_path = output_root / "rollback_decisions.csv"
    condition_path = output_root / "rollback_policy_conditions.csv"
    summary_path = output_root / "rollback_summary.json"
    _atomic_csv(pd.concat(all_decisions, ignore_index=True), decision_path)
    _atomic_csv(pd.DataFrame(all_conditions), condition_path)
    _atomic_json(summary, summary_path)
    _write_report(summary, output_root / "ROLLBACK_REPORT.md")
    print(f"[rollback-complete] {summary_path}", flush=True)
    return summary_path


def parse_args() -> Any:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
