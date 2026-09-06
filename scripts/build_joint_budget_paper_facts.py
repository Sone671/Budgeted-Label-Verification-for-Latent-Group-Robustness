#!/usr/bin/env python
"""Build the single machine-readable source for joint-budget paper numbers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.convex_two_action import (
    verify_convex_logistic_embedding,
    verify_nonorthogonal_logistic_stability,
)
from robust_verify.cost_budget_analysis import paired_summary
from robust_verify.three_action_theory import (
    verify_capacity_batch_three_world_extension,
    verify_finite_horizon_bellman_recovery,
    verify_per_label_batch_three_world_extension,
    verify_sharp_three_world_wga_template,
)
from robust_verify.two_action_theory import verify_same_marginal_lower_bound


JOINT_ROOT = ROOT / "joint_group_label_budget"
PHASE10_ANALYSIS = JOINT_ROOT / "outputs" / "phase10_cost_budget_map" / "analysis"
PHASE11_ANALYSIS = JOINT_ROOT / "outputs" / "phase11_acs_income" / "analysis"
PHASE11_RESULTS = (
    JOINT_ROOT / "outputs" / "phase11_acs_income" / "full" / "results.csv"
)
PHASE10_RESULTS = {
    dataset: JOINT_ROOT
    / "outputs"
    / "phase10_cost_budget_map"
    / dataset
    / "results.csv"
    for dataset in ("waterbirds", "celeba")
}
THEORY_FILES = (
    JOINT_ROOT / "SAME_MARGINAL_BINARY_AUDIT_THEOREM_ZH.md",
    JOINT_ROOT / "CONVEX_LOGISTIC_EMBEDDING_ZH.md",
    JOINT_ROOT / "NONORTHOGONAL_STABILITY_ZH.md",
    JOINT_ROOT / "RECOVERY_GUARANTEE_ZH.md",
    JOINT_ROOT / "THREE_WORLD_THEORY_DRAFT_ZH.md",
    JOINT_ROOT / "THREE_ACTION_COMPLETE_RESULTS_ZH.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _effect(record: pd.Series) -> dict[str, Any]:
    result = {
        "dataset": str(record["dataset"]),
        "noise_name": str(record["noise_name"]),
        "mean": float(record["strict_cost_mean"]),
        "ci_low": float(record["strict_cost_ci_low"]),
        "ci_high": float(record["strict_cost_ci_high"]),
        "mean_pp": 100.0 * float(record["strict_cost_mean"]),
        "ci_low_pp": 100.0 * float(record["strict_cost_ci_low"]),
        "ci_high_pp": 100.0 * float(record["strict_cost_ci_high"]),
        "p_value": float(record["strict_cost_p_value"]),
        "holm_p_value": float(record["strict_cost_p_holm"]),
        "cell_label": str(record["cell_label"]),
    }
    mechanism_columns = {
        "same_label_mean": "information_value_mean",
        "same_label_ci_low": "information_value_ci_low",
        "same_label_ci_high": "information_value_ci_high",
        "label_opportunity_cost_mean": "label_opportunity_cost_mean",
        "label_opportunity_cost_ci_low": "label_opportunity_cost_ci_low",
        "label_opportunity_cost_ci_high": "label_opportunity_cost_ci_high",
    }
    for source, target in mechanism_columns.items():
        if source in record.index and pd.notna(record[source]):
            result[target] = float(record[source])
            result[f"{target}_pp"] = 100.0 * float(record[source])
    return result


def _frozen_average_accuracy_effects(paths: list[Path]) -> list[dict[str, Any]]:
    """Summarize the paired average-accuracy contrast at the locked transfer cell."""

    columns = [
        "role",
        "dataset",
        "noise_name",
        "seed",
        "budget_fraction",
        "group_to_label_cost_ratio",
        "target_group_action_share",
        "average_accuracy",
    ]
    frame = pd.concat(
        [pd.read_csv(path, usecols=columns) for path in paths],
        ignore_index=True,
    )
    frame = frame[
        np.isclose(frame["budget_fraction"], 0.02)
        & np.isclose(frame["group_to_label_cost_ratio"], 0.1)
    ].copy()
    joint = frame[
        frame["role"].astype(str).eq("joint_sparse_attribute")
        & np.isclose(frame["target_group_action_share"], 0.5)
    ].copy()
    strict = frame[
        frame["role"].astype(str).eq("label_only_total_budget")
        & np.isclose(frame["target_group_action_share"], 0.5)
    ].copy()
    pair_columns = [
        "dataset",
        "noise_name",
        "seed",
        "budget_fraction",
        "group_to_label_cost_ratio",
    ]
    for role, subset in (("joint", joint), ("strict", strict)):
        if subset.duplicated(pair_columns, keep=False).any():
            raise ValueError(f"duplicate {role} average-accuracy rows at frozen cell")
    paired = joint[pair_columns + ["average_accuracy"]].rename(
        columns={"average_accuracy": "joint_average_accuracy"}
    )
    paired = paired.merge(
        strict[pair_columns + ["average_accuracy"]].rename(
            columns={"average_accuracy": "strict_average_accuracy"}
        ),
        on=pair_columns,
        how="left",
        validate="one_to_one",
    )
    if paired["strict_average_accuracy"].isna().any():
        raise ValueError("a frozen joint row lacks its strict average-accuracy control")
    paired["delta"] = (
        paired["joint_average_accuracy"] - paired["strict_average_accuracy"]
    )

    effects: list[dict[str, Any]] = []
    for (dataset, noise), group in paired.groupby(
        ["dataset", "noise_name"], sort=True
    ):
        summary = paired_summary(group["delta"].to_numpy(dtype=np.float64))
        effects.append(
            {
                "dataset": str(dataset),
                "noise_name": str(noise),
                **summary,
                "mean_pp": 100.0 * float(summary["mean"]),
                "ci_low_pp": 100.0 * float(summary["ci_low"]),
                "ci_high_pp": 100.0 * float(summary["ci_high"]),
            }
        )
    return effects


def _phase10_facts(
    summary: pd.DataFrame,
    decision: dict[str, Any],
    average_accuracy_effects: list[dict[str, Any]],
) -> dict[str, Any]:
    key_columns = [
        "noise_name",
        "budget_fraction",
        "group_to_label_cost_ratio",
        "target_group_action_share",
    ]
    stable = summary[
        (summary["strict_cost_mean"] > 0.0)
        & (summary["strict_cost_p_holm"] < 0.05)
    ]
    stable_sets = []
    for dataset in ("waterbirds", "celeba"):
        subset = stable[stable["dataset"].astype(str).eq(dataset)]
        stable_sets.append(set(map(tuple, subset[key_columns].itertuples(index=False, name=None))))
    common_stable = stable_sets[0].intersection(stable_sets[1])

    frozen = summary[
        np.isclose(summary["budget_fraction"], 0.02)
        & np.isclose(summary["group_to_label_cost_ratio"], 0.1)
        & np.isclose(summary["target_group_action_share"], 0.5)
    ].sort_values(["dataset", "noise_name"])
    if len(frozen) != 4:
        raise ValueError("the Phase-10 development-frozen cell must have four rows")

    labels: dict[str, dict[str, int]] = {}
    for dataset, group in summary.groupby("dataset", sort=True):
        counts = group["cell_label"].value_counts()
        labels[str(dataset)] = {
            name: int(counts.get(name, 0))
            for name in ("beneficial", "harmful", "unresolved", "incomplete")
        }

    controls: dict[str, Any] = {}
    for dataset, path in PHASE10_RESULTS.items():
        frame = pd.read_csv(path, usecols=["role"])
        counts = frame["role"].astype(str).value_counts()
        secondary = int(counts.get("label_only_same_label_count", 0))
        controls[dataset] = {
            "joint_rows": int(counts.get("joint_sparse_attribute", 0)),
            "strict_label_only_rows": int(counts.get("label_only_total_budget", 0)),
            "same_label_rows": secondary,
            "expected_same_label_rows": 900,
            "same_label_complete": secondary == 900,
        }

    return {
        "decision": str(decision["decision"]),
        "primary_cell_count": int(len(summary)),
        "mechanism_complete_cell_count": int(
            (summary["same_label_n_seeds"].astype(int) == 10).sum()
        ),
        "cell_labels_by_dataset": labels,
        "common_raw_beneficial_cell_count": int(
            decision["common_beneficial_cell_count"]
        ),
        "common_holm_stable_positive_cell_count": int(len(common_stable)),
        "common_holm_stable_noises": sorted({str(values[0]) for values in common_stable}),
        "development_frozen_setting": {
            "budget_fraction": 0.02,
            "group_to_label_cost_ratio": 0.1,
            "target_group_action_share": 0.5,
            "effects": [_effect(record) for _, record in frozen.iterrows()],
            "average_accuracy_effects": average_accuracy_effects,
        },
        "result_row_integrity": controls,
    }


def _phase11_facts(
    summary: pd.DataFrame,
    decision: dict[str, Any],
    average_accuracy_effects: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(summary) != 2:
        raise ValueError("Phase-11 must contain exactly two noise rows")
    return {
        "decision": str(decision["decision"]),
        "conjunctive_success": bool(decision["conjunctive_success"]),
        "beneficial_noises": list(decision["beneficial_noises"]),
        "integrity": decision["integrity"],
        "effects": [_effect(record) for _, record in summary.sort_values("noise_name").iterrows()],
        "average_accuracy_effects": average_accuracy_effects,
    }


def build_facts() -> dict[str, Any]:
    phase10_summary_path = PHASE10_ANALYSIS / "cell_summary.csv"
    phase10_decision_path = PHASE10_ANALYSIS / "decision.json"
    phase11_summary_path = PHASE11_ANALYSIS / "cell_summary.csv"
    phase11_decision_path = PHASE11_ANALYSIS / "decision.json"

    phase10_summary = pd.read_csv(phase10_summary_path)
    phase11_summary = pd.read_csv(phase11_summary_path)
    phase10_decision = json.loads(phase10_decision_path.read_text(encoding="utf-8"))
    phase11_decision = json.loads(phase11_decision_path.read_text(encoding="utf-8"))
    phase10_average_effects = _frozen_average_accuracy_effects(
        list(PHASE10_RESULTS.values())
    )
    phase11_average_effects = _frozen_average_accuracy_effects([PHASE11_RESULTS])

    source_paths = (
        phase10_summary_path,
        phase10_decision_path,
        phase11_summary_path,
        phase11_decision_path,
        *PHASE10_RESULTS.values(),
        PHASE11_RESULTS,
        *THEORY_FILES,
    )
    lower_bound = verify_same_marginal_lower_bound()
    convex = verify_convex_logistic_embedding()
    stability = verify_nonorthogonal_logistic_stability()
    sharp_three_world = verify_sharp_three_world_wga_template()
    capacity_batch_three_world = verify_capacity_batch_three_world_extension()
    per_label_batch_three_world = verify_per_label_batch_three_world_extension()
    finite_horizon_bellman = verify_finite_horizon_bellman_recovery()
    return {
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
        "theory": {
            "same_marginal_lower_bound": lower_bound,
            "convex_logistic_embedding": convex,
            "nonorthogonal_stability": stability,
            "sharp_three_world": sharp_three_world,
            "capacity_batch_three_world": capacity_batch_three_world,
            "per_label_batch_three_world": per_label_batch_three_world,
            "finite_horizon_bellman_recovery": finite_horizon_bellman,
            "recovery_regret_multiplier": 2,
            "exact_identification_gap_threshold": "strictly greater than 2E",
            "label_only_safety_gap_threshold": "strictly greater than 2E",
        },
        "phase10": _phase10_facts(
            phase10_summary,
            phase10_decision,
            phase10_average_effects,
        ),
        "phase11": _phase11_facts(
            phase11_summary,
            phase11_decision,
            phase11_average_effects,
        ),
    }


def _latex_macros(facts: dict[str, Any]) -> str:
    theory = facts["theory"]
    lower = theory["same_marginal_lower_bound"]
    stability = theory["nonorthogonal_stability"]
    sharp_three_world = theory["sharp_three_world"]
    capacity_batch_three_world = theory["capacity_batch_three_world"]
    per_label_batch_three_world = theory["per_label_batch_three_world"]
    phase10 = facts["phase10"]
    phase11_effects = {
        record["noise_name"]: record for record in facts["phase11"]["effects"]
    }
    phase10_frozen_effects = {
        (record["dataset"], record["noise_name"]): record
        for record in phase10["development_frozen_setting"]["effects"]
    }
    phase10_average_effects = {
        (record["dataset"], record["noise_name"]): record
        for record in phase10["development_frozen_setting"]["average_accuracy_effects"]
    }
    phase11_average_effects = {
        record["noise_name"]: record
        for record in facts["phase11"]["average_accuracy_effects"]
    }
    uniform = phase11_effects["uniform20"]
    minority = phase11_effects["minority_high_40"]
    uniform_average = phase11_average_effects["uniform20"]
    minority_average = phase11_average_effects["minority_high_40"]
    macros = {
        "JointTheoryPolicyVertices": lower["deterministic_policy_count"],
        "JointTheoryMinimaxRegret": lower["tight_randomized_worst_regret_exact"],
        "JointTheoryEpsilonStar": f'{stability["certified_spectral_radius"]:.6f}',
        "JointTheoryPerturbedFits": stability[
            "perturbed_full_retraining_fits_checked"
        ],
        "ThreeWorldSharpRegret": sharp_three_world["tight_minimax_regret"],
        "ThreeWorldCapacityPolicyCount": capacity_batch_three_world[
            "deterministic_policy_count"
        ],
        "ThreeWorldCapacityRegret": capacity_batch_three_world[
            "tight_minimax_regret"
        ],
        "ThreeWorldPerLabelPolicyCount": per_label_batch_three_world[
            "deterministic_policy_count"
        ],
        "ThreeWorldPerLabelRegret": per_label_batch_three_world[
            "tight_minimax_regret"
        ],
        "PhaseTenPrimaryCells": phase10["primary_cell_count"],
        "PhaseTenCommonRaw": phase10["common_raw_beneficial_cell_count"],
        "PhaseTenCommonHolm": phase10[
            "common_holm_stable_positive_cell_count"
        ],
        "ACSUniformMeanPP": f'{uniform["mean_pp"]:+.3f}',
        "ACSUniformLowPP": f'{uniform["ci_low_pp"]:+.3f}',
        "ACSUniformHighPP": f'{uniform["ci_high_pp"]:+.3f}',
        "ACSMinorityMeanPP": f'{minority["mean_pp"]:+.3f}',
        "ACSMinorityLowPP": f'{minority["ci_low_pp"]:+.3f}',
        "ACSMinorityHighPP": f'{minority["ci_high_pp"]:+.3f}',
        "ACSUniformAvgMeanPP": f'{uniform_average["mean_pp"]:+.3f}',
        "ACSUniformAvgLowPP": f'{uniform_average["ci_low_pp"]:+.3f}',
        "ACSUniformAvgHighPP": f'{uniform_average["ci_high_pp"]:+.3f}',
        "ACSMinorityAvgMeanPP": f'{minority_average["mean_pp"]:+.3f}',
        "ACSMinorityAvgLowPP": f'{minority_average["ci_low_pp"]:+.3f}',
        "ACSMinorityAvgHighPP": f'{minority_average["ci_high_pp"]:+.3f}',
    }
    dataset_names = {
        "waterbirds": "Waterbirds",
        "celeba": "CelebA",
        "acs_income": "ACS",
    }
    noise_names = {
        "uniform20": "Uniform",
        "minority_high_40": "Minority",
    }
    for dataset, dataset_name in dataset_names.items():
        for noise, noise_name in noise_names.items():
            if dataset == "acs_income":
                wga = phase11_effects[noise]
                average = phase11_average_effects[noise]
            else:
                wga = phase10_frozen_effects[(dataset, noise)]
                average = phase10_average_effects[(dataset, noise)]
            macros[f"Frozen{dataset_name}{noise_name}WGAEffect"] = (
                f'{wga["mean_pp"]:+.3f}\\,[{wga["ci_low_pp"]:+.3f},'
                f'{wga["ci_high_pp"]:+.3f}]'
            )
            macros[f"Frozen{dataset_name}{noise_name}AvgEffect"] = (
                f'{average["mean_pp"]:+.3f}\\,[{average["ci_low_pp"]:+.3f},'
                f'{average["ci_high_pp"]:+.3f}]'
            )
    lines = [
        "% Generated by scripts/build_joint_budget_paper_facts.py; do not edit."
    ]
    lines.extend(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=JOINT_ROOT / "paper_assets" / "paper_facts.json",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=JOINT_ROOT / "paper_assets" / "paper_facts.tex",
    )
    args = parser.parse_args()
    facts = build_facts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(_latex_macros(facts), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
