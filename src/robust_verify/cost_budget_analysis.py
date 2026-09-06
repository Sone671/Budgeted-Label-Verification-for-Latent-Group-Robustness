"""Locked paired analysis for the Phase-10 fixed cost--budget map."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


CELL_COLUMNS = [
    "dataset",
    "noise_name",
    "budget_fraction",
    "group_to_label_cost_ratio",
    "target_group_action_share",
]
PAIR_COLUMNS = CELL_COLUMNS + ["seed"]


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving NaN positions."""

    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return adjusted
    order = finite_positions[np.argsort(values[finite_positions], kind="stable")]
    running = 0.0
    total = len(order)
    for rank, position in enumerate(order):
        candidate = min(1.0, (total - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def paired_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize a seed-paired contrast with a two-sided 95% t interval."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean()) if n else math.nan
    if n < 2:
        return {
            "n_seeds": n,
            "mean": mean,
            "std": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_value": math.nan,
        }
    std = float(values.std(ddof=1))
    standard_error = std / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    if standard_error == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        p_value = float(stats.ttest_1samp(values, popmean=0.0).pvalue)
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - critical * standard_error,
        "ci_high": mean + critical * standard_error,
        "p_value": p_value,
    }


def _unique_role(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    subset = frame[frame["role"].astype(str) == role].copy()
    duplicates = subset.duplicated(PAIR_COLUMNS, keep=False)
    if duplicates.any():
        examples = subset.loc[duplicates, PAIR_COLUMNS].head().to_dict("records")
        raise ValueError(f"duplicate {role} rows for Phase-10 cells: {examples}")
    return subset


def pair_phase10_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Align joint rows with strict-cost and same-label controls."""

    required = set(PAIR_COLUMNS + ["role", "wga", "audit_pool_saturated"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Phase-10 results missing columns: {missing}")

    joint = _unique_role(frame, "joint_sparse_attribute")
    strict = _unique_role(frame, "label_only_total_budget")
    same = _unique_role(frame, "label_only_same_label_count")
    if not len(joint):
        return pd.DataFrame()

    joint_fields = PAIR_COLUMNS + [
        "wga",
        "audit_pool_saturated",
        "planned_group_audit_count",
        "planned_label_verification_count",
        "planned_realized_group_action_share",
    ]
    paired = joint[joint_fields].rename(columns={"wga": "joint_wga"})
    paired = paired.merge(
        strict[PAIR_COLUMNS + ["wga"]].rename(columns={"wga": "strict_control_wga"}),
        on=PAIR_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    paired = paired.merge(
        same[PAIR_COLUMNS + ["wga"]].rename(columns={"wga": "same_label_wga"}),
        on=PAIR_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if paired["strict_control_wga"].isna().any():
        raise ValueError("at least one joint cell lacks its strict-cost control")
    paired["strict_cost_delta"] = paired["joint_wga"] - paired["strict_control_wga"]
    paired["same_label_delta"] = paired["joint_wga"] - paired["same_label_wga"]
    paired["label_opportunity_cost"] = (
        paired["strict_control_wga"] - paired["same_label_wga"]
    )
    paired["decomposition_residual"] = paired["strict_cost_delta"] - (
        paired["same_label_delta"] - paired["label_opportunity_cost"]
    )
    return paired.sort_values(PAIR_COLUMNS, kind="stable").reset_index(drop=True)


def summarize_phase10_pairs(
    paired: pd.DataFrame,
    *,
    expected_seed_count: int = 10,
) -> pd.DataFrame:
    """Apply the frozen cell labels without interpreting incomplete cells."""

    rows: list[dict[str, Any]] = []
    for keys, group in paired.groupby(CELL_COLUMNS, sort=True, dropna=False):
        primary = paired_summary(group["strict_cost_delta"].to_numpy())
        secondary = paired_summary(group["same_label_delta"].to_numpy())
        opportunity = paired_summary(group["label_opportunity_cost"].to_numpy())
        residual = group["decomposition_residual"].to_numpy(dtype=np.float64)
        residual = residual[np.isfinite(residual)]
        row = dict(zip(CELL_COLUMNS, keys))
        row.update(
            {
                "n_seeds": primary["n_seeds"],
                "strict_cost_mean": primary["mean"],
                "strict_cost_std": primary["std"],
                "strict_cost_ci_low": primary["ci_low"],
                "strict_cost_ci_high": primary["ci_high"],
                "strict_cost_p_value": primary["p_value"],
                "same_label_n_seeds": secondary["n_seeds"],
                "same_label_mean": secondary["mean"],
                "same_label_ci_low": secondary["ci_low"],
                "same_label_ci_high": secondary["ci_high"],
                "label_opportunity_cost_mean": opportunity["mean"],
                "label_opportunity_cost_n_seeds": opportunity["n_seeds"],
                "label_opportunity_cost_ci_low": opportunity["ci_low"],
                "label_opportunity_cost_ci_high": opportunity["ci_high"],
                "decomposition_residual_max_abs": (
                    float(np.max(np.abs(residual))) if len(residual) else math.nan
                ),
                "audit_pool_saturated": bool(group["audit_pool_saturated"].any()),
                "planned_group_audit_count_min": int(
                    group["planned_group_audit_count"].min()
                ),
                "planned_group_audit_count_max": int(
                    group["planned_group_audit_count"].max()
                ),
                "planned_label_count_min": int(
                    group["planned_label_verification_count"].min()
                ),
                "planned_label_count_max": int(
                    group["planned_label_verification_count"].max()
                ),
            }
        )
        if int(primary["n_seeds"]) != expected_seed_count:
            label = "incomplete"
        elif row["audit_pool_saturated"]:
            label = "saturated"
        elif float(primary["ci_low"]) > 0.0:
            label = "beneficial"
        elif float(primary["ci_high"]) < 0.0:
            label = "harmful"
        else:
            label = "unresolved"
        row["cell_label"] = label
        rows.append(row)

    summary = pd.DataFrame(rows)
    if len(summary):
        summary["strict_cost_p_holm"] = holm_adjust(
            summary["strict_cost_p_value"].to_numpy(dtype=np.float64)
        )
    return summary.sort_values(CELL_COLUMNS, kind="stable").reset_index(drop=True)


def break_even_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Report a tested break-even ratio only for monotone complete sequences."""

    group_columns = [
        "dataset",
        "noise_name",
        "budget_fraction",
        "target_group_action_share",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in summary.groupby(group_columns, sort=True, dropna=False):
        ordered = group.sort_values("group_to_label_cost_ratio")
        complete = bool((ordered["cell_label"] != "incomplete").all())
        means = ordered["strict_cost_mean"].to_numpy(dtype=np.float64)
        signs = means >= 0.0
        monotone = bool(not np.any((~signs[:-1]) & signs[1:])) if len(signs) > 1 else True
        nonnegative = ordered.loc[signs, "group_to_label_cost_ratio"]
        break_even = (
            float(nonnegative.max())
            if complete and monotone and len(nonnegative)
            else math.nan
        )
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "complete": complete,
                "mean_signs_monotone": monotone,
                "break_even_cost_ratio": break_even,
                "tested_ratio_count": len(ordered),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def phase10_gate(summary: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the locked cross-dataset advance gate."""

    required_datasets = ("waterbirds", "celeba")
    present_datasets = set(summary["dataset"].astype(str)) if len(summary) else set()
    missing_datasets = sorted(set(required_datasets).difference(present_datasets))
    incomplete = (
        bool((summary["cell_label"] == "incomplete").any())
        if len(summary)
        else True
    ) or bool(missing_datasets)
    beneficial_keys: dict[str, set[tuple[Any, ...]]] = {}
    key_columns = [
        "noise_name",
        "budget_fraction",
        "group_to_label_cost_ratio",
        "target_group_action_share",
    ]
    for dataset in required_datasets:
        subset = summary[
            (summary["dataset"].astype(str) == dataset)
            & (summary["cell_label"].astype(str) == "beneficial")
        ]
        beneficial_keys[dataset] = set(
            map(tuple, subset[key_columns].itertuples(index=False, name=None))
        )
    common = beneficial_keys[required_datasets[0]].intersection(
        beneficial_keys[required_datasets[1]]
    )
    dataset_has_benefit = {
        dataset: bool(beneficial_keys[dataset]) for dataset in required_datasets
    }
    if incomplete:
        decision = "incomplete"
    elif all(dataset_has_benefit.values()) and common:
        decision = "advance_to_new_dataset_preregistration"
    else:
        decision = "stop_independent_empirical_main_conference_expansion"
    return {
        "decision": decision,
        "incomplete": incomplete,
        "missing_required_datasets": missing_datasets,
        "dataset_has_beneficial_cell": dataset_has_benefit,
        "common_beneficial_cell_count": len(common),
        "common_beneficial_cells": [
            dict(zip(key_columns, values)) for values in sorted(common)
        ],
    }


def phase11_confirmation_decision(summary: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen two-noise conjunctive ACS confirmation rule."""

    expected_noises = {"uniform20", "minority_high_40"}
    observed_noises = set(summary["noise_name"].astype(str)) if len(summary) else set()
    missing_noises = sorted(expected_noises.difference(observed_noises))
    incomplete = bool(missing_noises) or bool(
        (summary["cell_label"].astype(str) == "incomplete").any()
    )
    beneficial_noises = sorted(
        summary.loc[
            summary["cell_label"].astype(str) == "beneficial", "noise_name"
        ].astype(str)
    )
    if incomplete:
        decision = "incomplete"
    elif set(beneficial_noises) == expected_noises:
        decision = "strong_external_confirmation"
    elif beneficial_noises:
        decision = "partial_external_confirmation"
    else:
        decision = "external_confirmation_failed"
    return {
        "decision": decision,
        "incomplete": incomplete,
        "missing_noises": missing_noises,
        "beneficial_noises": beneficial_noises,
        "conjunctive_success": decision == "strong_external_confirmation",
    }
