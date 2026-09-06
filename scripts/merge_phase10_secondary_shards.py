#!/usr/bin/env python
"""Merge disjoint Phase-10 same-label shards without touching primary rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "dataset",
    "noise_name",
    "seed",
    "budget_fraction",
    "group_to_label_cost_ratio",
    "target_group_action_share",
    "role",
]
SECONDARY_ROLE = "label_only_same_label_count"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-secondary", type=int, default=900)
    args = parser.parse_args()

    original = pd.read_csv(args.main_results)
    primary_before = original[original["role"].astype(str) != SECONDARY_ROLE].copy()
    secondary_frames = [
        original[original["role"].astype(str) == SECONDARY_ROLE].copy()
    ]
    for shard_path in args.shards:
        shard = pd.read_csv(shard_path)
        secondary_frames.append(
            shard[shard["role"].astype(str) == SECONDARY_ROLE].copy()
        )

    secondary = pd.concat(secondary_frames, ignore_index=True, sort=False)
    comparison_columns = [
        "wga",
        "label_query_count",
        "planned_label_verification_count",
        "total_cost_used",
    ]
    conflicts = []
    for keys, group in secondary.groupby(KEY_COLUMNS, sort=False, dropna=False):
        if len(group) <= 1:
            continue
        for column in comparison_columns:
            if group[column].nunique(dropna=False) > 1:
                conflicts.append({"key": list(keys), "column": column})
    if conflicts:
        raise ValueError(f"conflicting duplicate secondary cells: {conflicts[:5]}")

    secondary = secondary.drop_duplicates(KEY_COLUMNS, keep="last")
    if len(secondary) != args.expected_secondary:
        raise ValueError(
            f"expected {args.expected_secondary} unique secondary rows, "
            f"found {len(secondary)}"
        )
    missing_queries = [
        str(path)
        for path in secondary["query_path"].astype(str)
        if not path or path == "nan" or not Path(path).exists()
    ]
    if missing_queries:
        raise FileNotFoundError(
            f"secondary rows reference missing query artifacts: {missing_queries[:5]}"
        )

    merged = pd.concat([primary_before, secondary], ignore_index=True, sort=False)
    if merged.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("merged Phase-10 table contains duplicate cell keys")
    primary_after = merged[merged["role"].astype(str) != SECONDARY_ROLE]
    if len(primary_after) != len(primary_before):
        raise AssertionError("merge changed the number of non-secondary rows")
    before_keys = set(map(tuple, primary_before[KEY_COLUMNS].itertuples(index=False, name=None)))
    after_keys = set(map(tuple, primary_after[KEY_COLUMNS].itertuples(index=False, name=None)))
    if before_keys != after_keys:
        raise AssertionError("merge changed primary/control cell keys")

    merged = merged.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    temporary = args.main_results.with_suffix(args.main_results.suffix + ".tmp")
    merged.to_csv(temporary, index=False)
    os.replace(temporary, args.main_results)
    print(
        json.dumps(
            {
                "status": "pass",
                "main_results": str(args.main_results),
                "total_rows": int(len(merged)),
                "non_secondary_rows_preserved": int(len(primary_after)),
                "unique_secondary_rows": int(len(secondary)),
                "secondary_query_artifacts_present": int(len(secondary)),
                "duplicate_cell_keys": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
