#!/usr/bin/env python
"""Outcome-free integrity preflight for the draft Phase-11 ACS confirmation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.config import load_config, output_layout
from robust_verify.cost_budget_map import allocate_action_mix
from robust_verify.data.acs_income import file_sha256
from robust_verify.features import load_features
from robust_verify.utils import budget_to_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config["project"]["status"] not in {
        "draft_preflight_only",
        "frozen_before_acs_wga",
    }:
        raise ValueError("unexpected Phase-11 project status")
    layout = output_layout(config["project"]["output_dir"])
    source = Path(config["data"]["root"]) / config["data"]["source_csv"]
    source_hash = file_sha256(source)
    expected_hash = str(config["data"]["source_sha256"]).upper()

    train_features, train_ids = load_features(layout["features"] / "train.npz")
    val_features, val_ids = load_features(layout["features"] / "val.npz")
    test_features, test_ids = load_features(layout["features"] / "test.npz")
    split_features = {
        "train": train_features,
        "val": val_features,
        "test": test_features,
    }
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    group_counts: dict[str, dict[str, int]] = {}
    public_leaks: dict[str, list[str]] = {}
    forbidden = {"clean_label", "place", "group", "is_minority", "SEX"}
    for split in ("train", "val", "test"):
        canonical = pd.read_csv(layout["canonical"] / f"{split}.csv")
        counts = canonical["group"].value_counts().sort_index()
        group_counts[split] = {str(int(key)): int(value) for key, value in counts.items()}
        if set(counts.index.astype(int)) != {0, 1, 2, 3}:
            raise ValueError(f"{split} does not contain all four frozen groups")
        if not np.array_equal(
            canonical["sample_id"].to_numpy(dtype=np.int64), split_ids[split]
        ):
            raise ValueError(f"{split} feature IDs do not align with canonical data")
        if not np.isfinite(split_features[split]).all():
            raise ValueError(f"{split} features contain non-finite values")

        public_path = layout["manifests"] / f"{split}_public.csv"
        if public_path.exists():
            public_columns = set(pd.read_csv(public_path, nrows=1).columns)
            public_leaks[split] = sorted(public_columns.intersection(forbidden))
            if public_leaks[split]:
                raise ValueError(f"private fields leaked into {split} public manifest")

    train_id_set = set(map(int, train_ids))
    for noise in config["scope"]["noises"]:
        for seed in config["scope"]["seeds"]:
            prefix = f"{noise}_seed{seed}"
            public_path = layout["manifests"] / f"{prefix}_train_public.csv"
            private_path = layout["manifests"] / f"{prefix}_train_private.csv"
            if not public_path.exists() or not private_path.exists():
                raise FileNotFoundError(f"missing frozen train manifest for {prefix}")
            public = pd.read_csv(public_path)
            private = pd.read_csv(private_path)
            leak_key = f"train:{prefix}"
            public_leaks[leak_key] = sorted(set(public.columns).intersection(forbidden))
            if public_leaks[leak_key]:
                raise ValueError(f"private fields leaked into {prefix} public manifest")
            if set(map(int, public["sample_id"])) != train_id_set:
                raise ValueError(f"public train IDs do not align for {prefix}")
            if set(map(int, private["sample_id"])) != train_id_set:
                raise ValueError(f"private train IDs do not align for {prefix}")
            if not set(public["noisy_label"].astype(int).unique()).issubset({0, 1}):
                raise ValueError(f"non-binary noisy labels in {prefix}")

    if len({array.shape[1] for array in split_features.values()}) != 1:
        raise ValueError("ACS feature dimensions differ across splits")

    budget_fraction = float(config["phase11"]["primary_joint_cell"]["total_budget_fraction"])
    ratio = float(config["phase11"]["primary_joint_cell"]["group_to_label_cost_ratio"])
    share = float(config["phase11"]["primary_joint_cell"]["target_group_action_share"])
    capacity = budget_to_count(budget_fraction, len(train_ids))
    allocation = allocate_action_mix(
        capacity,
        share,
        ratio,
        max_group_audits=len(val_ids),
    )
    if allocation.audit_pool_saturated:
        raise ValueError("frozen ACS primary cell saturates the validation audit pool")
    if allocation.total_cost_used != capacity:
        raise ValueError("frozen ACS primary cell does not use exact total cost")

    stage1_exists = (layout["stage1"] / "results.csv").exists()
    if config["project"]["status"] == "draft_preflight_only" and stage1_exists:
        raise ValueError("test WGA artifacts exist before Phase-11 config freeze")

    report = {
        "status": "pass",
        "project_status": config["project"]["status"],
        "source_path": str(source),
        "source_sha256": source_hash,
        "source_hash_matches": source_hash == expected_hash,
        "folktables_version": config["data"]["folktables_version"],
        "split_counts": {name: int(len(ids)) for name, ids in split_ids.items()},
        "group_counts": group_counts,
        "feature_dimension": int(train_features.shape[1]),
        "feature_shapes": {
            name: [int(value) for value in array.shape]
            for name, array in split_features.items()
        },
        "public_private_field_leaks": public_leaks,
        "budget_capacity": allocation.budget_capacity,
        "group_audit_count": allocation.group_audit_count,
        "label_verification_count": allocation.label_verification_count,
        "realized_group_action_share": allocation.realized_group_action_share,
        "total_cost_used": allocation.total_cost_used,
        "audit_pool_saturated": allocation.audit_pool_saturated,
        "stage1_wga_artifacts_exist": stage1_exists,
    }
    if not report["source_hash_matches"]:
        raise ValueError("ACS source hash does not match the frozen config")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
