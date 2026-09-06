"""Folktables ACSIncome metadata with a private binary sex attribute."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = ("AGEP", "WKHP")
CATEGORICAL_FEATURES = (
    "COW",
    "SCHL",
    "MAR",
    "OCCP",
    "POBP",
    "RELP",
    "RAC1P",
)
SOURCE_COLUMNS = (
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    "SEX",
    "PINCP",
    "PWGTP",
    "ST",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stratified_splits(groups: np.ndarray, seed: int) -> np.ndarray:
    indices = np.arange(len(groups), dtype=np.int64)
    train, heldout = train_test_split(
        indices,
        test_size=0.40,
        random_state=seed,
        shuffle=True,
        stratify=groups,
    )
    validation, test = train_test_split(
        heldout,
        test_size=0.50,
        random_state=seed + 1,
        shuffle=True,
        stratify=groups[heldout],
    )
    split = np.empty(len(groups), dtype=object)
    split[train] = "train"
    split[validation] = "val"
    split[test] = "test"
    return split.astype(str)


def load_acs_income_metadata(data_config: dict[str, Any]) -> pd.DataFrame:
    """Load the frozen 2018 1-Year California ACSIncome cohort.

    The filter matches ``folktables.acs.adult_filter``.  ``SEX`` is converted
    to the private binary ``place`` field and is deliberately omitted from the
    returned public feature columns.
    """

    root = Path(data_config["root"])
    source_path = Path(
        data_config.get("source_csv", "2018/1-Year/psam_p06.csv")
    )
    if not source_path.is_absolute():
        source_path = root / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"ACS person PUMS file not found: {source_path}")

    expected_hash = str(data_config.get("source_sha256", "")).upper()
    if expected_hash:
        actual_hash = file_sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"ACS source hash mismatch: expected {expected_hash}, got {actual_hash}"
            )

    raw = pd.read_csv(source_path, usecols=list(SOURCE_COLUMNS))
    state_code = int(data_config.get("state_code", 6))
    if not bool((raw["ST"].astype(int) == state_code).all()):
        raise ValueError(f"ACS source contains rows outside state code {state_code}")

    # Exact Folktables ACSIncome / Adult-style cohort filter.
    filtered = raw[
        (raw["AGEP"] > 16)
        & (raw["PINCP"] > 100)
        & (raw["WKHP"] > 0)
        & (raw["PWGTP"] >= 1)
    ].copy()
    if filtered[list(SOURCE_COLUMNS)].isna().any().any():
        missing = filtered[list(SOURCE_COLUMNS)].isna().sum()
        raise ValueError(
            f"frozen ACS columns contain missing values: {missing[missing > 0].to_dict()}"
        )
    if not set(filtered["SEX"].astype(int).unique()).issubset({1, 2}):
        raise ValueError("ACS SEX must use the frozen 1=male, 2=female coding")

    clean_label = (filtered["PINCP"] > 50_000).astype(np.int64).to_numpy()
    place = (filtered["SEX"].astype(int) == 2).astype(np.int64).to_numpy()
    group = 2 * clean_label + place
    split = _stratified_splits(group, int(data_config.get("split_seed", 20260730)))

    canonical = pd.DataFrame(
        {
            "sample_id": np.arange(len(filtered), dtype=np.int64),
            "image_relpath": "",
            "clean_label": clean_label,
            "place": place,
            "split": split,
        }
    )
    canonical["split_id"] = canonical["split"].map(
        {"train": 0, "val": 1, "test": 2}
    )
    canonical["group"] = group
    # Predeclared rare high-income female group.
    canonical["is_minority"] = (group == 3).astype(np.int64)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        canonical[column] = filtered[column].to_numpy()

    if "SEX" in canonical.columns:
        raise AssertionError("private SEX must not be present in canonical feature columns")
    return canonical.sort_values("sample_id").reset_index(drop=True)
