#!/usr/bin/env python
"""Strictly compare a completed frozen result matrix with its reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["noise_name", "seed", "method", "budget_fraction"]


def compare(reference_path: str | Path, candidate_path: str | Path, *, tolerance: float = 1e-7) -> dict:
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = [column for column in KEYS if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing comparison keys: {missing}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} has duplicate comparison keys")

    merged = reference.merge(candidate, on=KEYS, how="outer", suffixes=("_reference", "_candidate"), indicator=True)
    common = merged[merged["_merge"] == "both"]
    numeric = sorted(
        set(reference.select_dtypes(include=[np.number]).columns)
        .intersection(candidate.select_dtypes(include=[np.number]).columns)
        .difference(KEYS)
    )
    string_columns = sorted(
        set(reference.select_dtypes(exclude=[np.number]).columns)
        .intersection(candidate.select_dtypes(exclude=[np.number]).columns)
        .difference(KEYS)
    )
    max_error = 0.0
    numeric_mismatch_columns: dict[str, int] = {}
    for column in numeric:
        difference = np.abs(
            common[f"{column}_reference"].to_numpy(float)
            - common[f"{column}_candidate"].to_numpy(float)
        )
        max_error = max(max_error, float(np.nanmax(difference)) if len(difference) else 0.0)
        mismatches = int((difference > tolerance).sum())
        if mismatches:
            numeric_mismatch_columns[column] = mismatches
    string_mismatch_columns: dict[str, int] = {}
    for column in string_columns:
        mismatch = common[f"{column}_reference"].fillna("<NA>") != common[f"{column}_candidate"].fillna("<NA>")
        if mismatch.any():
            string_mismatch_columns[column] = int(mismatch.sum())
    report = {
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "matched_rows": len(common),
        "missing_or_extra_rows": int((merged["_merge"] != "both").sum()),
        "numeric_columns_compared": numeric,
        "max_absolute_numeric_error": max_error,
        "numeric_mismatch_columns": numeric_mismatch_columns,
        "string_mismatch_columns": string_mismatch_columns,
        "tolerance": tolerance,
    }
    report["passed"] = bool(
        report["missing_or_extra_rows"] == 0
        and not numeric_mismatch_columns
        and not string_mismatch_columns
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly compare two P0 result matrices.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, tolerance=args.tolerance)
    report.update({"reference": str(Path(args.reference)), "candidate": str(Path(args.candidate))})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
