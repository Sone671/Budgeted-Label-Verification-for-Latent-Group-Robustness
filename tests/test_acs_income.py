from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.data.acs_income import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    file_sha256,
    load_acs_income_metadata,
)
from robust_verify.features import extract_acs_tabular_features, load_features


def _write_synthetic_acs(path: Path) -> None:
    rows = []
    for label in (0, 1):
        for female in (0, 1):
            for repeat in range(20):
                rows.append(
                    {
                        "AGEP": 25 + repeat,
                        "WKHP": 20 + repeat,
                        "COW": 1 + repeat % 2,
                        "SCHL": 16 + repeat % 3,
                        "MAR": 1 + repeat % 2,
                        "OCCP": 100 + repeat % 4,
                        "POBP": 6 + repeat % 3,
                        "RELP": repeat % 3,
                        "RAC1P": 1 + repeat % 2,
                        "SEX": 2 if female else 1,
                        "PINCP": 60_000 if label else 30_000,
                        "PWGTP": 1,
                        "ST": 6,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_acs_loader_keeps_private_sex_out_of_features(tmp_path: Path) -> None:
    source = tmp_path / "acs.csv"
    _write_synthetic_acs(source)
    canonical = load_acs_income_metadata(
        {
            "root": str(tmp_path),
            "source_csv": source.name,
            "source_sha256": file_sha256(source),
            "state_code": 6,
            "split_seed": 20260730,
        }
    )
    assert len(canonical) == 80
    assert "SEX" not in canonical.columns
    assert set(canonical["place"]) == {0, 1}
    assert set(canonical["group"]) == {0, 1, 2, 3}
    assert canonical.groupby(["split", "group"]).size().min() > 0
    assert canonical.loc[canonical["group"] == 3, "is_minority"].eq(1).all()


def test_acs_features_fit_once_and_align_all_splits(tmp_path: Path) -> None:
    source = tmp_path / "acs.csv"
    _write_synthetic_acs(source)
    canonical = load_acs_income_metadata(
        {"root": str(tmp_path), "source_csv": source.name, "split_seed": 7}
    )
    output = tmp_path / "features"
    paths = extract_acs_tabular_features(
        canonical,
        output,
        {
            "numeric_columns": list(NUMERIC_FEATURES),
            "categorical_columns": list(CATEGORICAL_FEATURES),
        },
    )
    dimensions = set()
    for split, path in paths.items():
        features, sample_ids = load_features(path)
        expected_ids = canonical.loc[
            canonical["split"] == split, "sample_id"
        ].to_numpy(dtype=np.int64)
        assert np.array_equal(sample_ids, expected_ids)
        assert np.isfinite(features).all()
        dimensions.add(features.shape[1])
    assert len(dimensions) == 1
    metadata = (output / "acs_preprocessor.json").read_text(encoding="utf-8")
    assert "SEX" in metadata
    assert "categorical__SEX" not in metadata
