from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


IMAGE_COLUMNS = ("img_filename", "image", "filename", "path")
LABEL_COLUMNS = ("y", "label", "target")
ATTRIBUTE_COLUMNS = ("place", "a", "background", "attribute")
SPLIT_COLUMNS = ("split", "split_id")


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(
        f"Could not find {role} column. Tried {candidates}. "
        f"Available columns: {list(frame.columns)}"
    )


def load_waterbirds_metadata(data_config: dict[str, Any]) -> pd.DataFrame:
    root = Path(data_config["root"])
    metadata_path = Path(data_config.get("metadata_csv", "metadata.csv"))
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path

    if not metadata_path.exists():
        raise FileNotFoundError(f"Waterbirds metadata not found: {metadata_path}")

    raw = pd.read_csv(metadata_path)
    image_col = _find_column(raw, IMAGE_COLUMNS, "image")
    label_col = _find_column(raw, LABEL_COLUMNS, "label")
    attribute_col = _find_column(raw, ATTRIBUTE_COLUMNS, "spurious attribute")
    split_col = _find_column(raw, SPLIT_COLUMNS, "split")

    split_values = data_config.get(
        "split_values", {"train": 0, "val": 1, "test": 2}
    )
    inverse_split = {int(value): name for name, value in split_values.items()}

    canonical = pd.DataFrame(
        {
            "sample_id": range(len(raw)),
            "image_relpath": raw[image_col].astype(str),
            "clean_label": raw[label_col].astype(int),
            "place": raw[attribute_col].astype(int),
            "split_id": raw[split_col].astype(int),
        }
    )
    canonical["split"] = canonical["split_id"].map(inverse_split)

    if canonical["split"].isna().any():
        bad = sorted(canonical.loc[canonical["split"].isna(), "split_id"].unique())
        raise ValueError(f"Unrecognized split IDs in metadata: {bad}")

    canonical["group"] = 2 * canonical["clean_label"] + canonical["place"]
    canonical["is_minority"] = (
        canonical["clean_label"] != canonical["place"]
    ).astype(int)

    image_paths = canonical["image_relpath"].map(lambda value: root / value)
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image files from metadata were not found under {root}. "
            f"First missing paths:\n{preview}"
        )

    return canonical.sort_values("sample_id").reset_index(drop=True)


def write_canonical_splits(
    canonical: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        subset = canonical.loc[canonical["split"] == split].copy()
        path = output_dir / f"{split}.csv"
        subset.to_csv(path, index=False)
        paths[split] = path
    return paths
