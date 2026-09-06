from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _attr_to_binary(series: pd.Series) -> pd.Series:
    """Convert CelebA -1/1 attribute to 0/1."""
    return (series > 0).astype(int)


def load_celeba_metadata(data_config: dict[str, Any]) -> pd.DataFrame:
    """Load CelebA metadata and produce a canonical frame.

    Expected files under ``data_config["root"]``:
      - ``list_attr_celeba.csv``  (image_id + 40 attributes, -1/1)
      - ``list_eval_partition.csv`` (image_id + partition: 0=train 1=val 2=test)
      - ``img_align_celeba/`` image directory

    Configurable via ``data_config``:
      - ``target_attr`` (default ``"Blond_Hair"``)
      - ``spurious_attr`` (default ``"Male"``)
      - ``image_subdir`` (default ``"img_align_celeba"``)
      - ``metadata_csv`` (default ``"list_attr_celeba.csv"``)
      - ``partition_csv`` (default ``"list_eval_partition.csv"``)
      - ``split_values`` (default ``{train: 0, val: 1, test: 2}``)
      - ``minority_mode`` (default ``"label_neq_place"``):
          - ``"label_neq_place"``: minority = (clean_label != place) [original behavior]
          - ``"group_id"``: minority = (group == minority_group) [for rare-group validation]
      - ``minority_group`` (required when ``minority_mode == "group_id"``):
          int 0-3, the group id to designate as the rare minority group.
    """
    root = Path(data_config["root"])

    attr_path = Path(data_config.get("metadata_csv", "list_attr_celeba.csv"))
    if not attr_path.is_absolute():
        attr_path = root / attr_path
    if not attr_path.exists():
        raise FileNotFoundError(f"CelebA attributes not found: {attr_path}")

    partition_path = Path(data_config.get("partition_csv", "list_eval_partition.csv"))
    if not partition_path.is_absolute():
        partition_path = root / partition_path
    if not partition_path.exists():
        raise FileNotFoundError(f"CelebA partition not found: {partition_path}")

    target_attr = data_config.get("target_attr", "Blond_Hair")
    spurious_attr = data_config.get("spurious_attr", "Male")
    image_subdir = data_config.get("image_subdir", "img_align_celeba")

    attrs = pd.read_csv(attr_path)
    partitions = pd.read_csv(partition_path)

    if target_attr not in attrs.columns:
        raise ValueError(
            f"Target attribute {target_attr!r} not in attribute file. "
            f"Available: {list(attrs.columns)}"
        )
    if spurious_attr not in attrs.columns:
        raise ValueError(
            f"Spurious attribute {spurious_attr!r} not in attribute file. "
            f"Available: {list(attrs.columns)}"
        )

    merged = attrs[["image_id", target_attr, spurious_attr]].merge(
        partitions[["image_id", "partition"]], on="image_id", how="inner"
    )
    if len(merged) != len(attrs):
        raise ValueError(
            f"Merge mismatch: {len(attrs)} attributes vs {len(merged)} merged"
        )

    split_values = data_config.get("split_values", {"train": 0, "val": 1, "test": 2})
    inverse_split = {int(value): name for name, value in split_values.items()}

    canonical = pd.DataFrame(
        {
            "sample_id": range(len(merged)),
            "image_relpath": merged["image_id"].map(
                lambda name: f"{image_subdir}/{name}"
            ),
            "clean_label": _attr_to_binary(merged[target_attr]).to_numpy(),
            "place": _attr_to_binary(merged[spurious_attr]).to_numpy(),
            "split_id": merged["partition"].astype(int).to_numpy(),
        }
    )
    canonical["split"] = canonical["split_id"].map(inverse_split)

    if canonical["split"].isna().any():
        bad = sorted(canonical.loc[canonical["split"].isna(), "split_id"].unique())
        raise ValueError(f"Unrecognized split IDs in metadata: {bad}")

    canonical["group"] = 2 * canonical["clean_label"] + canonical["place"]

    minority_mode = data_config.get("minority_mode", "label_neq_place")
    if minority_mode == "label_neq_place":
        canonical["is_minority"] = (
            canonical["clean_label"] != canonical["place"]
        ).astype(int)
    elif minority_mode == "group_id":
        minority_group = int(data_config["minority_group"])
        canonical["is_minority"] = (
            canonical["group"] == minority_group
        ).astype(int)
    else:
        raise ValueError(
            f"unknown minority_mode: {minority_mode!r}; "
            f"expected 'label_neq_place' or 'group_id'"
        )

    image_paths = canonical["image_relpath"].map(lambda value: root / value)
    missing = [str(path) for path in image_paths[:200] if not path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Image files not found under {root}. First missing:\n{preview}"
        )

    return canonical.sort_values("sample_id").reset_index(drop=True)
