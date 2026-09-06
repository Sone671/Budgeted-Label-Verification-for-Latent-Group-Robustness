from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_IDENTITY_COLUMNS = [
    "male", "female", "LGBTQ", "christian", "muslim",
    "other_religions", "black", "white",
]


def load_civilcomments_metadata(data_config: dict[str, Any]) -> pd.DataFrame:
    root = Path(data_config["root"])

    csv_path = Path("all_data_with_identities.csv")
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CivilComments data not found: {csv_path}. "
            f"Download from https://wilds.stanford.edu/downloads/ and extract to {root}"
        )

    identity = data_config.get("identity", "black")
    if identity not in _IDENTITY_COLUMNS:
        raise ValueError(
            f"Unknown identity {identity!r}. Available: {_IDENTITY_COLUMNS}"
        )

    raw = pd.read_csv(csv_path)

    # Label: toxicity >= 0.5 is toxic (1)
    clean_label = (raw["toxicity"] >= 0.5).astype(int)

    # Spurious: identity mention >= 0.5
    place = (raw[identity] >= 0.5).astype(int)

    # Split mapping
    split_map = {"train": 0, "val": 1, "test": 2}
    split_id = raw["split"].map(split_map).fillna(-1).astype(int)
    unknown = split_id == -1
    if unknown.any():
        bad = sorted(raw.loc[unknown, "split"].unique())
        raise ValueError(f"Unrecognized split values: {bad}")

    # Optional subsample training set for MVP experiments
    max_train = data_config.get("max_train")
    if max_train is not None:
        train_mask = split_id == 0
        train_idx = raw.index[train_mask]
        if len(train_idx) > max_train:
            import numpy as _np
            rng = _np.random.default_rng(42)
            keep_idx = rng.choice(train_idx, size=max_train, replace=False)
            drop_idx = train_idx.difference(keep_idx)
            raw = raw.drop(drop_idx)
            split_id = split_id[raw.index]
            clean_label = clean_label[raw.index]
            place = place[raw.index]
            print(f"Subsampled training from {len(train_idx):,} to {max_train:,}")

    canonical = pd.DataFrame(
        {
            "sample_id": range(len(raw)),
            "image_relpath": "",  # text datasets have no image path
            "comment_text": raw["comment_text"].astype(str),
            "toxicity_score": raw["toxicity"].to_numpy(dtype=float),
            "clean_label": clean_label.to_numpy(dtype=int),
            "place": place.to_numpy(dtype=int),
            "split_id": split_id.to_numpy(dtype=int),
        }
    )
    canonical["split"] = canonical["split_id"].map({0: "train", 1: "val", 2: "test"})
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
            f"unknown minority_mode: {minority_mode!r}"
        )

    return canonical.sort_values("sample_id").reset_index(drop=True)
