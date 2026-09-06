from __future__ import annotations

import pandas as pd

from robust_verify.config import ensure_output_layout
from robust_verify.data.noise import (
    inject_noise,
    write_validation_and_test_manifests,
)
from robust_verify.data.waterbirds import (
    load_waterbirds_metadata,
    write_canonical_splits,
)
from robust_verify.features import extract_split_features
from robust_verify.utils import save_json


def _load_metadata(data_config: dict):
    dataset = data_config.get("dataset", "waterbirds")
    if dataset == "waterbirds":
        return load_waterbirds_metadata(data_config)
    if dataset == "celeba":
        from robust_verify.data.celeba import load_celeba_metadata
        return load_celeba_metadata(data_config)
    if dataset == "civilcomments":
        from robust_verify.data.civilcomments import load_civilcomments_metadata
        return load_civilcomments_metadata(data_config)
    if dataset == "cifar10n":
        from robust_verify.data.cifar_n import load_cifar10n_metadata
        return load_cifar10n_metadata(data_config)
    if dataset == "acs_income":
        from robust_verify.data.acs_income import load_acs_income_metadata

        return load_acs_income_metadata(data_config)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _apply_label_swap(canonical: pd.DataFrame) -> pd.DataFrame:
    """Swap clean_label 0↔1 and place 0↔1 to preserve group semantics.

    In Waterbirds-style encoding, ``group = 2*clean_label + place`` and
    ``is_minority = (clean_label != place)``.  Swapping both label and place
    changes only the class *encoding* while keeping the minority-group
    structure identical.  After re-encoding, groups are mapped:
    g0↔g3, g1↔g2 (equivalently, just recomputed from swapped components).
    """
    canonical = canonical.copy()
    if "place" in canonical.columns:
        canonical["place"] = 1 - canonical["place"].astype(int)
    canonical["clean_label"] = 1 - canonical["clean_label"].astype(int)
    # recompute derived columns
    if "place" in canonical.columns:
        canonical["group"] = 2 * canonical["clean_label"] + canonical["place"].astype(int)
        if "is_minority" in canonical.columns:
            canonical["is_minority"] = (
                canonical["clean_label"] != canonical["place"]
            ).astype(int)
    return canonical


def run_stage0(config: dict) -> None:
    layout = ensure_output_layout(config["project"]["output_dir"])
    canonical = _load_metadata(config["data"])

    if config["data"].get("label_swap", False):
        canonical = _apply_label_swap(canonical)

    split_paths = write_canonical_splits(canonical, layout["canonical"])
    write_validation_and_test_manifests(canonical, layout["manifests"])

    if not bool(config["features"].get("skip_extraction", False)):
        if config["data"].get("dataset") == "acs_income":
            from robust_verify.features import extract_acs_tabular_features

            extract_acs_tabular_features(
                canonical, layout["features"], config["features"]
            )
        else:
            for split, path in split_paths.items():
                frame = pd.read_csv(path)
                extract_split_features(
                    frame=frame,
                    data_root=config["data"]["root"],
                    output_path=layout["features"] / f"{split}.npz",
                    feature_config=config["features"],
                    seed=0,
                )

    train_frame = canonical.loc[canonical["split"] == "train"].copy()
    all_stats = []
    for noise_setting in config["noise"]["settings"]:
        noise_name = noise_setting["name"]
        for seed in config["experiment"]["seeds"]:
            public, private, stats = inject_noise(train_frame, noise_setting, int(seed))
            prefix = f"{noise_name}_seed{seed}"
            public.to_csv(layout["manifests"] / f"{prefix}_train_public.csv", index=False)
            private.to_csv(layout["manifests"] / f"{prefix}_train_private.csv", index=False)
            stats["noise_name"] = noise_name
            stats["noise_kind"] = noise_setting["kind"]
            all_stats.append(stats)

    pd.DataFrame(all_stats).to_csv(layout["root"] / "noise_statistics.csv", index=False)
    save_json(config, layout["root"] / "resolved_config.json")
