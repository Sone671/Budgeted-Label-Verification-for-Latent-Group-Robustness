from __future__ import annotations

import numpy as np
import pytest

from robust_verify.data.access import VerificationOracle
from robust_verify.data.cifar_n import (
    build_cifar10n_canonical,
    materialize_cifar10n_images,
)
from robust_verify.data.noise import inject_noise


def test_cifar10n_replay_keeps_clean_train_labels_private(tmp_path) -> None:
    canonical = build_cifar10n_canonical(
        clean_train_labels=np.array([0, 1, 2, 0]),
        noisy_train_labels=np.array([1, 1, 2, 2]),
        clean_test_labels=np.array([0, 1, 2]),
        validation_count=1,
    )

    assert canonical["sample_id"].tolist() == list(range(7))
    assert canonical["split"].value_counts().to_dict() == {"train": 4, "test": 2, "val": 1}

    train = canonical.loc[canonical["split"] == "train"].copy()
    public, private, stats = inject_noise(train, {"kind": "existing_real"}, seed=19)

    assert "clean_label" not in public.columns
    assert public["noisy_label"].tolist() == [1, 1, 2, 2]
    assert private["clean_label"].tolist() == [0, 1, 2, 0]
    assert private["is_noisy"].tolist() == [1, 0, 0, 1]
    assert stats["actual_noise_rate"] == 0.5

    private_path = tmp_path / "train_private.csv"
    private.to_csv(private_path, index=False)
    assert VerificationOracle(private_path).verify(np.array([0, 3])).tolist() == [0, 0]


def test_cifar10n_canonical_validates_alignment_and_materializes_images(tmp_path) -> None:
    with pytest.raises(ValueError, match="align"):
        build_cifar10n_canonical(
            clean_train_labels=np.array([0, 1]),
            noisy_train_labels=np.array([0]),
            clean_test_labels=np.array([0, 1]),
            validation_count=1,
        )

    materialize_cifar10n_images(
        tmp_path,
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        np.ones((1, 4, 4, 3), dtype=np.uint8) * 255,
    )
    assert (tmp_path / "images" / "train" / "00000.png").is_file()
    assert (tmp_path / "images" / "train" / "00001.png").is_file()
    assert (tmp_path / "images" / "eval" / "00000.png").is_file()
