from __future__ import annotations

import numpy as np
import pytest

from robust_verify.risk_stopping import (
    assert_nested_query_prefixes,
    audit_accepts,
    nominate_by_calibration,
    paired_stratified_bootstrap_risk_difference,
    sample_id_hash,
    stratified_validation_partition,
)


def test_stratified_partition_is_disjoint_exhaustive_and_order_stable() -> None:
    sample_ids = np.asarray([90, 10, 70, 20, 80, 30, 60, 40, 50, 0, 100, 110])
    labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    split = stratified_validation_partition(sample_ids, labels, seed=17)
    combined = np.concatenate([split.early_stop, split.calibration, split.audit])
    assert sorted(combined.tolist()) == list(range(len(sample_ids)))
    assert len(np.unique(combined)) == len(sample_ids)
    for positions in (split.early_stop, split.calibration, split.audit):
        assert set(labels[positions]) == {0, 1}

    order = np.asarray([4, 1, 9, 2, 8, 0, 11, 3, 10, 5, 7, 6])
    reordered = stratified_validation_partition(sample_ids[order], labels[order], seed=17)
    original_hashes = {
        sample_id_hash(sample_ids, positions)
        for positions in (split.early_stop, split.calibration, split.audit)
    }
    reordered_hashes = {
        sample_id_hash(sample_ids[order], positions)
        for positions in (reordered.early_stop, reordered.calibration, reordered.audit)
    }
    assert original_hashes == reordered_hashes


def test_partition_rejects_too_small_class() -> None:
    with pytest.raises(ValueError, match="at least three"):
        stratified_validation_partition(
            np.arange(5), np.asarray([0, 0, 1, 1, 1]), seed=0
        )


def test_nested_query_prefix_audit() -> None:
    assert_nested_query_prefixes(
        [np.asarray([3, 1]), np.asarray([3, 1, 8]), np.asarray([3, 1, 8, 2])]
    )
    with pytest.raises(ValueError, match="not an ordered prefix"):
        assert_nested_query_prefixes(
            [np.asarray([3, 1]), np.asarray([1, 3, 8])]
        )


def test_calibration_nomination_requires_strict_margin() -> None:
    assert nominate_by_calibration(0.8, 1.0, minimum_improvement=0.1)
    assert not nominate_by_calibration(0.9, 1.0, minimum_improvement=0.1)
    assert not nominate_by_calibration(1.1, 1.0)


def test_paired_bootstrap_detects_uniformly_better_predictions() -> None:
    labels = np.asarray([0, 1] * 60, dtype=np.int64)
    reference = np.tile(np.asarray([[0.60, 0.40], [0.40, 0.60]]), (60, 1))
    candidate = np.tile(np.asarray([[0.90, 0.10], [0.10, 0.90]]), (60, 1))
    result = paired_stratified_bootstrap_risk_difference(
        candidate,
        reference,
        labels,
        alphas=(0.1, 0.2),
        temperature=5.0,
        confidence=0.95,
        replicates=100,
        seed=3,
    )
    assert result.observed_difference < 0.0
    assert result.upper < 0.0
    assert audit_accepts(result)


def test_audit_rejects_no_improvement() -> None:
    labels = np.asarray([0, 1] * 20, dtype=np.int64)
    probabilities = np.tile(np.asarray([[0.75, 0.25], [0.25, 0.75]]), (20, 1))
    result = paired_stratified_bootstrap_risk_difference(
        probabilities,
        probabilities,
        labels,
        alphas=(0.1,),
        temperature=2.0,
        confidence=0.9,
        replicates=50,
        seed=9,
    )
    assert result.lower == pytest.approx(0.0)
    assert result.upper == pytest.approx(0.0)
    assert not audit_accepts(result)
