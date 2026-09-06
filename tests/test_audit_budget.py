import numpy as np
import pytest

from robust_verify.audit_budget import (
    allocate_stratified_scout,
    beta_residual_posterior,
    build_band_class_strata,
    ranking_band_boundaries,
)


def test_band_boundaries_cover_population():
    boundaries = ranking_band_boundaries(1000, (0.0, 0.01, 0.1, 1.0))
    assert boundaries.tolist() == [0, 10, 100, 1000]


def test_band_class_strata_are_exhaustive():
    ranking = np.array([5, 4, 3, 2, 1, 0])
    labels = np.array([0, 1, 0, 1, 0, 1])
    strata, boundaries = build_band_class_strata(ranking, labels, (0.0, 0.5, 1.0))
    selected = np.concatenate([stratum.indices for stratum in strata])
    assert boundaries.tolist() == [0, 3, 6]
    assert sorted(selected.tolist()) == list(range(6))
    assert {(item.band, item.observed_class) for item in strata} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_scout_allocation_covers_every_stratum_and_exact_total():
    allocation = allocate_stratified_scout(np.array([2, 10, 100]), 20)
    assert allocation.sum() == 20
    assert np.all(allocation >= 1)
    assert np.all(allocation <= np.array([2, 10, 100]))


def test_scout_allocation_rejects_too_small_budget():
    with pytest.raises(ValueError, match="cover every stratum"):
        allocate_stratified_scout(np.array([10, 10, 10]), 2)


def test_reviewed_band_has_zero_residual_contribution():
    common = dict(
        stratum_sizes=np.array([100, 100]),
        scout_counts=np.array([10, 10]),
        scout_errors=np.array([5, 0]),
        stratum_bands=np.array([0, 1]),
        prior_band_means=np.array([0.5, 0.0]),
        prior_strength=0.0,
        population_size=200,
        draws=5000,
        upper_probability=0.95,
        seed=3,
    )
    none_reviewed = beta_residual_posterior(
        **common, reviewed_bands=np.array([False, False])
    )
    first_reviewed = beta_residual_posterior(
        **common, reviewed_bands=np.array([True, False])
    )
    assert first_reviewed.mean < none_reviewed.mean
    assert first_reviewed.upper >= first_reviewed.mean
