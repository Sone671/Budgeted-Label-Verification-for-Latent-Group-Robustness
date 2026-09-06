import numpy as np
import pytest
from scipy.stats import hypergeom

from robust_verify.audit_budget import hypergeometric_upper_error_count


def test_hypergeometric_upper_bound_respects_feasible_endpoints():
    assert hypergeometric_upper_error_count(
        population_size=10, sample_size=0, observed_errors=0, alpha=0.05
    ) == 10
    assert hypergeometric_upper_error_count(
        population_size=10, sample_size=10, observed_errors=3, alpha=0.05
    ) == 3
    value = hypergeometric_upper_error_count(
        population_size=30, sample_size=8, observed_errors=2, alpha=0.05
    )
    assert 2 <= value <= 24


def test_hypergeometric_upper_bound_has_one_sided_coverage():
    population_size, true_errors, sample_size, alpha = 100, 25, 12, 0.05
    rng = np.random.default_rng(7)
    observed = hypergeom.rvs(
        population_size, true_errors, sample_size, size=10_000, random_state=rng
    )
    covered = [
        hypergeometric_upper_error_count(
            population_size=population_size,
            sample_size=sample_size,
            observed_errors=int(value),
            alpha=alpha,
        ) >= true_errors
        for value in observed
    ]
    assert np.mean(covered) >= 0.94


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 0, "sample_size": 0, "observed_errors": 0, "alpha": 0.05},
        {"population_size": 10, "sample_size": 11, "observed_errors": 0, "alpha": 0.05},
        {"population_size": 10, "sample_size": 3, "observed_errors": 4, "alpha": 0.05},
        {"population_size": 10, "sample_size": 3, "observed_errors": 1, "alpha": 0.0},
    ],
)
def test_hypergeometric_upper_bound_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        hypergeometric_upper_error_count(**kwargs)
