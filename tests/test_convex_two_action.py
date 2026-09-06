import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from robust_verify.convex_two_action import (
    certified_score_margin,
    convex_logistic_predictions,
    logistic_design_stability_radius,
    verify_convex_logistic_embedding,
    verify_nonorthogonal_logistic_stability,
)
from robust_verify.two_action_theory import SameMarginalWitness


def test_convex_embedding_matches_every_finite_witness_utility() -> None:
    result = verify_convex_logistic_embedding()
    assert result["input_dimension"] == 10
    assert result["query_sets_checked"] == 64
    assert result["world_response_utility_checks"] == 256
    assert result["tight_randomized_worst_regret_exact"] == "47/103"


def test_standard_sklearn_logistic_regression_has_the_same_sign_pattern() -> None:
    witness = SameMarginalWitness()
    query = (witness.a0_atoms[0], witness.c_atoms[1])
    labels = np.asarray(
        [int(atom in query) for atom in witness.all_atoms] + [1, 0, 0],
        dtype=np.int64,
    )
    design = np.eye(len(labels), dtype=np.float64)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=False,
        solver="lbfgs",
        tol=1e-12,
        max_iter=10_000,
    )
    model.fit(design, labels)
    predicted = model.predict(design)
    assert np.array_equal(predicted, labels)

    explicit = convex_logistic_predictions(witness, query)
    explicit_candidates = np.asarray(
        [explicit[f"repairable_{atom}"] for atom in witness.all_atoms]
    )
    assert np.array_equal(explicit_candidates, labels[: len(witness.all_atoms)])


def test_nonorthogonal_stability_has_positive_explicit_radius() -> None:
    radius = logistic_design_stability_radius(10, 1.0)
    assert radius == pytest.approx(0.07980813454411886)
    assert certified_score_margin(10, 1.0, 0.99 * radius) > 0.0
    assert certified_score_margin(10, 1.0, 1.01 * radius) < 0.0


def test_dense_nonorthogonal_designs_preserve_every_query_prediction() -> None:
    result = verify_nonorthogonal_logistic_stability(random_seeds=2)
    assert result["query_sets_per_design"] == 64
    assert result["perturbed_full_retraining_fits_checked"] == 128
    assert result["maximum_off_diagonal_training_gram_entry"] > 0.0
    assert result["minimum_actual_signed_score"] > 0.0
    assert result["all_certified_predictions_preserved"] is True
