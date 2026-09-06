"""Strongly convex linear-model realization of the two-action witness."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import exp, sqrt

import numpy as np

from robust_verify.two_action_theory import (
    SameMarginalWitness,
    same_marginal_test_support,
    same_marginal_utilities,
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def logistic_coordinate_optimum(label: int, l2_strength: float) -> float:
    """Unique coordinate optimum for one one-hot logistic training example."""

    if label not in (0, 1):
        raise ValueError("binary logistic label must be 0 or 1")
    if l2_strength <= 0:
        raise ValueError("l2_strength must be positive")

    def gradient(weight: float) -> float:
        return _sigmoid(weight) - label + l2_strength * weight

    lower = -max(2.0, 1.0 / l2_strength + 1.0)
    upper = -lower
    if gradient(lower) >= 0 or gradient(upper) <= 0:
        raise AssertionError("failed to bracket the strongly convex coordinate root")
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if gradient(middle) < 0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def convex_logistic_predictions(
    witness: SameMarginalWitness,
    query: tuple[int, ...],
    *,
    l2_strength: float = 1.0,
) -> dict[str, int]:
    """Fully retrain the orthogonal L2-logistic learner and predict test atoms."""

    query_set = set(query)
    if not query_set.issubset(witness.all_atoms):
        raise ValueError("query contains an atom outside the witness")
    labels = [int(atom in query_set) for atom in witness.all_atoms]
    # One always-correct positive anchor and two always-correct negative
    # anchors make all y x place groups nonempty.
    labels.extend([1, 0, 0])
    weights = [
        logistic_coordinate_optimum(label, l2_strength) for label in labels
    ]
    predictions = [int(weight >= 0.0) for weight in weights]
    names = [f"repairable_{atom}" for atom in witness.all_atoms]
    names.extend(
        [
            "correct_positive_anchor",
            "correct_negative_place0",
            "correct_negative_place1",
        ]
    )
    return dict(zip(names, predictions))


def convex_logistic_wga(
    witness: SameMarginalWitness,
    query: tuple[int, ...],
    *,
    world: int,
    audit_response: int,
    l2_strength: float = 1.0,
) -> Fraction:
    """Compute exact weighted WGA for the explicit convex linear learner."""

    predictions = convex_logistic_predictions(
        witness, query, l2_strength=l2_strength
    )
    support = same_marginal_test_support(witness, world, audit_response)
    correct: dict[tuple[int, int], int] = {}
    total: dict[tuple[int, int], int] = {}
    for feature, label, weight, place in support:
        key = (label, place)
        total[key] = total.get(key, 0) + weight
        correct[key] = correct.get(key, 0) + weight * int(
            predictions[feature] == label
        )
    if set(total) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise AssertionError("all four y x place groups must be nonempty")
    return min(Fraction(correct[key], total[key]) for key in total)


def verify_convex_logistic_embedding(
    witness: SameMarginalWitness | None = None,
    *,
    l2_strength: float = 1.0,
) -> dict[str, float | int | bool | str]:
    """Check that convex full retraining exactly realizes every witness utility."""

    witness = witness or SameMarginalWitness()
    label0_weight = logistic_coordinate_optimum(0, l2_strength)
    label1_weight = logistic_coordinate_optimum(1, l2_strength)
    if not label0_weight < 0.0 < label1_weight:
        raise AssertionError("logistic coordinate signs do not encode the label")

    query_sets = [
        query
        for size in range(witness.label_capacity + 1)
        for query in combinations(witness.all_atoms, size)
    ]
    utility_checks = 0
    for query in query_sets:
        predictions = convex_logistic_predictions(
            witness, query, l2_strength=l2_strength
        )
        predicted_query = {
            atom
            for atom in witness.all_atoms
            if predictions[f"repairable_{atom}"] == 1
        }
        if predicted_query != set(query):
            raise AssertionError("convex learner does not match the verified set")
        for response in (0, 1):
            expected0, expected1 = same_marginal_utilities(
                witness, query, response
            )
            actual0 = convex_logistic_wga(
                witness,
                query,
                world=0,
                audit_response=response,
                l2_strength=l2_strength,
            )
            actual1 = convex_logistic_wga(
                witness,
                query,
                world=1,
                audit_response=response,
                l2_strength=l2_strength,
            )
            if actual0 != expected0 or actual1 != expected1:
                raise AssertionError(
                    "explicit convex WGA differs from the finite witness utility"
                )
            utility_checks += 2

    return {
        "input_dimension": len(witness.all_atoms) + 3,
        "l2_strength": l2_strength,
        "strong_convexity_modulus": l2_strength,
        "label0_coordinate_weight": label0_weight,
        "label1_coordinate_weight": label1_weight,
        "query_sets_checked": len(query_sets),
        "world_response_utility_checks": utility_checks,
        "all_predictions_match_verified_labels": True,
        "all_wga_utilities_match_finite_witness": True,
        "tight_randomized_worst_regret_exact": str(
            witness.randomized_regret_bound
        ),
    }


def logistic_design_stability_radius(
    dimension: int,
    l2_strength: float,
) -> float:
    """Certified spectral-norm radius preserving every coordinate sign.

    Both the square training design and the matrix of test atom features may
    move this far from the identity.  The bound is uniform over every binary
    training-label vector.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if l2_strength <= 0:
        raise ValueError("l2_strength must be positive")
    margin = logistic_coordinate_optimum(1, l2_strength)
    root_dimension = sqrt(dimension)
    parameter_factor = root_dimension * (1.0 + margin / 4.0) / l2_strength
    linear = parameter_factor + margin * root_dimension
    discriminant = linear * linear + 4.0 * parameter_factor * margin
    return (-linear + sqrt(discriminant)) / (2.0 * parameter_factor)


def certified_score_margin(
    dimension: int,
    l2_strength: float,
    perturbation_radius: float,
) -> float:
    """Lower bound on every signed perturbed test score."""

    if perturbation_radius < 0:
        raise ValueError("perturbation_radius must be nonnegative")
    margin = logistic_coordinate_optimum(1, l2_strength)
    root_dimension = sqrt(dimension)
    parameter_factor = root_dimension * (1.0 + margin / 4.0) / l2_strength
    parameter_drift = parameter_factor * perturbation_radius
    score_drift = (
        (1.0 + perturbation_radius) * parameter_drift
        + perturbation_radius * margin * root_dimension
    )
    return margin - score_drift


def perturbed_logistic_optimum(
    design: np.ndarray,
    labels: np.ndarray,
    l2_strength: float,
) -> np.ndarray:
    """Newton-solve the uniquely optimal perturbed L2-logistic weights."""

    design = np.asarray(design, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != design.shape[1]:
        raise ValueError("design must be a square matrix of training rows")
    if labels.shape != (design.shape[0],):
        raise ValueError("labels must align with design rows")
    if not set(np.unique(labels).tolist()).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary")
    if l2_strength <= 0:
        raise ValueError("l2_strength must be positive")

    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(100):
        scores = design @ weights
        probabilities = np.where(
            scores >= 0,
            1.0 / (1.0 + np.exp(-scores)),
            np.exp(scores) / (1.0 + np.exp(scores)),
        )
        gradient = design.T @ (probabilities - labels) + l2_strength * weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design)
        hessian += l2_strength * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step) <= 1e-12:
            break
    else:
        raise RuntimeError("strongly convex Newton solve did not converge")
    return weights


def _spectral_scaled_noise(
    rng: np.random.Generator,
    dimension: int,
    radius: float,
) -> np.ndarray:
    noise = rng.normal(size=(dimension, dimension))
    spectral_norm = float(np.linalg.norm(noise, ord=2))
    return noise * (radius / spectral_norm)


def verify_nonorthogonal_logistic_stability(
    witness: SameMarginalWitness | None = None,
    *,
    l2_strength: float = 1.0,
    radius_fraction: float = 0.90,
    random_seeds: int = 4,
) -> dict[str, float | int | bool | str]:
    """Numerically check all query labels on dense certified perturbations."""

    witness = witness or SameMarginalWitness()
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between zero and one")
    dimension = len(witness.all_atoms) + 3
    certified_radius = logistic_design_stability_radius(dimension, l2_strength)
    tested_radius = radius_fraction * certified_radius
    margin_bound = certified_score_margin(
        dimension, l2_strength, tested_radius
    )
    if margin_bound <= 0:
        raise AssertionError("tested perturbation is outside the certified margin")

    query_sets = [
        query
        for size in range(witness.label_capacity + 1)
        for query in combinations(witness.all_atoms, size)
    ]
    minimum_actual_margin = float("inf")
    maximum_training_correlation = 0.0
    fits_checked = 0
    for seed in range(random_seeds):
        rng = np.random.default_rng(83_000 + seed)
        training_design = np.eye(dimension) + _spectral_scaled_noise(
            rng, dimension, tested_radius
        )
        test_design = np.eye(dimension) + _spectral_scaled_noise(
            rng, dimension, tested_radius
        )
        gram = training_design @ training_design.T
        off_diagonal = gram - np.diag(np.diag(gram))
        maximum_training_correlation = max(
            maximum_training_correlation,
            float(np.max(np.abs(off_diagonal))),
        )
        for query in query_sets:
            labels = np.asarray(
                [int(atom in query) for atom in witness.all_atoms] + [1, 0, 0],
                dtype=np.float64,
            )
            weights = perturbed_logistic_optimum(
                training_design, labels, l2_strength
            )
            scores = test_design @ weights
            signed_scores = np.where(labels == 1.0, scores, -scores)
            minimum_actual_margin = min(
                minimum_actual_margin, float(np.min(signed_scores))
            )
            if not bool((signed_scores > 0.0).all()):
                raise AssertionError("a certified non-orthogonal prediction changed sign")
            fits_checked += 1

    return {
        "input_dimension": dimension,
        "l2_strength": l2_strength,
        "certified_spectral_radius": certified_radius,
        "tested_spectral_radius": tested_radius,
        "certified_score_margin_at_tested_radius": margin_bound,
        "minimum_actual_signed_score": minimum_actual_margin,
        "maximum_off_diagonal_training_gram_entry": maximum_training_correlation,
        "random_dense_design_pairs": random_seeds,
        "query_sets_per_design": len(query_sets),
        "perturbed_full_retraining_fits_checked": fits_checked,
        "all_certified_predictions_preserved": True,
        "tight_randomized_worst_regret_exact": str(
            witness.randomized_regret_bound
        ),
    }
