import numpy as np

from robust_verify.rivet import (
    binary_rivet_scores,
    class_prior_adjusted_noise_score,
    empirical_cvar,
    feedback_exploration_indices,
    spectral_binary_rivet_scores,
    spectral_class_cvar,
    trust_region_candidate_selection,
)


def test_empirical_cvar_uses_worst_tail() -> None:
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9]], dtype=np.float32
    )
    labels = np.array([0, 0, 0, 1])
    losses = -np.log([0.9, 0.8, 0.4, 0.9])
    assert np.isclose(empirical_cvar(probabilities, labels, 0.5), np.mean(sorted(losses)[-2:]))


def test_rivet_is_invariant_to_binary_label_swap() -> None:
    rng = np.random.default_rng(7)
    train_features = rng.normal(size=(40, 3)).astype(np.float32)
    validation_features = rng.normal(size=(20, 3)).astype(np.float32)
    train_p1 = 1.0 / (1.0 + np.exp(-train_features[:, 0]))
    val_p1 = 1.0 / (1.0 + np.exp(-validation_features[:, 0]))
    train_probabilities = np.column_stack([1.0 - train_p1, train_p1])
    validation_probabilities = np.column_stack([1.0 - val_p1, val_p1])
    noisy_labels = (train_features[:, 1] > 0).astype(np.int64)
    validation_labels = (validation_features[:, 1] > 0).astype(np.int64)
    noise_score = rng.uniform(size=len(train_features))

    original = binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities,
        noisy_labels=noisy_labels,
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        validation_labels=validation_labels,
        noise_score=noise_score,
        alpha=0.2,
        damping=0.01,
    )
    swapped = binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities[:, ::-1],
        noisy_labels=1 - noisy_labels,
        validation_features=validation_features,
        validation_probabilities=validation_probabilities[:, ::-1],
        validation_labels=1 - validation_labels,
        noise_score=noise_score,
        alpha=0.2,
        damping=0.01,
    )

    assert np.allclose(original.leverage, swapped.leverage, atol=1e-7)
    assert np.allclose(original.utility, swapped.utility, atol=1e-7)


def test_rivet_preserves_precalibrated_noise_probability() -> None:
    rng = np.random.default_rng(11)
    train_features = rng.normal(size=(8, 3)).astype(np.float32)
    validation_features = rng.normal(size=(6, 3)).astype(np.float32)
    train_probabilities = rng.dirichlet([2.0, 2.0], size=8).astype(np.float32)
    validation_probabilities = rng.dirichlet([2.0, 2.0], size=6).astype(np.float32)
    noisy_labels = rng.integers(0, 2, size=8)
    validation_labels = rng.integers(0, 2, size=6)
    noise_probability = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.float64)

    result = binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities,
        noisy_labels=noisy_labels,
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        validation_labels=validation_labels,
        noise_score=noise_probability,
        rank_noise_score=False,
    )

    np.testing.assert_array_equal(result.noise_probability, noise_probability)


def test_class_prior_adjustment_upweights_high_loss_observed_class() -> None:
    noise_score = np.ones(8)
    losses = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    adjusted, metadata = class_prior_adjusted_noise_score(
        noise_score, losses, labels
    )

    assert metadata["weight_class_1"] > metadata["weight_class_0"]
    assert adjusted[labels == 1].mean() > adjusted[labels == 0].mean()


def test_feedback_exploration_is_unique_deterministic_and_keeps_top_priority() -> None:
    priority = np.linspace(0.0, 1.0, 40)
    labels = np.tile([0, 1], 20)

    first = feedback_exploration_indices(
        priority, labels, 12, seed=17, targeted_fraction=0.5
    )
    second = feedback_exploration_indices(
        priority, labels, 12, seed=17, targeted_fraction=0.5
    )

    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 12
    assert set(range(34, 40)).issubset(set(first.tolist()))


def test_spectral_class_cvar_is_a_smooth_upper_bound_and_swap_invariant() -> None:
    probabilities = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1])
    risk, _, _, cells, _ = spectral_class_cvar(
        probabilities, labels, alphas=(0.5, 1.0), temperature=4.0
    )
    swapped, _, _, swapped_cells, _ = spectral_class_cvar(
        probabilities[:, ::-1], 1 - labels, alphas=(0.5, 1.0), temperature=4.0
    )
    assert risk >= cells.max()
    assert np.isclose(risk, swapped)
    np.testing.assert_allclose(np.sort(cells), np.sort(swapped_cells))


def test_spectral_rivet_is_invariant_to_binary_label_swap() -> None:
    rng = np.random.default_rng(19)
    train_features = rng.normal(size=(40, 3)).astype(np.float32)
    validation_features = rng.normal(size=(24, 3)).astype(np.float32)
    train_p1 = 1.0 / (1.0 + np.exp(-train_features[:, 0]))
    val_p1 = 1.0 / (1.0 + np.exp(-validation_features[:, 0]))
    train_probabilities = np.column_stack([1.0 - train_p1, train_p1])
    validation_probabilities = np.column_stack([1.0 - val_p1, val_p1])
    noisy_labels = (train_features[:, 1] > 0).astype(np.int64)
    validation_labels = (validation_features[:, 1] > 0).astype(np.int64)
    noise_score = rng.uniform(size=len(train_features))

    original = spectral_binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities,
        noisy_labels=noisy_labels,
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        validation_labels=validation_labels,
        noise_score=noise_score,
        alphas=(0.1, 0.25),
        temperature=3.0,
        damping=0.01,
    )
    swapped = spectral_binary_rivet_scores(
        train_features=train_features,
        train_probabilities=train_probabilities[:, ::-1],
        noisy_labels=1 - noisy_labels,
        validation_features=validation_features,
        validation_probabilities=validation_probabilities[:, ::-1],
        validation_labels=1 - validation_labels,
        noise_score=noise_score,
        alphas=(0.1, 0.25),
        temperature=3.0,
        damping=0.01,
    )
    np.testing.assert_allclose(original.leverage, swapped.leverage, atol=1e-7)
    np.testing.assert_allclose(original.utility, swapped.utility, atol=1e-7)


def test_trust_region_selection_preserves_core_and_limits_replacements() -> None:
    prior_order = np.arange(20)
    prior_score = np.linspace(1.0, 0.0, 20)
    leverage = np.arange(20, dtype=np.float64)
    selected = trust_region_candidate_selection(
        prior_order,
        prior_score,
        leverage,
        10,
        candidate_multiplier=1.5,
        core_fraction=0.7,
    )
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10
    assert set(range(7)).issubset(set(selected.tolist()))
    assert len(set(selected.tolist()).difference(set(range(10)))) <= 3
