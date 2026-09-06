from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_verify.modern_baselines import (
    multiclass_repair_value_rank_scores,
    repair_value_rank_scores,
    reliability_gated_repair_value_scores,
)
from robust_verify.scoring import (
    ADAPTIVE_METHODS,
    METHOD_NAMES,
    build_adaptive_ranking,
    build_noise_gated_repair_value_ranking,
    build_rankings,
)


def _probabilities(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    logits = rng.normal(size=shape)
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def test_noise_gated_repair_value_is_registered_as_adaptive() -> None:
    assert "noise_gated_repair_value" in METHOD_NAMES
    assert "noise_gated_repair_value" in ADAPTIVE_METHODS


def test_hard_gate_preserves_anchors_and_excludes_outside_value_outlier() -> None:
    noise = np.linspace(1.0, 0.1, 10)
    repair = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 10.0])
    losses = np.linspace(0.0, 1.0, 10)

    ranking = build_noise_gated_repair_value_ranking(
        noise,
        repair,
        losses,
        budget_count=4,
        n_total=10,
        candidate_multiplier=2.0,
        noise_anchor_fraction=0.5,
    )

    assert sorted(ranking.tolist()) == list(range(10))
    assert set(ranking[:2]) == {0, 1}
    assert set(ranking[:4]) == {0, 1, 6, 7}
    assert 9 not in ranking[:4]


@pytest.mark.parametrize(
    ("candidate_multiplier", "anchor_fraction", "message"),
    [
        (0.9, 0.5, "candidate_multiplier"),
        (2.0, -0.1, "noise_anchor_fraction"),
        (2.0, 1.1, "noise_anchor_fraction"),
    ],
)
def test_hard_gate_rejects_invalid_constraints(
    candidate_multiplier: float, anchor_fraction: float, message: str
) -> None:
    values = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match=message):
        build_noise_gated_repair_value_ranking(
            values,
            values,
            values,
            budget_count=2,
            n_total=8,
            candidate_multiplier=candidate_multiplier,
            noise_anchor_fraction=anchor_fraction,
        )


def test_build_rankings_exports_v2_value_component_without_static_ranking() -> None:
    rng = np.random.default_rng(41)
    n_train, n_val, dim = 18, 12, 7
    train_y = np.asarray([0, 1] * 9)
    val_y = np.asarray([0] * 6 + [1] * 6)
    train_probs = _probabilities(rng, (n_train, 2))
    context = {
        "train_features": rng.normal(size=(n_train, dim)).astype(np.float32),
        "val_features": rng.normal(size=(n_val, dim)).astype(np.float32),
        "val_probs": _probabilities(rng, (n_val, 2)),
        "val_labels": val_y,
        "baseline_options": {"repair_value_tail_fraction": 0.20},
    }

    rankings, scores = build_rankings(
        methods=["noise_gated_repair_value"],
        probabilities=train_probs,
        labels=train_y,
        correctness_history=np.empty((n_train, 0), dtype=np.int8),
        private_frame=pd.DataFrame({"sample_id": np.arange(n_train)}),
        seed=41,
        context=context,
    )

    assert "noise_gated_repair_value" not in rankings
    component = scores["noise_gated_repair_value"]
    assert component.shape == (n_train,)
    assert np.all((0.0 <= component) & (component <= 1.0))

    ranking = build_adaptive_ranking(
        "noise_gated_repair_value",
        scores["noise_score"],
        train_probs,
        pd.DataFrame({"sample_id": np.arange(n_train)}),
        budget_count=4,
        n_total=n_train,
        losses=scores["loss"],
        labels=train_y,
        repair_value_scores=component,
        options={
            "repair_value_candidate_multiplier": 2.0,
            "repair_value_noise_anchor_fraction": 0.5,
        },
    )
    assert sorted(ranking.tolist()) == list(range(n_train))


def test_repair_value_component_prefers_helpful_binary_flip() -> None:
    component = repair_value_rank_scores(
        train_features=np.asarray([[1.0], [-1.0], [0.0]], dtype=np.float32),
        train_labels=np.asarray([0, 0, 0]),
        val_features=np.ones((4, 1), dtype=np.float32),
        val_probabilities=np.tile(np.asarray([[0.8, 0.2]]), (4, 1)),
        val_labels=np.ones(4, dtype=np.int64),
        tail_fraction=1.0,
    )
    assert component[0] > component[2] > component[1]


def test_multiclass_repair_value_reduces_to_binary() -> None:
    train_features = np.asarray([[1.0, 0.5], [-1.0, 0.5], [0.0, -1.0]], dtype=np.float32)
    train_labels = np.asarray([0, 1, 0], dtype=np.int64)
    train_probs = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]], dtype=np.float64)
    val_features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    val_probs = np.asarray(
        [[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6]], dtype=np.float64
    )
    val_labels = np.asarray([1, 1, 1, 1], dtype=np.int64)
    binary = repair_value_rank_scores(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probs,
        val_labels=val_labels,
        tail_fraction=1.0,
    )
    multiclass = multiclass_repair_value_rank_scores(
        train_features=train_features,
        train_probabilities=train_probs,
        train_labels=train_labels,
        val_features=val_features,
        val_probabilities=val_probs,
        val_labels=val_labels,
        tail_fraction=1.0,
    )
    assert np.allclose(binary, multiclass)


def test_reliability_gate_returns_legal_blend_metadata() -> None:
    rng = np.random.default_rng(53)
    n_train, n_val, dim = 20, 16, 5
    train_y = np.asarray([0, 1] * 10)
    train_probs = _probabilities(rng, (n_train, 2))
    val_y = np.asarray([0, 1] * 8)
    val_probs = _probabilities(rng, (n_val, 2))
    blended, metadata = reliability_gated_repair_value_scores(
        train_features=rng.normal(size=(n_train, dim)),
        train_probabilities=train_probs,
        train_labels=train_y,
        val_features=rng.normal(size=(n_val, dim)),
        val_probabilities=val_probs,
        val_labels=val_y,
        noise_posterior_proxy=np.linspace(0.0, 1.0, n_train),
        tail_fraction=0.2,
        num_folds=4,
    )
    assert blended.shape == (n_train,)
    assert np.all(np.isfinite(blended))
    assert 0.0 <= float(metadata["gate_alpha"]) <= 1.0
    assert 0.0 <= float(metadata["gate_reliability"]) <= 1.0
