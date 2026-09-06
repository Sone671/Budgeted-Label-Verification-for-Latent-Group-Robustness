from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from robust_verify.analysis import query_composition
from robust_verify.data.access import PrivateEvaluator, VerificationOracle
from robust_verify.scoring import build_rankings
from robust_verify.training import (
    make_initial_state,
    predict_from_state,
    train_linear_head,
)


def main() -> None:
    rng = np.random.default_rng(7)
    n_train, n_val, n_test, dim = 96, 32, 48, 12

    train_x = rng.normal(size=(n_train, dim)).astype(np.float32)
    val_x = rng.normal(size=(n_val, dim)).astype(np.float32)
    test_x = rng.normal(size=(n_test, dim)).astype(np.float32)

    true_w = rng.normal(size=(dim,))
    train_y = (train_x @ true_w > 0).astype(np.int64)
    val_y = (val_x @ true_w > 0).astype(np.int64)
    test_y = (test_x @ true_w > 0).astype(np.int64)

    noisy_y = train_y.copy()
    noisy_indices = rng.choice(n_train, size=15, replace=False)
    noisy_y[noisy_indices] = 1 - noisy_y[noisy_indices]

    group = 2 * test_y + (test_x[:, 0] > 0).astype(np.int64)
    train_group = 2 * train_y + (train_x[:, 0] > 0).astype(np.int64)
    train_minority = (train_y != (train_x[:, 0] > 0)).astype(np.int64)

    config = {
        "epochs": 4,
        "batch_size": 32,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "patience": 4,
        "selection_metric": "balanced_accuracy",
        "device": "cpu",
    }

    initial_state = make_initial_state(dim, 2, seed=3)
    probe = train_linear_head(
        train_x,
        noisy_y,
        val_x,
        val_y,
        config,
        seed=3,
        initial_state=initial_state,
        record_dynamics=True,
    )
    assert probe.correctness_history is not None

    private_train = pd.DataFrame(
        {
            "sample_id": np.arange(n_train),
            "clean_label": train_y,
            "place": train_group % 2,
            "group": train_group,
            "is_minority": train_minority,
            "is_noisy": noisy_y != train_y,
        }
    )

    methods = [
        "random",
        "loss",
        "entropy",
        "forgetting",
        "noise_score",
        "noise_score_coverage",
        "oracle_noise",
        "oracle_minority",
        "oracle_group_balanced",
    ]
    rankings, _ = build_rankings(
        methods,
        probe.train_probabilities,
        noisy_y,
        probe.correctness_history,
        private_train,
        seed=3,
    )
    assert all(len(ranking) == n_train for ranking in rankings.values())
    assert len(set(rankings["loss"].tolist())) == n_train

    composition = query_composition(rankings["oracle_noise"][:10], private_train)
    assert composition["num_corrected"] > 0

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        private_train.to_csv(temporary / "train_private.csv", index=False)
        oracle = VerificationOracle(temporary / "train_private.csv")
        queried = rankings["oracle_noise"][:10]
        verified = oracle.verify(queried)
        assert len(verified) == 10

        private_test = pd.DataFrame(
            {
                "sample_id": np.arange(n_test),
                "clean_label": test_y,
                "place": group % 2,
                "group": group,
                "is_minority": (test_y != group % 2).astype(int),
            }
        )
        private_test.to_csv(temporary / "test_private.csv", index=False)

        predictions, _ = predict_from_state(
            probe.best_state,
            test_x,
            dim,
            2,
            config,
        )
        metrics = PrivateEvaluator(temporary / "test_private.csv").evaluate(
            np.arange(n_test),
            predictions,
        )
        assert 0 <= metrics["wga"] <= 1

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
