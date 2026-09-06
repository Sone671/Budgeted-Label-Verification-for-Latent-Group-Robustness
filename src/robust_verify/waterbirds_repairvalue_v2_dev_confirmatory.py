"""LOCAL DEV integrity rules for Waterbirds noise-gated RepairValue-Q v2.

Dev rehearsal on development seeds 20--29 with scientific content identical to
the frozen v2 protocol. This is development evidence only and must never be
pooled with the formal 40--59 confirmation block. Only filesystem paths may be
overridden without creating a new protocol version.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_verify.absolute_utility import bootstrap_mean_ci, exact_two_sided_sign_test
from robust_verify.analysis import query_composition
from robust_verify.data.access import PrivateEvaluator
from robust_verify.utils import budget_to_count

PROTOCOL_VERSION = "waterbirds-e2e-repairvalue-v2-seeds-20-29-local-dev"
FROZEN_SEEDS = tuple(range(20, 30))
FROZEN_METHODS = (
    "random",
    "loss",
    "noise_score",
    "expected_repair_value",
    "noise_gated_repair_value",
)
FROZEN_BUDGET = 0.10
FROZEN_NOISE_NAME = "uniform20"
PRIMARY_METHOD = "noise_gated_repair_value"
PAIRED_BASELINE = "noise_score"
PRIMARY_MIN_POSITIVE_PAIRED_SEEDS = 15
REPAIR_VALUE_TAIL_FRACTION = 0.20
REPAIR_VALUE_CANDIDATE_MULTIPLIER = 2.0
REPAIR_VALUE_NOISE_ANCHOR_FRACTION = 0.50
PRECISION_NONINFERIORITY_MARGIN = 0.10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_RNG_SEED = 20260805

FROZEN_E2E = {
    "backbone": "resnet50",
    "pretrained": True,
    "input_size": 224,
    "augmentation": True,
    "batch_size": 48,
    "num_workers": 4,
    "learning_rate": 0.00003,
    "weight_decay": 0.0001,
    "epochs": 10,
    "patience": 3,
    "amp": True,
    "device": "cuda",
    "deterministic": True,
}

ARTIFACT_COLUMNS = (
    "query_artifact",
    "checkpoint",
    "prediction_artifact",
    "baseline_checkpoint",
    "baseline_prediction_artifact",
)

REQUIRED_RESULT_COLUMNS = {
    "noise_name",
    "noise_kind",
    "seed",
    "method",
    "budget_fraction",
    "budget_count",
    "num_queried",
    "num_corrected",
    "noise_precision",
    "minority_query_rate",
    "average_accuracy",
    "balanced_accuracy",
    "wga",
    "baseline_average_accuracy",
    "baseline_balanced_accuracy",
    "baseline_wga",
    "delta_average_accuracy",
    "delta_wga",
    *ARTIFACT_COLUMNS,
}


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_frozen_config(
    config: dict[str, Any], *, expected_seeds: Iterable[int] = FROZEN_SEEDS
) -> None:
    """Reject any scientific change to the registered follow-up protocol."""
    expected: dict[str, Any] = {
        "confirmation.protocol_version": PROTOCOL_VERSION,
        "confirmation.role": "prospective_method_confirmation",
        "confirmation.hypothesis_origin": (
            "posthoc_diagnosis_of_waterbirds_seed20_39_repairvalue_v1_failure"
        ),
        "confirmation.primary_method": PRIMARY_METHOD,
        "confirmation.paired_baseline": PAIRED_BASELINE,
        "confirmation.primary_estimand": (
            "mean_seed_paired_delta_wga_difference_vs_noise_score"
        ),
        "confirmation.primary_gate.require_bootstrap_ci_lower_above_zero": True,
        "confirmation.primary_gate.require_absolute_ci_lower_above_zero": True,
        "confirmation.primary_gate.min_positive_paired_seeds": (PRIMARY_MIN_POSITIVE_PAIRED_SEEDS),
        "confirmation.primary_gate.precision_noninferiority_margin": (
            PRECISION_NONINFERIORITY_MARGIN
        ),
        "data.dataset": "waterbirds",
        "data.metadata_csv": "metadata.csv",
        "data.split_values": {"train": 0, "val": 1, "test": 2},
        "features.skip_extraction": True,
        "training.selection_metric": "balanced_accuracy",
        **{f"end_to_end.{name}": value for name, value in FROZEN_E2E.items()},
    }
    problems: list[str] = []
    for dotted, wanted in expected.items():
        actual = _get(config, dotted)
        if not _same(actual, wanted):
            problems.append(f"{dotted}: expected {wanted!r}, found {actual!r}")

    wanted_noises = [{"name": FROZEN_NOISE_NAME, "kind": "uniform", "rate": 0.20}]
    noises = _get(config, "noise.settings")
    if noises != wanted_noises:
        problems.append(f"noise.settings: expected {wanted_noises!r}, found {noises!r}")

    seeds = tuple(int(value) for value in (_get(config, "experiment.seeds") or []))
    wanted_seeds = tuple(int(value) for value in expected_seeds)
    if seeds != wanted_seeds:
        problems.append(f"experiment.seeds: expected {wanted_seeds!r}, found {seeds!r}")

    budgets = tuple(float(value) for value in (_get(config, "experiment.budgets") or []))
    if budgets != (FROZEN_BUDGET,):
        problems.append(f"experiment.budgets: expected {(FROZEN_BUDGET,)!r}, found {budgets!r}")

    methods = tuple(str(value) for value in (_get(config, "experiment.methods") or []))
    if methods != FROZEN_METHODS:
        problems.append(f"experiment.methods: expected {FROZEN_METHODS!r}, found {methods!r}")

    options = _get(config, "experiment.baseline_options")
    wanted_options = {
        "repair_value_tail_fraction": REPAIR_VALUE_TAIL_FRACTION,
        "repair_value_candidate_multiplier": REPAIR_VALUE_CANDIDATE_MULTIPLIER,
        "repair_value_noise_anchor_fraction": REPAIR_VALUE_NOISE_ANCHOR_FRACTION,
    }
    if options != wanted_options:
        problems.append(
            f"experiment.baseline_options: expected {wanted_options!r}, found {options!r}"
        )

    if problems:
        raise ValueError(
            "Frozen Waterbirds RepairValue protocol mismatch:\n- " + "\n- ".join(problems)
        )


def protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the path-independent scientific protocol used for hashing."""
    validate_frozen_config(config)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "confirmation": config["confirmation"],
        "data": {
            key: config["data"][key]
            for key in (
                "dataset",
                "metadata_csv",
                "split_values",
            )
        },
        "features": {"skip_extraction": config["features"]["skip_extraction"]},
        "training": {"selection_metric": config["training"]["selection_metric"]},
        "end_to_end": {key: config["end_to_end"][key] for key in FROZEN_E2E},
        "noise": config["noise"],
        "experiment": {
            "seeds": list(FROZEN_SEEDS),
            "budgets": [FROZEN_BUDGET],
            "baseline_options": {
                "repair_value_tail_fraction": REPAIR_VALUE_TAIL_FRACTION,
                "repair_value_candidate_multiplier": REPAIR_VALUE_CANDIDATE_MULTIPLIER,
                "repair_value_noise_anchor_fraction": REPAIR_VALUE_NOISE_ANCHOR_FRACTION,
            },
            "methods": list(FROZEN_METHODS),
        },
    }


def protocol_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        protocol_payload(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_source_files(source_root: str | Path) -> list[Path]:
    root = Path(source_root)
    files = [
        root / "resolved_config.json",
        root / "noise_statistics.csv",
        *(root / "canonical" / f"{split}.csv" for split in ("train", "val", "test")),
        *(
            root / "manifests" / f"{split}_{scope}.csv"
            for split in ("val", "test")
            for scope in ("public", "private")
        ),
    ]
    for seed in FROZEN_SEEDS:
        files.extend(
            [
                root / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_public.csv",
                root / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_private.csv",
            ]
        )
    return files


def build_preparation_manifest(source_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate Stage-0 isolation and hash every prepared source artifact."""
    validate_frozen_config(config)
    root = Path(source_root).resolve()
    missing = [str(path) for path in expected_source_files(root) if not path.is_file()]
    if missing:
        raise ValueError("Stage-0 source is incomplete:\n- " + "\n- ".join(missing))

    expected_ids: np.ndarray | None = None
    for seed in FROZEN_SEEDS:
        public_path = root / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_public.csv"
        private_path = root / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_private.csv"
        public = pd.read_csv(public_path)
        private = pd.read_csv(private_path)
        if "clean_label" in public.columns or "group" in public.columns:
            raise ValueError(f"Private columns leaked into {public_path}")
        if not {"sample_id", "image_relpath", "noisy_label"}.issubset(public.columns):
            raise ValueError(f"Public manifest schema mismatch: {public_path}")
        if not {"sample_id", "clean_label", "group", "is_noisy"}.issubset(private.columns):
            raise ValueError(f"Private manifest schema mismatch: {private_path}")
        public_ids = np.sort(public["sample_id"].to_numpy(dtype=np.int64))
        private_ids = np.sort(private["sample_id"].to_numpy(dtype=np.int64))
        if not np.array_equal(public_ids, private_ids):
            raise ValueError(f"Public/private sample IDs differ for seed {seed}")
        if expected_ids is None:
            expected_ids = public_ids
        elif not np.array_equal(public_ids, expected_ids):
            raise ValueError(f"Training sample IDs differ across corruption seed {seed}")

    val_public = pd.read_csv(root / "manifests" / "val_public.csv", nrows=5)
    test_public = pd.read_csv(root / "manifests" / "test_public.csv", nrows=5)
    if "label" not in val_public.columns or "clean_label" in val_public.columns:
        raise ValueError("Validation public manifest must expose label but not clean_label")
    if "label" in test_public.columns or "clean_label" in test_public.columns:
        raise ValueError("Test public manifest must expose neither label nor clean_label")

    canonical_train = pd.read_csv(root / "canonical" / "train.csv")
    group_counts = {
        str(int(group)): int(count)
        for group, count in canonical_train.groupby("group").size().items()
    }
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in expected_source_files(root)
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(config),
        "source_root": str(root),
        "train_rows": len(canonical_train),
        "budget_count": budget_to_count(FROZEN_BUDGET, len(canonical_train)),
        "train_group_counts": group_counts,
        "seeds": list(FROZEN_SEEDS),
        "files": files,
    }


def verify_preparation_manifest(
    source_root: str | Path, manifest: dict[str, Any], config: dict[str, Any]
) -> None:
    root = Path(source_root).resolve()
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Preparation manifest protocol version mismatch")
    if manifest.get("protocol_sha256") != protocol_sha256(config):
        raise ValueError("Preparation manifest protocol hash mismatch")
    for record in manifest.get("files", []):
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Preparation artifact escapes source root: {path}") from exc
        if not path.is_file():
            raise ValueError(f"Missing prepared artifact: {path}")
        if int(record["bytes"]) != path.stat().st_size:
            raise ValueError(f"Prepared artifact size changed: {path}")
        if str(record["sha256"]) != sha256_file(path):
            raise ValueError(f"Prepared artifact hash changed: {path}")


def validate_result_frame(frame: pd.DataFrame, *, expected_seeds: Iterable[int]) -> None:
    """Validate the complete method/seed key space and primary metric invariants."""
    missing = sorted(REQUIRED_RESULT_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Result columns are missing: {missing}")
    seeds = tuple(int(value) for value in expected_seeds)
    expected_keys = {
        (seed, method, FROZEN_NOISE_NAME, FROZEN_BUDGET)
        for seed in seeds
        for method in FROZEN_METHODS
    }
    actual_keys = {
        (
            int(row.seed),
            str(row.method),
            str(row.noise_name),
            round(float(row.budget_fraction), 12),
        )
        for row in frame.itertuples(index=False)
    }
    if actual_keys != expected_keys or len(frame) != len(expected_keys):
        raise ValueError(
            "Result key space mismatch: "
            f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
        )
    if frame.duplicated(["seed", "method", "noise_name", "budget_fraction"]).any():
        raise ValueError("Duplicate RepairValue confirmatory result keys")

    numeric = [
        "budget_count",
        "num_queried",
        "num_corrected",
        "noise_precision",
        "average_accuracy",
        "wga",
        "baseline_average_accuracy",
        "baseline_wga",
        "delta_average_accuracy",
        "delta_wga",
    ]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("RepairValue confirmatory results contain non-finite values")
    if not np.allclose(
        frame["delta_wga"].to_numpy(float),
        frame["wga"].to_numpy(float) - frame["baseline_wga"].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("delta_wga is inconsistent with wga - baseline_wga")
    if not np.allclose(
        frame["delta_average_accuracy"].to_numpy(float),
        frame["average_accuracy"].to_numpy(float)
        - frame["baseline_average_accuracy"].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("delta_average_accuracy is inconsistent with its baseline")
    for seed, group in frame.groupby("seed"):
        for column in (
            "baseline_wga",
            "baseline_average_accuracy",
            "baseline_balanced_accuracy",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Seed {seed} has method-dependent {column}")


def _artifact_path(root: Path, value: Any) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise ValueError(f"Artifact path must be relative: {raw}")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes root: {raw}") from exc
    if not path.is_file():
        raise ValueError(f"Missing artifact: {path}")
    return path


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{name} mismatch: result={actual}, recomputed={expected}")


def validate_seed_artifacts(
    frame: pd.DataFrame, *, seed: int, shard_root: str | Path, source_root: str | Path
) -> dict[str, Any]:
    """Recompute query composition and test metrics from saved artifacts."""
    validate_result_frame(frame, expected_seeds=[seed])
    shard = Path(shard_root).resolve()
    source = Path(source_root).resolve()
    public = (
        pd.read_csv(source / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_public.csv")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    private = (
        pd.read_csv(source / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_private.csv")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    test_public = pd.read_csv(source / "manifests" / "test_public.csv").sort_values("sample_id")
    evaluator = PrivateEvaluator(source / "manifests" / "test_private.csv")
    expected_budget_count = budget_to_count(FROZEN_BUDGET, len(public))
    position_by_id = {
        int(sample_id): position
        for position, sample_id in enumerate(public["sample_id"].to_numpy(dtype=np.int64))
    }

    baseline_prediction_paths: set[Path] = set()
    baseline_checkpoint_paths: set[Path] = set()
    verified_files: set[Path] = set()
    for row in frame.to_dict("records"):
        method = str(row["method"])
        if int(row["budget_count"]) != expected_budget_count:
            raise ValueError(f"Seed {seed}/{method} has the wrong budget_count")
        if int(row["num_queried"]) != expected_budget_count:
            raise ValueError(f"Seed {seed}/{method} has the wrong num_queried")

        query_path = _artifact_path(shard, row["query_artifact"])
        with np.load(query_path, allow_pickle=False) as payload:
            if set(payload.files) != {"sample_ids"}:
                raise ValueError(f"Unexpected query artifact keys: {query_path}")
            query_ids = payload["sample_ids"].astype(np.int64, copy=False)
        if len(query_ids) != expected_budget_count or len(np.unique(query_ids)) != len(query_ids):
            raise ValueError(f"Invalid query count or duplicate IDs: {query_path}")
        missing_ids = sorted(set(map(int, query_ids)).difference(position_by_id))
        if missing_ids:
            raise ValueError(f"Query contains IDs outside public train: {missing_ids[:5]}")
        positions = np.asarray([position_by_id[int(value)] for value in query_ids], dtype=np.int64)
        composition = query_composition(positions, private)
        for name in (
            "num_queried",
            "num_corrected",
            "noise_precision",
            "minority_query_rate",
            "clean_minority_rate",
            "minority_recall",
            "group_query_entropy",
        ):
            _assert_close(f"seed {seed}/{method}/{name}", row[name], composition[name])

        prediction_path = _artifact_path(shard, row["prediction_artifact"])
        with np.load(prediction_path, allow_pickle=False) as payload:
            prediction_ids = payload["sample_ids"].astype(np.int64, copy=False)
            predictions = payload["predictions"].astype(np.int64, copy=False)
        expected_test_ids = test_public["sample_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(prediction_ids, expected_test_ids):
            raise ValueError(f"Test prediction IDs are incomplete: {prediction_path}")
        metrics = evaluator.evaluate(prediction_ids, predictions)
        for name, expected in metrics.items():
            _assert_close(f"seed {seed}/{method}/{name}", row[name], expected)

        baseline_prediction_path = _artifact_path(shard, row["baseline_prediction_artifact"])
        with np.load(baseline_prediction_path, allow_pickle=False) as payload:
            baseline_ids = payload["sample_ids"].astype(np.int64, copy=False)
            baseline_predictions = payload["predictions"].astype(np.int64, copy=False)
        if not np.array_equal(baseline_ids, expected_test_ids):
            raise ValueError(f"Baseline prediction IDs are incomplete: {baseline_prediction_path}")
        baseline_metrics = evaluator.evaluate(baseline_ids, baseline_predictions)
        for name, expected in baseline_metrics.items():
            _assert_close(
                f"seed {seed}/{method}/baseline_{name}", row[f"baseline_{name}"], expected
            )

        checkpoint_path = _artifact_path(shard, row["checkpoint"])
        baseline_checkpoint_path = _artifact_path(shard, row["baseline_checkpoint"])
        if checkpoint_path.stat().st_size == 0 or baseline_checkpoint_path.stat().st_size == 0:
            raise ValueError("A replay checkpoint is empty")
        baseline_prediction_paths.add(baseline_prediction_path)
        baseline_checkpoint_paths.add(baseline_checkpoint_path)
        verified_files.update(
            {
                query_path,
                prediction_path,
                baseline_prediction_path,
                checkpoint_path,
                baseline_checkpoint_path,
            }
        )

    if len(baseline_prediction_paths) != 1 or len(baseline_checkpoint_paths) != 1:
        raise ValueError(f"Seed {seed} methods do not share one no-correction baseline")
    return {
        "seed": seed,
        "rows": len(frame),
        "budget_count": expected_budget_count,
        "verified_files": sorted(str(path) for path in verified_files),
    }


def _paired_effect(frame: pd.DataFrame, method: str, baseline: str) -> np.ndarray:
    left = frame.loc[frame["method"].eq(method), ["seed", "delta_wga"]].rename(
        columns={"delta_wga": "left"}
    )
    right = frame.loc[frame["method"].eq(baseline), ["seed", "delta_wga"]].rename(
        columns={"delta_wga": "right"}
    )
    paired = left.merge(right, on="seed", how="inner", validate="one_to_one").sort_values("seed")
    return paired["left"].to_numpy(float) - paired["right"].to_numpy(float)


def summarize_confirmatory_results(frame: pd.DataFrame) -> dict[str, Any]:
    """Apply the registered Waterbirds noise-gated RepairValue v2 gates."""
    validate_result_frame(frame, expected_seeds=FROZEN_SEEDS)
    candidate = frame.loc[frame["method"].eq(PRIMARY_METHOD)].sort_values("seed")
    noise_score = frame.loc[frame["method"].eq(PAIRED_BASELINE)].sort_values("seed")
    absolute = candidate["delta_wga"].to_numpy(dtype=float)
    paired = _paired_effect(frame, PRIMARY_METHOD, PAIRED_BASELINE)
    random_paired = _paired_effect(frame, PRIMARY_METHOD, "random")
    loss_paired = _paired_effect(frame, PRIMARY_METHOD, "loss")
    v1_paired = _paired_effect(frame, PRIMARY_METHOD, "expected_repair_value")
    precision_difference = (
        candidate["noise_precision"].to_numpy(dtype=float)
        - noise_score["noise_precision"].to_numpy(dtype=float)
    )

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    paired_ci = bootstrap_mean_ci(paired, rng, BOOTSTRAP_REPLICATES)
    absolute_ci = bootstrap_mean_ci(absolute, rng, BOOTSTRAP_REPLICATES)
    random_ci = bootstrap_mean_ci(random_paired, rng, BOOTSTRAP_REPLICATES)
    loss_ci = bootstrap_mean_ci(loss_paired, rng, BOOTSTRAP_REPLICATES)
    v1_ci = bootstrap_mean_ci(v1_paired, rng, BOOTSTRAP_REPLICATES)
    precision_ci = bootstrap_mean_ci(precision_difference, rng, BOOTSTRAP_REPLICATES)
    positive_paired_count = int((paired > 0).sum())
    candidate_precision = float(candidate["noise_precision"].mean())
    noise_score_precision = float(noise_score["noise_precision"].mean())
    precision_drop = noise_score_precision - candidate_precision
    gates = {
        "paired_bootstrap_ci_lower_above_zero": bool(paired_ci[0] > 0),
        "absolute_bootstrap_ci_lower_above_zero": bool(absolute_ci[0] > 0),
        "positive_paired_seed_count_at_least_15": bool(
            positive_paired_count >= PRIMARY_MIN_POSITIVE_PAIRED_SEEDS
        ),
        "precision_noninferiority_ci_lower_at_least_negative_10pp": bool(
            precision_ci[0] >= -PRECISION_NONINFERIORITY_MARGIN
        ),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_method": PRIMARY_METHOD,
        "paired_baseline": PAIRED_BASELINE,
        "n_seeds": len(FROZEN_SEEDS),
        "seeds": list(FROZEN_SEEDS),
        "primary_estimand": (
            "mean paired delta_wga(noise-gated RepairValue-Q v2) "
            "minus delta_wga(NoiseScore)"
        ),
        "paired_vs_noise_score": {
            "mean_delta_wga_difference": float(paired.mean()),
            "bootstrap_95_ci": [float(paired_ci[0]), float(paired_ci[1])],
            "positive_seed_count": positive_paired_count,
            "negative_seed_count": int((paired < 0).sum()),
            "tie_seed_count": int((paired == 0).sum()),
            "exact_two_sided_sign_test_p": exact_two_sided_sign_test(paired),
        },
        "absolute_utility": {
            "mean_delta_wga": float(absolute.mean()),
            "bootstrap_95_ci": [float(absolute_ci[0]), float(absolute_ci[1])],
            "positive_seed_count": int((absolute > 0).sum()),
            "negative_seed_count": int((absolute < 0).sum()),
            "mean_noise_precision": candidate_precision,
            "exact_two_sided_sign_test_p": exact_two_sided_sign_test(absolute),
        },
        "precision_noninferiority": {
            "candidate_mean_noise_precision": candidate_precision,
            "noise_score_mean_noise_precision": noise_score_precision,
            "precision_drop": precision_drop,
            "paired_precision_difference_bootstrap_95_ci": [
                float(precision_ci[0]),
                float(precision_ci[1]),
            ],
            "noninferiority_margin": PRECISION_NONINFERIORITY_MARGIN,
        },
        "v2_vs_random_secondary": {
            "mean_delta_wga_difference": float(random_paired.mean()),
            "bootstrap_95_ci": [float(random_ci[0]), float(random_ci[1])],
            "positive_seed_count": int((random_paired > 0).sum()),
            "negative_seed_count": int((random_paired < 0).sum()),
            "tie_seed_count": int((random_paired == 0).sum()),
            "exact_two_sided_sign_test_p": exact_two_sided_sign_test(random_paired),
        },
        "v2_vs_loss_secondary": {
            "mean_delta_wga_difference": float(loss_paired.mean()),
            "bootstrap_95_ci": [float(loss_ci[0]), float(loss_ci[1])],
        },
        "v2_vs_v1_secondary": {
            "mean_delta_wga_difference": float(v1_paired.mean()),
            "bootstrap_95_ci": [float(v1_ci[0]), float(v1_ci[1])],
        },
        "primary_gate": gates,
        "primary_gate_passed": bool(all(gates.values())),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_rng_seed": BOOTSTRAP_RNG_SEED,
    }


def build_group_mechanism_audit(
    frame: pd.DataFrame, *, output_root: str | Path, source_root: str | Path
) -> pd.DataFrame:
    """Audit query/correction coverage of each seed's baseline worst group."""
    validate_result_frame(frame, expected_seeds=FROZEN_SEEDS)
    output = Path(output_root).resolve()
    source = Path(source_root).resolve()
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values(["seed", "method"]).to_dict("records"):
        seed = int(row["seed"])
        group_ids = sorted(
            int(name.removeprefix("baseline_group_").removesuffix("_accuracy"))
            for name in row
            if name.startswith("baseline_group_") and name.endswith("_accuracy")
        )
        if not group_ids:
            raise ValueError("Group-level baseline accuracy columns are absent")
        baseline_weak_group = min(
            group_ids, key=lambda group: (float(row[f"baseline_group_{group}_accuracy"]), group)
        )
        post_worst_group = min(
            group_ids, key=lambda group: (float(row[f"group_{group}_accuracy"]), group)
        )

        private = pd.read_csv(
            source / "manifests" / f"{FROZEN_NOISE_NAME}_seed{seed}_train_private.csv"
        ).set_index("sample_id")
        query_path = _artifact_path(output, row["query_artifact"])
        with np.load(query_path, allow_pickle=False) as payload:
            query_ids = payload["sample_ids"].astype(np.int64, copy=False)
        selected = private.loc[query_ids]
        weak_selected = selected["group"].astype(int).eq(baseline_weak_group)
        noisy_selected = selected["is_noisy"].astype(bool)
        weak_noisy_total = int(
            (
                private["group"].astype(int).eq(baseline_weak_group)
                & private["is_noisy"].astype(bool)
            ).sum()
        )
        weak_corrections = int((weak_selected & noisy_selected).sum())
        rows.append(
            {
                "seed": seed,
                "method": str(row["method"]),
                "baseline_worst_group": baseline_weak_group,
                "post_worst_group": post_worst_group,
                "worst_group_switched": bool(post_worst_group != baseline_weak_group),
                "weak_group_queries": int(weak_selected.sum()),
                "weak_group_query_rate": float(weak_selected.mean()),
                "weak_group_corrections": weak_corrections,
                "weak_group_noise_total": weak_noisy_total,
                "weak_group_noise_recall": float(weak_corrections / max(weak_noisy_total, 1)),
                "weak_group_accuracy_change": float(
                    row[f"group_{baseline_weak_group}_accuracy"]
                    - row[f"baseline_group_{baseline_weak_group}_accuracy"]
                ),
                "delta_wga": float(row["delta_wga"]),
                "noise_precision": float(row["noise_precision"]),
            }
        )
    return pd.DataFrame(rows)
