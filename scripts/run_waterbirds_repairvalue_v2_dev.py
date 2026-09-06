#!/usr/bin/env python
"""LOCAL DEV rehearsal of the Waterbirds noise-gated RepairValue-Q v2 protocol
on development seeds 20--29. Development evidence only; never pool with the
formal 40--59 confirmation block."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Required by deterministic CUDA matrix multiplications.  This must be set
# before the first CUDA context is created by preflight or formal training.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

from robust_verify.absolute_utility import audit_absolute_utility, prepare_results
from robust_verify.config import load_config
from robust_verify.end_to_end import build_image_classifier, run_end_to_end
from robust_verify.stage0 import run_stage0
from robust_verify.waterbirds_repairvalue_v2_dev_confirmatory import (
    ARTIFACT_COLUMNS,
    FROZEN_SEEDS,
    PRIMARY_METHOD,
    PROTOCOL_VERSION,
    build_group_mechanism_audit,
    build_preparation_manifest,
    protocol_sha256,
    sha256_file,
    summarize_confirmatory_results,
    validate_frozen_config,
    validate_result_frame,
    validate_seed_artifacts,
    verify_preparation_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "waterbirds_e2e_repairvalue_v2_seed20_29_dev.yaml"
# DEV-only relaxation of the formal server floor (6.0 GiB): the local RTX 3060
# Laptop reports 5.9995 GiB. Scientific protocol content is unchanged.
MIN_CUDA_MEMORY_GIB = 5.0
MIN_FREE_DISK_GIB = 30.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
    if args.output_root is not None:
        config["project"]["output_dir"] = str(Path(args.output_root).expanduser().resolve())
    validate_frozen_config(config)
    return config


def _output_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_dir"]).resolve()


def _source_root(config: dict[str, Any]) -> Path:
    return _output_root(config) / "source"


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ValueError(f"Cannot find an existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _data_preflight(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    root = Path(data["root"]).resolve()
    metadata = root / str(data["metadata_csv"])
    if not metadata.is_file():
        raise FileNotFoundError(f"Required Waterbirds metadata is missing: {metadata}")

    metadata_head = pd.read_csv(metadata, nrows=1)
    required_metadata = {"img_filename", "y", "place", "split"}
    if not required_metadata.issubset(metadata_head.columns):
        missing = sorted(required_metadata.difference(metadata_head.columns))
        raise ValueError(f"Waterbirds metadata lacks columns: {missing}")
    first_image = root / str(metadata_head.iloc[0]["img_filename"])
    if not first_image.is_file():
        raise FileNotFoundError(f"First Waterbirds image is unavailable: {first_image}")

    disk = shutil.disk_usage(_existing_ancestor(_output_root(config)))
    free_gib = disk.free / (1024**3)
    if free_gib < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"At least {MIN_FREE_DISK_GIB:.0f} GiB free disk is required; found {free_gib:.1f} GiB"
        )
    return {
        "data_root": str(root),
        "metadata_bytes": metadata.stat().st_size,
        "first_image": str(first_image),
        "free_disk_gib": free_gib,
    }


def _gpu_preflight(config: dict[str, Any], *, run_batch_smoke: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The formal run requires CUDA; torch.cuda.is_available() is false")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_gib = properties.total_memory / (1024**3)
    if total_gib < MIN_CUDA_MEMORY_GIB:
        raise RuntimeError(
            f"The protocol requires at least {MIN_CUDA_MEMORY_GIB:.0f} GiB VRAM; "
            f"found {total_gib:.1f} GiB on {properties.name}"
        )

    smoke_completed = False
    if run_batch_smoke:
        e2e = config["end_to_end"]
        model = build_image_classifier("resnet50", 2, pretrained=bool(e2e["pretrained"]))
        model.train().cuda()
        images = torch.randn(
            int(e2e["batch_size"]),
            3,
            int(e2e["input_size"]),
            int(e2e["input_size"]),
            device="cuda",
        )
        labels = torch.arange(len(images), device="cuda") % 2
        with torch.autocast(device_type="cuda", enabled=bool(e2e["amp"])):
            loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        smoke_completed = True
        del loss, labels, images, model
        torch.cuda.empty_cache()

    return {
        "gpu_name": properties.name,
        "cuda_device_index": device_index,
        "cuda_memory_gib": total_gib,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "batch_size_smoke_completed": smoke_completed,
    }


def _environment() -> dict[str, Any]:
    import torchvision

    code_paths = [
        REPOSITORY_ROOT / "src" / "robust_verify" / "waterbirds_repairvalue_v2_dev_confirmatory.py",
        REPOSITORY_ROOT / "src" / "robust_verify" / "end_to_end.py",
        REPOSITORY_ROOT / "src" / "robust_verify" / "scoring.py",
        REPOSITORY_ROOT / "src" / "robust_verify" / "modern_baselines.py",
        Path(__file__).resolve(),
    ]
    return {
        "created_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "code_sha256": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path) for path in code_paths
        },
    }


def command_preflight(config: dict[str, Any]) -> dict[str, Any]:
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(config),
        "data": _data_preflight(config),
        "gpu": _gpu_preflight(config, run_batch_smoke=True),
        "environment": _environment(),
    }
    output = _output_root(config) / "preflight.json"
    _write_json(output, report)
    print(
        f"Preflight PASS: {report['gpu']['gpu_name']} ({report['gpu']['cuda_memory_gib']:.1f} GiB)"
    )
    print(f"Wrote {output}")
    return report


def command_prepare(config: dict[str, Any]) -> dict[str, Any]:
    _data_preflight(config)
    source = _source_root(config)
    marker = source / "preparation_manifest.json"
    if marker.is_file():
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        verify_preparation_manifest(source, manifest, config)
        print(f"Prepared source already verified: {source}")
        return manifest
    if source.exists() and any(source.iterdir()):
        raise RuntimeError(
            f"Refusing to reuse an unsealed source directory: {source}. "
            "Choose a new --output-root; no partial source is deleted automatically."
        )

    stage_config = copy.deepcopy(config)
    stage_config["project"] = {"output_dir": str(source)}
    print(
        f"Generating fresh corruption manifests for seeds {FROZEN_SEEDS[0]}-{FROZEN_SEEDS[-1]}..."
    )
    run_stage0(stage_config)
    manifest = build_preparation_manifest(source, config)
    manifest["created_utc"] = _utc_now()
    _write_json(marker, manifest)
    verify_preparation_manifest(source, manifest, config)
    print(f"Stage-0 source sealed and verified: {marker}")
    return manifest


def _load_preparation(config: dict[str, Any]) -> dict[str, Any]:
    source = _source_root(config)
    marker = source / "preparation_manifest.json"
    if not marker.is_file():
        raise RuntimeError(f"Prepared source is absent; run prepare first: {marker}")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    verify_preparation_manifest(source, manifest, config)
    return manifest


def _attempts(output_root: Path, seed: int) -> list[Path]:
    seed_root = output_root / "shards" / f"seed_{seed}"
    return sorted(path for path in seed_root.glob("attempt_*") if path.is_dir())


def _completed_attempts(output_root: Path, seed: int) -> list[Path]:
    return [path for path in _attempts(output_root, seed) if (path / "COMPLETE.json").is_file()]


def _next_attempt(output_root: Path, seed: int) -> Path:
    indices = []
    for path in _attempts(output_root, seed):
        try:
            indices.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return output_root / "shards" / f"seed_{seed}" / f"attempt_{max(indices, default=0) + 1:03d}"


def _validate_completed_attempt(config: dict[str, Any], seed: int, attempt: Path) -> pd.DataFrame:
    complete_path = attempt / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("protocol_sha256") != protocol_sha256(config):
        raise ValueError(f"Completed seed {seed} has a different protocol hash")
    results_path = attempt / "end_to_end" / "results.csv"
    if complete.get("results_sha256") != sha256_file(results_path):
        raise ValueError(f"Completed seed {seed} results hash mismatch")
    frame = pd.read_csv(results_path)
    validate_seed_artifacts(frame, seed=seed, shard_root=attempt, source_root=_source_root(config))
    return frame


def _run_one_seed(config: dict[str, Any], seed: int) -> Path:
    output_root = _output_root(config)
    completed = _completed_attempts(output_root, seed)
    if len(completed) > 1:
        raise RuntimeError(f"Seed {seed} has multiple completed attempts: {completed}")
    if completed:
        _validate_completed_attempt(config, seed, completed[0])
        print(f"Seed {seed}: existing completed shard verified; skipping")
        return completed[0]

    attempt = _next_attempt(output_root, seed)
    attempt.mkdir(parents=True, exist_ok=False)
    shard_config = copy.deepcopy(config)
    shard_config["project"] = {
        "output_dir": str(attempt),
        "input_dir": str(_source_root(config)),
    }
    shard_config["experiment"]["seeds"] = [seed]
    validate_frozen_config(shard_config, expected_seeds=[seed])

    _write_json(attempt / "frozen_shard_config.json", shard_config)
    run_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(config),
        "seed": seed,
        "attempt": attempt.name,
        "status": "running",
        "source_preparation_sha256": sha256_file(
            _source_root(config) / "preparation_manifest.json"
        ),
        "environment": _environment(),
    }
    _write_json(attempt / "RUNNING.json", run_manifest)
    print(f"Seed {seed}: starting {attempt.name}")
    try:
        frame = run_end_to_end(shard_config)
        validation = validate_seed_artifacts(
            frame, seed=seed, shard_root=attempt, source_root=_source_root(config)
        )
        results_path = attempt / "end_to_end" / "results.csv"
        complete = {
            **run_manifest,
            "status": "complete",
            "completed_utc": _utc_now(),
            "results": results_path.relative_to(attempt).as_posix(),
            "results_sha256": sha256_file(results_path),
            "validation": validation,
        }
        _write_json(attempt / "COMPLETE.json", complete)
        (attempt / "RUNNING.json").unlink(missing_ok=True)
        print(f"Seed {seed}: PASS ({len(frame)} result rows)")
        return attempt
    except BaseException as exc:
        failed = {
            **run_manifest,
            "status": "failed",
            "failed_utc": _utc_now(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(attempt / "FAILED.json", failed)
        (attempt / "RUNNING.json").unlink(missing_ok=True)
        raise


def command_run(config: dict[str, Any], seeds: list[int] | None) -> list[Path]:
    _load_preparation(config)
    _gpu_preflight(config, run_batch_smoke=True)
    selected = list(FROZEN_SEEDS if not seeds else dict.fromkeys(seeds))
    unknown = sorted(set(selected).difference(FROZEN_SEEDS))
    if unknown:
        raise ValueError(f"Seeds are outside the frozen 40-59 set: {unknown}")
    return [_run_one_seed(config, seed) for seed in selected]


def _write_audit(output_root: Path, frame: pd.DataFrame) -> dict[str, Any]:
    audit_root = output_root / "aggregate" / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_results(frame, "waterbirds")
    audit = audit_absolute_utility(
        prepared,
        include_methods=[
            PRIMARY_METHOD,
            "random",
            "noise_score",
            "expected_repair_value",
        ],
        n_bootstrap=10_000,
        rng_seed=20260805,
    )
    audit.absolute_vs_no_correction.to_csv(
        audit_root / "absolute_vs_no_correction.csv", index=False
    )
    audit.paired_vs_loss.to_csv(audit_root / "paired_vs_loss.csv", index=False)
    audit.per_seed_absolute.to_csv(
        audit_root / "per_seed_absolute_vs_no_correction.csv", index=False
    )
    audit.per_seed_vs_loss.to_csv(audit_root / "per_seed_vs_loss.csv", index=False)
    audit.coverage.to_csv(audit_root / "coverage.csv", index=False)

    mechanism = build_group_mechanism_audit(
        frame, output_root=output_root, source_root=output_root / "source"
    )
    mechanism.to_csv(audit_root / "baseline_worst_group_mechanism.csv", index=False)

    decision = summarize_confirmatory_results(frame)
    _write_json(audit_root / "confirmatory_decision.json", decision)
    paired = decision["paired_vs_noise_score"]
    absolute = decision["absolute_utility"]
    precision = decision["precision_noninferiority"]
    lines = [
        "# Waterbirds noise-gated RepairValue-Q v2 confirmation decision",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`",
        "",
        f"Primary gate: **{'PASS' if decision['primary_gate_passed'] else 'NOT PASSED'}**",
        "",
        (
            "- Mean paired Delta-WGA (v2 minus NoiseScore): "
            f"{100 * paired['mean_delta_wga_difference']:+.3f} pp"
        ),
        (
            "- Paired seed-bootstrap 95% interval: "
            f"[{100 * paired['bootstrap_95_ci'][0]:+.3f}, "
            f"{100 * paired['bootstrap_95_ci'][1]:+.3f}] pp"
        ),
        f"- Positive paired seeds: {paired['positive_seed_count']}/20",
        "",
        (
            "- Secondary absolute RepairValue-Q Delta-WGA: "
            f"{100 * absolute['mean_delta_wga']:+.3f} pp"
        ),
        (
            "- Secondary absolute bootstrap interval: "
            f"[{100 * absolute['bootstrap_95_ci'][0]:+.3f}, "
            f"{100 * absolute['bootstrap_95_ci'][1]:+.3f}] pp"
        ),
        f"- RepairValue-Q query noise precision: {100 * absolute['mean_noise_precision']:.3f}%",
        (
            "- Precision drop versus NoiseScore: "
            f"{100 * precision['precision_drop']:+.3f} pp "
            f"(noninferiority margin {100 * precision['noninferiority_margin']:.1f} pp)"
        ),
        (
            "- Paired precision-difference bootstrap interval: "
            f"[{100 * precision['paired_precision_difference_bootstrap_95_ci'][0]:+.3f}, "
            f"{100 * precision['paired_precision_difference_bootstrap_95_ci'][1]:+.3f}] pp"
        ),
        "",
        (
            "All four primary gates are required: paired CI, at least 15/20 "
            "positive paired seeds, positive absolute CI, and precision "
            "noninferiority versus NoiseScore. Random, Loss, v1, and group "
            "composition are secondary."
        ),
        "",
    ]
    (audit_root / "CONFIRMATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def command_aggregate(config: dict[str, Any]) -> Path:
    _load_preparation(config)
    output_root = _output_root(config)
    frames: list[pd.DataFrame] = []
    files_to_hash: set[Path] = {_source_root(config) / "preparation_manifest.json"}
    for seed in FROZEN_SEEDS:
        completed = _completed_attempts(output_root, seed)
        if len(completed) != 1:
            raise RuntimeError(
                f"Seed {seed} must have exactly one completed attempt; found {completed}"
            )
        attempt = completed[0]
        frame = _validate_completed_attempt(config, seed, attempt).copy()
        shard_prefix = attempt.relative_to(output_root)
        for column in ARTIFACT_COLUMNS:
            frame[column] = frame[column].map(
                lambda value, prefix=shard_prefix: (prefix / str(value)).as_posix()
            )
        frame["dataset"] = "waterbirds"
        frame["protocol_version"] = PROTOCOL_VERSION
        frame["shard_root"] = shard_prefix.as_posix()
        frames.append(frame)
        files_to_hash.update(
            {
                attempt / "COMPLETE.json",
                attempt / "frozen_shard_config.json",
                attempt / "end_to_end" / "resolved_config.json",
                attempt / "end_to_end" / "results.csv",
            }
        )
        for column in ARTIFACT_COLUMNS:
            files_to_hash.update(output_root / value for value in frame[column].tolist())

    aggregate = (
        pd.concat(frames, ignore_index=True).sort_values(["seed", "method"]).reset_index(drop=True)
    )
    validate_result_frame(aggregate, expected_seeds=FROZEN_SEEDS)
    aggregate_root = output_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    results_path = aggregate_root / "results.csv"
    aggregate.to_csv(results_path, index=False)
    decision = _write_audit(output_root, aggregate)
    files_to_hash.update(
        {
            results_path,
            aggregate_root / "audit" / "confirmatory_decision.json",
            aggregate_root / "audit" / "absolute_vs_no_correction.csv",
            aggregate_root / "audit" / "paired_vs_loss.csv",
            aggregate_root / "audit" / "per_seed_absolute_vs_no_correction.csv",
            aggregate_root / "audit" / "per_seed_vs_loss.csv",
            aggregate_root / "audit" / "coverage.csv",
            aggregate_root / "audit" / "baseline_worst_group_mechanism.csv",
            aggregate_root / "audit" / "CONFIRMATORY_REPORT.md",
        }
    )
    artifact_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(config),
        "created_utc": _utc_now(),
        "primary_gate_passed": decision["primary_gate_passed"],
        "files": [
            {
                "path": path.resolve().relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files_to_hash)
        ],
    }
    _write_json(aggregate_root / "artifact_manifest.json", artifact_manifest)
    print(f"Aggregate PASS: {results_path} ({len(aggregate)} rows)")
    print(
        "Primary gate: "
        + ("PASS" if decision["primary_gate_passed"] else "NOT PASSED; retain the null result")
    )
    return results_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "run", "aggregate", "all"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", help="Server Waterbirds root; paths are the only override")
    parser.add_argument("--output-root", help="Fresh server output root")
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        choices=FROZEN_SEEDS,
        help="Run one frozen seed; repeat as needed. Omit to run all unfinished seeds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _resolved_config(args)
    if args.command == "preflight":
        command_preflight(config)
    elif args.command == "prepare":
        command_prepare(config)
    elif args.command == "run":
        command_run(config, args.seed)
    elif args.command == "aggregate":
        command_aggregate(config)
    elif args.command == "all":
        command_preflight(config)
        command_prepare(config)
        command_run(config, args.seed)
        command_aggregate(config)


if __name__ == "__main__":
    main()
