"""P0 reproducibility and protocol-control helpers.

This module deliberately contains no model-selection logic.  It records the
inputs and outputs of an experiment so that downstream analyses can determine
whether a result is eligible for confirmatory use.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RUN_MANIFEST_VERSION = "p0-run-manifest-v1"
ARTIFACT_MANIFEST_VERSION = "p0-artifact-manifest-v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash the fully resolved configuration used by one run."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _file_inventory(root: Path, patterns: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            records.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def write_or_validate_run_manifest(
    output_root: str | Path,
    config: dict[str, Any],
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze a run configuration and its immutable Stage-0 inputs.

    A resumed run is accepted only when both the resolved config and the
    feature/manifest fingerprints match.  The immutable source can differ from
    the output directory, which enables an isolated seed-0 reproduction.
    """
    output_root = Path(output_root)
    source_root = Path(source_root or output_root)
    path = output_root / "stage1" / "run_manifest.json"
    input_files = _file_inventory(
        source_root,
        ("features/*.npz", "manifests/*.csv"),
    )
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_VERSION,
        "config_sha256": config_fingerprint(config),
        "resolved_config": config,
        "source_root": str(source_root.resolve()),
        "input_files": input_files,
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable_keys = ("schema_version", "config_sha256", "source_root", "input_files")
        mismatch = [key for key in immutable_keys if existing.get(key) != payload.get(key)]
        if mismatch:
            raise RuntimeError(
                "Cannot resume a run with different frozen inputs: " + ", ".join(mismatch)
            )
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def write_query_artifact(
    path: str | Path,
    *,
    method: str,
    noise_name: str,
    seed: int,
    budget_fraction: float,
    sample_ids: np.ndarray,
    proxy_name: str,
    proxy_before: float,
    proxy_after: float,
    checkpoint_path: str | Path,
) -> None:
    """Write a self-describing companion for a saved query-ID array."""
    proxy_before = float(proxy_before)
    proxy_after = float(proxy_after)
    payload = {
        "schema_version": "p0-query-artifact-v1",
        "method": method,
        "noise_name": noise_name,
        "seed": int(seed),
        "budget_fraction": float(budget_fraction),
        "sample_ids": np.asarray(sample_ids, dtype=np.int64).tolist(),
        "proxy": {
            "name": proxy_name,
            "higher_is_better": True,
            "before": proxy_before,
            "after": proxy_after,
            "change": proxy_after - proxy_before,
        },
        "retrain_checkpoint": str(checkpoint_path).replace("\\", "/"),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def write_artifact_manifest(output_root: str | Path) -> Path:
    """Hash all P0 raw outputs once a Stage-1 run has completed."""
    root = Path(output_root)
    stage1 = root / "stage1"
    records = _file_inventory(
        root,
        (
            "stage1/results.csv",
            "stage1/score_correlations.csv",
            "stage1/v9_metadata.csv",
            "stage1/probes/*",
            "stage1/checkpoints/**/*",
            "stage1/queries/**/*",
        ),
    )
    payload = {
        "schema_version": ARTIFACT_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    path = stage1 / "artifact_manifest.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return path

