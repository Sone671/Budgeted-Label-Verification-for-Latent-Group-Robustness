#!/usr/bin/env python
"""Prepare the public/private dopanim metadata boundary.

One hard human label is selected per training observation using only the
public annotation record ID and its likelihood vector.  iNaturalist truth is
stored separately for post-freeze evaluation and is not used to select a
label, sample, split, or ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "dopanim_source"
DEFAULT_OUTPUT = ROOT / "outputs" / "dopanim_public"
CLASSES = (
    "German Yellowjacket", "European Paper Wasp", "Yellow-legged Hornet",
    "European Hornet", "Brown Hare", "Black-tailed Jackrabbit", "Marsh Rabbit",
    "Desert Cottontail", "European Rabbit", "Eurasian Red Squirrel",
    "American Red Squirrel", "Douglas' Squirrel", "Cheetah", "Jaguar", "Leopard",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _hard_label(likelihoods: dict[str, float]) -> int:
    candidates = [
        (float(likelihoods.get(class_name, -1.0)), -class_id, class_id)
        for class_id, class_name in enumerate(CLASSES)
    ]
    return int(max(candidates)[2])


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    public = output / "public"
    private = output / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    task_path = source / "task_data.json"
    annotation_path = source / "annotation_data.json"
    task_data = json.loads(task_path.read_text(encoding="utf-8"))
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))

    annotations_by_observation: dict[int, list[tuple[str, int]]] = {}
    for annotation_id, record in annotation_data.items():
        observation_id = int(record["observation_id"])
        label = _hard_label(record["likelihoods"])
        annotations_by_observation.setdefault(observation_id, []).append(
            (str(annotation_id), label)
        )

    rows: list[dict] = []
    truth: list[int] = []
    selected_annotation_ids: list[str] = []
    for raw_observation_id, record in task_data.items():
        if record.get("split") != "train":
            continue
        observation_id = int(raw_observation_id)
        candidates = annotations_by_observation.get(observation_id, [])
        if not candidates:
            continue
        selected_id, noisy_label = min(
            candidates,
            key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
        )
        taxon_name = str(record["taxon_name"])
        if taxon_name not in CLASS_TO_ID:
            raise RuntimeError(f"unknown private class for observation {observation_id}")
        sample_id = len(rows)
        rows.append(
            {
                "sample_id": sample_id,
                "observation_id": observation_id,
                "image_filename": f"{observation_id}.jpeg",
                "noisy_label": noisy_label,
                "public_split": "train",
            }
        )
        truth.append(CLASS_TO_ID[taxon_name])
        selected_annotation_ids.append(selected_id)

    if len(rows) < 1_000 or len(set(row["observation_id"] for row in rows)) != len(rows):
        raise RuntimeError("unexpected dopanim annotated training population")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(public / "manifest.csv", index=False)
    np.save(public / "sample_ids.npy", manifest["sample_id"].to_numpy(dtype=np.int64))
    np.save(public / "observed_labels.npy", manifest["noisy_label"].to_numpy(dtype=np.int64))
    np.save(private / "ground_truth_labels.npy", np.asarray(truth, dtype=np.int64))
    np.save(private / "audit_is_error.npy", manifest["noisy_label"].to_numpy(dtype=np.int64) != np.asarray(truth, dtype=np.int64))
    pd.DataFrame(
        {"sample_id": manifest["sample_id"], "annotation_id": selected_annotation_ids}
    ).to_csv(public / "selected_annotation_ids.csv", index=False)

    preparation = {
        "dataset": "dopanim",
        "class_count": len(CLASSES),
        "class_order": list(CLASSES),
        "population_count": len(manifest),
        "selection_rule": "minimum SHA256(annotation_id) per training observation; hard label is likelihood argmax with class-order tie break",
        "task_data_sha256": _sha256(task_path),
        "annotation_data_sha256": _sha256(annotation_path),
        "ground_truth_role": "private post-freeze evaluation only",
        "truth_statistics_not_recorded": True,
    }
    _atomic_json(preparation, output / "preparation_manifest.json")
    print(f"[dopanim-public-prepared] {output} population={len(manifest)}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
