#!/usr/bin/env python
"""Prepare public dopanim ImageNet features and fixed noisy-label probes.

This script consumes only the public manifest created by
``prepare_dopanim_public.py``.  The zip archive's class directories are used
only as opaque file locations; neither directory name nor iNaturalist truth
is passed to the model, written to public output, or used for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "dopanim_source"
DEFAULT_PUBLIC = ROOT / "outputs" / "dopanim_public"
DEFAULT_OUTPUT = ROOT / "outputs" / "dopanim_audit_public"
PROBE_SEEDS = (0, 1, 2)
ARCHIVE_MD5 = "95e628ed85927c1ac82d7680072d317b"


class DopanimImages(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, paths: list[Path], transform: object) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            converted = image.convert("RGB")
        return self.transform(converted), index


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _extract_train_images(archive_path: Path, manifest: pd.DataFrame, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {f"{int(value)}.jpeg" for value in manifest["observation_id"].to_numpy()}
    paths_by_name: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item for item in archive.infolist()
            if not item.is_dir() and Path(item.filename).name in expected
        ]
        if len(members) != len(expected):
            raise RuntimeError("train archive does not contain every public observation image")
        for item in members:
            basename = Path(item.filename).name
            target = destination / basename
            if not target.exists():
                with archive.open(item) as source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            paths_by_name[basename] = target
    if len(paths_by_name) != len(expected):
        raise RuntimeError("duplicated or missing dopanim image names")
    return [paths_by_name[f"{int(value)}.jpeg"] for value in manifest["observation_id"]]


@torch.inference_mode()
def _extract_features(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    workers: int,
    output_path: Path,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    features = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(dataset), 2048))
    for batch_index, (images, indices) in enumerate(loader):
        values = model(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
        features[indices.numpy()] = values
        if batch_index == 0 or (batch_index + 1) % 20 == 0:
            print(f"[dopanim-features] {min((batch_index + 1) * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    features.flush()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


def _train_full_probe(features: np.ndarray, labels: np.ndarray, *, seed: int) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    model = nn.Linear(x.shape[1], int(labels.max()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 512
    history = np.empty((len(labels), 5), dtype=np.int8)
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(history.shape[1]):
        model.train()
        order = torch.randperm(len(labels), generator=generator, device=device)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = torch.cat([model(x[start : start + batch_size]) for start in range(0, len(x), batch_size)]).argmax(dim=1)
        history[:, epoch] = (predictions == y).detach().cpu().numpy().astype(np.int8)
        print(f"[dopanim-probe] seed={seed} epoch={epoch + 1}/5 noisy_acc={history[:, epoch].mean():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x[start : start + batch_size]) for start in range(0, len(x), batch_size)])
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return state, probabilities, history


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source_root).resolve()
    public_root = Path(args.public_root).resolve()
    output = Path(args.output_root).resolve()
    archive = source / "train.zip"
    if not archive.exists():
        raise FileNotFoundError(f"waiting for verified archive: {archive}")
    if _hash(archive, "md5") != ARCHIVE_MD5:
        raise RuntimeError("dopanim train.zip MD5 does not match the frozen source record")
    manifest = pd.read_csv(public_root / "public" / "manifest.csv")
    labels = np.load(public_root / "public" / "observed_labels.npy", allow_pickle=False).astype(np.int64)
    if len(manifest) != len(labels) or not np.array_equal(manifest["sample_id"], np.arange(len(manifest))):
        raise RuntimeError("unexpected dopanim public manifest")
    if not ((0 <= labels).all() & (labels < 15).all()):
        raise RuntimeError("unexpected public human labels")

    image_paths = _extract_train_images(archive, manifest, output / "images_flat")
    feature_root = output / "features"
    probe_root = output / "probes"
    feature_root.mkdir(parents=True, exist_ok=True)
    probe_root.mkdir(parents=True, exist_ok=True)
    feature_path = feature_root / "train_resnet50_imagenet1k_v2.npy"
    if feature_path.exists() and not args.force_features:
        features = np.load(feature_path, mmap_mode="r")
        print("[dopanim-features] using cached array", flush=True)
    else:
        dataset = DopanimImages(image_paths, ResNet50_Weights.IMAGENET1K_V2.transforms())
        features = _extract_features(
            dataset,
            batch_size=int(args.feature_batch_size),
            workers=int(args.workers),
            output_path=feature_path,
        )
    if features.shape != (len(labels), 2048):
        raise RuntimeError(f"unexpected feature shape: {features.shape}")

    for seed in PROBE_SEEDS:
        dynamics_path = probe_root / f"probe_seed{seed}.dynamics.npz"
        checkpoint_path = probe_root / f"probe_seed{seed}.pt"
        if dynamics_path.exists() and checkpoint_path.exists() and not args.force_probes:
            print(f"[dopanim-probe] cached seed={seed}", flush=True)
            continue
        state, probabilities, history = _train_full_probe(np.asarray(features), labels, seed=seed)
        torch.save({"state": state, "epochs": 5}, checkpoint_path)
        np.savez_compressed(
            dynamics_path,
            sample_ids=np.arange(len(labels), dtype=np.int64),
            train_probabilities=probabilities,
            correctness_history=history,
        )

    result = {
        "dataset": "dopanim",
        "population_count": int(len(labels)),
        "class_count": 15,
        "public_manifest_sha256": _hash(public_root / "public" / "manifest.csv"),
        "observed_labels_sha256": _hash(public_root / "public" / "observed_labels.npy"),
        "train_archive_md5": ARCHIVE_MD5,
        "feature_path": str(feature_path),
        "feature_shape": list(features.shape),
        "probe_seeds": list(PROBE_SEEDS),
        "probe_training": "all public human labels; fixed five epochs; no clean validation or early stopping",
        "private_ground_truth_loaded": False,
    }
    _atomic_json(result, output / "preparation_manifest.json")
    print(f"[dopanim-audit-prepared] {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--public-root", default=str(DEFAULT_PUBLIC))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
