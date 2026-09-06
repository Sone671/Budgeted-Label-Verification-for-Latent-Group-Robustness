#!/usr/bin/env python
"""Prepare public Food-101N probes for frozen audit-budget confirmation.

Verification values are stored separately and never enter feature extraction,
public noisy-label probe training, or NoiseScore construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.training import train_linear_head  # noqa: E402


MISSING_VERIFIED_PATH = "hot_and_sour_soup/a49e6fb26e356c6c41a37b5c18176355.jpg"
PROBE_SEEDS = (0, 1, 2)


class Food101NImages(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, relative_paths: list[str], transform: object) -> None:
        self.root = root
        self.relative_paths = relative_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.relative_paths[index]) as image:
            converted = image.convert("RGB")
        return self.transform(converted), index


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


def _deterministic_validation_mask(paths: list[str], audit_ids: np.ndarray) -> np.ndarray:
    audit_mask = np.zeros(len(paths), dtype=bool)
    audit_mask[audit_ids] = True
    values = np.asarray(
        [int.from_bytes(hashlib.sha256(path.encode("utf-8")).digest()[:8], "big") for path in paths],
        dtype=np.uint64,
    )
    return (~audit_mask) & ((values % np.uint64(10)) == 0)


@torch.inference_mode()
def _extract_features(
    dataset: Dataset[tuple[torch.Tensor, int]],
    model: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
    output_path: Path,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(dataset), 2048)
    )
    for batch_index, (images, indices) in enumerate(loader):
        features = model(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
        output[indices.numpy()] = features
        if (batch_index + 1) % 100 == 0 or batch_index == 0:
            print(f"[food101n-features] {(batch_index + 1) * batch_size}/{len(dataset)}", flush=True)
    output.flush()
    return output


def _load_metadata(release_root: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    meta = release_root / "meta"
    classes = [line.strip() for line in (meta / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes and classes[0] == "class_name":
        classes = classes[1:]
    if len(classes) != 101 or len(set(classes)) != len(classes):
        raise RuntimeError("expected 101 unique Food-101N classes")
    class_to_id = {name: index for index, name in enumerate(classes)}
    image_frame = pd.read_csv(meta / "imagelist.tsv", sep="\t")
    image_paths = image_frame.iloc[:, 0].astype(str).tolist()
    if len(image_paths) != 310_009 or len(set(image_paths)) != len(image_paths):
        raise RuntimeError("unexpected Food-101N public image list")
    image_root = release_root / "images"
    if not all((image_root / path).exists() for path in image_paths):
        raise RuntimeError("public image list contains missing image files")
    noisy_labels = np.asarray([class_to_id[path.split("/", 1)[0]] for path in image_paths], dtype=np.int64)

    verified = pd.read_csv(meta / "verified_train.tsv", sep="\t")
    verified.columns = ["path", "verification"]
    if len(verified) != 52_868 or verified["path"].duplicated().any():
        raise RuntimeError("unexpected Food-101N verified training manifest")
    public_index = {path: index for index, path in enumerate(image_paths)}
    missing = verified.loc[~verified["path"].isin(public_index)]
    if missing["path"].tolist() != [MISSING_VERIFIED_PATH] or missing["verification"].tolist() != [0]:
        raise RuntimeError("unexpected verified/public image-list mismatch")
    verified = verified.loc[verified["path"] != MISSING_VERIFIED_PATH].copy()
    audit_ids = np.asarray([public_index[path] for path in verified["path"]], dtype=np.int64)
    is_error = (1 - verified["verification"].to_numpy(dtype=np.int8)).astype(bool)
    if len(audit_ids) != 52_867 or len(np.unique(audit_ids)) != len(audit_ids):
        raise RuntimeError("unexpected Food-101N audit population")
    if np.unique(noisy_labels[audit_ids]).size != 101:
        raise RuntimeError("audit population must cover all observed classes")
    return image_paths, noisy_labels, audit_ids, is_error, {
        "classes_sha256": _sha256(meta / "classes.txt"),
        "imagelist_sha256": _sha256(meta / "imagelist.tsv"),
        "verified_train_sha256": _sha256(meta / "verified_train.tsv"),
        "verified_val_sha256": _sha256(meta / "verified_val.tsv"),
        "missing_verified_path": MISSING_VERIFIED_PATH,
        "missing_verified_label": 0,
    }


def run(args: argparse.Namespace) -> None:
    release_root = Path(args.release_root).resolve()
    output_root = Path(args.output_root).resolve()
    feature_root = output_root / "features"
    probe_root = output_root / "probes"
    public_root = output_root / "public"
    private_root = output_root / "private"
    for directory in (feature_root, probe_root, public_root, private_root):
        directory.mkdir(parents=True, exist_ok=True)

    image_paths, noisy_labels, audit_ids, is_error, source_hashes = _load_metadata(release_root)
    validation_mask = _deterministic_validation_mask(image_paths, audit_ids)
    validation_ids = np.flatnonzero(validation_mask).astype(np.int64)
    train_ids = np.flatnonzero(~validation_mask).astype(np.int64)
    if len(validation_ids) == 0 or len(train_ids) == 0 or not np.isin(audit_ids, train_ids).all():
        raise RuntimeError("invalid public train/validation split")
    if np.unique(noisy_labels[validation_ids]).size != 101:
        raise RuntimeError("public validation split must cover all classes")

    public_manifest = pd.DataFrame(
        {
            "sample_id": np.arange(len(image_paths), dtype=np.int64),
            "path": image_paths,
            "noisy_label": noisy_labels,
            "public_probe_split": np.where(validation_mask, "validation", "train"),
        }
    )
    public_manifest.to_csv(public_root / "imagelist_public.csv", index=False)
    np.save(public_root / "train_sample_ids.npy", train_ids)
    np.save(public_root / "validation_sample_ids.npy", validation_ids)
    np.save(public_root / "audit_population_sample_ids.npy", audit_ids)
    np.save(private_root / "audit_is_error.npy", is_error)

    feature_path = feature_root / "all_resnet50_imagenet1k_v2.npy"
    if feature_path.exists() and not args.force_features:
        features = np.load(feature_path, mmap_mode="r")
        print("[food101n-features] using cached array", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.IMAGENET1K_V2
        dataset = Food101NImages(release_root / "images", image_paths, weights.transforms())
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        backbone.to(device).eval()
        print(f"[food101n-features] extracting {len(dataset)} images on {device}", flush=True)
        features = _extract_features(
            dataset,
            backbone,
            device,
            batch_size=int(args.feature_batch_size),
            workers=int(args.workers),
            output_path=feature_path,
        )
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if features.shape != (len(image_paths), 2048):
        raise RuntimeError(f"unexpected Food-101N feature shape: {features.shape}")

    training_config = {
        "epochs": 5,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "selection_metric": "balanced_accuracy",
        "device": "auto",
    }
    for seed in PROBE_SEEDS:
        checkpoint_path = probe_root / f"probe_seed{seed}.pt"
        dynamics_path = probe_root / f"probe_seed{seed}.dynamics.npz"
        if checkpoint_path.exists() and dynamics_path.exists() and not args.force_probes:
            print(f"[food101n-probe] cached seed={seed}", flush=True)
            continue
        print(f"[food101n-probe] training seed={seed}", flush=True)
        train_features = np.asarray(features[train_ids], dtype=np.float32)
        validation_features = np.asarray(features[validation_ids], dtype=np.float32)
        result = train_linear_head(
            train_features=train_features,
            train_labels=noisy_labels[train_ids],
            val_features=validation_features,
            val_labels=noisy_labels[validation_ids],
            config=training_config,
            seed=seed,
            record_dynamics=True,
        )
        torch.save(
            {
                "best_state": result.best_state,
                "initial_state": result.initial_state,
                "best_epoch": result.best_epoch,
                "best_validation_metric": result.best_validation_metric,
            },
            checkpoint_path,
        )
        np.savez_compressed(
            dynamics_path,
            sample_ids=train_ids,
            train_probabilities=result.train_probabilities.astype(np.float32),
            correctness_history=result.correctness_history.astype(np.int8),
            best_epoch=np.asarray(result.best_epoch, dtype=np.int32),
        )
        del train_features, validation_features, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "dataset": "Food-101N",
        "release_root": str(release_root),
        "image_count": len(image_paths),
        "class_count": 101,
        "audit_population_count": len(audit_ids),
        "source_hashes": source_hashes,
        "feature_path": str(feature_path),
        "feature_shape": list(features.shape),
        "probe_seeds": list(PROBE_SEEDS),
        "training": training_config,
        "public_validation_rule": "non-audit image with sha256(path) integer modulo 10 equal to zero",
        "verification_labels_role": "private audit simulation and post-freeze evaluation only",
    }
    _atomic_json(manifest, output_root / "preparation_manifest.json")
    print(f"[food101n-prepared] {output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        default=str(ROOT / "data" / "food101n_source" / "Food-101N_release"),
    )
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "sparsity_gated_food101n"))
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
