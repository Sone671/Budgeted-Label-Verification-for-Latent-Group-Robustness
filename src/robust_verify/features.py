from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from robust_verify.utils import resolve_device, seed_everything


def extract_acs_tabular_features(
    canonical: pd.DataFrame,
    output_dir: str | Path,
    feature_config: dict[str, Any],
) -> dict[str, Path]:
    """Fit the frozen ACS transformer on train only and transform all splits."""

    numeric = list(feature_config.get("numeric_columns", ["AGEP", "WKHP"]))
    categorical = list(
        feature_config.get(
            "categorical_columns",
            ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "RAC1P"],
        )
    )
    feature_columns = numeric + categorical
    missing = sorted(set(feature_columns).difference(canonical.columns))
    if missing:
        raise ValueError(f"ACS canonical data missing feature columns: {missing}")
    forbidden = {"SEX", "place", "group", "clean_label"}.intersection(
        feature_columns
    )
    if forbidden:
        raise ValueError(f"private ACS fields cannot enter public features: {forbidden}")

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("standardize", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    train = canonical.loc[canonical["split"] == "train"]
    if not len(train):
        raise ValueError("ACS training split is empty")
    transformer.fit(train[feature_columns])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    split_shapes: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        frame = canonical.loc[canonical["split"] == split]
        features = np.asarray(
            transformer.transform(frame[feature_columns]), dtype=np.float32
        )
        sample_ids = frame["sample_id"].to_numpy(dtype=np.int64)
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, features=features, sample_ids=sample_ids)
        paths[split] = path
        split_shapes[split] = [int(value) for value in features.shape]

    feature_names = transformer.get_feature_names_out().tolist()
    metadata = {
        "backbone": "acs_onehot_v1",
        "fit_split": "train",
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "forbidden_source_column": "SEX",
        "feature_dimension": len(feature_names),
        "feature_names": feature_names,
        "split_shapes": split_shapes,
    }
    (output_dir / "acs_preprocessor.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


class ImageManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: str | Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.root / str(row["image_relpath"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["sample_id"])


def build_backbone(pretrained: bool = True) -> tuple[nn.Module, Any, int]:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    transform = (
        weights.transforms()
        if weights is not None
        else ResNet50_Weights.DEFAULT.transforms()
    )
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, transform, feature_dim


def build_text_backbone(model_name: str = "all-MiniLM-L6-v2") -> tuple[Any, int]:
    # Try sentence-transformers first (better quality, requires network for first download)
    try:
        from sentence_transformers import SentenceTransformer
        import socket
        socket.setdefaulttimeout(5)

        model = SentenceTransformer(model_name, device="cpu")
        feature_dim = model.get_sentence_embedding_dimension()
        return model, feature_dim
    except Exception:
        pass

    # Fallback: TF-IDF vectorizer (no download needed)
    print(f"Warning: Could not load {model_name}, using TF-IDF fallback (dim=768)")
    model = TfidfVectorizer(max_features=768, stop_words="english")
    model._is_tfidf = True
    model._tfidf_dim = 768
    return model, 768


class TextManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        return str(row["comment_text"]), int(row["sample_id"])


@torch.inference_mode()
def extract_split_features(
    frame: pd.DataFrame,
    data_root: str | Path,
    output_path: str | Path,
    feature_config: dict[str, Any],
    seed: int = 0,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    seed_everything(seed)
    device = resolve_device(feature_config.get("device", "auto"))

    is_text = "comment_text" in frame.columns

    if is_text:
        text_model_name = feature_config.get("backbone", "all-MiniLM-L6-v2")
        model, feature_dim = build_text_backbone(text_model_name)
        dataset = TextManifestDataset(frame)
        loader = DataLoader(
            dataset,
            batch_size=int(feature_config.get("batch_size", 64)),
            shuffle=False,
            num_workers=0,
        )
    else:
        model, transform, feature_dim = build_backbone(
            pretrained=bool(feature_config.get("pretrained", True))
        )
        model.to(device)
        model.eval()
        dataset = ImageManifestDataset(frame, data_root, transform)
        loader = DataLoader(
            dataset,
            batch_size=int(feature_config.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(feature_config.get("num_workers", 4)),
            pin_memory=device.type == "cuda",
        )

    features = np.empty((len(dataset), feature_dim), dtype=np.float32)
    sample_ids = np.empty(len(dataset), dtype=np.int64)
    cursor = 0

    for batch in tqdm(loader, desc=f"Extracting {output_path.stem}"):
        if is_text:
            texts, ids = batch
            if getattr(model, "_is_tfidf", False):
                # TF-IDF: pre-fit on full frame for consistent dimensions across splits
                if not hasattr(model, "_fitted"):
                    all_texts = [str(frame.iloc[i]["comment_text"]) for i in range(len(frame))]
                    model.fit(all_texts)
                    model._fitted = True
                    feature_dim = len(model.get_feature_names_out())
                    features = np.empty((len(dataset), feature_dim), dtype=np.float32)
                batch_features = model.transform(list(texts)).toarray().astype(np.float32)
            else:
                batch_features = model.encode(
                    texts, batch_size=len(texts), show_progress_bar=False,
                    convert_to_tensor=True, device=device,
                ).cpu().numpy().astype(np.float32)
        else:
            images, ids = batch
            images = images.to(device, non_blocking=True)
            batch_features = model(images).detach().cpu().numpy().astype(np.float32)

        batch_size = len(ids)
        features[cursor : cursor + batch_size] = batch_features
        sample_ids[cursor : cursor + batch_size] = ids.numpy()
        cursor += batch_size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, features=features, sample_ids=sample_ids)
    return output_path


def load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path)
    return (
        archive["features"].astype(np.float32),
        archive["sample_ids"].astype(np.int64),
    )
