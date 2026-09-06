#!/usr/bin/env python
"""Generate k-means cluster assignments for adaptive coverage methods (v7/v8).

Usage:
    python scripts/generate_clusters.py \
        --features outputs/celeba_5seeds/features/train.npz \
        --k 20 \
        --output outputs/celeba_5seeds/cluster_assignments/clusters_k20.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cluster assignments")
    parser.add_argument("--features", required=True, help="Path to train.npz")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    archive = np.load(args.features)
    features = archive["features"].astype(np.float32)
    sample_ids = archive["sample_ids"].astype(np.int64)
    print(f"Loaded {len(features)} features, dim={features.shape[1]}")

    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        random_state=args.seed,
        batch_size=4096,
        n_init=10,
    )
    clusters = kmeans.fit_predict(features)
    print(f"Cluster sizes: {np.bincount(clusters)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sample_id": sample_ids, "cluster": clusters})
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cluster assignments to {output_path}")


if __name__ == "__main__":
    main()
