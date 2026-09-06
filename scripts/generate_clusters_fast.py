"""Fast k-means: subsample-fit + chunked GPU assign. Replaces slow sklearn version."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=20000)
    args = parser.parse_args()

    archive = np.load(args.features)
    features = archive["features"].astype(np.float32)
    sample_ids = archive["sample_ids"].astype(np.int64)
    n = len(features)
    print(f"Loaded {n} features, dim={features.shape[1]}")

    # Step 1: fit k-means on subsample (CPU, fast with 20k points)
    rng = np.random.default_rng(args.seed)
    if n > args.subsample:
        idx = rng.choice(n, args.subsample, replace=False)
        sub = features[idx]
    else:
        sub = features
    print(f"Fitting k-means on {len(sub)} subsample (k={args.k})...")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
    km.fit(sub)
    centroids = km.cluster_centers_.astype(np.float32)
    print(f"Fit done. Cluster balance on subsample: {np.bincount(km.labels_, minlength=args.k)}")

    # Step 2: assign all points to nearest centroid (GPU, chunked)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Assigning {n} points on {device}...")
    cent_t = torch.from_numpy(centroids).to(device)  # [k, d]
    labels = np.empty(n, dtype=np.int32)
    chunk = 8192
    for i in range(0, n, chunk):
        batch = torch.from_numpy(features[i:i+chunk]).to(device)
        dists = torch.cdist(batch, cent_t)  # [chunk, k]
        labels[i:i+chunk] = dists.argmin(dim=1).cpu().numpy()
    print(f"Final cluster sizes: {np.bincount(labels, minlength=args.k)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sample_id": sample_ids, "cluster": labels})
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cluster assignments to {output_path}")


if __name__ == "__main__":
    main()
