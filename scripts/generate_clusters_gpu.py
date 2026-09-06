"""GPU-accelerated k-means for cluster assignments (replaces slow sklearn version)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def gpu_kmeans(
    features: torch.Tensor,
    k: int,
    *,
    n_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Simple GPU k-means. Returns cluster labels [n]."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device, dtype=torch.float32)
    n = features.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)

    # k-means++ init via random subset
    init_idx = torch.randint(0, n, (k,), generator=gen, device=device)
    centroids = features[init_idx].clone()

    for _ in range(n_iter):
        # Assign: pairwise distances [n, k]
        dists = torch.cdist(features, centroids)
        labels = dists.argmin(dim=1)

        # Update
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = features[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]

        shift = (new_centroids - centroids).norm().item()
        centroids = new_centroids
        if shift < tol:
            break

    final_labels = torch.cdist(features, centroids).argmin(dim=1)
    return final_labels.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU k-means cluster assignments")
    parser.add_argument("--features", required=True, help="Path to train.npz")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=100)
    args = parser.parse_args()

    archive = np.load(args.features)
    features = torch.from_numpy(archive["features"].astype(np.float32))
    sample_ids = archive["sample_ids"].astype(np.int64)
    print(f"Loaded {len(features)} features, dim={features.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    features = features.to(device)

    clusters = gpu_kmeans(features, args.k, n_iter=args.n_iter, seed=args.seed, device=device)
    print(f"Cluster sizes: {np.bincount(clusters)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sample_id": sample_ids, "cluster": clusters})
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cluster assignments to {output_path}")


if __name__ == "__main__":
    main()
