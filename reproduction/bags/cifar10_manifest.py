#!/usr/bin/env python3
"""Build shared CIFAR-10/100 bag manifests for LLP baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _rounded_counts(weights: np.ndarray, bag_size: int, rng: np.random.Generator) -> np.ndarray:
    fractional = weights * bag_size
    counts = np.floor(fractional).astype(np.int64)
    for row in range(len(counts)):
        remainder = bag_size - int(counts[row].sum())
        if remainder:
            probabilities = fractional[row] - counts[row]
            if probabilities.sum() == 0:
                probabilities = weights[row]
            probabilities = probabilities / probabilities.sum()
            additions = rng.choice(weights.shape[1], remainder, replace=True, p=probabilities)
            np.add.at(counts[row], additions, 1)
    return counts


def _balance_columns(
    counts: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Preserve every bag size while matching each pool's available count."""
    counts = counts.copy()
    difference = targets.astype(np.int64) - counts.sum(axis=0)
    while np.any(difference > 0):
        destination = int(np.flatnonzero(difference > 0)[-1])
        sources = np.flatnonzero(difference < 0)
        if not len(sources):
            raise RuntimeError("Cannot balance bag count matrix")
        source = int(sources[-1])
        candidate_rows = np.flatnonzero(counts[:, source] > 0)
        row = int(rng.choice(candidate_rows))
        counts[row, source] -= 1
        counts[row, destination] += 1
        difference[source] += 1
        difference[destination] -= 1
    if not np.array_equal(counts.sum(axis=0), targets):
        raise RuntimeError("Balanced counts do not match pool sizes")
    return counts


def _bags_from_pools(
    pool_ids: np.ndarray,
    bag_size: int,
    alpha: float,
    rng: np.random.Generator,
    use_pool_prior: bool,
) -> np.ndarray:
    usable = len(pool_ids) // bag_size * bag_size
    selected = rng.permutation(len(pool_ids))[:usable]
    selected_pool_ids = pool_ids[selected]
    unique_ids = np.unique(selected_pool_ids)
    pools = []
    for pool_id in unique_ids:
        pool = selected[selected_pool_ids == pool_id].copy()
        rng.shuffle(pool)
        pools.append(pool)

    targets = np.asarray([len(pool) for pool in pools], dtype=np.int64)
    if use_pool_prior:
        parameters = np.maximum(alpha * targets / targets.sum(), 1e-12)
    else:
        # This matches the existing plench AlphaFirst implementation.
        parameters = np.full(len(pools), alpha, dtype=np.float64)
    weights = rng.dirichlet(parameters, size=usable // bag_size)
    counts = _balance_columns(_rounded_counts(weights, bag_size, rng), targets, rng)

    offsets = np.zeros(len(pools), dtype=np.int64)
    bags = np.empty((len(counts), bag_size), dtype=np.int64)
    for bag_id, row_counts in enumerate(counts):
        pieces = []
        for pool_index, count in enumerate(row_counts):
            start = offsets[pool_index]
            stop = start + int(count)
            pieces.append(pools[pool_index][start:stop])
            offsets[pool_index] = stop
        bag = np.concatenate(pieces)
        rng.shuffle(bag)
        bags[bag_id] = bag
    return bags


def build_bags(
    train_indices: np.ndarray,
    labels: np.ndarray,
    mode: str,
    bag_size: int,
    alpha: float,
    seed: int,
    clusters: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    usable = len(train_indices) // bag_size * bag_size
    if mode == "random":
        return rng.permutation(train_indices)[:usable].reshape(-1, bag_size)
    if mode == "alphafirst":
        local_bags = _bags_from_pools(
            labels[train_indices], bag_size, alpha, rng, use_pool_prior=False
        )
    elif mode == "cluster":
        if clusters is None:
            raise ValueError("cluster mode requires cluster assignments")
        local_bags = _bags_from_pools(
            clusters[train_indices], bag_size, alpha, rng, use_pool_prior=True
        )
    else:
        raise ValueError(f"Unknown bag mode: {mode}")
    return train_indices[local_bags]


def compute_proportions(
    bags: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    one_hot = np.eye(num_classes, dtype=np.float32)[labels[bags]]
    return one_hot.mean(axis=1)


def create_cluster_map(
    images: np.ndarray,
    output_path: Path,
    cluster_count: int,
    seed: int,
) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    flattened = images.reshape(len(images), -1).astype(np.float32) / 255.0
    reduced = PCA(n_components=128, random_state=seed).fit_transform(flattened)
    clusters = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        init_size=max(3 * cluster_count, 300),
        batch_size=1024,
        n_init=5,
        max_iter=100,
    ).fit_predict(reduced)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clusters=clusters.astype(np.int64),
        cluster_count=np.asarray(cluster_count),
        seed=np.asarray(seed),
        method=np.asarray("PCA128+MiniBatchKMeans"),
    )
    return clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100"),
        default="cifar10",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("random", "cluster", "alphafirst"),
        default=("random", "cluster", "alphafirst"),
    )
    parser.add_argument("--bag-sizes", nargs="+", type=int, default=(16, 32, 64, 128))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--cluster-count", type=int)
    parser.add_argument("--cluster-map", type=Path)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    from torchvision.datasets import CIFAR10, CIFAR100

    args = parse_args()
    if args.dataset == "cifar10":
        dataset_class = CIFAR10
        dataset_display = "CIFAR10"
        num_classes = 10
        default_cluster_count = 32
    else:
        dataset_class = CIFAR100
        dataset_display = "CIFAR100"
        num_classes = 100
        default_cluster_count = 256
    cluster_count = args.cluster_count or default_cluster_count
    dataset = dataset_class(args.data_dir, train=True, download=args.download)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    split_rng = np.random.default_rng(args.seed)
    split = split_rng.permutation(len(dataset))
    val_size = int(len(split) * args.val_fraction)
    val_indices = split[:val_size]
    train_indices = split[val_size:]

    clusters = None
    cluster_map_hash = None
    if "cluster" in args.modes:
        cluster_map = (
            args.cluster_map
            or args.output_dir / f"{dataset_display}_train_K{cluster_count}.npz"
        )
        if cluster_map.exists():
            clusters = np.load(cluster_map)["clusters"].astype(np.int64)
        else:
            clusters = create_cluster_map(
                dataset.data, cluster_map, cluster_count, args.seed
            )
        if len(clusters) != len(dataset):
            raise ValueError(
                f"Cluster map length does not match {dataset_display} train set"
            )
        cluster_map_hash = hashlib.sha256(cluster_map.read_bytes()).hexdigest()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for bag_size in args.bag_sizes:
            bags = build_bags(
                train_indices,
                labels,
                mode,
                bag_size,
                args.alpha,
                args.seed,
                clusters,
            )
            proportions = compute_proportions(bags, labels, num_classes)
            metadata = {
                "dataset": dataset_display,
                "mode": mode,
                "bag_size": bag_size,
                "alpha": args.alpha,
                "seed": args.seed,
                "num_classes": num_classes,
                "index_space": f"torchvision_{args.dataset}_train",
            }
            if mode == "cluster":
                metadata["cluster_map_sha256"] = cluster_map_hash
            path = (
                args.output_dir
                / f"{args.dataset}_{mode}_m{bag_size}_seed{args.seed}.npz"
            )
            np.savez_compressed(
                path,
                indices=bags,
                proportions=proportions,
                train_indices=train_indices,
                val_indices=val_indices,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            print(f"{path}: {len(bags)} bags, {len(np.unique(bags))} instances")


if __name__ == "__main__":
    main()
