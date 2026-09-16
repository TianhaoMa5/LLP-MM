#!/usr/bin/env python3
"""Prepare train-fitted ClusterBag assignments for Campo Verde or LEM."""

from __future__ import annotations

import argparse
import json

from plench.data.remote_sensing import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_INSTANCES_PER_EPOCH,
    load_remote_sensing_bundle,
    prepare_remote_sensing_cluster_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("CV", "LEM"))
    parser.add_argument("--data-dir", required=True, help="Dataset root containing processed/")
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-samples", type=int, default=DEFAULT_INSTANCES_PER_EPOCH)
    parser.add_argument("--predict-batch-size", type=int, default=65_536)
    args = parser.parse_args()

    bundle = load_remote_sensing_bundle(args.dataset, args.data_dir, seed=args.seed)
    manifest = prepare_remote_sensing_cluster_cache(
        bundle,
        cluster_count=args.clusters,
        seed=args.seed,
        fit_samples=args.fit_samples,
        predict_batch_size=args.predict_batch_size,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
