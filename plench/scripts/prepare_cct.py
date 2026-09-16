"""Prepare official CCT-20 images as variable-size feature LLP bags."""

from __future__ import annotations

import argparse
import json

from plench.data.cct_preparation import (
    DEFAULT_BBOX_SOURCE,
    DEFAULT_MAX_BAG_SIZE_RATIO,
    DEFAULT_MIN_BBOX_AREA,
    DEFAULT_MIN_BAG_SIZE_RATIO,
    DEFAULT_PCA_DIM,
    DEFAULT_TARGET_AVG_BAG_SIZE,
    prepare_cct,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen DINOv2 ViT-B/14 features, fit train-only PCA, and "
            "construct split-isolated variable-size CCT feature bags."
        )
    )
    parser.add_argument("--data_root", "--data-root", required=True)
    parser.add_argument(
        "--annotation_dir",
        "--annotation-dir",
        default=None,
        help="directory containing the five official CCT-20 annotation JSON files",
    )
    parser.add_argument(
        "--bbox_source", "--bbox-source",
        choices=["annotation"], default=DEFAULT_BBOX_SOURCE,
        help="category-agnostic crop geometry source; annotation is implemented now",
    )
    parser.add_argument(
        "--min_bbox_area", "--min-bbox-area",
        type=float, default=DEFAULT_MIN_BBOX_AREA,
        help="minimum annotation-space bbox width*height (reference default: 4096)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--extraction_batch_size", "--extraction-batch-size",
        type=int, default=64,
    )
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--pca_dim", "--pca-dim", type=int, default=DEFAULT_PCA_DIM)
    parser.add_argument(
        "--target_avg_bag_size", "--target-avg-bag-size",
        type=int, default=DEFAULT_TARGET_AVG_BAG_SIZE,
    )
    parser.add_argument(
        "--min_bag_size_ratio", "--min-bag-size-ratio",
        type=float, default=DEFAULT_MIN_BAG_SIZE_RATIO,
    )
    parser.add_argument(
        "--max_bag_size_ratio", "--max-bag-size-ratio",
        type=float, default=DEFAULT_MAX_BAG_SIZE_RATIO,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dinov2_repo", "--dinov2-repo", default=None,
        help="optional local facebookresearch/dinov2 checkout for offline model loading",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recompute feature, PCA, and feature-bag caches",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = prepare_cct(
        args.data_root,
        annotation_dir=args.annotation_dir,
        bbox_source=args.bbox_source,
        min_bbox_area=args.min_bbox_area,
        device=args.device,
        extraction_batch_size=args.extraction_batch_size,
        num_workers=args.num_workers,
        pca_dim=args.pca_dim,
        target_avg_bag_size=args.target_avg_bag_size,
        min_bag_size_ratio=args.min_bag_size_ratio,
        max_bag_size_ratio=args.max_bag_size_ratio,
        seed=args.seed,
        force=args.force,
        dinov2_repo=args.dinov2_repo,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
