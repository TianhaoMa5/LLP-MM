"""Prepare FLamby Fed-ISIC2019 as DINOv2-S/14 feature-dependent LLP bags."""

from __future__ import annotations

import argparse
import json

from plench.data.fed_isic2019_preparation import (
    DEFAULT_ENCODER,
    DEFAULT_MAX_BAG_SIZE,
    DEFAULT_MIN_BAG_SIZE,
    DEFAULT_SEED,
    DEFAULT_TARGET_BAG_SIZE,
    prepare_fed_isic2019,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract/load frozen DINOv2 ViT-S/14 embeddings and construct "
            "split-isolated, variable-size Fed-ISIC2019 feature bags."
        )
    )
    parser.add_argument("--data_root", "--data-root", required=True)
    parser.add_argument(
        "--metadata_csv", "--metadata-csv", default=None,
        help="FLamby dataset_creation_scripts/train_test_split; auto-discovered when omitted",
    )
    parser.add_argument(
        "--image_dir", "--image-dir", default=None,
        help="FLamby preprocessed image directory; auto-discovered when omitted",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--extraction_batch_size", "--extraction-batch-size", type=int, default=64
    )
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument(
        "--min_bag_size", "--min-bag-size", type=int, default=DEFAULT_MIN_BAG_SIZE
    )
    parser.add_argument(
        "--max_bag_size", "--max-bag-size", type=int, default=DEFAULT_MAX_BAG_SIZE
    )
    parser.add_argument(
        "--target_bag_size", "--target-bag-size", type=int,
        default=DEFAULT_TARGET_BAG_SIZE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dinov2_repo", "--dinov2-repo", default=None,
        help="optional local facebookresearch/dinov2 checkout for offline loading",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recompute compatible feature and bag caches",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = prepare_fed_isic2019(
        args.data_root,
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        encoder=args.encoder,
        device=args.device,
        extraction_batch_size=args.extraction_batch_size,
        num_workers=args.num_workers,
        min_bag_size=args.min_bag_size,
        max_bag_size=args.max_bag_size,
        target_bag_size=args.target_bag_size,
        seed=args.seed,
        force=args.force,
        dinov2_repo=args.dinov2_repo,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
