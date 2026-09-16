#!/usr/bin/env python3
"""Download and prepare Amazon-WILDS v2.1 reviewer-level natural bags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = ["1star", "2star", "3star", "4star", "5star"]
NUM_CLASSES = len(CLASS_NAMES)
OFFICIAL_SPLITS = ("train", "val", "id_val", "test", "id_test")


WILDS_VERSION = "2.1"
DOWNLOAD_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0xe3ed909786d34ee79d430d065582aa29/contents/blob/"
)
COMPRESSED_SIZE = 1_989_805_589
PREPROCESSING_VERSION = "amazon-wilds-v2.1-reviewer-natural-bags-v1"
SPLIT_ID_TO_NAME = {
    0: "train",
    1: "val",
    2: "id_val",
    3: "test",
    4: "id_test",
}
UNLABELED_SPLIT_ID_TO_NAME = {
    11: "val_unlabeled",
    12: "test_unlabeled",
    13: "extra_unlabeled",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"refusing link in official archive: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def _download_official(raw_root: Path, download: bool, force: bool) -> Path:
    data_dir = raw_root / f"amazon_v{WILDS_VERSION}"
    reviews = data_dir / "reviews.csv"
    split = data_dir / "splits" / "user.csv"
    release = data_dir / f"RELEASE_v{WILDS_VERSION}.txt"
    if reviews.is_file() and split.is_file() and release.is_file() and not force:
        print(f"Using existing official Amazon-WILDS data: {data_dir}")
        return data_dir
    if not download:
        raise FileNotFoundError(
            f"Amazon-WILDS v{WILDS_VERSION} not found at {data_dir}; rerun with --download"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "archive.tar.gz"
    if force and archive.exists():
        archive.unlink()
    if not archive.is_file():
        print(
            f"Downloading official Amazon-WILDS v{WILDS_VERSION} "
            f"({COMPRESSED_SIZE / 1e9:.2f} GB) to {archive}"
        )
        temporary = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=3600) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        if temporary.stat().st_size < int(COMPRESSED_SIZE * 0.95):
            raise RuntimeError(
                f"official archive is unexpectedly small: {temporary.stat().st_size} bytes"
            )
        temporary.replace(archive)
    print(f"Extracting {archive} into {data_dir}")
    _safe_extract(archive, data_dir)
    if not reviews.is_file() or not split.is_file():
        raise FileNotFoundError("official archive lacks reviews.csv or splits/user.csv")
    archive.unlink()
    if not release.is_file():
        release.write_text(
            f"Amazon-WILDS release v{WILDS_VERSION}; downloaded from the official WILDS bundle.\n",
            encoding="utf-8",
        )
    return data_dir


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": int(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def _fixed_list_array(pa: Any, matrix: np.ndarray, value_type: Any) -> Any:
    values = pa.array(matrix.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(values, matrix.shape[1])


def _write_parquet_atomic(table: Any, path: Path, pq: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _read_official_tables(data_dir: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Amazon-WILDS preprocessing requires pandas; install "
            "plench/requirements-amazon-wilds.txt"
        ) from exc
    reviews_path = data_dir / "reviews.csv"
    splits_path = data_dir / "splits" / "user.csv"
    print(f"Reading official reviews: {reviews_path}")
    reviews = pd.read_csv(
        reviews_path,
        dtype={
            "reviewerID": str,
            "asin": str,
            "reviewTime": str,
            "unixReviewTime": "int64",
            "reviewText": str,
            "summary": str,
            "verified": bool,
            "category": str,
            "reviewYear": "int64",
        },
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONNUMERIC,
    )
    splits = pd.read_csv(splits_path)
    if len(reviews) != len(splits):
        raise ValueError(
            f"official review/split row mismatch: {len(reviews)} vs {len(splits)}"
        )
    if "split" not in splits:
        raise ValueError("official WILDS user split lacks the split column")
    return reviews, splits, reviews_path, splits_path


def build_processed_cache(
    reviews: Any,
    splits: Any,
    data_root: Path,
    *,
    source_files: dict[str, str],
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build processed caches from row-aligned official review/split tables."""
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Amazon-WILDS preprocessing requires pandas and pyarrow; install "
            "plench/requirements-amazon-wilds.txt"
        ) from exc
    if len(reviews) != len(splits):
        raise ValueError("reviews and official split table must be row-aligned")
    required = {
        "reviewerID",
        "asin",
        "category",
        "reviewYear",
        "overall",
        "reviewText",
    }
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"official reviews table is missing columns: {missing}")

    all_split_values = pd.to_numeric(splits["split"], errors="raise").astype(np.int64)
    known_split_ids = {-1, *SPLIT_ID_TO_NAME, *UNLABELED_SPLIT_ID_TO_NAME}
    unknown_splits = sorted(set(all_split_values.tolist()) - known_split_ids)
    if unknown_splits:
        raise ValueError(f"unknown official WILDS split IDs: {unknown_splits}")
    # v2.1 co-locates the labeled benchmark and the official unlabeled
    # adaptation splits in one bundle.  This five-class task uses only labeled
    # splits 0..4; IDs 11..13 are counted below but never enter LLP bags or
    # instance-label evaluation.
    included = all_split_values.isin(SPLIT_ID_TO_NAME)
    frame = reviews.loc[included].copy().reset_index(names="original_index")
    split_values = all_split_values.loc[included].reset_index(drop=True)
    frame["split_id"] = split_values.to_numpy()
    frame["split"] = frame["split_id"].map(SPLIT_ID_TO_NAME)
    ratings = pd.to_numeric(frame["overall"], errors="raise").astype(np.int64)
    if not ratings.between(1, 5).all():
        raise ValueError("Amazon-WILDS ratings must be integers from 1 to 5")
    frame["label"] = ratings - 1
    if (frame["reviewerID"].astype(str).str.len() == 0).any():
        raise ValueError("Amazon-WILDS contains an empty reviewerID")
    # Preserve every official review, including a possible empty-text record.
    # Dropping it would alter its reviewer's natural bag and exact proportion.

    frame["reviewer_index"] = pd.factorize(frame["reviewerID"], sort=True)[0]
    frame["product_index"] = pd.factorize(frame["asin"], sort=True)[0]
    frame["category_index"] = pd.factorize(frame["category"], sort=True)[0]
    frame["bag_id"] = frame["reviewerID"].astype(str)
    frame["instance_id"] = [
        f"amazon-v{WILDS_VERSION}:{int(index)}" for index in frame["original_index"]
    ]
    frame["feature_index"] = np.arange(len(frame), dtype=np.int64)
    frame["natural_bag_size_in_split"] = (
        frame.groupby(["split", "reviewerID"])["reviewerID"].transform("size").astype(np.int64)
    )

    train = frame[frame["split"] == "train"].copy()
    if train.empty:
        raise ValueError("official Amazon-WILDS split contains no training reviews")
    grouped = train.groupby("reviewerID", sort=True, observed=True)
    bag_rows: list[dict[str, Any]] = []
    for reviewer_id, group in grouped:
        unique_reviewers = group["reviewerID"].nunique()
        if unique_reviewers != 1:
            raise AssertionError(f"bag {reviewer_id} contains {unique_reviewers} reviewers")
        counts = np.bincount(group["label"].to_numpy(np.int64), minlength=NUM_CLASSES)
        proportions = counts.astype(np.float64) / len(group)
        if len(proportions) != NUM_CLASSES or not np.isclose(proportions.sum(), 1.0, atol=1e-12):
            raise AssertionError(f"invalid proportion for reviewer {reviewer_id}")
        bag_rows.append(
            {
                "bag_id": str(reviewer_id),
                "reviewer_id": str(reviewer_id),
                "reviewer_index": int(group["reviewer_index"].iloc[0]),
                "n_instances": int(len(group)),
                "class_counts": counts.astype(np.int64),
                "class_proportions": proportions.astype(np.float32),
            }
        )
    if len(bag_rows) != train["reviewerID"].nunique():
        raise AssertionError("one training reviewer does not map to exactly one bag")
    if len({row["bag_id"] for row in bag_rows}) != len(bag_rows):
        raise AssertionError("Amazon-WILDS train bag IDs are not unique")

    # Recompute a deterministic sample as an explicit proportion audit.
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(bag_rows), size=min(20, len(bag_rows)), replace=False)
    train_by_reviewer = {str(key): value for key, value in grouped}
    for row_number in sampled:
        row = bag_rows[int(row_number)]
        labels = train_by_reviewer[row["reviewer_id"]]["label"].to_numpy(np.int64)
        recomputed = np.bincount(labels, minlength=NUM_CLASSES)
        if not np.array_equal(recomputed, row["class_counts"]):
            raise AssertionError(f"sampled bag count mismatch: {row['bag_id']}")
        if not np.allclose(recomputed / len(labels), row["class_proportions"], atol=1e-7):
            raise AssertionError(f"sampled bag proportion mismatch: {row['bag_id']}")

    bag_sizes = np.asarray([row["n_instances"] for row in bag_rows], dtype=np.int64)
    bag_proportions = np.stack([row["class_proportions"] for row in bag_rows]).astype(np.float64)
    entropies = -(bag_proportions * np.log(np.clip(bag_proportions, 1e-12, None))).sum(axis=1)
    train_labels = train["label"].to_numpy(np.int64)
    global_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    split_stats: dict[str, Any] = {}
    reviewer_sets: dict[str, set[str]] = {}
    for split_name in OFFICIAL_SPLITS:
        selected = frame[frame["split"] == split_name]
        reviewers = set(selected["reviewerID"].astype(str).tolist())
        reviewer_sets[split_name] = reviewers
        split_stats[split_name] = {
            "instances": int(len(selected)),
            "reviewers": int(len(reviewers)),
        }
    overlaps = {
        f"{left}__{right}": len(reviewer_sets[left] & reviewer_sets[right])
        for index, left in enumerate(OFFICIAL_SPLITS)
        for right in OFFICIAL_SPLITS[index + 1 :]
    }
    class_stats = {
        CLASS_NAMES[class_index]: {
            "global_instance_proportion": float(global_counts[class_index] / global_counts.sum()),
            "mean_bag_proportion": float(bag_proportions[:, class_index].mean()),
            "median_bag_proportion": float(np.median(bag_proportions[:, class_index])),
            "min_bag_proportion": float(bag_proportions[:, class_index].min()),
            "max_bag_proportion": float(bag_proportions[:, class_index].max()),
        }
        for class_index in range(NUM_CLASSES)
    }
    stats: dict[str, Any] = {
        "dataset": "Amazon-WILDS",
        "dataset_version": WILDS_VERSION,
        "task": "5-class review-text classification",
        "class_names": CLASS_NAMES,
        "bag_construction": "one reviewerID = one complete natural training bag",
        "splits": split_stats,
        "train_bags": len(bag_rows),
        "train_bag_size": _percentiles(bag_sizes),
        "train_class_proportions": class_stats,
        "train_bag_entropy": {
            "mean": float(entropies.mean()),
            "median": float(np.median(entropies)),
            "p10": float(np.percentile(entropies, 10)),
            "p90": float(np.percentile(entropies, 90)),
        },
        "near_pure_bags": {
            "max_proportion_gt_0.90": int((bag_proportions.max(axis=1) > 0.90).sum()),
            "fraction_max_proportion_gt_0.90": float((bag_proportions.max(axis=1) > 0.90).mean()),
            "max_proportion_gt_0.95": int((bag_proportions.max(axis=1) > 0.95).sum()),
            "fraction_max_proportion_gt_0.95": float((bag_proportions.max(axis=1) > 0.95).mean()),
        },
        "reviewer_size_thresholds": {
            f"at_least_{threshold}": int((bag_sizes >= threshold).sum())
            for threshold in (50, 75, 100, 200, 500)
        },
        "reviewer_overlap": overlaps,
        "included_instances": int(len(frame)),
        "excluded_split_minus_one_instances": int((all_split_values == -1).sum()),
        "official_unlabeled_splits": {
            name: int((all_split_values == split_id).sum())
            for split_id, name in UNLABELED_SPLIT_ID_TO_NAME.items()
        },
    }

    processed = data_root / "processed"
    instances_table = pa.table(
        {
            "instance_id": pa.array(frame["instance_id"].tolist(), type=pa.string()),
            "original_index": pa.array(frame["original_index"].to_numpy(np.int64)),
            "feature_index": pa.array(frame["feature_index"].to_numpy(np.int64)),
            "split": pa.array(frame["split"].tolist(), type=pa.string()),
            "split_id": pa.array(frame["split_id"].to_numpy(np.int8)),
            "reviewer_id": pa.array(frame["reviewerID"].astype(str).tolist(), type=pa.string()),
            "reviewer_index": pa.array(frame["reviewer_index"].to_numpy(np.int64)),
            "bag_id": pa.array(frame["bag_id"].tolist(), type=pa.string()),
            "natural_bag_size_in_split": pa.array(
                frame["natural_bag_size_in_split"].to_numpy(np.int64)
            ),
            "product_id": pa.array(frame["asin"].astype(str).tolist(), type=pa.string()),
            "product_index": pa.array(frame["product_index"].to_numpy(np.int64)),
            "category": pa.array(frame["category"].astype(str).tolist(), type=pa.string()),
            "category_index": pa.array(frame["category_index"].to_numpy(np.int64)),
            "year": pa.array(frame["reviewYear"].to_numpy(np.int64)),
            "unix_review_time": pa.array(frame["unixReviewTime"].to_numpy(np.int64)),
            "text": pa.array(frame["reviewText"].astype(str).tolist(), type=pa.large_string()),
            "label": pa.array(frame["label"].to_numpy(np.int8)),
        }
    )
    counts_matrix = np.stack([row["class_counts"] for row in bag_rows])
    proportions_matrix = np.stack([row["class_proportions"] for row in bag_rows])
    bags_table = pa.table(
        {
            "bag_id": pa.array([row["bag_id"] for row in bag_rows], type=pa.string()),
            "reviewer_id": pa.array(
                [row["reviewer_id"] for row in bag_rows], type=pa.string()
            ),
            "reviewer_index": pa.array(
                [row["reviewer_index"] for row in bag_rows], type=pa.int64()
            ),
            "n_instances": pa.array(
                [row["n_instances"] for row in bag_rows], type=pa.int64()
            ),
            "class_counts": _fixed_list_array(pa, counts_matrix, pa.int64()),
            "class_proportions": _fixed_list_array(
                pa, proportions_matrix, pa.float32()
            ),
        }
    )
    _write_parquet_atomic(instances_table, processed / "instances.parquet", pq)
    _write_parquet_atomic(bags_table, processed / "train_bags.parquet", pq)
    _atomic_json(processed / "amazon_wilds_stats.json", stats)
    _atomic_json(data_root / "splits" / "official_v2.1.json", {
        "dataset_version": WILDS_VERSION,
        "split_scheme": "official/user",
        "split_id_to_name": {str(key): value for key, value in SPLIT_ID_TO_NAME.items()},
        "counts": split_stats,
        "reviewer_overlap": overlaps,
    })
    metadata = {
        "dataset": "amazon_wilds",
        "dataset_version": WILDS_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "label_mapping": {str(stars): stars - 1 for stars in range(1, 6)},
        "reviewer_field": "reviewerID",
        "wilds_metadata_field": "user",
        "bag_id_definition": "original reviewerID",
        "natural_bags": True,
        "variable_bag_size": True,
        "target_bag_size": None,
        "training_labels_visible_to_llp": False,
        "official_split_scheme": "user",
        "official_splits_preserved": list(OFFICIAL_SPLITS),
        "official_unlabeled_splits_excluded": {
            str(split_id): name
            for split_id, name in UNLABELED_SPLIT_ID_TO_NAME.items()
        },
        "source_files": source_files,
        "source_sha256": source_hashes or {},
        "instances": int(len(frame)),
        "train_bags": len(bag_rows),
        "stats_file": "amazon_wilds_stats.json",
    }
    _atomic_json(processed / "metadata.json", metadata)

    print("Amazon-WILDS")
    print("Task: 5-class text classification")
    print("Bag construction: reviewerID = one complete natural training bag")
    for split_name in OFFICIAL_SPLITS:
        row = split_stats[split_name]
        suffix = f" bags={row['reviewers']}" if split_name == "train" else ""
        print(
            f"{split_name}: instances={row['instances']} reviewers={row['reviewers']}" + suffix
        )
    print("Train bag-size distribution:", json.dumps(stats["train_bag_size"], sort_keys=True))
    print("Train class-proportion statistics:", json.dumps(class_stats, sort_keys=True))
    print("Train bag entropy:", json.dumps(stats["train_bag_entropy"], sort_keys=True))
    print("Near-pure bags:", json.dumps(stats["near_pure_bags"], sort_keys=True))
    print("Reviewer thresholds:", json.dumps(stats["reviewer_size_thresholds"], sort_keys=True))
    print("Reviewer overlap:", json.dumps(overlaps, sort_keys=True))
    print("First 5 training bags:")
    for row in bag_rows[:5]:
        print(
            f"  {row['bag_id']}: N={row['n_instances']} "
            f"proportion={row['class_proportions'].tolist()}"
        )
    return stats


def prepare(data_root: Path, *, download: bool, force_download: bool) -> dict[str, Any]:
    raw_root = data_root / "raw"
    data_dir = _download_official(raw_root, download=download, force=force_download)
    reviews, splits, reviews_path, splits_path = _read_official_tables(data_dir)
    return build_processed_cache(
        reviews,
        splits,
        data_root,
        source_files={
            "reviews": str(reviews_path.relative_to(data_root)),
            "user_split": str(splits_path.relative_to(data_root)),
        },
        source_hashes={
            "reviews": _sha256(reviews_path),
            "user_split": _sha256(splits_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("plench/data/amazon_wilds")
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the official WILDS v2.1 archive if it is absent",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    prepare(
        args.data_root.expanduser().resolve(),
        download=bool(args.download or args.force_download),
        force_download=bool(args.force_download),
    )


if __name__ == "__main__":
    main()
