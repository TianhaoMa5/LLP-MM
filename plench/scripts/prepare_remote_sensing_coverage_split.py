#!/usr/bin/env python3
"""Create and optionally activate a fixed all-class-coverage CV/LEM split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np

from plench.data.remote_sensing import (
    PREPROCESSING_VERSION,
    SPLIT_NAMES,
    SPLIT_TO_CODE,
    _atomic_json,
    _atomic_npy,
    _load_json,
    _resolve_processed_dir,
    canonical_remote_dataset,
    compute_train_normalization,
    load_remote_sensing_bundle,
    normalization_fingerprint,
    print_split_statistics,
)
from plench.data.remote_sensing_splits import (
    DEFAULT_COVERAGE_SPLIT_RATIOS,
    build_all_class_coverage_split,
)


PROFILE_FILES = (
    "split_codes.npy",
    "split_manifest.json",
    "normalization.json",
    "preprocessing_manifest.json",
)


def _profile_hash(source_hash: str, split_manifest: Dict[str, Any]) -> str:
    payload = {
        "source_preprocessing_hash": source_hash,
        "split_profile": split_manifest["profile"],
        "split_manifest": split_manifest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _save_active_profile(processed: Path) -> str:
    manifest = _load_json(processed / "preprocessing_manifest.json", required=True)
    split_manifest = _load_json(processed / "split_manifest.json", required=True)
    profile = str(manifest.get("split_profile") or split_manifest.get("profile") or
                  f"paper_field_random_seed{int(split_manifest.get('seed', 0))}")
    destination = processed / "splits" / profile
    destination.mkdir(parents=True, exist_ok=True)
    for name in PROFILE_FILES:
        source = processed / name
        target = destination / name
        if not target.exists():
            shutil.copy2(source, target)
    return profile


def _activate_profile(processed: Path, profile_dir: Path) -> None:
    for name in PROFILE_FILES:
        source = profile_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        temporary = processed / f".{name}.activate.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(processed / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("CV", "LEM"))
    parser.add_argument("--data-dir", required=True, help="Dataset root containing processed/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        default=DEFAULT_COVERAGE_SPLIT_RATIOS,
        help="Final train/validation/test field-owner fractions (default: 0.70 0.10 0.20).",
    )
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    dataset = canonical_remote_dataset(args.dataset)
    processed = _resolve_processed_dir(args.data_dir, dataset)
    preprocessing_manifest = _load_json(processed / "preprocessing_manifest.json", required=True)
    if preprocessing_manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Coverage profiles require the current centre-catalogue preprocessing")
    class_mapping = _load_json(processed / "class_mapping.json", required=True)
    field_values = [str(value) for value in
                    _load_json(processed / "field_values.json", required=True)]
    image = np.load(processed / "image_stack.npy", mmap_mode="r", allow_pickle=False)
    rows = np.load(processed / "center_rows.npy", mmap_mode="r", allow_pickle=False)
    cols = np.load(processed / "center_cols.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(processed / "labels.npy", mmap_mode="r", allow_pickle=False)
    field_indices = np.load(processed / "field_indices.npy", mmap_mode="r", allow_pickle=False)
    num_classes = int(class_mapping["num_classes"])
    idx_to_class = {int(key): str(value) for key, value in class_mapping["idx_to_class"].items()}
    patch_size = int(preprocessing_manifest.get("patch_size", 21))
    train_fraction, val_fraction, test_fraction = (float(value) for value in args.ratios)
    if any(value <= 0 for value in (train_fraction, val_fraction, test_fraction)):
        parser.error("--ratios values must all be positive")
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0, atol=1e-9):
        parser.error("--ratios TRAIN VAL TEST must sum to 1")
    val_fraction_of_train = val_fraction / (train_fraction + val_fraction)
    ratio_suffix = "_".join(str(int(round(value * 100))) for value in args.ratios)

    profile = args.profile or f"all_class_coverage_{ratio_suffix}_seed{int(args.seed)}"
    split_codes, split_manifest = build_all_class_coverage_split(
        field_indices=field_indices,
        labels=labels,
        rows=rows,
        cols=cols,
        num_classes=num_classes,
        seed=args.seed,
        test_fraction=test_fraction,
        patch_size=patch_size,
        val_fraction_of_train=val_fraction_of_train,
        class_names=idx_to_class,
    )
    split_manifest.update({"dataset": dataset, "profile": profile})
    # Retain both catalogue indices (machine validation) and logical IDs
    # (human audit). The latter may be shapefile strings rather than integers.
    split_manifest["field_indices"] = split_manifest.pop("field_ids")
    split_manifest["field_ids"] = {
        name: [field_values[index] for index in indices]
        for name, indices in split_manifest["field_indices"].items()
    }
    split_manifest["field_owner_indices"] = split_manifest.pop("field_owner_ids")
    split_manifest["field_owner_ids"] = {
        name: [field_values[index] for index in indices]
        for name, indices in split_manifest["field_owner_indices"].items()
    }
    for record in split_manifest["spatial_exceptions"]:
        record["field_id"] = field_values[int(record["field_index"])]
        record["class_name"] = idx_to_class[int(record["class_index"])]

    source_hash = str(preprocessing_manifest.get(
        "source_preprocessing_hash", preprocessing_manifest["preprocessing_hash"]
    ))
    profile_manifest = dict(preprocessing_manifest)
    profile_manifest.update({
        "source_preprocessing_hash": source_hash,
        "split_profile": profile,
        "split_seed": int(args.seed),
        "split_strategy": split_manifest["strategy"],
    })
    profile_manifest["preprocessing_hash"] = _profile_hash(source_hash, split_manifest)
    train_indices = np.flatnonzero(split_codes == SPLIT_TO_CODE["train"]).astype(np.int64)
    mean, std = compute_train_normalization(image, rows, cols, train_indices, patch_size)
    normal = {
        "dataset": dataset,
        "split_profile": profile,
        "source": "all training-centred patches only",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fingerprint": normalization_fingerprint(
            profile_manifest, split_manifest, class_mapping
        ),
    }

    profile_dir = processed / "splits" / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npy(profile_dir / "split_codes.npy", split_codes)
    _atomic_json(profile_dir / "split_manifest.json", split_manifest)
    _atomic_json(profile_dir / "normalization.json", normal)
    _atomic_json(profile_dir / "preprocessing_manifest.json", profile_manifest)
    previous_profile = None
    if args.activate:
        previous_profile = _save_active_profile(processed)
        _activate_profile(processed, profile_dir)
        bundle = load_remote_sensing_bundle(dataset, args.data_dir)
        print_split_statistics(bundle)

    summary = {
        "dataset": dataset,
        "profile": profile,
        "profile_directory": str(profile_dir),
        "activated": bool(args.activate),
        "previous_profile_backed_up_as": previous_profile,
        "fully_field_disjoint": False,
        "spatial_exception_fields": [
            {
                "field_id": record["field_id"],
                "class_name": record["class_name"],
                "mode": record["mode"],
                "axis": record["axis"],
                "patch_support_disjoint": record["patch_support_disjoint"],
                "minimum_cross_split_center_separation": record[
                    "minimum_cross_split_center_separation"
                ],
                "valid_centres_by_split": record["valid_centres_by_split"],
                "dropped_guard_centres": record["dropped_guard_centres"],
            }
            for record in split_manifest["spatial_exceptions"]
        ],
        "field_owner_counts": split_manifest["field_owner_counts"],
        "actual_field_counts": split_manifest["field_counts"],
        "valid_center_counts": split_manifest["valid_center_counts"],
        "instances_per_class": split_manifest["instances_per_class"],
        "normalization": str(profile_dir / "normalization.json"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
