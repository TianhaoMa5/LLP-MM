#!/usr/bin/env python3
"""Build a CV/LEM raster stack and all valid patch centres.

Continuous image bands are bilinearly reprojected before patch extraction.
Categorical field/class membership is rasterized directly on the target grid,
equivalent to nearest-neighbour categorical resampling.  A patch is labelled by
its centre pixel and is not required to remain inside one polygon. The default
benchmark split is deterministic 70/10/20 with every class represented in
train, validation, and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from plench.data.remote_sensing import (
    PREPROCESSING_VERSION,
    REMOTE_SENSING_SPECS,
    SPLIT_NAMES,
    SPLIT_TO_CODE,
    canonical_remote_dataset,
    compute_train_normalization,
    deterministic_nested_field_split,
    normalization_fingerprint,
)
from plench.data.remote_sensing_splits import (
    DEFAULT_COVERAGE_SPLIT_RATIOS,
    build_all_class_coverage_split,
)


def _require_geo_packages():
    try:
        import fiona
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.features import rasterize
        from rasterio.warp import reproject, transform_geom
    except ImportError as exc:
        raise SystemExit("Install rasterio, fiona and affine for preprocessing") from exc
    return fiona, rasterio, Affine, Resampling, rasterize, reproject, transform_geom


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return path.name


def _source_record(path: Path, project_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _portable_path(path, project_root),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _property_counts(features: list[dict[str, Any]], properties: list[str]) -> dict[str, Any]:
    result = {}
    for column in properties:
        values = [feature["properties"].get(column) for feature in features]
        nonempty = [str(value).strip() for value in values if value is not None and str(value).strip()]
        result[column] = {
            "nonempty_rows": len(nonempty),
            "unique_nonempty": len(set(nonempty)),
        }
    return result


def _possible_field_columns(properties: list[str]) -> list[str]:
    pattern = re.compile(r"(^id$|fid|field|polygon|parcel|numb|code)", re.IGNORECASE)
    return [column for column in properties if pattern.search(column)]


def _choose_field_column(dataset: str, requested: str, schema_stats: dict[str, Any],
                         row_count: int) -> tuple[str, str]:
    if requested != "auto":
        if requested not in schema_stats:
            raise ValueError(f"Missing field ID column {requested!r}")
        return requested, "explicit command-line selection validated against schema"
    if dataset == "CV" and "Field_numb" in schema_stats:
        field_unique = schema_stats["Field_numb"]["unique_nonempty"]
        id_unique = schema_stats.get("Id", {}).get("unique_nonempty", 0)
        if field_unique > row_count * 0.5 and id_unique < row_count * 0.1:
            return "Field_numb", (
                f"Field_numb has {field_unique} logical values while Id has only {id_unique}; "
                "duplicate Field_numb rows are multipart pieces of the same logical field"
            )
    candidates = sorted(
        _possible_field_columns(list(schema_stats)),
        key=lambda name: schema_stats[name]["unique_nonempty"], reverse=True,
    )
    if not candidates:
        raise ValueError("No plausible field identifier found; pass --field-id-column")
    return candidates[0], "highest-cardinality plausible field identifier"


def _grid_from_sources(paths: list[Path], target_resolution: float | None, rasterio, Affine):
    with rasterio.open(paths[0]) as source:
        reference = {
            "crs": source.crs,
            "transform": source.transform,
            "width": source.width,
            "height": source.height,
            "bounds": source.bounds,
            "resolution": (abs(source.transform.a), abs(source.transform.e)),
        }
    for path in paths[1:]:
        with rasterio.open(path) as source:
            actual = (source.crs, source.transform, source.width, source.height)
            expected = (reference["crs"], reference["transform"], reference["width"], reference["height"])
            if actual != expected:
                raise ValueError(f"Raster {path} is not aligned with the first source raster")
    if target_resolution is None:
        transform = reference["transform"]
        width, height = reference["width"], reference["height"]
        resolution = reference["resolution"]
    else:
        left, bottom, right, top = reference["bounds"]
        width = int(math.ceil((right - left) / target_resolution))
        height = int(math.ceil((top - bottom) / target_resolution))
        transform = Affine(target_resolution, 0.0, left, 0.0, -target_resolution, top)
        resolution = (target_resolution, target_resolution)
    return reference, {
        "crs": reference["crs"], "transform": transform,
        "width": width, "height": height, "resolution": resolution,
    }


def _write_image_stack(paths: list[Path], output: Path, target: dict[str, Any],
                       rasterio, Resampling, reproject) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    band_records = []
    total_channels = 0
    for path in paths:
        with rasterio.open(path) as source:
            descriptions = list(source.descriptions)
            for band in range(1, source.count + 1):
                name = descriptions[band - 1] or f"{path.stem}_band{band}"
                band_records.append({"source": path.name, "source_band": band, "name": name})
                total_channels += 1
    stack = np.lib.format.open_memmap(
        output / "image_stack.npy", mode="w+", dtype=np.float32,
        shape=(total_channels, target["height"], target["width"]),
    )
    output_channel = 0
    for path in paths:
        with rasterio.open(path) as source:
            same_grid = (
                source.crs == target["crs"] and source.transform == target["transform"]
                and source.width == target["width"] and source.height == target["height"]
            )
            for band in range(1, source.count + 1):
                destination = stack[output_channel]
                if same_grid:
                    destination[:] = source.read(band, out_dtype="float32")
                    if source.nodata is not None:
                        destination[destination == source.nodata] = np.nan
                else:
                    destination.fill(np.nan)
                    reproject(
                        source=rasterio.band(source, band), destination=destination,
                        src_transform=source.transform, src_crs=source.crs,
                        src_nodata=source.nodata, dst_transform=target["transform"],
                        dst_crs=target["crs"], dst_nodata=np.nan,
                        resampling=Resampling.bilinear, init_dest_nodata=True,
                    )
                output_channel += 1
    stack.flush()
    return stack, [record["name"] for record in band_records], band_records


def _full_patch_valid_mask(stack: np.ndarray, patch_size: int) -> np.ndarray:
    invalid = np.zeros(stack.shape[1:], dtype=np.uint8)
    for channel in range(stack.shape[0]):
        invalid |= ~np.isfinite(stack[channel])
    integral = np.zeros((invalid.shape[0] + 1, invalid.shape[1] + 1), dtype=np.uint32)
    np.cumsum(invalid, axis=0, dtype=np.uint32, out=integral[1:, 1:])
    integral[1:, 1:] = np.cumsum(integral[1:, 1:], axis=1, dtype=np.uint32)
    window_invalid = (
        integral[patch_size:, patch_size:] - integral[:-patch_size, patch_size:]
        - integral[patch_size:, :-patch_size] + integral[:-patch_size, :-patch_size]
    )
    radius = patch_size // 2
    valid = np.zeros(invalid.shape, dtype=bool)
    valid[radius:invalid.shape[0] - radius, radius:invalid.shape[1] - radius] = window_invalid == 0
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(REMOTE_SENSING_SPECS))
    parser.add_argument("--raster", action="append", required=True, help="Repeat in final channel order")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--field-id-column", default="auto")
    parser.add_argument("--target-resolution", type=float, default=None,
                        help="Metres/pixel; CV defaults to 10, LEM preserves its aligned grid")
    parser.add_argument("--acquisition-date", action="append", default=None,
                        help="Repeat in chronological order; required for paper-aligned LEM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=("coverage_70_10_20", "paper_field_disjoint"),
        default="coverage_70_10_20",
        help="Default all-class 70/10/20 benchmark, or the original field-disjoint ratios.",
    )
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.patch_size != 21 or args.patch_size % 2 != 1:
        parser.error("The paper-aligned experiment requires patch_size=21")
    dataset = canonical_remote_dataset(args.dataset)
    if dataset == "CV" and args.target_resolution is None:
        args.target_resolution = 10.0
    raster_paths = [Path(path).expanduser().resolve() for path in args.raster]
    labels_path = Path(args.labels).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    fiona, rasterio, Affine, Resampling, rasterize, reproject, transform_geom = _require_geo_packages()
    with fiona.open(labels_path) as source:
        properties = list(source.schema["properties"])
        if args.label_column not in properties:
            raise ValueError(f"Missing label column {args.label_column!r}; available={properties}")
        vector_crs = source.crs
        features = [
            {"properties": dict(feature["properties"]), "geometry": feature["geometry"]}
            for feature in source
        ]
    schema_stats = _property_counts(features, properties)
    field_column, field_reason = _choose_field_column(dataset, args.field_id_column, schema_stats, len(features))
    possible = _possible_field_columns(properties)
    print(f"number of shapefile rows: {len(features)}")
    print(f"number of unique Id: {schema_stats.get('Id', {}).get('unique_nonempty', 'missing')}")
    print(f"number of unique Field_numb: {schema_stats.get('Field_numb', {}).get('unique_nonempty', 'missing')}")
    print("other possible field identifiers:")
    for column in possible:
        print(f"  {column}: {schema_stats[column]['unique_nonempty']} unique nonempty values")
    print(f"selected field identifier: {field_column} ({field_reason})")

    usable_rows = []
    raw_field_labels: dict[str, set[str]] = {}
    skipped_empty_label = 0
    for ordinal, feature in enumerate(features):
        raw_label = feature["properties"].get(args.label_column)
        raw_field = feature["properties"].get(field_column)
        if raw_label is None or not str(raw_label).strip():
            skipped_empty_label += 1
            continue
        if raw_field is None or not str(raw_field).strip():
            raise ValueError(f"Usable labeled row has empty {field_column}")
        label, field = str(raw_label).strip(), str(raw_field).strip()
        raw_field_labels.setdefault(field, set()).add(label)
        usable_rows.append((ordinal, field, label, feature["geometry"]))
    conflicting_raw_fields = {
        field: sorted(labels) for field, labels in raw_field_labels.items() if len(labels) > 1
    }
    # Field_numb is the correct logical grouping for 476/477 source values.
    # One source-data collision (486) is used by two distant polygons with
    # different May-2016 crops, so those two rows are explicitly disambiguated.
    usable = []
    field_label: dict[str, str] = {}
    collision_resolution = []
    for ordinal, raw_field, label, geometry in usable_rows:
        field = raw_field
        if raw_field in conflicting_raw_fields:
            field = f"{raw_field}#row{ordinal}"
            collision_resolution.append({
                "raw_field_id": raw_field, "resolved_field_id": field,
                "shapefile_row": ordinal, "class": label,
            })
        field_label[field] = label
        usable.append((field, label, geometry))
    if conflicting_raw_fields:
        print(f"field ID collisions with conflicting {args.label_column} labels: {conflicting_raw_fields}")
        print(f"collision resolution: {collision_resolution}")
    field_values = sorted(field_label, key=str)
    field_to_index = {field: index for index, field in enumerate(field_values)}
    class_names = sorted(set(field_label.values()))
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    class_mapping = {
        "dataset": dataset, "num_classes": len(class_names), "class_to_idx": class_to_idx,
        "idx_to_class": {str(index): name for name, index in class_to_idx.items()},
    }
    original_grid, target_grid = _grid_from_sources(raster_paths, args.target_resolution, rasterio, Affine)
    stack, inferred_channels, band_records = _write_image_stack(
        raster_paths, output, target_grid, rasterio, Resampling, reproject
    )
    if dataset == "CV" and stack.shape[0] != 7:
        raise ValueError(f"CV requires B1-B7 (7 channels), found {stack.shape[0]}")
    dates = list(args.acquisition_date or [])
    if dataset == "LEM":
        if not dates:
            raise ValueError("LEM requires explicit --acquisition-date entries")
        if len(dates) * 2 != stack.shape[0]:
            raise ValueError(f"LEM dates x VV/VH disagree with channel count: {len(dates)} x 2 != {stack.shape[0]}")
        selected_channels = [f"{date}_{polarization}" for date in dates for polarization in ("VV", "VH")]
    else:
        selected_channels = [f"Landsat7_B{index}" for index in range(1, 8)]

    shapes = []
    for field, _, geometry in usable:
        if vector_crs and target_grid["crs"] and vector_crs != target_grid["crs"]:
            geometry = transform_geom(vector_crs, target_grid["crs"], geometry)
        shapes.append((geometry, field_to_index[field]))
    field_raster = rasterize(
        shapes, out_shape=(target_grid["height"], target_grid["width"]),
        transform=target_grid["transform"], fill=-1, all_touched=False, dtype="int32",
    )
    patch_valid = _full_patch_valid_mask(stack, args.patch_size)
    valid_center_mask = (field_raster >= 0) & patch_valid
    rows, cols = np.nonzero(valid_center_mask)
    rows = rows.astype(np.int32, copy=False)
    cols = cols.astype(np.int32, copy=False)
    centre_fields = field_raster[rows, cols].astype(np.int32, copy=False)
    field_label_lookup = np.asarray([class_to_idx[field_label[field]] for field in field_values], dtype=np.int16)
    labels = field_label_lookup[centre_fields]
    if args.split_strategy == "coverage_70_10_20":
        train_fraction, val_fraction, test_fraction = DEFAULT_COVERAGE_SPLIT_RATIOS
        val_fraction_of_train = val_fraction / (train_fraction + val_fraction)
        split_codes, split_manifest = build_all_class_coverage_split(
            field_indices=centre_fields,
            labels=labels,
            rows=rows,
            cols=cols,
            num_classes=len(class_names),
            seed=args.seed,
            test_fraction=test_fraction,
            val_fraction_of_train=val_fraction_of_train,
            patch_size=args.patch_size,
            class_names={index: name for name, index in class_to_idx.items()},
        )
        split_manifest.update({
            "dataset": dataset,
            "profile": f"all_class_coverage_70_10_20_seed{int(args.seed)}",
            "field_id_column": field_column,
        })
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
            record["class_name"] = class_names[int(record["class_index"])]
    else:
        test_fraction = float(REMOTE_SENSING_SPECS[dataset]["test_fraction"])
        field_split = deterministic_nested_field_split(
            field_values, seed=args.seed, test_fraction=test_fraction
        )
        field_split_code = np.empty(len(field_values), dtype=np.uint8)
        for split_name in SPLIT_NAMES:
            for field in field_split[split_name]:
                field_split_code[field_to_index[field]] = SPLIT_TO_CODE[split_name]
        split_codes = field_split_code[centre_fields]
        split_manifest = {
            "dataset": dataset,
            "profile": f"paper_field_random_seed{int(args.seed)}",
            "strategy": "deterministic_random_logical_field_nested_train_val_test",
            "seed": int(args.seed),
            "fully_field_disjoint": True,
            "field_id_column": field_column,
            "field_ids": field_split,
            "field_counts": {name: len(field_split[name]) for name in SPLIT_NAMES},
            "valid_center_counts": {
                name: int(np.sum(split_codes == SPLIT_TO_CODE[name])) for name in SPLIT_NAMES
            },
        }
    np.save(output / "center_rows.npy", rows)
    np.save(output / "center_cols.npy", cols)
    np.save(output / "labels.npy", labels)
    np.save(output / "field_indices.npy", centre_fields)
    np.save(output / "split_codes.npy", split_codes)
    (output / "field_values.json").write_text(json.dumps(field_values, indent=2) + "\n", encoding="utf-8")
    (output / "class_mapping.json").write_text(
        json.dumps(class_mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    class_details = []
    for class_name, class_index in class_to_idx.items():
        class_fields = {field for field, label in field_label.items() if label == class_name}
        class_details.append({
            "class_index": class_index, "raw_class_name": class_name,
            "num_logical_fields": len(class_fields), "num_valid_pixels": int(np.sum(labels == class_index)),
        })
    source_files = [_source_record(path, project_root) for path in raster_paths]
    vector_sidecars = sorted(labels_path.parent.glob(labels_path.stem + ".*"))
    label_files = [_source_record(path, project_root) for path in vector_sidecars if path.is_file()]
    transform_values = [float(value) for value in target_grid["transform"][:6]]
    manifest_base = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": dataset,
        "source_version": "official CV release" if dataset == "CV" else "Planetary Computer RTC reconstruction",
        "source_files": source_files,
        "label_files": label_files,
        "label_column": args.label_column,
        "field_id_column": field_column,
        "field_id_selection_reason": field_reason,
        "field_id_collision_resolution": collision_resolution,
        "field_schema": {
            "num_rows": len(features), "num_unique_Id": schema_stats.get("Id", {}).get("unique_nonempty"),
            "num_unique_Field_numb": schema_stats.get("Field_numb", {}).get("unique_nonempty"),
            "possible_field_identifiers": {name: schema_stats[name] for name in possible},
        },
        "selected_bands": band_records,
        "selected_channels": selected_channels,
        "selected_acquisition_dates": dates,
        "polarization_order": ["VV", "VH"] if dataset == "LEM" else None,
        "num_acquisitions": len(dates) if dataset == "LEM" else 1,
        "num_channels": int(stack.shape[0]),
        "patch_size": args.patch_size,
        "patch_label_definition": "class of centre pixel; surrounding context may cross field boundaries",
        "continuous_resampling": "bilinear before patch extraction",
        "categorical_resampling": "nearest-neighbour semantics via target-grid pixel-centre polygon rasterization",
        "original_spatial_resolution_m": list(original_grid["resolution"]),
        "spatial_resolution_m": list(target_grid["resolution"]),
        "original_raster_dimensions": [original_grid["height"], original_grid["width"]],
        "grid": {
            "crs": str(target_grid["crs"]), "transform": transform_values,
            "dimensions": [target_grid["height"], target_grid["width"]],
        },
        "patch_footprint_m": [args.patch_size * target_grid["resolution"][0],
                              args.patch_size * target_grid["resolution"][1]],
        "num_shapefile_rows": len(features), "num_logical_fields": len(field_values),
        "num_valid_centres": len(rows), "skipped_empty_label_rows": skipped_empty_label,
        "class_mapping": class_mapping, "class_details": class_details,
        "split_seed": args.seed,
        "split_profile": split_manifest["profile"],
        "split_strategy": split_manifest["strategy"],
    }
    preprocessing_hash = hashlib.sha256(
        json.dumps(manifest_base, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_base, "preprocessing_hash": preprocessing_hash}
    normal_fingerprint = normalization_fingerprint(manifest, split_manifest, class_mapping)
    train_indices = np.flatnonzero(split_codes == SPLIT_TO_CODE["train"])
    mean, std = compute_train_normalization(stack, rows, cols, train_indices, args.patch_size)
    normalization = {
        "dataset": dataset, "source_split": "train_only",
        "definition": "exact moments over all pixels of every valid training-centred 21x21 patch",
        "fingerprint": normal_fingerprint, "num_train_centres": len(train_indices),
        "mean": mean.tolist(), "std": std.tolist(),
    }
    manifest["normalization"] = {"file": "normalization.json", "fingerprint": normal_fingerprint}
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    (output / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    stack.flush()

    print("class index -> raw class name -> logical fields -> valid pixels")
    for record in class_details:
        print(f"  {record['class_index']} -> {record['raw_class_name']} -> "
              f"{record['num_logical_fields']} -> {record['num_valid_pixels']}")
    print("final split counts")
    for name in SPLIT_NAMES:
        print(f"  {name}: fields={split_manifest['field_counts'][name]}, "
              f"valid_centres={split_manifest['valid_center_counts'][name]}")
    print(json.dumps({
        "dataset": dataset,
        "original_resolution_m": manifest["original_spatial_resolution_m"],
        "resampled_resolution_m": manifest["spatial_resolution_m"],
        "original_dimensions": manifest["original_raster_dimensions"],
        "resampled_dimensions": manifest["grid"]["dimensions"],
        "patch_footprint_m": manifest["patch_footprint_m"],
        "input_shape": [manifest["num_channels"], args.patch_size, args.patch_size],
        "normalization": normalization,
        "preprocessing_hash": preprocessing_hash,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
