"""Offline preparation for CCT-20 feature-dependent LLP bags.

The safety boundary in this module is intentionally explicit:

* :func:`build_feature_bags` accepts image features and bagging parameters only.
* split-specific membership is fixed before targets are read for proportions.
* camera/location metadata is parsed and retained, but is never passed to any
  clustering, merging, splitting, or nearest-centroid function.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_ENCODER = "dinov2_vitb14"
DEFAULT_PCA_DIM = 128
DEFAULT_TARGET_AVG_BAG_SIZE = 64
DEFAULT_MIN_BAG_SIZE_RATIO = 0.25
DEFAULT_MAX_BAG_SIZE_RATIO = 2.0
DEFAULT_MIN_BBOX_AREA = 4096.0
DEFAULT_BBOX_SOURCE = "annotation"
ACTIVE_CACHE_POINTER = "cct_active_cache.json"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CCT20_ANNOTATION_SPLITS = (
    ("train_annotations.json", "train", "train"),
    ("cis_val_annotations.json", "train", "cis_val"),
    ("trans_val_annotations.json", "train", "trans_val"),
    ("cis_test_annotations.json", "test", "cis_test"),
    ("trans_test_annotations.json", "test", "trans_test"),
)


@dataclass(frozen=True)
class CCTSample:
    sample_index: int
    crop_id: str
    image_path: str
    image_id: str
    original_image_id: str
    original_image_path: str
    annotation_id: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    source_image_width: int
    source_image_height: int
    annotation_image_width: int
    annotation_image_height: int
    bbox_source: str
    target: int
    category_id: int
    class_name: str
    split: str
    official_split: str
    location: str
    sequence_id: str
    frame_number: str
    datetime: str
    annotation_category_ids: str


@dataclass(frozen=True)
class CCTCropCandidate:
    crop_id: str
    original_image_id: str
    original_image_path: str
    annotation_id: str
    category_id: int
    class_name: str
    split: str
    official_split: str
    location: str
    sequence_id: str
    frame_number: str
    datetime: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    annotation_image_width: int
    annotation_image_height: int


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    )
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name, delete=False
    )
    try:
        with handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name, delete=False
    )
    try:
        with handle:
            np.savez(handle, **values)
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _unique_named_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Official CCT-20 annotation file not found: {filename}")
    if len(matches) > 1:
        raise ValueError(f"Multiple CCT annotation files named {filename}: {matches}")
    return matches[0]


def _build_image_index(data_root: Path) -> dict[str, Path]:
    supported = {".jpg", ".jpeg", ".png"}
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if path.name in result:
            duplicates.add(path.name)
        else:
            result[path.name] = path
    if duplicates:
        example = sorted(duplicates)[:5]
        raise ValueError(f"CCT image basenames are not unique: {example}")
    return result


def _resolve_image_path(
    data_root: Path,
    file_name: str,
    fallback_index: dict[str, Path] | None,
) -> tuple[Path, dict[str, Path] | None]:
    relative = Path(str(file_name))
    candidates = (
        data_root / relative,
        data_root / "images" / relative,
        data_root / "cct_images" / relative,
        data_root / "eccv_18_all_images_sm" / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, fallback_index
    if fallback_index is None:
        fallback_index = _build_image_index(data_root)
    match = fallback_index.get(relative.name)
    if match is None:
        raise FileNotFoundError(f"CCT image referenced by annotations is missing: {file_name}")
    return match, fallback_index


def _compact_number(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def _cache_namespace(
    *,
    bbox_source: str,
    min_bbox_area: float,
    pca_dim: int,
    target_avg_bag_size: int,
    min_bag_size_ratio: float,
    max_bag_size_ratio: float,
    seed: int,
) -> str:
    source = "ann" if bbox_source == "annotation" else bbox_source
    return (
        f"cct_bboxcrop_{source}_area{_compact_number(min_bbox_area)}_"
        f"{DEFAULT_ENCODER}_pca{int(pca_dim)}_avg{int(target_avg_bag_size)}_"
        f"min{_compact_number(min_bag_size_ratio)}_"
        f"max{_compact_number(max_bag_size_ratio)}_seed{int(seed)}_all_splits"
    )


def _crop_id(official_split: str, image_id: str, annotation_id: str) -> str:
    value = f"{official_split}\0{image_id}\0{annotation_id}".encode()
    return f"{official_split}_{hashlib.sha1(value).hexdigest()[:20]}"


def _parse_bbox(
    raw_bbox: Any,
    *,
    annotation_width: int,
    annotation_height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    # MegaDetector-style boxes are normalized; official CCT boxes are absolute.
    if max(abs(x), abs(y), abs(width), abs(height)) <= 1.000001:
        x *= annotation_width
        width *= annotation_width
        y *= annotation_height
        height *= annotation_height
    return x, y, width, height


def parse_official_cct20(
    data_root: str | os.PathLike[str],
    *,
    annotation_dir: str | os.PathLike[str] | None = None,
    min_bbox_area: float = DEFAULT_MIN_BBOX_AREA,
    bbox_source: str = DEFAULT_BBOX_SOURCE,
) -> tuple[list[CCTCropCandidate], list[str], dict[str, Any]]:
    """Parse official CCT-20 annotations into label-preserving crop candidates."""
    if bbox_source != "annotation":
        raise ValueError("Only --bbox-source=annotation is currently implemented")
    if min_bbox_area < 0:
        raise ValueError("min_bbox_area must be non-negative")
    root = Path(data_root).expanduser().resolve()
    annotations_root = (
        root if annotation_dir is None else Path(annotation_dir).expanduser().resolve()
    )
    files = [
        (_unique_named_file(annotations_root, filename), split, official_split)
        for filename, split, official_split in CCT20_ANNOTATION_SPLITS
    ]

    payloads: list[tuple[Path, str, str, dict[str, Any]]] = []
    reference_categories: list[dict[str, Any]] | None = None
    reference_mapping: dict[int, str] | None = None
    for path, split, official_split in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not all(key in payload for key in ("images", "annotations", "categories")):
            raise ValueError(f"{path} is not a COCO Camera Traps annotation file")
        categories = [dict(value) for value in payload["categories"]]
        mapping = {int(value["id"]): str(value["name"]) for value in categories}
        if len(mapping) != len(categories):
            raise ValueError(f"{path} contains duplicate category IDs")
        if reference_categories is None:
            reference_categories = categories
            reference_mapping = mapping
        elif mapping != reference_mapping:
            raise ValueError(f"CCT category mapping differs across split file {path}")
        payloads.append((path, split, official_split, payload))
    assert reference_categories is not None and reference_mapping is not None

    retained_categories = [
        value for value in reference_categories
        if str(value["name"]).strip().lower() != "empty"
    ]
    category_ids = [int(value["id"]) for value in retained_categories]
    class_names = [str(value["name"]) for value in retained_categories]
    if len(set(class_names)) != len(class_names):
        raise ValueError("CCT class names must be unique")
    if "empty" not in {
        str(value["name"]).strip().lower() for value in reference_categories
    }:
        raise ValueError("Official CCT class mapping did not contain empty")
    class_index = {category_id: index for index, category_id in enumerate(category_ids)}

    candidates: list[CCTCropCandidate] = []
    seen_image_ids: set[str] = set()
    seen_annotation_ids: set[tuple[str, str]] = set()
    fallback_index: dict[str, Path] | None = None
    total_images = 0
    total_annotations = 0
    annotations_with_bbox = 0
    invalid_bbox = 0
    small_bbox = 0
    empty_annotations = 0
    clipped_bbox = 0
    split_counts = {"train": 0, "test": 0}
    official_split_counts: dict[str, int] = {}
    for path, split, official_split, payload in payloads:
        total_images += len(payload["images"])
        total_annotations += len(payload["annotations"])
        images_by_id = {str(value["id"]): value for value in payload["images"]}
        for image in payload["images"]:
            image_id = str(image["id"])
            if image_id in seen_image_ids:
                raise ValueError(f"CCT image {image_id} occurs in more than one split")
            seen_image_ids.add(image_id)
        for annotation in payload["annotations"]:
            image_id = str(annotation["image_id"])
            annotation_id = str(annotation["id"])
            identity = (official_split, annotation_id)
            if identity in seen_annotation_ids:
                raise ValueError(f"{path}: duplicate annotation ID {annotation_id}")
            seen_annotation_ids.add(identity)
            image = images_by_id.get(image_id)
            if image is None:
                raise ValueError(f"{path}: annotation references missing image {image_id}")
            category_id = int(annotation["category_id"])
            class_name = reference_mapping.get(category_id)
            if class_name is None:
                raise ValueError(f"{path}: unknown category ID {category_id}")
            if class_name.strip().lower() == "empty":
                empty_annotations += 1
                continue
            annotation_width = int(image.get("width", 0))
            annotation_height = int(image.get("height", 0))
            if annotation_width <= 0 or annotation_height <= 0:
                invalid_bbox += 1
                continue
            bbox = _parse_bbox(
                annotation.get("bbox"),
                annotation_width=annotation_width,
                annotation_height=annotation_height,
            )
            if bbox is None:
                invalid_bbox += 1
                continue
            annotations_with_bbox += 1
            x, y, width, height = bbox
            area = width * height
            if area < float(min_bbox_area):
                small_bbox += 1
                continue
            if x < 0 or y < 0 or x + width > annotation_width or y + height > annotation_height:
                clipped_bbox += 1
            image_path, fallback_index = _resolve_image_path(
                root, str(image["file_name"]), fallback_index
            )
            relative_path = image_path.relative_to(root).as_posix()
            candidates.append(
                CCTCropCandidate(
                    crop_id=_crop_id(official_split, image_id, annotation_id),
                    original_image_id=image_id,
                    original_image_path=relative_path,
                    annotation_id=annotation_id,
                    category_id=category_id,
                    class_name=class_name,
                    split=split,
                    official_split=official_split,
                    location=str(image.get("location", "")),
                    sequence_id=str(image.get("seq_id", "")),
                    frame_number=str(image.get("frame_num", "")),
                    datetime=str(image.get("datetime", image.get("date_captured", ""))),
                    bbox_x=x,
                    bbox_y=y,
                    bbox_width=width,
                    bbox_height=height,
                    bbox_area=area,
                    annotation_image_width=annotation_width,
                    annotation_image_height=annotation_height,
                )
            )
            split_counts[split] += 1
            official_split_counts[official_split] = official_split_counts.get(official_split, 0) + 1

    split_rank = {"train": 0, "test": 1}
    official_rank = {value[2]: index for index, value in enumerate(CCT20_ANNOTATION_SPLITS)}
    candidates.sort(
        key=lambda row: (
            split_rank[row.split], official_rank[row.official_split],
            row.original_image_id, row.annotation_id,
        )
    )
    annotation_files: list[str] = []
    for path, _, _, _ in payloads:
        try:
            annotation_files.append(str(path.relative_to(root)))
        except ValueError:
            annotation_files.append(str(path))
    source = {
        "dataset": "Caltech Camera Traps-20",
        "format": "COCO Camera Traps JSON",
        "annotation_files": annotation_files,
        "bbox_source": bbox_source,
        "min_bbox_area": float(min_bbox_area),
        "original_frames": total_images,
        "total_annotations": total_annotations,
        "annotations_with_valid_bbox": annotations_with_bbox,
        "annotations_removed_small_bbox": small_bbox,
        "annotations_removed_invalid_bbox": invalid_bbox,
        "annotations_clipped_to_image_bounds": clipped_bbox,
        "empty_annotations_excluded": empty_annotations,
        "retained_crop_instances": len(candidates),
        "split_counts": split_counts,
        "official_split_counts": official_split_counts,
        "instance_definition": "one valid annotation bounding-box crop",
        "empty_is_ordinary_class": False,
    }
    return candidates, class_names, source


def _atomic_save_jpeg(path: Path, image: Image.Image, *, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name, suffix=".jpg", delete=False
    )
    handle.close()
    try:
        image.save(handle.name, format="JPEG", quality=int(quality), subsampling=0)
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), [0, .01, .1, .25, .5, .75, .9, .99, 1])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p01", "p10", "p25", "p50", "p75", "p90", "p99", "max"),
            result,
        )
    }


def _write_crop_diagnostics(
    data_root: Path,
    processed: Path,
    samples: Sequence[CCTSample],
    *,
    seed: int,
) -> dict[str, str]:
    if not samples:
        raise ValueError("Cannot create CCT crop diagnostics without retained crops")
    diagnostics = processed / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    ordered = sorted(samples, key=lambda sample: (sample.bbox_area, sample.crop_id))
    count = min(64, len(samples))
    rng = np.random.default_rng(int(seed))
    random_indices = np.sort(rng.choice(len(samples), size=count, replace=False))
    midpoint = len(ordered) // 2
    half = count // 2
    selections = {
        "random_retained_crops": [samples[int(index)] for index in random_indices],
        "smallest_retained_boxes": ordered[:count],
        "median_size_boxes": ordered[max(0, midpoint - half):max(0, midpoint - half) + count],
        "largest_retained_boxes": ordered[-count:],
    }
    outputs: dict[str, str] = {}
    manifest: dict[str, list[dict[str, Any]]] = {}
    tile = 112
    gap = 4
    columns = 8
    for name, selected in selections.items():
        rows = int(math.ceil(len(selected) / columns))
        canvas = Image.new(
            "RGB", (columns * tile + (columns + 1) * gap, rows * tile + (rows + 1) * gap),
            color=(25, 25, 25),
        )
        entries: list[dict[str, Any]] = []
        for number, sample in enumerate(selected):
            with Image.open(data_root / sample.image_path) as image:
                thumbnail = ImageOps.fit(image.convert("RGB"), (tile, tile), method=Image.Resampling.BICUBIC)
            left = gap + (number % columns) * (tile + gap)
            top = gap + (number // columns) * (tile + gap)
            canvas.paste(thumbnail, (left, top))
            entries.append(
                {
                    "tile": number,
                    "crop_id": sample.crop_id,
                    "class_name": sample.class_name,
                    "bbox_area": sample.bbox_area,
                    "crop_path": sample.image_path,
                }
            )
        output = diagnostics / f"{name}.jpg"
        _atomic_save_jpeg(output, canvas, quality=92)
        outputs[name] = output.relative_to(data_root).as_posix()
        manifest[name] = entries
    _json_dump(diagnostics / "diagnostic_manifest.json", manifest)
    return outputs


def materialize_cct_crops(
    data_root: Path,
    processed: Path,
    candidates: Sequence[CCTCropCandidate],
    class_names: Sequence[str],
    *,
    force: bool,
    seed: int,
) -> tuple[list[CCTSample], dict[str, Any]]:
    """Create native-resolution object crops from annotation-space boxes.

    CCT's public ``*_all_images_sm`` images are smaller than the dimensions in
    the official JSON. Coordinates are therefore converted from annotation
    space to the decoded source-image space before cropping.
    """
    class_index = {name: index for index, name in enumerate(class_names)}
    samples: list[CCTSample] = []
    invalid: list[dict[str, Any]] = []
    source_dimensions_differ = 0
    clamped_to_bounds = 0
    for candidate in candidates:
        source_path = data_root / candidate.original_image_path
        try:
            with Image.open(source_path) as source_image:
                rgb = source_image.convert("RGB")
        except (OSError, ValueError) as exc:
            invalid.append({"crop_id": candidate.crop_id, "reason": f"image_decode:{exc}"})
            continue
        source_width, source_height = rgb.size
        if source_width <= 0 or source_height <= 0:
            invalid.append({"crop_id": candidate.crop_id, "reason": "empty_source_image"})
            continue
        if (source_width, source_height) != (
            candidate.annotation_image_width, candidate.annotation_image_height
        ):
            source_dimensions_differ += 1
        scale_x = source_width / float(candidate.annotation_image_width)
        scale_y = source_height / float(candidate.annotation_image_height)
        raw_x1 = candidate.bbox_x * scale_x
        raw_y1 = candidate.bbox_y * scale_y
        raw_x2 = (candidate.bbox_x + candidate.bbox_width) * scale_x
        raw_y2 = (candidate.bbox_y + candidate.bbox_height) * scale_y
        x1 = max(0, min(source_width, int(math.floor(raw_x1))))
        y1 = max(0, min(source_height, int(math.floor(raw_y1))))
        x2 = max(0, min(source_width, int(math.ceil(raw_x2))))
        y2 = max(0, min(source_height, int(math.ceil(raw_y2))))
        if (x1, y1, x2, y2) != (
            int(math.floor(raw_x1)), int(math.floor(raw_y1)),
            int(math.ceil(raw_x2)), int(math.ceil(raw_y2)),
        ):
            clamped_to_bounds += 1
        if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
            invalid.append(
                {
                    "crop_id": candidate.crop_id,
                    "reason": "invalid_bbox_after_scaling",
                    "scaled_xyxy": [raw_x1, raw_y1, raw_x2, raw_y2],
                    "source_size": [source_width, source_height],
                }
            )
            continue
        assert 0 <= x1 < x2 <= source_width
        assert 0 <= y1 < y2 <= source_height
        crop_relative = Path(processed.name) / "crops" / candidate.official_split / f"{candidate.crop_id}.jpg"
        crop_path = data_root / crop_relative
        if force or not crop_path.is_file():
            _atomic_save_jpeg(crop_path, rgb.crop((x1, y1, x2, y2)), quality=95)
        sample_index = len(samples)
        samples.append(
            CCTSample(
                sample_index=sample_index,
                crop_id=candidate.crop_id,
                image_path=crop_relative.as_posix(),
                image_id=candidate.crop_id,
                original_image_id=candidate.original_image_id,
                original_image_path=candidate.original_image_path,
                annotation_id=candidate.annotation_id,
                bbox_x=candidate.bbox_x,
                bbox_y=candidate.bbox_y,
                bbox_width=candidate.bbox_width,
                bbox_height=candidate.bbox_height,
                bbox_area=candidate.bbox_area,
                source_image_width=source_width,
                source_image_height=source_height,
                annotation_image_width=candidate.annotation_image_width,
                annotation_image_height=candidate.annotation_image_height,
                bbox_source=DEFAULT_BBOX_SOURCE,
                target=class_index[candidate.class_name],
                category_id=candidate.category_id,
                class_name=candidate.class_name,
                split=candidate.split,
                official_split=candidate.official_split,
                location=candidate.location,
                sequence_id=candidate.sequence_id,
                frame_number=candidate.frame_number,
                datetime=candidate.datetime,
                annotation_category_ids=str(candidate.category_id),
            )
        )
    if invalid:
        rejection_path = processed / "crop_rejections.jsonl"
        rejection_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in invalid),
            encoding="utf-8",
        )
    if not samples:
        raise ValueError("CCT crop preparation retained no valid instances")
    diagnostics = _write_crop_diagnostics(data_root, processed, samples, seed=seed)
    crop_stats = {
        "retained_crop_instances": len(samples),
        "invalid_after_image_decode_or_scaling": len(invalid),
        "source_dimensions_differ_from_annotations": source_dimensions_differ,
        "boxes_clamped_to_source_bounds": clamped_to_bounds,
        "bbox_width_quantiles_annotation_pixels": _quantiles([sample.bbox_width for sample in samples]),
        "bbox_height_quantiles_annotation_pixels": _quantiles([sample.bbox_height for sample in samples]),
        "bbox_area_quantiles_annotation_pixels": _quantiles([sample.bbox_area for sample in samples]),
        "instances_per_class": {
            name: sum(sample.class_name == name for sample in samples)
            for name in class_names
        },
        "diagnostic_grids": diagnostics,
    }
    return samples, crop_stats


def load_cached_cct_crops(
    data_root: Path,
    processed: Path,
    candidates: Sequence[CCTCropCandidate],
    class_names: Sequence[str],
) -> tuple[list[CCTSample], dict[str, Any]]:
    """Resume preparation from a completely validated crop manifest.

    This avoids decoding all source frames again when a later DINO/PCA/bagging
    stage is interrupted. Every row is checked against the current annotation
    parse before the cached crops are accepted.
    """
    instances_path = processed / "instances.csv"
    diagnostics = processed / "diagnostics"
    grid_names = (
        "random_retained_crops",
        "smallest_retained_boxes",
        "median_size_boxes",
        "largest_retained_boxes",
    )
    required = [instances_path, diagnostics / "diagnostic_manifest.json"] + [
        diagnostics / f"{name}.jpg" for name in grid_names
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("The intermediate CCT crop cache is incomplete")
    with instances_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(candidates):
        raise ValueError("Cached CCT crop count differs from current annotations")
    int_fields = {
        "sample_index", "source_image_width", "source_image_height",
        "annotation_image_width", "annotation_image_height", "target", "category_id",
    }
    float_fields = {
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_area",
    }
    samples: list[CCTSample] = []
    class_index = {name: index for index, name in enumerate(class_names)}
    for index, (row, candidate) in enumerate(zip(rows, candidates)):
        values: dict[str, Any] = {}
        for name in CCTSample.__dataclass_fields__:
            if name not in row:
                raise ValueError(f"Cached CCT crop manifest is missing {name!r}")
            if name in int_fields:
                values[name] = int(row[name])
            elif name in float_fields:
                values[name] = float(row[name])
            else:
                values[name] = row[name]
        sample = CCTSample(**values)
        exact = (
            sample.sample_index == index
            and sample.crop_id == candidate.crop_id
            and sample.original_image_id == candidate.original_image_id
            and sample.original_image_path == candidate.original_image_path
            and sample.annotation_id == candidate.annotation_id
            and sample.category_id == candidate.category_id
            and sample.class_name == candidate.class_name
            and sample.target == class_index[candidate.class_name]
            and sample.split == candidate.split
            and sample.official_split == candidate.official_split
            and sample.bbox_source == DEFAULT_BBOX_SOURCE
        )
        geometry = np.allclose(
            [sample.bbox_x, sample.bbox_y, sample.bbox_width, sample.bbox_height],
            [candidate.bbox_x, candidate.bbox_y, candidate.bbox_width, candidate.bbox_height],
            rtol=0,
            atol=1e-9,
        )
        if not exact or not geometry:
            raise ValueError(
                f"Cached CCT crop row {index} differs from current annotation {candidate.annotation_id}"
            )
        if not (data_root / sample.image_path).is_file():
            raise FileNotFoundError(f"Cached CCT crop is missing: {sample.image_path}")
        samples.append(sample)
    crop_stats = {
        "retained_crop_instances": len(samples),
        "invalid_after_image_decode_or_scaling": 0,
        "source_dimensions_differ_from_annotations": sum(
            (sample.source_image_width, sample.source_image_height)
            != (sample.annotation_image_width, sample.annotation_image_height)
            for sample in samples
        ),
        "boxes_clamped_to_source_bounds": sum(
            candidate.bbox_x < 0
            or candidate.bbox_y < 0
            or candidate.bbox_x + candidate.bbox_width > candidate.annotation_image_width
            or candidate.bbox_y + candidate.bbox_height > candidate.annotation_image_height
            for candidate in candidates
        ),
        "bbox_width_quantiles_annotation_pixels": _quantiles([sample.bbox_width for sample in samples]),
        "bbox_height_quantiles_annotation_pixels": _quantiles([sample.bbox_height for sample in samples]),
        "bbox_area_quantiles_annotation_pixels": _quantiles([sample.bbox_area for sample in samples]),
        "instances_per_class": {
            name: sum(sample.class_name == name for sample in samples)
            for name in class_names
        },
        "diagnostic_grids": {
            name: (Path(processed.name) / "diagnostics" / f"{name}.jpg").as_posix()
            for name in grid_names
        },
    }
    print(f"CCT native crop cache hit and revalidated: {instances_path}", flush=True)
    return samples, crop_stats


def write_instances_csv(path: Path, samples: Sequence[CCTSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(samples[0]).keys())
    handle = tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=path.name, delete=False
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in samples)
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _sample_digest(samples: Sequence[CCTSample], *, include_targets: bool) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        # Frozen image embeddings depend on image identity/order, not on how an
        # official split is assigned to the downstream training protocol.
        values: list[Any] = [sample.sample_index, sample.image_id, sample.image_path]
        if include_targets:
            # Bag targets depend on both the downstream split and hidden label.
            values.extend([sample.split, sample.target])
        digest.update(("\0".join(str(value) for value in values) + "\n").encode())
    return digest.hexdigest()


def _feature_index_matches(path: Path, samples: Sequence[CCTSample]) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    if len(rows) != len(samples):
        return False
    try:
        return all(
            int(row.get("sample_index", -1)) == sample.sample_index
            and row.get("image_path") == sample.image_path
            for row, sample in zip(rows, samples)
        )
    except (TypeError, ValueError):
        return False


class _FeatureExtractionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        samples: Sequence[CCTSample],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.data_root = data_root
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(self.data_root / sample.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return sample.sample_index, tensor


def dinov2_preprocess() -> transforms.Compose:
    """Deterministic ImageNet/DINOv2 evaluation preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_dinov2_vitb14(
    *, dinov2_repo: str | os.PathLike[str] | None = None
) -> torch.nn.Module:
    if dinov2_repo is None:
        kwargs = {"source": "github"}
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", DEFAULT_ENCODER, trust_repo=True, **kwargs
            )
        except TypeError:
            model = torch.hub.load("facebookresearch/dinov2", DEFAULT_ENCODER, **kwargs)
    else:
        model = torch.hub.load(
            str(Path(dinov2_repo).expanduser().resolve()),
            DEFAULT_ENCODER,
            source="local",
        )
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DINOv2 feature encoder was not fully frozen")
    return model


def _encoder_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "embedding", "features"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported DINOv2 output type: {type(output)!r}")


def extract_or_load_dinov2_features(
    data_root: Path,
    samples: Sequence[CCTSample],
    processed: Path,
    *,
    device: str,
    extraction_batch_size: int,
    num_workers: int,
    force: bool,
    dinov2_repo: str | os.PathLike[str] | None = None,
    model: torch.nn.Module | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_path = processed / "dinov2_vitb14_embeddings.npy"
    index_path = processed / "feature_index.csv"
    manifest_path = processed / "feature_manifest.json"
    digest = _sample_digest(samples, include_targets=False)
    if not force and feature_path.is_file() and index_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = np.load(feature_path, mmap_mode="r")
        base_cache_valid = (
            manifest.get("encoder") == DEFAULT_ENCODER
            and manifest.get("l2_normalized") is True
            and features.ndim == 2
            and len(features) == len(samples)
            and int(manifest.get("embedding_dim", -1)) == features.shape[1]
        )
        if base_cache_valid and manifest.get("sample_digest") == digest:
            print(f"CCT DINOv2 cache hit: {feature_path}", flush=True)
            return features, manifest
        if base_cache_valid and _feature_index_matches(index_path, samples):
            # Older caches included downstream split names in this digest.
            # Row/path equality proves that every frozen image embedding still
            # corresponds to the same image, so only the manifest needs a safe
            # metadata migration when validation images become training data.
            manifest["sample_digest"] = digest
            manifest["sample_identity_fields"] = [
                "sample_index", "image_id", "image_path"
            ]
            _json_dump(manifest_path, manifest)
            print(
                f"CCT DINOv2 cache identity revalidated after split remapping: {feature_path}",
                flush=True,
            )
            return features, manifest
        raise ValueError(
            "Existing CCT feature cache is stale or incompatible; rerun with --force"
        )

    if extraction_batch_size <= 0:
        raise ValueError("extraction_batch_size must be positive")
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction was requested but CUDA is unavailable")
    encoder = load_dinov2_vitb14(dinov2_repo=dinov2_repo) if model is None else model
    encoder.requires_grad_(False)
    encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Feature encoder must have requires_grad=False")
    encoder.to(resolved_device)
    dataset = _FeatureExtractionDataset(data_root, samples, dinov2_preprocess())
    loader = DataLoader(
        dataset,
        batch_size=int(extraction_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=resolved_device.type == "cuda",
    )
    output_parts: list[np.ndarray] = []
    seen_indices: list[np.ndarray] = []
    started = time.time()
    print(
        f"Extracting {len(samples)} CCT DINOv2 features on {resolved_device} "
        f"with batch_size={extraction_batch_size}",
        flush=True,
    )
    with torch.no_grad():
        for batch_number, (indices, images) in enumerate(loader, start=1):
            output = _encoder_output(encoder(images.to(resolved_device, non_blocking=True)))
            if output.ndim != 2 or len(output) != len(images):
                raise ValueError(f"DINOv2 returned unexpected feature shape {tuple(output.shape)}")
            normalized = F.normalize(output.float(), p=2, dim=1)
            output_parts.append(normalized.cpu().numpy().astype(np.float32, copy=False))
            seen_indices.append(indices.numpy().astype(np.int64, copy=False))
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(loader):
                completed = min(batch_number * int(extraction_batch_size), len(samples))
                rate = completed / max(time.time() - started, 1e-9)
                print(
                    f"DINOv2 features: {completed}/{len(samples)} "
                    f"({rate:.1f} images/s)",
                    flush=True,
                )
    features = np.concatenate(output_parts, axis=0)
    extracted_indices = np.concatenate(seen_indices)
    expected_indices = np.arange(len(samples), dtype=np.int64)
    if not np.array_equal(extracted_indices, expected_indices):
        raise ValueError("DINOv2 feature extraction order does not match sample_index")
    if not np.isfinite(features).all():
        raise ValueError("DINOv2 feature cache contains non-finite values")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5):
        raise ValueError("DINOv2 features were not L2-normalized")
    _save_npy(feature_path, features)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "image_path"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {"sample_index": sample.sample_index, "image_path": sample.image_path}
            )
    manifest = {
        "encoder": DEFAULT_ENCODER,
        "pretrained": True,
        "frozen": True,
        "eval_mode": True,
        "no_grad": True,
        "requires_grad": False,
        "preprocess": "resize_256_bicubic_center_crop_224_imagenet_normalize",
        "l2_normalized": True,
        "instances": len(samples),
        "embedding_dim": int(features.shape[1]),
        "sample_digest": digest,
        "sample_identity_fields": ["sample_index", "image_id", "image_path"],
        "feature_file": feature_path.name,
        "index_file": index_path.name,
    }
    _json_dump(manifest_path, manifest)
    return np.load(feature_path, mmap_mode="r"), manifest


def fit_or_load_train_pca(
    features: np.ndarray,
    samples: Sequence[CCTSample],
    processed: Path,
    *,
    pca_dim: int,
    seed: int,
    feature_manifest: dict[str, Any],
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if pca_dim <= 0:
        raise ValueError("pca_dim must be positive")
    transform_path = processed / "pca_transform.npz"
    transformed_path = processed / "pca_features.npy"
    manifest_path = processed / "pca_manifest.json"
    train_indices = np.asarray(
        [sample.sample_index for sample in samples if sample.split == "train"],
        dtype=np.int64,
    )
    train_index_digest = hashlib.sha256(train_indices.tobytes()).hexdigest()
    if not force and transform_path.is_file() and transformed_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        transformed = np.load(transformed_path, mmap_mode="r")
        if (
            manifest.get("feature_sample_digest") == feature_manifest["sample_digest"]
            and manifest.get("train_index_digest") == train_index_digest
            and int(manifest.get("requested_pca_dim", -1)) == int(pca_dim)
            and int(manifest.get("seed", -1)) == int(seed)
            and transformed.ndim == 2
            and len(transformed) == len(samples)
            and transformed.shape[1] == int(manifest.get("effective_pca_dim", -1))
        ):
            print(f"CCT PCA cache hit: {transformed_path}", flush=True)
            return transformed, manifest
        print("CCT PCA cache is stale; refitting from the current training rows", flush=True)
    if len(train_indices) < 2:
        raise ValueError("CCT PCA needs at least two training images")
    effective_dim = min(int(pca_dim), len(train_indices), int(features.shape[1]))
    pca = PCA(
        n_components=effective_dim,
        svd_solver="randomized" if effective_dim < min(len(train_indices), features.shape[1]) else "full",
        random_state=int(seed),
        whiten=False,
    )
    # Split metadata selects rows; labels and locations are never consulted.
    print(
        f"Fitting label-blind PCA({effective_dim}) on {len(train_indices)} train features",
        flush=True,
    )
    pca.fit(np.asarray(features[train_indices], dtype=np.float32))
    transformed = pca.transform(np.asarray(features, dtype=np.float32)).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("PCA-transformed CCT features contain non-finite values")
    _save_npz(
        transform_path,
        mean_=pca.mean_.astype(np.float64),
        components_=pca.components_.astype(np.float64),
        explained_variance_=pca.explained_variance_.astype(np.float64),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float64),
        singular_values_=pca.singular_values_.astype(np.float64),
    )
    _save_npy(transformed_path, transformed)
    manifest = {
        "fit_split": "train",
        "fit_instances": int(len(train_indices)),
        "requested_pca_dim": int(pca_dim),
        "effective_pca_dim": int(effective_dim),
        "seed": int(seed),
        "labels_used": False,
        "locations_used": False,
        "feature_sample_digest": feature_manifest["sample_digest"],
        "train_index_digest": train_index_digest,
        "transform_file": transform_path.name,
        "transformed_feature_file": transformed_path.name,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    _json_dump(manifest_path, manifest)
    return np.load(transformed_path, mmap_mode="r"), manifest


def _derived_seed(seed: int, *parts: int) -> int:
    digest = hashlib.blake2b(
        ":".join(str(int(value)) for value in (seed, *parts)).encode(), digest_size=4
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _feature_lexicographic_split(
    features: np.ndarray, indices: np.ndarray, count: int
) -> list[np.ndarray]:
    values = np.asarray(features[indices], dtype=np.float64)
    # Feature coordinates determine the order.  The row index is only a final
    # deterministic tie-break for exactly identical feature vectors.
    keys: list[np.ndarray] = [indices]
    keys.extend(values[:, column] for column in range(values.shape[1] - 1, -1, -1))
    order = np.lexsort(tuple(keys))
    return [
        np.sort(indices[part].astype(np.int64, copy=False))
        for part in np.array_split(order, min(int(count), len(indices)))
        if len(part)
    ]


def _feature_partition(
    features: np.ndarray,
    indices: np.ndarray,
    count: int,
    *,
    seed: int,
) -> list[np.ndarray]:
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    count = min(max(1, int(count)), len(indices))
    if count == 1:
        return [indices]
    values = np.asarray(features[indices], dtype=np.float32)
    model = MiniBatchKMeans(
        n_clusters=count,
        random_state=int(seed),
        n_init=10,
        batch_size=min(len(indices), max(256, 8 * count)),
        max_iter=200,
        max_no_improvement=30,
    )
    labels = model.fit_predict(values)
    groups = [indices[labels == label] for label in range(count)]
    groups = [np.sort(group) for group in groups if len(group)]
    # Identical/near-identical vectors can leave MiniBatchKMeans with empty
    # clusters.  A deterministic feature-order partition prevents an infinite
    # oversized-cluster loop without consulting labels or metadata.
    if len(groups) != count or max(len(group) for group in groups) == len(indices):
        groups = _feature_lexicographic_split(features, indices, count)
    return groups


def _split_oversized(
    features: np.ndarray,
    indices: np.ndarray,
    *,
    target_avg_bag_size: int,
    max_bag_size: int,
    seed: int,
) -> list[np.ndarray]:
    queue = [np.sort(np.asarray(indices, dtype=np.int64))]
    result: list[np.ndarray] = []
    iteration = 0
    while queue:
        group = queue.pop(0)
        if len(group) <= max_bag_size:
            result.append(group)
            continue
        count = int(math.ceil(len(group) / target_avg_bag_size))
        parts = _feature_partition(
            features, group, count, seed=_derived_seed(seed, iteration, len(group))
        )
        iteration += 1
        if len(parts) < 2 or max(len(part) for part in parts) >= len(group):
            raise RuntimeError("Feature-only oversized-cluster splitting made no progress")
        queue[0:0] = parts
    return result


def _centroid(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(features[indices], dtype=np.float64).mean(axis=0)


def build_feature_bags(
    features: np.ndarray,
    *,
    target_avg_bag_size: int = DEFAULT_TARGET_AVG_BAG_SIZE,
    min_bag_size_ratio: float = DEFAULT_MIN_BAG_SIZE_RATIO,
    max_bag_size_ratio: float = DEFAULT_MAX_BAG_SIZE_RATIO,
    seed: int = 42,
) -> list[np.ndarray]:
    """Build variable-size bags from image features only.

    This API deliberately has no target, class, camera, or location parameter.
    """
    values = np.asarray(features)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    target = int(target_avg_bag_size)
    if target <= 0:
        raise ValueError("target_avg_bag_size must be positive")
    if not (0 < float(min_bag_size_ratio) <= 1):
        raise ValueError("min_bag_size_ratio must be in (0, 1]")
    if float(max_bag_size_ratio) < 1:
        raise ValueError("max_bag_size_ratio must be at least 1")
    min_size = max(1, int(math.floor(float(min_bag_size_ratio) * target)))
    max_size = max(1, int(math.ceil(float(max_bag_size_ratio) * target)))
    if min_size > max_size:
        raise ValueError("minimum CCT bag size exceeds maximum bag size")

    cluster_count = min(len(values), max(1, int(round(len(values) / target))))
    clusters = _feature_partition(
        values,
        np.arange(len(values), dtype=np.int64),
        cluster_count,
        seed=int(seed),
    )
    split_clusters: list[np.ndarray] = []
    for number, cluster in enumerate(clusters):
        split_clusters.extend(
            _split_oversized(
                values,
                cluster,
                target_avg_bag_size=target,
                max_bag_size=max_size,
                seed=_derived_seed(seed, 1, number),
            )
        )
    clusters = split_clusters

    max_iterations = max(10, 10 * len(values))
    for merge_iteration in range(max_iterations):
        small = [index for index, cluster in enumerate(clusters) if len(cluster) < min_size]
        if not small or len(clusters) == 1:
            break
        centroids = [_centroid(values, cluster) for cluster in clusters]
        source = min(
            small,
            key=lambda index: (
                len(clusters[index]),
                tuple(np.round(centroids[index], 12)),
                index,
            ),
        )
        admissible = [
            index
            for index, cluster in enumerate(clusters)
            if index != source and len(cluster) + len(clusters[source]) <= max_size
        ]
        candidates = admissible or [
            index for index in range(len(clusters)) if index != source
        ]
        destination = min(
            candidates,
            key=lambda index: (
                float(np.square(centroids[source] - centroids[index]).sum()),
                tuple(np.round(centroids[index], 12)),
                index,
            ),
        )
        merged = np.sort(np.concatenate([clusters[source], clusters[destination]]))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {source, destination}
        ]
        clusters.extend(
            _split_oversized(
                values,
                merged,
                target_avg_bag_size=target,
                max_bag_size=max_size,
                seed=_derived_seed(seed, 2, merge_iteration),
            )
        )
    else:
        raise RuntimeError("Feature-only small-cluster merging did not converge")

    clusters = [np.sort(np.asarray(cluster, dtype=np.int64)) for cluster in clusters]
    clusters.sort(key=lambda cluster: (int(cluster[0]), len(cluster)))
    flattened = np.concatenate(clusters)
    if len(flattened) != len(values) or not np.array_equal(
        np.sort(flattened), np.arange(len(values), dtype=np.int64)
    ):
        raise RuntimeError("Feature bag construction lost or duplicated instances")
    if max(len(cluster) for cluster in clusters) > max_size:
        raise RuntimeError("Feature bag construction left an oversized cluster")
    if len(values) >= min_size and min(len(cluster) for cluster in clusters) < min_size:
        raise RuntimeError("Feature bag construction left an avoidable undersized cluster")
    return clusters


def _bag_stats(sizes: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(list(sizes), dtype=np.int64)
    return {
        "bags": int(len(values)),
        "instances": int(values.sum()),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": int(values.max()),
    }


def build_split_bag_records(
    pca_features: np.ndarray,
    samples: Sequence[CCTSample],
    class_names: Sequence[str],
    *,
    target_avg_bag_size: int,
    min_bag_size_ratio: float,
    max_bag_size_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    targets = np.asarray([sample.target for sample in samples], dtype=np.int64)
    for split_number, split in enumerate(("train", "test")):
        positions = np.asarray(
            [sample.sample_index for sample in samples if sample.split == split],
            dtype=np.int64,
        )
        if len(positions) == 0:
            raise ValueError(f"CCT split {split!r} is empty")
        # Critical non-leakage boundary: only split-local image features enter.
        local_bags = build_feature_bags(
            np.asarray(pca_features[positions], dtype=np.float32),
            target_avg_bag_size=target_avg_bag_size,
            min_bag_size_ratio=min_bag_size_ratio,
            max_bag_size_ratio=max_bag_size_ratio,
            seed=_derived_seed(seed, 100, split_number),
        )
        global_bags = [np.sort(positions[local]) for local in local_bags]
        # IMPORTANT:
        # Ground-truth species labels are intentionally unavailable during
        # feature-bag construction. They are accessed only after bag
        # membership has been finalized to derive LLP proportions.
        for bag_number, indices in enumerate(global_bags):
            counts = np.bincount(targets[indices], minlength=len(class_names))
            records.append(
                {
                    "bag_id": f"{split}_feature_{bag_number:05d}",
                    "split": split,
                    "n_instances": int(len(indices)),
                    "instance_indices": indices.tolist(),
                    "class_counts": counts.tolist(),
                    "class_proportions": (counts / len(indices)).tolist(),
                }
            )
        split_stats[split] = _bag_stats(len(indices) for indices in global_bags)
    return records, split_stats


def prepare_cct(
    data_root: str | os.PathLike[str],
    *,
    annotation_dir: str | os.PathLike[str] | None = None,
    bbox_source: str = DEFAULT_BBOX_SOURCE,
    min_bbox_area: float = DEFAULT_MIN_BBOX_AREA,
    device: str = "auto",
    extraction_batch_size: int = 64,
    num_workers: int = 4,
    pca_dim: int = DEFAULT_PCA_DIM,
    target_avg_bag_size: int = DEFAULT_TARGET_AVG_BAG_SIZE,
    min_bag_size_ratio: float = DEFAULT_MIN_BAG_SIZE_RATIO,
    max_bag_size_ratio: float = DEFAULT_MAX_BAG_SIZE_RATIO,
    seed: int = 42,
    force: bool = False,
    dinov2_repo: str | os.PathLike[str] | None = None,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    processed = root / _cache_namespace(
        bbox_source=bbox_source,
        min_bbox_area=min_bbox_area,
        pca_dim=pca_dim,
        target_avg_bag_size=target_avg_bag_size,
        min_bag_size_ratio=min_bag_size_ratio,
        max_bag_size_ratio=max_bag_size_ratio,
        seed=seed,
    )
    processed.mkdir(parents=True, exist_ok=True)
    candidates, class_names, source = parse_official_cct20(
        root,
        annotation_dir=annotation_dir,
        min_bbox_area=min_bbox_area,
        bbox_source=bbox_source,
    )
    if not force and (processed / "instances.csv").is_file():
        samples, crop_stats = load_cached_cct_crops(
            root, processed, candidates, class_names
        )
    else:
        samples, crop_stats = materialize_cct_crops(
            root, processed, candidates, class_names, force=force, seed=seed
        )
    source["crop_statistics"] = crop_stats
    source["retained_crop_instances"] = len(samples)
    source["split_counts"] = {
        split: sum(sample.split == split for sample in samples)
        for split in ("train", "test")
    }
    source["official_split_counts"] = {
        official_split: sum(sample.official_split == official_split for sample in samples)
        for _, _, official_split in CCT20_ANNOTATION_SPLITS
    }
    print(
        f"Prepared CCT-20 bbox crops: instances={len(samples)} classes={len(class_names)} "
        f"splits={source['split_counts']}",
        flush=True,
    )
    print(
        "CCT bbox audit: "
        f"frames={source['original_frames']} "
        f"annotations={source['total_annotations']} "
        f"valid_bbox={source['annotations_with_valid_bbox']} "
        f"invalid_or_missing_bbox={source['annotations_removed_invalid_bbox']} "
        f"small_removed={source['annotations_removed_small_bbox']} "
        f"retained={len(samples)}",
        flush=True,
    )
    print(
        "CCT retained bbox quantiles: "
        f"width={crop_stats['bbox_width_quantiles_annotation_pixels']} "
        f"height={crop_stats['bbox_height_quantiles_annotation_pixels']} "
        f"area={crop_stats['bbox_area_quantiles_annotation_pixels']}",
        flush=True,
    )
    print(f"CCT instances per class: {crop_stats['instances_per_class']}", flush=True)
    write_instances_csv(processed / "instances.csv", samples)
    _json_dump(
        processed / "class_mapping.json",
        {
            "class_names": list(class_names),
            "name_to_index": {
                name: index for index, name in enumerate(class_names)
            },
            "original_category_id_to_training_index": {
                str(sample.category_id): int(sample.target) for sample in samples
            },
            "empty_excluded": True,
        },
    )

    features, feature_manifest = extract_or_load_dinov2_features(
        root,
        samples,
        processed,
        device=device,
        extraction_batch_size=extraction_batch_size,
        num_workers=num_workers,
        force=force,
        dinov2_repo=dinov2_repo,
        model=model,
    )
    feature_manifest.update(
        {
            "dataset": "CCT20",
            "instance_type": "bbox_crop",
            "bbox_source": bbox_source,
            "min_bbox_area": float(min_bbox_area),
            "cache_namespace": processed.name,
        }
    )
    _json_dump(processed / "feature_manifest.json", feature_manifest)
    pca_features, pca_manifest = fit_or_load_train_pca(
        features,
        samples,
        processed,
        pca_dim=pca_dim,
        seed=seed,
        feature_manifest=feature_manifest,
        force=force,
    )
    pca_manifest.update(
        {
            "instance_type": "bbox_crop",
            "bbox_source": bbox_source,
            "min_bbox_area": float(min_bbox_area),
            "cache_namespace": processed.name,
        }
    )
    _json_dump(processed / "pca_manifest.json", pca_manifest)

    bag_manifest_path = processed / "bag_manifest.json"
    bag_path = processed / "bags.json"
    requested = {
        "target_avg_bag_size": int(target_avg_bag_size),
        "min_bag_size_ratio": float(min_bag_size_ratio),
        "max_bag_size_ratio": float(max_bag_size_ratio),
        "seed": int(seed),
        "requested_pca_dim": int(pca_manifest["requested_pca_dim"]),
        "effective_pca_dim": int(pca_manifest["effective_pca_dim"]),
        "pca_feature_digest": feature_manifest["sample_digest"],
        "target_digest": _sample_digest(samples, include_targets=True),
        "instance_type": "bbox_crop",
        "bbox_source": bbox_source,
        "min_bbox_area": float(min_bbox_area),
        "cache_namespace": processed.name,
    }
    use_cached_bags = False
    if not force and bag_path.is_file() and bag_manifest_path.is_file():
        existing = json.loads(bag_manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in requested}
        if comparable == requested and existing.get("membership_inputs") == ["pca_image_features"]:
            use_cached_bags = True
            bag_manifest = existing
            print(f"CCT feature-bag cache hit: {bag_path}", flush=True)
        else:
            print("CCT bag cache is stale; rebuilding feature-only memberships", flush=True)
    if not use_cached_bags:
        print(
            "Building split-isolated CCT feature bags with "
            f"target_avg_bag_size={target_avg_bag_size}",
            flush=True,
        )
        records, split_stats = build_split_bag_records(
            pca_features,
            samples,
            class_names,
            target_avg_bag_size=int(target_avg_bag_size),
            min_bag_size_ratio=float(min_bag_size_ratio),
            max_bag_size_ratio=float(max_bag_size_ratio),
            seed=int(seed),
        )
        _json_dump(bag_path, records)
        bag_manifest = {
            **requested,
            "algorithm": "split-local MiniBatchKMeans with feature-only size safeguards",
            "membership_inputs": ["pca_image_features"],
            "labels_used_for_membership": False,
            "locations_used_for_membership": False,
            "camera_ids_used_for_membership": False,
            "proportions_computed_after_membership": True,
            "min_bag_size": max(
                1, int(math.floor(min_bag_size_ratio * target_avg_bag_size))
            ),
            "max_bag_size": max(
                1, int(math.ceil(max_bag_size_ratio * target_avg_bag_size))
            ),
            "split_isolation": True,
            "split_statistics": split_stats,
            "bags_file": bag_path.name,
        }
        _json_dump(bag_manifest_path, bag_manifest)
        print(f"CCT bag statistics: {split_stats}", flush=True)

    metadata = {
        "dataset_name": "Caltech Camera Traps",
        "dataset_variant": "CCT-20 official benchmark bbox-crop task",
        "instance_type": "bbox_crop",
        "bbox_source": bbox_source,
        "min_bbox_area": float(min_bbox_area),
        "processed_directory": processed.name,
        "instances": len(samples),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "source": source,
        "feature_manifest": feature_manifest,
        "pca_manifest": pca_manifest,
        "bag_manifest": bag_manifest,
        "classifier_preprocessing": {
            "backbone": "ImageNet-pretrained ResNet-18",
            "input_resolution": 112,
            "train": (
                "RandomResizedCrop(112,scale=(0.2,1.0)); HorizontalFlip(0.5); "
                "ColorJitter(0.4,0.4,0.4,0.1)@0.8; RandomGrayscale(0.2); "
                "ImageNet normalize"
            ),
            "evaluation": "Resize((112,112)); ImageNet normalize",
        },
    }
    _json_dump(processed / "metadata.json", metadata)
    _json_dump(
        root / ACTIVE_CACHE_POINTER,
        {
            "processed_directory": processed.name,
            "instance_type": "bbox_crop",
            "bbox_source": bbox_source,
            "min_bbox_area": float(min_bbox_area),
            "metadata_file": "metadata.json",
        },
    )
    return metadata
