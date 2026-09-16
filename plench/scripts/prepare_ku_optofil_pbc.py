"""Download and prepare KU-Optofil PBC as natural patient-level LLP bags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from PIL import Image


ZENODO_RECORD = "17333317"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
DOWNLOADS = {
    "dataset.zip": f"{ZENODO_BASE}/dataset.zip/content",
    "metadata.csv": f"{ZENODO_BASE}/metadata.csv/content",
    "metadata_with_patient_level_splits.csv": (
        f"{ZENODO_BASE}/metadata_with_patient_level_splits.csv/content"
    ),
}
CLASS_NAMES = [
    "band_neutrophil",
    "basophil",
    "blast",
    "eosinophil",
    "erythroblast",
    "giant_platelet",
    "lymphocyte",
    "metamyelocyte",
    "monocyte",
    "myelocyte",
    "platelet_cluster",
    "reactive_lymphocyte",
    "segmented_neutrophil",
]
PREPROCESSING_VERSION = "ku-optofil-pbc-natural-patient-bags-v1"
PAPER_REPORTED_INSTANCES = 31_489


def _normalise_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _detect_column(columns: Iterable[str], aliases: Iterable[str], role: str) -> str:
    lookup = {_normalise_column(column): column for column in columns}
    for alias in aliases:
        match = lookup.get(_normalise_column(alias))
        if match is not None:
            return match
    raise ValueError(
        f"Cannot detect {role} column. Metadata columns: {list(columns)!r}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "PLeNCH/1.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in dataset.zip: {member.filename!r}")
        handle.extractall(destination)


def _canonical_split(value: str) -> str:
    lookup = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    try:
        return lookup[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown patient split value: {value!r}") from exc


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot describe an empty split")
    ordered = sorted(values)
    return {
        "min": int(ordered[0]),
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "max": int(ordered[-1]),
    }


def prepare(data_root: str | Path, *, download: bool = True) -> dict:
    root = Path(data_root).expanduser().resolve()
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    for filename, url in DOWNLOADS.items():
        destination = raw / filename
        if not destination.is_file():
            if not download:
                raise FileNotFoundError(f"Missing required source file: {destination}")
            _download(url, destination)

    archive = raw / "dataset.zip"
    with zipfile.ZipFile(archive) as handle:
        archive_images = sorted(
            member.filename
            for member in handle.infolist()
            if not member.is_dir() and member.filename.lower().endswith((".jpg", ".jpeg"))
        )
    image_root = raw / "dataset"
    existing_images = list(image_root.rglob("*.jpg")) if image_root.is_dir() else []
    if len(existing_images) != len(archive_images):
        print(
            f"Extracting {archive.name}: archive images={len(archive_images)}, "
            f"existing images={len(existing_images)}"
        )
        _safe_extract(archive, raw)

    patient_metadata = raw / "metadata_with_patient_level_splits.csv"
    with patient_metadata.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        source_columns = list(reader.fieldnames or [])
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("Patient-level metadata is empty")
    print(f"Patient metadata columns: {source_columns}")

    patient_column = _detect_column(
        source_columns, ["patient_id", "patient", "patientid"], "patient ID"
    )
    filename_column = _detect_column(
        source_columns, ["image_name", "filename", "image", "file"], "image filename"
    )
    class_column = _detect_column(
        source_columns, ["cell_type", "class", "label", "cellclass"], "cell class"
    )
    split_column = _detect_column(
        source_columns, ["split", "patient_split", "patientlevelsplit"], "patient split"
    )
    path_column = _detect_column(
        source_columns, ["path", "directory", "folder"], "source image path"
    )

    image_paths = [path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}]
    image_dimensions: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    image_decode_errors: list[dict[str, str]] = []
    nonstandard_dimensions: list[dict[str, str | list[int]]] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                size = (int(image.size[0]), int(image.size[1]))
                image_dimensions[f"{size[0]}x{size[1]}"] += 1
                image_modes[str(image.mode)] += 1
                image.verify()
            if size != (368, 368):
                nonstandard_dimensions.append(
                    {"image_name": image_path.name, "dimensions": list(size)}
                )
        except Exception as exc:
            image_decode_errors.append(
                {"image_name": image_path.name, "error": str(exc)}
            )
    if image_decode_errors:
        raise ValueError(
            f"{len(image_decode_errors)} KU-Optofil images cannot be decoded: "
            f"{image_decode_errors[:5]}"
        )
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        by_name[path.name].append(path)

    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    instances: list[dict[str, str | int]] = []
    missing_images: list[str] = []
    patient_split: dict[str, str] = {}
    seen_filenames: set[str] = set()
    for index, row in enumerate(source_rows):
        patient_id = str(row[patient_column]).strip()
        filename = str(row[filename_column]).strip()
        class_name = str(row[class_column]).strip().lower().replace(" ", "_")
        split = _canonical_split(row[split_column])
        if not patient_id or not filename:
            raise ValueError(f"Empty patient/image identifier at metadata row {index + 2}")
        if class_name not in class_to_index:
            raise ValueError(f"Unknown KU-Optofil class {class_name!r}")
        if filename in seen_filenames:
            raise ValueError(f"Duplicate image filename in metadata: {filename}")
        seen_filenames.add(filename)
        previous = patient_split.setdefault(patient_id, split)
        if previous != split:
            raise ValueError(
                f"Patient leakage in official metadata: {patient_id} has {previous} and {split}"
            )
        matches = by_name.get(filename, [])
        if len(matches) != 1:
            missing_images.append(filename)
            continue
        image_path = matches[0]
        instances.append(
            {
                "instance_index": index,
                "patient_id": patient_id,
                "image_name": filename,
                "relative_path": image_path.relative_to(root).as_posix(),
                "source_path_field": str(row[path_column]),
                "class_name": class_name,
                "class_index": class_to_index[class_name],
                "split": split,
            }
        )
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} metadata images are missing or ambiguous: {missing_images[:10]}"
        )

    metadata_filenames = {str(row[filename_column]).strip() for row in source_rows}
    untracked_images = sorted(path.name for path in image_paths if path.name not in metadata_filenames)
    grouped: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in instances:
        grouped[str(row["patient_id"])].append(row)

    bags: list[dict] = []
    bag_stat_rows: list[dict[str, str | int | float]] = []
    for patient_id in sorted(grouped):
        rows = grouped[patient_id]
        counts = [0] * len(CLASS_NAMES)
        indices: list[int] = []
        for row in rows:
            counts[int(row["class_index"])] += 1
            indices.append(int(row["instance_index"]))
        total = len(rows)
        proportions = [count / total for count in counts]
        if abs(sum(proportions) - 1.0) >= 1e-12:
            raise AssertionError(f"Invalid bag proportions for patient {patient_id}")
        split = str(rows[0]["split"])
        bag = {
            "bag_id": patient_id,
            "patient_id": patient_id,
            "split": split,
            "n_instances": total,
            "instance_indices": indices,
            "class_counts": counts,
            "class_proportions": proportions,
        }
        bags.append(bag)
        stat_row: dict[str, str | int | float] = {
            "patient_id": patient_id,
            "split": split,
            "bag_size": total,
            "classes_present": sum(count > 0 for count in counts),
        }
        for class_name, count, proportion in zip(CLASS_NAMES, counts, proportions):
            stat_row[f"n_{class_name}"] = count
            stat_row[f"p_{class_name}"] = proportion
        bag_stat_rows.append(stat_row)

    split_patients = {
        split: {bag["patient_id"] for bag in bags if bag["split"] == split}
        for split in ("train", "val", "test")
    }
    assert split_patients["train"].isdisjoint(split_patients["val"])
    assert split_patients["train"].isdisjoint(split_patients["test"])
    assert split_patients["val"].isdisjoint(split_patients["test"])
    split_images = {
        split: {str(row["image_name"]) for row in instances if row["split"] == split}
        for split in ("train", "val", "test")
    }
    assert split_images["train"].isdisjoint(split_images["val"])
    assert split_images["train"].isdisjoint(split_images["test"])
    assert split_images["val"].isdisjoint(split_images["test"])

    class_counts = Counter(str(row["class_name"]) for row in instances)
    patients_per_class = {
        class_name: len(
            {str(row["patient_id"]) for row in instances if row["class_name"] == class_name}
        )
        for class_name in CLASS_NAMES
    }
    split_statistics: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        split_bags = [bag for bag in bags if bag["split"] == split]
        sizes = [int(bag["n_instances"]) for bag in split_bags]
        split_statistics[split] = {
            "patients": len(split_bags),
            "instances": sum(sizes),
            "bag_size": _distribution(sizes),
        }
    statistics = {
        "dataset": "KU-Optofil PBC",
        "classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "instances": len(instances),
        "patients": len(bags),
        "paper_reported_instances": PAPER_REPORTED_INSTANCES,
        "archive_images": len(image_paths),
        "untracked_archive_images": untracked_images,
        "metadata_missing_images": missing_images,
        "image_dimensions": dict(sorted(image_dimensions.items())),
        "image_modes": dict(sorted(image_modes.items())),
        "nonstandard_image_dimensions": nonstandard_dimensions,
        "image_decode_errors": image_decode_errors,
        "class_counts": {name: int(class_counts[name]) for name in CLASS_NAMES},
        "patients_per_class": patients_per_class,
        "classes_per_patient": _distribution(
            [sum(count > 0 for count in bag["class_counts"]) for bag in bags]
        ),
        "bag_size": _distribution([int(bag["n_instances"]) for bag in bags]),
        "splits": split_statistics,
        "patient_leakage": False,
        "image_leakage": False,
    }
    source_hashes = {
        filename: _sha256(raw / filename)
        for filename in DOWNLOADS
    }
    metadata = {
        "dataset": "ku_optofil_pbc",
        "preprocessing_version": PREPROCESSING_VERSION,
        "zenodo_record": ZENODO_RECORD,
        "doi": "10.5281/zenodo.17333317",
        "source_files": {
            filename: {"relative_path": f"raw/{filename}", "sha256": source_hashes[filename]}
            for filename in DOWNLOADS
        },
        "source_metadata_columns": source_columns,
        "detected_columns": {
            "patient_id": patient_column,
            "image_name": filename_column,
            "source_path": path_column,
            "cell_type": class_column,
            "patient_split": split_column,
        },
        "patient_split_file": "metadata_with_patient_level_splits.csv",
        "split_mapping": {"validation": "val"},
        "natural_bags": True,
        "bag_structure": "patient -> cells",
        "has_instance_labels": True,
        "use_instance_labels_for_training": False,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "image_channels": 3,
        "image_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "statistics": statistics,
    }

    instance_fields = list(instances[0])
    with (processed / "instances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=instance_fields)
        writer.writeheader()
        writer.writerows(instances)
    (processed / "bags.json").write_text(
        json.dumps(bags, indent=2, sort_keys=True), encoding="utf-8"
    )
    (processed / "class_mapping.json").write_text(
        json.dumps(class_to_index, indent=2, sort_keys=True), encoding="utf-8"
    )
    (processed / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (processed / "dataset_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (processed / "bag_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bag_stat_rows[0]))
        writer.writeheader()
        writer.writerows(bag_stat_rows)

    print("Dataset: KU-Optofil PBC")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Instances: {len(instances)}; patients/bags: {len(bags)}")
    for split in ("train", "val", "test"):
        current = split_statistics[split]
        bag_size = current["bag_size"]
        print(
            f"{split}: patients={current['patients']} cells={current['instances']} "
            f"bag min/median/mean/max={bag_size['min']}/{bag_size['median']}/"
            f"{bag_size['mean']:.2f}/{bag_size['max']}"
        )
    for index, name in enumerate(CLASS_NAMES):
        print(
            f"class {index:2d}: {name:24s} cells={class_counts[name]:5d} "
            f"patients={patients_per_class[name]:3d}"
        )
    if untracked_images:
        print(f"Untracked archive images ({len(untracked_images)}): {untracked_images}")
    if nonstandard_dimensions:
        print(
            f"Non-368x368 archive images ({len(nonstandard_dimensions)}): "
            f"{nonstandard_dimensions}"
        )
    print("Patient/image split leakage: none")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="plench/data/ku_optofil_pbc"
    )
    parser.add_argument(
        "--no-download", action="store_true", help="Require sources to exist locally"
    )
    args = parser.parse_args()
    prepare(args.data_root, download=not args.no_download)


if __name__ == "__main__":
    main()
