"""Prepare FLamby Fed-ISIC2019 as split-isolated feature-dependent LLP bags.

The public assignment boundary is deliberately label blind: :func:`assign_bags`
accepts only an embedding matrix and numerical bagging parameters.  Diagnosis
labels and medical-center identifiers are attached only after membership has
been fixed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .cct_preparation import build_feature_bags


DATASET_NAME = "FedISIC2019"
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_CLIENTS = 6
DEFAULT_ENCODER = "dinov2_vits14"
DEFAULT_MIN_BAG_SIZE = 16
DEFAULT_MAX_BAG_SIZE = 128
DEFAULT_TARGET_BAG_SIZE = 64
DEFAULT_SEED = 0
ACTIVE_CACHE_POINTER = "fed_isic2019_active_cache.json"
FEATURE_CACHE_SCHEMA = 1
BAG_CACHE_SCHEMA = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_OFFICIAL_INSTANCES = 23247
EXPECTED_OFFICIAL_SPLITS = {"train": 18597, "test": 4650}


@dataclass(frozen=True)
class FedISICSample:
    sample_index: int
    sample_id: str
    image_path: str
    target: int
    class_name: str
    client_id: int
    split: str
    official_fold2: str


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


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name, delete=False
    )
    handle.close()
    try:
        torch.save(value, handle.name)
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _write_instances(path: Path, samples: Sequence[FedISICSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FedISICSample.__dataclass_fields__)
    handle = tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=path.name, delete=False,
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


def _resolve_metadata_csv(root: Path, metadata_csv: str | os.PathLike[str] | None) -> Path:
    if metadata_csv is not None:
        path = Path(metadata_csv).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fed-ISIC2019 metadata CSV is missing: {path}")
        return path
    candidates = (
        root / "train_test_split",
        root / "dataset_creation_scripts" / "train_test_split",
        root / "flamby" / "train_test_split",
    )
    for path in candidates:
        if path.is_file():
            return path
    spec = importlib.util.find_spec("flamby")
    if spec is not None and spec.submodule_search_locations:
        for package_root in spec.submodule_search_locations:
            path = (
                Path(package_root) / "datasets" / "fed_isic2019"
                / "dataset_creation_scripts" / "train_test_split"
            )
            if path.is_file():
                return path
    raise FileNotFoundError(
        "FLamby's official Fed-ISIC2019 train_test_split was not found. Copy it "
        "under --data_root or pass --metadata_csv explicitly."
    )


def _resolve_image_root(root: Path, image_dir: str | os.PathLike[str] | None) -> Path:
    if image_dir is not None:
        path = Path(image_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Fed-ISIC2019 image directory is missing: {path}")
        return path
    candidates = (
        root / "ISIC_2019_Training_Input_preprocessed",
        root / "ISIC_2019_Training_Input",
        root / "images",
    )
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Fed-ISIC2019 images were not found under --data_root. Expected "
        "ISIC_2019_Training_Input_preprocessed/ (FLamby layout)."
    )


def parse_flamby_metadata(
    data_root: str | os.PathLike[str],
    *,
    metadata_csv: str | os.PathLike[str] | None = None,
    image_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[FedISICSample], dict[str, Any]]:
    """Parse and validate FLamby's fixed pooled train/test split and label order."""
    root = Path(data_root).expanduser().resolve()
    split_path = _resolve_metadata_csv(root, metadata_csv)
    images = _resolve_image_root(root, image_dir)
    with split_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Fed-ISIC2019 metadata is empty")
    required = {
        "image", *CLASS_NAMES, "UNK", "target", "center", "fold", "fold2"
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            "Fed-ISIC2019 metadata does not match FLamby's train_test_split; "
            f"missing={sorted(missing)}"
        )

    samples: list[FedISICSample] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row["image"]).strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Invalid or duplicate image ID at metadata row {row_number}")
        seen_ids.add(sample_id)
        try:
            target = int(row["target"])
            client_id = int(row["center"])
            one_hot = np.asarray([float(row[name]) for name in CLASS_NAMES])
            unknown = float(row["UNK"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric metadata at row {row_number}") from exc
        if target not in range(NUM_CLASSES):
            raise ValueError(f"Out-of-range target at row {row_number}: {target}")
        if client_id not in range(NUM_CLIENTS):
            raise ValueError(f"Out-of-range center at row {row_number}: {client_id}")
        if not np.all(np.isin(one_hot, [0.0, 1.0])) or not np.isclose(one_hot.sum(), 1.0):
            raise ValueError(f"The 8 official class columns are not one-hot at row {row_number}")
        if int(one_hot.argmax()) != target or not np.isclose(unknown, 0.0):
            raise ValueError(f"FLamby target/class columns disagree at row {row_number}")
        split = str(row["fold"]).strip()
        fold2 = str(row["fold2"]).strip()
        if split not in {"train", "test"} or fold2 != f"{split}_{client_id}":
            raise ValueError(f"Invalid FLamby fold/fold2 at row {row_number}")
        image_path = images / f"{sample_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Fed-ISIC2019 image is missing: {image_path}")
        try:
            relative_path = image_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("--image_dir must be inside --data_root for portable caches") from exc
        samples.append(
            FedISICSample(
                sample_index=len(samples),
                sample_id=sample_id,
                image_path=relative_path,
                target=target,
                class_name=CLASS_NAMES[target],
                client_id=client_id,
                split=split,
                official_fold2=fold2,
            )
        )

    split_counts = Counter(sample.split for sample in samples)
    client_counts = Counter(sample.client_id for sample in samples)
    source = {
        "dataset": DATASET_NAME,
        "metadata_format": "FLamby dataset_creation_scripts/train_test_split",
        "metadata_file": str(split_path),
        "image_directory": str(images.relative_to(root)),
        "instances": len(samples),
        "expected_official_instances": EXPECTED_OFFICIAL_INSTANCES,
        "matches_official_instance_count": len(samples) == EXPECTED_OFFICIAL_INSTANCES,
        "split_counts": {key: int(split_counts[key]) for key in ("train", "test")},
        "expected_official_split_counts": EXPECTED_OFFICIAL_SPLITS,
        "matches_official_split_counts": dict(split_counts) == EXPECTED_OFFICIAL_SPLITS,
        "client_counts": {str(key): int(client_counts[key]) for key in range(NUM_CLIENTS)},
        "official_split_has_validation": False,
        "validation_policy": "none; preserve FLamby's official pooled train/test split",
    }
    return samples, source


def _sample_digest(samples: Sequence[FedISICSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            (
                f"{sample.sample_index}\0{sample.sample_id}\0{sample.image_path}\0"
                f"{sample.target}\0{sample.client_id}\0{sample.split}\n"
            ).encode()
        )
    return digest.hexdigest()


def _feature_digest(features: torch.Tensor) -> str:
    values = features.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


class _FeatureDataset(Dataset):
    def __init__(
        self,
        root: Path,
        samples: Sequence[FedISICSample],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(self.root / sample.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return index, tensor


def dinov2_preprocess() -> transforms.Compose:
    """Deterministic DINOv2/ImageNet evaluation preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_dinov2_model(
    encoder: str = DEFAULT_ENCODER,
    *,
    dinov2_repo: str | os.PathLike[str] | None = None,
) -> torch.nn.Module:
    if dinov2_repo is None:
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", encoder, source="github", trust_repo=True
            )
        except TypeError:
            model = torch.hub.load("facebookresearch/dinov2", encoder, source="github")
    else:
        model = torch.hub.load(
            str(Path(dinov2_repo).expanduser().resolve()), encoder, source="local"
        )
    model.requires_grad_(False)
    model.eval()
    return model


def _encoder_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("x_norm_clstoken", "embedding", "features"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported DINOv2 output type: {type(output)!r}")


def _validate_feature_payload(
    payload: Any,
    samples: Sequence[FedISICSample],
    *,
    split: str,
    encoder: str,
) -> torch.Tensor:
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"{split} feature cache is not a versioned dictionary")
    metadata = payload["metadata"]
    expected = {
        "schema_version": FEATURE_CACHE_SCHEMA,
        "dataset": DATASET_NAME,
        "split": split,
        "encoder": encoder,
        "num_samples": len(samples),
        "sample_digest": _sample_digest(samples),
    }
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError(f"{split} feature cache metadata is stale or incompatible")
    features = payload.get("features")
    if not torch.is_tensor(features) or features.ndim != 2 or len(features) != len(samples):
        raise ValueError(f"{split} feature cache has an invalid feature tensor")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError(f"{split} feature cache must contain finite float32 embeddings")
    if int(metadata.get("embedding_dim", -1)) != int(features.shape[1]):
        raise ValueError(f"{split} feature cache embedding dimension is stale")
    if metadata.get("feature_digest") != _feature_digest(features):
        raise ValueError(f"{split} feature cache digest is stale")
    if not torch.allclose(torch.linalg.vector_norm(features, dim=1), torch.ones(len(features)), atol=2e-5):
        raise ValueError(f"{split} feature cache embeddings are not L2-normalized")
    expected_fields = {
        "labels": torch.tensor([sample.target for sample in samples], dtype=torch.long),
        "sample_indices": torch.tensor(
            [sample.sample_index for sample in samples], dtype=torch.long
        ),
        "client_ids": torch.tensor([sample.client_id for sample in samples], dtype=torch.long),
    }
    for key, expected_value in expected_fields.items():
        value = payload.get(key)
        if not torch.is_tensor(value) or not torch.equal(value, expected_value):
            raise ValueError(f"{split} feature cache field {key!r} is stale")
    if payload.get("sample_ids") != [sample.sample_id for sample in samples]:
        raise ValueError(f"{split} feature cache sample IDs are stale")
    if payload.get("image_paths") != [sample.image_path for sample in samples]:
        raise ValueError(f"{split} feature cache image paths are stale")
    return features


def extract_or_load_features(
    root: Path,
    samples: Sequence[FedISICSample],
    cache_path: Path,
    *,
    split: str,
    encoder: str = DEFAULT_ENCODER,
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 4,
    force: bool = False,
    dinov2_repo: str | os.PathLike[str] | None = None,
    model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if cache_path.is_file() and not force:
        payload = _torch_load(cache_path)
        features = _validate_feature_payload(
            payload, samples, split=split, encoder=encoder
        )
        print(f"Fed-ISIC2019 {split} feature cache hit: {cache_path}", flush=True)
        return features, dict(payload["metadata"])
    if batch_size <= 0:
        raise ValueError("feature extraction batch_size must be positive")
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto" else device
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction was requested but is unavailable")
    feature_encoder = (
        load_dinov2_model(encoder, dinov2_repo=dinov2_repo) if model is None else model
    )
    feature_encoder.requires_grad_(False)
    feature_encoder.eval()
    if any(parameter.requires_grad for parameter in feature_encoder.parameters()):
        raise RuntimeError("DINOv2 feature encoder must be completely frozen")
    feature_encoder.to(resolved_device)
    loader = DataLoader(
        _FeatureDataset(root, samples, dinov2_preprocess()),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=resolved_device.type == "cuda",
    )
    parts: list[torch.Tensor] = []
    seen: list[torch.Tensor] = []
    started = time.time()
    with torch.no_grad():
        for batch_number, (indices, images) in enumerate(loader, start=1):
            output = _encoder_output(
                feature_encoder(images.to(resolved_device, non_blocking=True))
            )
            if output.ndim != 2 or len(output) != len(images):
                raise ValueError(f"DINOv2 returned invalid shape {tuple(output.shape)}")
            parts.append(F.normalize(output.float(), p=2, dim=1).cpu())
            seen.append(indices.to(dtype=torch.long))
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(loader):
                completed = min(batch_number * int(batch_size), len(samples))
                rate = completed / max(time.time() - started, 1e-9)
                print(
                    f"Fed-ISIC2019 {split} DINOv2: {completed}/{len(samples)} "
                    f"({rate:.1f} images/s)",
                    flush=True,
                )
    features = torch.cat(parts, dim=0).contiguous()
    if not torch.equal(torch.cat(seen), torch.arange(len(samples))):
        raise RuntimeError("Feature extraction order does not match split-local sample order")
    metadata = {
        "schema_version": FEATURE_CACHE_SCHEMA,
        "dataset": DATASET_NAME,
        "split": split,
        "encoder": encoder,
        "pretrained": True,
        "frozen": True,
        "eval_mode": True,
        "no_grad": True,
        "preprocess": "resize_256_bicubic_center_crop_224_imagenet_normalize",
        "l2_normalized": True,
        "num_samples": len(samples),
        "embedding_dim": int(features.shape[1]),
        "sample_digest": _sample_digest(samples),
        "feature_digest": _feature_digest(features),
    }
    payload = {
        "metadata": metadata,
        "features": features,
        "labels": torch.tensor([sample.target for sample in samples], dtype=torch.long),
        "sample_indices": torch.tensor(
            [sample.sample_index for sample in samples], dtype=torch.long
        ),
        "sample_ids": [sample.sample_id for sample in samples],
        "client_ids": torch.tensor(
            [sample.client_id for sample in samples], dtype=torch.long
        ),
        "image_paths": [sample.image_path for sample in samples],
    }
    _torch_save(cache_path, payload)
    return features, metadata


def assign_bags(
    features: np.ndarray | torch.Tensor,
    *,
    min_bag_size: int = DEFAULT_MIN_BAG_SIZE,
    max_bag_size: int = DEFAULT_MAX_BAG_SIZE,
    target_bag_size: int = DEFAULT_TARGET_BAG_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[np.ndarray]:
    """Assign bags using features only; labels and centers are not accepted."""
    values = (
        features.detach().cpu().numpy() if torch.is_tensor(features)
        else np.asarray(features)
    )
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("features must be a non-empty finite two-dimensional matrix")
    minimum = int(min_bag_size)
    maximum = int(max_bag_size)
    target = int(target_bag_size)
    if not (1 <= minimum <= target <= maximum):
        raise ValueError("bag sizes must satisfy 1 <= min <= target <= max")
    if len(values) < minimum:
        raise ValueError(
            f"A split with {len(values)} samples cannot satisfy min_bag_size={minimum}"
        )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("features contain a zero vector and cannot be L2-normalized")
    normalized = (values / norms).astype(np.float32, copy=False)
    bags = build_feature_bags(
        normalized,
        target_avg_bag_size=target,
        min_bag_size_ratio=minimum / target,
        max_bag_size_ratio=maximum / target,
        seed=int(seed),
    )
    sizes = np.asarray([len(bag) for bag in bags], dtype=np.int64)
    if int(sizes.min()) < minimum or int(sizes.max()) > maximum:
        raise RuntimeError("Feature bag construction violated the hard size bounds")
    flattened = np.concatenate(bags)
    if len(flattened) != len(values) or not np.array_equal(
        np.sort(flattened), np.arange(len(values), dtype=np.int64)
    ):
        raise RuntimeError("Feature bag construction lost or duplicated samples")
    return bags


def compute_bag_proportions(
    bag_indices: Sequence[np.ndarray],
    labels: np.ndarray | torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Attach aggregate labels after feature-only membership is finalized."""
    values = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    counts: list[np.ndarray] = []
    proportions: list[np.ndarray] = []
    for indices in bag_indices:
        bag_counts = np.bincount(values[indices], minlength=int(num_classes)).astype(np.int64)
        if len(bag_counts) != int(num_classes):
            raise ValueError("labels contain an out-of-range class")
        counts.append(bag_counts)
        proportions.append(bag_counts.astype(np.float64) / len(indices))
    return counts, proportions


def _cosine_diagnostics(
    normalized_features: np.ndarray, bags: Sequence[np.ndarray]
) -> tuple[float, float]:
    values = np.asarray(normalized_features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / norms
    within_sum = 0.0
    within_pairs = 0
    for indices in bags:
        if len(indices) < 2:
            continue
        aggregate = values[indices].sum(axis=0)
        within_sum += (float(aggregate @ aggregate) - len(indices)) / 2.0
        within_pairs += len(indices) * (len(indices) - 1) // 2
    if within_pairs == 0 or len(values) < 2:
        raise ValueError("Cosine diagnostics require at least one pair")
    aggregate = values.sum(axis=0)
    all_pair_sum = (float(aggregate @ aggregate) - len(values)) / 2.0
    all_pairs = len(values) * (len(values) - 1) // 2
    return float(within_sum / within_pairs), float(all_pair_sum / all_pairs)


def _bag_payload(
    split: str,
    samples: Sequence[FedISICSample],
    features: torch.Tensor,
    *,
    encoder: str,
    min_bag_size: int,
    max_bag_size: int,
    target_bag_size: int,
    seed: int,
) -> dict[str, Any]:
    local_bags = assign_bags(
        features,
        min_bag_size=min_bag_size,
        max_bag_size=max_bag_size,
        target_bag_size=target_bag_size,
        seed=seed,
    )
    labels = np.asarray([sample.target for sample in samples], dtype=np.int64)
    counts, proportions = compute_bag_proportions(local_bags, labels)
    records: list[dict[str, Any]] = []
    for bag_number, (indices, bag_counts, proportion) in enumerate(
        zip(local_bags, counts, proportions)
    ):
        client_histogram = Counter(samples[int(index)].client_id for index in indices)
        dominant_client = min(
            client_histogram,
            key=lambda client: (-client_histogram[client], client),
        )
        records.append(
            {
                "bag_id": f"{split}_feature_{bag_number:05d}",
                "split": split,
                "local_indices": indices.tolist(),
                "instance_indices": [samples[int(index)].sample_index for index in indices],
                "bag_size": len(indices),
                "class_counts": bag_counts.tolist(),
                "class_proportions": proportion.tolist(),
                "client_histogram": {
                    str(client): int(client_histogram[client])
                    for client in sorted(client_histogram)
                },
                "num_unique_clients": len(client_histogram),
                "dominant_client": int(dominant_client),
            }
        )
    within, random_pair = _cosine_diagnostics(features.numpy(), local_bags)
    if within <= random_pair:
        raise RuntimeError(
            "Fed-ISIC2019 feature bags failed the similarity sanity check: "
            f"within={within:.6f} <= random_pair={random_pair:.6f}"
        )
    sizes = np.asarray([len(indices) for indices in local_bags], dtype=np.int64)
    proportion_array = np.asarray(proportions, dtype=np.float64)
    class_distribution = {
        name: {
            "min": float(proportion_array[:, class_index].min()),
            "median": float(np.median(proportion_array[:, class_index])),
            "mean": float(proportion_array[:, class_index].mean()),
            "max": float(proportion_array[:, class_index].max()),
        }
        for class_index, name in enumerate(CLASS_NAMES)
    }
    metadata = {
        "schema_version": BAG_CACHE_SCHEMA,
        "dataset": DATASET_NAME,
        "split": split,
        "encoder": encoder,
        "min_bag_size": int(min_bag_size),
        "max_bag_size": int(max_bag_size),
        "target_bag_size": int(target_bag_size),
        "seed": int(seed),
        "num_samples": len(samples),
        "num_bags": len(records),
        "sample_digest": _sample_digest(samples),
        "feature_digest": _feature_digest(features),
        "membership_inputs": ["l2_normalized_image_features"],
        "labels_used_for_membership": False,
        "client_ids_used_for_membership": False,
        "proportions_computed_after_membership": True,
        "bag_size": {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "mean": float(sizes.mean()),
            "max": int(sizes.max()),
            "histogram": {
                str(size): int(count) for size, count in sorted(Counter(sizes.tolist()).items())
            },
        },
        "class_proportion_distribution": class_distribution,
        "average_within_bag_cosine_similarity": within,
        "average_random_pair_cosine_similarity": random_pair,
        "within_exceeds_random": within > random_pair,
        "coverage_exactly_once": True,
    }
    return {"metadata": metadata, "bags": records}


def _validate_bag_payload(
    payload: Any,
    samples: Sequence[FedISICSample],
    features: torch.Tensor,
    *,
    split: str,
    encoder: str,
    min_bag_size: int,
    max_bag_size: int,
    target_bag_size: int,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"{split} bag cache is not a versioned dictionary")
    metadata = payload["metadata"]
    expected = {
        "schema_version": BAG_CACHE_SCHEMA,
        "dataset": DATASET_NAME,
        "split": split,
        "encoder": encoder,
        "min_bag_size": int(min_bag_size),
        "max_bag_size": int(max_bag_size),
        "target_bag_size": int(target_bag_size),
        "seed": int(seed),
        "num_samples": len(samples),
        "sample_digest": _sample_digest(samples),
        "feature_digest": _feature_digest(features),
        "membership_inputs": ["l2_normalized_image_features"],
    }
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError(f"{split} bag cache metadata is stale or incompatible")
    bags = payload.get("bags")
    if not isinstance(bags, list) or not bags:
        raise ValueError(f"{split} bag cache contains no bags")
    seen: list[int] = []
    cached_local_bags: list[np.ndarray] = []
    labels = np.asarray([sample.target for sample in samples], dtype=np.int64)
    for record in bags:
        local = np.asarray(record["local_indices"], dtype=np.int64)
        if len(local) < min_bag_size or len(local) > max_bag_size:
            raise ValueError(f"{record['bag_id']}: bag size violates cache bounds")
        if local.min() < 0 or local.max() >= len(samples):
            raise ValueError(f"{record['bag_id']}: local sample index is out of range")
        seen.extend(local.tolist())
        cached_local_bags.append(local)
        global_indices = [samples[int(index)].sample_index for index in local]
        if record.get("instance_indices") != global_indices:
            raise ValueError(f"{record['bag_id']}: global sample indices are stale")
        bag_counts = np.bincount(labels[local], minlength=NUM_CLASSES)
        if record.get("class_counts") != bag_counts.tolist():
            raise ValueError(f"{record['bag_id']}: class counts are stale")
        proportions = np.asarray(record.get("class_proportions"), dtype=np.float64)
        if len(proportions) != NUM_CLASSES or not np.isclose(proportions.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{record['bag_id']}: invalid class proportions")
        if not np.allclose(proportions, bag_counts / len(local), atol=1e-8):
            raise ValueError(f"{record['bag_id']}: class proportions are stale")
    if sorted(seen) != list(range(len(samples))):
        raise ValueError(f"{split} bag cache does not cover every sample exactly once")
    within, random_pair = _cosine_diagnostics(features.numpy(), cached_local_bags)
    if within <= random_pair:
        raise ValueError(f"{split} cached bags fail the feature-similarity sanity check")
    if not np.isclose(
        within, float(metadata.get("average_within_bag_cosine_similarity", math.nan)),
        atol=1e-10,
    ) or not np.isclose(
        random_pair, float(metadata.get("average_random_pair_cosine_similarity", math.nan)),
        atol=1e-10,
    ):
        raise ValueError(f"{split} cached cosine diagnostics are stale")
    return dict(metadata)


def _bag_namespace(
    encoder: str, min_bag_size: int, max_bag_size: int,
    target_bag_size: int, seed: int,
) -> str:
    return (
        f"feature_bags_{encoder}_min{int(min_bag_size)}_max{int(max_bag_size)}_"
        f"target{int(target_bag_size)}_seed{int(seed)}"
    )


def _print_summary(split: str, metadata: Mapping[str, Any]) -> None:
    sizes = metadata["bag_size"]
    print("Fed-ISIC2019", flush=True)
    print("--------------------------------", flush=True)
    print(f"split: {split}", flush=True)
    print(f"num instances: {metadata['num_samples']}", flush=True)
    print(f"num classes: {NUM_CLASSES}", flush=True)
    print(f"num bags: {metadata['num_bags']}", flush=True)
    print(
        "bag size: "
        f"min={sizes['min']} max={sizes['max']} mean={sizes['mean']:.1f} "
        f"median={sizes['median']:.1f}",
        flush=True,
    )
    print("feature encoder: DINOv2 ViT-S/14", flush=True)
    print(
        f"bagging: feature-dependent target={metadata['target_bag_size']} "
        f"seed={metadata['seed']}",
        flush=True,
    )
    print(
        "cosine similarity: "
        f"within={metadata['average_within_bag_cosine_similarity']:.6f} "
        f"random_pair={metadata['average_random_pair_cosine_similarity']:.6f}",
        flush=True,
    )


def prepare_fed_isic2019(
    data_root: str | os.PathLike[str],
    *,
    metadata_csv: str | os.PathLike[str] | None = None,
    image_dir: str | os.PathLike[str] | None = None,
    encoder: str = DEFAULT_ENCODER,
    device: str = "auto",
    extraction_batch_size: int = 64,
    num_workers: int = 4,
    min_bag_size: int = DEFAULT_MIN_BAG_SIZE,
    max_bag_size: int = DEFAULT_MAX_BAG_SIZE,
    target_bag_size: int = DEFAULT_TARGET_BAG_SIZE,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    dinov2_repo: str | os.PathLike[str] | None = None,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Extract/load features, construct/load bags, and publish a runtime cache."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Fed-ISIC2019 data root is missing: {root}")
    samples, source = parse_flamby_metadata(
        root, metadata_csv=metadata_csv, image_dir=image_dir
    )
    feature_root = root / "cache" / "fed_isic2019" / encoder
    feature_root.mkdir(parents=True, exist_ok=True)
    _write_instances(feature_root / "instances.csv", samples)
    bag_root = feature_root / _bag_namespace(
        encoder, min_bag_size, max_bag_size, target_bag_size, seed
    )
    bag_root.mkdir(parents=True, exist_ok=True)

    feature_manifests: dict[str, Any] = {}
    bag_manifests: dict[str, Any] = {}
    for split in ("train", "test"):
        split_samples = [sample for sample in samples if sample.split == split]
        if not split_samples:
            raise ValueError(f"Fed-ISIC2019 official split {split!r} is empty")
        feature_path = feature_root / f"{split}_features.pt"
        features, feature_metadata = extract_or_load_features(
            root,
            split_samples,
            feature_path,
            split=split,
            encoder=encoder,
            device=device,
            batch_size=extraction_batch_size,
            num_workers=num_workers,
            force=force,
            dinov2_repo=dinov2_repo,
            model=model,
        )
        feature_manifests[split] = {**feature_metadata, "file": str(feature_path.relative_to(root))}
        bag_path = bag_root / f"{split}_featurebags.pt"
        if bag_path.is_file() and not force:
            payload = _torch_load(bag_path)
            bag_metadata = _validate_bag_payload(
                payload,
                split_samples,
                features,
                split=split,
                encoder=encoder,
                min_bag_size=min_bag_size,
                max_bag_size=max_bag_size,
                target_bag_size=target_bag_size,
                seed=seed,
            )
            print(f"Fed-ISIC2019 {split} feature-bag cache hit: {bag_path}", flush=True)
        else:
            payload = _bag_payload(
                split,
                split_samples,
                features,
                encoder=encoder,
                min_bag_size=min_bag_size,
                max_bag_size=max_bag_size,
                target_bag_size=target_bag_size,
                seed=seed,
            )
            _torch_save(bag_path, payload)
            bag_metadata = dict(payload["metadata"])
        bag_manifests[split] = {**bag_metadata, "file": str(bag_path.relative_to(root))}
        _print_summary(split, bag_metadata)

    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_variant": "FLamby pooled fixed train/test split",
        "instances": len(samples),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "num_clients": NUM_CLIENTS,
        "official_validation_split": False,
        "source": source,
        "instances_file": str((feature_root / "instances.csv").relative_to(root)),
        "feature_manifests": feature_manifests,
        "bag_manifest": {
            "encoder": encoder,
            "min_bag_size": int(min_bag_size),
            "max_bag_size": int(max_bag_size),
            "target_bag_size": int(target_bag_size),
            "seed": int(seed),
            "split_isolation": True,
            "membership_inputs": ["l2_normalized_image_features"],
            "labels_used_for_membership": False,
            "client_ids_used_for_membership": False,
            "splits": bag_manifests,
        },
        "classifier_preprocessing": {
            "input_resolution": 224,
            "feature_encoder_is_downstream_model": False,
            "training_input": "original image with torchvision augmentation",
            "evaluation": "resize_256_bicubic_center_crop_224_imagenet_normalize",
        },
    }
    _json_dump(bag_root / "metadata.json", metadata)
    _json_dump(
        root / ACTIVE_CACHE_POINTER,
        {
            "dataset": DATASET_NAME,
            "metadata_file": str((bag_root / "metadata.json").relative_to(root)),
            "bag_directory": str(bag_root.relative_to(root)),
            "encoder": encoder,
            "min_bag_size": int(min_bag_size),
            "max_bag_size": int(max_bag_size),
            "target_bag_size": int(target_bag_size),
            "seed": int(seed),
        },
    )
    return metadata
