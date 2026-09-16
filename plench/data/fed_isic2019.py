"""Fed-ISIC2019 feature-bag runtime adapter for the existing PLeNCH pipeline."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .cct import (
    CCTInstanceBatchSampler,
    evaluate_cct,
    update_cct_algorithm,
)
from .fed_isic2019_preparation import (
    ACTIVE_CACHE_POINTER,
    CLASS_NAMES,
    DATASET_NAME,
    DEFAULT_ENCODER,
    DEFAULT_MAX_BAG_SIZE,
    DEFAULT_MIN_BAG_SIZE,
    DEFAULT_SEED,
    DEFAULT_TARGET_BAG_SIZE,
    NUM_CLASSES,
    NUM_CLIENTS,
)
from .ku_optofil_pbc import SUPPORTED_KU_METHODS


DATASET_ALIASES = {
    "fedisic2019": DATASET_NAME,
    "fed_isic2019": DATASET_NAME,
    "fed-isic2019": DATASET_NAME,
    "isic2019": DATASET_NAME,
    "fed-isic": DATASET_NAME,
}
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_FED_ISIC_METHODS = set(SUPPORTED_KU_METHODS)


def canonical_fed_isic2019_dataset(dataset: str) -> str:
    value = str(dataset).strip()
    if value == DATASET_NAME:
        return value
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown Fed-ISIC2019 alias: {dataset!r}") from exc


def is_fed_isic2019_dataset(dataset: str) -> bool:
    try:
        canonical_fed_isic2019_dataset(dataset)
    except ValueError:
        return False
    return True


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().resolve()
    for candidate in (
        path,
        path / "fed_isic2019",
        path / "Fed-ISIC2019",
        path / "isic2019",
    ):
        if (candidate / ACTIVE_CACHE_POINTER).is_file():
            return candidate
    return path


def _safe_relative(root: Path, value: Any, description: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Fed-ISIC2019 {description} is not a safe relative path")
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError(f"Fed-ISIC2019 {description} escapes the dataset root")
    return resolved


@dataclass(frozen=True)
class FedISICBag:
    bag_id: str
    indices: np.ndarray
    proportions: np.ndarray
    class_counts: np.ndarray
    split: str
    client_histogram: dict[int, int]
    num_unique_clients: int
    dominant_client: int


@dataclass
class FedISICBundle:
    dataset_root: Path
    sample_indices: np.ndarray
    sample_ids: np.ndarray
    image_paths: np.ndarray
    targets: np.ndarray
    client_ids: np.ndarray
    splits: np.ndarray
    official_folds: np.ndarray
    bags: list[FedISICBag]
    metadata: dict[str, Any]

    @property
    def class_names(self) -> list[str]:
        return list(CLASS_NAMES)

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return INPUT_SHAPE

    @property
    def has_instance_labels(self) -> bool:
        return True

    def bags_for_split(self, split: str) -> list[FedISICBag]:
        result = [bag for bag in self.bags if bag.split == split]
        if not result:
            raise ValueError(f"Fed-ISIC2019 split {split!r} contains no feature bags")
        return result


def load_fed_isic2019_bundle(root: str | os.PathLike[str]) -> FedISICBundle:
    dataset_root = _resolve_root(root)
    pointer_path = dataset_root / ACTIVE_CACHE_POINTER
    if not pointer_path.is_file():
        raise FileNotFoundError(
            "Fed-ISIC2019 prepared-cache pointer is missing. Run "
            "python -m plench.scripts.prepare_fed_isic2019 first."
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("dataset") != DATASET_NAME:
        raise ValueError("Fed-ISIC2019 cache pointer has the wrong dataset name")
    metadata_path = _safe_relative(
        dataset_root, pointer.get("metadata_file"), "metadata_file"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Fed-ISIC2019 metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("class_names") != CLASS_NAMES or metadata.get("num_classes") != NUM_CLASSES:
        raise ValueError("Fed-ISIC2019 class order differs from the FLamby contract")
    if metadata.get("num_clients") != NUM_CLIENTS:
        raise ValueError("Fed-ISIC2019 cache must preserve six FLamby centers")
    bag_manifest = metadata.get("bag_manifest", {})
    if bag_manifest.get("membership_inputs") != ["l2_normalized_image_features"]:
        raise ValueError("Fed-ISIC2019 bag cache is not feature-only")
    if bag_manifest.get("labels_used_for_membership") is not False:
        raise ValueError("Fed-ISIC2019 bag cache does not prove label-blind membership")
    if bag_manifest.get("client_ids_used_for_membership") is not False:
        raise ValueError("Fed-ISIC2019 default bags must not use client IDs")
    if bag_manifest.get("split_isolation") is not True:
        raise ValueError("Fed-ISIC2019 bag cache does not declare split isolation")

    instances_path = _safe_relative(
        dataset_root, metadata.get("instances_file"), "instances_file"
    )
    with instances_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Fed-ISIC2019 instance manifest is empty")
    sample_indices = np.asarray(
        [int(row["sample_index"]) for row in rows], dtype=np.int64
    )
    if not np.array_equal(sample_indices, np.arange(len(rows), dtype=np.int64)):
        raise ValueError("Fed-ISIC2019 sample indices must be contiguous and row-aligned")
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=object)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("Fed-ISIC2019 sample IDs must be unique")
    image_paths = np.asarray(
        [_safe_relative(dataset_root, row["image_path"], "image_path") for row in rows],
        dtype=object,
    )
    missing_images = [str(path) for path in image_paths if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Fed-ISIC2019 cache references {len(missing_images)} missing images: "
            f"{missing_images[:5]}"
        )
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    client_ids = np.asarray([int(row["client_id"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows], dtype=object)
    official_folds = np.asarray([row["official_fold2"] for row in rows], dtype=object)
    if targets.min() < 0 or targets.max() >= NUM_CLASSES:
        raise ValueError("Fed-ISIC2019 contains an out-of-range target")
    if client_ids.min() < 0 or client_ids.max() >= NUM_CLIENTS:
        raise ValueError("Fed-ISIC2019 contains an out-of-range client ID")
    if set(splits.tolist()) != {"train", "test"}:
        raise ValueError("Fed-ISIC2019 must preserve FLamby's train/test splits")
    if any(fold != f"{split}_{client}" for fold, split, client in zip(official_folds, splits, client_ids)):
        raise ValueError("Fed-ISIC2019 fold2/client metadata is inconsistent")

    bags: list[FedISICBag] = []
    seen: set[int] = set()
    minimum = int(bag_manifest["min_bag_size"])
    maximum = int(bag_manifest["max_bag_size"])
    split_payloads = bag_manifest.get("splits", {})
    for split in ("train", "test"):
        split_manifest = split_payloads.get(split, {})
        bag_path = _safe_relative(dataset_root, split_manifest.get("file"), f"{split} bags")
        payload = _torch_load(bag_path)
        if not isinstance(payload, dict) or payload.get("metadata") != {
            key: value for key, value in split_manifest.items() if key != "file"
        }:
            raise ValueError(f"Fed-ISIC2019 {split} bag file and metadata disagree")
        raw_bags = payload.get("bags")
        if not isinstance(raw_bags, list) or not raw_bags:
            raise ValueError(f"Fed-ISIC2019 {split} bag file is empty")
        for raw in raw_bags:
            indices = np.asarray(raw["instance_indices"], dtype=np.int64)
            if not (minimum <= len(indices) <= maximum):
                raise ValueError(f"{raw['bag_id']}: bag size is outside configured bounds")
            if indices.min() < 0 or indices.max() >= len(rows):
                raise ValueError(f"{raw['bag_id']}: instance index is out of range")
            if set(splits[indices].tolist()) != {split}:
                raise ValueError(f"{raw['bag_id']}: bag crosses official splits")
            overlap = seen.intersection(int(index) for index in indices)
            if overlap:
                raise ValueError(
                    f"Fed-ISIC2019 samples occur in multiple bags: {sorted(overlap)[:5]}"
                )
            seen.update(int(index) for index in indices)
            counts = np.asarray(raw["class_counts"], dtype=np.int64)
            proportions = np.asarray(raw["class_proportions"], dtype=np.float32)
            hidden_counts = np.bincount(targets[indices], minlength=NUM_CLASSES)
            if len(counts) != NUM_CLASSES or not np.array_equal(counts, hidden_counts):
                raise ValueError(f"{raw['bag_id']}: cached class counts are stale")
            if len(proportions) != NUM_CLASSES or not np.allclose(
                proportions, counts / len(indices), atol=1e-7
            ):
                raise ValueError(f"{raw['bag_id']}: cached class proportions are stale")
            if not np.isclose(proportions.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{raw['bag_id']}: class proportions do not sum to one")
            histogram = Counter(int(value) for value in client_ids[indices])
            cached_histogram = {int(key): int(value) for key, value in raw["client_histogram"].items()}
            if histogram != Counter(cached_histogram):
                raise ValueError(f"{raw['bag_id']}: cached client histogram is stale")
            dominant = min(histogram, key=lambda client: (-histogram[client], client))
            if int(raw["dominant_client"]) != dominant:
                raise ValueError(f"{raw['bag_id']}: cached dominant client is stale")
            bags.append(
                FedISICBag(
                    bag_id=str(raw["bag_id"]),
                    indices=indices,
                    proportions=proportions,
                    class_counts=counts,
                    split=split,
                    client_histogram=dict(sorted(histogram.items())),
                    num_unique_clients=len(histogram),
                    dominant_client=dominant,
                )
            )
    if seen != set(range(len(rows))):
        missing = sorted(set(range(len(rows))) - seen)
        raise ValueError(
            "Every Fed-ISIC2019 sample must occur in exactly one bag; "
            f"missing={missing[:5]}"
        )
    return FedISICBundle(
        dataset_root=dataset_root,
        sample_indices=sample_indices,
        sample_ids=sample_ids,
        image_paths=image_paths,
        targets=targets,
        client_ids=client_ids,
        splits=splits,
        official_folds=official_folds,
        bags=sorted(bags, key=lambda bag: bag.bag_id),
        metadata=metadata,
    )


def _image_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(50),
                transforms.ColorJitter(brightness=0.15, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.5, 2.0)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FedISIC2019FeatureBagDataset(Dataset):
    """One item is one immutable, variable-size, feature-defined image bag."""

    def __init__(
        self,
        bundle: FedISICBundle,
        split: str,
        *,
        paired_views: bool = False,
        expose_instance_labels: bool | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("Fed-ISIC2019 split must be train or test")
        self.bundle = bundle
        self.split = split
        self.bags = bundle.bags_for_split(split)
        self.transform = _image_transform(split)
        self.paired_views = bool(paired_views and split == "train")
        self.strong_transform = _image_transform("train") if self.paired_views else None
        self.expose_instance_labels = (
            split != "train" if expose_instance_labels is None
            else bool(expose_instance_labels)
        )
        if split == "train" and self.expose_instance_labels:
            raise ValueError("Fed-ISIC2019 training bags cannot expose instance labels")
        self.label_prob = [bag.proportions.tolist() for bag in self.bags]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, item: int) -> dict[str, Any]:
        bag = self.bags[item]
        images: list[torch.Tensor] = []
        strong_images: list[torch.Tensor] = []
        for index in bag.indices:
            with Image.open(Path(self.bundle.image_paths[int(index)])) as image:
                rgb = image.convert("RGB")
                images.append(self.transform(rgb))
                if self.strong_transform is not None:
                    strong_images.append(self.strong_transform(rgb))
        result: dict[str, Any] = {
            "x": torch.stack(images),
            "proportion": torch.from_numpy(bag.proportions.copy()),
            "bag_id": bag.bag_id,
            "sample_indices": self.bundle.sample_indices[bag.indices].tolist(),
            "sample_ids": self.bundle.sample_ids[bag.indices].tolist(),
            "image_paths": [str(self.bundle.image_paths[index]) for index in bag.indices],
            "client_ids": self.bundle.client_ids[bag.indices].tolist(),
            "client_histogram": dict(bag.client_histogram),
            "num_unique_clients": bag.num_unique_clients,
            "dominant_client": bag.dominant_client,
        }
        if strong_images:
            result["x_strong"] = torch.stack(strong_images)
        if self.expose_instance_labels:
            result["instance_labels"] = torch.from_numpy(
                self.bundle.targets[bag.indices].copy()
            )
            result["class_counts"] = torch.from_numpy(bag.class_counts.copy())
        return result


def collate_fed_isic2019_bags(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Fed-ISIC2019 batch")
    sizes = torch.tensor([len(item["x"]) for item in items], dtype=torch.long)
    result: dict[str, Any] = {
        "x": torch.cat([item["x"] for item in items], dim=0),
        "bag_index": torch.repeat_interleave(
            torch.arange(len(items), dtype=torch.long), sizes
        ),
        "bag_sizes": sizes,
        "proportion": torch.stack([item["proportion"] for item in items]),
        "bag_ids": [str(item["bag_id"]) for item in items],
        "sample_indices": [list(item["sample_indices"]) for item in items],
        "sample_ids": [list(item["sample_ids"]) for item in items],
        "image_paths": [list(item["image_paths"]) for item in items],
        "client_ids": [list(item["client_ids"]) for item in items],
        "client_histograms": [dict(item["client_histogram"]) for item in items],
        "num_unique_clients": [int(item["num_unique_clients"]) for item in items],
        "dominant_clients": [int(item["dominant_client"]) for item in items],
        "instance_weights": torch.ones(int(sizes.sum()), dtype=torch.float32),
    }
    if any("x_strong" in item for item in items):
        if not all("x_strong" in item for item in items):
            raise ValueError("Fed-ISIC2019 batch mixes paired and single views")
        result["x_strong"] = torch.cat([item["x_strong"] for item in items])
    if any("instance_labels" in item for item in items):
        if not all("instance_labels" in item for item in items):
            raise ValueError("Fed-ISIC2019 batch mixes hidden-label and label-free bags")
        result["instance_labels"] = torch.cat(
            [item["instance_labels"] for item in items]
        )
        result["inferred_class_counts"] = torch.stack(
            [item["class_counts"] for item in items]
        )
    return result


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _print_split(prefix: str, dataset: FedISIC2019FeatureBagDataset) -> None:
    sizes = np.asarray([len(bag.indices) for bag in dataset.bags], dtype=np.int64)
    print(
        f"{prefix}: bags={len(sizes)} instances={int(sizes.sum())} "
        f"size min/median/mean/max={int(sizes.min())}/{float(np.median(sizes)):.1f}/"
        f"{float(sizes.mean()):.2f}/{int(sizes.max())}",
        flush=True,
    )


def build_fed_isic2019_loaders(
    root: str,
    batch_size: int,
    *,
    seed: int = 0,
    num_workers: int = 0,
    paired_views: bool = False,
):
    bundle = load_fed_isic2019_bundle(root)
    train_dataset = FedISIC2019FeatureBagDataset(
        bundle, "train", paired_views=paired_views, expose_instance_labels=False
    )
    _print_split("Fed-ISIC2019 train", train_dataset)
    batch_sampler = CCTInstanceBatchSampler(
        [len(bag.indices) for bag in train_dataset.bags],
        target_instances=max(1, int(batch_size)),
        seed=int(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_fed_isic2019_bags,
    )
    return train_loader, None, bundle, bundle.input_shape


def build_fed_isic2019_eval_loader(
    root: str,
    *,
    split: str = "test",
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    bundle = load_fed_isic2019_bundle(root)
    dataset = FedISIC2019FeatureBagDataset(bundle, split, expose_instance_labels=True)
    _print_split(f"Fed-ISIC2019 {split}", dataset)
    if int(batch_size) != 1:
        print("Fed-ISIC2019 evaluation uses one complete bag per loader batch")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_fed_isic2019_bags,
    )


def update_fed_isic2019_algorithm(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, Any],
    device: str | torch.device,
    *,
    forward_chunk_size: int = 32,
    iteration: int = 0,
    softmatch_state: tuple[torch.Tensor, float, float] | None = None,
) -> Any:
    if algorithm_name not in SUPPORTED_FED_ISIC_METHODS:
        raise NotImplementedError(
            f"{algorithm_name} has no verified Fed-ISIC2019 variable-bag adapter"
        )
    if "instance_labels" in batch or "inferred_class_counts" in batch:
        raise ValueError("Fed-ISIC2019 training batches must hide instance labels")
    return update_cct_algorithm(
        algorithm,
        algorithm_name,
        batch,
        device,
        forward_chunk_size=forward_chunk_size,
        iteration=iteration,
        softmatch_state=softmatch_state,
    )


@torch.no_grad()
def evaluate_fed_isic2019(
    algorithm: Any,
    loader: DataLoader,
    device: str | torch.device,
    *,
    forward_chunk_size: int = 32,
) -> dict[str, Any]:
    return evaluate_cct(
        algorithm,
        loader,
        device,
        class_names=CLASS_NAMES,
        forward_chunk_size=forward_chunk_size,
    )


__all__ = [
    "CLASS_NAMES",
    "DATASET_NAME",
    "DEFAULT_ENCODER",
    "DEFAULT_MAX_BAG_SIZE",
    "DEFAULT_MIN_BAG_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_TARGET_BAG_SIZE",
    "FedISIC2019FeatureBagDataset",
    "build_fed_isic2019_eval_loader",
    "build_fed_isic2019_loaders",
    "canonical_fed_isic2019_dataset",
    "collate_fed_isic2019_bags",
    "evaluate_fed_isic2019",
    "is_fed_isic2019_dataset",
    "load_fed_isic2019_bundle",
    "update_fed_isic2019_algorithm",
]
