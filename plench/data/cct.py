"""Caltech Camera Traps feature-bag LLP runtime adapter.

The offline preparation pipeline lives in :mod:`plench.data.cct_preparation`.
This module only loads its immutable feature-defined bag cache, applies image
transforms, and exposes bag proportions to PLeNCH.  Hidden instance labels are
not returned by the training dataset and are stripped again by the shared
variable-image-bag update adapter as a defence in depth measure.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .ku_optofil_pbc import (
    PAIRED_VIEW_METHODS,
    SUPPORTED_KU_METHODS,
    update_ku_optofil_algorithm,
)


DATASET_NAME = "CCT"
DATASET_ALIASES = {
    "cct": DATASET_NAME,
    "cct20": DATASET_NAME,
    "cct-20": DATASET_NAME,
    "caltechcameratraps": DATASET_NAME,
    "caltech_camera_traps": DATASET_NAME,
    "caltech-camera-traps": DATASET_NAME,
}
INPUT_SHAPE = (3, 112, 112)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_CCT_METHODS = set(SUPPORTED_KU_METHODS)


def canonical_cct_dataset(dataset: str) -> str:
    value = str(dataset).strip()
    if value == DATASET_NAME:
        return value
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown Caltech Camera Traps alias: {dataset!r}") from exc


def is_cct_dataset(dataset: str) -> bool:
    try:
        canonical_cct_dataset(dataset)
    except ValueError:
        return False
    return True


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().resolve()
    for candidate in (path, path / "cct", path / "cct20", path / "CCT20"):
        if (candidate / "cct_active_cache.json").is_file():
            return candidate
    return path


def _resolve_processed(dataset_root: Path) -> Path:
    pointer_path = dataset_root / "cct_active_cache.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            "CCT bbox-crop cache pointer is missing. Run scripts/prepare_cct.py; "
            "the legacy processed/ whole-frame cache is intentionally rejected."
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("instance_type") != "bbox_crop":
        raise ValueError("The active CCT cache is not a bbox-crop cache")
    directory = str(pointer.get("processed_directory", ""))
    if not directory or Path(directory).is_absolute() or ".." in Path(directory).parts:
        raise ValueError("CCT active-cache pointer contains an unsafe processed directory")
    processed = (dataset_root / directory).resolve()
    if processed.parent != dataset_root:
        raise ValueError("CCT active-cache directory must be directly under the dataset root")
    return processed


@dataclass(frozen=True)
class CCTBag:
    bag_id: str
    indices: np.ndarray
    proportions: np.ndarray
    class_counts: np.ndarray
    split: str


@dataclass
class CCTBundle:
    dataset_root: Path
    processed_root: Path
    sample_indices: np.ndarray
    image_paths: np.ndarray
    image_ids: np.ndarray
    targets: np.ndarray
    splits: np.ndarray
    locations: np.ndarray
    bags: list[CCTBag]
    metadata: dict[str, Any]

    @property
    def class_names(self) -> list[str]:
        return [str(value) for value in self.metadata["class_names"]]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return INPUT_SHAPE

    @property
    def has_instance_labels(self) -> bool:
        return True

    def bags_for_split(self, split: str) -> list[CCTBag]:
        result = [bag for bag in self.bags if bag.split == split]
        if not result:
            raise ValueError(f"CCT split {split!r} contains no feature bags")
        return result


def load_cct_bundle(root: str | os.PathLike[str]) -> CCTBundle:
    dataset_root = _resolve_root(root)
    processed = _resolve_processed(dataset_root)
    required = [
        processed / "instances.csv",
        processed / "bags.json",
        processed / "metadata.json",
        processed / "bag_manifest.json",
        processed / "class_mapping.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CCT preprocessing is incomplete; missing: " + ", ".join(missing)
        )

    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("CCT metadata must contain a non-empty class_names list")
    if "empty" in {str(value).strip().lower() for value in class_names}:
        raise ValueError("The bbox-crop CCT task must exclude the empty class")
    if metadata.get("instance_type") != "bbox_crop":
        raise ValueError("CCT metadata does not declare bbox crops as instances")
    if float(metadata.get("min_bbox_area", -1)) < 0:
        raise ValueError("CCT metadata has an invalid bbox-area threshold")
    class_mapping = json.loads(
        (processed / "class_mapping.json").read_text(encoding="utf-8")
    )
    if class_mapping.get("class_names") != class_names:
        raise ValueError("CCT contiguous class mapping disagrees with metadata")
    if class_mapping.get("empty_excluded") is not True:
        raise ValueError("CCT crop cache did not explicitly exclude empty")

    with (processed / "instances.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CCT processed instance table is empty")
    required_crop_fields = {
        "crop_id", "original_image_id", "original_image_path", "annotation_id",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_area",
        "bbox_source", "target", "category_id", "split",
    }
    missing_fields = required_crop_fields.difference(rows[0])
    if missing_fields:
        raise ValueError(
            f"CCT instance manifest is not crop-level; missing fields={sorted(missing_fields)}"
        )
    sample_indices = np.asarray(
        [int(row["sample_index"]) for row in rows], dtype=np.int64
    )
    if not np.array_equal(sample_indices, np.arange(len(rows), dtype=np.int64)):
        raise ValueError("CCT sample_index must be contiguous and row-aligned")
    image_paths = np.asarray(
        [dataset_root / row["image_path"] for row in rows], dtype=object
    )
    missing_images = [str(path) for path in image_paths if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"CCT cache references {len(missing_images)} missing images: "
            f"{missing_images[:5]}"
        )
    image_ids = np.asarray([row["image_id"] for row in rows], dtype=object)
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("CCT image_id values must be unique")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    if targets.min() < 0 or targets.max() >= len(class_names):
        raise ValueError("CCT contains an out-of-range target")
    splits = np.asarray([row["split"] for row in rows], dtype=object)
    if set(splits.tolist()) != {"train", "test"}:
        raise ValueError("CCT instances must contain train and test splits only")
    locations = np.asarray([row.get("location", "") for row in rows], dtype=object)
    bbox_areas = np.asarray([float(row["bbox_area"]) for row in rows], dtype=np.float64)
    if (bbox_areas < float(metadata["min_bbox_area"])).any():
        raise ValueError("CCT crop manifest contains a box below min_bbox_area")
    if {row["bbox_source"] for row in rows} != {str(metadata["bbox_source"])}:
        raise ValueError("CCT crop manifest mixes or misdeclares bbox sources")

    raw_bags = json.loads((processed / "bags.json").read_text(encoding="utf-8"))
    if not isinstance(raw_bags, list) or not raw_bags:
        raise ValueError("CCT bags.json must contain a non-empty list")
    bags: list[CCTBag] = []
    seen: set[int] = set()
    for raw in raw_bags:
        indices = np.asarray(raw["instance_indices"], dtype=np.int64)
        proportions = np.asarray(raw["class_proportions"], dtype=np.float32)
        counts = np.asarray(raw["class_counts"], dtype=np.int64)
        split = str(raw["split"])
        bag_id = str(raw["bag_id"])
        if len(indices) == 0:
            raise ValueError(f"Empty CCT feature bag: {bag_id}")
        if indices.min() < 0 or indices.max() >= len(rows):
            raise ValueError(f"{bag_id}: instance index is out of range")
        if set(splits[indices].tolist()) != {split}:
            raise ValueError(f"{bag_id}: bag crosses official data splits")
        overlap = seen.intersection(int(index) for index in indices)
        if overlap:
            raise ValueError(
                f"CCT instances occur in multiple feature bags: {sorted(overlap)[:5]}"
            )
        seen.update(int(index) for index in indices)
        if len(proportions) != len(class_names) or len(counts) != len(class_names):
            raise ValueError(f"{bag_id}: cached class vector has the wrong length")
        hidden_counts = np.bincount(targets[indices], minlength=len(class_names))
        if not np.array_equal(hidden_counts, counts):
            raise ValueError(f"{bag_id}: cached class counts disagree with hidden labels")
        if int(counts.sum()) != len(indices):
            raise ValueError(f"{bag_id}: cached class counts do not equal bag size")
        if not np.allclose(counts / len(indices), proportions, atol=1e-7):
            raise ValueError(f"{bag_id}: cached class proportions are stale")
        bags.append(
            CCTBag(
                bag_id=bag_id,
                indices=indices,
                proportions=proportions,
                class_counts=counts,
                split=split,
            )
        )
    if seen != set(range(len(rows))):
        missing_indices = sorted(set(range(len(rows))) - seen)
        raise ValueError(
            "Every CCT instance must occur in exactly one split-isolated feature bag; "
            f"missing={missing_indices[:5]}"
        )

    bag_manifest = json.loads(
        (processed / "bag_manifest.json").read_text(encoding="utf-8")
    )
    if bag_manifest.get("membership_inputs") != ["pca_image_features"]:
        raise ValueError(
            "CCT bag cache does not declare the required label-blind membership contract"
        )
    forbidden = {"target", "label", "location", "camera_id"}
    declared = {str(value).lower() for value in bag_manifest["membership_inputs"]}
    if forbidden.intersection(declared):
        raise ValueError("CCT bag membership manifest contains a forbidden input")
    if bag_manifest.get("instance_type") != "bbox_crop":
        raise ValueError("CCT bag membership was not built from crop instances")

    return CCTBundle(
        dataset_root=dataset_root,
        processed_root=processed,
        sample_indices=sample_indices,
        image_paths=image_paths,
        image_ids=image_ids,
        targets=targets,
        splits=splits,
        locations=locations,
        bags=sorted(bags, key=lambda bag: bag.bag_id),
        metadata=metadata,
    )


def _image_transform(split: str):
    if split == "train":
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(112, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _strong_image_transform():
    # Paired-view algorithms receive an independent draw from the same
    # reference augmentation; no unrequested stronger augmentation is added.
    return _image_transform("train")


class CCTFeatureBagDataset(Dataset):
    """One item is one immutable, variable-size feature-defined CCT bag."""

    def __init__(
        self,
        bundle: CCTBundle,
        split: str,
        *,
        paired_views: bool = False,
        expose_instance_labels: bool | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("CCT split must be train or test")
        self.bundle = bundle
        self.split = split
        self.bags = bundle.bags_for_split(split)
        self.transform = _image_transform(split)
        self.paired_views = bool(paired_views and split == "train")
        self.strong_transform = (
            _strong_image_transform() if self.paired_views else None
        )
        self.expose_instance_labels = (
            split != "train"
            if expose_instance_labels is None
            else bool(expose_instance_labels)
        )
        if split == "train" and self.expose_instance_labels:
            raise ValueError("CCT training datasets cannot expose hidden instance labels")
        self.label_prob = [bag.proportions.tolist() for bag in self.bags]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, item: int) -> dict[str, Any]:
        bag = self.bags[item]
        images: list[torch.Tensor] = []
        strong_images: list[torch.Tensor] = []
        for index in bag.indices:
            path = Path(self.bundle.image_paths[int(index)])
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images.append(self.transform(rgb))
                if self.strong_transform is not None:
                    strong_images.append(self.strong_transform(rgb))
        if not images:
            raise ValueError(f"Empty CCT feature bag: {bag.bag_id}")
        result: dict[str, Any] = {
            "x": torch.stack(images),
            "proportion": torch.from_numpy(bag.proportions.copy()),
            "bag_id": bag.bag_id,
            "sample_indices": self.bundle.sample_indices[bag.indices].tolist(),
            "image_paths": [str(self.bundle.image_paths[index]) for index in bag.indices],
            "image_ids": self.bundle.image_ids[bag.indices].tolist(),
            "locations": self.bundle.locations[bag.indices].tolist(),
        }
        if strong_images:
            result["x_strong"] = torch.stack(strong_images)
        if self.expose_instance_labels:
            result["instance_labels"] = torch.from_numpy(
                self.bundle.targets[bag.indices].copy()
            )
            result["class_counts"] = torch.from_numpy(bag.class_counts.copy())
        return result


def collate_cct_bags(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty CCT batch")
    sizes = torch.tensor([len(item["x"]) for item in items], dtype=torch.long)
    if (sizes <= 0).any():
        raise ValueError("CCT contains an empty feature bag")
    result: dict[str, Any] = {
        "x": torch.cat([item["x"] for item in items], dim=0),
        "bag_index": torch.repeat_interleave(
            torch.arange(len(items), dtype=torch.long), sizes
        ),
        "bag_sizes": sizes,
        "proportion": torch.stack([item["proportion"] for item in items]),
        "bag_ids": [str(item["bag_id"]) for item in items],
        "sample_indices": [list(item["sample_indices"]) for item in items],
        "image_paths": [list(item["image_paths"]) for item in items],
        "image_ids": [list(item["image_ids"]) for item in items],
        "locations": [list(item["locations"]) for item in items],
        "instance_weights": torch.ones(int(sizes.sum()), dtype=torch.float32),
    }
    has_strong = ["x_strong" in item for item in items]
    if any(has_strong):
        if not all(has_strong):
            raise ValueError("CCT batch mixes single-view and paired-view bags")
        result["x_strong"] = torch.cat([item["x_strong"] for item in items], dim=0)
    has_labels = ["instance_labels" in item for item in items]
    if any(has_labels):
        if not all(has_labels):
            raise ValueError("CCT batch mixes hidden-label and label-free items")
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


def _print_split(prefix: str, dataset: CCTFeatureBagDataset) -> None:
    sizes = np.asarray([len(bag.indices) for bag in dataset.bags], dtype=np.int64)
    print(
        f"{prefix}: bags={len(sizes)} instances={int(sizes.sum())} "
        f"size min/median/mean/max={int(sizes.min())}/{float(np.median(sizes)):.1f}/"
        f"{float(sizes.mean()):.2f}/{int(sizes.max())}"
    )


class CCTInstanceBatchSampler:
    """Pack complete feature bags into a fixed number of instance-scale updates.

    A target of 1024 means approximately 1024 images per optimizer step. Bags
    remain indivisible, so exact update sizes vary slightly. Greedy balancing
    yields a stable number of steps per epoch while a new deterministic shuffle
    is used for every pass through the loader.
    """

    def __init__(self, bag_sizes: Sequence[int], target_instances: int, seed: int):
        self.bag_sizes = np.asarray(bag_sizes, dtype=np.int64)
        self.target_instances = int(target_instances)
        self.seed = int(seed)
        self.epoch = 0
        if len(self.bag_sizes) == 0 or (self.bag_sizes <= 0).any():
            raise ValueError("CCT instance batch sampler requires non-empty bags")
        if self.target_instances <= 0:
            raise ValueError("CCT target training instances per update must be positive")
        self.number_of_batches = max(
            1, int(np.ceil(self.bag_sizes.sum() / self.target_instances))
        )

    def __len__(self) -> int:
        return self.number_of_batches

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(
            len(self.bag_sizes)
        )
        self.epoch += 1
        batches: list[list[int]] = [[] for _ in range(self.number_of_batches)]
        totals = np.zeros(self.number_of_batches, dtype=np.int64)
        for index in order.tolist():
            destination = int(np.argmin(totals))
            batches[destination].append(int(index))
            totals[destination] += int(self.bag_sizes[index])
        for batch in batches:
            if batch:
                yield batch


def build_cct_loaders(
    root: str,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 0,
    paired_views: bool = False,
):
    bundle = load_cct_bundle(root)
    train_dataset = CCTFeatureBagDataset(
        bundle, "train", paired_views=paired_views, expose_instance_labels=False
    )
    _print_split("CCT train", train_dataset)
    batch_sampler = CCTInstanceBatchSampler(
        [len(bag.indices) for bag in train_dataset.bags],
        target_instances=max(1, int(batch_size)),
        seed=int(seed),
    )
    print(
        "CCT train batching: "
        f"updates_per_epoch={len(batch_sampler)} "
        f"target_instances_per_update={int(batch_size)} complete_bags_only=True"
    )
    common = dict(
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_cct_bags,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, **common
    )
    return train_loader, None, bundle, bundle.input_shape


def build_cct_eval_loader(
    root: str,
    *,
    split: str = "test",
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    bundle = load_cct_bundle(root)
    dataset = CCTFeatureBagDataset(bundle, split, expose_instance_labels=True)
    _print_split(f"CCT {split}", dataset)
    effective_batch_size = 1
    if int(batch_size) != effective_batch_size:
        print("CCT evaluation uses one complete feature bag per loader batch")
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_cct_bags,
    )


def update_cct_algorithm(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, Any],
    device: str | torch.device,
    *,
    forward_chunk_size: int = 32,
    iteration: int = 0,
    softmatch_state: tuple[torch.Tensor, float, float] | None = None,
) -> Any:
    """Run the shared variable-image-bag update without hidden CCT labels."""
    if algorithm_name not in SUPPORTED_CCT_METHODS:
        raise NotImplementedError(
            f"{algorithm_name} has no verified CCT variable-bag adapter. "
            f"Verified methods: {sorted(SUPPORTED_CCT_METHODS)}"
        )
    if "instance_labels" in batch or "inferred_class_counts" in batch:
        raise ValueError("CCT training batches must not expose hidden instance labels")
    stem_variant = getattr(
        getattr(algorithm, "featurizer", None), "stem_variant", None
    )
    uses_high_resolution_stem = (
        stem_variant == "modified_3x3_stride1_no_maxpool"
    )
    data_parallel_replicas = (
        len(algorithm.network.device_ids)
        if isinstance(algorithm.network, torch.nn.DataParallel)
        else 1
    )
    if algorithm_name == "PM" and not uses_high_resolution_stem:
        # At 112x112 the complete packed update fits on an A100. A single mixed-
        # bag forward also keeps BatchNorm statistics faithful to the intended
        # approximately-1024-image optimizer batch.
        logits = algorithm.predict(batch["x"].to(device))
        loss = algorithm.PM_Loss(
            logits,
            batch["proportion"].to(device),
            bag_sizes=batch["bag_sizes"].to(device),
            bag_index=batch["bag_index"].to(device),
        )
        return algorithm._backward_step(loss)
    full_batch_size = len(batch["x"])
    # The supplement's modified stem keeps 112x112 feature maps much longer
    # than the ImageNet stem.  Preserve every complete bag and the same packed
    # optimizer batch, but checkpoint bounded forward chunks to avoid changing
    # the experiment's effective instance batch solely for memory reasons.
    effective_chunk_size = (
        min(int(forward_chunk_size), full_batch_size)
        if uses_high_resolution_stem and data_parallel_replicas < 2
        else full_batch_size
    )
    return update_ku_optofil_algorithm(
        algorithm,
        algorithm_name,
        batch,
        device,
        forward_chunk_size=effective_chunk_size,
        iteration=iteration,
        softmatch_state=softmatch_state,
        activation_checkpoint=True,
    )


@torch.no_grad()
def evaluate_cct(
    algorithm: Any,
    loader: DataLoader,
    device: str | torch.device,
    *,
    class_names: Sequence[str],
    forward_chunk_size: int = 32,
) -> dict[str, Any]:
    was_training = algorithm.training
    algorithm.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    bag_prediction: list[np.ndarray] = []
    bag_target: list[np.ndarray] = []
    bag_ids: list[str] = []
    chunk_size = max(1, int(forward_chunk_size))
    for batch in loader:
        probability_parts: list[torch.Tensor] = []
        for start in range(0, len(batch["x"]), chunk_size):
            x = batch["x"][start : start + chunk_size].to(device)
            probability_parts.append(algorithm.predict(x).softmax(dim=1).cpu())
        probabilities = torch.cat(probability_parts, dim=0)
        target = batch["instance_labels"].numpy().astype(np.int64)
        prediction = probabilities.argmax(dim=1).numpy().astype(np.int64)
        targets.append(target)
        predictions.append(prediction)
        for bag_number, bag_id in enumerate(batch["bag_ids"]):
            mask = batch["bag_index"] == bag_number
            bag_prediction.append(probabilities[mask].mean(dim=0).numpy())
            bag_target.append(batch["proportion"][bag_number].numpy())
            bag_ids.append(str(bag_id))
    if not targets:
        raise ValueError("CCT evaluation loader is empty")
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (target, prediction), 1)
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_support = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        tp, predicted_support, out=np.zeros_like(tp), where=predicted_support > 0
    )
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    predicted_bags = np.asarray(bag_prediction, dtype=np.float64)
    target_bags = np.asarray(bag_target, dtype=np.float64)
    absolute = np.abs(predicted_bags - target_bags)
    result = {
        "instances": int(len(target)),
        "bags": len(bag_ids),
        "instance_accuracy": float(tp.sum() / max(1.0, support.sum())),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "bag_proportion_mae": float(absolute.mean()),
        "bag_proportion_rmse": float(
            np.sqrt(np.square(predicted_bags - target_bags).mean())
        ),
        "per_class": {
            str(name): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "bag_ids": bag_ids,
    }
    if was_training:
        algorithm.train()
    return result
