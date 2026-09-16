"""Amazon-WILDS reviewer-level natural-bag LLP adapter.

The official WILDS user split is preserved before bag construction.  Each
training reviewer is exactly one bag and all reviews from that reviewer remain
in the bag.  Hidden training ratings are stored for auditability but are never
returned by the LLP training dataset or collator.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ref2021 import SUPPORTED_REF_METHODS, ref2021_loss
from .remote_sensing import classification_metrics_from_confusion


DATASET_NAME = "AmazonWILDS"
DATASET_ALIASES = {
    "amazonwilds": DATASET_NAME,
    "amazon_wilds": DATASET_NAME,
    "amazon-wilds": DATASET_NAME,
}
CLASS_NAMES = ["1star", "2star", "3star", "4star", "5star"]
NUM_CLASSES = len(CLASS_NAMES)
OFFICIAL_SPLITS = ("train", "val", "id_val", "test", "id_test")
SUPPORTED_AMAZON_METHODS = set(SUPPORTED_REF_METHODS)


def canonical_amazon_wilds_dataset(dataset: str) -> str:
    value = str(dataset).strip()
    if value == DATASET_NAME:
        return value
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown Amazon-WILDS dataset alias: {dataset!r}") from exc


def is_amazon_wilds_dataset(dataset: str) -> bool:
    try:
        canonical_amazon_wilds_dataset(dataset)
    except ValueError:
        return False
    return True


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().resolve()
    candidates = (path, path / "amazon_wilds")
    for candidate in candidates:
        if (candidate / "processed" / "instances.parquet").is_file():
            return candidate
    return path


def _fixed_list_matrix(table: Any, name: str, dtype: np.dtype) -> np.ndarray:
    column = table[name].combine_chunks()
    if not hasattr(column.type, "list_size"):
        raise ValueError(f"{name} must be a fixed-size list column")
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype).reshape(len(column), column.type.list_size)


@dataclass(frozen=True)
class AmazonWILDSBag:
    bag_id: str
    reviewer_id: str
    reviewer_index: int
    indices: np.ndarray
    proportions: np.ndarray
    class_counts: np.ndarray


@dataclass
class AmazonWILDSBundle:
    dataset_root: Path
    features: np.ndarray
    instance_ids: np.ndarray
    labels: np.ndarray
    reviewer_ids: np.ndarray
    splits: np.ndarray
    train_bags: list[AmazonWILDSBag]
    metadata: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def has_instance_labels(self) -> bool:
        return True


def load_amazon_wilds_bundle(
    root: str | os.PathLike[str], *, require_features: bool = True
) -> AmazonWILDSBundle:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Amazon-WILDS requires pyarrow; install "
            "plench/requirements-amazon-wilds.txt"
        ) from exc

    dataset_root = _resolve_root(root)
    processed = dataset_root / "processed"
    metadata_path = processed / "metadata.json"
    instances_path = processed / "instances.parquet"
    bags_path = processed / "train_bags.parquet"
    stats_path = processed / "amazon_wilds_stats.json"
    features_path = dataset_root / "features" / "features.npy"
    feature_manifest_path = dataset_root / "features" / "feature_manifest.json"
    required = (metadata_path, instances_path, bags_path, stats_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Amazon-WILDS preprocessing is incomplete; missing: " + ", ".join(missing)
        )
    if require_features and not features_path.is_file():
        raise FileNotFoundError(
            f"missing {features_path}; run python -m "
            "plench.scripts.extract_amazon_wilds_features"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("class_names") != CLASS_NAMES:
        raise ValueError("Amazon-WILDS class order does not match 1..5 stars")
    if metadata.get("reviewer_field") != "reviewerID":
        raise ValueError("Amazon-WILDS reviewer field must be reviewerID")

    # The text column is intentionally not loaded here: it is large and the
    # learner consumes the frozen feature cache.  Keep bundle loading linear
    # in the compact metadata rather than materializing all review strings.
    instances = pq.read_table(
        instances_path,
        columns=["feature_index", "instance_id", "reviewer_id", "split", "label"],
    )
    bags_table = pq.read_table(bags_path)
    feature_indices = np.asarray(
        instances["feature_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    if not np.array_equal(feature_indices, np.arange(len(feature_indices))):
        raise ValueError("Amazon-WILDS feature_index must be contiguous and row-aligned")
    instance_ids = np.asarray(
        instances["instance_id"].combine_chunks().to_pylist(), dtype=object
    )
    reviewer_ids = np.asarray(
        instances["reviewer_id"].combine_chunks().to_pylist(), dtype=object
    )
    splits = np.asarray(instances["split"].combine_chunks().to_pylist(), dtype=object)
    labels = np.asarray(
        instances["label"].combine_chunks().to_numpy(), dtype=np.int64
    )
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError("Amazon-WILDS labels must be zero-based integers in [0,4]")

    if require_features:
        features = np.load(features_path, mmap_mode="r")
        if features.ndim != 2 or len(features) != len(instance_ids):
            raise ValueError("Amazon-WILDS feature cache shape disagrees with instances")
        manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if int(manifest["instances"]) != len(instance_ids):
            raise ValueError("Amazon-WILDS feature manifest instance count is stale")
        metadata = dict(metadata)
        metadata["feature_manifest"] = manifest
    else:
        features = np.empty((len(instance_ids), 0), dtype=np.float32)

    bag_ids = bags_table["bag_id"].combine_chunks().to_pylist()
    bag_reviewers = bags_table["reviewer_id"].combine_chunks().to_pylist()
    reviewer_indices = np.asarray(
        bags_table["reviewer_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    expected_sizes = np.asarray(
        bags_table["n_instances"].combine_chunks().to_numpy(), dtype=np.int64
    )
    proportions = _fixed_list_matrix(
        bags_table, "class_proportions", np.float32
    )
    class_counts = _fixed_list_matrix(bags_table, "class_counts", np.int64)
    train_mask = splits == "train"
    train_position_array = np.flatnonzero(train_mask)
    train_positions = set(train_position_array.tolist())
    positions_by_reviewer: dict[str, list[int]] = {}
    for position in train_position_array:
        positions_by_reviewer.setdefault(str(reviewer_ids[position]), []).append(
            int(position)
        )
    seen_positions: set[int] = set()
    train_bags: list[AmazonWILDSBag] = []
    for row, bag_id_value in enumerate(bag_ids):
        bag_id = str(bag_id_value)
        reviewer_id = str(bag_reviewers[row])
        indices = np.asarray(positions_by_reviewer.get(reviewer_id, ()), dtype=np.int64)
        if len(indices) != int(expected_sizes[row]):
            raise ValueError(
                f"{bag_id}: instances={len(indices)} but train_bags.parquet says "
                f"{expected_sizes[row]}"
            )
        if len(set(str(value) for value in reviewer_ids[indices])) != 1:
            raise ValueError(f"{bag_id}: one bag contains multiple reviewers")
        overlap = seen_positions.intersection(indices.tolist())
        if overlap:
            raise ValueError(f"Amazon-WILDS train instances occur in multiple bags: {sorted(overlap)[:5]}")
        seen_positions.update(indices.tolist())
        vector = proportions[row]
        counts = class_counts[row]
        if (vector < 0).any() or not np.isclose(vector.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{bag_id}: invalid class proportion {vector}")
        recomputed = np.bincount(labels[indices], minlength=NUM_CLASSES)
        if not np.array_equal(recomputed, counts):
            raise ValueError(f"{bag_id}: cached class counts do not match hidden labels")
        if not np.allclose(recomputed / len(indices), vector, atol=1e-7):
            raise ValueError(f"{bag_id}: cached proportions do not match hidden labels")
        train_bags.append(
            AmazonWILDSBag(
                bag_id=bag_id,
                reviewer_id=reviewer_id,
                reviewer_index=int(reviewer_indices[row]),
                indices=indices,
                proportions=vector.copy(),
                class_counts=counts.copy(),
            )
        )
    if seen_positions != train_positions:
        raise ValueError("some Amazon-WILDS training reviews do not belong to one reviewer bag")
    if len({bag.reviewer_id for bag in train_bags}) != len(train_bags):
        raise ValueError("one Amazon-WILDS reviewer maps to multiple bags")
    if len({bag.bag_id for bag in train_bags}) != len(train_bags):
        raise ValueError("Amazon-WILDS bag IDs are not unique")

    return AmazonWILDSBundle(
        dataset_root=dataset_root,
        features=features,
        instance_ids=instance_ids,
        labels=labels,
        reviewer_ids=reviewer_ids,
        splits=splits,
        train_bags=sorted(train_bags, key=lambda bag: bag.bag_id),
        metadata=metadata,
    )


class AmazonWILDSTrainDataset(Dataset):
    """One item is one complete training-reviewer bag; no rating is returned."""

    def __init__(
        self,
        bundle: AmazonWILDSBundle,
        *,
        seed: int = 42,
        num_reviewers: Optional[int] = None,
    ) -> None:
        bags = list(bundle.train_bags)
        if num_reviewers is not None:
            requested = int(num_reviewers)
            if requested <= 0:
                raise ValueError("num_reviewers must be positive or null")
            if requested > len(bags):
                raise ValueError(
                    f"num_reviewers={requested} exceeds {len(bags)} training reviewers"
                )
            rng = np.random.default_rng(int(seed))
            chosen = np.sort(rng.choice(len(bags), size=requested, replace=False))
            bags = [bags[int(index)] for index in chosen]
        self.bundle = bundle
        self.bags = bags
        self.label_prob = [bag.proportions.tolist() for bag in bags]
        self.num_reviewers = None if num_reviewers is None else int(num_reviewers)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, item: int) -> dict[str, Any]:
        bag = self.bags[item]
        features = np.asarray(self.bundle.features[bag.indices], dtype=np.float32).copy()
        if not np.isfinite(features).all():
            raise ValueError(f"{bag.bag_id}: non-finite feature values")
        # Deliberately no instance_label/class_counts key. Hidden ratings are
        # preprocessing/audit data and never enter the LLP update.
        return {
            "x": torch.from_numpy(features),
            "proportion": torch.from_numpy(bag.proportions.copy()),
            "bag_id": bag.bag_id,
            "reviewer_id": bag.reviewer_id,
            "instance_ids": self.bundle.instance_ids[bag.indices].tolist(),
            "instance_weights": torch.ones(len(bag.indices), dtype=torch.float32),
        }


class AmazonWILDSEvalDataset(Dataset):
    """Ordinary review-level labeled evaluation split."""

    def __init__(self, bundle: AmazonWILDSBundle, split: str) -> None:
        if split not in OFFICIAL_SPLITS[1:]:
            raise ValueError(f"Amazon-WILDS evaluation split must be one of {OFFICIAL_SPLITS[1:]}")
        self.bundle = bundle
        self.split = split
        self.indices = np.flatnonzero(bundle.splits == split)
        if len(self.indices) == 0:
            raise ValueError(f"Amazon-WILDS split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        feature = np.asarray(self.bundle.features[index], dtype=np.float32).copy()
        if not np.isfinite(feature).all():
            raise ValueError(f"{self.bundle.instance_ids[index]}: non-finite feature values")
        return {
            "x": torch.from_numpy(feature),
            "label": int(self.bundle.labels[index]),
            "instance_id": str(self.bundle.instance_ids[index]),
            "reviewer_id": str(self.bundle.reviewer_ids[index]),
        }


def collate_amazon_wilds_bags(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty Amazon-WILDS batch")
    sizes = torch.tensor([len(item["x"]) for item in items], dtype=torch.long)
    if bool((sizes <= 0).any()):
        raise ValueError("Amazon-WILDS contains an empty reviewer bag")
    return {
        "x": torch.cat([item["x"] for item in items], dim=0),
        "bag_index": torch.repeat_interleave(torch.arange(len(items)), sizes),
        "bag_sizes": sizes,
        "proportion": torch.stack([item["proportion"] for item in items]),
        "bag_ids": [str(item["bag_id"]) for item in items],
        "reviewer_ids": [str(item["reviewer_id"]) for item in items],
        "instance_ids": [list(item["instance_ids"]) for item in items],
        "instance_weights": torch.cat([item["instance_weights"] for item in items]),
    }


def collate_amazon_wilds_eval(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in items]),
        "label": torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
        "instance_ids": [str(item["instance_id"]) for item in items],
        "reviewer_ids": [str(item["reviewer_id"]) for item in items],
    }


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _print_train(dataset: AmazonWILDSTrainDataset) -> None:
    sizes = np.asarray([len(bag.indices) for bag in dataset.bags], dtype=np.int64)
    print(
        "Amazon-WILDS train: "
        f"reviewers=bags={len(sizes)} instances={int(sizes.sum())} "
        f"size min/median/mean/max={int(sizes.min())}/{float(np.median(sizes)):.1f}/"
        f"{float(sizes.mean()):.2f}/{int(sizes.max())}"
    )


def build_amazon_wilds_loaders(
    root: str,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 0,
    num_reviewers: Optional[int] = None,
):
    bundle = load_amazon_wilds_bundle(root)
    train_dataset = AmazonWILDSTrainDataset(
        bundle, seed=seed, num_reviewers=num_reviewers
    )
    val_dataset = AmazonWILDSEvalDataset(bundle, "val")
    _print_train(train_dataset)
    print(
        f"Amazon-WILDS val: instances={len(val_dataset)} "
        f"reviewers={len(set(bundle.reviewer_ids[val_dataset.indices].tolist()))}"
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_amazon_wilds_bags,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_amazon_wilds_eval,
    )
    return train_loader, val_loader, bundle, bundle.feature_dim


def build_amazon_wilds_eval_loader(
    root: str,
    *,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    bundle = load_amazon_wilds_bundle(root)
    dataset = AmazonWILDSEvalDataset(bundle, split)
    print(
        f"Amazon-WILDS {split}: instances={len(dataset)} "
        f"reviewers={len(set(bundle.reviewer_ids[dataset.indices].tolist()))}"
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_amazon_wilds_eval,
    )


def update_amazon_wilds_algorithm(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, Any],
    device: str | torch.device,
) -> dict[str, float]:
    if algorithm_name not in SUPPORTED_AMAZON_METHODS:
        raise NotImplementedError(
            f"{algorithm_name} has no verified Amazon-WILDS natural-bag adapter. "
            f"Verified methods: {sorted(SUPPORTED_AMAZON_METHODS)}"
        )
    allowed = {"x", "bag_index", "bag_sizes", "proportion", "instance_weights"}
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key in allowed
    }
    assert "label" not in moved and "instance_labels" not in moved
    loss = ref2021_loss(algorithm, algorithm_name, moved)
    return algorithm._backward_step(loss)


@torch.no_grad()
def evaluate_amazon_wilds(
    algorithm: Any,
    loader: DataLoader,
    device: str | torch.device,
) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    algorithm.eval()
    for batch in loader:
        logits = algorithm.predict(batch["x"].to(device))
        prediction = logits.argmax(dim=1).detach().cpu().numpy()
        target = batch["label"].numpy()
        np.add.at(confusion, (target, prediction), 1)
    return classification_metrics_from_confusion(confusion, CLASS_NAMES)
