"""KU-Optofil PBC natural patient-bag LLP dataset adapter.

Every item is one anonymized patient and every instance is one RGB cell image.
Ground-truth cell labels are returned for sanity checks and evaluation only;
``update_ku_optofil_algorithm`` deliberately removes them before constructing
the LLP loss.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

from .ref2021 import SUPPORTED_REF_METHODS, ref2021_loss


DATASET_NAME = "KUOptofilPBC"
DATASET_ALIASES = {
    "kuoptofilpbc": DATASET_NAME,
    "ku_optofil_pbc": DATASET_NAME,
    "ku-optofil-pbc": DATASET_NAME,
    "kuoptofil": DATASET_NAME,
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
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAIRED_VIEW_METHODS = {
    "LLP_AHIL",
    "LLP_DC",
    "LLP_FixMatch",
    "LLP_SoftMatch",
}
SUPPORTED_KU_METHODS = (
    set(SUPPORTED_REF_METHODS)
    | PAIRED_VIEW_METHODS
    | {"LLP_FlowLLP", "LLP_SimCLR", "LLP_VAT"}
)


def canonical_ku_optofil_dataset(dataset: str) -> str:
    value = str(dataset).strip()
    if value == DATASET_NAME:
        return value
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown KU-Optofil dataset alias: {dataset!r}") from exc


def is_ku_optofil_dataset(dataset: str) -> bool:
    try:
        canonical_ku_optofil_dataset(dataset)
    except ValueError:
        return False
    return True


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().resolve()
    for candidate in (path, path / "ku_optofil_pbc"):
        if (candidate / "processed" / "instances.csv").is_file():
            return candidate
    return path


@dataclass(frozen=True)
class KUOptofilBag:
    bag_id: str
    indices: np.ndarray
    proportions: np.ndarray
    class_counts: np.ndarray
    split: str


@dataclass
class KUOptofilBundle:
    dataset_root: Path
    image_paths: np.ndarray
    image_names: np.ndarray
    patient_ids: np.ndarray
    instance_labels: np.ndarray
    bags: list[KUOptofilBag]
    metadata: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return INPUT_SHAPE

    @property
    def has_instance_labels(self) -> bool:
        return True

    def bags_for_split(self, split: str) -> list[KUOptofilBag]:
        result = [bag for bag in self.bags if bag.split == split]
        if not result:
            raise ValueError(f"KU-Optofil split {split!r} contains no patient bags")
        return result


def load_ku_optofil_bundle(root: str | os.PathLike[str]) -> KUOptofilBundle:
    dataset_root = _resolve_root(root)
    processed = dataset_root / "processed"
    required = [
        processed / "instances.csv",
        processed / "bags.json",
        processed / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "KU-Optofil preprocessing is incomplete; missing: " + ", ".join(missing)
        )
    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("class_names") != CLASS_NAMES:
        raise ValueError("KU-Optofil class order differs from the benchmark contract")
    with (processed / "instances.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("KU-Optofil processed instance table is empty")
    indices = np.asarray([int(row["instance_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(rows), dtype=np.int64)):
        raise ValueError("KU-Optofil instance_index must be contiguous and row-aligned")
    image_paths = np.asarray(
        [dataset_root / row["relative_path"] for row in rows], dtype=object
    )
    missing_images = [str(path) for path in image_paths if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"KU-Optofil cache references {len(missing_images)} missing images: "
            f"{missing_images[:5]}"
        )
    image_names = np.asarray([row["image_name"] for row in rows], dtype=object)
    patient_ids = np.asarray([row["patient_id"] for row in rows], dtype=object)
    instance_labels = np.asarray([int(row["class_index"]) for row in rows], dtype=np.int64)
    if instance_labels.min() < 0 or instance_labels.max() >= NUM_CLASSES:
        raise ValueError("KU-Optofil contains an out-of-range cell label")

    raw_bags = json.loads((processed / "bags.json").read_text(encoding="utf-8"))
    bags: list[KUOptofilBag] = []
    seen: set[int] = set()
    for raw in raw_bags:
        bag_indices = np.asarray(raw["instance_indices"], dtype=np.int64)
        proportions = np.asarray(raw["class_proportions"], dtype=np.float32)
        counts = np.asarray(raw["class_counts"], dtype=np.int64)
        if len(bag_indices) == 0:
            raise ValueError(f"Empty patient bag: {raw['bag_id']}")
        if len(proportions) != NUM_CLASSES or len(counts) != NUM_CLASSES:
            raise ValueError(f"{raw['bag_id']}: expected {NUM_CLASSES} classes")
        if (proportions < 0).any() or not np.isclose(proportions.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{raw['bag_id']}: invalid class proportions")
        if int(counts.sum()) != len(bag_indices):
            raise ValueError(f"{raw['bag_id']}: class counts do not equal bag size")
        hidden_counts = np.bincount(instance_labels[bag_indices], minlength=NUM_CLASSES)
        if not np.array_equal(counts, hidden_counts):
            raise ValueError(f"{raw['bag_id']}: cached bag histogram is stale")
        overlap = seen.intersection(int(index) for index in bag_indices)
        if overlap:
            raise ValueError(f"KU-Optofil instances occur in multiple bags: {sorted(overlap)[:5]}")
        seen.update(int(index) for index in bag_indices)
        bag_patient_ids = set(patient_ids[bag_indices].tolist())
        if bag_patient_ids != {str(raw["patient_id"])}:
            raise ValueError(f"{raw['bag_id']}: patient membership mismatch")
        bags.append(
            KUOptofilBag(
                bag_id=str(raw["bag_id"]),
                indices=bag_indices,
                proportions=proportions,
                class_counts=counts,
                split=str(raw["split"]),
            )
        )
    if len(seen) != len(rows):
        raise ValueError("Some KU-Optofil instances do not belong to exactly one patient bag")
    split_sets = {
        split: {bag.bag_id for bag in bags if bag.split == split}
        for split in ("train", "val", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    image_split_sets = {
        split: {
            str(image_names[index])
            for bag in bags
            if bag.split == split
            for index in bag.indices
        }
        for split in ("train", "val", "test")
    }
    assert image_split_sets["train"].isdisjoint(image_split_sets["val"])
    assert image_split_sets["train"].isdisjoint(image_split_sets["test"])
    assert image_split_sets["val"].isdisjoint(image_split_sets["test"])
    return KUOptofilBundle(
        dataset_root=dataset_root,
        image_paths=image_paths,
        image_names=image_names,
        patient_ids=patient_ids,
        instance_labels=instance_labels,
        bags=bags,
        metadata=metadata,
    )


def _image_transform(split: str):
    operations: list[Any] = [transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2]))]
    if split == "train":
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def _strong_image_transform():
    """Morphology-preserving strong view for paired consistency methods."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class KUOptofilPBCDataset(Dataset):
    """One dataset item is one complete natural patient bag by default."""

    def __init__(
        self,
        bundle: KUOptofilBundle,
        split: str,
        *,
        seed: int = 42,
        train_instance_sample_size: Optional[int] = None,
        paired_views: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("KU-Optofil split must be train, val, or test")
        if split != "train" and train_instance_sample_size is not None:
            raise ValueError("KU-Optofil validation/test always use complete patient bags")
        if train_instance_sample_size is not None and int(train_instance_sample_size) <= 0:
            raise ValueError("train_instance_sample_size must be positive or null")
        self.bundle = bundle
        self.split = split
        self.bags = bundle.bags_for_split(split)
        self.seed = int(seed)
        self.epoch = 0
        self.train_instance_sample_size = (
            None if train_instance_sample_size is None else int(train_instance_sample_size)
        )
        self.transform = _image_transform(split)
        self.paired_views = bool(paired_views and split == "train")
        self.strong_transform = (
            _strong_image_transform() if self.paired_views else None
        )
        self.label_prob = [bag.proportions.tolist() for bag in self.bags]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.bags)

    def _indices(self, bag: KUOptofilBag) -> np.ndarray:
        requested = self.train_instance_sample_size
        if requested is None or len(bag.indices) <= requested:
            return bag.indices
        digest = hashlib.blake2b(
            f"{self.seed}:{self.epoch}:{bag.bag_id}".encode(), digest_size=8
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        return np.sort(rng.choice(bag.indices, size=requested, replace=False))

    def __getitem__(self, item: int) -> dict[str, Any]:
        bag = self.bags[item]
        indices = self._indices(bag)
        images: list[torch.Tensor] = []
        strong_images: list[torch.Tensor] = []
        for index in indices:
            path = Path(self.bundle.image_paths[int(index)])
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images.append(self.transform(rgb))
                if self.strong_transform is not None:
                    strong_images.append(self.strong_transform(rgb))
        if not images:
            raise ValueError(f"Empty KU-Optofil patient bag: {bag.bag_id}")
        result = {
            "x": torch.stack(images),
            "proportion": torch.from_numpy(bag.proportions.copy()),
            "bag_id": bag.bag_id,
            "instance_ids": self.bundle.image_names[indices].tolist(),
            "instance_labels": torch.from_numpy(
                self.bundle.instance_labels[indices].copy()
            ),
            "class_counts": torch.from_numpy(bag.class_counts.copy()),
        }
        if strong_images:
            result["x_strong"] = torch.stack(strong_images)
        return result


def collate_ku_optofil_bags(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty KU-Optofil batch")
    sizes = torch.tensor([len(item["x"]) for item in items], dtype=torch.long)
    if (sizes <= 0).any():
        raise ValueError("KU-Optofil contains an empty patient bag")
    result = {
        "x": torch.cat([item["x"] for item in items], dim=0),
        "bag_index": torch.repeat_interleave(
            torch.arange(len(items), dtype=torch.long), sizes
        ),
        "bag_sizes": sizes,
        "proportion": torch.stack([item["proportion"] for item in items]),
        "bag_ids": [str(item["bag_id"]) for item in items],
        "instance_ids": [list(item["instance_ids"]) for item in items],
        "instance_labels": torch.cat([item["instance_labels"] for item in items]),
        "instance_weights": torch.ones(int(sizes.sum()), dtype=torch.float32),
        "inferred_class_counts": torch.stack([item["class_counts"] for item in items]),
    }
    has_strong = ["x_strong" in item for item in items]
    if any(has_strong):
        if not all(has_strong):
            raise ValueError("KU-Optofil batch mixes single-view and paired-view bags")
        result["x_strong"] = torch.cat(
            [item["x_strong"] for item in items], dim=0
        )
    return result


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _print_split(prefix: str, dataset: KUOptofilPBCDataset) -> None:
    sizes = np.asarray([len(bag.indices) for bag in dataset.bags], dtype=np.int64)
    print(
        f"{prefix}: bags={len(sizes)} instances={int(sizes.sum())} "
        f"size min/median/mean/max={int(sizes.min())}/{float(np.median(sizes)):.1f}/"
        f"{float(sizes.mean()):.2f}/{int(sizes.max())}"
    )


def build_ku_optofil_loaders(
    root: str,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 0,
    train_instance_sample_size: Optional[int] = None,
    paired_views: bool = False,
):
    bundle = load_ku_optofil_bundle(root)
    train_dataset = KUOptofilPBCDataset(
        bundle,
        "train",
        seed=seed,
        train_instance_sample_size=train_instance_sample_size,
        paired_views=paired_views,
    )
    val_dataset = KUOptofilPBCDataset(bundle, "val", seed=seed)
    _print_split("KU-Optofil PBC train", train_dataset)
    _print_split("KU-Optofil PBC val", val_dataset)
    generator = torch.Generator().manual_seed(int(seed))
    common = dict(
        batch_size=max(1, int(batch_size)),
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_ku_optofil_bags,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, bundle, bundle.input_shape


def build_ku_optofil_eval_loader(
    root: str,
    *,
    split: str = "test",
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    bundle = load_ku_optofil_bundle(root)
    dataset = KUOptofilPBCDataset(bundle, split)
    _print_split(f"KU-Optofil PBC {split}", dataset)
    # One natural patient per loader batch bounds host memory while preserving
    # the complete bag. Model inference is additionally chunked below.
    effective_batch_size = 1
    if int(batch_size) != effective_batch_size:
        print("KU-Optofil evaluation uses one complete patient bag per loader batch")
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_ku_optofil_bags,
    )


@contextmanager
def _restore_batch_norm_running_stats(module: torch.nn.Module):
    """Undo BN-buffer mutations made by checkpoint recomputation.

    Keeping ``track_running_stats`` enabled makes the recomputed graph identical
    to the original graph, as required by non-reentrant checkpointing.  The
    buffer snapshots ensure that the replay itself does not count as a second
    training forward.
    """
    states = []
    for child in module.modules():
        if (
            isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
            and child.track_running_stats
        ):
            states.append(
                (
                    child,
                    child.running_mean.detach().clone(),
                    child.running_var.detach().clone(),
                    child.num_batches_tracked.detach().clone(),
                )
            )
    try:
        yield
    finally:
        with torch.no_grad():
            for child, running_mean, running_var, num_batches_tracked in states:
                child.running_mean.copy_(running_mean)
                child.running_var.copy_(running_var)
                child.num_batches_tracked.copy_(num_batches_tracked)


def update_ku_optofil_algorithm(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, Any],
    device: str | torch.device,
    *,
    forward_chunk_size: int = 32,
    iteration: int = 0,
    softmatch_state: Optional[tuple[torch.Tensor, float, float]] = None,
    activation_checkpoint: bool = False,
) -> Any:
    """Perform one LLP update without exposing cell labels to the loss."""
    if algorithm_name not in SUPPORTED_KU_METHODS:
        raise NotImplementedError(
            f"{algorithm_name} has no verified KU variable-bag adapter. "
            f"Verified methods: {sorted(SUPPORTED_KU_METHODS)}"
        )
    training_keys = {
        "bag_index",
        "bag_sizes",
        "proportion",
        "instance_weights",
    }
    training_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key in training_keys
    }
    assert "instance_labels" not in training_batch
    assert "inferred_class_counts" not in training_batch
    chunk_size = max(1, int(forward_chunk_size))

    def forward_chunks(values: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, len(values), chunk_size):
            chunk = values[start : start + chunk_size].to(device)
            if activation_checkpoint:
                output = checkpoint(
                    algorithm.predict,
                    chunk,
                    use_reentrant=False,
                    context_fn=lambda: (
                        nullcontext(),
                        _restore_batch_norm_running_stats(algorithm.network),
                    ),
                )
            else:
                output = algorithm.predict(chunk)
            chunks.append(output)
        return torch.cat(chunks, dim=0)

    if algorithm_name == "LLP_SimCLR":
        logits_chunks = []
        feature_chunks = []
        for start in range(0, len(batch["x"]), chunk_size):
            chunk_logits, chunk_features = algorithm.network(
                batch["x"][start : start + chunk_size].to(device)
            )
            logits_chunks.append(chunk_logits)
            feature_chunks.append(chunk_features)
        return algorithm.update_from_outputs(
            torch.cat(logits_chunks, dim=0),
            torch.cat(feature_chunks, dim=0),
            training_batch["proportion"],
            bag_sizes=training_batch["bag_sizes"],
            bag_index=training_batch["bag_index"],
        )

    if algorithm_name == "LLP_FlowLLP":
        feature_chunks = []
        logits_chunks = []
        for start in range(0, len(batch["x"]), chunk_size):
            chunk_features, chunk_logits = algorithm._forward_with_features(
                batch["x"][start : start + chunk_size].to(device)
            )
            feature_chunks.append(chunk_features)
            logits_chunks.append(chunk_logits)
        return algorithm.update_from_outputs(
            torch.cat(feature_chunks, dim=0),
            torch.cat(logits_chunks, dim=0),
            training_batch["proportion"],
            bag_sizes=training_batch["bag_sizes"],
            bag_index=training_batch["bag_index"],
        )

    logits_weak = forward_chunks(batch["x"])
    if algorithm_name in PAIRED_VIEW_METHODS:
        if "x_strong" not in batch:
            raise ValueError(
                f"{algorithm_name} requires aligned weak/strong KU views"
            )
        if len(batch["x_strong"]) != len(batch["x"]):
            raise ValueError("KU weak/strong views contain different instance counts")
        logits = torch.cat([logits_weak, forward_chunks(batch["x_strong"])], dim=0)
        sizes = training_batch["bag_sizes"]
        index = training_batch["bag_index"]
        proportions = training_batch["proportion"]
        if algorithm_name == "LLP_DC":
            loss, _, _ = algorithm.LLP_DC_Loss(
                logits, proportions, bag_sizes=sizes, bag_index=index
            )
        elif algorithm_name == "LLP_FixMatch":
            loss, _, _ = algorithm.LLP_FixMatch_Loss(
                logits, proportions, bag_sizes=sizes, bag_index=index
            )
        elif algorithm_name == "LLP_AHIL":
            loss, _, _ = algorithm.LLP_AHIL_Loss(
                logits, proportions, bag_sizes=sizes, bag_index=index
            )
        else:
            if softmatch_state is None:
                classes = proportions.shape[1]
                softmatch_state = (
                    torch.full((classes,), 1.0 / classes, device=device),
                    1.0 / classes,
                    1.0,
                )
            loss, _, _, ema, mean, variance = algorithm.LLP_SoftMatch_Loss(
                logits,
                proportions,
                *softmatch_state,
                bag_sizes=sizes,
                bag_index=index,
            )
            step = algorithm._backward_step(loss)
            return step, (ema, mean, variance)
        return algorithm._backward_step(loss)

    if algorithm_name == "LLP_VAT":
        consistency_sum = logits_weak.new_tensor(0.0)
        total = 0
        for start in range(0, len(batch["x"]), chunk_size):
            values = batch["x"][start : start + chunk_size].to(device)
            chunk_loss = algorithm.consistency_criterion(
                algorithm.network,
                values,
                disable_bn_ctx=algorithm._disable_tracking_bn_stats,
            )
            consistency_sum = consistency_sum + len(values) * chunk_loss
            total += len(values)
        consistency = consistency_sum / max(total, 1)
        alpha = algorithm.get_rampup_weight(
            algorithm.consistency, int(iteration), algorithm.consistency_rampup
        )
        loss = algorithm.PM_Loss(
            logits_weak,
            training_batch["proportion"],
            bag_sizes=training_batch["bag_sizes"],
            bag_index=training_batch["bag_index"],
        ) + alpha * consistency
        return algorithm._backward_step(loss)

    logits = logits_weak
    # Keep image tensors in host memory and transfer only bounded chunks. The
    # concatenated logits retain every chunk's autograd graph, so the LLP loss
    # still backpropagates through every cell in the complete natural bag.
    loss = ref2021_loss(
        algorithm, algorithm_name, training_batch, logits=logits
    )
    return algorithm._backward_step(loss)


def _confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (target, prediction), 1)
    return matrix


@torch.no_grad()
def evaluate_ku_optofil(
    algorithm: Any,
    loader: DataLoader,
    device: str | torch.device,
    *,
    forward_chunk_size: int = 32,
) -> dict[str, Any]:
    was_training = algorithm.training
    algorithm.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    bag_prediction: list[np.ndarray] = []
    bag_target: list[np.ndarray] = []
    bag_ids: list[str] = []
    for batch in loader:
        probability_parts: list[torch.Tensor] = []
        for start in range(0, len(batch["x"]), max(1, int(forward_chunk_size))):
            x = batch["x"][start : start + forward_chunk_size].to(device)
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
        raise ValueError("KU-Optofil evaluation loader is empty")
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    matrix = _confusion_matrix(target, prediction)
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_support = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted_support, out=np.zeros_like(tp), where=predicted_support > 0)
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
        "bag_proportion_rmse": float(np.sqrt(np.square(predicted_bags - target_bags).mean())),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "bag_ids": bag_ids,
    }
    if was_training:
        algorithm.train()
    return result
