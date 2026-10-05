"""REF 2021 UoA 11 natural-bag LLP adapter.

One official institution/UoA submission is one bag.  Individual outputs have
no public quality label; only the official Outputs Profile is exposed to the
trainer.  Frozen text features are loaded from a stable on-disk cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DATASET_NAME = "REF2021UOA11"
DATASET_ALIASES = {
    "ref2021uoa11": DATASET_NAME,
    "ref2021_uoa11": DATASET_NAME,
    "ref2021-uoa11": DATASET_NAME,
}
CLASS_NAMES = ["unclassified", "1star", "2star", "3star", "4star"]
NUM_CLASSES = len(CLASS_NAMES)
SUPPORTED_REF_METHODS = {
    "PM",
    "LLP_MM",
    "LLP_DSQ",
    "LLP_PVC",
    "LLP_PT",
    "LLP_FC",
    "ROT",
    "EasyLLP",
    "GeneralUPM",
    "NonClipOVR",
}


def canonical_ref2021_dataset(dataset: str) -> str:
    value = str(dataset).strip()
    if value == DATASET_NAME:
        return value
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown REF2021 dataset alias: {dataset!r}") from exc


def is_ref2021_dataset(dataset: str) -> bool:
    try:
        canonical_ref2021_dataset(dataset)
    except ValueError:
        return False
    return True


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().resolve()
    candidates = (path, path / "ref2021_uoa11")
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
class REF2021Bag:
    bag_id: str
    institution_name: str
    indices: np.ndarray
    proportions: np.ndarray
    inferred_counts: np.ndarray
    split: str


@dataclass
class REF2021Bundle:
    dataset_root: Path
    features: np.ndarray
    instance_ids: np.ndarray
    instance_weights: np.ndarray
    bags: list[REF2021Bag]
    metadata: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def has_instance_labels(self) -> bool:
        return False

    def bags_for_split(self, split: str) -> list[REF2021Bag]:
        result = [bag for bag in self.bags if bag.split == split]
        if not result:
            raise ValueError(f"REF2021 split {split!r} contains no bags")
        return result


def load_ref2021_bundle(
    root: str | os.PathLike[str], *, require_features: bool = True
) -> REF2021Bundle:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "REF2021 requires pyarrow; install plench/requirements-ref2021.txt"
        ) from exc

    dataset_root = _resolve_root(root)
    processed = dataset_root / "processed"
    metadata_path = processed / "metadata.json"
    instances_path = processed / "instances.parquet"
    bags_path = processed / "bags.parquet"
    features_path = dataset_root / "features" / "features.npy"
    feature_manifest_path = dataset_root / "features" / "feature_manifest.json"
    required = (metadata_path, instances_path, bags_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "REF2021 preprocessing is incomplete; missing: " + ", ".join(missing)
        )
    if require_features and not features_path.is_file():
        raise FileNotFoundError(
            f"missing {features_path}; run python -m plench.scripts.extract_ref2021_features"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("class_names") != CLASS_NAMES:
        raise ValueError("REF2021 class order does not match the benchmark contract")
    instances = pq.read_table(instances_path)
    bags_table = pq.read_table(bags_path)
    feature_indices = np.asarray(
        instances["feature_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    if not np.array_equal(feature_indices, np.arange(len(feature_indices))):
        raise ValueError("REF2021 feature_index must be contiguous and row-aligned")
    instance_ids = np.asarray(
        instances["instance_id"].combine_chunks().to_pylist(), dtype=object
    )
    instance_bag_ids = np.asarray(
        instances["bag_id"].combine_chunks().to_pylist(), dtype=object
    )
    instance_weights = np.asarray(
        instances["instance_weight"].combine_chunks().to_numpy(), dtype=np.float32
    )
    if (instance_weights <= 0).any() or not np.isfinite(instance_weights).all():
        raise ValueError("REF2021 instance weights must be finite and positive")

    if require_features:
        features = np.load(features_path, mmap_mode="r")
        if features.ndim != 2 or len(features) != len(instance_ids):
            raise ValueError("REF2021 feature cache shape disagrees with instances")
        if not np.isfinite(features).all():
            raise ValueError("REF2021 feature cache contains NaN or Inf")
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if int(feature_manifest["instances"]) != len(instance_ids):
            raise ValueError("REF2021 feature manifest instance count is stale")
        metadata = dict(metadata)
        metadata["feature_manifest"] = feature_manifest
    else:
        features = np.empty((len(instance_ids), 0), dtype=np.float32)

    bag_ids = bags_table["bag_id"].combine_chunks().to_pylist()
    names = bags_table["institution_name"].combine_chunks().to_pylist()
    splits = bags_table["split"].combine_chunks().to_pylist()
    proportions = _fixed_list_matrix(
        bags_table, "class_proportions", np.float32
    )
    inferred_counts = _fixed_list_matrix(
        bags_table, "inferred_class_counts", np.int64
    )
    expected_sizes = np.asarray(
        bags_table["n_instances"].combine_chunks().to_numpy(), dtype=np.int64
    )
    result_bags: list[REF2021Bag] = []
    seen_instances: set[int] = set()
    for row, bag_id_value in enumerate(bag_ids):
        bag_id = str(bag_id_value)
        indices = np.flatnonzero(instance_bag_ids == bag_id)
        if len(indices) != int(expected_sizes[row]):
            raise ValueError(
                f"{bag_id}: instances={len(indices)} but bags.parquet says {expected_sizes[row]}"
            )
        overlap = seen_instances.intersection(int(value) for value in indices)
        if overlap:
            raise ValueError(f"instances belong to more than one REF2021 bag: {sorted(overlap)[:5]}")
        seen_instances.update(int(value) for value in indices)
        vector = proportions[row]
        if (vector < 0).any() or not np.isclose(vector.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{bag_id}: invalid official proportion vector {vector}")
        result_bags.append(
            REF2021Bag(
                bag_id=bag_id,
                institution_name=str(names[row]),
                indices=indices,
                proportions=vector.copy(),
                inferred_counts=inferred_counts[row].copy(),
                split=str(splits[row]),
            )
        )
    if len(seen_instances) != len(instance_ids):
        raise ValueError("some REF2021 instances do not belong to a constructed bag")
    split_sets = {
        split: {bag.bag_id for bag in result_bags if bag.split == split}
        for split in ("train", "val", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    return REF2021Bundle(
        dataset_root=dataset_root,
        features=features,
        instance_ids=instance_ids,
        instance_weights=instance_weights,
        bags=result_bags,
        metadata=metadata,
    )


class REF2021UOA11Dataset(Dataset):
    """One item is one complete natural submission bag.

    Optional training subsampling is without replacement and never changes the
    official Outputs Profile. Validation and test datasets reject subsampling.
    """

    def __init__(
        self,
        bundle: REF2021Bundle,
        split: str,
        *,
        seed: int = 42,
        train_instance_sample_size: Optional[int] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("REF2021 split must be train, val, or test")
        if split != "train" and train_instance_sample_size is not None:
            raise ValueError("REF2021 evaluation always uses complete natural bags")
        if train_instance_sample_size is not None and int(train_instance_sample_size) <= 0:
            raise ValueError("train_instance_sample_size must be positive or null")
        self.bundle = bundle
        self.split = split
        self.bags = bundle.bags_for_split(split)
        self.seed = int(seed)
        self.train_instance_sample_size = (
            None if train_instance_sample_size is None else int(train_instance_sample_size)
        )
        self.epoch = 0
        self.label_prob = [bag.proportions.tolist() for bag in self.bags]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.bags)

    def _indices(self, bag: REF2021Bag) -> np.ndarray:
        size = self.train_instance_sample_size
        if size is None or len(bag.indices) <= size:
            return bag.indices
        digest = hashlib.blake2b(
            f"{self.seed}:{self.epoch}:{bag.bag_id}".encode(), digest_size=8
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        selected = rng.choice(bag.indices, size=size, replace=False)
        return np.sort(selected.astype(np.int64))

    def __getitem__(self, item: int) -> dict[str, Any]:
        bag = self.bags[item]
        indices = self._indices(bag)
        features = np.asarray(self.bundle.features[indices], dtype=np.float32).copy()
        if not np.isfinite(features).all():
            raise ValueError(f"{bag.bag_id}: non-finite feature values")
        return {
            "x": torch.from_numpy(features),
            "proportion": torch.from_numpy(bag.proportions.copy()),
            "bag_id": bag.bag_id,
            "instance_ids": self.bundle.instance_ids[indices].tolist(),
            "instance_weights": torch.from_numpy(
                self.bundle.instance_weights[indices].copy()
            ),
            "inferred_class_counts": torch.from_numpy(
                bag.inferred_counts.copy()
            ),
        }


def collate_ref2021_bags(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty REF2021 batch")
    sizes = torch.tensor([len(item["x"]) for item in items], dtype=torch.long)
    if (sizes <= 0).any():
        raise ValueError("REF2021 contains an empty bag")
    return {
        "x": torch.cat([item["x"] for item in items], dim=0),
        "bag_index": torch.repeat_interleave(
            torch.arange(len(items), dtype=torch.long), sizes
        ),
        "bag_sizes": sizes,
        "proportion": torch.stack([item["proportion"] for item in items]),
        "bag_ids": [str(item["bag_id"]) for item in items],
        "instance_ids": [list(item["instance_ids"]) for item in items],
        "instance_weights": torch.cat(
            [item["instance_weights"] for item in items]
        ),
        "inferred_class_counts": torch.stack(
            [item["inferred_class_counts"] for item in items]
        ),
    }


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _print_split(prefix: str, dataset: REF2021UOA11Dataset) -> None:
    sizes = np.asarray([len(bag.indices) for bag in dataset.bags], dtype=np.int64)
    print(
        f"{prefix}: bags={len(sizes)} instances={int(sizes.sum())} "
        f"size min/median/mean/max={int(sizes.min())}/{float(np.median(sizes)):.1f}/"
        f"{float(sizes.mean()):.2f}/{int(sizes.max())}"
    )


def build_ref2021_loaders(
    root: str,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 0,
    train_instance_sample_size: Optional[int] = None,
):
    bundle = load_ref2021_bundle(root)
    train_dataset = REF2021UOA11Dataset(
        bundle,
        "train",
        seed=seed,
        train_instance_sample_size=train_instance_sample_size,
    )
    val_dataset = REF2021UOA11Dataset(bundle, "val", seed=seed)
    _print_split("REF2021 UoA11 train", train_dataset)
    _print_split("REF2021 UoA11 val", val_dataset)
    generator = torch.Generator().manual_seed(int(seed))
    common = dict(
        batch_size=max(1, int(batch_size)),
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_ref2021_bags,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, bundle, bundle.feature_dim


def build_ref2021_eval_loader(
    root: str,
    *,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    bundle = load_ref2021_bundle(root)
    dataset = REF2021UOA11Dataset(bundle, split)
    _print_split(f"REF2021 UoA11 {split}", dataset)
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        collate_fn=collate_ref2021_bags,
    )


def _weighted_bag_means(
    values: torch.Tensor,
    bag_index: torch.Tensor,
    weights: torch.Tensor,
    number_of_bags: int,
) -> torch.Tensor:
    result = torch.zeros(
        (number_of_bags, values.shape[1]), dtype=values.dtype, device=values.device
    )
    weighted = values * weights[:, None]
    result.scatter_add_(0, bag_index[:, None].expand_as(weighted), weighted)
    denominator = torch.zeros(number_of_bags, dtype=values.dtype, device=values.device)
    denominator.scatter_add_(0, bag_index, weights)
    return result / denominator[:, None].clamp_min(1e-12)


def _bag_slices(bag_index: torch.Tensor, number_of_bags: int) -> list[torch.Tensor]:
    return [
        torch.nonzero(bag_index == bag, as_tuple=False).squeeze(1)
        for bag in range(number_of_bags)
    ]


def _largest_remainder_counts(proportion: torch.Tensor, total: int) -> torch.Tensor:
    raw = proportion * int(total)
    counts = raw.floor().to(torch.long)
    remainder = int(total) - int(counts.sum().item())
    if remainder:
        order = torch.argsort(raw - counts.to(raw), descending=True, stable=True)
        counts = counts.clone()
        counts[order[:remainder]] += 1
    return counts


def _ordered_factorial_moment_loss(
    probabilities: torch.Tensor,
    counts: torch.Tensor,
    order: int,
    max_tuple_samples: Optional[int] = None,
) -> torch.Tensor:
    """Ordered r-draw-without-replacement moment loss for one bag.

    All class tuples are used when ``classes ** order`` fits the configured
    budget. For genuinely high-dimensional moments, uniformly sampled ordered
    class tuples give an unbiased Monte Carlo estimate of the full mean-squared
    moment objective without materializing millions of tuples.
    """
    instances, classes = probabilities.shape
    if order < 1 or order > instances:
        raise ValueError("factorial moment order must be in [1, bag_size]")
    total_tuples = int(classes) ** int(order)
    if max_tuple_samples is not None and total_tuples > int(max_tuple_samples):
        if int(max_tuple_samples) <= 0:
            raise ValueError("max_tuple_samples must be positive or null")
        tuples = torch.randint(
            classes,
            (int(max_tuple_samples), order),
            device=probabilities.device,
        )
    else:
        tuples = torch.cartesian_prod(
            *[
                torch.arange(classes, device=probabilities.device)
                for _ in range(order)
            ]
        )
        if order == 1:
            tuples = tuples.reshape(-1, 1)
    tuple_count = len(tuples)
    states = 1 << order
    dp = probabilities.new_zeros((tuple_count, states))
    dp[:, 0] = 1.0
    source_masks: list[int] = []
    destination_masks: list[int] = []
    positions: list[int] = []
    for mask in range(states):
        for position in range(order):
            if mask & (1 << position):
                continue
            source_masks.append(mask)
            destination_masks.append(mask | (1 << position))
            positions.append(position)
    source = torch.tensor(source_masks, dtype=torch.long, device=probabilities.device)
    destination = torch.tensor(
        destination_masks, dtype=torch.long, device=probabilities.device
    )
    tuple_positions = torch.tensor(
        positions, dtype=torch.long, device=probabilities.device
    )
    for row in probabilities:
        contributions = dp[:, source] * row[tuples[:, tuple_positions]]
        updated = dp.clone()
        updated.scatter_add_(
            1, destination[None, :].expand(tuple_count, -1), contributions
        )
        dp = updated
    denominator = math.prod(range(instances - order + 1, instances + 1))
    predicted = dp[:, -1] / float(denominator)
    target = probabilities.new_ones(tuple_count)
    used = torch.zeros(
        (tuple_count, classes), dtype=torch.long, device=probabilities.device
    )
    counts_device = counts.to(device=probabilities.device, dtype=torch.long)
    for position in range(order):
        label = tuples[:, position]
        available = counts_device[label] - used.gather(1, label[:, None]).squeeze(1)
        target = target * available.clamp_min(0).to(target)
        used.scatter_add_(1, label[:, None], torch.ones_like(label[:, None]))
    target = target / float(denominator)
    return F.mse_loss(predicted, target)


def _llp_mm_variable_loss(
    probabilities: torch.Tensor,
    proportions: torch.Tensor,
    slices: Sequence[torch.Tensor],
    order: int,
    *,
    order_weights: Optional[Sequence[float]] = None,
    loss_type: str = "ce",
    moment_algorithm: str = "stable_dp",
    compute_dtype: str = "float64",
    ce_smoothing_tau: float = 1e-4,
) -> torch.Tensor:
    """Exact multi-order count-profile CE in cancellation-free float64 DP.

    The paper uses cross-entropy for every method and assigns each order the
    weight ``1 / order``.  Ordered-tuple sampling and MSE are intentionally not
    supported by this training path: both silently weaken high-order signals.
    Bag class counts are reconstructed solely from the observed proportions.
    """
    if loss_type != "ce":
        raise ValueError("LLP-MM requires cross-entropy loss")
    if moment_algorithm != "stable_dp":
        raise ValueError("LLP-MM requires exact stable_dp moment computation")
    if compute_dtype != "float64":
        raise ValueError("LLP-MM requires float64 moment computation")
    if not math.isclose(float(ce_smoothing_tau), 1e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("LLP-MM requires the paper CE smoothing tau=1e-4")
    if probabilities.dtype != torch.float64:
        raise ValueError("LLP-MM probabilities must be computed in float64")
    if not slices:
        raise ValueError("LLP-MM received no bags")

    # A natural patient can contain fewer cells than the configured order.
    # Keep it intact and normalize the weights over its feasible orders.
    # Ordinary bags retain the existing objective and configured weights.
    if any(len(indices) < int(order) for indices in slices):
        configured_weights = (
            [1.0 / float(order)] * int(order)
            if order_weights is None else list(order_weights)
        )
        per_bag = []
        for bag_number, indices in enumerate(slices):
            effective_order = min(int(order), len(indices))
            active_weights = configured_weights[:effective_order]
            if effective_order < int(order):
                active_total = sum(active_weights)
                if active_total <= 0:
                    raise ValueError('No positive weight for the orders feasible in a short bag')
                active_weights = [value * sum(configured_weights) / active_total
                                  for value in active_weights]
            per_bag.append(_llp_mm_variable_loss(
                probabilities[indices], proportions[bag_number:bag_number + 1],
                [torch.arange(len(indices), device=probabilities.device)],
                effective_order, order_weights=active_weights,
                loss_type=loss_type, moment_algorithm=moment_algorithm,
                compute_dtype=compute_dtype, ce_smoothing_tau=ce_smoothing_tau,
            ))
        return torch.stack(per_bag).mean()

    try:
        from mo_matching.llp.structured_multiclass import (
            variable_multiclass_llp_mm_loss,
        )
    except ModuleNotFoundError:
        # The repository is also runnable directly without an editable install.
        from src.mo_matching.llp.structured_multiclass import (
            variable_multiclass_llp_mm_loss,
        )

    bag_sizes = torch.tensor(
        [len(indices) for indices in slices],
        dtype=torch.long,
        device=probabilities.device,
    )
    bag_index = torch.empty(
        len(probabilities), dtype=torch.long, device=probabilities.device
    )
    counts: list[torch.Tensor] = []
    for bag_number, indices in enumerate(slices):
        bag_index[indices] = bag_number
        counts.append(
            _largest_remainder_counts(proportions[bag_number], len(indices))
        )
    class_counts = torch.stack(counts).to(
        device=probabilities.device, dtype=torch.long
    )
    strict_proportions = proportions.to(
        device=probabilities.device, dtype=torch.float64
    )
    weights = (
        [1.0 / float(order)] * int(order)
        if order_weights is None
        else list(order_weights)
    )
    return variable_multiclass_llp_mm_loss(
        probabilities,
        bag_index,
        bag_sizes,
        class_counts,
        strict_proportions,
        int(order),
        order_weights=weights,
        moment_algorithm=moment_algorithm,
        loss_type=loss_type,
        ce_smoothing_tau=float(ce_smoothing_tau),
    )


def ref2021_loss(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, torch.Tensor],
    *,
    logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Variable-natural-bag loss adapter around existing PLeNCH algorithms."""
    if algorithm_name not in SUPPORTED_REF_METHODS:
        raise NotImplementedError(
            f"{algorithm_name} has no verified variable-natural-bag REF adapter. "
            f"Verified methods: {sorted(SUPPORTED_REF_METHODS)}"
        )
    if logits is None:
        logits = algorithm.predict(batch["x"])
    proportions = batch["proportion"].to(logits)
    bag_index = batch["bag_index"]
    sizes = batch["bag_sizes"].to(logits)
    weights = batch["instance_weights"].to(logits)
    number_of_bags = len(proportions)
    if algorithm_name == "LLP_MM":
        strict_probabilities = logits.to(torch.float64).softmax(dim=1).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        return _llp_mm_variable_loss(
            strict_probabilities,
            proportions,
            _bag_slices(bag_index, number_of_bags),
            int(getattr(algorithm, "order", 3)),
            order_weights=getattr(algorithm, "order_weights", None),
            loss_type=getattr(algorithm, "moment_loss_type", "ce"),
            moment_algorithm=getattr(algorithm, "moment_algorithm", "stable_dp"),
            compute_dtype=getattr(algorithm, "moment_compute_dtype", "float64"),
            ce_smoothing_tau=getattr(
                algorithm, "moment_ce_smoothing_tau", 1e-4
            ),
        )
    probabilities = logits.softmax(dim=1).clamp_min(1e-12)
    means = _weighted_bag_means(
        probabilities, bag_index, weights, number_of_bags
    ).clamp_min(1e-12)
    if algorithm_name == "PM":
        return -(proportions * means.log()).sum(dim=1).mean()
    if algorithm_name == "LLP_DSQ":
        return algorithm.DSQ_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
            update_ema=bool(algorithm.training),
        )
    slices = _bag_slices(bag_index, number_of_bags)
    if algorithm_name == "LLP_PVC":
        return algorithm.PVC_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
        )
    if algorithm_name == "LLP_PT":
        return algorithm.PT_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
        )
    if algorithm_name == "LLP_FC":
        return algorithm.FC_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
        )
    if algorithm_name == "ROT":
        losses = []
        original = algorithm.bagsize
        try:
            for bag_number, indices in enumerate(slices):
                algorithm.bagsize = len(indices)
                loss, _ = algorithm.ROTLoss_Loss(
                    logits[indices], proportions[bag_number : bag_number + 1]
                )
                losses.append(loss)
        finally:
            algorithm.bagsize = original
        return torch.stack(losses).mean()
    prior = (proportions * sizes[:, None]).sum(dim=0) / sizes.sum()
    if algorithm_name == "EasyLLP":
        loss, _ = algorithm.EasyLLP_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
        )
        if getattr(algorithm, "flooding", False):
            b = float(getattr(algorithm, "flooding_b", 0.0))
            loss = (loss - b).abs() + b
        return loss
    if algorithm_name == "GeneralUPM":
        if number_of_bags < 2:
            raise ValueError("REF2021 GeneralUPM requires batchsize >= 2 natural bags")
        ell = -F.log_softmax(logits, dim=1)
        bag_sums = torch.zeros_like(proportions)
        bag_sums.scatter_add_(0, bag_index[:, None].expand_as(ell), ell)
        total_sum = bag_sums.sum(dim=0)
        outside_sizes = sizes.sum() - sizes
        outside_means = (total_sum[None, :] - bag_sums) / outside_sizes[:, None]
        centered = (proportions - prior[None, :]) * (
            bag_sums - sizes[:, None] * outside_means
        )
        base = (prior[None, :] * outside_means).sum(dim=1)
        loss = (centered.sum(dim=1) + base).mean()
        if getattr(algorithm, "flooding", False):
            b = float(getattr(algorithm, "flooding_b", 0.0))
            loss = (loss - b).abs() + b
        return loss
    if algorithm_name == "NonClipOVR":
        return algorithm.NonClipOVR_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
            leave_one_bag_out=True,
        )
    raise AssertionError("unreachable REF2021 algorithm branch")


def update_ref2021_algorithm(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, Any],
    device: str | torch.device,
) -> dict[str, float]:
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    loss = ref2021_loss(algorithm, algorithm_name, moved)
    return algorithm._backward_step(loss)


@torch.no_grad()
def evaluate_ref2021(
    algorithm: Any,
    loader: DataLoader,
    device: str | torch.device,
) -> dict[str, Any]:
    algorithm.eval()
    predicted: list[np.ndarray] = []
    target: list[np.ndarray] = []
    bag_ids: list[str] = []
    for batch in loader:
        x = batch["x"].to(device)
        bag_index = batch["bag_index"].to(device)
        weights = batch["instance_weights"].to(device)
        probabilities = algorithm.predict(x).softmax(dim=1)
        means = _weighted_bag_means(
            probabilities, bag_index, weights, len(batch["proportion"])
        )
        predicted.append(means.cpu().numpy())
        target.append(batch["proportion"].numpy())
        bag_ids.extend(str(value) for value in batch["bag_ids"])
    if not predicted:
        raise ValueError("REF2021 evaluation loader is empty")
    prediction = np.concatenate(predicted).astype(np.float64)
    truth = np.concatenate(target).astype(np.float64)
    prediction = prediction / prediction.sum(axis=1, keepdims=True)
    truth = truth / truth.sum(axis=1, keepdims=True)
    eps = 1e-12
    midpoint = 0.5 * (prediction + truth)
    kl = np.sum(truth * (np.log(truth + eps) - np.log(prediction + eps)), axis=1)
    js = 0.5 * np.sum(
        truth * (np.log(truth + eps) - np.log(midpoint + eps)), axis=1
    ) + 0.5 * np.sum(
        prediction * (np.log(prediction + eps) - np.log(midpoint + eps)), axis=1
    )
    absolute = np.abs(prediction - truth)
    result = {
        "bags": int(len(truth)),
        "bag_proportion_mae": float(absolute.mean()),
        "bag_proportion_rmse": float(np.sqrt(np.square(prediction - truth).mean())),
        "bag_proportion_l1": float(absolute.sum(axis=1).mean()),
        "bag_proportion_kl": float(kl.mean()),
        "bag_proportion_js": float(js.mean()),
        "ordinal_emd": float(
            np.abs(np.cumsum(prediction, axis=1)[:, :-1] - np.cumsum(truth, axis=1)[:, :-1]).sum(axis=1).mean()
        ),
        "per_class_mae": {
            name: float(absolute[:, index].mean())
            for index, name in enumerate(CLASS_NAMES)
        },
        "bag_ids": bag_ids,
    }
    algorithm.train()
    return result
