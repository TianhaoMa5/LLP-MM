"""Ardehaly--Culotta (2017) Twitter Race3 data support for PLeNCH.

This module deliberately contains no downloader for Twitter/X.  It consumes a
small canonical manifest so that an author-provided historical release can be
used without replacing it with newly scraped users.  The default representation
is a frozen, 2048-dimensional ImageNet Xception feature per Twitter user.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATASET_NAME = "TwitterEthnicity2017"
DATASET_ALIASES = {
    "twitterethnicity2017",
    "twitter_ethnicity_2017",
    "twitter-race3-2017",
    "twitter_race3_2017",
}
CLASS_NAMES = ("White", "Black", "Hispanic")
CLASS_TO_INDEX = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
FEATURE_DIM = 2048
PROPORTION_COLUMNS = ("p_white", "p_black", "p_hispanic")
TRAIN_SPLITS = {"train", "training"}
EVAL_SPLITS = {"evaluation", "eval", "test"}


def canonical_twitter_dataset(dataset: str) -> str:
    if dataset == DATASET_NAME or dataset.strip().lower() in DATASET_ALIASES:
        return DATASET_NAME
    raise ValueError(
        f"Unknown Twitter demographic dataset {dataset!r}; expected {DATASET_NAME!r}"
    )


def is_twitter_ethnicity_dataset(dataset: str) -> bool:
    try:
        canonical_twitter_dataset(dataset)
    except ValueError:
        return False
    return True


def _resolve_dataset_root(root: str | os.PathLike[str]) -> Path:
    root_path = Path(root).expanduser().resolve()
    candidates = (
        root_path,
        root_path / "twitter_ethnicity_2017",
        root_path / "TwitterEthnicity2017",
    )
    for candidate in candidates:
        if (candidate / "processed" / "manifest.csv").is_file():
            return candidate
        if (candidate / "manifest.csv").is_file():
            return candidate
    expected = root_path / "twitter_ethnicity_2017"
    raise FileNotFoundError(
        "DATA_MISSING: TwitterEthnicity2017 has not been prepared. Expected\n"
        f"  {expected / 'processed' / 'manifest.csv'}\n"
        f"  {expected / 'processed' / 'county_proportions.csv'}\n"
        f"  {expected / 'processed' / 'xception_features.npy'}\n"
        f"  {expected / 'processed' / 'instance_ids.npy'}\n"
        "See plench/docs/twitter_ethnicity_2017.md for the raw file schemas."
    )


def _processed_dir(dataset_root: Path) -> Path:
    nested = dataset_root / "processed"
    return nested if (nested / "manifest.csv").is_file() else dataset_root


def _load_string_array(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1:
        raise ValueError(f"{path.name} must be a one-dimensional array")
    return values.astype(str)


def _normalize_split(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in TRAIN_SPLITS:
        return "train"
    if lowered in EVAL_SPLITS:
        return "evaluation"
    raise ValueError(
        f"unknown manifest split {value!r}; use train or evaluation"
    )


def parse_race3_label(value: Any, *, allow_missing: bool) -> int:
    if pd.isna(value) or str(value).strip() == "":
        if allow_missing:
            return -1
        raise ValueError("evaluation label is missing")
    text = str(value).strip().lower()
    if text in CLASS_TO_INDEX:
        return CLASS_TO_INDEX[text]
    try:
        label = int(float(text))
    except ValueError as exc:
        raise ValueError(
            f"invalid Race3 label {value!r}; expected 0/1/2 or White/Black/Hispanic"
        ) from exc
    if label not in range(len(CLASS_NAMES)):
        raise ValueError(f"Race3 label must be in [0, 2], got {label}")
    return label


@dataclass(frozen=True)
class TwitterEthnicityBundle:
    dataset_root: Path
    processed_dir: Path
    manifest: pd.DataFrame
    features: Optional[np.ndarray]
    county_proportions: dict[str, np.ndarray]
    normalization_applied: bool

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    @property
    def feature_dim(self) -> int:
        if self.features is None:
            return FEATURE_DIM
        return int(self.features.shape[1])

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.manifest["split"].to_numpy() == "train")

    @property
    def evaluation_indices(self) -> np.ndarray:
        return np.flatnonzero(self.manifest["split"].to_numpy() == "evaluation")


def load_twitter_ethnicity_bundle(
    root: str | os.PathLike[str],
    *,
    require_features: bool = True,
) -> TwitterEthnicityBundle:
    dataset_root = _resolve_dataset_root(root)
    processed = _processed_dir(dataset_root)
    manifest_path = processed / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"instance_id": str, "county_fips": str})
    required = {"instance_id", "county_fips", "image_path", "split", "label"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"manifest.csv is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("manifest.csv is empty")
    manifest["instance_id"] = manifest["instance_id"].astype(str).str.strip()
    if (manifest["instance_id"] == "").any() or manifest["instance_id"].duplicated().any():
        raise ValueError("manifest instance_id values must be non-empty and unique")
    manifest["split"] = manifest["split"].map(_normalize_split)
    manifest["county_fips"] = manifest["county_fips"].fillna("").astype(str).str.strip()
    train_mask = manifest["split"].eq("train")
    if not train_mask.any():
        raise ValueError("manifest contains no training users")
    if (manifest.loc[train_mask, "county_fips"] == "").any():
        raise ValueError("every training user must have county_fips")
    labels = [
        parse_race3_label(value, allow_missing=split == "train")
        for value, split in zip(manifest["label"], manifest["split"])
    ]
    manifest["label"] = np.asarray(labels, dtype=np.int64)
    if (manifest.loc[train_mask, "label"] >= 0).any():
        raise ValueError(
            "training rows must not contain instance labels; evaluation users belong in split=evaluation"
        )

    county_path = processed / "county_proportions.csv"
    if not county_path.is_file():
        raise FileNotFoundError(f"DATA_MISSING: {county_path}")
    counties = pd.read_csv(county_path, dtype={"county_fips": str})
    county_required = {"county_fips", *PROPORTION_COLUMNS}
    county_missing = sorted(county_required.difference(counties.columns))
    if county_missing:
        raise ValueError(f"county_proportions.csv is missing columns: {county_missing}")
    counties["county_fips"] = counties["county_fips"].astype(str).str.strip()
    if counties["county_fips"].duplicated().any():
        raise ValueError("county_proportions.csv contains duplicate county_fips values")
    matrix = counties.loc[:, PROPORTION_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("county Race3 proportions must be finite and non-negative")
    sums = matrix.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError("every county must have positive Race3 proportion mass")

    metadata_path = processed / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    normalization_applied = bool(metadata.get("normalize_race3_proportions", False))
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=0):
        raise ValueError(
            "county Race3 target proportions do not sum to one. Re-run the preparer with "
            "--normalize-race3-proportions; normalization is never applied silently."
        )
    used_counties = set(manifest.loc[train_mask, "county_fips"])
    available_counties = set(counties["county_fips"])
    absent = sorted(used_counties.difference(available_counties))
    if absent:
        raise ValueError(f"training users reference counties without proportions: {absent[:10]}")
    proportion_lookup = {
        str(county): row.astype(np.float32)
        for county, row in zip(counties["county_fips"], matrix)
    }

    feature_path = processed / "xception_features.npy"
    ids_path = processed / "instance_ids.npy"
    features: Optional[np.ndarray] = None
    if feature_path.is_file() or ids_path.is_file():
        if not feature_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                "xception_features.npy and instance_ids.npy must either both exist or both be absent"
            )
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        instance_ids = _load_string_array(ids_path)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"xception_features.npy must have shape [N, {FEATURE_DIM}], got {features.shape}"
            )
        if len(features) != len(manifest) or len(instance_ids) != len(manifest):
            raise ValueError("manifest, features, and instance_ids lengths differ")
        if not np.array_equal(instance_ids, manifest["instance_id"].to_numpy(dtype=str)):
            raise ValueError("instance_ids.npy is not exactly aligned with manifest.csv")
        for start in range(0, len(features), 4096):
            if not np.isfinite(np.asarray(features[start : start + 4096])).all():
                raise ValueError("xception_features.npy contains NaN or infinity")
    elif require_features:
        raise FileNotFoundError(
            "DATA_MISSING: precomputed mode requires xception_features.npy and instance_ids.npy. "
            "Run plench.scripts.extract_twitter_xception_features after supplying real images."
        )

    if "twitter_user_id" in manifest.columns:
        train_users = set(
            manifest.loc[train_mask, "twitter_user_id"].dropna().astype(str)
        )
        eval_users = set(
            manifest.loc[~train_mask, "twitter_user_id"].dropna().astype(str)
        )
        overlap = sorted(train_users.intersection(eval_users))
        if overlap:
            raise ValueError(f"training/evaluation Twitter user overlap: {overlap[:10]}")

    return TwitterEthnicityBundle(
        dataset_root=dataset_root,
        processed_dir=processed,
        manifest=manifest.reset_index(drop=True),
        features=features,
        county_proportions=proportion_lookup,
        normalization_applied=normalization_applied,
    )


@dataclass(frozen=True)
class CountyBag:
    bag_id: str
    county_fips: str
    indices: np.ndarray
    proportions: np.ndarray


def _county_groups(bundle: TwitterEthnicityBundle, indices: Iterable[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index in indices:
        county = str(bundle.manifest.iloc[int(index)]["county_fips"])
        groups.setdefault(county, []).append(int(index))
    return groups


def build_county_bags(
    bundle: TwitterEthnicityBundle,
    indices: Sequence[int] | np.ndarray,
    *,
    max_bag_size: Optional[int],
    seed: int,
) -> list[CountyBag]:
    if max_bag_size is not None and int(max_bag_size) <= 0:
        raise ValueError("max_bag_size must be positive or None")
    bags: list[CountyBag] = []
    for county, county_indices in sorted(_county_groups(bundle, indices).items()):
        shuffled = np.asarray(county_indices, dtype=np.int64)
        rng = np.random.default_rng(int(seed) + int.from_bytes(county.encode("utf-8"), "little") % (2**31))
        rng.shuffle(shuffled)
        chunk_size = len(shuffled) if max_bag_size is None else int(max_bag_size)
        for chunk_index, start in enumerate(range(0, len(shuffled), chunk_size)):
            chunk = shuffled[start : start + chunk_size]
            bags.append(
                CountyBag(
                    bag_id=f"county:{county}:chunk:{chunk_index:03d}",
                    county_fips=county,
                    indices=chunk,
                    proportions=bundle.county_proportions[county].copy(),
                )
            )
    if not bags:
        raise ValueError("no county bags were constructed")
    return bags


def bag_diagnostics(bags: Sequence[CountyBag]) -> dict[str, Any]:
    sizes = np.asarray([len(bag.indices) for bag in bags], dtype=np.int64)
    proportions = np.stack([bag.proportions for bag in bags])
    return {
        "instances": int(sizes.sum()),
        "bags": int(len(bags)),
        "bag_size": {
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "std": float(sizes.std()),
            "q25": float(np.quantile(sizes, 0.25)),
            "q75": float(np.quantile(sizes, 0.75)),
            "q90": float(np.quantile(sizes, 0.90)),
        },
        "mean_county_proportions": {
            name: float(value) for name, value in zip(CLASS_NAMES, proportions.mean(axis=0))
        },
    }


def _print_bag_diagnostics(prefix: str, bags: Sequence[CountyBag]) -> None:
    report = bag_diagnostics(bags)
    sizes = report["bag_size"]
    print(
        f"{prefix}: instances={report['instances']} bags={report['bags']} "
        f"size min/mean/median/std/q90/max={sizes['min']}/{sizes['mean']:.2f}/"
        f"{sizes['median']:.2f}/{sizes['std']:.2f}/{sizes['q90']:.2f}/{sizes['max']}"
    )
    print(f"{prefix}: mean Census Race3 proportions={report['mean_county_proportions']}")


def _image_path(bundle: TwitterEthnicityBundle, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    raw_candidate = bundle.dataset_root / "raw" / path
    return raw_candidate if raw_candidate.is_file() else bundle.dataset_root / path


class TwitterCountyBagDataset(Dataset):
    """One item is one natural county bag (or a within-county max-size chunk)."""

    def __init__(
        self,
        bundle: TwitterEthnicityBundle,
        bags: Sequence[CountyBag],
        *,
        mode: str,
        representation: str = "precomputed_feature",
    ) -> None:
        if representation not in {"precomputed_feature", "end_to_end"}:
            raise ValueError("representation must be precomputed_feature or end_to_end")
        if representation == "precomputed_feature" and bundle.features is None:
            raise FileNotFoundError("precomputed_feature mode requires Xception features")
        self.bundle = bundle
        self.bags = list(bags)
        self.mode = mode
        self.representation = representation
        self.label_prob = [bag.proportions.tolist() for bag in self.bags]
        self._train_image_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.bags)

    def _instances(self, bag: CountyBag) -> torch.Tensor:
        if self.representation == "precomputed_feature":
            assert self.bundle.features is not None
            values = np.asarray(self.bundle.features[bag.indices], dtype=np.float32)
            return torch.from_numpy(values.copy())
        images = []
        for index in bag.indices:
            path = _image_path(self.bundle, self.bundle.manifest.iloc[int(index)]["image_path"])
            if not path.is_file():
                raise FileNotFoundError(f"DATA_MISSING: profile image {path}")
            with Image.open(path) as image:
                images.append(self._train_image_transform(image.convert("RGB")))
        return torch.stack(images)

    def __getitem__(self, item: int):
        bag = self.bags[item]
        weak = self._instances(bag)
        views = [weak, weak.clone()] if self.mode == "train_u_L^2P-AHIL" else [weak]
        # The two final arrays intentionally contain no instance targets.  PLeNCH
        # keeps the legacy five-field batch contract, but Twitter train labels are
        # never loaded or returned here.
        safe_indices = bag.indices.copy()
        hidden_labels = np.full(len(bag.indices), -1, dtype=np.int64)
        return views, bag.proportions.tolist(), safe_indices, int(item), hidden_labels


class TwitterEvaluationDataset(Dataset):
    def __init__(
        self,
        bundle: TwitterEthnicityBundle,
        representation: str = "precomputed_feature",
    ) -> None:
        self.bundle = bundle
        self.indices = bundle.evaluation_indices
        self.representation = representation
        if not len(self.indices):
            raise FileNotFoundError(
                "DATA_MISSING: no evaluation rows in manifest.csv; supply the historical "
                "320-user evaluation_users.csv to report instance metrics"
            )
        if (bundle.manifest.iloc[self.indices]["label"].to_numpy() < 0).any():
            raise ValueError("every evaluation row must have a Race3 label")
        if representation == "precomputed_feature" and bundle.features is None:
            raise FileNotFoundError("evaluation precomputed_feature mode requires features")
        self._image_transform = transforms.Compose(
            [
                transforms.Resize(342),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        row = self.bundle.manifest.iloc[index]
        if self.representation == "precomputed_feature":
            assert self.bundle.features is not None
            feature = torch.as_tensor(
                np.asarray(self.bundle.features[index], dtype=np.float32).copy()
            )
        else:
            path = _image_path(self.bundle, row["image_path"])
            if not path.is_file():
                raise FileNotFoundError(f"DATA_MISSING: profile image {path}")
            with Image.open(path) as image:
                feature = self._image_transform(image.convert("RGB"))
        return feature, int(row["label"])


def _partition_counties(
    bundle: TwitterEthnicityBundle,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = bundle.train_indices
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if holdout_fraction == 0.0:
        return train_indices, train_indices
    counties = np.asarray(sorted(set(bundle.manifest.iloc[train_indices]["county_fips"])))
    if len(counties) < 2:
        raise ValueError("county-level holdout requires at least two counties")
    rng = np.random.default_rng(seed)
    rng.shuffle(counties)
    val_count = min(len(counties) - 1, max(1, int(round(len(counties) * holdout_fraction))))
    val_counties = set(counties[:val_count])
    county_values = bundle.manifest.iloc[train_indices]["county_fips"].to_numpy()
    val_mask = np.asarray([county in val_counties for county in county_values])
    return train_indices[~val_mask], train_indices[val_mask]


def build_twitter_ethnicity_loaders(
    root: str,
    max_bag_size: int,
    batch_size: int,
    *,
    seed: int = 0,
    num_bags: Optional[int] = None,
    num_workers: int = 0,
    method: str = "DLLP",
    holdout_fraction: float = 0.0,
    representation: str = "precomputed_feature",
):
    bundle = load_twitter_ethnicity_bundle(
        root, require_features=representation == "precomputed_feature"
    )
    train_indices, val_indices = _partition_counties(bundle, holdout_fraction, seed)
    train_bags = build_county_bags(
        bundle, train_indices, max_bag_size=max_bag_size, seed=seed
    )
    val_bags = build_county_bags(
        bundle, val_indices, max_bag_size=max_bag_size, seed=seed + 1
    )
    if num_bags is not None:
        train_bags = train_bags[: int(num_bags)]
        if not train_bags:
            raise ValueError("num_bags selected zero Twitter county bags")
    _print_bag_diagnostics("TwitterEthnicity2017 train", train_bags)
    _print_bag_diagnostics("TwitterEthnicity2017 val", val_bags)
    if batch_size != 1:
        print(
            "TwitterEthnicity2017 uses DataLoader batch_size=1 because natural county "
            "bags and their final chunks have variable sizes."
        )
    train_dataset = TwitterCountyBagDataset(
        bundle, train_bags, mode=f"train_u_{method}", representation=representation
    )
    val_dataset = TwitterCountyBagDataset(
        bundle, val_bags, mode="train_x", representation=representation
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": 1,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    input_shape: int | tuple[int, int, int] = (
        FEATURE_DIM if representation == "precomputed_feature" else (3, 299, 299)
    )
    return train_loader, val_loader, bundle, input_shape


def build_twitter_evaluation_loader(
    root: str,
    batch_size: int,
    *,
    num_workers: int = 0,
    representation: str = "precomputed_feature",
) -> Optional[DataLoader]:
    bundle = load_twitter_ethnicity_bundle(
        root, require_features=representation == "precomputed_feature"
    )
    if not len(bundle.evaluation_indices):
        print(
            "DATA_MISSING: historical 320-user Twitter evaluation set is absent; "
            "instance accuracy/F1 will not be reported."
        )
        return None
    dataset = TwitterEvaluationDataset(bundle, representation=representation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )


@torch.no_grad()
def evaluate_twitter_race3(network, loader: DataLoader, device: str | torch.device) -> dict[str, Any]:
    network.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for features, target in loader:
        logits = network.predict(features.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(target.numpy())
    target = np.concatenate(labels)
    prediction = np.concatenate(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        prediction,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        target, prediction, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        target, prediction, average="weighted", zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(
            target, prediction, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
    }
    for index, name in enumerate(CLASS_NAMES):
        key = name.lower()
        result[f"{key}_precision"] = float(precision[index])
        result[f"{key}_recall"] = float(recall[index])
        result[f"{key}_f1"] = float(f1[index])
        result[f"{key}_support"] = int(support[index])
    network.train()
    return result


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
