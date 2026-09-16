"""Paper-aligned Campo Verde and LEM datasets for multi-class LLP.

The current contract stores one georeferenced image stack and a catalogue of
all valid annotated patch centres.  Patches are extracted lazily and training
centres are re-sampled and re-shuffled at every epoch.  A read-only legacy
``patches.npy`` fallback is retained so old outputs remain inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PREPROCESSING_VERSION = "remote-sensing-centres-v2-paper-aligned"
MAIN_BAG_SIZES = (32, 64, 128, 256)
DEFAULT_INSTANCES_PER_EPOCH = 200_000
DEFAULT_CLUSTER_COUNT = 32
CLUSTER_FEATURE = "train_normalized_center_pixel"
REMOTE_SENSING_BACKBONE = "RemoteResNet18"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_TO_CODE = {name: index for index, name in enumerate(SPLIT_NAMES)}

REMOTE_SENSING_SPECS: Dict[str, Dict[str, Any]] = {
    "CV": {
        "aliases": {"cv", "campo_verde", "campoverde"},
        "directory": "campo_verde",
        "channels": 7,
        "patch_size": 21,
        "test_fraction": 0.50,
        "description": "Campo Verde May-2016 Landsat optical stack resampled to 10 m",
    },
    "LEM": {
        "aliases": {"lem", "luis_eduardo_magalhaes", "luís_eduardo_magalhães"},
        "directory": "lem",
        # The paper-aligned channel count is read from the preprocessing
        # manifest.  The full local Dec-Apr sequence is 12 dates x VV/VH = 24.
        "channels": None,
        "patch_size": 21,
        "test_fraction": 0.25,
        "description": "LEM multi-temporal Sentinel-1 VV/VH stack",
    },
}


def canonical_remote_dataset(dataset: str) -> str:
    key = dataset.strip()
    for canonical, spec in REMOTE_SENSING_SPECS.items():
        if key.upper() == canonical or key.lower() in spec["aliases"]:
            return canonical
    raise ValueError(f"Unknown remote-sensing dataset {dataset!r}; expected CV or LEM")


def is_remote_sensing_dataset(dataset: str) -> bool:
    try:
        canonical_remote_dataset(dataset)
    except ValueError:
        return False
    return True


def enforce_remote_sensing_backbone(dataset: str, hparams: Mapping[str, Any]) -> None:
    """Reject backbone overrides for the paper-aligned CV/LEM pipeline."""
    if not is_remote_sensing_dataset(dataset):
        return
    configured = hparams.get("model")
    if configured != REMOTE_SENSING_BACKBONE:
        raise ValueError(
            f"{canonical_remote_dataset(dataset)} requires "
            f"model={REMOTE_SENSING_BACKBONE}; got {configured!r}. The "
            "remote-sensing backbone is fixed so all algorithms use the same "
            "ResNet18 architecture."
        )


def _atomic_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def _load_json(path: Path, required: bool = False) -> Any:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_processed_dir(root: str | os.PathLike[str], dataset: str) -> Path:
    spec = REMOTE_SENSING_SPECS[canonical_remote_dataset(dataset)]
    root_path = Path(root).expanduser().resolve()
    candidates = (root_path / "processed", root_path / spec["directory"] / "processed", root_path)
    for candidate in candidates:
        new_contract = all((candidate / name).is_file() for name in (
            "image_stack.npy", "center_rows.npy", "center_cols.npy", "labels.npy",
            "field_indices.npy", "split_codes.npy",
        ))
        legacy_contract = (candidate / "patches.npy").is_file() and (candidate / "labels.npy").is_file()
        if new_contract or legacy_contract:
            return candidate
    raise FileNotFoundError(
        f"No preprocessed {dataset} data under {root_path}. Run "
        "python -m plench.scripts.prepare_remote_sensing first."
    )


def deterministic_nested_field_split(
    field_ids: Sequence[Any], seed: int, test_fraction: float,
    val_fraction_of_train: float = 0.20,
) -> Dict[str, list[Any]]:
    """One deterministic random polygon split, with validation carved from train."""
    unique = np.asarray(sorted(set(field_ids), key=str), dtype=object)
    if len(unique) < 3:
        raise ValueError("At least three logical fields are required")
    rng = np.random.default_rng(seed)
    unique = unique[rng.permutation(len(unique))]
    n_test = max(1, min(len(unique) - 2, int(round(len(unique) * test_fraction))))
    original_train = unique[n_test:]
    n_val = max(1, min(len(original_train) - 1, int(round(len(original_train) * val_fraction_of_train))))
    result = {
        "train": original_train[n_val:].tolist(),
        "val": original_train[:n_val].tolist(),
        "test": unique[:n_test].tolist(),
    }
    train_fields, val_fields, test_fields = (set(result[name]) for name in SPLIT_NAMES)
    assert train_fields.isdisjoint(val_fields)
    assert train_fields.isdisjoint(test_fields)
    assert val_fields.isdisjoint(test_fields)
    assert train_fields | val_fields | test_fields == set(field_ids)
    return result


def normalization_fingerprint(manifest: Mapping[str, Any], split_manifest: Mapping[str, Any],
                              class_mapping: Mapping[str, Any]) -> str:
    payload = {
        "preprocessing_version": manifest.get("preprocessing_version"),
        "preprocessing_hash": manifest.get("preprocessing_hash"),
        "source_files": manifest.get("source_files"),
        "selected_acquisition_dates": manifest.get("selected_acquisition_dates"),
        "selected_channels": manifest.get("selected_channels"),
        "spatial_resolution_m": manifest.get("spatial_resolution_m"),
        "grid": manifest.get("grid"),
        "split_manifest_hash": hashlib.sha256(
            json.dumps(split_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "class_mapping_hash": hashlib.sha256(
            json.dumps(class_mapping, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _coverage_weights(rows: np.ndarray, cols: np.ndarray, height: int, width: int,
                      radius: int) -> np.ndarray:
    """Number of selected centred patches covering every raster pixel."""
    diff = np.zeros((height + 1, width + 1), dtype=np.int32)
    top, left = rows - radius, cols - radius
    bottom, right = rows + radius + 1, cols + radius + 1
    np.add.at(diff, (top, left), 1)
    np.add.at(diff, (bottom, left), -1)
    np.add.at(diff, (top, right), -1)
    np.add.at(diff, (bottom, right), 1)
    return diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]


def compute_train_normalization(image_stack: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                                train_indices: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact channel moments over every training-centred patch, without materialising patches."""
    radius = patch_size // 2
    selected_rows = np.asarray(rows[train_indices], dtype=np.int64)
    selected_cols = np.asarray(cols[train_indices], dtype=np.int64)
    weights = _coverage_weights(selected_rows, selected_cols, image_stack.shape[1], image_stack.shape[2], radius)
    pixel_count = int(len(train_indices) * patch_size * patch_size)
    channel_sum = np.empty(image_stack.shape[0], dtype=np.float64)
    channel_sq_sum = np.empty(image_stack.shape[0], dtype=np.float64)
    covered = weights > 0
    covered_weights = weights[covered].astype(np.float64, copy=False)
    for channel in range(image_stack.shape[0]):
        values = np.asarray(image_stack[channel][covered], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("A training patch contains non-finite source pixels")
        channel_sum[channel] = np.sum(values * covered_weights, dtype=np.float64)
        channel_sq_sum[channel] = np.sum(np.square(values) * covered_weights, dtype=np.float64)
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sq_sum / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class RemoteSensingBundle:
    dataset: str
    processed_dir: Path
    image_stack: Optional[np.ndarray]
    patches: Optional[np.ndarray]
    rows: Optional[np.ndarray]
    cols: Optional[np.ndarray]
    labels: np.ndarray
    field_indices: np.ndarray
    field_values: list[str]
    split_indices: Dict[str, np.ndarray]
    split_strategy: str
    mean: np.ndarray
    std: np.ndarray
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    channels: int
    patch_size: int
    manifest: Dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self.channels, self.patch_size, self.patch_size

    @property
    def field_ids(self) -> np.ndarray:
        values = np.asarray(self.field_values, dtype=object)
        return values[self.field_indices]

    def extract(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self.patches is not None:
            return np.asarray(self.patches[indices], dtype=np.float32)
        assert self.image_stack is not None and self.rows is not None and self.cols is not None
        radius = self.patch_size // 2
        result = np.empty((len(indices), self.channels, self.patch_size, self.patch_size), dtype=np.float32)
        for output_index, centre_index in enumerate(indices):
            row, col = int(self.rows[centre_index]), int(self.cols[centre_index])
            result[output_index] = self.image_stack[:, row - radius:row + radius + 1,
                                                    col - radius:col + radius + 1]
        return result


def _load_new_bundle(canonical: str, processed_dir: Path) -> RemoteSensingBundle:
    manifest = _load_json(processed_dir / "preprocessing_manifest.json", required=True)
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError(
            f"Unsupported preprocessing version {manifest.get('preprocessing_version')!r}; "
            f"expected {PREPROCESSING_VERSION!r}"
        )
    image_stack = np.load(processed_dir / "image_stack.npy", mmap_mode="r", allow_pickle=False)
    rows = np.load(processed_dir / "center_rows.npy", mmap_mode="r", allow_pickle=False)
    cols = np.load(processed_dir / "center_cols.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(processed_dir / "labels.npy", mmap_mode="r", allow_pickle=False).astype(np.int64, copy=False)
    field_indices = np.load(processed_dir / "field_indices.npy", mmap_mode="r", allow_pickle=False).astype(np.int64, copy=False)
    split_codes = np.load(processed_dir / "split_codes.npy", mmap_mode="r", allow_pickle=False)
    lengths = {len(rows), len(cols), len(labels), len(field_indices), len(split_codes)}
    if len(lengths) != 1:
        raise ValueError(f"Centre catalogue length mismatch: {lengths}")
    class_mapping = _load_json(processed_dir / "class_mapping.json", required=True)
    class_to_idx = {str(key): int(value) for key, value in class_mapping["class_to_idx"].items()}
    idx_to_class = {int(key): str(value) for key, value in class_mapping["idx_to_class"].items()}
    field_values = [str(value) for value in _load_json(processed_dir / "field_values.json", required=True)]
    split_manifest = _load_json(processed_dir / "split_manifest.json", required=True)
    patch_size = int(manifest.get("patch_size", 21))
    split_indices = {
        name: np.flatnonzero(split_codes == SPLIT_TO_CODE[name]).astype(np.int64)
        for name in SPLIT_NAMES
    }
    unexpected_codes = set(np.unique(split_codes).astype(int).tolist()) - {0, 1, 2, 255}
    if unexpected_codes:
        raise ValueError(f"Unexpected split codes: {sorted(unexpected_codes)}")
    if any(len(split_indices[name]) == 0 for name in SPLIT_NAMES):
        raise ValueError("train/val/test centre sets must all be non-empty")
    split_field_indices = [set(field_indices[split_indices[name]].astype(int).tolist())
                           for name in SPLIT_NAMES]
    allowed_overlap = set(int(value) for value in
                          split_manifest.get("spatial_exception_field_indices", []))
    observed_overlap: set[int] = set()
    for left_index in range(len(SPLIT_NAMES)):
        for right_index in range(left_index + 1, len(SPLIT_NAMES)):
            overlap = split_field_indices[left_index] & split_field_indices[right_index]
            observed_overlap.update(overlap)
            if not overlap.issubset(allowed_overlap):
                raise AssertionError(
                    f"Unexpected field leakage between {SPLIT_NAMES[left_index]} and "
                    f"{SPLIT_NAMES[right_index]}: {sorted(overlap - allowed_overlap)}"
                )
    if observed_overlap != allowed_overlap:
        raise AssertionError(
            "Declared spatial exception fields do not match observed split overlap: "
            f"declared={sorted(allowed_overlap)}, observed={sorted(observed_overlap)}"
        )
    exception_records = {
        int(record["field_index"]): record
        for record in split_manifest.get("spatial_exceptions", [])
    }
    if set(exception_records) != allowed_overlap:
        raise AssertionError("Every spatial exception field must have one manifest record")
    for field_index, record in exception_records.items():
        axis_values = rows if record["axis"] == "row" else cols
        recorded_separation = int(record["minimum_cross_split_center_separation"])
        if record.get("patch_support_disjoint", False) and recorded_separation < patch_size:
            raise AssertionError("A full-guard spatial exception has an invalid separation")
        present_splits = [
            name for name in SPLIT_NAMES
            if np.any((field_indices == field_index) & (split_codes == SPLIT_TO_CODE[name]))
        ]
        for left, right in zip(present_splits[:-1], present_splits[1:]):
            left_values = axis_values[(field_indices == field_index)
                                      & (split_codes == SPLIT_TO_CODE[left])]
            right_values = axis_values[(field_indices == field_index)
                                       & (split_codes == SPLIT_TO_CODE[right])]
            if int(right_values.min()) - int(left_values.max()) < recorded_separation:
                raise AssertionError(
                    f"Spatial exception field {field_index} violates its recorded guard "
                    f"between {left} and {right}"
                )
    normal = _load_json(processed_dir / "normalization.json", required=True)
    expected_fingerprint = normalization_fingerprint(manifest, split_manifest, class_mapping)
    if normal.get("fingerprint") != expected_fingerprint:
        raise ValueError("Stale normalization.json: fingerprint does not match preprocessing/split/classes")
    channels = int(image_stack.shape[0])
    expected_channels = manifest.get("num_channels")
    if expected_channels is not None and channels != int(expected_channels):
        raise ValueError(f"Stack/manifest channel mismatch: {channels} != {expected_channels}")
    if tuple(image_stack.shape[1:]) != tuple(manifest["grid"]["dimensions"]):
        raise ValueError("Stack dimensions disagree with preprocessing manifest")
    if np.any(rows < patch_size // 2) or np.any(rows >= image_stack.shape[1] - patch_size // 2):
        raise ValueError("Invalid centre row for patch extraction")
    if np.any(cols < patch_size // 2) or np.any(cols >= image_stack.shape[2] - patch_size // 2):
        raise ValueError("Invalid centre column for patch extraction")
    return RemoteSensingBundle(
        dataset=canonical, processed_dir=processed_dir, image_stack=image_stack, patches=None,
        rows=rows, cols=cols, labels=labels, field_indices=field_indices,
        field_values=field_values, split_indices=split_indices,
        split_strategy=str(split_manifest["strategy"]),
        mean=np.asarray(normal["mean"], dtype=np.float32),
        std=np.asarray(normal["std"], dtype=np.float32), class_to_idx=class_to_idx,
        idx_to_class=idx_to_class, channels=channels, patch_size=patch_size,
        manifest=manifest,
    )


def _load_legacy_bundle(canonical: str, processed_dir: Path, seed: int) -> RemoteSensingBundle:
    """Read old capped patches for audit only; new experiments must preprocess v2."""
    patches = np.load(processed_dir / "patches.npy", mmap_mode="r", allow_pickle=False)
    raw_labels = np.asarray(np.load(processed_dir / "labels.npy", mmap_mode="r", allow_pickle=False)).astype(str)
    raw_fields = np.asarray(np.load(processed_dir / "field_ids.npy", mmap_mode="r", allow_pickle=False)).astype(str)
    names = sorted(set(raw_labels.tolist()))
    class_to_idx = {name: index for index, name in enumerate(names)}
    labels = np.asarray([class_to_idx[value] for value in raw_labels], dtype=np.int64)
    field_values = sorted(set(raw_fields.tolist()))
    field_to_index = {value: index for index, value in enumerate(field_values)}
    field_indices = np.asarray([field_to_index[value] for value in raw_fields], dtype=np.int64)
    ratios = REMOTE_SENSING_SPECS[canonical]["test_fraction"]
    field_split = deterministic_nested_field_split(field_values, seed, ratios)
    split_indices = {name: np.flatnonzero(np.isin(raw_fields, fields)).astype(np.int64)
                     for name, fields in field_split.items()}
    train = np.asarray(patches[split_indices["train"]], dtype=np.float64)
    mean = train.mean(axis=(0, 2, 3)).astype(np.float32)
    std = train.std(axis=(0, 2, 3)).astype(np.float32)
    std[std < 1e-8] = 1.0
    return RemoteSensingBundle(
        dataset=canonical, processed_dir=processed_dir, image_stack=None, patches=patches,
        rows=None, cols=None, labels=labels, field_indices=field_indices,
        field_values=field_values, split_indices=split_indices,
        split_strategy="legacy_deterministic_field_audit_only", mean=mean, std=std,
        class_to_idx=class_to_idx, idx_to_class={index: name for name, index in class_to_idx.items()},
        channels=int(patches.shape[1]), patch_size=int(patches.shape[2]),
        manifest={"legacy": True, "warning": "capped fixed-patch data; do not use for main reproduction"},
    )


def load_remote_sensing_bundle(dataset: str, root: str | os.PathLike[str], seed: int = 0,
                               split_ratios: Optional[Sequence[float]] = None) -> RemoteSensingBundle:
    if split_ratios is not None:
        raise ValueError("Paper-aligned split ratios are recorded by preprocessing and cannot be overridden")
    canonical = canonical_remote_dataset(dataset)
    processed_dir = _resolve_processed_dir(root, canonical)
    if (processed_dir / "image_stack.npy").is_file():
        bundle = _load_new_bundle(canonical, processed_dir)
    else:
        bundle = _load_legacy_bundle(canonical, processed_dir, seed)
    expected = REMOTE_SENSING_SPECS[canonical]["channels"]
    if expected is not None and bundle.channels != expected:
        raise ValueError(f"{canonical} requires {expected} channels; found {bundle.channels}")
    return bundle


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cluster_cache_paths(bundle: RemoteSensingBundle, cluster_count: int = DEFAULT_CLUSTER_COUNT,
                        seed: int = 0) -> Tuple[Path, Path, Path]:
    stem = f"{CLUSTER_FEATURE}_k{int(cluster_count)}_seed{int(seed)}"
    root = bundle.processed_dir / "clusters"
    return root / f"{stem}_assignments.npy", root / f"{stem}_centres.npy", root / f"{stem}.json"


def remote_cluster_fingerprint(bundle: RemoteSensingBundle, cluster_count: int,
                               seed: int) -> str:
    """Fingerprint an unsupervised cluster cache against its complete data contract."""
    payload = {
        "dataset": bundle.dataset,
        "preprocessing_version": bundle.manifest.get("preprocessing_version"),
        "preprocessing_hash": bundle.manifest.get("preprocessing_hash"),
        "normalization_sha256": _file_sha256(bundle.processed_dir / "normalization.json"),
        "split_manifest_sha256": _file_sha256(bundle.processed_dir / "split_manifest.json"),
        "class_mapping_sha256": _file_sha256(bundle.processed_dir / "class_mapping.json"),
        "feature": CLUSTER_FEATURE,
        "cluster_count": int(cluster_count),
        "seed": int(seed),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _center_features(bundle: RemoteSensingBundle, indices: np.ndarray) -> np.ndarray:
    """Return low-cost, label-free spectral/temporal features for patch centres."""
    indices = np.asarray(indices, dtype=np.int64)
    if bundle.image_stack is not None:
        assert bundle.rows is not None and bundle.cols is not None
        values = np.asarray(
            bundle.image_stack[:, bundle.rows[indices], bundle.cols[indices]].T,
            dtype=np.float32,
        )
    else:
        assert bundle.patches is not None
        radius = bundle.patch_size // 2
        values = np.asarray(bundle.patches[indices, :, radius, radius], dtype=np.float32)
    return (values - bundle.mean[None, :]) / bundle.std[None, :]


def prepare_remote_sensing_cluster_cache(
    bundle: RemoteSensingBundle,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    seed: int = 0,
    fit_samples: int = DEFAULT_INSTANCES_PER_EPOCH,
    predict_batch_size: int = 65_536,
) -> Dict[str, Any]:
    """Fit train-only MiniBatchKMeans and cache assignments for every centre.

    Clustering never consumes class labels. Validation/test centres are only
    passed through the train-fitted clusterer so validation bags can use the
    same construction without fitting on held-out data.
    """
    if bundle.manifest.get("legacy"):
        raise ValueError("ClusterBag requires the paper-aligned centre catalogue")
    cluster_count = int(cluster_count)
    if cluster_count < 2:
        raise ValueError("cluster_count must be at least 2")
    train_indices = bundle.split_indices["train"]
    if len(train_indices) < cluster_count:
        raise ValueError("Training split has fewer centres than requested clusters")
    if fit_samples <= 0 or predict_batch_size <= 0:
        raise ValueError("fit_samples and predict_batch_size must be positive")
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError(
            "Preparing CV/LEM ClusterBag caches requires scikit-learn"
        ) from exc

    rng = np.random.default_rng(seed)
    fit_count = min(int(fit_samples), len(train_indices))
    fit_indices = rng.choice(train_indices, size=fit_count, replace=False)
    fit_features = _center_features(bundle, fit_indices)
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        batch_size=min(8192, fit_count),
        n_init=10,
        max_iter=200,
    )
    clusterer.fit(fit_features)

    assignments = np.empty(
        len(bundle.labels), dtype=np.int16 if cluster_count < 32768 else np.int32
    )
    for start in range(0, len(assignments), int(predict_batch_size)):
        stop = min(start + int(predict_batch_size), len(assignments))
        indices = np.arange(start, stop, dtype=np.int64)
        assignments[start:stop] = clusterer.predict(_center_features(bundle, indices))

    assignment_path, centre_path, manifest_path = cluster_cache_paths(bundle, cluster_count, seed)
    centres = np.asarray(clusterer.cluster_centers_, dtype=np.float32)
    _atomic_npy(assignment_path, assignments)
    _atomic_npy(centre_path, centres)
    train_counts = np.bincount(assignments[train_indices], minlength=cluster_count)
    split_counts = {
        name: np.bincount(assignments[indices], minlength=cluster_count).astype(int).tolist()
        for name, indices in bundle.split_indices.items()
    }
    manifest = {
        "dataset": bundle.dataset,
        "feature": CLUSTER_FEATURE,
        "feature_dimension": bundle.channels,
        "normalization": "training-patch channel mean/std",
        "fit_split": "train",
        "fit_samples": fit_count,
        "cluster_count": cluster_count,
        "seed": int(seed),
        "fingerprint": remote_cluster_fingerprint(bundle, cluster_count, seed),
        "assignments_file": assignment_path.name,
        "assignments_sha256": _file_sha256(assignment_path),
        "cluster_centres_file": centre_path.name,
        "cluster_centres_sha256": _file_sha256(centre_path),
        "inertia": float(clusterer.inertia_),
        "train_cluster_counts": train_counts.astype(int).tolist(),
        "split_cluster_counts": split_counts,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def load_remote_sensing_cluster_assignments(
    bundle: RemoteSensingBundle,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    assignment_path, centre_path, manifest_path = cluster_cache_paths(bundle, cluster_count, seed)
    if not assignment_path.is_file() or not centre_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing ClusterBag cache for {bundle.dataset}: {manifest_path}. Run "
            f"python -m plench.scripts.prepare_remote_sensing_clusters --dataset "
            f"{bundle.dataset} --data-dir <dataset-root> --clusters {int(cluster_count)} "
            f"--seed {int(seed)}"
        )
    manifest = _load_json(manifest_path, required=True)
    expected = remote_cluster_fingerprint(bundle, int(cluster_count), int(seed))
    if manifest.get("fingerprint") != expected:
        raise ValueError("Stale remote-sensing cluster cache: fingerprint mismatch")
    assignments = np.load(assignment_path, mmap_mode="r", allow_pickle=False)
    if assignments.shape != (len(bundle.labels),):
        raise ValueError("Cluster assignment count does not match centre catalogue")
    if np.any(assignments < 0) or np.any(assignments >= int(cluster_count)):
        raise ValueError("Cluster assignments contain out-of-range values")
    if manifest.get("assignments_sha256") != _file_sha256(assignment_path):
        raise ValueError("Cluster assignment cache checksum mismatch")
    return assignments, manifest


def split_statistics(bundle: RemoteSensingBundle) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dataset": bundle.dataset, "input_shape": list(bundle.input_shape),
        "num_classes": bundle.num_classes, "class_to_idx": bundle.class_to_idx,
        "split_strategy": bundle.split_strategy, "splits": {},
    }
    field_ids = bundle.field_ids
    for name, indices in bundle.split_indices.items():
        counts = np.bincount(bundle.labels[indices], minlength=bundle.num_classes)
        result["splits"][name] = {
            "num_fields": int(len(np.unique(field_ids[indices]))),
            "num_instances": int(len(indices)),
            "instances_per_class": {bundle.idx_to_class[index]: int(counts[index])
                                    for index in range(bundle.num_classes)},
        }
    return result


def print_split_statistics(bundle: RemoteSensingBundle) -> None:
    summary = split_statistics(bundle)
    print(f"{bundle.dataset}: input={bundle.input_shape}, classes={bundle.num_classes}, "
          f"split={bundle.split_strategy}")
    print(f"class_to_idx: {bundle.class_to_idx}")
    for name, stats in summary["splits"].items():
        print(f"{name}: fields={stats['num_fields']}, valid_centres={stats['num_instances']}, "
              f"per_class={stats['instances_per_class']}")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _balanced_group_bags(sampled: np.ndarray, group_ids: np.ndarray, bag_size: int,
                         alpha0: float, rng: np.random.Generator) -> np.ndarray:
    """Redistribute one sampled population into Dirichlet cluster mixtures.

    Row sums remain exactly ``bag_size`` and column sums remain exactly the
    sampled cluster counts. Thus ClusterBag changes bag membership but not the
    epoch's sampled instances or global class prior.
    """
    sampled = np.asarray(sampled, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    if sampled.ndim != 1 or group_ids.shape != sampled.shape:
        raise ValueError("sampled and group_ids must be aligned one-dimensional arrays")
    if len(sampled) == 0 or len(sampled) % int(bag_size):
        raise ValueError("sampled population must contain an integer number of bags")
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    if len(unique_groups) == 0 or unique_groups[0] < 0:
        raise ValueError("ClusterBag group IDs must be non-negative")
    num_bags = len(sampled) // int(bag_size)
    targets = np.bincount(inverse, minlength=len(unique_groups)).astype(np.int64)
    base = targets.astype(np.float64) / len(sampled)
    weights = rng.dirichlet(np.maximum(float(alpha0) * base, 1e-12), size=num_bags)
    fractional = weights * int(bag_size)
    counts = np.floor(fractional).astype(np.int64)
    remainders = int(bag_size) - counts.sum(axis=1)
    fractions = fractional - counts
    for bag_id, remainder in enumerate(remainders):
        if remainder:
            probabilities = fractions[bag_id]
            probabilities = probabilities / probabilities.sum()
            additions = rng.choice(
                len(unique_groups), size=int(remainder), replace=True, p=probabilities
            )
            np.add.at(counts[bag_id], additions, 1)

    # Move units within rows until every column equals the sampled cluster
    # population. Every move preserves the exact bag size.
    differences = targets - counts.sum(axis=0)
    deficits = [int(group) for group in np.flatnonzero(differences > 0)]
    deficit_position = 0
    for surplus_group in np.flatnonzero(differences < 0):
        remaining = int(-differences[surplus_group])
        rows = np.flatnonzero(counts[:, surplus_group] > 0)
        rng.shuffle(rows)
        for row in rows:
            movable = min(int(counts[row, surplus_group]), remaining)
            while movable > 0:
                while (deficit_position < len(deficits)
                       and differences[deficits[deficit_position]] == 0):
                    deficit_position += 1
                if deficit_position == len(deficits):
                    raise RuntimeError("Cluster count balancing exhausted deficit columns")
                deficit_group = deficits[deficit_position]
                moved = min(movable, int(differences[deficit_group]))
                counts[row, surplus_group] -= moved
                counts[row, deficit_group] += moved
                differences[surplus_group] += moved
                differences[deficit_group] -= moved
                remaining -= moved
                movable -= moved
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("Unable to balance sampled cluster counts")
    if not np.all(counts.sum(axis=1) == int(bag_size)) or not np.array_equal(
        counts.sum(axis=0), targets
    ):
        raise AssertionError("Invalid balanced ClusterBag count matrix")

    bags = np.empty((num_bags, int(bag_size)), dtype=np.int64)
    offsets = np.zeros(num_bags, dtype=np.int64)
    for group in range(len(unique_groups)):
        pool = sampled[inverse == group].copy()
        rng.shuffle(pool)
        pointer = 0
        for bag_id in np.flatnonzero(counts[:, group]):
            amount = int(counts[bag_id, group])
            start = int(offsets[bag_id])
            bags[bag_id, start:start + amount] = pool[pointer:pointer + amount]
            offsets[bag_id] += amount
            pointer += amount
        if pointer != len(pool):
            raise AssertionError("Cluster pool was not consumed exactly once")
    if not np.all(offsets == int(bag_size)):
        raise AssertionError("ClusterBag contains an incomplete row")
    for bag in bags:
        rng.shuffle(bag)
    return bags


class RemoteSensingBagDataset(Dataset):
    """Dynamically sampled exact-size bags; call :meth:`set_epoch` every epoch."""

    def __init__(self, bundle: RemoteSensingBundle, split: str, bag_size: int,
                 instances_per_epoch: Optional[int], seed: int, mode: str,
                 dynamic: bool = True, num_bags: Optional[int] = None,
                 bag_build: str = "random", cluster_assignments: Optional[np.ndarray] = None,
                 alpha0: float = 1.0):
        if bag_size not in MAIN_BAG_SIZES:
            raise ValueError(f"Main remote-sensing bag_size must be one of {MAIN_BAG_SIZES}")
        if num_bags is not None:
            if instances_per_epoch is not None:
                raise ValueError("Specify instances_per_epoch, not both it and num_bags")
            instances_per_epoch = int(num_bags) * int(bag_size)
        source_count = len(bundle.split_indices[split])
        if instances_per_epoch is None:
            instances_per_epoch = DEFAULT_INSTANCES_PER_EPOCH if dynamic else source_count
        full_count = (int(instances_per_epoch) // int(bag_size)) * int(bag_size)
        if full_count < bag_size:
            raise ValueError(f"{split} instances_per_epoch must provide at least one full bag")
        self.bundle = bundle
        self.split = split
        self.bag_size = int(bag_size)
        self.instances_per_epoch_requested = int(instances_per_epoch)
        self.instances_per_epoch = full_count
        self.num_bags = full_count // self.bag_size
        self.seed = int(seed)
        self.mode = mode
        self.dynamic = bool(dynamic)
        if bag_build not in {"random", "cluster", "alphafirst"}:
            raise ValueError(f"RemoteSensingBagDataset does not support bag_build={bag_build!r}")
        if alpha0 <= 0:
            raise ValueError("ClusterBag Dirichlet concentration alpha0 must be positive")
        if bag_build == "cluster" and cluster_assignments is None:
            raise ValueError("ClusterBag requires cluster_assignments")
        if cluster_assignments is not None and len(cluster_assignments) != len(bundle.labels):
            raise ValueError("cluster_assignments length does not match centre catalogue")
        self.bag_build = bag_build
        self.cluster_assignments = cluster_assignments
        self.alpha0 = float(alpha0)
        self.epoch = -1
        self.bag_indices = np.empty((0, self.bag_size), dtype=np.int64)
        self.label_prob: list[list[float]] = []
        self.set_epoch(0)
        if source_count < self.bag_size:
            raise ValueError(f"{bundle.dataset} {split} has fewer than {self.bag_size} valid centres")

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if not self.dynamic and self.epoch >= 0:
            return
        source = self.bundle.split_indices[self.split]
        rng = np.random.default_rng(self.seed + 1_000_003 * epoch)
        if len(source) >= self.instances_per_epoch:
            sampled = rng.choice(source, size=self.instances_per_epoch, replace=False)
        else:
            # Actual CV/LEM populations exceed 200k.  This branch supports tiny
            # fixtures while guaranteeing no duplicate within an individual bag.
            bags = [rng.choice(source, size=self.bag_size, replace=False) for _ in range(self.num_bags)]
            sampled = np.concatenate(bags)
        if self.bag_build in {"cluster", "alphafirst"}:
            if self.bag_build == "cluster":
                assert self.cluster_assignments is not None
                groups = np.asarray(self.cluster_assignments[sampled], dtype=np.int64)
            else:
                groups = np.asarray(self.bundle.labels[sampled], dtype=np.int64)
            self.bag_indices = _balanced_group_bags(
                sampled, groups, self.bag_size, self.alpha0, rng
            )
        else:
            self.bag_indices = sampled.reshape(self.num_bags, self.bag_size)
        hidden = self.bundle.labels[self.bag_indices]
        counts = np.stack([(hidden == class_index).sum(axis=1)
                           for class_index in range(self.bundle.num_classes)], axis=1)
        proportions = counts.astype(np.float64) / self.bag_size
        if proportions.shape != (self.num_bags, self.bundle.num_classes):
            raise AssertionError("Invalid bag-proportion dimensions")
        if not np.allclose(proportions.sum(axis=1), 1.0, atol=1e-12):
            raise AssertionError("Bag proportions must sum to one")
        self.label_prob = proportions.tolist()
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_bags

    def _normalized(self, indices: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(self.bundle.extract(indices).copy())
        mean = torch.as_tensor(self.bundle.mean).view(1, -1, 1, 1)
        std = torch.as_tensor(self.bundle.std).view(1, -1, 1, 1)
        return (tensor - mean) / std

    def _augment(self, tensor: torch.Tensor, bag_id: int, view: int) -> torch.Tensor:
        # Each instance gets an independent remote-sensing-safe rotation and
        # mirror.  The epoch participates in the seed, so views are not fixed.
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch + 104_729 * (bag_id + 1) + 7_919 * view
        )
        result = tensor.clone()
        horizontal = torch.rand(len(result), generator=generator) < 0.5
        vertical = torch.rand(len(result), generator=generator) < 0.5
        rotations = torch.randint(0, 4, (len(result),), generator=generator)
        result[horizontal] = result[horizontal].flip(-1)
        result[vertical] = result[vertical].flip(-2)
        for quarter_turns in (1, 2, 3):
            mask = rotations == quarter_turns
            result[mask] = torch.rot90(result[mask], quarter_turns, dims=(-2, -1))
        return result

    def __getitem__(self, bag_id: int):
        indices = self.bag_indices[bag_id]
        labels = self.bundle.labels[indices]
        normalized = self._normalized(indices)
        weak = self._augment(normalized, bag_id, view=0) if self.split == "train" else normalized
        if self.mode == "train_u_L^2P-AHIL":
            views = [weak, self._augment(normalized, bag_id, view=1)]
        else:
            views = [weak]
        # Labels are diagnostics only. train.py sends only views[0] and the
        # exact proportion vector to algorithm.update().
        return views, self.label_prob[bag_id], indices.copy(), int(bag_id), labels.copy()


class RemoteSensingEvalDataset(Dataset):
    def __init__(self, bundle: RemoteSensingBundle, split: str = "test"):
        self.bundle = bundle
        self.indices = bundle.split_indices[split]
        self.mean = torch.as_tensor(bundle.mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(bundle.std, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        patch = torch.from_numpy(self.bundle.extract(np.asarray([index]))[0].copy())
        return (patch - self.mean) / self.std, int(self.bundle.labels[index])


def build_remote_sensing_loaders(dataset: str, root: str, bag_size: int, batch_size: int,
                                 seed: int = 0, instances_per_epoch: int = DEFAULT_INSTANCES_PER_EPOCH,
                                 num_bags: Optional[int] = None, num_workers: int = 0,
                                 method: str = "DLLP", bag_build: str = "random",
                                 alpha0: float = 1.0,
                                 cluster_count: int = DEFAULT_CLUSTER_COUNT,
                                 cluster_seed: int = 0):
    bundle = load_remote_sensing_bundle(dataset, root, seed=seed)
    print_split_statistics(bundle)
    cluster_assignments = None
    if bag_build == "cluster":
        cluster_assignments, cluster_manifest = load_remote_sensing_cluster_assignments(
            bundle, cluster_count=cluster_count, seed=cluster_seed
        )
        print(
            f"{bundle.dataset} ClusterBag: K={cluster_count}, alpha0={alpha0}, "
            f"cluster_seed={cluster_seed}, experiment_seed={seed}, "
            f"feature={cluster_manifest['feature']}, fit_split=train"
        )
    elif bag_build not in {"random", "alphafirst"}:
        raise ValueError(
            f"{bundle.dataset} supports bag_build random, cluster, or alphafirst; "
            f"got {bag_build!r}"
        )
    if bag_build == "alphafirst":
        # AlphaFirst uses the hidden training class only to construct synthetic
        # bags. Labels are still withheld from FlowLLP and every other loss.
        cluster_assignments = np.asarray(bundle.labels, dtype=np.int64)
        print(
            f"{bundle.dataset} AlphaFirst: alpha0={alpha0}, "
            f"experiment_seed={seed}, class pools=train split only"
        )
    train_dataset = RemoteSensingBagDataset(
        bundle, "train", bag_size, instances_per_epoch=instances_per_epoch if num_bags is None else None,
        num_bags=num_bags, seed=seed, mode=f"train_u_{method}", dynamic=True,
        bag_build=bag_build, cluster_assignments=cluster_assignments, alpha0=alpha0,
    )
    val_dataset = RemoteSensingBagDataset(
        bundle, "val", bag_size, instances_per_epoch=min(len(bundle.split_indices["val"]), 20_000),
        seed=seed + 1, mode="train_x", dynamic=False,
        bag_build=bag_build, cluster_assignments=cluster_assignments, alpha0=alpha0,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader, bundle


def build_remote_sensing_eval_loader(dataset: str, root: str, batch_size: int,
                                     seed: int = 0, num_workers: int = 0):
    bundle = load_remote_sensing_bundle(dataset, root, seed=seed)
    return DataLoader(
        RemoteSensingEvalDataset(bundle, split="test"), batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
    )


def classification_metrics_from_confusion(confusion: np.ndarray,
                                          class_names: Sequence[str]) -> Dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.int64)
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    total = int(confusion.sum())
    return {
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "balanced_accuracy": float(recall[support > 0].mean()) if np.any(support > 0) else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(name): {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_remote_sensing(algorithm, loader: DataLoader, device: str,
                            class_names: Sequence[str]) -> Dict[str, Any]:
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    algorithm.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            logits = algorithm.predict(inputs.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
            targets_np = targets.numpy()
            np.add.at(confusion, (targets_np, predictions), 1)
    return classification_metrics_from_confusion(confusion, class_names)
