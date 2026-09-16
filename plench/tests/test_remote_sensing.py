import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from plench.core import hparams_registry
from plench.core.networks import Featurizer
from plench.data.remote_sensing import (
    PREPROCESSING_VERSION,
    RemoteSensingBagDataset,
    build_remote_sensing_loaders,
    classification_metrics_from_confusion,
    compute_train_normalization,
    deterministic_nested_field_split,
    enforce_remote_sensing_backbone,
    load_remote_sensing_cluster_assignments,
    load_remote_sensing_bundle,
    normalization_fingerprint,
    prepare_remote_sensing_cluster_cache,
)
from plench.data.remote_sensing_splits import (
    DEFAULT_COVERAGE_SPLIT_RATIOS,
    DROPPED_SPLIT_CODE,
    build_all_class_coverage_split,
    nested_split_target_counts,
    spatially_partition_field,
)


def _write_dataset(root: Path, dataset: str, channels: int):
    processed = root / "processed"
    processed.mkdir(parents=True)
    rng = np.random.default_rng(123)
    image = rng.normal(size=(channels, 100, 100)).astype(np.float32)
    np.save(processed / "image_stack.npy", image)

    field_values = [str(index) for index in range(60)]
    split_fields = deterministic_nested_field_split(
        field_values, seed=7, test_fraction=0.5 if dataset == "CV" else 0.25
    )
    field_to_split = {
        field: split_code
        for split_code, name in enumerate(("train", "val", "test"))
        for field in split_fields[name]
    }
    # Forty valid centres per logical field; classes repeat independently of split.
    field_indices = np.repeat(np.arange(60, dtype=np.int32), 40)
    flat_positions = np.arange(len(field_indices)) % (80 * 80)
    rows = (flat_positions // 80 + 10).astype(np.int32)
    cols = (flat_positions % 80 + 10).astype(np.int32)
    labels = (field_indices % 4).astype(np.int16)
    split_codes = np.asarray([field_to_split[field_values[index]] for index in field_indices], dtype=np.uint8)
    np.save(processed / "center_rows.npy", rows)
    np.save(processed / "center_cols.npy", cols)
    np.save(processed / "labels.npy", labels)
    np.save(processed / "field_indices.npy", field_indices)
    np.save(processed / "split_codes.npy", split_codes)
    (processed / "field_values.json").write_text(json.dumps(field_values))

    class_names = [f"class_{index}" for index in range(4)]
    class_mapping = {
        "dataset": dataset, "num_classes": 4,
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "idx_to_class": {str(index): name for index, name in enumerate(class_names)},
    }
    (processed / "class_mapping.json").write_text(json.dumps(class_mapping))
    split_manifest = {
        "dataset": dataset, "strategy": "test_fixture_field_disjoint", "seed": 7,
        "field_ids": split_fields,
    }
    (processed / "split_manifest.json").write_text(json.dumps(split_manifest))
    manifest = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_hash": hashlib.sha256(image[:1].tobytes()).hexdigest(),
        "dataset": dataset, "source_files": [{"path": "fixture", "sha256": "fixture"}],
        "selected_acquisition_dates": [] if dataset == "CV" else [f"date_{i}" for i in range(channels // 2)],
        "selected_channels": [f"channel_{i}" for i in range(channels)],
        "spatial_resolution_m": [10.0, 10.0], "num_channels": channels,
        "patch_size": 21, "grid": {"dimensions": [100, 100]},
    }
    fingerprint = normalization_fingerprint(manifest, split_manifest, class_mapping)
    train_indices = np.flatnonzero(split_codes == 0)
    mean, std = compute_train_normalization(image, rows, cols, train_indices, 21)
    (processed / "normalization.json").write_text(json.dumps({
        "fingerprint": fingerprint, "mean": mean.tolist(), "std": std.tolist(),
    }))
    (processed / "preprocessing_manifest.json").write_text(json.dumps(manifest))
    return processed


@pytest.mark.parametrize("dataset,channels", [("CV", 7), ("LEM", 24)])
def test_bundle_shape_field_split_and_train_only_normalization(tmp_path, dataset, channels):
    _write_dataset(tmp_path, dataset, channels)
    bundle = load_remote_sensing_bundle(dataset, tmp_path, seed=7)
    assert bundle.input_shape == (channels, 21, 21)
    assert bundle.num_classes == 4
    split_fields = [set(bundle.field_ids[bundle.split_indices[name]]) for name in ("train", "val", "test")]
    assert split_fields[0].isdisjoint(split_fields[1])
    assert split_fields[0].isdisjoint(split_fields[2])
    assert split_fields[1].isdisjoint(split_fields[2])
    expected_mean, expected_std = compute_train_normalization(
        bundle.image_stack, bundle.rows, bundle.cols, bundle.split_indices["train"], 21
    )
    np.testing.assert_allclose(bundle.mean, expected_mean)
    np.testing.assert_allclose(bundle.std, expected_std)


@pytest.mark.parametrize("dataset,channels", [("CV", 7), ("LEM", 24)])
@pytest.mark.parametrize("bag_size", [32, 64, 128, 256])
def test_exact_dynamic_bag_proportions(tmp_path, dataset, channels, bag_size):
    _write_dataset(tmp_path, dataset, channels)
    bundle = load_remote_sensing_bundle(dataset, tmp_path, seed=2)
    first = RemoteSensingBagDataset(
        bundle, "train", bag_size, instances_per_epoch=bag_size * 3,
        seed=11, mode="train_u_DLLP",
    )
    second = RemoteSensingBagDataset(
        bundle, "train", bag_size, instances_per_epoch=bag_size * 3,
        seed=11, mode="train_u_DLLP",
    )
    np.testing.assert_array_equal(first.bag_indices, second.bag_indices)
    epoch_zero = first.bag_indices.copy()
    first.set_epoch(1)
    assert not np.array_equal(epoch_zero, first.bag_indices)
    assert all(len(np.unique(indices)) == bag_size for indices in first.bag_indices)
    for bag_id, indices in enumerate(first.bag_indices):
        expected = np.bincount(bundle.labels[indices], minlength=bundle.num_classes) / bag_size
        np.testing.assert_allclose(first.label_prob[bag_id], expected, atol=0, rtol=0)
        assert len(first.label_prob[bag_id]) == bundle.num_classes
        assert sum(first.label_prob[bag_id]) == pytest.approx(1.0)


def test_cluster_bags_preserve_sampled_population_and_change_each_epoch(tmp_path):
    _write_dataset(tmp_path, "CV", 7)
    bundle = load_remote_sensing_bundle("CV", tmp_path, seed=7)
    # Deterministic synthetic cluster IDs are sufficient to test the sampler;
    # cluster-cache preparation is exercised separately below.
    assignments = np.arange(len(bundle.labels), dtype=np.int64) % 5
    random_bags = RemoteSensingBagDataset(
        bundle, "train", 32, instances_per_epoch=32 * 6,
        seed=11, mode="train_u_DLLP", bag_build="random",
    )
    cluster_bags = RemoteSensingBagDataset(
        bundle, "train", 32, instances_per_epoch=32 * 6,
        seed=11, mode="train_u_DLLP", bag_build="cluster",
        cluster_assignments=assignments, alpha0=0.3,
    )
    np.testing.assert_array_equal(
        np.sort(random_bags.bag_indices.ravel()),
        np.sort(cluster_bags.bag_indices.ravel()),
    )
    assert all(len(np.unique(bag)) == 32 for bag in cluster_bags.bag_indices)
    for bag_id, indices in enumerate(cluster_bags.bag_indices):
        exact = np.bincount(bundle.labels[indices], minlength=bundle.num_classes) / 32
        np.testing.assert_array_equal(cluster_bags.label_prob[bag_id], exact)
    epoch_zero = cluster_bags.bag_indices.copy()
    cluster_bags.set_epoch(1)
    assert not np.array_equal(epoch_zero, cluster_bags.bag_indices)


def test_alphafirst_bags_preserve_population_and_exact_proportions(tmp_path):
    _write_dataset(tmp_path, "CV", 7)
    bundle = load_remote_sensing_bundle("CV", tmp_path, seed=7)
    random_bags = RemoteSensingBagDataset(
        bundle, "train", 32, instances_per_epoch=32 * 6,
        seed=17, mode="train_u_DLLP", bag_build="random",
    )
    alpha_bags = RemoteSensingBagDataset(
        bundle, "train", 32, instances_per_epoch=32 * 6,
        seed=17, mode="train_u_DLLP", bag_build="alphafirst", alpha0=0.3,
    )
    np.testing.assert_array_equal(
        np.sort(random_bags.bag_indices.ravel()),
        np.sort(alpha_bags.bag_indices.ravel()),
    )
    assert all(len(np.unique(bag)) == 32 for bag in alpha_bags.bag_indices)
    for bag_id, indices in enumerate(alpha_bags.bag_indices):
        exact = np.bincount(bundle.labels[indices], minlength=bundle.num_classes) / 32
        np.testing.assert_array_equal(alpha_bags.label_prob[bag_id], exact)
    epoch_zero = alpha_bags.bag_indices.copy()
    alpha_bags.set_epoch(1)
    assert not np.array_equal(epoch_zero, alpha_bags.bag_indices)


def test_cluster_cache_and_loader_are_train_fitted_and_reproducible(tmp_path):
    pytest.importorskip("sklearn")
    _write_dataset(tmp_path, "CV", 7)
    bundle = load_remote_sensing_bundle("CV", tmp_path, seed=7)
    manifest = prepare_remote_sensing_cluster_cache(
        bundle, cluster_count=4, seed=7, fit_samples=256, predict_batch_size=128
    )
    assignments, loaded = load_remote_sensing_cluster_assignments(
        bundle, cluster_count=4, seed=7
    )
    assert manifest == loaded
    assert manifest["fit_split"] == "train"
    assert manifest["feature"] == "train_normalized_center_pixel"
    assert assignments.shape == bundle.labels.shape
    assert set(np.unique(assignments)).issubset(set(range(4)))

    train_loader, val_loader, _ = build_remote_sensing_loaders(
        "CV", str(tmp_path), bag_size=32, batch_size=2, seed=999,
        instances_per_epoch=96, num_workers=0, bag_build="cluster",
        alpha0=0.5, cluster_count=4, cluster_seed=7,
    )
    assert train_loader.dataset.bag_build == "cluster"
    assert val_loader.dataset.bag_build == "cluster"
    assert train_loader.dataset.num_bags == 3


def test_augmentation_changes_by_epoch_but_is_reproducible(tmp_path):
    _write_dataset(tmp_path, "CV", 7)
    bundle = load_remote_sensing_bundle("CV", tmp_path)
    bags = RemoteSensingBagDataset(
        bundle, "train", 32, instances_per_epoch=64, seed=5, mode="train_u_DLLP"
    )
    indices_zero = bags.bag_indices.copy()
    epoch_zero = bags[0][0][0].clone()
    bags.set_epoch(1)
    epoch_one = bags[0][0][0].clone()
    assert not torch.equal(epoch_zero, epoch_one)
    bags.set_epoch(0)
    np.testing.assert_array_equal(bags.bag_indices, indices_zero)
    assert torch.equal(bags[0][0][0], epoch_zero)


@pytest.mark.parametrize("dataset,shape", [("CV", (7, 21, 21)), ("LEM", (24, 21, 21))])
def test_remote_resnet18_forward(dataset, shape):
    hparams = hparams_registry.default_hparams("LLP_PVC", dataset)
    enforce_remote_sensing_backbone(dataset, hparams)
    assert hparams["model"] == "RemoteResNet18"
    model = Featurizer(shape, hparams).eval()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    with torch.no_grad():
        features = model(torch.zeros(2, *shape))
        output = torch.nn.Linear(model.n_outputs, 4)(features)
    assert features.shape == (2, 512)
    assert output.shape == (2, 4)


@pytest.mark.parametrize("dataset", ["CV", "LEM"])
def test_remote_backbone_override_is_rejected(dataset):
    hparams = hparams_registry.default_hparams("LLP_PVC", dataset)
    hparams["model"] = "ResNet"
    with pytest.raises(ValueError, match="requires model=RemoteResNet18"):
        enforce_remote_sensing_backbone(dataset, hparams)


def test_bag_loader_shape_and_no_instance_labels_in_views(tmp_path):
    _write_dataset(tmp_path, "CV", 7)
    bundle = load_remote_sensing_bundle("CV", tmp_path)
    bags = RemoteSensingBagDataset(
        bundle, "train", bag_size=32, instances_per_epoch=64, seed=0, mode="train_u_DLLP"
    )
    views, proportions, indices, bag_ids, hidden_labels = next(iter(DataLoader(bags, batch_size=2)))
    assert len(views) == 1
    assert views[0].shape == (2, 32, 7, 21, 21)
    assert len(proportions) == bundle.num_classes
    assert indices.shape == hidden_labels.shape == (2, 32)
    assert bag_ids.shape == (2,)


def test_200k_bag_counts_and_config_matrix():
    assert {size: 200_000 // size for size in (32, 64, 128, 256)} == {
        32: 6250, 64: 3125, 128: 1562, 256: 781,
    }
    config_dir = Path(__file__).parents[1] / "configs" / "remote_sensing"
    configs = sorted(config_dir.glob("*.json"))
    assert len(configs) == 8
    observed = set()
    for path in configs:
        config = json.loads(path.read_text())
        observed.add((config["dataset"], config["bagsize"]))
        assert config["algorithm"] == "LLP_PVC"
        assert config["instances_per_epoch"] == 200_000
        assert config["batchsize"] == 2
    assert observed == {(dataset, size) for dataset in ("CV", "LEM") for size in (32, 64, 128, 256)}


def test_stale_normalization_is_rejected(tmp_path):
    processed = _write_dataset(tmp_path, "CV", 7)
    manifest = json.loads((processed / "preprocessing_manifest.json").read_text())
    manifest["preprocessing_hash"] = "regenerated-same-shape"
    (processed / "preprocessing_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Stale normalization"):
        load_remote_sensing_bundle("CV", tmp_path)


def test_normalization_ignores_nodata_outside_training_patch_coverage():
    image = np.ones((2, 50, 50), dtype=np.float32)
    image[:, :5, :] = np.nan
    rows = np.asarray([25], dtype=np.int32)
    cols = np.asarray([25], dtype=np.int32)
    mean, std = compute_train_normalization(image, rows, cols, np.asarray([0]), 21)
    np.testing.assert_array_equal(mean, [1.0, 1.0])
    np.testing.assert_array_equal(std, [1.0, 1.0])


def _coverage_catalogue():
    # Four classes represented by 1, 2, 10, and 20 logical fields. The first
    # two classes mathematically require guarded within-field exceptions if all
    # three splits must contain every class.
    fields_per_class = (1, 2, 10, 20)
    field_indices = []
    labels = []
    rows = []
    cols = []
    field_index = 0
    for class_index, num_fields in enumerate(fields_per_class):
        for _ in range(num_fields):
            # Long row extent permits two independent 21-pixel guards.
            field_indices.extend([field_index] * 200)
            labels.extend([class_index] * 200)
            rows.extend(np.repeat(np.arange(15, 115), 2).tolist())
            cols.extend(np.tile([20, 21], 100).tolist())
            field_index += 1
    return tuple(np.asarray(values) for values in (field_indices, labels, rows, cols))


def test_fixed_all_class_coverage_split_is_reproducible_and_guarded():
    field_indices, labels, rows, cols = _coverage_catalogue()
    first_codes, first_manifest = build_all_class_coverage_split(
        field_indices, labels, rows, cols, num_classes=4, seed=42,
        test_fraction=0.5, patch_size=21,
    )
    second_codes, second_manifest = build_all_class_coverage_split(
        field_indices, labels, rows, cols, num_classes=4, seed=42,
        test_fraction=0.5, patch_size=21,
    )
    np.testing.assert_array_equal(first_codes, second_codes)
    assert first_manifest == second_manifest
    assert first_manifest["field_owner_counts"] == nested_split_target_counts(33, 0.5)
    assert first_manifest["dropped_guard_centres"] == int(
        np.sum(first_codes == DROPPED_SPLIT_CODE)
    )

    split_fields = {}
    for split_code, split_name in enumerate(("train", "val", "test")):
        selected = first_codes == split_code
        assert set(np.unique(labels[selected])) == {0, 1, 2, 3}
        split_fields[split_name] = set(np.unique(field_indices[selected]).tolist())
    allowed = set(first_manifest["spatial_exception_field_indices"])
    observed = ((split_fields["train"] & split_fields["val"])
                | (split_fields["train"] & split_fields["test"])
                | (split_fields["val"] & split_fields["test"]))
    assert observed == allowed
    assert len(first_manifest["spatial_exceptions"]) == 2

    assert nested_split_target_counts(478, 0.20, 0.125) == {
        "train": 334, "val": 48, "test": 96,
    }
    assert DEFAULT_COVERAGE_SPLIT_RATIOS == (0.70, 0.10, 0.20)
    assert nested_split_target_counts(794, 0.20, 0.125) == {
        "train": 556, "val": 79, "test": 159,
    }

    for record in first_manifest["spatial_exceptions"]:
        field = record["field_index"]
        axis = rows if record["axis"] == "row" else cols
        present = [
            split_code for split_code in range(3)
            if np.any((field_indices == field) & (first_codes == split_code))
        ]
        for left, right in zip(present[:-1], present[1:]):
            left_values = axis[(field_indices == field) & (first_codes == left)]
            right_values = axis[(field_indices == field) & (first_codes == right)]
            assert right_values.min() - left_values.max() >= 21


def test_loader_accepts_only_declared_guarded_spatial_field_overlap(tmp_path):
    processed = _write_dataset(tmp_path, "CV", 7)
    field_indices = np.load(processed / "field_indices.npy")
    # Produce a rare-class distribution while retaining all 60 fixture fields.
    field_classes = np.concatenate((np.repeat(0, 1), np.repeat(1, 2),
                                    np.repeat(2, 20), np.repeat(3, 37)))
    labels = field_classes[field_indices].astype(np.int16)
    rows = np.load(processed / "center_rows.npy")
    cols = np.load(processed / "center_cols.npy")
    # Give the three rare fields enough spatial extent for guarded cuts.
    for field in range(3):
        selected = np.flatnonzero(field_indices == field)
        rows[selected] = np.linspace(10, 89, len(selected), dtype=np.int32)
        cols[selected] = 20
    split_codes, split_manifest = build_all_class_coverage_split(
        field_indices, labels, rows, cols, num_classes=4, seed=42,
        test_fraction=0.5, patch_size=21,
    )
    split_manifest.update({"dataset": "CV", "profile": "fixture_coverage"})
    np.save(processed / "labels.npy", labels)
    np.save(processed / "center_rows.npy", rows)
    np.save(processed / "center_cols.npy", cols)
    np.save(processed / "split_codes.npy", split_codes)
    (processed / "split_manifest.json").write_text(json.dumps(split_manifest))
    manifest = json.loads((processed / "preprocessing_manifest.json").read_text())
    class_mapping = json.loads((processed / "class_mapping.json").read_text())
    fingerprint = normalization_fingerprint(manifest, split_manifest, class_mapping)
    image = np.load(processed / "image_stack.npy")
    train_indices = np.flatnonzero(split_codes == 0)
    mean, std = compute_train_normalization(image, rows, cols, train_indices, 21)
    (processed / "normalization.json").write_text(json.dumps({
        "fingerprint": fingerprint, "mean": mean.tolist(), "std": std.tolist(),
    }))
    bundle = load_remote_sensing_bundle("CV", tmp_path)
    for split_name in ("train", "val", "test"):
        assert set(np.unique(bundle.labels[bundle.split_indices[split_name]])) == {0, 1, 2, 3}


def test_compact_single_field_records_maximal_partial_guard():
    row_grid, col_grid = np.meshgrid(np.arange(17), np.arange(38), indexing="ij")
    rows = row_grid.ravel()[:608]
    cols = col_grid.ravel()[:608]
    blocks, details = spatially_partition_field(
        rows, cols, ("train", "val", "test"), (0.6, 0.15, 0.25), patch_size=21
    )
    assert all(len(block) > 0 for block in blocks)
    assert len(set(np.concatenate(blocks).tolist())) == sum(len(block) for block in blocks)
    assert not details["patch_support_disjoint"]
    assert details["minimum_cross_split_center_separation"] < 21
    assert details["achieved_guard_radius"] > 0


def test_imbalanced_metrics_include_all_requested_outputs():
    confusion = np.asarray([[8, 2], [1, 1]])
    metrics = classification_metrics_from_confusion(confusion, ["major", "rare"])
    assert metrics["overall_accuracy"] == pytest.approx(9 / 12)
    assert metrics["balanced_accuracy"] == pytest.approx((0.8 + 0.5) / 2)
    assert "macro_f1" in metrics
    assert set(metrics["per_class"]) == {"major", "rare"}
    assert metrics["confusion_matrix"] == confusion.tolist()
