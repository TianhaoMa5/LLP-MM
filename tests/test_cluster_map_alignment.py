"""Cluster IDs must follow image identity through selection and permutation."""

import json
import pickle

import numpy as np
import pytest

from mo_matching.data import images as LLP_load


def test_cluster_ids_follow_permuted_images_after_bag_size_truncation():
    clusters = np.array([2, 0, 1, 2, 1, 0, 2])
    image_order = np.array([4, 0, 5, 2, 1, 3])  # seventh row was dropped
    chosen = np.array([5, 1, 3, 0])
    aligned = LLP_load._cluster_ids_for_training_samples(
        clusters,
        dataset="CIFAR10",
        dataset_length=7,
        shuffled_indices=image_order,
    )
    expected = clusters[image_order[chosen]]
    np.testing.assert_array_equal(aligned[chosen], expected)
    assert not np.array_equal(clusters[chosen], expected)  # reproduces the old bug


def test_merged_mini_map_selects_first_500_of_each_600_before_shuffle():
    clusters = np.arange(60000, dtype=np.int64)
    order = np.array([49999, 500, 499, 0, 25001])
    aligned = LLP_load._cluster_ids_for_training_samples(
        clusters,
        dataset="miniImageNet",
        dataset_length=50000,
        shuffled_indices=order,
        metadata={"dataset": "miniImageNet", "split": "train+val+test", "N": 60000},
        map_indices=np.arange(60000),
    )
    expected = (order // 500) * 600 + order % 500
    np.testing.assert_array_equal(aligned, expected)
    assert np.all(aligned % 600 < 500)


@pytest.mark.parametrize(
    "length,metadata",
    [
        (60000, None),
        (59999, {"split": "train+val+test"}),
        (49984, {"split": "train"}),
        (50000, {"split": "train+val+test"}),
    ],
)
def test_ambiguous_or_mismatched_mini_maps_are_rejected(length, metadata):
    with pytest.raises(ValueError, match="length|projection"):
        LLP_load._cluster_ids_for_training_samples(
            np.zeros(length, dtype=np.int64),
            dataset="miniImageNet",
            dataset_length=50000,
            shuffled_indices=np.arange(16),
            metadata=metadata,
        )


def test_cluster_metadata_and_noncanonical_map_order_are_rejected():
    kwargs = dict(dataset="CIFAR10", dataset_length=6, shuffled_indices=np.arange(4))
    with pytest.raises(ValueError, match="dataset"):
        LLP_load._cluster_ids_for_training_samples(
            np.arange(6), metadata={"dataset": "CIFAR100"}, **kwargs
        )
    with pytest.raises(ValueError, match="canonical"):
        LLP_load._cluster_ids_for_training_samples(
            np.arange(6), map_indices=np.arange(6)[::-1], **kwargs
        )
    with pytest.raises(ValueError, match="length"):
        LLP_load._cluster_ids_for_training_samples(np.arange(4), **kwargs)


def _write_cifar(root, reverse_labels=False):
    data_dir = root / "cifar-10-batches-py"
    data_dir.mkdir(exist_ok=True)
    for part in range(5):
        rows = np.arange(part * 12, (part + 1) * 12)
        labels = rows % 3
        if reverse_labels:
            labels = 2 - labels
        with (data_dir / f"data_batch_{part + 1}").open("wb") as handle:
            pickle.dump(
                {
                    "data": np.repeat(rows[:, None], 3072, axis=1).astype(np.uint8),
                    "labels": labels.tolist(),
                },
                handle,
            )
    (root / "cluster_maps").mkdir(exist_ok=True)
    np.savez(
        root / "cluster_maps/CIFAR10_train_K32.npz",
        clusters=np.arange(60) // 20,
        indices=np.arange(60),
        meta=json.dumps({"dataset": "CIFAR10", "split": "train", "N": 60}),
    )


def _image_bags(pack):
    return tuple(tuple(int(image.flat[0]) for image in bag) for bag in pack[0])


def _load_cluster(root, seed):
    output = LLP_load.load_data_train(
        1.0,
        "cluster",
        10,
        0.3,
        dataset="CIFAR10",
        dspth=str(root),
        bagsize=8,
        seed=seed,
    )
    return _image_bags(output[0]), _image_bags(output[1])


def test_cluster_loader_seed_controls_both_splits_without_global_rng_or_labels(
    tmp_path,
):
    _write_cifar(tmp_path)
    previous_rng = np.random.get_state()
    try:
        np.random.seed(9)
        first = _load_cluster(tmp_path, 17)
        np.random.seed(919)
        np.random.normal(size=100)
        repeated = _load_cluster(tmp_path, 17)
        different = _load_cluster(tmp_path, 18)
        assert repeated == first
        assert different != first
        rows = [row for split in first for bag in split for row in bag]
        assert sorted(rows) == list(range(56))  # truncation is applied before shuffle
        assert not set(sum(first[0], ())).intersection(sum(first[1], ()))
        _write_cifar(tmp_path, reverse_labels=True)
        assert _load_cluster(tmp_path, 17) == first  # labels never determine membership
    finally:
        np.random.set_state(previous_rng)


def test_mini_loader_uses_identical_training_selection_for_images_and_map(
    tmp_path, monkeypatch
):
    data = np.arange(1200)[:, None]
    labels = np.repeat([0, 1], 600)
    monkeypatch.setattr(LLP_load, "merge_train_val_test", lambda root: (data, labels))
    (tmp_path / "cluster_maps").mkdir()
    np.savez(
        tmp_path / "cluster_maps/miniImageNet_train_K256.npz",
        clusters=np.arange(1200) // 100,
        indices=np.arange(1200),
        meta=json.dumps(
            {"dataset": "miniImageNet", "split": "train+val+test", "N": 1200}
        ),
    )
    result = LLP_load.load_data_train(
        1.0,
        "cluster",
        100,
        0.0,
        dataset="miniImageNet",
        dspth=str(tmp_path),
        bagsize=32,
        seed=3,
    )
    rows = [row for bag in _image_bags(result[0]) for row in bag]
    expected = np.concatenate([np.arange(500), np.arange(600, 1100)])[:992]
    np.testing.assert_array_equal(sorted(rows), expected)


def test_get_train_loader_passes_experiment_seed_to_cluster_constructor(monkeypatch):
    class ReachedConstructor(Exception):
        pass

    received = {}

    def capture(*args, **kwargs):
        received.update(kwargs)
        raise ReachedConstructor

    monkeypatch.setattr(LLP_load, "load_data_train", capture)
    monkeypatch.setattr(LLP_load, "_IMAGE_AUGMENTATION_IMPORT_ERROR", None)
    with pytest.raises(ReachedConstructor):
        LLP_load.get_train_loader(1.0, "cluster", 10, 0.0, "CIFAR10", 2, 8, seed=37)
    assert received["seed"] == 37
