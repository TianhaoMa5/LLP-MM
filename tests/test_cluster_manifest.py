import json

import numpy as np
import pytest

from mo_matching.data.cluster_manifest import freeze_cluster_manifest
from mo_matching.data.images import load_data_train
from test_cluster_map_alignment import _write_cifar, _image_bags


def test_manifest_tracks_actual_images_and_proportions_and_rejects_changed_seed(
    tmp_path,
):
    _write_cifar(tmp_path)
    path = tmp_path / "frozen.npz"
    kwargs = dict(
        dataset="CIFAR10",
        dspth=str(tmp_path),
        bagsize=8,
        seed=17,
        cluster_manifest=path,
    )
    first = load_data_train(1.0, "cluster", 10, 0.3, **kwargs)
    original_bytes = path.read_bytes()
    with np.load(path, allow_pickle=False) as manifest:
        np.testing.assert_array_equal(manifest["train_indices"], _image_bags(first[0]))
        np.testing.assert_array_equal(manifest["val_indices"], _image_bags(first[1]))
        indices = manifest["train_indices"]
        expected = np.stack([(indices % 3 == c).mean(1) for c in range(10)], axis=1)
        np.testing.assert_array_equal(manifest["train_proportions"], expected)
    load_data_train(1.0, "cluster", 10, 0.3, **kwargs)
    assert path.read_bytes() == original_bytes
    kwargs["seed"] = 18
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        load_data_train(1.0, "cluster", 10, 0.3, **kwargs)
    assert path.read_bytes() == original_bytes


def test_manifest_checks_array_contents_even_if_stored_digest_was_not_changed(tmp_path):
    path = tmp_path / "frozen.npz"
    kwargs = dict(
        metadata={"seed": 0},
        train_indices=[[0, 1]],
        val_indices=[],
        train_proportions=[[0.5, 0.5]],
        val_proportions=[],
    )
    freeze_cluster_manifest(path, **kwargs)
    with np.load(path, allow_pickle=False) as saved:
        contents = dict(saved)
    contents["train_indices"] = np.array([[1, 0]])
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="train_indices mismatch"):
        freeze_cluster_manifest(path, **kwargs)


def test_map_labels_are_checked_against_real_images(tmp_path):
    _write_cifar(tmp_path)
    path = tmp_path / "cluster_maps/CIFAR10_train_K32.npz"
    with np.load(path, allow_pickle=False) as saved:
        contents = dict(saved)
    contents["labels"] = np.arange(60) % 3
    np.savez(path, **contents)
    load_data_train(
        1.0, "cluster", 10, 0.0, dataset="CIFAR10", dspth=str(tmp_path), bagsize=8
    )
    contents["labels"] = (contents["labels"] + 1) % 3
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="map/image label alignment failed"):
        load_data_train(
            1.0, "cluster", 10, 0.0, dataset="CIFAR10", dspth=str(tmp_path), bagsize=8
        )
