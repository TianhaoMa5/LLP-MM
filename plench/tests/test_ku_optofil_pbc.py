import csv
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from plench.data.ku_optofil_pbc import (
    CLASS_NAMES,
    KUOptofilPBCDataset,
    KUOptofilBag,
    build_ku_optofil_loaders,
    canonical_ku_optofil_dataset,
    collate_ku_optofil_bags,
    load_ku_optofil_bundle,
    partition_ku_unknown_indices,
    update_ku_optofil_algorithm,
)


def _write_fixture(root: Path) -> Path:
    processed = root / "processed"
    image_root = root / "raw" / "dataset"
    processed.mkdir(parents=True)
    rows = []
    bags = []
    index = 0
    for split_number, (split, patients) in enumerate(
        [("train", ["p0"]), ("val", ["p1"]), ("test", ["p2"])]
    ):
        for patient in patients:
            bag_indices = []
            counts = [0] * len(CLASS_NAMES)
            for cell_number in range(3):
                class_index = (split_number + cell_number) % len(CLASS_NAMES)
                folder = image_root / split / CLASS_NAMES[class_index]
                folder.mkdir(parents=True, exist_ok=True)
                filename = f"{patient}_{cell_number}.jpg"
                Image.new("RGB", (8, 8), (class_index, 10, 20)).save(folder / filename)
                rows.append(
                    {
                        "instance_index": index,
                        "patient_id": patient,
                        "image_name": filename,
                        "relative_path": (folder / filename).relative_to(root).as_posix(),
                        "source_path_field": f"{split}/{CLASS_NAMES[class_index]}",
                        "class_name": CLASS_NAMES[class_index],
                        "class_index": class_index,
                        "split": split,
                    }
                )
                bag_indices.append(index)
                counts[class_index] += 1
                index += 1
            bags.append(
                {
                    "bag_id": patient,
                    "patient_id": patient,
                    "split": split,
                    "n_instances": len(bag_indices),
                    "instance_indices": bag_indices,
                    "class_counts": counts,
                    "class_proportions": [value / len(bag_indices) for value in counts],
                }
            )
    with (processed / "instances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (processed / "bags.json").write_text(json.dumps(bags))
    (processed / "metadata.json").write_text(
        json.dumps({"class_names": CLASS_NAMES})
    )
    return root


def test_registry_bundle_and_patient_split(tmp_path):
    root = _write_fixture(tmp_path / "ku")
    assert canonical_ku_optofil_dataset("ku_optofil_pbc") == "KUOptofilPBC"
    bundle = load_ku_optofil_bundle(root)
    assert bundle.has_instance_labels
    assert bundle.num_classes == 13
    split_ids = {
        split: {bag.bag_id for bag in bundle.bags_for_split(split)}
        for split in ("train", "val", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["val"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])


def test_train_test_only_merges_validation_patients_without_test_overlap(tmp_path):
    root = _write_fixture(tmp_path / 'ku')
    train, val, bundle, _ = build_ku_optofil_loaders(
        str(root), batch_size=2, num_workers=0, merge_validation_into_train=True,
    )
    assert val is None
    assert {bag.bag_id for bag in train.dataset.bags} == {'p0', 'p1'}
    indices = np.concatenate([bag.indices for bag in train.dataset.bags])
    test_indices = np.concatenate([bag.indices for bag in bundle.bags_for_split('test')])
    assert len(indices) == len(np.unique(indices)) == 6
    assert set(indices).isdisjoint(test_indices)
    assert len(indices) + len(test_indices) == len(bundle.image_paths)
    assert [bag.bag_id for bag in bundle.bags_for_split('val')] == ['p1']


def test_unknown_3204_partition_is_balanced_label_free_and_deterministic():
    indices = np.arange(3204)
    bags = partition_ku_unknown_indices(indices, 128, 0)
    assert len(bags) == 26
    assert sorted(map(len, bags)) == [123] * 20 + [124] * 6
    assert np.array_equal(np.sort(np.concatenate(bags)), indices)
    repeated = partition_ku_unknown_indices(indices[::-1], 128, 0)
    assert all(np.array_equal(a, b) for a, b in zip(bags, repeated))
    changed = partition_ku_unknown_indices(indices, 128, 1)
    assert not np.array_equal(bags[0], changed[0])


def test_unknown_split_retains_all_cells_known_patients_and_test(tmp_path):
    bundle = load_ku_optofil_bundle(_write_fixture(tmp_path / 'ku'))
    original_test = bundle.bags_for_split('test')[0]
    unknown_indices = np.arange(len(bundle.instance_labels), len(bundle.instance_labels) + 3204)
    unknown_labels = np.arange(3204) % len(CLASS_NAMES)
    counts = np.bincount(unknown_labels, minlength=len(CLASS_NAMES))
    bundle.patient_ids = np.concatenate([bundle.patient_ids, np.full(3204, 'unknown', dtype=object)])
    bundle.instance_labels = np.concatenate([bundle.instance_labels, unknown_labels])
    original_unknown = KUOptofilBag(
        'unknown', unknown_indices, (counts / 3204).astype(np.float32), counts, 'train',
    )
    bundle.bags.append(original_unknown)
    train = KUOptofilPBCDataset(
        bundle, 'train', seed=42, merge_validation_into_train=True,
        unknown_bag_max_size=128, unknown_bag_seed=0,
    )
    assert len(train.bags) == 28  # two known patients plus 26 synthetic bags
    assert train.bags[0] is bundle.bags_for_split('train')[0]
    assert train.bags[1] is bundle.bags_for_split('val')[0]
    assert bundle.bags_for_split('test')[0] is original_test
    all_indices = np.concatenate([bag.indices for bag in train.bags])
    assert len(all_indices) == len(np.unique(all_indices)) == 3210
    assert set(all_indices).isdisjoint(original_test.indices)
    for bag in train.bags[2:]:
        expected_counts = np.bincount(bundle.instance_labels[bag.indices], minlength=13)
        np.testing.assert_array_equal(bag.class_counts, expected_counts)
        np.testing.assert_allclose(bag.proportions, expected_counts / len(bag.indices))
        assert np.isclose(bag.proportions.sum(), 1)
    unchanged_default = KUOptofilPBCDataset(bundle, 'train')
    assert unchanged_default.bags[-1] is original_unknown
    # Changing hidden diagnoses or training RNG must not affect membership.
    bundle.instance_labels[unknown_indices] = (unknown_labels + 1) % 13
    changed = KUOptofilPBCDataset(
        bundle, 'train', seed=99, merge_validation_into_train=True,
        unknown_bag_max_size=128, unknown_bag_seed=0,
    )
    assert all(np.array_equal(a.indices, b.indices) for a, b in zip(train.bags, changed.bags))


def test_unknown_partition_rejects_invalid_inputs_and_test_changes(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        partition_ku_unknown_indices(np.arange(5), 0, 0)
    with pytest.raises(ValueError, match='repeated'):
        partition_ku_unknown_indices(np.array([0, 0]), 128, 0)
    bundle = load_ku_optofil_bundle(_write_fixture(tmp_path / 'ku'))
    with pytest.raises(ValueError, match='training-only'):
        KUOptofilPBCDataset(bundle, 'test', unknown_bag_max_size=128)


def test_collate_contains_eval_labels_but_update_filters_them(tmp_path):
    bundle = load_ku_optofil_bundle(_write_fixture(tmp_path / "ku"))
    dataset = KUOptofilPBCDataset(bundle, "train")
    batch = collate_ku_optofil_bags([dataset[0]])
    assert batch["x"].shape == (3, 3, 224, 224)
    assert batch["instance_labels"].shape == (3,)
    assert torch.allclose(batch["proportion"].sum(dim=1), torch.ones(1))

    class SpyAlgorithm:
        def __init__(self):
            self.network = torch.nn.Linear(3, 13)
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)
            self.seen_labels = False

        def predict(self, x):
            return self.network(x.mean(dim=(2, 3)))

        def _backward_step(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": float(loss.item())}

    algorithm = SpyAlgorithm()
    result = update_ku_optofil_algorithm(algorithm, "PM", batch, "cpu")
    assert np.isfinite(result["loss"])


def test_training_subsampling_is_without_replacement_and_dynamic(tmp_path):
    bundle = load_ku_optofil_bundle(_write_fixture(tmp_path / "ku"))
    dataset = KUOptofilPBCDataset(
        bundle, "train", seed=7, train_instance_sample_size=2
    )
    dataset.set_epoch(0)
    first = dataset[0]["instance_ids"]
    dataset.set_epoch(1)
    second = dataset[0]["instance_ids"]
    assert len(first) == len(set(first)) == 2
    assert len(second) == len(set(second)) == 2
    assert first != second


def test_checkpoint_preserves_full_bag_update_and_batchnorm_buffers():
    """Recomputing bounded forwards must retain every cell's gradient once."""
    class SmallAlgorithm:
        def __init__(self):
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 1),
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(4, 13),
            )
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)

        def predict(self, x):
            return self.network(x)

        def _backward_step(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": float(loss.detach())}

    torch.manual_seed(5)
    ordinary = SmallAlgorithm()
    checkpointed = copy.deepcopy(ordinary)
    batch = {
        "x": torch.randn(5, 3, 4, 4),
        "bag_index": torch.zeros(5, dtype=torch.long),
        "bag_sizes": torch.tensor([5]),
        "proportion": torch.full((1, 13), 1 / 13),
        "instance_weights": torch.ones(5),
    }
    expected = update_ku_optofil_algorithm(
        ordinary, "PM", batch, "cpu", forward_chunk_size=2
    )
    actual = update_ku_optofil_algorithm(
        checkpointed, "PM", batch, "cpu", forward_chunk_size=2,
        activation_checkpoint=True,
    )
    assert abs(expected["loss"] - actual["loss"]) < 1e-6
    for name, value in ordinary.network.state_dict().items():
        torch.testing.assert_close(value, checkpointed.network.state_dict()[name])


def test_mm_keeps_short_patients_and_uses_only_feasible_orders():
    from plench.data.ref2021 import _llp_mm_variable_loss

    torch.manual_seed(8)
    logits = torch.randn(6, 13, dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    proportions = torch.zeros(2, 13, dtype=torch.float64)
    proportions[0, 0] = 1
    proportions[1, :5] = 0.2
    slices = [torch.arange(1), torch.arange(1, 6)]
    loss = _llp_mm_variable_loss(probabilities, proportions, slices, 3)
    first = _llp_mm_variable_loss(
        probabilities[:1], proportions[:1], [torch.arange(1)], 1
    )
    second = _llp_mm_variable_loss(
        probabilities[1:], proportions[1:], [torch.arange(5)], 3
    )
    torch.testing.assert_close(loss, (first + second) / 2)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(dim=1) > 0).all()
