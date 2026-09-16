import numpy as np
import pytest


torch = pytest.importorskip("torch")

from plench.data.ref2021 import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    canonical_ref2021_dataset,
    collate_ref2021_bags,
)


def test_ref2021_registry_and_class_order():
    assert canonical_ref2021_dataset("ref2021_uoa11") == "REF2021UOA11"
    assert NUM_CLASSES == 5
    assert CLASS_NAMES == ["unclassified", "1star", "2star", "3star", "4star"]


def test_variable_natural_bag_collate_has_no_instance_labels():
    items = [
        {
            "x": torch.zeros(2, 4),
            "proportion": torch.tensor([0.0, 0.0, 0.25, 0.5, 0.25]),
            "bag_id": "a",
            "instance_ids": ["a0", "a1"],
            "instance_weights": torch.ones(2),
            "inferred_class_counts": torch.tensor([0, 0, 0, 1, 1]),
        },
        {
            "x": torch.ones(3, 4),
            "proportion": torch.tensor([0.0, 0.1, 0.2, 0.4, 0.3]),
            "bag_id": "b",
            "instance_ids": ["b0", "b1", "b2"],
            "instance_weights": torch.ones(3),
            "inferred_class_counts": torch.tensor([0, 0, 1, 1, 1]),
        },
    ]
    batch = collate_ref2021_bags(items)
    assert tuple(batch["x"].shape) == (5, 4)
    assert batch["bag_sizes"].tolist() == [2, 3]
    assert batch["bag_index"].tolist() == [0, 0, 1, 1, 1]
    assert tuple(batch["proportion"].shape) == (2, 5)
    assert "instance_labels" not in batch
    assert np.allclose(batch["proportion"].sum(dim=1).numpy(), 1.0)
