import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from plench.data.ku_optofil_pbc import (
    CLASS_NAMES,
    KUOptofilPBCDataset,
    canonical_ku_optofil_dataset,
    collate_ku_optofil_bags,
    load_ku_optofil_bundle,
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
