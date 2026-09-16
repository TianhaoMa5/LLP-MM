import csv
import inspect
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from plench.core import hparams_registry
from plench.data.fed_isic2019 import (
    FedISIC2019FeatureBagDataset,
    canonical_fed_isic2019_dataset,
    collate_fed_isic2019_bags,
    load_fed_isic2019_bundle,
)
from plench.data.fed_isic2019_preparation import (
    CLASS_NAMES,
    assign_bags,
    compute_bag_proportions,
    prepare_fed_isic2019,
)


class FakeDINOv2(torch.nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.fail = fail
        self.grad_enabled: list[bool] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.fail:
            raise AssertionError("a valid feature cache must skip encoder execution")
        self.grad_enabled.append(torch.is_grad_enabled())
        mean = images.mean(dim=(2, 3))
        std = images.std(dim=(2, 3))
        return torch.cat([mean, std, mean * std], dim=1) * self.scale


def _write_fixture(root: Path, *, samples_per_split: int = 64) -> Path:
    image_root = root / "ISIC_2019_Training_Input_preprocessed"
    image_root.mkdir(parents=True)
    fieldnames = [
        "image", *CLASS_NAMES, "UNK", "target", "center", "fold", "fold2"
    ]
    rows = []
    rng = np.random.default_rng(91)
    for split_number, split in enumerate(("train", "test")):
        for local_index in range(samples_per_split):
            target = local_index % len(CLASS_NAMES)
            center = local_index % 6
            sample_id = f"ISIC_fixture_{split_number}_{local_index:04d}"
            base = np.asarray(
                [
                    25 + 90 * (local_index // 16),
                    35 + 15 * (local_index % 4),
                    210 - 30 * (local_index // 16),
                ],
                dtype=np.int16,
            )
            pixels = np.clip(
                base + rng.integers(-3, 4, size=(32, 32, 3)), 0, 255
            ).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(image_root / f"{sample_id}.jpg")
            row = {name: "0.0" for name in CLASS_NAMES}
            row.update(
                {
                    "image": sample_id,
                    CLASS_NAMES[target]: "1.0",
                    "UNK": "0.0",
                    "target": str(target),
                    "center": str(center),
                    "fold": split,
                    "fold2": f"{split}_{center}",
                }
            )
            rows.append(row)
    with (root / "train_test_split").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return root


def _clustered_embeddings() -> np.ndarray:
    rng = np.random.default_rng(7)
    features = np.concatenate(
        [
            rng.normal(loc=cluster * 2.0, scale=0.15, size=(size, 16))
            for cluster, size in enumerate([120, 80, 60, 40, 20])
        ]
    ).astype(np.float32)
    features[:, 0] += 1.0
    return features


def test_assignment_is_label_blind_deterministic_bounded_and_complete():
    signature = inspect.signature(assign_bags)
    forbidden = {"target", "targets", "label", "labels", "client", "client_ids"}
    assert forbidden.isdisjoint(signature.parameters)
    features = _clustered_embeddings()
    first = assign_bags(features, seed=0)
    second = assign_bags(features, seed=0)
    assert [bag.tolist() for bag in first] == [bag.tolist() for bag in second]
    sizes = [len(bag) for bag in first]
    assert min(sizes) >= 16
    assert max(sizes) <= 128
    assert len(set(sizes)) > 1
    flattened = np.concatenate(first)
    assert len(flattened) == len(features)
    assert np.array_equal(np.sort(flattened), np.arange(len(features)))
    assert len(np.unique(flattened)) == len(features)


def test_proportions_are_attached_only_after_assignment():
    bags = assign_bags(_clustered_embeddings(), seed=3)
    labels = np.arange(320, dtype=np.int64) % 8
    counts, proportions = compute_bag_proportions(bags, labels)
    assert len(counts) == len(bags)
    for bag, count, proportion in zip(bags, counts, proportions):
        assert len(count) == 8
        assert len(proportion) == 8
        assert count.sum() == len(bag)
        assert np.isclose(proportion.sum(), 1.0, atol=1e-8)


def test_offline_cache_runtime_and_original_image_bags(tmp_path):
    root = _write_fixture(tmp_path / "fed_isic2019")
    encoder = FakeDINOv2()
    metadata = prepare_fed_isic2019(
        root,
        device="cpu",
        extraction_batch_size=16,
        num_workers=0,
        min_bag_size=8,
        max_bag_size=24,
        target_bag_size=16,
        seed=0,
        model=encoder,
    )
    assert encoder.grad_enabled and not any(encoder.grad_enabled)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert metadata["class_names"] == CLASS_NAMES
    assert metadata["num_classes"] == 8
    assert metadata["num_clients"] == 6
    assert metadata["official_validation_split"] is False
    assert metadata["source"]["split_counts"] == {"train": 64, "test": 64}
    assert metadata["bag_manifest"]["membership_inputs"] == [
        "l2_normalized_image_features"
    ]
    assert metadata["bag_manifest"]["labels_used_for_membership"] is False
    assert metadata["bag_manifest"]["client_ids_used_for_membership"] is False
    for split in ("train", "test"):
        feature_path = root / metadata["feature_manifests"][split]["file"]
        assert feature_path.name == f"{split}_features.pt"
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        assert payload["features"].shape[0] == 64
        assert payload["labels"].shape == (64,)
        assert payload["client_ids"].shape == (64,)
        assert len(payload["sample_ids"]) == 64
        assert len(payload["image_paths"]) == 64
        bag_info = metadata["bag_manifest"]["splits"][split]
        assert bag_info["coverage_exactly_once"]
        assert bag_info["bag_size"]["min"] >= 8
        assert bag_info["bag_size"]["max"] <= 24
        assert bag_info["average_within_bag_cosine_similarity"] > bag_info[
            "average_random_pair_cosine_similarity"
        ]

    bundle = load_fed_isic2019_bundle(root)
    assert canonical_fed_isic2019_dataset("fed_isic2019") == "FedISIC2019"
    assert bundle.num_classes == 8
    assert bundle.input_shape == (3, 224, 224)
    assert set(bundle.splits.tolist()) == {"train", "test"}
    assert sorted(index for bag in bundle.bags for index in bag.indices.tolist()) == list(
        range(128)
    )
    train = FedISIC2019FeatureBagDataset(bundle, "train")
    train_item = train[0]
    assert "instance_labels" not in train_item
    assert train_item["x"].shape[1:] == (3, 224, 224)
    assert sum(train_item["client_histogram"].values()) == len(train_item["x"])
    train_batch = collate_fed_isic2019_bags([train_item])
    assert "instance_labels" not in train_batch
    assert torch.allclose(train_batch["proportion"].sum(dim=1), torch.ones(1))
    test_item = FedISIC2019FeatureBagDataset(bundle, "test")[0]
    assert "instance_labels" in test_item
    assert test_item["instance_labels"].shape[0] == len(test_item["x"])

    cached = prepare_fed_isic2019(
        root,
        device="cpu",
        extraction_batch_size=16,
        num_workers=0,
        min_bag_size=8,
        max_bag_size=24,
        target_bag_size=16,
        seed=0,
        model=FakeDINOv2(fail=True),
    )
    assert cached["feature_manifests"] == metadata["feature_manifests"]
    assert cached["bag_manifest"] == metadata["bag_manifest"]


def test_fed_isic_hparams_keep_dinov2_separate_from_downstream_model():
    hparams = hparams_registry.default_hparams("PM", "FedISIC2019")
    assert hparams["model"] == "ImageNetResNet18"
    assert hparams["pretrained"] is True
    assert hparams["input_resolution"] == 224
