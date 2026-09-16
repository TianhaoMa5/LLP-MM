import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from plench.core import algorithms, hparams_registry
from plench.core.networks import Featurizer
from plench.data.cct import (
    CCTFeatureBagDataset,
    CCTInstanceBatchSampler,
    canonical_cct_dataset,
    collate_cct_bags,
    load_cct_bundle,
)
from plench.data.cct_preparation import (
    CCT20_ANNOTATION_SPLITS,
    CCTSample,
    build_feature_bags,
    build_split_bag_records,
    prepare_cct,
)


class FakeDINOv2(torch.nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.fail = fail
        self.grad_enabled: list[bool] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.fail:
            raise AssertionError("valid feature cache should skip encoder execution")
        self.grad_enabled.append(torch.is_grad_enabled())
        mean = images.mean(dim=(2, 3))
        std = images.std(dim=(2, 3))
        return torch.cat([mean, std, mean[:, :2] * std[:, :2]], dim=1) * self.scale


def _write_cct20_fixture(root: Path) -> Path:
    image_root = root / "eccv_18_all_images_sm"
    annotation_root = root / "eccv_18_annotation_files"
    image_root.mkdir(parents=True)
    annotation_root.mkdir(parents=True)
    categories = [{"id": 30, "name": "empty"}, {"id": 1, "name": "animal"}]
    for file_number, (filename, _, _) in enumerate(CCT20_ANNOTATION_SPLITS):
        count = 8 if filename == "train_annotations.json" else 4
        images = []
        annotations = []
        for item in range(count):
            image_id = f"image-{file_number}-{item}"
            image_name = f"{image_id}.jpg"
            yy, xx = np.mgrid[:32, :32]
            pixels = np.stack(
                [
                    (xx * (item + 1) + 17 * file_number) % 255,
                    (yy * (file_number + 1) + 29 * item) % 255,
                    ((xx + yy) * (item + file_number + 1)) % 255,
                ],
                axis=2,
            ).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(image_root / image_name)
            images.append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": 128,
                    "height": 128,
                    "location": str(100 + file_number),
                    "seq_id": f"sequence-{file_number}",
                    "frame_num": item,
                    "date_captured": "2020-01-01 00:00:00",
                }
            )
            annotations.append(
                {
                    "id": f"ann-{file_number}-{item}",
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [0.125, 0.125, 0.75, 0.75]
                    if file_number == 4 and item == count - 1
                    else [16, 16, 96, 96],
                }
            )
            if file_number == 0 and item == 0:
                annotations.append(
                    {"id": "empty-frame-marker", "image_id": image_id, "category_id": 30}
                )
        (annotation_root / filename).write_text(
            json.dumps(
                {
                    "info": {"description": "fixture"},
                    "categories": categories,
                    "images": images,
                    "annotations": annotations,
                }
            ),
            encoding="utf-8",
        )
    return root


def _synthetic_samples(target_offset: int = 0, location_offset: int = 0):
    result = []
    index = 0
    for split in ("train", "test"):
        for local in range(96):
            target = (local + target_offset) % 3
            result.append(
                CCTSample(
                    sample_index=index,
                    crop_id=f"crop-{index}",
                    image_path=f"images/{index}.jpg",
                    image_id=f"crop-{index}",
                    original_image_id=f"image-{index}",
                    original_image_path=f"source/{index}.jpg",
                    annotation_id=f"annotation-{index}",
                    bbox_x=0.0,
                    bbox_y=0.0,
                    bbox_width=100.0,
                    bbox_height=100.0,
                    bbox_area=10000.0,
                    source_image_width=100,
                    source_image_height=100,
                    annotation_image_width=100,
                    annotation_image_height=100,
                    bbox_source="annotation",
                    target=target,
                    category_id=target,
                    class_name=str(target),
                    split=split,
                    official_split=split,
                    location=str(location_offset + local % 7),
                    sequence_id="",
                    frame_number="",
                    datetime="",
                    annotation_category_ids=str(target),
                )
            )
            index += 1
    return result


def test_feature_builder_is_variable_bounded_deterministic_and_label_blind():
    signature = inspect.signature(build_feature_bags)
    assert not {"target", "targets", "label", "labels", "location"}.intersection(signature.parameters)
    rng = np.random.default_rng(7)
    features = np.concatenate(
        [
            rng.normal(loc=number * 3, scale=0.25, size=(size, 8))
            for number, size in enumerate([120, 80, 60, 40, 20])
        ]
    ).astype(np.float32)
    first = build_feature_bags(features, seed=42)
    second = build_feature_bags(features, seed=42)
    assert [bag.tolist() for bag in first] == [bag.tolist() for bag in second]
    sizes = [len(bag) for bag in first]
    assert len(set(sizes)) > 1
    assert min(sizes) >= 16
    assert max(sizes) <= 128
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(len(features)))


def test_membership_is_unchanged_when_hidden_labels_and_locations_change():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(288, 12)).astype(np.float32)
    original, original_stats = build_split_bag_records(
        features,
        _synthetic_samples(),
        ["a", "b", "c"],
        target_avg_bag_size=16,
        min_bag_size_ratio=0.25,
        max_bag_size_ratio=2.0,
        seed=5,
    )
    changed, changed_stats = build_split_bag_records(
        features,
        _synthetic_samples(target_offset=1, location_offset=500),
        ["a", "b", "c"],
        target_avg_bag_size=16,
        min_bag_size_ratio=0.25,
        max_bag_size_ratio=2.0,
        seed=5,
    )
    assert [record["instance_indices"] for record in original] == [
        record["instance_indices"] for record in changed
    ]
    assert original_stats == changed_stats
    assert any(
        left["class_counts"] != right["class_counts"]
        for left, right in zip(original, changed)
    )
    for record in original:
        assert {record["split"]} == {
            _synthetic_samples()[index].split for index in record["instance_indices"]
        }


def test_offline_pipeline_cache_and_runtime_hide_training_labels(tmp_path):
    root = _write_cct20_fixture(tmp_path / "cct20")
    encoder = FakeDINOv2()
    metadata = prepare_cct(
        root,
        device="cpu",
        extraction_batch_size=3,
        num_workers=0,
        pca_dim=4,
        target_avg_bag_size=4,
        min_bag_size_ratio=0.25,
        max_bag_size_ratio=2.0,
        seed=13,
        model=encoder,
    )
    assert encoder.grad_enabled and not any(encoder.grad_enabled)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert metadata["source"]["original_frames"] == 24
    assert metadata["source"]["total_annotations"] == 25
    assert metadata["source"]["annotations_with_valid_bbox"] == 24
    assert metadata["source"]["annotations_removed_small_bbox"] == 0
    assert metadata["source"]["empty_annotations_excluded"] == 1
    assert metadata["source"]["retained_crop_instances"] == 24
    assert metadata["feature_manifest"]["l2_normalized"]
    assert metadata["pca_manifest"]["fit_split"] == "train"
    assert not metadata["pca_manifest"]["labels_used"]
    assert metadata["bag_manifest"]["membership_inputs"] == ["pca_image_features"]
    assert metadata["bag_manifest"]["proportions_computed_after_membership"]

    bundle = load_cct_bundle(root)
    assert canonical_cct_dataset("cct20") == "CCT"
    assert bundle.class_names == ["animal"]
    assert bundle.input_shape == (3, 112, 112)
    assert bundle.metadata["instance_type"] == "bbox_crop"
    assert bundle.image_paths[0].name.endswith(".jpg")
    with Image.open(bundle.image_paths[0]) as crop:
        assert crop.size == (24, 24)
    train = CCTFeatureBagDataset(bundle, "train")
    train_item = train[0]
    assert "instance_labels" not in train_item
    train_batch = collate_cct_bags([train_item])
    assert "instance_labels" not in train_batch
    assert torch.allclose(train_batch["proportion"].sum(dim=1), torch.ones(1))
    test_item = CCTFeatureBagDataset(bundle, "test")[0]
    assert "instance_labels" in test_item
    assert set(bundle.splits.tolist()) == {"train", "test"}
    assert metadata["source"]["split_counts"]["train"] == 16
    assert "val" not in metadata["bag_manifest"]["split_statistics"]

    # A valid cache must avoid all encoder work unless --force is supplied.
    cached = prepare_cct(
        root,
        device="cpu",
        extraction_batch_size=3,
        num_workers=0,
        pca_dim=4,
        target_avg_bag_size=4,
        min_bag_size_ratio=0.25,
        max_bag_size_ratio=2.0,
        seed=13,
        model=FakeDINOv2(fail=True),
    )
    assert cached["feature_manifest"] == metadata["feature_manifest"]


def test_cct_hparams_use_image_backbone_and_variable_bag_batching():
    hparams = hparams_registry.default_hparams("PM", "CCT")
    assert hparams["model"] == "CCTResNet18"
    assert hparams["pretrained"] is True
    assert hparams["input_resolution"] == 112
    assert hparams["batch_size"] == 1024
    assert hparams["optimizer"] == "SGD"
    assert hparams["lr"] == 0.05


def test_cct_supplement_stem_resnet18_is_exact_and_unpretrained():
    model = Featurizer(
        (3, 112, 112),
        {"model": "ResNet", "pretrained": False, "input_resolution": 112},
    )
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert model.pretrained_weights is None
    assert model.stem_variant == "modified_3x3_stride1_no_maxpool"


def test_cct_transforms_match_reference_crop_pipeline(tmp_path):
    root = _write_cct20_fixture(tmp_path / "cct20")
    prepare_cct(
        root,
        device="cpu",
        extraction_batch_size=4,
        num_workers=0,
        pca_dim=4,
        target_avg_bag_size=4,
        min_bag_size_ratio=0.25,
        max_bag_size_ratio=2.0,
        seed=3,
        model=FakeDINOv2(),
    )
    bundle = load_cct_bundle(root)
    train_ops = CCTFeatureBagDataset(bundle, "train").transform.transforms
    assert isinstance(train_ops[0], transforms.RandomResizedCrop)
    assert train_ops[0].size == (112, 112)
    assert train_ops[0].scale == (0.2, 1.0)
    assert isinstance(train_ops[1], transforms.RandomHorizontalFlip)
    assert train_ops[1].p == 0.5
    assert isinstance(train_ops[2], transforms.RandomApply)
    assert train_ops[2].p == 0.8
    assert isinstance(train_ops[3], transforms.RandomGrayscale)
    assert train_ops[3].p == 0.2
    eval_ops = CCTFeatureBagDataset(bundle, "test").transform.transforms
    assert isinstance(eval_ops[0], transforms.Resize)
    assert eval_ops[0].size == (112, 112)


def test_cct_instance_batch_sampler_preserves_bags_and_instance_scale():
    bag_sizes = [16, 31, 48, 64, 79, 96, 112, 128]
    sampler = CCTInstanceBatchSampler(bag_sizes, target_instances=256, seed=7)
    batches = list(iter(sampler))
    assert len(batches) == int(np.ceil(sum(bag_sizes) / 256))
    assert sorted(index for batch in batches for index in batch) == list(
        range(len(bag_sizes))
    )
    totals = [sum(bag_sizes[index] for index in batch) for batch in batches]
    assert max(totals) - min(totals) <= max(bag_sizes)


def test_pm_loss_uses_real_variable_bag_layout():
    algorithm = algorithms.PM(
        epochs=2,
        input_shape=4,
        train_givenY=[[0.5, 0.5]],
        hparams={"model": "Linear", "lr": 0.01},
        bagsize=64,
    )
    outputs = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]
    )
    sizes = torch.tensor([2, 3])
    index = torch.repeat_interleave(torch.arange(2), sizes)
    proportions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    actual = algorithm.PM_Loss(
        outputs, proportions, bag_sizes=sizes, bag_index=index
    )
    probabilities = outputs.softmax(dim=1)
    expected = -0.5 * (
        probabilities[:2].mean(dim=0)[0].log()
        + probabilities[2:].mean(dim=0)[1].log()
    )
    torch.testing.assert_close(actual, expected)
