from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from plench.data.amazon_wilds import (  # noqa: E402
    AmazonWILDSEvalDataset,
    AmazonWILDSTrainDataset,
    build_amazon_wilds_loaders,
    collate_amazon_wilds_bags,
    load_amazon_wilds_bundle,
    update_amazon_wilds_algorithm,
)
from plench.scripts.prepare_amazon_wilds import build_processed_cache  # noqa: E402


def _fixture(root: Path) -> Path:
    reviewers = ["A", "A", "A", "B", "B", "C", "A", "D", "B", "E", "F"]
    ratings = [1, 1, 5, 2, 2, 3, 4, 5, 1, 4, 5]
    reviews = pd.DataFrame(
        {
            "reviewerID": reviewers,
            "asin": [f"P{index % 4}" for index in range(len(reviewers))],
            "category": ["books" if index % 2 else "music" for index in range(len(reviewers))],
            "reviewYear": [2012 + index % 3 for index in range(len(reviewers))],
            "unixReviewTime": [1_350_000_000 + index for index in range(len(reviewers))],
            "overall": ratings,
            "reviewText": [f"review text {index}" for index in range(len(reviewers))],
        }
    )
    # Official WILDS IDs: train, val, id_val, test, id_test; E is excluded and
    # F belongs to the co-located official unlabeled adaptation data.
    splits = pd.DataFrame({"split": [0, 0, 0, 0, 0, 1, 2, 3, 4, -1, 11]})
    build_processed_cache(
        reviews,
        splits,
        root,
        source_files={"reviews": "fixture/reviews.csv", "user_split": "fixture/user.csv"},
    )
    features_dir = root / "features"
    features_dir.mkdir(parents=True)
    features = np.arange(9 * 4, dtype=np.float32).reshape(9, 4) / 100
    np.save(features_dir / "features.npy", features)
    (features_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "amazon_wilds",
                "dataset_version": "2.1",
                "encoder": "diagnostic-hashing-4",
                "instances": 9,
                "feature_dimension": 4,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_preprocessing_builds_one_complete_bag_per_training_reviewer(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "amazon_wilds")
    bundle = load_amazon_wilds_bundle(root)

    assert len(bundle.train_bags) == 2
    by_reviewer = {bag.reviewer_id: bag for bag in bundle.train_bags}
    assert set(by_reviewer) == {"A", "B"}
    assert len(by_reviewer["A"].indices) == 3
    assert len(by_reviewer["B"].indices) == 2
    np.testing.assert_allclose(
        by_reviewer["A"].proportions, [2 / 3, 0, 0, 0, 1 / 3]
    )
    np.testing.assert_allclose(by_reviewer["B"].proportions, [0, 1, 0, 0, 0])
    for bag in bundle.train_bags:
        assert len(set(bundle.reviewer_ids[bag.indices].tolist())) == 1
        assert np.isclose(bag.proportions.sum(), 1.0)


def test_official_split_is_preserved_and_overlap_is_reported(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "amazon_wilds")
    bundle = load_amazon_wilds_bundle(root)
    stats = json.loads(
        (root / "processed" / "amazon_wilds_stats.json").read_text(encoding="utf-8")
    )

    assert {name: int((bundle.splits == name).sum()) for name in set(bundle.splits)} == {
        "train": 5,
        "val": 1,
        "id_val": 1,
        "test": 1,
        "id_test": 1,
    }
    assert stats["reviewer_overlap"]["train__val"] == 0
    assert stats["reviewer_overlap"]["train__id_val"] == 1
    assert stats["reviewer_overlap"]["train__id_test"] == 1
    assert stats["excluded_split_minus_one_instances"] == 1
    assert stats["official_unlabeled_splits"]["val_unlabeled"] == 1
    assert stats["official_unlabeled_splits"]["test_unlabeled"] == 0
    assert stats["official_unlabeled_splits"]["extra_unlabeled"] == 0


def test_training_batch_hides_instance_labels(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "amazon_wilds")
    bundle = load_amazon_wilds_bundle(root)
    dataset = AmazonWILDSTrainDataset(bundle)
    items = [dataset[index] for index in range(len(dataset))]
    batch = collate_amazon_wilds_bags(items)

    assert "label" not in items[0]
    assert "instance_labels" not in items[0]
    assert "label" not in batch
    assert "instance_labels" not in batch
    assert sorted(batch["bag_sizes"].tolist()) == [2, 3]
    assert batch["x"].shape == (5, 4)
    assert batch["proportion"].shape == (2, 5)


def test_num_reviewers_selects_complete_bags_deterministically(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "amazon_wilds")
    bundle = load_amazon_wilds_bundle(root)
    first = AmazonWILDSTrainDataset(bundle, seed=17, num_reviewers=1)
    second = AmazonWILDSTrainDataset(bundle, seed=17, num_reviewers=1)

    assert first.bags[0].bag_id == second.bags[0].bag_id
    original = next(bag for bag in bundle.train_bags if bag.bag_id == first.bags[0].bag_id)
    assert np.array_equal(first.bags[0].indices, original.indices)
    with pytest.raises(ValueError, match="exceeds"):
        AmazonWILDSTrainDataset(bundle, num_reviewers=3)


def test_evaluation_remains_instance_labeled(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "amazon_wilds")
    bundle = load_amazon_wilds_bundle(root)
    validation = AmazonWILDSEvalDataset(bundle, "val")
    test = AmazonWILDSEvalDataset(bundle, "test")

    assert len(validation) == 1 and validation[0]["label"] == 2
    assert len(test) == 1 and test[0]["label"] == 4


class _TinyAlgorithm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(4, 5)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(values)

    def _backward_step(self, loss: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach())}


def test_tiny_llp_update_receives_no_hidden_rating(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "amazon_wilds")
    train_loader, _, _, _ = build_amazon_wilds_loaders(
        str(root), batch_size=2, num_workers=0
    )
    batch = next(iter(train_loader))
    algorithm = _TinyAlgorithm()
    result = update_amazon_wilds_algorithm(algorithm, "PM", batch, "cpu")

    assert np.isfinite(result["loss"])
    assert "label" not in batch and "instance_labels" not in batch


def test_amazon_config_has_no_target_bag_size() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "amazon_wilds.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "bagsize" not in config
    assert "bag_size" not in config
    assert "target_bag_size" not in config
    assert "train_instance_sample_size" not in config
    assert config["num_reviewers"] is None
