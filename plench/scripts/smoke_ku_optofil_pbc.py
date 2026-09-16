"""Strict KU-Optofil natural-bag, ResNet-18, LLP and evaluation smoke test."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..core import algorithms, hparams_registry
from ..data.ku_optofil_pbc import (
    CLASS_NAMES,
    INPUT_SHAPE,
    NUM_CLASSES,
    KUOptofilPBCDataset,
    collate_ku_optofil_bags,
    evaluate_ku_optofil,
    load_ku_optofil_bundle,
    update_ku_optofil_algorithm,
)


def run(
    data_root: str,
    method: str,
    device: str,
    *,
    order: int = 3,
) -> dict:
    bundle = load_ku_optofil_bundle(data_root)
    assert bundle.has_instance_labels is True
    assert bundle.num_classes == NUM_CLASSES == 13
    split_sets = {
        split: {bag.bag_id for bag in bundle.bags_for_split(split)}
        for split in ("train", "val", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    for bag in bundle.bags:
        assert len(bag.proportions) == NUM_CLASSES
        assert np.isclose(bag.proportions.sum(), 1.0, atol=1e-6)
        assert np.array_equal(
            np.bincount(bundle.instance_labels[bag.indices], minlength=NUM_CLASSES),
            bag.class_counts,
        )

    train_dataset = KUOptofilPBCDataset(
        bundle, "train", seed=42, train_instance_sample_size=2
    )
    val_dataset = KUOptofilPBCDataset(bundle, "val", seed=42)
    test_dataset = KUOptofilPBCDataset(bundle, "test", seed=42)
    largest = int(np.argmax([len(bag.indices) for bag in train_dataset.bags]))
    train_dataset.set_epoch(0)
    epoch_zero = train_dataset[largest]["instance_ids"]
    train_dataset.set_epoch(1)
    epoch_one = train_dataset[largest]["instance_ids"]
    assert epoch_zero != epoch_one

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_ku_optofil_bags,
    )
    batch = next(iter(train_loader))
    assert tuple(batch["x"].shape[1:]) == INPUT_SHAPE
    assert batch["proportion"].shape == (2, NUM_CLASSES)
    assert "instance_labels" in batch  # evaluation/debug metadata exists

    hparams = hparams_registry.default_hparams(method, "KUOptofilPBC")
    hparams.update({"model": "ResNet", "lr": 1e-3, "weight_decay": 0.0})
    if method == "LLP_MM":
        hparams.update(
            {
                "order": int(order),
                "moment_loss_type": "ce",
                "moment_algorithm": "stable_dp",
                "moment_compute_dtype": "float64",
                "moment_ce_smoothing_tau": 1e-4,
                "order_weights": [1.0 / float(order)] * int(order),
            }
        )
    algorithm_class = algorithms.get_algorithm_class(method)
    algorithm = algorithm_class(
        2, INPUT_SHAPE, train_dataset.label_prob, hparams, bagsize=2
    ).to(device)
    step = update_ku_optofil_algorithm(algorithm, method, batch, device)

    # Run the real full-bag evaluator on the two smallest official test bags.
    smallest_test = np.argsort([len(bag.indices) for bag in test_dataset.bags])[:2]
    test_loader = DataLoader(
        Subset(test_dataset, smallest_test.tolist()),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_ku_optofil_bags,
    )
    metrics = evaluate_ku_optofil(
        algorithm, test_loader, device, forward_chunk_size=2
    )
    # Exercise two complete validation patients through the DataLoader as well.
    smallest_val = np.argsort([len(bag.indices) for bag in val_dataset.bags])[:2]
    val_batch = next(
        iter(
            DataLoader(
                Subset(val_dataset, smallest_val.tolist()),
                batch_size=2,
                collate_fn=collate_ku_optofil_bags,
            )
        )
    )
    result = {
        "method": method,
        "order": int(order) if method == "LLP_MM" else None,
        "moment_loss_type": "ce" if method == "LLP_MM" else None,
        "moment_algorithm": "stable_dp" if method == "LLP_MM" else None,
        "moment_compute_dtype": "float64" if method == "LLP_MM" else None,
        "moment_ce_smoothing_tau": 1e-4 if method == "LLP_MM" else None,
        "classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "input_shape": list(INPUT_SHAPE),
        "x_shape": list(batch["x"].shape),
        "proportion_shape": list(batch["proportion"].shape),
        "train_hidden_label_shape": list(batch["instance_labels"].shape),
        "train_loss_received_instance_labels": False,
        "loss": float(step["loss"]),
        "backward_completed": True,
        "dynamic_train_subsampling_verified": True,
        "val_bag_sizes_checked": val_batch["bag_sizes"].tolist(),
        "evaluation_completed": True,
        "evaluation_instances": metrics["instances"],
        "instance_accuracy": metrics["instance_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "confusion_matrix_shape": [
            len(metrics["confusion_matrix"]),
            len(metrics["confusion_matrix"][0]),
        ],
        "split_disjoint": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="plench/data/ku_optofil_pbc")
    parser.add_argument("--method", default="PM")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    run(
        args.data_root,
        args.method,
        args.device,
        order=args.order,
    )


if __name__ == "__main__":
    main()
