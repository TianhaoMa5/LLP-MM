#!/usr/bin/env python3
"""Strict REF2021 data, natural-bag, forward/loss/backward smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..core import algorithms, hparams_registry
from ..data.ref2021 import (
    CLASS_NAMES,
    NUM_CLASSES,
    REF2021UOA11Dataset,
    build_ref2021_eval_loader,
    build_ref2021_loaders,
    evaluate_ref2021,
    load_ref2021_bundle,
    update_ref2021_algorithm,
)


def smoke(data_root: Path, method: str, device: str) -> dict[str, object]:
    np.random.seed(42)
    torch.manual_seed(42)
    bundle = load_ref2021_bundle(data_root)
    assert bundle.num_classes == 5
    assert bundle.has_instance_labels is False
    assert bundle.feature_dim > 0
    assert np.isfinite(bundle.features).all()
    assert len(set(bundle.instance_ids.tolist())) == len(bundle.instance_ids)
    split_sets = {
        split: {bag.bag_id for bag in bundle.bags_for_split(split)}
        for split in ("train", "val", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    for bag in bundle.bags:
        assert len(bag.indices) > 0
        assert len(bag.proportions) == NUM_CLASSES
        assert (bag.proportions >= 0).all()
        assert abs(float(bag.proportions.sum()) - 1.0) < 1e-6
        assert int(bag.inferred_counts.sum()) == len(bag.indices)
        assert (bundle.instance_weights[bag.indices] > 0).all()

    sampled = REF2021UOA11Dataset(
        bundle, "train", seed=42, train_instance_sample_size=32
    )
    large_item = next(index for index, bag in enumerate(sampled.bags) if len(bag.indices) > 32)
    sampled.set_epoch(0)
    first_ids = sampled[large_item]["instance_ids"]
    sampled.set_epoch(1)
    second_ids = sampled[large_item]["instance_ids"]
    assert len(first_ids) == len(second_ids) == 32
    assert first_ids != second_ids

    train_loader, val_loader, _, input_dim = build_ref2021_loaders(
        str(data_root), batch_size=8, seed=42, num_workers=0
    )
    test_loader = build_ref2021_eval_loader(
        str(data_root), split="test", batch_size=8, num_workers=0
    )
    batch = next(iter(train_loader))
    assert batch["x"].ndim == 2
    assert batch["x"].shape[1] == input_dim
    assert batch["proportion"].shape[1] == NUM_CLASSES
    assert torch.allclose(
        batch["proportion"].sum(dim=1), torch.ones(len(batch["proportion"])), atol=1e-6
    )
    assert int(batch["bag_sizes"].sum()) == len(batch["x"])

    hparams = hparams_registry.default_hparams(method, "REF2021UOA11")
    hparams.update({"model": "MLP", "lr": 1e-3, "weight_decay": 1e-4})
    algorithm_class = algorithms.get_algorithm_class(method)
    model = algorithm_class(
        max(1, len(train_loader)),
        input_dim,
        train_loader.dataset.label_prob,
        hparams,
        64,
    ).to(device)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    update = update_ref2021_algorithm(model, method, batch, device)
    assert np.isfinite(float(update["loss"]))
    assert any(
        not torch.equal(old, new.detach().cpu())
        for old, new in zip(before, model.parameters())
    )
    val_metrics = evaluate_ref2021(model, val_loader, device)
    test_metrics = evaluate_ref2021(model, test_loader, device)
    for metrics in (val_metrics, test_metrics):
        for key in (
            "bag_proportion_mae",
            "bag_proportion_rmse",
            "bag_proportion_l1",
            "bag_proportion_kl",
            "bag_proportion_js",
            "ordinal_emd",
        ):
            assert np.isfinite(float(metrics[key]))
    result: dict[str, object] = {
        "method": method,
        "x_shape": list(batch["x"].shape),
        "proportion_shape": list(batch["proportion"].shape),
        "bag_sizes": batch["bag_sizes"].tolist(),
        "classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "feature_dimension": input_dim,
        "loss": float(update["loss"]),
        "backward_completed": True,
        "evaluation_completed": True,
        "dynamic_subsampling_verified": True,
        "split_disjoint": True,
        "val_metrics": {
            key: value
            for key, value in val_metrics.items()
            if key not in {"bag_ids", "per_class_mae"}
        },
        "test_metrics": {
            key: value
            for key, value in test_metrics.items()
            if key not in {"bag_ids", "per_class_mae"}
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("plench/data/ref2021_uoa11"))
    parser.add_argument("--method", default="PM")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = smoke(args.data_root.expanduser().resolve(), args.method, args.device)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
