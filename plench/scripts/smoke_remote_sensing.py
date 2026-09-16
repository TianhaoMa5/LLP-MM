#!/usr/bin/env python3
"""Strict CV/LEM preprocessing, dynamic-bag, and backbone smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from plench.core import hparams_registry
from plench.core.networks import Featurizer
from plench.data.remote_sensing import (
    DEFAULT_INSTANCES_PER_EPOCH,
    MAIN_BAG_SIZES,
    RemoteSensingBagDataset,
    SPLIT_NAMES,
    load_remote_sensing_bundle,
    split_statistics,
)


def check_dataset(dataset: str, root: str, seed: int, device: str):
    bundle = load_remote_sensing_bundle(dataset, root, seed=seed)
    manifest = bundle.manifest
    split_manifest = json.loads(
        (bundle.processed_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert bundle.patch_size == 21
    fields = [set(bundle.field_indices[bundle.split_indices[name]].astype(int).tolist())
              for name in SPLIT_NAMES]
    observed_overlap = ((fields[0] & fields[1]) | (fields[0] & fields[2])
                        | (fields[1] & fields[2]))
    allowed_overlap = set(int(value) for value in
                          split_manifest.get("spatial_exception_field_indices", []))
    assert observed_overlap == allowed_overlap
    if split_manifest.get("strategy") == "deterministic_all_class_coverage_field_spatial_v1":
        for split_name in SPLIT_NAMES:
            present = set(np.unique(bundle.labels[bundle.split_indices[split_name]]).tolist())
            assert present == set(range(bundle.num_classes))
    if dataset == "CV":
        assert manifest["field_id_column"] == "Field_numb"
        assert np.allclose(manifest["spatial_resolution_m"], [10.0, 10.0])
        assert bundle.channels == 7
    else:
        dates = manifest["selected_acquisition_dates"]
        assert len(dates) * 2 == bundle.channels
        assert manifest["polarization_order"] == ["VV", "VH"]
        assert manifest["selected_channels"] == [
            f"{date}_{polarization}" for date in dates for polarization in ("VV", "VH")
        ]

    bag_report = {}
    for bag_size in MAIN_BAG_SIZES:
        bags = RemoteSensingBagDataset(
            bundle, "train", bag_size, instances_per_epoch=DEFAULT_INSTANCES_PER_EPOCH,
            seed=seed, mode="train_u_DLLP", dynamic=True,
        )
        expected_bags = DEFAULT_INSTANCES_PER_EPOCH // bag_size
        assert len(bags) == expected_bags
        epoch_zero = bags.bag_indices.copy()
        first_indices = epoch_zero[0]
        assert len(first_indices) == bag_size
        assert len(np.unique(first_indices)) == bag_size
        hidden = bundle.labels[first_indices]
        exact = np.bincount(hidden, minlength=bundle.num_classes) / bag_size
        observed = np.asarray(bags.label_prob[0])
        assert len(observed) == bundle.num_classes
        assert abs(float(observed.sum()) - 1.0) < 1e-12
        np.testing.assert_array_equal(observed, exact)
        first_view = bags[0][0][0]
        bags.set_epoch(1)
        assert not np.array_equal(epoch_zero, bags.bag_indices)
        assert not torch.equal(first_view, bags[0][0][0])
        bag_report[str(bag_size)] = {
            "bags_per_epoch": expected_bags,
            "sampled_instances": expected_bags * bag_size,
            "remainder_dropped": DEFAULT_INSTANCES_PER_EPOCH % bag_size,
            "dynamic_epoch_sampling": True,
            "exact_proportions": True,
        }

    hparams = hparams_registry.default_hparams("LLP_PVC", dataset)
    backbone = Featurizer(bundle.input_shape, hparams).to(device).eval()
    classifier = torch.nn.Linear(backbone.n_outputs, bundle.num_classes).to(device).eval()
    indices = bundle.split_indices["train"][:2]
    inputs = torch.from_numpy(bundle.extract(indices).copy()).to(device)
    mean = torch.as_tensor(bundle.mean, device=device).view(1, -1, 1, 1)
    std = torch.as_tensor(bundle.std, device=device).view(1, -1, 1, 1)
    with torch.no_grad():
        features = backbone((inputs - mean) / std)
        logits = classifier(features)
    assert tuple(features.shape) == (2, 512)
    assert tuple(logits.shape) == (2, bundle.num_classes)
    return {
        **split_statistics(bundle),
        "manifest_checks": {
            "field_disjoint": not observed_overlap,
            "spatial_exception_field_indices": sorted(allowed_overlap),
            "all_classes_in_every_split": all(
                set(np.unique(bundle.labels[bundle.split_indices[name]]).tolist())
                == set(range(bundle.num_classes)) for name in SPLIT_NAMES
            ),
            "patch_size": 21,
            "resolution_m": manifest["spatial_resolution_m"],
            "selected_dates": manifest["selected_acquisition_dates"],
            "channel_order": manifest["selected_channels"],
        },
        "bags": bag_report,
        "backbone": {
            "architecture": "small-input ResNet18", "feature_shape": [2, 512],
            "classifier_shape": [2, bundle.num_classes], "device": device,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--lem-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    report = {
        dataset: check_dataset(dataset, root, args.seed, device)
        for dataset, root in (("CV", args.cv_root), ("LEM", args.lem_root))
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
