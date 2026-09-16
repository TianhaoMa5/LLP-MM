"""Tiny fully supervised CCT crop overfit check (diagnostic, not an LLP result)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from plench.core import networks
from plench.data.cct import _image_transform, load_cct_bundle


class _CropSubset(Dataset):
    def __init__(self, paths, targets, *, split: str) -> None:
        self.paths = list(paths)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.transform = _image_transform(split)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            value = self.transform(image.convert("RGB"))
        return value, self.targets[index]


def _accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            prediction = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((prediction == targets).sum())
            total += len(targets)
    return correct / max(1, total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that ImageNet-pretrained ResNet-18 at 112x112 can overfit a "
            "tiny labeled subset of the prepared CCT bbox crops."
        )
    )
    parser.add_argument("--data_root", "--data-root", required=True)
    parser.add_argument("--samples_per_class", "--samples-per-class", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_accuracy", "--min-accuracy", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no_pretrained", "--no-pretrained", action="store_true",
        help="test-only escape hatch; the reference diagnostic defaults to ImageNet weights",
    )
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples_per_class <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("samples_per_class, steps, and batch_size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bundle = load_cct_bundle(args.data_root)
    rng = np.random.default_rng(args.seed)
    selected: list[int] = []
    per_class: dict[str, int] = {}
    for class_index, class_name in enumerate(bundle.class_names):
        candidates = np.flatnonzero(
            (bundle.splits == "train") & (bundle.targets == class_index)
        )
        if not len(candidates):
            raise ValueError(f"No CCT training crop for class {class_name!r}")
        count = min(int(args.samples_per_class), len(candidates))
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.extend(int(index) for index in chosen)
        per_class[class_name] = count
    rng.shuffle(selected)

    train_dataset = _CropSubset(
        bundle.image_paths[selected], bundle.targets[selected], split="train"
    )
    eval_dataset = _CropSubset(
        bundle.image_paths[selected], bundle.targets[selected], split="test"
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=min(args.batch_size, len(eval_dataset)),
        shuffle=False,
        num_workers=0,
    )
    requested_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(requested_device)
    hparams = {
        "model": "CCTResNet18",
        "pretrained": not args.no_pretrained,
    }
    featurizer = networks.Featurizer(bundle.input_shape, hparams)
    model = torch.nn.Sequential(
        featurizer, networks.Classifier(featurizer.n_outputs, bundle.num_classes)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    iterator = iter(train_loader)
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, targets = next(iterator)
        model.train()
        logits = model(images.to(device))
        loss = F.cross_entropy(logits, targets.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        if step == 1 or step % 100 == 0 or step == args.steps:
            accuracy = _accuracy(model, eval_loader, device)
            print(
                f"supervised crop overfit step={step}/{args.steps} "
                f"loss={last_loss:.6f} deterministic_subset_accuracy={accuracy:.4f}",
                flush=True,
            )

    final_accuracy = _accuracy(model, eval_loader, device)
    result = {
        "diagnostic_only_not_llp": True,
        "cache_namespace": bundle.processed_root.name,
        "instance_type": "bbox_crop",
        "backbone": "resnet18",
        "pretrained_weights": getattr(featurizer, "pretrained_weights", None),
        "input_resolution": 112,
        "subset_instances": len(selected),
        "samples_per_class": per_class,
        "steps": int(args.steps),
        "optimizer": "AdamW",
        "learning_rate": float(args.lr),
        "final_loss": last_loss,
        "deterministic_subset_accuracy": final_accuracy,
        "minimum_required_accuracy": float(args.min_accuracy),
        "passed": final_accuracy >= float(args.min_accuracy),
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else bundle.processed_root / "diagnostics" / "tiny_supervised_overfit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"saved diagnostic: {output}", flush=True)
    if not result["passed"]:
        raise SystemExit(
            f"Tiny supervised CCT crop overfit failed: {final_accuracy:.4f} "
            f"< {args.min_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
