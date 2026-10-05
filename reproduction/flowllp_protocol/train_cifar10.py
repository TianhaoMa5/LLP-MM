#!/usr/bin/env python3
"""Run Particle Flow on CIFAR-10 using the MO-Matching experiment protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import ot
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def squared_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_norm = first.pow(2).sum(dim=1, keepdim=True)
    second_norm = second.pow(2).sum(dim=1, keepdim=True)
    return first_norm + second_norm.t() - 2 * first @ second.t()


def learn_labelprop_anchor(
    bags: list[dict],
    n_per_class_anchor: int,
    num_epochs: int,
    n_class: int,
    dim: int,
    lr: float,
    batch_bag: int,
    reg_label: float,
    clf_bag: nn.Module,
    reg_bagclf: float,
    **_: object,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Isolated upstream particle-flow routine without its broken demo imports."""
    device = torch.device("cpu")
    num_anchors = n_per_class_anchor * n_class
    embedding = nn.Embedding(num_anchors, dim, device=device)
    embedding.weight.data.normal_(mean=0.0, std=2.0)
    class_indices = {
        class_id: np.arange(
            n_per_class_anchor * class_id,
            n_per_class_anchor * (class_id + 1),
        )
        for class_id in range(n_class)
    }
    anchor_labels = torch.arange(n_class).repeat_interleave(n_per_class_anchor)
    optimizer = torch.optim.Adam(embedding.parameters(), lr=lr, betas=(0.9, 0.999))
    clf_bag.eval().to(device)

    for step in range(num_epochs):
        use_noisy_labels = reg_label > 0 and step > 0
        objective = torch.zeros((), device=device)
        order = np.random.permutation(len(bags))[:batch_bag]
        for bag_id in order:
            bag = bags[int(bag_id)]
            features = bag["data"].reshape(-1, dim).to(device)
            source_mass = torch.full((len(features),), 1.0 / len(features))
            target_mass = torch.zeros(num_anchors)
            for class_id in range(n_class):
                target_mass[class_indices[class_id]] = (
                    bag["prop"][class_id] / n_per_class_anchor
                )
            target_mass /= target_mass.sum()
            cost = squared_distance(features, embedding.weight)
            if use_noisy_labels:
                noisy = bag["y_pred_noisy"].to(device)
                mismatch = 1 - (
                    F.one_hot(noisy, n_class).float()
                    @ F.one_hot(anchor_labels, n_class).float().t()
                )
                cost = cost + reg_label * mismatch
            with torch.no_grad():
                transport = ot.emd(source_mass, target_mass, cost)
            objective = objective + (cost * transport).sum()
            objective = objective + reg_bagclf * F.cross_entropy(
                clf_bag(embedding.weight), anchor_labels
            )
        optimizer.zero_grad(set_to_none=True)
        clf_bag.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"particle_step={step + 1}/{num_epochs} loss={float(objective):.6f}")
    return embedding.weight.detach().numpy(), class_indices


class ParticleFlowClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.projector(self.feature_extractor(images))
        logits = self.classifier(features)
        if return_features:
            return features, logits
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bag-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=250)
    parser.add_argument("--finetune-epochs", type=int, default=250)
    parser.add_argument("--samples-per-step", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--cosine-mode", choices=("standard", "quarter"), default="standard")
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--drop-incomplete-batch", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--anchors-per-class", type=int, default=1000)
    parser.add_argument("--anchor-steps", type=int, default=3000)
    parser.add_argument("--anchor-learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-bag-batch", type=int, default=1)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)
    parser.add_argument("--lambda-bag", type=float, default=1.0)
    parser.add_argument("--reg-label", type=float, default=0.0)
    parser.add_argument("--reg-bag-classifier", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-bags", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, max_bags: int | None) -> tuple[np.ndarray, torch.Tensor]:
    manifest = np.load(path)
    indices = manifest["indices"].astype(np.int64)
    proportions = torch.as_tensor(manifest["proportions"], dtype=torch.float32)
    if max_bags is not None:
        indices = indices[:max_bags]
        proportions = proportions[:max_bags]
    if indices.ndim != 2 or proportions.shape != (len(indices), 10):
        raise ValueError(f"Invalid manifest shapes: {indices.shape}, {proportions.shape}")
    if len(np.unique(indices)) != indices.size:
        raise ValueError("Manifest reuses training instances")
    return indices, proportions


def make_datasets(data_dir: Path):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_aug = datasets.CIFAR10(data_dir, train=True, transform=train_transform)
    train_eval = datasets.CIFAR10(data_dir, train=True, transform=eval_transform)
    test = datasets.CIFAR10(data_dir, train=False, transform=eval_transform)
    return train_aug, train_eval, test


def load_images(dataset, indices: np.ndarray) -> torch.Tensor:
    return torch.stack([dataset[int(index)][0] for index in indices.reshape(-1)])


def iter_groups(
    num_bags: int,
    bags_per_step: int,
    rng: np.random.Generator,
    drop_incomplete_batch: bool = False,
):
    order = rng.permutation(num_bags)
    for start in range(0, num_bags, bags_per_step):
        if drop_incomplete_batch and start + bags_per_step > num_bags:
            break
        yield order[start : start + bags_per_step]


def bag_proportion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bag_size: int,
) -> torch.Tensor:
    predictions = F.softmax(logits.reshape(len(targets), bag_size, -1), dim=2).mean(dim=1)
    return -(targets * predictions.clamp_min(1e-7).log()).sum(dim=1).mean()


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    warmup_ratio: float = 0.0,
    cosine_mode: str = "standard",
):
    warmup_steps = int(total_steps * warmup_fraction)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return warmup_ratio + (1.0 - warmup_ratio) * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        if cosine_mode == "quarter":
            return math.cos(math.pi * progress / 2.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    model.train()
    return correct / total


def train_bag_epochs(
    model: ParticleFlowClassifier,
    dataset,
    bag_indices: np.ndarray,
    proportions: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    samples_per_step: int,
    rng: np.random.Generator,
    anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
    lambda_bag: float = 1.0,
    lambda_anchor: float = 0.1,
    drop_incomplete_batch: bool = False,
) -> None:
    bag_size = bag_indices.shape[1]
    bags_per_step = max(1, samples_per_step // bag_size)
    for epoch in range(epochs):
        started = time.monotonic()
        total_loss = 0.0
        steps = 0
        model.train()
        for group in iter_groups(
            len(bag_indices), bags_per_step, rng, drop_incomplete_batch
        ):
            images = load_images(dataset, bag_indices[group]).to(device)
            targets = proportions[group].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = lambda_bag * bag_proportion_loss(logits, targets, bag_size)
            if anchors is not None:
                anchor_features, anchor_labels = anchors
                anchor_logits = model.classifier(anchor_features.to(device))
                loss = loss + lambda_anchor * F.cross_entropy(
                    anchor_logits, anchor_labels.to(device)
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
            steps += 1
        print(
            f"epoch={epoch + 1}/{epochs} loss={total_loss / max(steps, 1):.6f} "
            f"lr={scheduler.get_last_lr()[0]:.8f} seconds={time.monotonic() - started:.2f}"
        )


def learn_particles(
    model: ParticleFlowClassifier,
    dataset,
    bag_indices: np.ndarray,
    proportions: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    bags = []
    model.eval()
    with torch.no_grad():
        for index, target in zip(bag_indices, proportions):
            images = load_images(dataset, index).to(device)
            features, logits = model(images, return_features=True)
            bags.append(
                {
                    "data": features.cpu(),
                    "label": torch.zeros(len(index), dtype=torch.long),
                    "prop": target.tolist(),
                    "y_pred_noisy": logits.argmax(dim=1).cpu(),
                }
            )

    classifier = model.classifier.cpu()
    anchors, class_indices = learn_labelprop_anchor(
        bags,
        n_per_class_anchor=args.anchors_per_class,
        num_epochs=args.anchor_steps,
        n_class=10,
        dim=args.latent_dim,
        lr=args.anchor_learning_rate,
        method="emd",
        batch_bag=min(args.anchor_bag_batch, len(bags)),
        reg_label=args.reg_label,
        clf_bag=classifier,
        reg_bagclf=args.reg_bag_classifier,
        debug=False,
    )
    labels = np.empty(len(anchors), dtype=np.int64)
    for class_id, indices in class_indices.items():
        labels[indices] = class_id
    model.to(device)
    return torch.from_numpy(anchors).float(), torch.from_numpy(labels).long()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment must run on a scheduled GPU node")

    bag_indices, proportions = load_manifest(args.bag_manifest, args.max_bags)
    train_aug, train_eval, test = make_datasets(args.data_dir)
    test_loader = DataLoader(
        test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = ParticleFlowClassifier(args.latent_dim, 10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    bags_per_step = max(1, args.samples_per_step // bag_indices.shape[1])
    steps_per_epoch = (
        len(bag_indices) // bags_per_step
        if args.drop_incomplete_batch
        else math.ceil(len(bag_indices) / bags_per_step)
    )
    if steps_per_epoch < 1:
        raise ValueError("No complete training batch is available")
    total_epochs = args.pretrain_epochs + args.finetune_epochs
    scheduler = make_scheduler(
        optimizer,
        total_epochs * steps_per_epoch,
        args.warmup_fraction,
        args.warmup_ratio,
        args.cosine_mode,
    )
    rng = np.random.default_rng(args.seed)

    print(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(args.data_dir),
                "bag_manifest": str(args.bag_manifest),
                "output_dir": str(args.output_dir),
                "device": str(device),
                "num_bags": len(bag_indices),
                "bag_size": int(bag_indices.shape[1]),
                "bags_per_step": bags_per_step,
                "effective_samples_per_full_step": bags_per_step
                * int(bag_indices.shape[1]),
            },
            sort_keys=True,
        )
    )

    train_bag_epochs(
        model,
        train_aug,
        bag_indices,
        proportions,
        optimizer,
        scheduler,
        device,
        args.pretrain_epochs,
        args.samples_per_step,
        rng,
        drop_incomplete_batch=args.drop_incomplete_batch,
    )
    anchors = learn_particles(
        model, train_eval, bag_indices, proportions, args, device
    )
    train_bag_epochs(
        model,
        train_aug,
        bag_indices,
        proportions,
        optimizer,
        scheduler,
        device,
        args.finetune_epochs,
        args.samples_per_step,
        rng,
        anchors=anchors,
        lambda_bag=args.lambda_bag,
        lambda_anchor=args.lambda_anchor,
        drop_incomplete_batch=args.drop_incomplete_batch,
    )
    accuracy = evaluate(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "accuracy": accuracy,
        "seed": args.seed,
        "bag_manifest": str(args.bag_manifest),
        "total_classifier_epochs": total_epochs,
        "training": {
            "samples_per_step": args.samples_per_step,
            "steps_per_epoch": steps_per_epoch,
            "nesterov": args.nesterov,
            "warmup_fraction": args.warmup_fraction,
            "warmup_ratio": args.warmup_ratio,
            "cosine_mode": args.cosine_mode,
            "drop_incomplete_batch": args.drop_incomplete_batch,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save({"model": model.state_dict(), "result": result}, args.output_dir / "model.pt")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
