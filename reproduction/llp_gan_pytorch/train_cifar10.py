#!/usr/bin/env python3
"""Modern PyTorch port of the authors' TensorFlow 1 CIFAR-10 LLP-GAN code."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512, bias=False),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.image = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 3, 5, 2, 2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        projected = self.project(noise)
        return self.image(projected.reshape(noise.shape[0], 512, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images).mean(dim=(2, 3))
        return self.classifier(features), features


@dataclass
class EpochMetrics:
    epoch: int
    discriminator_loss: float
    proportion_loss: float
    generator_loss: float
    test_error: float
    seconds: float


class ManifestBagDataset(Dataset):
    """Return complete bags and their observed proportions."""

    def __init__(
        self,
        dataset: Dataset,
        indices: np.ndarray,
        proportions: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.proportions = torch.as_tensor(proportions, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([self.dataset[int(i)][0] for i in self.indices[index]])
        return images, self.proportions[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("reproduction/data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction/artifacts/llpgan"),
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--bag-size", type=int, default=36)
    parser.add_argument(
        "--bag-manifest",
        type=Path,
        help="Shared .npz manifest. Its bag size overrides --bag-size.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limit bags per epoch for a fast scheduler smoke test.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    data_dir: Path,
    bag_size: int,
    bag_manifest: Path | None,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    loader_args = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if bag_manifest is not None:
        manifest = np.load(bag_manifest)
        indices = manifest["indices"].astype(np.int64)
        proportions = manifest["proportions"].astype(np.float32)
        if indices.ndim != 2 or proportions.shape != (len(indices), 10):
            raise ValueError(f"Invalid bag manifest shapes: {indices.shape}, {proportions.shape}")
        train_loader = DataLoader(
            ManifestBagDataset(train_data, indices, proportions),
            batch_size=None,
            shuffle=False,
            **loader_args,
        )
    else:
        # Preserve the official LLP-GAN fixed random bag ordering as a fallback.
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_data), generator=generator).tolist()
        train_data = Subset(train_data, indices)
        train_loader = DataLoader(
            train_data,
            batch_size=bag_size,
            shuffle=False,
            drop_last=True,
            **loader_args,
        )
    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_args,
    )
    return train_loader, test_loader


def evaluate(
    discriminator: Discriminator,
    loader: DataLoader,
    device: torch.device,
) -> float:
    discriminator.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits, _ = discriminator(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return 1.0 - correct / total


def save_state(
    output_dir: Path,
    generator: Generator,
    discriminator: Discriminator,
    args: argparse.Namespace,
    metrics: EpochMetrics,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "args": vars(args),
            "metrics": asdict(metrics),
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metrics)) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_loader, test_loader = make_loaders(
        args.data_dir,
        args.bag_size,
        args.bag_manifest,
        args.eval_batch_size,
        args.seed,
        args.num_workers,
    )
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999),
    )
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999),
    )

    for epoch in range(args.epochs):
        started = time.monotonic()
        totals = torch.zeros(4)
        steps = 0
        generator.train()
        discriminator.train()

        for images, bag_targets in train_loader:
            if args.max_steps is not None and steps >= args.max_steps:
                break
            images = images.to(device)
            current_size = images.shape[0]
            if args.bag_manifest is None:
                proportions = F.one_hot(
                    bag_targets.to(device), num_classes=10
                ).float().mean(dim=0)
            else:
                proportions = bag_targets.to(device)

            d_optimizer.zero_grad(set_to_none=True)
            noise = torch.empty(current_size, args.latent_dim, device=device).uniform_(-1, 1)
            fake_images = generator(noise).detach()
            real_logits, _ = discriminator(images)
            fake_logits, _ = discriminator(fake_images)
            real_energy = torch.logsumexp(real_logits, dim=1)
            fake_energy = torch.logsumexp(fake_logits, dim=1)
            unsupervised_loss = (
                -real_energy.mean()
                + F.softplus(real_energy).mean()
                + F.softplus(fake_energy).mean()
            )
            predicted_proportions = F.softmax(real_logits, dim=1).mean(dim=0)
            proportion_loss = -torch.sum(
                proportions * torch.log(predicted_proportions.clamp_min(1e-7))
            )
            discriminator_loss = unsupervised_loss + proportion_loss
            discriminator_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad(set_to_none=True)
            noise = torch.empty(current_size, args.latent_dim, device=device).uniform_(-1, 1)
            generated = generator(noise)
            _, fake_features = discriminator(generated)
            with torch.no_grad():
                _, real_features = discriminator(images)
            generator_loss = F.mse_loss(
                fake_features.mean(dim=0),
                real_features.mean(dim=0),
            )
            generator_loss.backward()
            g_optimizer.step()

            totals += torch.tensor(
                [
                    discriminator_loss.item(),
                    proportion_loss.item(),
                    generator_loss.item(),
                    current_size,
                ]
            )
            steps += 1

        if steps == 0:
            raise RuntimeError("No training bags were produced.")
        test_error = evaluate(discriminator, test_loader, device)
        metrics = EpochMetrics(
            epoch=epoch,
            discriminator_loss=(totals[0] / steps).item(),
            proportion_loss=(totals[1] / steps).item(),
            generator_loss=(totals[2] / steps).item(),
            test_error=test_error,
            seconds=time.monotonic() - started,
        )
        save_state(args.output_dir, generator, discriminator, args, metrics)
        print(json.dumps(asdict(metrics)))


if __name__ == "__main__":
    main()
