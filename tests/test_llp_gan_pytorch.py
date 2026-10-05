from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from reproduction.llp_gan_pytorch.core import (
    _optimizer_step,
    BagDataset,
    BagManifest,
    DCGANGenerator,
    StemResNet18,
    adversarial_loss,
    bag_moment_loss,
    bag_proportion_loss,
    create_random_manifest,
    feature_matching_loss,
    frozen_batchnorm_stats,
    frozen_discriminator,
    load_bag_manifest,
    make_warmup_cosine_scheduler,
    normalize_fake_images,
    train_iteration,
)
from reproduction.llp_gan_pytorch.train import promote_checkpoint, save_checkpoint
from mo_matching.llp.multiclass import MulticlassFactorialMomentLoss


class TensorImageDataset(Dataset):
    def __init__(self, count: int) -> None:
        self.images = torch.linspace(
            -1.0, 1.0, count * 3 * 32 * 32
        ).reshape(count, 3, 32, 32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], index % 3


class TinyFeatureDiscriminator(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = self.features(images).flatten(1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def test_generator_shape_range_and_initialization() -> None:
    torch.manual_seed(0)
    generator = DCGANGenerator(z_dim=16).eval()
    with torch.no_grad():
        images = generator(torch.randn(3, 16, 1, 1))
    assert images.shape == (3, 3, 32, 32)
    assert images.min().item() >= -1.0
    assert images.max().item() <= 1.0
    first_weight = next(
        module.weight
        for module in generator.modules()
        if isinstance(module, nn.ConvTranspose2d)
    )
    assert first_weight.mean().item() == pytest.approx(0.0, abs=2e-3)
    assert first_weight.std().item() == pytest.approx(0.02, rel=0.08)


def test_fake_normalization_matches_pixel_conversion() -> None:
    fake = torch.tensor(
        [
            [[[-1.0]], [[0.0]], [[1.0]]],
        ]
    )
    normalized = normalize_fake_images(
        fake,
        mean=(0.5, 0.25, 0.75),
        std=(0.5, 0.25, 0.25),
    )
    assert torch.allclose(
        normalized.flatten(),
        torch.tensor([-1.0, 1.0, 1.0]),
    )


def test_losses_are_finite_and_have_gradients() -> None:
    torch.manual_seed(1)
    real_logits = torch.randn(8, 3, requires_grad=True)
    fake_logits = torch.randn(8, 3, requires_grad=True)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.5, 0.5]])
    loss_prop = bag_proportion_loss(real_logits, targets, bag_size=4)
    loss_adv = adversarial_loss(real_logits, fake_logits)
    loss = loss_prop + loss_adv
    assert torch.isfinite(loss_prop)
    assert torch.isfinite(loss_adv)
    loss.backward()
    assert real_logits.grad is not None
    assert fake_logits.grad is not None
    assert real_logits.grad.norm().item() > 0
    assert fake_logits.grad.norm().item() > 0

    real_features = torch.randn(8, 5)
    fake_features = torch.randn(8, 5, requires_grad=True)
    loss_g = feature_matching_loss(real_features, fake_features)
    loss_g.backward()
    assert torch.isfinite(loss_g)
    assert fake_features.grad is not None
    assert fake_features.grad.norm().item() > 0


def test_bag_loss_rejects_flattening_across_bags() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        bag_proportion_loss(
            torch.randn(7, 3),
            torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.5, 0.5]]),
            bag_size=4,
        )


def test_order_one_moment_control_is_exact_proportion_loss() -> None:
    torch.manual_seed(17)
    logits = torch.randn(8, 3, requires_grad=True)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.5, 0.5]])
    historical = bag_proportion_loss(logits, targets, bag_size=4)
    nested_control = bag_moment_loss(
        logits,
        targets,
        bag_size=4,
        moment_criterion=None,
    )
    assert torch.equal(historical, nested_control)


def test_high_order_bag_moment_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(18)
    logits = torch.randn(8, 3, requires_grad=True)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.5, 0.5]])
    criterion = MulticlassFactorialMomentLoss(
        num_classes=3,
        max_order=2,
        bag_size=4,
        loss_type="ce",
    )
    loss = bag_moment_loss(
        logits,
        targets,
        bag_size=4,
        moment_criterion=criterion,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm().item() > 0


def test_checkpointed_bag_chunks_match_full_moment_objective() -> None:
    torch.manual_seed(19)
    full_logits = torch.randn(12, 3, requires_grad=True)
    chunked_logits = full_logits.detach().clone().requires_grad_(True)
    targets = torch.tensor(
        [
            [0.5, 0.25, 0.25],
            [0.0, 0.5, 0.5],
            [0.25, 0.5, 0.25],
        ]
    )
    criterion = MulticlassFactorialMomentLoss(
        num_classes=3,
        max_order=2,
        bag_size=4,
        loss_type="ce",
    )
    full = bag_moment_loss(
        full_logits,
        targets,
        bag_size=4,
        moment_criterion=criterion,
    )
    chunked = bag_moment_loss(
        chunked_logits,
        targets,
        bag_size=4,
        moment_criterion=criterion,
        moment_bag_chunk_size=1,
    )
    full.backward()
    chunked.backward()
    assert chunked.item() == pytest.approx(full.item(), rel=1e-6, abs=1e-7)
    assert torch.allclose(
        chunked_logits.grad,
        full_logits.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_manifest_loader_and_bag_dataset(tmp_path) -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
    indices = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    path = tmp_path / "bags.npz"
    np.savez(path, indices=indices)
    manifest = load_bag_manifest(
        path,
        labels=labels,
        dataset_size=len(labels),
        bag_size=4,
        num_classes=3,
    )
    assert manifest.proportions.shape == (2, 3)
    assert np.allclose(manifest.proportions.sum(axis=1), 1.0)
    images, proportions = BagDataset(TensorImageDataset(8), manifest)[0]
    assert images.shape == (4, 3, 32, 32)
    assert proportions.shape == (3,)


@pytest.mark.parametrize(
    ("indices", "proportions", "bag_size", "message"),
    [
        (
            np.asarray([[0, 1, 2, 8]], dtype=np.int64),
            np.asarray([[0.25, 0.25, 0.5]], dtype=np.float32),
            4,
            "outside",
        ),
        (
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            np.asarray([[0.25, 0.25, 0.25]], dtype=np.float32),
            4,
            "sum to one",
        ),
        (
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            np.asarray([[0.25, 0.25, 0.5]], dtype=np.float32),
            8,
            "bag size",
        ),
    ],
)
def test_manifest_validation_errors(
    tmp_path,
    indices: np.ndarray,
    proportions: np.ndarray,
    bag_size: int,
    message: str,
) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, indices=indices, proportions=proportions)
    with pytest.raises(ValueError, match=message):
        load_bag_manifest(
            path,
            labels=np.asarray([0, 1, 2, 0, 1, 2, 0, 1]),
            dataset_size=8,
            bag_size=bag_size,
            num_classes=3,
        )


def test_random_manifest_is_reproducible() -> None:
    labels = np.arange(64) % 4
    first = create_random_manifest(
        labels=labels,
        dataset_size=len(labels),
        bag_size=8,
        num_classes=4,
        seed=7,
    )
    second = create_random_manifest(
        labels=labels,
        dataset_size=len(labels),
        bag_size=8,
        num_classes=4,
        seed=7,
    )
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.proportions, second.proportions)


def test_frozen_discriminator_preserves_batchnorm_statistics() -> None:
    model = TinyFeatureDiscriminator(3)
    model.train()
    batch_norm = next(
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    )
    before = batch_norm.running_mean.clone()
    with frozen_discriminator(model):
        inputs = torch.randn(4, 3, 32, 32, requires_grad=True)
        logits = model(inputs)
        logits.sum().backward()
        assert inputs.grad is not None
        assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training
    assert torch.equal(before, batch_norm.running_mean)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_frozen_batchnorm_stats_retains_discriminator_gradients() -> None:
    model = TinyFeatureDiscriminator(3)
    model.train()
    batch_norm = next(
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    )
    before_mean = batch_norm.running_mean.clone()
    before_var = batch_norm.running_var.clone()
    inputs = torch.randn(4, 3, 32, 32, requires_grad=True)

    with frozen_batchnorm_stats(model):
        logits = model(inputs)
        logits.sum().backward()
        assert model.training
        assert batch_norm.training
        assert not batch_norm.track_running_stats

    assert model.training
    assert batch_norm.training
    assert batch_norm.track_running_stats
    assert torch.equal(before_mean, batch_norm.running_mean)
    assert torch.equal(before_var, batch_norm.running_var)
    assert inputs.grad is not None
    assert inputs.grad.norm().item() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_optimizer_step_clips_large_gradients() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = model.weight.detach().clone()
    loss = model(torch.full((1, 2), 1000.0)).sum()

    preclip_norm = _optimizer_step(
        loss,
        optimizer,
        scaler=None,
        parameters=model.parameters(),
        max_grad_norm=0.25,
    )

    update_norm = (model.weight.detach() - before).norm().item()
    assert preclip_norm > 1000
    assert update_norm == pytest.approx(0.25, rel=1e-5)


def test_one_iteration_updates_both_networks() -> None:
    torch.manual_seed(3)
    discriminator = TinyFeatureDiscriminator(3)
    generator = DCGANGenerator(z_dim=8)
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.01)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    bag_images = torch.randn(2, 4, 3, 32, 32)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]])
    moment_criterion = MulticlassFactorialMomentLoss(
        num_classes=3,
        max_order=2,
        bag_size=4,
        loss_type="ce",
    )
    before_d = discriminator.classifier.weight.detach().clone()
    before_g = generator.network[0].weight.detach().clone()

    metrics = train_iteration(
        discriminator=discriminator,
        generator=generator,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
        bag_images=bag_images,
        target_proportions=targets,
        device=torch.device("cpu"),
        bag_size=4,
        num_classes=3,
        z_dim=8,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        lambda_prop=1.0,
        lambda_adv=1.0,
        amp=False,
        moment_criterion=moment_criterion,
    )
    assert np.isfinite(
        [
            metrics.discriminator_loss,
            metrics.proportion_loss,
            metrics.adversarial_loss,
            metrics.generator_loss,
        ]
    ).all()
    assert metrics.discriminator_grad_norm > 0
    assert metrics.generator_grad_norm > 0
    assert not torch.equal(before_d, discriminator.classifier.weight)
    assert not torch.equal(before_g, generator.network[0].weight)


def test_scheduler_and_checkpoint_round_trip(tmp_path) -> None:
    discriminator = TinyFeatureDiscriminator(3)
    generator = DCGANGenerator(z_dim=8)
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.05)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=3e-4)
    scheduler = make_warmup_cosine_scheduler(
        optimizer_d, total_steps=4, warmup_steps=0
    )
    for _ in range(2):
        optimizer_d.step()
        scheduler.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        epoch=2,
        discriminator=discriminator,
        generator=generator,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
        scheduler_d=scheduler,
        scaler_d=None,
        scaler_g=None,
        config={"dataset": "synthetic"},
        validation_accuracy=0.4,
        test_accuracy=0.5,
    )
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["epoch"] == 2
    assert restored["validation_accuracy"] == 0.4
    assert restored["test_accuracy"] == 0.5
    assert set(restored) >= {
        "discriminator",
        "generator",
        "optimizer_d",
        "optimizer_g",
        "scheduler_d",
        "config",
    }
    json.dumps(restored["config"])
    promoted = tmp_path / "checkpoint_best.pt"
    promote_checkpoint(path, promoted)
    promoted_state = torch.load(promoted, map_location="cpu", weights_only=False)
    assert promoted_state["epoch"] == 2


def test_stem_resnet18_output_shapes() -> None:
    pytest.importorskip("torchvision")
    model = StemResNet18(num_classes=10).eval()
    with torch.no_grad():
        logits, features = model(
            torch.randn(2, 3, 32, 32), return_features=True
        )
    assert logits.shape == (2, 10)
    assert features.shape == (2, 512)
    assert model.backbone.conv1.kernel_size == (3, 3)
    assert model.backbone.conv1.stride == (1, 1)
    assert isinstance(model.backbone.maxpool, nn.Identity)
