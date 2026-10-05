"""Models, losses, manifests, and one-step training for unified LLP-GAN."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


DATASET_SPECS = {
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        num_classes=100,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
}


class FeatureDiscriminator(Protocol):
    training: bool

    def __call__(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def train(self, mode: bool = True) -> nn.Module: ...

    def eval(self) -> nn.Module: ...


class StemResNet18(nn.Module):
    """Torchvision ResNet-18 with the standard CIFAR stem."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise RuntimeError(
                "StemResNet18 requires torchvision. Install the LLP-GAN "
                "requirements before training."
            ) from exc

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError(
                "StemResNet18 expects images shaped [N, 3, 32, 32], "
                f"got {tuple(images.shape)}"
            )
        features = self.backbone(images)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class DCGANGenerator(nn.Module):
    """DCGAN generator mapping normal noise to 32x32 RGB images in [-1, 1]."""

    def __init__(self, z_dim: int = 100) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        self.z_dim = z_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        expected = ("N", self.z_dim, 1, 1)
        if (
            noise.ndim != 4
            or noise.shape[1] != self.z_dim
            or noise.shape[2:] != (1, 1)
        ):
            raise ValueError(
                f"Generator expects noise shaped {expected}, got {tuple(noise.shape)}"
            )
        return self.network(noise)


def normalize_fake_images(
    fake_tanh: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Map generated [-1, 1] pixels into the real-image normalized domain."""
    if fake_tanh.ndim != 4 or fake_tanh.shape[1] != 3:
        raise ValueError(
            "Generated images must have shape [N, 3, H, W], "
            f"got {tuple(fake_tanh.shape)}"
        )
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean and std must contain three values and std must be positive")
    pixels = (fake_tanh + 1.0) / 2.0
    mean_tensor = pixels.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = pixels.new_tensor(std).view(1, 3, 1, 1)
    return (pixels - mean_tensor) / std_tensor


@dataclass(frozen=True)
class BagManifest:
    indices: np.ndarray
    proportions: np.ndarray
    validation_indices: np.ndarray
    source: str

    @property
    def bag_size(self) -> int:
        return int(self.indices.shape[1])

    @property
    def num_bags(self) -> int:
        return int(self.indices.shape[0])


def _validate_integer_indices(
    values: np.ndarray,
    *,
    name: str,
    dataset_size: int,
    dimensions: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-D, got shape {array.shape}")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must use an integer dtype, got {array.dtype}")
    indices = array.astype(np.int64, copy=False)
    if indices.size and (indices.min() < 0 or indices.max() >= dataset_size):
        raise ValueError(
            f"{name} contains indices outside [0, {dataset_size - 1}]"
        )
    return np.ascontiguousarray(indices)


def _compute_proportions(
    indices: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    bag_labels = labels[indices]
    one_hot = np.eye(num_classes, dtype=np.float32)[bag_labels]
    return one_hot.mean(axis=1, dtype=np.float32)


def _validate_proportions(
    values: np.ndarray,
    *,
    num_bags: int,
    num_classes: int,
) -> np.ndarray:
    proportions = np.asarray(values, dtype=np.float32)
    expected = (num_bags, num_classes)
    if proportions.shape != expected:
        raise ValueError(
            f"proportions must have shape {expected}, got {proportions.shape}"
        )
    if not np.isfinite(proportions).all():
        raise ValueError("proportions contains non-finite values")
    if (proportions < -1e-7).any():
        raise ValueError("proportions contains negative values")
    row_sums = proportions.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Every proportions row must sum to one")
    return np.ascontiguousarray(proportions)


def load_bag_manifest(
    path: Path,
    *,
    labels: np.ndarray,
    dataset_size: int,
    bag_size: int,
    num_classes: int,
) -> BagManifest:
    """Load and validate the shared NPZ bag format."""
    if not path.is_file():
        raise FileNotFoundError(f"Bag manifest not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "indices" not in archive:
            raise ValueError(f"{path} does not contain an 'indices' array")
        indices = _validate_integer_indices(
            archive["indices"],
            name="indices",
            dataset_size=dataset_size,
            dimensions=2,
        )
        if indices.shape[1] != bag_size:
            raise ValueError(
                f"Manifest bag size is {indices.shape[1]}, expected {bag_size}"
            )
        if not len(indices):
            raise ValueError("Manifest contains no bags")
        if "proportions" in archive:
            proportions = _validate_proportions(
                archive["proportions"],
                num_bags=len(indices),
                num_classes=num_classes,
            )
        else:
            proportions = _compute_proportions(indices, labels, num_classes)
        if "val_indices" in archive:
            validation_indices = _validate_integer_indices(
                archive["val_indices"],
                name="val_indices",
                dataset_size=dataset_size,
                dimensions=1,
            )
        else:
            validation_indices = np.empty(0, dtype=np.int64)

    if validation_indices.size and np.intersect1d(
        indices.reshape(-1), validation_indices
    ).size:
        raise ValueError("Manifest training bags overlap val_indices")
    return BagManifest(
        indices=indices,
        proportions=proportions,
        validation_indices=validation_indices,
        source=str(path.resolve()),
    )


def create_random_manifest(
    *,
    labels: np.ndarray,
    dataset_size: int,
    bag_size: int,
    num_classes: int,
    seed: int,
) -> BagManifest:
    """Create the same deterministic shuffle-and-partition random bags."""
    if len(labels) != dataset_size:
        raise ValueError("labels length does not match dataset_size")
    rng = np.random.default_rng(seed)
    usable = dataset_size // bag_size * bag_size
    if usable == 0:
        raise ValueError("Dataset is smaller than one bag")
    indices = rng.permutation(dataset_size)[:usable].reshape(-1, bag_size)
    proportions = _compute_proportions(indices, labels, num_classes)
    return BagManifest(
        indices=np.ascontiguousarray(indices, dtype=np.int64),
        proportions=proportions,
        validation_indices=np.empty(0, dtype=np.int64),
        source=f"deterministic-random-seed-{seed}",
    )


class BagDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Apply the image transform independently and return complete bags."""

    def __init__(self, dataset: Dataset, manifest: BagManifest) -> None:
        self.dataset = dataset
        self.indices = manifest.indices
        self.proportions = torch.from_numpy(manifest.proportions)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack(
            [self.dataset[int(instance_index)][0] for instance_index in self.indices[index]]
        )
        return images, self.proportions[index]


def bag_proportion_loss(
    real_logits: torch.Tensor,
    target_proportions: torch.Tensor,
    bag_size: int,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Exact bag-level cross-entropy used by the released LLP-GAN code."""
    if real_logits.ndim != 2:
        raise ValueError("real_logits must have shape [N, K]")
    if target_proportions.ndim != 2:
        raise ValueError("target_proportions must have shape [B, K]")
    num_bags, num_classes = target_proportions.shape
    if real_logits.shape != (num_bags * bag_size, num_classes):
        raise ValueError(
            "Logits/target/bag dimensions are inconsistent: "
            f"logits={tuple(real_logits.shape)}, targets="
            f"{tuple(target_proportions.shape)}, bag_size={bag_size}"
        )
    probabilities = F.softmax(real_logits.float(), dim=1)
    predictions = probabilities.reshape(num_bags, bag_size, num_classes).mean(dim=1)
    targets = target_proportions.to(device=predictions.device, dtype=predictions.dtype)
    return -(targets * predictions.clamp_min(epsilon).log()).sum(dim=1).mean()


def bag_moment_loss(
    real_logits: torch.Tensor,
    target_proportions: torch.Tensor,
    bag_size: int,
    moment_criterion: nn.Module | None = None,
    moment_bag_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return first- or multi-order supervision on complete real bags.

    ``None`` deliberately takes the historical LLP-GAN proportion-loss path.
    This makes ``moment_order=1`` an exact nested control rather than merely a
    numerically similar invocation of the general factorial-moment code.
    Higher-order criteria receive only real-image class probabilities and bag
    proportions; generated images have no trustworthy proportion target.
    """
    if moment_criterion is None:
        return bag_proportion_loss(
            real_logits,
            target_proportions,
            bag_size,
        )
    if real_logits.ndim != 2:
        raise ValueError("real_logits must have shape [N, K]")
    if target_proportions.ndim != 2:
        raise ValueError("target_proportions must have shape [B, K]")
    num_bags, num_classes = target_proportions.shape
    if real_logits.shape != (num_bags * bag_size, num_classes):
        raise ValueError(
            "Logits/target/bag dimensions are inconsistent: "
            f"logits={tuple(real_logits.shape)}, targets="
            f"{tuple(target_proportions.shape)}, bag_size={bag_size}"
        )
    probabilities = F.softmax(real_logits.float(), dim=1)
    targets = target_proportions.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    if moment_bag_chunk_size is None or moment_bag_chunk_size >= num_bags:
        return moment_criterion(targets, probabilities)
    if moment_bag_chunk_size <= 0:
        raise ValueError("moment_bag_chunk_size must be positive when provided")

    probability_bags = probabilities.reshape(
        num_bags,
        bag_size,
        num_classes,
    )
    weighted_losses = []
    for start in range(0, num_bags, moment_bag_chunk_size):
        stop = min(start + moment_bag_chunk_size, num_bags)
        chunk_targets = targets[start:stop]
        chunk_probabilities = probability_bags[start:stop].reshape(
            (stop - start) * bag_size,
            num_classes,
        )
        if chunk_probabilities.requires_grad:
            chunk_loss = checkpoint(
                moment_criterion,
                chunk_targets,
                chunk_probabilities,
                use_reentrant=False,
            )
        else:
            chunk_loss = moment_criterion(
                chunk_targets,
                chunk_probabilities,
            )
        weighted_losses.append(chunk_loss * ((stop - start) / num_bags))
    return torch.stack(weighted_losses).sum()


def adversarial_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Implicit K+1-class semi-supervised GAN discriminator loss."""
    if real_logits.ndim != 2 or fake_logits.ndim != 2:
        raise ValueError("real_logits and fake_logits must both be 2-D")
    if real_logits.shape[1] != fake_logits.shape[1]:
        raise ValueError("Real and fake logits must have the same class dimension")
    real_energy = torch.logsumexp(real_logits.float(), dim=1)
    fake_energy = torch.logsumexp(fake_logits.float(), dim=1)
    real_loss = (-real_energy + F.softplus(real_energy)).mean()
    fake_loss = F.softplus(fake_energy).mean()
    return real_loss + fake_loss


def feature_matching_loss(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
) -> torch.Tensor:
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("Feature matching expects [N, F] tensors")
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("Real and fake feature dimensions do not match")
    return F.mse_loss(
        fake_features.float().mean(dim=0),
        real_features.float().mean(dim=0),
    )


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@contextmanager
def frozen_batchnorm_stats(module: nn.Module) -> Iterator[None]:
    """Use batch statistics without updating BatchNorm running buffers."""
    batch_norms = [
        child
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    tracking_states = [child.track_running_stats for child in batch_norms]
    for child in batch_norms:
        child.track_running_stats = False
    try:
        yield
    finally:
        for child, was_tracking in zip(batch_norms, tracking_states):
            child.track_running_stats = was_tracking


@contextmanager
def frozen_discriminator(discriminator: nn.Module) -> Iterator[None]:
    """Freeze parameters and BN statistics while retaining input gradients."""
    was_training = discriminator.training
    requires_grad = [parameter.requires_grad for parameter in discriminator.parameters()]
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, original in zip(discriminator.parameters(), requires_grad):
            parameter.requires_grad_(original)
        discriminator.train(was_training)


def gradient_l2_norm(parameters: Iterator[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def _optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    parameters: Iterator[nn.Parameter],
    max_grad_norm: float | None = None,
) -> float:
    parameter_list = list(parameters)
    if scaler is None:
        loss.backward()
        grad_norm = gradient_l2_norm(iter(parameter_list))
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameter_list,
                max_grad_norm,
                error_if_nonfinite=True,
            )
        optimizer.step()
        return grad_norm
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = gradient_l2_norm(iter(parameter_list))
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            parameter_list,
            max_grad_norm,
            error_if_nonfinite=False,
        )
    scaler.step(optimizer)
    scaler.update()
    return grad_norm


@dataclass(frozen=True)
class IterationMetrics:
    discriminator_loss: float
    proportion_loss: float
    adversarial_loss: float
    generator_loss: float
    discriminator_grad_norm: float
    generator_grad_norm: float
    real_samples: int


def train_iteration(
    *,
    discriminator: nn.Module,
    generator: DCGANGenerator,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    bag_images: torch.Tensor,
    target_proportions: torch.Tensor,
    device: torch.device,
    bag_size: int,
    num_classes: int,
    z_dim: int,
    mean: Sequence[float],
    std: Sequence[float],
    lambda_prop: float,
    lambda_adv: float,
    amp: bool,
    moment_criterion: nn.Module | None = None,
    moment_bag_chunk_size: int | None = None,
    scaler_d: torch.cuda.amp.GradScaler | None = None,
    scaler_g: torch.cuda.amp.GradScaler | None = None,
    d_max_grad_norm: float | None = None,
    g_max_grad_norm: float | None = None,
) -> IterationMetrics:
    """Perform one discriminator update followed by one generator update."""
    if bag_images.ndim != 5 or bag_images.shape[1:] != (bag_size, 3, 32, 32):
        raise ValueError(
            "bag_images must have shape [B, bag_size, 3, 32, 32], "
            f"got {tuple(bag_images.shape)}"
        )
    if target_proportions.shape != (bag_images.shape[0], num_classes):
        raise ValueError(
            "target_proportions has the wrong shape: "
            f"{tuple(target_proportions.shape)}"
        )
    if lambda_prop < 0 or lambda_adv < 0:
        raise ValueError("Loss weights must be non-negative")
    if d_max_grad_norm is not None and d_max_grad_norm <= 0:
        raise ValueError("d_max_grad_norm must be positive when provided")
    if g_max_grad_norm is not None and g_max_grad_norm <= 0:
        raise ValueError("g_max_grad_norm must be positive when provided")

    num_real = int(bag_images.shape[0] * bag_size)
    real_images = bag_images.reshape(num_real, 3, 32, 32).to(
        device, non_blocking=True
    )
    target_proportions = target_proportions.to(device, non_blocking=True)
    autocast_enabled = bool(amp and device.type == "cuda")

    discriminator.train()
    generator.train()
    optimizer_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        noise = torch.randn(num_real, z_dim, 1, 1, device=device)
        fake_tanh = generator(noise)
        fake_images = normalize_fake_images(fake_tanh, mean, std)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):
        real_logits = discriminator(real_images)
        # Running statistics are classifier state and should describe real
        # CIFAR images. Fake batches still train every discriminator
        # parameter, including BatchNorm affine weights, without contaminating
        # those statistics.
        with frozen_batchnorm_stats(discriminator):
            fake_logits = discriminator(fake_images.detach())
        if not isinstance(real_logits, torch.Tensor) or not isinstance(
            fake_logits, torch.Tensor
        ):
            raise TypeError("Discriminator must return logits unless features are requested")
        loss_prop = bag_moment_loss(
            real_logits,
            target_proportions,
            bag_size,
            moment_criterion,
            moment_bag_chunk_size,
        )
        loss_adv = adversarial_loss(real_logits, fake_logits)
        loss_d = lambda_prop * loss_prop + lambda_adv * loss_adv
    d_grad_norm = _optimizer_step(
        loss_d,
        optimizer_d,
        scaler_d,
        discriminator.parameters(),
        d_max_grad_norm,
    )

    optimizer_g.zero_grad(set_to_none=True)
    noise = torch.randn(num_real, z_dim, 1, 1, device=device)
    with frozen_discriminator(discriminator):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            generated_tanh = generator(noise)
            generated = normalize_fake_images(generated_tanh, mean, std)
            with torch.no_grad():
                real_result = discriminator(real_images, return_features=True)
            fake_result = discriminator(generated, return_features=True)
            if not (
                isinstance(real_result, tuple)
                and isinstance(fake_result, tuple)
                and len(real_result) == 2
                and len(fake_result) == 2
            ):
                raise TypeError(
                    "Discriminator must return (logits, features) when requested"
                )
            _, real_features = real_result
            _, fake_features = fake_result
            loss_g = feature_matching_loss(real_features, fake_features)
        g_grad_norm = _optimizer_step(
            loss_g,
            optimizer_g,
            scaler_g,
            generator.parameters(),
            g_max_grad_norm,
        )

    values = (loss_d, loss_prop, loss_adv, loss_g)
    if not all(torch.isfinite(value).item() for value in values):
        raise FloatingPointError("LLP-GAN produced a non-finite loss")
    return IterationMetrics(
        discriminator_loss=float(loss_d.detach()),
        proportion_loss=float(loss_prop.detach()),
        adversarial_loss=float(loss_adv.detach()),
        generator_loss=float(loss_g.detach()),
        discriminator_grad_norm=d_grad_norm,
        generator_grad_norm=g_grad_norm,
        real_samples=num_real,
    )
