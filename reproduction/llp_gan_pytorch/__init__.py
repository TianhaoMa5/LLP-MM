"""Unified PyTorch LLP-GAN components for CIFAR experiments."""

from .core import (
    DATASET_SPECS,
    BagDataset,
    BagManifest,
    DCGANGenerator,
    DatasetSpec,
    StemResNet18,
    adversarial_loss,
    bag_proportion_loss,
    create_random_manifest,
    feature_matching_loss,
    load_bag_manifest,
    make_warmup_cosine_scheduler,
    normalize_fake_images,
    train_iteration,
)

__all__ = [
    "DATASET_SPECS",
    "BagDataset",
    "BagManifest",
    "DCGANGenerator",
    "DatasetSpec",
    "StemResNet18",
    "adversarial_loss",
    "bag_proportion_loss",
    "create_random_manifest",
    "feature_matching_loss",
    "load_bag_manifest",
    "make_warmup_cosine_scheduler",
    "normalize_fake_images",
    "train_iteration",
]
