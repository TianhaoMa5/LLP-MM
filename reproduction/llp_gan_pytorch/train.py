"""Training entry point for Stem-ResNet18 LLP-GAN on CIFAR-10/100."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    from mo_matching.llp.multiclass import MulticlassFactorialMomentLoss
except ModuleNotFoundError as exc:
    if exc.name != "mo_matching":
        raise
    # Direct source-tree runs do not require an editable package install.
    from src.mo_matching.llp.multiclass import MulticlassFactorialMomentLoss

from .core import (
    DATASET_SPECS,
    BagDataset,
    BagManifest,
    DCGANGenerator,
    StemResNet18,
    create_random_manifest,
    load_bag_manifest,
    make_warmup_cosine_scheduler,
    train_iteration,
)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    steps: int
    discriminator_loss: float
    proportion_loss: float
    adversarial_loss: float
    generator_loss: float
    discriminator_grad_norm: float
    generator_grad_norm: float
    discriminator_lr: float
    validation_accuracy: float | None
    test_accuracy: float
    seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified PyTorch LLP-GAN baseline for CIFAR-10/100"
    )
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_SPECS), default="cifar10"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("reproduction/data")
    )
    parser.add_argument("--bag-file", type=Path)
    parser.add_argument(
        "--bag-type",
        choices=("random", "cluster", "alpha_first", "custom"),
        required=True,
    )
    parser.add_argument(
        "--bag-size", type=int, choices=(16, 32, 64, 128), required=True
    )
    parser.add_argument("--samples-per-step", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--d-lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument(
        "--g-optimizer", choices=("adam", "sgd"), default="adam"
    )
    parser.add_argument("--g-lr", type=float, default=3e-4)
    parser.add_argument("--g-beta1", type=float, default=0.5)
    parser.add_argument("--g-beta2", type=float, default=0.999)
    parser.add_argument("--g-momentum", type=float, default=0.9)
    parser.add_argument("--g-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lambda-prop",
        "--lambda-mm",
        dest="lambda_prop",
        type=float,
        default=1.0,
        help=(
            "Weight of real-bag supervision. --lambda-mm is an alias used "
            "when --moment-order is greater than one."
        ),
    )
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument(
        "--moment-order",
        type=int,
        default=1,
        help=(
            "Maximum exact categorical factorial-moment order. One preserves "
            "the historical LLP-GAN proportion loss exactly."
        ),
    )
    parser.add_argument(
        "--mm-order-weights",
        type=float,
        nargs="+",
        help=(
            "Optional non-negative weights for orders 1..moment-order. "
            "Weights are normalized internally; the default is uniform."
        ),
    )
    parser.add_argument(
        "--mm-power-chunk-size",
        type=int,
        default=64,
        help="Composition chunk size used by exact multi-order moment matching.",
    )
    parser.add_argument(
        "--mm-bag-chunk-size",
        type=int,
        default=0,
        help=(
            "Checkpoint the exact moment objective in chunks of this many "
            "bags; zero evaluates all bags together. This changes memory and "
            "recomputation only, not the batch objective."
        ),
    )
    parser.add_argument(
        "--mm-implementation",
        choices=("factorial_mass", "paper_dp"),
        default="factorial_mass",
        help=(
            "Exact moment implementation. paper_dp uses the optimized "
            "LLPHighOrderLoss from the paper's plench runtime."
        ),
    )
    parser.add_argument(
        "--d-max-grad-norm",
        type=float,
        default=0.0,
        help="Clip discriminator gradients to this norm; zero disables clipping.",
    )
    parser.add_argument(
        "--g-max-grad-norm",
        type=float,
        default=0.0,
        help="Clip generator gradients to this norm; zero disables clipping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save a numbered checkpoint every N epochs; zero disables it.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string such as cuda, cuda:0, or cpu; default auto.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Bound steps per epoch for a diagnostic smoke test.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        help="Bound validation/test batches for a diagnostic smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing artifacts in an existing run directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_step <= 0:
        raise ValueError("--samples-per-step must be positive")
    if args.samples_per_step % args.bag_size:
        raise ValueError(
            "--samples-per-step must be exactly divisible by --bag-size"
        )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs)")
    if min(args.d_lr, args.g_lr) <= 0:
        raise ValueError("Learning rates must be positive")
    if args.z_dim <= 0:
        raise ValueError("--z-dim must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be positive")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if args.lambda_prop < 0 or args.lambda_adv < 0:
        raise ValueError("Loss weights must be non-negative")
    if not 1 <= args.moment_order <= 14:
        raise ValueError("--moment-order must be in [1, 14]")
    if args.moment_order > args.bag_size:
        raise ValueError("--moment-order cannot exceed --bag-size")
    if args.mm_order_weights is not None:
        if len(args.mm_order_weights) != args.moment_order:
            raise ValueError(
                "--mm-order-weights must contain exactly --moment-order values"
            )
        weights = np.asarray(args.mm_order_weights, dtype=np.float64)
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError(
                "--mm-order-weights must be finite, non-negative, and nonzero"
            )
    if args.mm_power_chunk_size <= 0:
        raise ValueError("--mm-power-chunk-size must be positive")
    if args.mm_bag_chunk_size < 0:
        raise ValueError("--mm-bag-chunk-size cannot be negative")
    if args.d_max_grad_norm < 0 or args.g_max_grad_norm < 0:
        raise ValueError("Gradient clipping norms cannot be negative")
    if args.bag_type != "random" and args.bag_file is None:
        raise ValueError(
            f"--bag-type {args.bag_type} requires --bag-file; "
            "it will not fall back to random bags"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def make_datasets(args: argparse.Namespace):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torchvision. Run "
            "'bash reproduction/scripts/setup_llpgan.sh' first."
        ) from exc

    spec = DATASET_SPECS[args.dataset]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )
    dataset_class = (
        datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    )
    download = not args.no_download
    train_aug = dataset_class(
        args.data_root, train=True, transform=train_transform, download=download
    )
    train_eval = dataset_class(
        args.data_root, train=True, transform=eval_transform, download=download
    )
    test = dataset_class(
        args.data_root, train=False, transform=eval_transform, download=download
    )
    return train_aug, train_eval, test


def resolve_manifest(
    args: argparse.Namespace,
    labels: np.ndarray,
) -> BagManifest:
    spec = DATASET_SPECS[args.dataset]
    if args.bag_file is not None:
        return load_bag_manifest(
            args.bag_file,
            labels=labels,
            dataset_size=len(labels),
            bag_size=args.bag_size,
            num_classes=spec.num_classes,
        )
    print(
        "warning: --bag-file was omitted; constructing deterministic random "
        "bags for this smoke run. Use the shared NPZ manifest for reported runs.",
        flush=True,
    )
    return create_random_manifest(
        labels=labels,
        dataset_size=len(labels),
        bag_size=args.bag_size,
        num_classes=spec.num_classes,
        seed=args.seed,
    )


def make_loaders(
    args: argparse.Namespace,
    train_aug,
    train_eval,
    test,
    manifest: BagManifest,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    bags_per_step = args.samples_per_step // args.bag_size
    loader_generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        BagDataset(train_aug, manifest),
        batch_size=bags_per_step,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **common,
    )
    if not len(train_loader):
        raise ValueError(
            f"Manifest has {manifest.num_bags} bags but one step requires "
            f"{bags_per_step}"
        )
    validation_loader = None
    if manifest.validation_indices.size:
        validation_loader = DataLoader(
            Subset(train_eval, manifest.validation_indices.tolist()),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **common,
        )
    test_loader = DataLoader(
        test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader


def make_generator_optimizer(
    args: argparse.Namespace,
    generator: nn.Module,
) -> torch.optim.Optimizer:
    if args.g_optimizer == "adam":
        return torch.optim.Adam(
            generator.parameters(),
            lr=args.g_lr,
            betas=(args.g_beta1, args.g_beta2),
            weight_decay=args.g_weight_decay,
        )
    return torch.optim.SGD(
        generator.parameters(),
        lr=args.g_lr,
        momentum=args.g_momentum,
        weight_decay=args.g_weight_decay,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        logits = model(images.to(device, non_blocking=True))
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Classifier evaluation expected a logits tensor")
        predictions = logits.argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    model.train(was_training)
    if total == 0:
        raise RuntimeError("Evaluation loader produced no examples")
    return correct / total


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def default_output_dir(args: argparse.Namespace) -> Path:
    return (
        Path("reproduction/artifacts/llpgan_unified")
        / args.dataset
        / args.bag_type
        / f"m{args.bag_size}"
        / f"seed{args.seed}"
    )


def prepare_output(
    args: argparse.Namespace,
    manifest: BagManifest,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    output_dir = args.output_dir or default_output_dir(args)
    protected = (
        output_dir / "config.json",
        output_dir / "metrics.jsonl",
        output_dir / "checkpoint_last.pt",
    )
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError(
            f"Run artifacts already exist in {output_dir}; choose a new "
            "--output-dir or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _json_ready(vars(args))
    config.update(
        {
            "output_dir": str(output_dir),
            "device": str(device),
            "num_classes": DATASET_SPECS[args.dataset].num_classes,
            "num_bags": manifest.num_bags,
            "manifest_source": manifest.source,
            "validation_examples": int(manifest.validation_indices.size),
            "amp_enabled": bool(not args.no_amp and device.type == "cuda"),
            "bag_supervision": (
                "proportion_ce"
                if args.moment_order == 1
                else "exact_multiclass_factorial_moment_ce"
            ),
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.save(output_dir / "used_bag_indices.npy", manifest.indices)
    metrics_path = output_dir / "metrics.jsonl"
    if args.overwrite:
        metrics_path.write_text("", encoding="utf-8")
    return output_dir, config


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    discriminator: nn.Module,
    generator: nn.Module,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler,
    scaler_d: torch.cuda.amp.GradScaler | None,
    scaler_g: torch.cuda.amp.GradScaler | None,
    config: dict[str, Any],
    validation_accuracy: float | None,
    test_accuracy: float,
) -> None:
    payload = {
        "epoch": epoch,
        "discriminator": discriminator.state_dict(),
        "generator": generator.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
        "scaler_g": scaler_g.state_dict() if scaler_g is not None else None,
        "config": config,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def promote_checkpoint(source: Path, destination: Path) -> None:
    """Atomically retain a checkpoint without serializing it a second time."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = DATASET_SPECS[args.dataset]
    train_aug, train_eval, test = make_datasets(args)
    labels = np.asarray(train_aug.targets, dtype=np.int64)
    manifest = resolve_manifest(args, labels)
    train_loader, validation_loader, test_loader = make_loaders(
        args, train_aug, train_eval, test, manifest, device
    )
    output_dir, config = prepare_output(args, manifest, device)

    discriminator = StemResNet18(spec.num_classes).to(device)
    generator = DCGANGenerator(args.z_dim).to(device)
    if args.moment_order == 1:
        moment_criterion = None
    elif args.mm_implementation == "paper_dp":
        try:
            from plench.core.order import LLPHighOrderLoss
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "--mm-implementation paper_dp requires the paper's plench "
                "runtime on PYTHONPATH"
            ) from exc
        moment_criterion = LLPHighOrderLoss(
            C=spec.num_classes,
            max_order=args.moment_order,
            bag_size=args.bag_size,
            loss_type="ce",
            weight_mode="uniform",
            order_weights=args.mm_order_weights,
            reduce="mean",
        ).to(device)
    else:
        moment_criterion = MulticlassFactorialMomentLoss(
            num_classes=spec.num_classes,
            max_order=args.moment_order,
            bag_size=args.bag_size,
            loss_type="ce",
            order_weights=args.mm_order_weights,
            power_chunk_size=args.mm_power_chunk_size,
        ).to(device)
    optimizer_d = torch.optim.SGD(
        discriminator.parameters(),
        lr=args.d_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=not args.no_nesterov,
    )
    optimizer_g = make_generator_optimizer(args, generator)

    steps_per_epoch = len(train_loader)
    if args.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler_d = make_warmup_cosine_scheduler(
        optimizer_d,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    amp_enabled = bool(not args.no_amp and device.type == "cuda")
    scaler_d = (
        torch.cuda.amp.GradScaler(enabled=True) if amp_enabled else None
    )
    scaler_g = (
        torch.cuda.amp.GradScaler(enabled=True) if amp_enabled else None
    )

    print(
        json.dumps(
            {
                "event": "run_start",
                "dataset": args.dataset,
                "device": str(device),
                "bags": manifest.num_bags,
                "bag_size": manifest.bag_size,
                "bags_per_step": args.samples_per_step // args.bag_size,
                "samples_per_step": args.samples_per_step,
                "steps_per_epoch": steps_per_epoch,
                "manifest": manifest.source,
                "moment_order": args.moment_order,
                "mm_implementation": args.mm_implementation,
                "mm_order_weights": (
                    args.mm_order_weights
                    if args.mm_order_weights is not None
                    else [1.0 / args.moment_order] * args.moment_order
                ),
                "lambda_mm": args.lambda_prop,
                "lambda_adv": args.lambda_adv,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best_validation = -float("inf")
    best_epoch = None
    best_test = None
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        totals = np.zeros(6, dtype=np.float64)
        completed_steps = 0
        for step_index, (bag_images, target_proportions) in enumerate(train_loader):
            if step_index >= steps_per_epoch:
                break
            if step_index == 0:
                print(
                    json.dumps(
                        {
                            "event": "batch_shapes",
                            "bag_images": list(bag_images.shape),
                            "bag_proportions": list(target_proportions.shape),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            iteration = train_iteration(
                discriminator=discriminator,
                generator=generator,
                optimizer_d=optimizer_d,
                optimizer_g=optimizer_g,
                bag_images=bag_images,
                target_proportions=target_proportions,
                device=device,
                bag_size=args.bag_size,
                num_classes=spec.num_classes,
                z_dim=args.z_dim,
                mean=spec.mean,
                std=spec.std,
                lambda_prop=args.lambda_prop,
                lambda_adv=args.lambda_adv,
                amp=amp_enabled,
                moment_criterion=moment_criterion,
                moment_bag_chunk_size=args.mm_bag_chunk_size or None,
                scaler_d=scaler_d,
                scaler_g=scaler_g,
                d_max_grad_norm=args.d_max_grad_norm or None,
                g_max_grad_norm=args.g_max_grad_norm or None,
            )
            scheduler_d.step()
            totals += np.asarray(
                [
                    iteration.discriminator_loss,
                    iteration.proportion_loss,
                    iteration.adversarial_loss,
                    iteration.generator_loss,
                    iteration.discriminator_grad_norm,
                    iteration.generator_grad_norm,
                ]
            )
            completed_steps += 1
            if (
                (step_index + 1) % args.log_interval == 0
                or step_index + 1 == steps_per_epoch
            ):
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step_index + 1,
                            "steps": steps_per_epoch,
                            **asdict(iteration),
                            "d_lr": scheduler_d.get_last_lr()[0],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if completed_steps != steps_per_epoch:
            raise RuntimeError(
                f"Expected {steps_per_epoch} steps, completed {completed_steps}"
            )

        validation_accuracy = (
            evaluate(
                discriminator,
                validation_loader,
                device,
                args.max_eval_batches,
            )
            if validation_loader is not None
            else None
        )
        test_accuracy = evaluate(
            discriminator,
            test_loader,
            device,
            args.max_eval_batches,
        )
        averages = totals / completed_steps
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            steps=completed_steps,
            discriminator_loss=float(averages[0]),
            proportion_loss=float(averages[1]),
            adversarial_loss=float(averages[2]),
            generator_loss=float(averages[3]),
            discriminator_grad_norm=float(averages[4]),
            generator_grad_norm=float(averages[5]),
            discriminator_lr=float(scheduler_d.get_last_lr()[0]),
            validation_accuracy=validation_accuracy,
            test_accuracy=test_accuracy,
            seconds=time.monotonic() - started,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(epoch_metrics), sort_keys=True) + "\n")

        save_arguments = {
            "epoch": epoch,
            "discriminator": discriminator,
            "generator": generator,
            "optimizer_d": optimizer_d,
            "optimizer_g": optimizer_g,
            "scheduler_d": scheduler_d,
            "scaler_d": scaler_d,
            "scaler_g": scaler_g,
            "config": config,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
        }
        last_checkpoint = output_dir / "checkpoint_last.pt"
        save_checkpoint(last_checkpoint, **save_arguments)
        if args.save_every and epoch % args.save_every == 0:
            promote_checkpoint(
                last_checkpoint,
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )

        if validation_accuracy is not None:
            is_best = validation_accuracy > best_validation
        else:
            # Without a validation split, "best" is defined as the final
            # checkpoint. Test accuracy is never used for model selection.
            is_best = epoch == args.epochs
        if is_best:
            best_validation = (
                validation_accuracy
                if validation_accuracy is not None
                else -float("inf")
            )
            best_epoch = epoch
            best_test = test_accuracy
            promote_checkpoint(last_checkpoint, output_dir / "checkpoint_best.pt")
        print(json.dumps(asdict(epoch_metrics), sort_keys=True), flush=True)

    result = {
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            None if best_validation == -float("inf") else best_validation
        ),
        "test_accuracy_at_best": best_test,
        "selection_metric": (
            "validation_accuracy" if validation_loader is not None else "final_epoch"
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "run_complete", **result}, sort_keys=True))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
