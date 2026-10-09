"""Variable-bag adapters around the existing exact multi-class LLP-MM loss."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .multiclass import (
    MulticlassFactorialMomentLoss,
    compositions,
    target_count_masses,
    variable_stable_scan_classifier_masses,
)

MM_ORDER_SCALE_MODES = (
    "none",
    "inverse_ordered_cell_rmse",
    "inverse_ordered_cell_mse",
)
MM_LOSS_TYPES = ("mse", "ce")

BAG_WEIGHT_MODES = ("uniform", "sqrt_size", "size")


def bag_weights_from_sizes(
    bag_sizes: torch.Tensor,
    mode: str = "uniform",
) -> torch.Tensor:
    """Return non-negative per-bag weights for an instance-level objective.

    ``uniform`` preserves the original LLP objective. ``sqrt_size`` is a
    conservative compromise for long-tailed bags, while ``size`` makes each
    instance contribute equally after aggregating its bag loss.
    """

    if bag_sizes.ndim != 1 or not len(bag_sizes):
        raise ValueError("bag_sizes must be a non-empty vector")
    if mode not in BAG_WEIGHT_MODES:
        raise ValueError(
            f"unknown bag weight mode {mode!r}; expected one of {BAG_WEIGHT_MODES}"
        )
    sizes = bag_sizes.to(torch.float64)
    if not torch.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("bag sizes must be finite and positive")
    if mode == "uniform":
        return torch.ones_like(sizes)
    if mode == "sqrt_size":
        return sizes.sqrt()
    return sizes


def _weighted_bag_mean(
    values: torch.Tensor,
    bag_weights: torch.Tensor | None,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("per-bag losses must be a vector")
    if bag_weights is None:
        return values.mean()
    weights = bag_weights.to(dtype=values.dtype, device=values.device)
    if weights.shape != values.shape:
        raise ValueError("bag weights must match the number of bags")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("bag weights must be finite and non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("bag weights must have positive total mass")
    return (values * weights).sum() / weights.sum()


def multiclass_order_multiplier(
    moment_order: int,
    num_classes: int,
    scale_mode: str = "none",
) -> float:
    """Absolute scale for an order-specific auxiliary moment loss."""
    if moment_order < 1:
        raise ValueError("moment_order must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if scale_mode not in MM_ORDER_SCALE_MODES:
        raise ValueError(
            f"unknown MM order scale mode {scale_mode!r}; "
            f"expected one of {MM_ORDER_SCALE_MODES}"
        )
    exponent_multiplier = {
        "none": 0,
        "inverse_ordered_cell_rmse": 1,
        "inverse_ordered_cell_mse": 2,
    }[scale_mode]
    return float(num_classes) ** (
        exponent_multiplier * (moment_order - 1)
    )


def calibrated_multiclass_order_weights(
    moment_order: int,
    num_classes: int,
    order_weights: Sequence[float] | None = None,
    scale_mode: str = "none",
) -> tuple[float, ...] | None:
    """Convert semantic per-order weights to effective LLP-MM weights.

    The exact order-r objective is a mean over ``C**r`` ordered categorical
    cells. Its raw MSE and gradient therefore become much smaller as ``r``
    grows. ``inverse_ordered_cell_rmse`` multiplies order ``r`` by
    ``C**(r - 1)``; the stronger ``inverse_ordered_cell_mse`` uses
    ``C**(2 * (r - 1))``. These modes change only the relative optimization
    scale; exact factorial-moment targets and predictions remain unchanged.
    """
    if moment_order < 1:
        raise ValueError("moment_order must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if scale_mode not in MM_ORDER_SCALE_MODES:
        raise ValueError(
            f"unknown MM order scale mode {scale_mode!r}; "
            f"expected one of {MM_ORDER_SCALE_MODES}"
        )
    if order_weights is None and scale_mode == "none":
        return None
    base = (
        torch.ones(moment_order, dtype=torch.float64)
        if order_weights is None
        else torch.tensor(tuple(order_weights), dtype=torch.float64)
    )
    if len(base) != moment_order:
        raise ValueError(f"order_weights must have length {moment_order}")
    if (
        not torch.isfinite(base).all()
        or (base < 0).any()
        or float(base.sum()) <= 0.0
    ):
        raise ValueError("order weights must be finite, non-negative, and nonzero")
    if scale_mode in {
        "inverse_ordered_cell_rmse",
        "inverse_ordered_cell_mse",
    }:
        scale = torch.tensor(
            [
                multiclass_order_multiplier(order, num_classes, scale_mode)
                for order in range(1, moment_order + 1)
            ],
            dtype=torch.float64,
        )
        base = base * scale
    effective = base / base.sum()
    return tuple(float(value) for value in effective.tolist())


def scheduled_multiclass_order_weights(
    moment_order: int,
    target_weights: Sequence[float] | None,
    warmup_epochs: int,
    ramp_epochs: int,
    epoch: int,
    *,
    total_epochs: int | None = None,
    recovery_epochs: int = 0,
) -> tuple[float, ...] | None:
    """Warm up, introduce high orders, and optionally recover on order one."""
    if min(warmup_epochs, ramp_epochs, recovery_epochs, epoch) < 0:
        raise ValueError("MM schedule values must be non-negative")
    if recovery_epochs:
        if total_epochs is None or total_epochs <= 0:
            raise ValueError("MM recovery requires a positive total_epochs")
        if warmup_epochs + ramp_epochs + recovery_epochs > total_epochs:
            raise ValueError("MM warmup, ramp, and recovery stages overlap")
    if warmup_epochs == 0 and ramp_epochs == 0 and recovery_epochs == 0:
        return None if target_weights is None else tuple(target_weights)
    if moment_order < 1:
        raise ValueError("moment_order must be positive")
    target = (
        [1.0 / moment_order] * moment_order
        if target_weights is None
        else [float(value) for value in target_weights]
    )
    if len(target) != moment_order:
        raise ValueError(f"target_weights must have length {moment_order}")
    start = [1.0] + [0.0] * (moment_order - 1)
    if recovery_epochs and epoch >= int(total_epochs) - recovery_epochs:
        recovery_progress = (
            epoch - (int(total_epochs) - recovery_epochs) + 1
        ) / recovery_epochs
        return tuple(
            (1.0 - recovery_progress) * target_value
            + recovery_progress * start_value
            for start_value, target_value in zip(start, target)
        )
    if epoch < warmup_epochs:
        progress = 0.0
    elif ramp_epochs == 0:
        progress = 1.0
    else:
        progress = min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
    return tuple(
        (1.0 - progress) * start_value + progress * target_value
        for start_value, target_value in zip(start, target)
    )








def variable_multiclass_llp_mm_loss(
    probabilities: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    class_counts: torch.Tensor,
    class_proportions: torch.Tensor,
    moment_order: int,
    order_weights: Sequence[float] | None = None,
    bag_weights: torch.Tensor | None = None,
    moment_algorithm: str = "newton",
    loss_type: str = "mse",
    ce_smoothing_tau: float = 0.0,
    class_prior: torch.Tensor | None = None,
    ce_class_balance_power: float = 0.0,
) -> torch.Tensor:
    """Average the existing exact fixed-size objective over variable-size bags.

    The mathematical implementation remains
    :class:`MulticlassFactorialMomentLoss`; this function only slices the
    flattened batch into its complete bags and instantiates the original loss
    with each observed bag size. ``mse`` preserves the original C**r
    normalization and makes order one the bag-proportion MSE. ``ce`` applies
    cross-entropy directly to the same exact joint count-profile masses,
    providing a stronger proper-distribution gradient at high orders.
    """
    if not len(bag_sizes):
        raise ValueError("LLP-MM received an empty bag batch")
    if int(bag_sizes.min().item()) < moment_order:
        raise ValueError("bag_size must be at least moment_order")
    if class_counts.shape != class_proportions.shape:
        raise ValueError("class count and proportion shapes differ")
    if loss_type not in MM_LOSS_TYPES:
        raise ValueError(
            f"unknown MM loss type {loss_type!r}; expected one of "
            f"{MM_LOSS_TYPES}"
        )
    if (
        not 0.0 <= float(ce_smoothing_tau) < 1.0
        or not torch.isfinite(torch.tensor(float(ce_smoothing_tau)))
    ):
        raise ValueError("CE smoothing tau must be finite and in [0, 1)")
    if ce_smoothing_tau and loss_type != "ce":
        raise ValueError("CE smoothing tau requires CE loss")
    if (
        not 0.0 <= float(ce_class_balance_power) <= 2.0
        or not torch.isfinite(torch.tensor(float(ce_class_balance_power)))
    ):
        raise ValueError("CE class-balance power must be finite and in [0, 2]")
    if ce_class_balance_power and loss_type != "ce":
        raise ValueError("class-balanced moment loss requires CE")
    if ce_class_balance_power and moment_algorithm != "stable_scan":
        raise ValueError(
            "class-balanced variable moment CE requires stable_scan"
        )
    observed = torch.bincount(bag_index, minlength=len(bag_sizes))
    if not torch.equal(
        observed.to(device=bag_sizes.device, dtype=bag_sizes.dtype), bag_sizes
    ):
        raise ValueError("bag_index counts do not match bag_sizes")
    if moment_algorithm == "stable_scan":
        number_of_classes = int(probabilities.shape[1])
        predicted = variable_stable_scan_classifier_masses(
            probabilities,
            bag_index,
            bag_sizes,
            moment_order,
        )
        targets = target_count_masses(class_counts, moment_order)
        if order_weights is None:
            weights = torch.full(
                (moment_order,),
                1.0 / float(moment_order),
                dtype=torch.float64,
                device=probabilities.device,
            )
        else:
            if len(order_weights) != moment_order:
                raise ValueError(
                    f"order_weights must have length {moment_order}"
                )
            weights = torch.as_tensor(
                order_weights,
                dtype=torch.float64,
                device=probabilities.device,
            )
            if (
                not torch.isfinite(weights).all()
                or (weights < 0).any()
                or float(weights.sum()) <= 0.0
            ):
                raise ValueError(
                    "order weights must be finite, non-negative, and nonzero"
                )
            weights = weights / weights.sum()

        per_bag_loss = torch.zeros(
            len(bag_sizes),
            dtype=torch.float64,
            device=probabilities.device,
        )
        for order, (target, prediction) in enumerate(
            zip(targets, predicted), start=1
        ):
            weight = weights[order - 1]
            if float(weight) == 0.0:
                continue
            if loss_type == "ce":
                if ce_smoothing_tau:
                    k = compositions(
                        number_of_classes,
                        order,
                        device=prediction.device,
                    )
                    factorials = torch.tensor(
                        [math.factorial(value) for value in range(order + 1)],
                        dtype=torch.float64,
                        device=prediction.device,
                    )
                    permutation_count = (
                        math.factorial(order) / factorials[k].prod(dim=1)
                    )
                    background = permutation_count / float(
                        number_of_classes**order
                    )
                    prediction = (
                        (1.0 - float(ce_smoothing_tau)) * prediction
                        + float(ce_smoothing_tau) * background.unsqueeze(0)
                    )
                cell_weights: torch.Tensor | float = 1.0
                if ce_class_balance_power:
                    if class_prior is None:
                        raise ValueError(
                            "class-balanced moment CE requires class_prior"
                        )
                    prior = class_prior.to(
                        dtype=torch.float64,
                        device=prediction.device,
                    )
                    if (
                        prior.shape != (number_of_classes,)
                        or not torch.isfinite(prior).all()
                        or (prior <= 0).any()
                    ):
                        raise ValueError(
                            "class_prior must be a positive finite class vector"
                        )
                    prior = prior / prior.sum()
                    k = compositions(
                        number_of_classes,
                        order,
                        device=prediction.device,
                    ).to(torch.float64)
                    log_weights = (
                        -float(ce_class_balance_power)
                        * (k * prior.log().unsqueeze(0)).sum(dim=1)
                        / float(order)
                    )
                    log_weights = log_weights - log_weights.max()
                    cell_weights = log_weights.exp().unsqueeze(0)
                weighted_target = target * cell_weights
                order_loss = -(
                    weighted_target * prediction.clamp_min(1e-15).log()
                ).sum(dim=1) / weighted_target.sum(
                    dim=1
                ).clamp_min(1e-15)
            else:
                k = compositions(
                    number_of_classes,
                    order,
                    device=prediction.device,
                )
                permutation_count = (
                    math.factorial(order)
                    / torch.tensor(
                        [
                            math.prod(
                                math.factorial(int(value)) for value in row
                            )
                            for row in k.cpu().tolist()
                        ],
                        dtype=torch.float64,
                        device=prediction.device,
                    )
                )
                order_loss = (
                    (target - prediction).square()
                    / permutation_count.unsqueeze(0)
                ).sum(dim=1) / float(number_of_classes**order)
            per_bag_loss = per_bag_loss + weight * order_loss
        return _weighted_bag_mean(
            per_bag_loss, bag_weights
        ).to(probabilities.dtype)

    losses: list[torch.Tensor] = []
    number_of_classes = int(probabilities.shape[1])
    for bag_number, size_tensor in enumerate(bag_sizes):
        size = int(size_tensor.item())
        criterion = MulticlassFactorialMomentLoss(
            number_of_classes,
            moment_order,
            size,
            loss_type=loss_type,
            order_weights=order_weights,
            moment_algorithm=moment_algorithm,
            ce_smoothing_tau=ce_smoothing_tau,
        ).to(probabilities.device)
        bag_probabilities = probabilities[bag_index == bag_number].unsqueeze(0)
        losses.append(
            criterion(
                class_proportions[bag_number : bag_number + 1],
                bag_probabilities,
                label_counts=class_counts[bag_number : bag_number + 1],
            )
        )
    return _weighted_bag_mean(torch.stack(losses), bag_weights)
