from __future__ import annotations
import math
from typing import Any, Mapping, Optional, Sequence
import torch
import torch.nn.functional as F

SUPPORTED_METHODS = {
    "GeneralUPM",
    "ROT",
    "LLP_MM",
    "LLP_PVC",
    "EasyLLP",
    "PM",
    "LLP_FC",
    "LLP_DSQ",
}


def _weighted_bag_means(
    values: torch.Tensor,
    bag_index: torch.Tensor,
    weights: torch.Tensor,
    number_of_bags: int,
) -> torch.Tensor:
    result = torch.zeros(
        (number_of_bags, values.shape[1]), dtype=values.dtype, device=values.device
    )
    weighted = values * weights[:, None]
    result.scatter_add_(0, bag_index[:, None].expand_as(weighted), weighted)
    denominator = torch.zeros(number_of_bags, dtype=values.dtype, device=values.device)
    denominator.scatter_add_(0, bag_index, weights)
    return result / denominator[:, None].clamp_min(1e-12)


def _bag_slices(bag_index: torch.Tensor, number_of_bags: int) -> list[torch.Tensor]:
    return [
        torch.nonzero(bag_index == bag, as_tuple=False).squeeze(1)
        for bag in range(number_of_bags)
    ]


def _largest_remainder_counts(proportion: torch.Tensor, total: int) -> torch.Tensor:
    raw = proportion * int(total)
    counts = raw.floor().to(torch.long)
    remainder = int(total) - int(counts.sum().item())
    if remainder:
        order = torch.argsort(raw - counts.to(raw), descending=True, stable=True)
        counts = counts.clone()
        counts[order[:remainder]] += 1
    return counts


def _llp_mm_variable_loss(
    probabilities: torch.Tensor,
    proportions: torch.Tensor,
    slices: Sequence[torch.Tensor],
    order: int,
    *,
    order_weights: Optional[Sequence[float]] = None,
    loss_type: str = "ce",
    moment_algorithm: str = "stable_dp",
    compute_dtype: str = "float64",
    ce_smoothing_tau: float = 0.0001,
) -> torch.Tensor:
    """Exact multi-order count-profile CE in cancellation-free float64 DP.

    The paper uses cross-entropy for every method and assigns each order the
    weight ``1 / order``.  Ordered-tuple sampling and MSE are intentionally not
    supported by this training path: both silently weaken high-order signals.
    Bag class counts are reconstructed solely from the observed proportions.
    """
    if loss_type != "ce":
        raise ValueError("LLP-MM requires cross-entropy loss")
    if moment_algorithm != "stable_dp":
        raise ValueError("LLP-MM requires exact stable_dp moment computation")
    if compute_dtype != "float64":
        raise ValueError("LLP-MM requires float64 moment computation")
    if not math.isclose(float(ce_smoothing_tau), 0.0001, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("LLP-MM requires the paper CE smoothing tau=1e-4")
    if probabilities.dtype != torch.float64:
        raise ValueError("LLP-MM probabilities must be computed in float64")
    if not slices:
        raise ValueError("LLP-MM received no bags")
    if any((len(indices) < int(order) for indices in slices)):
        configured_weights = (
            [1.0 / float(order)] * int(order)
            if order_weights is None
            else list(order_weights)
        )
        per_bag = []
        for bag_number, indices in enumerate(slices):
            effective_order = min(int(order), len(indices))
            active_weights = configured_weights[:effective_order]
            if effective_order < int(order):
                active_total = sum(active_weights)
                if active_total <= 0:
                    raise ValueError(
                        "No positive weight for the orders feasible in a short bag"
                    )
                active_weights = [
                    value * sum(configured_weights) / active_total
                    for value in active_weights
                ]
            per_bag.append(
                _llp_mm_variable_loss(
                    probabilities[indices],
                    proportions[bag_number : bag_number + 1],
                    [torch.arange(len(indices), device=probabilities.device)],
                    effective_order,
                    order_weights=active_weights,
                    loss_type=loss_type,
                    moment_algorithm=moment_algorithm,
                    compute_dtype=compute_dtype,
                    ce_smoothing_tau=ce_smoothing_tau,
                )
            )
        return torch.stack(per_bag).mean()
    try:
        from mo_matching.llp.structured_multiclass import (
            variable_multiclass_llp_mm_loss,
        )
    except ModuleNotFoundError:
        from src.mo_matching.llp.structured_multiclass import (
            variable_multiclass_llp_mm_loss,
        )
    bag_sizes = torch.tensor(
        [len(indices) for indices in slices],
        dtype=torch.long,
        device=probabilities.device,
    )
    bag_index = torch.empty(
        len(probabilities), dtype=torch.long, device=probabilities.device
    )
    counts: list[torch.Tensor] = []
    for bag_number, indices in enumerate(slices):
        bag_index[indices] = bag_number
        counts.append(_largest_remainder_counts(proportions[bag_number], len(indices)))
    class_counts = torch.stack(counts).to(device=probabilities.device, dtype=torch.long)
    strict_proportions = proportions.to(
        device=probabilities.device, dtype=torch.float64
    )
    weights = (
        [1.0 / float(order)] * int(order)
        if order_weights is None
        else list(order_weights)
    )
    return variable_multiclass_llp_mm_loss(
        probabilities,
        bag_index,
        bag_sizes,
        class_counts,
        strict_proportions,
        int(order),
        order_weights=weights,
        moment_algorithm=moment_algorithm,
        loss_type=loss_type,
        ce_smoothing_tau=float(ce_smoothing_tau),
    )


def natural_bag_loss(
    algorithm: Any,
    algorithm_name: str,
    batch: Mapping[str, torch.Tensor],
    *,
    logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Variable-natural-bag loss adapter around existing paper algorithms."""
    if algorithm_name not in SUPPORTED_METHODS:
        raise NotImplementedError(
            f"{algorithm_name} has no verified variable-natural-bag natural-bag adapter. Verified methods: {sorted(SUPPORTED_METHODS)}"
        )
    if logits is None:
        logits = algorithm.predict(batch["x"])
    proportions = batch["proportion"].to(logits)
    bag_index = batch["bag_index"]
    sizes = batch["bag_sizes"].to(logits)
    weights = batch["instance_weights"].to(logits)
    number_of_bags = len(proportions)
    if algorithm_name == "LLP_MM":
        strict_probabilities = (
            logits.to(torch.float64)
            .softmax(dim=1)
            .clamp_min(torch.finfo(torch.float64).tiny)
        )
        return _llp_mm_variable_loss(
            strict_probabilities,
            proportions,
            _bag_slices(bag_index, number_of_bags),
            int(getattr(algorithm, "order", 3)),
            order_weights=getattr(algorithm, "order_weights", None),
            loss_type=getattr(algorithm, "moment_loss_type", "ce"),
            moment_algorithm=getattr(algorithm, "moment_algorithm", "stable_dp"),
            compute_dtype=getattr(algorithm, "moment_compute_dtype", "float64"),
            ce_smoothing_tau=getattr(algorithm, "moment_ce_smoothing_tau", 0.0001),
        )
    probabilities = logits.softmax(dim=1).clamp_min(1e-12)
    means = _weighted_bag_means(
        probabilities, bag_index, weights, number_of_bags
    ).clamp_min(1e-12)
    if algorithm_name == "PM":
        return -(proportions * means.log()).sum(dim=1).mean()
    if algorithm_name == "LLP_DSQ":
        return algorithm.DSQ_Loss(
            logits,
            proportions,
            bag_sizes=sizes,
            bag_index=bag_index,
            update_ema=bool(algorithm.training),
        )
    slices = _bag_slices(bag_index, number_of_bags)
    if algorithm_name == "LLP_PVC":
        return algorithm.PVC_Loss(
            logits, proportions, bag_sizes=sizes, bag_index=bag_index
        )
    if algorithm_name == "LLP_FC":
        return algorithm.FC_Loss(
            logits, proportions, bag_sizes=sizes, bag_index=bag_index
        )
    if algorithm_name == "ROT":
        losses = []
        original = algorithm.bagsize
        try:
            for bag_number, indices in enumerate(slices):
                algorithm.bagsize = len(indices)
                loss, _ = algorithm.ROTLoss_Loss(
                    logits[indices], proportions[bag_number : bag_number + 1]
                )
                losses.append(loss)
        finally:
            algorithm.bagsize = original
        return torch.stack(losses).mean()
    prior = (proportions * sizes[:, None]).sum(dim=0) / sizes.sum()
    if algorithm_name == "EasyLLP":
        loss, _ = algorithm.EasyLLP_Loss(
            logits, proportions, bag_sizes=sizes, bag_index=bag_index
        )
        if getattr(algorithm, "flooding", False):
            b = float(getattr(algorithm, "flooding_b", 0.0))
            loss = (loss - b).abs() + b
        return loss
    if algorithm_name == "GeneralUPM":
        if number_of_bags < 2:
            raise ValueError(
                "Natural-bag GeneralUPM requires batchsize >= 2 natural bags"
            )
        ell = -F.log_softmax(logits, dim=1)
        bag_sums = torch.zeros_like(proportions)
        bag_sums.scatter_add_(0, bag_index[:, None].expand_as(ell), ell)
        total_sum = bag_sums.sum(dim=0)
        outside_sizes = sizes.sum() - sizes
        outside_means = (total_sum[None, :] - bag_sums) / outside_sizes[:, None]
        centered = (proportions - prior[None, :]) * (
            bag_sums - sizes[:, None] * outside_means
        )
        base = (prior[None, :] * outside_means).sum(dim=1)
        loss = (centered.sum(dim=1) + base).mean()
        if getattr(algorithm, "flooding", False):
            b = float(getattr(algorithm, "flooding_b", 0.0))
            loss = (loss - b).abs() + b
        return loss
    raise AssertionError("unreachable Natural-bag algorithm branch")
