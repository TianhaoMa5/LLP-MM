"""Binary variable-bag adaptations of the existing MO-Matching LLP objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

MAX_MOMENT_ORDER = 14


def bag_means(
    probabilities: torch.Tensor, bag_index: torch.Tensor, number_of_bags: int
) -> torch.Tensor:
    probabilities = probabilities.reshape(-1)
    totals = torch.zeros(
        number_of_bags, dtype=probabilities.dtype, device=probabilities.device
    )
    counts = torch.zeros_like(totals)
    totals.scatter_add_(0, bag_index, probabilities)
    counts.scatter_add_(0, bag_index, torch.ones_like(probabilities))
    return totals / counts.clamp_min(1)






class _BagMeanCalibratedSigmoid(torch.autograd.Function):
    """Implicitly differentiated sigmoid with one fixed mean per bag."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        bag_index: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        number_of_bags = int(targets.numel())
        # The intercept solve has a unique monotone root. Bisection is slower
        # than Newton but cannot diverge for saturated model logits.
        lower = torch.full_like(targets, -40.0)
        upper = torch.full_like(targets, 40.0)
        for _ in range(48):
            middle = (lower + upper) * 0.5
            adjusted = torch.sigmoid(logits + middle[bag_index])
            means = bag_means(adjusted, bag_index, number_of_bags)
            too_high = means > targets
            upper = torch.where(too_high, middle, upper)
            lower = torch.where(too_high, lower, middle)
        offsets = (lower + upper) * 0.5
        adjusted = torch.sigmoid(logits + offsets[bag_index])
        ctx.save_for_backward(adjusted, bag_index)
        ctx.number_of_bags = number_of_bags
        return adjusted

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        adjusted, bag_index = ctx.saved_tensors
        slope = adjusted * (1.0 - adjusted)
        slope_sums = torch.zeros(
            ctx.number_of_bags,
            dtype=slope.dtype,
            device=slope.device,
        )
        weighted_gradient_sums = torch.zeros_like(slope_sums)
        slope_sums.scatter_add_(0, bag_index, slope)
        weighted_gradient_sums.scatter_add_(0, bag_index, gradient * slope)
        correction = weighted_gradient_sums / slope_sums.clamp_min(1e-12)
        # This is the implicit Jacobian of the equality-constrained intercept:
        # the gradient along a constant logit shift in any bag is exactly zero.
        gradient_logits = slope * (gradient - correction[bag_index])
        return gradient_logits, None, None


def calibrate_logits_to_bag_proportions(
    logits: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    label_proportions: torch.Tensor,
) -> torch.Tensor:
    """Apply a differentiable per-bag logit shift to match each known mean.

    This is an intercept-only calibration used by conditional higher-order
    matching. It preserves the within-bag ordering of instance predictions
    while removing the first-moment mismatch from the higher-order branch.
    Consequently, that branch learns count-distribution shape rather than
    repeatedly optimizing the same bag mean at every order.
    """
    logits = logits.reshape(-1)
    label_proportions = label_proportions.reshape(-1).to(
        device=logits.device, dtype=logits.dtype
    )
    if len(label_proportions) != len(bag_sizes):
        raise ValueError("one label proportion is required for every bag")
    if ((label_proportions < 0) | (label_proportions > 1)).any():
        raise ValueError("label proportions must lie in [0, 1]")
    observed_sizes = torch.bincount(
        bag_index, minlength=len(bag_sizes)
    ).to(device=bag_sizes.device, dtype=bag_sizes.dtype)
    if not torch.equal(observed_sizes, bag_sizes):
        raise ValueError(
            "bag_index instance counts do not match the reported bag_sizes"
        )

    targets = label_proportions.clamp(1e-6, 1.0 - 1e-6)
    return _BagMeanCalibratedSigmoid.apply(logits, bag_index, targets)


def calibrate_probabilities_to_bag_proportions(
    probabilities: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    label_proportions: torch.Tensor,
) -> torch.Tensor:
    """Probability-input convenience wrapper around implicit calibration."""
    epsilon = torch.finfo(probabilities.dtype).eps
    logits = torch.logit(
        probabilities.reshape(-1).clamp(epsilon, 1.0 - epsilon)
    )
    return calibrate_logits_to_bag_proportions(
        logits, bag_index, bag_sizes, label_proportions
    )


def _predicted_count_distributions(
    probabilities: torch.Tensor, max_order: int
) -> list[torch.Tensor]:
    """Return exact count distributions for orders 1..``max_order``.

    For a subset size ``r``, the required probabilities are the coefficients of
    the polynomial

        e_r((1 - p_1) + p_1 z, ..., (1 - p_m) + p_m z) / C(m, r),

    where ``e_r`` is the elementary symmetric polynomial.  We evaluate these
    polynomials on roots of unity, apply normalized Newton identities, and
    recover every coefficient with one inverse FFT per order.  Normalizing the
    identities by ``C(m, r)`` keeps intermediate values bounded, while the
    roots-of-unity interpolation avoids an ill-conditioned Vandermonde solve.
    All work is float64/complex128 and remains differentiable.
    """
    bag_index = torch.zeros(
        probabilities.numel(), dtype=torch.long, device=probabilities.device
    )
    bag_sizes = torch.tensor(
        [probabilities.numel()], dtype=torch.long, device=probabilities.device
    )
    return [
        distribution[0]
        for distribution in _predicted_count_distributions_batched(
            probabilities, bag_index, bag_sizes, max_order
        )
    ]


def _predicted_count_distributions_batched(
    probabilities: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    max_order: int,
) -> list[torch.Tensor]:
    """Vectorized counterpart of :func:`_predicted_count_distributions`."""
    m_min = int(bag_sizes.min().item())
    if not 1 <= max_order <= MAX_MOMENT_ORDER:
        raise ValueError(
            f"moment_order must be in [1, {MAX_MOMENT_ORDER}]; got {max_order}"
        )
    if m_min < max_order:
        raise ValueError(
            f"bag_size={m_min} is smaller than moment order={max_order}"
        )

    number_of_bags = int(bag_sizes.numel())
    observed_sizes = torch.bincount(bag_index, minlength=number_of_bags)
    if not torch.equal(
        observed_sizes.to(device=bag_sizes.device, dtype=bag_sizes.dtype), bag_sizes
    ):
        raise ValueError(
            "bag_index instance counts do not match the reported bag_sizes"
        )
    sequences = [
        probabilities[bag_index == bag_number]
        for bag_number in range(number_of_bags)
    ]
    p = pad_sequence(sequences, batch_first=True).to(dtype=torch.float64)
    valid = (
        torch.arange(p.shape[1], device=p.device).unsqueeze(0)
        < bag_sizes.unsqueeze(1)
    )
    p = p.clamp(1e-12, 1.0 - 1e-12)
    q = 1.0 - p
    number_of_nodes = max_order + 1
    angles = (
        -2.0
        * math.pi
        * torch.arange(number_of_nodes, dtype=torch.float64, device=p.device)
        / number_of_nodes
    )
    roots = torch.polar(torch.ones_like(angles), angles).to(torch.complex128)
    evaluated_variables = (
        q.to(torch.complex128).unsqueeze(1)
        + roots.unsqueeze(0).unsqueeze(2) * p.to(torch.complex128).unsqueeze(1)
    ) * valid.unsqueeze(1)
    power_sums: list[torch.Tensor | None] = [None]
    power_sums.extend(
        evaluated_variables.pow(order).sum(dim=2)
        for order in range(1, max_order + 1)
    )

    normalized_elementary = [
        torch.ones(
            (number_of_bags, number_of_nodes),
            dtype=torch.complex128,
            device=probabilities.device,
        )
    ]
    bag_sizes_float = bag_sizes.to(dtype=torch.float64)
    for order in range(1, max_order + 1):
        evaluated = torch.zeros_like(normalized_elementary[0])
        for power in range(1, order + 1):
            coefficient = torch.ones_like(bag_sizes_float)
            for offset in range(power):
                coefficient = coefficient * (
                    float(order - offset)
                    / (bag_sizes_float - float(order) + 1.0 + float(offset))
                )
            coefficient = coefficient / float(order)
            sign = 1.0 if power % 2 else -1.0
            evaluated = (
                evaluated
                + sign
                * coefficient.unsqueeze(1)
                * normalized_elementary[order - power]
                * power_sums[power]
            )
        normalized_elementary.append(evaluated)

    distributions = []
    for order in range(1, max_order + 1):
        coefficients = torch.fft.ifft(
            normalized_elementary[order], dim=1
        ).real[:, : order + 1]
        coefficients = coefficients.clamp_min(1e-12)
        distributions.append(
            coefficients / coefficients.sum(dim=1, keepdim=True)
        )
    return distributions


def _predicted_count_distribution(
    probabilities: torch.Tensor, order: int
) -> torch.Tensor:
    """Probability of 0..order positives in a uniform subset without replacement."""
    return _predicted_count_distributions(probabilities, order)[-1]


def _true_count_distribution(
    bag_size: int, label_count: int, order: int, device: torch.device
) -> torch.Tensor:
    if bag_size < order:
        raise ValueError(f"bag_size={bag_size} is smaller than moment order={order}")
    if label_count < 0 or label_count > bag_size:
        raise ValueError(f"label_count={label_count} is outside [0, {bag_size}]")
    denominator = math.comb(bag_size, order)
    values = []
    for positives in range(order + 1):
        if positives > label_count or order - positives > bag_size - label_count:
            values.append(0.0)
        else:
            values.append(
                math.comb(label_count, positives)
                * math.comb(bag_size - label_count, order - positives)
                / denominator
            )
    return torch.tensor(values, dtype=torch.float64, device=device)


def llp_mm_loss(
    probabilities: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    label_counts: torch.Tensor,
    moment_order: int,
    order_weights: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Exact binary LLP-MM objective, averaging CE for orders 1..``moment_order``.

    This is the variable-bag binary specialization of the repository's existing
    ``LLPHighOrderLoss(loss_type="ce", weight_mode="uniform")``.  Its targets
    are exact hypergeometric factorial moments and its predicted moments are the
    corresponding without-replacement subset probabilities.  No instance label
    is accepted by this interface.
    """
    if not 1 <= moment_order <= MAX_MOMENT_ORDER:
        raise ValueError(
            f"moment_order must be in [1, {MAX_MOMENT_ORDER}]; got {moment_order}"
        )
    if order_weights is None:
        normalized_weights = torch.full(
            (moment_order,), 1.0 / moment_order, dtype=torch.float64
        )
    else:
        if len(order_weights) != moment_order:
            raise ValueError(
                f"order_weights must have length {moment_order}; got {len(order_weights)}"
            )
        normalized_weights = torch.tensor(order_weights, dtype=torch.float64)
        if not torch.isfinite(normalized_weights).all():
            raise ValueError("order_weights must be finite")
        if (normalized_weights < 0).any() or normalized_weights.sum() <= 0:
            raise ValueError("order_weights must be non-negative with a positive sum")
        normalized_weights = normalized_weights / normalized_weights.sum()
    probabilities = probabilities.reshape(-1)
    if not len(bag_sizes):
        raise ValueError("LLP-MM received an empty bag batch")
    minimum_bag_size = int(bag_sizes.min().item())
    if minimum_bag_size < moment_order:
        raise ValueError(
            f"bag_size={minimum_bag_size} is smaller than "
            f"moment order={moment_order}"
        )
    # Scheduled MM training starts from the BCE-equivalent weight vector
    # [1, 0, ..., 0]. Avoid constructing higher-order count distributions until
    # their weight becomes non-zero. This preserves the configured order and its
    # bag-size validation while making the BCE pretraining stage as cheap as
    # actual DLLP-BCE.
    active_max_order = int(
        torch.nonzero(normalized_weights > 0, as_tuple=False)[-1].item()
    ) + 1
    predicted_distributions = _predicted_count_distributions_batched(
        probabilities, bag_index, bag_sizes, active_max_order
    )
    order_losses = []
    for order, predicted in enumerate(predicted_distributions, start=1):
        targets = torch.stack(
            [
                _true_count_distribution(
                    int(size.item()),
                    int(count.item()),
                    order,
                    predicted.device,
                )
                for size, count in zip(bag_sizes, label_counts)
            ]
        )
        order_losses.append(-(targets * predicted.log()).sum(dim=1))
    weights = normalized_weights[:active_max_order].to(device=probabilities.device)
    return (
        (torch.stack(order_losses, dim=1) * weights.unsqueeze(0))
        .sum(dim=1)
        .mean()
        .to(dtype=probabilities.dtype)
    )


def conditional_llp_mm_loss(
    probabilities: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    label_counts: torch.Tensor,
    moment_order: int,
    order_weights: list[float] | tuple[float, ...] | None = None,
    *,
    logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match the first moment and conditional higher-order shape jointly.

    The first-order term uses the model probabilities directly. For orders
    two and above, a differentiable bagwise logit shift first makes the
    predicted bag mean equal the design proportion. Higher-order count
    distributions therefore cannot improve merely by repeating the
    first-order direction. The interface accepts only bag counts, never
    instance labels, and all terms are optimized in one fixed-stage loss.
    """
    if not 1 <= moment_order <= MAX_MOMENT_ORDER:
        raise ValueError(
            f"moment_order must be in [1, {MAX_MOMENT_ORDER}]; got {moment_order}"
        )
    if order_weights is None:
        weights = torch.full(
            (moment_order,), 1.0 / moment_order, dtype=torch.float64
        )
    else:
        if len(order_weights) != moment_order:
            raise ValueError(
                f"order_weights must have length {moment_order}; "
                f"got {len(order_weights)}"
            )
        weights = torch.tensor(order_weights, dtype=torch.float64)
        if not torch.isfinite(weights).all():
            raise ValueError("order_weights must be finite")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "order_weights must be non-negative with a positive sum"
            )
        weights = weights / weights.sum()

    proportions = label_counts.to(dtype=probabilities.dtype) / bag_sizes.to(
        dtype=probabilities.dtype
    )
    means = bag_means(probabilities, bag_index, len(proportions))
    first_order = F.binary_cross_entropy(
        means.clamp(1e-12, 1.0 - 1e-12), proportions.to(dtype=means.dtype)
    )
    if moment_order == 1 or float(weights[1:].sum()) == 0.0:
        return first_order

    calibrated = (
        calibrate_logits_to_bag_proportions(
            logits, bag_index, bag_sizes, proportions
        )
        if logits is not None
        else calibrate_probabilities_to_bag_proportions(
            probabilities, bag_index, bag_sizes, proportions
        )
    )
    high_total = weights[1:].sum()
    high_weights = torch.cat((torch.zeros(1, dtype=weights.dtype), weights[1:]))
    high_order = llp_mm_loss(
        calibrated,
        bag_index,
        bag_sizes,
        label_counts,
        moment_order,
        high_weights.tolist(),
    )
    return (
        weights[0].to(device=probabilities.device, dtype=probabilities.dtype)
        * first_order
        + high_total.to(device=probabilities.device, dtype=probabilities.dtype)
        * high_order
    )
