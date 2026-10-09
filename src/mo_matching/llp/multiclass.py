"""Exact compressed multi-class factorial-moment matching.

This is the categorical (not one-vs-rest) objective defined in the paper.  A
compressed state is a class multiplicity vector ``k`` with ``sum(k) == r``.
The number of states is ``comb(C + r - 1, r)`` instead of ``C**r`` and no
instance-pair matrix is materialized.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn as nn

MAX_MULTICLASS_MOMENT_ORDER = 14
MULTICLASS_MOMENT_ALGORITHMS = ("newton", "stable_dp", "stable_scan")


@lru_cache(maxsize=None)
def _composition_tuples(classes: int, order: int) -> tuple[tuple[int, ...], ...]:
    if classes <= 0 or order < 0:
        raise ValueError("classes must be positive and order non-negative")
    output: list[tuple[int, ...]] = []

    def visit(class_index: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if class_index == classes - 1:
            output.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            visit(class_index + 1, remaining - value, (*prefix, value))

    visit(0, order, ())
    return tuple(output)


def compositions(
    classes: int, order: int, *, device: torch.device | None = None
) -> torch.Tensor:
    values = _composition_tensor_cpu(classes, order)
    if device is None or torch.device(device).type == "cpu":
        return values
    return values.to(device=device)


@lru_cache(maxsize=None)
def _composition_tensor_cpu(classes: int, order: int) -> torch.Tensor:
    """Cache immutable CPU index tensors used repeatedly by high orders."""
    return torch.tensor(_composition_tuples(classes, order), dtype=torch.long)


@lru_cache(maxsize=None)
def _polynomial_product_map(
    classes: int, left_order: int, right_order: int
) -> tuple[int, ...]:
    output_order = left_order + right_order
    output_index = {
        value: index
        for index, value in enumerate(_composition_tuples(classes, output_order))
    }
    return tuple(
        output_index[tuple(a + b for a, b in zip(left, right))]
        for left in _composition_tuples(classes, left_order)
        for right in _composition_tuples(classes, right_order)
    )


@lru_cache(maxsize=None)
def _polynomial_product_map_tensor_cpu(
    classes: int, left_order: int, right_order: int
) -> torch.Tensor:
    return torch.tensor(
        _polynomial_product_map(classes, left_order, right_order),
        dtype=torch.long,
    )


@lru_cache(maxsize=128)
def _polynomial_product_map_tensor_device(
    classes: int, left_order: int, right_order: int, device: torch.device
) -> torch.Tensor:
    """Share immutable product indices across DP nodes on the same device.

    Scatter backward retains these indices. Copying them separately at every
    tree node made a long patient bag retain hundreds of identical GPU maps.
    """
    return _polynomial_product_map_tensor_cpu(
        classes, left_order, right_order
    ).to(device=device)


def _multiply_polynomials(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    classes: int,
    left_order: int,
    right_order: int,
) -> torch.Tensor:
    batch = left.shape[0]
    output_size = len(_composition_tuples(classes, left_order + right_order))
    products = (left.unsqueeze(2) * right.unsqueeze(1)).reshape(batch, -1)
    destination = _polynomial_product_map_tensor_cpu(
        classes, left_order, right_order
    )
    if left.device.type != "cpu":
        destination = _polynomial_product_map_tensor_device(
            classes, left_order, right_order, left.device
        )
    output = torch.zeros(
        (batch, output_size), dtype=left.dtype, device=left.device
    )
    return output.scatter_add(
        1, destination.unsqueeze(0).expand(batch, -1), products
    )


def _multinomial_coefficients(k: torch.Tensor, order: int) -> torch.Tensor:
    factorial = torch.tensor(
        [math.factorial(value) for value in range(order + 1)],
        dtype=torch.float64,
        device=k.device,
    )
    return math.factorial(order) / factorial[k].prod(dim=1)


def _power_sum_polynomial(
    probabilities: torch.Tensor, order: int, *, chunk_size: int = 64
) -> torch.Tensor:
    """Coefficients of ``sum_i (sum_c p_ic u_c)**order``."""
    batch, _, classes = probabilities.shape
    k = compositions(classes, order, device=probabilities.device)
    multinomial = _multinomial_coefficients(k, order).to(probabilities.dtype)
    pieces: list[torch.Tensor] = []
    for start in range(0, len(k), chunk_size):
        exponents = k[start : start + chunk_size]
        monomials = probabilities.unsqueeze(2).pow(
            exponents.view(1, 1, -1, classes)
        ).prod(dim=-1)
        pieces.append(monomials.sum(dim=1))
    return torch.cat(pieces, dim=1) * multinomial.unsqueeze(0)


def _combination_ratio(m: int, r_minus_k: int, r: int) -> float:
    """Stable ``comb(m, r-k) / comb(m, r)`` for small orders and large m."""
    k = r - r_minus_k
    numerator = 1.0
    denominator = 1.0
    for offset in range(k):
        numerator *= float(r - offset)
        denominator *= float(m - r + 1 + offset)
    return numerator / denominator


def _first_two_classifier_masses(
    probabilities: torch.Tensor, max_order: int
) -> list[torch.Tensor]:
    """Fast aggregate formulas for the smoke-test-critical orders 1 and 2."""
    batch, m, classes = probabilities.shape
    totals = probabilities.sum(dim=1)
    first_k = compositions(classes, 1, device=probabilities.device)
    first = torch.stack(
        [totals[:, int(row.argmax().item())] / float(m) for row in first_k], dim=1
    )
    result = [first]
    if max_order == 1:
        return result

    gram_diagonal = torch.einsum(
        "bmc,bmd->bcd", probabilities, probabilities
    )
    second_values: list[torch.Tensor] = []
    denominator = float(math.comb(m, 2))
    for row in compositions(classes, 2, device=probabilities.device):
        nonzero = torch.nonzero(row, as_tuple=False).reshape(-1)
        if len(nonzero) == 1:
            class_id = int(nonzero[0].item())
            coefficient = (
                totals[:, class_id].square()
                - gram_diagonal[:, class_id, class_id]
            ) / 2.0
        else:
            first_class, second_class = (int(value.item()) for value in nonzero)
            coefficient = (
                totals[:, first_class] * totals[:, second_class]
                - gram_diagonal[:, first_class, second_class]
            )
        second_values.append(coefficient / denominator)
    result.append(torch.stack(second_values, dim=1))
    return result


def _stable_product_tree_classifier_masses(
    probabilities: torch.Tensor, max_order: int
) -> list[torch.Tensor]:
    """Cancellation-free normalized subset DP for categorical count masses.

    Each node stores the count-profile distribution of a uniformly sampled
    subset from its instances.  Merging two nodes is a positive mixture with
    hypergeometric weights.  A balanced tree keeps both the floating-point
    reduction depth and the autograd graph depth logarithmic in bag size.
    """
    batch, bag_size, classes = probabilities.shape
    first_k = compositions(classes, 1, device=probabilities.device)
    class_indices = first_k.argmax(dim=1)
    linear_terms = probabilities.index_select(2, class_indices)
    one = torch.ones(
        (batch, 1), dtype=probabilities.dtype, device=probabilities.device
    )
    nodes: list[tuple[int, list[torch.Tensor]]] = [
        (1, [one, linear_terms[:, instance_index, :]])
        for instance_index in range(bag_size)
    ]

    while len(nodes) > 1:
        merged_nodes: list[tuple[int, list[torch.Tensor]]] = []
        for node_index in range(0, len(nodes), 2):
            if node_index + 1 == len(nodes):
                merged_nodes.append(nodes[node_index])
                continue
            left_size, left_masses = nodes[node_index]
            right_size, right_masses = nodes[node_index + 1]
            merged_size = left_size + right_size
            merged_max_order = min(max_order, merged_size)
            merged_masses: list[torch.Tensor] = []
            for order in range(merged_max_order + 1):
                components: list[torch.Tensor] = []
                minimum_left_order = max(0, order - (len(right_masses) - 1))
                maximum_left_order = min(order, len(left_masses) - 1)
                denominator = math.comb(merged_size, order)
                for left_order in range(
                    minimum_left_order, maximum_left_order + 1
                ):
                    right_order = order - left_order
                    mixture_weight = (
                        math.comb(left_size, left_order)
                        * math.comb(right_size, right_order)
                        / denominator
                    )
                    product = _multiply_polynomials(
                        left_masses[left_order],
                        right_masses[right_order],
                        classes=classes,
                        left_order=left_order,
                        right_order=right_order,
                    )
                    components.append(product * mixture_weight)
                merged = (
                    components[0]
                    if len(components) == 1
                    else torch.stack(components, dim=0).sum(dim=0)
                )
                merged_masses.append(
                    merged
                    / merged.sum(dim=1, keepdim=True).clamp_min(
                        torch.finfo(merged.dtype).tiny
                    )
                )
            merged_nodes.append((merged_size, merged_masses))
        nodes = merged_nodes

    return nodes[0][1][1 : max_order + 1]


def _stable_scan_classifier_masses(
    probabilities: torch.Tensor, max_order: int
) -> list[torch.Tensor]:
    """Cancellation-free normalized subset DP scanned over instances.

    After processing ``m`` instances, ``masses[r]`` is the categorical
    count-profile distribution of a uniformly sampled size-``r`` subset.
    Adding instance ``m + 1`` gives the exact positive recurrence

    ``Q_new[r] = (1-r/(m+1)) Q_old[r] + r/(m+1) (Q_old[r-1] * p_new)``.

    Unlike the Newton recurrence, this path never subtracts nearly equal
    terms.  Multiplication is only by a degree-one categorical polynomial,
    which is substantially cheaper than the general balanced product tree
    for the sixth- and seventh-order variable-size UNSW bags.
    """
    batch, bag_size, classes = probabilities.shape
    first_k = compositions(classes, 1, device=probabilities.device)
    class_indices = first_k.argmax(dim=1)
    linear_terms = probabilities.index_select(2, class_indices)
    one = torch.ones(
        (batch, 1), dtype=probabilities.dtype, device=probabilities.device
    )
    masses: list[torch.Tensor] = [one]

    for instance_index in range(bag_size):
        processed = instance_index + 1
        linear = linear_terms[:, instance_index, :]
        maximum_order = min(max_order, processed)
        for order in range(maximum_order, 0, -1):
            included = _multiply_polynomials(
                masses[order - 1],
                linear,
                classes=classes,
                left_order=order - 1,
                right_order=1,
            )
            inclusion_probability = float(order) / float(processed)
            if order < len(masses):
                updated = (
                    (1.0 - inclusion_probability) * masses[order]
                    + inclusion_probability * included
                )
                masses[order] = updated
            else:
                # A new order appears only when ``order == processed``, so
                # its inclusion probability is exactly one.
                masses.append(included)
            masses[order] = masses[order] / masses[order].sum(
                dim=1, keepdim=True
            ).clamp_min(torch.finfo(masses[order].dtype).tiny)

    return masses[1:]


def variable_stable_scan_classifier_masses(
    instance_probabilities: torch.Tensor,
    bag_index: torch.Tensor,
    bag_sizes: torch.Tensor,
    max_order: int,
) -> list[torch.Tensor]:
    """Exact stable-scan masses for a batch of variable-size bags.

    This is the same positive normalized-subset recurrence as
    :func:`_stable_scan_classifier_masses`, evaluated for every bag in
    parallel.  Padding is masked: once a bag ends, all of its states remain
    unchanged.  Consequently padding cannot contribute probability mass or a
    gradient.
    """
    if instance_probabilities.ndim != 2:
        raise ValueError(
            "instance_probabilities must have shape [instances, classes]"
        )
    if bag_index.ndim != 1 or len(bag_index) != len(instance_probabilities):
        raise ValueError("bag_index must identify every instance")
    if bag_sizes.ndim != 1 or not len(bag_sizes):
        raise ValueError("bag_sizes must be a non-empty vector")
    if not 1 <= max_order <= MAX_MULTICLASS_MOMENT_ORDER:
        raise ValueError(
            f"max_order must be in [1, {MAX_MULTICLASS_MOMENT_ORDER}]"
        )
    sizes = bag_sizes.to(device=bag_index.device, dtype=torch.long)
    if int(sizes.min().item()) < max_order:
        raise ValueError("every bag size must be at least max_order")
    observed = torch.bincount(bag_index, minlength=len(sizes))
    if not torch.equal(observed.to(sizes), sizes):
        raise ValueError("bag_index counts do not match bag_sizes")

    probabilities = instance_probabilities.to(torch.float64)
    if not torch.isfinite(probabilities).all():
        raise ValueError("instance probabilities contain NaN or Inf")
    probabilities = probabilities.clamp_min(0.0)
    probability_sums = probabilities.sum(dim=1, keepdim=True)
    if (probability_sums <= 0).any():
        raise ValueError("each instance probability row must have positive mass")
    probabilities = probabilities / probability_sums

    bags = len(sizes)
    maximum_size = int(sizes.max().item())
    classes = probabilities.shape[1]
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=bag_index.device),
            sizes.cumsum(dim=0)[:-1],
        )
    )
    positions = (
        torch.arange(
            len(probabilities), dtype=torch.long, device=bag_index.device
        )
        - offsets.index_select(0, bag_index)
    )
    padded = probabilities.new_zeros((bags, maximum_size, classes))
    padded[bag_index, positions] = probabilities

    first_k = compositions(classes, 1, device=probabilities.device)
    class_indices = first_k.argmax(dim=1)
    linear_terms = padded.index_select(2, class_indices)
    one = torch.ones(
        (bags, 1), dtype=probabilities.dtype, device=probabilities.device
    )
    masses: list[torch.Tensor] = [one]
    masses.extend(
        probabilities.new_zeros(
            (bags, len(_composition_tuples(classes, order)))
        )
        for order in range(1, max_order + 1)
    )
    tiny = torch.finfo(probabilities.dtype).tiny

    for instance_index in range(maximum_size):
        processed = instance_index + 1
        active = sizes > instance_index
        linear = linear_terms[:, instance_index, :]
        for order in range(min(max_order, processed), 0, -1):
            included = _multiply_polynomials(
                masses[order - 1],
                linear,
                classes=classes,
                left_order=order - 1,
                right_order=1,
            )
            inclusion_probability = float(order) / float(processed)
            updated = (
                (1.0 - inclusion_probability) * masses[order]
                + inclusion_probability * included
            )
            updated = updated / updated.sum(
                dim=1, keepdim=True
            ).clamp_min(tiny)
            masses[order] = torch.where(
                active.unsqueeze(1), updated, masses[order]
            )

    return masses[1:]


def classifier_count_masses(
    instance_probabilities: torch.Tensor,
    max_order: int,
    *,
    power_chunk_size: int = 64,
    algorithm: str = "newton",
) -> list[torch.Tensor]:
    """Return exact categorical count-profile masses for orders 1..R.

    ``instance_probabilities`` has shape ``[B, m, C]``.  At order ``r``, the
    output has shape ``[B, comb(C+r-1, r)]`` and sums to one.  It is the
    coefficient vector of the normalized elementary-symmetric generating
    polynomial for a uniform subset of ``r`` distinct instances.
    """
    if instance_probabilities.ndim != 3:
        raise ValueError("instance_probabilities must have shape [B, m, C]")
    if not 1 <= max_order <= MAX_MULTICLASS_MOMENT_ORDER:
        raise ValueError(
            f"max_order must be in [1, {MAX_MULTICLASS_MOMENT_ORDER}]"
        )
    if algorithm not in MULTICLASS_MOMENT_ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {MULTICLASS_MOMENT_ALGORITHMS}"
        )
    batch, m, classes = instance_probabilities.shape
    if m < max_order:
        raise ValueError(f"bag size {m} is smaller than moment order {max_order}")
    probabilities = instance_probabilities.to(torch.float64)
    if not torch.isfinite(probabilities).all():
        raise ValueError("instance probabilities contain NaN or Inf")
    probabilities = probabilities.clamp_min(0.0)
    probability_sums = probabilities.sum(dim=-1, keepdim=True)
    if (probability_sums <= 0).any():
        raise ValueError("each instance probability row must have positive mass")
    probabilities = probabilities / probability_sums

    if algorithm == "stable_dp":
        masses = _stable_product_tree_classifier_masses(
            probabilities, max_order
        )
    elif algorithm == "stable_scan":
        masses = _stable_scan_classifier_masses(
            probabilities, max_order
        )
    elif max_order <= 2:
        masses = _first_two_classifier_masses(probabilities, max_order)
    else:
        power_sums: list[torch.Tensor | None] = [None]
        power_sums.extend(
            _power_sum_polynomial(
                probabilities, order, chunk_size=power_chunk_size
            )
            for order in range(1, max_order + 1)
        )
        elementary: list[torch.Tensor] = [
            torch.ones((batch, 1), dtype=probabilities.dtype, device=probabilities.device)
        ]
        for order in range(1, max_order + 1):
            value = torch.zeros(
                (
                    batch,
                    len(_composition_tuples(classes, order)),
                ),
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            for power in range(1, order + 1):
                product = _multiply_polynomials(
                    elementary[order - power],
                    power_sums[power],
                    classes=classes,
                    left_order=order - power,
                    right_order=power,
                )
                scale = _combination_ratio(m, order - power, order) / float(order)
                value = value + (1.0 if power % 2 else -1.0) * scale * product
            elementary.append(value)
        masses = elementary[1:]

    normalized: list[torch.Tensor] = []
    for order, mass in enumerate(masses, start=1):
        if (mass < -1e-8).any():
            minimum = float(mass.min().detach().cpu())
            raise FloatingPointError(
                f"order-{order} classifier moment has negative mass {minimum}"
            )
        mass = mass.clamp_min(0.0)
        normalized.append(mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-15))
    return normalized


def exact_counts_from_proportions(
    proportions: torch.Tensor,
    bag_size: int,
    *,
    tolerance: float = 1e-2,
) -> torch.Tensor:
    counts_float = proportions.to(torch.float64) * float(bag_size)
    counts = counts_float.round()
    maximum_error = float((counts_float - counts).abs().max().detach().cpu())
    if maximum_error > tolerance:
        raise ValueError(
            "chip_exact proportions do not encode integral pixel counts: "
            f"max |alpha*m-round(alpha*m)|={maximum_error:.6g} > {tolerance}"
        )
    if not torch.equal(
        counts.sum(dim=1).to(torch.long),
        torch.full(
            (len(counts),), bag_size, dtype=torch.long, device=counts.device
        ),
    ):
        raise ValueError("rounded chip_exact class counts do not sum to bag_size")
    return counts.to(torch.long)


def rounded_counts_from_proportions(
    proportions: torch.Tensor, bag_size: int
) -> torch.Tensor:
    """Largest-remainder pseudo-counts for an explicitly approximate protocol."""
    values = proportions.to(torch.float64) * float(bag_size)
    floors = values.floor().to(torch.long)
    missing = bag_size - floors.sum(dim=1)
    fractions = values - floors
    output = floors.clone()
    for row in range(len(output)):
        amount = int(missing[row].item())
        if amount < 0 or amount > output.shape[1]:
            raise ValueError("invalid largest-remainder pseudo-count state")
        if amount:
            selected = torch.topk(fractions[row], amount).indices
            output[row, selected] += 1
    if not torch.equal(
        output.sum(dim=1),
        torch.full(
            (len(output),), bag_size, dtype=torch.long, device=output.device
        ),
    ):
        raise RuntimeError("pseudo-count rounding failed to preserve bag size")
    return output


def target_count_masses(
    label_counts: torch.Tensor, max_order: int
) -> list[torch.Tensor]:
    """Multivariate-hypergeometric masses for fixed or variable bag sizes."""
    if label_counts.ndim != 2:
        raise ValueError("label_counts must have shape [B, C]")
    counts = label_counts.to(torch.long)
    if (counts < 0).any():
        raise ValueError("label counts must be non-negative")
    batch, classes = counts.shape
    bag_sizes = counts.sum(dim=1)
    if int(bag_sizes.min().item()) < max_order:
        raise ValueError("bag size is smaller than moment order")

    choose = torch.ones(
        (batch, classes, max_order + 1),
        dtype=torch.float64,
        device=counts.device,
    )
    counts_float = counts.to(torch.float64)
    for value in range(1, max_order + 1):
        choose[..., value] = (
            choose[..., value - 1]
            * (counts_float - float(value - 1)).clamp_min(0.0)
            / float(value)
        )

    outputs: list[torch.Tensor] = []
    for order in range(1, max_order + 1):
        k = compositions(classes, order, device=counts.device)
        mass = torch.ones(
            (batch, len(k)), dtype=torch.float64, device=counts.device
        )
        for class_id in range(classes):
            mass = mass * choose[:, class_id, :].gather(
                1, k[:, class_id].unsqueeze(0).expand(batch, -1)
            )
        denominator = torch.tensor(
            [
                math.comb(int(bag_size), order)
                for bag_size in bag_sizes.detach().cpu().tolist()
            ],
            dtype=torch.float64,
            device=counts.device,
        )
        mass = mass / denominator.unsqueeze(1)
        outputs.append(mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-15))
    return outputs


class MulticlassFactorialMomentLoss(nn.Module):
    """Exact joint categorical factorial-moment loss for fixed-size dense bags."""

    def __init__(
        self,
        num_classes: int,
        max_order: int,
        bag_size: int,
        *,
        loss_type: str = "ce",
        order_weights: Sequence[float] | None = None,
        counts_tolerance: float = 1e-2,
        power_chunk_size: int = 64,
        moment_algorithm: str = "newton",
        ce_smoothing_tau: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be at least two")
        if not 1 <= max_order <= MAX_MULTICLASS_MOMENT_ORDER:
            raise ValueError(
                f"max_order must be in [1, {MAX_MULTICLASS_MOMENT_ORDER}]"
            )
        if bag_size < max_order:
            raise ValueError("bag_size must be at least max_order")
        if loss_type not in {"ce", "mse"}:
            raise ValueError("loss_type must be ce or mse")
        if not 0.0 <= float(ce_smoothing_tau) < 1.0:
            raise ValueError("ce_smoothing_tau must be in [0, 1)")
        if ce_smoothing_tau and loss_type != "ce":
            raise ValueError("ce_smoothing_tau requires CE loss")
        if moment_algorithm not in MULTICLASS_MOMENT_ALGORITHMS:
            raise ValueError(
                "moment_algorithm must be one of "
                f"{MULTICLASS_MOMENT_ALGORITHMS}"
            )
        if order_weights is None:
            weights = torch.full((max_order,), 1.0 / max_order, dtype=torch.float64)
        else:
            if len(order_weights) != max_order:
                raise ValueError(f"order_weights must have length {max_order}")
            weights = torch.tensor(order_weights, dtype=torch.float64)
            if (
                not torch.isfinite(weights).all()
                or (weights < 0).any()
                or weights.sum() <= 0
            ):
                raise ValueError("order weights must be finite, non-negative, and nonzero")
            weights = weights / weights.sum()
        self.num_classes = int(num_classes)
        self.max_order = int(max_order)
        self.bag_size = int(bag_size)
        self.loss_type = loss_type
        self.counts_tolerance = float(counts_tolerance)
        self.power_chunk_size = int(power_chunk_size)
        self.moment_algorithm = str(moment_algorithm)
        self.ce_smoothing_tau = float(ce_smoothing_tau)
        self.register_buffer("order_weights", weights, persistent=True)

    def forward(
        self,
        label_proportions: torch.Tensor,
        instance_probabilities: torch.Tensor,
        *,
        label_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if label_proportions.ndim != 2:
            raise ValueError("label_proportions must have shape [B, C]")
        if instance_probabilities.ndim == 2:
            expected = len(label_proportions) * self.bag_size
            if len(instance_probabilities) != expected:
                raise ValueError(
                    f"expected {expected} flattened instances, got "
                    f"{len(instance_probabilities)}"
                )
            instance_probabilities = instance_probabilities.reshape(
                len(label_proportions), self.bag_size, self.num_classes
            )
        if instance_probabilities.shape != (
            len(label_proportions),
            self.bag_size,
            self.num_classes,
        ):
            raise ValueError(
                "instance probabilities must have shape "
                f"[B, {self.bag_size}, {self.num_classes}], got "
                f"{tuple(instance_probabilities.shape)}"
            )
        if label_proportions.shape[1] != self.num_classes:
            raise ValueError("proportion class dimension does not match num_classes")
        alpha = label_proportions.to(instance_probabilities.device)
        if not torch.isfinite(alpha).all() or (alpha < 0).any():
            raise ValueError("label proportions must be finite and non-negative")
        if not torch.allclose(
            alpha.sum(dim=1),
            torch.ones(len(alpha), dtype=alpha.dtype, device=alpha.device),
            atol=1e-4,
            rtol=0.0,
        ):
            raise ValueError("label proportion rows must sum to one")
        predicted = classifier_count_masses(
            instance_probabilities,
            self.max_order,
            power_chunk_size=self.power_chunk_size,
            algorithm=self.moment_algorithm,
        )
        if self.max_order == 1:
            first_k = compositions(
                self.num_classes, 1, device=alpha.device
            )
            targets = [
                torch.stack(
                    [
                        alpha[:, int(row.argmax().item())].to(torch.float64)
                        for row in first_k
                    ],
                    dim=1,
                )
            ]
        else:
            counts = (
                exact_counts_from_proportions(
                    alpha, self.bag_size, tolerance=self.counts_tolerance
                )
                if label_counts is None
                else label_counts.to(device=alpha.device, dtype=torch.long)
            )
            if counts.shape != alpha.shape or (counts < 0).any():
                raise ValueError(
                    "label_counts must be non-negative with shape [B, C]"
                )
            if not torch.equal(
                counts.sum(dim=1),
                torch.full(
                    (len(counts),),
                    self.bag_size,
                    dtype=torch.long,
                    device=counts.device,
                ),
            ):
                raise ValueError("label_counts rows must sum to bag_size")
            targets = target_count_masses(counts, self.max_order)
        loss = torch.zeros((), dtype=torch.float64, device=alpha.device)
        for order, (target, prediction) in enumerate(
            zip(targets, predicted), start=1
        ):
            weight = self.order_weights[order - 1]
            if float(weight) == 0.0:
                continue
            if self.loss_type == "ce":
                if self.ce_smoothing_tau:
                    k = compositions(
                        self.num_classes, order, device=prediction.device
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
                    background = permutation_count / float(
                        self.num_classes**order
                    )
                    prediction = (
                        (1.0 - self.ce_smoothing_tau) * prediction
                        + self.ce_smoothing_tau * background.unsqueeze(0)
                    )
                order_loss = -(
                    target * prediction.clamp_min(1e-15).log()
                ).sum(dim=1)
            else:
                k = compositions(
                    self.num_classes, order, device=prediction.device
                )
                permutation_count = (
                    math.factorial(order)
                    / torch.tensor(
                        [
                            math.prod(math.factorial(int(value)) for value in row)
                            for row in k.cpu().tolist()
                        ],
                        dtype=torch.float64,
                        device=prediction.device,
                    )
                )
                # Exact mean squared discrepancy across the C**r ordered entries.
                order_loss = (
                    (target - prediction).square() / permutation_count.unsqueeze(0)
                ).sum(dim=1) / float(self.num_classes**order)
            loss = loss + weight * order_loss.mean()
        return loss.to(instance_probabilities.dtype)


def dense_instance_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("dense classifier logits must have shape [B, C, H, W]")
    return (
        logits.softmax(dim=1)
        .permute(0, 2, 3, 1)
        .reshape(logits.shape[0], -1, logits.shape[1])
    )
