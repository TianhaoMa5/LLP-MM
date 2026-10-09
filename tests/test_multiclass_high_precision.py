from __future__ import annotations

import itertools
import math
from decimal import Decimal, getcontext

import pytest
import torch

from mo_matching.llp.multiclass import (
    classifier_count_masses,
    compositions,
    variable_stable_scan_classifier_masses,
)
from mo_matching.llp.structured_multiclass import (
    variable_multiclass_llp_mm_loss,
)


def _decimal_subset_reference(
    probabilities: list[list[str]], order: int
) -> torch.Tensor:
    getcontext().prec = 80
    decimal_probabilities = [
        [Decimal(value) for value in row] for row in probabilities
    ]
    classes = len(decimal_probabilities[0])
    states = [
        tuple(int(value) for value in row)
        for row in compositions(classes, order).tolist()
    ]
    state_index = {state: index for index, state in enumerate(states)}
    masses = [Decimal(0) for _ in states]
    for subset in itertools.combinations(range(len(probabilities)), order):
        for assignments in itertools.product(range(classes), repeat=order):
            count_profile = [0] * classes
            probability = Decimal(1)
            for instance_index, class_index in zip(subset, assignments):
                count_profile[class_index] += 1
                probability *= decimal_probabilities[instance_index][class_index]
            masses[state_index[tuple(count_profile)]] += probability
    denominator = Decimal(math.comb(len(probabilities), order))
    return torch.tensor(
        [float(value / denominator) for value in masses], dtype=torch.float64
    )


def test_stable_dp_matches_80_digit_subset_reference_through_order_five() -> None:
    probability_strings = [
        ["0.9700000000001", "0.0199999999999", "0.0100000000000"],
        ["0.0000000000010", "0.8999999999990", "0.1000000000000"],
        ["0.4999999999999", "0.0000000000001", "0.5000000000000"],
        ["0.0001000000000", "0.0002000000000", "0.9997000000000"],
        ["0.3333333333333", "0.3333333333333", "0.3333333333334"],
        ["0.8000000000000", "0.1500000000000", "0.0500000000000"],
    ]
    probabilities = torch.tensor(
        [[float(value) for value in row] for row in probability_strings],
        dtype=torch.float64,
    ).unsqueeze(0)

    masses = classifier_count_masses(
        probabilities, 5, algorithm="stable_dp"
    )

    for order, mass in enumerate(masses, start=1):
        reference = _decimal_subset_reference(probability_strings, order)
        torch.testing.assert_close(mass[0], reference, rtol=2e-14, atol=2e-15)
        assert float(mass.sum()) == pytest.approx(1.0, abs=3e-15)
        assert (mass >= 0).all()


def test_stable_scan_matches_80_digit_subset_reference_through_order_five() -> None:
    probability_strings = [
        ["0.9700000000001", "0.0199999999999", "0.0100000000000"],
        ["0.0000000000010", "0.8999999999990", "0.1000000000000"],
        ["0.4999999999999", "0.0000000000001", "0.5000000000000"],
        ["0.0001000000000", "0.0002000000000", "0.9997000000000"],
        ["0.3333333333333", "0.3333333333333", "0.3333333333334"],
        ["0.8000000000000", "0.1500000000000", "0.0500000000000"],
    ]
    probabilities = torch.tensor(
        [[float(value) for value in row] for row in probability_strings],
        dtype=torch.float64,
    ).unsqueeze(0)

    masses = classifier_count_masses(
        probabilities, 5, algorithm="stable_scan"
    )

    for order, mass in enumerate(masses, start=1):
        reference = _decimal_subset_reference(probability_strings, order)
        torch.testing.assert_close(mass[0], reference, rtol=2e-14, atol=2e-15)
        assert float(mass.sum()) == pytest.approx(1.0, abs=3e-15)
        assert (mass >= 0).all()


def test_stable_scan_matches_balanced_stable_dp_at_order_seven() -> None:
    torch.manual_seed(27)
    probabilities = torch.rand(2, 20, 7, dtype=torch.float64)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    tree = classifier_count_masses(
        probabilities, 7, algorithm="stable_dp"
    )
    scan = classifier_count_masses(
        probabilities, 7, algorithm="stable_scan"
    )

    for tree_mass, scan_mass in zip(tree, scan):
        torch.testing.assert_close(scan_mass, tree_mass, rtol=2e-13, atol=2e-15)


def test_variable_stable_scan_matches_independent_bags_and_gradients() -> None:
    torch.manual_seed(31)
    sizes = torch.tensor([7, 9, 12], dtype=torch.long)
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    values = (torch.rand(int(sizes.sum()), 7, dtype=torch.float64) + 0.1)
    variable_values = values.clone().requires_grad_()
    reference_values = values.clone().requires_grad_()

    variable = variable_stable_scan_classifier_masses(
        variable_values, bag_index, sizes, 7
    )
    references: list[list[torch.Tensor]] = []
    offset = 0
    for size in sizes.tolist():
        references.append(
            classifier_count_masses(
                reference_values[offset : offset + size].unsqueeze(0),
                7,
                algorithm="stable_scan",
            )
        )
        offset += size

    reference = [
        torch.cat([bag[order] for bag in references], dim=0)
        for order in range(7)
    ]
    for actual_mass, reference_mass in zip(variable, reference):
        torch.testing.assert_close(
            actual_mass, reference_mass, rtol=2e-13, atol=2e-15
        )

    variable_loss = sum(value.square().sum() for value in variable)
    reference_loss = sum(value.square().sum() for value in reference)
    variable_gradient = torch.autograd.grad(
        variable_loss, variable_values
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_loss, reference_values
    )[0]
    torch.testing.assert_close(
        variable_gradient, reference_gradient, rtol=3e-12, atol=2e-14
    )


def test_batched_variable_order_seven_loss_matches_per_bag_reference() -> None:
    torch.manual_seed(37)
    sizes = torch.tensor([7, 9], dtype=torch.long)
    bag_index = torch.repeat_interleave(torch.arange(2), sizes)
    counts = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 1, 2],
        ],
        dtype=torch.long,
    )
    proportions = counts / sizes.unsqueeze(1)
    logits = torch.randn(
        int(sizes.sum()), 7, dtype=torch.float64, requires_grad=True
    )
    reference_logits = logits.detach().clone().requires_grad_()
    weights = [0.0] * 6 + [1.0]

    actual = variable_multiclass_llp_mm_loss(
        logits.softmax(dim=1),
        bag_index,
        sizes,
        counts,
        proportions,
        7,
        weights,
        moment_algorithm="stable_scan",
        loss_type="ce",
    )
    reference = variable_multiclass_llp_mm_loss(
        reference_logits.softmax(dim=1),
        bag_index,
        sizes,
        counts,
        proportions,
        7,
        weights,
        moment_algorithm="stable_dp",
        loss_type="ce",
    )
    torch.testing.assert_close(actual, reference, rtol=2e-13, atol=2e-14)
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    reference_gradient = torch.autograd.grad(
        reference, reference_logits
    )[0]
    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=3e-11, atol=2e-13
    )


def test_class_balanced_high_order_ce_is_finite_and_default_is_exact() -> None:
    torch.manual_seed(41)
    sizes = torch.tensor([7, 9], dtype=torch.long)
    bag_index = torch.repeat_interleave(torch.arange(2), sizes)
    counts = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 1, 2],
        ],
        dtype=torch.long,
    )
    proportions = counts / sizes.unsqueeze(1)
    logits = torch.randn(
        int(sizes.sum()), 7, dtype=torch.float64, requires_grad=True
    )
    probabilities = logits.softmax(dim=1)
    weights = [0.0] * 6 + [1.0]
    prior = torch.tensor(
        [0.4, 0.2, 0.14, 0.1, 0.08, 0.05, 0.03],
        dtype=torch.float64,
    )

    default = variable_multiclass_llp_mm_loss(
        probabilities,
        bag_index,
        sizes,
        counts,
        proportions,
        7,
        weights,
        moment_algorithm="stable_scan",
        loss_type="ce",
    )
    explicit_zero = variable_multiclass_llp_mm_loss(
        probabilities,
        bag_index,
        sizes,
        counts,
        proportions,
        7,
        weights,
        moment_algorithm="stable_scan",
        loss_type="ce",
        class_prior=prior,
        ce_class_balance_power=0.0,
    )
    balanced = variable_multiclass_llp_mm_loss(
        probabilities,
        bag_index,
        sizes,
        counts,
        proportions,
        7,
        weights,
        moment_algorithm="stable_scan",
        loss_type="ce",
        class_prior=prior,
        ce_class_balance_power=0.5,
    )
    torch.testing.assert_close(default, explicit_zero, rtol=0, atol=0)
    assert torch.isfinite(balanced)
    balanced.backward()
    assert torch.isfinite(logits.grad).all()


def test_stable_dp_high_order_gradient_passes_double_precision_gradcheck() -> None:
    torch.manual_seed(9)
    probabilities = (
        torch.rand(1, 5, 3, dtype=torch.float64) + 0.2
    ).requires_grad_()

    def highest_order(value: torch.Tensor) -> torch.Tensor:
        return classifier_count_masses(
            value, 4, algorithm="stable_dp"
        )[-1]

    assert torch.autograd.gradcheck(
        highest_order,
        (probabilities,),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-4,
        fast_mode=True,
    )


def test_stable_dp_maximum_supported_order_is_finite_and_differentiable() -> None:
    logits = torch.linspace(
        -40.0, 40.0, 28, dtype=torch.float64
    ).reshape(1, 14, 2)
    logits.requires_grad_()
    masses = classifier_count_masses(
        logits.softmax(dim=-1), 14, algorithm="stable_dp"
    )
    assert len(masses) == 14
    for mass in masses:
        assert torch.isfinite(mass).all()
        assert (mass >= 0).all()
        assert float(mass.sum().detach()) == pytest.approx(1.0, abs=5e-15)
    sum(mass.square().sum() for mass in masses).backward()
    assert torch.isfinite(logits.grad).all()


def test_unknown_moment_algorithm_is_rejected() -> None:
    with pytest.raises(ValueError, match="algorithm must be one of"):
        classifier_count_masses(
            torch.full((1, 5, 3), 1 / 3),
            3,
            algorithm="unstable",
        )
