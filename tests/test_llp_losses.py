from __future__ import annotations

import itertools

import pytest
import torch

from mo_matching.llp.losses import (
    _predicted_count_distribution,
    _predicted_count_distributions_batched,
    calibrate_logits_to_bag_proportions,
    calibrate_probabilities_to_bag_proportions,
    conditional_llp_mm_loss,
    dllp_bce_loss,
    llp_mm_loss,
)


def test_llp_mm_order_one_matches_dllp_bce() -> None:
    probabilities = torch.tensor(
        [0.1, 0.7, 0.2, 0.9, 0.8, 0.6, 0.4, 0.3], requires_grad=True
    )
    bag_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    bag_sizes = torch.tensor([4, 4])
    label_counts = torch.tensor([1, 3])
    proportions = label_counts.float() / bag_sizes.float()
    mm = llp_mm_loss(probabilities, bag_index, bag_sizes, label_counts, 1)
    bce = dllp_bce_loss(probabilities, bag_index, proportions)
    assert torch.allclose(mm, bce, atol=1e-6, rtol=1e-5)
    mm.backward()
    assert torch.isfinite(probabilities.grad).all()


@pytest.mark.parametrize("order", [2, 3, 8, 9, 10, 14])
@pytest.mark.parametrize("label_count", [0, 16])
def test_high_order_extreme_bags_are_finite(order: int, label_count: int) -> None:
    probabilities = torch.linspace(0.01, 0.99, 16, requires_grad=True)
    loss = llp_mm_loss(
        probabilities,
        torch.zeros(16, dtype=torch.long),
        torch.tensor([16]),
        torch.tensor([label_count]),
        order,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()


def test_high_order_rejects_too_small_bag() -> None:
    with pytest.raises(ValueError, match="smaller than moment order"):
        llp_mm_loss(
            torch.tensor([0.2, 0.8]),
            torch.tensor([0, 0]),
            torch.tensor([2]),
            torch.tensor([1]),
            3,
        )


def test_order_four_distribution_matches_brute_force() -> None:
    probabilities = torch.tensor(
        [0.07, 0.19, 0.31, 0.48, 0.62, 0.76], dtype=torch.float64
    )
    expected = torch.zeros(5, dtype=torch.float64)
    for subset in itertools.combinations(range(len(probabilities)), 4):
        subset_distribution = torch.ones(1, dtype=torch.float64)
        for index in subset:
            probability = probabilities[index]
            subset_distribution = (
                torch.cat((subset_distribution, torch.zeros(1))) * (1 - probability)
                + torch.cat((torch.zeros(1), subset_distribution)) * probability
            )
        expected += subset_distribution
    expected /= len(list(itertools.combinations(range(len(probabilities)), 4)))

    actual = _predicted_count_distribution(probabilities, 4)

    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-10)


def test_high_order_rejects_unsupported_order() -> None:
    with pytest.raises(ValueError, match=r"\[1, 14\]"):
        llp_mm_loss(
            torch.full((16,), 0.5),
            torch.zeros(16, dtype=torch.long),
            torch.tensor([16]),
            torch.tensor([8]),
            15,
        )


def test_batched_high_order_matches_individual_variable_bags() -> None:
    first = torch.linspace(0.05, 0.85, 8, dtype=torch.float64)
    second = torch.linspace(0.15, 0.95, 10, dtype=torch.float64)
    probabilities = torch.cat((first, second))
    bag_index = torch.tensor([0] * len(first) + [1] * len(second))
    bag_sizes = torch.tensor([len(first), len(second)])

    batched = _predicted_count_distributions_batched(
        probabilities, bag_index, bag_sizes, 8
    )

    for order, distributions in enumerate(batched, start=1):
        assert torch.allclose(
            distributions[0],
            _predicted_count_distribution(first, order),
            atol=1e-11,
            rtol=1e-10,
        )
        assert torch.allclose(
            distributions[1],
            _predicted_count_distribution(second, order),
            atol=1e-11,
            rtol=1e-10,
        )


def test_order_weights_can_select_first_order_exactly() -> None:
    probabilities = torch.tensor(
        [0.1, 0.7, 0.2, 0.9, 0.8, 0.6, 0.4, 0.3], requires_grad=True
    )
    bag_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    bag_sizes = torch.tensor([4, 4])
    label_counts = torch.tensor([1, 3])
    proportions = label_counts.float() / bag_sizes.float()

    weighted = llp_mm_loss(
        probabilities,
        bag_index,
        bag_sizes,
        label_counts,
        3,
        order_weights=[1.0, 0.0, 0.0],
    )
    reference = dllp_bce_loss(
        probabilities, bag_index, proportions
    )

    assert torch.allclose(weighted, reference, atol=1e-6, rtol=1e-5)


def test_order_weights_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="length 3"):
        llp_mm_loss(
            torch.full((4,), 0.5),
            torch.zeros(4, dtype=torch.long),
            torch.tensor([4]),
            torch.tensor([2]),
            3,
            order_weights=[1.0, 1.0],
        )


def test_bag_logit_calibration_matches_means_and_preserves_order() -> None:
    probabilities = torch.tensor(
        [0.05, 0.2, 0.7, 0.9, 0.1, 0.4, 0.6, 0.95],
        requires_grad=True,
    )
    bag_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    bag_sizes = torch.tensor([4, 4])
    targets = torch.tensor([0.25, 0.75])
    calibrated = calibrate_probabilities_to_bag_proportions(
        probabilities, bag_index, bag_sizes, targets
    )
    assert calibrated[:4].mean().item() == pytest.approx(0.25, abs=1e-6)
    assert calibrated[4:].mean().item() == pytest.approx(0.75, abs=1e-6)
    assert torch.equal(
        calibrated[:4].argsort(), probabilities[:4].argsort()
    )
    calibrated.square().sum().backward()
    assert torch.isfinite(probabilities.grad).all()


def test_conditional_high_order_shape_is_bag_shift_invariant() -> None:
    logits = torch.tensor(
        [-2.0, -0.5, 0.7, 1.8, -1.4, -0.2, 0.9, 2.2],
        requires_grad=True,
    )
    bag_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    bag_sizes = torch.tensor([4, 4])
    label_counts = torch.tensor([1, 3])
    shifted_logits = logits + torch.tensor(
        [1.3, 1.3, 1.3, 1.3, -0.8, -0.8, -0.8, -0.8]
    )
    weights = [0.0, 0.0, 1.0]
    reference = conditional_llp_mm_loss(
        torch.sigmoid(logits),
        bag_index,
        bag_sizes,
        label_counts,
        3,
        weights,
    )
    shifted = conditional_llp_mm_loss(
        torch.sigmoid(shifted_logits),
        bag_index,
        bag_sizes,
        label_counts,
        3,
        weights,
    )
    assert shifted.item() == pytest.approx(reference.detach().item(), abs=2e-6)
    reference.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:4].sum().item() == pytest.approx(0.0, abs=2e-6)
    assert logits.grad[4:].sum().item() == pytest.approx(0.0, abs=2e-6)


def test_implicit_logit_calibration_has_exact_zero_sum_gradient() -> None:
    logits = torch.linspace(-8.0, 8.0, 16, requires_grad=True)
    bag_index = torch.tensor([0] * 8 + [1] * 8)
    bag_sizes = torch.tensor([8, 8])
    calibrated = calibrate_logits_to_bag_proportions(
        logits,
        bag_index,
        bag_sizes,
        torch.tensor([3 / 8, 1 / 8]),
    )
    assert calibrated[:8].mean().item() == pytest.approx(3 / 8, abs=1e-6)
    assert calibrated[8:].mean().item() == pytest.approx(1 / 8, abs=1e-6)
    calibrated.square().sum().backward()
    assert logits.grad[:8].sum().item() == pytest.approx(0.0, abs=1e-7)
    assert logits.grad[8:].sum().item() == pytest.approx(0.0, abs=1e-7)
