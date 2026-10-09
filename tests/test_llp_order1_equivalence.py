from __future__ import annotations

import torch

from mo_matching.llp.structured_multiclass import (
    bag_weights_from_sizes,
    variable_multiclass_llp_mm_loss,
)


def test_multiclass_order_one_matches_bag_proportion_mse() -> None:
    logits = torch.randn(11, 4, dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    bag_index = torch.tensor([0] * 5 + [1] * 6)
    sizes = torch.tensor([5, 6])
    counts = torch.tensor([[5, 0, 0, 0], [1, 2, 0, 3]])
    proportions = counts / sizes[:, None]
    mm = variable_multiclass_llp_mm_loss(
        probabilities, bag_index, sizes, counts, proportions, 1
    )
    means = torch.stack((probabilities[:5].mean(0), probabilities[5:].mean(0)))
    expected = torch.nn.functional.mse_loss(means, proportions)
    assert torch.allclose(mm, expected, atol=1e-6, rtol=1e-5)
    mm.backward()
    assert torch.isfinite(logits.grad).all()


def test_size_weighted_order_one_matches_size_weighted_proportion_mse() -> None:
    logits = torch.randn(14, 4, dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    bag_index = torch.tensor([0] * 2 + [1] * 4 + [2] * 8)
    sizes = torch.tensor([2, 4, 8])
    counts = torch.tensor(
        [[2, 0, 0, 0], [1, 2, 0, 1], [1, 2, 3, 2]]
    )
    proportions = counts / sizes[:, None]
    weights = bag_weights_from_sizes(sizes, "size")
    mm = variable_multiclass_llp_mm_loss(
        probabilities,
        bag_index,
        sizes,
        counts,
        proportions,
        1,
        bag_weights=weights,
    )
    means = torch.stack([
        probabilities[bag_index == i].mean(0) for i in range(len(sizes))
    ])
    per_bag = (means - proportions).square().mean(1)
    expected = (per_bag * weights).sum() / weights.sum()
    assert torch.allclose(mm, expected, atol=1e-6, rtol=1e-5)
    mm.backward()
    assert torch.isfinite(logits.grad).all()


def test_sqrt_size_weighted_ce_matches_manual_bag_weighting() -> None:
    probabilities = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=torch.float64,
    )
    bag_index = torch.tensor([0, 0, 1, 1, 1])
    proportions = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    weights = bag_weights_from_sizes(torch.tensor([2, 3]), "sqrt_size")
    observed = variable_multiclass_llp_mm_loss(
        probabilities, bag_index, torch.tensor([2, 3]),
        torch.tensor([[1, 1], [0, 3]]), proportions, 1,
        bag_weights=weights, moment_algorithm="stable_dp", loss_type="ce",
    )
    means = torch.stack((probabilities[:2].mean(0), probabilities[2:].mean(0)))
    per_bag = -(proportions * means.log()).sum(1)
    expected = (per_bag * weights).sum() / weights.sum()
    assert torch.allclose(observed, expected)
