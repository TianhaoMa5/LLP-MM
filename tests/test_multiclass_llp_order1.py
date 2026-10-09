from __future__ import annotations

import torch

from mo_matching.llp.structured_multiclass import (
    variable_multiclass_llp_mm_loss,
)


def test_multiclass_llp_mm_order_one_equals_proportion_mse() -> None:
    logits = torch.randn(12, 7, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    bag_index = torch.tensor([0] * 5 + [1] * 7)
    sizes = torch.tensor([5, 7])
    counts = torch.tensor([[5, 0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 3, 0, 1]])
    proportions = counts / sizes[:, None]
    actual = variable_multiclass_llp_mm_loss(
        probabilities, bag_index, sizes, counts, proportions, 1
    )
    means = torch.stack((probabilities[:5].mean(0), probabilities[5:].mean(0)))
    expected = torch.nn.functional.mse_loss(means, proportions)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
