from __future__ import annotations

import pytest
import torch

from mo_matching.llp.structured_multiclass import variable_multiclass_llp_mm_loss


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(
    "counts",
    [
        [20, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 18, 0, 1],
        [1, 2, 3, 4, 5, 2, 3],
    ],
)
def test_multiclass_unsw_moments_are_finite(order: int, counts: list[int]) -> None:
    logits = torch.randn(20, 7, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    count_tensor = torch.tensor([counts])
    size = torch.tensor([20])
    loss = variable_multiclass_llp_mm_loss(
        probabilities,
        torch.zeros(20, dtype=torch.long),
        size,
        count_tensor,
        count_tensor / size[:, None],
        order,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
