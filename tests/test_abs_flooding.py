"""ABS must reverse the gradient when an unbiased risk becomes negative."""

import pytest
import torch

from mo_matching.core.algorithms import EasyLLP, GeneralUPM


@pytest.mark.parametrize("algorithm_class", [EasyLLP, GeneralUPM])
@pytest.mark.parametrize(
    "enabled,threshold", [(False, 0.0), (True, -1.0), (True, 0.0), (True, 0.5)]
)
def test_negative_risk_update_obeys_abs_and_flooding(
    algorithm_class, enabled, threshold
):
    algorithm = algorithm_class.__new__(algorithm_class)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = 4
    algorithm.loss_type = "ce"
    algorithm.flooding = enabled
    algorithm.flooding_b = threshold
    algorithm.predict = lambda images: images
    algorithm._backward_step = lambda loss: loss
    logits = torch.tensor(
        [[4.0, -4.0]] * 4 + [[-4.0, 4.0]] * 4, dtype=torch.float64, requires_grad=True
    )
    proportions = torch.eye(2, dtype=torch.float64)
    if algorithm_class is EasyLLP:
        raw, _ = algorithm.EasyLLP_Loss(logits, proportions)
    else:
        raw = algorithm.GeneralUPM_Loss(logits, proportions)
    assert raw.item() < 0
    (raw_gradient,) = torch.autograd.grad(raw, logits)
    actual = algorithm.update((logits, proportions))
    (actual_gradient,) = torch.autograd.grad(actual, logits)
    corrected = enabled and threshold >= 0
    expected = (
        (raw.detach() - threshold).abs() + threshold if corrected else raw.detach()
    )
    torch.testing.assert_close(actual.detach(), expected)
    torch.testing.assert_close(
        actual_gradient, -raw_gradient if corrected else raw_gradient
    )
