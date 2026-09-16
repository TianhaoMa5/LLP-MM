from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from plench.data.ref2021 import _llp_mm_variable_loss  # noqa: E402


def _two_bag_inputs():
    logits = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    proportions = torch.tensor(
        [[0.5, 1.0 / 3.0, 1.0 / 6.0], [0.2, 0.4, 0.4]],
        dtype=torch.float64,
    )
    slices = [torch.arange(0, 6), torch.arange(6, 11)]
    return logits, probabilities, proportions, slices


def test_order_one_strict_ce_matches_paper_smoothed_cross_entropy() -> None:
    logits, probabilities, proportions, slices = _two_bag_inputs()
    loss = _llp_mm_variable_loss(
        probabilities,
        proportions,
        slices,
        1,
        order_weights=[1.0],
        loss_type="ce",
        moment_algorithm="stable_dp",
        compute_dtype="float64",
        ce_smoothing_tau=1e-4,
    )
    means = torch.stack(
        [probabilities[indices].mean(dim=0) for indices in slices]
    )
    smoothed = (1.0 - 1e-4) * means + 1e-4 / means.shape[1]
    expected = -(proportions * smoothed.log()).sum(dim=1).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    assert loss.dtype == torch.float64
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_order_six_strict_ce_is_finite_and_differentiable() -> None:
    logits = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    loss = _llp_mm_variable_loss(
        probabilities,
        torch.tensor([[0.25, 0.25, 0.5]], dtype=torch.float64),
        [torch.arange(12)],
        6,
        order_weights=[1.0 / 6.0] * 6,
        loss_type="ce",
        moment_algorithm="stable_dp",
        compute_dtype="float64",
        ce_smoothing_tau=1e-4,
    )
    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    (
        "probability_dtype",
        "loss_type",
        "moment_algorithm",
        "compute_dtype",
        "ce_smoothing_tau",
    ),
    [
        (torch.float32, "ce", "stable_dp", "float64", 1e-4),
        (torch.float64, "mse", "stable_dp", "float64", 1e-4),
        (torch.float64, "ce", "newton", "float64", 1e-4),
        (torch.float64, "ce", "stable_dp", "float32", 1e-4),
        (torch.float64, "ce", "stable_dp", "float64", 0.0),
    ],
)
def test_strict_llp_mm_rejects_non_paper_modes(
    probability_dtype,
    loss_type,
    moment_algorithm,
    compute_dtype,
    ce_smoothing_tau,
) -> None:
    _, probabilities, proportions, slices = _two_bag_inputs()
    with pytest.raises(ValueError):
        _llp_mm_variable_loss(
            probabilities.to(probability_dtype),
            proportions,
            slices,
            1,
            loss_type=loss_type,
            moment_algorithm=moment_algorithm,
            compute_dtype=compute_dtype,
            ce_smoothing_tau=ce_smoothing_tau,
        )
