from __future__ import annotations

import pytest
import torch

from mo_matching.llp.structured_multiclass import (
    calibrated_multiclass_order_weights,
    multiclass_order_multiplier,
    scheduled_multiclass_order_weights,
    variable_multiclass_llp_mm_loss,
)


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(
    "counts",
    [
        [50, 0, 0, 0, 0],
        [0, 0, 0, 0, 50],
        [25, 25, 0, 0, 0],
        [1, 2, 3, 4, 40],
    ],
)
def test_variable_multiclass_extreme_moments_are_finite(
    order: int, counts: list[int]
) -> None:
    logits = torch.randn(50, 5, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    count_tensor = torch.tensor([counts])
    size = torch.tensor([50])
    proportions = count_tensor / size[:, None]
    loss = variable_multiclass_llp_mm_loss(
        probabilities,
        torch.zeros(50, dtype=torch.long),
        size,
        count_tensor,
        proportions,
        order,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_order_rejects_bag_smaller_than_moment() -> None:
    with pytest.raises(ValueError, match="at least moment_order"):
        variable_multiclass_llp_mm_loss(
            torch.full((2, 3), 1 / 3),
            torch.zeros(2, dtype=torch.long),
            torch.tensor([2]),
            torch.tensor([[1, 1, 0]]),
            torch.tensor([[0.5, 0.5, 0.0]]),
            3,
        )


@pytest.mark.parametrize(
    ("order", "weights"),
    [
        (2, [0.1, 0.9]),
        (3, [0.1, 0.2, 0.7]),
        (4, [0.05, 0.1, 0.15, 0.7]),
        (5, [0.05, 0.05, 0.1, 0.2, 0.6]),
        (6, [0.02, 0.03, 0.05, 0.1, 0.2, 0.6]),
        (7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        (8, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_highest_order_weighted_objective_is_finite(
    order: int, weights: list[float]
) -> None:
    logits = torch.randn(24, 4, requires_grad=True)
    probabilities = logits.softmax(dim=1)
    counts = torch.tensor([[2, 4, 6, 12]])
    size = torch.tensor([24])
    loss = variable_multiclass_llp_mm_loss(
        probabilities,
        torch.zeros(24, dtype=torch.long),
        size,
        counts,
        counts / size[:, None],
        order,
        weights,
        moment_algorithm="stable_dp",
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("order", [6, 7])
def test_isolated_high_order_cross_entropy_is_finite(order: int) -> None:
    logits = torch.randn(24, 4, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([[2, 4, 6, 12]])
    weights = [0.0] * (order - 1) + [1.0]
    loss = variable_multiclass_llp_mm_loss(
        logits.softmax(dim=1),
        torch.zeros(24, dtype=torch.long),
        torch.tensor([24]),
        counts,
        counts / 24.0,
        order,
        weights,
        moment_algorithm="stable_dp",
        loss_type="ce",
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_inverse_cell_mse_scaling_makes_high_orders_effectively_larger() -> None:
    weights = calibrated_multiclass_order_weights(
        3,
        5,
        [1.0, 1.0, 1.0],
        "inverse_ordered_cell_mse",
    )
    assert weights == pytest.approx(
        [1.0 / 651.0, 25.0 / 651.0, 625.0 / 651.0]
    )
    assert sum(weights) == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_order_one_scaling_preserves_proportion_mse_weight() -> None:
    assert calibrated_multiclass_order_weights(
        1, 4, None, "inverse_ordered_cell_mse"
    ) == pytest.approx((1.0,))


def test_inverse_cell_rmse_scaling_is_gentler_than_mse_scaling() -> None:
    rmse_weights = calibrated_multiclass_order_weights(
        3, 5, [1.0, 1.0, 1.0], "inverse_ordered_cell_rmse"
    )
    mse_weights = calibrated_multiclass_order_weights(
        3, 5, [1.0, 1.0, 1.0], "inverse_ordered_cell_mse"
    )
    assert rmse_weights == pytest.approx((1 / 31, 5 / 31, 25 / 31))
    assert rmse_weights[2] < mse_weights[2]


def test_absolute_high_order_multiplier_compensates_cell_mse() -> None:
    assert multiclass_order_multiplier(1, 5, "inverse_ordered_cell_mse") == 1
    assert multiclass_order_multiplier(3, 5, "inverse_ordered_cell_rmse") == 25
    assert multiclass_order_multiplier(3, 5, "inverse_ordered_cell_mse") == 625


def test_high_order_curriculum_warms_up_then_reaches_target() -> None:
    target = (0.1, 0.2, 0.7)
    assert scheduled_multiclass_order_weights(3, target, 2, 2, 0) == (
        1.0,
        0.0,
        0.0,
    )
    assert scheduled_multiclass_order_weights(3, target, 2, 2, 2) == (
        pytest.approx(0.55),
        pytest.approx(0.1),
        pytest.approx(0.35),
    )
    assert scheduled_multiclass_order_weights(3, target, 2, 2, 3) == target


def test_high_order_curriculum_recovers_to_first_order() -> None:
    target = (0.1, 0.2, 0.7)
    assert scheduled_multiclass_order_weights(
        3,
        target,
        2,
        2,
        7,
        total_epochs=10,
        recovery_epochs=2,
    ) == target
    assert scheduled_multiclass_order_weights(
        3,
        target,
        2,
        2,
        9,
        total_epochs=10,
        recovery_epochs=2,
    ) == pytest.approx((1.0, 0.0, 0.0))
