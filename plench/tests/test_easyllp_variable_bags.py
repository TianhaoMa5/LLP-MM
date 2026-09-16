from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core.algorithms import EasyLLP  # noqa: E402
from plench.data.ref2021 import ref2021_loss  # noqa: E402


def _algorithm(*, bag_size: int, loss_type: str = "ce") -> EasyLLP:
    algorithm = EasyLLP.__new__(EasyLLP)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.loss_type = loss_type
    return algorithm


def _legacy_easyllp_loss(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
    loss_type: str,
) -> torch.Tensor:
    prior = proportions.mean(dim=0)
    bag_weight = proportions * bag_size - (bag_size - 1) * prior
    instance_weight = bag_weight.repeat_interleave(bag_size, dim=0)
    log_probabilities = F.log_softmax(logits, dim=1)
    if loss_type == "ce":
        return -(instance_weight * log_probabilities).sum(dim=1).mean()
    return ((1.0 - log_probabilities.exp()).square() * instance_weight).mean()


@pytest.mark.parametrize("loss_type", ["ce", "se"])
def test_easyllp_fixed_size_matches_legacy_loss(loss_type: str) -> None:
    torch.manual_seed(7)
    bag_size = 4
    proportions = torch.tensor(
        [[0.50, 0.25, 0.25], [0.00, 0.50, 0.50], [0.25, 0.25, 0.50]],
        dtype=torch.float64,
    )
    logits = torch.randn(3 * bag_size, 3, dtype=torch.float64, requires_grad=True)
    algorithm = _algorithm(bag_size=bag_size, loss_type=loss_type)

    legacy = _legacy_easyllp_loss(logits, proportions, bag_size, loss_type)
    implicit_fixed, _ = algorithm.EasyLLP_Loss(logits, proportions)
    explicit_variable, _ = algorithm.EasyLLP_Loss(
        logits,
        proportions,
        bag_sizes=torch.tensor([bag_size, bag_size, bag_size]),
    )

    torch.testing.assert_close(implicit_fixed, legacy)
    torch.testing.assert_close(explicit_variable, legacy)


def test_easyllp_variable_sizes_use_per_bag_correction_and_instance_mean() -> None:
    torch.manual_seed(11)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )
    logits = torch.randn(int(sizes.sum()), 3, dtype=torch.float64, requires_grad=True)
    algorithm = _algorithm(bag_size=999)

    actual, _ = algorithm.EasyLLP_Loss(
        logits, proportions, bag_sizes=sizes
    )

    sizes_float = sizes.to(proportions)
    prior = (sizes_float[:, None] * proportions).sum(dim=0) / sizes_float.sum()
    expected_bag_weights = (
        sizes_float[:, None] * proportions
        - (sizes_float[:, None] - 1.0) * prior
    )
    expected_instance_weights = expected_bag_weights.repeat_interleave(sizes, dim=0)
    expected = -(
        expected_instance_weights * F.log_softmax(logits, dim=1)
    ).sum(dim=1).mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_easyllp_bag_index_is_permutation_invariant() -> None:
    torch.manual_seed(19)
    sizes = torch.tensor([1, 3, 2])
    proportions = torch.tensor(
        [[0.0, 1.0], [1 / 3, 2 / 3], [0.5, 0.5]], dtype=torch.float64
    )
    contiguous_index = torch.repeat_interleave(torch.arange(3), sizes)
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    permutation = torch.tensor([4, 0, 2, 5, 1, 3])
    algorithm = _algorithm(bag_size=1)

    contiguous, _ = algorithm.EasyLLP_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        bag_index=contiguous_index,
    )
    permuted, _ = algorithm.EasyLLP_Loss(
        logits[permutation],
        proportions,
        bag_sizes=sizes,
        bag_index=contiguous_index[permutation],
    )

    torch.testing.assert_close(permuted, contiguous)


def test_easyllp_singleton_and_single_bag_are_finite() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    proportions = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    algorithm = _algorithm(bag_size=1)

    loss, _ = algorithm.EasyLLP_Loss(
        logits, proportions, bag_sizes=torch.tensor([1])
    )
    expected = F.cross_entropy(logits, torch.tensor([1]))

    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)


def test_easyllp_rejects_inconsistent_bag_metadata() -> None:
    algorithm = _algorithm(bag_size=2)
    logits = torch.randn(5, 2)
    proportions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        algorithm.EasyLLP_Loss(
            logits, proportions, bag_sizes=torch.tensor([2, 2])
        )

    with pytest.raises(ValueError, match="do not match"):
        algorithm.EasyLLP_Loss(
            logits,
            proportions,
            bag_sizes=torch.tensor([2, 3]),
            bag_index=torch.tensor([0, 0, 0, 1, 1]),
        )


def test_natural_bag_adapter_delegates_to_easyllp_loss() -> None:
    torch.manual_seed(23)
    sizes = torch.tensor([3, 5, 8])
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]]
    )
    logits = torch.randn(int(sizes.sum()), 3)
    algorithm = _algorithm(bag_size=32)
    batch = {
        "proportion": proportions,
        "bag_sizes": sizes,
        "bag_index": bag_index,
        "instance_weights": torch.ones(int(sizes.sum())),
    }

    direct, _ = algorithm.EasyLLP_Loss(
        logits, proportions, bag_sizes=sizes, bag_index=bag_index
    )
    adapted = ref2021_loss(
        algorithm, "EasyLLP", batch, logits=logits
    )

    torch.testing.assert_close(adapted, direct)
