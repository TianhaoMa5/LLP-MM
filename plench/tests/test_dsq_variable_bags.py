from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core.algorithms import LLP_DSQ  # noqa: E402
from plench.data.ref2021 import ref2021_loss  # noqa: E402


def _algorithm(
    *, bag_size: int, num_classes: int, ema_beta: float = 0.0
) -> LLP_DSQ:
    algorithm = LLP_DSQ.__new__(LLP_DSQ)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.num_classes = num_classes
    algorithm.dsq_ema_beta = ema_beta
    algorithm.register_buffer(
        "_dsq_prediction_ema", torch.zeros(num_classes), persistent=False
    )
    algorithm.register_buffer(
        "_dsq_ema_initialized", torch.tensor(False), persistent=False
    )
    algorithm.register_buffer(
        "_dsq_dataset_prior", torch.empty(0), persistent=False
    )
    algorithm.register_buffer(
        "_dsq_dataset_mean_k_minus_1",
        torch.tensor(float("nan")),
        persistent=False,
    )
    return algorithm


def _legacy_dsq_loss(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    number_of_bags = len(logits) // bag_size
    classes = logits.shape[1]
    predicted = probabilities.view(number_of_bags, bag_size, classes).mean(dim=1)
    term1 = bag_size * (predicted - proportions).square().mean(dim=1).mean()
    correction = (
        probabilities.mean(dim=0) - proportions.mean(dim=0)
    ).square().mean()
    return (term1 - (bag_size - 1) * correction) * classes


def _variable_expected(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    sizes: torch.Tensor,
    *,
    prior: torch.Tensor | None = None,
    mean_k_minus_1: float | None = None,
    correction_prediction_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    chunks = probabilities.split(sizes.tolist())
    predicted = torch.stack([chunk.mean(dim=0) for chunk in chunks])
    sizes_float = sizes.to(proportions)
    term1 = (
        sizes_float * (predicted - proportions).square().mean(dim=1)
    ).mean()
    if prior is None:
        prior = (
            sizes_float[:, None] * proportions
        ).sum(dim=0) / sizes_float.sum()
    if mean_k_minus_1 is None:
        mean_k_minus_1 = float((sizes_float - 1.0).mean())
    if correction_prediction_mean is None:
        correction_prediction_mean = probabilities.mean(dim=0)
    correction = (
        correction_prediction_mean - prior
    ).square().mean()
    return (term1 - mean_k_minus_1 * correction) * logits.shape[1]


def test_dsq_fixed_size_matches_legacy_loss() -> None:
    torch.manual_seed(31)
    bag_size = 8
    proportions = torch.tensor(
        [
            [0.50, 0.25, 0.25],
            [0.00, 0.50, 0.50],
            [0.25, 0.25, 0.50],
            [0.75, 0.25, 0.00],
        ],
        dtype=torch.float64,
    )
    logits = torch.randn(4 * bag_size, 3, dtype=torch.float64)
    algorithm = _algorithm(bag_size=bag_size, num_classes=3)

    legacy = _legacy_dsq_loss(logits, proportions, bag_size)
    implicit_fixed = algorithm.DSQ_Loss(logits, proportions)
    explicit_variable = algorithm.DSQ_Loss(
        logits,
        proportions,
        bag_sizes=torch.full((4,), bag_size),
    )

    torch.testing.assert_close(implicit_fixed, legacy)
    torch.testing.assert_close(explicit_variable, legacy)


def test_dsq_variable_sizes_use_each_k_and_mean_k_minus_one() -> None:
    torch.manual_seed(37)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )
    logits = torch.randn(int(sizes.sum()), 3, dtype=torch.float64, requires_grad=True)
    algorithm = _algorithm(bag_size=999, num_classes=3)

    actual = algorithm.DSQ_Loss(logits, proportions, bag_sizes=sizes)
    expected = _variable_expected(logits, proportions, sizes)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dsq_prior_is_pooled_by_instance_count() -> None:
    sizes = torch.tensor([2, 8])
    proportions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 0.5]]
        + [[0.2, -0.2]] * 4
        + [[-0.1, 0.1]] * 4,
        dtype=torch.float64,
    )
    algorithm = _algorithm(bag_size=2, num_classes=2)

    actual = algorithm.DSQ_Loss(logits, proportions, bag_sizes=sizes)
    pooled_prior = torch.tensor([0.2, 0.8], dtype=torch.float64)
    pooled_expected = _variable_expected(
        logits, proportions, sizes, prior=pooled_prior
    )
    equal_bag_expected = _variable_expected(
        logits,
        proportions,
        sizes,
        prior=torch.tensor([0.5, 0.5], dtype=torch.float64),
    )

    torch.testing.assert_close(actual, pooled_expected)
    assert not torch.isclose(actual, equal_bag_expected)


def test_dsq_uses_configured_dataset_statistics() -> None:
    torch.manual_seed(41)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3], [0.2, 0.8], [0.75, 0.25]], dtype=torch.float64
    )
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    algorithm = _algorithm(bag_size=3, num_classes=2)
    dataset_prior = torch.tensor([0.35, 0.65])
    dataset_mean_k_minus_1 = 12.25
    algorithm.configure_dataset_statistics(
        dataset_prior, dataset_mean_k_minus_1
    )

    actual = algorithm.DSQ_Loss(logits, proportions, bag_sizes=sizes)
    expected = _variable_expected(
        logits,
        proportions,
        sizes,
        prior=dataset_prior.to(proportions),
        mean_k_minus_1=dataset_mean_k_minus_1,
    )

    torch.testing.assert_close(actual, expected)


def test_dsq_ema_uses_pooled_instance_prediction_mean() -> None:
    sizes = torch.tensor([2, 4])
    proportions = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    first_logits = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0]] + [[0.5, -0.5]] * 4,
        requires_grad=True,
    )
    second_logits = torch.tensor(
        [[-0.5, 0.5], [0.25, -0.25]] + [[-1.0, 1.0]] * 4,
        requires_grad=True,
    )
    algorithm = _algorithm(bag_size=2, num_classes=2, ema_beta=0.5)

    first_loss = algorithm.DSQ_Loss(
        first_logits, proportions, bag_sizes=sizes, update_ema=True
    )
    first_mean = F.softmax(first_logits, dim=1).mean(dim=0)
    torch.testing.assert_close(algorithm._dsq_prediction_ema, first_mean.detach())

    second_loss = algorithm.DSQ_Loss(
        second_logits, proportions, bag_sizes=sizes, update_ema=True
    )
    second_mean = F.softmax(second_logits, dim=1).mean(dim=0)
    expected_ema = 0.5 * first_mean.detach() + 0.5 * second_mean
    expected_second = _variable_expected(
        second_logits,
        proportions,
        sizes,
        correction_prediction_mean=expected_ema,
    )

    assert torch.isfinite(first_loss)
    torch.testing.assert_close(second_loss, expected_second)
    torch.testing.assert_close(algorithm._dsq_prediction_ema, expected_ema.detach())
    second_loss.backward()
    assert second_logits.grad is not None
    assert torch.isfinite(second_logits.grad).all()


def test_dsq_singleton_single_bag_is_finite() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    proportions = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    algorithm = _algorithm(bag_size=1, num_classes=3, ema_beta=0.99)

    loss = algorithm.DSQ_Loss(
        logits,
        proportions,
        bag_sizes=torch.tensor([1]),
        update_ema=True,
    )
    expected = (
        F.softmax(logits, dim=1) - proportions
    ).square().sum()

    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)


def test_dsq_bag_index_is_instance_permutation_invariant() -> None:
    torch.manual_seed(43)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3], [0.2, 0.8], [0.75, 0.25]], dtype=torch.float64
    )
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    permutation = torch.randperm(len(logits))
    algorithm = _algorithm(bag_size=3, num_classes=2)

    contiguous = algorithm.DSQ_Loss(
        logits, proportions, bag_sizes=sizes, bag_index=bag_index
    )
    permuted = algorithm.DSQ_Loss(
        logits[permutation],
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index[permutation],
    )

    torch.testing.assert_close(permuted, contiguous)


def test_natural_bag_adapter_delegates_to_dsq_loss() -> None:
    torch.manual_seed(47)
    sizes = torch.tensor([3, 5, 8])
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]]
    )
    logits = torch.randn(int(sizes.sum()), 3)
    algorithm = _algorithm(bag_size=32, num_classes=3)
    algorithm.eval()
    batch = {
        "proportion": proportions,
        "bag_sizes": sizes,
        "bag_index": bag_index,
        "instance_weights": torch.ones(int(sizes.sum())),
    }

    direct = algorithm.DSQ_Loss(
        logits, proportions, bag_sizes=sizes, bag_index=bag_index
    )
    adapted = ref2021_loss(
        algorithm, "LLP_DSQ", batch, logits=logits
    )

    torch.testing.assert_close(adapted, direct)
