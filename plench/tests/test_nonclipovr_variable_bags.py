from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core.algorithms import NonClipOVR  # noqa: E402
from plench.data.ref2021 import ref2021_loss  # noqa: E402


def _algorithm(*, bag_size: int, num_classes: int) -> NonClipOVR:
    algorithm = NonClipOVR.__new__(NonClipOVR)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.num_classes = num_classes
    algorithm.detach_global = False
    algorithm.eps = 1e-12
    algorithm.register_buffer(
        "_nonclip_dataset_prior", torch.empty(0), persistent=False
    )
    return algorithm


def _legacy_loss(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    number_of_bags = len(logits) // bag_size
    classes = logits.shape[1]
    bag_sum = probabilities.view(number_of_bags, bag_size, classes).sum(dim=1)
    prior = proportions.mean(dim=0)
    prediction_mean = probabilities.mean(dim=0)
    residual = (
        bag_size * (proportions - prior[None, :])
        - (bag_sum - bag_size * prediction_mean[None, :])
    )
    return (
        (residual.square() / bag_size).mean()
        + (prediction_mean - prior).square().mean()
    ) * classes


def _paper_aligned_expected(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    sizes: torch.Tensor,
    *,
    prior: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    chunks = probabilities.split(sizes.tolist())
    bag_sums = torch.stack([chunk.sum(dim=0) for chunk in chunks])
    sizes_float = sizes.to(probabilities)
    bag_means = bag_sums / sizes_float[:, None]
    if prior is None:
        prior = (
            sizes_float[:, None] * proportions
        ).sum(dim=0) / sizes_float.sum()
    total_sum = bag_sums.sum(dim=0)
    outside_sizes = sizes_float.sum() - sizes_float
    leave_one_out = (
        total_sum[None, :] - bag_sums
    ) / outside_sizes[:, None]
    offset = leave_one_out - prior[None, :]
    return (
        sizes_float
        * (proportions - bag_means + offset).square().sum(dim=1)
        + offset.square().sum(dim=1)
    ).mean()


def test_nonclipovr_fixed_size_legacy_mode_is_exactly_compatible() -> None:
    torch.manual_seed(53)
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

    legacy = _legacy_loss(logits, proportions, bag_size)
    compatible_fixed = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=torch.full((4,), bag_size),
        leave_one_bag_out=False,
    )

    torch.testing.assert_close(compatible_fixed, legacy)


def test_nonclipovr_fixed_size_paper_loo_formula() -> None:
    torch.manual_seed(59)
    bag_size = 8
    sizes = torch.full((4,), bag_size)
    proportions = torch.tensor(
        [
            [0.50, 0.50],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.00, 1.00],
        ],
        dtype=torch.float64,
    )
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    algorithm = _algorithm(bag_size=bag_size, num_classes=2)

    actual = algorithm.NonClipOVR_Loss(logits, proportions)
    expected = _paper_aligned_expected(logits, proportions, sizes)

    torch.testing.assert_close(actual, expected)


def test_nonclipovr_variable_sizes_use_each_k_and_pooled_loo() -> None:
    torch.manual_seed(61)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )
    logits = torch.randn(
        int(sizes.sum()), 3, dtype=torch.float64, requires_grad=True
    )
    algorithm = _algorithm(bag_size=999, num_classes=3)

    actual = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
    )
    expected = _paper_aligned_expected(logits, proportions, sizes)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_nonclipovr_prior_is_pooled_by_instance_count() -> None:
    sizes = torch.tensor([2, 8])
    proportions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    logits = torch.tensor(
        [[1.0, -1.0], [-0.5, 0.5]] + [[0.2, -0.2]] * 8,
        dtype=torch.float64,
    )
    algorithm = _algorithm(bag_size=2, num_classes=2)

    actual = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        leave_one_bag_out=True,
    )
    pooled_expected = _paper_aligned_expected(
        logits,
        proportions,
        sizes,
        prior=torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    equal_bag_expected = _paper_aligned_expected(
        logits,
        proportions,
        sizes,
        prior=torch.tensor([0.5, 0.5], dtype=torch.float64),
    )

    torch.testing.assert_close(actual, pooled_expected)
    assert not torch.isclose(actual, equal_bag_expected)


def test_nonclipovr_uses_configured_training_prior() -> None:
    torch.manual_seed(67)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3], [0.2, 0.8], [0.75, 0.25]], dtype=torch.float64
    )
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    algorithm = _algorithm(bag_size=3, num_classes=2)
    dataset_prior = torch.tensor([0.35, 0.65])
    algorithm.configure_dataset_prior(dataset_prior)

    actual = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        leave_one_bag_out=True,
    )
    expected = _paper_aligned_expected(
        logits,
        proportions,
        sizes,
        prior=dataset_prior.to(proportions),
    )

    torch.testing.assert_close(actual, expected)


def test_nonclipovr_leave_one_out_is_instance_weighted() -> None:
    sizes = torch.tensor([2, 8, 10])
    first_coordinate_means = [0.1, 0.5, 0.9]
    probabilities = torch.cat(
        [
            torch.tensor([[mean, 0.4]]).repeat(int(size), 1)
            for mean, size in zip(first_coordinate_means, sizes.tolist())
        ],
        dim=0,
    ).to(torch.float64)
    logits = torch.logit(probabilities)
    proportions = torch.tensor(
        [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]], dtype=torch.float64
    )
    algorithm = _algorithm(bag_size=2, num_classes=2)

    actual = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        leave_one_bag_out=True,
    )
    expected = _paper_aligned_expected(logits, proportions, sizes)
    torch.testing.assert_close(actual, expected)

    correct_mu_minus_first = (8 * 0.5 + 10 * 0.9) / 18
    equal_bag_mu_minus_first = (0.5 + 0.9) / 2
    assert correct_mu_minus_first == pytest.approx(13 / 18)
    assert correct_mu_minus_first != pytest.approx(equal_bag_mu_minus_first)


def test_nonclipovr_bag_index_is_instance_permutation_invariant() -> None:
    torch.manual_seed(71)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3], [0.2, 0.8], [0.75, 0.25]], dtype=torch.float64
    )
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    permutation = torch.randperm(len(logits))
    algorithm = _algorithm(bag_size=3, num_classes=2)

    contiguous = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index,
        leave_one_bag_out=True,
    )
    permuted = algorithm.NonClipOVR_Loss(
        logits[permutation],
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index[permutation],
        leave_one_bag_out=True,
    )

    torch.testing.assert_close(permuted, contiguous)


def test_nonclipovr_rejects_padding_rows_not_counted_as_valid() -> None:
    algorithm = _algorithm(bag_size=5, num_classes=2)
    proportions = torch.tensor([[0.4, 0.6], [0.6, 0.4]])
    real_logits = torch.randn(10, 2)
    padded_logits = torch.cat([real_logits, torch.zeros(6, 2)], dim=0)

    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        algorithm.NonClipOVR_Loss(
            padded_logits,
            proportions,
            bag_sizes=torch.tensor([5, 5]),
            leave_one_bag_out=True,
        )


def test_nonclipovr_single_bag_loo_has_clear_error() -> None:
    algorithm = _algorithm(bag_size=5, num_classes=2)
    with pytest.raises(ValueError, match="at least 2 bags"):
        algorithm.NonClipOVR_Loss(
            torch.randn(5, 2),
            torch.tensor([[0.4, 0.6]]),
            bag_sizes=torch.tensor([5]),
            leave_one_bag_out=True,
        )


def test_natural_bag_adapter_delegates_to_nonclipovr_loss() -> None:
    torch.manual_seed(73)
    sizes = torch.tensor([3, 5, 8])
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]]
    )
    logits = torch.randn(int(sizes.sum()), 3)
    algorithm = _algorithm(bag_size=32, num_classes=3)
    batch = {
        "proportion": proportions,
        "bag_sizes": sizes,
        "bag_index": bag_index,
        "instance_weights": torch.ones(int(sizes.sum())),
    }

    direct = algorithm.NonClipOVR_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index,
        leave_one_bag_out=True,
    )
    adapted = ref2021_loss(
        algorithm, "NonClipOVR", batch, logits=logits
    )

    torch.testing.assert_close(adapted, direct)
