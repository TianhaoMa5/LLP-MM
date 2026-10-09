from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mo_matching.core.FFT import compute_CC_loss_fft_precise_batched  # noqa: E402
from mo_matching.core.algorithms import LLP_PVC  # noqa: E402
from mo_matching.data.natural_loss import natural_bag_loss  # noqa: E402


def _algorithm(*, bag_size: int) -> LLP_PVC:
    algorithm = LLP_PVC.__new__(LLP_PVC)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    return algorithm


def _legacy_loss(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    number_of_bags = len(logits) // bag_size
    classes = logits.shape[1]
    grouped = probabilities.view(number_of_bags, bag_size, classes)
    return compute_CC_loss_fft_precise_batched(grouped, proportions, reduce="mean")


def _variable_expected(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(logits)
    chunks = probabilities.split(sizes.tolist())
    losses = torch.stack(
        [
            compute_CC_loss_fft_precise_batched(
                chunk[None, :, :],
                proportions[index : index + 1],
                reduce=None,
            )[0]
            for index, chunk in enumerate(chunks)
        ]
    )
    weights = sizes.to(losses)
    return (weights * losses).sum() / weights.sum(), losses


def test_pvc_fixed_size_matches_legacy_loss() -> None:
    torch.manual_seed(83)
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
    algorithm = _algorithm(bag_size=bag_size)

    legacy = _legacy_loss(logits, proportions, bag_size)
    implicit_fixed = algorithm.PVC_Loss(logits, proportions)
    explicit_fixed = algorithm.PVC_Loss(
        logits,
        proportions,
        bag_sizes=torch.full((4,), bag_size),
    )

    torch.testing.assert_close(implicit_fixed, legacy)
    torch.testing.assert_close(explicit_fixed, legacy)


def test_pvc_variable_sizes_use_each_count_and_instance_weighted_mean() -> None:
    torch.manual_seed(89)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )
    logits = torch.randn(int(sizes.sum()), 3, dtype=torch.float64, requires_grad=True)
    algorithm = _algorithm(bag_size=999)

    actual = algorithm.PVC_Loss(logits, proportions, bag_sizes=sizes)
    expected, per_bag = _variable_expected(logits, proportions, sizes)

    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, per_bag.mean())
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_pvc_bag_index_is_instance_permutation_invariant() -> None:
    torch.manual_seed(97)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3], [0.2, 0.8], [0.75, 0.25]], dtype=torch.float64
    )
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    logits = torch.randn(int(sizes.sum()), 2, dtype=torch.float64)
    permutation = torch.randperm(len(logits))
    algorithm = _algorithm(bag_size=3)

    contiguous = algorithm.PVC_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index,
    )
    permuted = algorithm.PVC_Loss(
        logits[permutation],
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index[permutation],
    )

    torch.testing.assert_close(permuted, contiguous)


def test_pvc_supports_singleton_and_single_bag() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    proportions = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    algorithm = _algorithm(bag_size=1)

    actual = algorithm.PVC_Loss(logits, proportions, bag_sizes=torch.tensor([1]))
    expected, _ = _variable_expected(logits, proportions, torch.tensor([1]))

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)


def test_pvc_rejects_inconsistent_bag_metadata_and_padding_rows() -> None:
    algorithm = _algorithm(bag_size=2)
    proportions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        algorithm.PVC_Loss(
            torch.randn(5, 2),
            proportions,
            bag_sizes=torch.tensor([2, 2]),
        )

    with pytest.raises(ValueError, match="do not match"):
        algorithm.PVC_Loss(
            torch.randn(5, 2),
            proportions,
            bag_sizes=torch.tensor([2, 3]),
            bag_index=torch.tensor([0, 0, 0, 1, 1]),
        )


def test_natural_bag_adapter_delegates_to_pvc_loss() -> None:
    torch.manual_seed(101)
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

    direct = algorithm.PVC_Loss(
        logits,
        proportions,
        bag_sizes=sizes,
        bag_index=bag_index,
    )
    adapted = natural_bag_loss(algorithm, "LLP_PVC", batch, logits=logits)

    torch.testing.assert_close(adapted, direct)
