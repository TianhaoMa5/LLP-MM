import math

import pytest
import torch

from plench.core.gaussian_count import gaussian_count_nll


def test_binary_gaussian_count_nll_matches_closed_form():
    probabilities = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    proportions = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    floor = 1.0 / 12.0
    actual = gaussian_count_nll(logits, proportions, bag_size=2, count_variance_floor=floor)
    variance = 0.7 * 0.3 + 0.4 * 0.6 + floor
    expected = 0.5 * (0.1**2 / variance + math.log(variance) + math.log(2 * math.pi))
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_multiclass_gaussian_count_nll_has_finite_gradients():
    logits = torch.randn(2 * 16, 8, requires_grad=True)
    proportions = torch.tensor(
        [[8, 4, 2, 1, 1, 0, 0, 0], [0, 1, 1, 2, 4, 4, 2, 2]],
        dtype=torch.float64,
    ) / 16
    loss = gaussian_count_nll(logits, proportions, bag_size=16)
    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_gaussian_count_nll_rejects_invalid_bag_layout_and_proportions():
    logits = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="B \\* bag_size"):
        gaussian_count_nll(logits, torch.tensor([[0.5, 0.5, 0.0]]), bag_size=2)
    with pytest.raises(ValueError, match="sum to one"):
        gaussian_count_nll(logits, torch.tensor([[0.5, 0.25, 0.0]]), bag_size=3)


def test_gaussian_algorithm_is_registered():
    from plench.core import algorithms, hparams_registry

    assert algorithms.get_algorithm_class("LLP_Gaussian") is algorithms.LLP_Gaussian
    hparams = hparams_registry.default_hparams("LLP_Gaussian", "miniImageNet")
    assert hparams["gaussian_count_variance_floor"] == pytest.approx(1 / 12)

    hparams.update(model="MLP", optimizer="SGD", lr=0.01, momentum=0.9)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
    algorithm = algorithms.LLP_Gaussian(3, (8, 4), targets.tolist(), hparams, 4)
    result = algorithm.update((torch.randn(8, 4), targets))
    assert math.isfinite(result["loss"])


def test_microbatched_gaussian_update_matches_full_logical_batch():
    from plench.core import algorithms

    targets = torch.tensor(
        [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25],
         [0.0, 0.75, 0.25], [0.25, 0.25, 0.5]]
    )
    images = torch.randn(16, 4)
    hparams = {
        "model": "Linear", "optimizer": "SGD", "lr": 0.01,
        "momentum": 0.9, "weight_decay": 0.0,
    }
    full = algorithms.LLP_Gaussian(10, (8, 4), targets.tolist(), hparams, 4)
    chunked = algorithms.LLP_Gaussian(
        10, (8, 4), targets.tolist(),
        {**hparams, "gaussian_bags_per_microbatch": 2}, 4,
    )
    chunked.load_state_dict(full.state_dict())
    full_result = full.update((images, targets))
    chunked_result = chunked.update((images, targets))
    assert chunked_result["loss"] == pytest.approx(full_result["loss"], abs=1e-7)
    for full_parameter, chunked_parameter in zip(full.network.parameters(), chunked.network.parameters()):
        torch.testing.assert_close(chunked_parameter, full_parameter, atol=1e-7, rtol=1e-7)
