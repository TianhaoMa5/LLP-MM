"""Check the recovered historical DP against an independent ordered-sample oracle."""

import itertools

import pytest
import torch

from plench.core.order import LLPHighOrderLoss
from reproduction.llp_gan_pytorch.core import bag_moment_loss


def _ordered_reference(logits, labels, order, tau=1e-4):
    probabilities = logits.softmax(dim=-1)
    classes = logits.shape[-1]
    total = logits.new_zeros(())
    for degree in range(1, order + 1):
        draws = list(itertools.permutations(range(len(labels)), degree))
        loss = logits.new_zeros(())
        for outcome in itertools.product(range(classes), repeat=degree):
            predicted = torch.stack([
                torch.stack([probabilities[i, c] for i, c in zip(draw, outcome)]).prod()
                for draw in draws
            ]).mean()
            target = sum(tuple(labels[i] for i in draw) == outcome for draw in draws) / len(draws)
            loss = loss - target * ((1 - tau) * predicted + tau / classes ** degree).log()
        total = total + loss / order
    return total


@pytest.mark.parametrize("order", [1, 2, 3])
def test_historical_paper_dp_matches_ordered_sampling_values_and_gradients(order):
    torch.manual_seed(17)
    logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    labels = [0, 0, 1, 2]
    proportions = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float64)
    criterion = LLPHighOrderLoss(C=3, max_order=order, bag_size=4)
    actual = criterion(proportions, logits.softmax(dim=-1))
    reference = _ordered_reference(logits, labels, order)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
    actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual_gradient, reference_gradient, atol=1e-11, rtol=1e-11)


def test_gan_historical_paper_dp_chunking_preserves_loss_and_gradient():
    torch.manual_seed(23)
    logits = torch.randn(12, 3, requires_grad=True)
    proportions = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.0, 0.25, 0.75]])
    criterion = LLPHighOrderLoss(C=3, max_order=3, bag_size=4)
    full = bag_moment_loss(logits, proportions, 4, criterion)
    chunked = bag_moment_loss(logits, proportions, 4, criterion, moment_bag_chunk_size=2)
    torch.testing.assert_close(chunked, full)
    full_gradient, = torch.autograd.grad(full, logits, retain_graph=True)
    chunked_gradient, = torch.autograd.grad(chunked, logits)
    assert torch.isfinite(full_gradient).all()
    torch.testing.assert_close(chunked_gradient, full_gradient)
