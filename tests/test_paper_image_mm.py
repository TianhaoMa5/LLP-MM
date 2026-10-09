import torch

from mo_matching.core.algorithms import Algorithm, LLP_MM
from mo_matching.core.order import LLPHighOrderLoss


def test_fixed_image_dispatch_matches_supplement_loss_and_gradient(monkeypatch):
    def initialize(self, epochs, shape, proportions, hparams, bagsize):
        torch.nn.Module.__init__(self)
        self.num_classes = len(proportions[0])
        self.bagsize = bagsize

    monkeypatch.setattr(Algorithm, "__init__", initialize)
    alpha = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]], dtype=torch.float64)
    algorithm = LLP_MM(
        10, (3,), alpha, {"order": 3, "moment_implementation": "paper_image"}, 4
    )
    algorithm.predict = lambda x: x
    algorithm._backward_step = lambda loss: loss
    torch.manual_seed(6)
    logits = torch.randn(8, 3, requires_grad=True)
    actual = algorithm.update((logits, alpha))
    reference = LLPHighOrderLoss(
        C=3,
        max_order=3,
        bag_size=4,
        loss_type="ce",
        weight_mode="uniform",
        reduce="mean",
    )
    expected = reference(alpha, logits.softmax(1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual_grad,) = torch.autograd.grad(actual, logits, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
