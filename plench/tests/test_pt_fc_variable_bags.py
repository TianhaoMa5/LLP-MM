from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core.algorithms import LLP_FC, LLP_PT  # noqa: E402
from plench.data.ref2021 import ref2021_loss  # noqa: E402


def _hard_argmax(probs: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
    return F.one_hot(probs.argmax(dim=1), num_classes=probs.shape[1]).to(probs)


def _pt(*, bag_size: int) -> LLP_PT:
    algorithm = LLP_PT.__new__(LLP_PT)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm._eps = 1e-12
    algorithm.emd_hard_assign = _hard_argmax
    return algorithm


def _fc(*, bag_size: int, mode: str = "uniform") -> LLP_FC:
    algorithm = LLP_FC.__new__(LLP_FC)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.mode = mode
    algorithm.group_weight = 0.7
    algorithm.entropy_weight = 0.13
    algorithm.pi_grad_steps = 10
    algorithm.pi_grad_lr = 0.1
    algorithm._eps = 1e-12
    return algorithm


def _legacy_pt_loss(
    logits: torch.Tensor, proportions: torch.Tensor, bag_size: int
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1).clamp_min(1e-12)
    bags = len(logits) // bag_size
    probability_bags = probabilities.view(bags, bag_size, -1)
    logit_bags = logits.view(bags, bag_size, -1)
    ce_losses = []
    rce_losses = []
    for bag in range(bags):
        soft_targets = _hard_argmax(probability_bags[bag], proportions[bag])
        ce_losses.append(
            -(soft_targets * F.log_softmax(logit_bags[bag], dim=1))
            .sum(dim=1)
            .mean()
        )
        rce_losses.append(
            -(
                probability_bags[bag]
                * soft_targets.clamp_min(1e-12).log()
            )
            .sum(dim=1)
            .mean()
        )
    return torch.stack(ce_losses).mean() + torch.stack(rce_losses).mean()


def _variable_pt_expected(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1).clamp_min(1e-12)
    probability_bags = probabilities.split(sizes.tolist())
    logit_bags = logits.split(sizes.tolist())
    ce_losses = []
    rce_losses = []
    for bag, (bag_probs, bag_logits) in enumerate(
        zip(probability_bags, logit_bags)
    ):
        soft_targets = _hard_argmax(bag_probs, proportions[bag])
        ce_losses.append(
            -(soft_targets * F.log_softmax(bag_logits, dim=1))
            .sum(dim=1)
            .mean()
        )
        rce_losses.append(
            -(bag_probs * soft_targets.clamp_min(1e-12).log())
            .sum(dim=1)
            .mean()
        )
    return torch.stack(ce_losses).mean() + torch.stack(rce_losses).mean()


def _legacy_fc_loss(
    algorithm: LLP_FC,
    logits: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> torch.Tensor:
    eps = algorithm._eps
    probabilities = logits.softmax(dim=1).clamp_min(eps)
    bags, classes = len(proportions), logits.shape[1]
    groups = bags // classes
    if groups == 0:
        prediction = probabilities.view(bags, bag_size, classes).mean(dim=1)
        target = proportions.clamp_min(eps)
        target = target / target.sum(dim=1, keepdim=True)
        kl = F.kl_div(prediction.clamp_min(eps).log(), target, reduction="batchmean")
        entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
        return kl + algorithm.entropy_weight * entropy

    permutation = torch.randperm(bags)[: groups * classes].view(groups, classes)
    bag_probabilities = probabilities.view(bags, bag_size, classes)
    total = logits.new_tensor(0.0)
    for group in range(groups):
        bag_ids = permutation[group]
        pi_matrix = proportions[bag_ids]
        alpha = torch.full(
            (classes,),
            1.0 / classes,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        eta = (pi_matrix.t() @ alpha).clamp_min(eps)
        transition = pi_matrix * alpha[:, None]
        transition = transition / eta[None, :]
        transition = transition / transition.sum(dim=0, keepdim=True).clamp_min(eps)
        for noisy_class in range(classes):
            noisy = (
                bag_probabilities[bag_ids[noisy_class]] @ transition.t()
            ).clamp_min(eps)
            total = total - noisy[:, noisy_class].log().mean()
    fc = total / (groups * classes)
    entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
    return algorithm.group_weight * fc + algorithm.entropy_weight * entropy


def _variable_fc_expected(
    algorithm: LLP_FC,
    logits: torch.Tensor,
    proportions: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    eps = algorithm._eps
    probabilities = logits.softmax(dim=1).clamp_min(eps)
    bags, classes = len(proportions), logits.shape[1]
    groups = bags // classes
    permutation = torch.randperm(bags)[: groups * classes].view(groups, classes)
    bag_probabilities = probabilities.split(sizes.tolist())
    total = logits.new_tensor(0.0)
    for group in range(groups):
        bag_ids = permutation[group]
        pi_matrix = proportions[bag_ids]
        alpha = torch.full(
            (classes,),
            1.0 / classes,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        eta = (pi_matrix.t() @ alpha).clamp_min(eps)
        transition = pi_matrix * alpha[:, None]
        transition = transition / eta[None, :]
        transition = transition / transition.sum(dim=0, keepdim=True).clamp_min(eps)
        for noisy_class in range(classes):
            noisy = (
                bag_probabilities[int(bag_ids[noisy_class].item())]
                @ transition.t()
            ).clamp_min(eps)
            total = total - noisy[:, noisy_class].log().mean()
    fc = total / (groups * classes)
    entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
    return algorithm.group_weight * fc + algorithm.entropy_weight * entropy


def test_pt_fixed_size_matches_legacy_loss() -> None:
    torch.manual_seed(109)
    bag_size = 4
    proportions = torch.tensor(
        [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5]],
        dtype=torch.float64,
    )
    logits = torch.randn(3 * bag_size, 3, dtype=torch.float64)
    algorithm = _pt(bag_size=bag_size)

    expected = _legacy_pt_loss(logits, proportions, bag_size)
    implicit = algorithm.PT_Loss(logits, proportions)
    explicit = algorithm.PT_Loss(
        logits, proportions, bag_sizes=torch.full((3,), bag_size)
    )

    torch.testing.assert_close(implicit, expected)
    torch.testing.assert_close(explicit, expected)


def test_pt_variable_sizes_use_each_complete_natural_bag() -> None:
    torch.manual_seed(113)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )
    logits = torch.randn(int(sizes.sum()), 3, dtype=torch.float64, requires_grad=True)
    algorithm = _pt(bag_size=999)

    actual = algorithm.PT_Loss(logits, proportions, bag_sizes=sizes)
    expected = _variable_pt_expected(logits, proportions, sizes)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_fc_fixed_size_matches_legacy_loss() -> None:
    torch.manual_seed(127)
    bag_size = 4
    proportions = torch.tensor(
        [
            [0.50, 0.25, 0.25],
            [0.00, 0.50, 0.50],
            [0.25, 0.25, 0.50],
            [0.75, 0.25, 0.00],
            [0.25, 0.50, 0.25],
            [0.00, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    logits = torch.randn(6 * bag_size, 3, dtype=torch.float32)
    algorithm = _fc(bag_size=bag_size)

    torch.manual_seed(131)
    expected = _legacy_fc_loss(algorithm, logits, proportions, bag_size)
    torch.manual_seed(131)
    implicit = algorithm.FC_Loss(logits, proportions)
    torch.manual_seed(131)
    explicit = algorithm.FC_Loss(
        logits, proportions, bag_sizes=torch.full((6,), bag_size)
    )

    torch.testing.assert_close(implicit, expected)
    torch.testing.assert_close(explicit, expected)


def test_fc_variable_sizes_and_instance_pooled_prior() -> None:
    torch.manual_seed(137)
    sizes = torch.tensor([2, 8])
    proportions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = torch.randn(10, 2, requires_grad=True)
    algorithm = _fc(bag_size=999, mode="approx")
    captured_priors = []

    def capture_prior(pi_matrix: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        captured_priors.append(prior.detach().clone())
        return torch.full(
            (pi_matrix.shape[0],),
            1.0 / pi_matrix.shape[0],
            dtype=pi_matrix.dtype,
            device=pi_matrix.device,
        )

    algorithm.estimate_alpha_approx = capture_prior
    torch.manual_seed(139)
    expected = _variable_fc_expected(algorithm, logits, proportions, sizes)
    torch.manual_seed(139)
    loss = algorithm.FC_Loss(logits, proportions, bag_sizes=sizes)

    torch.testing.assert_close(captured_priors[0], torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("method", ["LLP_PT", "LLP_FC"])
def test_natural_bag_adapter_delegates_to_variable_loss(method: str) -> None:
    torch.manual_seed(139)
    sizes = torch.tensor([3, 5, 8])
    bag_index = torch.repeat_interleave(torch.arange(3), sizes)
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]]
    )
    logits = torch.randn(int(sizes.sum()), 3)
    algorithm = _pt(bag_size=32) if method == "LLP_PT" else _fc(bag_size=32)
    batch = {
        "proportion": proportions,
        "bag_sizes": sizes,
        "bag_index": bag_index,
        "instance_weights": torch.ones(int(sizes.sum())),
    }

    torch.manual_seed(149)
    if method == "LLP_PT":
        direct = algorithm.PT_Loss(
            logits, proportions, bag_sizes=sizes, bag_index=bag_index
        )
    else:
        direct = algorithm.FC_Loss(
            logits, proportions, bag_sizes=sizes, bag_index=bag_index
        )
    torch.manual_seed(149)
    adapted = ref2021_loss(algorithm, method, batch, logits=logits)

    torch.testing.assert_close(adapted, direct)


@pytest.mark.parametrize("algorithm_name", ["PT", "FC"])
def test_variable_losses_reject_padding_and_inconsistent_membership(
    algorithm_name: str,
) -> None:
    algorithm = _pt(bag_size=2) if algorithm_name == "PT" else _fc(bag_size=2)
    loss_function = algorithm.PT_Loss if algorithm_name == "PT" else algorithm.FC_Loss
    proportions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        loss_function(
            torch.randn(5, 2), proportions, bag_sizes=torch.tensor([2, 2])
        )
    with pytest.raises(ValueError, match="do not match"):
        loss_function(
            torch.randn(5, 2),
            proportions,
            bag_sizes=torch.tensor([2, 3]),
            bag_index=torch.tensor([0, 0, 0, 1, 1]),
        )
