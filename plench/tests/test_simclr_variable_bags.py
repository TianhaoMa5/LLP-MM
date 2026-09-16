from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core.algorithms import LLP_SimCLR  # noqa: E402
from plench.data.ku_optofil_pbc import (  # noqa: E402
    SUPPORTED_KU_METHODS,
    update_ku_optofil_algorithm,
)


class _ToyNetwork(torch.nn.Module):
    def __init__(self, feature_dimension: int, classes: int) -> None:
        super().__init__()
        self.featurizer = torch.nn.Identity()
        self.classifier = torch.nn.Linear(feature_dimension, classes)

    def forward(self, values: torch.Tensor):
        features = self.featurizer(values)
        return self.classifier(features), features


class _CounterScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _algorithm(
    bag_size: int,
    *,
    classes: int = 3,
    feature_dimension: int = 4,
) -> LLP_SimCLR:
    algorithm = LLP_SimCLR.__new__(LLP_SimCLR)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.num_classes = classes
    algorithm.entropy_weight = 0.17
    algorithm.match_weight = 0.4
    algorithm.match_threshold = 0.2
    algorithm.match_min_ratio = 0.4
    algorithm._eps = 1e-12
    algorithm.network = _ToyNetwork(feature_dimension, classes)
    algorithm.featurizer = algorithm.network.featurizer
    algorithm.classifier = algorithm.network.classifier
    algorithm.optimizer = torch.optim.SGD(algorithm.network.parameters(), lr=0.05)
    algorithm.scheduler = _CounterScheduler()
    return algorithm


def _legacy_stage2(
    algorithm: LLP_SimCLR,
    proportions: torch.Tensor,
    probabilities: torch.Tensor,
    features: torch.Tensor,
    bags: int,
    bag_size: int,
    classes: int,
) -> torch.Tensor:
    match_total = probabilities.new_tensor(0.0)
    predicted = probabilities.argmax(dim=1).view(bags, bag_size)
    feature_bags = features.view(bags, bag_size, -1)
    minimum = max(1, int(math.ceil(bag_size * algorithm.match_min_ratio)))
    for bag in range(bags):
        candidates = torch.nonzero(
            proportions[bag] > algorithm.match_threshold, as_tuple=False
        ).squeeze(1)
        for label in candidates.tolist():
            selected = torch.nonzero(
                predicted[bag] == label, as_tuple=False
            ).squeeze(1)
            if selected.numel() < minimum:
                continue
            logits = algorithm.classifier(feature_bags[bag, selected].detach())
            targets = torch.full(
                (len(logits),), label, dtype=torch.long, device=logits.device
            )
            nll = F.nll_loss(F.log_softmax(logits, dim=1), targets)
            algorithm.optimizer.zero_grad()
            (algorithm.match_weight * nll).backward()
            algorithm.optimizer.step()
            algorithm.scheduler.step()
            match_total = match_total + nll.detach()
    return algorithm.match_weight * match_total / bags


def test_simclr_fixed_stage1_matches_legacy_formula() -> None:
    torch.manual_seed(181)
    size = 4
    proportions = torch.tensor(
        [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5]],
        dtype=torch.float64,
    )
    logits = torch.randn(3 * size, 3, dtype=torch.float64)
    probabilities = logits.softmax(dim=1).clamp_min(1e-12)
    algorithm = _algorithm(size)

    normalized = proportions / proportions.sum(dim=1, keepdim=True)
    prediction = probabilities.view(3, size, 3).mean(dim=1).clamp_min(1e-12)
    expected = F.kl_div(prediction.log(), normalized, reduction="batchmean")
    expected = expected + algorithm.entropy_weight * (
        -(probabilities * probabilities.log()).sum(dim=1).mean()
    )
    actual = algorithm.stage1_loss(proportions, probabilities, 3, size, 3)

    torch.testing.assert_close(actual, expected)


def test_simclr_fixed_stage2_matches_legacy_updates() -> None:
    torch.manual_seed(191)
    bags, size, classes = 3, 4, 3
    proportions = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]
    )
    probabilities = torch.randn(bags * size, classes).softmax(dim=1)
    features = torch.randn(bags * size, 4)
    legacy = _algorithm(size)
    variable = _algorithm(size)
    variable.network.load_state_dict(legacy.network.state_dict())

    expected = _legacy_stage2(
        legacy, proportions, probabilities, features, bags, size, classes
    )
    actual = variable.stage2_loss(
        proportions,
        probabilities,
        features,
        bags,
        size,
        classes,
        probabilities.device,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        variable.classifier.weight, legacy.classifier.weight
    )
    torch.testing.assert_close(variable.classifier.bias, legacy.classifier.bias)
    assert variable.scheduler.steps == legacy.scheduler.steps


def test_simclr_variable_stage1_uses_real_bag_means() -> None:
    torch.manual_seed(193)
    sizes = torch.tensor([3, 5, 8])
    proportions = torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )
    logits = torch.randn(int(sizes.sum()), 3, dtype=torch.float64)
    probabilities = logits.softmax(dim=1).clamp_min(1e-12)
    index = torch.repeat_interleave(torch.arange(3), sizes)
    algorithm = _algorithm(999)
    actual = algorithm.stage1_loss(
        proportions,
        probabilities,
        3,
        999,
        3,
        bag_sizes=sizes,
        bag_index=index,
    )

    predictions = torch.stack(
        [chunk.mean(dim=0) for chunk in probabilities.split(sizes.tolist())]
    )
    normalized = proportions / proportions.sum(dim=1, keepdim=True)
    expected = F.kl_div(
        predictions.clamp_min(1e-12).log(), normalized, reduction="batchmean"
    ) + algorithm.entropy_weight * (
        -(probabilities * probabilities.log()).sum(dim=1).mean()
    )
    torch.testing.assert_close(actual, expected)


def test_simclr_stage2_uses_each_bag_size_for_match_threshold() -> None:
    sizes = torch.tensor([3, 5, 8])
    index = torch.repeat_interleave(torch.arange(3), sizes)
    algorithm = _algorithm(999, classes=2)
    algorithm.match_threshold = 0.5
    algorithm.match_min_ratio = 0.5
    proportions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    # Predicted class-0 counts are 2, 2, and 4.  Per-bag minima are
    # ceil(3/2)=2, ceil(5/2)=3, and ceil(8/2)=4, so only bags 0 and 2 update.
    predicted_labels = torch.tensor(
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    )
    probabilities = F.one_hot(predicted_labels, num_classes=2).float() * 0.8 + 0.1
    features = torch.randn(int(sizes.sum()), 4)

    loss = algorithm.stage2_loss(
        proportions,
        probabilities,
        features,
        3,
        999,
        2,
        probabilities.device,
        bag_sizes=sizes,
        bag_index=index,
    )

    assert torch.isfinite(loss)
    assert algorithm.scheduler.steps == 2


def test_simclr_variable_stage_losses_are_permutation_invariant() -> None:
    torch.manual_seed(197)
    sizes = torch.tensor([3, 5, 8])
    index = torch.repeat_interleave(torch.arange(3), sizes)
    proportions = torch.tensor(
        [[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]]
    )
    probabilities = torch.randn(int(sizes.sum()), 3).softmax(dim=1)
    permutation = torch.randperm(len(probabilities))
    algorithm = _algorithm(999)

    contiguous = algorithm.stage1_loss(
        proportions,
        probabilities,
        3,
        999,
        3,
        bag_sizes=sizes,
        bag_index=index,
    )
    shuffled = algorithm.stage1_loss(
        proportions,
        probabilities[permutation],
        3,
        999,
        3,
        bag_sizes=sizes,
        bag_index=index[permutation],
    )
    torch.testing.assert_close(shuffled, contiguous)


def test_ku_simclr_adapter_runs_chunked_two_stage_update() -> None:
    torch.manual_seed(199)
    sizes = torch.tensor([3, 5, 8])
    total = int(sizes.sum())
    algorithm = _algorithm(999)
    algorithm.match_threshold = 2.0  # isolate the stage-1 backward in this smoke test
    batch = {
        "x": torch.randn(total, 4),
        "bag_index": torch.repeat_interleave(torch.arange(3), sizes),
        "bag_sizes": sizes,
        "proportion": torch.tensor(
            [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]]
        ),
        "instance_weights": torch.ones(total),
    }

    result = update_ku_optofil_algorithm(
        algorithm,
        "LLP_SimCLR",
        batch,
        "cpu",
        forward_chunk_size=4,
    )

    assert "LLP_SimCLR" in SUPPORTED_KU_METHODS
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert algorithm.classifier.weight.grad is not None
    assert algorithm.scheduler.steps == 1


def test_simclr_rejects_padding_rows_and_wrong_feature_count() -> None:
    algorithm = _algorithm(2)
    proportions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        algorithm.stage1_loss(
            proportions,
            torch.randn(5, 2).softmax(dim=1),
            2,
            2,
            2,
            bag_sizes=torch.tensor([2, 2]),
        )
    with pytest.raises(ValueError, match="features must have shape"):
        algorithm.update_from_outputs(
            torch.randn(4, 2, requires_grad=True),
            torch.randn(3, 4),
            proportions,
        )
