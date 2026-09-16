from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.lib.misc import val_DSQ, val_easy  # noqa: E402


class _IdentityNetwork:
    def __init__(self) -> None:
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _SizedIterator:
    def __init__(self, batches):
        self._batches = list(batches)
        self._index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._batches)

    def __next__(self):
        if self._index >= len(self._batches):
            raise StopIteration
        value = self._batches[self._index]
        self._index += 1
        return value


class _Loader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return _SizedIterator(self._batches)


def _batch(bags: list[torch.Tensor], proportions: torch.Tensor):
    class_major = [proportions[:, index].tolist() for index in range(proportions.shape[1])]
    return ((bags,), class_major, None, None, None)


def _hard_predictions(bags: list[torch.Tensor], classes: int) -> torch.Tensor:
    logits = torch.cat(bags, dim=0)
    return F.one_hot(logits.argmax(dim=1), num_classes=classes).float()


def test_easy_evaluator_uses_each_bag_size_and_instance_mean() -> None:
    bags = [
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([[0.0, 2.0]] * 5),
        torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
    ]
    sizes = torch.tensor([2, 5, 3])
    proportions = torch.tensor([[0.5, 0.5], [0.2, 0.8], [2 / 3, 1 / 3]])
    sizes_float = sizes.to(proportions)
    prior = (sizes_float[:, None] * proportions).sum(dim=0) / sizes_float.sum()
    network = _IdentityNetwork()

    actual = val_easy(
        SimpleNamespace(n_classes=2),
        prior.numpy(),
        network,
        _Loader([_batch(bags, proportions)]),
        "cpu",
    )

    bag_weight = (
        sizes_float[:, None] * proportions
        - (sizes_float[:, None] - 1.0) * prior
    )
    instance_weight = bag_weight.repeat_interleave(sizes, dim=0)
    hard = _hard_predictions(bags, 2)
    zero_one = 1.0 - hard
    expected = (instance_weight * zero_one).sum(dim=1).mean().item()

    assert actual == pytest.approx(expected)
    assert network.training


def test_easy_evaluator_fixed_size_matches_legacy_bag_mean() -> None:
    bags = [torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)]
    proportions = torch.tensor(
        [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5]]
    )
    prior = proportions.mean(dim=0)
    actual = val_easy(
        SimpleNamespace(n_classes=3),
        prior.numpy(),
        _IdentityNetwork(),
        _Loader([_batch(bags, proportions)]),
        "cpu",
    )

    bag_weight = 4 * proportions - 3 * prior
    instance_weight = bag_weight.repeat_interleave(4, dim=0)
    zero_one = 1.0 - _hard_predictions(bags, 3)
    per_instance = (instance_weight * zero_one).sum(dim=1)
    legacy = per_instance.view(3, 4).mean(dim=1).mean().item()

    assert actual == pytest.approx(legacy)


def _expected_dsq(
    bags: list[torch.Tensor], proportions: torch.Tensor
) -> float:
    classes = proportions.shape[1]
    sizes = torch.tensor([len(bag) for bag in bags], dtype=torch.float32)
    hard_chunks = [
        F.one_hot(bag.argmax(dim=1), num_classes=classes).float()
        for bag in bags
    ]
    predicted = torch.stack([chunk.mean(dim=0) for chunk in hard_chunks])
    term1 = (sizes * (predicted - proportions).square().mean(dim=1)).mean()
    pooled_prediction = torch.cat(hard_chunks).mean(dim=0)
    pooled_prior = (sizes[:, None] * proportions).sum(dim=0) / sizes.sum()
    correction = (pooled_prediction - pooled_prior).square().mean()
    loss = term1 - (sizes - 1.0).mean() * correction
    return float(loss * classes / 2)


def test_dsq_evaluator_variable_sizes_matches_dataset_level_formula() -> None:
    bags = [
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([[0.0, 2.0]] * 5),
        torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
    ]
    proportions = torch.tensor([[0.5, 0.5], [0.2, 0.8], [2 / 3, 1 / 3]])

    actual = val_DSQ(
        SimpleNamespace(n_classes=2),
        _IdentityNetwork(),
        _Loader([_batch(bags, proportions)]),
        "cpu",
    )

    assert actual == pytest.approx(_expected_dsq(bags, proportions))


def test_dsq_evaluator_is_invariant_to_loader_batch_partition() -> None:
    bags = [
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([[0.0, 2.0]] * 5),
        torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[0.0, 2.0]]),
    ]
    proportions = torch.tensor(
        [[0.5, 0.5], [0.2, 0.8], [2 / 3, 1 / 3], [0.0, 1.0]]
    )
    args = SimpleNamespace(n_classes=2)

    one_batch = val_DSQ(
        args,
        _IdentityNetwork(),
        _Loader([_batch(bags, proportions)]),
        "cpu",
    )
    two_batches = val_DSQ(
        args,
        _IdentityNetwork(),
        _Loader(
            [
                _batch(bags[:2], proportions[:2]),
                _batch(bags[2:], proportions[2:]),
            ]
        ),
        "cpu",
    )

    assert one_batch == pytest.approx(two_batches)
    assert one_batch == pytest.approx(_expected_dsq(bags, proportions))


def test_dsq_evaluator_fixed_size_matches_legacy_one_batch_formula() -> None:
    torch.manual_seed(107)
    bags = [torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)]
    proportions = torch.tensor(
        [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5]]
    )
    actual = val_DSQ(
        SimpleNamespace(n_classes=3),
        _IdentityNetwork(),
        _Loader([_batch(bags, proportions)]),
        "cpu",
    )

    hard = _hard_predictions(bags, 3).view(3, 4, 3)
    predicted = hard.mean(dim=1)
    term1 = 4 * (predicted - proportions).square().mean(dim=1).mean()
    correction = (
        hard.reshape(-1, 3).mean(dim=0) - proportions.mean(dim=0)
    ).square().mean()
    legacy = float((term1 - 3 * correction) * 3 / 2)

    assert actual == pytest.approx(legacy)
