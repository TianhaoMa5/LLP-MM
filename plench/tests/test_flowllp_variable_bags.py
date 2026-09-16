from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core import flowllp_algorithm, hparams_registry  # noqa: E402
from plench.core.algorithms import ALGORITHMS, LLP_FlowLLP  # noqa: E402
from plench.data.ku_optofil_pbc import (  # noqa: E402
    SUPPORTED_KU_METHODS,
    update_ku_optofil_algorithm,
)


class _CounterScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _minimal_flow(
    *,
    bag_size: int = 999,
    classes: int = 3,
    input_dimension: int = 4,
    latent_dimension: int = 2,
) -> LLP_FlowLLP:
    algorithm = LLP_FlowLLP.__new__(LLP_FlowLLP)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.num_classes = classes
    algorithm.flow_latent_dim = latent_dimension
    algorithm.flow_anchors_per_class = 2
    algorithm.flow_particle_steps = 1
    algorithm.flow_particle_lr = 1e-3
    algorithm.flow_anchor_bag_batch = 3
    algorithm.flow_lambda_bag = 1.0
    algorithm.flow_lambda_anchor = 0.1
    algorithm.flow_reg_label = 0.0
    algorithm.flow_reg_bag_classifier = 0.0
    algorithm.flow_anchor_batch_size = 4
    algorithm.flow_pretrain_steps = 10
    algorithm.flow_cache_start = 0
    algorithm.featurizer = torch.nn.Linear(input_dimension, latent_dimension)
    algorithm.projector = torch.nn.Identity()
    algorithm.classifier = torch.nn.Linear(latent_dimension, classes)
    algorithm.network = torch.nn.Sequential(
        algorithm.featurizer, algorithm.projector, algorithm.classifier
    )
    algorithm.optimizer = torch.optim.SGD(algorithm.network.parameters(), lr=0.05)
    algorithm.scheduler = _CounterScheduler()
    number_of_anchors = classes * algorithm.flow_anchors_per_class
    algorithm.register_buffer(
        "flow_anchor_features", torch.zeros(number_of_anchors, latent_dimension)
    )
    algorithm.register_buffer(
        "flow_anchor_labels",
        torch.arange(classes).repeat_interleave(
            algorithm.flow_anchors_per_class
        ),
    )
    algorithm.register_buffer(
        "flow_anchors_ready", torch.tensor(False, dtype=torch.bool)
    )
    algorithm.register_buffer("flow_step", torch.tensor(0, dtype=torch.long))
    algorithm._flow_bag_cache = []
    return algorithm


def _proportions() -> torch.Tensor:
    return torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )


def test_flowllp_is_registered_with_original_hparams() -> None:
    hparams = hparams_registry.default_hparams("LLP_FlowLLP", "KUOptofilPBC")
    assert "LLP_FlowLLP" in ALGORITHMS
    assert "LLP_FlowLLP" in SUPPORTED_KU_METHODS
    assert hparams["model"] == "ResNet"
    assert hparams["flow_latent_dim"] == 50
    assert hparams["flow_pretrain_fraction"] == 0.5
    assert hparams["flow_anchors_per_class"] == 1000
    assert hparams["flow_particle_steps"] == 3000
    assert hparams["flow_lambda_bag"] == 1.0
    assert hparams["flow_lambda_anchor"] == 0.1


def test_flowllp_random_hparams_are_reproducible_and_nonconstant() -> None:
    first = hparams_registry.random_hparams("LLP_FlowLLP", "CV", 123)
    repeated = hparams_registry.random_hparams("LLP_FlowLLP", "CV", 123)
    assert first == repeated

    samples = [
        hparams_registry.random_hparams("LLP_FlowLLP", "CV", seed)
        for seed in range(1, 9)
    ]
    tuned = [
        "flow_latent_dim",
        "flow_pretrain_fraction",
        "flow_anchors_per_class",
        "flow_particle_steps",
        "flow_particle_lr",
        "flow_anchor_bag_batch",
        "flow_lambda_bag",
        "flow_lambda_anchor",
        "flow_reg_label",
        "flow_reg_bag_classifier",
        "flow_anchor_batch_size",
    ]
    assert all(len({sample[name] for sample in samples}) > 1 for name in tuned)
    assert all(sample["model"] == "RemoteResNet18" for sample in samples)
    assert all(0 < sample["flow_pretrain_fraction"] < 1 for sample in samples)
    assert all(sample["flow_particle_lr"] > 0 for sample in samples)


def test_flowllp_fixed_bag_loss_matches_original_formula() -> None:
    torch.manual_seed(211)
    bag_size = 4
    algorithm = _minimal_flow(bag_size=bag_size)
    proportions = torch.tensor(
        [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5]],
        dtype=torch.float64,
    )
    logits = torch.randn(3 * bag_size, 3, dtype=torch.float64)
    _, index = algorithm._layout(len(logits), proportions, None, None)
    actual = algorithm._bag_proportion_loss(logits, proportions, index)
    prediction = logits.softmax(dim=1).view(3, bag_size, 3).mean(dim=1)
    expected = -(proportions * prediction.clamp_min(1e-7).log()).sum(dim=1).mean()
    torch.testing.assert_close(actual, expected)


def test_flowllp_variable_bag_loss_and_cache_use_actual_sizes() -> None:
    torch.manual_seed(223)
    sizes = torch.tensor([3, 5, 8])
    index = torch.repeat_interleave(torch.arange(3), sizes)
    proportions = _proportions()
    algorithm = _minimal_flow()
    features = torch.randn(int(sizes.sum()), 2)
    logits = torch.randn(int(sizes.sum()), 3)

    _, resolved_index = algorithm._layout(
        len(logits), proportions, sizes, index
    )
    actual = algorithm._bag_proportion_loss(
        logits, proportions, resolved_index
    )
    prediction = torch.stack(
        [chunk.mean(dim=0) for chunk in logits.softmax(dim=1).split(sizes.tolist())]
    )
    expected = -(proportions * prediction.clamp_min(1e-7).log()).sum(dim=1).mean()
    torch.testing.assert_close(actual, expected)

    algorithm._cache_particle_bags(
        features, logits, proportions, resolved_index
    )
    assert [len(bag["data"]) for bag in algorithm._flow_bag_cache] == [3, 5, 8]
    for bag_id, cached in enumerate(algorithm._flow_bag_cache):
        torch.testing.assert_close(cached["prop"], proportions[bag_id].float())
        assert len(cached["y_pred_noisy"]) == int(sizes[bag_id])


def test_flowllp_particle_transport_uses_unit_mass_per_actual_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = torch.tensor([3, 5, 8])
    index = torch.repeat_interleave(torch.arange(3), sizes)
    algorithm = _minimal_flow()
    proportions = _proportions().float()
    features = torch.randn(int(sizes.sum()), 2)
    logits = torch.randn(int(sizes.sum()), 3)
    algorithm._cache_particle_bags(features, logits, proportions, index)
    observed = []

    def emd(source, target, _cost):
        observed.append((source.detach().clone(), target.detach().clone()))
        return torch.outer(source, target)

    if flowllp_algorithm.ot is None:
        monkeypatch.setattr(
            flowllp_algorithm, "ot", SimpleNamespace(emd=emd)
        )
    else:
        monkeypatch.setattr(flowllp_algorithm.ot, "emd", emd)

    algorithm._learn_particles()

    assert sorted(len(source) for source, _ in observed) == [3, 5, 8]
    for source, target in observed:
        torch.testing.assert_close(source, torch.full_like(source, 1 / len(source)))
        torch.testing.assert_close(source.sum(), torch.tensor(1.0))
        torch.testing.assert_close(target.sum(), torch.tensor(1.0))
    assert bool(algorithm.flow_anchors_ready.item())
    assert algorithm._flow_bag_cache == []


def test_flowllp_variable_update_caches_and_backpropagates() -> None:
    torch.manual_seed(227)
    sizes = torch.tensor([3, 5, 8])
    index = torch.repeat_interleave(torch.arange(3), sizes)
    algorithm = _minimal_flow()
    inputs = torch.randn(int(sizes.sum()), 4)
    features, logits = algorithm._forward_with_features(inputs)

    result = algorithm.update_from_outputs(
        features,
        logits,
        _proportions(),
        bag_sizes=sizes,
        bag_index=index,
    )

    assert result["flow_stage"] == 1.0
    assert result["anchor_loss"] == 0.0
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert [len(bag["data"]) for bag in algorithm._flow_bag_cache] == [3, 5, 8]
    assert int(algorithm.flow_step.item()) == 1
    assert algorithm.scheduler.steps == 1
    assert algorithm.featurizer.weight.grad is not None


def test_ku_flowllp_adapter_runs_chunked_natural_bag_update() -> None:
    torch.manual_seed(229)
    sizes = torch.tensor([3, 5, 8])
    total = int(sizes.sum())
    algorithm = _minimal_flow()
    batch = {
        "x": torch.randn(total, 4),
        "bag_index": torch.repeat_interleave(torch.arange(3), sizes),
        "bag_sizes": sizes,
        "proportion": _proportions().float(),
        "instance_weights": torch.ones(total),
    }

    result = update_ku_optofil_algorithm(
        algorithm,
        "LLP_FlowLLP",
        batch,
        "cpu",
        forward_chunk_size=4,
    )

    assert result["flow_stage"] == 1.0
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert [len(bag["data"]) for bag in algorithm._flow_bag_cache] == [3, 5, 8]
    assert algorithm.featurizer.weight.grad is not None


def test_flowllp_rejects_padding_and_inconsistent_membership() -> None:
    algorithm = _minimal_flow()
    proportions = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        algorithm._layout(
            5, proportions, torch.tensor([2, 2]), None
        )
    with pytest.raises(ValueError, match="do not match"):
        algorithm._layout(
            5,
            proportions,
            torch.tensor([2, 3]),
            torch.tensor([0, 0, 0, 1, 1]),
        )
