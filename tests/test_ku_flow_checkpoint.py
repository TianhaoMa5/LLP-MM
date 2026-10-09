"""FlowLLP checkpointing preserves gradients, BatchNorm state, and particle cache."""

import copy

import pytest
import torch
from torch import nn

from mo_matching.core.algorithms import LLP_FlowLLP, WarmupCosineLrScheduler
from mo_matching.data import ku_optofil_pbc


def _flow(*, anchors_ready):
    hparams = {
        "model": "Linear",
        "optimizer": "Adam",
        "lr": 0.001,
        "flow_latent_dim": 2,
        "flow_anchors_per_class": 2,
        "flow_pretrain_fraction": 0.5,
        "steps_per_epoch": 10,
    }
    algorithm = LLP_FlowLLP(100, (12, 4), [[0.5, 0.25, 0.25]], hparams, 4)
    algorithm.featurizer = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Tanh())
    algorithm.network = nn.Sequential(
        algorithm.featurizer, algorithm.projector, algorithm.classifier
    )
    algorithm.optimizer = torch.optim.Adam(algorithm.network.parameters(), lr=0.001)
    algorithm.scheduler = WarmupCosineLrScheduler(
        algorithm.optimizer, 100, warmup_iter=0
    )
    algorithm.flow_step.fill_(
        algorithm.flow_pretrain_steps if anchors_ready else algorithm.flow_cache_start
    )
    algorithm.flow_anchors_ready.fill_(anchors_ready)
    algorithm.flow_anchor_features.normal_()
    return algorithm


@pytest.mark.parametrize("anchors_ready", [False, True])
def test_ku_flow_checkpoint_matches_forward_backward_and_cache(
    monkeypatch, anchors_ready
):
    torch.manual_seed(109)
    reference = _flow(anchors_ready=anchors_ready)
    checkpointed = copy.deepcopy(reference)
    # Rebind optimizer/scheduler wrappers to the copied parameters.
    checkpointed.optimizer = torch.optim.Adam(
        checkpointed.network.parameters(), lr=0.001
    )
    checkpointed.scheduler = WarmupCosineLrScheduler(
        checkpointed.optimizer, 100, warmup_iter=0
    )
    sizes = torch.tensor([3, 4, 5])
    batch = {
        "x": torch.randn(12, 4),
        "bag_sizes": sizes,
        "bag_index": torch.repeat_interleave(torch.arange(3), sizes),
        "proportion": torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25], [0.2, 0.2, 0.6]]
        ),
    }
    calls = []
    original_checkpoint = ku_optofil_pbc.checkpoint

    def record_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(ku_optofil_pbc, "checkpoint", record_checkpoint)
    torch.manual_seed(113)
    expected = ku_optofil_pbc.update_ku_optofil_algorithm(
        reference,
        "LLP_FlowLLP",
        batch,
        "cpu",
        forward_chunk_size=4,
        activation_checkpoint=False,
    )
    assert not calls
    torch.manual_seed(113)
    actual = ku_optofil_pbc.update_ku_optofil_algorithm(
        checkpointed,
        "LLP_FlowLLP",
        batch,
        "cpu",
        forward_chunk_size=4,
        activation_checkpoint=True,
    )
    assert len(calls) == 3
    assert all(function.__self__ is checkpointed for function in calls)
    assert actual == pytest.approx(expected, abs=1e-7)
    for left, right in zip(reference.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(left, right)
        assert left.grad is not None and torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, checkpointed.state_dict()[name])
    assert reference.featurizer[1].num_batches_tracked.item() == 3
    expected_bags = 0 if anchors_ready else 3
    assert (
        len(reference._flow_bag_cache)
        == len(checkpointed._flow_bag_cache)
        == expected_bags
    )
    for left, right in zip(reference._flow_bag_cache, checkpointed._flow_bag_cache):
        for key in ("data", "prop", "y_pred_noisy"):
            torch.testing.assert_close(left[key], right[key])
