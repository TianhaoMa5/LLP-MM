from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from plench.data.ku_optofil_pbc import update_ku_optofil_algorithm  # noqa: E402


class _ToyAlgorithm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.BatchNorm1d(4),
            torch.nn.GELU(),
            torch.nn.Linear(4, 2),
        )
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.1)

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    def _backward_step(self, loss: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach())}


def _run_update(batch: dict[str, torch.Tensor], *, checkpoint: bool):
    torch.manual_seed(911)
    algorithm = _ToyAlgorithm()
    result = update_ku_optofil_algorithm(
        algorithm,
        "PM",
        batch,
        "cpu",
        forward_chunk_size=4,
        activation_checkpoint=checkpoint,
    )
    return algorithm, result


def test_cct_checkpoint_matches_direct_update_and_preserves_bn_count() -> None:
    torch.manual_seed(313)
    sizes = torch.tensor([4, 8])
    batch = {
        "x": torch.randn(12, 4),
        "bag_index": torch.repeat_interleave(torch.arange(2), sizes),
        "bag_sizes": sizes,
        "proportion": torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        "instance_weights": torch.ones(12),
    }

    direct, direct_result = _run_update(batch, checkpoint=False)
    replayed, replayed_result = _run_update(batch, checkpoint=True)

    assert replayed_result["loss"] == pytest.approx(direct_result["loss"])
    for direct_value, replayed_value in zip(
        direct.state_dict().values(), replayed.state_dict().values()
    ):
        torch.testing.assert_close(replayed_value, direct_value)
    assert int(replayed.network[1].num_batches_tracked) == 3
