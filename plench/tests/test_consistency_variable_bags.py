from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from plench.core.algorithms import (  # noqa: E402
    LLP_AHIL,
    LLP_DC,
    LLP_FixMatch,
    LLP_SoftMatch,
    LLP_VAT,
)
from plench.data.ku_optofil_pbc import (  # noqa: E402
    SUPPORTED_KU_METHODS,
    collate_ku_optofil_bags,
    update_ku_optofil_algorithm,
)


def _vat(bag_size: int) -> LLP_VAT:
    algorithm = LLP_VAT.__new__(LLP_VAT)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    return algorithm


def _fixmatch(bag_size: int) -> LLP_FixMatch:
    algorithm = LLP_FixMatch.__new__(LLP_FixMatch)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.mu_u = 0.7
    algorithm.tau = 0.55
    algorithm.eps = 1e-12
    return algorithm


def _softmatch(bag_size: int) -> LLP_SoftMatch:
    algorithm = LLP_SoftMatch.__new__(LLP_SoftMatch)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.eps = 1e-12
    algorithm.ema_p = 0.9
    algorithm.softmatch_k = 2.0
    algorithm.softmatch_s = 4.0
    algorithm.softmatch_eps = 1e-6
    algorithm.mu_u = 0.6
    return algorithm


def _ahil(bag_size: int) -> LLP_AHIL:
    algorithm = LLP_AHIL.__new__(LLP_AHIL)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.beta_i = 1.3
    algorithm.beta_b = 1.7
    algorithm.mu_u = 0.8
    algorithm.tau = 0.0
    algorithm.eps = 1e-12
    return algorithm


def _dc(bag_size: int) -> LLP_DC:
    algorithm = LLP_DC.__new__(LLP_DC)
    torch.nn.Module.__init__(algorithm)
    algorithm.bagsize = bag_size
    algorithm.lam_u = 0.5
    algorithm.thr = 0.35
    algorithm._solver_epsilon = 1e-12
    algorithm._solver_cost_scale = 10000

    def hard_argmax(
        probabilities: torch.Tensor,
        _proportions: torch.Tensor,
        bagsize: int,
        n_classes: int,
        **_kwargs,
    ) -> torch.Tensor:
        assert len(probabilities) == bagsize
        return F.one_hot(
            probabilities.argmax(dim=1), num_classes=n_classes
        ).to(torch.int32).cpu()

    algorithm.solve_optimal_onehot_with_proportions_torch = hard_argmax
    return algorithm


def _fixed_proportions() -> torch.Tensor:
    return torch.tensor(
        [[0.50, 0.25, 0.25], [0.00, 0.50, 0.50], [0.25, 0.25, 0.50]],
        dtype=torch.float64,
    )


def _variable_proportions() -> torch.Tensor:
    return torch.tensor(
        [[1 / 3, 2 / 3, 0.0], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]],
        dtype=torch.float64,
    )


def _reference_fixmatch(
    algorithm: LLP_FixMatch,
    outputs: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bags, classes = proportions.shape
    probabilities = outputs.softmax(dim=1).view(2 * bags, bag_size, classes)
    weak = probabilities[:bags]
    bag_prediction = weak.mean(dim=1).clamp_min(algorithm.eps)
    bag_loss = -(proportions * bag_prediction.log()).sum(dim=1).mean()
    pseudo = weak.detach().reshape(bags * bag_size, classes)
    confidence, labels = pseudo.max(dim=1)
    strong_logits = outputs.view(2 * bags, bag_size, classes)[bags:].reshape(
        bags * bag_size, classes
    )
    per_instance = F.cross_entropy(strong_logits, labels, reduction="none")
    mask = (confidence >= algorithm.tau).to(per_instance)
    instance_loss = (per_instance * mask).mean()
    return (
        bag_loss + algorithm.mu_u * instance_loss,
        bag_loss,
        instance_loss,
    )


def _reference_dc(
    algorithm: LLP_DC,
    outputs: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bags, classes = proportions.shape
    weak_logits = outputs[: bags * bag_size]
    strong_logits = outputs[bags * bag_size :]
    weak = weak_logits.softmax(dim=1).view(bags, bag_size, classes)
    pseudo = F.one_hot(
        weak.reshape(-1, classes).argmax(dim=1), num_classes=classes
    ).to(outputs)
    predictions = weak.mean(dim=1).clamp_min(algorithm._solver_epsilon)
    bag_loss = -(proportions * predictions.log()).sum(dim=1).mean()
    confidence = (weak.reshape(-1, classes) * pseudo).sum(dim=1)
    mask = (confidence >= algorithm.thr).to(outputs)
    instance_loss = (
        F.cross_entropy(strong_logits, pseudo.argmax(dim=1), reduction="none")
        * mask
    ).mean()
    return bag_loss + algorithm.lam_u * instance_loss, bag_loss, instance_loss


def _reference_softmatch(
    algorithm: LLP_SoftMatch,
    outputs: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
    state: tuple[torch.Tensor, float, float],
):
    bags, classes = proportions.shape
    weak = outputs.softmax(dim=1).view(2 * bags, bag_size, classes)[:bags]
    bag_prediction = weak.mean(dim=1).clamp_min(algorithm.eps)
    bag_loss = -(proportions * bag_prediction.log()).sum(dim=1).mean()
    weak_instances = weak.detach().reshape(-1, classes)
    labels = weak_instances.argmax(dim=1)
    strong_logits = outputs.view(2 * bags, bag_size, classes)[bags:].reshape(
        -1, classes
    )
    per_instance = F.cross_entropy(strong_logits, labels, reduction="none")
    ema, mean, variance = algorithm.update_prob_t(*state, weak_instances)
    _, mask = algorithm.calculate_mask(weak_instances, mean, variance)
    instance_loss = (per_instance * mask.to(per_instance)).mean()
    return (
        bag_loss + algorithm.mu_u * instance_loss,
        bag_loss,
        instance_loss,
        ema,
        mean,
        variance,
    )


def _reference_ahil(
    algorithm: LLP_AHIL,
    outputs: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bags, classes = proportions.shape
    probabilities = outputs.softmax(dim=1).view(2 * bags, bag_size, classes)
    weak, strong = probabilities[:bags], probabilities[bags:]
    bag_prediction = weak.mean(dim=1).clamp_min(algorithm.eps)
    bag_loss = -(proportions * bag_prediction.log()).sum(dim=1).mean()
    labels = weak.detach().reshape(-1, classes).argmax(dim=1)
    strong_logits = outputs.view(2 * bags, bag_size, classes)[bags:].reshape(
        -1, classes
    )
    per_instance = F.cross_entropy(strong_logits, labels, reduction="none")
    with torch.no_grad():
        strong_instances = strong.reshape(-1, classes).clamp_min(algorithm.eps)
        entropy_i = algorithm.calc_instance_entropy(strong_instances)
        lambda_i = algorithm.normal(entropy_i, 0, algorithm.beta_i)
        entropy_b = algorithm.calc_bag_entropy(strong)
        predicted = strong_instances.argmax(dim=1)
        one_hot = F.one_hot(predicted, num_classes=classes).to(strong_instances)
        log_base = torch.log(
            torch.tensor(float(classes), dtype=proportions.dtype)
        )
        lambda_b = []
        for bag, chunk in enumerate(one_hot.split(bag_size)):
            indices = chunk.argmax(dim=1)
            chunk_mean = chunk.mean(dim=0)
            label_proportion = proportions[bag].to(chunk_mean)
            errors = 1 - (chunk_mean - label_proportion).abs().pow(1 / log_base)
            optimum = algorithm.calc_opt_entropy(label_proportion * bag_size)
            normalized = algorithm.normal(
                entropy_b[bag].to(optimum), optimum, algorithm.beta_b
            )
            _ = errors[indices]
            lambda_b.append(normalized[indices].to(per_instance))
        weights = lambda_i.to(per_instance) * torch.cat(lambda_b)
    instance_loss = (per_instance * weights).mean()
    return bag_loss + algorithm.mu_u * instance_loss, bag_loss, instance_loss


def test_vat_pm_fixed_and_variable_bag_means() -> None:
    torch.manual_seed(151)
    fixed_size = 4
    fixed_proportions = _fixed_proportions()
    fixed_logits = torch.randn(12, 3, dtype=torch.float64)
    algorithm = _vat(fixed_size)
    expected_fixed = -(
        fixed_proportions
        * fixed_logits.softmax(dim=1).view(3, fixed_size, 3).mean(dim=1).log()
    ).sum(dim=1).mean()
    torch.testing.assert_close(
        algorithm.PM_Loss(fixed_logits, fixed_proportions), expected_fixed
    )

    sizes = torch.tensor([3, 5, 8])
    proportions = _variable_proportions()
    logits = torch.randn(int(sizes.sum()), 3, dtype=torch.float64)
    predictions = torch.stack(
        [chunk.mean(dim=0) for chunk in logits.softmax(dim=1).split(sizes.tolist())]
    )
    expected_variable = -(proportions * predictions.log()).sum(dim=1).mean()
    actual = algorithm.PM_Loss(logits, proportions, bag_sizes=sizes)
    torch.testing.assert_close(actual, expected_variable)


def test_fixed_size_consistency_losses_match_legacy_formulas() -> None:
    torch.manual_seed(157)
    size = 4
    proportions = _fixed_proportions()
    outputs = torch.randn(2 * len(proportions) * size, 3, dtype=torch.float64)

    fix = _fixmatch(size)
    for actual, expected in zip(
        fix.LLP_FixMatch_Loss(outputs, proportions),
        _reference_fixmatch(fix, outputs, proportions, size),
    ):
        torch.testing.assert_close(actual, expected)

    dc = _dc(size)
    for actual, expected in zip(
        dc.LLP_DC_Loss(outputs, proportions),
        _reference_dc(dc, outputs, proportions, size),
    ):
        torch.testing.assert_close(actual, expected)

    ahil = _ahil(size)
    for actual, expected in zip(
        ahil.LLP_AHIL_Loss(outputs, proportions),
        _reference_ahil(ahil, outputs, proportions, size),
    ):
        torch.testing.assert_close(actual, expected)


def test_softmatch_fixed_size_matches_legacy_formula_and_state() -> None:
    torch.manual_seed(163)
    size = 4
    proportions = _fixed_proportions()
    outputs = torch.randn(2 * len(proportions) * size, 3, dtype=torch.float64)
    algorithm = _softmatch(size)
    state = (torch.full((3,), 1 / 3, dtype=torch.float64), 1 / 3, 1.0)
    expected = _reference_softmatch(algorithm, outputs, proportions, size, state)
    actual = algorithm.LLP_SoftMatch_Loss(outputs, proportions, *state)
    for actual_value, expected_value in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(actual_value, expected_value)
    assert actual[4] == pytest.approx(expected[4])
    assert actual[5] == pytest.approx(expected[5])


@pytest.mark.parametrize("method", ["LLP_DC", "LLP_FixMatch", "LLP_AHIL"])
def test_variable_paired_losses_are_bag_index_permutation_invariant(method: str) -> None:
    torch.manual_seed(167)
    sizes = torch.tensor([3, 5, 8])
    proportions = _variable_proportions()
    total = int(sizes.sum())
    outputs = torch.randn(2 * total, 3, dtype=torch.float64, requires_grad=True)
    index = torch.repeat_interleave(torch.arange(3), sizes)
    permutation = torch.randperm(total)
    permuted = torch.cat(
        [outputs[:total][permutation], outputs[total:][permutation]], dim=0
    )
    algorithm = {
        "LLP_DC": _dc(999),
        "LLP_FixMatch": _fixmatch(999),
        "LLP_AHIL": _ahil(999),
    }[method]
    loss_function = getattr(
        algorithm,
        {
            "LLP_DC": "LLP_DC_Loss",
            "LLP_FixMatch": "LLP_FixMatch_Loss",
            "LLP_AHIL": "LLP_AHIL_Loss",
        }[method],
    )

    contiguous = loss_function(
        outputs, proportions, bag_sizes=sizes, bag_index=index
    )[0]
    shuffled = loss_function(
        permuted,
        proportions,
        bag_sizes=sizes,
        bag_index=index[permutation],
    )[0]

    torch.testing.assert_close(shuffled, contiguous)
    contiguous.backward()
    assert outputs.grad is not None
    assert torch.isfinite(outputs.grad).all()


def test_softmatch_variable_bags_and_state_are_permutation_invariant() -> None:
    torch.manual_seed(173)
    sizes = torch.tensor([3, 5, 8])
    proportions = _variable_proportions()
    total = int(sizes.sum())
    outputs = torch.randn(2 * total, 3, dtype=torch.float64)
    index = torch.repeat_interleave(torch.arange(3), sizes)
    permutation = torch.randperm(total)
    permuted = torch.cat(
        [outputs[:total][permutation], outputs[total:][permutation]], dim=0
    )
    state = (torch.full((3,), 1 / 3, dtype=torch.float64), 1 / 3, 1.0)
    algorithm = _softmatch(999)
    contiguous = algorithm.LLP_SoftMatch_Loss(
        outputs, proportions, *state, bag_sizes=sizes, bag_index=index
    )
    shuffled = algorithm.LLP_SoftMatch_Loss(
        permuted,
        proportions,
        *state,
        bag_sizes=sizes,
        bag_index=index[permutation],
    )
    for first, second in zip(contiguous[:4], shuffled[:4]):
        torch.testing.assert_close(first, second)
    assert contiguous[4] == pytest.approx(shuffled[4])
    assert contiguous[5] == pytest.approx(shuffled[5])


def test_paired_losses_reject_padding_rows() -> None:
    algorithm = _fixmatch(2)
    proportions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match=r"sum\(bag_sizes\)"):
        algorithm.LLP_FixMatch_Loss(
            torch.randn(10, 2), proportions, bag_sizes=torch.tensor([2, 2])
        )


def test_ku_collate_preserves_aligned_paired_views_and_supported_methods() -> None:
    items = []
    for bag, size in enumerate((2, 5)):
        items.append(
            {
                "x": torch.full((size, 3), float(bag)),
                "x_strong": torch.full((size, 3), float(bag + 10)),
                "proportion": torch.tensor([0.5, 0.5]),
                "bag_id": str(bag),
                "instance_ids": [f"{bag}-{i}" for i in range(size)],
                "instance_labels": torch.zeros(size, dtype=torch.long),
                "class_counts": torch.tensor([size, 0]),
            }
        )
    batch = collate_ku_optofil_bags(items)
    torch.testing.assert_close(batch["bag_sizes"], torch.tensor([2, 5]))
    torch.testing.assert_close(
        batch["bag_index"], torch.tensor([0, 0, 1, 1, 1, 1, 1])
    )
    assert len(batch["x"]) == len(batch["x_strong"]) == 7
    assert {
        "LLP_AHIL",
        "LLP_DC",
        "LLP_FixMatch",
        "LLP_SoftMatch",
        "LLP_VAT",
    }.issubset(SUPPORTED_KU_METHODS)


def _attach_toy_network(algorithm) -> None:
    algorithm.network = torch.nn.Linear(3, 3)

    def backward(loss: torch.Tensor) -> dict[str, float]:
        algorithm.network.zero_grad()
        loss.backward()
        return {"loss": float(loss.detach())}

    algorithm._backward_step = backward


@pytest.mark.parametrize(
    "method", ["LLP_AHIL", "LLP_DC", "LLP_FixMatch", "LLP_SoftMatch", "LLP_VAT"]
)
def test_ku_variable_adapter_runs_forward_and_backward(method: str) -> None:
    torch.manual_seed(179)
    sizes = torch.tensor([3, 5, 8])
    total = int(sizes.sum())
    proportions = _variable_proportions().to(torch.float32)
    batch = {
        "x": torch.randn(total, 3),
        "bag_index": torch.repeat_interleave(torch.arange(3), sizes),
        "bag_sizes": sizes,
        "proportion": proportions,
        "instance_weights": torch.ones(total),
    }
    if method != "LLP_VAT":
        batch["x_strong"] = torch.randn(total, 3)

    if method == "LLP_AHIL":
        algorithm = _ahil(999)
    elif method == "LLP_DC":
        algorithm = _dc(999)
    elif method == "LLP_FixMatch":
        algorithm = _fixmatch(999)
    elif method == "LLP_SoftMatch":
        algorithm = _softmatch(999)
    else:
        algorithm = _vat(999)
        algorithm.consistency = 1.0
        algorithm.consistency_rampup = 0
        algorithm.consistency_criterion = LLP_VAT.VATLoss(
            xi=1e-3, eps=0.1, ip=1
        )
    _attach_toy_network(algorithm)

    result = update_ku_optofil_algorithm(
        algorithm,
        method,
        batch,
        "cpu",
        forward_chunk_size=4,
        iteration=3,
        softmatch_state=(torch.full((3,), 1 / 3), 1 / 3, 1.0)
        if method == "LLP_SoftMatch"
        else None,
    )
    step = result[0] if method == "LLP_SoftMatch" else result
    assert torch.isfinite(torch.tensor(step["loss"]))
    assert algorithm.network.weight.grad is not None
    assert torch.isfinite(algorithm.network.weight.grad).all()
