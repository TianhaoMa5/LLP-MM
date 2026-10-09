import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mo_matching.core import algorithms
from mo_matching.core.hparams_registry import default_hparams
from mo_matching.run import METHODS, build_command, build_parser


def command(dataset="CIFAR10", method="LLP-MM", *extra):
    return build_command(
        build_parser().parse_args(
            [
                "--dataset",
                dataset,
                "--method",
                method,
                "--data-root",
                "data",
                "--output-dir",
                "out",
                *extra,
            ]
        )
    )


def hparams(argv):
    return json.loads(argv[argv.index("--hparams") + 1])


@pytest.mark.parametrize(
    "dataset,classes,order",
    [("CIFAR10", 10, 8), ("CIFAR100", 100, 3), ("miniImageNet", 100, 3)],
)
@pytest.mark.parametrize("size", [16, 32, 64, 128])
def test_image_settings(dataset, classes, order, size):
    argv = command(dataset, "LLP-MM", "--bag-size", str(size))
    assert int(argv[argv.index("--batchsize") + 1]) * size == 1024
    assert argv[argv.index("--n-classes") + 1] == str(classes)
    assert argv[argv.index("--epochs") + 1] == "500"
    h = hparams(argv)
    assert h["order"] == order
    assert h["order_weights"] == [1 / order] * order
    assert h["moment_implementation"] == "paper_image"
    assert h["optimizer"] == "SGD"


def test_abs_only_changes_flooding_and_pvc_learning_rate():
    for name in ["EasyLLP", "GeneralUPM"]:
        plain = hparams(command(method=name))
        absolute = hparams(command(method=name + "-ABS"))
        assert not plain["flooding"] and absolute["flooding"]
        absolute["flooding"] = False
        assert plain == absolute
    assert hparams(command(method="LLP-PVC"))["lr"] == 0.005


def test_ku_and_order_ablation_protocol():
    for order in [3, 5, 8]:
        argv = command("KUOptofilPBC", "LLP-MM", "--order", str(order))
        h = hparams(argv)
        assert h["model"] == "ImageNetResNet18" and h["pretrained"] is True
        assert h["optimizer"] == "Adam" and h["lr"] == 0.001
        assert h["cosine_mode"] == "standard" and h["warmup_fraction"] == 0.05
        assert argv[argv.index("--epochs") + 1] == "100"
        assert argv[argv.index("--batchsize") + 1] == "4"
        assert "--ku-merge-validation-into-train" in argv
        assert argv[argv.index("--ku-unknown-bag-max-size") + 1] == "128"
        assert h["order"] == order
    argv = command("CIFAR10", "LLP-MM", "--order", "1")
    assert argv[argv.index("--algorithm") + 1] == "PM"
    assert hparams(command("CIFAR10", "LLP-MM", "--order", "13"))["order"] == 13


def test_gan_pair_shares_protocol_and_manifest():
    a = command("CIFAR10", "LLP-GAN", "--bag-file", "bags.npz")
    b = command("CIFAR10", "MM+GAN", "--bag-file", "bags.npz")
    i = a.index("--moment-order") + 1
    assert a[i] == "1" and b[i] == "8"
    a[i] = b[i]
    assert a == b
    with pytest.raises(ValueError, match="same --bag-file"):
        command(method="MM+GAN")
    with pytest.raises(ValueError, match="CIFAR10"):
        command("CIFAR100", "LLP-GAN", "--bag-file", "bags.npz")


@pytest.mark.parametrize("method", ["GeneralUPM", "GeneralUPM-ABS"])
def test_ku_generalupm_keeps_at_least_two_bags_in_every_batch(method):
    argv = command("KUOptofilPBC", method)
    batch_size = int(argv[argv.index("--batchsize") + 1])
    assert batch_size == 5
    assert 245 % batch_size == 0
    assert argv[argv.index("--checkpoint_freq") + 1] == "49"


@pytest.mark.parametrize("method", [m for m in METHODS if METHODS[m] is not None])
def test_every_paper_method_can_take_an_update(method):
    torch.manual_seed(31)
    np.random.seed(31)
    argv = command(method=method)
    h = hparams(argv)
    h.update(
        model="Linear",
        order=2,
        order_weights=[0.5, 0.5],
        flow_latent_dim=2,
        flow_particle_steps=1,
        flow_anchors_per_class=2,
    )
    name = METHODS[method]
    proportions = [[0.75, 0.25], [0.25, 0.75]]
    model = algorithms.get_algorithm_class(name)(20, (8, 4), proportions, h, 4)
    before = [p.detach().clone() for p in model.network.parameters()]
    result = model.update(
        (torch.randn(8, 4), torch.tensor(proportions, dtype=torch.float64))
    )
    assert math.isfinite(result["loss"])
    assert all(torch.isfinite(p).all() for p in model.network.parameters())
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.network.parameters())
    )


def test_registries_contain_only_paper_methods_and_datasets():
    assert set(algorithms.ALGORITHMS) == set(METHODS.values()) - {None}
    with pytest.raises(ValueError, match="Unsupported"):
        default_hparams("LLP_PT", "CIFAR10")
    with pytest.raises(ValueError, match="Unsupported"):
        default_hparams("PM", "SVHN")
