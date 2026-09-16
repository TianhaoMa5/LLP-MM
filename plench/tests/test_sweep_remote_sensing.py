import pytest
import os
from pathlib import Path
import subprocess
import sys

import torch

from plench.core.algorithms import LLP_MM
from plench.sweep import Job, make_args_list


def _remote_args(**overrides):
    values = {
        "n_trials_from": 0,
        "n_trials": 2,
        "dataset_names": ["CV", "LEM"],
        "algorithms": ["LLP_PVC"],
        "n_hparams_from": 0,
        "n_hparams": 1,
        "epochs": 3,
        "data_dir": "/datasets/plench",
        "holdout_fraction": 0.0,
        "hparams": None,
        "skip_model_save": False,
        "batchsize": 2,
        "bagsize": 32,
        "checkpoint_freq": 100,
        "bag_build": "cluster",
        "pi": 0.5,
        "instances_per_epoch": 200_000,
        "num_workers": 6,
        "cluster_seed": 0,
    }
    values.update(overrides)
    return make_args_list(**values)


def test_cv_lem_sweep_uses_metadata_and_separate_cluster_seed(tmp_path):
    jobs = _remote_args()
    assert len(jobs) == 4
    assert len({job["seed"] for job in jobs}) == 4
    for args in jobs:
        assert "n-classes" not in args
        assert args["instances-per-epoch"] == 200_000
        assert args["num-workers"] == 6
        assert args["cluster-seed"] == 0
        command = Job(args, str(tmp_path)).command_str
        assert "--instances-per-epoch 200000" in command
        assert "--cluster-seed 0" in command
        assert "--bag_build cluster" in command


def test_remote_sweep_rejects_unsupported_bag_construction():
    with pytest.raises(ValueError, match="bag size"):
        _remote_args(bagsize=16)
    assert len(_remote_args(bag_build="alphafirst")) == 4
    with pytest.raises(ValueError, match="random, cluster, or alphafirst"):
        _remote_args(bag_build="unknown")


def test_llp_mm_fixed_bag_update_runs_backward():
    algorithm = LLP_MM(
        epochs=4,
        input_shape=(1, 4),
        train_givenY=[[2 / 3, 1 / 3], [1 / 3, 2 / 3]],
        hparams={
            "model": "Linear", "lr": 1e-3, "weight_decay": 0.0,
            "order": 2,
            "moment_loss_type": "ce",
            "moment_algorithm": "stable_dp",
            "moment_compute_dtype": "float64",
            "moment_ce_smoothing_tau": 1e-4,
            "order_weights": [0.5, 0.5],
        },
        bagsize=3,
    )
    before = algorithm.classifier.weight.detach().clone()
    result = algorithm.update((
        torch.randn(6, 4),
        torch.tensor([[2 / 3, 1 / 3], [1 / 3, 2 / 3]]),
    ))
    assert result["loss"] >= 0
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert not torch.equal(before, algorithm.classifier.weight.detach())
    assert algorithm.moment_loss_type == "ce"
    assert algorithm.moment_algorithm == "stable_dp"
    assert algorithm.moment_compute_dtype == "float64"
    assert algorithm.moment_ce_smoothing_tau == pytest.approx(1e-4)


def test_all_method_remote_template_dry_run(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "run_cv_lem_all_methods.sh"
    environment = dict(os.environ)
    environment.update({
        "DATASETS": "CV LEM",
        "BAG_SIZES": "32",
        "OUTPUT_ROOT": str(tmp_path / "commands"),
        "COMMAND_LAUNCHER": "dummy",
        "N_TRIALS": "1",
        "N_HPARAMS": "1",
        "PLENCH_PYTHON": sys.executable,
    })
    completed = subprocess.run(
        ["bash", str(script)], check=True, text=True, capture_output=True,
        env=environment,
    )
    commands = [line for line in completed.stdout.splitlines()
                if " -m plench.train " in line]
    assert len(commands) == 2 * 17
    assert any(
        "--algorithm LLP_MM" in command
        and '"moment_loss_type":"ce"' in command
        and '"moment_algorithm":"stable_dp"' in command
        and '"moment_compute_dtype":"float64"' in command
        and '"moment_ce_smoothing_tau":0.0001' in command
        for command in commands
    )
    assert any("--algorithm LLP_FlowLLP" in command for command in commands)


def test_flowllp_remote_smoke_matrix_template_dry_run(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "run_cv_lem_flowllp_tuning.sh"
    environment = dict(os.environ)
    environment.update({
        "DATASETS": "CV LEM",
        "BAG_SIZES": "32",
        "BAG_BUILDS": "random cluster",
        "OUTPUT_ROOT": str(tmp_path / "flow-commands"),
        "COMMAND_LAUNCHER": "dummy",
        "N_TRIALS": "1",
        "N_HPARAMS": "1",
        "PLENCH_PYTHON": sys.executable,
    })
    completed = subprocess.run(
        ["bash", str(script)], check=True, text=True, capture_output=True,
        env=environment,
    )
    commands = [line for line in completed.stdout.splitlines()
                if " -m plench.train " in line]
    assert len(commands) == 4
    assert {"CV", "LEM"} == {
        command.split("--dataset ", 1)[1].split()[0] for command in commands
    }
    assert {"random", "cluster"} == {
        command.split("--bag_build ", 1)[1].split()[0] for command in commands
    }
