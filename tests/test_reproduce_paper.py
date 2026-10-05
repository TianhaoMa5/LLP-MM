"""Validate the release plan without data downloads or training."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reproduce_paper", REPO_ROOT / "scripts/reproduce_paper.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def one_run(**kwargs):
    return runner.build_plan(datasets=["CIFAR10"], methods=["PM"], bag_modes=["random"],
                             bag_sizes=[32], seeds=[0], **kwargs)


def test_complete_matrix_is_unique_and_uses_paper_optimization():
    runs = runner.build_plan()
    assert len(runs) == 3 * 3 * 4 * 11 * 3 == 1188
    assert len({run["run_id"] for run in runs}) == len(runs)
    assert len({run["output_dir"] for run in runs}) == len(runs)
    for run in runs:
        hp = run["hparams"]
        assert hp["optimizer"] == "SGD"
        assert hp["momentum"] == 0.9
        assert hp["weight_decay"] == 5e-4
        assert hp["lr"] == (0.005 if run["method"] == "LLP-PVC" else 0.05)
        command = run["command"]
        assert command[command.index("--epochs") + 1] == "500"
        assert int(command[command.index("--batchsize") + 1]) * run["bag_size"] == 1024
        assert int(command[command.index("--n-classes") + 1]) == (10 if run["dataset"] == "CIFAR10" else 100)
        assert float(command[command.index("--pi") + 1]) == (10.0 if run["bag_mode"] == "alphafirst" else 1.0)
        assert run["verification_status"] == "reconstructed_unverified"


def test_moment_orders_and_abs_methods_are_distinct():
    for run in runner.build_plan(bag_modes=["random"], bag_sizes=[16], seeds=[0]):
        hp = run["hparams"]
        if run["method"] == "LLP-MM":
            order = 8 if run["dataset"] == "CIFAR10" else 3
            assert hp["order"] == order
            assert hp["order_weights"] == [1.0 / order] * order
            assert hp["moment_loss_type"] == "ce"
            assert hp["moment_algorithm"] == "stable_dp"
            assert hp["moment_compute_dtype"] == "float64"
        if run["algorithm"] in {"EasyLLP", "GeneralUPM"}:
            assert hp["flooding"] is run["method"].endswith("-ABS")
            assert hp["flooding_b"] == 0.0


def test_cli_default_prints_only_and_handles_paths_with_spaces(tmp_path):
    data = tmp_path / "data with spaces"
    outputs = tmp_path / "new outputs"
    result = subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts/reproduce_paper.py"),
        "--dataset", "CIFAR100", "--method", "LLP-MM", "--bag-mode", "alphafirst",
        "--bag-size", "64", "--seed", "2", "--data-root", str(data),
        "--output-root", str(outputs),
    ], capture_output=True, text=True, check=True, cwd=tmp_path)
    assert "Selected 1 runs" in result.stderr
    assert "-m plench.train" in result.stdout
    assert "CIFAR100/alphafirst/bag64/LLP-MM/seed2" in result.stdout
    assert not outputs.exists()
    assert not data.exists()


def test_existing_output_blocks_every_subprocess(tmp_path, monkeypatch):
    runs = one_run(output_root=tmp_path / "outputs", data_root=tmp_path)
    target = Path(runs[0]["output_dir"])
    target.mkdir(parents=True)
    marker = target / "existing.txt"
    marker.write_text("keep me")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.execute_plan(runs)
    assert calls == []
    assert marker.read_text() == "keep me"


def test_execution_records_failure_and_stops_the_selection(tmp_path, monkeypatch):
    runs = runner.build_plan(datasets=["CIFAR10"], methods=["PM"], bag_modes=["random"],
                             bag_sizes=[32], seeds=[0, 1], output_root=tmp_path / "outputs",
                             data_root=tmp_path)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 17)
    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_plan(runs)
    assert len(calls) == 1
    output = Path(runs[0]["output_dir"])
    assert json.loads((output / "reproduction_status.json").read_text())["returncode"] == 17
    assert not Path(runs[1]["output_dir"]).exists()


def test_invalid_programmatic_filters_are_rejected():
    with pytest.raises(ValueError, match="Invalid method"):
        runner.build_plan(methods=["missing"])
