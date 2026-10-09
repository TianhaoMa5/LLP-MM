"""Exercise the real image loader, ResNet, optimizer, and CLI in a tiny run."""

import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np

from test_cluster_map_alignment import _write_cifar


def test_image_cli_completes_one_update_and_records_smoke(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_cifar(data)
    with (data / "cifar-10-batches-py/test_batch").open("wb") as f:
        pickle.dump(
            {"data": np.zeros((3, 3072), dtype=np.uint8), "labels": [0, 1, 2]}, f
        )
    output = tmp_path / "run"
    cmd = [
        sys.executable,
        "-m",
        "mo_matching.train",
        "--data_dir",
        str(data),
        "--output_dir",
        str(output),
        "--dataset",
        "CIFAR10",
        "--algorithm",
        "PM",
        "--bagsize",
        "8",
        "--batchsize",
        "2",
        "--n-classes",
        "10",
        "--num-workers",
        "0",
        "--bag_build",
        "cluster",
        "--cluster-manifest",
        str(data / "bags.npz"),
        "--max-steps",
        "1",
        "--skip_model_save",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "smoke_done").is_file() and not (output / "done").exists()
    rows = [json.loads(l) for l in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and np.isfinite(rows[0]["loss"])
    assert np.isfinite(rows[0]["test_acc"])
    assert rows[0]["hparams"]["optimizer"] == "SGD"
    assert (output / "cluster_manifest.json").exists()
    result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=30)
    assert result.returncode != 0
    assert "not empty" in result.stderr
