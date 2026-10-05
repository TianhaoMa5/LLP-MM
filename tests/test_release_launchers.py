"""Check experiment boundaries and result-selection correctness without real data."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_gan
import reproduce_ku
import summarize_ku


def test_gan_pairs_share_manifests_and_keep_historical_objective(tmp_path):
    args = argparse.Namespace(method=None, bag_mode=None, bag_size=None, seed=None,
                              data_root=tmp_path / "data", bag_root=tmp_path / "bags",
                              output_root=tmp_path / "out", num_workers=0)
    plan = list(reproduce_gan.commands(args))
    assert len(plan) == 72
    for left, right in zip(plan[:36], plan[36:]):
        assert left["manifest"] == right["manifest"]
        assert left["output"] != right["output"]
        assert "--mm-implementation" not in left["command"]
        assert right["command"][right["command"].index("--mm-implementation") + 1] == "paper_dp"


def test_ku_protocol_uses_100_epochs_and_absolute_risk(tmp_path):
    for method in reproduce_ku.METHODS:
        cfg = reproduce_ku.make_config(method, 0, tmp_path, tmp_path)
        hp = json.loads(cfg["hparams"])
        assert cfg["epochs"] == 100
        assert hp["optimizer"] == "Adam" and hp["warmup_fraction"] == 0.05
        if method.startswith(("EasyLLP", "GeneralUPM")):
            assert hp["flooding"] == method.endswith("-ABS")
            assert hp["flooding_b"] == 0
        assert "--ku-unknown-bag-max-size" in reproduce_ku.command(cfg)


def test_gan_rejects_wrong_dirichlet_protocol_even_if_filename_matches(tmp_path):
    path = tmp_path / "cifar10_alphafirst_m16_seed0.npz"
    meta = {"dataset": "CIFAR10", "mode": "alphafirst", "seed": 0,
            "bag_size": 16, "num_classes": 10, "alpha": 1.0}
    np.savez(path, indices=np.arange(50000).reshape(-1, 16), metadata=json.dumps(meta))
    item = {"manifest": str(path), "bag_mode": "alphafirst", "bag_size": 16, "seed": 0}
    with pytest.raises(ValueError, match="alpha"):
        reproduce_gan.validate_manifest(item)
    meta["alpha"] = 10.0
    np.savez(path, indices=np.arange(50000).reshape(-1, 16), metadata=json.dumps(meta))
    assert len(reproduce_gan.validate_manifest(item)["sha256_file_bytes"]) == 64
    np.savez(path, indices=np.arange(32).reshape(2, 16), metadata=json.dumps(meta))
    with pytest.raises(ValueError, match="complete CIFAR-10"):
        reproduce_gan.validate_manifest(item)


def test_ku_uses_all_metrics_at_same_selected_epoch_and_rejects_partial(tmp_path):
    cfg = reproduce_ku.make_config("PM", 0, tmp_path, tmp_path)
    rows = [{"epoch": e, "args": cfg, "hparams": json.loads(cfg["hparams"]),
             "test_macro_f1": 0.9 if e in (23, 30) else 0.1,
             "test_acc": e / 100, "test_balanced_accuracy": 0.5, "test_weighted_f1": 0.6}
            for e in range(1, 101)]
    (tmp_path / "done").write_text("")
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    selected = summarize_ku.selected_run(tmp_path, "PM", 0)
    assert selected["epoch"] == 23 and selected["test_acc"] == 0.23
    rows[0]["hparams"]["optimizer"] = "SGD"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="optimizer"):
        summarize_ku.selected_run(tmp_path, "PM", 0)
    rows[0]["hparams"]["optimizer"] = "Adam"
    path.write_text("\n".join(json.dumps(r) for r in rows[:-1]))
    with pytest.raises(ValueError, match="Missing epochs"):
        summarize_ku.selected_run(tmp_path, "PM", 0)
