import json
import math
import pytest
from mo_matching.summarize import read_run, summarize


def write_run(root, seed, values, *, setting="image", failure=False):
    root.mkdir()
    args = {
        "dataset": "KUOptofilPBC" if setting == "ku" else "CIFAR10",
        "algorithm": "PM",
        "epochs": 100 if setting == "ku" else 500,
        "seed": seed,
        "bag_build": "cluster",
        "bagsize": 32,
    }
    (root / "run_config.json").write_text(json.dumps({"args": args, "hparams": {}}))
    (root / "results.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    (root / ("numerical_failure.json" if failure else "done")).write_text("{}")
    return root


def test_accuracy_selects_best_and_uses_sample_std(tmp_path):
    paths = []
    for i, v in enumerate([0.9, 0.92, 0.94]):
        paths.append(
            write_run(
                tmp_path / str(i),
                i,
                [{"epoch": 100, "test_acc": v}, {"epoch": 500, "test_acc": 0.8}],
            )
        )
    result = summarize(paths, "image")
    assert result["aggregate_completed_only"]["test_acc"] == pytest.approx(
        {"mean_percent": 92, "std_percent": 2}
    )
    assert [r["epoch"] for r in result["runs"]] == [100] * 3


def test_ku_keeps_all_metrics_at_one_macro_checkpoint(tmp_path):
    rows = [
        {
            "epoch": 50,
            "test_acc": 0.8,
            "test_macro_f1": 0.7,
            "test_balanced_accuracy": 0.6,
            "test_weighted_f1": 0.75,
        },
        {
            "epoch": 100,
            "test_acc": 0.9,
            "test_macro_f1": 0.65,
            "test_balanced_accuracy": 0.61,
            "test_weighted_f1": 0.8,
        },
    ]
    path = write_run(tmp_path / "ku", 0, rows, setting="ku")
    selected = read_run(path, "ku")
    assert selected["epoch"] == 50
    assert selected["metrics"]["test_acc"] == 0.8
    assert (
        summarize([path], "ku")["aggregate_completed_only"]["test_macro_f1"][
            "std_percent"
        ]
        == 0
    )


def test_gan_selects_each_seeds_peak_and_uses_sample_std(tmp_path):
    paths = []
    for seed, peak in enumerate([0.9, 0.92, 0.94]):
        path = tmp_path / str(seed)
        path.mkdir()
        config = {
            "dataset": "cifar10",
            "bag_type": "cluster",
            "bag_size": 32,
            "moment_order": 8,
            "epochs": 500,
            "seed": seed,
        }
        (path / "config.json").write_text(json.dumps(config))
        rows = [
            {"epoch": 200, "test_accuracy": peak},
            {"epoch": 500, "test_accuracy": 0.8},
        ]
        (path / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (path / "done").write_text("done")
        paths.append(path)
    result = summarize(paths, "gan")
    assert result["aggregate_completed_only"]["test_accuracy"] == pytest.approx(
        {"mean_percent": 92, "std_percent": 2}
    )
    assert [run["epoch"] for run in result["runs"]] == [200] * 3


def test_divergence_is_flagged_and_not_mixed_with_normal_mean(tmp_path):
    normal = write_run(tmp_path / "normal", 0, [{"epoch": 500, "test_acc": 0.9}])
    failed = write_run(
        tmp_path / "failed", 1, [{"epoch": 30, "test_acc": 0.7}], failure=True
    )
    with pytest.raises(ValueError, match="include-diverged"):
        summarize([normal, failed], "image")
    r = summarize([normal, failed], "image", True)
    assert (
        r["diverged"] == 1
        and r["aggregate_completed_only"]["test_acc"]["mean_percent"] == 90
    )
    assert r["runs"][1]["metrics"]["test_acc"] == 0.7
    assert (
        r["aggregate_best_finite_including_diverged"]["test_acc"]["mean_percent"] == 80
    )


def test_rejects_incomplete_smoke_and_duplicate_seeds(tmp_path):
    path = write_run(tmp_path / "a", 0, [{"epoch": 499, "test_acc": 0.9}])
    with pytest.raises(ValueError, match="completed epochs"):
        read_run(path, "image")
    (path / "smoke_done").write_text("done")
    with pytest.raises(ValueError, match="Smoke"):
        read_run(path, "image")
    b = write_run(tmp_path / "b", 0, [{"epoch": 500, "test_acc": 0.9}])
    c = write_run(tmp_path / "c", 0, [{"epoch": 500, "test_acc": 0.8}])
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize([b, c], "image")
