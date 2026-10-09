"""Summarize completed runs with the checkpoint conventions used in the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read_run(path, setting, include_diverged=False):
    path = Path(path)
    if (path / "smoke_done").exists():
        raise ValueError(f"Smoke run is not a formal result: {path}")
    failed = (path / "numerical_failure.json").exists()
    if failed and not include_diverged:
        raise ValueError(f"Numerical failure requires --include-diverged: {path}")
    if not failed and not (path / "done").exists():
        raise ValueError(f"Run is unfinished: {path}")
    metrics = path / ("metrics.jsonl" if setting == "gan" else "results.jsonl")
    rows = [
        json.loads(line) for line in metrics.read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"No evaluation records: {path}")
    config = json.loads(
        (path / ("config.json" if setting == "gan" else "run_config.json")).read_text()
    )
    args = config if setting == "gan" else config["args"]
    if (
        args.get("max_steps") is not None
        or args.get("max_train_steps") is not None
        or args.get("max_eval_batches") is not None
    ):
        raise ValueError(f"Diagnostic run is not a formal result: {path}")
    expected = 100 if setting == "ku" else 500
    if not failed and (
        int(args["epochs"]) != expected
        or not math.isclose(float(rows[-1]["epoch"]), expected, abs_tol=1e-6)
    ):
        raise ValueError(f"Expected {expected} completed epochs: {path}")
    key = (
        "test_accuracy"
        if setting == "gan"
        else "test_macro_f1"
        if setting == "ku"
        else "test_acc"
    )
    valid = [
        r
        for r in rows
        if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])
    ]
    if not valid:
        raise ValueError(f"No finite {key}: {path}")
    row = max(valid, key=lambda r: r[key])
    keys = (
        ("test_acc", "test_balanced_accuracy", "test_macro_f1", "test_weighted_f1")
        if setting == "ku"
        else (key,)
    )
    values = {k: float(row[k]) for k in keys}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError(f"Non-finite or invalid selected metrics: {path}")
    if setting == "gan":
        identity = (
            args["dataset"],
            args["bag_type"],
            args["bag_size"],
            args["moment_order"],
        )
    else:
        h = config["hparams"]
        identity = (
            args["dataset"],
            args["algorithm"],
            args["bag_build"],
            args["bagsize"],
            h.get("order") if args["algorithm"] == "LLP_MM" else None,
            h.get("flooding", False),
        )
    return {
        "path": str(path),
        "seed": args["seed"],
        "epoch": row["epoch"],
        "status": "diverged" if failed else "completed",
        "metrics": values,
        "identity": identity,
    }


def summarize(paths, setting, include_diverged=False):
    runs = [read_run(p, setting, include_diverged) for p in paths]
    if len({tuple(r["identity"]) for r in runs}) != 1:
        raise ValueError("Do not aggregate different configurations")
    if len({r["seed"] for r in runs}) != len(runs):
        raise ValueError("Duplicate seed; choose one owned run per configuration")
    normal = [r for r in runs if r["status"] == "completed"]
    aggregate = {}
    if normal:
        for key in normal[0]["metrics"]:
            values = [r["metrics"][key] * 100 for r in normal]
            spread = (
                statistics.pstdev(values)
                if setting == "ku"
                else statistics.stdev(values)
                if len(values) > 1
                else None
            )
            aggregate[key] = {
                "mean_percent": statistics.mean(values),
                "std_percent": spread,
            }
    best_finite = {}
    if include_diverged:
        for key in runs[0]["metrics"]:
            values = [r["metrics"][key] * 100 for r in runs]
            spread = (
                statistics.pstdev(values)
                if setting == "ku"
                else statistics.stdev(values)
                if len(values) > 1
                else None
            )
            best_finite[key] = {
                "mean_percent": statistics.mean(values),
                "std_percent": spread,
            }
    return {
        "setting": setting,
        "completed": len(normal),
        "diverged": len(runs) - len(normal),
        "aggregate_completed_only": aggregate,
        "aggregate_best_finite_including_diverged": best_finite,
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--setting", choices=("image", "ku", "gan"), required=True)
    p.add_argument("--include-diverged", action="store_true")
    args = p.parse_args()
    try:
        result = summarize(args.runs, args.setting, args.include_diverged)
    except (ValueError, KeyError, OSError) as e:
        p.error(str(e))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
