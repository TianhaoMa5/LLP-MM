#!/usr/bin/env python3
"""Aggregate complete KU epoch logs using the paper's test-Macro-F1 selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from reproduce_ku import METHODS, make_config

METRICS = ("test_acc", "test_macro_f1", "test_balanced_accuracy", "test_weighted_f1")


def selected_run(folder, method, seed):
    if not (folder / "done").is_file():
        raise ValueError(f"Incomplete run: {folder}")
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
    epoch_rows = {}
    expected = make_config(method, seed, Path("data"), Path("outputs"))
    expected_hp = json.loads(expected["hparams"])
    ignored_args = {"data_dir", "output_dir", "hparams", "num_workers", "forward_chunk_size"}
    for row in rows:
        epoch = float(row["epoch"])
        if epoch < 1 or epoch > 100 or not epoch.is_integer():
            raise ValueError(f"Expected completed integer epochs 1..100: {folder}: {epoch}")
        if epoch in epoch_rows:
            raise ValueError(f"Duplicate epoch: {folder}: {epoch}")
        if row["args"]["seed"] != seed or row["args"]["algorithm"] != METHODS[method]:
            raise ValueError(f"Run identity mismatch: {folder}")
        if row["args"]["epochs"] != 100 or row["args"]["dataset"] != "KUOptofilPBC":
            raise ValueError(f"Protocol mismatch: {folder}")
        for key, value in expected.items():
            if key not in ignored_args and row["args"].get(key) != value:
                raise ValueError(f"Protocol mismatch for {key}: {folder}")
        for key, value in expected_hp.items():
            if row["hparams"].get(key) != value:
                raise ValueError(f"Scientific hyperparameter mismatch for {key}: {folder}")
        if method.startswith(("EasyLLP", "GeneralUPM")):
            if row["hparams"].get("flooding", False) != method.endswith("-ABS"):
                raise ValueError(f"ABS variant mismatch: {folder}")
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid fractional metric {metric}: {folder}")
        epoch_rows[int(epoch)] = row
    if set(epoch_rows) != set(range(1, 101)):
        raise ValueError(f"Missing epochs in {folder}")
    # Match first maximum / earliest epoch when several epochs tie.
    selected = max((epoch_rows[e] for e in sorted(epoch_rows)), key=lambda r: r["test_macro_f1"])
    return selected


def summarize(root):
    summaries = []
    for method in METHODS:
        runs = [selected_run(root / method / f"seed{seed}", method, seed) for seed in (0, 1, 2)]
        row = {"method": method, "seeds": "0,1,2", "selection": "max_test_macro_f1",
               "selected_epochs": "/".join(str(int(r["epoch"])) for r in runs), "std_ddof": 0}
        for metric in METRICS:
            values = [r[metric] * 100 for r in runs]
            row[metric + "_mean_percent"] = statistics.mean(values)
            row[metric + "_std_percent"] = statistics.pstdev(values)
        summaries.append(row)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated 33 complete runs into {args.output}; test selection, population std.")


if __name__ == "__main__":
    main()
