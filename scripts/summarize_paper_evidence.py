#!/usr/bin/env python3
"""Validate archived GAN per-seed evidence and reproduce its summary CSV.

This script does not train models or turn aggregate-only KU data into raw runs.
It requires all expected, nonempty paired bag hashes. The archived hash scope
is not known precisely enough to compare with newly generated NPZ file bytes.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODES = ("random", "cluster", "alphafirst")
METHODS = ("LLP-GAN", "MM+GAN")
SIZES = (16, 32, 64, 128)
SEEDS = (0, 1, 2)
FIELDS = (
    "dataset", "bag_mode", "bag_size", "method", "n_runs", "seeds",
    "selection", "epoch", "mean_accuracy_percent",
    "sample_std_accuracy_percent", "std_ddof", "evidence_level",
)


def summarize(payload: dict) -> list[dict]:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported evidence schema")
    expected = set(itertools.product(METHODS, MODES, SIZES, SEEDS))
    indexed = {}
    runs = payload.get("runs", [])
    if len(runs) != 72:
        raise ValueError(f"Expected 72 runs; received {len(runs)}")
    for row in runs:
        key = (row["method"], row["bag_mode"], row["bag_size"], row["seed"])
        if key not in expected or row.get("dataset") != "cifar10":
            raise ValueError(f"Unexpected run: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate run: {key}")
        value = row.get("accuracy_percent")
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid percentage for {key}")
        digest = row.get("bag_hash")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Missing or invalid bag hash for {key}; empty hashes cannot verify a pair")
        if row.get("selection") != "final_epoch" or row.get("epoch") != 500:
            raise ValueError(f"Unexpected checkpoint selection for {key}")
        evidence_kind = row.get("epoch_evidence_kind")
        if evidence_kind not in {"retained_result_fields", "companion_audit_statement"}:
            raise ValueError(f"Missing checkpoint provenance for {key}")
        provenance_id = row.get("epoch_source_artifact")
        provenance = payload.get("provenance", {}).get(provenance_id, {})
        if re.fullmatch(r"[0-9a-f]{64}", provenance.get("artifact_sha256", "")) is None:
            raise ValueError(f"Missing source artifact hash for {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Evidence does not cover the complete expected grid")
    for mode, size, seed in itertools.product(MODES, SIZES, SEEDS):
        left = indexed[(METHODS[0], mode, size, seed)]["bag_hash"]
        right = indexed[(METHODS[1], mode, size, seed)]["bag_hash"]
        if left != right:
            raise ValueError(f"Paired bag hash mismatch: {mode}, {size}, seed {seed}")

    references = {}
    for row in payload.get("archived_summary_reference", []):
        key = (row["method"], row["bag_mode"], row["bag_size"])
        if key in references:
            raise ValueError(f"Duplicate archived aggregate: {key}")
        references[key] = row
    if set(references) != set(itertools.product(METHODS, MODES, SIZES)):
        raise ValueError("Expected all 24 archived summary references")

    output = []
    for mode, size, method in itertools.product(MODES, SIZES, METHODS):
        values = [indexed[(method, mode, size, seed)]["accuracy_percent"] for seed in SEEDS]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        reference = references[(method, mode, size)]
        for field, actual in (("mean_accuracy_percent", mean), ("sample_std_accuracy_percent", sd)):
            if not math.isclose(actual, reference[field], rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Recomputed {field} disagrees with archive: {method}, {mode}, {size}")
        output.append({
            "dataset": "cifar10", "bag_mode": mode, "bag_size": size,
            "method": method, "n_runs": 3, "seeds": "0;1;2",
            "selection": "final_epoch", "epoch": 500,
            "mean_accuracy_percent": f"{mean:.10f}",
            "sample_std_accuracy_percent": f"{sd:.10f}",
            "std_ddof": 1, "evidence_level": "recomputed_from_archived_per_seed_extract",
        })
    return output


def csv_text(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/gan_cifar10_runs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gan_cifar10_summary.csv")
    parser.add_argument("--check", action="store_true", help="Check the existing CSV without writing")
    args = parser.parse_args()
    rows = summarize(json.loads(args.input.read_text()))
    content = csv_text(rows)
    if args.check:
        if not args.output.is_file() or args.output.read_text() != content:
            raise SystemExit("Summary CSV is missing or does not match recomputed evidence")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print("Verified 72 unique runs, 36 nonempty paired bag hashes, seeds 0/1/2, and 24 aggregate rows.")
    print("This verifies archived evidence aggregation; it does not verify a fresh training reproduction.")


if __name__ == "__main__":
    main()
