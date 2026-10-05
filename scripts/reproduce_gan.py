#!/usr/bin/env python3
"""Portable CIFAR-10 GAN paper commands. A plan is not a completed experiment."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("LLP-GAN", "MM+GAN")
MODES = ("random", "cluster", "alphafirst")


def validate_manifest(item):
    import numpy as np
    path = Path(item["manifest"])
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive:
            raise ValueError(f"Paper manifests require explicit construction metadata: {path}")
        meta = json.loads(archive["metadata"].item())
        expected = {"dataset": "CIFAR10", "mode": item["bag_mode"], "seed": item["seed"],
                    "bag_size": item["bag_size"], "num_classes": 10,
                    "alpha": 10.0 if item["bag_mode"] == "alphafirst" else 1.0}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Manifest {key}: expected {value!r}, got {meta.get(key)!r}: {path}")
        indices = archive["indices"]
        if indices.ndim != 2 or indices.shape[1] != item["bag_size"] or not indices.size:
            raise ValueError(f"Invalid bag dimensions: {path}")
        if indices.size != 50000 // item["bag_size"] * item["bag_size"]:
            raise ValueError(f"Paper run requires the complete CIFAR-10 training population: {path}")
        if not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= 50000:
            raise ValueError(f"Invalid CIFAR-10 indices: {path}")
        if len(np.unique(indices)) != indices.size:
            raise ValueError(f"Repeated training indices: {path}")
        if "val_indices" in archive and archive["val_indices"].size:
            raise ValueError(f"Paper GAN runs require no validation split: {path}")
    return {"path": str(path), "sha256_file_bytes": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": meta}


def commands(args):
    for method, mode, size, seed in itertools.product(
        [args.method] if args.method else METHODS,
        [args.bag_mode] if args.bag_mode else MODES,
        [args.bag_size] if args.bag_size else (16, 32, 64, 128),
        [args.seed] if args.seed is not None else (0, 1, 2),
    ):
        manifest = args.bag_root / f"cifar10_{mode}_m{size}_seed{seed}.npz"
        output = args.output_root / method / mode / f"m{size}" / f"seed{seed}"
        cmd = [sys.executable, "-m", "reproduction.llp_gan_pytorch.train",
               "--dataset", "cifar10", "--data-root", str(args.data_root),
               "--bag-type", "alpha_first" if mode == "alphafirst" else mode,
               "--bag-size", str(size), "--bag-file", str(manifest),
               "--output-dir", str(output), "--seed", str(seed),
               "--epochs", "500", "--samples-per-step", "1024",
               "--warmup-epochs", "5", "--d-lr", "0.05", "--momentum", "0.9",
               "--weight-decay", "0.0005", "--g-optimizer", "adam", "--g-lr", "0.0003",
               "--g-beta1", "0.5", "--g-beta2", "0.999", "--z-dim", "100",
               "--lambda-prop", "1", "--lambda-adv", "1", "--d-max-grad-norm", "5",
               "--g-max-grad-norm", "5", "--no-amp", "--no-download",
               "--num-workers", str(args.num_workers)]
        if method == "MM+GAN":
            cmd += ["--moment-order", "8", "--mm-implementation", "paper_dp"]
        yield {"method": method, "bag_mode": mode, "bag_size": size, "seed": seed,
               "manifest": str(manifest), "output": str(output), "command": cmd}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--bag-root", type=Path, default=ROOT / "data/bags/cifar10")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/gan")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--bag-mode", choices=MODES)
    p.add_argument("--bag-size", type=int, choices=(16, 32, 64, 128))
    p.add_argument("--seed", type=int, choices=(0, 1, 2))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--run", action="store_true", help="Execute selected runs sequentially")
    p.add_argument("--json", action="store_true", help="Print the full machine-readable plan")
    args = p.parse_args()
    for name in ("data_root", "bag_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    plan = list(commands(args))
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        for item in plan:
            print(shlex.join(item["command"]))
        print(f"# {len(plan)} planned runs; no training unless --run is passed.")
    if not args.run:
        return
    # Preflight the complete selected batch before launching any training.
    audits = {}
    for item in plan:
        if not Path(item["manifest"]).is_file():
            p.error(f"Missing shared bag manifest: {item['manifest']}")
        if Path(item["output"]).exists():
            p.error(f"Refusing to overwrite an existing run: {item['output']}")
        try:
            audits[item["manifest"]] = validate_manifest(item)
        except (ValueError, KeyError) as exc:
            p.error(str(exc))
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / ("manifest_audit_" + hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16] + ".json")
    audit_path.write_text(json.dumps(list(audits.values()), indent=2) + "\n")
    for item in plan:
        subprocess.run(item["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
