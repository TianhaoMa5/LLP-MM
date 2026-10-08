#!/usr/bin/env python3
"""Plan or run the reconstructed 1,188-run image benchmark, without a scheduler.

Planning is the default and does not load datasets or create output files.
Training requires --run. Existing run directories are never overwritten.
See configs/paper_protocol.json for the unresolved historical provenance.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "paper_protocol.json"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _selected(values: Sequence[Any] | None, allowed: Sequence[Any], name: str) -> list[Any]:
    selected = list(allowed) if values is None else list(dict.fromkeys(values))
    invalid = [value for value in selected if value not in allowed]
    if not selected or invalid:
        raise ValueError(f"Invalid {name}: {invalid or selected}; choose from {list(allowed)}")
    return selected


def build_plan(
    *,
    protocol: dict[str, Any] | None = None,
    datasets: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    bag_modes: Sequence[str] | None = None,
    bag_sizes: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    data_root: Path = REPO_ROOT / "data",
    output_root: Path = REPO_ROOT / "outputs" / "paper",
    python: str = sys.executable,
    num_workers: int = 4,
    order: int | None = None,
) -> list[dict[str, Any]]:
    protocol = load_protocol() if protocol is None else protocol
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    axes = (
        _selected(datasets, list(protocol["datasets"]), "dataset"),
        _selected(bag_modes, list(protocol["bag_modes"]), "bag mode"),
        _selected(bag_sizes, protocol["bag_sizes"], "bag size"),
        _selected(methods, list(protocol["methods"]), "method"),
        _selected(seeds, protocol["seeds"], "seed"),
    )
    if order is not None and (axes[3] != ["LLP-MM"] or not 1 <= order <= 13):
        raise ValueError("An order override requires only LLP-MM and an order from 1 to 13")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    training = protocol["training"]
    runs = []
    for dataset, mode, size, method, seed in itertools.product(*axes):
        dataset_spec = protocol["datasets"][dataset]
        method_spec = protocol["methods"][method]
        samples_per_step = int(training["samples_per_step"])
        if samples_per_step % size:
            raise ValueError("samples_per_step must be divisible by every bag size")
        hparams = {
            **training["hparams"],
            "model": dataset_spec["model"],
            **method_spec.get("hparams", {}),
        }
        if method == "LLP-MM":
            selected_order = dataset_spec["moment_order"] if order is None else order
            hparams.update(order=selected_order, order_weights=[1.0 / selected_order] * selected_order)
        method_dir = method if order is None else f"{method}-order{order}"
        run_id = f"{dataset}/{mode}/bag{size}/{method_dir}/seed{seed}"
        output_dir = output_root / run_id
        command = [
            str(python), "-m", "plench.train",
            "--dataset", dataset,
            "--algorithm", method_spec["algorithm"],
            "--data_dir", str(data_root),
            "--output_dir", str(output_dir),
            "--bag_build", mode,
            "--pi", str(protocol["bag_modes"][mode]),
            "--bagsize", str(size),
            "--batchsize", str(samples_per_step // size),
            "--n-classes", str(dataset_spec["num_classes"]),
            "--epochs", str(training["epochs"]),
            "--checkpoint_freq", str(training["checkpoint_freq"]),
            "--seed", str(seed),
            "--trial_seed", str(seed),
            "--hparams_seed", "0",
            "--num-workers", str(num_workers),
            "--hparams", json.dumps(hparams, sort_keys=True, separators=(",", ":")),
        ]
        runs.append({
            "run_id": run_id,
            "protocol_id": protocol["protocol_id"],
            "verification_status": protocol["verification_status"],
            "dataset": dataset,
            "method": method,
            "algorithm": method_spec["algorithm"],
            "bag_mode": mode,
            "bag_size": size,
            "seed": seed,
            "hparams": hparams,
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "command": command,
        })
    return runs


def validate_outputs(runs: Sequence[dict[str, Any]]) -> None:
    """Preflight the entire selection before starting any training."""
    outputs = [Path(run["output_dir"]) for run in runs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("The plan contains duplicate output directories")
    occupied = [str(path) for path in outputs if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing run directories; choose a fresh --output-root: "
            + ", ".join(occupied[:5])
        )


def execute_plan(runs: Sequence[dict[str, Any]]) -> None:
    validate_outputs(runs)
    missing = sorted({run["data_root"] for run in runs if not Path(run["data_root"]).is_dir()})
    if missing:
        raise FileNotFoundError("Prepare --data-root before training: " + ", ".join(missing))
    for run in runs:
        output_dir = Path(run["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "reproduction_plan.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8"
        )
        print(shlex.join(run["command"]), flush=True)
        result = subprocess.run(run["command"], cwd=REPO_ROOT, check=False)
        status = {
            "run_id": run["run_id"],
            "returncode": result.returncode,
            "training_done_marker": (output_dir / "done").is_file(),
            "verification_status": "numerical_paper_match_unverified",
        }
        (output_dir / "reproduction_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, run["command"])
        if not status["training_done_marker"]:
            raise RuntimeError(f"Training exited without a done marker: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List selected run IDs without training")
    mode.add_argument("--dry-run", action="store_true", help="Print commands without training (default)")
    mode.add_argument("--run", action="store_true", help="Execute all selected runs sequentially")
    parser.add_argument("--dataset", nargs="+", choices=list(protocol["datasets"]))
    parser.add_argument("--method", nargs="+", choices=list(protocol["methods"]))
    parser.add_argument("--bag-mode", nargs="+", choices=list(protocol["bag_modes"]))
    parser.add_argument("--bag-size", nargs="+", type=int, choices=protocol["bag_sizes"])
    parser.add_argument("--seed", nargs="+", type=int, choices=protocol["seeds"])
    parser.add_argument("--order", type=int, choices=range(1, 14), help="LLP-MM order ablation; use PM for the figure's order-one point")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="Shared dataset directory used by the PLeNCH loaders")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "paper")
    parser.add_argument("--python", default=sys.executable, help="Python executable for training")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        runs = build_plan(
            datasets=args.dataset, methods=args.method, bag_modes=args.bag_mode,
            bag_sizes=args.bag_size, seeds=args.seed, data_root=args.data_root,
            output_root=args.output_root, python=args.python, num_workers=args.num_workers,
            order=args.order,
        )
        print(
            f"Selected {len(runs)} runs. Reconstructed paper protocol; historical numerical "
            "and shared-bag equality remain unverified. See configs/paper_protocol.json.",
            file=sys.stderr,
        )
        if args.run:
            execute_plan(runs)
        else:
            for run in runs:
                print(run["run_id"] if args.list else shlex.join(run["command"]))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
