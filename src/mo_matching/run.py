"""Run one paper configuration; never submit a grid or overwrite a run."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import json

DATASETS = ("CIFAR10", "CIFAR100", "miniImageNet", "KUOptofilPBC")
METHODS = {
    "PM": "PM",
    "DSQ": "LLP_DSQ",
    "LLP-PVC": "LLP_PVC",
    "LLP-FC": "LLP_FC",
    "ROT": "ROT",
    "EasyLLP": "EasyLLP",
    "EasyLLP-ABS": "EasyLLP",
    "GeneralUPM": "GeneralUPM",
    "GeneralUPM-ABS": "GeneralUPM",
    "FlowLLP": "LLP_FlowLLP",
    "LLP-MM": "LLP_MM",
    "LLP-GAN": None,
    "MM+GAN": None,
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--method", choices=tuple(METHODS), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--bag-type", choices=("random", "cluster", "alphafirst"), default="cluster"
    )
    p.add_argument("--bag-size", type=int, choices=(16, 32, 64, 128), default=32)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--order", type=int, help="MM order ablation; one uses PM")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bag-file", type=Path, help="Shared NPZ manifest for the GAN pair")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One update; excluded from formal summaries",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the fully resolved command"
    )
    return p


def build_command(args):
    from .core.hparams_registry import default_hparams

    method = args.method
    if args.order is not None and args.order < 1:
        raise ValueError("Order must be positive")
    if args.order is not None and method not in {"LLP-MM", "MM+GAN"}:
        raise ValueError("--order applies only to moment matching")
    if args.order == 1 and method == "LLP-MM":
        method = "PM"
    if method in {"LLP-GAN", "MM+GAN"}:
        if args.dataset != "CIFAR10":
            raise ValueError("The paper GAN comparison uses CIFAR10")
        if args.bag_file is None:
            raise ValueError("The GAN pair requires the same --bag-file")
        command = [
            sys.executable,
            "-m",
            "mo_matching.gan.train",
            "--dataset",
            "cifar10",
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(args.output_dir),
            "--bag-type",
            "alpha_first" if args.bag_type == "alphafirst" else args.bag_type,
            "--bag-size",
            str(args.bag_size),
            "--bag-file",
            str(args.bag_file),
            "--moment-order",
            str(args.order or 8) if method == "MM+GAN" else "1",
            "--mm-implementation",
            "paper_dp",
            "--no-amp",
            "--no-nesterov",
            "--epochs",
            "1" if args.smoke else "500",
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.smoke:
            command += [
                "--max-train-steps",
                "1",
                "--max-eval-batches",
                "1",
                "--samples-per-step",
                str(2 * args.bag_size),
            ]
        return command
    natural = args.dataset == "KUOptofilPBC"
    algorithm = METHODS[method]
    bags_per_update = 5 if natural and algorithm == "GeneralUPM" else 4
    h = default_hparams(algorithm, args.dataset)
    h.update(flooding=method.endswith("-ABS"), flooding_b=0.0)
    if args.order is not None:
        h["order"] = args.order
    if algorithm == "LLP_MM":
        h["order_weights"] = [1.0 / h["order"]] * h["order"]
    if args.smoke and algorithm == "LLP_FlowLLP":
        h.update(flow_particle_steps=1, flow_anchors_per_class=2)
    command = [
        sys.executable,
        "-m",
        "mo_matching.train",
        "--dataset",
        args.dataset,
        "--algorithm",
        algorithm,
        "--data_dir",
        str(args.data_root),
        "--output_dir",
        str(args.output_dir),
        "--bag_build",
        "random" if natural else args.bag_type,
        "--bagsize",
        str(args.bag_size),
        "--batchsize",
        str(bags_per_update) if natural else str(1024 // args.bag_size),
        "--n-classes",
        "13" if natural else "10" if args.dataset == "CIFAR10" else "100",
        "--epochs",
        "100" if natural else "500",
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--hparams",
        json.dumps(h),
        "--checkpoint_freq",
        str((245 + bags_per_update - 1) // bags_per_update) if natural else "1000",
        "--pi",
        "10" if args.bag_type == "alphafirst" else "1",
    ]
    if natural:
        command += [
            "--ku-merge-validation-into-train",
            "--ku-unknown-bag-max-size",
            "128",
            "--ku-unknown-bag-seed",
            "0",
            "--forward-chunk-size",
            "32",
        ]
    elif args.bag_type == "cluster":
        manifest = (
            args.data_root
            / "bags"
            / f"{args.dataset}_bag{args.bag_size}_seed{args.seed}.npz"
        )
        command += ["--cluster-manifest", str(manifest)]
    if args.smoke:
        command += ["--max-steps", "1", "--skip-final-test"]
    return command


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        command = build_command(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        import shlex

        print(shlex.join(command))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; use a new directory")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
