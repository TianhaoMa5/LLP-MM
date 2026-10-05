#!/usr/bin/env python3
"""Reconstructed KU paper protocol; preserve test-Macro-F1 selection explicitly."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"PM": "PM", "DSQ": "LLP_DSQ", "LLP-PVC": "LLP_PVC", "LLP-FC": "LLP_FC",
           "ROT": "ROT", "EasyLLP": "EasyLLP", "EasyLLP-ABS": "EasyLLP",
           "GeneralUPM": "GeneralUPM", "GeneralUPM-ABS": "GeneralUPM",
           "FlowLLP": "LLP_FlowLLP", "LLP-MM": "LLP_MM"}


def make_config(method, seed, data_root, output_root):
    bags_per_step = 5 if method.startswith("GeneralUPM") else 4
    hp = {"model": "ImageNetResNet18", "pretrained": True, "input_resolution": 224,
          "optimizer": "Adam", "lr": 0.001, "weight_decay": 0.0005,
          "warmup_fraction": 0.05, "warmup_ratio": 0.1, "warmup": "linear",
          "cosine_mode": "standard", "ku_activation_checkpoint": True,
          "ku_select_by_validation_macro_f1": False, "ku_test_every_checkpoint": True}
    if method == "LLP-MM":
        hp.update(order=8, order_weights=[0.125] * 8, moment_loss_type="ce",
                  moment_algorithm="stable_dp", moment_compute_dtype="float64",
                  moment_ce_smoothing_tau=0.0001)
    if method.startswith(("EasyLLP", "GeneralUPM")):
        hp.update(flooding=method.endswith("-ABS"), flooding_b=0.0)
    if method == "FlowLLP":
        hp.update(flow_pretrain_fraction=0.5, flow_latent_dim=50,
                  flow_anchors_per_class=1000, flow_particle_steps=3000,
                  flow_particle_lr=0.001, flow_lambda_anchor=0.1)
    return {"dataset": "KUOptofilPBC", "algorithm": METHODS[method],
            "data_dir": str(data_root), "output_dir": str(output_root / method / f"seed{seed}"),
            "batchsize": bags_per_step, "bagsize": 32, "n_classes": 13,
            "ku_merge_validation_into_train": True, "ku_unknown_bag_max_size": 128,
            "ku_unknown_bag_seed": 0, "variable_bag_size": True, "natural_bags": False,
            "train_instance_sample_size": None, "bag_build": "random", "holdout_fraction": 0.0,
            "epochs": 100, "checkpoint_freq": (245 + bags_per_step - 1) // bags_per_step,
            "seed": seed, "trial_seed": seed, "hparams_seed": 0, "num_workers": 4,
            "forward_chunk_size": 32, "hparams": json.dumps(hp, sort_keys=True)}


def command(config):
    cmd = [sys.executable, "-m", "plench.train"]
    aliases = {"n_classes": "n-classes", "num_workers": "num-workers",
               "ku_merge_validation_into_train": "ku-merge-validation-into-train",
               "ku_unknown_bag_max_size": "ku-unknown-bag-max-size",
               "ku_unknown_bag_seed": "ku-unknown-bag-seed", "variable_bag_size": "variable-bag-size",
               "forward_chunk_size": "forward-chunk-size"}
    for key, value in config.items():
        if value is None or value is False:
            continue
        cmd.append("--" + aliases.get(key, key))
        if value is not True:
            cmd.append(str(value))
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/ku_optofil_pbc")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/ku")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--seed", type=int, choices=(0, 1, 2))
    p.add_argument("--run", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    configs = [make_config(m, s, args.data_root, args.output_root)
               for m, s in itertools.product([args.method] if args.method else METHODS,
                                             [args.seed] if args.seed is not None else (0, 1, 2))]
    if args.json:
        print(json.dumps(configs, indent=2))
    else:
        for cfg in configs:
            print(shlex.join(command(cfg)))
        print(f"# {len(configs)} reconstructed runs; aggregation must select test Macro-F1, ddof=0.")
    if args.run:
        if not args.data_root.is_dir():
            p.error(f"Missing prepared KU dataset: {args.data_root}")
        for cfg in configs:
            if Path(cfg["output_dir"]).exists():
                p.error(f"Refusing existing output: {cfg['output_dir']}")
        for cfg in configs:
            subprocess.run(command(cfg), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
