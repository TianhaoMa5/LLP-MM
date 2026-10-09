import argparse
import collections
import json
import math
import os
import random
import sys
import time
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from .core import hparams_registry, algorithms
from .lib import misc
from .data.images import get_train_loader, get_val_loader
from .data.ku_optofil_pbc import (
    canonical_ku_optofil_dataset,
    evaluate_ku_optofil,
    is_ku_optofil_dataset,
    load_ku_optofil_bundle,
    update_ku_optofil_algorithm,
)

if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Learning from Label Proportions")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file; explicit command-line flags override its values",
    )
    parser.add_argument("--data_dir", type=str, default="data_dir")
    parser.add_argument(
        "--data_root",
        "--data-root",
        dest="data_root",
        type=str,
        default=None,
        help="dataset root containing official images and prepared caches",
    )
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--algorithm", type=str, default="LLP_MM")
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help='JSON-serialized hparams dict, e.g. \'{"model":"ResNet","lr":0.01}\'',
    )
    parser.add_argument(
        "--batchsize", type=int, default=32, help="Mini-batch size for training"
    )
    parser.add_argument(
        "--bagsize", type=int, default=32, help="Bag size for LLP / bag-level training"
    )
    parser.add_argument(
        "--n-classes", type=int, default=None, help="number of classes; inferred for KU"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1000,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--output_dir", type=str, default="train_output")
    parser.add_argument("--holdout_fraction", type=float, default=0.0)
    parser.add_argument("--skip_model_save", action="store_true")
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Replicate the classifier across all CUDA devices visible to this process",
    )
    parser.add_argument(
        "--bag_build",
        type=str,
        default="random",
        choices=["random", "cluster", "alphafirst"],
    )
    parser.add_argument(
        "--pi",
        type=float,
        default=1.0,
        help="Dirichlet concentration (alpha0). Smaller -> more dominant (peaky) bags; larger -> more uniform.",
    )
    parser.add_argument(
        "--num-bags", type=int, default=None, help="Optional diagnostic bag limit"
    )
    parser.add_argument(
        "--train-instance-sample-size",
        type=int,
        default=None,
        help="Natural-bag datasets: optional without-replacement training instances per complete bag; null preserves the full bag",
    )
    parser.add_argument(
        "--ku-merge-validation-into-train",
        action="store_true",
        help="KU only: train on official train + validation patients and retain the official test split",
    )
    parser.add_argument(
        "--ku-unknown-bag-max-size",
        type=int,
        default=None,
        help="KU only: split the missing-patient-ID training group into balanced sub-bags; known patients and test unchanged",
    )
    parser.add_argument(
        "--ku-unknown-bag-seed",
        type=int,
        default=0,
        help="KU unknown-group assignment seed, independent of model seed",
    )
    parser.add_argument(
        "--forward-chunk-size",
        type=int,
        default=32,
        help="KU-Optofil train/evaluation device-forward chunk size; patient bags remain complete",
    )
    parser.add_argument(
        "--natural-bags",
        action="store_true",
        help="Declare that dataset bags are natural rather than synthetic",
    )
    parser.add_argument(
        "--variable-bag-size",
        action="store_true",
        help="Declare variable-size natural bags",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Skip instance-level final test evaluation for short diagnostic runs",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cluster-manifest",
        type=str,
        default=None,
        help="Path to shared bag assignments",
    )
    if config_args.config:
        with open(config_args.config, encoding="utf-8") as config_file:
            config_defaults = json.load(config_file)
        if not isinstance(config_defaults, dict):
            parser.error("--config must contain one JSON object")
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config_defaults) - valid_dests)
        if unknown:
            parser.error(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**config_defaults)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Diagnostic update limit; never a formal completion",
    )
    args = parser.parse_args()
    if args.dataset not in {"miniImageNet", "CIFAR10", "CIFAR100", "KUOptofilPBC"}:
        parser.error("Dataset is not part of the paper")
    if args.algorithm not in algorithms.ALGORITHMS:
        parser.error("Algorithm is not part of the paper")
    if is_ku_optofil_dataset(args.dataset):
        args.dataset = canonical_ku_optofil_dataset(args.dataset)
        metadata = load_ku_optofil_bundle(args.data_dir)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with KU-Optofil ({metadata.num_classes})"
            )
        if (
            args.train_instance_sample_size is not None
            and args.train_instance_sample_size <= 0
        ):
            parser.error(
                "--train-instance-sample-size must be positive or omitted/null"
            )
        if args.forward_chunk_size <= 0:
            parser.error("--forward-chunk-size must be positive")
    elif args.n_classes is None:
        args.n_classes = 10 if args.dataset == "CIFAR10" else 100
    start_step = 0
    algorithm_dict = None
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        parser.error("Output directory is not empty; use a new run directory")
    os.makedirs(args.output_dir, exist_ok=True)
    done_file = os.path.join(args.output_dir, "done")
    if os.path.exists(done_file):
        print(f"Output already exists at {args.output_dir}, skipping.")
        sys.exit(0)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.ku_merge_validation_into_train and hparams.get(
        "ku_select_by_validation_macro_f1", False
    ):
        parser.error("KU train+test protocol cannot select a model using validation")
    hparams["n_classes"] = args.n_classes
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as handle:
        json.dump({"args": vars(args), "hparams": hparams}, handle, indent=2)
    print("HParams:")
    for k, v in sorted(hparams.items()):
        print("\t{}: {}".format(k, v))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    method = "DLLP"
    backbone = hparams.get("model")
    (
        train_loader,
        val_loader,
        train_label_prob,
        dataset_length,
        dataset_length_var,
        input_dim,
        prior_train,
        prior_val,
        prior_all,
    ) = get_train_loader(
        pi=args.pi,
        bag_build=args.bag_build,
        classes=args.n_classes,
        holdout_fraction=args.holdout_fraction,
        dataset=args.dataset,
        batch_size=args.batchsize,
        bag_size=args.bagsize,
        root=args.data_dir,
        method=method,
        supervised=False,
        backbone=backbone,
        seed=args.seed,
        num_bags=args.num_bags,
        num_workers=args.num_workers,
        train_instance_sample_size=args.train_instance_sample_size,
        ku_merge_validation_into_train=args.ku_merge_validation_into_train,
        ku_unknown_bag_max_size=args.ku_unknown_bag_max_size,
        ku_unknown_bag_seed=args.ku_unknown_bag_seed,
        cluster_manifest=args.cluster_manifest,
    )
    if args.cluster_manifest is not None:
        with np.load(args.cluster_manifest, allow_pickle=False) as manifest:
            args.cluster_manifest_sha256 = str(manifest["sha256"].item())
        with open(
            os.path.join(args.output_dir, "cluster_manifest.json"), "w"
        ) as handle:
            json.dump(
                {"path": args.cluster_manifest, "sha256": args.cluster_manifest_sha256},
                handle,
                indent=2,
            )
    if is_ku_optofil_dataset(args.dataset) and args.ku_unknown_bag_max_size is not None:
        assignment_manifest = {
            "mode": "known_patient_bags_plus_synthetic_unknown_subbags",
            "unknown_bag_max_size": args.ku_unknown_bag_max_size,
            "unknown_bag_seed": args.ku_unknown_bag_seed,
            "merge_validation_into_train": args.ku_merge_validation_into_train,
            "bags": [
                {
                    "bag_id": bag.bag_id,
                    "source_split": bag.split,
                    "indices": bag.indices.tolist(),
                    "class_counts": bag.class_counts.tolist(),
                    "proportions": bag.proportions.tolist(),
                }
                for bag in train_loader.dataset.bags
            ],
        }
        with open(
            os.path.join(args.output_dir, "training_bag_assignments.json"), "w"
        ) as handle:
            json.dump(assignment_manifest, handle, sort_keys=True)
    if args.skip_final_test and is_ku_optofil_dataset(args.dataset):
        test_loader = None
    else:
        test_loader = get_val_loader(
            dataset=args.dataset,
            batch_size=64,
            num_workers=args.num_workers,
            root=args.data_dir,
            n_classes=args.n_classes,
            backbone=backbone,
            seed=args.seed,
        )
    steps_per_epoch = len(train_loader)
    hparams["steps_per_epoch"] = steps_per_epoch
    n_epochs = args.epochs
    n_steps = n_epochs * steps_per_epoch
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        n_steps = min(n_steps, args.max_steps)
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        n_steps, input_dim, train_label_prob, hparams, args.bagsize
    )
    if args.algorithm == "LLP_DSQ":
        dsq_proportions = np.asarray(train_label_prob, dtype=np.float64)
        dataset_bags = getattr(train_loader.dataset, "bags", None)
        if dataset_bags is not None and len(dataset_bags) == len(dsq_proportions):
            requested_size = getattr(
                train_loader.dataset, "train_instance_sample_size", None
            )
            dsq_bag_sizes = np.asarray(
                [
                    min(len(bag.indices), int(requested_size))
                    if requested_size is not None
                    else len(bag.indices)
                    for bag in dataset_bags
                ],
                dtype=np.float64,
            )
        else:
            dsq_bag_sizes = np.full(
                len(dsq_proportions), int(args.bagsize), dtype=np.float64
            )
        if len(dsq_bag_sizes) == 0 or np.any(dsq_bag_sizes <= 0):
            raise ValueError("LLP_DSQ requires non-empty training bags")
        dsq_class_prior = np.average(dsq_proportions, axis=0, weights=dsq_bag_sizes)
        dsq_mean_k_minus_1 = float(np.mean(dsq_bag_sizes - 1.0))
        algorithm.configure_dataset_statistics(dsq_class_prior, dsq_mean_k_minus_1)
        print(
            f"LLP_DSQ dataset statistics: mean_k_minus_1={dsq_mean_k_minus_1:.6f} class_prior={dsq_class_prior.tolist()}"
        )
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
    algorithm.to(device)
    if args.data_parallel:
        visible_cuda_devices = torch.cuda.device_count()
        if device != "cuda" or visible_cuda_devices < 2:
            parser.error(
                f"--data-parallel requires at least two visible CUDA devices; found {visible_cuda_devices}"
            )
        algorithm.network = torch.nn.DataParallel(algorithm.network)
        print(
            f"Classifier data parallelism: replicas={visible_cuda_devices} logical_devices={list(range(visible_cuda_devices))}",
            flush=True,
        )
    train_minibatches_iterator = iter(train_loader)
    checkpoint_vals = collections.defaultdict(lambda: [])
    checkpoint_freq = args.checkpoint_freq

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": input_dim,
            "model_num_classes": args.n_classes,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    ku_select_validation = is_ku_optofil_dataset(args.dataset) and bool(
        hparams.get("ku_select_by_validation_macro_f1", False)
    )
    ku_completed_epoch_logging = ku_select_validation or (
        is_ku_optofil_dataset(args.dataset) and args.ku_merge_validation_into_train
    )
    ku_best_validation = None
    for step in range(start_step, n_steps):
        algorithm.train()
        step_start_time = time.time()
        if is_ku_optofil_dataset(args.dataset):
            try:
                ku_batch = next(train_minibatches_iterator)
            except StopIteration:
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(step // steps_per_epoch)
                train_minibatches_iterator = iter(train_loader)
                ku_batch = next(train_minibatches_iterator)
            ku_result = update_ku_optofil_algorithm(
                algorithm,
                args.algorithm,
                ku_batch,
                device,
                forward_chunk_size=args.forward_chunk_size,
                activation_checkpoint=bool(
                    hparams.get("ku_activation_checkpoint", False)
                ),
                iteration=step,
                softmatch_state=None,
            )
            step_vals = ku_result
            checkpoint_vals["step_time"].append(time.time() - step_start_time)
        else:
            try:
                var1, var2, var3, var4, var5 = next(train_minibatches_iterator)
            except StopIteration:
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(step // steps_per_epoch)
                train_minibatches_iterator = iter(train_loader)
                var1, var2, var3, var4, var5 = next(train_minibatches_iterator)
            var1 = var1[0]
            length = len(var2[0])
            imsw = []
            for i in range(length):
                imsw.append(var1[i])
            ims_u_weak = torch.cat(imsw, dim=0)
            imgs = ims_u_weak.to(device)
            label_proportions = []
            for i in range(length):
                label_proportions.append([var2[j][i] for j in range(args.n_classes)])
            proportion = torch.as_tensor(
                label_proportions, dtype=torch.float64, device=device
            )
            minibatches_device = (imgs, proportion)
            step_vals = algorithm.update(minibatches_device)
            checkpoint_vals["step_time"].append(time.time() - step_start_time)
        for key, val in step_vals.items():
            if key == "loss" and (not math.isfinite(float(val))):
                failure = {
                    "reason": "nonfinite_training_loss",
                    "step": step,
                    "metric": key,
                    "value": str(val),
                }
                with open(
                    os.path.join(args.output_dir, "numerical_failure.json"), "w"
                ) as f:
                    json.dump(failure, f, indent=2)
                raise FloatingPointError(f"Non-finite training loss: {failure}")
            checkpoint_vals[key].append(val)
        checkpoint_due = (
            (step + 1) % checkpoint_freq == 0
            if ku_completed_epoch_logging
            else step % checkpoint_freq == 0
        )
        if checkpoint_due or step == n_steps - 1:
            results = {"step": step, "epoch": (step + 1) / steps_per_epoch}
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            ku_val_details = None
            if is_ku_optofil_dataset(args.dataset):
                if val_loader is not None:
                    ku_val_details = evaluate_ku_optofil(
                        algorithm,
                        val_loader,
                        device,
                        forward_chunk_size=args.forward_chunk_size,
                    )
                    for key, value in ku_val_details.items():
                        if key not in {"bag_ids", "per_class", "confusion_matrix"}:
                            results[f"val_{key}"] = value
            else:
                val_PM = misc.val_PM(args, algorithm, val_loader, device)
                results["val_PM"] = val_PM
                val_easy = misc.val_easy(args, prior_all, algorithm, val_loader, device)
                results["val_easy"] = val_easy
                val_DSQ = misc.val_DSQ(args, algorithm, val_loader, device)
                results["val_DSQ"] = val_DSQ
                val_generalUPM = misc.val_generalUPM(
                    args, algorithm, val_loader, device
                )
                results["val_GeneralUPM"] = val_generalUPM
            ku_test_details = None
            if test_loader is None:
                results["test_acc"] = None
            elif is_ku_optofil_dataset(args.dataset):
                if test_loader is not None and (
                    step == n_steps - 1
                    or hparams.get("ku_test_every_checkpoint", False)
                ):
                    ku_test_details = evaluate_ku_optofil(
                        algorithm,
                        test_loader,
                        device,
                        forward_chunk_size=args.forward_chunk_size,
                    )
                    results["test_acc"] = ku_test_details["instance_accuracy"]
                    results["test_macro_f1"] = ku_test_details["macro_f1"]
                    results["test_weighted_f1"] = ku_test_details["weighted_f1"]
                    results["test_balanced_accuracy"] = ku_test_details[
                        "balanced_accuracy"
                    ]
                    results["test_bag_proportion_mae"] = ku_test_details[
                        "bag_proportion_mae"
                    ]
                    results["test_bag_proportion_rmse"] = ku_test_details[
                        "bag_proportion_rmse"
                    ]
                else:
                    ku_test_details = None
                    results["test_acc"] = None
                    results["test_macro_f1"] = None
                    results["test_weighted_f1"] = None
            else:
                results["test_acc"] = misc.accuracy(algorithm, test_loader, device)
            results["mem_gb"] = torch.cuda.max_memory_allocated() / (
                1024.0 * 1024.0 * 1024.0
            )
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)
            if is_ku_optofil_dataset(args.dataset) and ku_test_details is not None:
                results["test_per_class"] = ku_test_details["per_class"]
                results["test_confusion_matrix"] = ku_test_details["confusion_matrix"]
            if ku_select_validation:
                validation_score = float(ku_val_details["macro_f1"])
                if np.isfinite(validation_score) and (
                    ku_best_validation is None
                    or validation_score > ku_best_validation["val_macro_f1"]
                ):
                    ku_best_validation = {
                        "step": step,
                        "epoch": results["epoch"],
                        "val_macro_f1": validation_score,
                        "val_instance_accuracy": ku_val_details["instance_accuracy"],
                        "checkpoint": "model_best_val_macro_f1.pkl",
                    }
                    save_checkpoint(ku_best_validation["checkpoint"])
                    with open(
                        os.path.join(args.output_dir, "best_validation.json"), "w"
                    ) as handle:
                        json.dump(ku_best_validation, handle, indent=2)
                results["best_validation"] = (
                    dict(ku_best_validation) if ku_best_validation else None
                )
            results.update({"hparams": hparams, "args": vars(args)})
            epochs_path = os.path.join(args.output_dir, "results.jsonl")
            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, cls=misc.NpEncoder, sort_keys=True) + "\n")
            algorithm_dict = algorithm.state_dict()
            checkpoint_vals = collections.defaultdict(lambda: [])
            if args.save_model_every_checkpoint:
                save_checkpoint(f"model_step{step}.pkl")
    if is_ku_optofil_dataset(args.dataset) and args.ku_merge_validation_into_train:
        save_checkpoint("model_final.pkl")
        with open(os.path.join(args.output_dir, "final_test.json"), "w") as handle:
            json.dump(
                {
                    "selection": "final_epoch",
                    "epoch": n_epochs,
                    "test": ku_test_details,
                    "hparams": hparams,
                    "args": vars(args),
                    "split_protocol": "official_train_plus_validation_vs_official_test",
                },
                handle,
                indent=2,
                cls=misc.NpEncoder,
            )
    if ku_select_validation and ku_best_validation is not None:
        save_checkpoint("model_final.pkl")
        best_path = os.path.join(args.output_dir, ku_best_validation["checkpoint"])
        saved = torch.load(best_path, map_location=device, weights_only=False)
        algorithm.load_state_dict(saved["model_dict"])
        selected = {
            "selection": "validation_macro_f1",
            "best_validation": ku_best_validation,
            "test": evaluate_ku_optofil(
                algorithm,
                test_loader,
                device,
                forward_chunk_size=args.forward_chunk_size,
            )
            if test_loader is not None
            else None,
            "hparams": hparams,
            "args": vars(args),
        }
        with open(os.path.join(args.output_dir, "selected_test.json"), "w") as handle:
            json.dump(selected, handle, indent=2, cls=misc.NpEncoder)
    with open(
        os.path.join(
            args.output_dir, "smoke_done" if args.max_steps is not None else "done"
        ),
        "w",
    ) as f:
        f.write("done")
