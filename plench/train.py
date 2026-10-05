import argparse
import collections
import json
import os

# HuggingFace tokenizers + DataLoader fork: avoid deadlock warnings
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import sys
import time
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from .data import datasets
from .core import hparams_registry, algorithms
from .lib import misc
from .data.LLP_load import get_train_loader, get_val_loader
from .data.remote_sensing import (
    MAIN_BAG_SIZES,
    canonical_remote_dataset,
    enforce_remote_sensing_backbone,
    evaluate_remote_sensing,
    is_remote_sensing_dataset,
    load_remote_sensing_bundle,
)
from .data.twitter_ethnicity_2017 import (
    canonical_twitter_dataset,
    evaluate_twitter_race3,
    is_twitter_ethnicity_dataset,
    load_twitter_ethnicity_bundle,
)
from .data.ref2021 import (
    canonical_ref2021_dataset,
    evaluate_ref2021,
    is_ref2021_dataset,
    load_ref2021_bundle,
    update_ref2021_algorithm,
)
from .data.ku_optofil_pbc import (
    canonical_ku_optofil_dataset,
    evaluate_ku_optofil,
    is_ku_optofil_dataset,
    load_ku_optofil_bundle,
    update_ku_optofil_algorithm,
)
from .data.amazon_wilds import (
    canonical_amazon_wilds_dataset,
    evaluate_amazon_wilds,
    is_amazon_wilds_dataset,
    load_amazon_wilds_bundle,
    update_amazon_wilds_algorithm,
)
from .data.cct import (
    canonical_cct_dataset,
    evaluate_cct,
    is_cct_dataset,
    load_cct_bundle,
    update_cct_algorithm,
)
from .data.fed_isic2019 import (
    canonical_fed_isic2019_dataset,
    evaluate_fed_isic2019,
    is_fed_isic2019_dataset,
    load_fed_isic2019_bundle,
    update_fed_isic2019_algorithm,
)

# python -m plench.train --data_dir data/CIFAR10 --dataset CIFAR10 --algorithm LLP_PVC --batchsize 8 --bagsize 16 --steps 600000 --output_dir ./train_output --skip_model_save

if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description='Learning from Label Proportions')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON config file; explicit command-line flags override its values')
    parser.add_argument('--data_dir', type=str, default="data_dir")
    parser.add_argument(
        '--data_root', '--data-root', dest='data_root', type=str, default=None,
        help='dataset root containing official images and prepared caches',
    )
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--algorithm', type=str, default="LLP_ORDER")
    parser.add_argument('--hparams', type=str, default=None,
        help='JSON-serialized hparams dict, e.g. \'{"model":"ResNet","lr":0.01}\'')
    parser.add_argument('--batchsize', type=int, default=32, help='Mini-batch size for training')
    parser.add_argument('--bagsize', type=int, default=32, help='Bag size for LLP / bag-level training')
    parser.add_argument('--n-classes', type=int, default=None,
                        help='number of classes; inferred from metadata for CV/LEM')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--epochs', type=int, default=500, help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=1000, help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument(
        '--data-parallel', action='store_true',
        help='Replicate the classifier across all CUDA devices visible to this process',
    )
    parser.add_argument('--bag_build', type=str, default='random', choices=['random', 'cluster', 'alphafirst', 'feature'])
    parser.add_argument('--pi', type=float, default=1.0, help='Dirichlet concentration (alpha0). Smaller -> more dominant (peaky) bags; larger -> more uniform.')
    parser.add_argument('--num-bags', type=int, default=None,
                        help='Legacy override; remote-sensing main experiments use --instances-per-epoch')
    parser.add_argument('--num-reviewers', type=int, default=None,
                        help='Amazon-WILDS only: select complete training-reviewer bags; null uses all reviewers')
    parser.add_argument('--instances-per-epoch', type=int, default=200000,
                        help='CV/LEM sampled training instances per epoch; small bag remainder is dropped')
    parser.add_argument('--cluster-seed', type=int, default=0,
                        help='CV/LEM ClusterBag cache seed; independent of experiment seed')
    parser.add_argument('--target_avg_bag_size', '--target-avg-bag-size', type=int, default=64,
                        help='feature clustering scale; not a fixed bag size')
    parser.add_argument('--min_bag_size', '--min-bag-size', type=int, default=16,
                        help='Fed-ISIC2019 hard minimum feature-bag size')
    parser.add_argument('--max_bag_size', '--max-bag-size', type=int, default=128,
                        help='Fed-ISIC2019 hard maximum feature-bag size')
    parser.add_argument('--min_bag_size_ratio', '--min-bag-size-ratio', type=float, default=0.25,
                        help='CCT feature-bag lower size safeguard relative to the target average')
    parser.add_argument('--max_bag_size_ratio', '--max-bag-size-ratio', type=float, default=2.0,
                        help='CCT feature-bag upper size safeguard relative to the target average')
    parser.add_argument('--pca_dim', '--pca-dim', type=int, default=128,
                        help='CCT train-fitted PCA dimension recorded by preparation')
    parser.add_argument('--train-instance-sample-size', type=int, default=None,
                        help='Natural-bag datasets: optional without-replacement training instances per complete bag; null preserves the full bag')
    parser.add_argument('--ku-merge-validation-into-train', action='store_true',
                        help='KU only: train on official train + validation patients and retain the official test split')
    parser.add_argument('--ku-unknown-bag-max-size', type=int, default=None,
                        help='KU only: split the missing-patient-ID training group into balanced sub-bags; known patients and test unchanged')
    parser.add_argument('--ku-unknown-bag-seed', type=int, default=0,
                        help='KU unknown-group assignment seed, independent of model seed')
    parser.add_argument('--forward-chunk-size', type=int, default=32,
                        help='KU-Optofil train/evaluation device-forward chunk size; patient bags remain complete')
    parser.add_argument('--feature-encoder', type=str, default=None,
                        help='REF2021 frozen feature encoder recorded in the cache manifest')
    parser.add_argument('--feature-cache', action='store_true',
                        help='REF2021 uses a frozen on-disk feature cache')
    parser.add_argument('--natural-bags', action='store_true',
                        help='Declare that dataset bags are natural rather than synthetic')
    parser.add_argument('--variable-bag-size', action='store_true',
                        help='Declare variable-size natural bags')
    parser.add_argument('--uoa', type=int, default=None)
    parser.add_argument('--split-seed', type=int, default=None)
    parser.add_argument('--skip-final-test', action='store_true',
                        help='Skip instance-level final test evaluation for short diagnostic runs')
    parser.add_argument('--num-workers', type=int, default=4)

    if config_args.config:
        with open(config_args.config, encoding='utf-8') as config_file:
            config_defaults = json.load(config_file)
        if not isinstance(config_defaults, dict):
            parser.error('--config must contain one JSON object')
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config_defaults) - valid_dests)
        if unknown:
            parser.error(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if is_fed_isic2019_dataset(args.dataset):
        args.dataset = canonical_fed_isic2019_dataset(args.dataset)
        if args.data_root is not None:
            args.data_dir = args.data_root
        metadata = load_fed_isic2019_bundle(args.data_dir)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with Fed-ISIC2019 "
                f"({metadata.num_classes})"
            )
        bag_manifest = metadata.metadata["bag_manifest"]
        requested_values = {
            "min_bag_size": int(args.min_bag_size),
            "max_bag_size": int(args.max_bag_size),
            "target_bag_size": int(args.target_avg_bag_size),
            "seed": int(args.seed),
        }
        cached_values = {
            key: int(bag_manifest[key]) for key in requested_values
        }
        if requested_values != cached_values:
            parser.error(
                "Fed-ISIC2019 training parameters disagree with the prepared bag "
                f"cache: requested={requested_values}, cached={cached_values}"
            )
        if args.bag_build != "feature":
            parser.error("Fed-ISIC2019 requires --bag_build feature")
        if args.train_instance_sample_size is not None:
            parser.error("Fed-ISIC2019 feature bags cannot be truncated")
        cached_encoder = bag_manifest["encoder"]
        if args.feature_encoder is not None and args.feature_encoder != cached_encoder:
            parser.error(
                f"--feature-encoder={args.feature_encoder!r} disagrees with cache "
                f"encoder {cached_encoder!r}"
            )
        args.bagsize = args.target_avg_bag_size
    elif is_cct_dataset(args.dataset):
        args.dataset = canonical_cct_dataset(args.dataset)
        if args.data_root is not None:
            args.data_dir = args.data_root
        metadata = load_cct_bundle(args.data_dir)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with CCT "
                f"({metadata.num_classes})"
            )
        bag_manifest = metadata.metadata["bag_manifest"]
        pca_manifest = metadata.metadata["pca_manifest"]
        cached_values = {
            "target_avg_bag_size": int(bag_manifest["target_avg_bag_size"]),
            "min_bag_size_ratio": float(bag_manifest["min_bag_size_ratio"]),
            "max_bag_size_ratio": float(bag_manifest["max_bag_size_ratio"]),
            "pca_dim": int(pca_manifest["requested_pca_dim"]),
        }
        requested_values = {
            "target_avg_bag_size": int(args.target_avg_bag_size),
            "min_bag_size_ratio": float(args.min_bag_size_ratio),
            "max_bag_size_ratio": float(args.max_bag_size_ratio),
            "pca_dim": int(args.pca_dim),
        }
        if requested_values != cached_values:
            parser.error(
                "CCT training parameters disagree with the immutable prepared cache: "
                f"requested={requested_values}, cached={cached_values}"
            )
        if args.bag_build != "feature":
            parser.error("CCT requires --bag_build feature; random bags are forbidden")
        if args.train_instance_sample_size is not None:
            parser.error("CCT feature bags cannot be truncated")
        cached_encoder = metadata.metadata["feature_manifest"]["encoder"]
        if args.feature_encoder is not None and args.feature_encoder != cached_encoder:
            parser.error(
                f"--feature-encoder={args.feature_encoder!r} disagrees with cache "
                f"encoder {cached_encoder!r}"
            )
        # Legacy algorithm constructors still require one scalar ``bagsize``.
        # For CCT this is only a nominal scale; every loss update receives the
        # real per-bag sizes from the immutable variable-size cache.
        args.bagsize = args.target_avg_bag_size
    elif is_amazon_wilds_dataset(args.dataset):
        args.dataset = canonical_amazon_wilds_dataset(args.dataset)
        metadata = load_amazon_wilds_bundle(args.data_dir, require_features=True)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with Amazon-WILDS "
                f"({metadata.num_classes})"
            )
        if args.train_instance_sample_size is not None:
            parser.error(
                'Amazon-WILDS preserves complete reviewer bags and does not support '
                '--train-instance-sample-size'
            )
        if args.num_reviewers is not None and args.num_reviewers <= 0:
            parser.error('--num-reviewers must be positive or omitted/null')
        cached_encoder = metadata.metadata['feature_manifest']['encoder']
        if args.feature_encoder is not None and args.feature_encoder != cached_encoder:
            parser.error(
                f"--feature-encoder={args.feature_encoder!r} disagrees with cache "
                f"encoder {cached_encoder!r}"
            )
    elif is_ku_optofil_dataset(args.dataset):
        args.dataset = canonical_ku_optofil_dataset(args.dataset)
        metadata = load_ku_optofil_bundle(args.data_dir)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with KU-Optofil "
                f"({metadata.num_classes})"
            )
        if args.train_instance_sample_size is not None and args.train_instance_sample_size <= 0:
            parser.error('--train-instance-sample-size must be positive or omitted/null')
        if args.forward_chunk_size <= 0:
            parser.error('--forward-chunk-size must be positive')
    elif is_ref2021_dataset(args.dataset):
        args.dataset = canonical_ref2021_dataset(args.dataset)
        metadata = load_ref2021_bundle(args.data_dir, require_features=True)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with REF2021 "
                f"({metadata.num_classes})"
            )
        if args.train_instance_sample_size is not None and args.train_instance_sample_size <= 0:
            parser.error('--train-instance-sample-size must be positive or omitted/null')
        if args.uoa not in {None, 11}:
            parser.error('REF2021UOA11 requires uoa=11')
        if args.split_seed is not None and args.split_seed != int(metadata.metadata['split_seed']):
            parser.error(
                f"--split-seed={args.split_seed} disagrees with processed split "
                f"seed {metadata.metadata['split_seed']}"
            )
        cached_encoder = metadata.metadata['feature_manifest']['encoder']
        if args.feature_encoder is not None and args.feature_encoder != cached_encoder:
            parser.error(
                f"--feature-encoder={args.feature_encoder!r} disagrees with cache "
                f"encoder {cached_encoder!r}"
            )
    elif is_twitter_ethnicity_dataset(args.dataset):
        args.dataset = canonical_twitter_dataset(args.dataset)
        metadata = load_twitter_ethnicity_bundle(args.data_dir, require_features=False)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with Twitter Race3 "
                f"({metadata.num_classes})"
            )
    elif is_remote_sensing_dataset(args.dataset):
        args.dataset = canonical_remote_dataset(args.dataset)
        if args.bagsize not in MAIN_BAG_SIZES:
            parser.error(f"CV/LEM bagsize must be one of {MAIN_BAG_SIZES}; got {args.bagsize}")
        metadata = load_remote_sensing_bundle(args.dataset, args.data_dir, seed=args.seed)
        if args.n_classes is None:
            args.n_classes = metadata.num_classes
        elif args.n_classes != metadata.num_classes:
            parser.error(
                f"--n-classes={args.n_classes} disagrees with {args.dataset} metadata "
                f"({metadata.num_classes})"
            )
    elif args.n_classes is None:
        args.n_classes = 10
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    done_file = os.path.join(args.output_dir, 'done')
    if os.path.exists(done_file):
        print(f"Output already exists at {args.output_dir}, skipping.")
        sys.exit(0)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.ku_merge_validation_into_train and hparams.get('ku_select_by_validation_macro_f1', False):
        parser.error('KU train+test protocol cannot select a model using validation')
    if is_cct_dataset(args.dataset):
        cct_backbone_configs = {
            "CCTResNet18": True,
            # This is the exact ResNet-18 setting in the supplied supplement:
            # weights=None, 3x3/stride-1 stem, and no max-pool.
            "ResNet": False,
        }
        model_name = hparams.get("model")
        expected_pretrained = cct_backbone_configs.get(model_name)
        if (
            expected_pretrained is None
            or hparams.get("pretrained") is not expected_pretrained
            or hparams.get("input_resolution") != 112
        ):
            parser.error(
                "CCT bbox-crop experiments require one of two controlled ResNet-18@112 "
                "configurations: CCTResNet18 with pretrained=true, or the supplied-"
                "supplement ResNet modified stem with pretrained=false; received="
                f"{{'model': {model_name!r}, 'pretrained': {hparams.get('pretrained')!r}, "
                f"'input_resolution': {hparams.get('input_resolution')!r}}}"
            )
    try:
        enforce_remote_sensing_backbone(args.dataset, hparams)
    except ValueError as exc:
        parser.error(str(exc))
    hparams['n_classes'] = args.n_classes
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if args.algorithm in {"LLP_AHIL", "LLP_DC", "LLP_SoftMatch", "LLP_FixMatch"}:
        method = "L^2P-AHIL"
    else:
        method = "DLLP"
    backbone = hparams.get('model')
    train_loader, val_loader, train_label_prob, dataset_length, dataset_length_var, input_dim, prior_train, prior_val, prior_all = get_train_loader(
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
        instances_per_epoch=args.instances_per_epoch,
        train_instance_sample_size=args.train_instance_sample_size,
        num_reviewers=args.num_reviewers,
        cluster_seed=args.cluster_seed,
        target_avg_bag_size=args.target_avg_bag_size,
        ku_merge_validation_into_train=args.ku_merge_validation_into_train,
        ku_unknown_bag_max_size=args.ku_unknown_bag_max_size,
        ku_unknown_bag_seed=args.ku_unknown_bag_seed,
    )
    if is_ku_optofil_dataset(args.dataset) and args.ku_unknown_bag_max_size is not None:
        assignment_manifest = {
            'mode': 'known_patient_bags_plus_synthetic_unknown_subbags',
            'unknown_bag_max_size': args.ku_unknown_bag_max_size,
            'unknown_bag_seed': args.ku_unknown_bag_seed,
            'merge_validation_into_train': args.ku_merge_validation_into_train,
            'bags': [
                {'bag_id': bag.bag_id, 'source_split': bag.split,
                 'indices': bag.indices.tolist(),
                 'class_counts': bag.class_counts.tolist(),
                 'proportions': bag.proportions.tolist()}
                for bag in train_loader.dataset.bags
            ],
        }
        with open(os.path.join(args.output_dir, 'training_bag_assignments.json'), 'w') as handle:
            json.dump(assignment_manifest, handle, sort_keys=True)
    if args.algorithm == "NonClipOVR":
        loader_batch_size = getattr(train_loader, "batch_size", None)
        if loader_batch_size is None:
            loader_batch_size = getattr(
                getattr(train_loader, "batch_sampler", None), "batch_size", None
            )
        if loader_batch_size is not None and int(loader_batch_size) < 2:
            raise ValueError(
                "NonClipOVR requires at least 2 bags per training minibatch "
                "for the leave-one-bag-out prediction mean; increase --batchsize"
            )
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "drop_last"):
            batch_sampler.drop_last = True
    if args.skip_final_test and (
        is_remote_sensing_dataset(args.dataset)
        or is_fed_isic2019_dataset(args.dataset)
        or is_cct_dataset(args.dataset)
        or is_ref2021_dataset(args.dataset)
        or is_ku_optofil_dataset(args.dataset)
        or is_amazon_wilds_dataset(args.dataset)
    ):
        test_loader = None
    else:
        test_loader = get_val_loader(
            dataset=args.dataset, batch_size=64, num_workers=args.num_workers,
            root=args.data_dir, n_classes=args.n_classes, backbone=backbone, seed=args.seed,
        )

    steps_per_epoch = len(train_loader)
    hparams['steps_per_epoch'] = steps_per_epoch
    n_epochs = args.epochs
    n_steps = n_epochs * steps_per_epoch

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(n_steps,input_dim, train_label_prob, hparams,args.bagsize)

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
        dsq_class_prior = np.average(
            dsq_proportions, axis=0, weights=dsq_bag_sizes
        )
        dsq_mean_k_minus_1 = float(np.mean(dsq_bag_sizes - 1.0))
        algorithm.configure_dataset_statistics(
            dsq_class_prior, dsq_mean_k_minus_1
        )
        print(
            "LLP_DSQ dataset statistics: "
            f"mean_k_minus_1={dsq_mean_k_minus_1:.6f} "
            f"class_prior={dsq_class_prior.tolist()}"
        )

    if args.algorithm == "NonClipOVR":
        nonclip_proportions = np.asarray(train_label_prob, dtype=np.float64)
        dataset_bags = getattr(train_loader.dataset, "bags", None)
        if (
            dataset_bags is not None
            and len(dataset_bags) == len(nonclip_proportions)
        ):
            requested_size = getattr(
                train_loader.dataset, "train_instance_sample_size", None
            )
            nonclip_bag_sizes = np.asarray(
                [
                    min(len(bag.indices), int(requested_size))
                    if requested_size is not None
                    else len(bag.indices)
                    for bag in dataset_bags
                ],
                dtype=np.float64,
            )
        else:
            nonclip_bag_sizes = np.full(
                len(nonclip_proportions), int(args.bagsize), dtype=np.float64
            )
        if len(nonclip_bag_sizes) == 0 or np.any(nonclip_bag_sizes <= 0):
            raise ValueError("NonClipOVR requires non-empty training bags")
        nonclip_class_prior = np.average(
            nonclip_proportions, axis=0, weights=nonclip_bag_sizes
        )
        algorithm.configure_dataset_prior(nonclip_class_prior)
        print(
            "NonClipOVR pooled training prior: "
            f"{nonclip_class_prior.tolist()}"
        )

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if args.data_parallel:
        visible_cuda_devices = torch.cuda.device_count()
        if device != "cuda" or visible_cuda_devices < 2:
            parser.error(
                "--data-parallel requires at least two visible CUDA devices; "
                f"found {visible_cuda_devices}"
            )
        algorithm.network = torch.nn.DataParallel(algorithm.network)
        print(
            "Classifier data parallelism: "
            f"replicas={visible_cuda_devices} logical_devices="
            f"{list(range(visible_cuda_devices))}",
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
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    ku_select_validation = is_ku_optofil_dataset(args.dataset) and bool(
        hparams.get('ku_select_by_validation_macro_f1', False)
    )
    ku_completed_epoch_logging = ku_select_validation or (
        is_ku_optofil_dataset(args.dataset) and args.ku_merge_validation_into_train
    )
    ku_best_validation = None
    ulb_prob_t = torch.ones((args.n_classes)).to(device) / args.n_classes
    prob_max_mu_t = 1.0 / args.n_classes
    prob_max_var_t = 1.0
    for step in range(start_step, n_steps):
        algorithm.train()
        step_start_time = time.time()
        if is_fed_isic2019_dataset(args.dataset):
            try:
                fed_isic_batch = next(train_minibatches_iterator)
            except StopIteration:
                train_minibatches_iterator = iter(train_loader)
                fed_isic_batch = next(train_minibatches_iterator)
            fed_isic_result = update_fed_isic2019_algorithm(
                algorithm,
                args.algorithm,
                fed_isic_batch,
                device,
                forward_chunk_size=args.forward_chunk_size,
                iteration=step,
                softmatch_state=(
                    ulb_prob_t, prob_max_mu_t, prob_max_var_t
                ) if args.algorithm == "LLP_SoftMatch" else None,
            )
            if args.algorithm == "LLP_SoftMatch":
                step_vals, softmatch_state = fed_isic_result
                ulb_prob_t, prob_max_mu_t, prob_max_var_t = softmatch_state
            else:
                step_vals = fed_isic_result
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        elif is_cct_dataset(args.dataset):
            try:
                cct_batch = next(train_minibatches_iterator)
            except StopIteration:
                train_minibatches_iterator = iter(train_loader)
                cct_batch = next(train_minibatches_iterator)
            cct_result = update_cct_algorithm(
                algorithm,
                args.algorithm,
                cct_batch,
                device,
                forward_chunk_size=args.forward_chunk_size,
                iteration=step,
                softmatch_state=(
                    ulb_prob_t, prob_max_mu_t, prob_max_var_t
                ) if args.algorithm == "LLP_SoftMatch" else None,
            )
            if args.algorithm == "LLP_SoftMatch":
                step_vals, softmatch_state = cct_result
                ulb_prob_t, prob_max_mu_t, prob_max_var_t = softmatch_state
            else:
                step_vals = cct_result
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        elif is_amazon_wilds_dataset(args.dataset):
            try:
                amazon_batch = next(train_minibatches_iterator)
            except StopIteration:
                train_minibatches_iterator = iter(train_loader)
                amazon_batch = next(train_minibatches_iterator)
            step_vals = update_amazon_wilds_algorithm(
                algorithm, args.algorithm, amazon_batch, device
            )
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        elif is_ku_optofil_dataset(args.dataset):
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
                activation_checkpoint=bool(hparams.get('ku_activation_checkpoint', False)),
                iteration=step,
                softmatch_state=(
                    ulb_prob_t, prob_max_mu_t, prob_max_var_t
                ) if args.algorithm == "LLP_SoftMatch" else None,
            )
            if args.algorithm == "LLP_SoftMatch":
                step_vals, softmatch_state = ku_result
                ulb_prob_t, prob_max_mu_t, prob_max_var_t = softmatch_state
            else:
                step_vals = ku_result
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        elif is_ref2021_dataset(args.dataset):
            try:
                ref_batch = next(train_minibatches_iterator)
            except StopIteration:
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(step // steps_per_epoch)
                train_minibatches_iterator = iter(train_loader)
                ref_batch = next(train_minibatches_iterator)
            step_vals = update_ref2021_algorithm(
                algorithm, args.algorithm, ref_batch, device
            )
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        elif args.algorithm not in {"LLP_AHIL", "LLP_DC", "LLP_SoftMatch", "LLP_FixMatch"}:
            try:
                (var1, var2,var3,var4,var5 ) = next(train_minibatches_iterator)
            except StopIteration:
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(step // steps_per_epoch)
                train_minibatches_iterator = iter(train_loader)
                (var1, var2,var3,var4,var5) = next(train_minibatches_iterator)

            # var1 是一个 list/tuple，里面第 0 个是 weak 的 bag tensor: [length, bagsize, C, H, W] 或类似
            var1 = var1[0]  # 取 weak
            # length = batch 里有多少个 bag（你原来用 var2[0] 的长度）
            length = len(var2[0])
            if is_twitter_ethnicity_dataset(args.dataset):
                # Twitter county chunks are variable-size and are loaded one bag
                # at a time. Existing PLeNCH objectives can therefore retain
                # their fixed-size math while using the current natural chunk.
                algorithm.bagsize = int(var1.shape[1])

            # ---- imgs：把 length 个 bag 的 weak 拼成 [length*bagsize, C, H, W] ----
            imsw = []
            for i in range(length):
                imsw.append(var1[i])  # var1[i]: [bagsize, C, H, W]（或 MNIST 那种）
            ims_u_weak = torch.cat(imsw, dim=0)  # [length*bagsize, C, H, W]
            # MNIST 系列你的代码里要 permute 一下就保留
            if args.dataset in ["MNIST", "FashionMNIST", "KMNIST", "EMNISTBalanced"]:
                ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)

            imgs = ims_u_weak.to(device)
            # ---- proportion：从 var2 组装成 [length, n_classes] ----
            # var2 的结构：var2[j][i] = 第 i 个 bag 的第 j 类比例
            label_proportions = []
            for i in range(length):
                label_proportions.append([var2[j][i] for j in range(args.n_classes)])

            proportion = torch.as_tensor(label_proportions, dtype=torch.float64, device=device)
            if args.algorithm == 'LLP_VAT':
                minibatches_device = (imgs, proportion,step)
            else:
                minibatches_device = (imgs, proportion)


            step_vals = algorithm.update(minibatches_device)

            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        else:
            try:
                (var1, var2, var3, var4, var5) = next(train_minibatches_iterator)
            except StopIteration:
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(step // steps_per_epoch)
                train_minibatches_iterator = iter(train_loader)
                (var1, var2, var3, var4, var5) = next(train_minibatches_iterator)

            # var1 是一个 list/tuple，里面第 0 个是 weak 的 bag tensor: [length, bagsize, C, H, W] 或类似
            # length = batch 里有多少个 bag（你原来用 var2[0] 的长度）
            length = len(var2[0])

            # ---- imgs：把 length 个 bag 的 weak 拼成 [length*bagsize, C, H, W] ----
            imsw = []
            ims_u_weak1,ims_u_strong0 = var1
            if is_twitter_ethnicity_dataset(args.dataset):
                algorithm.bagsize = int(ims_u_weak1.shape[1])
            imsw, imss0, labels_real, labels_idx, indices_u = [], [], [], [], []

            for i in range(length):
                imsw.append(ims_u_weak1[i])
                imss0.append(ims_u_strong0[i])
                labels_real.append(var3[i])
                labels_idx.append(var4[i])
                indices_u.append(var5[i])
            ims_u_weak = torch.cat(imsw, dim=0)
            ims_u_strong0 = torch.cat(imss0, dim=0)

            # MNIST 系列你的代码里要 permute 一下就保留
            if args.dataset in ["MNIST", "FashionMNIST", "KMNIST", "EMNISTBalanced"]:
                ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)

            imgs = torch.cat([ims_u_weak, ims_u_strong0], dim=0).to(device)
            # ---- proportion：从 var2 组装成 [length, n_classes] ----
            # var2 的结构：var2[j][i] = 第 i 个 bag 的第 j 类比例
            label_proportions = []
            for i in range(length):
                label_proportions.append([var2[j][i] for j in range(args.n_classes)])

            proportion = torch.as_tensor(label_proportions, dtype=torch.float32, device=device)  # [length, n_classes]
            # 你之前用 double 就：
            # proportion = proportion.double()

            # 现在你只要这俩
            if args.algorithm == 'LLP_SoftMatch':
                minibatches_device = (imgs, proportion, ulb_prob_t, prob_max_mu_t, prob_max_var_t)
                step_vals,ulb_prob_t, prob_max_mu_t, prob_max_var_t = algorithm.update(minibatches_device)

            else:
                minibatches_device = (imgs, proportion)
                step_vals = algorithm.update(minibatches_device)

            # 调算法（update 里也要对应接收 (imgs, proportion)）

            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)


        checkpoint_due = (
            (step + 1) % checkpoint_freq == 0
            if ku_completed_epoch_logging else step % checkpoint_freq == 0
        )
        if checkpoint_due or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': (step + 1) / steps_per_epoch if ku_completed_epoch_logging else step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            ref_val_details = None
            ku_val_details = None
            amazon_val_details = None
            cct_val_details = None
            if is_fed_isic2019_dataset(args.dataset):
                # FLamby exposes a fixed train/test split and no validation set.
                pass
            elif is_cct_dataset(args.dataset):
                # There is intentionally no CCT validation loader: official
                # train/cis-val/trans-val data are all training bags.
                pass
            elif is_amazon_wilds_dataset(args.dataset):
                amazon_val_details = evaluate_amazon_wilds(
                    algorithm, val_loader, device
                )
                for key, value in amazon_val_details.items():
                    if key not in {"per_class", "confusion_matrix"}:
                        results[f"val_{key}"] = value
            elif is_ku_optofil_dataset(args.dataset):
                if val_loader is not None:
                    ku_val_details = evaluate_ku_optofil(
                        algorithm, val_loader, device,
                        forward_chunk_size=args.forward_chunk_size,
                    )
                    for key, value in ku_val_details.items():
                        if key not in {"bag_ids", "per_class", "confusion_matrix"}:
                            results[f"val_{key}"] = value
            elif is_ref2021_dataset(args.dataset):
                ref_val_details = evaluate_ref2021(algorithm, val_loader, device)
                for key, value in ref_val_details.items():
                    if key not in {"bag_ids", "per_class_mae"}:
                        results[f"val_{key}"] = value
                results['val_per_class_mae'] = ref_val_details['per_class_mae']
            else:
                val_PM = misc.val_PM(args,algorithm, val_loader, device)
                results['val_PM'] = val_PM
                val_easy = misc.val_easy(args,prior_all,algorithm, val_loader, device)
                results['val_easy'] = val_easy
                val_DSQ = misc.val_DSQ(args,algorithm, val_loader, device)
                results['val_DSQ'] = val_DSQ

                val_generalUPM = misc.val_generalUPM(args, algorithm, val_loader, device)
                results['val_GeneralUPM'] = val_generalUPM

            remote_metric_details = None
            ku_test_details = None
            if test_loader is None:
                results['test_acc'] = None
            elif is_fed_isic2019_dataset(args.dataset):
                # Monitoring only: test labels never affect optimization or scheduling.
                fed_isic_test_details = evaluate_fed_isic2019(
                    algorithm,
                    test_loader,
                    device,
                    forward_chunk_size=args.forward_chunk_size,
                )
                results['test_monitoring_only'] = True
                results['test_acc'] = fed_isic_test_details['instance_accuracy']
                results['test_macro_f1'] = fed_isic_test_details['macro_f1']
                results['test_weighted_f1'] = fed_isic_test_details['weighted_f1']
                results['test_bag_proportion_mae'] = fed_isic_test_details['bag_proportion_mae']
                results['test_bag_proportion_rmse'] = fed_isic_test_details['bag_proportion_rmse']
                results['test_per_class'] = fed_isic_test_details['per_class']
                results['test_confusion_matrix'] = fed_isic_test_details['confusion_matrix']
            elif is_cct_dataset(args.dataset):
                # User-requested monitoring only: these test metrics are never
                # fed into training, checkpoint selection, or the scheduler.
                cct_test_details = evaluate_cct(
                    algorithm,
                    test_loader,
                    device,
                    class_names=metadata.class_names,
                    forward_chunk_size=args.forward_chunk_size,
                )
                results['test_monitoring_only'] = True
                results['test_acc'] = cct_test_details['instance_accuracy']
                results['test_macro_f1'] = cct_test_details['macro_f1']
                results['test_weighted_f1'] = cct_test_details['weighted_f1']
                results['test_bag_proportion_mae'] = cct_test_details['bag_proportion_mae']
                results['test_bag_proportion_rmse'] = cct_test_details['bag_proportion_rmse']
                results['test_per_class'] = cct_test_details['per_class']
                results['test_confusion_matrix'] = cct_test_details['confusion_matrix']
            elif is_amazon_wilds_dataset(args.dataset):
                if step == n_steps - 1:
                    amazon_test_details = evaluate_amazon_wilds(
                        algorithm, test_loader, device
                    )
                    results['test_acc'] = amazon_test_details['overall_accuracy']
                    results['test_balanced_accuracy'] = amazon_test_details['balanced_accuracy']
                    results['test_macro_f1'] = amazon_test_details['macro_f1']
                    results['test_per_class'] = amazon_test_details['per_class']
                    results['test_confusion_matrix'] = amazon_test_details['confusion_matrix']
                else:
                    results['test_acc'] = None
                    results['test_balanced_accuracy'] = None
                    results['test_macro_f1'] = None
            elif is_ku_optofil_dataset(args.dataset):
                if test_loader is not None and (
                    step == n_steps - 1 or hparams.get('ku_test_every_checkpoint', False)
                ):
                    ku_test_details = evaluate_ku_optofil(
                        algorithm,
                        test_loader,
                        device,
                        forward_chunk_size=args.forward_chunk_size,
                    )
                    results['test_acc'] = ku_test_details['instance_accuracy']
                    results['test_macro_f1'] = ku_test_details['macro_f1']
                    results['test_weighted_f1'] = ku_test_details['weighted_f1']
                    results['test_balanced_accuracy'] = ku_test_details['balanced_accuracy']
                    results['test_bag_proportion_mae'] = ku_test_details['bag_proportion_mae']
                    results['test_bag_proportion_rmse'] = ku_test_details['bag_proportion_rmse']
                else:
                    ku_test_details = None
                    results['test_acc'] = None
                    results['test_macro_f1'] = None
                    results['test_weighted_f1'] = None
            elif is_ref2021_dataset(args.dataset):
                ref_test_details = evaluate_ref2021(algorithm, test_loader, device)
                for key, value in ref_test_details.items():
                    if key not in {"bag_ids", "per_class_mae"}:
                        results[f"test_{key}"] = value
                results['test_per_class_mae'] = ref_test_details['per_class_mae']
                # REF publishes no individual output star ratings.
                results['test_acc'] = None
                results['test_macro_f1'] = None
            elif is_twitter_ethnicity_dataset(args.dataset):
                twitter_metrics = evaluate_twitter_race3(algorithm, test_loader, device)
                results.update({f"test_{key}": value for key, value in twitter_metrics.items()})
                results['test_acc'] = twitter_metrics['accuracy']
            elif is_remote_sensing_dataset(args.dataset):
                # The centre-level test population can contain millions of
                # patches. Evaluate it once at the final checkpoint, not at
                # every training log checkpoint.
                if step == n_steps - 1:
                    class_names = [metadata.idx_to_class[index] for index in range(metadata.num_classes)]
                    remote_metric_details = evaluate_remote_sensing(
                        algorithm, test_loader, device, class_names
                    )
                    results['test_acc'] = remote_metric_details['overall_accuracy']
                    results['test_balanced_accuracy'] = remote_metric_details['balanced_accuracy']
                    results['test_macro_f1'] = remote_metric_details['macro_f1']
                else:
                    results['test_acc'] = None
                    results['test_balanced_accuracy'] = None
                    results['test_macro_f1'] = None
            else:
                results['test_acc'] = misc.accuracy(algorithm, test_loader, device)
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            if remote_metric_details is not None:
                results['test_per_class'] = remote_metric_details['per_class']
                results['test_confusion_matrix'] = remote_metric_details['confusion_matrix']
            if is_ku_optofil_dataset(args.dataset) and ku_test_details is not None:
                results['test_per_class'] = ku_test_details['per_class']
                results['test_confusion_matrix'] = ku_test_details['confusion_matrix']
            if ku_select_validation:
                validation_score = float(ku_val_details['macro_f1'])
                if np.isfinite(validation_score) and (
                    ku_best_validation is None or validation_score > ku_best_validation['val_macro_f1']
                ):
                    ku_best_validation = {
                        'step': step,
                        'epoch': results['epoch'],
                        'val_macro_f1': validation_score,
                        'val_instance_accuracy': ku_val_details['instance_accuracy'],
                        'checkpoint': 'model_best_val_macro_f1.pkl',
                    }
                    save_checkpoint(ku_best_validation['checkpoint'])
                    with open(os.path.join(args.output_dir, 'best_validation.json'), 'w') as handle:
                        json.dump(ku_best_validation, handle, indent=2)
                results['best_validation'] = dict(ku_best_validation) if ku_best_validation else None

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, cls=misc.NpEncoder, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    if is_ku_optofil_dataset(args.dataset) and args.ku_merge_validation_into_train:
        save_checkpoint('model_final.pkl')
        with open(os.path.join(args.output_dir, 'final_test.json'), 'w') as handle:
            json.dump({
                'selection': 'final_epoch', 'epoch': n_epochs,
                'test': ku_test_details, 'hparams': hparams, 'args': vars(args),
                'split_protocol': 'official_train_plus_validation_vs_official_test',
            }, handle, indent=2, cls=misc.NpEncoder)
    if ku_select_validation and ku_best_validation is not None:
        save_checkpoint('model_final.pkl')
        best_path = os.path.join(args.output_dir, ku_best_validation['checkpoint'])
        saved = torch.load(best_path, map_location=device, weights_only=False)
        algorithm.load_state_dict(saved['model_dict'])
        selected = {
            'selection': 'validation_macro_f1',
            'best_validation': ku_best_validation,
            'test': evaluate_ku_optofil(
                algorithm, test_loader, device, forward_chunk_size=args.forward_chunk_size
            ) if test_loader is not None else None,
            'hparams': hparams,
            'args': vars(args),
        }
        with open(os.path.join(args.output_dir, 'selected_test.json'), 'w') as handle:
            json.dump(selected, handle, indent=2, cls=misc.NpEncoder)
    # save_checkpoint('model.pkl')
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
