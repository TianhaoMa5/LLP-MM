"""
Run sweeps
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import numpy as np
import shlex
import sys
try:
    from .lib import misc, command_launchers
except ImportError:  # Support ``python plench/sweep.py`` from the project parent.
    from lib import misc, command_launchers
import tqdm


REMOTE_SENSING_DATASETS = {"CV", "LEM"}
REMOTE_SENSING_BAG_SIZES = {32, 64, 128, 256}

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        # Flags that train accepts without a value
        flag_only = {'skip_model_save', 'save_model_every_checkpoint'}
        python_executable = os.environ.get('PLENCH_PYTHON', sys.executable)
        command = [shlex.quote(python_executable), '-m', 'plench.train']
        for k, v in sorted(self.train_args.items()):
            if k in flag_only:
                if v:
                    command.append(f'--{k}')
                continue
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            else:
                v = str(v)
            command.append(f'--{k}')
            command.append(v)
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['hparams_seed'],
            self.train_args['trial_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def _n_classes_for_dataset(dataset):
    """Return default number of classes for a dataset (for train --n-classes)."""
    if dataset in REMOTE_SENSING_DATASETS:
        return None
    if dataset in {
        "TwitterEthnicity2017",
        "twitter_ethnicity_2017",
        "twitter-race3-2017",
        "twitter_race3_2017",
    }:
        return 3
    if dataset == "CIFAR100":
        return 100
    if dataset == "EMNISTBalanced":
        return 26
    return 10


def make_args_list(
    n_trials_from,
    n_trials,
    dataset_names,
    algorithms,
    n_hparams_from,
    n_hparams,
    epochs,
    data_dir,
    holdout_fraction,
    hparams,
    skip_model_save,
    batchsize=64,
    bagsize=16,
    checkpoint_freq=1000,
    bag_build="alphafirst",
    pi=1.0,
    instances_per_epoch=200000,
    num_workers=4,
    cluster_seed=0,
):
    args_list = []
    for trial_seed in range(n_trials_from, n_trials):
        for dataset in dataset_names:
            if dataset in REMOTE_SENSING_DATASETS:
                if bagsize not in REMOTE_SENSING_BAG_SIZES:
                    raise ValueError(
                        f"{dataset} bag size must be one of "
                        f"{sorted(REMOTE_SENSING_BAG_SIZES)}; got {bagsize}"
                    )
                if bag_build not in {"random", "cluster", "alphafirst"}:
                    raise ValueError(
                        f"{dataset} sweep supports bag_build random, cluster, or alphafirst; "
                        f"got {bag_build!r}"
                    )
            for algorithm in algorithms:
                for hparams_seed in range(n_hparams_from, n_hparams):
                    train_args = {}
                    train_args["dataset"] = dataset
                    train_args["algorithm"] = algorithm
                    train_args["holdout_fraction"] = holdout_fraction
                    train_args["hparams_seed"] = hparams_seed
                    train_args["data_dir"] = data_dir
                    train_args["trial_seed"] = trial_seed
                    train_args["seed"] = misc.seed_hash(dataset, algorithm, hparams_seed, trial_seed)
                    train_args["epochs"] = epochs
                    train_args["batchsize"] = batchsize
                    train_args["bagsize"] = bagsize
                    n_classes = _n_classes_for_dataset(dataset)
                    if n_classes is not None:
                        train_args["n-classes"] = n_classes
                    train_args["checkpoint_freq"] = checkpoint_freq
                    train_args["bag_build"] = bag_build
                    train_args["pi"] = pi
                    if dataset in REMOTE_SENSING_DATASETS:
                        train_args["instances-per-epoch"] = instances_per_epoch
                        train_args["num-workers"] = num_workers
                        if bag_build == "cluster":
                            train_args["cluster-seed"] = cluster_seed
                    if hparams is not None:
                        train_args["hparams"] = hparams
                    if skip_model_save:
                        train_args["skip_model_save"] = True
                    args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete','delete_completed'])
    parser.add_argument('--datasets', nargs='+', type=str, required=True)
    parser.add_argument('--algorithms', nargs='+', type=str, required=True)
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials_from', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1024, help='Training epochs (passed to train)')
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.1)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--bagsize', type=int, default=32)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)
    parser.add_argument('--bag_build', type=str, default='alphafirst', choices=['random', 'cluster', 'alphafirst'])
    parser.add_argument('--pi', type=float, default=1.0, help='Dirichlet concentration for bag construction')
    parser.add_argument('--instances-per-epoch', '--instances_per_epoch', dest='instances_per_epoch',
                        type=int, default=200000,
                        help='CV/LEM sampled training instances per epoch')
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--cluster-seed', '--cluster_seed', dest='cluster_seed', type=int, default=0,
                        help='CV/LEM KMeans cache seed, independent of each job seed')


    args = parser.parse_args()
    args_list = make_args_list(
        n_trials_from=args.n_trials_from,
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        epochs=args.epochs,
        data_dir=args.data_dir,
        holdout_fraction=args.holdout_fraction,
        hparams=args.hparams,
        skip_model_save=args.skip_model_save,
        batchsize=args.batchsize,
        bagsize=args.bagsize,
        checkpoint_freq=args.checkpoint_freq,
        bag_build=args.bag_build,
        pi=args.pi,
        instances_per_epoch=args.instances_per_epoch,
        num_workers=args.num_workers,
        cluster_seed=args.cluster_seed,
    )
    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    # if delete incomplete
    if len([j for j in jobs if j.state == Job.INCOMPLETE]) > 0:
        for j_delete in [j for j in jobs if j.state == Job.INCOMPLETE]:
            print(j_delete)

    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)

    elif args.command == 'delete_completed':
        to_delete = [j for j in jobs if j.state == Job.DONE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)



# Run examples

### 20News
## case 1
## python sweep.py launch --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets 20News --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/20News  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets 20News --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/20News  --command_launcher dummy --skip_model_save


## case 2
# python sweep.py launch --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save


## case 3
# python sweep.py launch --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save






### CIFAR100
## case 1
# python sweep.py launch --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save


## case 2
# python sweep.py launch --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save


## case 3
# python sweep.py launch --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets CIFAR100 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR100  --command_launcher dummy --skip_model_save


### CIFAR10
## case 1
# python sweep.py launch --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

## case 2
# python sweep.py launch --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

# case 3
# python sweep.py launch --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

# case 1
# python sweep.py launch --epochs 500 --bagsize 32 --batchsize 32 --bag_build cluster --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build cluster --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

# case 2
# python sweep.py launch --epochs 500 --bagsize 128 --batchsize 8 --bag_build cluster --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build cluster --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

# case 3
# python sweep.py launch --epochs 500 --bagsize 256 --batchsize 4 --bag_build cluster --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build cluster --pi 1.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save


# case 1
# python sweep.py launch --epochs 500 --bagsize 32 --batchsize 32 --bag_build alphafirst --pi 10.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build alphafirst --pi 10.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

# case 2
# python sweep.py launch --epochs 500 --bagsize 128 --batchsize 8 --bag_build alphafirst --pi 10.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build alphafirst --pi 10.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save

# case 3
# python sweep.py launch --epochs 500 --bagsize 256 --batchsize 4 --bag_build alphafirst --pi 10.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build alphafirst --pi 10.0 --datasets CIFAR10 --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/CIFAR10  --command_launcher dummy --skip_model_save




# FashionMNIST
# case 1
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build alphafirst --pi 10.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save

# case 2
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build alphafirst --pi 10.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save

# case 3
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build alphafirst --pi 10.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save


# delete_incomplete

# case 1
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build random --pi 1.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save

# case 2
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build random --pi 1.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save

# case 3
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build random --pi 1.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save




# case 1
# python sweep.py delete_incomplete --epochs 500 --bagsize 32 --batchsize 32 --bag_build cluster --pi 1.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save

# case 2
# python sweep.py delete_incomplete --epochs 500 --bagsize 128 --batchsize 8 --bag_build cluster --pi 1.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save

# case 3
# python sweep.py delete_incomplete --epochs 500 --bagsize 256 --batchsize 4 --bag_build cluster --pi 1.0 --datasets FashionMNIST --algorithms LLP_PVC PM LLP_DSQ LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT --n_trials 3 --n_hparams 20 --output_dir /path/to/LLP-MM/plench/output/ --data_dir /path/to/data/FashionMNIST  --command_launcher dummy --skip_model_save
