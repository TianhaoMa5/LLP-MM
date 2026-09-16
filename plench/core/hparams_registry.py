import numpy as np
from ..lib import misc

MLP_DATASET = [
    "TwitterEthnicity2017",
    "REF2021UOA11",
    "AmazonWILDS",
]

RESNET_DATASET = [
    "CIFAR10",
    "KUOptofilPBC",
    "CCT",
    "FedISIC2019",
]

TEXT_DATASET = ["AGNEWS", "Yelp", "Yahoo", "20News"]
REMOTE_SENSING_DATASET = ["CV", "LEM"]


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    if dataset in MLP_DATASET:
        _hparam('model', 'MLP', lambda r: 'MLP')
    elif dataset == "FedISIC2019":
        _hparam('model', 'ImageNetResNet18', lambda r: 'ImageNetResNet18')
        _hparam('pretrained', True, lambda r: True)
        _hparam('input_resolution', 224, lambda r: 224)
    elif dataset == "CCT":
        _hparam('model', 'CCTResNet18', lambda r: 'CCTResNet18')
        _hparam('pretrained', True, lambda r: True)
        _hparam('input_resolution', 112, lambda r: 112)
    elif dataset in RESNET_DATASET or dataset == "CIFAR100":
        _hparam('model', 'ResNet', lambda r: 'ResNet')
    elif dataset == "miniImageNet":
        _hparam('model', 'ResNet_miniImageNet', lambda r: 'ResNet_miniImageNet')
    elif dataset in TEXT_DATASET:
        # Default to BERT-base; override with --hparams '{"model":"BERT_small"}' if needed
        _hparam('model', 'BERT_base', lambda r: r.choice(['BERT_base', 'BERT_small']))
        # Number of bottom transformer layers to freeze (0 = full fine-tune)
        _hparam('bert_freeze_layers', 0, lambda r: int(r.choice([0, 6])))
    elif dataset in REMOTE_SENSING_DATASET:
        _hparam('model', 'RemoteResNet18', lambda r: 'RemoteResNet18')

    if dataset == "CCT":
        _hparam('lr', 5e-3 if algorithm == 'LLP_PVC' else 5e-2,
                lambda r: 5e-3 if algorithm == 'LLP_PVC' else 5e-2)
    elif dataset in {"REF2021UOA11", "AmazonWILDS", "KUOptofilPBC", "FedISIC2019"}:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4, -2.5))
    elif algorithm == 'LLP_PVC':
        _hparam('lr', 5e-3, lambda r: 5e-3)
    elif dataset in TEXT_DATASET:
        # BERT typically needs a much smaller LR
        _hparam('lr', 2e-5, lambda r: 10 ** r.uniform(-5, -4))
    else:
        _hparam('lr', 5e-2, lambda r: 5e-2)

    _hparam('weight_decay', 1e-2 if dataset in TEXT_DATASET else 5e-4,
            lambda r: 5e-4)
    if dataset == "CCT":
        _hparam('optimizer', 'SGD', lambda r: 'SGD')
        _hparam('momentum', 0.9, lambda r: 0.9)
        _hparam('nesterov', False, lambda r: False)
        _hparam('warmup_fraction', 0.08, lambda r: 0.08)
        _hparam('warmup_ratio', 0.0, lambda r: 0.0)
        _hparam('warmup', 'linear', lambda r: 'linear')
        _hparam('cosine_mode', 'standard', lambda r: 'standard')
        _hparam('batch_size', 1024, lambda r: 1024)
    elif dataset == "KUOptofilPBC":
        _hparam('batch_size', 1, lambda r: int(r.choice([1, 2])))
    elif dataset == "FedISIC2019":
        _hparam('batch_size', 512, lambda r: int(r.choice([256, 512, 1024])))
    elif dataset in RESNET_DATASET:
        _hparam('batch_size', 256, lambda r: 2**int(r.uniform(6, 9)))
    elif dataset in MLP_DATASET:
        _hparam(
            'batch_size',
            8 if dataset in {"REF2021UOA11", "AmazonWILDS"} else 128,
            lambda r: 2**int(r.uniform(2, 5)) if dataset in {"REF2021UOA11", "AmazonWILDS"} else 2**int(r.uniform(5, 8)),
        )
    elif dataset in TEXT_DATASET:
        _hparam('batch_size', 32, lambda r: 2**int(r.uniform(4, 6)))
    elif dataset in REMOTE_SENSING_DATASET:
        _hparam('batch_size', 8, lambda r: 2**int(r.uniform(2, 5)))

    # algorithm-specific hyperparameters
    if algorithm == 'LLP_FlowLLP':
        # The first value remains the verified formal default. The second is
        # used only when hparams_seed > 0, so sweeps tune FlowLLP without
        # changing default or single-run behaviour.
        _hparam('flow_latent_dim', 50,
                lambda r: int(r.choice([32, 50, 64, 128])))
        _hparam('flow_pretrain_fraction', 0.5,
                lambda r: float(r.choice([0.3, 0.5, 0.7])))
        _hparam('flow_anchors_per_class', 1000,
                lambda r: int(r.choice([128, 256, 512, 1000])))
        _hparam('flow_particle_steps', 3000,
                lambda r: int(r.choice([500, 1000, 2000, 3000])))
        _hparam('flow_particle_lr', 1e-3,
                lambda r: float(10 ** r.uniform(-4, -2)))
        _hparam('flow_anchor_bag_batch', 1,
                lambda r: int(r.choice([1, 2, 4])))
        _hparam('flow_lambda_bag', 1.0,
                lambda r: float(10 ** r.uniform(-0.5, 0.5)))
        _hparam('flow_lambda_anchor', 0.1,
                lambda r: float(10 ** r.uniform(-2, -0.3)))
        _hparam('flow_reg_label', 0.0,
                lambda r: float(r.choice([0.0, 0.1, 1.0])))
        _hparam('flow_reg_bag_classifier', 1.0,
                lambda r: float(r.choice([0.0, 0.1, 1.0])))
        _hparam('flow_anchor_batch_size', 128,
                lambda r: int(r.choice([64, 128, 256])))
    elif algorithm == 'LLP_SimCLR':
        if dataset == "CIFAR10":
            _hparam('feat_dim', 128, lambda r: 128)
        elif dataset == "CIFAR100":
            _hparam('feat_dim',64, lambda r: 64)
        elif dataset in REMOTE_SENSING_DATASET:
            _hparam('feat_dim', 128, lambda r: 128)
        elif dataset in {"KUOptofilPBC", "CCT", "FedISIC2019"}:
            _hparam('feat_dim', 128, lambda r: 128)
        elif dataset == "TwitterEthnicity2017":
            _hparam('feat_dim', 128, lambda r: 128)
        _hparam('entropy_weight', 0.01, lambda r: 10 ** r.uniform(-3, -1))
        _hparam('match_weight', 5e-4, lambda r: 10 ** r.uniform(-6, -4))
        _hparam('match_threshold', 0.03, lambda r: 10 ** r.uniform(-2, 0))
        _hparam('match_min_ratio', 0.03, lambda r: 10 ** r.uniform(-2, 0))
    elif algorithm in {'LLP_ORDER', 'LLP_MM'}:
        _hparam('order', 3, lambda r: 3)
        if algorithm == 'LLP_MM':
            _hparam('moment_loss_type', 'ce', lambda r: 'ce')
            _hparam('moment_algorithm', 'stable_dp', lambda r: 'stable_dp')
            _hparam('moment_compute_dtype', 'float64', lambda r: 'float64')
            _hparam('moment_ce_smoothing_tau', 1e-4, lambda r: 1e-4)
            _hparam('order_weights', None, lambda r: None)
    elif algorithm == 'LLP_DSQ':
        # COLT 2024 Appendix G uses beta=0.99 for the streaming approximation
        # of the full-training-set prediction mean.
        _hparam('dsq_ema_beta', 0.99, lambda r: 0.99)
    elif algorithm == 'LLP_FC':
        _hparam('mode', 'approx', lambda r: r.choice(['approx', 'uniform']))
        _hparam('group_weight', 1.0, lambda r: 10 ** r.uniform(-1, 1))
        _hparam('entropy_weight', 0.0, lambda r: 10 ** r.uniform(-3, -1))
        _hparam('pi_grad_steps', 20, lambda r: int(r.uniform(10, 30)))
        _hparam('pi_grad_lr', 0.1, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == 'LLP_DC':
        _hparam('lam_u', 0.5, lambda r: r.uniform(0.3, 1))
        _hparam('thr', 0.95, lambda r: r.uniform(0.6, 0.97))

    elif algorithm == 'LLP_FixMatch':
        _hparam('mu_u', 0.5, lambda r: r.uniform(0.3, 1.0))
        _hparam('tau', 0.95, lambda r: r.uniform(0.6, 0.97))

    elif algorithm == 'LLP_SoftMatch':
        _hparam('ema_p', 0.999, lambda r: 0.999)  # 固定
        _hparam('softmatch_k', 2.0, lambda r: r.uniform(0.5, 5.0))
        _hparam('softmatch_s', 4.0, lambda r: r.uniform(0.5, 20.0))
        _hparam('mu_u', 0.5, lambda r: r.uniform(0.3, 1.0))

    elif algorithm == 'LLP_AHIL':
        _hparam('beta_i', 1.0, lambda r: r.uniform(0.5, 40.0))
        _hparam('beta_b', 1.0, lambda r: r.uniform(0.5, 40.0))
        _hparam('mu_u', 1.0, lambda r: r.uniform(0.3, 1.0))

    elif algorithm == 'ROT':
        # alpha in [0,1]
        _hparam('alpha', 0.6, lambda r: r.uniform(0.3, 0.7))
        _hparam('sinkhorn_iterations', 3, lambda r: int(r.uniform(3, 30)))

    elif algorithm == 'EasyLLP':
        _hparam('loss_type', 'ce', lambda r: r.choice(['ce', 'se']))
        _hparam('flooding', True, lambda r: bool(r.choice([False, True])))
        _hparam('flooding_b', 0.0, lambda r: 10 ** r.uniform(-3, -0.3))  # ~[1e-3, 0.5]
        # 如果 flooding=False，一般 flooding_b 会被忽略；保留也无害

    elif algorithm == 'GeneralUPM':
        _hparam('flooding', True, lambda r: bool(r.choice([False, True])))
        _hparam('flooding_b', 0.0, lambda r: 10 ** r.uniform(-3, -0.3))  # ~[1e-3, 0.5]

    elif algorithm == 'LLP_VAT':
        _hparam('consistency', 1.0, lambda r: 10 ** r.uniform(-1, 1))
        _hparam('consistency_rampup', 1000, lambda r: int(r.choice([0, 500, 1000, 2000, 4000])))

        _hparam('vat_xi', 1e-3, lambda r: 10 ** r.uniform(-4, -1))
        _hparam('vat_eps', 1.0, lambda r: 10 ** r.uniform(-0.5, 0.5))
        _hparam('vat_ip', 1, lambda r: int(r.choice([1, 2])))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
