"""Frozen defaults for the methods and datasets evaluated in the paper."""


def default_hparams(algorithm, dataset):
    from .algorithms import ALGORITHMS

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if dataset not in {"CIFAR10", "CIFAR100", "miniImageNet", "KUOptofilPBC"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    natural = dataset == "KUOptofilPBC"
    h = {
        "model": "ImageNetResNet18"
        if natural
        else "ResNet_miniImageNet"
        if dataset == "miniImageNet"
        else "ResNet",
        "optimizer": "Adam" if natural else "SGD",
        "lr": 0.001 if natural else 0.005 if algorithm == "LLP_PVC" else 0.05,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "nesterov": False,
        "warmup": "linear",
        "warmup_fraction": 0.05 if natural else 0.08,
        "warmup_ratio": 0.1 if natural else 0.0005,
        "cosine_mode": "standard" if natural else "legacy_quarter",
        "order": 8 if dataset in {"CIFAR10", "KUOptofilPBC"} else 3,
        "moment_implementation": "variable_bag" if natural else "paper_image",
        "moment_loss_type": "ce",
        "moment_algorithm": "stable_dp",
        "moment_compute_dtype": "float64",
        "moment_ce_smoothing_tau": 0.0001,
        "order_weights": None,
        "loss_type": "ce",
        "flooding": False,
        "flooding_b": 0.0,
        "dsq_ema_beta": 0.99,
        "mode": "approx",
        "group_weight": 1.0,
        "entropy_weight": 0.0,
        "pi_grad_steps": 20,
        "pi_grad_lr": 0.1,
        "alpha": 0.6,
        "sinkhorn_iterations": 3,
    }
    if natural:
        h.update(
            pretrained=True,
            ku_activation_checkpoint=True,
            ku_test_every_checkpoint=True,
            ku_select_by_validation_macro_f1=False,
        )
    if algorithm == "LLP_FlowLLP":
        h.update(
            flow_latent_dim=50,
            flow_pretrain_fraction=0.5,
            flow_anchors_per_class=1000,
            flow_particle_steps=3000,
            flow_particle_lr=0.001,
            flow_anchor_bag_batch=1,
            flow_lambda_bag=1.0,
            flow_lambda_anchor=0.1,
            flow_reg_label=0.0,
            flow_reg_bag_classifier=1.0,
            flow_anchor_batch_size=128,
        )
        if not natural:
            h.update(nesterov=True, warmup_ratio=0.001)
    return h
