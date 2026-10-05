from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import random
import scipy.sparse as sp
from contextlib import contextmanager

from scipy.optimize import minimize, Bounds, LinearConstraint
try:
    import ot
except ImportError:
    ot = None
import os
from sklearn.metrics import euclidean_distances
from .FFT import compute_CC_loss_fft_precise_batched
from . import networks
from .gaussian_count import gaussian_count_nll
import collections
from torch.optim.lr_scheduler import _LRScheduler

# ---- OR-Tools import compatibility ----
try:
    # Newer OR-Tools
    from ortools.graph.python import min_cost_flow as _mcf
    _HAS_NEW_ORTOOLS = True
except Exception:
    _HAS_NEW_ORTOOLS = False
    try:
        # Older OR-Tools
        from ortools.graph import pywrapgraph as _pywrapgraph
    except Exception as e:
        _pywrapgraph = None

ALGORITHMS = [
    'PM',
    'LLP_MM',
    'LLP_Gaussian',
    'LLP_DSQ',
    'LLP_PVC',
    'LLP_SimCLR',
    'LLP_PT',
    'LLP_FC',
    'ROT',
    'EasyLLP',
    'GeneralUPM',
    'NonClipOVR',
    'LLP_AHIL',
    'LLP_DC',
    'LLP_FixMatch',
    'LLP_SoftMatch',
    'LLP_VAT',
    'LLP_FlowLLP',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]



class WarmupCosineLrScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            cosine_mode='legacy_quarter',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        self.cosine_mode = cosine_mode
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = max(1, self.max_iter - self.warmup_iter)
            progress = min(1.0, max(0.0, real_iter / real_max_iter))
            if self.cosine_mode == 'standard':
                ratio = 0.5 * (1.0 + np.cos(np.pi * progress))
            elif self.cosine_mode == 'legacy_quarter':
                ratio = np.cos(np.pi * progress / 2.0)
            else:
                raise ValueError(f"Unknown cosine mode: {self.cosine_mode!r}")

        return ratio

    def get_warmup_ratio(self):
        # progress: 0 -> 1
        t = (self.last_epoch + 1) / max(1, self.warmup_iter)

        if self.warmup == "linear":
            # 从 warmup_ratio 线性升到 1
            return self.warmup_ratio + (1 - self.warmup_ratio) * t

        if self.warmup == "exp":
            # 从 warmup_ratio 指数升到 1（更常用）
            return self.warmup_ratio ** (1 - t)

        raise ValueError("warmup must be 'linear' or 'exp'")


def _make_optimizer(parameters, hparams):
    optimizer_name = str(hparams.get("optimizer", "Adam")).lower()
    common = {
        "lr": float(hparams["lr"]),
        "weight_decay": float(hparams.get("weight_decay", 1e-4)),
    }
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=float(hparams.get("momentum", 0.9)),
            nesterov=bool(hparams.get("nesterov", False)),
            **common,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **common)
    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a learning from label proportions algorithm.
    """
    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_data = input_shape
        self.num_classes = len(train_givenY[0])
        self.bagsize = bagsize

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, self.num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.optimizer = _make_optimizer(self.network.parameters(), self.hparams)

        total_iters = int(epochs)
        warmup_iters = int(
            round(total_iters * float(self.hparams.get("warmup_fraction", 0.0)))
        )
        self.scheduler = WarmupCosineLrScheduler(
            self.optimizer,
            total_iters,
            warmup_iter=warmup_iters,
            warmup_ratio=float(self.hparams.get("warmup_ratio", 5e-4)),
            warmup=str(self.hparams.get("warmup", "linear")),
            cosine_mode=str(self.hparams.get("cosine_mode", "legacy_quarter")),
        )

        train_givenY = np.array(train_givenY, dtype=np.float32)
        self.label_proportions = torch.from_numpy(train_givenY)

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError

    def predict(self, x):
        return self.network(x)

    def _backward_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss.item()}


def _resolve_bag_layout(
    number_of_instances: int,
    proportions: torch.Tensor,
    number_of_classes: int,
    device: torch.device,
    fixed_bag_size: int,
    bag_sizes: torch.Tensor | None,
    bag_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate flattened valid-instance natural-bag metadata."""
    if proportions.dim() != 2 or proportions.shape[1] != number_of_classes:
        raise ValueError(
            f"proportions must have shape [B,{number_of_classes}], "
            f"got {tuple(proportions.shape)}"
        )
    number_of_bags = proportions.shape[0]
    if bag_sizes is None:
        size = int(fixed_bag_size)
        if size <= 0:
            raise ValueError(f"bagsize must be positive, got {size}")
        if number_of_instances != number_of_bags * size:
            raise ValueError(
                f"N={number_of_instances} must equal "
                f"B*bagsize={number_of_bags * size}"
            )
        sizes = torch.full(
            (number_of_bags,), size, dtype=torch.long, device=device
        )
    else:
        sizes_value = torch.as_tensor(bag_sizes, device=device)
        if sizes_value.dim() != 1 or sizes_value.numel() != number_of_bags:
            raise ValueError(
                f"bag_sizes must have shape [{number_of_bags}], "
                f"got {tuple(sizes_value.shape)}"
            )
        if sizes_value.is_floating_point() and not torch.equal(
            sizes_value, sizes_value.round()
        ):
            raise ValueError("bag_sizes must contain integer values")
        sizes = sizes_value.to(torch.long)
        if bool((sizes <= 0).any()):
            raise ValueError("bag_sizes must be positive")
        if int(sizes.sum().item()) != number_of_instances:
            raise ValueError(
                f"sum(bag_sizes)={int(sizes.sum().item())} "
                f"must equal N={number_of_instances}"
            )
    if bag_index is None:
        index = torch.repeat_interleave(
            torch.arange(number_of_bags, device=device), sizes
        )
    else:
        index = torch.as_tensor(bag_index, dtype=torch.long, device=device)
        if index.dim() != 1 or index.numel() != number_of_instances:
            raise ValueError(
                f"bag_index must have shape [{number_of_instances}], "
                f"got {tuple(index.shape)}"
            )
        if bool(((index < 0) | (index >= number_of_bags)).any()):
            raise ValueError("bag_index contains an out-of-range bag id")
        if not torch.equal(torch.bincount(index, minlength=number_of_bags), sizes):
            raise ValueError("bag_index counts do not match bag_sizes")
    return sizes, index


def _split_paired_logits(
    outputs: torch.Tensor,
    proportions: torch.Tensor,
    fixed_bag_size: int,
    bag_sizes: torch.Tensor | None,
    bag_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[all weak, all strong]`` logits using valid-instance metadata."""
    if outputs.shape[0] % 2:
        raise ValueError("paired weak/strong outputs must contain an even number of rows")
    valid_instances = outputs.shape[0] // 2
    sizes, index = _resolve_bag_layout(
        valid_instances,
        proportions,
        outputs.shape[1],
        outputs.device,
        fixed_bag_size,
        bag_sizes,
        bag_index,
    )
    return outputs[:valid_instances], outputs[valid_instances:], sizes, index


def _bag_means(
    values: torch.Tensor, bag_index: torch.Tensor, number_of_bags: int
) -> torch.Tensor:
    result = values.new_zeros((number_of_bags, values.shape[1]))
    result.scatter_add_(0, bag_index[:, None].expand_as(values), values)
    counts = torch.bincount(bag_index, minlength=number_of_bags).to(values)
    return result / counts[:, None]

class LLP_VAT(Algorithm):
    """
    PM + VAT consistency
    Reference: On learning from label proportions, ArXiv 2014.
    """

    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        hp = hparams if isinstance(hparams, dict) else vars(hparams)

        self.consistency = float(hp.get("consistency", 1.0))
        self.consistency_rampup = int(hp.get("consistency_rampup", 1000))

        self.vat_xi = float(hp.get("vat_xi", 1e-3))
        self.vat_eps = float(hp.get("vat_eps", 1.0))
        self.vat_ip = int(hp.get("vat_ip", 1))

        # VAT loss as a submodule (same logic, just kept inside LLP_VAT)
        self.consistency_criterion = self.VATLoss(xi=self.vat_xi, eps=self.vat_eps, ip=self.vat_ip)

    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    @staticmethod
    def get_algorithm_class(algorithm_name):
        """Return the algorithm class with the given name."""
        if algorithm_name not in globals():
            raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
        return globals()[algorithm_name]

    @staticmethod
    def get_rampup_weight(weight, iteration, rampup):
        return weight * LLP_VAT.sigmoid_rampup(iteration, rampup)

    @contextmanager
    def _disable_tracking_bn_stats(self, model):
        def switch_attr(m):
            if hasattr(m, "track_running_stats"):
                m.track_running_stats ^= True

        model.apply(switch_attr)
        try:
            yield
        finally:
            model.apply(switch_attr)

    class VATLoss(nn.Module):
        def __init__(self, xi=10.0, eps=1.0, ip=1):
            """
            VAT loss
            :param xi: hyperparameter of VAT (default: 10.0)
            :param eps: hyperparameter of VAT (default: 1.0)
            :param ip: iteration times of computing adv noise (default: 1)
            """
            super().__init__()
            self.xi = xi
            self.eps = eps
            self.ip = ip

        @staticmethod
        def _l2_normalize(d):
            d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
            d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
            return d

        def forward(self, model, x, disable_bn_ctx):
            # pred (no grad)
            with torch.no_grad():
                output_tuple = model(x)
                logits = output_tuple[0] if isinstance(output_tuple, (tuple, list)) else output_tuple
                if logits.dim() == 1:  # e.g. [C]
                    logits = logits.unsqueeze(0)  # -> [1, C]
                pred = F.softmax(logits, dim=-1)

            # prepare random unit tensor
            d = torch.randn_like(x)
            d = self._l2_normalize(d)

            with disable_bn_ctx(model):
                # calc adversarial direction
                for _ in range(self.ip):
                    d.requires_grad_()

                    pred_hat = model(x + self.xi * d)
                    pred_hat = pred_hat[0] if isinstance(pred_hat, (tuple, list)) else pred_hat
                    logp_hat = F.log_softmax(pred_hat, dim=-1)

                    adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                    adv_distance.backward()

                    d = self._l2_normalize(d.grad)
                    model.zero_grad()

                # calc LDS
                r_adv = d * self.eps
                pred_hat = model(x + r_adv)
                pred_hat = pred_hat[0] if isinstance(pred_hat, (tuple, list)) else pred_hat
                logp_hat = F.log_softmax(pred_hat, dim=-1)
                lds = F.kl_div(logp_hat, pred, reduction="batchmean")

            return lds

    def update(self, minibatches):
        x, proportions, it = minibatches

        consistency_loss = self.consistency_criterion(
            self.network, x, disable_bn_ctx=self._disable_tracking_bn_stats
        )

        alpha = self.get_rampup_weight(self.consistency, it, self.consistency_rampup)
        consistency_loss = alpha * consistency_loss

        loss = self.PM_Loss(self.predict(x), proportions)
        loss = loss + consistency_loss

        return self._backward_step(loss)

    def PM_Loss(
        self,
        outputs,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        proportions = proportions.to(device)

        probs = F.softmax(outputs, dim=1)

        N, C = probs.shape
        _, index = _resolve_bag_layout(
            N,
            proportions,
            C,
            device,
            self.bagsize,
            bag_sizes,
            bag_index,
        )
        bag_preds = _bag_means(probs, index, len(proportions)).clamp_min(1e-12)

        loss = -(proportions * torch.log(bag_preds)).sum(dim=1).mean()
        return loss


class PM(Algorithm):
    """
    PM
    Reference: On learning from label proportions, ArXiv 2014.
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super(PM, self).__init__(epochs,input_shape, train_givenY, hparams, bagsize)

    def update(self, minibatches):
        x,proportions= minibatches
        loss = self.PM_Loss(self.predict(x),proportions)
        return self._backward_step(loss)

    def PM_Loss(
        self,
        outputs,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        proportions = proportions.to(device)
        probs = F.softmax(outputs, dim=1)

        N, C = probs.shape
        _, index = _resolve_bag_layout(
            N,
            proportions,
            C,
            device,
            self.bagsize,
            bag_sizes,
            bag_index,
        )
        bag_preds = _bag_means(probs, index, len(proportions)).clamp_min(1e-12)
        loss = -(proportions * bag_preds.log()).sum(dim=1).mean()
        return loss


class LLP_Gaussian(Algorithm):
    """Second-order Gaussian likelihood for the observed bag-label counts."""

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)
        self.count_variance_floor = float(
            hparams.get("gaussian_count_variance_floor", 1.0 / 12.0)
        )
        self.bags_per_microbatch = hparams.get("gaussian_bags_per_microbatch")
        if self.bags_per_microbatch is not None:
            self.bags_per_microbatch = int(self.bags_per_microbatch)
            if self.bags_per_microbatch < 1:
                raise ValueError("gaussian_bags_per_microbatch must be positive")

    def update(self, minibatches):
        images, proportions = minibatches
        number_of_bags = int(proportions.shape[0])
        if self.bags_per_microbatch is None or self.bags_per_microbatch >= number_of_bags:
            loss = gaussian_count_nll(
                self.predict(images),
                proportions,
                self.bagsize,
                self.count_variance_floor,
            )
            return self._backward_step(loss)

        # The Gaussian likelihood is a mean of independent bag terms.  Sum
        # their weighted gradients before a single optimizer/scheduler step,
        # preserving the logical bag batch on memory-limited GPUs.
        self.optimizer.zero_grad()
        loss_value = 0.0
        for first in range(0, number_of_bags, self.bags_per_microbatch):
            last = min(first + self.bags_per_microbatch, number_of_bags)
            bag_loss = gaussian_count_nll(
                self.predict(images[first * self.bagsize:last * self.bagsize]),
                proportions[first:last],
                self.bagsize,
                self.count_variance_floor,
            )
            weight = (last - first) / number_of_bags
            (bag_loss * weight).backward()
            loss_value += bag_loss.detach().item() * weight
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss_value}


class LLP_MM(Algorithm):
    """Exact multi-class factorial moment matching with CE supervision.

    Natural bags use the adapter in :func:`plench.data.ref2021.ref2021_loss`.
    The shared update below applies the same factorial-moment objective to the
    fixed-size bags used by CV/LEM and the original synthetic benchmarks.  All
    moment probabilities and the stable DP are evaluated in float64; the image
    backbone may remain float32.
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)
        self.order = int(hparams.get("order", 3))
        self.moment_implementation = str(hparams.get("moment_implementation", "variable_bag"))
        if self.moment_implementation not in {"variable_bag", "paper_image"}:
            raise ValueError("Unknown LLP-MM moment_implementation")
        self.moment_loss_type = str(hparams.get("moment_loss_type", "ce"))
        self.moment_algorithm = str(hparams.get("moment_algorithm", "stable_dp"))
        self.moment_compute_dtype = str(
            hparams.get("moment_compute_dtype", "float64")
        )
        self.moment_ce_smoothing_tau = float(
            hparams.get("moment_ce_smoothing_tau", 1e-4)
        )
        if self.order < 1:
            raise ValueError("LLP_MM order must be positive")
        configured_weights = hparams.get("order_weights")
        self.order_weights = (
            tuple([1.0 / float(self.order)] * self.order)
            if configured_weights is None
            else tuple(float(value) for value in configured_weights)
        )
        if self.moment_loss_type != "ce":
            raise ValueError("LLP_MM requires moment_loss_type='ce'")
        if self.moment_algorithm != "stable_dp":
            raise ValueError("LLP_MM requires moment_algorithm='stable_dp'")
        if self.moment_compute_dtype != "float64":
            raise ValueError("LLP_MM requires moment_compute_dtype='float64'")
        if not math.isclose(
            self.moment_ce_smoothing_tau, 1e-4, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("LLP_MM requires moment_ce_smoothing_tau=1e-4")
        if len(self.order_weights) != self.order:
            raise ValueError("LLP_MM order_weights must have length order")
        if (
            any(not math.isfinite(value) or value < 0 for value in self.order_weights)
            or sum(self.order_weights) <= 0
        ):
            raise ValueError("LLP_MM order_weights must be finite and non-negative")

        if self.moment_implementation == "paper_image":
            from .order import LLPHighOrderLoss
            self.paper_image_criterion = LLPHighOrderLoss(
                C=self.num_classes, max_order=self.order, bag_size=bagsize,
                loss_type="ce", weight_mode="uniform",
                order_weights=self.order_weights, reduce="mean",
            )

    def update(self, minibatches):
        from ..data.ref2021 import _llp_mm_variable_loss

        x, proportions = minibatches
        logits = self.predict(x)
        if self.moment_implementation == "paper_image":
            # Match the fixed-image implementation shipped in the supplement.
            loss = self.paper_image_criterion(
                proportions.to(device=logits.device), F.softmax(logits, dim=1)
            )
            return self._backward_step(loss)
        proportions = proportions.to(device=logits.device, dtype=logits.dtype)
        number_of_bags = int(proportions.shape[0])
        if number_of_bags <= 0 or logits.shape[0] != number_of_bags * int(self.bagsize):
            raise ValueError(
                "LLP_MM fixed-bag update expects B*bagsize flattened instances; "
                f"got N={logits.shape[0]}, B={number_of_bags}, bagsize={self.bagsize}"
            )
        slices = [
            torch.arange(
                bag * int(self.bagsize),
                (bag + 1) * int(self.bagsize),
                device=logits.device,
            )
            for bag in range(number_of_bags)
        ]
        loss = _llp_mm_variable_loss(
            logits.to(torch.float64).softmax(dim=1).clamp_min(
                torch.finfo(torch.float64).tiny
            ),
            proportions,
            slices,
            self.order,
            order_weights=self.order_weights,
            loss_type=self.moment_loss_type,
            moment_algorithm=self.moment_algorithm,
            compute_dtype=self.moment_compute_dtype,
            ce_smoothing_tau=self.moment_ce_smoothing_tau,
        )
        return self._backward_step(loss)

class LLP_DC(Algorithm):
    """
    LLP_DC:
      - weak side: per-bag optimal one-hot assignment under label proportions (min-cost flow)
      - bag loss: LLP loss between bag mean probs and given proportions
      - strong side: instance pseudo-label loss using the optimal one-hot as pseudo label
      - weight: confidence threshold mask (thr)
      - total: loss_prop + lam_u * loss_u

    hparams only:
      - lam_u
      - thr
    """

    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        hp = hparams if isinstance(hparams, dict) else vars(hparams)
        self.lam_u = float(hp.get("lam_u", 1.0))
        self.thr = float(hp.get("thr", 0.95))

        # 固定默认（不放进 hparams）
        self._solver_epsilon = 1e-12
        self._solver_cost_scale = 10000

        if (not _HAS_NEW_ORTOOLS) and (_pywrapgraph is None):
            raise ImportError(
                "OR-Tools not found. Please install ortools, e.g. `pip install ortools`."
            )

    # -----------------------------
    # OR-Tools solver (封装进类里)
    # -----------------------------
    @staticmethod
    def solve_optimal_onehot_with_proportions_torch(
        softmax_tensor: torch.Tensor,
        proportions: torch.Tensor,
        bagsize: int,
        n_classes: int,
        epsilon: float = 1e-12,
        cost_scale: int = 10000,
    ) -> torch.Tensor:
        """
        Solve optimal one-hot assignment under class-count constraints:
            minimize sum_{i,j} x_{ij} * (-log p_{ij})
            s.t. each sample assigned exactly one class,
                 each class assigned target_counts[j] samples.

        Returns:
            best_onehot: [bagsize, n_classes] int32 on CPU
        """
        assert isinstance(softmax_tensor, torch.Tensor)
        assert isinstance(proportions, torch.Tensor)
        assert softmax_tensor.shape == (bagsize, n_classes), \
            f"softmax_tensor should be {(bagsize, n_classes)}, got {tuple(softmax_tensor.shape)}"
        assert proportions.shape == (n_classes,), \
            f"proportions should be {(n_classes,)}, got {tuple(proportions.shape)}"

        # to CPU numpy
        softmax_cpu = softmax_tensor.detach().cpu().numpy()
        proportions_cpu = proportions.detach().cpu().numpy()

        # sum check (allow tiny numerical error)
        s = float(proportions_cpu.sum())
        if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"proportions sum must be ~1, got {s}")

        # 1) target counts by rounding
        raw = proportions_cpu * bagsize
        target_counts = np.floor(raw).astype(int)  # start with floor
        remaining = bagsize - int(target_counts.sum())

        # distribute remaining based on largest fractional parts
        frac = raw - np.floor(raw)
        if remaining > 0:
            idx = np.argsort(-frac)  # descending
            for k in idx[:remaining]:
                target_counts[k] += 1
        elif remaining < 0:
            # too many assigned due to numeric issues; remove from smallest fractional parts
            idx = np.argsort(frac)  # ascending
            for k in idx[:(-remaining)]:
                if target_counts[k] > 0:
                    target_counts[k] -= 1

        # safety
        assert target_counts.sum() == bagsize, f"target_counts sum {target_counts.sum()} != {bagsize}"

        # 2) build min-cost flow
        if _HAS_NEW_ORTOOLS:
            mcf = _mcf.SimpleMinCostFlow()
            add_arc = mcf.add_arc_with_capacity_and_unit_cost
            set_supply = mcf.set_node_supply
            solve = mcf.solve
            OPTIMAL = mcf.OPTIMAL
            num_arcs = mcf.num_arcs
            flow = mcf.flow
            tail = mcf.tail
            head = mcf.head
        else:
            mcf = _pywrapgraph.SimpleMinCostFlow()
            add_arc = mcf.AddArcWithCapacityAndUnitCost
            set_supply = mcf.SetNodeSupply
            solve = mcf.Solve
            OPTIMAL = mcf.OPTIMAL
            num_arcs = mcf.NumArcs
            flow = mcf.Flow
            tail = mcf.Tail
            head = mcf.Head

        S = 0
        T = bagsize + n_classes + 1

        def sample_node(i):  # 0..bagsize-1 -> 1..bagsize
            return i + 1

        def class_node(j):   # 0..C-1 -> bagsize+1 .. bagsize+C
            return bagsize + 1 + j

        # S -> samples
        for i in range(bagsize):
            add_arc(S, sample_node(i), 1, 0)

        # samples -> classes
        for i in range(bagsize):
            for j in range(n_classes):
                p_ij = float(max(softmax_cpu[i, j], epsilon))
                cost = int(-math.log(p_ij) * cost_scale)
                add_arc(sample_node(i), class_node(j), 1, cost)

        # classes -> T
        for j in range(n_classes):
            add_arc(class_node(j), T, int(target_counts[j]), 0)

        set_supply(S, bagsize)
        set_supply(T, -bagsize)

        status = solve()
        if status != OPTIMAL:
            raise RuntimeError("OR-Tools: failed to find optimal flow.")

        best_onehot = torch.zeros((bagsize, n_classes), dtype=torch.int32)
        for i in range(num_arcs()):
            if flow(i) > 0:
                u = tail(i)
                v = head(i)
                if 1 <= u <= bagsize and (bagsize + 1) <= v <= (bagsize + n_classes):
                    sample_idx = u - 1
                    class_idx = v - (bagsize + 1)
                    best_onehot[sample_idx, class_idx] = 1

        return best_onehot

    # -----------------------------
    # losses
    # -----------------------------
    @staticmethod
    def llp_loss(proportion: torch.Tensor, bag_prob: torch.Tensor):
        # proportion, bag_prob: [C]
        return -(proportion * bag_prob.log()).sum()

    @staticmethod
    def soft_ce_with_onehot(logits: torch.Tensor, onehot: torch.Tensor):
        # logits: [N,C], onehot: [N,C]
        logp = F.log_softmax(logits, dim=1)
        return -(onehot * logp).sum(dim=1)  # [N]

    # -----------------------------
    # training step
    # -----------------------------
    def update(self, minibatches):
        x, proportions = minibatches
        logits = self.predict(x)

        loss, loss_prop, loss_u = self.LLP_DC_Loss(logits, proportions)
        return self._backward_step(loss)

    def LLP_DC_Loss(
        self,
        logits_all,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        """
        logits_all: [2*N, C], ordered as all weak then all strong instances.
        proportions: [B0, C]
        """
        device = logits_all.device
        proportions = proportions.to(device)

        _, C = logits_all.shape
        B0 = proportions.shape[0]
        logits_u_w, logits_u_s, sizes, index = _split_paired_logits(
            logits_all,
            proportions,
            self.bagsize,
            bag_sizes,
            bag_index,
        )

        # --------- 以下逻辑完全照你原来的 DC，不动 ----------
        probs_w = torch.softmax(logits_u_w, dim=1)  # [N,C], N=B0*s

        pseudo_onehot = torch.zeros_like(probs_w)
        loss_prop_list = []

        for b in range(B0):
            positions = torch.nonzero(index == b, as_tuple=False).squeeze(1)
            bag_probs = probs_w[positions]  # [k_i,C]
            bag_prop = proportions[b]  # [C]
            size = int(sizes[b].item())

            opt_onehot = self.solve_optimal_onehot_with_proportions_torch(
                bag_probs, bag_prop, bagsize=size, n_classes=C,
                epsilon=self._solver_epsilon,
                cost_scale=self._solver_cost_scale
            ).to(device=device, dtype=logits_u_w.dtype)  # [k_i,C] float on device

            pseudo_onehot[positions] = opt_onehot

            bag_mean = bag_probs.mean(dim=0).clamp_min(self._solver_epsilon)  # [C]
            loss_prop_list.append(self.llp_loss(bag_prop.to(bag_mean.dtype), bag_mean))

        loss_prop = torch.stack(loss_prop_list).mean()

        confidence = (probs_w * pseudo_onehot).sum(dim=1)  # [N]
        mask = (confidence >= self.thr).to(dtype=logits_u_w.dtype)  # [N]

        per_inst = self.soft_ce_with_onehot(logits_u_s, pseudo_onehot)  # [N]
        loss_u = (per_inst * mask).mean()

        loss = loss_prop + self.lam_u * loss_u
        return loss, loss_prop, loss_u

class LLP_SoftMatch(Algorithm):
    """
    LLP + SoftMatch-style (instance pseudo-label version):

    - front half (weak): bag-level LLP loss
    - back  half (strong): instance pseudo-label loss from weak instance prediction
    - weight is SoftMatch soft mask from weak instance probs
    - total = bag_loss + mu_u * inst_loss
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)
        self.eps = 1e-12
        hp = hparams if isinstance(hparams, dict) else vars(hparams)
        self.ema_p = float(hp.get("ema_p", 0.999))
        self.softmatch_k = float(hp.get("softmatch_k", 2.0))
        self.softmatch_s = float(hp.get("softmatch_s", 4.0))
        self.softmatch_eps = float(hp.get("softmatch_eps", 1e-6))
        self.mu_u = float(hp.get("mu_u", 0.5))

    def update(self, minibatches):
        x, proportions, ulb_prob_t, prob_max_mu_t, prob_max_var_t = minibatches
        outputs = self.predict(x)
        loss, bag_loss, inst_loss, ulb_prob_t, prob_max_mu_t, prob_max_var_t = \
            self.LLP_SoftMatch_Loss(outputs, proportions, ulb_prob_t, prob_max_mu_t, prob_max_var_t)

        return self._backward_step(loss), ulb_prob_t, prob_max_mu_t, prob_max_var_t

    @torch.no_grad()
    def update_prob_t(self, ema_probs, mean, var, ulb_probs):
        ulb_prob_t = ulb_probs.mean(0)
        ema_probs = self.ema_p * ema_probs + (1 - self.ema_p) * ulb_prob_t

        max_probs, _ = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_var_t = torch.var(max_probs, unbiased=True)

        mean = self.ema_p * mean + (1 - self.ema_p) * prob_max_mu_t.item()
        var = self.ema_p * var + (1 - self.ema_p) * prob_max_var_t.item()
        return ema_probs, mean, var

    @torch.no_grad()
    def calculate_mask(self, probs, mean, var):
        max_probs, _ = probs.max(dim=-1)
        mu = mean
        v = float(var)
        v = max(v, self.softmatch_eps)

        denom = (self.softmatch_k * v) / self.softmatch_s
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / denom))
        return max_probs.detach(), mask.detach()

    def LLP_SoftMatch_Loss(
        self,
        outputs,
        proportions,
        ulb_prob_t,
        prob_max_mu_t,
        prob_max_var_t,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        proportions = proportions.to(device)

        B0 = proportions.shape[0]
        logits_w, logits_s, _, index = _split_paired_logits(
            outputs,
            proportions,
            self.bagsize,
            bag_sizes,
            bag_index,
        )
        p_w = F.softmax(logits_w, dim=1)

        # 1) bag loss on weak bags
        bag_pred_w = _bag_means(p_w, index, B0).clamp_min(self.eps)
        bag_loss = -(proportions * bag_pred_w.log()).sum(dim=1).mean()

        # 2) instance pseudo labels from weak instance predictions
        probs_w_inst = p_w.detach()
        _, y_inst = probs_w_inst.max(dim=1)               # [B0*s]

        per_inst_loss = F.cross_entropy(logits_s, y_inst, reduction="none")

        # SoftMatch mask from weak instance probs
        ulb_prob_t, prob_max_mu_t, prob_max_var_t = self.update_prob_t(
            ulb_prob_t, prob_max_mu_t, prob_max_var_t, probs_w_inst
        )

        max_probs, mask = self.calculate_mask(probs_w_inst, prob_max_mu_t, prob_max_var_t)
        mask = mask.to(dtype=per_inst_loss.dtype, device=device)

        inst_loss = (per_inst_loss * mask).mean()

        total_loss = bag_loss + self.mu_u * inst_loss
        return total_loss, bag_loss, inst_loss, ulb_prob_t, prob_max_mu_t, prob_max_var_t
class LLP_FixMatch(Algorithm):
    """
    LLP + FixMatch-style (instance pseudo-label version):

    - front half (weak): bag-level LLP loss
    - back  half (strong): instance pseudo-label loss from weak instance prediction
    - weight is threshold mask on weak instance confidence
    - total = bag_loss + mu_u * inst_loss
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)

        hp = hparams if isinstance(hparams, dict) else vars(hparams)
        self.mu_u = float(hp.get("mu_u", 1.0))
        self.tau = float(hp.get("tau", 0.95))
        self.eps = 1e-12

    def update(self, minibatches):
        x, proportions = minibatches
        outputs = self.predict(x)
        loss, bag_loss, inst_loss = self.LLP_FixMatch_Loss(outputs, proportions)
        return self._backward_step(loss)

    def LLP_FixMatch_Loss(
        self,
        outputs,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        proportions = proportions.to(device)

        B0 = proportions.shape[0]
        logits_w, logits_s, _, index = _split_paired_logits(
            outputs,
            proportions,
            self.bagsize,
            bag_sizes,
            bag_index,
        )
        p_w = F.softmax(logits_w, dim=1)

        # 1) bag loss on weak bags
        bag_pred_w = _bag_means(p_w, index, B0).clamp_min(self.eps)
        bag_loss = -(proportions * bag_pred_w.log()).sum(dim=1).mean()

        # 2) instance pseudo labels from weak instance predictions
        pseudo_inst = p_w.detach()
        conf_inst, y_inst = pseudo_inst.max(dim=1)     # [B0*s], [B0*s]

        per_inst_loss = F.cross_entropy(logits_s, y_inst, reduction="none")

        # instance-level FixMatch mask
        # instance-level FixMatch mask (script-style weighting)
        if self.tau > 0:
            mask = (conf_inst >= self.tau).float().to(device=device, dtype=per_inst_loss.dtype)
        else:
            mask = torch.ones(len(logits_s), device=device, dtype=per_inst_loss.dtype)

        inst_loss = (per_inst_loss * mask).mean()

        total_loss = bag_loss + self.mu_u * inst_loss
        return total_loss, bag_loss, inst_loss
class LLP_AHIL(Algorithm):
    """
    LLP + AHIL (instance pseudo-label version):
      - front half (weak): bag-level LLP loss
      - back  half (strong): instance-level pseudo-label loss from weak instance prediction
      - instance loss weights = lambda_i(entropy_i, beta_i) * lambda_b(entropy_b, beta_b)
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)

        hp = hparams if isinstance(hparams, dict) else vars(hparams)

        self.beta_i = float(hp.get("beta_i", 1.0))
        self.beta_b = float(hp.get("beta_b", 1.0))
        self.mu_u = float(hp.get("mu_u", 1.0))

        self.tau = float(hp.get("tau", 0.0))
        self.eps = float(hp.get("eps", 1e-12))

    def update(self, minibatches):
        x, proportions = minibatches
        outputs = self.predict(x)
        loss, bag_loss, inst_loss = self.LLP_AHIL_Loss(outputs, proportions)
        return self._backward_step(loss)

    @staticmethod
    def normal(h, h_tilde, beta):
        return torch.exp(-(h - h_tilde) ** 2 / beta)

    @staticmethod
    def calc_bag_entropy(probs):
        probs_class_normal = torch.nn.functional.normalize(probs, p=1, dim=1)
        return -torch.sum(
            probs_class_normal * torch.log(probs_class_normal + 1e-8),
            dim=1
        )

    @staticmethod
    def calc_instance_entropy(probs):
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    @staticmethod
    def calc_opt_entropy(nn):
        return torch.log(nn + 1e-8)

    def LLP_AHIL_Loss(
        self,
        outputs,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        proportions = proportions.to(device)

        _, C = outputs.shape
        B0 = proportions.shape[0]
        logits_w, logits_s, sizes, index = _split_paired_logits(
            outputs,
            proportions,
            self.bagsize,
            bag_sizes,
            bag_index,
        )
        p_w = F.softmax(logits_w, dim=1)
        p_s = F.softmax(logits_s, dim=1)

        # 1) bag-level LLP loss on weak half
        bag_pred_w = _bag_means(p_w, index, B0).clamp_min(self.eps)
        bag_loss = -(proportions * bag_pred_w.log()).sum(dim=1).mean()

        # 2) instance pseudo labels from weak instance predictions
        pseudo_inst = p_w.detach()
        _, y_inst = pseudo_inst.max(dim=1)              # [B0*s]

        per_inst_loss = F.cross_entropy(logits_s, y_inst, reduction="none")

        # 3) keep old AHIL weighting logic
        with torch.no_grad():
            probs_s = p_s.clamp_min(self.eps)

            # instance-level entropy weight
            entropy_i = self.calc_instance_entropy(probs_s)
            lambda_i = self.normal(entropy_i, h_tilde=0, beta=self.beta_i)
            lambda_i = lambda_i.to(dtype=per_inst_loss.dtype)

            # hard labels from strong predictions
            _, pred_idx = torch.max(probs_s, dim=1)
            one_hot = torch.zeros_like(probs_s).scatter_(1, pred_idx.unsqueeze(1), 1)

            lambda_b = torch.zeros_like(per_inst_loss)

            log_base = torch.log(torch.tensor(float(C), device=device, dtype=proportions.dtype))

            for i in range(B0):
                positions = torch.nonzero(index == i, as_tuple=False).squeeze(1)
                chunk = one_hot[positions]
                _, class_indices = torch.max(chunk, dim=1)
                chunk_mean = torch.mean(chunk, dim=0)

                label_proportion = proportions[i].to(dtype=chunk_mean.dtype)
                difference = chunk_mean - label_proportion
                abs_difference = torch.abs(difference)
                errors = 1 - abs_difference.pow(1 / log_base)

                opt_entropy_b = self.calc_opt_entropy(
                    label_proportion * sizes[i].to(label_proportion)
                )
                entropy_b = self.calc_bag_entropy(
                    probs_s[positions].unsqueeze(0)
                ).squeeze(0)

                entropy_b_normal = self.normal(
                    entropy_b.to(dtype=opt_entropy_b.dtype),
                    h_tilde=opt_entropy_b,
                    beta=self.beta_b
                )

                selected_errors = errors[class_indices]
                selected_entropy_b_normal = entropy_b_normal[class_indices]

                selected_errors = selected_errors.to(dtype=per_inst_loss.dtype)
                selected_entropy_b_normal = selected_entropy_b_normal.to(dtype=per_inst_loss.dtype)

                lambda_b[positions] = selected_entropy_b_normal
            lambda_total = lambda_i * lambda_b
            mask = torch.ones_like(lambda_total)

        denom = mask.sum().clamp_min(1.0)
        inst_loss = (per_inst_loss * lambda_total).mean()

        total_loss = bag_loss + self.mu_u * inst_loss
        return total_loss, bag_loss, inst_loss
class NonClipOVR(Algorithm):
    """ICML 2025 unclipped square-loss estimator with OVR outputs.

    The paper proves the binary estimator.  This class retains the existing
    multiclass one-vs-rest vector extension and applies the same estimator to
    every coordinate.  Natural bags are represented as flattened instances
    plus ``bag_sizes``/``bag_index``; no padded entry participates in a mean.
    """
    def __init__(self,epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        self.detach_global = self.hparams.get("detach_global", False)
        self.eps = self.hparams.get("eps", 1e-12)
        self.register_buffer(
            "_nonclip_dataset_prior", torch.empty(0), persistent=False
        )

    def configure_dataset_prior(self, class_prior):
        """Set the pooled instance-level label prior from the training set."""
        prior = torch.as_tensor(class_prior, dtype=torch.float32)
        if prior.dim() != 1 or prior.numel() != self.num_classes:
            raise ValueError(
                f"NonClipOVR class_prior must have shape [{self.num_classes}], "
                f"got {tuple(prior.shape)}"
            )
        if not torch.isfinite(prior).all() or bool((prior < 0).any()):
            raise ValueError(
                "NonClipOVR class_prior must be finite and non-negative"
            )
        if not torch.isclose(prior.sum(), prior.new_tensor(1.0), atol=1e-5):
            raise ValueError("NonClipOVR class_prior must sum to one")
        self._nonclip_dataset_prior = prior.to(
            self._nonclip_dataset_prior.device
        )

    def update(self, minibatches):
        x, proportions = minibatches
        logits = self.predict(x)
        loss = self.NonClipOVR_Loss(
            logits, proportions, leave_one_bag_out=True
        )
        return self._backward_step(loss)

    def NonClipOVR_Loss(
        self,
        outputs: torch.Tensor,
        proportions: torch.Tensor,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
        *,
        leave_one_bag_out: bool = True,
    ) -> torch.Tensor:
        """Compute the fixed- or variable-bag NonClipOVR objective.

        ``leave_one_bag_out=True`` is the paper-aligned training path: for
        every bag, ``mu_h`` is the pooled prediction mean over all valid
        instances in the *other* bags.  Passing ``False`` retains the
        historical fixed-size plug-in calculation for compatibility and an
        exact regression check against old checkpoints.
        """
        if outputs.dim() != 2:
            raise ValueError(
                f"outputs must be 2D [N,C], got {tuple(outputs.shape)}"
            )
        device = outputs.device
        proportions = proportions.to(outputs)

        probs = torch.sigmoid(outputs)  # [N, C]
        N, C = probs.shape
        if proportions.dim() != 2:
            raise ValueError(
                f"proportions must be 2D [B,C], got {tuple(proportions.shape)}"
            )
        B = proportions.shape[0]
        if B <= 0:
            raise ValueError("NonClipOVR requires at least one non-empty bag")
        if proportions.shape[1] != C:
            raise ValueError(
                f"proportions must have C={C} columns, got {tuple(proportions.shape)}"
            )

        if bag_sizes is None:
            fixed_size = int(self.bagsize)
            if fixed_size <= 0:
                raise ValueError(f"bagsize must be positive, got {fixed_size}")
            if N != B * fixed_size:
                raise ValueError(
                    f"N={N} must equal B*bagsize={B * fixed_size}"
                )
            sizes_long = torch.full(
                (B,), fixed_size, dtype=torch.long, device=device
            )
        else:
            sizes_value = torch.as_tensor(bag_sizes, device=device)
            if sizes_value.dim() != 1 or sizes_value.numel() != B:
                raise ValueError(
                    f"bag_sizes must have shape [{B}], got {tuple(sizes_value.shape)}"
                )
            if sizes_value.is_floating_point():
                rounded = sizes_value.round()
                if not torch.equal(sizes_value, rounded):
                    raise ValueError("bag_sizes must contain integer values")
                sizes_long = rounded.to(torch.long)
            else:
                sizes_long = sizes_value.to(torch.long)
            if bool((sizes_long <= 0).any()):
                raise ValueError("NonClipOVR does not support empty bags")
            if int(sizes_long.sum().item()) != N:
                raise ValueError(
                    f"sum(bag_sizes)={int(sizes_long.sum().item())} must equal N={N}"
                )

        if bag_index is None:
            bag_index_long = torch.repeat_interleave(
                torch.arange(B, device=device), sizes_long
            )
        else:
            bag_index_long = torch.as_tensor(
                bag_index, dtype=torch.long, device=device
            )
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        sizes = sizes_long.to(probs)
        bag_prediction_sums = torch.zeros(
            (B, C), dtype=probs.dtype, device=device
        )
        bag_prediction_sums.scatter_add_(
            0, bag_index_long[:, None].expand_as(probs), probs
        )
        bag_prediction_means = bag_prediction_sums / sizes[:, None]

        if self._nonclip_dataset_prior.numel() == C:
            p_hat = self._nonclip_dataset_prior.to(proportions)
        else:
            p_hat = (
                sizes[:, None] * proportions
            ).sum(dim=0) / sizes.sum()

        if leave_one_bag_out:
            if B < 2:
                raise ValueError(
                    "NonClipOVR leave-one-bag-out requires at least 2 bags "
                    "per minibatch"
                )
            total_prediction_sum = bag_prediction_sums.sum(dim=0)
            outside_sizes = sizes.sum() - sizes
            if bool((outside_sizes <= 0).any()):
                raise ValueError(
                    "NonClipOVR leave-one-bag-out found no outside instances"
                )
            prediction_mean = (
                total_prediction_sum[None, :] - bag_prediction_sums
            ) / outside_sizes[:, None]
        else:
            # Exact legacy fixed-size plug-in behavior: the current bag is
            # included in the global prediction mean.
            prediction_mean = probs.mean(dim=0).expand(B, -1)

        if self.detach_global:
            p_hat = p_hat.detach()
            prediction_mean = prediction_mean.detach()

        offset = prediction_mean - p_hat[None, :]
        centered_residual = proportions - bag_prediction_means + offset
        loss_per_bag = (
            sizes * centered_residual.pow(2).sum(dim=1)
            + offset.pow(2).sum(dim=1)
        )
        return loss_per_bag.mean()


class ROT(Algorithm):
    """
    ROTLoss = alpha * loss_prop + (1-alpha) * loss_u

    hparams:
      - alpha: bag loss weight (default 0.6). instance weight is (1-alpha)
      - sinkhorn_iterations: (default 3)
      - eps: (default 1e-12)
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        # ---- hyperparams ----
        self.alpha = float(self.hparams.get("alpha", 0.6))
        self.alpha = max(0.0, min(1.0, self.alpha))  # clamp to [0,1]
        self.sinkhorn_iterations = int(self.hparams.get("sinkhorn_iterations", 3))
        self.eps = float(self.hparams.get("eps", 1e-12))

    @staticmethod
    @torch.no_grad()
    def _distributed_sinkhorn(Q: torch.Tensor, sinkhorn_iterations: int,
                              target_counts: torch.Tensor, eps: float = 1e-12):
        """
        Q: [m, C] non-negative (usually probs)
        target_counts: [C] desired column sums (counts per class), e.g. proportions * m
        Returns: adjusted Q with column sums matching target_counts (approximately),
                 then row-normalized outside if needed.
        """
        Q = Q.clamp_min(eps)

        target_counts = target_counts.clamp_min(0)
        if target_counts.sum() <= eps:
            target_counts = torch.ones_like(target_counts)

        # normalize to probability mass (sum=1) for stability
        target_mass = target_counts / target_counts.sum().clamp_min(eps)

        for _ in range(sinkhorn_iterations):
            Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)  # row norm
            Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(eps)  # col norm
            Q = Q * target_mass.unsqueeze(0)                   # set col marginals

        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(eps)
        Q = Q * target_mass.unsqueeze(0)
        return Q

    @staticmethod
    def _bag_ce_loss(proportion: torch.Tensor, pred_mean: torch.Tensor, eps: float = 1e-12):
        """
        Cross-entropy between bag proportions and mean prediction in the bag.
        proportion: [C]
        pred_mean:  [C]
        """
        proportion = proportion.clamp_min(0)
        proportion = proportion / proportion.sum().clamp_min(eps)

        pred_mean = pred_mean.clamp_min(eps)
        pred_mean = pred_mean / pred_mean.sum().clamp_min(eps)

        return -(proportion * pred_mean.log()).sum()

    def update(self, minibatches):
        x, proportions = minibatches
        logits = self.predict(x)

        loss, stats = self.ROTLoss_Loss(logits, proportions)
        out = self._backward_step(loss)
        out.update(stats)
        return out

    def ROTLoss_Loss(self, logits: torch.Tensor, proportions: torch.Tensor):
        device = logits.device
        proportions = proportions.to(device)

        N, C = logits.shape
        m = self.bagsize
        assert N % m == 0, f"N={N} 不能被 bagsize={m} 整除"
        B = N // m
        if proportions.shape != (B, C):
            raise ValueError(f"proportions should be [B,C]=[{B},{C}], got {tuple(proportions.shape)}")

        probs = F.softmax(logits, dim=1)      # [N,C]
        probs_bag = probs.view(B, m, C)       # [B,m,C]
        pred_mean = probs_bag.mean(dim=1)     # [B,C]

        # -------- loss_prop (bag-level CE) --------
        loss_prop = 0.0
        for j in range(B):
            loss_prop = loss_prop + self._bag_ce_loss(proportions[j], pred_mean[j], eps=self.eps)
        loss_prop = loss_prop / B

        # -------- pseudo labels via sinkhorn --------
        with torch.no_grad():
            probs_det = probs.detach()
            pseudo_all, score_all = [], []
            for j in range(B):
                chunk = probs_det[j * m:(j + 1) * m]  # [m,C]
                target_counts = (proportions[j].clamp_min(0) * float(m)).to(chunk.device).to(chunk.dtype)

                adjusted = self._distributed_sinkhorn(
                    chunk, self.sinkhorn_iterations, target_counts, eps=self.eps
                )  # [m,C]

                adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(self.eps)  # row-normalize
                scores, pseudo = adjusted.max(dim=1)  # [m], [m]
                pseudo_all.append(pseudo)
                score_all.append(scores)

            pseudo = torch.cat(pseudo_all, dim=0)  # [N]
            scores = torch.cat(score_all, dim=0)   # [N]

        # -------- loss_u (instance-level CE on pseudo labels) --------
        loss_u = F.cross_entropy(logits, pseudo, reduction="mean")

        # -------- combined loss --------
        alpha_bag = self.alpha
        alpha_inst = 1.0 - self.alpha
        loss = alpha_bag * loss_prop + alpha_inst * loss_u

        stats = {
            "loss_prop": float(loss_prop.item()),
            "loss_u": float(loss_u.item()),
        }
        return loss, stats


class EasyLLP(Algorithm):
    """
    EasyLLP (minimal, raw-weight version)

    Weight:
      w_{b,c} = m_b*alpha_{b,c} - (m_b-1)*prior_c,
      prior_c = sum_b(m_b*alpha_{b,c}) / sum_b(m_b)
      w_{i,c} = w_{bag(i),c}

    Loss options:
      - "ce":  L = - mean_i sum_c w_{i,c} * log p_{i,c}
      - "se":  L = mean_{i,c} w_{i,c} * (1 - p_{i,c})^2

    ``bag_sizes`` and ``bag_index`` are optional for backward compatibility.
    When omitted, the legacy fixed-size layout defined by ``self.bagsize`` is
    used.  Corrected weights are intentionally allowed to be negative.
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        self.loss_type = str(self.hparams.get("loss_type", "ce")).lower()  # "ce" or "se"

        # ---- flooding hyperparams ----
        self.flooding = bool(self.hparams.get("flooding", False))
        self.flooding_b = float(self.hparams.get("flooding_b", 0.0))

    @staticmethod
    def _apply_flooding(loss: torch.Tensor, b: float) -> torch.Tensor:
        # b=0 is the absolute-risk correction used by the ABS baselines.
        # Preserve the historical negative-threshold no-op behavior.
        if b < 0:
            return loss
        return (loss - b).abs() + b

    def update(self, minibatches):
        x, proportions = minibatches  # proportions: [B,C]
        logits = self.predict(x)      # logits: [N,C]

        loss, stats = self.EasyLLP_Loss(logits, proportions)
        if self.flooding:
            loss = self._apply_flooding(loss, self.flooding_b)

        out = self._backward_step(loss)
        # out.update({k: (v if not isinstance(v, torch.Tensor) else float(v.item())) for k, v in stats.items()})
        return out

    def EasyLLP_Loss(
        self,
        outputs: torch.Tensor,
        proportions: torch.Tensor,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        """Compute the EasyLLP risk for fixed or variable-size bags.

        Args:
            outputs: Flattened instance logits with shape ``[N, C]``.
            proportions: Bag proportions with shape ``[B, C]``.
            bag_sizes: Number of valid instances in every bag.  If omitted,
                every bag is assumed to have legacy size ``self.bagsize``.
            bag_index: Bag membership for each flattened instance.  If
                omitted, instances must be contiguous by bag and membership
                is constructed from ``bag_sizes``.

        The pooled prior and final loss are both instance-weighted.  No clamp
        or renormalization is applied to the corrected EasyLLP coefficients.
        """
        if outputs.dim() != 2:
            raise ValueError(f"outputs must be 2D [N,C], got {tuple(outputs.shape)}")
        device = outputs.device
        proportions = proportions.to(device)

        log_p = F.log_softmax(outputs, dim=1)  # [N,C]
        probs = log_p.exp()                   # [N,C]

        N, C = probs.shape
        if proportions.dim() != 2:
            raise ValueError(f"proportions must be 2D [B,C], got {tuple(proportions.shape)}")
        B = proportions.shape[0]
        if B <= 0:
            raise ValueError("EasyLLP requires at least one non-empty bag")
        if proportions.shape[1] != C:
            raise ValueError(
                f"proportions must have C={C} columns, got {tuple(proportions.shape)}"
            )

        if bag_sizes is None:
            m = int(self.bagsize)
            if m <= 0:
                raise ValueError(f"bagsize must be positive, got {m}")
            if N != B * m:
                raise ValueError(f"N={N} must equal B*bagsize={B * m}")
            sizes_long = torch.full((B,), m, dtype=torch.long, device=device)
        else:
            sizes_value = torch.as_tensor(bag_sizes, device=device)
            if sizes_value.dim() != 1 or sizes_value.numel() != B:
                raise ValueError(
                    f"bag_sizes must have shape [{B}], got {tuple(sizes_value.shape)}"
                )
            if sizes_value.is_floating_point():
                rounded = sizes_value.round()
                if not torch.equal(sizes_value, rounded):
                    raise ValueError("bag_sizes must contain integer values")
                sizes_long = rounded.to(torch.long)
            else:
                sizes_long = sizes_value.to(torch.long)
            if bool((sizes_long <= 0).any()):
                raise ValueError("EasyLLP does not support empty bags")
            if int(sizes_long.sum().item()) != N:
                raise ValueError(
                    f"sum(bag_sizes)={int(sizes_long.sum().item())} must equal N={N}"
                )

        if bag_index is None:
            bag_index_long = torch.repeat_interleave(
                torch.arange(B, device=device), sizes_long
            )
        else:
            bag_index_long = torch.as_tensor(
                bag_index, dtype=torch.long, device=device
            )
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        sizes = sizes_long.to(dtype=proportions.dtype)
        prior = (sizes[:, None] * proportions).sum(dim=0) / sizes.sum()
        bag_weight = (
            sizes[:, None] * proportions
            - (sizes[:, None] - 1.0) * prior
        )
        instance_weight = bag_weight[bag_index_long]

        stats = {"loss_type": self.loss_type}

        if self.loss_type == "se":
            loss = ((1.0 - probs).pow(2) * instance_weight).mean()
        elif self.loss_type == "ce":
            loss = -(instance_weight * log_p).sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown loss_type={self.loss_type}, choose 'ce' or 'se'.")

        return loss, stats


class GeneralUPM(Algorithm):
    """
    Implements Eq.(28): full-histogram multi-class bag-level estimator.
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        self.flooding = bool(self.hparams.get("flooding", False))
        self.flooding_b = float(self.hparams.get("flooding_b", 0.0))

    @staticmethod
    def _apply_flooding(loss: torch.Tensor, b: float) -> torch.Tensor:
        # b=0 is the absolute-risk correction used by the ABS baselines.
        # Preserve the historical negative-threshold no-op behavior.
        if b < 0:
            return loss
        return (loss - b).abs() + b

    def update(self, minibatches):
        x, proportions = minibatches  # proportions: [B,C]
        logits = self.predict(x)      # logits: [N,C]

        loss = self.GeneralUPM_Loss(logits, proportions)
        if self.flooding:
            loss = self._apply_flooding(loss, self.flooding_b)

        return self._backward_step(loss)

    def GeneralUPM_Loss(self, outputs, proportions):
        device = outputs.device
        proportions = proportions.to(device)

        N, C = outputs.shape
        m = self.bagsize
        if N % m != 0:
            raise ValueError(f"N={N} must be divisible by bagsize={m}")
        B = N // m

        if proportions.dim() != 2 or proportions.shape != (B, C):
            raise ValueError(f"proportions must be [B,C]=[{B},{C}], got {tuple(proportions.shape)}")

        log_p = F.log_softmax(outputs, dim=1)  # [N,C]
        ell = -log_p                           # [N,C]

        p_hat = proportions.mean(dim=0)        # [C]
        E_hat = ell.mean(dim=0)                # [C]

        ell_bag = ell.view(B, m, C)            # [B,m,C]
        centered = ell_bag - E_hat.view(1, 1, C)
        sum_centered = centered.sum(dim=1)     # [B,C]

        term1 = ((proportions - p_hat.view(1, C)) * sum_centered).sum(dim=1)  # [B]
        term2 = (p_hat * E_hat).sum()          # scalar

        loss_per_bag = term1 + term2
        return loss_per_bag.mean()
class LLP_DSQ(Algorithm):
    """
    LLP_DSQ
    Reference: Optimistic rates for learning from label proportions, COLT 2024.

    This is the variable-size extension of the original DSQ debiasing
    objective.  Each fixed-size factor ``k`` is replaced by the corresponding
    valid bag size ``k_i`` and the global coefficient becomes
    ``mean_i(k_i - 1)``.  Equal-size bags exactly recover the fixed-size
    objective.
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super(LLP_DSQ, self).__init__(epochs,input_shape, train_givenY, hparams, bagsize)

        # The paper's streaming implementation uses beta=0.99.  A value of
        # zero is the legacy behavior: use the current pooled minibatch mean.
        self.dsq_ema_beta = float(self.hparams.get("dsq_ema_beta", 0.0))
        if not 0.0 <= self.dsq_ema_beta < 1.0:
            raise ValueError("dsq_ema_beta must be in [0, 1)")

        # These buffers are deterministic/streaming training auxiliaries, not
        # model parameters.  Keeping them non-persistent preserves old DSQ
        # checkpoint compatibility.
        self.register_buffer(
            "_dsq_prediction_ema",
            torch.zeros(self.num_classes),
            persistent=False,
        )
        self.register_buffer(
            "_dsq_ema_initialized",
            torch.tensor(False),
            persistent=False,
        )
        self.register_buffer(
            "_dsq_dataset_prior",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_dsq_dataset_mean_k_minus_1",
            torch.tensor(float("nan")),
            persistent=False,
        )

    def configure_dataset_statistics(self, class_prior, mean_k_minus_1):
        """Configure training-set DSQ statistics calculated by the loader.

        ``class_prior`` must be the pooled instance-level prior and
        ``mean_k_minus_1`` must average over training bags, not minibatches.
        """
        prior = torch.as_tensor(class_prior, dtype=torch.float32)
        if prior.dim() != 1 or prior.numel() != self.num_classes:
            raise ValueError(
                f"DSQ class_prior must have shape [{self.num_classes}], "
                f"got {tuple(prior.shape)}"
            )
        if not torch.isfinite(prior).all() or bool((prior < 0).any()):
            raise ValueError("DSQ class_prior must be finite and non-negative")
        if not torch.isclose(prior.sum(), prior.new_tensor(1.0), atol=1e-5):
            raise ValueError("DSQ class_prior must sum to one")
        coefficient = float(mean_k_minus_1)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("DSQ mean_k_minus_1 must be finite and non-negative")
        self._dsq_dataset_prior = prior.to(self._dsq_prediction_ema.device)
        self._dsq_dataset_mean_k_minus_1.fill_(coefficient)

    def reset_prediction_ema(self):
        self._dsq_prediction_ema.zero_()
        self._dsq_ema_initialized.fill_(False)

    def update(self, minibatches):
        x, proportions = minibatches
        loss = self.DSQ_Loss(
            self.predict(x),
            proportions,
            update_ema=True,
        )
        return self._backward_step(loss)

    def _prediction_mean_for_correction(
        self,
        batch_prediction_mean: torch.Tensor,
        update_ema: bool,
    ) -> torch.Tensor:
        """Return the plug-in prediction mean used by the DSQ correction.

        Historical EMA state is detached, while the current pooled minibatch
        observation remains in the autograd graph, matching the streaming
        approximation described in the COLT 2024 experimental appendix.
        """
        if not update_ema or self.dsq_ema_beta == 0.0:
            correction_mean = batch_prediction_mean
        elif not bool(self._dsq_ema_initialized.item()):
            correction_mean = batch_prediction_mean
        else:
            correction_mean = (
                self.dsq_ema_beta * self._dsq_prediction_ema.detach()
                + (1.0 - self.dsq_ema_beta) * batch_prediction_mean
            )

        if update_ema:
            with torch.no_grad():
                self._dsq_prediction_ema.copy_(correction_mean.detach())
                self._dsq_ema_initialized.fill_(True)
        return correction_mean

    def DSQ_Loss(
        self,
        outputs: torch.Tensor,
        proportions: torch.Tensor,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
        update_ema: bool = False,
    ) -> torch.Tensor:
        """Compute DSQ over flattened fixed-size or natural variable bags."""
        if outputs.dim() != 2:
            raise ValueError(f"outputs must be 2D [N,C], got {tuple(outputs.shape)}")
        device = outputs.device
        proportions = proportions.to(device)
        probs = F.softmax(outputs, dim=1)  # [N, C]

        N, C = probs.shape
        if proportions.dim() != 2:
            raise ValueError(
                f"proportions must be 2D [B,C], got {tuple(proportions.shape)}"
            )
        B = proportions.shape[0]
        if B <= 0:
            raise ValueError("DSQ requires at least one non-empty bag")
        if proportions.shape[1] != C:
            raise ValueError(
                f"proportions must have C={C} columns, got {tuple(proportions.shape)}"
            )

        if bag_sizes is None:
            fixed_size = int(self.bagsize)
            if fixed_size <= 0:
                raise ValueError(f"bagsize must be positive, got {fixed_size}")
            if N != B * fixed_size:
                raise ValueError(f"N={N} must equal B*bagsize={B * fixed_size}")
            sizes_long = torch.full(
                (B,), fixed_size, dtype=torch.long, device=device
            )
        else:
            sizes_value = torch.as_tensor(bag_sizes, device=device)
            if sizes_value.dim() != 1 or sizes_value.numel() != B:
                raise ValueError(
                    f"bag_sizes must have shape [{B}], got {tuple(sizes_value.shape)}"
                )
            if sizes_value.is_floating_point():
                rounded = sizes_value.round()
                if not torch.equal(sizes_value, rounded):
                    raise ValueError("bag_sizes must contain integer values")
                sizes_long = rounded.to(torch.long)
            else:
                sizes_long = sizes_value.to(torch.long)
            if bool((sizes_long <= 0).any()):
                raise ValueError("DSQ does not support empty bags")
            if int(sizes_long.sum().item()) != N:
                raise ValueError(
                    f"sum(bag_sizes)={int(sizes_long.sum().item())} must equal N={N}"
                )

        if bag_index is None:
            bag_index_long = torch.repeat_interleave(
                torch.arange(B, device=device), sizes_long
            )
        else:
            bag_index_long = torch.as_tensor(
                bag_index, dtype=torch.long, device=device
            )
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        sizes = sizes_long.to(dtype=proportions.dtype)
        prediction_sums = torch.zeros(
            (B, C), dtype=probs.dtype, device=device
        )
        prediction_sums.scatter_add_(
            0, bag_index_long[:, None].expand_as(probs), probs
        )
        predicted_proportions = prediction_sums / sizes.to(probs)[:, None]

        # T1 is a bag mean.  k_i appears exactly once inside each bag term.
        per_bag_mse = (predicted_proportions - proportions).pow(2).mean(dim=1)
        term1 = (sizes * per_bag_mse).mean()

        # The minibatch observation is pooled across all valid instances, not
        # an unweighted mean of bag prediction means.
        batch_prediction_mean = probs.mean(dim=0)
        correction_prediction_mean = self._prediction_mean_for_correction(
            batch_prediction_mean, bool(update_ema)
        )

        if self._dsq_dataset_prior.numel() == C:
            global_prior = self._dsq_dataset_prior.to(proportions)
        else:
            global_prior = (
                sizes[:, None] * proportions
            ).sum(dim=0) / sizes.sum()

        if torch.isfinite(self._dsq_dataset_mean_k_minus_1):
            mean_k_minus_1 = self._dsq_dataset_mean_k_minus_1.to(proportions)
        else:
            mean_k_minus_1 = (sizes - 1.0).mean()

        correction_mse = (
            correction_prediction_mean - global_prior
        ).pow(2).mean()
        term2 = mean_k_minus_1 * correction_mse

        # Multiplying by C converts the class-mean MSE convention retained by
        # the legacy multiclass code into the squared L2 norm in the paper.
        return (term1 - term2) * C
def init_last_linear_bias_sigmoid_to_1_over_k(model: nn.Module, num_classes: int):
    K = int(num_classes)
    if K <= 1:
        return
    b0 = -math.log(K - 1)

    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is not None and last_linear.bias is not None:
        with torch.no_grad():
            last_linear.bias.fill_(b0)


class LLP_PVC(Algorithm):
    """
    LLP_PVC
    Reference: Learning from Label Proportions via Proportional Value Classification, ICLR2026.

    PVC is defined independently for every bag and therefore naturally
    supports different bag sizes.  Variable bags are represented by flattened
    valid instances plus ``bag_sizes`` and ``bag_index``.  Their per-bag count
    likelihoods are aggregated with an instance-weighted mean, matching the
    variable-length helper released by the authors.
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super(LLP_PVC, self).__init__(epochs,input_shape, train_givenY, hparams, bagsize)
        init_last_linear_bias_sigmoid_to_1_over_k(self.network, self.num_classes)

    def update(self, minibatches):
        x,proportions= minibatches
        loss = self.PVC_Loss(self.predict(x),proportions)
        return self._backward_step(loss)

    def PVC_Loss(
        self,
        outputs: torch.Tensor,
        proportions: torch.Tensor,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute PVC count-likelihood loss for fixed or variable bags.

        Each target count is derived using that bag's actual size ``k_i``.
        Bags of equal size are grouped so the existing batched FFT kernel is
        retained.  The final reduction is ``sum_i k_i * loss_i / sum_i k_i``;
        when all bags have the same size this is exactly the legacy bag mean.
        """
        if outputs.dim() != 2:
            raise ValueError(
                f"outputs must be 2D [N,C], got {tuple(outputs.shape)}"
            )
        device = outputs.device
        proportions = proportions.to(outputs)
        probs = torch.sigmoid(outputs)

        N, C = probs.shape
        if proportions.dim() != 2:
            raise ValueError(
                f"proportions must be 2D [B,C], got {tuple(proportions.shape)}"
            )
        B = proportions.shape[0]
        if B <= 0:
            raise ValueError("LLP_PVC requires at least one non-empty bag")
        if proportions.shape[1] != C:
            raise ValueError(
                f"proportions must have C={C} columns, got {tuple(proportions.shape)}"
            )

        if bag_sizes is None:
            fixed_size = int(self.bagsize)
            if fixed_size <= 0:
                raise ValueError(f"bagsize must be positive, got {fixed_size}")
            if N != B * fixed_size:
                raise ValueError(
                    f"N={N} must equal B*bagsize={B * fixed_size}"
                )
            sizes_long = torch.full(
                (B,), fixed_size, dtype=torch.long, device=device
            )
        else:
            sizes_value = torch.as_tensor(bag_sizes, device=device)
            if sizes_value.dim() != 1 or sizes_value.numel() != B:
                raise ValueError(
                    f"bag_sizes must have shape [{B}], got {tuple(sizes_value.shape)}"
                )
            if sizes_value.is_floating_point():
                rounded = sizes_value.round()
                if not torch.equal(sizes_value, rounded):
                    raise ValueError("bag_sizes must contain integer values")
                sizes_long = rounded.to(torch.long)
            else:
                sizes_long = sizes_value.to(torch.long)
            if bool((sizes_long <= 0).any()):
                raise ValueError("LLP_PVC does not support empty bags")
            if int(sizes_long.sum().item()) != N:
                raise ValueError(
                    f"sum(bag_sizes)={int(sizes_long.sum().item())} must equal N={N}"
                )

        if bag_index is None:
            bag_index_long = torch.repeat_interleave(
                torch.arange(B, device=device), sizes_long
            )
        else:
            bag_index_long = torch.as_tensor(
                bag_index, dtype=torch.long, device=device
            )
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        # The FFT kernel requires a rectangular [B_group, k, C] tensor.  Only
        # bags with the same k_i are stacked together; no padding is created.
        loss_chunks = []
        bag_id_chunks = []
        for current_size in torch.unique(sizes_long, sorted=True).tolist():
            bag_ids = torch.nonzero(
                sizes_long == int(current_size), as_tuple=False
            ).squeeze(1)
            instance_indices = torch.stack(
                [
                    torch.nonzero(
                        bag_index_long == bag_id, as_tuple=False
                    ).squeeze(1)
                    for bag_id in bag_ids
                ],
                dim=0,
            )
            grouped_probabilities = probs[instance_indices]
            grouped_losses = compute_CC_loss_fft_precise_batched(
                grouped_probabilities,
                proportions[bag_ids],
                reduce=None,
            )
            loss_chunks.append(grouped_losses)
            bag_id_chunks.append(bag_ids)

        unordered_ids = torch.cat(bag_id_chunks)
        unordered_losses = torch.cat(loss_chunks)
        loss_per_bag = unordered_losses[torch.argsort(unordered_ids)]
        sizes = sizes_long.to(loss_per_bag)
        return (sizes * loss_per_bag).sum() / sizes.sum()


class LLP_PT(Algorithm):
    """
    LLP_PT
    Reference: Progressive Training for Learning from Label Proportions, TNNLS 2025.
    """
    def __init__(self,epochs, input_shape, train_givenY, hparams, bagsize):
        super(LLP_PT, self).__init__(epochs,input_shape, train_givenY, hparams, bagsize)
        self._eps = 1e-12

    def update(self, minibatches):
        x, proportions = minibatches
        logits = self.predict(x)
        loss = self.PT_Loss(logits, proportions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item()}

    def PT_Loss(
        self,
        outputs,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        proportions = proportions.to(device=device, dtype=torch.float32)
        probs = F.softmax(outputs, dim=1).clamp_min(self._eps)

        N, C = probs.shape
        B = proportions.shape[0]
        if proportions.dim() != 2 or proportions.shape[1] != C:
            raise ValueError(
                f"proportions must have shape [B,{C}], got {tuple(proportions.shape)}"
            )
        if bag_sizes is None:
            fixed_size = int(self.bagsize)
            if fixed_size <= 0:
                raise ValueError(f"bagsize must be positive, got {fixed_size}")
            if N != B * fixed_size:
                raise ValueError(
                    f"N={N} must equal B*bagsize={B * fixed_size}"
                )
            sizes_long = torch.full(
                (B,), fixed_size, dtype=torch.long, device=device
            )
        else:
            sizes_value = torch.as_tensor(bag_sizes, device=device)
            if sizes_value.dim() != 1 or sizes_value.numel() != B:
                raise ValueError(
                    f"bag_sizes must have shape [{B}], got {tuple(sizes_value.shape)}"
                )
            if sizes_value.is_floating_point():
                if not torch.equal(sizes_value, sizes_value.round()):
                    raise ValueError("bag_sizes must contain integer values")
            sizes_long = sizes_value.to(torch.long)
            if bool((sizes_long <= 0).any()):
                raise ValueError("bag_sizes must be positive")
            if int(sizes_long.sum().item()) != N:
                raise ValueError(
                    f"sum(bag_sizes)={int(sizes_long.sum().item())} must equal N={N}"
                )
        if bag_index is None:
            bag_index_long = torch.repeat_interleave(
                torch.arange(B, device=device), sizes_long
            )
        else:
            bag_index_long = torch.as_tensor(
                bag_index, dtype=torch.long, device=device
            )
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        ce_list, rce_list = [], []
        for b in range(B):
            indices = torch.nonzero(
                bag_index_long == b, as_tuple=False
            ).squeeze(1)
            probs_bag = probs[indices]
            logits_bag = outputs[indices]
            target_prop = proportions[b].clamp_min(self._eps)
            target_prop = target_prop / target_prop.sum()
            soft_targets = self.emd_hard_assign(
                probs_bag, target_prop
            )  # [k_i, C] hard one-hot

            logp = F.log_softmax(logits_bag, dim=1)
            ce = -(soft_targets * logp).sum(dim=1).mean()

            st_clamped = soft_targets.clamp_min(self._eps)
            rce = -(probs_bag * torch.log(st_clamped)).sum(dim=1).mean()

            ce_list.append(ce)
            rce_list.append(rce)

        ce_loss = torch.stack(ce_list).mean() if ce_list else outputs.new_tensor(0.0)
        rce_loss = torch.stack(rce_list).mean() if rce_list else outputs.new_tensor(0.0)
        total_loss = ce_loss + rce_loss

        return total_loss

    def emd_hard_assign(self, probs, target_prop):
        """
        Emulate LLP_PT.optimize_ot: exact EMD then argmax -> one-hot.
        probs: [s, C] (on device), target_prop: [C]
        returns one-hot tensor [s, C] on same device.
        """
        if ot is None:
            raise ImportError("LLP_PT requires POT; install it with `pip install POT`")
        s, C = probs.shape
        PS = probs.detach().cpu().numpy()
        a = target_prop.detach().cpu().numpy().astype(float)
        b = np.ones(s, dtype=float) / s
        M = -np.log(PS + self._eps)
        T = ot.emd(b, a, M)  # shape s x C
        argmaxes = np.argmax(T, axis=1)
        out = np.eye(C, dtype=np.float32)[argmaxes]
        return torch.from_numpy(out).to(probs.device, dtype=probs.dtype)

class LLP_SimCLR(Algorithm):
    """
    LLP_SimCLR
    Reference: A two-stage training framework with feature-label matching mechanism for learning from label proportions, ACML 2021.
    """
    class SupConResNet(nn.Module):
        """backbone + projection head"""
        def __init__(self, name_classes, input_shape, hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(self.featurizer.n_outputs, name_classes)
            feature_dimension = int(
                hparams.get("feat_dim", self.featurizer.n_outputs)
            )
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, feature_dimension))

        def forward(self, x):
            feat = self.featurizer(x)
            logits = self.classifier(feat)
            # Stage two updates the main classifier, so it must consume the
            # same backbone feature space as that classifier.  The previous
            # projection-head output could have a different dimension.
            return logits, feat

    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super(LLP_SimCLR, self).__init__(epochs,input_shape, train_givenY, hparams, bagsize)
        self.entropy_weight = self.hparams["entropy_weight"]
        self.match_weight = self.hparams["match_weight"]
        self.match_threshold = self.hparams["match_threshold"]
        self.match_min_ratio = self.hparams["match_min_ratio"]
        self.network = self.SupConResNet(self.num_classes, input_shape, self.hparams)
        self.featurizer = self.network.featurizer
        self.classifier = self.network.classifier
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams.get("weight_decay", 1e-4),
        )
        self._eps = 1e-12



    def update(self, minibatches):
        x, proportions = minibatches
        logits, features = self.network(x)
        return self.update_from_outputs(logits, features, proportions)

    def update_from_outputs(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        proportions: torch.Tensor,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run the original two-stage update on flattened valid instances."""
        device = logits.device
        proportions = proportions.to(device=device, dtype=torch.float32)
        probs = F.softmax(logits, dim=1).clamp_min(self._eps)

        N, C = probs.shape
        if features.dim() != 2 or features.shape[0] != N:
            raise ValueError(
                f"features must have shape [N,D] with N={N}, "
                f"got {tuple(features.shape)}"
            )
        B = proportions.shape[0]
        sizes, index = _resolve_bag_layout(
            N,
            proportions,
            C,
            device,
            self.bagsize,
            bag_sizes,
            bag_index,
        )

        base_loss = self.stage1_loss(
            proportions,
            probs,
            B,
            self.bagsize,
            C,
            bag_sizes=sizes,
            bag_index=index,
        )
        self.optimizer.zero_grad()
        base_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        match_loss = self.stage2_loss(
            proportions,
            probs,
            features,
            B,
            self.bagsize,
            C,
            device,
            bag_sizes=sizes,
            bag_index=index,
        )

        total_loss = base_loss + match_loss
        return {"loss": total_loss.item()}

    def stage1_loss(
        self,
        proportions,
        probs,
        B,
        s,
        C,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        proportions = proportions.clamp_min(self._eps)
        proportions = proportions / proportions.sum(dim=1, keepdim=True).clamp_min(self._eps)
        _, index = _resolve_bag_layout(
            len(probs),
            proportions,
            C,
            probs.device,
            s,
            bag_sizes,
            bag_index,
        )
        pred_prop = _bag_means(probs, index, B).clamp_min(self._eps)
        kl_loss = F.kl_div(torch.log(pred_prop), proportions, reduction="batchmean")
        entropy_loss = -(probs * torch.log(probs)).sum(dim=1).mean()
        base_loss = kl_loss + self.entropy_weight * entropy_loss
        return base_loss

    def stage2_loss(
        self,
        proportions,
        probs,
        features,
        B,
        s,
        C,
        device,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        sizes, index = _resolve_bag_layout(
            len(probs),
            proportions,
            C,
            probs.device,
            s,
            bag_sizes,
            bag_index,
        )
        match_total = probs.new_tensor(0.0)
        pred_cls = probs.argmax(dim=1)

        for b in range(B):
            positions = torch.nonzero(index == b, as_tuple=False).squeeze(1)
            bag_predictions = pred_cls[positions]
            bag_features = features[positions]
            min_count = max(
                1,
                int(math.ceil(int(sizes[b].item()) * self.match_min_ratio)),
            )
            candidate_labels = torch.nonzero(proportions[b] > self.match_threshold, as_tuple=False).squeeze(1)
            if candidate_labels.numel() == 0:
                continue

            for label in candidate_labels.tolist():
                idx = torch.nonzero(
                    bag_predictions == label, as_tuple=False
                ).squeeze(1)
                if idx.numel() < min_count:
                    continue

                selected_features = bag_features[idx]
                selected_logits = self.classifier(selected_features.detach())
                targets = torch.full((selected_logits.size(0),), label, dtype=torch.long, device=device)
                nll = F.nll_loss(F.log_softmax(selected_logits, dim=1), targets)

                self.optimizer.zero_grad()
                (self.match_weight * nll).backward()
                self.optimizer.step()
                self.scheduler.step()

                match_total = match_total + nll.detach()

        match_loss = self.match_weight * (match_total / B if B > 0 else probs.new_tensor(0.0))
        return match_loss

    def predict(self, x):
        logits, features = self.network(x)
        return logits

class LLP_FC(Algorithm):
    """
    LLP_FC
    Reference: Learning from label proportions by learning with label noise, NeurIPS 2022.
    """
    def __init__(self, epochs,input_shape, train_givenY, hparams, bagsize):
        super(LLP_FC, self).__init__(epochs,input_shape, train_givenY, hparams, bagsize)
        self.mode = self.hparams["mode"]
        self.group_weight = self.hparams["group_weight"]
        self.entropy_weight = self.hparams["entropy_weight"]
        self.pi_grad_steps = self.hparams["pi_grad_steps"]
        self.pi_grad_lr = self.hparams["pi_grad_lr"]
        self._eps = 1e-12

    def update(self, minibatches):
        x, proportions = minibatches
        logits = self.predict(x)
        loss = self.FC_Loss(logits, proportions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item()}

    def FC_Loss(
        self,
        outputs,
        proportions,
        bag_sizes: torch.Tensor | None = None,
        bag_index: torch.Tensor | None = None,
    ):
        device = outputs.device
        eps = self._eps
        proportions = proportions.to(device=device, dtype=torch.float32)
        probs = F.softmax(outputs, dim=1).clamp_min(eps)

        N, C = probs.shape
        B = proportions.shape[0]
        if proportions.dim() != 2 or proportions.shape[1] != C:
            raise ValueError(
                f"proportions must have shape [B,{C}], got {tuple(proportions.shape)}"
            )
        if bag_sizes is None:
            fixed_size = int(self.bagsize)
            if fixed_size <= 0:
                raise ValueError(f"bagsize must be positive, got {fixed_size}")
            if N != B * fixed_size:
                raise ValueError(
                    f"N={N} must equal B*bagsize={B * fixed_size}"
                )
            sizes_long = torch.full(
                (B,), fixed_size, dtype=torch.long, device=device
            )
        else:
            sizes_value = torch.as_tensor(bag_sizes, device=device)
            if sizes_value.dim() != 1 or sizes_value.numel() != B:
                raise ValueError(
                    f"bag_sizes must have shape [{B}], got {tuple(sizes_value.shape)}"
                )
            if sizes_value.is_floating_point():
                if not torch.equal(sizes_value, sizes_value.round()):
                    raise ValueError("bag_sizes must contain integer values")
            sizes_long = sizes_value.to(torch.long)
            if bool((sizes_long <= 0).any()):
                raise ValueError("bag_sizes must be positive")
            if int(sizes_long.sum().item()) != N:
                raise ValueError(
                    f"sum(bag_sizes)={int(sizes_long.sum().item())} must equal N={N}"
                )
        if bag_index is None:
            bag_index_long = torch.repeat_interleave(
                torch.arange(B, device=device), sizes_long
            )
        else:
            bag_index_long = torch.as_tensor(
                bag_index, dtype=torch.long, device=device
            )
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        bag_probabilities = [
            probs[bag_index_long == bag_id] for bag_id in range(B)
        ]

        # Need at least one complete group of C bags.
        G = B // C
        if G == 0:
            # Fallback to bag-level KL when a mini-batch is too small for LLPFC grouping.
            bag_pred = torch.stack(
                [bag.mean(dim=0) for bag in bag_probabilities]
            ).clamp_min(eps)
            target = proportions.clamp_min(eps)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
            kl_loss = F.kl_div(torch.log(bag_pred), target, reduction="batchmean")
            entropy_loss = -(probs * torch.log(probs)).sum(dim=1).mean()
            total_loss = kl_loss + self.entropy_weight * entropy_loss
            return total_loss


        # Randomly partition B bags into G groups of C bags; drop remainder.
        bag_perm = torch.randperm(B, device=device)[: G * C].view(G, C)
        sizes = sizes_long.to(proportions)
        pi_global = (sizes[:, None] * proportions).sum(dim=0) / sizes.sum()

        total_fc = outputs.new_tensor(0.0)
        group_count = 0

        for g in range(G):
            bag_ids = bag_perm[g]
            pi_mat = proportions[bag_ids]  # [C, C], row c = \hat pi_{i,c}

            if self.mode == "approx":
                alpha = self.estimate_alpha_approx(pi_mat, pi_global)
            else:
                alpha = torch.full((C,), 1.0 / C, device=device, dtype=probs.dtype)
            eta = torch.matmul(pi_mat.t(), alpha).clamp_min(eps)  # [C]
            T = pi_mat * alpha.unsqueeze(1)
            T = T / eta.unsqueeze(0)
            T = T / T.sum(dim=0, keepdim=True).clamp_min(eps)  # numerically column-stochastic

            # Forward correction on each noisy bag c in this group.
            # bag slot c is treated as noisy label c.
            for c in range(C):
                inst_prob = bag_probabilities[int(bag_ids[c].item())]  # [k_i, C]
                noisy_prob = torch.matmul(inst_prob, T.t()).clamp_min(eps)  # [k_i, C]
                total_fc = total_fc - torch.log(noisy_prob[:, c]).mean()
            group_count += 1

        fc_loss = total_fc / max(group_count * C, 1)
        entropy_loss = -(probs * torch.log(probs)).sum(dim=1).mean()
        total_loss = self.group_weight * fc_loss + self.entropy_weight * entropy_loss

        return total_loss

    def estimate_alpha_approx(self, pi_mat, pi_global):
        """
        Match original LLP_FC: solve min 0.5||A x - b||^2 s.t. x in simplex
        via scipy trust-constr (approx_noisy_prior).
        """
        C = pi_mat.size(0)
        A = pi_mat.t().detach().cpu().numpy()  # [C, C]
        b = pi_global.detach().cpu().numpy()

        def ls_error(x, A, b):
            return 0.5 * np.sum((np.matmul(A, x) - b) ** 2)

        def grad(x, A, b):
            return np.matmul(np.matmul(np.transpose(A), A), x) - np.matmul(np.transpose(A), b)

        def hess(x, A, b):
            return np.matmul(np.transpose(A), A)

        x0 = np.random.rand(C)
        x0 /= np.sum(x0)

        res = minimize(
            ls_error,
            x0,
            args=(A, b),
            method="trust-constr",
            jac=grad,
            hess=hess,
            bounds=Bounds(np.zeros(C), np.ones(C)),
            constraints=LinearConstraint(np.ones(C), np.ones(1), np.ones(1)),
        )

        alpha = torch.from_numpy(res.x).to(pi_mat.device, dtype=pi_mat.dtype)
        return alpha

    def project_simplex(self, v):
        """
        Euclidean projection onto simplex {x >= 0, sum x = 1}.
        """
        n = v.numel()
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0
        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho.to(v.dtype) + 1.0)
        w = torch.clamp(v - theta, min=0.0)
        return w / w.sum().clamp_min(self._eps)


# FlowLLP lives in a separate module because its particle-learning lifecycle is
# substantially larger than a single LLP loss, while it still uses the common
# Algorithm registry and training entry point.
from .flowllp_algorithm import build_flowllp_algorithm

LLP_FlowLLP = build_flowllp_algorithm(
    Algorithm,
    WarmupCosineLrScheduler,
    _resolve_bag_layout,
    _bag_means,
)
