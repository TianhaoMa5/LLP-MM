from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from .FFT import compute_CC_loss_fft_precise_batched
from . import networks


class WarmupCosineLrScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        warmup_iter,
        warmup_ratio=5e-4,
        warmup="exp",
        cosine_mode="legacy_quarter",
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
            if self.cosine_mode == "standard":
                ratio = 0.5 * (1.0 + np.cos(np.pi * progress))
            elif self.cosine_mode == "legacy_quarter":
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
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, self.num_classes
        )
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
                f"N={number_of_instances} must equal B*bagsize={number_of_bags * size}"
            )
        sizes = torch.full((number_of_bags,), size, dtype=torch.long, device=device)
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


def _bag_means(
    values: torch.Tensor, bag_index: torch.Tensor, number_of_bags: int
) -> torch.Tensor:
    result = values.new_zeros((number_of_bags, values.shape[1]))
    result.scatter_add_(0, bag_index[:, None].expand_as(values), values)
    counts = torch.bincount(bag_index, minlength=number_of_bags).to(values)
    return result / counts[:, None]


class PM(Algorithm):
    """
    PM
    Reference: On learning from label proportions, ArXiv 2014.
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super(PM, self).__init__(epochs, input_shape, train_givenY, hparams, bagsize)

    def update(self, minibatches):
        x, proportions = minibatches
        loss = self.PM_Loss(self.predict(x), proportions)
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


class LLP_MM(Algorithm):
    """Exact multi-class factorial moment matching with CE supervision.

    Natural bags use the adapter in :func:`mo_matching.data.natural_loss.natural_bag_loss`.
    The shared update below applies the same factorial-moment objective to the
    fixed-size bags used by CV/LEM and the original synthetic benchmarks.  All
    moment probabilities and the stable DP are evaluated in float64; the image
    backbone may remain float32.
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)
        self.order = int(hparams.get("order", 3))
        self.moment_implementation = str(
            hparams.get("moment_implementation", "variable_bag")
        )
        if self.moment_implementation not in {"variable_bag", "paper_image"}:
            raise ValueError("Unknown LLP-MM moment_implementation")
        self.moment_loss_type = str(hparams.get("moment_loss_type", "ce"))
        self.moment_algorithm = str(hparams.get("moment_algorithm", "stable_dp"))
        self.moment_compute_dtype = str(hparams.get("moment_compute_dtype", "float64"))
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
                C=self.num_classes,
                max_order=self.order,
                bag_size=bagsize,
                loss_type="ce",
                weight_mode="uniform",
                order_weights=self.order_weights,
                reduce="mean",
            )

    def update(self, minibatches):
        from ..data.natural_loss import _llp_mm_variable_loss

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
            logits.to(torch.float64)
            .softmax(dim=1)
            .clamp_min(torch.finfo(torch.float64).tiny),
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


class ROT(Algorithm):
    """
    ROTLoss = alpha * loss_prop + (1-alpha) * loss_u

    hparams:
      - alpha: bag loss weight (default 0.6). instance weight is (1-alpha)
      - sinkhorn_iterations: (default 3)
      - eps: (default 1e-12)
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)

        # ---- hyperparams ----
        self.alpha = float(self.hparams.get("alpha", 0.6))
        self.alpha = max(0.0, min(1.0, self.alpha))  # clamp to [0,1]
        self.sinkhorn_iterations = int(self.hparams.get("sinkhorn_iterations", 3))
        self.eps = float(self.hparams.get("eps", 1e-12))

    @staticmethod
    @torch.no_grad()
    def _distributed_sinkhorn(
        Q: torch.Tensor,
        sinkhorn_iterations: int,
        target_counts: torch.Tensor,
        eps: float = 1e-12,
    ):
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
            Q = Q * target_mass.unsqueeze(0)  # set col marginals

        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(eps)
        Q = Q * target_mass.unsqueeze(0)
        return Q

    @staticmethod
    def _bag_ce_loss(
        proportion: torch.Tensor, pred_mean: torch.Tensor, eps: float = 1e-12
    ):
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
            raise ValueError(
                f"proportions should be [B,C]=[{B},{C}], got {tuple(proportions.shape)}"
            )

        probs = F.softmax(logits, dim=1)  # [N,C]
        probs_bag = probs.view(B, m, C)  # [B,m,C]
        pred_mean = probs_bag.mean(dim=1)  # [B,C]

        # -------- loss_prop (bag-level CE) --------
        loss_prop = 0.0
        for j in range(B):
            loss_prop = loss_prop + self._bag_ce_loss(
                proportions[j], pred_mean[j], eps=self.eps
            )
        loss_prop = loss_prop / B

        # -------- pseudo labels via sinkhorn --------
        with torch.no_grad():
            probs_det = probs.detach()
            pseudo_all, score_all = [], []
            for j in range(B):
                chunk = probs_det[j * m : (j + 1) * m]  # [m,C]
                target_counts = (
                    (proportions[j].clamp_min(0) * float(m))
                    .to(chunk.device)
                    .to(chunk.dtype)
                )

                adjusted = self._distributed_sinkhorn(
                    chunk, self.sinkhorn_iterations, target_counts, eps=self.eps
                )  # [m,C]

                adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(
                    self.eps
                )  # row-normalize
                scores, pseudo = adjusted.max(dim=1)  # [m], [m]
                pseudo_all.append(pseudo)
                score_all.append(scores)

            pseudo = torch.cat(pseudo_all, dim=0)  # [N]
            scores = torch.cat(score_all, dim=0)  # [N]

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

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)

        self.loss_type = str(
            self.hparams.get("loss_type", "ce")
        ).lower()  # "ce" or "se"

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
        logits = self.predict(x)  # logits: [N,C]

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
        probs = log_p.exp()  # [N,C]

        N, C = probs.shape
        if proportions.dim() != 2:
            raise ValueError(
                f"proportions must be 2D [B,C], got {tuple(proportions.shape)}"
            )
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
            bag_index_long = torch.as_tensor(bag_index, dtype=torch.long, device=device)
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
        bag_weight = sizes[:, None] * proportions - (sizes[:, None] - 1.0) * prior
        instance_weight = bag_weight[bag_index_long]

        stats = {"loss_type": self.loss_type}

        if self.loss_type == "se":
            loss = ((1.0 - probs).pow(2) * instance_weight).mean()
        elif self.loss_type == "ce":
            loss = -(instance_weight * log_p).sum(dim=1).mean()
        else:
            raise ValueError(
                f"Unknown loss_type={self.loss_type}, choose 'ce' or 'se'."
            )

        return loss, stats


class GeneralUPM(Algorithm):
    """
    Implements Eq.(28): full-histogram multi-class bag-level estimator.
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super().__init__(epochs, input_shape, train_givenY, hparams, bagsize)

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
        logits = self.predict(x)  # logits: [N,C]

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
            raise ValueError(
                f"proportions must be [B,C]=[{B},{C}], got {tuple(proportions.shape)}"
            )

        log_p = F.log_softmax(outputs, dim=1)  # [N,C]
        ell = -log_p  # [N,C]

        p_hat = proportions.mean(dim=0)  # [C]
        E_hat = ell.mean(dim=0)  # [C]

        ell_bag = ell.view(B, m, C)  # [B,m,C]
        centered = ell_bag - E_hat.view(1, 1, C)
        sum_centered = centered.sum(dim=1)  # [B,C]

        term1 = ((proportions - p_hat.view(1, C)) * sum_centered).sum(dim=1)  # [B]
        term2 = (p_hat * E_hat).sum()  # scalar

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

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super(LLP_DSQ, self).__init__(
            epochs, input_shape, train_givenY, hparams, bagsize
        )

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
            sizes_long = torch.full((B,), fixed_size, dtype=torch.long, device=device)
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
            bag_index_long = torch.as_tensor(bag_index, dtype=torch.long, device=device)
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
        prediction_sums = torch.zeros((B, C), dtype=probs.dtype, device=device)
        prediction_sums.scatter_add_(0, bag_index_long[:, None].expand_as(probs), probs)
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
            global_prior = (sizes[:, None] * proportions).sum(dim=0) / sizes.sum()

        if torch.isfinite(self._dsq_dataset_mean_k_minus_1):
            mean_k_minus_1 = self._dsq_dataset_mean_k_minus_1.to(proportions)
        else:
            mean_k_minus_1 = (sizes - 1.0).mean()

        correction_mse = (correction_prediction_mean - global_prior).pow(2).mean()
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

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super(LLP_PVC, self).__init__(
            epochs, input_shape, train_givenY, hparams, bagsize
        )
        init_last_linear_bias_sigmoid_to_1_over_k(self.network, self.num_classes)

    def update(self, minibatches):
        x, proportions = minibatches
        loss = self.PVC_Loss(self.predict(x), proportions)
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
            raise ValueError(f"outputs must be 2D [N,C], got {tuple(outputs.shape)}")
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
                raise ValueError(f"N={N} must equal B*bagsize={B * fixed_size}")
            sizes_long = torch.full((B,), fixed_size, dtype=torch.long, device=device)
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
            bag_index_long = torch.as_tensor(bag_index, dtype=torch.long, device=device)
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
                    torch.nonzero(bag_index_long == bag_id, as_tuple=False).squeeze(1)
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


class LLP_FC(Algorithm):
    """
    LLP_FC
    Reference: Learning from label proportions by learning with label noise, NeurIPS 2022.
    """

    def __init__(self, epochs, input_shape, train_givenY, hparams, bagsize):
        super(LLP_FC, self).__init__(
            epochs, input_shape, train_givenY, hparams, bagsize
        )
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
                raise ValueError(f"N={N} must equal B*bagsize={B * fixed_size}")
            sizes_long = torch.full((B,), fixed_size, dtype=torch.long, device=device)
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
            bag_index_long = torch.as_tensor(bag_index, dtype=torch.long, device=device)
            if bag_index_long.dim() != 1 or bag_index_long.numel() != N:
                raise ValueError(
                    f"bag_index must have shape [{N}], got {tuple(bag_index_long.shape)}"
                )
            if bool(((bag_index_long < 0) | (bag_index_long >= B)).any()):
                raise ValueError("bag_index contains an out-of-range bag id")
            observed_sizes = torch.bincount(bag_index_long, minlength=B)
            if not torch.equal(observed_sizes, sizes_long):
                raise ValueError("bag_index counts do not match bag_sizes")

        bag_probabilities = [probs[bag_index_long == bag_id] for bag_id in range(B)]

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
            T = T / T.sum(dim=0, keepdim=True).clamp_min(
                eps
            )  # numerically column-stochastic

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
            return np.matmul(np.matmul(np.transpose(A), A), x) - np.matmul(
                np.transpose(A), b
            )

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


from .flowllp_algorithm import build_flowllp_algorithm

LLP_FlowLLP = build_flowllp_algorithm(
    Algorithm, WarmupCosineLrScheduler, _resolve_bag_layout, _bag_means
)

ALGORITHMS = [
    "EasyLLP",
    "GeneralUPM",
    "LLP_DSQ",
    "LLP_FC",
    "LLP_FlowLLP",
    "LLP_MM",
    "LLP_PVC",
    "PM",
    "ROT",
]


def get_algorithm_class(name):
    if name not in ALGORITHMS:
        raise ValueError(f"Unsupported paper algorithm: {name}")
    return globals()[name]
