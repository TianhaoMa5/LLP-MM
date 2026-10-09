"""Fixed-size LLP-MM loss used in the paper image experiments."""

from typing import Optional, Sequence, Union, List, Tuple

from functools import lru_cache

import math

import torch

import torch.nn as nn

@lru_cache(None)
def _compositions(C: int, r: int) -> torch.Tensor:
    """
    所有非负整数向量 k∈N^C, sum k=r
    返回 [Nr, C] int64
    """
    out: List[Tuple[int, ...]] = []

    def rec(i: int, left: int, cur: List[int]):
        if i == C - 1:
            out.append(tuple(cur + [left]))
            return
        for v in range(left + 1):
            rec(i + 1, left - v, cur + [v])

    rec(0, r, [])
    return torch.tensor(out, dtype=torch.long)  # [Nr,C]

@lru_cache(None)
def _build_maps(C: int, R: int):
    """
    为 DP 构建从 degree d-1 的 composition 加一个 e_c 到 degree d 的索引映射
    返回 maps[d][c]: LongTensor [N_{d-1}] -> index in degree d
    """
    comps = [None] * (R + 1)
    index = [None] * (R + 1)
    for d in range(R + 1):
        kd = _compositions(C, d)
        comps[d] = kd
        index[d] = {tuple(row.tolist()): i for i, row in enumerate(kd)}

    maps = [[None for _ in range(C)] for __ in range(R + 1)]
    for d in range(1, R + 1):
        prev = comps[d - 1]
        for c in range(C):
            nxt_idx = []
            for row in prev:
                rr = row.clone()
                rr[c] += 1
                nxt_idx.append(index[d][tuple(rr.tolist())])
            maps[d][c] = torch.tensor(nxt_idx, dtype=torch.long)  # [N_{d-1}]
    return comps, maps

def _factorial_table(n: int, device, dtype) -> torch.Tensor:
    """fact[j] = j!, j=0..n  (CPU tensor, moved to device)"""
    facts = [1] * (n + 1)
    for i in range(1, n + 1):
        facts[i] = facts[i - 1] * i
    return torch.tensor(facts, device=device, dtype=dtype)

def _normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    s = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / s

def _dp_scatter_fast(
    P: torch.Tensor,
    idx_flat_list: List[torch.Tensor],
    degree_sizes: List[int],
    R: int,
) -> List[torch.Tensor]:
    """
    Flat-scatter DP with pre-built device index buffers.

    Same math as _dp_multivariate_coeffs_scatter, but idx_flat tensors are
    passed in from module buffers (already on device, no repeated .to(device)).

    P:              [B, m, C], float64
    idx_flat_list:  list length R+1; idx_flat_list[d] is [C * N_{d-1}] int64
    degree_sizes:   list length R+1; degree_sizes[d] = N_d
    R:              max order

    Returns coeffs list length R+1; coeffs[d]: [B, N_d]
    """
    B, m_int, C = P.shape
    device, dtype = P.device, P.dtype

    coeffs: List[torch.Tensor] = [None] * (R + 1)
    coeffs[0] = torch.ones((B, 1), device=device, dtype=dtype)
    for d in range(1, R + 1):
        coeffs[d] = torch.zeros((B, degree_sizes[d]), device=device, dtype=dtype)

    for i in range(m_int):
        max_d = min(R, i + 1)
        new: List[torch.Tensor] = list(coeffs)
        p_i = P[:, i, :]

        for d in range(max_d, 0, -1):
            prev  = coeffs[d - 1]
            src   = (p_i.unsqueeze(2) * prev.unsqueeze(1)).reshape(B, -1)
            idx   = idx_flat_list[d].unsqueeze(0).expand(B, -1)
            delta = torch.zeros(B, degree_sizes[d], device=device, dtype=dtype)
            delta.scatter_add_(1, idx, src)
            new[d] = coeffs[d] + delta   # out-of-place: autograd-safe

        coeffs = new

    return coeffs

class LLPHighOrderLoss(nn.Module):
    """
    Exact high-order LLP loss as nn.Module.

    All static structures (compositions, idx_flat, prod_fact, perm_count) are
    precomputed once in __init__ and registered as persistent=False buffers,
    so they travel with .cuda()/.to(device) without any repeated .to() or
    construction cost inside forward.

    Key optimisations vs the function API:
      - No .to(device) inside forward (buffers already on device).
      - Falling factorial table computed once per forward (shared across orders).
      - Factorial / perm_count lookup uses pre-built table in buffers.
      - Single DP pass covers all orders.
      - CE uses correct per-entry predicted probability: pp = v_pred / Zp.

    Parameters
    ----------
    C            : number of classes
    max_order    : R — highest order to include (1..14)
    bag_size     : fixed integer bag size m
    loss_type    : "ce" | "mse"
    weight_mode  : "uniform" | "inv_order"
    order_weights: optional length-R sequence; None → uniform
    eps, tau     : numerical stability
    reduce       : "mean" | "sum" | None
    """

    def __init__(
        self,
        C: int,
        max_order: int,
        bag_size: int,
        loss_type: str = "ce",
        weight_mode: str = "uniform",
        order_weights: Optional[Sequence[float]] = None,
        eps: float = 1e-200,
        tau: float = 1e-4,
        reduce: str = "mean",
    ):
        super().__init__()

        if not (1 <= max_order <= 14):
            raise ValueError(f"max_order must be in [1, 14], got {max_order}")
        if weight_mode not in ("uniform", "inv_order"):
            raise ValueError("weight_mode must be 'uniform' or 'inv_order'")
        if loss_type not in ("ce", "mse"):
            raise ValueError("loss_type must be 'ce' or 'mse'")

        self.C        = C
        self.R        = max_order
        self.m        = float(int(bag_size))
        self.m_int    = int(bag_size)
        self.loss_type  = loss_type
        self.weight_mode = weight_mode
        self.eps      = eps
        self.tau      = tau
        self.reduce   = reduce

        R = max_order
        if order_weights is None:
            w = [1.0 / R] * R
        else:
            if len(order_weights) < R:
                raise ValueError(f"order_weights length must be >= max_order={R}")
            w = [float(order_weights[i]) for i in range(R)]
            s = sum(w)
            if s <= 0:
                raise ValueError("Sum of order_weights must be > 0")
            w = [wi / s for wi in w]
        self.w: List[float] = w

        # ── Static structures (built on CPU, migrate with .to(device)) ────────
        comps, maps = _build_maps(C, R)
        self.degree_sizes: List[int] = [comps[d].shape[0] for d in range(R + 1)]

        # idx_flat[d]: flat scatter index, shape [C * N_{d-1}]
        for d in range(1, R + 1):
            idx_flat_d = torch.cat([maps[d][c] for c in range(C)], dim=0)
            self.register_buffer(f"idx_flat_{d}", idx_flat_d, persistent=False)

        # k_r, prod_fact_r, perm_count_r for r = 1..R
        fact_cpu = _factorial_table(R, device=torch.device("cpu"), dtype=torch.float64)
        for r in range(1, R + 1):
            k_r = _compositions(C, r)                        # [Nr, C] int64
            self.register_buffer(f"k_{r}", k_r, persistent=False)

            prod_fact_r = fact_cpu[k_r].prod(dim=1)          # [Nr] float64
            self.register_buffer(f"prod_fact_{r}", prod_fact_r, persistent=False)

            perm_count_r = (math.factorial(r) / prod_fact_r) # [Nr] float64
            self.register_buffer(f"perm_count_{r}", perm_count_r, persistent=False)

    def _idx_flat_list(self) -> List[torch.Tensor]:
        return [None] + [getattr(self, f"idx_flat_{d}") for d in range(1, self.R + 1)]

    def forward(
        self,
        labels_proportion: torch.Tensor,   # [B, C]
        y: torch.Tensor,                   # [B * m, C]
    ) -> torch.Tensor:
        B, C = labels_proportion.shape
        device   = y.device
        eps      = self.eps
        dtype_w  = torch.float64

        alpha_true = labels_proportion.to(device=device, dtype=dtype_w).clamp_min(0.0)
        alpha_true = _normalize_rows(alpha_true, eps)       # [B, C]

        # Prepare instance probability matrix P
        P = y.to(dtype=dtype_w).clamp_min(eps)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
        P = P.reshape(B, self.m_int, C)                     # [B, m, C]

        R, m = self.R, self.m

        # ── One DP pass covers all orders ─────────────────────────────────
        coeffs = _dp_scatter_fast(P, self._idx_flat_list(), self.degree_sizes, R)

        # ── Falling factorial table: computed once, shared across orders ──
        # fall[b, c, j] = ∏_{t=0}^{j-1} (alpha_true[b,c] - t/m),  j=0..R
        fall = torch.ones((B, C, R + 1), device=device, dtype=dtype_w)
        for j in range(1, R + 1):
            fall[..., j] = fall[..., j - 1] * (alpha_true - (j - 1) / m)

        loss_per_bag = torch.zeros(B, device=device, dtype=dtype_w)

        for r in range(1, R + 1):
            if self.w[r - 1] == 0.0:
                continue

            k          = getattr(self, f"k_{r}")           # [Nr, C] — on device
            prod_fact  = getattr(self, f"prod_fact_{r}")   # [Nr]
            perm_count = getattr(self, f"perm_count_{r}")  # [Nr]
            Nr         = k.shape[0]
            coeff_r    = coeffs[r]                         # [B, Nr], carries grad

            # v_pred: out-of-place chain from P → grad safe
            v_pred = (coeff_r * prod_fact.unsqueeze(0) / (m ** r)).clamp_min(eps)

            # v_true: from pre-built fall table (no grad needed)
            v_true = torch.ones((B, Nr), device=device, dtype=dtype_w)
            for c in range(C):
                kc = k[:, c]   # [Nr]
                v_true = v_true * fall[:, c, :].gather(
                    1, kc.unsqueeze(0).expand(B, -1)
                )
            v_true = v_true.clamp_min(0.0)

            mt = v_true * perm_count.unsqueeze(0)   # [B, Nr]
            mp = v_pred * perm_count.unsqueeze(0)   # [B, Nr]

            if self.loss_type == "ce":
                num_entries = float(C ** r)
                Zt = mt.sum(dim=1, keepdim=True).clamp_min(eps)
                Zp = mp.sum(dim=1, keepdim=True).clamp_min(eps)
                pt = mt / Zt
                # pp = per-entry predicted probability (correct: v_pred/Zp, not mp/Zp)
                pp = v_pred / Zp
                if self.tau > 0:
                    pp = (1.0 - self.tau) * pp + self.tau / num_entries
                lr = -(pt * torch.log(pp.clamp_min(eps))).sum(dim=1)
            else:   # mse
                diff2 = (v_true - v_pred) ** 2
                lr = (diff2 * perm_count.unsqueeze(0)).sum(dim=1) / float(C ** r)

            if self.weight_mode == "inv_order":
                lr = lr / float(r)

            loss_per_bag = loss_per_bag + self.w[r - 1] * lr

        if self.reduce is None:
            return loss_per_bag.to(dtype=labels_proportion.dtype)
        if self.reduce == "mean":
            return loss_per_bag.mean().to(dtype=labels_proportion.dtype)
        if self.reduce == "sum":
            return loss_per_bag.sum().to(dtype=labels_proportion.dtype)
        raise ValueError("reduce must be None|'mean'|'sum'")
