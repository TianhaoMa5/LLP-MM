
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


@lru_cache(None)
def _build_pred_maps(C: int, R: int):
    """
    为 gather DP 构建 predecessor 索引。

    对 degree d 的每个 composition k，枚举所有 k_c > 0 的前驱：
        predecessor = k - e_c，对应 degree d-1 中的索引。

    Returns (CPU tensors, @lru_cache 缓存):
      comps:      list[R+1], comps[d]: [N_d, C] int64
      pred_idx:   list[R+1], pred_idx[d]:  [N_d, L_d] int64  — 前驱在 d-1 的位置
      pred_cls:   list[R+1], pred_cls[d]:  [N_d, L_d] int64  — 对应的 class c
      pred_mask:  list[R+1], pred_mask[d]: [N_d, L_d] bool   — 填充列为 False
    其中 L_d = min(C, d)（最多 d 个非零分量）。
    """
    comps = [None] * (R + 1)
    index = [None] * (R + 1)
    for d in range(R + 1):
        kd = _compositions(C, d)
        comps[d] = kd
        index[d] = {tuple(row.tolist()): i for i, row in enumerate(kd)}

    pred_idx  = [None] * (R + 1)
    pred_cls  = [None] * (R + 1)
    pred_mask = [None] * (R + 1)

    for d in range(1, R + 1):
        kd = comps[d]
        Nd = kd.shape[0]
        Ld = min(C, d)

        idx  = torch.zeros((Nd, Ld), dtype=torch.long)
        cls  = torch.zeros((Nd, Ld), dtype=torch.long)
        mask = torch.zeros((Nd, Ld), dtype=torch.bool)

        for n, row in enumerate(kd):
            t = 0
            for c in range(C):
                if int(row[c].item()) > 0:
                    prev = row.clone()
                    prev[c] -= 1
                    idx[n, t]  = index[d - 1][tuple(prev.tolist())]
                    cls[n, t]  = c
                    mask[n, t] = True
                    t += 1

        pred_idx[d]  = idx
        pred_cls[d]  = cls
        pred_mask[d] = mask

    return comps, pred_idx, pred_cls, pred_mask


def _factorial_table(n: int, device, dtype) -> torch.Tensor:
    """fact[j] = j!, j=0..n  (CPU tensor, moved to device)"""
    facts = [1] * (n + 1)
    for i in range(1, n + 1):
        facts[i] = facts[i - 1] * i
    return torch.tensor(facts, device=device, dtype=dtype)


def _prod_factorial_from_k(k: torch.Tensor, r: int, dtype) -> torch.Tensor:
    """
    k: [Nr, C] int64
    returns ∏_c k_c! for each row: [Nr]
    避免 k.tolist() → GPU->CPU 同步；用 gather 代替。
    """
    fact = _factorial_table(r, device=k.device, dtype=dtype)  # [r+1]
    return fact[k].prod(dim=1)                                 # [Nr]


def _dp_multivariate_coeffs_scatter(P: torch.Tensor, R: int) -> List[torch.Tensor]:
    """Scatter-add DP (保留作对比 baseline，不对外暴露)。"""
    B, m_int, C = P.shape
    comps, maps = _build_maps(C, R)
    device, dtype = P.device, P.dtype

    idx_flat: List[Optional[torch.Tensor]] = [None] * (R + 1)
    for d in range(1, R + 1):
        idx_flat[d] = torch.cat(
            [maps[d][c].to(device) for c in range(C)], dim=0
        )

    coeffs: List[torch.Tensor] = [None] * (R + 1)
    coeffs[0] = torch.ones((B, 1), device=device, dtype=dtype)
    for d in range(1, R + 1):
        coeffs[d] = torch.zeros((B, comps[d].shape[0]), device=device, dtype=dtype)

    for i in range(m_int):
        max_d = min(R, i + 1)
        new: List[torch.Tensor] = list(coeffs)
        p_i = P[:, i, :]

        for d in range(max_d, 0, -1):
            prev   = coeffs[d - 1]
            N_prev = prev.shape[1]
            src    = (p_i.unsqueeze(2) * prev.unsqueeze(1)).reshape(B, -1)
            delta  = torch.zeros(B, comps[d].shape[0], device=device, dtype=dtype)
            delta.scatter_add_(1, idx_flat[d].unsqueeze(0).expand(B, -1), src)
            new[d] = coeffs[d] + delta

        coeffs = new

    return coeffs


def _dp_multivariate_coeffs(P: torch.Tensor, R: int) -> List[torch.Tensor]:
    """
    Exact gather DP for ∏_{i=1}^m (1 + Σ_c p_{i,c} z_c), truncated to degree ≤ R.

    P: [B, m, C]
    Returns coeffs[d]: [B, N_d].

    Gather 递推（等价于 scatter 版但避免 scatter_add_ 的写冲突开销）：

        new_d[k] = old_d[k] + Σ_{c: k_c>0} old_{d-1}[k-e_c] * p_i[c]

    优化：
      - 所有静态 index 搬到 device 一次（每个 forward call）；
      - 用 predecessor gather 代替 scatter，合并所有前驱为一次 gather + sum；
      - 浅拷贝 + max_d 剪枝（同 scatter 版）。

    Autograd 安全：
      - contrib 由 gather + multiply + sum 组成，全为 out-of-place；
      - new[d] = coeffs[d] + contrib，out-of-place；
      - pred_idx/pred_cls/pred_mask 均无梯度。
    """
    B, m_int, C = P.shape
    device, dtype = P.device, P.dtype

    comps, pred_idx_cpu, pred_cls_cpu, pred_mask_cpu = _build_pred_maps(C, R)

    # 每个 forward 只搬一次：后续可改成 module buffer 进一步消除此开销
    pred_idx:  List[Optional[torch.Tensor]] = [None] * (R + 1)
    pred_cls:  List[Optional[torch.Tensor]] = [None] * (R + 1)
    pred_mask: List[Optional[torch.Tensor]] = [None] * (R + 1)
    for d in range(1, R + 1):
        pred_idx[d]  = pred_idx_cpu[d].to(device)
        pred_cls[d]  = pred_cls_cpu[d].to(device)
        pred_mask[d] = pred_mask_cpu[d].to(device=device, dtype=dtype)

    coeffs: List[torch.Tensor] = [None] * (R + 1)
    coeffs[0] = torch.ones((B, 1), device=device, dtype=dtype)
    for d in range(1, R + 1):
        coeffs[d] = torch.zeros((B, comps[d].shape[0]), device=device, dtype=dtype)

    for i in range(m_int):
        max_d = min(R, i + 1)
        new: List[torch.Tensor] = list(coeffs)
        p_i = P[:, i, :]   # [B, C]

        for d in range(max_d, 0, -1):
            prev = coeffs[d - 1]                    # [B, N_{d-1}]

            # prev_vals[b, n, l] = prev[b, pred_idx[d][n, l]]
            prev_vals = prev[:, pred_idx[d]]         # [B, N_d, L_d]

            # p_vals[b, n, l]   = p_i[b, pred_cls[d][n, l]]
            p_vals = p_i[:, pred_cls[d]]             # [B, N_d, L_d]

            # 无效填充列（mask=0）贡献为 0，sum 后自然消掉
            contrib = (prev_vals * p_vals * pred_mask[d].unsqueeze(0)).sum(dim=-1)

            new[d] = coeffs[d] + contrib             # out-of-place

        coeffs = new

    return coeffs


def _true_tensor_values_from_alpha(alpha: torch.Tensor, m: float, k: torch.Tensor) -> torch.Tensor:
    """
    精确的 T_true(entry)（按你现在的定义：ordered distinct / m^r）
    entry 的值只取决于 multiplicity k（sum k=r）。
    alpha: [B,C]
    k: [Nr,C] int
    返回 v_true: [B,Nr]
    v_true(k) = ∏_c ∏_{t=0}^{k_c-1} (alpha_c - t/m)
    """
    B, C = alpha.shape
    R = int(k.sum(dim=1).max().item())
    device, dtype = alpha.device, alpha.dtype
    invm = 1.0 / m

    # fall[b,c,j] = ∏_{t=0}^{j-1} (alpha[b,c] - t/m), j=0..R
    fall = torch.ones((B, C, R + 1), device=device, dtype=dtype)
    for j in range(1, R + 1):
        fall[..., j] = fall[..., j - 1] * (alpha - (j - 1) * invm)

    v = torch.ones((B, k.shape[0]), device=device, dtype=dtype)
    for c in range(C):
        kc = k[:, c]  # [Nr]
        v *= fall[:, c, :].gather(1, kc.unsqueeze(0).expand(B, -1))
    return v.clamp_min(0.0)


def _pred_tensor_values_from_coeff(
    coeff_r: torch.Tensor,
    k: torch.Tensor,
    m: float,
    r: Optional[int] = None,
) -> torch.Tensor:
    """
    coeff_r: [B, Nr] 是 monomial z^k 的系数（来自 ∏(1+Σ p z)）
    对于 ordered distinct / m^r：
      v_pred(k) = (∏ k_c!) * coeff_r(k) / m^r

    优化：用 _prod_factorial_from_k 代替 k.tolist()，避免 GPU->CPU 同步。
    r 可由 caller 直接传入，避免一次 .item() 调用；不传时自动推断（向后兼容）。
    """
    device, dtype = coeff_r.device, coeff_r.dtype
    if r is None:
        r = int(k[0].sum().item())
    prod_fact = _prod_factorial_from_k(k, r=r, dtype=dtype)   # [Nr]
    return coeff_r * prod_fact.unsqueeze(0) / (m ** r)


def _grouped_loss_order_r(alpha: torch.Tensor,
                          P: torch.Tensor,
                          m: float,
                          r: int,
                          loss_type: str,
                          eps: float,
                          tau: float = 1e-4) -> torch.Tensor:
    """
    精确计算 r阶张量的 CE/MSE，但不显式展开 C^r。
    返回 [B]
    """
    B, _, C = P.shape
    k = _compositions(C, r).to(P.device)           # [Nr,C]
    Nr = k.shape[0]

    # 预测：先 DP 求所有阶的系数，再取 r 阶
    coeffs = _dp_multivariate_coeffs(P, R=r)
    coeff_r = coeffs[r]                            # [B, Nr]
    v_pred = _pred_tensor_values_from_coeff(coeff_r, k, m, r=r).clamp_min(eps)  # [B,Nr]

    # 真值：
    v_true = _true_tensor_values_from_alpha(alpha, m, k)                    # [B,Nr]

    # perm_count = r! / ∏ k_c!
    prod_fact_int = _prod_factorial_from_k(k, r=r, dtype=torch.float64)    # [Nr]
    perm_count = (math.factorial(r) / prod_fact_int).to(P.dtype)            # [Nr]

    mt = v_true * perm_count.unsqueeze(0)   # [B,Nr]
    mp = v_pred * perm_count.unsqueeze(0)   # [B,Nr]

    if loss_type == "ce":
        # 和你 _ce_from_mass 一致：先归一化再做 CE（这是对 C^r entries 的精确 CE）
        num_entries = float(C ** r)
        Zt = mt.sum(dim=1, keepdim=True).clamp_min(eps)
        Zp = mp.sum(dim=1, keepdim=True).clamp_min(eps)
        pt = mt / Zt
        pp = mp / Zp
        if tau > 0:
            pp = (1.0 - tau) * pp + tau / num_entries
        return -(pt * torch.log(pp.clamp_min(eps))).sum(dim=1)

    elif loss_type == "mse":
        # 对所有 C^r entries 的均值：sum(count * diff^2) / C^r
        diff2 = (v_true - v_pred) ** 2
        return (diff2 * perm_count.unsqueeze(0)).sum(dim=1) / float(C ** r)

    else:
        raise ValueError("loss_type must be 'ce' or 'mse'")

def _parse_max_order(max_order: Union[int, float, torch.Tensor]) -> int:
    if isinstance(max_order, torch.Tensor):
        if max_order.numel() != 1:
            raise ValueError(f"max_order tensor must be scalar, got {tuple(max_order.shape)}")
        max_order = int(max_order.item())
    else:
        max_order = int(max_order)
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if max_order > 14:
        raise ValueError("This implementation supports max_order up to 14.")
    return max_order


def _ensure_scalar_bag_size(bag_sizes, device, dtype) -> float:
    """
    Require a fixed bag size m (scalar). This matches your 'N / bag_size = #bags' setting.
    """
    if bag_sizes is None:
        raise ValueError("bag_sizes must be provided as a scalar (e.g., 16/32/64/128) when y is [N,C].")

    if isinstance(bag_sizes, (int, float)):
        m = float(bag_sizes)
    elif isinstance(bag_sizes, torch.Tensor):
        if bag_sizes.numel() == 1:
            m = float(bag_sizes.item())
        else:
            # If you later need variable bag sizes, we can add another function.
            raise ValueError("This function expects scalar bag_sizes. Got a tensor with more than 1 element.")
    else:
        # list/tuple not supported here because your packing rule is fixed-size
        raise ValueError("This function expects scalar bag_sizes (int/float/0-d tensor).")

    if m <= 0:
        raise ValueError(f"bag_sizes must be positive, got {m}")
    return m

def _ce_from_mass(mass_true: torch.Tensor,
                  mass_pred: torch.Tensor,
                  eps: float,
                  tau: float = 1e-4) -> torch.Tensor:
    """
    mass_true, mass_pred: [B, ...] nonnegative "masses"
    returns per-bag CE: [B]
    CE between normalized distributions derived from masses.
    tau: label smoothing on pred distribution.
    """
    B = mass_true.shape[0]
    num_entries = float(mass_true[0].numel())

    mt = mass_true.clamp_min(0.0)
    mp = mass_pred.clamp_min(eps)

    Zt = mt.reshape(B, -1).sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1]
    Zp = mp.reshape(B, -1).sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1]

    pt = (mt.reshape(B, -1) / Zt)                                   # [B,D]
    pp = (mp.reshape(B, -1) / Zp)                                   # [B,D]

    # smoothing to avoid log(0) dominance
    if tau > 0:
        pp = (1.0 - tau) * pp + tau / num_entries

    return -(pt * torch.log(pp.clamp_min(eps))).sum(dim=1)           # [B]
def _order2_true(alpha: torch.Tensor, m: float) -> torch.Tensor:
    # [B,C,C]
    B, C = alpha.shape
    I = torch.eye(C, device=alpha.device, dtype=alpha.dtype)[None, :, :]  # [1,C,C]
    outer = alpha[:, :, None] * alpha[:, None, :]                         # [B,C,C]
    return outer - (alpha / m)[:, :, None] * I

def _normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    s = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / s


def _order2_true(alpha: torch.Tensor, m: float) -> torch.Tensor:
    # [B,C,C]
    M2 = alpha.unsqueeze(2) * alpha.unsqueeze(1)
    diag = alpha * (alpha - 1.0 / m)
    M2 = M2.clone()
    M2.diagonal(dim1=1, dim2=2).copy_(diag)  # labels side (no grad), safe
    return M2

def _order3_ce_chunked(alpha: torch.Tensor,
                       P: torch.Tensor,
                       m: float,
                       eps: float,
                       chunk_k: int = 8,
                       tau: float = 1e-4) -> torch.Tensor:
    """
    Per-bag CE between normalized 3rd-order masses (true vs pred), chunked over c-dim.
    alpha: [B,C] normalized true proportions
    P:     [B,m,C] instance probs
    return: [B]
    """
    B, C = alpha.shape
    device, dtype = alpha.device, alpha.dtype

    S1 = P.sum(dim=1)                               # [B,C]
    A  = torch.einsum("bmc,bmd->bcd", P, P)         # [B,C,C]

    I = torch.eye(C, device=device, dtype=dtype)    # [C,C]
    I_ab = I[None, :, :, None]                      # [1,C,C,1]

    invm  = 1.0 / m
    invm2 = 1.0 / (m * m)
    invm3 = 1.0 / (m * m * m)

    # ----- pass 1: compute Zt, Zp -----
    Zt = torch.zeros((B,), device=device, dtype=dtype)
    Zp = torch.zeros((B,), device=device, dtype=dtype)

    for start in range(0, C, chunk_k):
        end = min(C, start + chunk_k)
        idx = torch.arange(start, end, device=device, dtype=torch.long)  # [K]
        P_k = P[:, :, start:end]                                          # [B,m,K]
        S1_k = S1.gather(1, idx.view(1, -1).expand(B, -1))                 # [B,K]

        # pred mass chunk: [B,C,C,K]
        outer3 = S1[:, :, None, None] * S1[:, None, :, None] * S1_k[:, None, None, :]
        term12 = A[:, :, :, None] * S1_k[:, None, None, :]

        A_ak = torch.einsum("bma,bmk->bak", P, P_k)                        # [B,C,K]
        term13 = A_ak[:, :, None, :] * S1[:, None, :, None]

        A_bk = torch.einsum("bmd,bmk->bdk", P, P_k)                        # [B,C,K]
        term23 = S1[:, :, None, None] * A_bk[:, None, :, :]

        S3 = torch.einsum("bmk,bma,bmd->badk", P_k, P, P)                  # [B,C,C,K]
        T_pred = (outer3 - term12 - term13 - term23 + 2.0 * S3) * invm3
        T_pred = T_pred.clamp_min(eps)

        # true mass chunk: [B,C,C,K]
        alpha_k = alpha.gather(1, idx.view(1, -1).expand(B, -1))            # [B,K]
        base = alpha[:, :, None, None] * alpha[:, None, :, None] * alpha_k[:, None, None, :]

        I_c  = I[:, start:end]                                              # [C,K]
        I_ac = I_c[None, :, None, :]                                        # [1,C,1,K]
        I_bc = I_c[None, None, :, :]                                        # [1,1,C,K]

        corr12  = (-invm)  * (alpha[:, :, None] * alpha_k[:, None, :])[:, :, None, :] * I_ab
        corr13  = (-invm)  * (alpha[:, :, None, None] * alpha[:, None, :, None]) * I_ac
        corr23  = (-invm)  * (alpha[:, :, None, None] * alpha[:, None, :, None]) * I_bc
        corr123 = (2.0 * invm2) * alpha[:, :, None, None] * I_ab * I_ac

        T_true = (base + corr12 + corr13 + corr23 + corr123).clamp_min(0.0)

        Zp = Zp + T_pred.sum(dim=(1, 2, 3))
        Zt = Zt + T_true.sum(dim=(1, 2, 3))

    Zp = Zp.clamp_min(eps)
    Zt = Zt.clamp_min(eps)

    # ----- pass 2: compute CE -----
    num_entries = float(C * C * C)
    ce = torch.zeros((B,), device=device, dtype=dtype)

    for start in range(0, C, chunk_k):
        end = min(C, start + chunk_k)
        idx = torch.arange(start, end, device=device, dtype=torch.long)

        P_k = P[:, :, start:end]
        S1_k = S1.gather(1, idx.view(1, -1).expand(B, -1))

        outer3 = S1[:, :, None, None] * S1[:, None, :, None] * S1_k[:, None, None, :]
        term12 = A[:, :, :, None] * S1_k[:, None, None, :]

        A_ak = torch.einsum("bma,bmk->bak", P, P_k)
        term13 = A_ak[:, :, None, :] * S1[:, None, :, None]

        A_bk = torch.einsum("bmd,bmk->bdk", P, P_k)
        term23 = S1[:, :, None, None] * A_bk[:, None, :, :]

        S3 = torch.einsum("bmk,bma,bmd->badk", P_k, P, P)
        T_pred = (outer3 - term12 - term13 - term23 + 2.0 * S3) * invm3
        T_pred = T_pred.clamp_min(eps)

        alpha_k = alpha.gather(1, idx.view(1, -1).expand(B, -1))
        base = alpha[:, :, None, None] * alpha[:, None, :, None] * alpha_k[:, None, None, :]

        I_c  = I[:, start:end]
        I_ac = I_c[None, :, None, :]
        I_bc = I_c[None, None, :, :]

        corr12  = (-invm)  * (alpha[:, :, None] * alpha_k[:, None, :])[:, :, None, :] * I_ab
        corr13  = (-invm)  * (alpha[:, :, None, None] * alpha[:, None, :, None]) * I_ac
        corr23  = (-invm)  * (alpha[:, :, None, None] * alpha[:, None, :, None]) * I_bc
        corr123 = (2.0 * invm2) * alpha[:, :, None, None] * I_ab * I_ac

        T_true = (base + corr12 + corr13 + corr23 + corr123).clamp_min(0.0)

        pt = T_true / Zt[:, None, None, None]
        pp = T_pred / Zp[:, None, None, None]
        if tau > 0:
            pp = (1.0 - tau) * pp + tau / num_entries

        ce = ce - (pt * torch.log(pp.clamp_min(eps))).sum(dim=(1, 2, 3))

    return ce  # [B]

def _order2_pred(P: torch.Tensor, m: float) -> torch.Tensor:
    # P: [B,m,C]
    S1 = P.sum(dim=1)  # [B,C]
    S2 = torch.einsum("bmc,bmd->bcd", P, P)  # [B,C,C] sum_i p_i ⊗ p_i
    return (S1.unsqueeze(2) * S1.unsqueeze(1) - S2) / (m * m)


def _order3_mse_chunked(alpha: torch.Tensor, P: torch.Tensor, m: float, chunk_k: int = 8) -> torch.Tensor:
    """
    Per-bag MSE between true and pred 3rd-order factorial moment tensors (ordered indices),
    without materializing [B,C,C,C].

    alpha: [B,C] true proportions (should be normalized, alpha=z/m ideally)
    P:     [B,m,C] instance softmax probs
    m:     scalar bag size
    return: [B] mse
    """
    B, C = alpha.shape
    device, dtype = alpha.device, alpha.dtype

    # pred shared stats
    S1 = P.sum(dim=1)                                # [B,C]
    A  = torch.einsum("bmc,bmd->bcd", P, P)          # [B,C,C]  sum_i p_i ⊗ p_i

    I = torch.eye(C, device=device, dtype=dtype)     # [C,C]

    se_sum = torch.zeros((B,), device=device, dtype=dtype)
    total_entries = float(C * C * C)

    invm  = 1.0 / m
    invm2 = 1.0 / (m * m)
    invm3 = 1.0 / (m * m * m)

    for start in range(0, C, chunk_k):
        end = min(C, start + chunk_k)
        K = end - start
        idx = torch.arange(start, end, device=device, dtype=torch.long)   # [K]

        # ---------- pred slice T_pred: [B,C,C,K] ----------
        P_k = P[:, :, start:end]                                         # [B,m,K]
        S1_k = S1.gather(1, idx.view(1, -1).expand(B, -1))                # [B,K]

        # S1⊗S1⊗S1_k
        outer3 = S1[:, :, None, None] * S1[:, None, :, None] * S1_k[:, None, None, :]   # [B,C,C,K]

        # A_ab * S1_k
        term12 = A[:, :, :, None] * S1_k[:, None, None, :]                                 # [B,C,C,K]

        # A_ak * S1_b  where A_ak = sum_i p_i[a] p_i[k]
        A_ak = torch.einsum("bma,bmk->bak", P, P_k)                                         # [B,C,K]
        term13 = A_ak[:, :, None, :] * S1[:, None, :, None]                                 # [B,C,C,K]

        # A_bk * S1_a  where A_bk = sum_i p_i[b] p_i[k]
        A_bk = torch.einsum("bmd,bmk->bdk", P, P_k)                                         # [B,C,K]  ✅修正
        term23 = S1[:, :, None, None] * A_bk[:, None, :, :]                                 # [B,C,C,K]

        # S3_abk = sum_i p_i[k] p_i[a] p_i[b]
        S3 = torch.einsum("bmk,bma,bmd->badk", P_k, P, P)                                   # [B,C,C,K]

        T_pred = (outer3 - term12 - term13 - term23 + 2.0 * S3) * invm3                     # [B,C,C,K]

        # ---------- true slice T_true: [B,C,C,K] ----------
        alpha_k = alpha.gather(1, idx.view(1, -1).expand(B, -1))                             # [B,K]

        # base: alpha_a alpha_b alpha_k
        base = alpha[:, :, None, None] * alpha[:, None, :, None] * alpha_k[:, None, None, :]  # [B,C,C,K]

        # masks for equalities with c in [start,end)
        I_ab = I[None, :, :, None]                                  # [1,C,C,1]  mask a==b
        I_c  = I[:, start:end]                                      # [C,K]      mask a==c or b==c depends on broadcast
        I_ac = I_c[None, :, None, :]                                # [1,C,1,K]  mask a==c
        I_bc = I_c[None, None, :, :]                                # [1,1,C,K]  mask b==c

        # When alpha = z/m (exact counts), the ordered 3rd factorial moments satisfy:
        # T_true = alpha_a alpha_b alpha_c
        #         - (1/m)[ 1_{a=b} alpha_a alpha_c + 1_{a=c} alpha_a alpha_b + 1_{b=c} alpha_b alpha_a ]
        #         + (2/m^2) 1_{a=b=c} alpha_a
        # (this is the clean correction form; avoids messy diag2/diag3 bookkeeping)
        corr12 = (-invm)  * (alpha[:, :, None] * alpha_k[:, None, :])[:, :, None, :] * I_ab     # a==b
        corr13 = (-invm)  * (alpha[:, :, None, None] * alpha[:, None, :, None]) * I_ac           # a==c
        corr23 = (-invm)  * (alpha[:, :, None, None] * alpha[:, None, :, None]) * I_bc           # b==c
        corr123 = (2.0 * invm2) * alpha[:, :, None, None] * I_ab * I_ac                           # a==b==c

        T_true = base + corr12 + corr13 + corr23 + corr123                                       # [B,C,C,K]

        diff = T_true - T_pred
        se_sum = se_sum + (diff * diff).sum(dim=(1, 2, 3))

    return se_sum / total_entries   # [B]



def llp_high_order_loss_batch_taylor(
    labels_proportion: torch.Tensor,   # [B, C] 每个 bag 的真实比例
    y: torch.Tensor,                   # 支持 [B,C] 或者 [N,C] 的 instance softmax（按bag从上到下堆）
    max_order: Union[int, float, torch.Tensor] = 1,
    loss_type: str = "ce",             # "ce" or "mse"
    order_weights: Optional[Sequence[float]] = None,  # 例如 [1,0,0] 或 [0,1,1]
    weight_mode: str = "uniform",      # NEW: "uniform" or "inv_order" (每阶 / r)
    bag_sizes: Union[torch.Tensor, Sequence[int], int, float, None] = None,
    reduce: str = "mean",
    eps: float = 1e-200,
) -> torch.Tensor:
    """
    接口保持不变：返回选定(1..max_order)阶的加权平均 loss。
    额外支持 weight_mode:
      - "uniform": 原样（order_weights=None 时均匀权重）
      - "inv_order": 在加权前把第 r 阶 loss 除以 r（l_r <- l_r / r）
    注意：本函数依赖外部你已有的这些函数：
      _parse_max_order, _ensure_scalar_bag_size, _normalize_rows,
      _order2_true, _order2_pred, _ce_from_mass, _order3_ce_chunked, _order3_mse_chunked,
      _dp_multivariate_coeffs, _compositions, _pred_tensor_values_from_coeff, _true_tensor_values_from_alpha
    """

    if weight_mode not in ("uniform", "inv_order"):
        raise ValueError("weight_mode must be 'uniform' or 'inv_order'")

    max_order = _parse_max_order(max_order)

    if labels_proportion.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"labels_proportion and y must be 2D, got {tuple(labels_proportion.shape)}, {tuple(y.shape)}"
        )

    B, C = labels_proportion.shape
    device = y.device
    dtype_work = torch.float64

    alpha_true = labels_proportion.to(device=device, dtype=dtype_work).clamp_min(0.0)
    alpha_true = _normalize_rows(alpha_true, eps)  # [B,C]

    # ----- weights: allow selecting 1..max_order by setting some to 0
    if order_weights is None:
        w = [1.0 / max_order] * max_order
    else:
        if len(order_weights) < max_order:
            raise ValueError(f"order_weights length must be >= max_order={max_order}")
        w = [float(order_weights[i]) for i in range(max_order)]
        s = sum(w)
        if s <= 0:
            raise ValueError("Sum of selected order_weights must be > 0.")
        # normalize to keep overall scale stable
        w = [wi / s for wi in w]

    # ----- Case 1: y is bag-level [B,C] (只能稳定算1阶)
    if y.shape == (B, C):
        alpha_hat = y.to(dtype=dtype_work).clamp_min(eps)
        alpha_hat = _normalize_rows(alpha_hat, eps)

        if loss_type == "ce":
            l1 = -(alpha_true * torch.log(alpha_hat)).sum(dim=1)  # [B]
        elif loss_type == "mse":
            l1 = ((alpha_true - alpha_hat) ** 2).mean(dim=1)      # [B]
        else:
            raise ValueError("loss_type must be 'ce' or 'mse'")

        if weight_mode == "inv_order":
            l1 = l1 / 1.0

        loss_per_bag = w[0] * l1

        if max_order > 1:
            raise ValueError(
                "For order>=2 you must pass instance softmax y as [N,C] and provide scalar bag_sizes=m."
            )

    else:
        # ----- Case 2: y is instance-level [N,C], pack by fixed bag_size
        m = _ensure_scalar_bag_size(bag_sizes, device=device, dtype=dtype_work)
        m_int = int(round(m))
        if abs(m - m_int) > 1e-6:
            raise ValueError(f"bag_sizes must be an integer bag size, got {m}")
        m = float(m_int)

        N, C2 = y.shape
        if C2 != C:
            raise ValueError(f"y has C={C2}, but labels_proportion has C={C}")
        if N != B * m_int:
            raise ValueError(f"Expect N == B*m. Got N={N}, B={B}, m={m_int} => B*m={B*m_int}")

        P = y.to(dtype=dtype_work)
        # normalize each instance row just in case
        P = P.clamp_min(eps)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
        P = P.reshape(B, m_int, C)  # [B,m,C]

        # ── 统一路径：一次 DP 覆盖所有阶，然后逐阶计算 grouped exact loss ──────
        # 对 r=1: v_pred(e_c) = coeff_1[c]/m = alpha_hat[c]，与原 order-1 等价
        # 对 r=2: grouped exact 等价于原 _order2_true/_order2_pred + CE/MSE
        # 对 r=3: grouped exact 等价于原 chunked order-3
        # 对 r≥4: 同原 order4+ 路径
        # 数学完全不变；perm_count 恢复 C^r 有序 entry 的精确 mass/count。
        _tau = 1e-4
        coeffs = _dp_multivariate_coeffs(P, R=max_order)
        loss_per_bag = torch.zeros(B, device=device, dtype=dtype_work)

        for r in range(1, max_order + 1):
            if w[r - 1] == 0.0:
                continue

            k = _compositions(C, r).to(device)   # [Nr, C]，CPU cached → device
            coeff_r = coeffs[r]                   # [B, Nr]，携带 y 的梯度

            # v_pred: out-of-place，保留 autograd 链
            v_pred = _pred_tensor_values_from_coeff(coeff_r, k, m, r=r).clamp_min(eps)
            v_true = _true_tensor_values_from_alpha(alpha_true, m, k)  # 静态，无梯度

            # perm_count = r! / ∏ k_c!，静态权重，无梯度
            prod_fact  = _prod_factorial_from_k(k, r=r, dtype=torch.float64)
            perm_count = (math.factorial(r) / prod_fact).to(dtype_work)  # [Nr]

            mt = v_true * perm_count.unsqueeze(0)   # [B, Nr]
            mp = v_pred * perm_count.unsqueeze(0)   # [B, Nr]

            if loss_type == "ce":
                num_entries = float(C ** r)
                Zt = mt.sum(dim=1, keepdim=True).clamp_min(eps)
                Zp = mp.sum(dim=1, keepdim=True).clamp_min(eps)
                # pt: group-level true probability (= per-entry true prob × perm_count)
                pt = mt / Zt
                # pp: per-entry predicted probability — do NOT use mp/Zp here.
                # mp/Zp would add log(perm_count) to each CE term (wrong).
                pp = v_pred / Zp
                if _tau > 0:
                    pp = (1.0 - _tau) * pp + _tau / num_entries
                lr = -(pt * torch.log(pp.clamp_min(eps))).sum(dim=1)   # [B]
            elif loss_type == "mse":
                diff2 = (v_true - v_pred) ** 2
                lr = (diff2 * perm_count.unsqueeze(0)).sum(dim=1) / float(C ** r)
            else:
                raise ValueError("loss_type must be 'ce' or 'mse'")

            if weight_mode == "inv_order":
                lr = lr / float(r)

            loss_per_bag = loss_per_bag + w[r - 1] * lr

    # ----- reduce (same as your old)
    if reduce is None:
        return loss_per_bag.to(dtype=labels_proportion.dtype)
    if reduce == "mean":
        return loss_per_bag.mean().to(dtype=labels_proportion.dtype)
    if reduce == "sum":
        return loss_per_bag.sum().to(dtype=labels_proportion.dtype)
    raise ValueError("reduce must be None|'mean'|'sum'")


# ═══════════════════════════════════════════════════════════════════════════════
# Module-based exact high-order loss
# ═══════════════════════════════════════════════════════════════════════════════

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
