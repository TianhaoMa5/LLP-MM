import torch
from typing import Optional

# ============================================================
# 0) proportions -> integer counts (do NOT renormalize)
# ============================================================
def proportions_to_counts_exact(
    proportions: torch.Tensor,  # [C], assumed correct
    s: int,
) -> torch.Tensor:
    """
    Convert proportions to integer counts k with minimal disturbance.
    Does NOT renormalize proportions.
    Fixes tiny float rounding so sum(k) == s.
    """
    # proportions: keep on same device
    k = torch.round(proportions * s).to(torch.long)  # [C]
    diff = int(s - int(k.sum().item()))
    if diff != 0:
        # put mismatch to the largest-proportion class (minimal disturbance)
        j = int(torch.argmax(proportions).item())
        k[j] += diff
    k = torch.clamp(k, 0, s)
    return k
def proportions_to_counts_exact_vmap(
    proportions: torch.Tensor,  # [C]
    s: int,
) -> torch.Tensor:
    # round to nearest integer counts
    k = torch.round(proportions * float(s)).to(torch.long)  # [C]

    # tensor scalar diff (no .item())
    diff = (s - k.sum()).to(torch.long)  # 0-dim tensor

    # index of largest proportion (tensor scalar)
    j = torch.argmax(proportions)        # 0-dim long tensor

    # add diff to k[j] WITHOUT python indexing / item()
    # (scatter_add expects index to have same dim as src)
    add = torch.zeros_like(k)
    add.scatter_(0, j.view(1), diff.view(1))
    k = k + add

    # clamp to [0, s]
    k = torch.clamp(k, 0, s)
    return k


# ============================================================
# 1) FFT convolution (IMPORTANT: detach the stabilization scale)
# ============================================================
def batch_fft_convolve_one_pass(
    A: torch.Tensor,  # [C, n_pair, L]
    B: torch.Tensor,  # [C, n_pair, L]
    eps: float = 1e-300,
):
    """
    Return:
      conv_scaled : [C, n_pair, L_a+L_b-1]
      log_scale   : [C]  (log of stabilization scale per C)
    """
    next_len = A.size(-1) + B.size(-1) - 1
    N = 1 << ((next_len - 1).bit_length())

    fftA = torch.fft.rfft(A, n=N, dim=-1)
    fftB = torch.fft.rfft(B, n=N, dim=-1)
    conv_full = torch.fft.irfft(fftA * fftB, n=N, dim=-1)
    conv = conv_full[..., :next_len]  # [C, n_pair, next_len]

    # numerical stabilization ONLY: detach so it won't kill gradients via amax
    max_val = conv.detach().abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)  # [C,1,1]
    conv_scaled = conv / max_val
    log_scale = max_val.squeeze(-1).squeeze(-1).log()  # [C]

    return conv_scaled.contiguous(), log_scale


# ============================================================
# 2) Multiply (1-p + p z) polynomials per class via FFT merges
# ============================================================
def coeff_product_fast_norm(p_c_batch: torch.Tensor):
    """
    p_c_batch : [C, s]  (float64 recommended)
    Return:
      coeffs_scaled : [C, s+1]
      log_scale_tot : [C]
    """
    C, s = p_c_batch.shape
    polys = torch.stack((1.0 - p_c_batch, p_c_batch), dim=2)  # [C, s, 2]
    cur_len = 2
    log_scale_tot = torch.zeros(C, 1, device=p_c_batch.device, dtype=p_c_batch.dtype)  # [C,1]

    while polys.size(1) > 1:
        if polys.size(1) & 1:
            pad = polys.new_zeros(C, 1, cur_len)
            pad[..., 0] = 1.0
            polys = torch.cat((polys, pad), dim=1)

        polys = polys.view(C, -1, 2, cur_len)  # [C, n_pair, 2, cur_len]
        A = polys[:, :, 0]  # [C, n_pair, cur_len]
        B = polys[:, :, 1]

        conv, log_s = batch_fft_convolve_one_pass(A, B)  # conv: [C, n_pair, next_len]
        log_scale_tot = log_scale_tot + log_s.unsqueeze(1)

        cur_len = conv.size(-1)
        polys = conv.view(C, -1, cur_len)  # [C, n_poly, cur_len]

    coeffs = polys.squeeze(1)             # [C, s+1]
    log_scale_tot = log_scale_tot.squeeze(1)  # [C]
    return coeffs, log_scale_tot


# ============================================================
# 3) Single-bag CC loss (IMPORTANT: no "+1e-12" smoothing)
# ============================================================
def compute_CC_loss_fft_precise(
    softmax_p: torch.Tensor,      # [s, C] probabilities
    proportions: torch.Tensor,    # [C] assumed correct
) -> torch.Tensor:
    """
    Returns scalar:
      loss = - sum_c log P(K_c = k_c)
    where K_c is Poisson-binomial count from Bernoulli probs softmax_p[:,c].
    """
    assert softmax_p.dim() == 2, f"softmax_p expected [s,C], got {softmax_p.shape}"
    assert proportions.dim() == 1, f"proportions expected [C], got {proportions.shape}"

    s, C = softmax_p.shape
    dev = softmax_p.device
    dt = torch.float64
    tiny = torch.finfo(torch.float64).tiny  # ~2e-308
    # [C, s]

    p = softmax_p.to(dt).t().clamp(min=tiny, max=1.0 - tiny)

    # integer targets (do not renorm proportions)
    k_c = proportions_to_counts_exact_vmap(proportions.to(dev), s)  # [C]

    # coefficients
    coeffs, log_sc = coeff_product_fast_norm(p)  # [C, s+1], [C]

    # IMPORTANT: do NOT add 1e-12 here (kills gradients for s=128)
    coeffs_pos = coeffs.clamp_min(1e-80)  # only fix tiny negatives from FFT noise

    log_coeffs = torch.log(coeffs_pos) + log_sc.unsqueeze(1)  # [C, s+1]

    # normalize per class in log-spac
    logZ = torch.logsumexp(log_coeffs, dim=1, keepdim=True)    # [C,1]
    log_coeffs_norm = log_coeffs - logZ

    idx = torch.arange(C, device=dev)
    log_a_k = log_coeffs_norm[idx, k_c]  # prob >= ~1e-35

    loss = -log_a_k.sum()

    return loss.to(softmax_p.dtype)


# ============================================================
# 4) Batched wrapper
# ============================================================
def compute_CC_loss_fft_precise_batched(
    softmax_p_batch: torch.Tensor,     # [B, s, C]
    proportions_batch: torch.Tensor,   # [B, C]
    reduce: Optional[str] = "mean",    # "mean" recommended
) -> torch.Tensor:
    assert softmax_p_batch.dim() == 3, f"softmax_p_batch expected [B,s,C], got {softmax_p_batch.shape}"
    assert proportions_batch.dim() == 2, f"proportions_batch expected [B,C], got {proportions_batch.shape}"
    B, s, C = softmax_p_batch.shape
    assert proportions_batch.shape == (B, C)

    dev = softmax_p_batch.device

    try:
        from torch.func import vmap
        def _one(softmax_p, proportions):
            return compute_CC_loss_fft_precise(softmax_p, proportions)
        loss_b = vmap(_one)(softmax_p_batch, proportions_batch)  # [B]
    except Exception:
        loss_list = []
        for b in range(B):
            loss_list.append(compute_CC_loss_fft_precise(
                softmax_p_batch[b],
                proportions_batch[b],
            ))
        loss_b = torch.stack(loss_list, dim=0).to(dev)

    if reduce is None:
        return loss_b.to(softmax_p_batch.dtype)
    if reduce == "mean":
        return loss_b.mean().to(softmax_p_batch.dtype)
    if reduce == "sum":
        return loss_b.sum().to(softmax_p_batch.dtype)
    raise ValueError("reduce must be None | 'mean' | 'sum'")
