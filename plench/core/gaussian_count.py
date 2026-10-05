"""Gaussian approximation to a bag's Poisson-multinomial label counts."""

import math

import torch


def gaussian_count_nll(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    bag_size: int,
    count_variance_floor: float = 1.0 / 12.0,
) -> torch.Tensor:
    """Mean per-coordinate Gaussian negative log likelihood for fixed-size bags.

    Conditional independence gives count mean ``sum_i p_i`` and covariance
    ``sum_i (diag(p_i) - p_i p_i^T)``.  Counts sum to ``bag_size``, so the full
    covariance is singular.  Dropping the final class gives a nonredundant
    coordinate system; the variance floor models count quantization and keeps
    nearly deterministic predictions numerically well-conditioned.
    """
    if logits.ndim != 2 or proportions.ndim != 2:
        raise ValueError("logits and proportions must be matrices")
    bags, classes = proportions.shape
    if classes < 2 or logits.shape[1] != classes:
        raise ValueError("Gaussian count likelihood requires matching C >= 2")
    if bag_size <= 0 or logits.shape[0] != bags * bag_size:
        raise ValueError("logits must contain exactly B * bag_size instances")
    if not math.isfinite(count_variance_floor) or count_variance_floor <= 0:
        raise ValueError("count_variance_floor must be positive and finite")

    # Double precision is important for the Cholesky solve and log determinant.
    probabilities = logits.to(torch.float64).softmax(dim=-1)
    targets = proportions.to(device=logits.device, dtype=torch.float64)
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("proportions must be finite")
    if bool((targets < -1e-6).any()) or bool((targets > 1 + 1e-6).any()):
        raise ValueError("proportions must lie in [0, 1]")
    if not bool(torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-5, rtol=0)):
        raise ValueError("each bag proportion vector must sum to one")

    reduced = probabilities.reshape(bags, bag_size, classes)[..., :-1]
    mean = reduced.sum(dim=1)
    covariance = torch.diag_embed(mean) - reduced.transpose(1, 2) @ reduced
    dimension = classes - 1
    covariance = covariance + count_variance_floor * torch.eye(
        dimension, device=logits.device, dtype=torch.float64
    )
    residual = bag_size * targets[:, :-1] - mean
    factor = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
    quadratic = (residual * solved).sum(-1)
    log_determinant = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return (0.5 * (quadratic + log_determinant + dimension * math.log(2 * math.pi))).mean() / dimension
