"""Binary and joint multi-class LLP moment-matching objectives."""

from .losses import dllp_bce_loss, dllp_mse_loss, llp_mm_loss
from .multiclass import (
    MulticlassFactorialMomentLoss,
    classifier_count_masses,
    dense_instance_probabilities,
    exact_counts_from_proportions,
    rounded_counts_from_proportions,
    target_count_masses,
)

__all__ = [
    "MulticlassFactorialMomentLoss",
    "classifier_count_masses",
    "dense_instance_probabilities",
    "dllp_bce_loss",
    "dllp_mse_loss",
    "exact_counts_from_proportions",
    "llp_mm_loss",
    "rounded_counts_from_proportions",
    "target_count_masses",
]
