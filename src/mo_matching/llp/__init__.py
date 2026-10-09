"""Binary and joint multi-class LLP moment-matching objectives."""

from .losses import llp_mm_loss
from .paper import LLPHighOrderLoss
from .structured_multiclass import variable_multiclass_llp_mm_loss
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
    "LLPHighOrderLoss",
    "classifier_count_masses",
    "dense_instance_probabilities",
    "exact_counts_from_proportions",
    "llp_mm_loss",
    "rounded_counts_from_proportions",
    "target_count_masses",
    "variable_multiclass_llp_mm_loss",
]
