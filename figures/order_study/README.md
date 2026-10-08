Updated Cluster Bag order figure

48 verified order-ablation runs plus 8 verified PM order-1 points and 8 verified seed-0 baseline runs (CIFAR10 order 8; CIFAR100 order 3), matched by frozen manifest digest. Existing points retain their prior final test_acc values; order13 uses best logged test_acc. All use 500 epochs and uniform order weights.

The original plotting script was unavailable. Other curves were recovered from the vector PDF paths and linear axis ticks, rounded to 0.01; original_vector_data.json preserves them. Order 1 uses the corresponding PM seed-0 result as explicitly requested. CIFAR10 order 13 is complete for all bags and uses best logged test accuracy: 94.24, 91.87, 89.05, 72.21%. Runtime is original, not from the new concurrent runs.

The figure was uploaded to the existing Overleaf sections file on 2026-10-08; online compile verification is recorded separately. The PDF has the original basename for replacement.

User override (2026-10-08): CIFAR10 / bag128 / order2 = 71.91%, replacing 73.80%. Raw rerun evidence and seed were not provided. This point is user-reported, not newly independently validated; provenance is recorded in user_rerun_overrides.json. No seed or weight annotations were added to the figure.
