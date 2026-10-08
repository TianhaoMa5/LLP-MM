# Reproducibility record

Release review: **2026-10-08**. Extends original LLP-MM commit
`03e3d21832d8227b77cabfb9cbe5906e5259160a`, preserving separate fixed-image
historical and float64 stable-DP natural-bag kernels.

## Validation

Tests cover independent moment values/gradients, order-one equivalence, variable
bags, seeded Cluster alignment, manifest validation, ABS gradients, FlowLLP
optimizer/scheduler settings, GAN pairing, order-launch isolation, and result
selection and aggregation. [public_release_validation.json](results/public_release_validation.json)
records the final test, CLI, wheel and isolated-package checks. CUDA and optional
GeoPandas checks skip when unavailable. Linux CI runs the dataset-free checks.

The code review does not claim fresh training or complete provenance for every
historical paper experiment. The corrected Cluster grid does have 330 validated
normal results and 66 verified numerical divergences; compact histories and
failure records are bundled. See [CURRENT_RESULTS.md](docs/CURRENT_RESULTS.md).
The earlier [coverage audit](docs/PAPER_COVERAGE.md) describes a historical
snapshot; its Cluster-completion gaps are superseded by the current record.

## Frozen settings

- Image tables: 500 epochs, 1,024 images/update, original backbones and optimizer
  settings, uniform MM weights and default orders 8/3/3.
- Fixed-image MM: `paper_image`; KU: float64 stable DP, CE and smoothing `1e-4`.
- ABS: flooding enabled at zero threshold. Non-ABS disables flooding.
- Corrected Cluster: aligned source indices, deterministic seeds, shared frozen
  manifests and independent numerical-failure records.
- KU: 100 epochs, maximum **test Macro-F1** and population standard deviation.
  Cluster: maximum **test accuracy** and sample standard deviation. These are
  paper reporting conventions, not validation-only selection.

The retained miniImageNet map was fitted on the merged population, including
held-out images. Correcting its projection does not make it train-only clustering.
Ordering and provenance are documented in [DATASETS.md](docs/DATASETS.md).

The order plot retains the original recovered runtime curve. One accuracy point
is user reported and explicitly labeled in its data provenance. New concurrent
training times do not replace original single-GPU timing measurements.

Earlier code differences remain in [VERSION_DIFFERENCES.md](docs/VERSION_DIFFERENCES.md).
They do not imply every historical run used the same faulty source version.
