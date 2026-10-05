# Reproducibility record

## Validation performed for this release

The clean release was validated on 2026-09-16 with Python 3.11.15. The full
dataset-free regression suite completed with:

```text
147 passed, 5 warnings
```

The five warnings are the same PyTorch performance warning emitted by the
batched higher-order FFT helper; there were no correctness failures. The CLI
entry point `python -m plench.train --help` also completed successfully.

The most relevant regression coverage includes:

- deterministic, label-blind Fed-ISIC2019 feature-bag assignment;
- complete, non-overlapping sample coverage and bag sizes in `[16, 128]`;
- eight-dimensional proportions that sum to one;
- feature-cache and bag-cache validation;
- LLP-MM cross-entropy moment losses using stable float64 dynamic programming;
- variable-bag behavior for PM and all reported comparison methods.

These tests use synthetic fixtures and therefore do not require or redistribute
Fed-ISIC2019. A complete medical-image rerun still requires the user to obtain
the official FLamby data and pretrained model weights.

## Tested environment

```text
Python       3.11.15
PyTorch      2.13.0
torchvision  0.28.0
NumPy        2.4.6
SciPy        1.17.1
scikit-learn 1.9.0
pandas       2.3.3
Pillow       12.3.0
PyYAML       6.0.3
tqdm         4.69.1
h5py         3.16.0
pytest       9.1.1
```

## Determinism and data isolation

Fed-ISIC2019 uses the official FLamby labels and pooled train/test split. Train
and test features and bags are constructed independently. Feature extraction
uses a frozen encoder in evaluation mode with deterministic preprocessing.
K-means receives the configured seed. Bag membership is computed from features
without labels or medical-center IDs; labels are attached only afterward to
compute aggregate proportions.

Feature and bag caches include compatibility metadata and content fingerprints.
Every loaded cache is rechecked for sample identity, bounds, coverage,
proportions, and the expected feature-similarity advantage over random pairs.

## Paper configuration mapping

The semi-real-world appendix uses the following directories:

```text
plench/configs/fed_isic2019_adam100_seed0/
plench/configs/fed_isic2019_adam100_seed1/
plench/configs/fed_isic2019_adam100_seed2/
```

Each directory contains PM, DSQ, LLP-PVC, LLP-FC, ROT, EasyLLP,
EasyLLP-ABS, GeneralUPM, GeneralUPM-ABS, and LLP-MM. Output directories and the
dataset root can be overridden without changing the optimization settings.

## What is and is not verified

Verified here: source importability, command-line parsing, all 147 regression
tests, deterministic synthetic bag construction, configuration consistency,
and absence of private data or machine-specific paths in the release.

Not rerun during release packaging: the complete three-seed, 100-epoch
Fed-ISIC2019 GPU experiment. The compact CSV records the paper values, while a
fresh full rerun produces the auditable per-epoch JSONL and checkpoints under
the configured output directories.
