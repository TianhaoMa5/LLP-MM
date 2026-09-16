# LLP-MM

Anonymous implementation and reproduction package for **Learning from Label
Proportions via Moment Matching (LLP-MM)**. The repository contains the LLP-MM
objective, comparison methods, shared training/evaluation code, experiment
configurations, and the Fed-ISIC2019 feature-bag pipeline used in the paper.

## Installation

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the code without downloading a dataset:

```bash
python -m pytest plench/tests -q
python -m plench.train --help
```

The release was checked with all 147 tests passing. The exact tested package
versions and the scope of this validation are recorded in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Fed-ISIC2019 semi-real-world experiment

Follow the official
[FLamby Fed-ISIC2019 instructions](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019)
to obtain the images and metadata. Raw medical images are not redistributed.

Prepare deterministic DINOv2 ViT-S/14 features and feature-dependent bags:

```bash
python -m plench.scripts.prepare_fed_isic2019 \
  --data_root /path/to/fed_isic2019 \
  --device auto \
  --extraction_batch_size 64 \
  --num_workers 4 \
  --encoder dinov2_vits14 \
  --min_bag_size 16 \
  --max_bag_size 128 \
  --target_bag_size 64 \
  --seed 0
```

Run one paper seed (all ten methods, in the table order):

```bash
bash scripts/run_fed_isic2019_seed.sh 0 /path/to/fed_isic2019
```

Use seeds `0`, `1`, and `2` for the reported experiment. The canonical configs
are under `plench/configs/fed_isic2019_adam100_seed{0,1,2}` and use:

- Adam, learning rate `0.001`, weight decay `5e-4`;
- 100 epochs and an approximate 512-image optimizer-step budget;
- ImageNet-pretrained ResNet-18 at 224-pixel resolution;
- variable feature-bag sizes in `[16, 128]`, target size 64;
- order 8, cross-entropy moment losses, float64 stable dynamic programming for
  LLP-MM.

The downstream model is trained from images. Frozen DINOv2 embeddings determine
bag membership only. See [`plench/README_FED_ISIC2019.md`](plench/README_FED_ISIC2019.md)
for the dataset, cache, leakage, and bag-validity contracts. Aggregate appendix
results are stored in [`results/fed_isic2019_summary.csv`](results/fed_isic2019_summary.csv).

## Repository structure

```text
plench/core/        LLP objectives, including LLP-MM
plench/data/        datasets, loaders, and bag construction
plench/configs/     experiment configurations
plench/scripts/     preparation and diagnostic utilities
plench/tests/       synthetic, dataset-free regression tests
results/            compact paper-level result summaries
scripts/            reproducible launch helpers
```

Generated datasets, feature caches, checkpoints, logs, and full prediction
files are intentionally excluded from Git.
