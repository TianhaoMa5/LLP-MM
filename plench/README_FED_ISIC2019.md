# Fed-ISIC2019 feature-dependent LLP bags

This adapter adds FLamby's Fed-ISIC2019 pooled image-classification dataset to
the existing PLeNCH training path. It does not redefine the labels or create a
new validation split.

## Dataset contract

- 23,247 dermoscopic images retained by FLamby.
- Six medical centers/clients.
- Eight classes in the exact FLamby column/target order:
  `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`.
- Fixed FLamby pooled split: 18,597 train and 4,650 test images.
- FLamby provides no validation fold for this dataset. The adapter therefore
  leaves validation absent instead of silently moving official training or test
  examples. Test bags are used only for evaluation/monitoring and never for
  optimization or scheduler decisions.

The authoritative metadata is FLamby's
[`dataset_creation_scripts/train_test_split`](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split).
The parser validates the eight one-hot columns, integer `target`, `center`,
`fold`, and `fold2` for every row.

Follow [FLamby's Fed-ISIC2019 download and licence instructions](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019)
to obtain and preprocess the images. A normal data root is:

```text
fed_isic2019/
├── ISIC_2019_Training_Input_preprocessed/
│   ├── ISIC_0000000.jpg
│   └── ...
└── train_test_split
```

`train_test_split` can instead be supplied with `--metadata_csv`; an installed
FLamby package is also auto-detected. The image directory can be supplied with
`--image_dir`, but it must remain below `--data_root` so cache paths are
portable.

## Feature-bag preparation

```text
image
  -> deterministic Resize(256)/CenterCrop(224)/ImageNet normalization
  -> frozen pretrained DINOv2 ViT-S/14
  -> L2-normalized embedding
  -> split-local feature clustering
  -> feature-only merge/split safeguards
  -> fixed variable-size membership
  -> labels attached afterward to compute 8-class proportions
```

The public `assign_bags(features, ...)` function has no label, target, center,
or client argument. Train and test are extracted, cached, and clustered
separately. Center IDs are preserved for analysis, but never enter the default
assignment function.

Prepare the data once:

```bash
python3 -m plench.scripts.prepare_fed_isic2019 \
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

The preparation command prints split-level bag count and min/median/mean/max
size, the exact size histogram, class-proportion summaries in the metadata,
and two cosine diagnostics. The feature-dependence sanity check compares the
average of all within-bag pairs against the exact expected similarity of a
uniform random pair from the same split.

Generated files follow the existing cache-under-dataset-root convention:

```text
cache/fed_isic2019/dinov2_vits14/
├── instances.csv
├── train_features.pt
├── test_features.pt
└── feature_bags_dinov2_vits14_min16_max128_target64_seed0/
    ├── train_featurebags.pt
    ├── test_featurebags.pt
    └── metadata.json
fed_isic2019_active_cache.json
```

Each feature cache stores the embedding, label, global sample index, sample ID,
client ID, and relative image path. Each bag cache fingerprints the encoder,
size bounds, target size, seed, sample count/identity, and exact feature bytes.
An incompatible existing cache is rejected rather than reused.

Every bag cache is checked for:

- size in `[16, 128]` with defaults;
- complete, non-overlapping coverage of its split;
- an 8-dimensional class proportion vector summing to one;
- cached class counts/proportions matching hidden instance labels;
- split isolation and a label/client-blind membership declaration.

## Train through PLeNCH

After preparation, edit only `data_root` in the supplied config or override it
on the command line:

```bash
python3 -m plench.train \
  --config plench/configs/fed_isic2019_feature_bags.json \
  --data_root /path/to/fed_isic2019
```

Equivalent key arguments are:

```bash
python3 -m plench.train \
  --dataset fed_isic2019 \
  --data_root /path/to/fed_isic2019 \
  --bag_build feature \
  --min_bag_size 16 \
  --max_bag_size 128 \
  --target_avg_bag_size 64 \
  --feature_encoder dinov2_vits14 \
  --seed 0
```

The downstream model is the existing image LLP model (default:
ImageNet-pretrained ResNet-18 at 224 pixels). It reads the original images with
training augmentation. Cached DINOv2 embeddings define membership only and are
not substituted for downstream model inputs.

`batchsize` remains an approximate image budget per optimizer update. Complete
variable-size bags are greedily packed and are never split, so the exact number
of images in an update varies.
