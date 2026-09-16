# Campo Verde and LEM paper-aligned LLP pipeline

This pipeline follows Scenario III of La Rosa, Oliveira & Ghamisi as closely as
the available sources permit. It uses centre-labelled 21×21 patches, logical
field-disjoint splits, exact bag proportions, and 200,000 newly sampled patch
centres per epoch. The only supported main bag sizes are 32, 64, 128, and 256.

## Processed contract

`<dataset>/processed/` contains:

- `image_stack.npy`: `[C,H,W]` float32 aligned raster stack;
- `center_rows.npy`, `center_cols.npy`: every valid annotated patch centre;
- `labels.npy`: hidden centre-pixel class indices;
- `field_indices.npy`, `field_values.json`: logical field membership;
- `split_codes.npy`, `split_manifest.json`: fixed field-disjoint train/val/test split;
- `class_mapping.json`: explicit raw class names and contiguous indices;
- `normalization.json`: exact train-patch-only channel moments with a strong cache fingerprint;
- `preprocessing_manifest.json`: source hashes, grid, dates/channels, classes, split seed, and version.

Patches are extracted lazily. The entire 21×21 window need not be inside one
field: the class is the class of its centre pixel. Continuous bands use bilinear
resampling before extraction; categorical field/class membership is rasterized
on the target grid with nearest-neighbour semantics.

At epoch `e`, the loader uses a deterministic `seed + e` realization to sample,
augment, shuffle, and partition approximately 200,000 centres. For each bag it
computes `bincount(hidden_labels, minlength=num_classes) / bag_size`. Hidden
instance labels are retained only for this construction and diagnostics;
`algorithm.update()` receives only image instances and bag proportions.

## CV preprocessing

CV uses the May-2016 reference column, Landsat-7 B1–B7, and `Field_numb` as the
logical field identifier. `Id` is not valid: it has only two values in the 513
shapefile rows. One source collision reuses Field_numb 486 for two distant
polygons with different May labels; the manifest records their deterministic
row-suffixed IDs instead of merging them. The 30 m optical grid is bilinearly resampled to 10 m before
building the valid-centre catalogue, so 21 pixels cover about 210×210 m.

From `/path/to/LLP-MM`:

```bash
python3 -m plench.scripts.prepare_remote_sensing \
  --dataset CV \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B1_cut.tif \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B2_cut.tif \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B3_cut.tif \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B4_cut.tif \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B5_cut.tif \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B6_cut.tif \
  --raster plench/data/campo_verde/raw/landsat/7_May/20160505_merge_B7_cut.tif \
  --labels plench/data/campo_verde/raw/reference/Reference/CampoVerde_Oct2015_Jul2016.shp \
  --label-column May_2016 --field-id-column auto --target-resolution 10 \
  --seed 0 --project-root . --output plench/data/campo_verde/processed
```

The split first holds out 50% of logical fields for test, then carves validation
from the original training half. It is one deterministic random field split,
not a class-aware search.

## LEM acquisition and preprocessing

The available official Dec-Apr calendar has 12 descending relative-orbit-126
acquisitions. All 12 were found in Planetary Computer RTC with both VV and VH:

```text
2017-12-09  2017-12-21
2018-01-02  2018-01-14  2018-01-26
2018-02-07  2018-02-19
2018-03-03  2018-03-15  2018-03-27
2018-04-08  2018-04-20
```

Channel order is chronological, with `VV, VH` within each date: 12×2 = 24
channels. The legitimate `not identified` label remains in the main 14-class
mapping. Other empty/background values are not conflated with it. The RTC stack
is a reproducible public reconstruction and is not byte-for-byte identical to
the retired INPE image binaries.

```bash
python3 -m plench.scripts.download_lem_rtc \
  --labels plench/data/lem/raw/reference/classes_mensal_LEM_buffer_cut_v2/classes_mensal_LEM_buffer_cut_v2.shp \
  --output plench/data/lem/raw/rtc

python3 -m plench.scripts.prepare_remote_sensing \
  --dataset LEM \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20171209_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20171221_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180102_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180114_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180126_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180207_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180219_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180303_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180315_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180327_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180408_vv_vh_db.tif \
  --raster plench/data/lem/raw/rtc/lem_s1_rtc_20180420_vv_vh_db.tif \
  --labels plench/data/lem/raw/reference/classes_mensal_LEM_buffer_cut_v2/classes_mensal_LEM_buffer_cut_v2.shp \
  --label-column Feb_2018 --field-id-column Id \
  --acquisition-date 2017-12-09 --acquisition-date 2017-12-21 \
  --acquisition-date 2018-01-02 --acquisition-date 2018-01-14 --acquisition-date 2018-01-26 \
  --acquisition-date 2018-02-07 --acquisition-date 2018-02-19 \
  --acquisition-date 2018-03-03 --acquisition-date 2018-03-15 --acquisition-date 2018-03-27 \
  --acquisition-date 2018-04-08 --acquisition-date 2018-04-20 \
  --seed 0 --project-root . --output plench/data/lem/processed
```

LEM first holds out 25% of logical fields for test, then carves validation from
the original 75% training side.

## Sanity check and experiments

The smoke command validates manifests, field disjointness, every bag size,
dynamic epochs, exact proportions, and one ResNet18/classifier forward pass:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m plench.scripts.smoke_remote_sensing \
  --cv-root plench/data/campo_verde --lem-root plench/data/lem --device cuda \
  --output plench/outputs/paper_remote_sensing_smoke.json
```

The eight main commands are:

```bash
python3 -m plench.train --config plench/configs/remote_sensing/cv_bag32.json
python3 -m plench.train --config plench/configs/remote_sensing/cv_bag64.json
python3 -m plench.train --config plench/configs/remote_sensing/cv_bag128.json
python3 -m plench.train --config plench/configs/remote_sensing/cv_bag256.json
python3 -m plench.train --config plench/configs/remote_sensing/lem_bag32.json
python3 -m plench.train --config plench/configs/remote_sensing/lem_bag64.json
python3 -m plench.train --config plench/configs/remote_sensing/lem_bag128.json
python3 -m plench.train --config plench/configs/remote_sensing/lem_bag256.json
```

These perform long training and should be launched only after the smoke report
succeeds. Final centre-level test evaluation reports overall accuracy, balanced
accuracy, macro-F1, per-class precision/recall/F1, and the confusion matrix.
