# Twitter Ethnicity 2017 (Race3)

This adapter targets the real-world Twitter demographic LLP experiment from:

Ehsan Mohammady Ardehaly and Aron Culotta. “Co-training for Demographic
Classification Using Deep Learning from Label Proportions.” ICDM Workshops,
2017. [Paper](https://arxiv.org/abs/1709.04108).

The classification unit is a Twitter **user**, not a tweet. The default task is
Race3 with `0 = White`, `1 = Black`, and `2 = Hispanic`. A user's profile image
is the primary view. Aggregated tweets (up to 200 per user in the paper) are an
optional future/adaptation view and are not required by this image benchmark.

## Data availability

`DATA_MISSING` as of 2026-08-10: no original Twitter user/image release or
author replication archive was found in this repository, the paper's arXiv
record, or the author's publication page. The paper says that replication code
and data would be made available upon publication, but the audit did not locate
that release. PLeNCH therefore does **not** download current X/Twitter users,
does not fabricate county assignments or labels, and does not substitute a
different demographic image dataset.

A public-source audit also found that the ArchiveTeam Twitter Stream Grab TAR
objects are currently marked private. GESIS dataset `10.7802/1166` contains
2014/2015 U.S. geotagged tweet IDs organized by county, but access is restricted
and the records still require policy-compliant rehydration. UCI's public Twitter
Geospatial Data contains coordinates, timestamps, and time zones only: it has
no tweet IDs, user IDs, or historical profile images. Consequently none of
these can by itself produce the user-level image benchmark.

The paper reports approximately 10,500 retained one-face Twitter users in 85
county bags (about 124 users per county), plus a manually labeled evaluation set
of 320 users. The reported Race3 evaluation distribution is 210 White, 80
Black, and 30 Hispanic users. County bag targets come from U.S. Census county
race/ethnicity statistics.

## How the paper constructs the data

The paper's reported construction is:

1. collect about 120,000 tweets with the Twitter Streaming API;
2. use tweet geolocation to associate each user with a U.S. county;
3. remove users without a profile image (about 33,000 records remain);
4. run Viola-Jones face detection and retain training images with exactly one
   detected face (about 10,500 users remain);
5. optionally download up to 200 recent tweets for each retained user;
6. group retained users by county, producing 85 natural bags with a reported
   mean size of 124; and
7. attach county Census race/ethnicity shares as each bag's weak label.

The paper cites Mohammady and Culotta (2014) for the county constraint. That
work specifies the official `CC-EST2012-ALLDATA` source and constructs county
FIPS as the two-digit state code plus three-digit county code. For the three
retained categories, its stated definitions map to:

- White: `NHWAC_MALE + NHWAC_FEMALE` (non-Hispanic White alone or in
  combination);
- Black: `NHBAC_MALE + NHBAC_FEMALE` (non-Hispanic Black alone or in
  combination); and
- Hispanic: `H_MALE + H_FEMALE` (Hispanic of any race).

Each count is divided by `TOT_POP`, selecting county totals for the July 1,
2012 estimate (`SUMLEV=50`, `YEAR=5`, `AGEGRP=0`). Because omitted groups and
"in combination" categories mean these three raw shares need not sum to one,
the downloader never renormalizes them. The separate preparation command below
requires an explicit normalization flag before they become a three-class LLP
target.

The Census part is reproducible directly from the official source:

```bash
python -m plench.scripts.build_twitter_census_2012 \
  --data-root data/twitter_ethnicity_2017
```

If `raw/users.csv` already exists, only its county FIPS values are written;
otherwise all counties are written. The command also writes a provenance JSON.
An already downloaded official CSV can be supplied with `--input-csv`.

What this command cannot reconstruct is the paper's exact historical sample:
the publication does not report the collection dates, the 85 county IDs, the
retained Twitter user/tweet IDs, the contemporaneous profile-image files, or
the identities and labels of the 320-person evaluation set. Recollecting
current X accounts would therefore create a new adaptation, not the published
Twitter benchmark.

## Historical reconstruction pipeline

For an authorized historical Twitter v1 JSON archive, PLeNCH can construct a
clearly labeled reconstruction. It does not query the current X API. Accepted
inputs are JSONL, gzip/bzip2-compressed JSONL, and streaming TAR archives whose
members use those formats. Only exact tweet coordinates are accepted; place
bounding-box centroids are deliberately not treated as user locations.

First map historical users to Census/TIGER county polygons:

```bash
python -m plench.scripts.construct_twitter_ethnicity_2017 extract \
  --archive /authorized/archive/twitter-stream.tar \
  --county-file /reference/tl_2012_us_county.shp \
  --data-root data/twitter_ethnicity_2017
```

This writes `raw/archive_candidates.csv`. If a user has exact-geotagged tweets
in more than one county, the modal county is used, with FIPS as the deterministic
tie breaker. `--write-observed-text` stores up to 200 texts seen in the supplied
stream, but provenance explicitly records that these are not equivalent to the
paper's separate “recent 200 tweets” API lookup.

Next download only the profile-image URLs recorded in that historical JSON:

```bash
python -m plench.scripts.construct_twitter_ethnicity_2017 images \
  --data-root data/twitter_ethnicity_2017
```

Finally run the OpenCV Viola-Jones cascade and keep exactly one detected face:

```bash
python -m plench.scripts.construct_twitter_ethnicity_2017 faces \
  --data-root data/twitter_ethnicity_2017
```

The last stage writes `raw/users.csv`, `raw/face_detection_audit.csv`, and a
provenance JSON containing the exact cascade parameters. The paper does not
publish those detector parameters, so this pipeline does not claim bit-exact
sample equivalence. No instance-level race/ethnicity label is created at any
stage. Run this CPU-heavy face stage as a PBS job on Miyabi/compute server rather than on a
login node.

## Required historical files

Place an authentic historical release under:

```text
data/twitter_ethnicity_2017/
  raw/
    images/
    tweets/                              # optional text view
    users.csv
    census_county_proportions.csv
    evaluation_users.csv                # optional for training, required for metrics
```

`users.csv` requires:

```text
instance_id, twitter_user_id, county_fips, image_path, text_path
```

- `instance_id` or `twitter_user_id` is required; `instance_id` is preferred.
- `county_fips` and `image_path` are required.
- `text_path` and `twitter_user_id` may be blank if the release anonymizes or
  omits them.
- Training labels must not be present.

`census_county_proportions.csv` requires:

```text
county_fips,p_white,p_black,p_hispanic
```

The preparer checks finite, non-negative values and does not assume that the
three columns sum to one. If explicit Race3 renormalization is wanted, pass
`--normalize-race3-proportions`; both raw values and their original sum are
retained in the processed file. Without that flag, non-unit sums are rejected.

`evaluation_users.csv`, when available, requires:

```text
instance_id,twitter_user_id,image_path,label
```

Labels may be `0/1/2` or `White/Black/Hispanic`. Evaluation users are checked
for identifier overlap with training users. Their labels are loaded only by the
evaluation dataset and are never returned by the LLP training loader.

## Prepare and extract features

```bash
python -m plench.scripts.prepare_twitter_ethnicity_2017 \
  --data-root data/twitter_ethnicity_2017 \
  --normalize-race3-proportions

pip install -e '.[twitter]'

python -m plench.scripts.extract_twitter_xception_features \
  --data-root data/twitter_ethnicity_2017
```

The processed contract is:

```text
processed/
  manifest.csv
  county_proportions.csv
  metadata.json
  xception_features.npy     # [N, 2048], float32
  instance_ids.npy          # exact manifest order
```

Feature extraction uses an ImageNet-pretrained Xception without its classifier,
299×299 RGB input, and global average pooling. This frozen 2048-dimensional
representation is the benchmark-friendly PLeNCH adaptation. It is not a claim
that the paper used precomputed features: the paper trained an Xception-based
Deep LLP model and froze its early blocks.

An optional end-to-end path is available by overriding the model:

```bash
python -m plench.train \
  --config plench/configs/twitter_ethnicity_2017/race3_xception_bs32.json \
  --hparams '{"model":"Xception","lr":0.0001,"weight_decay":0.0001}'
```

It reads the original images, applies 299×299 Xception preprocessing, uses
ImageNet initialization, and freezes the stem plus the first two named Xception
blocks. This is the paper-faithful option; frozen features remain the default.

## Natural county bags

PLeNCH accepts a maximum bag size of 32, 64, 128, or optionally 256. Users are
shuffled reproducibly **within each county** and large counties are split into
chunks of at most that size. Every chunk inherits the same county Census target.
Small counties and final remainder chunks stay smaller; users are never mixed
across counties. Because those bags are variable-size, the adapter loads one bag
per DataLoader batch and updates PLeNCH's fixed-size objective parameter to the
current chunk size.

Run one of the checked-in configurations:

```bash
python -m plench.train \
  --config plench/configs/twitter_ethnicity_2017/race3_xception_bs32.json

python -m plench.train \
  --config plench/configs/twitter_ethnicity_2017/race3_xception_bs64.json

python -m plench.train \
  --config plench/configs/twitter_ethnicity_2017/race3_xception_bs128.json
```

The default `PM` objective is the Batch Averager / proportion cross-entropy
baseline: it averages instance softmax outputs and minimizes cross-entropy with
the county target (equivalent to KL up to the target entropy). The dataset goes
through the same PLeNCH algorithm interface, so another existing method can be
selected with `--algorithm` while retaining the same features and bags.

The shared default feature head follows the existing PLeNCH MLP convention
(`2048 -> 500 -> ReLU -> 3`). A direct `2048 -> 3` linear ablation is available
with `--hparams '{"model":"Linear","lr":0.001}'`. All LLP methods receive the
same selected head in a comparison.

When the historical evaluation set is present, PLeNCH reports accuracy, macro
F1, weighted F1, per-class precision/recall/F1/support, and a confusion matrix.
If it is absent, training remains available and the log explicitly reports
`DATA_MISSING`; no instance-level score is invented.
