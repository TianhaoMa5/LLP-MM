#!/usr/bin/env bash
# Reconstruct Amazon-WILDS v2.1 when the official CodaLab bundle is unavailable.
#
# This script intentionally operates on the remote data root passed as $1. It
# consumes the official per-category Amazon Review Data (2018) 5-core files and
# the official WILDS preprocessing source checkout already placed below that
# root. Expensive stages have completion markers so interrupted runs can resume.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_ROOT" >&2
  exit 2
fi

data_root=$1
reconstruction_root="$data_root/reconstruction"
source_raw="$reconstruction_root/raw"
wilds_source="$reconstruction_root/wilds-src"
work_root="$reconstruction_root/work"
work_data="$work_root/amazon/data"
target="$data_root/raw/amazon_v2.1"
python_bin=${AMAZON_WILDS_PYTHON:-python}
preprocess_source="$wilds_source/dataset_preprocessing/amazon_yelp"

categories=(
  AMAZON_FASHION All_Beauty Appliances Arts_Crafts_and_Sewing Automotive
  Books CDs_and_Vinyl Cell_Phones_and_Accessories Clothing_Shoes_and_Jewelry
  Digital_Music Electronics Gift_Cards Grocery_and_Gourmet_Food
  Home_and_Kitchen Industrial_and_Scientific Kindle_Store Luxury_Beauty
  Magazine_Subscriptions Movies_and_TV Musical_Instruments Office_Products
  Patio_Lawn_and_Garden Pet_Supplies Prime_Pantry Software Sports_and_Outdoors
  Tools_and_Home_Improvement Toys_and_Games Video_Games
)

while [[ ! -f "$source_raw/DOWNLOAD_COMPLETE" ]]; do
  printf '[%s] waiting for source downloads\n' "$(date -Iseconds)"
  sleep 60
done

if [[ -s "$target/reviews.csv" && -s "$target/splits/user.csv" ]]; then
  echo "A ready official copy arrived first; reconstruction is not needed."
  exit 0
fi

if [[ ! -f "$preprocess_source/process_amazon.py" ]]; then
  echo "missing official WILDS preprocessing checkout: $wilds_source" >&2
  exit 3
fi

mkdir -p "$work_data/preprocessing/token_length" "$work_data/splits"
if [[ ! -e "$work_data/raw" ]]; then
  ln -s "$source_raw" "$work_data/raw"
fi

# The UCSD server exposes the files as ``<category>_5.json.gz``. The official
# WILDS downloader stores the exact same payload at ``<category>.json.gz``
# before the preprocessing functions read it. Keep the downloaded filenames
# intact and provide the names expected by the WILDS source as symlinks.
for category in "${categories[@]}"; do
  source_file="$source_raw/${category}_5.json.gz"
  expected_file="$source_raw/${category}.json.gz"
  if [[ ! -s "$source_file" ]]; then
    echo "missing downloaded category file: $source_file" >&2
    exit 4
  fi
  if [[ ! -e "$expected_file" ]]; then
    ln -s "${category}_5.json.gz" "$expected_file"
  fi
done

export python_bin preprocess_source work_data
tokenize_category() {
  local category=$1
  cd "$preprocess_source"
  "$python_bin" -c \
    'import process_amazon as p, sys; p.compute_token_length(sys.argv[1], sys.argv[2])' \
    "$work_data" "$category"
}
export -f tokenize_category

if [[ ! -f "$reconstruction_root/TOKEN_LENGTHS_COMPLETE" ]]; then
  for category in "${categories[@]}"; do
    if [[ ! -s "$work_data/preprocessing/token_length/${category}.csv" ]]; then
      printf '%s\n' "$category"
    fi
  done | xargs -r -n1 -P16 bash -c 'tokenize_category "$1"' _
  touch "$reconstruction_root/TOKEN_LENGTHS_COMPLETE"
fi

if [[ ! -f "$reconstruction_root/KCORE_COMPLETE" ]]; then
  cd "$preprocess_source"
  "$python_bin" -c \
    'import process_amazon as p, sys; p.process_k_core(sys.argv[1], 30)' \
    "$work_data"
  touch "$reconstruction_root/KCORE_COMPLETE"
fi

if [[ ! -f "$reconstruction_root/USER_SPLIT_COMPLETE" ]]; then
  cd "$preprocess_source"
  "$python_bin" generate_splits_amazon.py --root_dir "$work_root"
  touch "$reconstruction_root/USER_SPLIT_COMPLETE"
fi

# v2.1 first adds the official unlabeled splits and then applies the v2.0
# 25%-reviewer source subsampling. Reversing these stages changes the extra
# reviewer population.
if [[ ! -f "$reconstruction_root/UNLABELED_SPLITS_COMPLETE" ]]; then
  cd "$preprocess_source"
  "$python_bin" create_unlabeled_amazon.py "$work_data"
  touch "$reconstruction_root/UNLABELED_SPLITS_COMPLETE"
fi

if [[ ! -f "$reconstruction_root/V21_SUBSAMPLE_COMPLETE" ]]; then
  cd "$preprocess_source"
  "$python_bin" subsample_amazon.py "$work_data" 0.25
  touch "$reconstruction_root/V21_SUBSAMPLE_COMPLETE"
fi

if [[ -s "$target/reviews.csv" && -s "$target/splits/user.csv" ]]; then
  echo "A ready official copy arrived before publish; keeping that copy."
  exit 0
fi

mkdir -p "$target/splits"
install -m 0644 "$work_data/reviews.csv" "$target/reviews.csv"
install -m 0644 "$work_data/splits/user.csv" "$target/splits/user.csv"
wilds_commit=$(git -C "$wilds_source" rev-parse HEAD)
printf '%s\n' \
  "Amazon-WILDS v2.1 reconstructed from official WILDS preprocessing commit $wilds_commit" \
  "and the official UCSD Amazon Review Data (2018) per-category 5-core files." \
  > "$target/RELEASE_v2.1.txt"

echo "Reconstructed Amazon-WILDS source tables are ready at $target"
