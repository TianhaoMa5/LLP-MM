#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SEED DATA_ROOT [CUDA_DEVICE]" >&2
  exit 2
fi

seed="$1"
data_root="$2"
cuda_device="${3:-}"

case "$seed" in
  0|1|2) ;;
  *) echo "SEED must be 0, 1, or 2" >&2; exit 2 ;;
esac

config_dir="plench/configs/fed_isic2019_adam100_seed${seed}"
methods=(
  pm
  dsq
  llp_pvc
  llp_fc
  rot
  easyllp
  easyllp_abs
  generalupm
  generalupm_abs
  llp_mm_order8
)

for method in "${methods[@]}"; do
  config="$config_dir/$method.json"
  echo "Running seed=$seed method=$method"
  if [[ -n "$cuda_device" ]]; then
    CUDA_VISIBLE_DEVICES="$cuda_device" python -m plench.train \
      --config "$config" --data_root "$data_root"
  else
    python -m plench.train --config "$config" --data_root "$data_root"
  fi
done
