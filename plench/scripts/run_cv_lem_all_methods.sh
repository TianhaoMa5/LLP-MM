#!/usr/bin/env bash
set -euo pipefail

# Full CV/LEM LLP benchmark template. It uses the active preprocessed split,
# which is all_class_coverage_70_10_20_seed42 by default.
PLENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(dirname "$PLENCH_DIR")"
cd "$PROJECT_ROOT"

export PLENCH_PYTHON="${PLENCH_PYTHON:-$(command -v python3)}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

DATA_DIR="${DATA_DIR:-$PLENCH_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PLENCH_DIR/outputs/sweeps/cv_lem_all_methods_70_10_20}"
COMMAND_LAUNCHER="${COMMAND_LAUNCHER:-dummy}"
DATASETS="${DATASETS:-CV LEM}"
BAG_SIZES="${BAG_SIZES:-32 64 128 256}"
BAG_BUILD="${BAG_BUILD:-random}"
INSTANCES_PER_EPOCH="${INSTANCES_PER_EPOCH:-200000}"
EPOCHS="${EPOCHS:-100}"
BATCHSIZE="${BATCHSIZE:-2}"
N_TRIALS="${N_TRIALS:-1}"
N_HPARAMS="${N_HPARAMS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
PI="${PI:-1.0}"
INCLUDE_FLOWLLP="${INCLUDE_FLOWLLP:-1}"
INCLUDE_LLP_MM="${INCLUDE_LLP_MM:-1}"
LLP_MM_ORDER="${LLP_MM_ORDER:-6}"

read -r -a dataset_array <<< "$DATASETS"
read -r -a bag_size_array <<< "$BAG_SIZES"

algorithms=(
  PM LLP_DSQ LLP_PVC LLP_SimCLR LLP_PT LLP_FC ROT EasyLLP GeneralUPM
  NonClipOVR LLP_AHIL LLP_DC LLP_FixMatch LLP_SoftMatch LLP_VAT
)
if [[ "$INCLUDE_FLOWLLP" == "1" ]]; then
  algorithms+=(LLP_FlowLLP)
fi

if [[ "$COMMAND_LAUNCHER" != "dummy" ]]; then
  "$PLENCH_PYTHON" - <<'PY'
import importlib.util
missing = [name for name in ("ot", "ortools") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing all-method dependencies: " + ", ".join(missing) +
                     ". Install POT and ortools before launching.")
PY
fi

for bag_size in "${bag_size_array[@]}"; do
  full_bags=$((INSTANCES_PER_EPOCH / bag_size))
  steps_per_epoch=$(((full_bags + BATCHSIZE - 1) / BATCHSIZE))
  checkpoint_freq=$((10 * steps_per_epoch))
  output_dir="$OUTPUT_ROOT/bag${bag_size}_${BAG_BUILD}"

  "$PLENCH_PYTHON" "$PLENCH_DIR/sweep.py" launch \
    --datasets "${dataset_array[@]}" \
    --algorithms "${algorithms[@]}" \
    --data_dir "$DATA_DIR" \
    --output_dir "$output_dir" \
    --bag_build "$BAG_BUILD" --pi "$PI" --cluster-seed "$CLUSTER_SEED" \
    --bagsize "$bag_size" --batchsize "$BATCHSIZE" \
    --instances-per-epoch "$INSTANCES_PER_EPOCH" \
    --epochs "$EPOCHS" --checkpoint_freq "$checkpoint_freq" \
    --n_trials "$N_TRIALS" --n_hparams "$N_HPARAMS" \
    --num-workers "$NUM_WORKERS" \
    --command_launcher "$COMMAND_LAUNCHER" \
    --skip_confirmation --skip_model_save

  if [[ "$INCLUDE_LLP_MM" == "1" ]]; then
    mm_weights="$($PLENCH_PYTHON -c "import json; n=$LLP_MM_ORDER; print(json.dumps([1.0/n]*n))")"
    mm_hparams="{\"model\":\"RemoteResNet18\",\"order\":$LLP_MM_ORDER,\"moment_loss_type\":\"ce\",\"moment_algorithm\":\"stable_dp\",\"moment_compute_dtype\":\"float64\",\"moment_ce_smoothing_tau\":0.0001,\"order_weights\":$mm_weights}"
    "$PLENCH_PYTHON" "$PLENCH_DIR/sweep.py" launch \
      --datasets "${dataset_array[@]}" \
      --algorithms LLP_MM \
      --data_dir "$DATA_DIR" \
      --output_dir "$output_dir" \
      --bag_build "$BAG_BUILD" --pi "$PI" --cluster-seed "$CLUSTER_SEED" \
      --bagsize "$bag_size" --batchsize "$BATCHSIZE" \
      --instances-per-epoch "$INSTANCES_PER_EPOCH" \
      --epochs "$EPOCHS" --checkpoint_freq "$checkpoint_freq" \
      --n_trials "$N_TRIALS" --n_hparams "$N_HPARAMS" \
      --num-workers "$NUM_WORKERS" --hparams "$mm_hparams" \
      --command_launcher "$COMMAND_LAUNCHER" \
      --skip_confirmation --skip_model_save
  fi
done

echo "Prepared CV/LEM all-method sweep under: $OUTPUT_ROOT"
if [[ "$COMMAND_LAUNCHER" == "dummy" ]]; then
  echo "Dry run only. Set COMMAND_LAUNCHER=multi_gpu and CUDA_VISIBLE_DEVICES before launching."
fi
