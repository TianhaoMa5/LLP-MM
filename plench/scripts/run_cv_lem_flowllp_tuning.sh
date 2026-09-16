#!/usr/bin/env bash
set -euo pipefail

# Reproducible FlowLLP random-hyperparameter search for CV/LEM. Seed zero is
# the verified formal default; positive hparam seeds sample the registry space.
PLENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(dirname "$PLENCH_DIR")"
cd "$PROJECT_ROOT"

export PLENCH_PYTHON="${PLENCH_PYTHON:-$(command -v python3)}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

DATA_DIR="${DATA_DIR:-$PLENCH_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PLENCH_DIR/outputs/sweeps/cv_lem_flowllp_tuning}"
COMMAND_LAUNCHER="${COMMAND_LAUNCHER:-dummy}"
DATASETS="${DATASETS:-CV LEM}"
BAG_SIZES="${BAG_SIZES:-32 64 128 256}"
BAG_BUILDS="${BAG_BUILDS:-random cluster alphafirst}"
INSTANCES_PER_EPOCH="${INSTANCES_PER_EPOCH:-200000}"
EPOCHS="${EPOCHS:-100}"
BATCHSIZE="${BATCHSIZE:-2}"
N_TRIALS="${N_TRIALS:-1}"
N_HPARAMS="${N_HPARAMS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
RANDOM_PI="${RANDOM_PI:-1.0}"
CLUSTER_PI="${CLUSTER_PI:-1.0}"
ALPHAFIRST_PI="${ALPHAFIRST_PI:-10.0}"

read -r -a dataset_array <<< "$DATASETS"
read -r -a bag_size_array <<< "$BAG_SIZES"
read -r -a bag_build_array <<< "$BAG_BUILDS"

if [[ "$COMMAND_LAUNCHER" != "dummy" ]]; then
  "$PLENCH_PYTHON" - <<'PY'
import importlib.util
if importlib.util.find_spec("ot") is None:
    raise SystemExit("FlowLLP requires POT. Install requirements-flowllp.txt first.")
PY
fi

for bag_build in "${bag_build_array[@]}"; do
  case "$bag_build" in
    random) pi="$RANDOM_PI" ;;
    cluster) pi="$CLUSTER_PI" ;;
    alphafirst) pi="$ALPHAFIRST_PI" ;;
    *) echo "Unsupported BAG_BUILDS entry: $bag_build" >&2; exit 2 ;;
  esac
  for bag_size in "${bag_size_array[@]}"; do
    full_bags=$((INSTANCES_PER_EPOCH / bag_size))
    steps_per_epoch=$(((full_bags + BATCHSIZE - 1) / BATCHSIZE))
    checkpoint_freq=$((10 * steps_per_epoch))

    "$PLENCH_PYTHON" "$PLENCH_DIR/sweep.py" launch \
      --datasets "${dataset_array[@]}" \
      --algorithms LLP_FlowLLP \
      --data_dir "$DATA_DIR" \
      --output_dir "$OUTPUT_ROOT/bag${bag_size}_${bag_build}" \
      --bag_build "$bag_build" --pi "$pi" --cluster-seed "$CLUSTER_SEED" \
      --bagsize "$bag_size" --batchsize "$BATCHSIZE" \
      --instances-per-epoch "$INSTANCES_PER_EPOCH" \
      --epochs "$EPOCHS" --checkpoint_freq "$checkpoint_freq" \
      --n_trials "$N_TRIALS" --n_hparams "$N_HPARAMS" \
      --num-workers "$NUM_WORKERS" \
      --command_launcher "$COMMAND_LAUNCHER" \
      --skip_confirmation --skip_model_save
  done
done

echo "Prepared FlowLLP tuning sweep under: $OUTPUT_ROOT"
echo "Each dataset/bag/build setting contains $N_HPARAMS hparam seeds x $N_TRIALS trials."
if [[ "$COMMAND_LAUNCHER" == "dummy" ]]; then
  echo "Dry run only. Set COMMAND_LAUNCHER=multi_gpu and CUDA_VISIBLE_DEVICES to launch."
fi
