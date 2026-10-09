# LLP-MM

Implementation of **Rethinking Learning from Label Proportions via Moment
Matching**, including LLP-MM, baseline comparisons, order ablations,
patient-grouped experiments, and GAN compatibility experiments.

## Install

Use Python 3.11. Run the commands below from the repository root in Bash.

```bash
git clone https://github.com/TianhaoMa5/LLP-MM.git
cd LLP-MM
python -m pip install -e '.[paper]'
```

Choose the GPU for training:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Prepare data

### CIFAR-10 and CIFAR-100

```bash
python - <<'PY'
from torchvision.datasets import CIFAR10, CIFAR100
for dataset in (CIFAR10, CIFAR100):
    for train in (True, False):
        dataset('./data', train=train, download=True)
PY
```

Training supports `random`, `cluster`, and `alphafirst` bags. The required
assignment maps are included in the package.

### miniImageNet

```bash
mkdir -p ./data/miniimagenet
```

Place the cache files in this directory:

```text
data/miniimagenet/mini-imagenet-cache-train.pkl
data/miniimagenet/mini-imagenet-cache-val.pkl
data/miniimagenet/mini-imagenet-cache-test.pkl
```

Use caches containing `image_data` and `class_dict` in the original ordering.
The loader merges the caches and uses 500 training and 100 test images per class.
See [Data preparation](docs/DATASETS.md#miniimagenet) for the required cache format and split.

### KU-Optofil PBC

Download and prepare the images and patient metadata:

```bash
python -m mo_matching.prepare_ku --data-root ./data/ku_optofil_pbc
```

For offline preparation, put `dataset.zip`, `metadata.csv`, and
`metadata_with_patient_level_splits.csv` in `data/ku_optofil_pbc/raw/`, then run:

```bash
python -m mo_matching.prepare_ku --data-root ./data/ku_optofil_pbc --no-download
```

## Image benchmarks

Methods: **LLP-MM, PM, DSQ, LLP-PVC, LLP-FC, ROT, EasyLLP, EasyLLP-ABS,
GeneralUPM, GeneralUPM-ABS, FlowLLP**, plus **LLP-GAN and MM+GAN** for the
GAN comparison below.

The runner uses 500 epochs, 1024 images per update, bag sizes 16/32/64/128,
and seeds 0/1/2. Bag types are `random`, `cluster`, and `alphafirst`.
Default LLP-MM orders are 8/3/3 for CIFAR-10/CIFAR-100/miniImageNet, with equal
moment weights. ABS variants apply absolute loss at threshold zero.

Run one configuration:

```bash
python -m mo_matching.run --dataset CIFAR10 --method LLP-MM \
  --data-root ./data --output-dir ./outputs/CIFAR10_random_LLP-MM_b32_s0 \
  --bag-type random --bag-size 32 --seed 0
```

Run the complete image benchmark matrix. The loops execute sequentially; narrow
the lists to run a subset. Use a new output directory for a repeat experiment.

```bash
for dataset in CIFAR10 CIFAR100 miniImageNet; do
  for mode in random cluster alphafirst; do
    for size in 16 32 64 128; do
      for seed in 0 1 2; do
        for method in LLP-MM PM DSQ LLP-PVC LLP-FC ROT EasyLLP EasyLLP-ABS \
                      GeneralUPM GeneralUPM-ABS FlowLLP; do
          python -m mo_matching.run --dataset "$dataset" --method "$method" \
            --data-root ./data --bag-type "$mode" --bag-size "$size" --seed "$seed" \
            --output-dir "./outputs/${dataset}_${mode}_${method}_b${size}_s${seed}"
        done
      done
    done
  done
done
```

## Order ablation

Order one uses PM. The image order curves use seed 0.

CIFAR-10, orders 1 through 13:

```bash
for mode in random cluster alphafirst; do
  for size in 16 32 64 128; do
    for order in {1..13}; do
      python -m mo_matching.run --dataset CIFAR10 --method LLP-MM \
        --data-root ./data --bag-type "$mode" --bag-size "$size" --seed 0 \
        --order "$order" \
        --output-dir "./outputs/order_CIFAR10_${mode}_b${size}_o${order}_s0"
    done
  done
done
```

CIFAR-100, orders 1/2/3:

```bash
for mode in random cluster alphafirst; do
  for size in 16 32 64 128; do
    for order in 1 2 3; do
      python -m mo_matching.run --dataset CIFAR100 --method LLP-MM \
        --data-root ./data --bag-type "$mode" --bag-size "$size" --seed 0 \
        --order "$order" \
        --output-dir "./outputs/order_CIFAR100_${mode}_b${size}_o${order}_s0"
    done
  done
done
```

## Patient-grouped experiments

KU-Optofil PBC uses patient bags, ImageNet-pretrained ResNet-18, Adam, and
100 epochs. The runner uses four bags per update, or five for GeneralUPM and
GeneralUPM-ABS. The unknown-patient subgroup limit is 128; image `--bag-size`
does not change patient grouping.

LLP-MM, orders 3/5/8 and seeds 0/1/2:

```bash
for order in 3 5 8; do
  for seed in 0 1 2; do
    python -m mo_matching.run --dataset KUOptofilPBC --method LLP-MM \
      --data-root ./data/ku_optofil_pbc --order "$order" --seed "$seed" \
      --output-dir "./outputs/ku_LLP-MM_o${order}_s${seed}"
  done
done
```

All ten baselines, seeds 0/1/2:

```bash
for method in PM DSQ LLP-PVC LLP-FC ROT EasyLLP EasyLLP-ABS \
              GeneralUPM GeneralUPM-ABS FlowLLP; do
  for seed in 0 1 2; do
    python -m mo_matching.run --dataset KUOptofilPBC --method "$method" \
      --data-root ./data/ku_optofil_pbc --seed "$seed" \
      --output-dir "./outputs/ku_${method}_s${seed}"
  done
done
```

## GAN compatibility

LLP-GAN and MM+GAN use CIFAR-10 and the **same bag file** for each comparison.
First construct shared bags for all sizes and seeds. Random and Cluster use
concentration 1; Alpha-First uses 10.

```bash
for seed in 0 1 2; do
  python -m mo_matching.gan.bags --data-dir ./data --output-dir ./data/gan_bags \
    --modes random cluster --bag-sizes 16 32 64 128 --alpha 1 --seed "$seed"
  python -m mo_matching.gan.bags --data-dir ./data --output-dir ./data/gan_bags \
    --modes alphafirst --bag-sizes 16 32 64 128 --alpha 10 --seed "$seed"
done
```

Train both methods for 500 epochs; MM+GAN uses order eight:

```bash
for mode in random cluster alphafirst; do
  for size in 16 32 64 128; do
    for seed in 0 1 2; do
      for method in LLP-GAN MM+GAN; do
        python -m mo_matching.run --dataset CIFAR10 --method "$method" \
          --data-root ./data --bag-type "$mode" --bag-size "$size" --seed "$seed" \
          --bag-file "./data/gan_bags/cifar10_${mode}_m${size}_seed${seed}.npz" \
          --output-dir "./outputs/gan_${mode}_${method}_b${size}_s${seed}"
      done
    done
  done
done
```

## Check a configuration

Print the resolved settings without training:

```bash
python -m mo_matching.run --dataset CIFAR10 --method LLP-MM \
  --data-root ./data --bag-type random --bag-size 32 --seed 0 \
  --output-dir ./outputs/check_mm --dry-run
```

After preparing the data, run one diagnostic update:

```bash
python -m mo_matching.run --dataset CIFAR10 --method LLP-MM \
  --data-root ./data --bag-type random --bag-size 32 --seed 0 \
  --output-dir ./outputs/smoke_mm --smoke
```

Smoke runs check the data and model setup; they are not full experiments.
FlowLLP smoke also reduces its anchor and particle settings.

View command-line options:

```bash
python -m mo_matching.run --help
python -m mo_matching.prepare_ku --help
python -m mo_matching.gan.bags --help
```

## License

[MIT](LICENSE). [Third-party notices](THIRD_PARTY.md) and dataset terms apply
separately.
