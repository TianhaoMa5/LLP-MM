# Dataset preparation

Use your own data root. Raw datasets, pretrained weights and generated bags
are excluded from Git.

## CIFAR-10 and CIFAR-100

The image loader expects torchvision's Python distributions and cluster maps:

```text
data/cifar-10-batches-py/
data/cifar-100-python/
data/cluster_maps/CIFAR10_train_K32.npz
data/cluster_maps/CIFAR100_train_K256.npz
```

Download both official train/test splits and install the retained assignments:

```bash
python -c "from torchvision.datasets import CIFAR10,CIFAR100; [cls('./data', train=train, download=True) for cls in (CIFAR10,CIFAR100) for train in (True,False)]"
mkdir -p data/cluster_maps
cp plench/data_dir/cluster_maps/CIFAR10_train_K32.npz data/cluster_maps/
cp plench/data_dir/cluster_maps/CIFAR100_train_K256.npz data/cluster_maps/
```

Generate all 36 shared CIFAR-10 GAN manifests:

```bash
for seed in 0 1 2; do
  python -m reproduction.bags.cifar10_manifest \
    --dataset cifar10 --data-dir ./data --output-dir ./data/bags/cifar10 \
    --cluster-map ./data/cluster_maps/CIFAR10_train_K32.npz \
    --modes random cluster --bag-sizes 16 32 64 128 --alpha 1 --seed "$seed"
  python -m reproduction.bags.cifar10_manifest \
    --dataset cifar10 --data-dir ./data --output-dir ./data/bags/cifar10 \
    --modes alphafirst --bag-sizes 16 32 64 128 --alpha 10 --seed "$seed"
done
```

The main-table PLeNCH loader constructs bags internally; these GAN NPZs are not
automatically injected into it. Matching parameters does not prove historical
instance-wise bag identity.

## miniImageNet

Obtain the original experiment's cache distribution:

```text
data/miniimagenet/mini-imagenet-cache-train.pkl
data/miniimagenet/mini-imagenet-cache-val.pkl
data/miniimagenet/mini-imagenet-cache-test.pkl
data/cluster_maps/miniImageNet_train_K256.npz
```

Each pickle contains `image_data` and `class_dict`. The retained loader merges
the three caches, maps class names in sorted order, and uses the first 500 and
last 100 samples of each contiguous 600-image block for train/test. It depends on
historical cache ordering; a shuffled cache is not equivalent. Recover the exact
cache hashes before claiming historical identity. Use trusted original pickles.

```bash
cp plench/data_dir/cluster_maps/miniImageNet_train_K256.npz data/cluster_maps/
```

Map SHA-256:
`5f173fc015dd749587bf4af4a6fc9f8c4d6c35c0bcfb29ff0e351ecc3d4904aa`.
This retained map contains 60,000 merged train/val/test assignments. The corrected loader projects it onto the 50,000 training images before applying the sample shuffle. Its metadata records clustering of the full merged population, including held-out images; this inherited transductive construction is not train-only clustering. The original cache ordering and historical assignments still require verification.

## KU-Optofil PBC

The preparation script downloads dataset and patient metadata from Zenodo record
`17333317`, checks split isolation and writes source hashes:

```bash
python -m plench.scripts.prepare_ku_optofil_pbc --data-root ./data/ku_optofil_pbc
```

Offline: put `dataset.zip`, `metadata.csv`, and
`metadata_with_patient_level_splits.csv` in `data/ku_optofil_pbc/raw/` and add
`--no-download`. The paper launcher merges official train/validation data,
then splits only the unknown-patient training group with cap 128/grouping seed 0.
It records `training_bag_assignments.json`. Known-patient and test bags stay intact.
Unknown subbags are synthetic groups, not recovered patient identities.
