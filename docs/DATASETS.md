# Data preparation

Raw datasets and pretrained weights are obtained separately. Only the three
small cluster-assignment maps needed by the image experiments are bundled.

## CIFAR-10 / CIFAR-100

Download the official Python distributions:

```bash
python -c "from torchvision.datasets import CIFAR10,CIFAR100; [cls('./data', train=t, download=True) for cls in (CIFAR10,CIFAR100) for t in (True,False)]"
```

The loader reads `data/cifar-10-batches-py/` and `data/cifar-100-python/`.

## miniImageNet

Place the original cache distribution under:

```text
data/miniimagenet/mini-imagenet-cache-train.pkl
data/miniimagenet/mini-imagenet-cache-val.pkl
data/miniimagenet/mini-imagenet-cache-test.pkl
```

Each trusted pickle must contain `image_data` and `class_dict`. The experiment
loader merges train/val/test, sorts class names for IDs, and uses the first 500
and last 100 images of each contiguous 600-image block for training/testing.
This depends on the original cache ordering. A differently ordered distribution
is not the same experiment.

## Image bag construction

Select the bag type with `--bag-type`:

- `random`: shuffle the training instances and partition them into bags.
- `cluster`: sample from feature clusters with Dirichlet mixing weights of total concentration 1 and a base measure proportional to cluster sizes.
- `alphafirst`: sample class proportions with Dirichlet concentration 10 per class, then draw instances accordingly.

Use the same dataset, bag type, size, and seed for every compared method.

## KU-Optofil PBC

Download and prepare the patient metadata and images from Zenodo record 17333317:

```bash
python -m mo_matching.prepare_ku --data-root ./data/ku_optofil_pbc
```

For offline preparation, put `dataset.zip`, `metadata.csv`, and
`metadata_with_patient_level_splits.csv` in `data/ku_optofil_pbc/raw/` and append
`--no-download`. Preparation validates patient split isolation and records source
hashes.

The paper runner merges official training and validation patients, retains all
known-patient bags, and divides the missing-patient training group into balanced
subgroups using fixed shuffle seed zero and limit 128. It preserves complete test
bags. Cell labels are used for bag-proportion construction and evaluation, not
instance-level training supervision. Actual training assignments are saved per run.

## GAN comparison

Use one shared bag file for LLP-GAN and MM+GAN in each comparison. Commands
for all three bag types are listed in [README](../README.md#gan-compatibility).
