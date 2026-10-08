# LLP-MM

Research code for **Rethinking Learning from Label Proportions via Moment Matching**.
LLP-MM, eleven comparison methods, LLP-GAN/MM+GAN, portable experiment launchers,
and checked result extracts are included.

## Install

Python 3.11 is tested. Full training requires a suitable CUDA PyTorch build.

```bash
git clone https://github.com/TianhaoMa5/LLP-MM.git
cd LLP-MM
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q
```

No private parent checkout is needed. See [dataset preparation](docs/DATASETS.md).
Raw images and pretrained checkpoints are acquired separately.

## Checked results

The corrected Cluster grid is complete: **396 configurations, 330 normal
500-epoch completions, 66 verified numerical divergences, zero missing**.
The published table uses each run's maximum finite logged test accuracy,
including saved maxima before divergence. Divergence remains explicitly recorded.

```bash
python scripts/summarize_cluster.py --check
python scripts/summarize_cluster.py
python scripts/summarize_paper_evidence.py --check
```

- [Cluster per-run results and metric histories](results/cluster_best_runs.json)
- [Cluster table: mean and sample standard deviation](results/cluster_best_summary.csv)
- [Completion counts](results/cluster_completion.json)
- [GAN per-seed results](results/gan_cifar10_runs.json)
- [KU paper summary](results/ku_paper_summary.csv)
- [Order-study inputs and plotting program](figures/order_study)

Cluster selects peak **test accuracy**; KU selects peak **test Macro-F1**.
These preserve the paper convention, rather than validation-only selection.
The order figure retains the original runtime curve, without treating new
concurrent measurements as comparable single-GPU timings.
See [current evidence](docs/CURRENT_RESULTS.md).

## Image experiments

```bash
# List the 1,188-run image matrix without training.
python scripts/reproduce_paper.py --list

# Print one training command; add --run to execute.
python scripts/reproduce_paper.py --dataset CIFAR10 --method LLP-MM \
  --bag-mode random --bag-size 16 --seed 0 \
  --data-root ./data --output-root ./outputs/paper
```

[paper_protocol.json](configs/paper_protocol.json) fixes 500 epochs, 1,024
images/update, optimizers, backbones and MM orders 8/3/3. Fixed-image runs use
`paper_image`; natural KU bags use float64 stable DP. Do not interchange these
kernels when comparing historical timings.

Prepare frozen Cluster manifests on an allocated compute node. All methods in a
condition share the same manifest:

```bash
python scripts/cluster_rerun.py --prepare CIFAR10 \
  --data-root ./data --output-root ./outputs/cluster
python scripts/cluster_rerun.py --list \
  --data-root ./data --output-root ./outputs/cluster
python scripts/cluster_rerun.py --index 30 --smoke \
  --data-root ./data --output-root ./outputs/cluster
python scripts/cluster_rerun.py --index 30 \
  --data-root ./data --output-root ./outputs/cluster
```

Repeat preparation for CIFAR100 and miniImageNet. The runner verifies manifest
content, records source/environment hashes, refuses existing outputs and stops
on nonfinite loss. Divergence and infrastructure failures remain distinct.
Smoke outputs are stored separately.

For an order ablation, use an isolated output directory:

```bash
python scripts/reproduce_paper.py --dataset CIFAR10 --method LLP-MM \
  --bag-mode cluster --bag-size 16 32 64 128 --seed 0 --order 13
python -m pip install -e '.[plot]'
python figures/order_study/plot.py
```

The figure uses PM for order one. The bag128/order2 value 71.91% is explicitly
recorded as a user-reported rerun without supplied raw evidence.

## GAN and KU

```bash
python scripts/reproduce_gan.py --method MM+GAN --bag-mode cluster \
  --bag-size 32 --seed 0 --data-root ./data --bag-root ./data/bags/cifar10
python -m plench.scripts.prepare_ku_optofil_pbc --data-root ./data/ku_optofil_pbc
python scripts/reproduce_ku.py --method LLP-MM --seed 0

# Omitting --seed plans all three seeds. Default MM order is eight.
python scripts/reproduce_ku.py --method LLP-MM --order 3
python scripts/reproduce_ku.py --method LLP-MM --order 5
```

Commands plan by default; add `--run` to train. GAN methods share canonical NPZ
bags. KU uses 100 epochs and label-independent unknown-patient grouping.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records validation;
[VERSION_DIFFERENCES.md](docs/VERSION_DIFFERENCES.md) explains historical fixes.
Other inherited adapters are extensions, not additional completed paper runs.

## License

Original project code and the authors' changes use the [MIT License](LICENSE).
Third-party components retain their own licenses and notices; see
[THIRD_PARTY.md](THIRD_PARTY.md). Dataset licenses are separate.
