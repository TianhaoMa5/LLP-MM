# LLP-MM

Research code for **Rethinking Learning from Label Proportions via Moment Matching**.
This branch extends the original LLP-MM release to the manuscript snapshot audited on
2026-10-05: eleven comparison methods, LLP-GAN and MM+GAN, portable launchers,
and explicitly labelled archived evidence.

**Reproduction candidate: every paper result has not yet been reproduced.**
The scope is 1,188 synthetic-image runs, 72 CIFAR-10 GAN runs, and 33 KU runs,
plus the order/runtime study. See [paper coverage](docs/PAPER_COVERAGE.md) for
the table-to-code mapping and missing evidence. Historical ABS results need
revalidation after a loss correction. The order/runtime raw data and plotting
program have not been recovered.

## Install and verify

Run commands from the repository root. Python 3.11 is tested. Full training
requires a PyTorch build appropriate for your CUDA system.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q
python -m plench.train --help
python scripts/summarize_paper_evidence.py --check
```

Installation includes `plench`, `reproduction`, and the `mo_matching` moment-loss
dependency. No private parent checkout or cluster account is required.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records validation and its limitations.
Full training needs separately obtained datasets and GPU time.

## Synthetic image tables

Prepare [datasets and cluster maps](docs/DATASETS.md), then inspect the matrix:

```bash
python scripts/reproduce_paper.py --list
python scripts/reproduce_paper.py \
  --dataset CIFAR10 --method LLP-MM --bag-mode random --bag-size 16 --seed 0 \
  --data-root ./data --output-root ./outputs/paper
```

Add `--run` to execute. All three launchers print a plan by default, execute
sequentially when requested, stop on failure, and refuse existing run directories.
Use a compute node or suitable workstation; scheduler submission is the caller's
responsibility.

[paper_protocol.json](configs/paper_protocol.json) explicitly fixes SGD, 500 epochs,
1,024 images/update, PVC's learning rate, and LLP-MM orders 8/3/3. It distinguishes
manuscript parameters from inherited implementation choices. This is a reconstructed
protocol; original source/config/result records remain incomplete.

## GAN compatibility

Generate shared CIFAR-10 manifests following [DATASETS.md](docs/DATASETS.md):

```bash
python scripts/reproduce_gan.py --method MM+GAN --bag-mode cluster \
  --bag-size 32 --seed 0 --data-root ./data --bag-root ./data/bags/cifar10
```

Add `--run` to execute. Both methods consume the same manifests. Preflight checks
dataset, seed, bag size, mode, concentration, complete population size and unique
indices. Alpha-first requires concentration 10. New manifest file hashes are
recorded. MM+GAN explicitly selects the restored historical `paper_dp` kernel.

The [72 archived records](results/gan_cifar10_runs.json) regenerate
[24 aggregate rows](results/gan_cifar10_summary.csv):

```bash
python scripts/summarize_paper_evidence.py
```

These are archived observations, not newly trained results. Metadata distinguishes
retained result fields from companion audit evidence and records the uncertain
byte scope of archived bag hashes.

## KU-Optofil PBC

```bash
python -m plench.scripts.prepare_ku_optofil_pbc --data-root ./data/ku_optofil_pbc
python scripts/reproduce_ku.py --method LLP-MM --seed 0 \
  --data-root ./data/ku_optofil_pbc --output-root ./outputs/ku
```

Add `--run` to execute; omit method/seed filters to plan all 33 runs. This fixes
100 epochs, ImageNet-pretrained standard-stem ResNet-18, Adam, five warmup epochs,
and label-independent unknown-patient splitting. Older generic KU configs remain
historical examples, not the paper recipe.

After completing all runs:

```bash
python scripts/summarize_ku.py ./outputs/ku --output ./outputs/ku_summary.csv
```

The paper selects maximum **test Macro-F1**, takes all four metrics at that epoch,
and uses population standard deviation. The collector preserves this convention,
checks the scientific settings and all 100 epochs, and rejects incomplete runs.
This is not validation-set selection. The included [KU CSV](results/ku_paper_summary.csv)
is aggregate-only; one of the 33 historical epoch logs has been recovered as a sanitized metric extract.

## Corrections affecting historical results

- EasyLLP/GeneralUPM with flooding enabled at threshold zero now apply absolute
  loss. The old helper returned the original loss.
- FlowLLP honors explicit optimizer and scheduling settings; the old implementation
  silently constructed SGD with a fixed schedule.
- The image-table launcher explicitly uses SGD instead of the generic Adam default.
- Cluster assignments are aligned with shuffled sample identities and the miniImageNet
  training subset; cluster bag sampling uses the configured seed. The legacy loader
  misaligned these indices and created an unseeded random generator.

Tests verify these behaviors. Historical numerical results still require reruns
or comparison with their original runtime sources.

## Earlier release and attribution

The earlier Fed-ISIC2019 experiment remains under
`scripts/run_fed_isic2019_seed.sh`, its existing configs and
[dataset instructions](plench/README_FED_ISIC2019.md). It is outside the audited
current manuscript. Other inherited adapters are extensions, not additional
completed paper experiments.

[THIRD_PARTY.md](THIRD_PARTY.md) records attribution and unresolved licensing
provenance. This candidate does not assign a blanket license to inherited code.

Additional source recovery: [RECOVERED_LOGS.md](docs/RECOVERED_LOGS.md) records 12/1,188 main-table run logs and 1/33 KU logs, with source hashes and independently checked aggregates.
