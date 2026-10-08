# Current experiment evidence — 2026-10-08

This supersedes the completion counts in the 2026-10-05 audit. Completed Cluster
replacements do not certify every historical Random/Alpha-first or KU experiment.

## Corrected Cluster table

The grid is 3 datasets × 4 bag sizes × 11 methods × 3 seeds = 396 runs:
330 normal 500-epoch completions and 66 verified numerical divergences, zero
missing or running. Smoke results are excluded. The original FlowLLP OOM failure
was preserved; an independently authorized recovery supplies its formal result.

`results/cluster_best_runs.json` contains normal-run compact metric histories,
scientific settings, original log hashes, bag digests and divergence provenance.
Machine paths and account details are omitted. Sixty-four first divergence
steps are exact; two are known only at the first observed nonfinite checkpoint.

The table uses each run's highest finite logged **test accuracy**, including
saved pre-divergence maxima with the failure marker retained. Mean and sample
standard deviation (`ddof=1`) use all three maxima. Final-epoch accuracy remains
separate; divergence is never a finite 500-epoch completion. Recompute every
cell using `python scripts/summarize_cluster.py --check`.

## Order figure

`figures/order_study/cluster_bag_verified.csv` contains Cluster orders 1–13 for
CIFAR10 and 1–3 for CIFAR100, at bags 16/32/64/128. Order one uses the matched PM
seed-zero run. Order thirteen uses best logged accuracy: **94.24, 91.87, 89.05,
72.21%**. Other updated points retain their final-epoch convention.

CIFAR10/bag128/order2 = **71.91%** is a user-reported additional rerun; its raw
log was not supplied. `user_rerun_overrides.json` retains that distinction.
Other bag-mode curves and runtime were recovered from the original vector
figure. Runtime is not newly measured from concurrent reruns.

## KU and GAN

The KU paper summary selects maximum test Macro-F1 and all metrics from that
epoch. Order-three/five/eight launchers support three seeds and distinct output
directories. Historical KU aggregates without complete logs remain labeled as
aggregate evidence. Selecting on test performance is not validation-only
checkpoint selection.

The 72 GAN run extracts and 24 aggregates remain checked by
`summarize_paper_evidence.py --check`; they were not newly trained during this
release review.
