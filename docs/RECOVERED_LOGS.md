# Recovered experiment logs

This release contains sanitized numerical extracts from **13 existing original
training logs**: 12 image-table runs and one KU run. No training was performed to
produce these files. The extraction records a SHA256 of each original log and
retains only scientific configuration fields and numerical evidence. Original
machine paths and remote identities are omitted.

## Image main tables: 12 of 1,188 runs

`results/main_table_recovered_runs.json` contains CIFAR100, alpha-first bags,
FlowLLP, bag sizes 16/32/64/128, and seeds 0/1/2. Each source log has 25
checkpoint records, a configured budget of 500 epochs, a final logged epoch of
499.9791666666667, pi=10, and an observed completion marker. The final logged
test accuracy is used for this recovered slice. Unrecorded optimizer/scheduler
fields are left absent; current code defaults are not substituted for them.

`results/main_table_recovered_summary.csv` is recomputed from those 12 extracted
values. Its standard deviation uses **ddof=1** (sample standard deviation), which
matches all four manuscript cells at the reported two-decimal precision:

| Bag size | Test accuracy, mean +/- SD (%) |
| --- | --- |
| 16 | 70.90 +/- 0.24 |
| 32 | 65.85 +/- 0.46 |
| 64 | 50.37 +/- 1.17 |
| 128 | 2.79 +/- 0.53 |

This establishes numerical traceability for four of 396 main-table cells.
It does not recover the other 1,176 runs or prove a fresh code rerun reproduces
the historical values. Do not apply this slice's ddof=1 rule automatically to
other tables; the KU manuscript summary uses ddof=0.

## KU: one of 33 runs

`results/ku_recovered_run.json` retains all 100 epochs of accuracy, Macro-F1,
balanced accuracy and weighted F1 for LLP-MM order 8, seed 0. The recorded
configuration uses Adam, learning rate 0.001, weight decay 0.0005, 100 epochs,
equal weights across orders 1 through 8, CE, stable-DP and float64 moments.
Training and validation are merged, and the unknown-patient group is split
with maximum bag size 128 and assignment seed 0.

Selecting the maximum **test** Macro-F1 returns epoch 99, matching the saved
manuscript-source summary for this seed. Every selected metric comes from that
same epoch. Test-based checkpoint selection is explicit and must not be
presented as validation-based selection. The other 32 complete epoch logs were
not found locally in the inspected scope, so the three-seed paper aggregate
cannot be recomputed from this file alone.

## Search scope and exclusions

The bounded search covered the original development workspace and the older
archived PLeNCH checkout under the local project-code collection. It inventoried
JSON/JSONL, CSV/TSV, TXT and LOG filenames, then inspected relevant image and KU
results. It excluded raw data/data_dir, virtual environments, release-package
copies, third-party/upstream sources, Git internals and Python caches. No home
folder scan, image scan, remote fetch or training was performed.

The two inspected roots contained 8,011 candidate text/result files; only five
were in the older archived PLeNCH checkout. That older checkout contains a
37-group aggregate results.txt and one unrelated CIFAR10 SimCLR debug run
(50 epochs, randomized hyperparameters). It does not provide per-seed main-table
logs. Another 21-group aggregate results.txt in the development workspace also
lacks complete per-seed traceability. Their different method/protocol coverage
was not merged into the recovered numerical evidence.

The development workspace contains 30 JSONL files: the 12 qualifying FlowLLP
logs, the qualifying KU seed-0 log, six older KU diagnostics, eight CCT curves,
two Fed-ISIC2019 curves and one GAN smoke log. The six older KU diagnostics use
seed 42 and only 3, 10 or 20 epochs (including order 3 or order 6 LLP-MM); they
are excluded from the 100-epoch, seeds 0/1/2, order-8 paper comparison.

Saved KU summaries describe all 33 runs and identify remote sources, but these
are aggregate records rather than the missing complete epoch logs. The recovery
count therefore remains 1/33 for KU.
