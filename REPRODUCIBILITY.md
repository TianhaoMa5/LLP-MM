# Reproducibility record

Prepared on 2026-10-05 from original LLP-MM commit
`03e3d21832d8227b77cabfb9cbe5906e5259160a` and selected updated research sources.
The [2026-09-16 record](docs/REPRODUCIBILITY_20260916.md) is historical documentation,
not validation of this source revision.

## Verification

Dataset-free tests cover moment values/gradients, order-one equivalence, variable
and singleton bags, patient grouping, ABS negative-risk gradients, FlowLLP optimizer
and schedule selection, launch plans, GAN manifest checks and KU aggregation.
An independent ordered-sampling oracle checks the restored historical `paper_dp`
kernel's values and gradients.

The GAN evidence validator checks 72 unique runs, 36 nonempty paired bag hashes,
checkpoint provenance, and 24 recomputed means/sample standard deviations. They
match the archived summaries and audited manuscript display values. This is
evidence aggregation, not fresh model training.

A fresh Python 3.11 environment was installed from this repository. Its initial
check exposed undeclared PyArrow; that dependency is now included, as are POT and
OpenCV. The final checks are recorded in
[release_validation.json](results/release_validation.json). Wheel construction
checks package inclusion. The [tested macOS environment](requirements-tested-macos-arm64.txt)
is an observation, not a Linux/CUDA lockfile. CUDA is unavailable on the test host.
The optional geospatial test skips when GeoPandas is absent.

## Changes and provenance

The historical flooding helper returned raw loss for `b <= 0`. With flooding
enabled, the corrected helper applies `abs(loss - b) + b` for `b >= 0`; non-ABS
launchers explicitly disable flooding. Negative thresholds retain their old no-op
behavior. Historical nominal ABS rows need numerical revalidation.

FlowLLP now honors explicit optimizer/schedule parameters. Omitted parameters
retain the legacy SGD/Nesterov/quarter-cosine defaults. KU explicitly requests
Adam and five warmup epochs. This fixes the executable recipe, but does not
establish the optimizer actually used by each historical run.

`plench/core/order.py` was restored byte-for-byte from the historical packaged
FlowLLP-complete snapshot. SHA-256:
`d7c02eb70190e7713c1c64ee6ddffec31b3e66f94b9347e63d5aa3e9c94c4563`.
The GAN `paper_dp` branch uses it. Equality to every remote historical source
revision has not been established. The newer float64 stable-DP LLP-MM path
cannot substitute as evidence for historical runtime measurements.

## Remaining work

See [PAPER_COVERAGE.md](docs/PAPER_COVERAGE.md) and
[paper_coverage.json](configs/paper_coverage.json):

1. Recover main-table original configurations, source identities and per-seed
   scores, including formal miniImageNet alpha-first FlowLLP evidence.
2. Recover the remaining 32 KU epoch logs and verify all metrics at the recorded selected epochs.
3. Recover order/runtime measurements, timing procedure and plot source.
4. Rerun or source-verify historical ABS and optimizer/scheduler discrepancies.
5. Validate GPU training and resolve public-release license/provenance.

No experiment is declared reproduced solely because its imports, tests,
generated commands or scheduler submissions succeeded.

[Recovered log extracts](docs/RECOVERED_LOGS.md) cover 12 of 1,188 main-table runs and one of 33 KU runs. Their original log file hashes are retained; they do not constitute new training.

## Cluster-bag alignment and determinism correction

The inherited loader shuffled image arrays before constructing cluster bags but
indexed the original cluster array with shuffled-space indices. It also created
an unseeded NumPy Generator. The miniImageNet map describes all 60,000 merged
images, while training uses the first 500 of each 600-image block. The corrected
loader projects that map onto the 50,000-image training subset, then applies the
sample permutation and uses the configured seed. These fixes change historical
cluster assignments. Historical Cluster Bag numbers require source verification
or reruns; a matching random seed in an old configuration is not sufficient.
