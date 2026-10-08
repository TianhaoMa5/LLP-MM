# Paper coverage and evidence

> Historical 2026-10-05 audit. See [CURRENT_RESULTS.md](CURRENT_RESULTS.md) for the completed corrected Cluster grid and updated order figure; its completion counts supersede the earlier Cluster gaps below. Other historical provenance limitations remain.

**This release does not yet establish complete reproduction of every reported experiment.** It contains implementations and partial archived evidence. Recomputing an archived mean is different from reproducing training on a clean machine. The unresolved items below must remain visible when describing the release.

Audit date: 2026-10-05. The audited paper is *Rethinking Learning from Label Proportions via Moment Matching*, using the `gan-maintext-20261005/updated` source snapshot and its `main.tex` entry point. The original workspace-root manuscript was older: in particular, it used CIFAR-10 order six and omitted the later GAN study. The current paper uses **order eight** on CIFAR-10. This audit distributes no manuscript TeX or PDF files.

The machine-readable inventory is [paper_coverage.json](../configs/paper_coverage.json). It records source-file SHA-256 values, manuscript locations, methods, units, coverage counts, and limitations without machine-specific paths. Its snapshot includes the verified online version plus the subsequent caption and GAN implementation-detail edits. The later edits did **not** remove the order/runtime figure. The alternative `ICLRVersion` branch excludes GAN compatibility; the `main.tex` scope below includes it.

## What must be reproduced

| Paper item | Source location in audited snapshot | Scope | Evidence in this release |
| --- | --- | --- | --- |
| Random + Cluster table | `sections/experiments.tex:1–102`, `tab:cifar_random_cluster` | 3 datasets × 2 bag modes × 4 sizes × 11 methods = 264 conditions; 792 runs at three runs per condition | **RED:** complete raw-run provenance not bundled |
| Alpha-first table | `sections/experiments.tex:104–156`, `tab:cifar_alphafirst` | 3 datasets × 4 sizes × 11 methods = 132 conditions; 396 runs | **RED:** complete raw-run provenance not bundled |
| CIFAR-10 GAN compatibility | `sections/gan_compatibility.tex:1–26`, `tab:gan-compatibility` | 3 bag modes × 4 sizes × 2 methods × 3 seeds = 72 runs | Per-seed archive extract; means and sample standard deviations independently recomputed |
| KU-Optofil PBC | `sections/fed_isic2019.tex:7–38`, `tab:ku-optofil-semi-real` | 11 methods × 3 runs; 4 metrics, 44 reported cells | **RED:** aggregate-only extract; no reconstruction of per-seed scores |
| Order and runtime figure | `sections/experiments.tex:203–223`, `fig:order-effect-runtime-allbags` | CIFAR-10 orders 1–13 and CIFAR-100 orders 1–3, four bag sizes, three bag modes; 192 accuracy conditions | **RED — MISSING:** raw measurements, seeds, timing procedure, and plotting recipe |
| Bag construction illustration | `sections/introduction.tex:13–14`, `fig:bag_con` | Conceptual illustration | Not an experiment; image not redistributed |

The three-run table matrices total **1,293 runs** before the order sweep: 1,188 synthetic main-table runs, 72 GAN runs, and 33 KU runs. This is a target inventory, **not a completed-run claim**. Order-sweep seed count is unknown, and its overlap with other configurations is not counted again here. The appendix has theoretical proofs and implementation settings, but no additional experimental tables.

## Method-to-code mapping

| Paper method | Training algorithm or entry point | Required distinction |
| --- | --- | --- |
| PM | `plench.core.algorithms.PM` | First-order proportion matching |
| DSQ | `plench.core.algorithms.LLP_DSQ` | MSE surrogate |
| LLP-PVC | `plench.core.algorithms.LLP_PVC` | Synthetic image learning rate 0.005 |
| LLP-FC | `plench.core.algorithms.LLP_FC` | Approximate prior mode, correction weight 1, entropy weight 0 |
| ROT | `plench.core.algorithms.ROT` | Keep frozen protocol parameters |
| EasyLLP | `plench.core.algorithms.EasyLLP` | Explicit `flooding=false` |
| EasyLLP-ABS | Same class | `flooding=true`, `flooding_b=0`; historical values require revalidation |
| GeneralUPM | `plench.core.algorithms.GeneralUPM` | Explicit `flooding=false` |
| GeneralUPM-ABS | Same class | `flooding=true`, `flooding_b=0`; historical values require revalidation |
| FlowLLP | `plench/core/flowllp_algorithm.py` | 50-dimensional latent space, 1,000 anchors/class, anchor loss 0.1, 3,000 particle steps |
| LLP-MM | `plench.core.algorithms.LLP_MM`, `plench/core/order.py`, `plench/data/ref2021.py` | Preserve the configured historical image kernel versus variable-bag stable DP; these are not interchangeable runtime evidence |
| LLP-GAN | `train_llpgan_cifar.py` / `reproduction/llp_gan_pytorch/train.py` | Moment order 1 |
| MM+GAN | Same entry point | CIFAR-10 order 8; archived full configs specify `paper_dp` |

Shared backbones are in `plench/core/networks.py`; KU grouping is in `plench/data/ku_optofil_pbc.py`. The variable-bag factorial-moment implementation also uses `src/mo_matching/llp/multiclass.py`. Canonical CIFAR bag generation is in `reproduction/bags/cifar10_manifest.py`.

The historical GAN `paper_dp` import was absent from the audited workspace's active `plench/core`, although an older packaged copy existed. Restoring a kernel and passing correctness checks does not establish that every historical run used the same source revision or runtime path. Preserve implementation provenance when rerunning or comparing speed.

## Frozen scientific settings

**Synthetic image tables.** CIFAR-10/100 use modified-stem ResNet-18; miniImageNet uses modified-stem ResNet-34. Train for 500 epochs with 1,024 images per update, SGD momentum 0.9 and weight decay 0.0005. Learning rate is 0.005 for PVC and 0.05 for the other methods, with linear warmup followed by cosine decay. LLP-MM uses cross-entropy, equal weights across orders 1 through `s`, and `s=8/3/3` for CIFAR-10/CIFAR-100/miniImageNet. DSQ uses MSE. See `sections/appendix.tex:1355–1367`.

Bag sizes are 16, 32, 64, and 128. The inherited common protocol uses alpha-first concentration 10, cluster concentration 1, trial seeds 0/1/2, and warmup fraction 0.08. These details come from earlier code/protocol artifacts and are not all specified explicitly in the latest manuscript. The main tables say standard deviation without specifying the denominator; recover the original aggregation rule before claiming numerical identity.

**GAN comparison.** The two methods share bags, modified-stem ResNet-18 discriminator, generator, 500 epochs, and 1,024 real images per update. The generator uses Adam at 0.0003, betas `(0.5, 0.999)`, and 100-dimensional noise. Bag and adversarial weights are both 1; MM+GAN uses eight equally weighted CE moment terms. The archived full configs additionally record five warmup epochs, no AMP, and gradient clipping at 5. Selection is final epoch 500; report sample standard deviation (`ddof=1`). See `sections/appendix.tex:1371–1373`.

**KU-Optofil PBC.** The official training and validation partitions are merged: 25,719 training cells and 5,765 test cells, 13 classes. Known patients form 219 training bags; 3,204 cells with unknown patient IDs are shuffled independently of labels into 26 bags of 123–124 cells. There are 245 training and 56 test bags. The model is standard-stem ImageNet-pretrained ResNet-18 with 224-pixel inputs, ImageNet normalization, and horizontal/vertical flips. Adam uses learning rate 0.001 for 100 epochs and five warmup epochs. LLP-MM uses order eight and float64 stable DP; short bags retain feasible orders and renormalize weights. FlowLLP uses 50 pretraining and 50 joint-training epochs. The archive reports five bags/update for GeneralUPM and its ABS variant, four for the others, and forward chunks of 8 or 32. See `sections/appendix.tex:1379–1389`.

**KU checkpoint selection uses the highest test Macro-F1 in each run**, and all four metrics come from that same checkpoint. Report population standard deviation (`ddof=0`). This reproduces the reported protocol; it is not validation selection and is not an unbiased held-out estimate. Do not silently change selection rules and call the resulting scores the same experiment. The existing generic KU examples with 200 epochs or seed 42 are not the frozen paper recipe.

## Included evidence and verification

[gan_cifar10_runs.json](../results/gan_cifar10_runs.json) has exactly 72 unique method/mode/size/seed records. Scientific config fields are copied only when observed; missing fields are not inferred from defaults. Machine-specific data/output/source paths are removed. The original complete artifact's SHA-256 and source-record positions remain available as provenance.

Sixty rows retain direct `result.json` selection fields. The remaining twelve MM+GAN Cluster rows have abbreviated configs and no retained result object. Their epoch 500 and final-epoch selection are supported by a separately hashed companion audit statement, recorded as `epoch_evidence_kind=companion_audit_statement`. They are not presented as direct per-row result metadata.

All 36 GAN/MM pairs have nonempty, identical archived bag hashes. The original hash computation's exact byte scope was not retained and local NPZ files were unavailable for a byte-level check. **These are expected archived identifiers, not a guarantee that rerunning NPZ serialization produces the same file bytes.** An empty hash must never count as a verified match.

Recompute or verify the 24 method/configuration aggregate rows:

```bash
python scripts/summarize_paper_evidence.py
python scripts/summarize_paper_evidence.py --check
```

The script checks the exact 72-run grid, seeds 0/1/2, uniqueness, valid percentages, nonempty paired hashes, checkpoint provenance, and agreement with the archived aggregates. It computes means and sample standard deviations from the per-seed accuracy values and writes [gan_cifar10_summary.csv](../results/gan_cifar10_summary.csv). It does not train a model.

[ku_paper_summary.csv](../results/ku_paper_summary.csv) is explicitly **aggregate-only**. It transcribes the eleven rows from the hashed `macro-selected-sources.json` artifact and retains the reported selected epochs in seed 0/1/2 order. The mean/std values have not been recomputed from all 33 raw logs, and no per-seed scores are invented. One complete LLP-MM seed-0 log was subsequently recovered; its maximum test Macro-F1 is at epoch 99. See [RECOVERED_LOGS.md](RECOVERED_LOGS.md). Its ABS rows are marked as requiring historical revalidation.

## Version-specific qualification

The original experiment ABS implementation and a packaged Cluster implementation already contain the intended behavior. Defects found in the initial LLP-MM release must not be generalized to all historical runs. See [VERSION_DIFFERENCES.md](VERSION_DIFFERENCES.md).

## Open reproduction blockers

- **RED — Cluster index/seed defects.** The inherited loader did not map shuffled image indices back to the original cluster array, and initialized the cluster Generator without a seed. The miniImageNet map is for 60,000 merged images while its train split contains 50,000. The corrected release aligns these index spaces and seeds sampling; historical Cluster rows need original-runtime verification or reruns. The retained miniImageNet map was fitted on the full merged population, not training images only.

- **RED — Release-copy ABS behavior.** The initial LLP-MM release flooding helper returned raw loss when `b=0`; nominal ABS runs therefore did not necessarily apply absolute value. Correcting the release code does not repair the historical numbers. EasyLLP-ABS and GeneralUPM-ABS in synthetic and KU tables need reruns or source-level revalidation before their numerical reproducibility can be claimed.
- **RED — Main-table provenance.** Recover all relevant raw scores, complete configs, source revisions, seed definitions, checkpoint selection rules, and aggregation conventions. Twelve CIFAR-100 alpha-first FlowLLP run logs were recovered and their four aggregate cells match the manuscript; 1,176 main-table run logs remain unrecovered. See [RECOVERED_LOGS.md](RECOVERED_LOGS.md).
- **RED — KU raw logs.** Recover the remaining 32 epoch logs (LLP-MM seed 0 has been recovered) and verify the recorded test-selected epochs and all four metrics at each selected checkpoint. The present CSV is only an aggregate reference.
- **RED — Order/runtime evidence.** Recover the measurement table, original plot script, seeds, warmup/timing method, and exact implementation. The paper reports an NVIDIA A100-SXM4-40GB and AMD EPYC 7742 CPU. The current figure cannot be regenerated from this release; do not extract plotted pixels and describe them as measured values.
- **RED — FlowLLP miniImageNet alpha-first.** The checked older local summary covers 32/36 FlowLLP configurations and lacks four formal miniImageNet alpha-first aggregates. Old `pi=1` alpha-first runs do not satisfy the `pi=10` protocol and cannot fill the gap.
- **AMBER — Historical bag identity.** The latest manuscript explicitly states that archived CIFAR-100 and miniImageNet outputs lack bag-manifest hashes (`sections/experiments.tex:177`). Matching construction parameters does not prove instance-wise identical bags.
- **AMBER — Clean-environment execution.** Artifact checks and smoke tests are not the full GPU experiment matrix. Record future reruns separately from archived values, with source/config/data hashes and realized seed/selection metadata.

## Scope boundary

CIFAR-100 GAN compatibility, LLP-Gaussian, Fed-ISIC2019, CCT, UNSW-NB15, MiniCriteo, text sweeps, CV/LEM, AI4Arctic, Sentinel2, REF2021, Amazon/MovieLens/ESCI, and Huebner2017 are not experimental results in this audited manuscript. Existing implementations or outputs for these studies must be labelled extensions or exploratory work. The historical `fed_isic2019.tex` filename now contains the **KU-Optofil PBC** table; its filename is not evidence for a Fed-ISIC paper experiment.

中文说明：当前只核验了部分历史证据和实现，尚不能宣称论文全部实验已完整复现；红色缺口必须保留。
