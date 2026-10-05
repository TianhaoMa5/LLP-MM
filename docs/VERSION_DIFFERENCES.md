# Version differences clarified after the release audit

The user's original usage of ABS is correct: enabling flooding with threshold
zero produces absolute loss in the inspected experiment-source copy and packaged
FlowLLP-complete source. Both directly return `(loss - b).abs() + b`.
The initial LLP-MM publication commit `03e3d21832d8227b77cabfb9cbe5906e5259160a`
instead has an early return for `b <= 0`. The candidate fixes that release-copy
behavior. This finding does not establish a defect in every historical ABS run.

Cluster implementations also differ. The inspected experiment-source copy and
initial LLP-MM release shuffle image rows but read the original cluster map using
shuffled-row positions, and construct an unseeded Generator. However, the current
extracted directory `MO-Matching-FlowLLP-complete-pi10-20260724-v1` already carries
`source_indices` through shuffling, indexes the map with those source indices,
and passes a seeded Generator. Its directory name is not proof of its historical
modification date or of which code was executed on a server.

The same-named ZIP was subsequently inspected separately: its
`plench/data/LLP_load.py` still has the unseeded Generator and
`clusters_all[local2global]` lookup. The ZIP and extracted directory are therefore
different source snapshots, despite sharing a name. Their loader SHA-256 values
are `0f90f1406c7664427019c42f59363542d0694fc97bbd26319cd91763a5efd771`
(ZIP member) and
`ffcac093004e68d668e36f854c5bf0dc5b8e4b18b8f668add128195f5d468a7f`
(extracted directory). Neither snapshot alone establishes the runtime version
used for a particular paper result.

## Cluster impact and resolution

The faulty lookup changes the cluster identities used to construct bags. Images
and class labels still undergo the same permutation, and bag proportions are
computed from the labels of the selected images. Thus this is a bag-construction
protocol defect, not evidence that image class labels were swapped. Its numerical
effect and direction have not been measured. If methods used different source
versions or different realized bags, their comparability also requires checking.

The lookup defect is specific to the affected Cluster loader. It does not by
itself invalidate Random, alpha-first, KU patient grouping, or the separate GAN
canonical-manifest path. Those paths have their own provenance requirements.

For historical Cluster runs, first recover the executed loader revision and,
where available, the realized bag manifest. Correctly aligned runs do not need
rerunning because of this defect. Confirmed affected comparisons should be
rerun using one frozen manifest per dataset/bag-size/seed shared across methods;
rerun all methods being compared in each affected condition, not just LLP-MM.
Preserve both old and corrected results, and update the corresponding paper
cells and Cluster ablations if the evidence changes. If source provenance is
unrecoverable, a controlled replacement run is needed to support a verified
reproduction claim. A pilot diagnoses behavior but does not validate an entire
paper grid. Seeding the corrected implementation does not reconstruct old bags
created with an unseeded Generator.

## Corrected Cluster rerun protocol

The image-table rerun explicitly selects `moment_implementation=paper_image` for
LLP-MM. This dispatches to the supplementary archive's restored
`LLPHighOrderLoss` implementation, with CE and uniform weights over orders
1 through 8 for CIFAR-10 and 1 through 3 for CIFAR-100/miniImageNet. The separate
variable-bag implementation remains available for natural-bag studies. The
dispatch has a loss-and-gradient regression check against the supplementary
criterion. This distinction matters for runtime and historical source fidelity.

`scripts/cluster_rerun.py` prepares and verifies frozen bag manifests before
training. Manifests contain original image-row indices and class proportions,
and their canonical content digest can be checked across machines. A mismatch
aborts training. The rerun writes source hashes, package versions, full commands,
GPU information and a validated completion record. Short smoke outputs are
stored separately and are never counted as 500-epoch results.

Thus, the candidate repairs concrete defects in the code prepared for release.
The effect on paper results remains a **per-run source-provenance question**.
Retrieve the runtime source revision and bag manifest before deciding which
historical experiments actually require rerunning. Existing flags requesting
historical revalidation mean that this source check is outstanding; they do not
assert that the original ABS or Cluster experiment was incorrectly implemented.
