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
copy inside `MO-Matching-FlowLLP-complete-pi10-20260724-v1` already carries
`source_indices` through shuffling, indexes the map with those source indices,
and passes a seeded Generator. Its directory name is not proof of its historical
modification date or of which code was executed on a server.

Thus, the candidate repairs concrete defects in the code prepared for release.
The effect on paper results remains a **per-run source-provenance question**.
Retrieve the runtime source revision and bag manifest before deciding which
historical experiments actually require rerunning. Existing flags requesting
historical revalidation mean that this source check is outstanding; they do not
assert that the original ABS or Cluster experiment was incorrectly implemented.
