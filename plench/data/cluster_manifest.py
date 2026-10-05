"""Freeze and verify realized Cluster bags before any optimizer step."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def freeze_cluster_manifest(path, *, metadata, train_indices, val_indices,
                            train_proportions, val_proportions):
    """Atomically create once; subsequently require identical contents.

    Indices refer to original dataset rows (merged rows for miniImageNet).
    The digest covers canonical JSON plus named, shaped, little-endian arrays,
    not ZIP timestamps. It can be compared between independent machines.
    """
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    arrays = {
        "train_indices": np.asarray(train_indices, dtype="<i8"),
        "val_indices": np.asarray(val_indices, dtype="<i8"),
        "train_proportions": np.asarray(train_proportions, dtype="<f8"),
        "val_proportions": np.asarray(val_proportions, dtype="<f8"),
    }
    digest = hashlib.sha256(metadata_json.encode())
    for name, value in sorted(arrays.items()):
        digest.update(json.dumps([name, list(value.shape), value.dtype.str]).encode())
        digest.update(value.tobytes(order="C"))
    identity = digest.hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata=metadata_json, sha256=identity, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # never replace an existing manifest
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as saved:
        if saved["metadata"].item() != metadata_json or saved["sha256"].item() != identity:
            raise ValueError(f"Cluster manifest identity mismatch: {path}")
        for name, value in arrays.items():
            if not np.array_equal(saved[name], value):
                raise ValueError(f"Cluster manifest {name} mismatch: {path}")
    print(f"Cluster manifest verified: sha256={identity} file={path}", flush=True)
    return identity
