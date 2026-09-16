#!/usr/bin/env python3
"""Extract and cache one frozen embedding per REF2021 submitted output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hashing_features(texts: Sequence[str], dimension: int) -> np.ndarray:
    """Dependency-free deterministic diagnostic encoder, never selected silently."""
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    token_pattern = re.compile(r"[a-z0-9]+")
    for row, text in enumerate(texts):
        tokens = token_pattern.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % dimension
            sign = 1.0 if (value >> 63) == 0 else -1.0
            matrix[row, index] += sign
        norm = float(np.linalg.norm(matrix[row]))
        if norm:
            matrix[row] /= norm
    return matrix


def extract_features(
    data_root: Path,
    encoder: str,
    batch_size: int,
    device: str | None,
    force: bool,
) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "REF2021 feature extraction requires pyarrow; install "
            "plench/requirements-ref2021.txt"
        ) from exc
    instances_path = data_root / "processed" / "instances.parquet"
    if not instances_path.is_file():
        raise FileNotFoundError(
            f"missing {instances_path}; run python -m plench.scripts.prepare_ref2021 first"
        )
    output_dir = data_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.npy"
    manifest_path = output_dir / "feature_manifest.json"
    if output_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("encoder") == encoder:
            print(f"Using existing REF2021 feature cache: {output_path}")
            return manifest
        raise FileExistsError(
            f"feature cache uses {manifest.get('encoder')!r}, requested {encoder!r}; pass --force"
        )
    table = pq.read_table(instances_path, columns=["instance_id", "bag_id", "text", "feature_index"])
    feature_index = np.asarray(table["feature_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(feature_index, np.arange(len(table))):
        raise ValueError("instances.parquet feature_index is not stable/contiguous")
    instance_ids = [str(value) for value in table["instance_id"].to_pylist()]
    bag_ids = [str(value) for value in table["bag_id"].to_pylist()]
    texts = [str(value or "") for value in table["text"].to_pylist()]
    if any(not text.strip() for text in texts):
        raise ValueError("REF2021 contains an empty constructed instance text")

    if encoder.startswith("diagnostic-hashing-"):
        try:
            dimension = int(encoder.rsplit("-", 1)[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("diagnostic hashing encoder must end in its dimension") from exc
        if dimension <= 0:
            raise ValueError("diagnostic hashing dimension must be positive")
        features = _hashing_features(texts, dimension)
        backend = "dependency-free hashing diagnostic (not the main experiment encoder)"
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "The main REF2021 encoder requires sentence-transformers. Install "
                "plench/requirements-ref2021.txt, or explicitly use "
                "--encoder diagnostic-hashing-384 for a non-main smoke test."
            ) from exc
        model_name = encoder.removeprefix("sentence-transformers/")
        model = SentenceTransformer(model_name, device=device)
        features = model.encode(
            texts,
            batch_size=int(batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        backend = "sentence-transformers"
    if features.ndim != 2 or features.shape[0] != len(texts):
        raise ValueError("encoder returned an invalid REF2021 feature matrix")
    if not np.isfinite(features).all():
        raise ValueError("encoder returned NaN or Inf REF2021 features")
    temporary = output_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, features.astype(np.float32, copy=False))
    temporary.replace(output_path)
    manifest: dict[str, object] = {
        "dataset": "ref2021_uoa11",
        "encoder": encoder,
        "backend": backend,
        "frozen": True,
        "cache": True,
        "instances": len(texts),
        "feature_dimension": int(features.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "instance_id_sha256": _sha256_strings(instance_ids),
        "bag_id_sha256": _sha256_strings(bag_ids),
        "text_sha256": _sha256_strings(texts),
        "feature_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ref2021_uoa11")
    parser.add_argument("--data-root", type=Path, default=Path("plench/data/ref2021_uoa11"))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if str(args.dataset).lower().replace("-", "_") not in {
        "ref2021_uoa11",
        "ref2021uoa11",
    }:
        raise ValueError("this extractor only supports ref2021_uoa11")
    extract_features(
        args.data_root.expanduser().resolve(),
        args.encoder,
        args.batch_size,
        args.device,
        args.force,
    )


if __name__ == "__main__":
    main()
