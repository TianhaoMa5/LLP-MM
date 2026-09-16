#!/usr/bin/env python3
"""Extract one frozen text embedding per processed Amazon-WILDS review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_string_hash(digest: Any, values: list[str]) -> None:
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")


def _hashing_features(texts: list[str], dimension: int) -> np.ndarray:
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    pattern = re.compile(r"[a-z0-9]+")
    for row, text in enumerate(texts):
        for token in pattern.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            matrix[row, value % dimension] += 1.0 if (value >> 63) == 0 else -1.0
        norm = float(np.linalg.norm(matrix[row]))
        if norm:
            matrix[row] /= norm
    return matrix


def extract_features(
    data_root: Path,
    encoder: str,
    batch_size: int,
    parquet_batch_size: int,
    device: str | None,
    force: bool,
) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Amazon-WILDS feature extraction requires pyarrow; install "
            "plench/requirements-amazon-wilds.txt"
        ) from exc
    instances_path = data_root / "processed" / "instances.parquet"
    if not instances_path.is_file():
        raise FileNotFoundError(
            f"missing {instances_path}; run python -m plench.scripts.prepare_amazon_wilds"
        )
    output_dir = data_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.npy"
    manifest_path = output_dir / "feature_manifest.json"
    if output_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("encoder") == encoder:
            print(f"Using existing Amazon-WILDS feature cache: {output_path}")
            return manifest
        raise FileExistsError(
            f"feature cache uses {manifest.get('encoder')!r}; pass --force to replace it"
        )

    parquet = pq.ParquetFile(instances_path)
    total = int(parquet.metadata.num_rows)
    if total <= 0:
        raise ValueError("Amazon-WILDS processed cache contains no reviews")
    diagnostic = encoder.startswith("diagnostic-hashing-")
    if diagnostic:
        try:
            feature_dimension = int(encoder.rsplit("-", 1)[1])
        except ValueError as exc:
            raise ValueError("diagnostic hashing encoder must end in its dimension") from exc
        if feature_dimension <= 0:
            raise ValueError("diagnostic hashing dimension must be positive")
        model = None
        backend = "dependency-free hashing diagnostic (not the main experiment encoder)"
        max_seq_length = None
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install plench/requirements-amazon-wilds.txt or explicitly use "
                "--encoder diagnostic-hashing-384 for a non-main smoke test"
            ) from exc
        model_name = encoder.removeprefix("sentence-transformers/")
        model = SentenceTransformer(model_name, device=device)
        feature_dimension = int(model.get_sentence_embedding_dimension())
        backend = "sentence-transformers"
        max_seq_length = int(model.max_seq_length)

    temporary = output_path.with_suffix(".npy.tmp")
    features = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(total, feature_dimension),
    )
    instance_hash = hashlib.sha256()
    bag_hash = hashlib.sha256()
    text_hash = hashlib.sha256()
    offset = 0
    for batch_number, batch in enumerate(
        parquet.iter_batches(
            batch_size=max(1, int(parquet_batch_size)),
            columns=["instance_id", "bag_id", "text", "feature_index"],
        )
    ):
        rows = batch.to_pydict()
        indices = np.asarray(rows["feature_index"], dtype=np.int64)
        expected = np.arange(offset, offset + len(indices), dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise ValueError("Amazon-WILDS feature_index is not stable/contiguous")
        texts = [str(value or "") for value in rows["text"]]
        if diagnostic:
            encoded = _hashing_features(texts, feature_dimension)
        else:
            assert model is not None
            encoded = model.encode(
                texts,
                batch_size=max(1, int(batch_size)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        if encoded.shape != (len(texts), feature_dimension):
            raise ValueError("encoder returned an invalid Amazon-WILDS feature matrix")
        if not np.isfinite(encoded).all():
            raise ValueError("encoder returned NaN or Inf Amazon-WILDS features")
        features[offset : offset + len(texts)] = encoded
        _update_string_hash(instance_hash, [str(value) for value in rows["instance_id"]])
        _update_string_hash(bag_hash, [str(value) for value in rows["bag_id"]])
        _update_string_hash(text_hash, texts)
        offset += len(texts)
        print(f"Amazon-WILDS features: {offset}/{total} reviews", flush=True)
    if offset != total:
        raise ValueError(f"feature extraction wrote {offset} rows, expected {total}")
    features.flush()
    del features
    temporary.replace(output_path)
    manifest: dict[str, object] = {
        "dataset": "amazon_wilds",
        "dataset_version": "2.1",
        "encoder": encoder,
        "backend": backend,
        "frozen": True,
        "cache": True,
        "instances": total,
        "feature_dimension": feature_dimension,
        "dtype": "float32",
        "normalized": True,
        "max_seq_length": max_seq_length,
        "instance_id_sha256": instance_hash.hexdigest(),
        "bag_id_sha256": bag_hash.hexdigest(),
        "text_sha256": text_hash.hexdigest(),
        "feature_sha256": _sha256_file(output_path),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("plench/data/amazon_wilds")
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parquet-batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract_features(
        args.data_root.expanduser().resolve(),
        args.encoder,
        args.batch_size,
        args.parquet_batch_size,
        args.device,
        args.force,
    )


if __name__ == "__main__":
    main()
