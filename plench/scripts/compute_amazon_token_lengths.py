#!/usr/bin/env python3
"""Compute WILDS Amazon BERT token lengths with ordered worker processes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Iterator


_TOKENIZER = None


def _init_worker(model_name: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import BertTokenizerFast

    _TOKENIZER = BertTokenizerFast.from_pretrained(model_name)


def _tokenize(texts: list[str]) -> list[int]:
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    tokens = _TOKENIZER(
        texts,
        padding="do_not_pad",
        truncation="do_not_truncate",
        return_token_type_ids=False,
        return_attention_mask=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=True,
    )
    return list(tokens["length"])


def _text_batches(path: Path, batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    with gzip.open(path, "rb") as source:
        for line in source:
            review = json.loads(line)
            text = review.get("reviewText", "")
            if not isinstance(text, str) or not text.strip():
                text = ""
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def compute(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str,
    workers: int,
    batch_size: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    context = mp.get_context("spawn")
    count = 0
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(model_name,),
    ) as pool, temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["token_counts"])
        results: Iterable[list[int]] = pool.imap(
            _tokenize,
            _text_batches(input_path, batch_size),
            chunksize=1,
        )
        for batch_index, lengths in enumerate(results, start=1):
            writer.writerows((length,) for length in lengths)
            count += len(lengths)
            if batch_index % 1000 == 0:
                print(f"processed_reviews={count}", flush=True)
    temporary.replace(output_path)
    print(f"wrote {count} token lengths to {output_path}", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    compute(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        model_name=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
