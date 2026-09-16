#!/usr/bin/env python3
"""Resume WILDS Amazon preprocessing after the legacy CSV writer fails.

The official preprocessing can finish the expensive graph/k-core computation
and then fail while serializing a review whose text requires CSV escaping. This
script reuses the resulting ``users.txt`` and ``products.txt`` and repeats only
the second, streaming pass with all fields quoted. Parsed values and filtering
remain identical to the official implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = [
    "reviewerID",
    "asin",
    "overall",
    "reviewTime",
    "unixReviewTime",
    "reviewText",
    "summary",
    "verified",
    "category",
]


def recover(data_dir: Path, official_source: Path) -> None:
    sys.path.insert(0, str(official_source))
    import pandas as pd
    import process_amazon as official

    user_path = Path(official.user_list_path(str(data_dir)))
    product_path = Path(official.product_list_path(str(data_dir)))
    if not user_path.is_file() or not product_path.is_file():
        raise FileNotFoundError("30-core users.txt/products.txt are required")
    user_ids = set(pd.read_csv(user_path, names=["user_id"])["user_id"])
    product_ids = set(pd.read_csv(product_path, names=["product_id"])["product_id"])
    if not user_ids or not product_ids:
        raise ValueError("30-core user/product sets must be non-empty")

    duplicates_path = Path(official.reviews_with_duplicates_path(str(data_dir)))
    temporary_duplicates = duplicates_path.with_suffix(".csv.tmp")
    kept = 0
    missing_summary = 0
    with temporary_duplicates.open("w", newline="", encoding="utf-8") as output:
        # QUOTE_ALL is semantically equivalent after CSV parsing and avoids a
        # Python _csv edge case in the legacy QUOTE_NONNUMERIC writer.
        writer = csv.DictWriter(output, FIELDS, quoting=csv.QUOTE_ALL)
        for category in official.CATEGORIES:
            lengths = pd.read_csv(
                official.token_length_path(str(data_dir), category)
            )["token_counts"].to_numpy()
            seen = 0
            category_kept = 0
            for index, review in enumerate(
                official.parse(official.raw_reviews_path(str(data_dir), category))
            ):
                seen += 1
                text = review.get("reviewText")
                if not isinstance(text, str) or not text.strip():
                    continue
                if lengths[index] > 512:
                    continue
                if (
                    review["reviewerID"] not in user_ids
                    or review["asin"] not in product_ids
                ):
                    continue
                row = {
                    field: category if field == "category" else review.get(field, "")
                    for field in FIELDS
                }
                if "summary" not in review:
                    missing_summary += 1
                writer.writerow(row)
                kept += 1
                category_kept += 1
            if len(lengths) != seen:
                raise AssertionError(
                    f"token/review row mismatch for {category}: {len(lengths)} vs {seen}"
                )
            print(
                f"category={category} source_rows={seen} kept_rows={category_kept}",
                flush=True,
            )
    temporary_duplicates.replace(duplicates_path)

    frame = pd.read_csv(
        duplicates_path,
        names=FIELDS,
        dtype={
            "reviewerID": str,
            "asin": str,
            "reviewTime": str,
            "unixReviewTime": int,
            "reviewText": str,
            "summary": str,
            "verified": bool,
            "category": str,
        },
        keep_default_na=False,
        na_values=[],
    )
    frame["reviewYear"] = frame["reviewTime"].map(
        lambda value: int(value.split(",")[-1])
    )
    before_dedup = len(frame)
    frame = frame.drop_duplicates(["asin", "reviewerID", "overall", "reviewTime"])
    reviews_path = Path(official.reviews_path(str(data_dir)))
    temporary_reviews = reviews_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary_reviews, index=False, quoting=csv.QUOTE_ALL)
    temporary_reviews.replace(reviews_path)
    print(
        json.dumps(
            {
                "kcore_users": len(user_ids),
                "kcore_products": len(product_ids),
                "kept_with_duplicates": kept,
                "reviews_after_dedup": len(frame),
                "duplicates_removed": before_dedup - len(frame),
                "missing_summary": missing_summary,
                "reviews_path": str(reviews_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    args = parser.parse_args()
    recover(
        args.data_dir.expanduser().resolve(),
        args.official_source.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
