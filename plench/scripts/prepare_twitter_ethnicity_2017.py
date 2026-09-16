"""Validate historical Twitter Race3 files and create PLeNCH's manifest contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plench.data.twitter_ethnicity_2017 import (
    CLASS_NAMES,
    PROPORTION_COLUMNS,
    parse_race3_label,
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"DATA_MISSING: {path}")
    frame = pd.read_csv(path, dtype={"instance_id": str, "twitter_user_id": str, "county_fips": str})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _canonical_user_id(frame: pd.DataFrame, source: str) -> pd.Series:
    if "instance_id" in frame.columns:
        result = frame["instance_id"].astype(str).str.strip()
    elif "twitter_user_id" in frame.columns:
        result = frame["twitter_user_id"].astype(str).str.strip()
    else:
        raise ValueError(f"{source} needs instance_id or twitter_user_id")
    if (result == "").any() or result.duplicated().any():
        raise ValueError(f"{source} identifiers must be non-empty and unique")
    return result


def prepare(
    dataset_root: Path,
    *,
    normalize_race3_proportions: bool,
    overwrite: bool,
) -> Path:
    root = dataset_root.expanduser().resolve()
    raw = root / "raw"
    processed = root / "processed"
    manifest_path = processed / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild it")

    users = _read_csv(
        raw / "users.csv",
        {"county_fips", "image_path"},
    )
    users["instance_id"] = _canonical_user_id(users, "users.csv")
    users["county_fips"] = users["county_fips"].fillna("").astype(str).str.strip()
    if (users["county_fips"] == "").any():
        raise ValueError("users.csv: every training user needs county_fips")
    if "label" in users.columns and users["label"].notna().any():
        raise ValueError("users.csv must not contain instance labels")

    census = _read_csv(
        raw / "census_county_proportions.csv",
        {"county_fips", *PROPORTION_COLUMNS},
    )
    census["county_fips"] = census["county_fips"].astype(str).str.strip()
    if census["county_fips"].duplicated().any():
        raise ValueError("census_county_proportions.csv contains duplicate county_fips")
    raw_proportions = census.loc[:, PROPORTION_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_proportions).all() or (raw_proportions < 0).any():
        raise ValueError("Census Race3 proportions must be finite and non-negative")
    raw_sums = raw_proportions.sum(axis=1)
    if (raw_sums <= 0).any():
        raise ValueError("every county needs positive Race3 mass")
    if normalize_race3_proportions:
        target_proportions = raw_proportions / raw_sums[:, None]
    else:
        target_proportions = raw_proportions.copy()
        if not np.allclose(raw_sums, 1.0, atol=1e-6, rtol=0):
            examples = census.loc[~np.isclose(raw_sums, 1.0, atol=1e-6), "county_fips"].head(5).tolist()
            raise ValueError(
                "Race3 Census values do not sum to one for all counties. "
                "Pass --normalize-race3-proportions to opt in explicitly; "
                f"example counties: {examples}"
            )
    available_counties = set(census["county_fips"])
    missing_counties = sorted(set(users["county_fips"]).difference(available_counties))
    if missing_counties:
        raise ValueError(f"users.csv references counties absent from Census data: {missing_counties[:10]}")

    train = pd.DataFrame(
        {
            "instance_id": users["instance_id"],
            "twitter_user_id": users.get("twitter_user_id", pd.Series([""] * len(users))),
            "county_fips": users["county_fips"],
            "image_path": users["image_path"].astype(str),
            "text_path": users.get("text_path", pd.Series([""] * len(users))).fillna("").astype(str),
            "split": "train",
            "label": "",
        }
    )

    evaluation_path = raw / "evaluation_users.csv"
    evaluation = pd.DataFrame(columns=train.columns)
    if evaluation_path.is_file():
        raw_evaluation = _read_csv(evaluation_path, {"image_path", "label"})
        raw_evaluation["instance_id"] = _canonical_user_id(raw_evaluation, "evaluation_users.csv")
        parsed_labels = [
            parse_race3_label(value, allow_missing=False) for value in raw_evaluation["label"]
        ]
        evaluation = pd.DataFrame(
            {
                "instance_id": raw_evaluation["instance_id"],
                "twitter_user_id": raw_evaluation.get(
                    "twitter_user_id", pd.Series([""] * len(raw_evaluation))
                ),
                "county_fips": raw_evaluation.get(
                    "county_fips", pd.Series([""] * len(raw_evaluation))
                ).fillna("").astype(str),
                "image_path": raw_evaluation["image_path"].astype(str),
                "text_path": raw_evaluation.get(
                    "text_path", pd.Series([""] * len(raw_evaluation))
                ).fillna("").astype(str),
                "split": "evaluation",
                "label": parsed_labels,
            }
        )
        identifier_overlap = sorted(set(train["instance_id"]).intersection(evaluation["instance_id"]))
        if identifier_overlap:
            raise ValueError(f"training/evaluation instance overlap: {identifier_overlap[:10]}")
        train_twitter_ids = set(train["twitter_user_id"].dropna().astype(str)) - {""}
        eval_twitter_ids = set(evaluation["twitter_user_id"].dropna().astype(str)) - {""}
        twitter_overlap = sorted(train_twitter_ids.intersection(eval_twitter_ids))
        if twitter_overlap:
            raise ValueError(f"training/evaluation Twitter user overlap: {twitter_overlap[:10]}")

    manifest = pd.concat([train, evaluation], ignore_index=True)
    county_output = census[["county_fips"]].copy()
    for index, column in enumerate(PROPORTION_COLUMNS):
        county_output[column] = target_proportions[:, index]
        county_output[f"raw_{column}"] = raw_proportions[:, index]
    county_output["raw_race3_sum"] = raw_sums

    processed.mkdir(parents=True, exist_ok=True)
    _atomic_csv(manifest, manifest_path)
    _atomic_csv(county_output, processed / "county_proportions.csv")
    _atomic_json(
        {
            "dataset": "TwitterEthnicity2017",
            "task": "race3",
            "class_to_index": {name: index for index, name in enumerate(CLASS_NAMES)},
            "modality": "image",
            "default_representation": "precomputed_feature",
            "feature_type": "xception_imagenet_global_average_pool",
            "feature_dim": 2048,
            "normalize_race3_proportions": bool(normalize_race3_proportions),
            "normalization_logged": True,
            "training_instances": int(len(train)),
            "evaluation_instances": int(len(evaluation)),
            "evaluation_available": bool(len(evaluation)),
            "historical_author_data_required": True,
            "source_authenticity_verified_by_preparer": False,
            "name_bags_enabled": False,
            "search_bags_enabled": False,
        },
        processed / "metadata.json",
    )
    print(f"Wrote {manifest_path} ({len(train)} train, {len(evaluation)} evaluation users)")
    if not len(evaluation):
        print(
            "DATA_MISSING: raw/evaluation_users.csv is absent. Training preparation succeeded, "
            "but instance-level Twitter evaluation is unavailable."
        )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--normalize-race3-proportions",
        action="store_true",
        help="explicitly renormalize White/Black/Hispanic Census mass to sum to one",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.data_root,
        normalize_race3_proportions=args.normalize_race3_proportions,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
