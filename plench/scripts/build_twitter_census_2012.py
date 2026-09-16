"""Build the county Race3 priors used by the Twitter LLP benchmark.

The 2017 paper refers to the county construction in Mohammady and Culotta
(2014).  That paper identifies the exact Census vintage and demographic
categories.  This script downloads the official 2012 county estimates (or
reads a previously downloaded copy) and writes PLeNCH's raw Census contract.

It does not download, infer, or fabricate any Twitter users.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd


CENSUS_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2010-2012/"
    "counties/asrh/cc-est2012-alldata.csv"
)
SOURCE_YEAR = 5  # July 1, 2012 estimate in CC-EST2012-ALLDATA.
TOTAL_AGE_GROUP = 0
SOURCE_COLUMNS = {
    "white": ("NHWAC_MALE", "NHWAC_FEMALE"),
    "black": ("NHBAC_MALE", "NHBAC_FEMALE"),
    "hispanic": ("H_MALE", "H_FEMALE"),
}


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


def download_census_csv(destination: Path, *, overwrite: bool = False) -> Path:
    """Download the official 2012 all-state county characteristics CSV."""

    destination = destination.expanduser().resolve()
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.download")
    request = urllib.request.Request(
        CENSUS_URL,
        headers={"User-Agent": "PLeNCH-TwitterEthnicity2017/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Census download returned an empty file: {CENSUS_URL}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _required_source_columns() -> set[str]:
    demographic = {column for pair in SOURCE_COLUMNS.values() for column in pair}
    return {"SUMLEV", "STATE", "COUNTY", "YEAR", "AGEGRP", "TOT_POP", *demographic}


def build_race3_proportions(
    census: pd.DataFrame,
    *,
    county_fips: set[str] | None = None,
) -> pd.DataFrame:
    """Convert CC-EST2012-ALLDATA rows to raw White/Black/Hispanic shares.

    White and Black follow the categories stated in the cited 2014 paper:
    non-Hispanic and alone-or-in-combination. Hispanic is Hispanic of any race.
    The values are deliberately not renormalized after dropping other groups.
    """

    missing = sorted(_required_source_columns().difference(census.columns))
    if missing:
        raise ValueError(f"Census source is missing columns: {missing}")

    numeric_keys = census[["SUMLEV", "YEAR", "AGEGRP"]].apply(
        pd.to_numeric, errors="coerce"
    )
    selected = census.loc[
        (numeric_keys["SUMLEV"] == 50)
        & (numeric_keys["YEAR"] == SOURCE_YEAR)
        & (numeric_keys["AGEGRP"] == TOTAL_AGE_GROUP)
    ].copy()
    if selected.empty:
        raise ValueError("no 2012 total-age county rows (SUMLEV=50, YEAR=5, AGEGRP=0)")

    state = selected["STATE"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    county = selected["COUNTY"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
    selected["county_fips"] = state + county
    if selected["county_fips"].duplicated().any():
        duplicates = selected.loc[selected["county_fips"].duplicated(), "county_fips"].tolist()
        raise ValueError(f"duplicate 2012 county rows: {duplicates[:10]}")

    total = pd.to_numeric(selected["TOT_POP"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("TOT_POP must be finite and positive for every selected county")

    output = pd.DataFrame({"county_fips": selected["county_fips"].to_numpy()})
    for target, columns in SOURCE_COLUMNS.items():
        counts = sum(
            pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=np.float64)
            for column in columns
        )
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f"invalid Census counts for {target}: {columns}")
        output[f"p_{target}"] = counts / total

    if county_fips is not None:
        wanted = {str(value).strip().zfill(5) for value in county_fips}
        available = set(output["county_fips"])
        missing_counties = sorted(wanted.difference(available))
        if missing_counties:
            raise ValueError(f"requested FIPS absent from Census source: {missing_counties[:10]}")
        output = output[output["county_fips"].isin(wanted)].copy()

    return output.sort_values("county_fips").reset_index(drop=True)


def _counties_from_users(path: Path) -> set[str]:
    users = pd.read_csv(path, dtype={"county_fips": str})
    if "county_fips" not in users.columns:
        raise ValueError(f"{path} is missing county_fips")
    counties = set(users["county_fips"].dropna().astype(str).str.strip()) - {""}
    if not counties:
        raise ValueError(f"{path} contains no county_fips values")
    return counties


def build(
    dataset_root: Path,
    *,
    input_csv: Path | None = None,
    users_csv: Path | None = None,
    overwrite: bool = False,
) -> Path:
    root = dataset_root.expanduser().resolve()
    raw = root / "raw"
    output_path = raw / "census_county_proportions.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    if input_csv is None:
        input_path = download_census_csv(root / "source" / "cc-est2012-alldata.csv")
        source_kind = "downloaded_official_csv"
    else:
        input_path = input_csv.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        source_kind = "user_supplied_official_csv"

    if users_csv is None:
        default_users = raw / "users.csv"
        users_csv = default_users if default_users.is_file() else None
    requested_counties = _counties_from_users(users_csv) if users_csv is not None else None

    source = pd.read_csv(input_path, low_memory=False)
    output = build_race3_proportions(source, county_fips=requested_counties)
    _atomic_csv(output, output_path)
    _atomic_json(
        {
            "artifact": "census_county_proportions.csv",
            "source_kind": source_kind,
            "source_url": CENSUS_URL,
            "source_file": str(input_path),
            "source_vintage": "CC-EST2012-ALLDATA, July 1 2012 estimate",
            "source_filter": {"SUMLEV": 50, "YEAR": SOURCE_YEAR, "AGEGRP": TOTAL_AGE_GROUP},
            "county_fips_definition": "zero_pad(STATE,2) + zero_pad(COUNTY,3)",
            "race3_count_fields": {
                target: list(columns) for target, columns in SOURCE_COLUMNS.items()
            },
            "denominator": "TOT_POP",
            "race3_renormalized": False,
            "county_filter": "raw/users.csv" if requested_counties is not None else "all counties",
            "counties_written": int(len(output)),
            "twitter_records_created": False,
        },
        raw / "census_county_proportions.provenance.json",
    )
    print(f"Wrote {output_path} ({len(output)} counties; raw Race3 shares, not renormalized)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="use an existing official cc-est2012-alldata.csv instead of downloading it",
    )
    parser.add_argument(
        "--users-csv",
        type=Path,
        help="optional users.csv whose county_fips values should be retained",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(
        args.data_root,
        input_csv=args.input_csv,
        users_csv=args.users_csv,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
