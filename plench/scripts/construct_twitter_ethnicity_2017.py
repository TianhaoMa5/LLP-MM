"""Reconstruct the Twitter Race3 training set from an authorized historical archive.

This script never queries the current X/Twitter API and never invents users,
coordinates, images, counties, or instance labels.  It accepts historical v1
Twitter JSON captured at collection time and builds auditable intermediate
artifacts for the image-only LLP benchmark.
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
from typing import BinaryIO, Iterable, Iterator

import pandas as pd
from PIL import Image, UnidentifiedImageError
import requests


CANDIDATE_COLUMNS = [
    "instance_id",
    "twitter_user_id",
    "county_fips",
    "profile_image_url",
    "first_observed_tweet_id",
    "first_observed_at",
    "geotagged_tweet_count",
    "text_path",
]
IMAGE_COLUMNS = [*CANDIDATE_COLUMNS, "image_path", "image_sha256", "download_status"]


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


def _text_stream(binary: BinaryIO, name: str) -> io.TextIOWrapper:
    lower = name.lower()
    if lower.endswith(".gz"):
        binary = gzip.GzipFile(fileobj=binary)
    elif lower.endswith(".bz2"):
        binary = bz2.BZ2File(binary)
    return io.TextIOWrapper(binary, encoding="utf-8", errors="replace")


def _regular_text_stream(path: Path):
    lower = path.name.lower()
    if lower.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if lower.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_archive_lines(paths: Iterable[Path]) -> Iterator[tuple[str, int, str]]:
    """Yield source name, line number, and text from JSONL, compressed JSONL, or TAR."""

    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DATA_MISSING: historical Twitter archive {path}")
        if tarfile.is_tarfile(path):
            with tarfile.open(path, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    lower = member.name.lower()
                    if not lower.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz", ".json.bz2", ".jsonl.bz2")):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    with _text_stream(extracted, member.name) as stream:
                        for line_number, line in enumerate(stream, start=1):
                            yield f"{path}:{member.name}", line_number, line
        else:
            with _regular_text_stream(path) as stream:
                for line_number, line in enumerate(stream, start=1):
                    yield str(path), line_number, line


def _tweet_coordinates(tweet: dict) -> tuple[float, float] | None:
    coordinates = tweet.get("coordinates")
    values = coordinates.get("coordinates") if isinstance(coordinates, dict) else None
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    try:
        longitude, latitude = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        return None
    return longitude, latitude


def _historical_profile_url(user: dict) -> str:
    for key in ("profile_image_url_https", "profile_image_url"):
        value = user.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_historical_tweet(tweet: dict) -> dict | None:
    """Parse the fields required from a historical Twitter API v1 tweet."""

    if not isinstance(tweet, dict) or "delete" in tweet:
        return None
    user = tweet.get("user")
    coordinates = _tweet_coordinates(tweet)
    if not isinstance(user, dict) or coordinates is None:
        return None
    user_id = str(user.get("id_str") or user.get("id") or "").strip()
    image_url = _historical_profile_url(user)
    if not user_id or not image_url or "default_profile_images" in image_url:
        return None
    tweet_id = str(tweet.get("id_str") or tweet.get("id") or "").strip()
    text = tweet.get("full_text") or tweet.get("text") or ""
    longitude, latitude = coordinates
    return {
        "twitter_user_id": user_id,
        "profile_image_url": image_url,
        "tweet_id": tweet_id,
        "created_at": str(tweet.get("created_at") or ""),
        "longitude": longitude,
        "latitude": latitude,
        "text": str(text),
    }


def _load_counties(path: Path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("county mapping requires geopandas") from exc

    county_path = path.expanduser().resolve()
    if not county_path.exists():
        raise FileNotFoundError(f"DATA_MISSING: Census/TIGER county geometry {county_path}")
    counties = gpd.read_file(county_path)
    geoid_column = next((name for name in ("GEOID", "GEOID10", "GEOID20") if name in counties), None)
    if geoid_column is None:
        raise ValueError(f"{county_path} needs a GEOID, GEOID10, or GEOID20 column")
    if counties.crs is None:
        raise ValueError(f"{county_path} has no declared coordinate reference system")
    counties = counties[[geoid_column, "geometry"]].rename(columns={geoid_column: "county_fips"})
    counties["county_fips"] = counties["county_fips"].astype(str).str.zfill(5)
    if counties["county_fips"].duplicated().any():
        raise ValueError("county geometry contains duplicate FIPS codes")
    return counties.to_crs("EPSG:4326")


def extract_candidates(
    archive_paths: list[Path],
    county_file: Path,
    data_root: Path,
    *,
    write_observed_text: bool = False,
    max_tweets_per_user: int = 200,
    overwrite: bool = False,
) -> Path:
    """Map exact historical tweet coordinates to counties and deduplicate users."""

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("candidate extraction requires geopandas") from exc

    root = data_root.expanduser().resolve()
    raw = root / "raw"
    output_path = raw / "archive_candidates.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to rebuild it")

    records: list[dict] = []
    malformed = 0
    json_objects = 0
    for _, _, line in iter_archive_lines(archive_paths):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        json_objects += 1
        parsed = parse_historical_tweet(payload)
        if parsed is not None:
            records.append(parsed)
    if not records:
        raise RuntimeError(
            "DATA_MISSING: no historical v1 tweets with exact coordinates and a non-default "
            "profile image URL were found"
        )

    counties = _load_counties(county_file)
    frame = pd.DataFrame(records)
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, counties, how="inner", predicate="within")
    if joined.empty:
        raise RuntimeError("DATA_MISSING: none of the exact tweet coordinates mapped to a U.S. county")
    joined = joined.sort_index(kind="stable")

    county_counts: dict[str, Counter] = defaultdict(Counter)
    first: dict[str, dict] = {}
    texts: dict[str, list[str]] = defaultdict(list)
    for row in joined.itertuples(index=False):
        user_id = str(row.twitter_user_id)
        county_fips = str(row.county_fips).zfill(5)
        county_counts[user_id][county_fips] += 1
        if user_id not in first:
            first[user_id] = {
                "profile_image_url": row.profile_image_url,
                "first_observed_tweet_id": row.tweet_id,
                "first_observed_at": row.created_at,
            }
        if write_observed_text and len(texts[user_id]) < max_tweets_per_user and row.text:
            texts[user_id].append(row.text.replace("\x00", ""))

    text_root = raw / "tweets"
    rows = []
    for user_id in sorted(first):
        # Pick the county with most exact-geotagged tweets; break ties by FIPS.
        county_fips, count = sorted(
            county_counts[user_id].items(), key=lambda item: (-item[1], item[0])
        )[0]
        instance_id = "twitter-" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
        text_path = ""
        if write_observed_text and texts[user_id]:
            text_root.mkdir(parents=True, exist_ok=True)
            destination = text_root / f"{instance_id}.txt"
            temporary = destination.with_suffix(f".txt.{os.getpid()}.tmp")
            temporary.write_text("\n".join(texts[user_id]) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
            text_path = str(destination.relative_to(raw))
        rows.append(
            {
                "instance_id": instance_id,
                "twitter_user_id": user_id,
                "county_fips": county_fips,
                **first[user_id],
                "geotagged_tweet_count": count,
                "text_path": text_path,
            }
        )
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    _atomic_csv(candidates, output_path)
    _atomic_json(
        {
            "artifact": "archive_candidates.csv",
            "construction": "historical_reconstruction_not_author_release",
            "current_x_api_queried": False,
            "archive_files": [str(path.expanduser().resolve()) for path in archive_paths],
            "county_geometry": str(county_file.expanduser().resolve()),
            "coordinate_policy": "exact Twitter v1 coordinates only; place bounding boxes rejected",
            "multiple_county_policy": "modal exact-geotagged county; ties resolved by lowest FIPS",
            "json_objects": json_objects,
            "malformed_lines_skipped": malformed,
            "eligible_geotagged_records": len(records),
            "county_mapped_records": int(len(joined)),
            "unique_users": int(len(candidates)),
            "observed_stream_text_written": bool(write_observed_text),
            "observed_stream_text_is_recent_200_api_equivalent": False,
            "max_observed_tweets_per_user": max_tweets_per_user if write_observed_text else 0,
        },
        raw / "archive_candidates.provenance.json",
    )
    print(f"Wrote {output_path} ({len(candidates)} historical users)")
    return output_path


def _full_size_profile_url(url: str) -> str:
    return re.sub(r"_normal(?=\.[A-Za-z0-9]+(?:[?#]|$))", "", url)


def _validated_image(content: bytes) -> tuple[bytes, str]:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=95)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("response is not a decodable image") from exc
    return output.getvalue(), ".jpg"


def download_images(
    candidates_csv: Path,
    data_root: Path,
    *,
    timeout: float = 20,
    overwrite: bool = False,
) -> Path:
    """Download the profile URLs recorded in historical JSON, without using an API."""

    root = data_root.expanduser().resolve()
    raw = root / "raw"
    output_path = raw / "image_downloads.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to rebuild it")
    candidates = pd.read_csv(
        candidates_csv.expanduser().resolve(),
        dtype={"instance_id": str, "twitter_user_id": str, "county_fips": str},
        keep_default_na=False,
    )
    missing = sorted(set(CANDIDATE_COLUMNS).difference(candidates.columns))
    if missing:
        raise ValueError(f"{candidates_csv} is missing columns: {missing}")

    image_root = raw / "images" / "historical"
    image_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "PLeNCH-TwitterEthnicity2017-historical-reconstruction/1.0"
    output_rows = []
    for row in candidates.to_dict(orient="records"):
        url = str(row["profile_image_url"])
        attempts = list(dict.fromkeys([_full_size_profile_url(url), url]))
        status = "download_failed"
        image_path = ""
        digest = ""
        for attempt in attempts:
            try:
                response = session.get(attempt, timeout=timeout)
                response.raise_for_status()
                encoded, suffix = _validated_image(response.content)
            except (requests.RequestException, ValueError):
                continue
            digest = hashlib.sha256(encoded).hexdigest()
            destination = image_root / f"{row['instance_id']}{suffix}"
            temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
            temporary.write_bytes(encoded)
            os.replace(temporary, destination)
            image_path = str(destination.relative_to(raw))
            status = "downloaded_historical_url"
            break
        output_rows.append({**row, "image_path": image_path, "image_sha256": digest, "download_status": status})

    output = pd.DataFrame(output_rows, columns=IMAGE_COLUMNS)
    _atomic_csv(output, output_path)
    _atomic_json(
        {
            "artifact": "image_downloads.csv",
            "construction": "historical_reconstruction_not_author_release",
            "current_x_api_queried": False,
            "source": "profile_image_url(_https) stored in historical tweet JSON",
            "full_size_policy": "remove terminal _normal before extension, then fall back to recorded URL",
            "attempted": int(len(output)),
            "downloaded": int((output["download_status"] == "downloaded_historical_url").sum()),
            "failed": int((output["download_status"] != "downloaded_historical_url").sum()),
        },
        raw / "image_downloads.provenance.json",
    )
    print(f"Wrote {output_path}: {(output['download_status'] == 'downloaded_historical_url').sum()}/{len(output)} images")
    return output_path


def filter_single_face(
    image_downloads_csv: Path,
    data_root: Path,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: int = 30,
    overwrite: bool = False,
) -> Path:
    """Retain users with exactly one OpenCV Viola-Jones frontal-face detection."""

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Viola-Jones filtering requires OpenCV; install the twitter extras"
        ) from exc

    root = data_root.expanduser().resolve()
    raw = root / "raw"
    output_path = raw / "users.csv"
    audit_path = raw / "face_detection_audit.csv"
    if (output_path.exists() or audit_path.exists()) and not overwrite:
        raise FileExistsError(f"{output_path} or {audit_path} exists; pass --overwrite to rebuild")
    downloads = pd.read_csv(
        image_downloads_csv.expanduser().resolve(),
        dtype={"instance_id": str, "twitter_user_id": str, "county_fips": str},
        keep_default_na=False,
    )
    missing = sorted(set(IMAGE_COLUMNS).difference(downloads.columns))
    if missing:
        raise ValueError(f"{image_downloads_csv} is missing columns: {missing}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"could not load OpenCV cascade {cascade_path}")

    audit_rows = []
    retained = []
    for row in downloads.to_dict(orient="records"):
        relative = str(row["image_path"])
        count = -1
        status = "not_downloaded"
        if relative:
            image = cv2.imread(str(raw / relative), cv2.IMREAD_COLOR)
            if image is None:
                status = "decode_failed"
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_face_size, min_face_size),
                )
                count = len(faces)
                status = "retained_exactly_one" if count == 1 else "rejected_face_count"
        audit_rows.append(
            {
                "instance_id": row["instance_id"],
                "twitter_user_id": row["twitter_user_id"],
                "county_fips": row["county_fips"],
                "image_path": relative,
                "face_count": count,
                "status": status,
            }
        )
        if count == 1:
            retained.append(
                {
                    "instance_id": row["instance_id"],
                    "twitter_user_id": row["twitter_user_id"],
                    "county_fips": row["county_fips"],
                    "image_path": relative,
                    "text_path": row["text_path"],
                }
            )

    users = pd.DataFrame(
        retained,
        columns=["instance_id", "twitter_user_id", "county_fips", "image_path", "text_path"],
    )
    audit = pd.DataFrame(audit_rows)
    _atomic_csv(audit, audit_path)
    _atomic_csv(users, output_path)
    _atomic_json(
        {
            "artifact": "users.csv",
            "construction": "historical_reconstruction_not_author_release",
            "detector": "OpenCV Viola-Jones Haar cascade",
            "cascade": cascade_path.name,
            "scale_factor": scale_factor,
            "min_neighbors": min_neighbors,
            "min_face_size": [min_face_size, min_face_size],
            "retention_policy": "exactly one detected frontal face",
            "images_audited": int(len(audit)),
            "users_retained": int(len(users)),
            "instance_labels_created": False,
            "paper_parameter_match_claimed": False,
        },
        raw / "users.provenance.json",
    )
    print(f"Wrote {output_path} ({len(users)}/{len(audit)} exactly-one-face users)")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    extract_parser = subparsers.add_parser("extract", help="extract county-mapped users")
    extract_parser.add_argument("--archive", type=Path, nargs="+", required=True)
    extract_parser.add_argument("--county-file", type=Path, required=True)
    extract_parser.add_argument("--data-root", type=Path, required=True)
    extract_parser.add_argument("--write-observed-text", action="store_true")
    extract_parser.add_argument("--max-tweets-per-user", type=int, default=200)
    extract_parser.add_argument("--overwrite", action="store_true")

    image_parser = subparsers.add_parser("images", help="download recorded historical profile URLs")
    image_parser.add_argument("--candidates-csv", type=Path)
    image_parser.add_argument("--data-root", type=Path, required=True)
    image_parser.add_argument("--timeout", type=float, default=20)
    image_parser.add_argument("--overwrite", action="store_true")

    face_parser = subparsers.add_parser("faces", help="retain exactly-one-face users")
    face_parser.add_argument("--image-downloads-csv", type=Path)
    face_parser.add_argument("--data-root", type=Path, required=True)
    face_parser.add_argument("--scale-factor", type=float, default=1.1)
    face_parser.add_argument("--min-neighbors", type=int, default=5)
    face_parser.add_argument("--min-face-size", type=int, default=30)
    face_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.data_root.expanduser().resolve()
    if args.stage == "extract":
        extract_candidates(
            args.archive,
            args.county_file,
            root,
            write_observed_text=args.write_observed_text,
            max_tweets_per_user=args.max_tweets_per_user,
            overwrite=args.overwrite,
        )
    elif args.stage == "images":
        download_images(
            args.candidates_csv or root / "raw" / "archive_candidates.csv",
            root,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
    else:
        filter_single_face(
            args.image_downloads_csv or root / "raw" / "image_downloads.csv",
            root,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_face_size=args.min_face_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
