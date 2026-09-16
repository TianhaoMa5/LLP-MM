#!/usr/bin/env python3
"""Fetch the full public Dec-Apr Sentinel-1 RTC sequence for LEM.

The retired INPE download page lists the Sentinel-1 acquisition calendar, but
its image binaries are no longer available from the archived site. This script
queries the Microsoft Planetary Computer Sentinel-1 RTC STAC collection for all
12 available official-calendar dates from December 2017 through April 2018. It clips the COG
assets to the supplied LEM field polygons, converts linear gamma-nought values
to dB, and writes aligned two-band (VV, VH) GeoTIFFs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = (
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/"
    "sentinel1euwestrtc/sentinel1-grd-rtc"
)
DEFAULT_DATES = (
    "2017-12-09", "2017-12-21",
    "2018-01-02", "2018-01-14", "2018-01-26",
    "2018-02-07", "2018-02-19",
    "2018-03-03", "2018-03-15", "2018-03-27",
    "2018-04-08", "2018-04-20",
)


def _json_request(url: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def _search(date: str, bbox_wgs84, relative_orbit: int):
    start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    result = _json_request(
        f"{STAC_ROOT}/search",
        {
            "collections": ["sentinel-1-rtc"],
            "bbox": list(bbox_wgs84),
            "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
            "limit": 100,
        },
    )
    candidates = [
        feature for feature in result.get("features", [])
        if feature["properties"].get("sat:relative_orbit") == relative_orbit
        and feature["properties"].get("sat:orbit_state") == "descending"
        and {"vv", "vh"}.issubset(feature.get("assets", {}))
    ]
    if len(candidates) != 1:
        ids = [feature.get("id") for feature in candidates]
        raise RuntimeError(
            f"Expected one descending relative-orbit-{relative_orbit} item for {date}; "
            f"found {len(candidates)}: {ids}"
        )
    return candidates[0]


def _target_grid(bounds, resolution: float, margin: float):
    minx, miny, maxx, maxy = bounds
    minx = math.floor((minx - margin) / resolution) * resolution
    miny = math.floor((miny - margin) / resolution) * resolution
    maxx = math.ceil((maxx + margin) / resolution) * resolution
    maxy = math.ceil((maxy + margin) / resolution) * resolution
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    return (minx, miny, maxx, maxy), width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="LEM field polygon Shapefile")
    parser.add_argument("--output", required=True)
    parser.add_argument("--date", action="append", dest="dates",
                        help="YYYY-MM-DD; repeat in the desired temporal order")
    parser.add_argument("--relative-orbit", type=int, default=126)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=110.0,
                        help="Extra metres around field bounds (>= 10 pixels for 21x21 patches)")
    parser.add_argument("--list-only", action="store_true",
                        help="Validate STAC availability and print the date/VV/VH table without downloading")
    args = parser.parse_args()
    dates = tuple(args.dates or DEFAULT_DATES)
    if not dates or len(set(dates)) != len(dates):
        parser.error("At least one unique --date is required")
    if tuple(sorted(dates)) != dates:
        parser.error("--date values must be chronological")

    try:
        import fiona
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import transform_bounds
    except ImportError as exc:
        raise SystemExit("Install rasterio and fiona before downloading LEM RTC data") from exc

    labels_path = Path(args.labels).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with fiona.open(labels_path) as fields:
        source_crs = fields.crs
        field_bounds = fields.bounds
    target_crs = "EPSG:32723"
    if source_crs:
        field_bounds = transform_bounds(source_crs, target_crs, *field_bounds)
    target_bounds, width, height = _target_grid(field_bounds, args.resolution, args.margin)
    bbox_wgs84 = transform_bounds(target_crs, "EPSG:4326", *target_bounds)
    transform = Affine(args.resolution, 0, target_bounds[0], 0, -args.resolution, target_bounds[3])

    features = [(date, _search(date, bbox_wgs84, args.relative_orbit)) for date in dates]
    print("date          VV available   VH available   used?", flush=True)
    for date, feature in features:
        print(f"{date}    {'vv' in feature['assets']!s:<12}   {'vh' in feature['assets']!s:<12}   yes", flush=True)
    if args.list_only:
        return

    token = _json_request(TOKEN_URL)["token"]
    records = []
    common_profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 2,
        "dtype": "float32", "crs": target_crs, "transform": transform,
        "nodata": -32768.0, "tiled": True, "blockxsize": 512, "blockysize": 512,
        "compress": "zstd", "predictor": 3, "BIGTIFF": "IF_SAFER",
    }
    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }

    for date, feature in features:
        destination = output / f"lem_s1_rtc_{date.replace('-', '')}_vv_vh_db.tif"
        print(f"{date}: {feature['id']} -> {destination}", flush=True)
        reuse = False
        if destination.is_file():
            with rasterio.open(destination) as existing:
                reuse = (
                    existing.count == 2 and existing.width == width and existing.height == height
                    and existing.crs == target_crs and existing.transform == transform
                    and tuple(existing.descriptions) == ("VV", "VH")
                )
            if not reuse:
                raise RuntimeError(f"Existing {destination} does not match the requested aligned grid")
            print("  reusing aligned existing file", flush=True)
        if not reuse:
            with rasterio.Env(**env_options), rasterio.open(destination, "w", **common_profile) as dst:
                for band_number, polarization in enumerate(("vv", "vh"), start=1):
                    signed_url = feature["assets"][polarization]["href"] + "?" + token
                    with rasterio.open(signed_url) as src, WarpedVRT(
                        src,
                        crs=target_crs,
                        transform=transform,
                        width=width,
                        height=height,
                        src_nodata=src.nodata,
                        nodata=-32768.0,
                        resampling=Resampling.bilinear,
                    ) as vrt:
                        linear = vrt.read(1, masked=True, out_dtype="float32")
                    values = np.full(linear.shape, -32768.0, dtype=np.float32)
                    valid = ~np.ma.getmaskarray(linear) & np.isfinite(linear.data) & (linear.data > 0)
                    values[valid] = 10.0 * np.log10(linear.data[valid])
                    dst.write(values, band_number)
                    dst.set_band_description(band_number, polarization.upper())
                dst.update_tags(
                    source_collection="Microsoft Planetary Computer sentinel-1-rtc",
                    source_item=feature["id"],
                    source_datetime=feature["properties"]["datetime"],
                    scale="gamma0_dB",
                )
        records.append({
            "date": date,
            "item_id": feature["id"],
            "datetime": feature["properties"]["datetime"],
            "relative_orbit": feature["properties"]["sat:relative_orbit"],
            "orbit_state": feature["properties"]["sat:orbit_state"],
            "output": destination.name,
            "assets": {key: feature["assets"][key]["href"] for key in ("vv", "vh")},
        })

    manifest = {
        "dataset": "LEM",
        "source": "Microsoft Planetary Computer sentinel-1-rtc",
        "source_stac": STAC_ROOT,
        "source_license": "CC-BY-4.0",
        "selection": "all available official-calendar acquisitions from Dec 2017 through Apr 2018",
        "dates": list(dates),
        "polarization_order": ["VV", "VH"],
        "channel_order": [f"{date}_{pol}" for date in dates for pol in ("VV", "VH")],
        "processing": "Planetary Computer RTC linear gamma0 -> 10*log10 -> aligned EPSG:32723 crop",
        "labels": labels_path.name,
        "bounds": list(target_bounds),
        "resolution": args.resolution,
        "shape_each": [2, height, width],
        "records": records,
    }
    manifest_path = output / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
