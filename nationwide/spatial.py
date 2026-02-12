"""Tile discovery: STAC queries, HEAD-scan URL resolution, elevation filter."""

from __future__ import annotations

import json
import logging
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Swisstopo endpoints ─────────────────────────────────────────────────────

_SI_TEMPLATE = (
    "https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/"
    "swissimage-dop10_{year}_{coord}/"
    "swissimage-dop10_{year}_{coord}_0.1_2056.tif"
)

_DSM_TEMPLATE = (
    "https://data.geo.admin.ch/ch.swisstopo.swisssurface3d-raster/"
    "swisssurface3d-raster_{year}_{coord}/"
    "swisssurface3d-raster_{year}_{coord}_0.5_2056_5728.tif"
)

_DHM25_URL = "https://data.geo.admin.ch/ch.swisstopo.digitales-hoehenmodell_25/data.zip"

_STAC_BASE = "https://data.geo.admin.ch/api/stac/v0.9"
_SI_COLLECTION = "ch.swisstopo.swissimage-dop10"
_DSM_COLLECTION = "ch.swisstopo.swisssurface3d-raster"

_COORD_RE = re.compile(r"(\d{4}-\d{4})")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})


# ── HEAD-scan URL resolution (used for --coords / --elevation-json) ─────────

def _find_latest_url(
    template: str, coord: str, max_year: int = 2026, min_year: int = 2017,
) -> str | None:
    """HEAD-scan Swisstopo CDN backward from max_year; return first 200 URL."""
    for year in range(max_year, min_year - 1, -1):
        url = template.format(year=year, coord=coord)
        try:
            resp = _SESSION.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def resolve_tile_urls(
    coord: str, max_year: int = 2026,
) -> tuple[str, str] | None:
    """Find latest SwissIMAGE RGB and swissSURFACE3D DSM URLs for a tile.

    Returns (rgb_url, dsm_url) or None if either is missing.
    """
    rgb_url = _find_latest_url(_SI_TEMPLATE, coord, max_year)
    if rgb_url is None:
        return None
    dsm_url = _find_latest_url(_DSM_TEMPLATE, coord, max_year)
    if dsm_url is None:
        return None
    return (rgb_url, dsm_url)


def resolve_batch(
    coords: list[str], max_year: int = 2026, threads: int = 8,
) -> dict[str, tuple[str, str]]:
    """Resolve URLs for many tiles in parallel via HEAD requests."""
    results: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(resolve_tile_urls, coord, max_year): coord
            for coord in coords
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Resolving URLs", unit="tile",
        ):
            coord = futures[future]
            try:
                pair = future.result()
                if pair is not None:
                    results[coord] = pair
            except Exception as exc:
                log.warning(f"{coord} URL resolution failed: {exc}")
    return results


# ── STAC tile discovery (used for --bbox) ────────────────────────────────────

def _stac_paginate(
    collection: str, bbox: str, limit: int = 100,
) -> Iterator[dict]:
    """Yield all STAC items from a collection within a bbox, handling pagination."""
    url: str | None = f"{_STAC_BASE}/collections/{collection}/items"
    params: dict = {"bbox": bbox, "limit": limit}
    while url:
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        yield from data.get("features", [])
        url = None
        params = {}
        for link in data.get("links", []):
            if link.get("rel") == "next":
                url = link["href"]
                break


def _extract_stac_tiles(items: Iterator[dict]) -> dict[str, str]:
    """Extract (coord -> asset_url) mapping from STAC items.

    Parses item IDs for XXXX-YYYY coordinates and picks the first .tif
    asset URL per coordinate (STAC typically returns newest first).
    """
    result: dict[str, str] = {}
    for item in items:
        item_id = item.get("id", "")
        m = _COORD_RE.search(item_id)
        if not m:
            continue
        coord = m.group(1)
        if coord in result:
            continue  # Keep first (typically newest)
        for asset in item.get("assets", {}).values():
            href = asset.get("href", "")
            if href.endswith(".tif"):
                result[coord] = href
                break
    return result


def query_stac_bbox(bbox: str) -> list[tuple[str, str, str]]:
    """Query STAC API for tile pairs in a WGS84 bounding box.

    Args:
        bbox: "west,south,east,north" (e.g. "7.0,46.5,8.0,47.0").

    Returns:
        List of (coord, rgb_url, dsm_url) tuples.
    """
    log.info(f"Querying STAC for SwissIMAGE tiles in bbox={bbox} ...")
    rgb_tiles = _extract_stac_tiles(_stac_paginate(_SI_COLLECTION, bbox))
    log.info(f"  Found {len(rgb_tiles)} SwissIMAGE tiles")

    log.info(f"Querying STAC for swissSURFACE3D tiles in bbox={bbox} ...")
    dsm_tiles = _extract_stac_tiles(_stac_paginate(_DSM_COLLECTION, bbox))
    log.info(f"  Found {len(dsm_tiles)} swissSURFACE3D tiles")

    common = sorted(set(rgb_tiles) & set(dsm_tiles))
    log.info(f"  Matched pairs: {len(common)}")
    return [(c, rgb_tiles[c], dsm_tiles[c]) for c in common]


# ── Elevation filter ─────────────────────────────────────────────────────────

def load_filtered_coords(
    elevation_json: Path, min_elevation: float,
) -> list[str]:
    """Load tile_elevations.json, return coords above threshold as XXXX-YYYY."""
    with open(elevation_json) as f:
        elevations: dict[str, float] = json.load(f)
    return sorted(
        k.replace("_", "-") for k, v in elevations.items() if v >= min_elevation
    )


def download_dhm25(dest: Path) -> Path:
    """Download DHM25/200 DEM zip and extract the GeoTIFF."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "dhm25_200.zip"

    if dest.exists():
        log.info(f"DEM already exists: {dest}")
        return dest

    log.info(f"Downloading DHM25/200 from {_DHM25_URL} ...")
    resp = _SESSION.get(_DHM25_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="DHM25") as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    log.info("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
        if not tif_names:
            raise RuntimeError(f"No .tif found in {zip_path}")
        zf.extract(tif_names[0], dest.parent)
        extracted = dest.parent / tif_names[0]
        if extracted != dest:
            extracted.rename(dest)

    zip_path.unlink()
    log.info(f"DEM saved to {dest}")
    return dest


def compute_tile_elevations(
    dem_path: Path,
    grid_step_m: int = 1000,
) -> dict[str, float]:
    """Compute max elevation per 1km LV95 grid cell from DHM25/200 DEM.

    Returns dict mapping "XXXX_YYYY" -> max_elevation_m.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        nodata = src.nodata
        bounds = src.bounds

    x_min = int(bounds.left // grid_step_m) * grid_step_m
    x_max = int(np.ceil(bounds.right / grid_step_m)) * grid_step_m
    y_min = int(bounds.bottom // grid_step_m) * grid_step_m
    y_max = int(np.ceil(bounds.top / grid_step_m)) * grid_step_m

    result = {}
    xs = range(x_min, x_max, grid_step_m)
    ys = range(y_min, y_max, grid_step_m)

    for gx in tqdm(xs, desc="Grid cols"):
        for gy in ys:
            cell_left, cell_right = gx, gx + grid_step_m
            cell_bottom, cell_top = gy, gy + grid_step_m

            col_start, row_start = ~transform * (cell_left, cell_top)
            col_end, row_end = ~transform * (cell_right, cell_bottom)

            r0 = max(0, int(row_start))
            r1 = min(dem.shape[0], int(np.ceil(row_end)))
            c0 = max(0, int(col_start))
            c1 = min(dem.shape[1], int(np.ceil(col_end)))

            if r0 >= r1 or c0 >= c1:
                continue

            window = dem[r0:r1, c0:c1]
            if nodata is not None:
                valid = window[window != nodata]
            else:
                valid = window.ravel()

            if len(valid) == 0:
                continue

            max_elev = float(valid.max())
            tile_id = f"{gx // 1000}_{gy // 1000}"
            result[tile_id] = round(max_elev, 1)

    return result
