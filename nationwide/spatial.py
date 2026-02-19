"""Tile discovery: STAC queries, HEAD-scan URL resolution."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import duckdb
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── STAC cache (DuckDB in cache dir) ─────────────────────────────────────────

_stac_cache_path: Path | None = None


def set_stac_cache_dir(cache_dir: Path) -> None:
    """Set directory for the STAC response cache (call before query_stac_bbox)."""
    global _stac_cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    _stac_cache_path = cache_dir / "stac_cache.duckdb"


def _load_stac_cache(bbox: str) -> list[tuple[str, str, str]] | None:
    """Load cached STAC results for the given bbox, or None if not cached."""
    if _stac_cache_path is None or not _stac_cache_path.exists():
        return None
    try:
        con = duckdb.connect(str(_stac_cache_path), read_only=True)
        row = con.execute(
            "SELECT n_tiles FROM stac_cache_meta WHERE bbox = ?", [bbox],
        ).fetchone()
        if row is None:
            con.close()
            return None
        tiles = con.execute(
            "SELECT coord, rgb_url, dsm_url FROM stac_cache WHERE bbox = ? ORDER BY coord",
            [bbox],
        ).fetchall()
        con.close()
        if len(tiles) != row[0]:
            log.warning("STAC cache count mismatch, re-querying")
            return None
        return [(c, r, d) for c, r, d in tiles]
    except Exception as exc:
        log.warning(f"STAC cache read failed: {exc}")
        return None


def _save_stac_cache(bbox: str, tiles: list[tuple[str, str, str]]) -> None:
    """Persist STAC results to the cache DuckDB."""
    if _stac_cache_path is None:
        return
    try:
        con = duckdb.connect(str(_stac_cache_path))
        con.execute("""
            CREATE TABLE IF NOT EXISTS stac_cache (
                bbox     VARCHAR,
                coord    VARCHAR,
                rgb_url  VARCHAR NOT NULL,
                dsm_url  VARCHAR NOT NULL,
                PRIMARY KEY (bbox, coord)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS stac_cache_meta (
                bbox       VARCHAR PRIMARY KEY,
                n_tiles    INTEGER,
                cached_at  TIMESTAMP DEFAULT current_timestamp
            )
        """)
        # Replace any existing cache for this bbox
        con.execute("DELETE FROM stac_cache WHERE bbox = ?", [bbox])
        con.execute("DELETE FROM stac_cache_meta WHERE bbox = ?", [bbox])
        con.executemany(
            "INSERT INTO stac_cache (bbox, coord, rgb_url, dsm_url) VALUES (?, ?, ?, ?)",
            [(bbox, c, r, d) for c, r, d in tiles],
        )
        con.execute(
            "INSERT INTO stac_cache_meta (bbox, n_tiles) VALUES (?, ?)",
            [bbox, len(tiles)],
        )
        con.close()
        log.info(f"STAC cache saved: {len(tiles)} tile pairs for bbox={bbox}")
    except Exception as exc:
        log.warning(f"STAC cache write failed: {exc}")

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

_STAC_BASE = "https://data.geo.admin.ch/api/stac/v0.9"
_SI_COLLECTION = "ch.swisstopo.swissimage-dop10"
_DSM_COLLECTION = "ch.swisstopo.swisssurface3d-raster"

_COORD_RE = re.compile(r"(\d{4}-\d{4})")

# Union of SwissIMAGE + swissSURFACE3D spatial extents (WGS84), from:
#   GET https://data.geo.admin.ch/api/stac/v0.9/collections/{collection}
#   → extent.spatial.bbox
# SwissIMAGE:     [5.9503666, 45.8151271, 10.4998461, 47.8091281]
# swissSURFACE3D: [5.9503666, 45.7213375, 10.4998461, 47.8216742]
# We use the union so the STAC query returns all tiles from both layers;
# query_stac_bbox() then intersects the results to keep only matched pairs.
SWITZERLAND_BBOX = "5.95,45.72,10.50,47.83"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})


# ── HEAD-scan URL resolution (used for --coords) ────────────────────────────

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

    Uses a local DuckDB cache (in the tile cache dir) to avoid re-querying
    the STAC API on subsequent runs with the same bbox.

    Args:
        bbox: "west,south,east,north" (e.g. "7.0,46.5,8.0,47.0").

    Returns:
        List of (coord, rgb_url, dsm_url) tuples.
    """
    cached = _load_stac_cache(bbox)
    if cached is not None:
        log.info(f"STAC cache hit: {len(cached)} tile pairs for bbox={bbox}")
        return cached

    log.info(f"Querying STAC for SwissIMAGE tiles in bbox={bbox} ...")
    rgb_tiles = _extract_stac_tiles(_stac_paginate(_SI_COLLECTION, bbox))
    log.info(f"  Found {len(rgb_tiles)} SwissIMAGE tiles")

    log.info(f"Querying STAC for swissSURFACE3D tiles in bbox={bbox} ...")
    dsm_tiles = _extract_stac_tiles(_stac_paginate(_DSM_COLLECTION, bbox))
    log.info(f"  Found {len(dsm_tiles)} swissSURFACE3D tiles")

    common = sorted(set(rgb_tiles) & set(dsm_tiles))
    log.info(f"  Matched pairs: {len(common)}")

    result = [(c, rgb_tiles[c], dsm_tiles[c]) for c in common]
    _save_stac_cache(bbox, result)
    return result
