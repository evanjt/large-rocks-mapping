"""Tile discovery: STAC queries, HEAD-scan URL resolution."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import requests
from tqdm import tqdm

from nationwide.cache import load_stac_cache, save_stac_cache
from utils.constants import (
    COORD_RE, DSM_COLLECTION, DSM_TEMPLATE,
    SI_COLLECTION, SI_TEMPLATE, STAC_BASE,
    SWITZERLAND_BBOX,
)

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})


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
    rgb_url = _find_latest_url(SI_TEMPLATE, coord, max_year)
    if rgb_url is None:
        return None
    dsm_url = _find_latest_url(DSM_TEMPLATE, coord, max_year)
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


def _stac_paginate(
    collection: str, bbox: str, limit: int = 100,
) -> Iterator[dict]:
    """Yield all STAC items from a collection within a bbox, handling pagination."""
    url: str | None = f"{STAC_BASE}/collections/{collection}/items"
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
        m = COORD_RE.search(item_id)
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
    cached = load_stac_cache(bbox)
    if cached is not None:
        log.info(f"STAC cache hit: {len(cached)} tile pairs for bbox={bbox}")
        return cached

    log.info(f"Querying STAC for SwissIMAGE tiles in bbox={bbox} ...")
    rgb_tiles = _extract_stac_tiles(_stac_paginate(SI_COLLECTION, bbox))
    log.info(f"  Found {len(rgb_tiles)} SwissIMAGE tiles")

    log.info(f"Querying STAC for swissSURFACE3D tiles in bbox={bbox} ...")
    dsm_tiles = _extract_stac_tiles(_stac_paginate(DSM_COLLECTION, bbox))
    log.info(f"  Found {len(dsm_tiles)} swissSURFACE3D tiles")

    common = sorted(set(rgb_tiles) & set(dsm_tiles))
    log.info(f"  Matched pairs: {len(common)}")

    result = [(c, rgb_tiles[c], dsm_tiles[c]) for c in common]
    save_stac_cache(bbox, result)
    return result
