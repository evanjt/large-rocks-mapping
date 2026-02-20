"""Caching: file-based LRU tile cache and STAC response cache (DuckDB)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from urllib.parse import urlparse

import duckdb

log = logging.getLogger(__name__)


class TileCache:
    """File-based LRU cache for downloaded tiles.

    Keys are the URL filename (already unique for Swisstopo tiles).
    LRU tracked via file mtime (touch on hit, evict oldest).
    """

    def __init__(self, cache_dir: Path, max_bytes: int) -> None:
        self._dir = cache_dir
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(url: str) -> str:
        return Path(urlparse(url).path).name

    def get(self, url: str) -> bytes | None:
        path = self._dir / self._key(url)
        if not path.exists():
            return None
        path.touch()
        return path.read_bytes()

    def put(self, url: str, data: bytes) -> None:
        path = self._dir / self._key(url)
        path.write_bytes(data)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        with self._lock:
            files = list(self._dir.iterdir())
            total = sum(f.stat().st_size for f in files)
            if total <= self._max_bytes:
                return
            files.sort(key=lambda f: f.stat().st_mtime)
            for f in files:
                if total <= self._max_bytes:
                    break
                size = f.stat().st_size
                f.unlink()
                total -= size
                log.debug("Cache evicted %s (%d MB)", f.name, size // 1_000_000)


_tile_cache: TileCache | None = None
_cache_config: tuple[str, int] | None = None


def init_cache(cache_dir: Path, max_gb: float) -> None:
    """Initialise the module-level tile cache."""
    global _tile_cache, _cache_config
    if max_gb <= 0:
        _tile_cache = None
        _cache_config = None
        return
    max_bytes = int(max_gb * 1_000_000_000)
    _tile_cache = TileCache(cache_dir, max_bytes)
    _cache_config = (str(cache_dir), max_bytes)
    log.info("Tile cache: %s (max %.1f GB)", cache_dir, max_gb)


def get_cache_config() -> tuple[str, int] | None:
    """Return (cache_dir, max_bytes) for passing to worker initializers."""
    return _cache_config


def reinit_cache(cache_dir: str | None, max_bytes: int) -> None:
    """Re-create the tile cache in a worker process."""
    global _tile_cache
    if cache_dir is not None and max_bytes > 0:
        _tile_cache = TileCache(Path(cache_dir), max_bytes)
    else:
        _tile_cache = None


def cache_get(url: str) -> bytes | None:
    """Return cached bytes for url, or None."""
    if _tile_cache is None:
        return None
    return _tile_cache.get(url)


def cache_put(url: str, data: bytes) -> None:
    """Store downloaded bytes in the cache."""
    if _tile_cache is not None:
        _tile_cache.put(url, data)


# --- STAC response cache (DuckDB) ---

_stac_cache_path: Path | None = None


def set_stac_cache_dir(cache_dir: Path) -> None:
    """Set directory for the STAC response cache (call before query_stac_bbox)."""
    global _stac_cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    _stac_cache_path = cache_dir / "stac_cache.duckdb"


def load_stac_cache(bbox: str) -> list[tuple[str, str, str]] | None:
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


def save_stac_cache(bbox: str, tiles: list[tuple[str, str, str]]) -> None:
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
