"""File-based LRU tile cache for downloaded Swisstopo tiles."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from urllib.parse import urlparse

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
