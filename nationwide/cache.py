import duckdb
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
        try:
            data = path.read_bytes()
            if not data:
                path.unlink(missing_ok=True)
                return None
            path.touch()
            return data
        except (FileNotFoundError, OSError):
            return None

    def put(self, url: str, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            path = self._dir / self._key(url)
            path.write_bytes(data)
        self._evict_if_needed()

    def path(self, url: str) -> Path | None:
        """Return filesystem path to cached file without reading it."""
        p = self._dir / self._key(url)
        try:
            if p.exists() and p.stat().st_size > 0:
                p.touch()
                return p
        except OSError:
            pass
        return None

    def _evict_if_needed(self) -> None:
        with self._lock:
            files = []
            total = 0
            for f in self._dir.iterdir():
                try:
                    files.append((f, f.stat()))
                except (FileNotFoundError, OSError):
                    continue
            total = sum(s.st_size for _, s in files)
            if total <= self._max_bytes:
                return
            files.sort(key=lambda x: x[1].st_mtime)
            for f, s in files:
                if total <= self._max_bytes:
                    break
                try:
                    f.unlink()
                except (FileNotFoundError, OSError):
                    continue
                total -= s.st_size


class PatchStore:
    """File store for preprocessed patches. NO size limit, NO LRU eviction.

    Patches are the expensive output of preprocessing — they must never be
    evicted by the download cache's LRU policy.
    """

    def __init__(self, patch_dir: Path) -> None:
        self._dir = patch_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        if not data:
            return
        (self._dir / key).write_bytes(data)

    def get(self, key: str) -> bytes | None:
        path = self._dir / key
        try:
            data = path.read_bytes()
            return data if data else None
        except (FileNotFoundError, OSError):
            return None

    def path(self, key: str) -> Path | None:
        p = self._dir / key
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except OSError:
            pass
        return None


_tile_cache: TileCache | None = None
_patch_store: PatchStore | None = None
_cache_config: tuple[str, int] | None = None


def _migrate_flat_cache(cache_dir: Path) -> None:
    """Move files from flat cache layout to tiered subdirectories.

    Idempotent: files already in subdirectories are not in the root.
    """
    downloads_dir = cache_dir / "downloads"
    patches_dir = cache_dir / "patches"
    downloads_dir.mkdir(exist_ok=True)
    patches_dir.mkdir(exist_ok=True)

    moved = 0
    for f in cache_dir.iterdir():
        if f.is_dir():
            continue
        suffix = f.suffix.lower()
        if suffix in (".tif", ".npy"):
            f.rename(downloads_dir / f.name)
            moved += 1
        # .duckdb and other files stay in root

    if moved:
        log.info("Cache migration: moved %d download files to downloads/", moved)


def init_cache(cache_dir: Path, max_gb: float) -> None:
    """Initialise the module-level tile cache and patch store.

    Downloads (raw .tif) go in downloads/ with LRU eviction.
    Patches (.patchbin) go in patches/ with NO eviction.
    """
    global _tile_cache, _patch_store, _cache_config
    if max_gb <= 0:
        _tile_cache = None
        _patch_store = None
        _cache_config = None
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    _migrate_flat_cache(cache_dir)

    max_bytes = int(max_gb * 1_000_000_000)
    _tile_cache = TileCache(cache_dir / "downloads", max_bytes)
    _patch_store = PatchStore(cache_dir / "patches")
    _cache_config = (str(cache_dir), max_bytes)

    dl_dir = cache_dir / "downloads"
    dl_bytes = sum(f.stat().st_size for f in dl_dir.iterdir() if f.is_file())
    patch_dir = cache_dir / "patches"
    patch_count = sum(1 for f in patch_dir.iterdir() if f.is_file())
    patch_bytes = sum(f.stat().st_size for f in patch_dir.iterdir() if f.is_file())

    log.info(
        "Tile cache: %s (downloads: %.1f/%.1f GB | patches: %.1f GB, %d files)",
        cache_dir, dl_bytes / 1e9, max_gb, patch_bytes / 1e9, patch_count,
    )


def get_cache_config() -> tuple[str, int] | None:
    """Return (cache_dir, max_bytes) for passing to worker initializers."""
    return _cache_config


def reinit_cache(cache_dir: str | None, max_bytes: int) -> None:
    """Re-create the tile cache and patch store in a worker process."""
    global _tile_cache, _patch_store
    if cache_dir is not None and max_bytes > 0:
        root = Path(cache_dir)
        _tile_cache = TileCache(root / "downloads", max_bytes)
        _patch_store = PatchStore(root / "patches")
    else:
        _tile_cache = None
        _patch_store = None


def cache_path(url: str) -> Path | None:
    """Return filesystem path to cached file without reading bytes."""
    if _tile_cache is None:
        return None
    return _tile_cache.path(url)


def cache_get(url: str) -> bytes | None:
    """Return cached bytes for url, or None."""
    if _tile_cache is None:
        return None
    return _tile_cache.get(url)


def cache_put(url: str, data: bytes) -> None:
    """Store downloaded bytes in the cache."""
    if _tile_cache is not None:
        _tile_cache.put(url, data)


# --- Patch store (no eviction) ---


def patch_path(key: str) -> Path | None:
    """Return filesystem path to a cached patch file, or None."""
    if _patch_store is None:
        return None
    return _patch_store.path(key)


def patch_get(key: str) -> bytes | None:
    """Return cached patch bytes for key, or None."""
    if _patch_store is None:
        return None
    return _patch_store.get(key)


def patch_put(key: str, data: bytes) -> None:
    """Store patch bytes (never evicted)."""
    if _patch_store is not None:
        _patch_store.put(key, data)


# --- STAC response cache (DuckDB) ---

_stac_cache_path: Path | None = None


def set_stac_cache_dir(cache_dir: Path) -> None:
    """Set directory for the STAC response cache """
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
            "SELECT coord, rgb_url, dsm_url FROM stac_cache WHERE bbox = ? ORDER BY coord",  # noqa
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


def load_neighbor_cache() -> dict[str, tuple[str, str]]:
    """Load cached neighbor tile URLs. Returns {coord: (rgb_url, dsm_url)}."""
    if _stac_cache_path is None or not _stac_cache_path.exists():
        return {}
    try:
        con = duckdb.connect(str(_stac_cache_path), read_only=True)
        rows = con.execute("SELECT coord, rgb_url, dsm_url FROM neighbor_cache").fetchall()
        con.close()
        return {c: (r, d) for c, r, d in rows}
    except Exception:
        return {}


def save_neighbor_cache(resolved: dict[str, tuple[str, str]]) -> None:
    """Persist resolved neighbor tile URLs to cache."""
    if _stac_cache_path is None or not resolved:
        return
    try:
        con = duckdb.connect(str(_stac_cache_path))
        con.execute("""
            CREATE TABLE IF NOT EXISTS neighbor_cache (
                coord    VARCHAR PRIMARY KEY,
                rgb_url  VARCHAR NOT NULL,
                dsm_url  VARCHAR NOT NULL
            )
        """)
        con.executemany(
            "INSERT OR REPLACE INTO neighbor_cache (coord, rgb_url, dsm_url) VALUES (?, ?, ?)",
            [(c, r, d) for c, (r, d) in resolved.items()],
        )
        con.close()
    except Exception as exc:
        log.warning(f"Neighbor cache write failed: {exc}")


def load_neighbor_misses() -> set[str]:
    """Load coords previously resolved as missing (no tile on server)."""
    if _stac_cache_path is None or not _stac_cache_path.exists():
        return set()
    try:
        con = duckdb.connect(str(_stac_cache_path), read_only=True)
        rows = con.execute("SELECT coord FROM neighbor_misses").fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def save_neighbor_misses(coords: set[str]) -> None:
    """Persist coords confirmed missing from the server."""
    if _stac_cache_path is None or not coords:
        return
    try:
        con = duckdb.connect(str(_stac_cache_path))
        con.execute("""
            CREATE TABLE IF NOT EXISTS neighbor_misses (
                coord VARCHAR PRIMARY KEY
            )
        """)
        con.executemany(
            "INSERT OR REPLACE INTO neighbor_misses (coord) VALUES (?)",
            [(c,) for c in coords],
        )
        con.close()
    except Exception as exc:
        log.warning(f"Neighbor misses cache write failed: {exc}")


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
            "INSERT INTO stac_cache (bbox, coord, rgb_url, dsm_url) VALUES (?, ?, ?, ?)",  # noqa
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
