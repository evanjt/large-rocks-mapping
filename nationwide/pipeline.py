import concurrent.futures
import logging
import os
import queue
import threading
import time
import typer
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from nationwide.db import (
    get_cached_elevations,
    get_processed_tiles,
    init_db,
    mark_tile_done,
    save_elevation,
    write_detections,
)
from nationwide.cache import (
    get_cache_config, init_cache, load_neighbor_cache, load_neighbor_misses,
    patch_path, save_neighbor_cache, save_neighbor_misses, set_stac_cache_dir,
)
from utils.constants import SWITZERLAND_BBOX, TILE_SIZE_PX
from nationwide.processing import (
    _patch_cache_key,
    build_batch_tensor,
    check_elevation,
    check_gdaldem,
    check_gdalbuildvrt,
    dedup_detections,
    infer_on_tensor,
    process_tile_from_cache,
    reinit_session,
    run_inference,
)
from nationwide.spatial import load_url_csvs, query_stac_bbox, resolve_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_process(
    coord: str, rgb_url: str, dsm_url: str,
    neighbor_right: tuple[str, str] | None = None,
    neighbor_bottom: tuple[str, str] | None = None,
    neighbor_corner: tuple[str, str] | None = None,
    discard_patches: bool = False,
    cache_patches: bool = True,
) -> tuple[str, list | int | Exception]:
    """Process single tile, catching exceptions.

    With discard_patches=True, returns (coord, n_patches) — avoids pickling
    ~20 MB of numpy data when only the caching side-effect matters.
    With cache_patches=False, patches are not saved to PatchStore.
    """
    try:
        patches = process_tile_from_cache(
            coord, rgb_url, dsm_url,
            neighbor_right=neighbor_right,
            neighbor_bottom=neighbor_bottom,
            neighbor_corner=neighbor_corner,
            cache_patches=cache_patches,
        )
        return (coord, len(patches) if discard_patches else patches)
    except Exception as exc:
        return (coord, exc)


def _has_cached_patches(entry: tuple) -> bool:
    """Check if a tile's fused patches are already in the PatchStore."""
    pk = _patch_cache_key(entry[0], any(entry[3:]))
    return (patch_path(pk) is not None
)


def _resolve_tiles_from_cli(
    coords, bbox, all_switzerland, rgb_url_csv, dsm_url_csv,
    download_threads, rgb_year, dsm_year,
) -> list[tuple[str, str, str]]:
    """Resolve tile URLs from CLI arguments. Shared by run and fill-cache."""
    if rgb_url_csv and dsm_url_csv:
        return load_url_csvs(rgb_url_csv, dsm_url_csv)
    if all_switzerland:
        return query_stac_bbox(SWITZERLAND_BBOX, rgb_year=rgb_year, dsm_year=dsm_year)
    if bbox:
        return query_stac_bbox(bbox, rgb_year=rgb_year, dsm_year=dsm_year)
    if coords:
        log.info("Resolving tile URLs ...")
        url_map = resolve_batch(list(coords), threads=download_threads)
        log.info(f"Resolved {len(url_map)}/{len(coords)} tile URL pairs")
        return [(c, r, d) for c, (r, d) in url_map.items()]
    log.error("No tile source. Pass --coords, --bbox, --all, or --rgb-urls/--dsm-urls.")
    raise typer.Exit(code=1)


def _resolve_tile_source(
    tile_source: list[tuple[str, str, str]],
    skip_coords: set[str],
    elevation_db,
    min_elevation: float,
    download_threads: int,
) -> tuple[list[tuple], int, int]:
    """Filter tiles by checkpoint/cache + elevation, resolve neighbor URLs.

    Args:
        tile_source: All resolved (coord, rgb_url, dsm_url) tuples.
        skip_coords: Coordinates to skip (already processed or cached).
        elevation_db: DuckDB connection for elevation caching.
        min_elevation: Skip tiles below this elevation (0 = disabled).
        download_threads: Threads for neighbor URL resolution.

    Returns:
        (tiles_with_neighbors, n_skipped, n_elevation_filtered)
    """
    remaining = [(c, r, d) for c, r, d in tile_source if c not in skip_coords]
    n_skipped = len(tile_source) - len(remaining)
    if n_skipped:
        log.info(f"Skipping {n_skipped} already-processed tiles")

    # --- Elevation pre-filter ---
    n_elevation = 0
    if min_elevation > 0 and remaining:
        cached_elevs = get_cached_elevations(elevation_db)
        uncached = [(c, r, d) for c, r, d in remaining if c not in cached_elevs]

        if uncached:
            log.info(f"Checking elevations: {len(cached_elevs)} cached, {len(uncached)} to download ...")

            def _get_elev(tile):
                try:
                    return (tile[0], check_elevation(tile[2]))
                except Exception:
                    return (tile[0], float("inf"))

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                results = list(tqdm(
                    pool.map(_get_elev, uncached),
                    total=len(uncached), desc="Elevation filter", unit="tile",
                ))
            for coord, elev in results:
                cached_elevs[coord] = elev
                save_elevation(elevation_db, coord, elev)
        else:
            log.info(f"Elevation filter: all {len(remaining)} tiles cached")

        before = len(remaining)
        remaining = [(c, r, d) for c, r, d in remaining
                     if cached_elevs.get(c, float("inf")) >= min_elevation]
        n_elevation = before - len(remaining)
        if n_elevation:
            log.info(f"Eliminated {n_elevation} tiles below {min_elevation:.0f}m")

    log.info(f"Tiles to process: {len(remaining)}")

    if not remaining:
        return [], n_skipped, n_elevation

    # --- Neighbor URL resolution ---
    url_lookup: dict[str, tuple[str, str]] = {c: (r, d) for c, r, d in tile_source}
    cached_neighbors = load_neighbor_cache()
    url_lookup.update(cached_neighbors)
    known_misses = load_neighbor_misses()

    neighbor_coords: set[str] = set()
    for c, _, _ in tile_source:
        parts = c.split("-")
        x, y = int(parts[0]), int(parts[1])
        for nc in [f"{x+1}-{y}", f"{x}-{y-1}", f"{x+1}-{y-1}"]:
            if nc not in url_lookup and nc not in known_misses:
                neighbor_coords.add(nc)

    if neighbor_coords:
        log.info(f"Resolving {len(neighbor_coords)} neighbor tile URLs ({len(cached_neighbors)} cached, {len(known_misses)} known missing) ...")
        resolved = resolve_batch(list(neighbor_coords), threads=download_threads)
        url_lookup.update(resolved)
        save_neighbor_cache({**cached_neighbors, **resolved})
        new_misses = neighbor_coords - set(resolved)
        if new_misses:
            save_neighbor_misses(known_misses | new_misses)
        log.info(f"  Resolved {len(resolved)}/{len(neighbor_coords)} neighbor tiles")

    def _neighbor_urls(coord: str) -> tuple:
        parts = coord.split("-")
        x, y = int(parts[0]), int(parts[1])
        return (
            url_lookup.get(f"{x+1}-{y}"),
            url_lookup.get(f"{x}-{y-1}"),
            url_lookup.get(f"{x+1}-{y-1}"),
        )

    tiles_with_neighbors = [
        (c, r, d, *_neighbor_urls(c)) for c, r, d in remaining
    ]
    return tiles_with_neighbors, n_skipped, n_elevation


# ---------------------------------------------------------------------------
# Pipeline internals
# ---------------------------------------------------------------------------


def _producer_thread(
    tiles: list[tuple],
    tile_queue: queue.Queue,
    workers: int,
    cache_patches: bool = True,
) -> None:
    """Download + preprocess tiles, feed results into tile_queue.

    Each worker checks the patch cache first — cached tiles return
    immediately. Uncached tiles are downloaded, processed, and cached.
    Uses FIRST_COMPLETED so the GPU never waits for a slow tile.
    """
    cc = get_cache_config()
    init_args = (cc[0], cc[1]) if cc else (None, 0)
    pbar = tqdm(total=len(tiles), desc="Preprocessing", unit="tile",
                position=1, leave=False)

    def _submit(entry):
        return pool.submit(_safe_process, *entry, cache_patches=cache_patches)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=reinit_session,
        initargs=init_args,
    ) as pool:
        pending: dict[concurrent.futures.Future, bool] = {}
        tile_iter = iter(tiles)

        # Seed the pool with 2x workers to keep it busy
        for _ in range(workers * 2):
            entry = next(tile_iter, None)
            if entry is None:
                break
            pending[_submit(entry)] = True

        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                del pending[future]
                tile_queue.put(future.result())
                pbar.update(1)

                entry = next(tile_iter, None)
                if entry is not None:
                    pending[_submit(entry)] = True

    pbar.close()
    tile_queue.put(None)  # Sentinel


def _writer_thread(
    con,
    write_queue: queue.Queue,
    stats: dict,
    pbar,
    no_dedup: bool = False,
) -> None:
    """Background thread: dedup + write detections + checkpoint to DuckDB."""
    while True:
        item = write_queue.get()
        if item is None:
            break
        tile_id, detections = item
        try:
            tile_dets = detections if no_dedup else dedup_detections(detections, distance_m=7.5)
            n = write_detections(con, tile_dets)
            mark_tile_done(con, tile_id, n)
            stats["total_detections"] += n
            stats["tiles_processed"] += 1
        except Exception as exc:
            log.error(f"Writer failed for tile {tile_id}: {exc}")
            stats["tiles_failed"] += 1
        pbar.update(1)
        pbar.set_postfix(
            dets=stats["total_detections"],
            ok=stats["tiles_processed"],
            fail=stats["tiles_failed"],
        )


def _collect_batch(
    tile_queue: queue.Queue,
    max_tiles: int,
    write_queue: queue.Queue,
) -> tuple[list[tuple[str, list]], list[tuple], bool]:
    """Drain up to max_tiles from queue. Blocks until full or producer done."""
    batch: list[tuple[str, list]] = []
    all_patches: list[tuple] = []
    sentinel = False

    while len(batch) < max_tiles:
        item = tile_queue.get()  # block until a tile is ready

        if item is None:
            sentinel = True
            break

        coord, result = item
        tile_id = coord.replace("-", "_")

        if isinstance(result, Exception):
            log.warning(f"Tile {coord} failed: {result}")
            continue
        if not result:
            write_queue.put((tile_id, []))
            continue

        batch.append((tile_id, result))
        all_patches.extend(result)

    return batch, all_patches, sentinel


def run_pipeline(
    model_path: Path,
    output_db: Path,
    tiles: list[tuple],
    device: str = "auto",
    workers: int = 8,
    conf: float = 0.10,
    iou: float = 0.70,
    max_batch_tiles: int = 16,
    queue_maxsize: int = 0,
    no_dedup: bool = False,
    cache_patches: bool = False,
) -> None:
    """Producer-consumer pipeline: preprocess -> infer -> dedup -> store.

    Tiles are preprocessed in parallel workers, batched, and sent through
    double-buffered GPU inference. With cache_patches=True, fused patches
    are saved to the PatchStore for reuse by future runs.
    """
    if not tiles:
        log.info("No tiles to process.")
        return

    if max_batch_tiles <= 0:
        max_batch_tiles = 8
    if queue_maxsize <= 0:
        queue_maxsize = workers * 3

    con = init_db(output_db)

    # --- Start producer thread ---
    tile_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
    producer = threading.Thread(
        target=_producer_thread,
        args=(tiles, tile_queue, workers, cache_patches),
        daemon=True,
    )
    producer.start()

    # --- Load YOLO model ---
    import torch
    from ultralytics import YOLO

    log.info(f"Loading model from {model_path} on {device} ...")
    model = YOLO(model_path)
    if device != "cpu":
        model.to(device)
        log.info("Warming up GPU ...")
        dummy = torch.zeros(1, 3, TILE_SIZE_PX, TILE_SIZE_PX, device=device)
        model.predict(source=dummy, conf=conf, iou=iou, imgsz=TILE_SIZE_PX,
                      save=False, verbose=False)
        del dummy
        torch.cuda.empty_cache()

    # --- Start writer thread ---
    stats = {"total_detections": 0, "tiles_processed": 0, "tiles_failed": 0}
    pbar = tqdm(total=len(tiles), desc="Processing tiles", unit="tile")
    write_queue: queue.Queue = queue.Queue()
    writer = threading.Thread(
        target=_writer_thread,
        args=(con, write_queue, stats, pbar, no_dedup),
        daemon=True,
    )
    writer.start()

    t0 = time.time()

    # --- Double-buffered inference loop ---
    # GPU runs predict() on batch N while the main thread collects tiles
    # and builds tensor for batch N+1. PyTorch releases the GIL during
    # CUDA kernels, so a background thread can drive the GPU concurrently.
    infer_executor = ThreadPoolExecutor(max_workers=1)

    batch, batch_patches, sentinel_received = _collect_batch(
        tile_queue, max_batch_tiles, write_queue,
    )
    if batch_patches:
        tensor, meta = build_batch_tensor(batch_patches, device)
    else:
        tensor, meta = None, None

    while batch:
        pbar.write(f"  >> {len(batch)} tiles, {len(batch_patches)} patches")

        infer_future = infer_executor.submit(
            infer_on_tensor, model, tensor, meta, conf, iou,
        )

        if not sentinel_received:
            next_batch, next_patches, sentinel_received = _collect_batch(
                tile_queue, max_batch_tiles, write_queue,
            )
            if next_patches:
                next_tensor, next_meta = build_batch_tensor(next_patches, device)
            else:
                next_tensor, next_meta = None, None
        else:
            next_batch, next_patches = [], []
            next_tensor, next_meta = None, None

        try:
            detections = infer_future.result()
        except (RuntimeError, Exception) as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                log.warning("OOM in double-buffer, falling back to chunked inference")
                detections = run_inference(
                    model, batch_patches, conf=conf, iou=iou, device=device,
                    max_patches_per_call=len(batch_patches) // 2,
                )
            else:
                raise

        det_by_tile: dict[str, list] = defaultdict(list)
        for det in detections:
            det_by_tile[det.tile_id].append(det)
        for tile_id, _patches in batch:
            write_queue.put((tile_id, det_by_tile.get(tile_id, [])))

        batch, batch_patches = next_batch, next_patches
        tensor, meta = next_tensor, next_meta

    infer_executor.shutdown(wait=True)

    # --- Shutdown ---
    write_queue.put(None)
    writer.join()
    pbar.close()
    producer.join()

    elapsed = time.time() - t0
    tiles_processed = stats["tiles_processed"]
    tiles_failed = stats["tiles_failed"]
    total_detections = stats["total_detections"]
    rate = tiles_processed / elapsed if elapsed > 0 else 0

    db_total = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    # Auto-export GPKG
    gpkg_path = output_db.with_suffix(".gpkg")
    rows = con.execute(
        "SELECT tile_id, patch_id, easting, northing, confidence, "
        "bbox_w_m, bbox_h_m, class_id, rgb_source, dsm_source FROM detections",
    ).fetchall()
    con.close()

    if rows:
        import geopandas as gpd
        from shapely.geometry import box

        records = []
        for tile_id, patch_id, e, n, conf_val, w_m, h_m, cls, rgb_src, dsm_src in rows:
            hw, hh = w_m / 2, h_m / 2
            records.append({
                "tile_id": tile_id,
                "patch_id": patch_id,
                "confidence": round(conf_val, 4),
                "bbox_w_m": round(w_m, 2),
                "bbox_h_m": round(h_m, 2),
                "class_id": cls,
                "rgb_source": rgb_src,
                "dsm_source": dsm_src,
                "geometry": box(e - hw, n - hh, e + hw, n + hh),
            })
        gdf = gpd.GeoDataFrame(records, crs="EPSG:2056")
        gdf.to_file(gpkg_path, driver="GPKG", layer="rock_detections")

    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info(f"  Tiles: {tiles_processed} processed, {tiles_failed} failed")
    log.info(f"  Detections: {total_detections} (DB total: {db_total})")
    log.info(f"  Time: {elapsed:.0f}s ({rate:.2f} tiles/s)")
    log.info(f"  Output: {output_db}")
    if rows:
        log.info(f"  GPKG:   {gpkg_path}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

app = typer.Typer(help="Large-scale rock detection on Swisstopo imagery.")


@app.command()
def export_tiles(
    bbox: Optional[str] = typer.Option(None, help="WGS84 bounding box: west,south,east,north"),
    all_switzerland: bool = typer.Option(False, "--all", help="Export all of Switzerland"),
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates (e.g. 2587-1133)"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Cache directory (for STAC cache)"),
    download_threads: int = typer.Option(8, help="Thread count for URL resolution"),
    rgb_year: int = typer.Option(0, "--rgb-year", help="SwissIMAGE year (0=newest)"),
    dsm_year: int = typer.Option(0, "--dsm-year", help="swissALTI3D year (0=newest)"),
) -> None:
    """Export tile list as CSV (coord,rgb_url,dsm_url)."""
    set_stac_cache_dir(cache_dir)
    tile_list = _resolve_tiles_from_cli(
        coords, bbox, all_switzerland, None, None,
        download_threads, rgb_year, dsm_year,
    )
    for coord, rgb_url, dsm_url in tile_list:
        print(f"{coord},{rgb_url},{dsm_url}")
    log.info(f"Exported {len(tile_list)} tiles")


@app.command("fill-cache")
def fill_cache(
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates (e.g. 2587-1133)"),
    bbox: Optional[str] = typer.Option(None, help="WGS84 bounding box: west,south,east,north"),
    all_switzerland: bool = typer.Option(False, "--all", help="Process all of Switzerland"),
    min_elevation: float = typer.Option(1500, help="Skip tiles below this elevation (0=disabled)"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Tile cache directory"),
    cache_gb: float = typer.Option(500.0, help="Max download cache in GB"),
    workers: int = typer.Option(12, help="Number of parallel workers"),
    download_threads: int = typer.Option(8, help="Download threads for URL resolution"),
    rgb_year: int = typer.Option(0, "--rgb-year", help="SwissIMAGE year (0=newest)"),
    dsm_year: int = typer.Option(0, "--dsm-year", help="swissALTI3D year (0=newest)"),
    rgb_url_csv: Optional[Path] = typer.Option(None, "--rgb-urls", help="CSV of SwissIMAGE URLs"),
    dsm_url_csv: Optional[Path] = typer.Option(None, "--dsm-urls", help="CSV of swissALTI3D URLs"),
) -> None:
    """Fill the patch cache without GPU inference.

    Uses the same processing as `run` — cached patches are loaded
    automatically on the next `run` invocation.
    """
    check_gdaldem()
    init_cache(cache_dir, cache_gb)
    set_stac_cache_dir(cache_dir)

    tile_list = _resolve_tiles_from_cli(
        coords, bbox, all_switzerland, rgb_url_csv, dsm_url_csv,
        download_threads, rgb_year, dsm_year,
    )
    if not tile_list:
        log.error("No tiles resolved.")
        raise typer.Exit(code=1)

    log.info(f"Total tile pairs: {len(tile_list)}")

    elev_db = init_db(cache_dir / "elevations.duckdb")
    tiles, n_skipped, n_elev = _resolve_tile_source(
        tile_list, set(), elev_db, min_elevation, download_threads,
    )
    elev_db.close()

    # Filter already-cached patches
    remaining = [t for t in tiles if not _has_cached_patches(t)]
    n_cached = len(tiles) - len(remaining)
    if n_cached:
        log.info(f"Skipping {n_cached} tiles with cached patches")

    if not remaining:
        log.info("All tiles already cached!")
        return

    log.info(f"Tiles to cache: {len(remaining)}")

    cc = get_cache_config()
    init_args = (cc[0], cc[1]) if cc else (None, 0)
    pbar = tqdm(total=len(remaining), desc="Caching", unit="tile")
    done, failed = 0, 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=reinit_session,
        initargs=init_args,
    ) as pool:
        futures = {
            pool.submit(_safe_process, *entry, discard_patches=True): entry
            for entry in remaining
        }
        for future in concurrent.futures.as_completed(futures):
            coord, result = future.result()
            if isinstance(result, Exception):
                failed += 1
                log.warning(f"Tile {coord} failed: {result}")
            else:
                done += 1
            pbar.update(1)
            pbar.set_postfix(ok=done, fail=failed)

    pbar.close()
    log.info(f"Cache fill complete: {done} processed, {failed} failed, {n_cached} already cached")


@app.command()
def run(
    ctx: typer.Context,
    model: Optional[Path] = typer.Option(None, help="Path to YOLO .pt model"),
    output: Path = typer.Option(Path("detections.duckdb"), help="DuckDB output path"),
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates (e.g. 2587-1133)"),
    bbox: Optional[str] = typer.Option(None, help="WGS84 bounding box: west,south,east,north"),
    all_switzerland: bool = typer.Option(False, "--all", help="Process all of Switzerland"),
    min_elevation: float = typer.Option(1500, help="Skip tiles below this elevation in meters (0=disabled)"),
    device: str = typer.Option("auto", help="PyTorch device (auto, cuda:0, cpu, etc.)"),
    download_threads: int = typer.Option(8, help="Download thread count"),
    max_tiles: int = typer.Option(0, help="Limit tiles (0=all)"),
    conf: float = typer.Option(0.10, help="Confidence threshold"),
    iou: float = typer.Option(0.70, help="IoU threshold"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Tile cache directory"),
    cache_gb: float = typer.Option(500.0, help="Max download cache in GB (0 to disable)"),
    max_batch_tiles: int = typer.Option(16, help="Max tiles per GPU batch"),
    queue_size: int = typer.Option(0, help="Producer-consumer queue size (0=auto)"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable spatial deduplication"),
    save_patches: bool = typer.Option(False, "--save-patches", help="Cache fused patches to disk (uses ~1TB for all of Switzerland)"),
    rgb_year: int = typer.Option(0, "--rgb-year", help="SwissIMAGE year (0=newest)"),
    dsm_year: int = typer.Option(0, "--dsm-year", help="swissALTI3D year (0=newest)"),
    rgb_url_csv: Optional[Path] = typer.Option(None, "--rgb-urls", help="CSV of SwissIMAGE URLs"),
    dsm_url_csv: Optional[Path] = typer.Option(None, "--dsm-urls", help="CSV of swissALTI3D URLs"),
) -> None:
    """Run the full rock detection pipeline on Swisstopo tiles."""
    if model is None:
        print(ctx.get_help())
        raise typer.Exit()

    check_gdaldem()
    check_gdalbuildvrt()
    init_cache(cache_dir, cache_gb)
    set_stac_cache_dir(cache_dir)

    tile_list = _resolve_tiles_from_cli(
        coords, bbox, all_switzerland, rgb_url_csv, dsm_url_csv,
        download_threads, rgb_year, dsm_year,
    )
    if not tile_list:
        log.error("No tiles resolved.")
        raise typer.Exit(code=1)

    if max_tiles > 0:
        tile_list = tile_list[:max_tiles]

    if not model.exists():
        log.error(f"Model not found: {model}")
        raise typer.Exit(code=1)

    # Fail fast: verify GPU before slow network/elevation work
    import torch
    from ultralytics import YOLO
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        log.error(f"Device {device} requested but no GPU available.")
        raise typer.Exit(code=1)
    log.info(f"Device: {device}")
    test_model = YOLO(model)
    if device != "cpu":
        test_model.to(device)
        del test_model
        torch.cuda.empty_cache()
    else:
        del test_model
    log.info("Model and GPU check passed")

    log.info(f"Total tile pairs: {len(tile_list)}")

    # Resolve: checkpoint filter + elevation + neighbor URLs
    con = init_db(output)
    done_coords = {t.replace("_", "-") for t in get_processed_tiles(con)}

    tiles, n_skipped, n_elev = _resolve_tile_source(
        tile_list, done_coords, con, min_elevation, download_threads,
    )
    con.close()

    if not tiles:
        log.info("All tiles already processed!")
        return

    if save_patches:
        cached_count = sum(1 for t in tiles if _has_cached_patches(t))
        if cached_count:
            log.info(f"Patch cache: {cached_count}/{len(tiles)} tiles cached (will skip preprocessing)")

    run_pipeline(
        model_path=model,
        output_db=output,
        tiles=tiles,
        device=device,
        workers=download_threads,
        conf=conf,
        iou=iou,
        max_batch_tiles=max_batch_tiles,
        queue_maxsize=queue_size,
        no_dedup=no_dedup,
        cache_patches=save_patches,
    )
