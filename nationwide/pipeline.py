import concurrent.futures
import logging
import os
import queue
import threading
import time
import typer
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from nationwide.db import (
    get_processed_tiles,
    init_db,
    mark_tile_done,
    write_detections,
)
from nationwide.cache import get_cache_config, init_cache, set_stac_cache_dir
from utils.constants import SWITZERLAND_BBOX, TILE_SIZE_PX
from nationwide.processing import (
    build_batch_tensor,
    check_elevation,
    dedup_detections,
    infer_on_tensor,
    process_tile,
    reinit_session,
    run_inference,
)
from nationwide.spatial import query_stac_bbox, resolve_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _safe_process(
    coord: str, rgb_url: str, dsm_url: str,
) -> tuple[str, list | Exception]:
    """Process single tile, catching exceptions for graceful error handling"""
    try:
        return (coord, process_tile(coord, rgb_url, dsm_url))
    except Exception as exc:
        return (coord, exc)


def _producer_thread(
    tiles: list[tuple[str, str, str]],
    tile_queue: queue.Queue,
    download_threads: int,
) -> None:
    """Download+preprocess tiles in parallel processes, put results in queue.

    Uses ProcessPoolExecutor for true CPU parallelism — avoids GIL
    contention during rasterio TIFF decompression and numpy processing.
    """
    cc = get_cache_config()
    init_args = (cc[0], cc[1]) if cc else (None, 0)

    with ProcessPoolExecutor(
        max_workers=download_threads,
        initializer=reinit_session,
        initargs=init_args,
    ) as pool:
        src = iter(tiles)
        pending: dict[concurrent.futures.Future, bool] = {}

        for _ in range(download_threads):
            try:
                c, r, d = next(src)
                pending[pool.submit(_safe_process, c, r, d)] = True
            except StopIteration:
                break

        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            future = done.pop()
            del pending[future]
            tile_queue.put(future.result())

            try:
                c, r, d = next(src)
                pending[pool.submit(_safe_process, c, r, d)] = True
            except StopIteration:
                pass

    tile_queue.put(None)  # Sentinel


def _writer_thread(
    con,
    write_queue: queue.Queue,
    stats: dict,
    pbar,
) -> None:
    """Background thread: dedup + write detections + checkpoint to DuckDB."""
    while True:
        item = write_queue.get()
        if item is None:
            break
        tile_id, detections = item
        try:
            tile_dets = dedup_detections(detections, distance_m=7.5)
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
    batch_timeout: float,
    stats: dict,
    pbar,
    write_queue: queue.Queue,
) -> tuple[list[tuple[str, list]], list[tuple], bool]:
    """Drain up to max_tiles from queue. Returns (batch, patches, sentinel_received).

    Blocks on the first tile, then drains with timeout for the rest.
    Handles failures (log + update stats) and empty tiles (send to writer).
    """
    batch: list[tuple[str, list]] = []
    batch_patches: list[tuple] = []
    tiles_in_batch = 0
    sentinel_received = False

    # Block indefinitely for the first tile
    first = tile_queue.get()
    if first is None:
        return batch, batch_patches, True

    pending = [first]

    while tiles_in_batch < max_tiles:
        # Drain pending items
        while pending and tiles_in_batch < max_tiles:
            item = pending.pop(0)
            if item is None:
                sentinel_received = True
                break

            coord, result = item
            tile_id = coord.replace("-", "_")

            if isinstance(result, Exception):
                stats["tiles_failed"] += 1
                log.warning(f"Tile {coord} failed: {result}")
                pbar.update(1)
                pbar.set_postfix(
                    dets=stats["total_detections"],
                    ok=stats["tiles_processed"],
                    fail=stats["tiles_failed"],
                )
                continue

            patches = result
            if not patches:
                write_queue.put((tile_id, []))
                continue

            batch.append((tile_id, patches))
            batch_patches.extend(patches)
            tiles_in_batch += 1

        if sentinel_received or tiles_in_batch >= max_tiles:
            break

        # Wait for more tiles up to batch_timeout
        try:
            nxt = tile_queue.get(timeout=batch_timeout)
        except queue.Empty:
            break
        pending.append(nxt)

    return batch, batch_patches, sentinel_received


def run_pipeline(
    model_path: Path,
    output_db: Path,
    tile_source: list[tuple[str, str, str]],
    device: str = "cuda:0",
    download_threads: int = 8,
    conf: float = 0.10,
    iou: float = 0.40,
    min_elevation: float = 0,
    max_batch_tiles: int = 8,
    queue_maxsize: int = 0,
    batch_timeout: float = 0.5,
) -> None:
    """Main pipeline: download+preprocess -> infer -> dedup -> store.

    Uses a bounded producer-consumer queue: download_threads workers
    preprocess tiles, a queue buffers results, and the main thread
    runs batched GPU inference. Checkpoints to DuckDB for resume.
    """
    log.info(f"Total tile pairs: {len(tile_source)}")

    if not tile_source:
        log.error("No tiles to process.")
        raise typer.Exit(code=1)

    if max_batch_tiles <= 0:
        max_batch_tiles = 8

    if queue_maxsize <= 0:
        queue_maxsize = download_threads * 3

    # --- 1. Init DB, checkpoint filter ---
    con = init_db(output_db)

    done_set = get_processed_tiles(con)
    remaining = [
        (c, r, d) for c, r, d in tile_source
        if c.replace("-", "_") not in done_set
    ]
    skipped = len(tile_source) - len(remaining)
    if skipped:
        log.info(f"Checkpoint: skipping {skipped} already-processed tiles")

    # --- 2. Elevation pre-filter ---
    elevation_skipped = 0
    if min_elevation > 0 and remaining:
        from concurrent.futures import ThreadPoolExecutor

        log.info(f"Checking elevations for {len(remaining)} tiles ...")

        def _elev_ok(tile: tuple[str, str, str]) -> bool:
            try:
                return check_elevation(tile[2], min_elevation)
            except Exception:
                return True  # keep on error, let pipeline handle it

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            passes = list(tqdm(
                pool.map(_elev_ok, remaining),
                total=len(remaining), desc="Elevation filter", unit="tile",
            ))
        before = len(remaining)
        remaining = [t for t, ok in zip(remaining, passes) if ok]
        elevation_skipped = before - len(remaining)
        log.info(f"Eliminated {elevation_skipped} tiles below {min_elevation:.0f}m")

    log.info(f"Tiles to process: {len(remaining)}")

    if not remaining:
        log.info("All tiles already processed!")
        con.close()
        return

    # --- 3. Start producer (downloads tiles while model loads) ---
    tile_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)

    producer = threading.Thread(
        target=_producer_thread,
        args=(remaining, tile_queue, download_threads),
        daemon=True,
    )
    producer.start()

    # --- 4. Load YOLO model ---
    log.info(f"Loading model from {model_path} on {device} ...")
    import torch
    from ultralytics import YOLO
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    if device != "cpu":
        model.to(device)

    # --- 5. GPU warmup ---
    if device != "cpu":
        log.info("Warming up GPU ...")
        dummy = torch.zeros(1, 3, TILE_SIZE_PX, TILE_SIZE_PX, device=device)
        model.predict(
            source=dummy, conf=conf, iou=iou, imgsz=TILE_SIZE_PX,
            save=False, verbose=False,
        )
        del dummy
        torch.cuda.empty_cache()

    # --- 6. Start background DB writer ---
    stats = {"total_detections": 0, "tiles_processed": 0, "tiles_failed": 0}
    pbar = tqdm(total=len(remaining), desc="Processing tiles", unit="tile")
    write_queue: queue.Queue = queue.Queue()

    writer = threading.Thread(
        target=_writer_thread,
        args=(con, write_queue, stats, pbar),
        daemon=True,
    )
    writer.start()

    t0 = time.time()

    # --- 7. Double-buffered consumer loop ---
    # While GPU runs predict() on batch N, the main thread collects tiles
    # and builds the tensor for batch N+1.  PyTorch releases the GIL during
    # CUDA kernels, so a background thread can drive the GPU concurrently.
    from concurrent.futures import ThreadPoolExecutor as _InferPool

    infer_executor = _InferPool(max_workers=1)

    # Prefetch first batch
    batch, batch_patches, sentinel_received = _collect_batch(
        tile_queue, max_batch_tiles, batch_timeout, stats, pbar, write_queue,
    )
    if batch_patches:
        tensor, meta = build_batch_tensor(batch_patches, device)
    else:
        tensor, meta = None, None

    while batch:
        pbar.write(f"  >> Batch: {len(batch)} tiles, {len(batch_patches)} patches")

        # Submit inference to background thread (GIL released during CUDA)
        infer_future = infer_executor.submit(
            infer_on_tensor, model, tensor, meta, conf, iou,
        )

        # While GPU works: collect + prepare next batch on CPU
        if not sentinel_received:
            next_batch, next_patches, sentinel_received = _collect_batch(
                tile_queue, max_batch_tiles, batch_timeout,
                stats, pbar, write_queue,
            )
            if next_patches:
                next_tensor, next_meta = build_batch_tensor(next_patches, device)
            else:
                next_tensor, next_meta = None, None
        else:
            next_batch, next_patches = [], []
            next_tensor, next_meta = None, None

        # Wait for inference result
        try:
            detections = infer_future.result()
        except (RuntimeError, Exception) as exc:
            if "out of memory" in str(exc).lower():
                import torch
                torch.cuda.empty_cache()
                log.warning("OOM in double-buffer, falling back to run_inference")
                detections = run_inference(
                    model, batch_patches, conf=conf, iou=iou, device=device,
                    max_patches_per_call=len(batch_patches) // 2,
                )
            else:
                raise

        # Distribute detections to writer by tile
        det_by_tile: dict[str, list] = defaultdict(list)
        for det in detections:
            det_by_tile[det.tile_id].append(det)

        for tile_id, patches in batch:
            del patches
            write_queue.put((tile_id, det_by_tile.get(tile_id, [])))

        # Swap: next becomes current
        batch, batch_patches = next_batch, next_patches
        tensor, meta = next_tensor, next_meta

    infer_executor.shutdown(wait=False)

    # --- 8. Shutdown ---
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
        "bbox_w_m, bbox_h_m, class_id FROM detections",
    ).fetchall()
    con.close()

    if rows:
        import geopandas as gpd
        from shapely.geometry import box

        records = []
        for tile_id, patch_id, e, n, conf_val, w_m, h_m, cls in rows:
            hw, hh = w_m / 2, h_m / 2
            records.append({
                "tile_id": tile_id,
                "patch_id": patch_id,
                "confidence": round(conf_val, 4),
                "bbox_w_m": round(w_m, 2),
                "bbox_h_m": round(h_m, 2),
                "class_id": cls,
                "geometry": box(e - hw, n - hh, e + hw, n + hh),
            })
        gdf = gpd.GeoDataFrame(records, crs="EPSG:2056")
        gdf.to_file(gpkg_path, driver="GPKG", layer="rock_detections")

    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info(f"  Tiles: {tiles_processed} processed, {tiles_failed} failed, {skipped} checkpointed, {elevation_skipped} below elevation")
    log.info(f"  Detections: {total_detections} (DB total: {db_total})")
    log.info(f"  Time: {elapsed:.0f}s ({rate:.2f} tiles/s)")
    log.info(f"  Output: {output_db}")
    if rows:
        log.info(f"  GPKG:   {gpkg_path}")
    log.info("=" * 60)


app = typer.Typer(help="Nationwide rock detection pipeline.")


@app.command()
def run(
    model: Path = typer.Option(..., help="Path to YOLO .pt model"),
    output: Path = typer.Option(Path("detections.duckdb"), help="DuckDB output path"),
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates (e.g. 2587-1133)"),
    bbox: Optional[str] = typer.Option(None, help="WGS84 bounding box: west,south,east,north"),
    all_switzerland: bool = typer.Option(False, "--all", help="Process all of Switzerland"),
    min_elevation: float = typer.Option(1500, help="Skip tiles below this elevation in meters (0=disabled)"),
    device: str = typer.Option("cuda:0", help="PyTorch device"),
    download_threads: int = typer.Option(8, help="Download thread count"),
    max_tiles: int = typer.Option(0, help="Limit tiles (0=all)"),
    conf: float = typer.Option(0.10, help="Confidence threshold"),
    iou: float = typer.Option(0.40, help="IoU threshold"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Tile cache directory"),
    cache_gb: float = typer.Option(10.0, help="Max tile cache size in GB (0 to disable)"),
    max_batch_tiles: int = typer.Option(8, help="Max tiles per GPU batch"),
    queue_size: int = typer.Option(0, help="Producer-consumer queue size (0=auto: download_threads*3)"),
    batch_timeout: float = typer.Option(0.5, help="Seconds to wait for more tiles before running inference (0=no wait)"),
) -> None:
    """Run the nationwide rock detection pipeline."""
    init_cache(cache_dir, cache_gb)
    set_stac_cache_dir(cache_dir)

    if all_switzerland:
        bbox = SWITZERLAND_BBOX
    if bbox:
        tile_list = query_stac_bbox(bbox)
    elif coords:
        log.info("Resolving tile URLs ...")
        url_map = resolve_batch(list(coords), threads=download_threads)
        log.info(f"Resolved {len(url_map)}/{len(coords)} tile URL pairs")
        tile_list = [(c, r, d) for c, (r, d) in url_map.items()]
    else:
        log.error("No tile source. Pass --coords or --bbox.")
        raise typer.Exit(code=1)

    if not tile_list:
        log.error("No tiles resolved. Check network, coordinates, or bbox.")
        raise typer.Exit(code=1)

    if max_tiles > 0:
        tile_list = tile_list[:max_tiles]

    if not model.exists():
        log.error(f"Model not found: {model}. Place a .pt file in models/.")
        raise typer.Exit(code=1)

    run_pipeline(
        model_path=model,
        output_db=output,
        tile_source=tile_list,
        device=device,
        download_threads=download_threads,
        conf=conf,
        iou=iou,
        min_elevation=min_elevation,
        max_batch_tiles=max_batch_tiles,
        queue_maxsize=queue_size,
        batch_timeout=batch_timeout,
    )
