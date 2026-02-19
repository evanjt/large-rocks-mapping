"""Pipeline orchestrator and CLI."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from nationwide.db import (
    get_processed_tiles,
    init_db,
    mark_tile_done,
    write_detections,
)
from nationwide.processing import (
    dedup_detections,
    get_cache_config,
    init_cache,
    process_tile,
    reinit_session,
    run_inference,
)
from nationwide.spatial import (
    SWITZERLAND_BBOX,
    query_stac_bbox,
    resolve_batch,
    set_stac_cache_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Producer-consumer ────────────────────────────────────────────────────────

def _safe_process(
    coord: str, rgb_url: str, dsm_url: str, min_elevation: float = 0,
) -> tuple[str, list | Exception]:
    """Process a single tile, catching exceptions for graceful error handling.

    Module-level function so it can be pickled by ProcessPoolExecutor.
    """
    try:
        return (coord, process_tile(coord, rgb_url, dsm_url, min_elevation=min_elevation))
    except Exception as exc:
        return (coord, exc)


def _producer_thread(
    tiles: list[tuple[str, str, str]],
    tile_queue: queue.Queue,
    download_threads: int,
    min_elevation: float = 0,
) -> None:
    """Download+preprocess tiles in parallel processes, put results in queue.

    Uses ProcessPoolExecutor for true CPU parallelism — avoids GIL
    contention during rasterio TIFF decompression and numpy processing.
    The bounded queue's put() blocks when full, providing backpressure.
    """
    # Pass cache config so worker processes can reconstruct the tile cache
    cc = get_cache_config()
    init_args = (cc[0], cc[1]) if cc else (None, 0)

    with ProcessPoolExecutor(
        max_workers=download_threads,
        initializer=reinit_session,
        initargs=init_args,
    ) as pool:
        src = iter(tiles)
        pending: dict[concurrent.futures.Future, bool] = {}

        # Seed pool with initial work
        for _ in range(download_threads):
            try:
                c, r, d = next(src)
                pending[pool.submit(_safe_process, c, r, d, min_elevation)] = True
            except StopIteration:
                break

        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            future = done.pop()
            del pending[future]
            tile_queue.put(future.result())  # Blocks when queue full

            # Refill from source
            try:
                c, r, d = next(src)
                pending[pool.submit(_safe_process, c, r, d, min_elevation)] = True
            except StopIteration:
                pass

    tile_queue.put(None)  # Sentinel: all tiles processed


def run_pipeline(
    model_path: Path,
    output_db: Path,
    tile_source: list[tuple[str, str, str]],
    device: str = "cuda:0",
    download_threads: int = 8,
    conf: float = 0.10,
    iou: float = 0.40,
    min_elevation: float = 0,
    max_batch_tiles: int = 0,
    queue_maxsize: int = 0,
    enable_monitor: bool = True,
) -> None:
    """Main pipeline: download+preprocess -> infer -> dedup -> store.

    Uses a bounded producer-consumer queue to limit memory usage:
    download_threads workers preprocess tiles, a queue buffers results,
    and the main thread runs batched GPU inference.
    Checkpoints progress to DuckDB for resume on restart.

    If max_batch_tiles=0 (default), auto-detects optimal GPU batch size.
    If queue_maxsize=0 (default), uses download_threads * 3.
    """
    log.info(f"Total tile pairs: {len(tile_source)}")

    if not tile_source:
        log.error("No tiles to process.")
        raise typer.Exit(code=1)

    # Load model
    log.info(f"Loading model from {model_path} on {device} ...")
    import torch
    from ultralytics import YOLO
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    if device != "cpu":
        model.to(device)

    # Auto-tune GPU batch size
    if max_batch_tiles <= 0 and device != "cpu":
        from nationwide.benchmark import probe_gpu_max_batch
        gpu_result = probe_gpu_max_batch(model, device)
        max_batch_tiles = gpu_result.max_batch_tiles
        log.info(f"Auto-tuned: max_batch_tiles={max_batch_tiles} ({gpu_result.max_patches} patches)")
    elif max_batch_tiles <= 0:
        max_batch_tiles = 4  # Conservative CPU default

    # Dynamic queue sizing
    if queue_maxsize <= 0:
        queue_maxsize = download_threads * 3

    # Init DB (includes processed_tiles table for checkpoint/resume)
    con = init_db(output_db)

    # Checkpoint: skip already-processed tiles
    done_set = get_processed_tiles(con)
    remaining = [
        (c, r, d) for c, r, d in tile_source
        if c.replace("-", "_") not in done_set
    ]
    skipped = len(tile_source) - len(remaining)
    if skipped:
        log.info(f"Checkpoint: skipping {skipped} already-processed tiles")
    log.info(f"Tiles to process: {len(remaining)}")

    if not remaining:
        log.info("All tiles already processed!")
        con.close()
        return

    # Bounded queue: buffer tiles for producer-consumer backpressure
    tile_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)

    # Start resource monitor
    monitor = None
    if enable_monitor:
        from nationwide.monitor import ResourceMonitor
        monitor = ResourceMonitor(tile_queue=tile_queue)
        monitor.start()

    producer = threading.Thread(
        target=_producer_thread,
        args=(remaining, tile_queue, download_threads, min_elevation),
        daemon=True,
    )
    producer.start()

    total_detections = 0
    tiles_processed = 0
    tiles_failed = 0
    t0 = time.time()

    # Timing accumulators
    t_wait_total = 0.0
    t_infer_total = 0.0
    t_post_total = 0.0
    n_batches = 0
    total_patches_inferred = 0

    pbar = tqdm(total=len(remaining), desc="Processing tiles", unit="tile")

    MAX_BATCH_TILES = max_batch_tiles
    max_patches = MAX_BATCH_TILES * 16  # patches per tile

    while True:
        # ── Accumulate a batch of tiles ──────────────────────────────
        batch: list[tuple[str, list]] = []  # (tile_id, patches)
        batch_patches: list[tuple] = []     # flat list for run_inference
        tiles_in_batch = 0

        # First tile: block until work arrives (or sentinel)
        t_wait_start = time.time()
        item = tile_queue.get()
        t_wait_total += time.time() - t_wait_start
        if item is None:
            break

        # Process first item + drain more without blocking
        pending_items = [item]
        while tiles_in_batch < MAX_BATCH_TILES:
            if not pending_items:
                # Try to grab more without stalling the GPU
                try:
                    nxt = tile_queue.get(block=False)
                except queue.Empty:
                    break
                if nxt is None:
                    pending_items.append(nxt)
                    break
                pending_items.append(nxt)
                continue

            item = pending_items.pop(0)
            if item is None:
                # Put sentinel back so outer loop exits
                pending_items.insert(0, None)
                break

            coord, result = item
            tile_id = coord.replace("-", "_")

            if isinstance(result, Exception):
                tiles_failed += 1
                log.warning(f"Tile {coord} failed: {result}")
                pbar.update(1)
                continue

            patches = result
            if not patches:
                mark_tile_done(con, tile_id, 0)
                tiles_processed += 1
                pbar.update(1)
                continue

            batch.append((tile_id, patches))
            batch_patches.extend(patches)
            tiles_in_batch += 1

        if not batch:
            # Check if we hit the sentinel during accumulation
            if pending_items and pending_items[0] is None:
                break
            continue

        # ── Run inference on accumulated patches ─────────────────────
        t_infer_start = time.time()
        detections = run_inference(
            model, batch_patches, conf=conf, iou=iou, device=device,
            max_patches_per_call=max_patches,
        )
        t_infer_total += time.time() - t_infer_start
        total_patches_inferred += len(batch_patches)
        del batch_patches

        # ── Split detections by tile_id, dedup + write per tile ──────
        t_post_start = time.time()
        det_by_tile: dict[str, list] = defaultdict(list)
        for det in detections:
            det_by_tile[det.tile_id].append(det)

        for tile_id, patches in batch:
            del patches  # Free patch memory
            tile_dets = dedup_detections(det_by_tile.get(tile_id, []), distance_m=7.5)
            n = write_detections(con, tile_dets)
            mark_tile_done(con, tile_id, n)
            total_detections += n
            tiles_processed += 1
            pbar.update(1)
            pbar.set_postfix(dets=total_detections, ok=tiles_processed, fail=tiles_failed)
        t_post_total += time.time() - t_post_start
        n_batches += 1

        # If sentinel was found during batch accumulation, exit
        if pending_items and pending_items[0] is None:
            break

    pbar.close()
    producer.join()

    # Stop resource monitor
    if monitor is not None:
        monitor.stop()

    elapsed = time.time() - t0
    rate = tiles_processed / elapsed if elapsed > 0 else 0

    db_total = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    avg_batch = tiles_processed / n_batches if n_batches else 0
    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info(f"  Tiles processed: {tiles_processed}")
    log.info(f"  Tiles failed:    {tiles_failed}")
    log.info(f"  Tiles skipped:   {skipped}")
    log.info(f"  Total detections: {total_detections}")
    log.info(f"  DB total:        {db_total}")
    log.info(f"  Time:            {elapsed:.0f}s ({rate:.2f} tiles/s)")
    log.info(f"  Output:          {output_db}")
    log.info(f"  Batch config:    max_batch_tiles={MAX_BATCH_TILES} ({max_patches} patches), queue={queue_maxsize}")
    log.info(f"  Consumer timing ({n_batches} batches, avg {avg_batch:.1f} tiles/batch):")
    log.info(f"    Queue wait:    {t_wait_total:.1f}s ({t_wait_total/elapsed*100:.0f}%)")
    log.info(f"    Inference:     {t_infer_total:.1f}s ({t_infer_total/elapsed*100:.0f}%) [{total_patches_inferred} patches]")
    log.info(f"    Dedup+write:   {t_post_total:.1f}s ({t_post_total/elapsed*100:.0f}%)")

    if monitor is not None:
        rs = monitor.summary()
        log.info(f"  Resource usage ({rs.n_samples} samples over {rs.duration_s:.0f}s):")
        log.info(f"    GPU util:      avg {rs.gpu_util_avg:.0f}%, max {rs.gpu_util_max:.0f}%")
        log.info(f"    GPU VRAM:      max {rs.gpu_vram_max_mb} / {rs.gpu_vram_total_mb} MB")
        log.info(f"    CPU:           avg {rs.cpu_avg:.0f}%, max {rs.cpu_max:.0f}%")
        log.info(f"    RAM:           max {rs.ram_max_mb} / {rs.ram_total_mb} MB")
        log.info(f"    Network:       {rs.net_total_mb:.0f} MB total, {rs.net_avg_mbps:.0f} Mbps avg")
        log.info(f"    Queue depth:   avg {rs.queue_avg:.1f}, max {rs.queue_max}")

    log.info("=" * 60)

    con.close()


# ── Typer CLI ────────────────────────────────────────────────────────────────

app = typer.Typer(help="Nationwide rock detection pipeline.")


@app.command()
def run(
    model: Path = typer.Option(..., help="Path to YOLO .pt model"),
    output: Path = typer.Option(Path("detections.duckdb"), help="DuckDB output path"),
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates (e.g. 2587-1133)"),
    bbox: Optional[str] = typer.Option(None, help="WGS84 bounding box: west,south,east,north"),
    all_switzerland: bool = typer.Option(False, "--all", help="Process all of Switzerland"),
    min_elevation: float = typer.Option(0, help="Skip tiles below this elevation in meters (0=disabled, e.g. 1500)"),
    device: str = typer.Option("cuda:0", help="PyTorch device"),
    download_threads: int = typer.Option(8, help="Download thread count"),
    max_tiles: int = typer.Option(0, help="Limit tiles (0=all)"),
    conf: float = typer.Option(0.10, help="Confidence threshold"),
    iou: float = typer.Option(0.40, help="IoU threshold"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Tile cache directory"),
    cache_gb: float = typer.Option(10.0, help="Max tile cache size in GB (0 to disable)"),
    max_batch_tiles: int = typer.Option(0, help="Max tiles per GPU batch (0=auto-detect)"),
    queue_size: int = typer.Option(0, help="Producer-consumer queue size (0=auto: download_threads*3)"),
    no_monitor: bool = typer.Option(False, help="Disable resource monitoring"),
) -> None:
    """Run the nationwide rock detection pipeline."""
    init_cache(cache_dir, cache_gb)
    set_stac_cache_dir(cache_dir)

    # Build tile source from the chosen input method
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
        enable_monitor=not no_monitor,
    )


@app.command()
def benchmark(
    model: Path = typer.Option(..., help="Path to YOLO .pt model"),
    device: str = typer.Option("cuda:0", help="PyTorch device"),
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates for network test (e.g. 2587-1133)"),
    bbox: Optional[str] = typer.Option(None, help="WGS84 bbox for network test (provides more URLs)"),
    download_threads: int = typer.Option(8, help="Current download thread count (for recommendation)"),
) -> None:
    """Run GPU and optional network benchmarks, print recommended parameters."""
    if not model.exists():
        log.error(f"Model not found: {model}")
        raise typer.Exit(code=1)

    import torch
    from ultralytics import YOLO
    from nationwide.benchmark import (
        measure_download_throughput,
        probe_gpu_max_batch,
        recommend_params,
    )

    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading model from {model} on {device} ...")
    yolo_model = YOLO(model)
    if device != "cpu":
        yolo_model.to(device)

    gpu_result = probe_gpu_max_batch(yolo_model, device)

    net_result = None
    if coords:
        url_map = resolve_batch(list(coords), threads=4)
        # Collect RGB URLs (largest files) for throughput test
        rgb_urls = [r for _, (r, _) in url_map.items()]
        if rgb_urls:
            net_result = measure_download_throughput(rgb_urls)
    elif bbox:
        tile_list = query_stac_bbox(bbox)
        rgb_urls = [r for _, r, _ in tile_list[:40]]
        if rgb_urls:
            net_result = measure_download_throughput(rgb_urls)

    params = recommend_params(gpu_result, net_result)

    print()
    print("=" * 50)
    print("  Benchmark Results")
    print("=" * 50)
    print(f"  GPU max batch:     {gpu_result.max_patches} patches ({gpu_result.max_batch_tiles} tiles)")
    print(f"  GPU peak VRAM:     {gpu_result.peak_vram_mb} MB")
    if net_result:
        print(f"  Per-thread speed:  {net_result.mbps_per_thread} MB/s")
        print(f"  Best thread count: {net_result.recommended_threads}")
    print()
    print("  Recommended parameters:")
    print(f"    --max-batch-tiles {params.max_batch_tiles}")
    print(f"    --download-threads {params.download_threads}")
    print(f"    --queue-size {params.queue_maxsize}")
    print("=" * 50)


@app.command()
def export(
    input_db: Path = typer.Option(..., "--input", help="DuckDB database path"),
    output: Path = typer.Option(..., help="Output path (.gpkg or .geojson)"),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence filter"),
    geometry: str = typer.Option("polygon", help="Geometry type: 'point' or 'polygon'"),
) -> None:
    """Export detections from DuckDB to GeoPackage or GeoJSON (EPSG:2056).

    Format is detected from the output file extension:
      .gpkg     → GeoPackage (default, recommended)
      .geojson  → GeoJSON
    """
    import duckdb

    if not input_db.exists():
        log.error(f"Database not found: {input_db}")
        raise typer.Exit(code=1)

    ext = output.suffix.lower()
    if ext not in (".gpkg", ".geojson", ".json"):
        log.error(f"Unsupported output format '{ext}'. Use .gpkg or .geojson")
        raise typer.Exit(code=1)

    con = duckdb.connect(str(input_db), read_only=True)
    rows = con.execute(
        "SELECT tile_id, patch_id, easting, northing, confidence, "
        "bbox_w_m, bbox_h_m, class_id FROM detections "
        "WHERE confidence >= ? ORDER BY confidence DESC",
        [min_confidence],
    ).fetchall()
    con.close()

    if not rows:
        log.error("No detections match the filter.")
        raise typer.Exit(code=1)

    use_polygon = geometry.lower().startswith("poly")
    output.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".gpkg":
        _export_gpkg(rows, output, use_polygon)
    else:
        _export_geojson(rows, output, use_polygon)

    log.info(f"Exported {len(rows)} detections to {output} ({geometry}, EPSG:2056)")


def _export_gpkg(
    rows: list[tuple], output: Path, use_polygon: bool,
) -> None:
    """Write detections to a GeoPackage file."""
    import geopandas as gpd
    from shapely.geometry import Point, box

    records = []
    for tile_id, patch_id, e, n, conf, w_m, h_m, cls in rows:
        if use_polygon:
            hw, hh = w_m / 2, h_m / 2
            geom = box(e - hw, n - hh, e + hw, n + hh)
        else:
            geom = Point(e, n)
        records.append({
            "tile_id": tile_id,
            "patch_id": patch_id,
            "confidence": round(conf, 4),
            "bbox_w_m": round(w_m, 2),
            "bbox_h_m": round(h_m, 2),
            "class_id": cls,
            "geometry": geom,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:2056")
    gdf.to_file(output, driver="GPKG", layer="rock_detections")


def _export_geojson(
    rows: list[tuple], output: Path, use_polygon: bool,
) -> None:
    """Write detections to a GeoJSON file."""
    features = []
    for tile_id, patch_id, e, n, conf, w_m, h_m, cls in rows:
        if use_polygon:
            hw, hh = w_m / 2, h_m / 2
            coords = [[
                [e - hw, n - hh],
                [e + hw, n - hh],
                [e + hw, n + hh],
                [e - hw, n + hh],
                [e - hw, n - hh],
            ]]
            geom = {"type": "Polygon", "coordinates": coords}
        else:
            geom = {"type": "Point", "coordinates": [e, n]}

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "tile_id": tile_id,
                "patch_id": patch_id,
                "confidence": round(conf, 4),
                "bbox_w_m": round(w_m, 2),
                "bbox_h_m": round(h_m, 2),
                "class_id": cls,
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "rock_detections",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::2056"},
        },
        "features": features,
    }

    with open(output, "w") as f:
        json.dump(geojson, f)


