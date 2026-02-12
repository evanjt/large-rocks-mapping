"""Pipeline orchestrator and CLI."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    init_cache,
    process_tile,
    run_inference,
)
from nationwide.spatial import (
    query_stac_bbox,
    resolve_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Producer-consumer ────────────────────────────────────────────────────────

def _producer_thread(
    tiles: list[tuple[str, str, str]],
    tile_queue: queue.Queue,
    download_threads: int,
    min_elevation: float = 0,
) -> None:
    """Download+preprocess tiles in parallel, put results in bounded queue.

    Uses a ThreadPoolExecutor with `download_threads` workers. Submits
    tiles one-for-one as each completes, so at most `download_threads`
    tiles are in flight. The bounded queue's put() blocks when full,
    providing backpressure to the producer.
    """

    def _safe_process(
        coord: str, rgb_url: str, dsm_url: str,
    ) -> tuple[str, list | Exception]:
        try:
            return (coord, process_tile(coord, rgb_url, dsm_url, min_elevation=min_elevation))
        except Exception as exc:
            return (coord, exc)

    with ThreadPoolExecutor(max_workers=download_threads) as pool:
        src = iter(tiles)
        pending: dict[concurrent.futures.Future, bool] = {}

        # Seed pool with initial work
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
            # Process one completed future at a time to minimize peak memory
            future = done.pop()
            del pending[future]
            tile_queue.put(future.result())  # Blocks when queue full

            # Refill from source
            try:
                c, r, d = next(src)
                pending[pool.submit(_safe_process, c, r, d)] = True
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
) -> None:
    """Main pipeline: download+preprocess -> infer -> dedup -> store.

    Uses a bounded producer-consumer queue to limit memory usage:
    download_threads workers preprocess tiles, a queue buffers results
    (maxsize=2*download_threads), and the main thread runs batched GPU
    inference (up to 4 tiles / 64 patches per call).
    Checkpoints progress to DuckDB for resume on restart.
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

    # Bounded queue: buffer enough tiles so all producer workers can deposit
    # without blocking. With 8 threads, maxsize=16 → ~300 MB peak memory.
    tile_queue: queue.Queue = queue.Queue(maxsize=download_threads * 2)

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

    pbar = tqdm(total=len(remaining), desc="Processing tiles", unit="tile")

    MAX_BATCH_TILES = 4  # Accumulate up to 4 tiles (64 patches) per GPU call

    while True:
        # ── Accumulate a batch of tiles ──────────────────────────────
        batch: list[tuple[str, list]] = []  # (tile_id, patches)
        batch_patches: list[tuple] = []     # flat list for run_inference
        tiles_in_batch = 0

        # First tile: block until work arrives (or sentinel)
        item = tile_queue.get()
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
        detections = run_inference(model, batch_patches, conf=conf, iou=iou, device=device)
        del batch_patches

        # ── Split detections by tile_id, dedup + write per tile ──────
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

        # If sentinel was found during batch accumulation, exit
        if pending_items and pending_items[0] is None:
            break

    pbar.close()
    producer.join()

    elapsed = time.time() - t0
    rate = tiles_processed / elapsed if elapsed > 0 else 0

    db_total = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info(f"  Tiles processed: {tiles_processed}")
    log.info(f"  Tiles failed:    {tiles_failed}")
    log.info(f"  Tiles skipped:   {skipped}")
    log.info(f"  Total detections: {total_detections}")
    log.info(f"  DB total:        {db_total}")
    log.info(f"  Time:            {elapsed:.0f}s ({rate:.2f} tiles/s)")
    log.info(f"  Output:          {output_db}")
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
    min_elevation: float = typer.Option(0, help="Skip tiles below this elevation in meters (0=disabled, e.g. 1500)"),
    device: str = typer.Option("cuda:0", help="PyTorch device"),
    download_threads: int = typer.Option(8, help="Download thread count"),
    max_tiles: int = typer.Option(0, help="Limit tiles (0=all)"),
    conf: float = typer.Option(0.10, help="Confidence threshold"),
    iou: float = typer.Option(0.40, help="IoU threshold"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Tile cache directory"),
    cache_gb: float = typer.Option(10.0, help="Max tile cache size in GB (0 to disable)"),
) -> None:
    """Run the nationwide rock detection pipeline."""
    init_cache(cache_dir, cache_gb)

    # Build tile source from the chosen input method
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
    )


@app.command()
def export(
    input_db: Path = typer.Option(..., "--input", help="DuckDB database path"),
    output: Path = typer.Option(..., help="Output GeoJSON path"),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence filter"),
    geometry: str = typer.Option("polygon", help="Geometry type: 'point' or 'polygon'"),
) -> None:
    """Export detections from DuckDB to GeoJSON (EPSG:2056)."""
    import duckdb

    if not input_db.exists():
        log.error(f"Database not found: {input_db}")
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

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(geojson, f)

    log.info(f"Exported {len(features)} detections to {output} ({geometry}, EPSG:2056)")


