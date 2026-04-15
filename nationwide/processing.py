import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import rasterio
import rasterio.io
import requests
from requests.adapters import HTTPAdapter
from rasterio.enums import Resampling
from rasterio.windows import Window
from nationwide.cache import cache_get, cache_put, reinit_cache
from nationwide.db import Detection
from utils.constants import (
    FUSION_CHANNEL, NEIGHBOR_STRIP_DSM, NEIGHBOR_STRIP_RGB,
    SRC_CROP_DSM, SRC_CROP_RGB,
    SRC_STRIDE_DSM, SRC_STRIDE_RGB, TILE_PX_DSM, TILE_PX_RGB,
    TILE_SIZE_PX,
)

log = logging.getLogger(__name__)

_POOL_SIZE = os.cpu_count() or 8
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
_SESSION.mount("https://", HTTPAdapter(pool_maxsize=_POOL_SIZE, pool_connections=_POOL_SIZE))

def check_gdaldem() -> None:
    """Fail early if gdaldem is missing."""
    if shutil.which("gdaldem") is None:
        raise RuntimeError("gdaldem not found on PATH — install GDAL CLI tools")


def reinit_session(
    cache_dir: str | None = None, cache_max_bytes: int = 0,
) -> None:
    """Reinitialize HTTP session and tile cache in worker processes.

    ProcessPoolExecutor workers get a fresh module import —
    must re-create session and cache in each worker.
    """
    global _SESSION
    _SESSION = requests.Session()
    _SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
    _SESSION.mount("https://", HTTPAdapter(pool_maxsize=_POOL_SIZE, pool_connections=_POOL_SIZE))
    reinit_cache(cache_dir, cache_max_bytes)


def download_to_memory(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    """Download a file into memory with retries. Uses tile cache when enabled."""
    cached = cache_get(url)
    if cached is not None:
        return cached

    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            if len(resp.content) == 0:
                raise IOError("Empty response")
            cache_put(url, resp.content)
            return resp.content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("Unreachable")


def generate_hillshade(dsm_path: str | Path) -> np.ndarray:
    """Generate hillshade via gdaldem -combined -az 315 -alt 45.

    Combined hillshade blends slope and aspect-based directional lighting.
    Matches the training data preprocessing for best.pt.
    """
    fd, out_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        cmd = [
            "gdaldem", "hillshade", str(dsm_path), out_path,
            "-combined", "-az", "315", "-alt", "45",
            "-compute_edges", "-of", "GTiff", "-q",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        with rasterio.open(out_path) as src:
            return src.read(1)
    finally:
        Path(out_path).unlink(missing_ok=True)


def _get_hillshade(dsm_bytes: bytes, dsm_url: str) -> tuple[np.ndarray, rasterio.Affine]:
    """Generate hillshade from DSM bytes, caching in the tile cache.

    Cache key: dsm filename with '_hs' suffix (e.g. swissalti3d_..._hs.npy).
    Returns (hillshade_array, dsm_transform).
    """
    from nationwide.cache import cache_get, cache_put

    hs_cache_key = dsm_url.replace(".tif", "_hs.npy")
    cached = cache_get(hs_cache_key)
    if cached is not None:
        with rasterio.io.MemoryFile(dsm_bytes) as mf:
            with mf.open() as src:
                transform = src.transform
        hillshade = np.frombuffer(cached, dtype=np.uint8).reshape(
            TILE_PX_DSM, TILE_PX_DSM,
        )
        return hillshade, transform

    fd, dsm_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    with open(dsm_path, "wb") as f:
        f.write(dsm_bytes)
    try:
        with rasterio.open(dsm_path) as src:
            transform = src.transform
        hillshade = generate_hillshade(dsm_path)
    finally:
        Path(dsm_path).unlink(missing_ok=True)

    cache_put(hs_cache_key, hillshade.tobytes())
    return hillshade, transform


def crop_patches(
    data: np.ndarray,
    transform: rasterio.Affine,
    crop_px: int,
    stride_px: int,
    out_px: int,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int]]:
    """Extract overlapping patches from a raster array.

    Returns list of (patch_array, patch_transform, row_idx, col_idx).
    """
    is_2d = data.ndim == 2
    h, w = data.shape[-2:]

    patches = []
    row_idx = 0
    y_off = 0
    while y_off + crop_px <= h:
        col_idx = 0
        x_off = 0
        while x_off + crop_px <= w:
            if is_2d:
                window = data[y_off:y_off + crop_px, x_off:x_off + crop_px]
            else:
                window = data[:, y_off:y_off + crop_px, x_off:x_off + crop_px]

            if crop_px != out_px:
                if is_2d:
                    patch = cv2.resize(
                        window, (out_px, out_px), interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    bands = [
                        cv2.resize(window[b], (out_px, out_px), interpolation=cv2.INTER_CUBIC)
                        for b in range(window.shape[0])
                    ]
                    patch = np.stack(bands)
            else:
                patch = window.copy()

            x_map = transform.c + x_off * transform.a
            y_map = transform.f + y_off * transform.e
            patch_res_x = (crop_px * transform.a) / out_px
            patch_res_y = (crop_px * transform.e) / out_px
            patch_transform = rasterio.Affine(
                patch_res_x, 0, x_map, 0, patch_res_y, y_map,
            )

            patches.append((patch, patch_transform, row_idx, col_idx))
            col_idx += 1
            x_off += stride_px
        row_idx += 1
        y_off += stride_px

    return patches


def _crop_resample_rgb(
    rgb_path: str, width: int, height: int,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int]]:
    """Crop+resample RGB patches via per-window rasterio reads.

    Each patch is independently read from the full-resolution (10cm) source
    and resampled to 640x640 using GDAL's cubic kernel (via rasterio).
    Pixel-identical to gdal_translate -srcwin -outsize -r cubic, validated
    across 22,224 patches with zero differences.
    """
    patches = []
    with rasterio.open(rgb_path) as src:
        row_idx = 0
        y_off = 0
        while y_off + SRC_CROP_RGB <= height:
            col_idx = 0
            x_off = 0
            while x_off + SRC_CROP_RGB <= width:
                patch = src.read(
                    indexes=[1, 2, 3],
                    window=Window(x_off, y_off, SRC_CROP_RGB, SRC_CROP_RGB),
                    out_shape=(3, TILE_SIZE_PX, TILE_SIZE_PX),
                    resampling=Resampling.cubic,
                )
                patch_transform = rasterio.Affine(
                    src.transform.a * SRC_CROP_RGB / TILE_SIZE_PX, 0,
                    src.transform.c + x_off * src.transform.a,
                    0, src.transform.e * SRC_CROP_RGB / TILE_SIZE_PX,
                    src.transform.f + y_off * src.transform.e,
                )
                patches.append((patch, patch_transform, row_idx, col_idx))
                col_idx += 1
                x_off += SRC_STRIDE_RGB
            row_idx += 1
            y_off += SRC_STRIDE_RGB
    return patches


def yolo_to_map_coords(
    cx: float, cy: float, w: float, h: float,
    img_size: int, transform: rasterio.Affine,
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized coords to EPSG:2056 centroid + bbox meters."""
    px, py = cx * img_size, cy * img_size
    pw, ph = w * img_size, h * img_size
    easting, northing = transform * (px, py)
    width_m = abs(pw * transform.a)
    height_m = abs(ph * transform.e)
    return easting, northing, width_m, height_m


def _extract_detections(
    results, meta: list[tuple[rasterio.Affine, int, int, str, str, str]],
) -> list[Detection]:
    """Extract Detection objects from YOLO results with coordinate transform."""
    detections = []
    for result, (transform, row, col, tile_id, rgb_url, dsm_url) in zip(results, meta):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        patch_id = f"{tile_id}_{row}_{col}"
        for i in range(len(boxes)):
            xywhn = boxes.xywhn[i].cpu().numpy()
            cx, cy, w, h = xywhn
            conf_val = float(boxes.conf[i].cpu())
            cls = int(boxes.cls[i].cpu())
            easting, northing, w_m, h_m = yolo_to_map_coords(
                cx, cy, w, h, TILE_SIZE_PX, transform,
            )
            detections.append(Detection(
                tile_id=tile_id, patch_id=patch_id,
                easting=float(easting), northing=float(northing),
                confidence=conf_val, bbox_w_m=float(w_m), bbox_h_m=float(h_m),
                class_id=cls, rgb_source=rgb_url, dsm_source=dsm_url,
            ))
    return detections


def dedup_detections(
    detections: list[Detection], distance_m: float = 7.5,
) -> list[Detection]:
    """NMS-like spatial dedup on EPSG:2056 centroids. Keeps highest confidence."""
    if len(detections) <= 1:
        return detections

    tiles: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        tiles[det.tile_id].append(det)

    kept: list[Detection] = []
    for tile_id, tile_dets in tiles.items():
        if len(tile_dets) <= 1:
            kept.extend(tile_dets)
            continue

        indexed = sorted(
            enumerate(tile_dets), key=lambda x: x[1].confidence, reverse=True,
        )
        suppressed: set[int] = set()

        for i, (idx_i, det_i) in enumerate(indexed):
            if idx_i in suppressed:
                continue
            for j in range(i + 1, len(indexed)):
                idx_j, det_j = indexed[j]
                if idx_j in suppressed:
                    continue
                dist = np.sqrt(
                    (det_i.easting - det_j.easting) ** 2
                    + (det_i.northing - det_j.northing) ** 2
                )
                if dist < distance_m:
                    suppressed.add(idx_j)

        for idx, det in enumerate(tile_dets):
            if idx not in suppressed:
                kept.append(det)

    return kept


def check_elevation(dsm_url: str) -> float:
    """Download DSM tile and return its max elevation."""
    dsm_bytes = download_to_memory(dsm_url)
    with rasterio.io.MemoryFile(dsm_bytes) as memfile:
        with memfile.open() as src:
            return float(src.read(1).max())


def _read_strip(tile_bytes: bytes, band_count: int,
                 col_off: int, row_off: int,
                 width: int, height: int) -> np.ndarray:
    """Read a pixel strip from tile bytes via MemoryFile."""
    with rasterio.io.MemoryFile(tile_bytes) as mf:
        with mf.open() as src:
            return src.read(
                indexes=list(range(1, band_count + 1)),
                window=Window(col_off, row_off, width, height),
            )


def _stitch_and_write(center_bytes: bytes, band_count: int,
                      right_bytes: bytes | None,
                      bottom_bytes: bytes | None,
                      corner_bytes: bytes | None,
                      tile_px: int, strip_px: int) -> tuple[str, rasterio.Affine]:
    """Stitch center tile with neighbor strips and write to temp file.

    Returns (temp_file_path, transform). Caller must delete the file.
    """
    with rasterio.io.MemoryFile(center_bytes) as mf:
        with mf.open() as src:
            center = src.read(indexes=list(range(1, band_count + 1)))
            transform = src.transform
            dtype = src.dtypes[0]
            crs = src.crs

    has_right = right_bytes is not None
    has_bottom = bottom_bytes is not None
    has_corner = corner_bytes is not None and has_right and has_bottom

    ext_w = tile_px + (strip_px if has_right else 0)
    ext_h = tile_px + (strip_px if has_bottom else 0)

    if band_count == 1:
        extended = np.zeros((ext_h, ext_w), dtype=center.dtype)
        extended[:tile_px, :tile_px] = center[0]
    else:
        extended = np.zeros((band_count, ext_h, ext_w), dtype=center.dtype)
        extended[:, :tile_px, :tile_px] = center
    del center

    if has_right:
        strip = _read_strip(right_bytes, band_count, 0, 0, strip_px, tile_px)
        if band_count == 1:
            extended[:tile_px, tile_px:] = strip[0]
        else:
            extended[:, :tile_px, tile_px:] = strip

    if has_bottom:
        strip = _read_strip(bottom_bytes, band_count, 0, 0, tile_px, strip_px)
        if band_count == 1:
            extended[tile_px:, :tile_px] = strip[0]
        else:
            extended[:, tile_px:, :tile_px] = strip

    if has_corner:
        strip = _read_strip(corner_bytes, band_count, 0, 0, strip_px, strip_px)
        if band_count == 1:
            extended[tile_px:, tile_px:] = strip[0]
        else:
            extended[:, tile_px:, tile_px:] = strip

    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    with rasterio.open(
        path, "w", driver="GTiff",
        width=ext_w, height=ext_h,
        count=band_count, dtype=dtype,
        crs=crs, transform=transform,
    ) as dst:
        if band_count == 1:
            dst.write(extended, 1)
        else:
            dst.write(extended)

    return path, transform


def _patch_cache_key(coord: str, has_neighbors: bool) -> str:
    """Cache key for fused patches. Includes neighbor flag so cached
    patches with/without stitching don't collide."""
    suffix = "_stitched" if has_neighbors else ""
    return f"patches_{coord}{suffix}.npz"


def _save_patch_cache(key: str, results: list[tuple]) -> None:
    """Save fused patches + transforms to tile cache."""
    arrays = []
    transforms = []
    rows_cols = []
    for patch, tf, row, col, *_ in results:
        arrays.append(patch)
        transforms.append((tf.a, tf.c, tf.e, tf.f))
        rows_cols.append((row, col))
    from nationwide.cache import cache_put
    import io
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        patches=np.stack(arrays),
        transforms=np.array(transforms),
        rows_cols=np.array(rows_cols),
    )
    cache_put(key, buf.getvalue())


def _load_patch_cache(
    key: str, tile_id: str, rgb_url: str, dsm_url: str,
) -> list[tuple] | None:
    """Load cached fused patches. Returns None on miss."""
    from nationwide.cache import cache_get
    data = cache_get(key)
    if data is None:
        return None
    try:
        import io
        npz = np.load(io.BytesIO(data))
        patches = npz["patches"]
        transforms = npz["transforms"]
        rows_cols = npz["rows_cols"]
        results = []
        for i in range(len(patches)):
            a, c, e, f = transforms[i]
            tf = rasterio.Affine(a, 0, c, 0, e, f)
            row, col = int(rows_cols[i][0]), int(rows_cols[i][1])
            results.append((patches[i], tf, row, col, tile_id, rgb_url, dsm_url))
        return results
    except Exception:
        return None


def process_tile(
    coord: str,
    rgb_url: str,
    dsm_url: str,
    neighbor_right: tuple[str, str] | None = None,
    neighbor_bottom: tuple[str, str] | None = None,
    neighbor_corner: tuple[str, str] | None = None,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int, str, str, str]]:
    """Download, hillshade, crop, and fuse one tile with cross-tile stitching.

    Caches the final fused patches so subsequent runs skip all preprocessing.
    """
    has_neighbors = bool(neighbor_right or neighbor_bottom or neighbor_corner)

    # Derive tile_id from coord (e.g. "2587-1133" → "2587_1133")
    tile_id = coord.replace("-", "_")

    # Check patch cache — if hit, skip ALL preprocessing
    cache_key = _patch_cache_key(coord, has_neighbors)
    cached = _load_patch_cache(cache_key, tile_id, rgb_url, dsm_url)
    if cached is not None:
        return cached

    # Download center + neighbor tiles in parallel
    urls_to_fetch = [rgb_url, dsm_url]
    labels = ["rgb", "dsm"]
    if neighbor_right:
        urls_to_fetch.extend(neighbor_right)
        labels.extend(["rgb_right", "dsm_right"])
    if neighbor_bottom:
        urls_to_fetch.extend(neighbor_bottom)
        labels.extend(["rgb_bottom", "dsm_bottom"])
    if neighbor_corner:
        urls_to_fetch.extend(neighbor_corner)
        labels.extend(["rgb_corner", "dsm_corner"])

    with ThreadPoolExecutor(max_workers=len(urls_to_fetch)) as dl_pool:
        futures = {label: dl_pool.submit(download_to_memory, url)
                   for label, url in zip(labels, urls_to_fetch)}
        downloaded = {label: f.result() for label, f in futures.items()}

    rgb_bytes = downloaded["rgb"]
    dsm_bytes = downloaded["dsm"]
    dsm_right = downloaded.get("dsm_right")
    dsm_bottom = downloaded.get("dsm_bottom")
    dsm_corner = downloaded.get("dsm_corner")
    rgb_right = downloaded.get("rgb_right")
    rgb_bottom = downloaded.get("rgb_bottom")
    rgb_corner = downloaded.get("rgb_corner")

    # --- Generate all hillshades in parallel (center + neighbors) ---
    hs_jobs: dict[str, tuple[bytes, str]] = {"center": (dsm_bytes, dsm_url)}
    if dsm_right:
        hs_jobs["right"] = (dsm_right, neighbor_right[1])
    if dsm_bottom:
        hs_jobs["bottom"] = (dsm_bottom, neighbor_bottom[1])
    if dsm_corner:
        hs_jobs["corner"] = (dsm_corner, neighbor_corner[1])

    with ThreadPoolExecutor(max_workers=len(hs_jobs)) as hs_pool:
        hs_futures = {
            name: hs_pool.submit(_get_hillshade, data, url)
            for name, (data, url) in hs_jobs.items()
        }
        hs_results = {name: f.result() for name, f in hs_futures.items()}

    center_hs, dsm_transform = hs_results["center"]
    hs_patches = crop_patches(
        center_hs, dsm_transform,
        crop_px=SRC_CROP_DSM, stride_px=SRC_STRIDE_DSM, out_px=TILE_SIZE_PX,
    )

    # --- Cross-boundary hillshade: stitch neighbor hillshades ---
    if has_neighbors:
        right_hs = hs_results["right"][0] if "right" in hs_results else None
        bottom_hs = hs_results["bottom"][0] if "bottom" in hs_results else None
        corner_hs = hs_results["corner"][0] if "corner" in hs_results else None

        ext_h = TILE_PX_DSM + (NEIGHBOR_STRIP_DSM if bottom_hs is not None else 0)
        ext_w = TILE_PX_DSM + (NEIGHBOR_STRIP_DSM if right_hs is not None else 0)
        extended_hs = np.zeros((ext_h, ext_w), dtype=center_hs.dtype)
        extended_hs[:TILE_PX_DSM, :TILE_PX_DSM] = center_hs
        if right_hs is not None:
            extended_hs[:TILE_PX_DSM, TILE_PX_DSM:] = right_hs[:, :NEIGHBOR_STRIP_DSM]
        if bottom_hs is not None:
            extended_hs[TILE_PX_DSM:, :TILE_PX_DSM] = bottom_hs[:NEIGHBOR_STRIP_DSM, :]
        if corner_hs is not None and right_hs is not None and bottom_hs is not None:
            extended_hs[TILE_PX_DSM:, TILE_PX_DSM:] = corner_hs[:NEIGHBOR_STRIP_DSM, :NEIGHBOR_STRIP_DSM]

        ext_hs_patches = crop_patches(
            extended_hs, dsm_transform,
            crop_px=SRC_CROP_DSM, stride_px=SRC_STRIDE_DSM, out_px=TILE_SIZE_PX,
        )
        hs_patches.extend(p for p in ext_hs_patches if p[2] >= 4 or p[3] >= 4)
        del extended_hs, right_hs, bottom_hs, corner_hs

    del center_hs, dsm_bytes, dsm_right, dsm_bottom, dsm_corner

    # --- Center RGB: from original file (preserves compression/block layout) ---
    fd_rgb, rgb_center_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd_rgb)
    with open(rgb_center_path, "wb") as f:
        f.write(rgb_bytes)

    try:
        with rasterio.open(rgb_center_path) as src:
            rgb_width, rgb_height = src.width, src.height
        rgb_patches = _crop_resample_rgb(rgb_center_path, rgb_width, rgb_height)

        # --- Cross-boundary RGB: stitch neighbor strips ---
        if has_neighbors:
            rgb_ext_path, _ = _stitch_and_write(
                rgb_bytes, 3, rgb_right, rgb_bottom, rgb_corner,
                TILE_PX_RGB, NEIGHBOR_STRIP_RGB,
            )
            try:
                with rasterio.open(rgb_ext_path) as src:
                    ext_w, ext_h = src.width, src.height
                ext_rgb_patches = _crop_resample_rgb(rgb_ext_path, ext_w, ext_h)
                rgb_patches.extend(
                    p for p in ext_rgb_patches if p[2] >= 4 or p[3] >= 4
                )
            finally:
                Path(rgb_ext_path).unlink(missing_ok=True)
    finally:
        Path(rgb_center_path).unlink(missing_ok=True)

    del rgb_bytes, rgb_right, rgb_bottom, rgb_corner

    # --- Fuse RGB + hillshade ---
    results = []
    for (hs_patch, hs_tf, row, col), (rgb_patch, _, _, _) in zip(
        hs_patches, rgb_patches,
    ):
        rgb_patch[FUSION_CHANNEL] = hs_patch
        results.append((rgb_patch, hs_tf, row, col, tile_id, rgb_url, dsm_url))

    # Cache the fused patches for future runs
    _save_patch_cache(cache_key, results)

    return results


def build_batch_tensor(
    patches: list[tuple[np.ndarray, rasterio.Affine, int, int, str, str, str]],
    device: str = "cuda:0",
) -> tuple:
    """Stack patches into a GPU-ready float tensor. Returns (tensor, meta_list).

    This is the CPU-bound half of inference: np.stack + float conversion + H2D
    transfer. Designed to run on the main thread while the GPU processes the
    previous batch in a background thread.
    """
    import torch

    meta = []
    arrays = []
    for patch_data, transform, row, col, tile_id, rgb_url, dsm_url in patches:
        arrays.append(patch_data)
        meta.append((transform, row, col, tile_id, rgb_url, dsm_url))
    batch = torch.from_numpy(np.stack(arrays)).float().div_(255.0).to(device)
    return batch, meta


def infer_on_tensor(
    model,
    batch,
    meta: list,
    conf: float = 0.10,
    iou: float = 0.40,
) -> list[Detection]:
    """Run YOLO predict on a pre-built GPU tensor, extract detections.

    This is the GPU-bound half of inference. PyTorch releases the GIL during
    CUDA kernels, so this can run in a background thread while the main thread
    builds the next tensor on CPU.
    """
    results = model.predict(
        source=batch, conf=conf, iou=iou, imgsz=TILE_SIZE_PX,
        save=False, verbose=False,
    )
    return _extract_detections(results, meta)


def run_inference(
    model,
    patches: list[tuple[np.ndarray, rasterio.Affine, int, int, str, str, str]],
    conf: float = 0.10,
    iou: float = 0.40,
    device: str = "cuda:0",
    max_patches_per_call: int = 0,
) -> list[Detection]:
    """Run YOLO on fused patches, return Detections in EPSG:2056.

    Passes a pre-built (N, 3, H, W) tensor for single batched forward pass.
    If max_patches_per_call > 0, splits into chunks to avoid OOM.
    """
    if not patches:
        return []

    import torch

    meta = []
    arrays = []
    for patch_data, transform, row, col, tile_id, rgb_url, dsm_url in patches:
        arrays.append(patch_data)
        meta.append((transform, row, col, tile_id, rgb_url, dsm_url))

    chunk_size = len(arrays) if max_patches_per_call <= 0 else max_patches_per_call

    all_results = []
    oom_fallback_detections: list[Detection] | None = None

    for start in range(0, len(arrays), chunk_size):
        end = min(start + chunk_size, len(arrays))
        chunk_arrays = arrays[start:end]

        batch = torch.from_numpy(np.stack(chunk_arrays)).float().div_(255.0).to(device)

        try:
            results = model.predict(
                source=batch, conf=conf, iou=iou, imgsz=TILE_SIZE_PX,
                save=False, verbose=False,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            del batch
            torch.cuda.empty_cache()
            half = len(chunk_arrays) // 2
            log.warning("OOM on %d patches, retrying remaining with chunk=%d", len(chunk_arrays), half)
            if half == 0:
                raise
            oom_fallback_detections = run_inference(
                model, patches[start:], conf, iou, device, half,
            )
            break

        all_results.extend(zip(results, meta[start:end]))
        del batch

    detections = _extract_detections(
        [r for r, _ in all_results],
        [m for _, m in all_results],
    )

    if oom_fallback_detections is not None:
        detections.extend(oom_fallback_detections)

    return detections
