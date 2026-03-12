import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import rasterio
import rasterio.io
import requests
from requests.adapters import HTTPAdapter
from nationwide.cache import cache_get, cache_put, reinit_cache
from nationwide.db import Detection
from rasterio.enums import Resampling
from utils.constants import (
    DSM_RES, FUSION_CHANNEL, SRC_CROP_DSM,
    SRC_STRIDE_DSM, SWISSIMAGE_RES, TARGET_RES, TILE_SIZE_PX,
)

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
_SESSION.mount("https://", HTTPAdapter(pool_maxsize=4, pool_connections=4))


def reinit_session(cache_dir: str | None = None, cache_max_bytes: int = 0) -> None:
    """Reinitialize HTTP session and tile cache in worker processes.

    ProcessPoolExecutor workers get a fresh module import —
    must re-create session and cache in each worker.
    """
    global _SESSION
    _SESSION = requests.Session()
    _SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
    _SESSION.mount("https://", HTTPAdapter(pool_maxsize=4, pool_connections=4))
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


def generate_hillshade(
    dsm: np.ndarray,
    res: float = 0.5,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Overhead hillshade from slope magnitude: shade = 255 * cos(slope).

    Validated against 16 training patches of tile 2581-1126:
    r=0.999+, MAE=0.029, 99.87% of pixels within 1.0 of training values.
    """
    dsm = dsm.astype(np.float32) * z_factor

    padded = np.pad(dsm, 1, mode="edge")

    dzdx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * res)

    dzdy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
        - (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8.0 * res)

    shade = 1.0 / np.sqrt(1.0 + dzdx**2 + dzdy**2)

    shade = np.clip(shade * 255.0, 0, 255).astype(np.uint8)
    return shade


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


def fuse_rgb_hillshade(
    rgb: np.ndarray, hillshade: np.ndarray, channel: int = 1,
) -> np.ndarray:
    """Replace one RGB channel with hillshade. Arrays are (bands, H, W)."""
    fused = rgb.copy()
    fused[channel] = hillshade.astype(rgb.dtype)
    return fused


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


def check_elevation(dsm_url: str, min_elevation: float) -> bool:
    """Download DSM tile and check if any point reaches min_elevation."""
    dsm_bytes = download_to_memory(dsm_url)
    with rasterio.io.MemoryFile(dsm_bytes) as memfile:
        with memfile.open() as src:
            return float(src.read(1).max()) >= min_elevation


def process_tile(
    coord: str,
    rgb_url: str,
    dsm_url: str,
    min_elevation: float = 0,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int, str]]:
    """Download, hillshade, crop, and fuse one tile entirely in memory.

    Returns list of (fused_patch, transform, row, col, tile_id).
    Each fused_patch is (3, 640, 640) uint8.
    """
    with ThreadPoolExecutor(max_workers=2) as dl_pool:
        rgb_future = dl_pool.submit(download_to_memory, rgb_url)
        dsm_future = dl_pool.submit(download_to_memory, dsm_url)
        rgb_bytes = rgb_future.result()
        dsm_bytes = dsm_future.result()

    with rasterio.io.MemoryFile(dsm_bytes) as memfile:
        with memfile.open() as src:
            dsm_data = src.read(1)
            dsm_transform = src.transform
            dsm_bounds = src.bounds
    del dsm_bytes

    if min_elevation > 0:
        max_elev = float(dsm_data.max())
        if max_elev < min_elevation:
            log.debug(f"Tile {coord} below {min_elevation}m (max={max_elev:.0f}m), skipping")
            del dsm_data, rgb_bytes
            return []

    hillshade = generate_hillshade(dsm_data, res=DSM_RES)
    del dsm_data

    # DSM already at target res — no resampling needed
    hs_patches = crop_patches(
        hillshade, dsm_transform,
        crop_px=SRC_CROP_DSM, stride_px=SRC_STRIDE_DSM, out_px=TILE_SIZE_PX,
    )
    del hillshade

    with rasterio.io.MemoryFile(rgb_bytes) as memfile:
        with memfile.open() as src:
            bands = min(src.count, 3)
            target_h = int(src.height * SWISSIMAGE_RES / TARGET_RES)
            target_w = int(src.width * SWISSIMAGE_RES / TARGET_RES)
            rgb_data = src.read(
                indexes=list(range(1, bands + 1)),
                out_shape=(bands, target_h, target_w),
                resampling=Resampling.bilinear,
            )
            rgb_transform = src.transform * src.transform.scale(
                src.width / target_w, src.height / target_h,
            )
    del rgb_bytes

    rgb_patches = crop_patches(
        rgb_data, rgb_transform,
        crop_px=SRC_CROP_DSM, stride_px=SRC_STRIDE_DSM, out_px=TILE_SIZE_PX,
    )
    del rgb_data

    # LV95 km grid
    grid_x = int(dsm_bounds.left) // 1000
    grid_y = int(dsm_bounds.bottom) // 1000
    tile_id = f"{grid_x}_{grid_y}"

    results = []
    for (hs_patch, hs_tf, row, col), (rgb_patch, _, _, _) in zip(
        hs_patches, rgb_patches,
    ):
        rgb_patch[FUSION_CHANNEL] = hs_patch
        results.append((rgb_patch, hs_tf, row, col, tile_id))

    return results


def build_batch_tensor(
    patches: list[tuple[np.ndarray, rasterio.Affine, int, int, str]],
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
    for patch_data, transform, row, col, tile_id in patches:
        arrays.append(patch_data)
        meta.append((transform, row, col, tile_id))
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
    detections = []
    for result, (transform, row, col, tile_id) in zip(results, meta):
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
                class_id=cls,
            ))
    return detections


def run_inference(
    model,
    patches: list[tuple[np.ndarray, rasterio.Affine, int, int, str]],
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
    for patch_data, transform, row, col, tile_id in patches:
        arrays.append(patch_data)
        meta.append((transform, row, col, tile_id))

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

    detections = []
    for result, (transform, row, col, tile_id) in all_results:
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
                class_id=cls,
            ))

    if oom_fallback_detections is not None:
        detections.extend(oom_fallback_detections)

    return detections
