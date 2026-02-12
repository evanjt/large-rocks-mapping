"""Tile processing: download, hillshade, crop, fuse, inference, dedup."""

from __future__ import annotations

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

from nationwide.db import Detection

log = logging.getLogger(__name__)

# ── Patch geometry constants (match training data) ───────────────────────────

TILE_SIZE_PX = 640
OVERLAP_PX = 210
TARGET_RES = 0.5          # m/pixel (output)
SWISSIMAGE_RES = 0.1      # m/pixel (source RGB)
DSM_RES = 0.5             # m/pixel (source DSM)
FUSION_CHANNEL = 1        # Replace green channel

TILE_GROUND_M = TILE_SIZE_PX * TARGET_RES          # 320m
STRIDE_PX = TILE_SIZE_PX - OVERLAP_PX              # 430px
STRIDE_GROUND_M = STRIDE_PX * TARGET_RES           # 215m
SRC_CROP_RGB = int(TILE_GROUND_M / SWISSIMAGE_RES) # 3200px
SRC_STRIDE_RGB = int(STRIDE_GROUND_M / SWISSIMAGE_RES)  # 2150px
SRC_CROP_DSM = int(TILE_GROUND_M / DSM_RES)        # 640px
SRC_STRIDE_DSM = int(STRIDE_GROUND_M / DSM_RES)    # 430px

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
# 8 download threads × 2 parallel downloads per tile = 16 concurrent connections
_SESSION.mount("https://", HTTPAdapter(pool_maxsize=20, pool_connections=20))


# ── Download ─────────────────────────────────────────────────────────────────

def download_to_memory(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    """Download a file into memory with retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            if len(resp.content) == 0:
                raise IOError("Empty response")
            return resp.content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("Unreachable")


# ── Hillshade ────────────────────────────────────────────────────────────────

def generate_hillshade(
    dsm: np.ndarray,
    res: float = 0.5,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Pure-numpy Igor hillshade (no subprocess, no gdaldem dependency).

    Igor's formula produces a diffuse overhead illumination that closely
    matches the training data. Output is uint8 (0-255).

    ~3.6x faster than the gdaldem subprocess approach.
    """
    dsm = dsm.astype(np.float64) * z_factor

    # Gradient in x and y (Horn's method using 3x3 kernel)
    padded = np.pad(dsm, 1, mode="edge")

    dzdx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * res)

    dzdy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
        - (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8.0 * res)

    # Igor's formula: overhead diffuse illumination from slope magnitude
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    shade = np.cos(slope_rad)

    # Scale to 0-254 (GDAL reserves 0 for nodata in some modes)
    shade = np.clip(shade * 254.0 + 1.0, 1, 255).astype(np.uint8)
    return shade


# ── Crop patches ─────────────────────────────────────────────────────────────

def crop_patches(
    data: np.ndarray,
    transform: rasterio.Affine,
    crop_px: int,
    stride_px: int,
    out_px: int,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int]]:
    """Extract overlapping patches from a raster array.

    Args:
        data: (bands, H, W) or (H, W) array.
        transform: Affine transform of the source raster.
        crop_px: Window size in source pixels.
        stride_px: Stride in source pixels.
        out_px: Output patch size (resampled if != crop_px).

    Returns:
        List of (patch_array, patch_transform, row_idx, col_idx).
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

            # Resample if needed
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

            # Compute transform for this patch
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


# ── Fusion ───────────────────────────────────────────────────────────────────

def fuse_rgb_hillshade(
    rgb: np.ndarray, hillshade: np.ndarray, channel: int = 1,
) -> np.ndarray:
    """Replace one RGB channel with hillshade. Arrays are (bands, H, W)."""
    fused = rgb.copy()
    fused[channel] = hillshade.astype(rgb.dtype)
    return fused


# ── YOLO coordinate transform ───────────────────────────────────────────────

def yolo_to_map_coords(
    cx: float, cy: float, w: float, h: float,
    img_size: int, transform: rasterio.Affine,
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized coords to EPSG:2056 centroid + bbox meters.

    Returns (easting, northing, width_m, height_m).
    """
    px, py = cx * img_size, cy * img_size
    pw, ph = w * img_size, h * img_size
    easting, northing = transform * (px, py)
    width_m = abs(pw * transform.a)
    height_m = abs(ph * transform.e)
    return easting, northing, width_m, height_m


# ── Deduplication ────────────────────────────────────────────────────────────

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

        # Sort by confidence descending
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


# ── Tile processing (in-memory) ──────────────────────────────────────────────

def process_tile(
    coord: str,
    rgb_url: str,
    dsm_url: str,
    min_elevation: float = 0,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int, str]]:
    """Download, hillshade, crop, and fuse one tile entirely in memory.

    No temp files: downloads to bytes, opens via rasterio MemoryFile,
    computes hillshade in numpy, crops and fuses in-memory.

    If min_elevation > 0, checks the DSM max value and skips processing
    if the tile is below the threshold.

    Returns list of (fused_patch, transform, row, col, tile_id).
    Each fused_patch is (3, 640, 640) uint8.
    """
    # Download both files in parallel
    with ThreadPoolExecutor(max_workers=2) as dl_pool:
        rgb_future = dl_pool.submit(download_to_memory, rgb_url)
        dsm_future = dl_pool.submit(download_to_memory, dsm_url)
        rgb_bytes = rgb_future.result()
        dsm_bytes = dsm_future.result()

    # Read DSM from memory, compute hillshade
    with rasterio.io.MemoryFile(dsm_bytes) as memfile:
        with memfile.open() as src:
            dsm_data = src.read(1)
            dsm_transform = src.transform
            dsm_bounds = src.bounds
    del dsm_bytes

    # Elevation gate: skip tiles below threshold
    if min_elevation > 0:
        max_elev = float(dsm_data.max())
        if max_elev < min_elevation:
            log.info(f"Tile {coord} below {min_elevation}m (max={max_elev:.0f}m), skipping")
            del dsm_data, rgb_bytes
            return []

    hillshade = generate_hillshade(dsm_data, res=DSM_RES)
    del dsm_data

    # Crop hillshade patches (DSM already at target res)
    hs_patches = crop_patches(
        hillshade, dsm_transform,
        crop_px=SRC_CROP_DSM, stride_px=SRC_STRIDE_DSM, out_px=TILE_SIZE_PX,
    )
    del hillshade

    # Read RGB from memory
    with rasterio.io.MemoryFile(rgb_bytes) as memfile:
        with memfile.open() as src:
            rgb_data = src.read()
            rgb_transform = src.transform
    del rgb_bytes
    if rgb_data.shape[0] > 3:
        rgb_data = rgb_data[:3]

    # Crop RGB patches (higher res, needs resampling)
    rgb_patches = crop_patches(
        rgb_data, rgb_transform,
        crop_px=SRC_CROP_RGB, stride_px=SRC_STRIDE_RGB, out_px=TILE_SIZE_PX,
    )
    del rgb_data

    # Tile ID from DSM bounds (LV95 km grid)
    grid_x = int(dsm_bounds.left) // 1000
    grid_y = int(dsm_bounds.bottom) // 1000
    tile_id = f"{grid_x}_{grid_y}"

    # Fuse matching patches
    results = []
    for (hs_patch, hs_tf, row, col), (rgb_patch, _, _, _) in zip(
        hs_patches, rgb_patches,
    ):
        fused = fuse_rgb_hillshade(rgb_patch, hs_patch, FUSION_CHANNEL)
        results.append((fused, hs_tf, row, col, tile_id))

    return results


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(
    model,
    patches: list[tuple[np.ndarray, rasterio.Affine, int, int, str]],
    conf: float = 0.10,
    iou: float = 0.40,
    device: str = "cuda:0",
) -> list[Detection]:
    """Run YOLO on fused patches, return Detections in EPSG:2056."""
    if not patches:
        return []

    images = []
    meta = []
    for patch_data, transform, row, col, tile_id in patches:
        # (3, H, W) RGB -> (H, W, 3) BGR for YOLO
        if patch_data.ndim == 3 and patch_data.shape[0] == 3:
            img = np.transpose(patch_data, (1, 2, 0))
            img = img[..., ::-1].copy()
        else:
            img = patch_data
        images.append(img)
        meta.append((transform, row, col, tile_id))

    results = model.predict(
        source=images, conf=conf, iou=iou, imgsz=TILE_SIZE_PX,
        device=device, save=False, verbose=False,
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
