"""Auto-tuning: GPU batch size probe, network throughput, parameter recommendations."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

PATCHES_PER_TILE = 16  # 4x4 grid from 2000x2000 tile
PATCH_SIZE = 640


@dataclass
class GPUProbeResult:
    """Result of GPU batch size probing."""
    max_batch_tiles: int
    max_patches: int
    peak_vram_mb: int


@dataclass
class NetworkProbeResult:
    """Result of network throughput probing."""
    mbps_per_thread: float
    recommended_threads: int


@dataclass
class PipelineParams:
    """Recommended pipeline parameters."""
    max_batch_tiles: int
    download_threads: int
    queue_maxsize: int


def _get_vram_used_mb(device: str) -> int:
    """Get current GPU VRAM usage in MB via torch."""
    import torch
    idx = int(device.split(":")[-1]) if ":" in device else 0
    return torch.cuda.memory_allocated(idx) // (1024 * 1024)


def _get_vram_total_mb(device: str) -> int:
    """Get total GPU VRAM in MB via torch."""
    import torch
    idx = int(device.split(":")[-1]) if ":" in device else 0
    props = torch.cuda.get_device_properties(idx)
    return props.total_memory // (1024 * 1024)


def probe_gpu_max_batch(
    model,
    device: str = "cuda:0",
    min_patches: int = 16,
    max_patches: int = 512,
    step: int = 16,
    safety_factor: float = 0.85,
) -> GPUProbeResult:
    """Binary search for the largest batch size that fits in GPU VRAM.

    Warms up CUDA/cuDNN with a single-patch inference, then binary searches
    in range [min_patches, max_patches] stepping by `step` (= 1 tile worth).
    Applies a safety factor and rounds down to a multiple of step.

    Takes ~3-5 seconds total (~6 binary search iterations x ~0.5s each).
    """
    import torch

    log.info("Probing GPU max batch size ...")
    vram_total = _get_vram_total_mb(device)
    log.info(f"  GPU VRAM total: {vram_total} MB")

    # Warmup: single patch to prime CUDA kernels and cuDNN autotuner
    dummy = torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)
    model.predict(source=dummy, conf=0.10, imgsz=PATCH_SIZE, save=False, verbose=False)
    del dummy
    torch.cuda.empty_cache()
    baseline_vram = _get_vram_used_mb(device)
    log.info(f"  Baseline VRAM after warmup: {baseline_vram} MB")

    # Binary search
    lo = min_patches // step
    hi = max_patches // step
    best = lo  # Fallback to minimum

    while lo <= hi:
        mid = (lo + hi) // 2
        n_patches = mid * step
        try:
            torch.cuda.empty_cache()
            dummy = torch.zeros(
                n_patches, 3, PATCH_SIZE, PATCH_SIZE,
                dtype=torch.float32, device=device,
            )
            model.predict(
                source=dummy, conf=0.10, imgsz=PATCH_SIZE,
                save=False, verbose=False,
            )
            peak = _get_vram_used_mb(device)
            del dummy
            torch.cuda.empty_cache()
            log.info(f"  {n_patches} patches: OK ({peak} MB VRAM)")
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                log.info(f"  {n_patches} patches: OOM")
                del dummy  # noqa: F821 — may not exist if alloc failed
                torch.cuda.empty_cache()
                hi = mid - 1
            else:
                raise

    # Apply safety factor
    safe_patches = int(best * step * safety_factor)
    safe_patches = max(step, (safe_patches // step) * step)  # Round down to step
    safe_tiles = safe_patches // PATCHES_PER_TILE

    peak_vram = _get_vram_used_mb(device)
    result = GPUProbeResult(
        max_batch_tiles=safe_tiles,
        max_patches=safe_patches,
        peak_vram_mb=peak_vram,
    )
    log.info(
        f"  GPU probe result: {safe_tiles} tiles ({safe_patches} patches), "
        f"safety={safety_factor:.0%}"
    )
    return result


def _download_one(url: str) -> int:
    """Download a single URL, return size in bytes."""
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent": "rock-bench/0.1"})
    resp = sess.get(url, timeout=120)
    resp.raise_for_status()
    return len(resp.content)


def measure_download_throughput(
    urls: list[str],
    thread_counts: tuple[int, ...] = (1, 4, 8, 16),
) -> NetworkProbeResult:
    """Measure download throughput at multiple thread counts.

    Tests each thread count with enough URLs to get a stable measurement,
    finds the sweet spot where adding threads stops helping.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log.info("Measuring download throughput ...")

    best_mbps = 0.0
    best_threads = 1
    results: list[tuple[int, float]] = []

    for n_threads in thread_counts:
        n_files = min(len(urls), max(n_threads * 2, 6))
        test_urls = urls[:n_files]

        total_bytes = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = {pool.submit(_download_one, u): u for u in test_urls}
            for f in as_completed(futs):
                total_bytes += f.result()

        elapsed = time.time() - t0
        total_mb = total_bytes / (1024 * 1024)
        mbps = total_mb / elapsed if elapsed > 0 else 0
        results.append((n_threads, mbps))
        log.info(
            f"  {n_threads:2d} threads: {n_files} files, "
            f"{total_mb:.0f} MB in {elapsed:.1f}s = {mbps:.0f} MB/s"
        )

        if mbps > best_mbps:
            best_mbps = mbps
            best_threads = n_threads

    # Find the thread count where download throughput plateaus.
    # "Plateau" = less than 10% improvement over previous.
    plateau_threads = results[0][0]
    plateau_mbps = results[0][1]
    for n, mbps in results[1:]:
        if mbps > plateau_mbps * 1.10:
            plateau_mbps = mbps
            plateau_threads = n
        else:
            break

    # Pipeline workers do download + CPU preprocessing (hillshade, crop, fuse).
    # CPU work ~doubles the per-tile time vs raw download, so we need ~2x
    # the download-saturating thread count to keep the pipeline fed.
    # Cap at CPU count since each worker is CPU-bound during preprocessing.
    cpus = os.cpu_count() or 8
    recommended_threads = min(plateau_threads * 2, cpus)

    result = NetworkProbeResult(
        mbps_per_thread=round(best_mbps / best_threads, 1),
        recommended_threads=recommended_threads,
    )
    log.info(
        f"  Download plateau: {plateau_mbps:.0f} MB/s at {plateau_threads} threads | "
        f"Recommended workers (incl. CPU overhead): {recommended_threads}"
    )
    return result


def recommend_params(
    gpu_result: GPUProbeResult,
    net_result: NetworkProbeResult | None = None,
    cpu_count: int | None = None,
) -> PipelineParams:
    """Combine probe results into recommended pipeline parameters."""
    cpus = cpu_count or os.cpu_count() or 8

    if net_result is not None:
        download_threads = net_result.recommended_threads
    else:
        # Default heuristic: half of CPU cores, clamped
        download_threads = max(4, min(32, cpus // 2))

    queue_maxsize = download_threads * 3

    return PipelineParams(
        max_batch_tiles=gpu_result.max_batch_tiles,
        download_threads=download_threads,
        queue_maxsize=queue_maxsize,
    )
