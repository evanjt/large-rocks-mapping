"""Background resource monitor: GPU, CPU, RAM, network, queue depth."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Sample:
    """Single point-in-time resource snapshot."""
    timestamp: float
    gpu_util_pct: float = 0.0
    gpu_vram_used_mb: int = 0
    gpu_vram_total_mb: int = 0
    cpu_pct: float = 0.0
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    net_recv_mb: float = 0.0
    queue_depth: int = 0


@dataclass
class ResourceSummary:
    """Aggregated stats from monitoring samples."""
    n_samples: int = 0
    gpu_util_avg: float = 0.0
    gpu_util_max: float = 0.0
    gpu_vram_max_mb: int = 0
    gpu_vram_total_mb: int = 0
    cpu_avg: float = 0.0
    cpu_max: float = 0.0
    ram_max_mb: int = 0
    ram_total_mb: int = 0
    net_total_mb: float = 0.0
    net_avg_mbps: float = 0.0
    queue_avg: float = 0.0
    queue_max: int = 0
    duration_s: float = 0.0


def _query_nvidia_smi() -> tuple[float, int, int]:
    """Query GPU utilization and VRAM via nvidia-smi. Returns (util%, used_mb, total_mb)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 0.0, 0, 0
        # Parse first GPU line
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return float(parts[0]), int(parts[1]), int(parts[2])
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return 0.0, 0, 0


class ResourceMonitor:
    """Background daemon thread that samples system resources periodically.

    Stores up to 1000 samples in a deque. Provides summary() for
    avg/max aggregation across the monitoring window.
    """

    def __init__(
        self,
        interval: float = 5.0,
        maxlen: int = 1000,
        tile_queue: Any = None,
    ) -> None:
        self._interval = interval
        self._tile_queue = tile_queue
        self._samples: deque[Sample] = deque(maxlen=maxlen)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil_available = False
        self._net_baseline: float = 0.0
        try:
            import psutil  # noqa: F401
            self._psutil_available = True
        except ImportError:
            log.warning("psutil not installed — CPU/RAM/network monitoring disabled")

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._psutil_available:
            import psutil
            counters = psutil.net_io_counters()
            self._net_baseline = counters.bytes_recv if counters else 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Resource monitor started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self._take_sample()
                self._samples.append(sample)
            except Exception:
                pass  # Never crash the monitor thread
            self._stop_event.wait(self._interval)

    def _take_sample(self) -> Sample:
        gpu_util, gpu_used, gpu_total = _query_nvidia_smi()

        cpu_pct = 0.0
        ram_used = 0
        ram_total = 0
        net_recv_mb = 0.0

        if self._psutil_available:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_used = mem.used // (1024 * 1024)
            ram_total = mem.total // (1024 * 1024)
            counters = psutil.net_io_counters()
            if counters:
                net_recv_mb = (counters.bytes_recv - self._net_baseline) / (1024 * 1024)

        queue_depth = 0
        if self._tile_queue is not None:
            try:
                queue_depth = self._tile_queue.qsize()
            except Exception:
                pass

        return Sample(
            timestamp=time.time(),
            gpu_util_pct=gpu_util,
            gpu_vram_used_mb=gpu_used,
            gpu_vram_total_mb=gpu_total,
            cpu_pct=cpu_pct,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            net_recv_mb=net_recv_mb,
            queue_depth=queue_depth,
        )

    def summary(self) -> ResourceSummary:
        """Compute aggregate statistics from collected samples."""
        samples = list(self._samples)
        if not samples:
            return ResourceSummary()

        duration = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0.0

        gpu_utils = [s.gpu_util_pct for s in samples]
        cpu_pcts = [s.cpu_pct for s in samples]
        queue_depths = [s.queue_depth for s in samples]

        net_total_mb = samples[-1].net_recv_mb if samples else 0.0
        net_avg_mbps = (net_total_mb / duration * 8) if duration > 0 else 0.0

        return ResourceSummary(
            n_samples=len(samples),
            gpu_util_avg=sum(gpu_utils) / len(gpu_utils),
            gpu_util_max=max(gpu_utils),
            gpu_vram_max_mb=max(s.gpu_vram_used_mb for s in samples),
            gpu_vram_total_mb=samples[-1].gpu_vram_total_mb,
            cpu_avg=sum(cpu_pcts) / len(cpu_pcts),
            cpu_max=max(cpu_pcts),
            ram_max_mb=max(s.ram_used_mb for s in samples),
            ram_total_mb=samples[-1].ram_total_mb,
            net_total_mb=round(net_total_mb, 1),
            net_avg_mbps=round(net_avg_mbps, 1),
            queue_avg=sum(queue_depths) / len(queue_depths),
            queue_max=max(queue_depths),
            duration_s=round(duration, 1),
        )
