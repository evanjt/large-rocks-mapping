"""Lock the 5×5 = 25 patch count per tile from `process_tile_from_cache`.

Regression anchor for v1.0.3's `76cbf75` "Optimise the on-fly preprocessing",
which clipped the RGB/hillshade iteration dimensions based on which neighbor
tiles happened to be present in the local download cache at call time. On
an interior tile whose neighbors weren't in the current batch, patches
dropped to 4×5, 5×4, or 4×4 — silently losing 5–9 patches per tile and ~17%
of nationwide detections.

These tests exercise `process_tile_from_cache` end-to-end on synthetic
TIFFs (no network, no model) across every neighbor-availability permutation
and assert the output is always 25 patches of shape (3, 640, 640) uint8.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from nationwide import processing
from utils.constants import TILE_PX_DSM, TILE_PX_RGB

CRS = "EPSG:2056"
CENTER_COORD = "2600-1140"


def _write_dsm(path: Path, origin_x_km: int, origin_y_km: int) -> None:
    h = w = TILE_PX_DSM
    data = np.fromfunction(lambda y, x: (y + x).astype(np.float32) * 0.5, (h, w))
    transform = Affine(0.5, 0, origin_x_km * 1000, 0, -0.5, (origin_y_km + 1) * 1000)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=1, dtype="float32",
        crs=CRS, transform=transform,
        compress="LZW",
    ) as dst:
        dst.write(data, 1)


def _write_rgb(path: Path, origin_x_km: int, origin_y_km: int) -> None:
    h = w = TILE_PX_RGB
    transform = Affine(0.1, 0, origin_x_km * 1000, 0, -0.1, (origin_y_km + 1) * 1000)
    row = np.full(w, 128, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=3, dtype="uint8",
        crs=CRS, transform=transform,
        compress="LZW", tiled=True,
    ) as dst:
        band = np.broadcast_to(row, (h, w))
        dst.write(band, 1)
        dst.write(band, 2)
        dst.write(band, 3)


def _url(kind: str, cx: int, cy: int) -> str:
    return f"http://fake/{kind}_{cx}-{cy}.tif"


@pytest.fixture
def synth_tiles(tmp_path, monkeypatch):
    url_to_path: dict[str, Path] = {}

    def fake_ensure_cached(url, retries=3, timeout=120):
        p = url_to_path.get(url)
        if p is None:
            raise RuntimeError(f"test: unmapped url {url!r}")
        return p

    monkeypatch.setattr(processing, "ensure_cached", fake_ensure_cached)

    def add(kind: str, cx: int, cy: int) -> str:
        url = _url(kind, cx, cy)
        if url in url_to_path:
            return url
        p = tmp_path / f"{kind}_{cx}_{cy}.tif"
        (_write_dsm if kind == "dsm" else _write_rgb)(p, cx, cy)
        url_to_path[url] = p
        return url

    return add


@pytest.mark.parametrize(
    "has_right,has_bottom,has_corner",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ],
    ids=["isolated", "right-only", "bottom-only", "no-corner", "full"],
)
def test_process_tile_always_returns_25_patches(
    synth_tiles, has_right: bool, has_bottom: bool, has_corner: bool,
):
    cx, cy = (int(x) for x in CENTER_COORD.split("-"))
    rgb_url = synth_tiles("rgb", cx, cy)
    dsm_url = synth_tiles("dsm", cx, cy)

    def _nb(nx: int, ny: int) -> tuple[str, str]:
        return synth_tiles("rgb", nx, ny), synth_tiles("dsm", nx, ny)

    nr = _nb(cx + 1, cy) if has_right else None
    nb = _nb(cx, cy - 1) if has_bottom else None
    nc = _nb(cx + 1, cy - 1) if has_corner else None

    patches = processing.process_tile_from_cache(
        CENTER_COORD, rgb_url, dsm_url,
        neighbor_right=nr, neighbor_bottom=nb, neighbor_corner=nc,
        cache_patches=False,
    )

    assert len(patches) == 25, (
        f"Expected 25 (5×5) patches; got {len(patches)} with "
        f"right={has_right} bottom={has_bottom} corner={has_corner}"
    )
    for patch, _tf, row, col, tile_id, *_ in patches:
        assert patch.shape == (3, 640, 640), patch.shape
        assert patch.dtype == np.uint8
        assert tile_id == CENTER_COORD.replace("-", "_")
        assert 0 <= row <= 4
        assert 0 <= col <= 4

    rows_cols = {(r, c) for _, _, r, c, *_ in patches}
    assert rows_cols == {(r, c) for r in range(5) for c in range(5)}


def test_patch_transforms_are_deterministic(synth_tiles):
    cx, cy = (int(x) for x in CENTER_COORD.split("-"))
    rgb_url = synth_tiles("rgb", cx, cy)
    dsm_url = synth_tiles("dsm", cx, cy)

    first = processing.process_tile_from_cache(
        CENTER_COORD, rgb_url, dsm_url, cache_patches=False,
    )
    second = processing.process_tile_from_cache(
        CENTER_COORD, rgb_url, dsm_url, cache_patches=False,
    )
    assert len(first) == len(second) == 25
    for a, b in zip(first, second):
        assert (a[2], a[3]) == (b[2], b[3])
        assert tuple(a[1])[:6] == tuple(b[1])[:6]


def _write_rgb_jpeg(path: Path, origin_x_km: int, origin_y_km: int) -> None:
    """Write a 10000×10000 RGB with JPEG-in-TIFF compression (Swisstopo format)."""
    h = w = TILE_PX_RGB
    transform = Affine(0.1, 0, origin_x_km * 1000, 0, -0.1, (origin_y_km + 1) * 1000)
    row_r = np.linspace(0, 255, w, dtype=np.uint8)
    row_g = np.linspace(64, 192, w, dtype=np.uint8)
    row_b = np.linspace(128, 255, w, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=3, dtype="uint8",
        crs=CRS, transform=transform,
        compress="jpeg", jpeg_quality=90,
        photometric="ycbcr", tiled=True, blockxsize=512, blockysize=512,
    ) as dst:
        dst.write(np.broadcast_to(row_r, (h, w)), 1)
        dst.write(np.broadcast_to(row_g, (h, w)), 2)
        dst.write(np.broadcast_to(row_b, (h, w)), 3)


def test_rgb_cubic_resampling_parity_with_uncompressed(tmp_path, monkeypatch):
    """Regression anchor for the JPEG-vs-uncompressed cubic resampling bug.

    GDAL cubic resampling of a JPEG-compressed source produces different pixel
    values than cubic resampling of the same pixels stored uncompressed
    (max Δ ≈ 47 on uint8 per band, >90% of pixels differ), even though a raw
    windowed read returns identical pixels. v1.0.2 dodged this by materialising
    an uncompressed intermediate before resampling. If a future refactor removes
    the `_materialize_uncompressed` step, detection counts would silently
    regress by ~17% at nationwide scale.

    This test asserts the pipeline's RGB bands (0 and 2; band 1 is replaced by
    hillshade in fusion) are bit-identical to what cubic resampling an
    uncompressed copy of the source produces.
    """
    from rasterio.windows import Window
    from rasterio.enums import Resampling

    cx, cy = (int(x) for x in CENTER_COORD.split("-"))
    rgb_path = tmp_path / "center_jpeg.tif"
    dsm_path = tmp_path / "center_dsm.tif"
    _write_rgb_jpeg(rgb_path, cx, cy)
    _write_dsm(dsm_path, cx, cy)

    rgb_url, dsm_url = "http://fake/rgb.tif", "http://fake/dsm.tif"
    monkeypatch.setattr(
        processing, "ensure_cached",
        lambda url, **_kw: {rgb_url: rgb_path, dsm_url: dsm_path}[url],
    )

    patches = processing.process_tile_from_cache(
        CENTER_COORD, rgb_url, dsm_url, cache_patches=False,
    )
    assert len(patches) == 25

    # Reference: materialise uncompressed, cubic-resample window (0,0,3200,3200) → 640×640.
    uncompressed = processing._materialize_uncompressed(str(rgb_path))
    try:
        with rasterio.open(uncompressed) as src:
            ref_rgb = src.read(
                indexes=[1, 2, 3],
                window=Window(0, 0, 3200, 3200),
                out_shape=(3, 640, 640),
                resampling=Resampling.cubic,
            )
    finally:
        Path(uncompressed).unlink(missing_ok=True)

    patch_00 = next(p for p in patches if p[2] == 0 and p[3] == 0)[0]
    # Fusion overwrites band 1 (green) with hillshade — compare bands 0 and 2 only.
    assert np.array_equal(patch_00[0], ref_rgb[0]), "band 0 (R) differs from uncompressed reference"
    assert np.array_equal(patch_00[2], ref_rgb[2]), "band 2 (B) differs from uncompressed reference"
