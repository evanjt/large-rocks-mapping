"""Validate hillshade implementation against QGIS training data.

The training patches were generated in QGIS with azimuth=0, vertical_angle=0,
which produces overhead illumination (equivalent to gdaldem -alt 90).

These tests ensure:
1. gdaldem -alt 90 (overhead) matches the QGIS ground truth used to train the model.
2. The overhead hillshade is direction-agnostic.
3. The wrong hillshade (combined -az 315 -alt 45) does NOT match.
4. Combined mode is aspect-sensitive (sun direction matters).
5. Combined and overhead produce different results.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

import nationwide.processing as proc
from nationwide.processing import generate_hillshade

FIXTURES = Path(__file__).parent / "fixtures" / "hillshade"
DSM_DIR = FIXTURES / "dsm"
EXPECTED_DIR = FIXTURES / "expected"

PATCH_NAMES = [
    "2581_1126_0_0.tif",
    "2586_1132_3_1.tif",
    "2640_1105_0_0.tif",
    "2780_1138_1_2.tif",
    "2593_1130_1_2.tif",
]


def _load_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _write_dsm(path: Path, data: np.ndarray, res: float = 0.5) -> None:
    """Write a numpy array as a single-band GeoTIFF with correct pixel size."""
    h, w = data.shape
    transform = Affine(res, 0, 0, 0, -res, h * res)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=1,
        dtype=data.dtype, transform=transform,
    ) as dst:
        dst.write(data, 1)


def _set_mode(mode: str, az: float = 315.0, alt: float = 45.0) -> None:
    """Set hillshade config via module globals (avoids gdaldem PATH check)."""
    proc._hs_mode = mode
    proc._hs_azimuth = az
    proc._hs_altitude = alt


@pytest.mark.parametrize("patch_name", PATCH_NAMES)
def test_hillshade_matches_qgis_training_data(patch_name):
    """gdaldem -alt 90 must closely match the QGIS-generated training data.

    Thresholds: r > 0.999, MAE < 1.0, 99%+ pixels within 2.0.
    gdaldem uses slightly different rounding than QGIS at pixel level,
    so the within-2 tolerance is used instead of within-1.
    """
    _set_mode("overhead")
    dsm_path = DSM_DIR / patch_name
    expected = _load_tif(EXPECTED_DIR / patch_name).astype(np.float32)

    actual = generate_hillshade(dsm_path).astype(np.float32)

    assert actual.shape == expected.shape

    r = np.corrcoef(actual.ravel(), expected.ravel())[0, 1]
    assert r > 0.999, f"Pearson r={r:.6f}, expected > 0.999"

    mae = np.mean(np.abs(actual - expected))
    assert mae < 1.0, f"MAE={mae:.3f}, expected < 1.0"

    within_2 = np.mean(np.abs(actual - expected) <= 2.0)
    assert within_2 > 0.99, f"Only {within_2:.4%} pixels within 2.0, expected > 99%"


def test_hillshade_overhead_not_directional():
    """Overhead hillshade depends only on slope magnitude, not aspect.

    A plane tilted the same amount in 4 different directions must produce
    the same brightness.
    """
    _set_mode("overhead")
    size = 100
    res = 0.5

    brightnesses = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        slope_per_pixel = 0.5
        y_coords, x_coords = np.mgrid[0:size, 0:size]
        dsm = (dx * x_coords + dy * y_coords).astype(np.float32) * slope_per_pixel

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            dsm_path = Path(f.name)
        try:
            _write_dsm(dsm_path, dsm, res=res)
            hs = generate_hillshade(dsm_path)
        finally:
            dsm_path.unlink(missing_ok=True)

        center = hs[20:80, 20:80]
        brightnesses.append(float(center.mean()))

    for i in range(1, 4):
        assert abs(brightnesses[i] - brightnesses[0]) < 0.5, (
            f"Direction {i} brightness {brightnesses[i]:.2f} differs from "
            f"direction 0 brightness {brightnesses[0]:.2f} — hillshade is "
            f"directional, not overhead"
        )


def test_wrong_hillshade_does_not_match():
    """gdaldem -combined -az 315 -alt 45 produces wrong hillshade for training data.

    The directional hillshade has much lower correlation with the QGIS training data.
    """
    _set_mode("combined", az=315.0, alt=45.0)
    dsm_path = DSM_DIR / PATCH_NAMES[0]
    expected = _load_tif(EXPECTED_DIR / PATCH_NAMES[0]).astype(np.float32)

    wrong_hs = generate_hillshade(dsm_path).astype(np.float32)

    r = np.corrcoef(wrong_hs.ravel(), expected.ravel())[0, 1]
    mae = np.mean(np.abs(wrong_hs - expected))

    assert r < 0.95, (
        f"Wrong hillshade r={r:.4f} is too high — expected < 0.95 to confirm "
        f"directional hillshade differs from training data"
    )
    assert mae > 10, (
        f"Wrong hillshade MAE={mae:.2f} is too low — expected > 10"
    )


def test_combined_is_aspect_sensitive():
    """Combined hillshade with sun from NW (az=315) lights west-facing slopes more."""
    _set_mode("combined", az=315.0, alt=45.0)
    size = 100
    res = 0.5

    def _brightness(dx, dy):
        y_coords, x_coords = np.mgrid[0:size, 0:size]
        dsm = (dx * x_coords + dy * y_coords).astype(np.float32) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            dsm_path = Path(f.name)
        try:
            _write_dsm(dsm_path, dsm, res=res)
            hs = generate_hillshade(dsm_path)
        finally:
            dsm_path.unlink(missing_ok=True)
        return float(hs[20:80, 20:80].mean())

    # West-facing slope (rises to the east, faces the NW sun)
    west_facing = _brightness(1, 0)
    # East-facing slope (rises to the west, faces away from sun)
    east_facing = _brightness(-1, 0)

    assert west_facing != east_facing, (
        f"Combined hillshade should be aspect-sensitive but both slopes "
        f"have brightness {west_facing:.1f}"
    )


def test_combined_differs_from_overhead():
    """On real terrain, combined output should differ substantially from overhead."""
    dsm_path = DSM_DIR / PATCH_NAMES[0]

    _set_mode("overhead")
    overhead_hs = generate_hillshade(dsm_path).astype(np.float32)

    _set_mode("combined", az=315.0, alt=45.0)
    combined_hs = generate_hillshade(dsm_path).astype(np.float32)

    r = np.corrcoef(overhead_hs.ravel(), combined_hs.ravel())[0, 1]
    assert r < 0.95, (
        f"Combined vs overhead r={r:.4f} — expected < 0.95 to confirm they differ"
    )
