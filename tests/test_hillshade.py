"""Validate hillshade implementation against QGIS training data.

The training patches were generated in QGIS with azimuth=0, vertical_angle=0,
which produces overhead illumination: shade = 255 * cos(slope).  Our pipeline
computes this as 255 / sqrt(1 + dzdx^2 + dzdy^2) via Horn's gradient method.

These tests ensure:
1. Our hillshade matches the QGIS ground truth used to train the model.
2. The optimized formula is mathematically identical to the original.
3. The hillshade is direction-agnostic (overhead, not directional).
4. The wrong hillshade (gdaldem -az 315 -alt 45) does NOT match.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

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


@pytest.mark.parametrize("patch_name", PATCH_NAMES)
def test_hillshade_matches_qgis_training_data(patch_name):
    """Our hillshade must closely match the QGIS-generated training data.

    Thresholds based on full 992-patch validation:
    r=0.999, MAE=0.586, 99.4%+ pixels within 1.0.
    """
    dsm = _load_tif(DSM_DIR / patch_name).astype(np.float32)
    expected = _load_tif(EXPECTED_DIR / patch_name).astype(np.float32)

    actual = generate_hillshade(dsm, res=0.5).astype(np.float32)

    assert actual.shape == expected.shape

    # Pearson correlation
    r = np.corrcoef(actual.ravel(), expected.ravel())[0, 1]
    assert r > 0.998, f"Pearson r={r:.6f}, expected > 0.998"

    # Mean absolute error
    mae = np.mean(np.abs(actual - expected))
    assert mae < 1.0, f"MAE={mae:.3f}, expected < 1.0"

    # 99% of pixels within 1.0
    within_1 = np.mean(np.abs(actual - expected) <= 1.0)
    assert within_1 > 0.99, f"Only {within_1:.4%} pixels within 1.0, expected > 99%"


def test_hillshade_formula_equivalence():
    """Optimized 1/sqrt(1+x) must produce identical output to cos(arctan(sqrt(x)))."""
    dsm = _load_tif(DSM_DIR / PATCH_NAMES[0]).astype(np.float32)
    res = 0.5

    padded = np.pad(dsm, 1, mode="edge")

    dzdx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * res)

    dzdy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
        - (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8.0 * res)

    sum_sq = dzdx**2 + dzdy**2

    # Original formula: cos(arctan(sqrt(sum_sq)))
    original = np.cos(np.arctan(np.sqrt(sum_sq)))

    # Optimized formula: 1/sqrt(1 + sum_sq)
    optimized = 1.0 / np.sqrt(1.0 + sum_sq)

    # Mathematically identical; float32 intermediate precision causes ~1.5e-7 diffs
    np.testing.assert_allclose(original, optimized, atol=2e-7, rtol=0)


def test_hillshade_overhead_not_directional():
    """Overhead hillshade depends only on slope magnitude, not aspect.

    A plane tilted the same amount in 4 different directions must produce
    the same brightness.
    """
    size = 100
    res = 0.5

    brightnesses = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        # Create a plane with constant slope in direction (dx, dy)
        slope_per_pixel = 0.5  # meters per pixel
        y_coords, x_coords = np.mgrid[0:size, 0:size]
        dsm = (dx * x_coords + dy * y_coords).astype(np.float32) * slope_per_pixel
        hs = generate_hillshade(dsm, res=res)
        # Take center region to avoid edge effects
        center = hs[20:80, 20:80]
        brightnesses.append(float(center.mean()))

    # All four orientations should give the same brightness
    for i in range(1, 4):
        assert abs(brightnesses[i] - brightnesses[0]) < 0.5, (
            f"Direction {i} brightness {brightnesses[i]:.2f} differs from "
            f"direction 0 brightness {brightnesses[0]:.2f} — hillshade is "
            f"directional, not overhead"
        )


def test_wrong_hillshade_does_not_match():
    """gdaldem -az 315 -alt 45 -combined produces wrong hillshade.

    This documents the known bug in the bash pipeline. The directional
    hillshade has much lower correlation with the QGIS training data.
    """
    dsm_path = DSM_DIR / PATCH_NAMES[0]
    expected = _load_tif(EXPECTED_DIR / PATCH_NAMES[0]).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        cmd = (
            f"gdaldem hillshade {dsm_path} {tmp.name} "
            f"-az 315 -alt 45 -combined -compute_edges "
            f"-of GTiff -q"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            pytest.skip(f"gdaldem not available: {result.stderr.decode()}")

        wrong_hs = _load_tif(Path(tmp.name)).astype(np.float32)

    r = np.corrcoef(wrong_hs.ravel(), expected.ravel())[0, 1]
    mae = np.mean(np.abs(wrong_hs - expected))

    assert r < 0.95, (
        f"Wrong hillshade r={r:.4f} is too high — expected < 0.95 to confirm "
        f"directional hillshade differs from training data"
    )
    assert mae > 10, (
        f"Wrong hillshade MAE={mae:.2f} is too low — expected > 10"
    )
