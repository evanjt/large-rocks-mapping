"""Shared constants for training and inference pipelines.

These values are baked into the trained model and must not be changed
without retraining. Origins documented inline.
"""

# YOLO input size — model trained on 640×640 patches.
# See utils/helpers.py (patch_size=640), utils/arg_parser.py (--imgsz 640).
TILE_SIZE_PX = 640

# Patch overlap — ensures rocks near tile edges appear fully in ≥1 patch.
# Yields stride=430px, 4×4=16 patches per 1km tile.
OVERLAP_PX = 210

# Output resolution — training images are 50cm/pixel.
TARGET_RES = 0.5  # m/pixel

# SwissIMAGE DOP10 native resolution.
SWISSIMAGE_RES = 0.1  # m/pixel

# swissSURFACE3D raster native resolution.
DSM_RES = 0.5  # m/pixel

# Green channel replacement — best fusion strategy per baseline experiments.
# See src/baseline/baseline_dataset.py:58.
FUSION_CHANNEL = 1
