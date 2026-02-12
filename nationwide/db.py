"""DuckDB storage: schema, checkpoint, detection writes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

_DB_SCHEMA_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    tile_id     VARCHAR,
    patch_id    VARCHAR,
    easting     DOUBLE,
    northing    DOUBLE,
    confidence  FLOAT,
    bbox_w_m    FLOAT,
    bbox_h_m    FLOAT,
    class_id    SMALLINT,
    inserted_at TIMESTAMP DEFAULT current_timestamp
);
"""

_DB_SCHEMA_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS processed_tiles (
    tile_id       VARCHAR PRIMARY KEY,
    n_detections  INTEGER,
    processed_at  TIMESTAMP DEFAULT current_timestamp
);
"""


@dataclass
class Detection:
    """A single detection in EPSG:2056 coordinates."""
    tile_id: str
    patch_id: str
    easting: float
    northing: float
    confidence: float
    bbox_w_m: float
    bbox_h_m: float
    class_id: int = 0


def init_db(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Create or open the detections database and ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_DB_SCHEMA_DETECTIONS)
    con.execute(_DB_SCHEMA_CHECKPOINTS)
    return con


def get_processed_tiles(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Return set of tile_ids that have already been processed."""
    rows = con.execute("SELECT tile_id FROM processed_tiles").fetchall()
    return {r[0] for r in rows}


def mark_tile_done(
    con: duckdb.DuckDBPyConnection, tile_id: str, n_detections: int,
) -> None:
    """Record a tile as processed in the checkpoint table."""
    con.execute(
        "INSERT INTO processed_tiles (tile_id, n_detections) VALUES (?, ?) "
        "ON CONFLICT (tile_id) DO UPDATE SET "
        "n_detections = excluded.n_detections, processed_at = NOW()",
        [tile_id, n_detections],
    )


def write_detections(
    con: duckdb.DuckDBPyConnection, detections: list[Detection],
) -> int:
    """Bulk-insert detections using executemany. Returns row count."""
    if not detections:
        return 0
    rows = [
        (d.tile_id, d.patch_id, d.easting, d.northing,
         d.confidence, d.bbox_w_m, d.bbox_h_m, d.class_id)
        for d in detections
    ]
    con.executemany(
        "INSERT INTO detections "
        "(tile_id, patch_id, easting, northing, confidence, bbox_w_m, bbox_h_m, class_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)
