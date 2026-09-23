"""GeoPandas/Shapely rewrite of the 500 m British National Grid cell assignment.

Report-compliance demonstration only (DA1 proposal item: "GeoPandas/Shapely
for spatial processing"). Does NOT replace, import, or get imported by
src/spatial_grid.py, and its output never touches any frozen classifier
artifact. See tests/report_compliance/test_grid_geopandas_equivalence.py for
the exact cell-ID equivalence proof against the frozen implementation.

Unlike src/spatial_grid.py (manual pyproj.Transformer + np.floor arithmetic),
this module:
  1. Represents collisions as a GeoDataFrame of Points and reprojects with
     GeoDataFrame.to_crs (GeoPandas/pyproj machinery).
  2. Builds one Shapely `box` polygon per occupied cell.
  3. Assigns each point to a cell via a genuine spatial join
     (`geopandas.sjoin`, predicate="within"), not by re-using index
     arithmetic -- this actually tests that the polygon geometrically
     contains the point, rather than just repeating the same floor-division.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GRID_SIZE_METERS, WGS84_CRS, UK_PROJECTED_CRS  # noqa: E402


def build_cell_polygons(ix, iy, cell=GRID_SIZE_METERS):
    """One Shapely box polygon per (ix, iy) grid index pair, in projected metres."""
    ix = np.asarray(ix)
    iy = np.asarray(iy)
    return [
        box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell)
        for i, j in zip(ix, iy)
    ]


def assign_grid_geopandas(df, cell=GRID_SIZE_METERS):
    """GeoPandas/Shapely equivalent of spatial_grid.assign_grid.

    Adds grid_ix, grid_iy, grid_id columns, matching the frozen
    implementation's naming and cell-index convention exactly (same origin,
    same cell size, same floor-division direction) so the two can be
    compared cell-for-cell.
    """
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84_CRS,
    ).to_crs(UK_PROJECTED_CRS)

    ix = np.floor(points.geometry.x.to_numpy() / cell).astype(int)
    iy = np.floor(points.geometry.y.to_numpy() / cell).astype(int)
    points["grid_ix"] = ix
    points["grid_iy"] = iy

    occupied = pd.DataFrame({"grid_ix": ix, "grid_iy": iy}).drop_duplicates().reset_index(drop=True)
    occupied["geometry"] = build_cell_polygons(occupied["grid_ix"], occupied["grid_iy"], cell)
    occupied["cell_grid_id"] = occupied["grid_ix"].astype(str) + "_" + occupied["grid_iy"].astype(str)
    cells_gdf = gpd.GeoDataFrame(occupied, geometry="geometry", crs=UK_PROJECTED_CRS)

    joined = gpd.sjoin(
        points,
        cells_gdf[["grid_ix", "grid_iy", "cell_grid_id", "geometry"]],
        how="left",
        predicate="within",
        lsuffix="pt",
        rsuffix="cell",
    )
    # A point that lands exactly on a shared cell boundary can spatially
    # satisfy "within" for more than one adjacent polygon (box() geometry is
    # closed on all edges). Break ties by preferring the match whose cell
    # indices equal the point's own floor-division indices, which is always
    # present because that polygon was built from this point's own (ix, iy).
    joined["is_own_cell"] = (joined["grid_ix_pt"] == joined["grid_ix_cell"]) & (
        joined["grid_iy_pt"] == joined["grid_iy_cell"]
    )
    joined = joined.sort_values("is_own_cell", ascending=False)
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()

    result = df.copy()
    result["grid_ix"] = ix
    result["grid_iy"] = iy
    result["grid_id"] = joined["cell_grid_id"].to_numpy()
    return result
