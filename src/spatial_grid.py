"""Assign collisions to 500 m x 500 m grid cells in the British National Grid.

Coordinates are projected from WGS84 (lat/lon) to EPSG:27700 (easting/northing,
metres) so cells are true squares; the old lat/lon-degree grid produced cells
0.5 km tall but only ~0.29 km wide at UK latitudes.

Outputs
  data/interim/collisions_with_grid.csv  collisions + grid_id, grid_e, grid_n
  data/interim/grid_cells.csv            one row per cell: id, centre easting/northing, centre lat/lon
  data/interim/grid_summary.csv          per-cell collision statistics
"""

import numpy as np
import pandas as pd
from pyproj import Transformer

from config import (
    CLEAN_COLLISIONS_PATH,
    INTERIM_DATA_DIR,
    GRID_SIZE_METERS,
    WGS84_CRS,
    UK_PROJECTED_CRS,
)

_TO_BNG = Transformer.from_crs(WGS84_CRS, UK_PROJECTED_CRS, always_xy=True)
_TO_WGS = Transformer.from_crs(UK_PROJECTED_CRS, WGS84_CRS, always_xy=True)


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def assign_grid(df, cell=GRID_SIZE_METERS):
    """Add grid_ix, grid_iy (integer cell indices) and grid_id to a frame with longitude/latitude."""
    easting, northing = _TO_BNG.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    df = df.copy()
    df["grid_ix"] = np.floor(easting / cell).astype(int)
    df["grid_iy"] = np.floor(northing / cell).astype(int)
    df["grid_id"] = df["grid_ix"].astype(str) + "_" + df["grid_iy"].astype(str)
    return df


def cell_table(df, cell=GRID_SIZE_METERS):
    """One row per occupied cell with its geometric centre (not the mean of collision points)."""
    cells = df[["grid_id", "grid_ix", "grid_iy"]].drop_duplicates("grid_id").reset_index(drop=True)
    cells["centre_easting"] = (cells["grid_ix"] + 0.5) * cell
    cells["centre_northing"] = (cells["grid_iy"] + 0.5) * cell
    lon, lat = _TO_WGS.transform(cells["centre_easting"].to_numpy(), cells["centre_northing"].to_numpy())
    cells["grid_latitude"] = lat
    cells["grid_longitude"] = lon
    return cells


def grid_summary(df):
    return (
        df.groupby("grid_id")
        .agg(
            collision_count=("collision_index", "count"),
            total_casualties=("number_of_casualties", "sum"),
            average_severity=("collision_severity", "mean"),
            first_collision=("datetime", "min"),
            last_collision=("datetime", "max"),
        )
        .reset_index()
        .sort_values("collision_count", ascending=False)
    )


def main():
    print_section("SPATIAL GRID (EPSG:27700, 500 m squares)")
    df = pd.read_csv(CLEAN_COLLISIONS_PATH, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = assign_grid(df)
    cells = cell_table(df)
    print(f"Collisions: {len(df):,}   occupied cells: {len(cells):,}")
    print(f"Collisions per cell: median {df.groupby('grid_id').size().median():.0f}, max {df.groupby('grid_id').size().max():,}")

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_DATA_DIR / "collisions_with_grid.csv", index=False)
    cells.to_csv(INTERIM_DATA_DIR / "grid_cells.csv", index=False)
    grid_summary(df).to_csv(INTERIM_DATA_DIR / "grid_summary.csv", index=False)
    print("Saved collisions_with_grid.csv, grid_cells.csv, grid_summary.csv")


if __name__ == "__main__":
    main()
