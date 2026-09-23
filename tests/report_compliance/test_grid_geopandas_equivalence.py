"""Concrete pass/fail proof that the GeoPandas/Shapely grid (report-compliance
demonstration) assigns EXACTLY the same grid_id as the frozen pyproj-based
grid (src/spatial_grid.py), which this test only reads -- it never modifies
data/interim/collisions_clean.csv or any other frozen artifact.
"""

import numpy as np
import pandas as pd
import pytest

from spatial_grid import assign_grid
from report_compliance.spatial_grid_geopandas import assign_grid_geopandas
from config import CLEAN_COLLISIONS_PATH, GRID_SIZE_METERS


@pytest.fixture(scope="module")
def sample_points():
    df = pd.read_csv(CLEAN_COLLISIONS_PATH, low_memory=False, usecols=["longitude", "latitude"])
    return df.sample(n=20_000, random_state=42).reset_index(drop=True)


def test_exact_cell_id_match_on_real_data_sample(sample_points):
    """Every one of 20,000 sampled real collision points must land in the
    identical grid_id under both implementations -- 0 mismatches, not a
    tolerance band."""
    frozen = assign_grid(sample_points)
    geo = assign_grid_geopandas(sample_points)

    mismatches = (frozen["grid_id"].to_numpy() != geo["grid_id"].to_numpy()).sum()
    assert mismatches == 0, f"{mismatches} / {len(sample_points)} grid_id mismatches"
    # also confirm the underlying integer indices agree, not just the string id
    np.testing.assert_array_equal(frozen["grid_ix"].to_numpy(), geo["grid_ix"].to_numpy())
    np.testing.assert_array_equal(frozen["grid_iy"].to_numpy(), geo["grid_iy"].to_numpy())


def test_exact_cell_id_match_on_full_dataset():
    """Same check over the entire cleaned dataset (slower, run once)."""
    df = pd.read_csv(CLEAN_COLLISIONS_PATH, low_memory=False, usecols=["longitude", "latitude"])
    frozen = assign_grid(df)
    geo = assign_grid_geopandas(df)
    mismatches = (frozen["grid_id"].to_numpy() != geo["grid_id"].to_numpy()).sum()
    assert mismatches == 0, f"{mismatches} / {len(df)} grid_id mismatches"


def test_synthetic_cell_boundary_points():
    """Points placed exactly on projected cell boundaries are the one case
    where the spatial-join tie-break logic could disagree with floor
    division; construct them directly in WGS84 near a known easting/northing
    grid line and confirm both implementations still agree."""
    from pyproj import Transformer

    to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    cell = GRID_SIZE_METERS
    # a grid of eastings/northings straddling several cell boundaries
    offsets = [-1.0, 0.0, 1.0, cell - 1.0, cell, cell + 1.0]
    eastings = [400_000 + o for o in offsets]
    northings = [300_000 + o for o in offsets]
    rows = []
    for e in eastings:
        for n in northings:
            lon, lat = to_wgs.transform(e, n)
            rows.append({"longitude": lon, "latitude": lat})
    df = pd.DataFrame(rows)

    frozen = assign_grid(df)
    geo = assign_grid_geopandas(df)
    mismatches = (frozen["grid_id"].to_numpy() != geo["grid_id"].to_numpy()).sum()
    assert mismatches == 0, f"{mismatches} / {len(df)} boundary-point mismatches"
