"""Global Moran's I of per-cell collision counts: how strongly do neighbouring cells resemble each other?

Cells: the modelled cells (>= 3 collisions in 2021-2023). Variable: each cell's 2021-2023 collision count (training
years only). Weights: queen contiguity on the 500 m grid, restricted to modelled neighbours, row-standardised;
cells without a modelled neighbour are left out of the statistic and counted. p-value: 499 random permutations
of the values over the cells. Read-only; writes only outputs/dashboard/morans_i.json.
"""

import datetime
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from ._common import DASHBOARD_DIR, PROJECT_ROOT
except ImportError:  # run as a script
    from _common import DASHBOARD_DIR, PROJECT_ROOT

GRID = PROJECT_ROOT / "data" / "interim" / "collisions_with_grid.csv"
OUT = DASHBOARD_DIR / "morans_i.json"
N_PERM = 499
SEED = 42
MIN_COLLISIONS = 3


def main():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(GRID, usecols=["grid_id", "grid_ix", "grid_iy", "collision_year"])
    c = c[c["collision_year"] <= 2023]
    cells = c.groupby("grid_id").agg(n=("grid_id", "size"), ix=("grid_ix", "first"), iy=("grid_iy", "first"))
    cells = cells[cells["n"] >= MIN_COLLISIONS].reset_index()
    key = cells["ix"].to_numpy(np.int64) * 1_000_000 + cells["iy"].to_numpy(np.int64)
    index = pd.Series(np.arange(len(cells)), index=key)

    rows, cols = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = index.reindex(key + dx * 1_000_000 + dy)
            ok = nb.notna().to_numpy()
            rows.append(np.arange(len(cells))[ok]); cols.append(nb.to_numpy()[ok].astype(int))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    w = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cells),) * 2)
    deg = np.asarray(w.sum(axis=1)).ravel()
    keep = deg > 0
    w = sp.diags(1 / deg[keep]) @ w[keep][:, keep]
    x = cells["n"].to_numpy(float)[keep]

    def moran(v):
        z = v - v.mean()
        return float(len(z) * (z @ (w @ z)) / (w.sum() * (z @ z)))

    i_obs = moran(x)
    rng = np.random.default_rng(SEED)
    perm = np.array([moran(rng.permutation(x)) for _ in range(N_PERM)])
    p = float((1 + (perm >= i_obs).sum()) / (N_PERM + 1))
    expected = -1 / (len(x) - 1)

    out = {
        "morans_i": i_obs, "expected_under_randomness": expected, "p_value": p, "permutations": N_PERM,
        "n_cells_used": int(len(x)), "n_cells_without_modelled_neighbour": int((~keep).sum()),
        "variable": "per-cell collision count, 2021-2023", "weights": "queen contiguity, modelled neighbours only, row-standardised",
        "interpretation": (
            f"Moran's I = {i_obs:.3f} (expected {expected:.4f} if cells were independent; permutation p = {p:.3f}): "
            f"significant positive spatial autocorrelation, so neighbouring cells have similar collision counts and "
            f"treating cells as independent understates uncertainty."
            if i_obs > expected and p < 0.05 else
            f"Moran's I = {i_obs:.3f} (expected {expected:.4f}; permutation p = {p:.3f}): no significant positive spatial autocorrelation detected."
        ),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(out["interpretation"], f"[{len(x):,} cells, {out['n_cells_without_modelled_neighbour']:,} isolated]")


if __name__ == "__main__":
    main()
