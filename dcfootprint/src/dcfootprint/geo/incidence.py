"""L1/L4 — incidence matrices (the exact coupling structure).

A_zone (facility x grid zone) and A_basin (facility x basin), 0/1. Basin/zone
loads are exact linear maps of site loads: W_b = sum_f A_basin[f,b] * W_f. This is
v2's L4 *core*; a learned GNN for unknown spillover is a separate Ceiling arm.
"""
from __future__ import annotations

import pandas as pd


def build_incidence(fac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (A_zone, A_basin) as facility x {zone|basin} 0/1 frames."""
    z = fac.dropna(subset=["zone_id"])
    A_zone = pd.crosstab(z["facility_id"], z["zone_id"]).clip(upper=1)
    b = fac.dropna(subset=["basin_id"])
    A_basin = pd.crosstab(b["facility_id"], b["basin_id"].astype("int64")).clip(upper=1)
    return A_zone, A_basin


def aggregate_to_basin(A_basin: pd.DataFrame, site_values: pd.Series) -> pd.Series:
    """Exact map site -> basin: basin total = sum over facilities in that basin."""
    aligned = A_basin.reindex(site_values.index).fillna(0)
    return aligned.mul(site_values, axis=0).sum(axis=0)
