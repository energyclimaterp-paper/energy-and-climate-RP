"""L0 — AWARE 2.0 intermediate variables (Seitfudem, Berger, Müller Schmied & Boulay 2025,
doi:10.5281/zenodo.15133241): the absolute water remaining per basin-month.

    AMD_final  [m3 / m2 / month] = availability - human water consumption - EWR, per area
    area       [m2]
    remaining  [m3 / month]      = AMD * area   (negative = already over-committed)

Keyed by AWARE Basin_ID, which equals the GPKG feature id used as basin_id in geo/join.py.
Cached next to the workbook as parquet (the xlsx is slow).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_remaining(path: str | Path | None = None) -> pd.DataFrame:
    """[basin_id, month, amd_m3_per_m2, area_m2, remaining_m3]."""
    if path is None:
        p = _cfg_params()
        path = _repo_root() / p["routing"]["aware_intermediate_path"]
    path = Path(path)
    cache = path.with_name("aware_amd_remaining.parquet")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache)
    amd = pd.read_excel(path, sheet_name="AMD_final")
    area = pd.read_excel(path, sheet_name="basin_area").rename(columns={"area": "area_m2"})
    long = amd.melt(id_vars="Basin_ID", value_vars=_MONTHS, var_name="mon", value_name="amd_m3_per_m2")
    long["month"] = long["mon"].map({m: i + 1 for i, m in enumerate(_MONTHS)})
    long = long.merge(area, on="Basin_ID", how="left").rename(columns={"Basin_ID": "basin_id"})
    long["remaining_m3"] = long["amd_m3_per_m2"] * long["area_m2"]
    out = long[["basin_id", "month", "amd_m3_per_m2", "area_m2", "remaining_m3"]].astype({"basin_id": "int64"})
    from dcfootprint.validation import schemas
    out = schemas.AwareRemaining.validate(out)
    out.to_parquet(cache)
    return out
