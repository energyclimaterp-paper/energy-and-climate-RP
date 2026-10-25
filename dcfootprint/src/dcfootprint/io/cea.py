"""L0 — CEA CO2 Baseline Database (V22): India's official grid emission factor.
Primary India carbon source (national-annual, consumption-relevant on the
synchronous national grid).

We take the **grid emission rate incl. RES & captive** (emissions / total generation
incl. renewables) — the right factor for grid-consumption carbon — NOT the plain
"Weighted Average Emission Rate" (which excludes RES from the denominator and so runs
~0.82, overstating consumption carbon). The incl-RES rate (~0.68) cross-validates
against Ember's independent CO2-intensity (~0.6).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root

_EF_SUBSTR = "Grid Emission Rate (Incl"   # Weighted Average Grid Emission Rate (Incl. RES,Captive)
_HDR_SUBSTR = "Emission Factors (tCO2/MWh)"


def _row_containing(r: pd.DataFrame, substr: str) -> int:
    hits = r.index[r.apply(lambda row: row.astype(str).str.contains(substr, regex=False).any(), axis=1)]
    if len(hits) == 0:
        raise ValueError(f"CEA Results: no row contains {substr!r}")
    return int(hits[0])


def india_grid_ef(path: str | Path | None = None) -> dict:
    """{'ef_tco2_per_mwh', 'fy', 'all_years'} — national grid EF (incl. RES), latest FY.
    Label-anchored parse (not row-hardcoded), so it survives sheet edits."""
    path = Path(path) if path else _repo_root() / "data" / "cea" / "CEA_Database_V22.xlsx"
    r = pd.read_excel(path, sheet_name="Results", header=None)
    hdr_row, ef_row = _row_containing(r, _HDR_SUBSTR), _row_containing(r, _EF_SUBSTR)
    years, vals = r.loc[hdr_row], r.loc[ef_row]
    year_cols = [c for c in r.columns
                 if isinstance(years[c], str) and "-" in years[c] and years[c][:4].isdigit()]
    series = {years[c]: float(vals[c]) for c in year_cols if pd.notna(vals[c])}
    latest_fy = sorted(series)[-1]
    return {"ef_tco2_per_mwh": series[latest_fy], "fy": latest_fy, "all_years": series}


if __name__ == "__main__":
    ef = india_grid_ef()
    print(f"India grid EF (CEA V22, incl. RES): {ef['ef_tco2_per_mwh']:.4f} tCO2/MWh  (FY {ef['fy']})")
    print("all years:", {k: round(v, 3) for k, v in ef["all_years"].items()})
