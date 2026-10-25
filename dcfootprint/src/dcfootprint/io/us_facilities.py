"""L0/L1 — US facility spine from Compute Atlas (pinned release v1.34.0, CC BY 4.0).

Kept: facilityType = data_center, status = operational, a cited capacityMw.operational
(235 of 747 operational datacenters in v1.34.0; "capacity is omitted when it cannot be
sourced" — absent means unpublished, not zero, and is counted, never imputed).

Capacity basis (Compute Atlas' schema does not say IT vs facility power). Decided from the
records' own notes (evidence, 2026-09-27):
  * of 28 records whose notes state an explicit IT MW figure, 26 record exactly that figure
    (and in 9 records quoting both, the smaller IT figure was chosen over utility power);
  * records quoting only a utility/total/facility figure record that figure (18 exact).
So each record is classified:
  it            recorded value equals a stated IT figure                    -> IT MW as is
  facility      recorded value equals a stated utility/total/facility figure -> / PUE(type)
  unspecified   neither stated                                               -> treated as IT
                (the dataset's demonstrated preference; the facility reading is the sensitivity)
State = Compute Atlas' 2-letter state mapped to the Ember name; GADM USA disagreements are
reported (not overridden), as for India's workbook states.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

_NUM = r"(\d+(?:\.\d+)?)\s*(?:mw|megawatts?)"
_IT = re.compile(_NUM + r"\s*(?:of\s+)?(?:critical\s+)?it\b|(?:critical\s+)?it\s+(?:load|capacity|power)\s+(?:of\s+)?"
                 + _NUM, re.I)
_UTIL = re.compile(_NUM + r"\s*(?:of\s+)?(?:total\s+)?(?:utility|grid|facility|total)\s*(?:power|capacity|feed|service)?"
                   r"|(?:utility|total|facility)\s+(?:power|capacity)\s+(?:of\s+)?" + _NUM, re.I)


def _figures(pattern, text: str) -> list[float]:
    return [float(a or b) for a, b in pattern.findall(text or "")]


def classify_basis(recorded: float, notes: str) -> str:
    close = lambda xs: any(np.isclose(recorded, x, rtol=1e-6) for x in xs)
    if close(_figures(_IT, notes)):
        return "it"
    if close(_figures(_UTIL, notes)):
        return "facility"
    return "unspecified"


def load_records() -> pd.DataFrame:
    rc = _cfg_params()["region_config"]["US"]
    recs = json.load(open(_repo_root() / rc["compute_atlas_path"], encoding="utf-8"))
    return pd.json_normalize(recs)


def build(unspecified_as: str = "it") -> pd.DataFrame:
    """Facilities-contract frame for the US (+ capacity basis columns).
    unspecified_as = 'it' (primary) | 'facility' (sensitivity)."""
    from dcfootprint.account.energy import infer_facility_type
    from dcfootprint.io import ember
    df = load_records()
    op = df[(df["facilityType"] == "data_center") & (df["status"] == "operational")].copy()
    n_operational = len(op)
    op = op[op["capacityMw.operational"].notna()].copy()
    op["capacity_recorded_mw"] = op["capacityMw.operational"].astype(float)
    op["capacity_basis"] = [classify_basis(r, n if isinstance(n, str) else "")
                            for r, n in zip(op["capacity_recorded_mw"], op["notes"])]
    op["facility_type"] = [infer_facility_type(o or "", "") for o in op["operator"].fillna("")]
    pue_cfg = _cfg_params()["energy"]["pue"]["by_facility_type"]
    pue = op["facility_type"].map(lambda t: pue_cfg.get(t, pue_cfg["unknown"])["default"])
    to_facility = (op["capacity_basis"] == "facility") | ((op["capacity_basis"] == "unspecified") & (unspecified_as == "facility"))
    op["capacity_mw"] = np.where(to_facility, op["capacity_recorded_mw"] / pue, op["capacity_recorded_mw"])

    raw = ember.load_raw("US")
    code2name = (raw.dropna(subset=["State code"]).drop_duplicates("State code")
                 .set_index("State code")["State"].to_dict())
    out = pd.DataFrame({
        "facility_id": "us-" + op["id"].astype(str),
        "operator": op["operator"], "operator_family": op["operator"],
        "region": "US", "state": op["location.state"].map(code2name),
        "latitude": op["location.lat"].astype(float), "longitude": op["location.lon"].astype(float),
        "capacity_mw": op["capacity_mw"].astype(float), "zone_id": op["location.state"].map(code2name),
        "basin_id": pd.array([pd.NA] * len(op), dtype="Int64"),
        "city": op["location.city"], "status": "Operational",
        "capacity_recorded_mw": op["capacity_recorded_mw"], "capacity_basis": op["capacity_basis"],
        "facility_type_hint": op["facility_type"], "location_precision": op["location.precision"],
        "source_url": op["sources"].map(lambda s: s[0]["url"] if s else None),
        "geocode_method": "compute_atlas",
    }).reset_index(drop=True)
    out.attrs["meta"] = {"n_operational_datacenters": int(n_operational), "n_with_capacity": int(len(out)),
                         "basis_counts": out["capacity_basis"].value_counts().to_dict(),
                         "unspecified_as": unspecified_as}
    from dcfootprint.validation import schemas
    return schemas.Facilities.validate(out)


def gadm_state_check(fac: pd.DataFrame) -> pd.DataFrame:
    """Report-only: Compute Atlas state vs GADM USA state from coordinates."""
    import geopandas as gpd
    from dcfootprint.io.gem import _TO_EMBER
    rc = _cfg_params()["region_config"]["US"]
    adm1 = gpd.read_file(_repo_root() / rc["gadm_path"], layer="ADM_ADM_1")[["NAME_1", "geometry"]]
    pts = gpd.GeoDataFrame(fac[["facility_id", "state"]], crs="EPSG:4326",
                           geometry=gpd.points_from_xy(fac["longitude"], fac["latitude"]))
    j = gpd.sjoin(pts, adm1, how="left", predicate="within").drop_duplicates("facility_id")
    j["gadm_state"] = j["NAME_1"].map(lambda s: _TO_EMBER.get(s, s) if isinstance(s, str) else s)
    j["match"] = j["gadm_state"] == j["state"]
    return pd.DataFrame(j[["facility_id", "state", "gadm_state", "match"]])


if __name__ == "__main__":
    f = build()
    print(f.attrs["meta"])
    print(f"IT MW total {f.capacity_mw.sum():,.0f} (recorded {f.capacity_recorded_mw.sum():,.0f}); states {f.state.nunique()}")
    chk = gadm_state_check(f)
    print("GADM state mismatches:", int((~chk["match"]).sum()), "of", len(chk))
    print(chk[~chk["match"]].head(10).to_string(index=False))
