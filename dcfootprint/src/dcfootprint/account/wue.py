"""L2 - on-site WUE per facility-month: constant (round 5) or climate-dependent.

water_onsite.wue_model:
  constant  every facility takes wue_default (1.9 L/kWh) unless its operator discloses a fleet WUE.
  climate   facilities without a disclosed operator WUE take the cooling-tower WUE of their own
            location and month:
                WUE(Tw) = max(0, 0.0005112 Tw^2 - 0.04982 Tw + 2.387)     [L per IT kWh, Tw in F]
            the "fixed cold-water temperature" configuration of the empirical commercial-cooling-
            tower model in Gupta, Hossen, Li, Ren & Islam (2024), "A dataset for research on water
            sustainability", ACM e-Energy '24, pp. 442-446 (eq. 2 as restated in arXiv:2412.03716).
            Tw = monthly mean wet-bulb temperature at the facility for the account year, NASA POWER
            T2MWET (MERRA-2), cached under data/climate/. Disclosed operator WUEs are measured fleet
            values and are kept.
The curve is for water-cooled (cooling tower) plants; it does not model air-cooled chillers, which
use little water (see validation/reported.py for what operators report).
Evaluating the curve at the monthly MEAN wet-bulb (not hour by hour) is a stated approximation.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root

_POWER = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2MWET&community=RE"
          "&longitude={lon:.4f}&latitude={lat:.4f}&start={year}&end={year}&format=JSON")


def gupta_cold_water_wue(tw_c) -> np.ndarray:
    """Cooling-tower on-site WUE (L/kWh IT) at wet-bulb temperature tw_c (deg C)."""
    tw_f = np.asarray(tw_c, float) * 9.0 / 5.0 + 32.0
    return np.maximum(0.0005112 * tw_f ** 2 - 0.04982 * tw_f + 2.387, 0.0)


def _fetch_point(lat: float, lon: float, year: int) -> dict[int, float]:
    """Monthly T2MWET at one point. Fails fast (3 tries, 30 s each) with a clear message rather than
    silently falling back to the default WUE, which would change the results without notice."""
    url = _POWER.format(lat=lat, lon=lon, year=year)
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dcfootprint"}), timeout=30) as r:
                v = json.load(r)["properties"]["parameter"]["T2MWET"]
            out = {int(k[-2:]): float(x) for k, x in v.items() if 1 <= int(k[-2:]) <= 12}
            if len(out) != 12 or any(x <= -900 for x in out.values()):    # -999 = POWER fill value
                raise ValueError(f"incomplete T2MWET at {lat},{lon} {year}")
            return out
        except (OSError, ValueError) as e:
            if i == 2:
                raise RuntimeError(
                    f"NASA POWER wet-bulb unavailable for {lat},{lon} {year} ({e}). Run once online to fill "
                    f"data/climate/, or set water_onsite.wue_model: constant in parameters.yaml.") from e
            time.sleep(3 * (i + 1))
    raise RuntimeError("unreachable")


def wet_bulb_monthly(points: pd.DataFrame, year: int) -> pd.DataFrame:
    """[lat_r, lon_r, month, tw_c] for the unique points (latitude/longitude rounded to 0.01 deg),
    fetched once from NASA POWER and cached in data/climate/nasa_power_t2mwet_<year>.parquet."""
    cache = _repo_root() / "data" / "climate" / f"nasa_power_t2mwet_{year}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["lat_r", "lon_r", "month", "tw_c"])
    pts = points[["latitude", "longitude"]].dropna().round(2).drop_duplicates()
    pts.columns = ["lat_r", "lon_r"]
    known = set(map(tuple, have[["lat_r", "lon_r"]].drop_duplicates().to_numpy()))
    new = []
    for lat, lon in pts.itertuples(index=False):
        if (lat, lon) in known:
            continue
        for m, v in _fetch_point(lat, lon, year).items():
            new.append({"lat_r": lat, "lon_r": lon, "month": m, "tw_c": v})
        time.sleep(0.2)
    if new:
        have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
        have.astype({"lat_r": float, "lon_r": float, "month": int, "tw_c": float}).to_parquet(cache)
    return have.astype({"lat_r": float, "lon_r": float, "month": int, "tw_c": float})


def facility_month_wue(fac: pd.DataFrame, params: dict, year: int) -> pd.DataFrame:
    """[facility_id, month, wue_l_per_kwh, wue_source, tw_c] for every facility x month."""
    w = params["water_onsite"]
    model = w.get("wue_model", "constant")
    default = float(w["wue_default"]["default"])
    byop = {k.lower(): float(v) for k, v in w.get("wue_by_operator", {}).items()}
    f = fac[["facility_id", "operator", "latitude", "longitude"]].drop_duplicates("facility_id").copy()
    opl = f["operator"].fillna("").str.lower()
    f["op_wue"] = np.nan
    for op, val in byop.items():
        f.loc[opl.str.contains(op, regex=False), "op_wue"] = val
    fm = f.merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    fm["tw_c"] = np.nan
    if model == "climate":
        tw = wet_bulb_monthly(f, year)
        fm["lat_r"], fm["lon_r"] = fm["latitude"].round(2), fm["longitude"].round(2)
        fm = fm.drop(columns="tw_c").merge(tw, on=["lat_r", "lon_r", "month"], how="left")
        clim = gupta_cold_water_wue(fm["tw_c"])
        fm["wue_l_per_kwh"] = np.where(fm["op_wue"].notna(), fm["op_wue"], np.where(fm["tw_c"].notna(), clim, default))
        fm["wue_source"] = np.where(fm["op_wue"].notna(), "operator_disclosed",
                                    np.where(fm["tw_c"].notna(), "climate_gupta2024", "default_no_climate"))
    elif model == "constant":
        fm["wue_l_per_kwh"] = fm["op_wue"].fillna(default)
        fm["wue_source"] = np.where(fm["op_wue"].notna(), "operator_disclosed", "default")
    else:
        raise ValueError(f"unknown water_onsite.wue_model {model!r} (constant | climate)")
    return fm[["facility_id", "month", "wue_l_per_kwh", "wue_source", "tw_c"]]


def recompute_onsite(account: pd.DataFrame, wue_fm: pd.DataFrame) -> pd.DataFrame:
    """Account with on-site (scope-1) water and its scarcity rebuilt from a new facility-month WUE;
    scope-2 columns untouched, totals rebuilt. Same formulas as account/build.py."""
    a = account.drop(columns=[c for c in ("wue_l_per_kwh", "wue_source", "tw_c") if c in account.columns])
    a = a.merge(wue_fm, on=["facility_id", "month"], how="left", validate="many_to_one")
    if a["wue_l_per_kwh"].isna().any():
        raise KeyError("WUE missing for some facility-months")
    a["water_onsite_l"] = a["wue_l_per_kwh"] * a["e_it_mwh"] * 1000.0
    a["water_scarcity_onsite_l_eq"] = a["water_onsite_l"] * a["cf"]
    a["water_phys_l"] = a["water_onsite_l"] + a["water_grid_l"]
    a["water_scarcity_l_eq"] = a["water_scarcity_onsite_l_eq"] + a["water_scarcity_grid_l_eq"]
    return a
