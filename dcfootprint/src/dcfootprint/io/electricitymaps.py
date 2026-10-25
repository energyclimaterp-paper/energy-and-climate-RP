"""L0 — India CONSUMPTION-based grid carbon from Electricity Maps (addresses R2).

The account's default India carbon uses Ember STATE-GENERATION intensity (a state's own
plants). A datacenter draws the POOLED regional grid, so the decision-relevant intensity is
CONSUMPTION-based (import/export flow-traced). Electricity Maps serves flow-traced LIFECYCLE
carbon intensity for India's five regional grids (IN-NO/WE/SO/EA/NE), hourly; we average to
monthly and map onto states.

Caveat to state in the paper: EM intensity is `lifecycle` (incl. upstream), while Ember's is
direct-generation, so the EM-vs-Ember gap blends (consumption vs generation) with (lifecycle
vs direct). The consumption basis is the R2 fix; the lifecycle component is a separate,
stated modelling choice.

Key: read from data/electricitymaps.key (gitignored) or $ELECTRICITYMAPS_API_KEY; never
logged. Hourly series cached to data/em/ so a run hits the API once.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root

_BASE = "https://api.electricitymap.org/v3"
EM_ZONES = ["IN-NO", "IN-WE", "IN-SO", "IN-EA", "IN-NE"]

# Indian state -> regional load-dispatch grid = the five EM India zones. Spellings match the
# account's state labels; unmapped states fall back to the national IN zone (see map_states).
STATE_TO_EM = {
    "Delhi": "IN-NO", "Haryana": "IN-NO", "Punjab": "IN-NO", "Rajasthan": "IN-NO",
    "Uttar Pradesh": "IN-NO", "Uttarakhand": "IN-NO", "Himachal Pradesh": "IN-NO",
    "Jammu and Kashmir": "IN-NO", "Chandigarh": "IN-NO",
    "Maharashtra": "IN-WE", "Gujarat": "IN-WE", "Madhya Pradesh": "IN-WE",
    "Chhattisgarh": "IN-WE", "Goa": "IN-WE", "Dadra and Nagar Haveli": "IN-WE",
    "Tamil Nadu": "IN-SO", "Karnataka": "IN-SO", "Kerala": "IN-SO", "Andhra Pradesh": "IN-SO",
    "Telangana": "IN-SO", "Puducherry": "IN-SO",
    "West Bengal": "IN-EA", "Bihar": "IN-EA", "Jharkhand": "IN-EA", "Odisha": "IN-EA",
    "Sikkim": "IN-EA",
    "Assam": "IN-NE", "Meghalaya": "IN-NE", "Tripura": "IN-NE", "Manipur": "IN-NE",
    "Mizoram": "IN-NE", "Nagaland": "IN-NE", "Arunachal Pradesh": "IN-NE",
}


def _key() -> str:
    p = _repo_root() / "data" / "electricitymaps.key"
    k = p.read_text(encoding="utf-8-sig").strip() if p.exists() else ""
    return k or os.environ.get("ELECTRICITYMAPS_API_KEY", "").strip()


def _get(path: str, _tries: int = 3) -> dict:
    for i in range(_tries):
        try:
            req = urllib.request.Request(_BASE + path, headers={"auth-token": _key()})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < _tries - 1:
                time.sleep(2 * (i + 1)); continue
            raise


def fetch_zone_monthly(zone: str, year: int) -> pd.DataFrame:
    """[zone, month, ci_consumption_gco2_kwh, n_hours] — monthly mean of hourly flow-traced
    lifecycle CI; fetched in monthly chunks, cached to data/em/."""
    cache = _repo_root() / "data" / "em" / f"ci_{zone}_{year}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    frames, fmt = [], "%Y-%m-%dT%H:%M:%SZ"
    t = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    stop = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
    while t < stop:                                   # API caps past-range at 10 days of hourly data
        e = min(t + pd.Timedelta(days=10), stop)
        resp = _get(f"/carbon-intensity/past-range?zone={zone}&start={t.strftime(fmt)}&end={e.strftime(fmt)}")
        frames.append(pd.DataFrame(resp.get("data", [])))
        time.sleep(0.1)
        t = e
    raw = pd.concat(frames, ignore_index=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.drop_duplicates("datetime")
    raw["month"] = raw["datetime"].dt.month
    out = (raw.groupby("month")
           .agg(ci_consumption_gco2_kwh=("carbonIntensity", "mean"),
                n_hours=("carbonIntensity", "size")).reset_index())
    out.insert(0, "zone", zone)
    out.to_parquet(cache)
    return out


def india_region_month_ci(year: int = 2024) -> pd.DataFrame:
    """[zone, month, ci_consumption_gco2_kwh, n_hours] for all five India regional grids."""
    return pd.concat([fetch_zone_monthly(z, year) for z in EM_ZONES], ignore_index=True)


def india_state_month_ci(year: int = 2024) -> pd.DataFrame:
    """[state, month, ci_consumption_gco2_kwh] — each state gets its EM regional grid's
    monthly consumption CI. Unmapped states are omitted (caller falls back to Ember)."""
    reg = india_region_month_ci(year).set_index(["zone", "month"])["ci_consumption_gco2_kwh"]
    rows = [{"state": s, "month": m, "ci_consumption_gco2_kwh": float(reg.get((z, m)))}
            for s, z in STATE_TO_EM.items() for m in range(1, 13) if (z, m) in reg.index]
    return pd.DataFrame(rows)


def apply_consumption_ci(ci: pd.DataFrame, year: int) -> pd.DataFrame:
    """Override Ember state-GENERATION CI with EM regional CONSUMPTION CI where available.
    `ci` = [zone_id (= state), month, ci_gco2_per_kwh, ci_source]; states not in STATE_TO_EM
    (or months EM lacks) keep the Ember value. ci_source flags which rows were replaced."""
    import numpy as np
    cons = india_state_month_ci(year).rename(columns={"state": "zone_id"})
    m = ci.merge(cons, on=["zone_id", "month"], how="left")
    use = m["ci_consumption_gco2_kwh"].notna().to_numpy()
    out = ci.copy()
    out["ci_gco2_per_kwh"] = np.where(use, m["ci_consumption_gco2_kwh"].to_numpy(),
                                      ci["ci_gco2_per_kwh"].to_numpy())
    out["ci_source"] = np.where(use, "em_consumption", ci["ci_source"].astype(str).to_numpy())
    return out


if __name__ == "__main__":
    reg = india_region_month_ci(2024)
    ann = reg.groupby("zone")["ci_consumption_gco2_kwh"].mean().round(0)
    print("EM 2024 consumption CI (annual mean, gCO2/kWh) by India region:")
    print(ann.to_string())
    print("\nmonthly min/max per zone:")
    print(reg.groupby("zone")["ci_consumption_gco2_kwh"].agg(["min", "max"]).round(0).to_string())
