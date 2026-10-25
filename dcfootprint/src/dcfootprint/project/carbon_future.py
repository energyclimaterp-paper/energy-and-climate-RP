"""L3 — future grid carbon for Q1 siting: official 2030/2050 paths as factors on today's zone CI.

A zone's carbon intensity in scenario year Y = its account-year CI (Ember, or Electricity Maps for India)
x f(Y), with f(Y) = path(Y) / path(account year) from the region's official source (siting.carbon_future):
  India  CEA National Electricity Plan 2022-32 (national average emission factor; base year from the CEA
         CO2 Baseline Database). One factor for every state; held after FY 2031-32 (no official path).
  US     NREL Cambium 2022, Mid-case, long-run marginal CO2 from direct combustion, by state (the metric
         NREL recommends for a new long-lived load; used here only as each state's relative change).
  EU     EU Reference Scenario 2020, carbon intensity of electricity and steam production, by country.
Only the RELATIVE change is taken from each path, so the account's measured spatial pattern is kept and the
path's own level (marginal vs average, net vs gross) does not enter. Zones a source does not cover get the
median factor of the covered zones, flagged in `source`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

YEARS = (2030, 2050)
US_STATES = {"AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California", "CO": "Colorado",
             "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "IA": "Iowa", "ID": "Idaho",
             "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
             "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
             "MO": "Missouri", "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
             "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada",
             "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
             "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
             "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
             "WV": "West Virginia", "WY": "Wyoming"}
EU_CODES = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany",
            "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
            "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg",
            "LV": "Latvia", "MT": "Malta", "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
            "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia"}


def _interp(points: dict[int, float], year: float, hold: bool) -> float:
    xs = sorted(points)
    if hold and year > xs[-1]:
        return float(points[xs[-1]])
    return float(np.interp(year, xs, [points[x] for x in xs]))


def _india(cfg: dict, base_year: int) -> pd.DataFrame:
    from dcfootprint.io.cea import india_grid_ef
    ef = india_grid_ef()["all_years"]
    fy = f"{base_year}-{str(base_year + 1)[-2:]}"
    base = float(ef[fy])
    pts = {int(k): float(v) for k, v in cfg["points"].items()}
    rows = [{"zone_id": "*", "year": y, "path_base": base, "path_year": _interp({base_year: base, **pts}, y, True),
             "source": f"CEA NEP 2022-32 (national); base CEA database FY {fy}"
                       + ("; held after FY 2031-32" if y > max(pts) else "")} for y in YEARS]
    return pd.DataFrame(rows)


def _us(cfg: dict, base_year: int) -> pd.DataFrame:
    path = _repo_root() / cfg["path"]
    raw = pd.read_excel(path, sheet_name="Data - Annual", header=None)
    head = raw.iloc[:5]
    metric = head.iloc[2].astype(str).str.startswith(cfg["metric"])
    blocks = metric[metric].index                         # first column of each metric block
    start = int(blocks[0])
    end = next((int(c) for c in head.columns if c > start and pd.notna(head.loc[2, c])
                and not str(head.loc[2, c]).startswith(cfg["metric"])), raw.shape[1])
    scen = head.loc[3, start:end - 1].ffill()
    yrs = pd.to_numeric(head.loc[4, start:end - 1], errors="coerce")
    cols = [c for c in range(start, end) if scen[c] == cfg["scenario"] and pd.notna(yrs[c])]
    states = raw.loc[5:, 1]
    keep = states.isin(US_STATES)
    vals = raw.loc[5:, cols][keep].apply(pd.to_numeric, errors="coerce")
    vals.columns = [int(yrs[c]) for c in cols]
    vals.index = states[keep].map(US_STATES)
    rows = []
    for st, r in vals.iterrows():
        pts = r.dropna().to_dict()
        for y in YEARS:
            rows.append({"zone_id": st, "year": y, "path_base": _interp(pts, base_year, False),
                         "path_year": _interp(pts, y, False),
                         "source": f"NREL Cambium 2022 {cfg['scenario']} LRMER, {cfg['metric'].lower()}"})
    return pd.DataFrame(rows)


def _eu(cfg: dict, base_year: int) -> pd.DataFrame:
    path = _repo_root() / cfg["path"]
    rows = []
    for code, name in EU_CODES.items():
        s = pd.read_excel(path, sheet_name=f"{code}_B", header=None)
        years = pd.to_numeric(s.iloc[1, 1:11], errors="coerce")
        hit = s.index[s[0].astype(str).str.strip() == cfg["row_label"]]
        if not len(hit):
            continue
        v = pd.to_numeric(s.iloc[int(hit[0]), 1:11], errors="coerce")
        pts = {int(y): float(x) for y, x in zip(years, v) if pd.notna(y) and pd.notna(x)}
        for y in YEARS:
            rows.append({"zone_id": name, "year": y, "path_base": _interp(pts, base_year, False),
                         "path_year": _interp(pts, y, False), "source": "EU Reference Scenario 2020"})
    return pd.DataFrame(rows)


@lru_cache(maxsize=4)
def path_factors(region: str) -> pd.DataFrame:
    """[zone_id, year, path_base, path_year, factor, source] for YEARS. zone_id '*' = applies to every zone.
    Empty when the region has no configured path or its file is missing (carbon then stays at the account year)."""
    cfg_all = _cfg_params()
    cfg = cfg_all["siting"].get("carbon_future", {}).get(region)
    if not cfg:
        return pd.DataFrame(columns=["zone_id", "year", "path_base", "path_year", "factor", "source"])
    base_year = int(cfg_all["region_config"][region]["account_year"])
    loader = {"national_path": _india, "cambium_lrmer_states": _us, "eu_ref2020": _eu}[cfg["kind"]]
    try:
        t = loader(cfg, base_year)
    except FileNotFoundError:
        return pd.DataFrame(columns=["zone_id", "year", "path_base", "path_year", "factor", "source"])
    t["factor"] = np.where(t["path_base"] > 0, t["path_year"] / t["path_base"].where(t["path_base"] > 0), 1.0)
    return t.reset_index(drop=True)


def zone_factors(region: str, zones) -> pd.DataFrame:
    """[zone_id, carbon_f2030, carbon_f2050, carbon_path_source] for the given zones. Zones the source does
    not cover get the median factor of the covered zones (flagged); no path at all -> factor 1 (flagged)."""
    t = path_factors(region)
    zones = pd.Index(sorted(set(zones)))
    out = pd.DataFrame({"zone_id": zones})
    if t.empty:
        for y in YEARS:
            out[f"carbon_f{y}"] = 1.0
        out["carbon_path_source"] = "none (carbon held at the account year)"
        return out
    if (t["zone_id"] == "*").all():
        for y in YEARS:
            out[f"carbon_f{y}"] = float(t.loc[t["year"] == y, "factor"].iloc[0])
        out["carbon_path_source"] = t["source"].iloc[-1]
        return out
    wide = t.pivot(index="zone_id", columns="year", values="factor")
    src = t.groupby("zone_id")["source"].first()
    for y in YEARS:
        out[f"carbon_f{y}"] = out["zone_id"].map(wide[y]).fillna(float(wide[y].median()))
    out["carbon_path_source"] = out["zone_id"].map(src).fillna("fallback: median factor of the covered zones")
    return out
