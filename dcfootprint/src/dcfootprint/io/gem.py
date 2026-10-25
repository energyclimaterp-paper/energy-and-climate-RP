"""L0 — GEM Global Integrated Power Tracker: power plants with location, type, capacity.

Used for (a) scope-2 scarcity at the basins where electricity is GENERATED (geo/
generation_basins.py) and (b) the Q1 candidate grid + grid fossil share (decisions/
siting.py). The 23 MB workbook is slow to read, so the India operating subset is cached
next to it as parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

_COLS = {"GEM unit/phase ID": "plant_id", "Type": "type", "Plant / Project name": "plant",
         "Capacity (MW)": "capacity_mw", "Status": "status", "Latitude": "latitude", "Longitude": "longitude",
         "Subnational unit (state, province)": "gem_state"}
_FOSSIL = {"coal", "oil/gas"}


def region_plants(region: str = "India", status: str = "operating") -> pd.DataFrame:
    """GEM plants of every country in region_config[region].gem_countries, state of record assigned."""
    rc = _cfg_params()["region_config"][region]
    return assign_state_of_record(load_plants(list(rc["gem_countries"]), status, tag=region, assign=False), region)


def load_plants(country="India", status: str = "operating", path: str | Path | None = None,
                tag: str | None = None, assign: bool = True) -> pd.DataFrame:
    """[plant_id, type, plant, capacity_mw, status, latitude, longitude, gem_state, country, fossil]
    for one country or a list (cached per `tag`); India state of record assigned if assign."""
    root = _repo_root()
    if path is None:
        params = _cfg_params()
        path = root / params["grid"]["gem_path"]
    path = Path(path)
    countries = [country] if isinstance(country, str) else list(country)
    tag = tag or countries[0]
    cache = path.with_name(f"gem_{tag.lower().replace(' ', '_')}_{status}_v4.parquet")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        g = pd.read_parquet(cache)
        return assign_state_of_record(g) if assign else g
    g = pd.read_excel(path, sheet_name="Power facilities", usecols=["Country/area", *_COLS])
    g = g[g["Country/area"].isin(countries) & (g["Status"] == status)].rename(columns=_COLS)
    g = g.rename(columns={"Country/area": "country"})
    g["capacity_mw"] = pd.to_numeric(g["capacity_mw"], errors="coerce")
    g["latitude"] = pd.to_numeric(g["latitude"], errors="coerce")
    g["longitude"] = pd.to_numeric(g["longitude"], errors="coerce")
    g = g.dropna(subset=["capacity_mw", "latitude", "longitude"])
    g = g[g["capacity_mw"] > 0].reset_index(drop=True)   # 0-MW 'operating' units (seen in EU) carry no weight
    g["fossil"] = g["type"].isin(_FOSSIL)
    g.to_parquet(cache)
    return assign_state_of_record(g) if assign else g


# --------------------------------------------------------------------------- #
# State of record: GADM 4.1 from coordinates (decision 2026-09-27)
# --------------------------------------------------------------------------- #
# GEM / Ember spelling -> acceptable GADM 4.1 NAME_1 values (GADM predates the 2019/2020 UT
# reorganisations: Ladakh sits inside Jammu and Kashmir; DNH and Daman & Diu are separate)
GADM_ALIASES = {
    "Delhi": {"NCT of Delhi"},
    "National Capital Territory of Delhi": {"NCT of Delhi"},
    "Andaman and Nicobar Islands": {"Andaman and Nicobar"},
    "Dadra and Nagar Haveli and Daman and Diu": {"Dadra and Nagar Haveli", "Daman and Diu"},
    "Ladakh": {"Jammu and Kashmir"},
    "Orissa": {"Odisha"},
    "Pondicherry": {"Puducherry"},
    "Uttaranchal": {"Uttarakhand"},
}
US_GADM_ALIASES = {"Arizona and Nevada": {"Arizona", "Nevada"}}
# GADM / GEM spelling -> Ember zone name (the pipeline's zone key)
_TO_EMBER = {"District of Columbia": "Washington, D.C.", "Czech Republic": "Czechia","NCT of Delhi": "Delhi", "National Capital Territory of Delhi": "Delhi",
             "Andaman and Nicobar Islands": "Andaman and Nicobar",
             "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
             "Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
             "Orissa": "Odisha", "Pondicherry": "Puducherry", "Uttaranchal": "Uttarakhand"}


def assign_state_of_record(plants: pd.DataFrame, region: str = "India") -> pd.DataFrame:
    """Add state (the pipeline's), gadm_state, state_source, state_note.
    Rules, in order:
      1. grid.gem_state_overrides[plant_id]           -> that state            (override)
      2. grid.gem_state_ambiguous[plant_id]           -> NaN, both kept in note (ambiguous;
                                                          excluded from every single-state view)
      3. point outside every GADM polygon             -> GEM's label           (gem_label_outside_gadm)
      4. GEM label consistent with GADM (alias table) -> GEM's label           (gadm_agrees; keeps
                                                          finer post-2019 names such as Ladakh)
      5. otherwise                                    -> GADM state            (gadm)
    Names are normalised to Ember's spelling."""
    import geopandas as gpd
    root = _repo_root()
    gcfg = _cfg_params()["grid"]
    rc = _cfg_params()["region_config"][region]
    p = plants.copy()
    if rc["tier"] == "country":                      # EU: the zone IS the plant's country
        p["gadm_state"] = None
        p["state"] = p["country"].map(lambda c: _TO_EMBER.get(c, c))
        p["state_source"], p["state_note"] = "country", ""
        return p
    overrides = {str(k): v for k, v in (gcfg.get("gem_state_overrides") or {}).items()}
    ambiguous = {str(k): v for k, v in (gcfg.get("gem_state_ambiguous") or {}).items()}
    aliases = {**GADM_ALIASES, **US_GADM_ALIASES}
    adm1 = gpd.read_file(root / rc["gadm_path"], layer="ADM_ADM_1")[["NAME_1", "geometry"]]

    pts = gpd.GeoDataFrame(p[["plant_id"]], crs="EPSG:4326",
                           geometry=gpd.points_from_xy(p["longitude"], p["latitude"]))
    j = gpd.sjoin(pts, adm1, how="left", predicate="within")
    p["gadm_state"] = j[~j.index.duplicated(keep="first")]["NAME_1"].reindex(p.index).values

    def ember(s):
        return _TO_EMBER.get(s, s) if isinstance(s, str) else s

    state, source, note = [], [], []
    for pid, gem_l, gadm_s in zip(p["plant_id"].astype(str), p["gem_state"], p["gadm_state"]):
        agrees = isinstance(gadm_s, str) and gadm_s in aliases.get(gem_l, {gem_l})
        if pid in overrides:
            state.append(ember(overrides[pid]["state"])); source.append("override")
            note.append(overrides[pid].get("note", ""))
        elif pid in ambiguous:
            state.append(None); source.append("ambiguous"); note.append(ambiguous[pid])
        elif not isinstance(gadm_s, str):
            state.append(ember(gem_l)); source.append("gem_label_outside_gadm"); note.append("")
        elif agrees and gem_l in US_GADM_ALIASES:      # compound label (e.g. Hoover Dam): coordinates pick the state
            state.append(ember(gadm_s)); source.append("gadm_agrees"); note.append(f"GEM label: {gem_l}")
        elif agrees:
            state.append(ember(gem_l)); source.append("gadm_agrees"); note.append("")
        else:
            state.append(ember(gadm_s)); source.append("gadm"); note.append(f"GEM label: {gem_l}")
    p["state"], p["state_source"], p["state_note"] = state, source, note
    return p


def state_fossil_share(plants: pd.DataFrame) -> pd.DataFrame:
    """[state, fossil_share, capacity_mw] — operating fossil capacity / total capacity."""
    s = plants.groupby("state").apply(
        lambda d: pd.Series({"fossil_share": d.loc[d["fossil"], "capacity_mw"].sum() / d["capacity_mw"].sum(),
                             "capacity_mw": d["capacity_mw"].sum()}), include_groups=False)
    return s.reset_index()


if __name__ == "__main__":
    p = load_plants()
    print(f"India operating plants: {len(p)}  ({p['capacity_mw'].sum()/1000:,.0f} GW)")
    print("state of record:", p["state_source"].value_counts().to_dict())
    print(p.groupby("type")["capacity_mw"].sum().div(1000).round(1).to_string())
    print(state_fossil_share(p).sort_values("capacity_mw", ascending=False).head(10).round(2).to_string(index=False))
