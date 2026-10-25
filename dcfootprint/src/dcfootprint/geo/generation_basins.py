"""L1 — scope-2 scarcity at the GENERATION basins (fix for red-team R3).

Scope-2 water is evaporated at power plants, not at the datacenter, so it must be
scarcity-weighted at the basins where the zone's electricity is generated:

    CF_gen(z, fuel, m) = capacity-weighted mean AWARE CF(m) over operating GEM plants of
                         `fuel` in zone z   (fallback: the same over all of India)
    S_grid(f, m)       = E_grid(f, m) * sum_fuel share(z, fuel, m) * macknick[fuel] * CF_gen(z, fuel, m)

Capacity weighting is a proxy for generation weighting within a fuel (no plant-level
generation data). zone = state (grid.zone: state) or all of India (grid.zone: national).
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd


def plant_basins(plants: pd.DataFrame, basins: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-polygon: each GEM plant -> AWARE basin_id (plants in no basin dropped)."""
    pts = gpd.GeoDataFrame(plants.copy(), crs="EPSG:4326",
                           geometry=gpd.points_from_xy(plants["longitude"], plants["latitude"]))
    j = gpd.sjoin(pts, basins[["basin_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


def fuel_basin_cf(plants_b: pd.DataFrame, cf: pd.DataFrame, type_to_fuel: dict,
                  zone: str = "state") -> pd.DataFrame:
    """[zone_id, fuel, month, cf_gen, cf_gen_source] capacity-weighted generation-basin CF."""
    p = plants_b.copy()
    p["fuel"] = p["type"].map(type_to_fuel)
    p = p.dropna(subset=["fuel"])
    p = p.merge(cf, on="basin_id", how="inner")                 # plant x month
    p["w"] = p["capacity_mw"] * p["cf"]

    def _wmean(df, keys):
        g = df.groupby(keys, as_index=False).agg(w=("w", "sum"), cap=("capacity_mw", "sum"))
        g["cf_gen"] = g["w"] / g["cap"]
        return g.drop(columns=["w", "cap"])

    nat = _wmean(p, ["fuel", "month"]).rename(columns={"cf_gen": "cf_nat"})
    if zone == "national":
        nat["cf_gen_source"] = "gem_national"
        return nat.rename(columns={"cf_nat": "cf_gen"})
    st = _wmean(p, ["state", "fuel", "month"]).rename(columns={"state": "zone_id"})
    zones = sorted(p["state"].dropna().unique())
    full = (pd.MultiIndex.from_product([zones, nat["fuel"].unique(), range(1, 13)],
                                       names=["zone_id", "fuel", "month"]).to_frame(index=False)
            .merge(st, on=["zone_id", "fuel", "month"], how="left")
            .merge(nat, on=["fuel", "month"], how="left"))
    full["cf_gen_source"] = full["cf_gen"].notna().map({True: "gem_state", False: "gem_national_fill"})
    full["cf_gen"] = full["cf_gen"].fillna(full["cf_nat"])
    return full.drop(columns="cf_nat")


def zone_month_grid_water(shares: pd.DataFrame, fuel_cf: pd.DataFrame, ewif_coeff: dict,
                          zone: str = "state") -> pd.DataFrame:
    """[zone_id, month, ewif_l_per_mwh, ewif_hydro_l_per_mwh, sewif_l_eq_per_mwh,
    sewif_hydro_l_eq_per_mwh] — physical and scarcity-weighted scope-2 intensity, with the
    hydro part split out for the hydro-coefficient sensitivity."""
    s = shares.copy()
    s["coeff"] = s["fuel"].map(ewif_coeff)
    if s["coeff"].isna().any():
        raise KeyError(f"Fuels with no EWIF coefficient: {sorted(s.loc[s['coeff'].isna(), 'fuel'].unique())}")
    if zone == "national":
        s = s.merge(fuel_cf[["fuel", "month", "cf_gen"]], on=["fuel", "month"], how="left")
    else:
        s = s.merge(fuel_cf[["zone_id", "fuel", "month", "cf_gen"]], on=["zone_id", "fuel", "month"], how="left")
        # a zone with no GEM plants at all (not in fuel_cf) -> national fuel CF
        nat = fuel_cf.groupby(["fuel", "month"], as_index=False)["cf_gen"].mean().rename(columns={"cf_gen": "cf_n"})
        s = s.merge(nat, on=["fuel", "month"], how="left")
        s["cf_gen"] = s["cf_gen"].fillna(s["cf_n"])
    s["cf_gen"] = s["cf_gen"].fillna(0.0)          # fuel with no plant anywhere (e.g. zero-coeff wind)
    s["w"] = s["share"] * s["coeff"]
    s["sw"] = s["w"] * s["cf_gen"]
    hyd = s["fuel"] == "Hydro"
    out = s.groupby(["zone_id", "month"], as_index=False).agg(ewif_l_per_mwh=("w", "sum"), sewif_l_eq_per_mwh=("sw", "sum"))
    h = (s[hyd].groupby(["zone_id", "month"], as_index=False)
         .agg(ewif_hydro_l_per_mwh=("w", "sum"), sewif_hydro_l_eq_per_mwh=("sw", "sum")))
    out = out.merge(h, on=["zone_id", "month"], how="left").fillna(
        {"ewif_hydro_l_per_mwh": 0.0, "sewif_hydro_l_eq_per_mwh": 0.0})
    return out
