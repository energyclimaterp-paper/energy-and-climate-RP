"""L0/L1 - US grid carbon and power-plant water from plant-level public data (eGRID + EIA).

Replaces, for the US, the Ember state generation mix x Macknick 2012 fuel coefficients with the
plant-level sources the US literature uses (Siddik et al. 2021; Talukder et al. 2026):

  zone      eGRID subregion (EPA's location-based grid unit for US scope-2 accounting).
            Plants: eGRID's own SUBRGN; plants new since that eGRID year: point-in-polygon on the
            eGRID subregion map. Datacenters: point-in-polygon on the same map.
  carbon    plant-month CO2 = EIA-923 monthly net generation x the plant's eGRID annual CO2 output
            rate (biomass- and CHP-adjusted, as eGRID publishes it); plants not in eGRID take the
            eGRID subregion x fuel-category mean rate. Subregion CI = sum CO2 / sum generation.
            eGRID itself is annual; the EIA-923 monthly generation gives the seasonal shape.
            Check: on the eGRID year this reproduces eGRID's subregion rates (egrid_reproduction).
  water     thermoelectric plants: REPORTED monthly cooling-water consumption (EIA thermoelectric
            cooling water data, from EIA-923 schedule 8D / EIA-860 cooling systems); reporting gaps
            filled with the reported intensity (gal/MWh) of the same technology and cooling type.
            Other plants: Macknick et al. 2012 coefficient of the fuel (water_grid.ewif_coeff_L_per_MWh)
            for steam / combined-cycle prime movers; zero for simple-cycle turbines, engines, wind,
            batteries and pumped storage; hydro reservoir evaporation kept separate (hydro column).
  scarcity  each plant's water x the AWARE 2.0 monthly CF of the basin the plant sits in.

All tables are cached as parquet next to the EIA files (derived data, rebuilt when inputs change).
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import data_path
from dcfootprint.settings import params as _cfg_params

LB_PER_MWH_TO_G_PER_KWH = 0.45359237           # 1 lb/MWh = 0.4536 kg/MWh = 0.4536 g/kWh
GAL_TO_L = 3.785411784
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
THERMAL_PM = {"ST", "CA", "CT", "CS"}           # steam turbine and combined-cycle parts: need cooling
# EIA-923 MER fuel code -> the Ember fuel keys of water_grid.ewif_coeff_L_per_MWh
MER_TO_FUEL = {"COL": "Coal", "NG": "Gas", "NUC": "Nuclear", "HYC": "Hydro", "HPS": "Hydro",
               "SUN": "Solar", "WND": "Wind", "GEO": "Other Renewables", "WWW": "Bioenergy",
               "WOO": "Bioenergy", "MLG": "Bioenergy", "ORW": "Bioenergy", "DFO": "Other Fossil",
               "RFO": "Other Fossil", "PC": "Other Fossil", "OOG": "Other Fossil", "OTH": "Other Fossil",
               "WOC": "Coal", "WAS": "Other Fossil"}
# eGRID plant fuel category -> the same fuel keys (for the mean-rate fill of plants not in eGRID)
EGRID_FUELCT_TO_FUEL = {"COAL": "Coal", "GAS": "Gas", "OIL": "Other Fossil", "OFSL": "Other Fossil",
                        "OTHF": "Other Fossil", "NUCLEAR": "Nuclear", "HYDRO": "Hydro", "SOLAR": "Solar",
                        "WIND": "Wind", "BIOMASS": "Bioenergy", "GEOTHERMAL": "Other Renewables"}


def _cfg() -> dict:
    return _cfg_params()["region_config"]["US"]


def _resolve(pattern_key: str, year: int) -> Path:
    pat = str(_repo_root() / _cfg()[pattern_key].format(year=year))
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"raw input not on disk: {pat} (download as in data/README.md)")
    return Path(hits[-1])


def _cached(path: Path, name: str, build) -> pd.DataFrame:
    cache = path.with_name(name)
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache)
    out = build()
    out.to_parquet(cache)
    return out


# --------------------------------------------------------------------------- inputs
def egrid_plants() -> pd.DataFrame:
    """[plant_id, subregion, latitude, longitude, fuel, egrid_gen_mwh, co2_rate_g_per_kwh]."""
    path = data_path(_cfg()["egrid_path"])

    def build():
        p = pd.read_excel(path, sheet_name="PLNT" + _egrid_tag(), header=1)
        out = pd.DataFrame({
            "plant_id": p["ORISPL"].astype("int64"), "subregion": p["SUBRGN"].astype(str),
            "latitude": pd.to_numeric(p["LAT"], errors="coerce"),
            "longitude": pd.to_numeric(p["LON"], errors="coerce"),
            "fuel": p["PLFUELCT"].map(EGRID_FUELCT_TO_FUEL),
            "egrid_gen_mwh": pd.to_numeric(p["PLNGENAN"], errors="coerce"),
            "co2_rate_g_per_kwh": pd.to_numeric(p["PLCO2RTA"], errors="coerce") * LB_PER_MWH_TO_G_PER_KWH})
        # eGRID leaves the rate blank for plants with no CO2 (wind, solar, nuclear, hydro): that is 0, not unknown
        co2 = pd.to_numeric(p["PLCO2AN"], errors="coerce").fillna(0.0).to_numpy()
        out.loc[out["co2_rate_g_per_kwh"].isna() & (co2 <= 0), "co2_rate_g_per_kwh"] = 0.0
        return out.drop_duplicates("plant_id")
    return _cached(path, f"egrid_plants_{_egrid_tag()}.parquet", build)


def egrid_subregion_rates() -> pd.DataFrame:
    """[subregion, egrid_gen_mwh, egrid_co2_rate_g_per_kwh] as eGRID publishes them (SRL sheet)."""
    path = data_path(_cfg()["egrid_path"])

    def build():
        s = pd.read_excel(path, sheet_name="SRL" + _egrid_tag(), header=1)
        return pd.DataFrame({"subregion": s["SUBRGN"].astype(str),
                             "egrid_gen_mwh": pd.to_numeric(s["SRNGENAN"], errors="coerce"),
                             "egrid_co2_rate_g_per_kwh": pd.to_numeric(s["SRCO2RTA"], errors="coerce")
                             * LB_PER_MWH_TO_G_PER_KWH})
    return _cached(path, f"egrid_subregions_{_egrid_tag()}.parquet", build)


def _egrid_tag() -> str:
    return str(int(_cfg()["egrid_year"]))[-2:]


def eia923_generation(year: int) -> pd.DataFrame:
    """[plant_id, month, prime_mover, fuel, netgen_mwh] from EIA-923 page 1 (generation and fuel),
    one row per plant x prime mover x fuel x month; state-level increment rows (plant 99999) dropped."""
    path = _resolve("eia923_path", year)

    def build():
        g = pd.read_excel(path, sheet_name="Page 1 Generation and Fuel Data", header=5)
        g.columns = [str(c).replace("\n", " ").strip() for c in g.columns]
        g = g[g["Plant Id"] != 99999]
        long = g.melt(id_vars=["Plant Id", "Reported Prime Mover", "MER Fuel Type Code"],
                      value_vars=[f"Netgen {m}" for m in MONTHS], var_name="m", value_name="netgen_mwh")
        long["month"] = long["m"].str.replace("Netgen ", "").map({m: i + 1 for i, m in enumerate(MONTHS)})
        long["netgen_mwh"] = pd.to_numeric(long["netgen_mwh"], errors="coerce").fillna(0.0)
        out = long.rename(columns={"Plant Id": "plant_id", "Reported Prime Mover": "prime_mover",
                                   "MER Fuel Type Code": "mer_fuel"})
        out["plant_id"] = out["plant_id"].astype("int64")
        out["fuel"] = out["mer_fuel"].map(MER_TO_FUEL).fillna("Other Fossil")
        return out[["plant_id", "month", "prime_mover", "mer_fuel", "fuel", "netgen_mwh"]]
    return _cached(path, f"eia923_generation_{year}.parquet", build)


def eia860_plants() -> pd.DataFrame:
    """[plant_id, latitude, longitude] from EIA-860 schedule 2 (locations of plants new since eGRID)."""
    path = data_path(_cfg()["eia860_plant_path"])

    def build():
        p = pd.read_excel(path, header=1)
        return pd.DataFrame({"plant_id": pd.to_numeric(p["Plant Code"], errors="coerce"),
                             "latitude": pd.to_numeric(p["Latitude"], errors="coerce"),
                             "longitude": pd.to_numeric(p["Longitude"], errors="coerce")}
                            ).dropna(subset=["plant_id"]).astype({"plant_id": "int64"}).drop_duplicates("plant_id")
    return _cached(path, "eia860_plants.parquet", build)


def cooling_water(year: int) -> pd.DataFrame:
    """[plant_id, month, water_l, water_reported_l, thermal_gen_mwh] plant-month cooling-water
    consumption. Rows without a reported volume take the reported intensity (consumption / net
    generation) of the same technology x cooling type that year (else the technology's)."""
    path = _resolve("eia_cooling_path", year)

    def build():
        c = pd.read_excel(path, header=2)
        c.columns = [str(x).strip() for x in c.columns]
        num = lambda s: pd.to_numeric(s, errors="coerce")
        gen = sum(num(c[k]).fillna(0.0) for k in [
            "Net Generation from Steam Turbines (MWh)",
            "Net Generation Associated with Single Shaft Combined Cycle Units (MWh)",
            "Net Generation Associated with Combined Cycle Gas Turbines (MWh)"]).clip(lower=0)
        d = pd.DataFrame({"plant_id": num(c["Plant Code"]).astype("int64"), "month": num(c["Month"]).astype("int64"),
                          "tech": c["Generator Primary Technology"].astype(str),
                          "cooling": c["860 Cooling Type 1"].astype(str), "gen": gen,
                          "wc_l": num(c["Water Consumption Volume (Million Gallons)"]) * 1e6 * GAL_TO_L})
        rep = d[d["wc_l"].notna() & (d["gen"] > 0)]
        s_tc = rep.groupby(["tech", "cooling"])[["wc_l", "gen"]].sum()
        s_t = rep.groupby("tech")[["wc_l", "gen"]].sum()
        k_tc = (s_tc["wc_l"] / s_tc["gen"]).rename("k_tc")
        k_t = (s_t["wc_l"] / s_t["gen"]).rename("k_t")
        d = d.join(k_tc, on=["tech", "cooling"]).join(k_t, on="tech")
        d["k_fill"] = d["k_tc"].fillna(d["k_t"]).fillna(0.0)
        d["water_l"] = d["wc_l"].fillna(d["gen"] * d["k_fill"])
        return (d.groupby(["plant_id", "month"], as_index=False)
                .agg(water_l=("water_l", "sum"), water_reported_l=("wc_l", "sum"), thermal_gen_mwh=("gen", "sum")))
    return _cached(path, f"eia_cooling_water_{year}.parquet", build)


@lru_cache(maxsize=1)
def subregion_shapes():
    import geopandas as gpd
    g = gpd.read_file(data_path(_cfg()["egrid_subregions_path"])).to_crs("EPSG:4326")
    col = next(c for c in g.columns if c.upper() in ("SUBRGN", "ZIPSUBREGI", "SUBREGION", "NAME"))
    return g[[col, "geometry"]].rename(columns={col: "subregion"})


def assign_subregion(df: pd.DataFrame, lat: str = "latitude", lon: str = "longitude") -> pd.Series:
    """Point-in-polygon on the eGRID subregion map; NaN where a point falls in no subregion."""
    import geopandas as gpd
    ok = df[lat].notna() & df[lon].notna()
    pts = gpd.GeoDataFrame(df.loc[ok, []], geometry=gpd.points_from_xy(df.loc[ok, lon], df.loc[ok, lat]), crs="EPSG:4326")
    j = gpd.sjoin(pts, subregion_shapes(), how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j["subregion"].reindex(df.index)


# --------------------------------------------------------------------------- plant-month table
def plant_month(year: int) -> pd.DataFrame:
    """Plant x month: [plant_id, month, subregion, latitude, longitude, gen_mwh, co2_t, water_l,
    water_hydro_l, water_source]. Generation clipped at 0 (pumped storage, station service)."""
    coeff = _cfg_params()["water_grid"]["ewif_coeff_L_per_MWh"]
    g = eia923_generation(year).copy()
    g["gen"] = g["netgen_mwh"].clip(lower=0)
    eg = egrid_plants()
    loc = eg[["plant_id", "subregion", "latitude", "longitude"]]
    new = eia860_plants()
    new = new[~new["plant_id"].isin(loc["plant_id"])].copy()
    new["subregion"] = assign_subregion(new)
    loc = pd.concat([loc, new[["plant_id", "subregion", "latitude", "longitude"]]], ignore_index=True)

    cw = cooling_water(year)
    in_cooling = set(cw["plant_id"])
    pm = g["prime_mover"].astype(str)
    thermal = pm.isin(THERMAL_PM) & ~g["fuel"].isin(["Solar", "Wind", "Hydro"])
    covered = thermal & g["plant_id"].isin(in_cooling)          # the cooling file reports this water
    hydro = pm.eq("HY")
    k = g["fuel"].map(coeff).fillna(0.0)
    g["water_macknick_l"] = np.where(thermal & ~covered, g["gen"] * k, 0.0)
    g.loc[pm.eq("PV") | (g["fuel"].eq("Solar") & ~hydro), "water_macknick_l"] = g["gen"] * coeff["Solar"]
    g.loc[g["fuel"].eq("Other Renewables"), "water_macknick_l"] = g["gen"] * coeff["Other Renewables"]
    g["water_hydro_l"] = np.where(hydro, g["gen"] * coeff["Hydro"], 0.0)

    # CO2: eGRID plant rate; plants without one take the subregion x fuel mean rate of eGRID plants
    rate = eg.set_index("plant_id")["co2_rate_g_per_kwh"]
    g["rate"] = g["plant_id"].map(rate)
    pmo = (g.groupby(["plant_id", "month"], as_index=False)
           .agg(gen_mwh=("gen", "sum"), rate=("rate", "first"), water_macknick_l=("water_macknick_l", "sum"),
                water_hydro_l=("water_hydro_l", "sum")))
    dominant = (g.sort_values("gen").drop_duplicates(["plant_id", "month"], keep="last")
                [["plant_id", "month", "fuel"]])                          # the plant-month's largest fuel
    pmo = pmo.merge(dominant, on=["plant_id", "month"], how="left")
    pmo = pmo.merge(loc, on="plant_id", how="left").merge(cw[["plant_id", "month", "water_l", "water_reported_l"]],
                                                          on=["plant_id", "month"], how="left")
    e = eg.dropna(subset=["co2_rate_g_per_kwh"]).assign(w=lambda d: d["egrid_gen_mwh"].clip(lower=0))
    e = e.assign(rw=e["co2_rate_g_per_kwh"] * e["w"]).groupby(["subregion", "fuel"])[["rw", "w"]].sum()
    fill = (e["rw"] / e["w"]).where(e["w"] > 0).rename("rate_fill")          # generation-weighted mean rate
    pmo = pmo.join(fill, on=["subregion", "fuel"])
    pmo["rate_source"] = np.where(pmo["rate"].notna(), "egrid_plant", "egrid_subregion_fuel_mean")
    pmo["rate"] = pmo["rate"].fillna(pmo["rate_fill"]).fillna(0.0)
    pmo["co2_t"] = pmo["gen_mwh"] * pmo["rate"] / 1000.0                       # MWh x g/kWh = kg -> /1000 t
    pmo["water_cooling_l"] = pmo["water_l"].fillna(0.0)
    pmo["water_l"] = pmo["water_cooling_l"] + pmo["water_macknick_l"]
    pmo["water_source"] = np.where(pmo["water_reported_l"].fillna(0) > 0, "eia_reported",
                                   np.where(pmo["water_cooling_l"] > 0, "eia_cooling_imputed", "macknick_or_none"))
    return pmo[["plant_id", "month", "subregion", "latitude", "longitude", "fuel", "gen_mwh", "co2_t",
                "rate_source", "water_l", "water_cooling_l", "water_hydro_l", "water_source"]]


# --------------------------------------------------------------------------- zone-month tables
def subregion_month(year: int, basin_cf: pd.DataFrame | None = None, basins=None) -> pd.DataFrame:
    """[zone_id, month, ci_gco2_per_kwh, ci_source, gen_mwh, ewif_l_per_mwh, ewif_hydro_l_per_mwh,
    sewif_l_eq_per_mwh, sewif_hydro_l_eq_per_mwh] per eGRID subregion. Scope-2 scarcity weights
    each plant's water by its own basin's monthly AWARE CF (basin_cf + basins required for it)."""
    p = plant_month(year)
    p = p[p["subregion"].notna() & (p["subregion"] != "nan")]
    if basin_cf is not None:
        from dcfootprint.geo import join as gj
        loc = p.drop_duplicates("plant_id")[["plant_id", "latitude", "longitude"]].rename(columns={"plant_id": "facility_id"})
        b = gj.assign_basin(loc, basins)[["facility_id", "basin_id"]].rename(columns={"facility_id": "plant_id"})
        p = p.merge(b, on="plant_id", how="left")
        cf = basin_cf.assign(basin_id=basin_cf["basin_id"].astype("int64"))
        p = p.merge(cf, on=["basin_id", "month"], how="left")
        # plants outside every basin (offshore, missing coords) take the subregion-month mean CF of located water
        w = p["water_l"] + p["water_hydro_l"]
        agg = (p.assign(wcf=(w * p["cf"]).fillna(0.0), ww=w.where(p["cf"].notna(), 0.0))
               .groupby(["subregion", "month"])[["wcf", "ww"]].sum())
        mean_cf = (agg["wcf"] / agg["ww"]).where(agg["ww"] > 0).rename("cf_mean")
        p = p.join(mean_cf, on=["subregion", "month"])
        p["cf"] = p["cf"].fillna(p["cf_mean"]).fillna(0.0)
    else:
        p["cf"] = np.nan
    p["s_l"] = p["water_l"] * p["cf"]
    p["s_hydro_l"] = p["water_hydro_l"] * p["cf"]
    z = p.groupby(["subregion", "month"], as_index=False).agg(
        gen_mwh=("gen_mwh", "sum"), co2_t=("co2_t", "sum"), water_l=("water_l", "sum"),
        water_hydro_l=("water_hydro_l", "sum"), s_l=("s_l", "sum"), s_hydro_l=("s_hydro_l", "sum"))
    z = z[z["gen_mwh"] > 0]
    out = pd.DataFrame({
        "zone_id": z["subregion"], "month": z["month"].astype(int),
        "ci_gco2_per_kwh": z["co2_t"] / z["gen_mwh"] * 1000.0, "ci_source": "egrid_eia",
        "gen_mwh": z["gen_mwh"],
        "ewif_l_per_mwh": (z["water_l"] + z["water_hydro_l"]) / z["gen_mwh"],
        "ewif_hydro_l_per_mwh": z["water_hydro_l"] / z["gen_mwh"],
        "sewif_l_eq_per_mwh": (z["s_l"] + z["s_hydro_l"]) / z["gen_mwh"],
        "sewif_hydro_l_eq_per_mwh": z["s_hydro_l"] / z["gen_mwh"]})
    return out.reset_index(drop=True)


def egrid_reproduction() -> pd.DataFrame:
    """Method check on the eGRID year: annual subregion CO2 rate from EIA-923 generation x eGRID
    plant rates vs eGRID's published subregion rate. [subregion, ours, egrid, rel_diff]."""
    yr = int(_cfg()["egrid_year"])
    z = subregion_month(yr)
    ann = z.assign(co2=z["ci_gco2_per_kwh"] * z["gen_mwh"]).groupby("zone_id").agg(co2=("co2", "sum"), gen=("gen_mwh", "sum"))
    ours = (ann["co2"] / ann["gen"]).rename("ours_g_per_kwh")
    e = egrid_subregion_rates().set_index("subregion")["egrid_co2_rate_g_per_kwh"]
    d = pd.concat([ours, e.rename("egrid_g_per_kwh")], axis=1, join="inner").reset_index(names="subregion")
    d["rel_diff"] = d["ours_g_per_kwh"] / d["egrid_g_per_kwh"] - 1
    return d


def seasonal_naive_ci(year: int) -> pd.DataFrame:
    """[zone_id, month, ci_hat] - the controller's one-year-ahead forecast for the US eGRID basis:
    the same subregion-month CI computed for year - 1 (EIA-923 of that year)."""
    prev = subregion_month(year - 1)
    return prev[["zone_id", "month", "ci_gco2_per_kwh"]].rename(columns={"ci_gco2_per_kwh": "ci_hat"})
