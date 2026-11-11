"""L1 - how much does city-centroid geocoding move the scarcity result?

India facilities sit at ATLAS city centroids (no street addresses in the master; researched site
coordinates can be supplied in datasets/india_dc_facilities/geocodes.csv). A facility may really
be anywhere in its metro area, so its true AWARE basin may differ from the centroid's. This bounds
that error: for every centroid-geocoded facility, take every AWARE basin within `radius_km` of the
point and recompute its scope-1 scarcity-weighted water with the lowest- and the highest-scarcity
basin (annual sum, so the monthly CF profile is kept). Scope-2 is untouched (it is weighted at the
generation basins, not at the facility).
"""
from __future__ import annotations

import pandas as pd


def basins_within(points: pd.DataFrame, basins, radius_km: float) -> pd.DataFrame:
    """[facility_id, basin_id] for every AWARE basin intersecting a radius_km disc around each point."""
    import geopandas as gpd
    pts = gpd.GeoDataFrame(points[["facility_id"]].copy(), crs="EPSG:4326",
                           geometry=gpd.points_from_xy(points["longitude"], points["latitude"]))
    utm = pts.estimate_utm_crs()                                   # metric buffer
    disc = pts.to_crs(utm).buffer(radius_km * 1000.0).to_crs("EPSG:4326")
    j = gpd.sjoin(gpd.GeoDataFrame(pts[["facility_id"]], geometry=disc, crs="EPSG:4326"),
                  basins[["basin_id", "geometry"]], how="inner", predicate="intersects")
    return pd.DataFrame(j[["facility_id", "basin_id"]]).drop_duplicates()


def centroid_bounds(account: pd.DataFrame, facilities: pd.DataFrame, basins, cf: pd.DataFrame,
                    radius_km: float = 25.0) -> tuple[pd.DataFrame, dict]:
    """(per-facility table, summary). Only facilities whose geocode_method is a city centroid are
    varied; others keep their basin in both bounds."""
    fac = facilities[facilities["facility_id"].isin(account["facility_id"])].drop_duplicates("facility_id")
    cen = fac[fac["geocode_method"].astype(str).str.contains("centroid")]
    cand = basins_within(cen, basins, radius_km)
    onsite = account.groupby(["facility_id", "month"])["water_onsite_l"].sum().rename("w").reset_index()
    c = cand.merge(onsite, on="facility_id").merge(cf, on=["basin_id", "month"], how="left")
    # a basin without an AWARE CF in some month (ocean, undefined) is not a scorable location: the
    # account drops such facility-months, so it is excluded here rather than scored as CF = 0
    no_cf = c.loc[c["cf"].isna(), ["facility_id", "basin_id"]].drop_duplicates()
    c = c.merge(no_cf.assign(_drop=True), on=["facility_id", "basin_id"], how="left")
    c = c[c["_drop"].isna()].drop(columns="_drop")
    c["s"] = c["w"] * c["cf"]
    per_basin = c.groupby(["facility_id", "basin_id"], as_index=False).agg(s_l_eq=("s", "sum"),
                                                                          cf_mean=("cf", "mean"))
    assigned = account.groupby("facility_id").agg(basin_id=("basin_id", "first"),
                                                  s_assigned=("water_scarcity_onsite_l_eq", "sum"),
                                                  city=("city", "first"))
    rng = per_basin.groupby("facility_id").agg(n_basins=("basin_id", "nunique"), s_min=("s_l_eq", "min"),
                                               s_max=("s_l_eq", "max"), cf_min=("cf_mean", "min"),
                                               cf_max=("cf_mean", "max"))
    t = assigned.join(rng, how="left")
    t["n_basins"] = t["n_basins"].fillna(1).astype(int)
    t["s_min"] = t["s_min"].fillna(t["s_assigned"])
    t["s_max"] = t["s_max"].fillna(t["s_assigned"])
    t = t.reset_index()
    tot = account["water_scarcity_l_eq"].sum()
    s1 = t["s_assigned"].sum()
    by_city = t.groupby("city").agg(n_facilities=("facility_id", "size"), n_basins=("n_basins", "max"),
                                    cf_min=("cf_min", "min"), cf_max=("cf_max", "max")).reset_index()
    summary = {
        "radius_km": radius_km, "n_centroid_facilities": len(cen),
        "n_facilities_with_alternative_basin": int((t["n_basins"] > 1).sum()),
        "scope1_scarcity_m3eq": round(float(s1) / 1000.0),
        "scope1_scarcity_range_m3eq": [round(float(t["s_min"].sum()) / 1000.0), round(float(t["s_max"].sum()) / 1000.0)],
        "total_scarcity_m3eq": round(float(tot) / 1000.0),
        "total_scarcity_range_m3eq": [round(float(tot - s1 + t["s_min"].sum()) / 1000.0),
                                      round(float(tot - s1 + t["s_max"].sum()) / 1000.0)],
        "total_scarcity_range_pct": [round(float((t["s_min"].sum() - s1) / tot * 100), 1),
                                     round(float((t["s_max"].sum() - s1) / tot * 100), 1)],
        "by_city": by_city.round(2).to_dict(orient="records"),
    }
    return t, summary

