"""L1 — spatial linkage: facility point -> AWARE basin (point-in-polygon) + the
basin's 12 monthly scarcity CFs. Also carries zone_id (state) for later state-level
joins. AWARE's Basin_ID is the GPKG feature id (read with pyogrio fid_as_index), so basin_id
joins directly to the AWARE intermediate-variable tables (io/aware.py).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from dcfootprint.io.facilities import _repo_root

AWARE_LAYER = "AWARE20_Native_CFs_geospatial"
_CF_MONTHS = ["CF_Jan", "CF_Feb", "CF_Mar", "CF_Apr", "CF_May", "CF_Jun",
              "CF_Jul", "CF_Aug", "CF_Sep", "CF_Oct", "CF_Nov", "CF_Dec"]


def load_aware_basins(path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Read the AWARE native-CF basins; keep monthly CFs + geometry. basin_id = the GPKG's
    native feature id, which IS AWARE's Basin_ID (geopandas.read_file drops it; pyogrio with
    fid_as_index keeps it — verified 9,406/9,406 against the xlsx in AWARE2.0_AUDIT.md)."""
    import pyogrio
    path = Path(path) if path else _repo_root() / "data" / "aware" / "AWARE20_Native_CFs_geospatial.gpkg"
    g = pyogrio.read_dataframe(path, layer=AWARE_LAYER, fid_as_index=True)
    g["basin_id"] = g.index.to_numpy().astype("int64")
    g = g.reset_index(drop=True)
    return g[["basin_id", *_CF_MONTHS, "geometry"]]


def basin_monthly_cf(basins: gpd.GeoDataFrame) -> pd.DataFrame:
    """Long CF table [basin_id, month(1-12), cf] from the wide CF_Jan..CF_Dec cols."""
    long = basins.drop(columns="geometry").melt(
        id_vars="basin_id", value_vars=_CF_MONTHS, var_name="cf_month", value_name="cf")
    long["month"] = long["cf_month"].map({m: i + 1 for i, m in enumerate(_CF_MONTHS)})
    long = long.dropna(subset=["cf"])              # ocean / undefined basins have no CF
    from dcfootprint.validation import schemas
    return schemas.BasinMonthlyCF.validate(
        long[["basin_id", "month", "cf"]].sort_values(["basin_id", "month"]).reset_index(drop=True))


def assign_basin(facilities: pd.DataFrame, basins: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-polygon: facility lat/lon -> basin_id. Facilities without coords, or
    whose point falls in no basin, keep basin_id = <NA>."""
    df = facilities.copy()
    geo = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    pts = gpd.GeoDataFrame(
        geo[["facility_id"]],
        geometry=gpd.points_from_xy(geo["longitude"], geo["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, basins[["basin_id", "geometry"]], how="left", predicate="within")
    # a point on a border can hit >1 polygon -> keep the first
    joined = joined.drop_duplicates("facility_id")[["facility_id", "basin_id"]]
    df = df.drop(columns=[c for c in ["basin_id"] if c in df.columns])
    df = df.merge(joined, on="facility_id", how="left")
    df["basin_id"] = df["basin_id"].astype("Int64")
    return df


def assign_zone_and_basin(facilities: pd.DataFrame) -> pd.DataFrame:
    """L1 output: facilities + zone_id (state) + basin_id (AWARE point-in-polygon)."""
    df = facilities.copy()
    df["zone_id"] = df["state"].astype("string")          # India state; carbon is national in v1
    basins = load_aware_basins()
    return assign_basin(df, basins)


def build(config: dict | None = None) -> pd.DataFrame:
    """Load the facility spine, attach zone_id + basin_id. Reads the L0 parquet if
    present, else builds it in-memory."""
    from dcfootprint.io import facilities as fac_io
    out = _repo_root() / "dcfootprint" / "outputs" / "interim" / "facilities.parquet"
    fac = pd.read_parquet(out) if out.exists() else fac_io.build(config)
    return assign_zone_and_basin(fac)


if __name__ == "__main__":
    basins = load_aware_basins()
    cf = basin_monthly_cf(basins)
    fac = build()
    matched = fac["basin_id"].notna()
    geoc = fac["latitude"].notna()
    print(f"AWARE basins: {len(basins)}  | monthly-CF rows: {len(cf)}  | basins with CF: {cf['basin_id'].nunique()}")
    print(f"facilities: {len(fac)}  geocoded: {geoc.sum()}  basin-matched: {matched.sum()} "
          f"({matched.sum()}/{geoc.sum()} of geocoded)")
    print("sample basin_ids:", fac.loc[matched, "basin_id"].head(8).tolist())
    print("CF range:", round(cf["cf"].min(), 2), "->", round(cf["cf"].max(), 2))

    outdir = _repo_root() / "dcfootprint" / "outputs" / "interim"
    outdir.mkdir(parents=True, exist_ok=True)
    fac.to_parquet(outdir / "facilities_geocoded.parquet")
    cf.to_parquet(outdir / "basin_cf_monthly.parquet")
    print(f"wrote {outdir/'facilities_geocoded.parquet'} and basin_cf_monthly.parquet")
