"""GADM state verification of GEM power-plant state labels (REPORT ONLY — corrects nothing).

For every operating GEM plant in India (the set the pipeline uses), point-in-polygon its
lat/lon against GADM 4.1 ADM_ADM_1 (NAME_1) and compare with GEM's own
"Subnational unit (state, province)". Writes

    dcfootprint/results/gem_gadm_state_diff.csv       one row per plant
    dcfootprint/results/gem_gadm_state_mismatches.csv mismatches only

Label spelling differences are NOT counted as mismatches; they go through the explicit
alias table below (GADM 4.1 predates the 2019/2020 UT reorganisations: Ladakh is inside
Jammu and Kashmir, and Dadra & Nagar Haveli / Daman & Diu are still separate). For each
mismatch the distance from the point to GEM's labelled state polygon is given, so a
border-precision slip (a few km) can be told apart from a gross label or coordinate error.

Run: PYTHONPATH=dcfootprint/src python dcfootprint/experiments/gadm_gem_state_check.py
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import yaml

from dcfootprint.io.facilities import _repo_root

from dcfootprint.io.gem import GADM_ALIASES as ALIASES   # one alias table, shared with the pipeline
BORDER_KM = 5.0


def load_gem_india_operating() -> pd.DataFrame:
    root = _repo_root()
    p = yaml.safe_load((root / "dcfootprint" / "config" / "parameters.yaml").read_text(encoding="utf-8"))
    cols = {"GEM unit/phase ID": "plant_id", "GEM location ID": "location_id", "Plant / Project name": "plant",
            "Type": "type", "Capacity (MW)": "capacity_mw", "Status": "status", "Country/area": "country",
            "Latitude": "lat", "Longitude": "lon", "Location accuracy": "location_accuracy",
            "Subnational unit (state, province)": "gem_state"}
    g = pd.read_excel(root / p["grid"]["gem_path"], sheet_name="Power facilities", usecols=list(cols)).rename(columns=cols)
    g = g[(g["country"] == "India") & (g["status"] == "operating")].drop(columns=["country", "status"])
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g["capacity_mw"] = pd.to_numeric(g["capacity_mw"], errors="coerce")
    return g.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def main() -> pd.DataFrame:
    root = _repo_root()
    adm1 = gpd.read_file(root / "data" / "gadm" / "gadm41_IND.gpkg", layer="ADM_ADM_1")[["NAME_1", "geometry"]]
    plants = load_gem_india_operating()
    pts = gpd.GeoDataFrame(plants, crs="EPSG:4326", geometry=gpd.points_from_xy(plants["lon"], plants["lat"]))

    j = gpd.sjoin(pts, adm1, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    plants["gadm_state"] = j["NAME_1"].reindex(plants.index).values

    unknown = sorted(set(plants["gem_state"].dropna()) - set(adm1["NAME_1"]) - set(ALIASES))
    if unknown:
        print("WARNING: GEM state names with no GADM match or alias:", unknown)

    ok_names = plants["gem_state"].map(lambda s: ALIASES.get(s, {s}))
    plants["match"] = [pd.notna(g) and g in ok for g, ok in zip(plants["gadm_state"], ok_names)]
    plants["issue"] = "ok"
    plants.loc[plants["gem_state"].isna(), "issue"] = "no_gem_state"
    plants.loc[plants["gadm_state"].isna(), "issue"] = "outside_gadm_india"
    plants.loc[~plants["match"] & plants["gem_state"].notna() & plants["gadm_state"].notna(), "issue"] = "state_mismatch"

    # distance (km) from each mismatched point to the polygon(s) of the state GEM claims
    eq = adm1.to_crs("EPSG:7755")                                     # India, metres
    pts_m = pts.to_crs("EPSG:7755")
    dist = []
    for i, r in plants.iterrows():
        if r["issue"] != "state_mismatch":
            dist.append(None); continue
        shapes = eq[eq["NAME_1"].isin(ALIASES.get(r["gem_state"], {r["gem_state"]}))]
        dist.append(round(float(shapes.distance(pts_m.geometry.iloc[i]).min()) / 1000.0, 1) if len(shapes) else None)
    plants["km_to_gem_state"] = dist
    plants.loc[(plants["issue"] == "state_mismatch") & (plants["km_to_gem_state"] <= BORDER_KM), "issue"] = "state_mismatch_near_border"

    out = plants[["plant_id", "location_id", "plant", "type", "capacity_mw", "lat", "lon", "location_accuracy",
                  "gem_state", "gadm_state", "match", "issue", "km_to_gem_state"]]
    res = root / "dcfootprint" / "results"
    out.to_csv(res / "gem_gadm_state_diff.csv", index=False)
    mm = out[out["issue"] != "ok"].sort_values(["issue", "km_to_gem_state"], ascending=[True, False])
    mm.to_csv(res / "gem_gadm_state_mismatches.csv", index=False)

    n, bad = len(out), int((out["issue"] != "ok").sum())
    print(f"{bad} of {n} operating GEM plants disagree with GADM ({bad / n:.1%}); "
          f"{out.loc[out['issue'] != 'ok', 'capacity_mw'].sum():,.0f} of {out['capacity_mw'].sum():,.0f} MW")
    print(out["issue"].value_counts().to_string())
    return out


if __name__ == "__main__":
    main()
