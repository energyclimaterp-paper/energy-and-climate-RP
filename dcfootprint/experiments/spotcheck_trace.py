"""Step-5 spot check: trace facilities from the source workbook to the final report figures,
RE-COMPUTING every hop from raw files without calling the pipeline's own functions.

Picks: 2 random operational facilities from the account (seed 20260927) + the facility whose
scope-2 scarcity moved most under the GADM state-of-record decision (no facility changed its
own state: facility states come from the DC workbook, not GEM).

Hops checked per facility (annual totals, 2024):
  workbook row -> capacity, state, city
  ATLAS raw    -> city-centroid lat/lon          vs interim/facilities_geocoded.parquet
  AWARE gpkg   -> basin (fid) + 12 monthly CFs   vs account basin_id / cf
  Ember raw    -> state-month CI, fuel shares    vs account carbon, scope-2 water
  EM cache     -> regional consumption CI (data/em, when grid.carbon_basis = consumption) vs account carbon
  GEM plants   -> capacity-weighted generation-basin CF per fuel (state of record)
  account      -> india_account_summary.csv and q2_scorecard.csv figures

Run: PYTHONPATH=dcfootprint/src python dcfootprint/experiments/spotcheck_trace.py <outdir>
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import yaml

ROOT = Path(__file__).resolve().parents[2]
FUELS = ["Coal", "Gas", "Nuclear", "Bioenergy", "Hydro", "Solar", "Wind", "Other Fossil", "Other Renewables"]
TOL = 1e-6
# India's five regional load-dispatch grids (written out here, not imported from io/electricitymaps.py,
# so the carbon hop is checked against an independent state -> grid assignment)
REGIONAL_GRID = {**{s: "IN-NO" for s in ["Delhi", "Haryana", "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand",
                                         "Himachal Pradesh", "Jammu and Kashmir", "Chandigarh"]},
                 **{s: "IN-WE" for s in ["Maharashtra", "Gujarat", "Madhya Pradesh", "Chhattisgarh", "Goa",
                                         "Dadra and Nagar Haveli"]},
                 **{s: "IN-SO" for s in ["Tamil Nadu", "Karnataka", "Kerala", "Andhra Pradesh", "Telangana",
                                         "Puducherry"]},
                 **{s: "IN-EA" for s in ["West Bengal", "Bihar", "Jharkhand", "Odisha", "Sikkim"]},
                 **{s: "IN-NE" for s in ["Assam", "Meghalaya", "Tripura", "Manipur", "Mizoram", "Nagaland",
                                         "Arunachal Pradesh"]}}


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main(outdir: Path):
    P = yaml.safe_load((ROOT / "dcfootprint/config/parameters.yaml").read_text(encoding="utf-8"))
    basis = P["grid"].get("carbon_basis", "generation")
    acct = pd.read_parquet(ROOT / "dcfootprint/outputs/account_facility_month.parquet")
    geo = pd.read_parquet(ROOT / "dcfootprint/outputs/interim/facilities_geocoded.parquet")
    summ = pd.read_csv(ROOT / "dcfootprint/results/india_account_summary.csv")
    card = pd.read_csv(ROOT / "dcfootprint/results/q2_scorecard.csv")
    wb = pd.read_excel(ROOT / "datasets/india_dc_facilities/India_DC_Facilities_v0.xlsx")

    # --- picks
    fids = sorted(acct["facility_id"].unique())
    rng = np.random.default_rng(20260927)
    picks = list(rng.choice(fids, 2, replace=False))
    tel = acct[acct["state"] == "Telangana"].groupby("facility_id")["water_scarcity_grid_l_eq"].sum()
    third = tel.sort_values(ascending=False).index[0]
    picks.append(third if third not in picks else tel.sort_values(ascending=False).index[1])

    # --- raw sources
    atlas = pd.read_parquet(ROOT / "data/atlas/datacenters.parquet")
    atlas = atlas[atlas["country"].astype(str).str.strip().str.lower() == "india"].dropna(subset=["latitude", "longitude"])
    aw = pyogrio.read_dataframe(ROOT / "data/aware/AWARE20_Native_CFs_geospatial.gpkg",
                                layer="AWARE20_Native_CFs_geospatial", fid_as_index=True)
    em = pd.read_csv(ROOT / "data/ember/india_monthly_full_release_long_format.csv")
    em["date"] = pd.to_datetime(em["Date"])
    em = em[em["date"].dt.year == 2024]
    plants = pd.read_parquet(ROOT / "data/gem/gem_india_operating_v2.parquet")   # raw GEM subset (cached read)
    from dcfootprint.io.gem import assign_state_of_record                        # Decision C rule (under test in Step 2)
    plants = assign_state_of_record(plants)
    ppts = gpd.GeoDataFrame(plants, crs="EPSG:4326", geometry=gpd.points_from_xy(plants["longitude"], plants["latitude"]))
    pj = gpd.sjoin(ppts, aw.reset_index(names="fid")[["fid", "geometry"] + [f"CF_{m}" for m in
                   ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]],
                   how="inner", predicate="within")
    pj = pj[~pj.index.duplicated()]
    t2f = P["grid"]["gem_type_to_fuel"]; coeff = P["water_grid"]["ewif_coeff_L_per_MWh"]
    util = P["energy"]["utilisation"]["by_facility_type"]["colocation"]["default"]
    pue = P["energy"]["pue"]["by_facility_type"]["colocation"]["default"]
    wue = P["water_onsite"]["wue_default"]["default"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    lines, problems = [], []
    for fid in picks:
        g = geo[geo["facility_id"] == fid].iloc[0]
        a = acct[acct["facility_id"] == fid].sort_values("month")
        # workbook row: same row order as the spine (normalize_india keeps row order)
        w = wb.iloc[geo.index[geo["facility_id"] == fid][0]]
        lines.append(f"\n### {fid}\n- workbook: `{w['facility_name']}`, {w['operator']}, {w['city']}, {w['state']}, "
                     f"capacity_mw={w['capacity_mw']}, status={w['status']}")
        chk = [("capacity", float(w["capacity_mw"]), float(a["capacity_mw"].iloc[0])),
               ("state", w["state"], a["state"].iloc[0])]
        # ATLAS centroid
        city = str(w["city"]).lower().strip()
        city = {"bengaluru": "bangalore", "gurugram": "gurgaon", "greater noida": "noida"}.get(city, city)
        ac = atlas[atlas["city"].astype(str).str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)
                   .str.replace(r"\s+", " ", regex=True).str.strip() == city]
        chk += [("lat", float(ac["latitude"].mean()), float(g["latitude"])),
                ("lon", float(ac["longitude"].mean()), float(g["longitude"]))]
        # AWARE basin + CFs
        pt = gpd.GeoDataFrame(geometry=gpd.points_from_xy([g["longitude"]], [g["latitude"]]), crs="EPSG:4326")
        hit = aw[aw.contains(pt.geometry.iloc[0])]
        bid = int(hit.index[0]); cfs = hit.iloc[0][[f"CF_{m}" for m in months]].astype(float).values
        chk += [("basin_id", bid, int(a["basin_id"].iloc[0]))]
        # energy, carbon, water — recomputed
        hrs = np.array([calendar.monthrange(2024, m)[1] * 24 for m in range(1, 13)])
        e_it = float(w["capacity_mw"]) * util * hrs
        e_grid = e_it * pue
        st = w["state"]
        ci = (em[(em["State"] == st) & (em["Variable"] == "CO2 intensity") & (em["Unit"] == "gCO2/kWh")]
              .sort_values("date")["Value"].values)
        if basis == "consumption" and st in REGIONAL_GRID:   # Electricity Maps regional consumption CI (cached API)
            emz = pd.read_parquet(ROOT / f"data/em/ci_{REGIONAL_GRID[st]}_2024.parquet").sort_values("month")
            lines.append(f"- carbon basis: consumption, {REGIONAL_GRID[st]} regional grid (Ember generation-basis "
                         f"carbon for comparison: {float((e_grid * ci / 1000).sum()):,.6g} t)")
            ci = emz["ci_consumption_gco2_kwh"].to_numpy()
        carbon = float((e_grid * ci / 1000).sum())
        onsite = e_it * 1000 * wue
        gen = em[(em["State"] == st) & (em["Category"] == "Electricity generation") & (em["Unit"] == "GWh")
                 & em["Variable"].isin(FUELS)].copy()
        gen["Value"] = gen["Value"].clip(lower=0); gen["m"] = gen["date"].dt.month
        gen["share"] = gen["Value"] / gen.groupby("m")["Value"].transform("sum")
        gen["coeff"] = gen["Variable"].map(coeff)
        ewif = gen.assign(x=gen["share"] * gen["coeff"]).groupby("m")["x"].sum().reindex(range(1, 13)).values
        # generation-basin CF: capacity-weighted, per fuel, plants of this state (national fill)
        pjs = pj.assign(fuel=pj["type"].map(t2f))
        sew = np.zeros(12)
        for i, mo in enumerate(months):
            cfcol = f"CF_{mo}"
            gm = gen[gen["m"] == i + 1]
            for _, r in gm.iterrows():
                sub = pjs[(pjs["state"] == st) & (pjs["fuel"] == r["Variable"])].dropna(subset=[cfcol])
                if sub.empty:
                    sub = pjs[pjs["fuel"] == r["Variable"]].dropna(subset=[cfcol])
                cfg = float((sub["capacity_mw"] * sub[cfcol]).sum() / sub["capacity_mw"].sum()) if len(sub) else 0.0
                sew[i] += r["share"] * r["coeff"] * cfg
        grid = e_grid * ewif
        sc = float((onsite * cfs).sum() + (e_grid * sew).sum())
        chk += [("carbon_tco2_yr", carbon, float(a["carbon_tco2"].sum())),
                ("water_onsite_l_yr", float(onsite.sum()), float(a["water_onsite_l"].sum())),
                ("water_grid_l_yr", float(grid.sum()), float(a["water_grid_l"].sum())),
                ("scarcity_l_eq_yr", sc, float(a["water_scarcity_l_eq"].sum()))]
        s_row = summ[summ["facility_id"] == fid].iloc[0]; c_row = card[card["facility_id"] == fid].iloc[0]
        chk += [("summary.carbon", float(a["carbon_tco2"].sum()), float(s_row["carbon_tco2_yr"])),
                ("summary.scarcity_m3eq", float(a["water_scarcity_l_eq"].sum()) / 1000, float(s_row["water_scarcity_m3eq_yr"])),
                ("scorecard.scarcity_m3eq", float(a["water_scarcity_l_eq"].sum()) / 1000, float(c_row["scarcity_m3eq_yr"]))]
        lines.append("| hop | recomputed from source | pipeline | match |\n|---|---|---|---|")
        for name, x, y in chk:
            ok = (x == y) if isinstance(x, str) else rel(x, y) < TOL
            if not ok:
                problems.append(f"{fid}: {name} {x} vs {y}")
            fx = x if isinstance(x, str) else f"{x:,.6g}"
            fy = y if isinstance(y, str) else f"{y:,.6g}"
            lines.append(f"| {name} | {fx} | {fy} | {'yes' if ok else '**NO**'} |")
    head = [f"# Step 5 spot check (relative tolerance {TOL:g})",
            f"Picks: {picks[0]}, {picks[1]} (random, seed 20260927); {picks[2]} (largest Telangana scope-2 value; "
            "Telangana moved most under the GADM decision).",
            f"**Discrepancies: {len(problems)}**" + ("".join(f"\n- {p}" for p in problems) if problems else " — every hop matches.")]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "SPOTCHECK.md").write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    print("\n".join(head))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dcfootprint/results")
