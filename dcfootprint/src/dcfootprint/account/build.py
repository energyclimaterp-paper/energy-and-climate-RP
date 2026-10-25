"""L2 build — assemble the facility-month account (product ②, the dataset).

Wires the layers on real data:
  energy (E_IT, E_grid)  ->  carbon  E_grid * CI(zone, month)            (Ember zone-month)
                          ->  water   scope-1 WUE*E_IT  +  scope-2 EWIF(zone, month)*E_grid
                          ->  scarcity scope-1 * AWARE_CF(facility basin, month)
                                       scope-2 * CF of the basins where the zone's power is
                                                 generated (GEM plants; geo/generation_basins.py)
                          ->  inference scoping scalar (column)
zone = grid.zone in parameters.yaml (state | national). Output validated against
schemas.FacilityMonthAccount, then written; a small git-committed summary goes to results/.

Present-account scope: Operational, costed (capacity known), basin-resolved facilities.
Pipeline (announced/under-construction) is held for the L3 projection, not the present account.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.io import ember, gem
from dcfootprint.account import energy as energy_mod, carbon as carbon_mod
from dcfootprint.settings import params as _cfg_params

_CONTRACT_COLS = ["facility_id", "date", "region", "e_it_mwh", "e_grid_mwh", "carbon_tco2",
                  "water_onsite_l", "water_grid_l", "water_phys_l",
                  "water_scarcity_onsite_l_eq", "water_scarcity_grid_l_eq", "water_scarcity_l_eq",
                  "inference_share"]


def load_params() -> dict:
    return _cfg_params()


@lru_cache(maxsize=8)
def grid_tables(zone: str, year: int, region: str = "India") -> dict:
    """Zone-month grid inputs shared by the account, routing and siting:
    ci [zone_id, month, ci_gco2_per_kwh, ci_source], grid_water [zone_id, month, ewif...,
    sewif...], plants (GEM operating, with basin_id), fuel_cf."""
    from dcfootprint.geo import join as gj, generation_basins as gb
    params = load_params()
    raw = ember.load_raw(region)
    ci = ember.zone_month_ci(raw, year, zone)
    shares = ember.zone_month_fuel_shares(raw, year, zone)
    basins = gj.load_aware_basins()
    cf = gj.basin_monthly_cf(basins)
    plants = gb.plant_basins(gem.region_plants(region), basins)
    fuel_cf = gb.fuel_basin_cf(plants, cf, params["grid"]["gem_type_to_fuel"], zone)
    gw = gb.zone_month_grid_water(shares, fuel_cf, params["water_grid"]["ewif_coeff_L_per_MWh"], zone)
    from dcfootprint.validation import schemas
    schemas.GridWater.validate(gw)
    schemas.GemPlants.validate(plants)
    return {"ci": ci, "shares": shares, "grid_water": gw, "plants": plants, "fuel_cf": fuel_cf, "basin_cf": cf}


def _wue_by_facility_id(fac: pd.DataFrame, params: dict) -> dict:
    wcfg = params["water_onsite"]
    default = float(wcfg["wue_default"]["default"])
    byop = {k.lower(): v for k, v in wcfg.get("wue_by_operator", {}).items()}
    wue = pd.Series(default, index=fac.index, dtype=float)
    opl = fac["operator"].fillna("").str.lower()
    for op, val in byop.items():
        wue.loc[opl.str.contains(op, regex=False)] = float(val)
    return dict(zip(fac["facility_id"], wue))


def facility_month_account(config: dict | None = None, region: str = "India") -> pd.DataFrame:
    """Facility-month account for a facility-tier region (India, US)."""
    params = load_params()
    root = _repo_root()
    zone = params["grid"]["zone"]
    year = int(params["region_config"][region]["account_year"])
    if region == "India":
        fac = pd.read_parquet(root / "dcfootprint" / "outputs" / "interim" / "facilities_geocoded.parquet")
        acct_fac = fac[(fac["status"] == "Operational") & fac["capacity_mw"].notna() & fac["basin_id"].notna()].copy()
        n_op_total = int((fac["status"] == "Operational").sum())
        # Hyperscale cloud regions are EXCLUDED (decision 2026-09-27): no region-level capacity is
        # publicly disclosed; announced investment figures are multi-year capital commitments, not
        # operational MW, and are not used as a proxy. They drop out via capacity_mw = NaN; counted here.
        op = fac[fac["status"] == "Operational"]
        hyperscale_regions = op["operator_family"].fillna("").str.contains("hyperscale", case=False)
        extra_meta = {
            "n_hyperscale_regions_excluded": int(hyperscale_regions.sum()),
            "n_hyperscale_regions_with_capacity": int((hyperscale_regions & op["capacity_mw"].notna()).sum()),
            "n_operational_uncosted_other": int((~hyperscale_regions & op["capacity_mw"].isna()).sum()),
            "n_operational_no_basin": int((op["capacity_mw"].notna() & op["basin_id"].isna()).sum()),
        }
    elif region == "US":
        from dcfootprint.io import us_facilities
        from dcfootprint.geo import join as gj
        spine = us_facilities.build()
        fac = gj.assign_basin(spine, gj.load_aware_basins())
        acct_fac = fac[fac["basin_id"].notna()].copy()
        n_op_total = int(spine.attrs["meta"]["n_operational_datacenters"])
        extra_meta = {"us_capacity_basis": spine.attrs["meta"]["basis_counts"],
                      "us_unspecified_basis_treated_as": spine.attrs["meta"]["unspecified_as"],
                      "n_operational_uncosted": n_op_total - int(spine.attrs["meta"]["n_with_capacity"]),
                      "n_costed_no_basin": int(fac["basin_id"].isna().sum())}
    else:
        raise ValueError(f"facility_month_account: region {region!r} is not facility-tier")

    g = grid_tables(zone, year, region)
    cf = g["basin_cf"].copy()

    ci = g["ci"]
    carbon_basis = params["grid"].get("carbon_basis", "generation")
    if carbon_basis == "consumption" and region == "India":
        try:                                           # R2: consumption-based India carbon (Electricity Maps)
            from dcfootprint.io import electricitymaps as em
            ci = em.apply_consumption_ci(ci, year)
        except Exception as exc:                       # no EM key/cache -> fall back to generation, flagged
            carbon_basis = f"generation (consumption unavailable: {type(exc).__name__})"
    elif carbon_basis == "consumption":                # consumption CI exists for India's grids only
        carbon_basis = "generation (consumption basis is India-only)"

    e = energy_mod.energy_account(acct_fac, params, year)
    e["zone_id"] = e["state"].astype(str)
    e = carbon_mod.carbon_account(e, ci)

    # --- scope-1 (on-site) water, scarcity at the facility's own basin ---
    e["wue_l_per_kwh"] = e["facility_id"].map(_wue_by_facility_id(acct_fac, params))
    e["water_onsite_l"] = e["wue_l_per_kwh"] * e["e_it_mwh"] * 1000.0          # MWh -> kWh
    e["basin_id"] = e["basin_id"].astype("int64")
    cf["basin_id"] = cf["basin_id"].astype("int64")
    e = e.merge(cf, on=["basin_id", "month"], how="left")
    missing_cf = int(e["cf"].isna().sum())
    e = e[e["cf"].notna()].copy()
    e["water_scarcity_onsite_l_eq"] = e["water_onsite_l"] * e["cf"]

    # --- scope-2 (grid) water, scarcity at the generation basins of the zone ---
    e = e.merge(g["grid_water"], on=["zone_id", "month"], how="left")
    e["water_grid_l"] = e["ewif_l_per_mwh"] * e["e_grid_mwh"]
    e["water_grid_hydro_l"] = e["ewif_hydro_l_per_mwh"] * e["e_grid_mwh"]
    e["water_scarcity_grid_l_eq"] = e["sewif_l_eq_per_mwh"] * e["e_grid_mwh"]
    e["water_scarcity_grid_hydro_l_eq"] = e["sewif_hydro_l_eq_per_mwh"] * e["e_grid_mwh"]
    e["cf_grid_eff"] = (e["sewif_l_eq_per_mwh"] / e["ewif_l_per_mwh"]).where(e["ewif_l_per_mwh"] > 0, 0.0)

    e["water_phys_l"] = e["water_onsite_l"] + e["water_grid_l"]
    e["water_scarcity_l_eq"] = e["water_scarcity_onsite_l_eq"] + e["water_scarcity_grid_l_eq"]

    e["inference_share"] = float(params["inference"]["sectoral_share"]["default"])
    e["date"] = e["month"].map(lambda m: pd.Timestamp(year, int(m), 1))
    e["region"] = region

    keep = _CONTRACT_COLS + ["state", "zone_id", "basin_id", "facility_type", "capacity_mw", "operator", "city",
                             "cf", "cf_grid_eff", "ci_gco2_per_kwh", "ci_source", "pue", "wue_l_per_kwh",
                             "water_grid_hydro_l", "water_scarcity_grid_hydro_l_eq", "month"]
    account = e[keep].reset_index(drop=True)

    from dcfootprint.validation import schemas
    schemas.FacilityMonthAccount.validate(account[_CONTRACT_COLS])   # contract enforced

    account.attrs["meta"] = {
        "account_year": year, "grid_zone": zone,
        "ci_mean_gco2_per_kwh": round(float((account["carbon_tco2"].sum() * 1000) / account["e_grid_mwh"].sum()), 1),
        "region": region,
        "n_facilities": int(acct_fac["facility_id"].nunique()), "n_operational_total": n_op_total,
        **extra_meta,
        "dropped_facility_months_no_cf": missing_cf,
        "carbon_basis": carbon_basis,
        "ci_state_fill_months": int((~account["ci_source"].isin(["ember_state", "em_consumption"])).sum()) if zone == "state" else 0,
    }
    try:                                           # optional cross-check, CEA not required
        from dcfootprint.io import cea
        account.attrs["meta"]["cea_national_ef_crosscheck"] = round(cea.india_grid_ef()["ef_tco2_per_mwh"], 4)
    except Exception:
        account.attrs["meta"]["cea_national_ef_crosscheck"] = None
    return account


def _write_results(account: pd.DataFrame) -> Path:
    """Small, git-committed summary artefacts in dcfootprint/results/."""
    meta = account.attrs["meta"]
    ann = account.groupby(["facility_id", "operator", "city", "state", "basin_id"], as_index=False).agg(
        carbon_tco2_yr=("carbon_tco2", "sum"),
        water_phys_m3_yr=("water_phys_l", lambda s: s.sum() / 1000.0),
        water_onsite_m3_yr=("water_onsite_l", lambda s: s.sum() / 1000.0),
        water_scarcity_m3eq_yr=("water_scarcity_l_eq", lambda s: s.sum() / 1000.0),
        water_scarcity_onsite_m3eq_yr=("water_scarcity_onsite_l_eq", lambda s: s.sum() / 1000.0),
        water_scarcity_grid_m3eq_yr=("water_scarcity_grid_l_eq", lambda s: s.sum() / 1000.0),
        capacity_mw=("capacity_mw", "first"),
    ).sort_values("water_scarcity_m3eq_yr", ascending=False)

    resdir = _repo_root() / "dcfootprint" / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    ann.to_csv(resdir / "india_account_summary.csv", index=False)

    inf = account["inference_share"].iloc[0]
    tot_c = ann["carbon_tco2_yr"].sum()
    tot_wp, tot_w1 = ann["water_phys_m3_yr"].sum(), ann["water_onsite_m3_yr"].sum()
    tot_ws = ann["water_scarcity_m3eq_yr"].sum()
    tot_ws1 = ann["water_scarcity_onsite_m3eq_yr"].sum()
    by_state = ann.groupby("state")["water_scarcity_m3eq_yr"].sum().sort_values(ascending=False)

    basis = str(meta.get("carbon_basis", "generation"))
    carbon_caveat = (
        "- India carbon is **consumption-based** (Electricity Maps regional flow-traced, lifecycle) -> removes "
        "the R2 generation-vs-consumption caveat; generation basis via `grid.carbon_basis: generation` is the "
        "sensitivity (EM is lifecycle vs Ember direct, so the level blends the two; the re-ranking is the "
        "consumption signal).\n"
        if basis.startswith("consumption") else
        "- state CI is Ember GENERATION-based (a state's own plants), not consumption-based (R2); set "
        "grid.carbon_basis: consumption (Electricity Maps) for the consumption reading.\n"
    )
    md = [
        "# India AI-datacenter account — v2 results (present, operational)\n",
        f"*Generated by `dcfootprint.account.build` on real data. Account year {meta['account_year']}; "
        f"grid zone = **{meta['grid_zone']}**; carbon basis = **{meta.get('carbon_basis', 'generation')}** "
        f"(energy-weighted mean {meta['ci_mean_gco2_per_kwh']} gCO2/kWh).*\n",
        f"- **Facilities in account:** {meta['n_facilities']} (operational, costed, basin-resolved) "
        f"of {meta['n_operational_total']} operational total — coverage gap = un-costed / un-geocoded.",
        f"- **Total carbon:** {tot_c:,.0f} tCO2/yr  (inference-attributed ~{tot_c*inf:,.0f} tCO2/yr at {inf:.0%})",
        f"- **Total physical water:** {tot_wp:,.0f} m3/yr  (scope-1 on-site {tot_w1:,.0f} = {tot_w1/tot_wp:.0%})",
        f"- **Total scarcity-weighted water:** {tot_ws:,.0f} m3-eq/yr  *(primary metric)* — "
        f"scope-1 {tot_ws1:,.0f} ({tot_ws1/tot_ws:.0%}), scope-2 at generation basins {tot_ws-tot_ws1:,.0f}\n",
        "## Scarcity-weighted water by state (top 6)\n",
        by_state.head(6).apply(lambda v: f"{v:,.0f} m3-eq/yr").to_string(),
        "\n\n## Top 8 facilities by scarcity-weighted water\n",
        ann.head(8)[["operator", "city", "state", "capacity_mw", "carbon_tco2_yr", "water_scarcity_m3eq_yr"]].to_string(index=False),
        "\n\n### Caveats (v2)\n",
        "- util/PUE are modeled scalars (R1) -> absolutes are indicative; lead with spatial/relative results.\n"
        + carbon_caveat +
        "- scope-2 scarcity uses capacity-weighted CF of the zone's GEM plants per fuel (proxy for generation weighting).\n"
        "- coordinates are ATLAS city-centroids (facilities in one city share a basin); precise geocode pending.\n",
    ]
    (resdir / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    return resdir


if __name__ == "__main__":
    account = facility_month_account()
    meta = account.attrs["meta"]
    out = _repo_root() / "dcfootprint" / "outputs" / "account_facility_month.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    account.to_parquet(out)
    resdir = _write_results(account)

    ann_c = account["carbon_tco2"].sum()
    ann_wp = account["water_phys_l"].sum() / 1000.0
    ann_ws = account["water_scarcity_l_eq"].sum() / 1000.0
    print(f"ACCOUNT rows: {len(account)}  facilities: {meta['n_facilities']}  months: {account['date'].nunique()}")
    print(f"meta: {meta}")
    print(f"TOTAL carbon:  {ann_c:,.0f} tCO2/yr")
    print(f"TOTAL water:   {ann_wp:,.0f} m3/yr physical | {ann_ws:,.0f} m3-eq/yr scarcity-weighted")
    print(f"contract: PASS (FacilityMonthAccount)  | wrote {out.name} + results/RESULTS.md + india_account_summary.csv")
