"""L2 — EU country-month account (country tier: facility data is confidential under
Reg (EU) 2024/1364; only Member State aggregates are published).

Source: Commission, "Assessment of the energy performance and sustainability of data centres in
EU — First Technical Report" (EY/AIT/Borderstep, July 2025), data of the first reporting period
(calendar 2023), extracted to data/eu_eed/eu_member_state_2023.csv (Tables 13 and 24; the
extraction reproduces the report's Table 22 EU totals: 3,738.86 MW, 14,088 GWh, 6,223,391 m3).

Per Member State c with reporting datacentres, month m of 2023 (months weighted by hours):
  E_grid(c,m)  = reported energy EDC(c) * hours(m)/8760           MEASURED
  E_IT         = E_grid / PUE_EU, PUE_EU = 1.36 (Table 23, energy-weighted, n=681) — EU average
                 applied to every country (per-country PUE is only in an unlabelled figure)
  Carbon       = E_grid * CI(c, m)                                Ember Europe, CO2-equivalent
  W_onsite     = reported water input WIN(c) * hours(m)/8760      MEASURED; categories 1+2 of
                 EN 50600-4-9 = water INPUT (withdrawal), not consumption -> flagged
  W_grid       = EWIF(c, m) * E_grid                              Ember EU fuel mix x Macknick
  S_onsite     = W_onsite * AWARE country CF (non-agricultural, month)
  S_grid       = E_grid * sEWIF(c, m)                             GEM plants of c -> AWARE basins
Coverage: only the reporting datacentres (Table 13: 770 of an estimated 2,161, 36 %); countries
with none reporting are not in the account.
"""
from __future__ import annotations

import calendar

import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

PUE_EU = 1.36          # Table 23, energy-weighted average of reporting DCs (n = 681)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def aware_country_cf(iso3: list[str]) -> pd.DataFrame:
    rc = _cfg_params()["region_config"]["EU"]
    x = pd.read_excel(_repo_root() / rc["aware_country_path"], sheet_name=rc["aware_country_sheet"])
    x = x[x["GLAM_ISO3"].isin(iso3)]
    long = x.melt(id_vars="GLAM_ISO3", value_vars=_MONTHS, var_name="mon", value_name="cf")
    long["month"] = long["mon"].map({m: i + 1 for i, m in enumerate(_MONTHS)})
    return long.rename(columns={"GLAM_ISO3": "iso3"})[["iso3", "month", "cf"]]


def country_month_account() -> pd.DataFrame:
    from dcfootprint.account.build import grid_tables, _CONTRACT_COLS
    from dcfootprint.account import carbon as carbon_mod
    from dcfootprint.io import ember
    P = _cfg_params()
    rc = P["region_config"]["EU"]
    year, zone = int(rc["account_year"]), P["grid"]["zone"]
    eu = pd.read_csv(_repo_root() / rc["eu_eed_path"])
    rep = eu[eu["reported_energy_gwh"].notna()].copy()

    raw = ember.load_raw("EU")
    iso = (raw.dropna(subset=["ISO 3 code"]).drop_duplicates("State").set_index("State")["ISO 3 code"].to_dict())
    rep["iso3"] = rep["country"].map(iso)
    hrs = pd.DataFrame({"month": range(1, 13), "hours": [calendar.monthrange(year, m)[1] * 24 for m in range(1, 13)]})
    frac = hrs.assign(frac=hrs["hours"] / hrs["hours"].sum())[["month", "frac"]]
    e = rep.merge(frac, how="cross")
    e["zone_id"] = e["country"]
    e["e_grid_mwh"] = e["reported_energy_gwh"] * 1000.0 * e["frac"]
    e["e_it_mwh"] = e["e_grid_mwh"] / PUE_EU
    g = grid_tables(zone, year, "EU")
    e = carbon_mod.carbon_account(e, g["ci"])
    e["water_onsite_l"] = e["reported_water_input_m3"] * 1000.0 * e["frac"]
    e = e.merge(aware_country_cf(rep["iso3"].dropna().tolist()), on=["iso3", "month"], how="left")
    missing_cf = sorted(e.loc[e["cf"].isna(), "country"].unique())
    if missing_cf:
        raise KeyError(f"no AWARE country CF for {missing_cf}")
    e["water_scarcity_onsite_l_eq"] = e["water_onsite_l"] * e["cf"]
    e = e.merge(g["grid_water"], on=["zone_id", "month"], how="left")
    e["water_grid_l"] = e["ewif_l_per_mwh"] * e["e_grid_mwh"]
    e["water_grid_hydro_l"] = e["ewif_hydro_l_per_mwh"] * e["e_grid_mwh"]
    e["water_scarcity_grid_l_eq"] = e["sewif_l_eq_per_mwh"] * e["e_grid_mwh"]
    e["water_scarcity_grid_hydro_l_eq"] = e["sewif_hydro_l_eq_per_mwh"] * e["e_grid_mwh"]
    e["cf_grid_eff"] = (e["sewif_l_eq_per_mwh"] / e["ewif_l_per_mwh"]).where(e["ewif_l_per_mwh"] > 0, 0.0)
    e["water_phys_l"] = e["water_onsite_l"] + e["water_grid_l"]
    e["water_scarcity_l_eq"] = e["water_scarcity_onsite_l_eq"] + e["water_scarcity_grid_l_eq"]
    e["inference_share"] = float(P["inference"]["sectoral_share"]["default"])
    e["date"] = e["month"].map(lambda m: pd.Timestamp(year, int(m), 1))
    e["region"] = "EU"
    e["facility_id"] = "eu-" + e["country"].str.lower().str.replace(" ", "-")
    e["state"], e["basin_id"], e["facility_type"] = e["country"], pd.NA, "country_aggregate"
    e["capacity_mw"] = e["reported_it_mw"]
    e["operator"] = e["reporting_n_dc"].map(lambda n: f"{int(n)} reporting datacentres")
    e["city"], e["pue"] = None, PUE_EU
    e["wue_l_per_kwh"] = e["water_onsite_l"] / (e["e_it_mwh"] * 1000.0)
    keep = _CONTRACT_COLS + ["state", "zone_id", "basin_id", "facility_type", "capacity_mw", "operator", "city",
                             "cf", "cf_grid_eff", "ci_gco2_per_kwh", "ci_source", "pue", "wue_l_per_kwh",
                             "water_grid_hydro_l", "water_scarcity_grid_hydro_l_eq", "month",
                             "reporting_share_pct", "est_n_dc", "reporting_n_dc"]
    account = e[keep].reset_index(drop=True)
    from dcfootprint.validation import schemas
    schemas.FacilityMonthAccount.validate(account[_CONTRACT_COLS])
    account.attrs["meta"] = {
        "region": "EU", "account_year": year, "tier": "country",
        "n_countries_reporting": int(rep["country"].nunique()), "n_countries_total": int(len(eu)),
        "n_reporting_dc": int(eu["reporting_n_dc"].sum()), "n_estimated_dc": int(eu["est_n_dc"].sum()),
        "reported_energy_gwh": float(rep["reported_energy_gwh"].sum()), "pue_eu_assumed": PUE_EU,
        "ci_mean_gco2_per_kwh": round(float(account["carbon_tco2"].sum() * 1000 / account["e_grid_mwh"].sum()), 1),
    }
    return account


if __name__ == "__main__":
    a = country_month_account()
    print(a.attrs["meta"])
    t = a.groupby("state").agg(gwh=("e_grid_mwh", lambda s: s.sum() / 1e3), carbon=("carbon_tco2", "sum"),
                               scar=("water_scarcity_l_eq", lambda s: s.sum() / 1e3)).sort_values("scar", ascending=False)
    print(t.round(0).head(10).to_string())
    print(f"TOTAL carbon {a.carbon_tco2.sum():,.0f} t | physical {a.water_phys_l.sum() / 1e3:,.0f} m3 | "
          f"scarcity {a.water_scarcity_l_eq.sum() / 1e3:,.0f} m3-eq")
