"""US and EU runs of the decision pipeline (India runs through pipeline.run, unchanged).

    US  facility tier (Compute Atlas): account, calibration vs LBNL, levers, Q2, Q1, Q3 routing,
        uncertainty, GADM state check of the sites
    EU  country tier (EED Member State aggregates, 2023): account, coverage, levers, Q2 (countries),
        Q1 (country x basin cells), uncertainty with measured energy/water.
        Q3 is NOT run: with no site locations there are no basins to hold queues and no facilities
        to route between; routing across whole countries would be an artefact of aggregation.
Outputs: dcfootprint/results/<us|eu>/ ; accounts in dcfootprint/outputs/account_<region>.parquet.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root

ROOT = _repo_root()
OUT = ROOT / "dcfootprint" / "outputs"
LBNL_2023_TWH = 176.0     # Shehabi et al. (2024), 2024 US Data Center Energy Usage Report (LBNL), 2023 total


def _check(schema: str, df: pd.DataFrame) -> pd.DataFrame:
    from dcfootprint.validation import schemas
    getattr(schemas, schema).validate(df)
    return df


def run_region(region: str) -> dict:
    res = ROOT / "dcfootprint" / "results" / region.lower()
    res.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status, art = [], {}

    def stage(name, fn):
        try:
            art[name] = fn(); status.append((name, "OK"))
        except Exception as e:
            status.append((name, f"FAIL: {type(e).__name__}: {e}")); traceback.print_exc(); art[name] = None
        return art[name]

    def _account():
        if region == "US":
            from dcfootprint.account.build import facility_month_account
            a = facility_month_account(region="US")
        elif region == "EU":
            from dcfootprint.account.eu import country_month_account
            a = country_month_account()
        else:
            raise ValueError(region)
        a.to_parquet(OUT / f"account_{region.lower()}.parquet")
        summ = a.groupby(["facility_id", "operator", "state"], as_index=False, dropna=False).agg(
            carbon_tco2_yr=("carbon_tco2", "sum"), e_grid_mwh_yr=("e_grid_mwh", "sum"),
            water_phys_m3_yr=("water_phys_l", lambda s: s.sum() / 1e3),
            water_onsite_m3_yr=("water_onsite_l", lambda s: s.sum() / 1e3),
            water_scarcity_m3eq_yr=("water_scarcity_l_eq", lambda s: s.sum() / 1e3),
            water_scarcity_onsite_m3eq_yr=("water_scarcity_onsite_l_eq", lambda s: s.sum() / 1e3),
            water_scarcity_grid_m3eq_yr=("water_scarcity_grid_l_eq", lambda s: s.sum() / 1e3),
            capacity_mw=("capacity_mw", "first")).sort_values("water_scarcity_m3eq_yr", ascending=False)
        summ.to_csv(res / f"{region.lower()}_account_summary.csv", index=False)
        return a
    account = stage(f"{region} L2 account", _account)

    def _calibrate():
        m = dict(account.attrs.get("meta", {}))
        twh = float(account["e_grid_mwh"].sum() / 1e6)
        if region == "US":
            m.update({"bottom_up_twh": round(twh, 2), "lbnl_2023_twh": LBNL_2023_TWH,
                      "energy_coverage_vs_lbnl": round(twh / LBNL_2023_TWH, 3)})
        else:
            m.update({"reported_twh": round(twh, 3),
                      "dc_coverage_reporting_over_estimated": round(m["n_reporting_dc"] / m["n_estimated_dc"], 3)})
        (res / "calibration.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        return m
    cal = stage(f"{region} L2.5 calibrate/coverage", _calibrate)

    def _gadm():
        from dcfootprint.io import us_facilities
        from dcfootprint.io.gem import region_plants
        us_facilities.gadm_state_check(us_facilities.build()).to_csv(res / "us_site_gadm_state_check.csv", index=False)
        p = region_plants("US")
        p[p["state_source"] != "gadm_agrees"][["plant_id", "plant", "type", "capacity_mw", "latitude", "longitude",
                                                "gem_state", "gadm_state", "state", "state_source", "state_note"]] \
            .to_csv(res / "us_gem_gadm_state_diff.csv", index=False)
        return True
    if region == "US":
        stage("US L1 GADM checks", _gadm)

    def _levers():
        from dcfootprint.counterfactual import levers
        lv = _check("LeverSavings", levers.rank_lever_savings(account)); lv.to_csv(res / "lever_savings.csv", index=False); return lv
    lv = stage(f"{region} L7 levers", _levers)

    def _q2():
        from dcfootprint.decisions import scorecard
        s = _check("Q2Scorecard", scorecard.scorecard(account)); s.to_csv(res / "q2_scorecard.csv", index=False); return s
    q2 = stage(f"{region} L7 Q2 scorecard", _q2)

    def _q1():
        from dcfootprint.decisions import siting
        both = siting.rank_sites(account)
        _check("Q1Siting", both["headline"]).to_csv(res / "q1_siting.csv", index=False)
        if len(both["small_grids"]):
            _check("Q1Siting", both["small_grids"]).to_csv(res / "q1_siting_small_grids.csv", index=False)
        if len(both["scenarios"]):
            _check("Q1Siting", both["scenarios"]).to_csv(res / "q1_siting_scenarios.csv", index=False)
        return both
    q1 = stage(f"{region} L7 Q1 siting", _q1)

    rt = None
    if region == "US":
        def _q3():
            from dcfootprint.routing import lyapunov
            from dcfootprint.geo.incidence import build_incidence
            fac = account.drop_duplicates("facility_id")[["facility_id", "basin_id", "zone_id"]]
            A_basin = build_incidence(fac)[1]
            r = pd.concat([lyapunov.compare(account, legal=True, A_basin=A_basin),
                           lyapunov.compare(account, legal=False, A_basin=A_basin)], ignore_index=True)
            _check("RoutingComparison", r).to_csv(res / "routing_comparison.csv", index=False)
            bs = lyapunov.budget_sweep(account, A_basin=A_basin); bs.to_csv(res / "routing_budget_sweep.csv", index=False)
            return {"table": r, "budget_sweep": bs}
        rt = stage("US L6 routing (Q3)", _q3)

    def _unc():
        from dcfootprint.uncertainty.monte_carlo import monte_carlo
        u = monte_carlo(account, measured=(region == "EU"))
        (res / "uncertainty.json").write_text(json.dumps(u, indent=2), encoding="utf-8"); return u
    unc = stage(f"{region} L8 uncertainty", _unc)

    stage(f"{region} L9 report", lambda: _report(region, res, account, cal, lv, q2, q1, rt, unc))
    stage(f"{region} L9 figures", lambda: __import__("dcfootprint.viz.figures", fromlist=["make_region"]).make_region(region, res))
    return {"status": status, "artifacts": art}


def _report(region, res, account, cal, lv, q2, q1, rt, unc) -> Path:
    c = account["carbon_tco2"].sum(); wp = account["water_phys_l"].sum() / 1e3; ws = account["water_scarcity_l_eq"].sum() / 1e3
    ws1 = account["water_scarcity_onsite_l_eq"].sum() / 1e3
    m = account.attrs.get("meta", {})
    L = [f"# {region} datacenter account — results\n",
         f"*Generated by `dcfootprint.regions.run_region('{region}')`. Account year {m.get('account_year')}, "
         f"tier: {'facility (Compute Atlas v1.34.0)' if region == 'US' else 'country (EED Member State aggregates)'}.*\n",
         f"## Account\n- {account['facility_id'].nunique()} {'facilities' if region == 'US' else 'Member States'}; "
         f"**{c:,.0f} tCO2{'e' if region == 'EU' else ''}/yr** (mean grid CI {m.get('ci_mean_gco2_per_kwh')} g/kWh); "
         f"**{wp:,.0f} m3/yr** physical; **{ws:,.0f} m3-eq/yr** scarcity-weighted (scope-1 {ws1 / ws:.0%}, scope-2 {1 - ws1 / ws:.0%})."]
    if cal:
        if region == "US":
            L.append(f"- Coverage: {cal['n_facilities']} of {cal['n_operational_total']} operational Compute Atlas datacenters "
                     f"carry a cited capacity; bottom-up {cal['bottom_up_twh']} TWh = {cal['energy_coverage_vs_lbnl']:.0%} of the LBNL 2023 "
                     f"US total ({cal['lbnl_2023_twh']:.0f} TWh). Capacity basis: {cal.get('us_capacity_basis')} "
                     f"(unspecified treated as IT).")
        else:
            L.append(f"- Coverage: {cal['n_reporting_dc']} reporting of an estimated {cal['n_estimated_dc']} datacentres "
                     f"({cal['dc_coverage_reporting_over_estimated']:.0%}); {cal['n_countries_reporting']} of {cal['n_countries_total']} "
                     f"Member States report. Energy and water are MEASURED (reported); water is input (withdrawal). "
                     f"IT energy uses the EU-average PUE {cal['pue_eu_assumed']}. Emissions are CO2-equivalent.")
    if lv is not None:
        L.append("## Levers\n" + "\n".join(f"- {r['lever']}: {r['pct_of_scarcity_baseline']}% of scarcity water, "
                                            f"{r['pct_of_carbon_baseline']}% of carbon" for _, r in lv.iterrows()))
    if q2 is not None:
        L.append(f"## Q2 scorecard\n- {len(q2)} scored; {int(q2['harm_flag'].sum())} harm-flagged; top scarcity: "
                 + ", ".join(f"{r['operator']} ({r['state']})" for _, r in q2.head(3).iterrows()))
    if q1 is not None:
        h = q1["headline"]
        L.append(f"## Q1 siting\n- {len(h)} candidate cells on grids >= 10 TWh/yr; top 5: "
                 + "; ".join(f"{r['state']} basin {int(r['basin_id'])}" for _, r in h.head(5).iterrows()))
        from dcfootprint.decisions.siting import scenario_summary
        if len(q1.get("scenarios", [])):
            L.append(scenario_summary(q1))
    if rt is not None:
        t = rt["table"]; t = t[t["legal"]].set_index("policy")
        L.append(f"## Q3 routing (stylised)\n- scarcity water saved vs static: greedy {t.loc['greedy', 'scarcity_saving_pct_vs_static']}%, "
                 f"lyapunov {t.loc['lyapunov', 'scarcity_saving_pct_vs_static']}%, oracle {t.loc['oracle', 'scarcity_saving_pct_vs_static']}%; "
                 f"carbon lyapunov {t.loc['lyapunov', 'carbon_saving_pct_vs_static']}%; lyapunov gap to oracle "
                 f"{t.loc['lyapunov', 'penalty_gap_pct_vs_oracle']}%.")
    elif region == "EU":
        L.append("## Q3 routing\n- Not run for the EU: country-tier data has no site locations or basin queues.")
    if unc:
        L.append(f"## Uncertainty\n- scarcity 90% interval {unc['scarcity_m3eq_yr']['p05']:,.0f} - {unc['scarcity_m3eq_yr']['p95']:,.0f} m3-eq/yr "
                 f"({'EWIF/hydro/inference only: energy and water measured' if unc.get('measured_energy_and_water') else 'triangular bands'}); "
                 f"Sobol {unc['scarcity_sobol_first_order']}.")
    (res / "RESULTS_FULL.md").write_text("\n\n".join(L) + "\n", encoding="utf-8")
    return res / "RESULTS_FULL.md"


if __name__ == "__main__":
    import sys
    for reg in (sys.argv[1:] or ["US", "EU"]):
        out = run_region(reg)
        print(f"\n===== {reg} STATUS =====")
        for n, s in out["status"]:
            print(f"  {'[OK]' if s == 'OK' else '[XX]'} {n:<32} {s}")
