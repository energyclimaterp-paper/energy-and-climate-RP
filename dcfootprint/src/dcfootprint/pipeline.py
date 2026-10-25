"""End-to-end pipeline: runs L0->L9 in order, validates, and writes every output
to dcfootprint/results/ + a combined RESULTS_FULL.md. Each stage is isolated, so a
failure is reported (with its traceback) without aborting the rest; a status table
prints at the end. This is the deterministic core; forecasting is real (rolling-origin
SARIMA vs seasonal-naive) and Q3 routing is the stylised Lyapunov controller.

Layer names follow ARCHITECTURE v2: L4 = spatial coupling (incidence matrices, used by
routing); the counterfactual levers are part of L7 (decision layer).

Run:  PYTHONPATH=dcfootprint/src python -m dcfootprint.pipeline
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root

ROOT = _repo_root()
INTERIM = ROOT / "dcfootprint" / "outputs" / "interim"
OUT = ROOT / "dcfootprint" / "outputs"
RES = ROOT / "dcfootprint" / "results"


def _check(schema: str, df: pd.DataFrame) -> pd.DataFrame:
    """Contract at the output boundary: a failing schema fails the stage (raises)."""
    from dcfootprint.validation import schemas
    getattr(schemas, schema).validate(df)
    return df


def run() -> dict:
    INTERIM.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    status, art = [], {}

    def stage(name, fn):
        try:
            art[name] = fn()
            status.append((name, "OK"))
        except Exception as e:
            status.append((name, f"FAIL: {type(e).__name__}: {e}"))
            traceback.print_exc()
            art[name] = None
        return art[name]

    # --- L0/L1: facility spine + geo join ---
    def _spine():
        from dcfootprint.io import facilities as fac_io
        from dcfootprint.geo import join as gj
        fac = fac_io.build()
        fac.to_parquet(INTERIM / "facilities.parquet")
        basins = gj.load_aware_basins()
        cf = gj.basin_monthly_cf(basins)
        cf.to_parquet(INTERIM / "basin_cf_monthly.parquet")
        fac_geo = gj.assign_zone_and_basin(fac)
        fac_geo.to_parquet(INTERIM / "facilities_geocoded.parquet")
        return fac_geo
    fac_geo = stage("L0/L1 facilities+geo", _spine)

    # --- L2: account ---
    def _account():
        from dcfootprint.account import build
        acct = build.facility_month_account()
        acct.to_parquet(OUT / "account_facility_month.parquet")
        build._write_results(acct)
        return acct
    account = stage("L2 account", _account)

    # --- L4 incidence (exact coupling; consumed by L6 routing) + L2.5 calibration ---
    incidence = stage("L4 incidence", lambda: __import__("dcfootprint.geo.incidence", fromlist=["build_incidence"]).build_incidence(fac_geo))
    A_basin = incidence[1] if incidence is not None else None
    cal = stage("L2.5 calibrate", lambda: __import__("dcfootprint.account.calibrate", fromlist=["calibrate_capacity"]).calibrate_capacity(fac_geo))

    # --- L3 forecast + recharge ---
    def _forecast():
        from dcfootprint.project import forecast
        r = forecast.forecast_ci(); _check("ForecastCI", r["forecast"]).to_csv(RES / "forecast_ci.csv", index=False)
        bt = r["backtest"]
        pd.DataFrame([{"model": m, "fold_origin": o, "rmse": v}
                      for m, vs in bt["fold_rmse"].items() for o, v in zip(bt["fold_origins"], vs)]
                     ).to_csv(RES / "forecast_backtest.csv", index=False)
        r["core_benchmark"] = forecast.core_benchmark()
        from dcfootprint.project.hierarchy import chosen_onestep_ci
        from dcfootprint.settings import params as _p
        g = _p()["grid"]
        ci, tab, method = chosen_onestep_ci(tuple(sorted(account["zone_id"].astype(str).unique())),
                                            int(g["account_year"]), g["zone"])
        tab.to_csv(RES / "forecast_ci_method_backtest.csv", index=False)
        ci.to_csv(RES / "forecast_ci_onestep_q3.csv", index=False)
        r["q3_method"] = method
        if r["core_benchmark"].get("rmse_by_model"):
            pd.DataFrame(r["core_benchmark"]["rmse_by_model"]).to_csv(RES / "forecast_core_benchmark.csv", index=False)
        return r
    fc = stage("L3 forecast", _forecast)
    rech = stage("L3 recharge", lambda: __import__("dcfootprint.project.recharge", fromlist=["recharge_climatology"]).recharge_climatology())

    # --- L5 policy-gap ---
    def _policy():
        from dcfootprint.policy import gap, rag_bridge
        gap.registry().to_csv(RES / "policy_registry.csv", index=False)      # the single source
        gap.four_axis_matrix().to_csv(RES / "regulation_matrix.csv", index=False)   # derived
        gap.HARD_CONSTRAINTS.to_csv(RES / "regulation_constraints.csv", index=False)  # binding view
        ov = gap.gap_overlay(account)
        try:                                        # attach the RAG corpus as cited evidence
            rag_bridge.load_manifest().to_csv(RES / "policy_corpus.csv", index=False)
            rag_bridge.cited_constraints().to_csv(RES / "regulation_constraints_cited.csv", index=False)
            ov["rag_corpus"] = rag_bridge.corpus_summary()
        except Exception as e:
            ov["rag_corpus"] = {"available": False, "reason": str(e)}
        return ov
    gap_overlay = stage("L5 policy-gap", _policy)

    # --- L6 routing (Q3), with and without legal limits ---
    def _routing():
        from dcfootprint.routing import lyapunov
        with_l = lyapunov.compare(account, legal=True, A_basin=A_basin)
        without_l = lyapunov.compare(account, legal=False, A_basin=A_basin)
        r = pd.concat([with_l, without_l], ignore_index=True)
        _check("RoutingComparison", r).to_csv(RES / "routing_comparison.csv", index=False)
        lyapunov.v_sweep(account, A_basin=A_basin).to_csv(RES / "routing_v_sweep.csv", index=False)
        bs = lyapunov.budget_sweep(account, A_basin=A_basin); bs.to_csv(RES / "routing_budget_sweep.csv", index=False)
        return {"table": r, "unstabilisable": with_l.attrs["n_unstabilisable_basins"],
                "n_basins": with_l.attrs["n_basins"], "overdraft": with_l.attrs["overdraft_basin_months"],
                "alpha": with_l.attrs["budget_alpha"], "fixed_floor": with_l.attrs["fixed_only_peak_queue_m3"],
                "budget_sweep": bs}
    routing = stage("L6 routing (Q3)", _routing)

    # --- L7 decisions: levers, scorecard (Q2), siting (Q1) ---
    def _levers():
        from dcfootprint.counterfactual import levers
        lv = _check("LeverSavings", levers.rank_lever_savings(account)); lv.to_csv(RES / "lever_savings.csv", index=False); return lv
    levers_df = stage("L7 levers (counterfactual)", _levers)

    def _scorecard():
        from dcfootprint.decisions import scorecard
        s = _check("Q2Scorecard", scorecard.scorecard(account)); s.to_csv(RES / "q2_scorecard.csv", index=False); return s
    scard = stage("L7 Q2 scorecard", _scorecard)

    def _siting():
        from dcfootprint.decisions import siting
        both = siting.rank_sites(account)
        _check("Q1Siting", both["headline"]).to_csv(RES / "q1_siting.csv", index=False)
        _check("Q1Siting", both["small_grids"]).to_csv(RES / "q1_siting_small_grids.csv", index=False)
        _check("Q1Siting", both["scenarios"]).to_csv(RES / "q1_siting_scenarios.csv", index=False)
        return both
    sites = stage("L7 Q1 siting", _siting)

    # --- L8 uncertainty ---
    def _unc():
        from dcfootprint.uncertainty.monte_carlo import monte_carlo
        u = monte_carlo(account)
        (RES / "uncertainty.json").write_text(json.dumps(u, indent=2), encoding="utf-8")
        return u
    unc = stage("L8 uncertainty", _unc)

    # --- L9 combined report ---
    stage("L9 report", lambda: _write_report(account, cal, fc, rech, levers_df, gap_overlay, routing, scard, sites, unc))
    stage("L9 figures", lambda: __import__("dcfootprint.viz.figures", fromlist=["make_all"]).make_all(RES))

    print("\n===== PIPELINE STATUS =====")
    for n, s in status:
        print(f"  {'[OK]' if s == 'OK' else '[XX]'} {n:<28} {s}")
    ok = sum(1 for _, s in status if s == "OK")
    print(f"  {ok}/{len(status)} stages OK")
    return {"status": status, "artifacts": art}


def _write_report(account, cal, fc, rech, levers_df, gap_overlay, routing, scard, sites, unc) -> Path:
    L = ["# India AI-datacenter environmental pipeline — full results\n",
         "*Deterministic account -> levers -> policy-gap, + forecasting (L3), Q1 siting, Q2 scorecard, "
         "Q3 routing (stylised), uncertainty. Generated by `dcfootprint.pipeline`.*\n"]
    if account is not None:
        m = account.attrs.get("meta", {})
        c = account["carbon_tco2"].sum(); wp = account["water_phys_l"].sum() / 1e3; ws = account["water_scarcity_l_eq"].sum() / 1e3
        w1 = account["water_onsite_l"].sum() / 1e3; ws1 = account["water_scarcity_onsite_l_eq"].sum() / 1e3
        L.append(f"## L2 Account\n- {account['facility_id'].nunique()} facilities x 12 months ({m.get('account_year')}); "
                 f"grid zone **{m.get('grid_zone')}** (Ember zone-month CI, mean {m.get('ci_mean_gco2_per_kwh')} gCO2/kWh).\n"
                 f"- **{c:,.0f} tCO2/yr**; **{wp:,.0f} m3/yr** physical (scope-1 {w1/wp:.0%}); "
                 f"**{ws:,.0f} m3-eq/yr** scarcity-weighted (scope-1 at facility basin {ws1/ws:.0%}, "
                 f"scope-2 at generation basins {1-ws1/ws:.0%}).")
    if cal: L.append(f"## L2.5 Calibration\n- bottom-up {cal['bottom_up_operational_mw']} MW vs CEEW/JLL {cal['ceew_jll_range_mw']} -> {cal['verdict']} (ratio {cal['ratio_vs_low']}).")
    if fc:
        bt = fc["backtest"]
        L.append(f"## L3 Forecast\n- rolling-origin backtest ({len(bt['fold_origins'])} x 12-month folds from {bt['fold_origins'][0]}): "
                 f"winner **{bt['winner']}**; skill vs seasonal naive {bt['skill_vs_naive']}. "
                 f"Bands = empirical backtest error quantiles. earlier cached benchmark reused: "
                 f"{'yes — its Nexus rows are Chronos-2 in the Nexus slot, labelled as such' if fc.get('core_benchmark') else 'n/a'}.")
    if rech is not None: L.append(f"## L3 Recharge (R-hat)\n- monsoon-peaked seasonal recharge (peak month {int(rech.loc[rech['recharge_mult'].idxmax(),'month'])}); shapes the Q3 basin budgets.")
    if gap_overlay:
        rc = gap_overlay.get("rag_corpus", {})
        rc_line = (f" Evidence: RAG corpus of {rc['n_documents']} cited policy docs "
                   f"({rc['n_in_force']} in force) across {len(rc['jurisdictions'])} jurisdictions — "
                   f"the regulated perimeter, none mandating the four axes."
                   if isinstance(rc, dict) and rc.get("n_documents") else "")
        L.append(f"## L5 Policy-gap\n- **{gap_overlay['pct_burden_in_blind_spot']:.0f}%** of burden in a "
                 f"regulatory blind spot; axes mandated anywhere: {gap_overlay['axes_mandated_anywhere']}/4.{rc_line}")
    if routing is not None:
        t = routing["table"]; tl = t[t["legal"]].set_index("policy")
        bs = routing["budget_sweep"]
        L.append("## L6 Q3 Routing (stylised)\n"
                 f"- Budget = alpha x AWARE AMD x basin area: the water left after human consumption and environmental "
                 f"water requirements (AWARE 2.0). Headline alpha = {routing['alpha']:g}, the parameter-free bound. "
                 f"{routing['overdraft']} of {routing['n_basins'] * 12} occupied basin-months have NO water left (AMD <= 0), "
                 f"so any datacenter draw there is an overdraft.\n"
                 f"- vs static, scarcity-water saving: greedy {tl.loc['greedy','scarcity_saving_pct_vs_static']}%, "
                 f"lyapunov {tl.loc['lyapunov','scarcity_saving_pct_vs_static']}%, oracle {tl.loc['oracle','scarcity_saving_pct_vs_static']}%; "
                 f"carbon change lyapunov {tl.loc['lyapunov','carbon_saving_pct_vs_static']}% (negative = more carbon).\n"
                 f"- peak basin overdraft queue: static {tl.loc['static','peak_basin_queue_m3']:,.0f} m3, greedy "
                 f"{tl.loc['greedy','peak_basin_queue_m3']:,.0f}, lyapunov {tl.loc['lyapunov','peak_basin_queue_m3']:,.0f}; "
                 f"lyapunov penalty is {tl.loc['lyapunov','penalty_gap_pct_vs_oracle']}% above the offline oracle at equal queue peaks.\n"
                 f"- **Finding: routing alone cannot clear the overdraft.** The non-shiftable load by itself overdraws "
                 f"every basin-month with AMD <= 0, a floor of {routing['fixed_floor']:,.0f} m3 peak overdraft that no "
                 f"router can remove; lyapunov reaches {tl.loc['lyapunov','peak_basin_queue_m3']:,.0f} m3. Over the full year "
                 f"the budget covers the load in {routing['n_basins'] - routing['unstabilisable']}/{routing['n_basins']} basins at "
                 f"alpha={routing['alpha']:g} (the overdraft is seasonal), falling to "
                 f"{routing['n_basins'] - int(bs['unstabilisable_basins'].max())}/{routing['n_basins']} at alpha={bs['budget_alpha'].min():g}. "
                 f"Siting and capacity limits are needed, not only load shifting (routing_budget_sweep.csv). "
                 f"*Synthetic demand; conditional on R-hat.*")
    if levers_df is not None:
        top = levers_df.iloc[0]
        L.append("## L7 Levers — which lever pays\n" + "\n".join(
            f"- {r['lever']}: {r['water_scarcity_saved_m3eq_yr']:,.0f} m3-eq/yr ({r['pct_of_scarcity_baseline']}% of scarcity), "
            f"{r['carbon_saved_tco2_yr']:,.0f} tCO2/yr ({r['pct_of_carbon_baseline']}%)" for _, r in levers_df.iterrows()))
    if scard is not None:
        L.append(f"## L7 Q2 Scorecard\n- {len(scard)} facilities scored; **{int(scard['harm_flag'].sum())}** harm-flagged; "
                 f"recommended levers: {scard.loc[scard['harm_flag'], 'recommended_lever'].value_counts().to_dict()}.")
    if sites is not None:
        h, sg = sites["headline"], sites["small_grids"]
        lines = [f"  {int(r['rank'])}. {r['state']} basin {int(r['basin_id'])}: CF {r['cf_mean']:.1f}, CI {r['ci_mean']:.0f} g/kWh, "
                 f"{int(r['overdraft_months'])} overdraft months, rank band {r['rank_p10']:.0f}-{r['rank_p90']:.0f}"
                 for _, r in h.head(5).iterrows()]
        L.append(f"## L7 Q1 Siting\n- **Headline: {len(h)} candidate state x basin cells on grids >= 10 TWh/yr** "
                 f"({int((h['n_existing_dc'] > 0).sum())} already hold a DC), minimax regret over 4 criteria "
                 f"(new facility's scarcity water, carbon, marginal basin overdraft, grid fossil share). Top 5:\n"
                 + "\n".join(lines)
                 + (f"\n- Excluded small grids ({len(sg)} cells, q1_siting_small_grids.csv): own-generation CI "
                    f"(e.g. {sg.iloc[0]['state']} {sg.iloc[0]['ci_mean']:.0f} g/kWh) is not what a new load would draw; "
                    f"reported, not recommended." if len(sg) else ""))
        from dcfootprint.decisions.siting import scenario_summary
        if len(sites.get("scenarios", [])):
            L.append(scenario_summary(sites))
    if unc:
        L.append(f"## L8 Uncertainty\n- scarcity-weighted water 90% interval ({unc['distribution']} draws, mode = point estimate): "
                 f"{unc['scarcity_m3eq_yr']['p05']:,.0f} - {unc['scarcity_m3eq_yr']['p95']:,.0f} m3-eq/yr; MC mean = "
                 f"{unc['scarcity_mean_over_point']}x the point estimate (WUE band 0.7-2.5 L/kWh, mode 1.9).\n"
                 f"- first-order Sobol: {unc['scarcity_sobol_first_order']}.\n"
                 f"- **Sensitivity variant, no hydro reservoir evaporation:** {unc['scarcity_at_hydro_0_m3eq_yr']:,.0f} m3-eq/yr. "
                 f"The primary figure keeps Macknick 2012's hydro evaporation; attributing multi-purpose reservoir "
                 f"evaporation wholly to power is contested in the literature.")
    meta = account.attrs.get("meta", {}) if account is not None else {}
    basis = str(meta.get("carbon_basis", "generation"))
    ci_caveat = ("- state CI is consumption-based (Electricity Maps regional, import-adjusted, lifecycle): the shift from Ember "
                 "blends consumption-vs-generation with lifecycle-vs-direct; US/EU stay on Ember generation (direct).\n"
                 if basis.startswith("consumption") else
                 f"- state CI is Ember generation-based (R2; carbon_basis = {basis}): small grids that import power show "
                 "unrepresentative CI.\n")
    L.append("\n### Caveats\n- absolutes are calibrated ranges (util/PUE/WUE assumed); lead with relative/spatial results.\n"
             + ci_caveat +
             "- Q3 routing is a stylised controller (synthetic demand, budgets from AWARE AMD) — conditional on R-hat.\n"
             f"- {meta.get('n_hyperscale_regions_excluded', '?')} hyperscale cloud regions excluded — capacity data not publicly "
             "disclosed at region level; announced investment figures are multi-year capital commitments, not current "
             "operational capacity, and are not used as a proxy. All facilities in the account are colocation. "
             f"(Operational rows: {meta.get('n_operational_total', '?')}; in account {meta.get('n_facilities', '?')}; "
             f"hyperscale regions excluded {meta.get('n_hyperscale_regions_excluded', '?')}; other rows without capacity "
             f"{meta.get('n_operational_uncosted_other', '?')}; costed but no basin {meta.get('n_operational_no_basin', '?')}.)\n"
             "- Datacenter coordinates are city-centroids. GEM plant states are GADM 4.1-derived from coordinates, with 3 "
             "reviewed overrides; Lower Sileru hydro (Andhra Pradesh | Telangana) is unresolved and excluded from "
             "single-state views (config grid.gem_state_*).")
    (RES / "RESULTS_FULL.md").write_text("\n\n".join(L), encoding="utf-8")
    return RES / "RESULTS_FULL.md"


if __name__ == "__main__":
    run()
