"""Round 7: bug fixes and consistency checks on the round-6 results.

Writes dcfootprint/results/round7_fixes/ (round5_final/ and round6_robustness/ are read, never edited).

Fixes applied, each isolated as a step in headline_steps.csv:
  capacity basis  India rows stated as total facility power (capacity_basis operational_total) are
                  converted to IT capacity with C_IT = C / PUE (account/energy.py). Round 5 and 6
                  treated them as IT, overstating 16 of 71 facilities by the PUE (1.6x).
  WUE clamp       the Gupta et al. cooling-tower curve is held at its minimum below 9.3 C wet-bulb
                  instead of rising again in cold months (account/wue.py).
  ZLD             zero liquid discharge saves recovered blowdown only, 0.92 / 5 cycles = 18.4% of
                  on-site water, not 92% (counterfactual/levers.zld_saving_fraction); also the
                  Rajasthan legal rule in routing and siting.
  siting basis    a new facility is charged the CI the account charges (India: consumption), not
                  Ember own-generation CI (decisions/siting.py); India is re-ranked here from the frozen
                  round-5 candidate cells, because a fresh run needs the GEM workbook.
  Sobol           Saltelli/Jansen first-order and total indices with bootstrap intervals, on the
                  headline total (uncertainty/monte_carlo.py); routing Monte Carlo indices bias-corrected.
  groundwater     aquifer shares on ON-SITE scarcity-weighted water only (decisions/equity.py).
  baseline        routing also reported against a status-quo baseline (no load moves).
Also: the consistent generation-basis India routing (results/generation_basis_check/), the scale-free
forecast significance tests (experiments/forecast_significance.py), and a run manifest.

India starts from the frozen round-5 account, as in round 6 (no GEM workbook or Electricity Maps key
needed). The US account is rebuilt from raw data on eGRID + EIA.

    PYTHONPATH=dcfootprint/src python dcfootprint/experiments/robustness_round7.py [--skip-routing-mc]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root

ROOT = _repo_root()
RES = ROOT / "dcfootprint" / "results"
R5, R6, GB = RES / "round5_final", RES / "round6_robustness", RES / "generation_basis_check"
OUT = RES / "round7_fixes"


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n", encoding="utf-8")


def _totals(a: pd.DataFrame) -> dict:
    return {"n_facilities": int(a["facility_id"].nunique()),
            "capacity_it_mw": round(float(a.drop_duplicates("facility_id")["capacity_mw"].sum()), 1),
            "electricity_twh": round(float(a["e_grid_mwh"].sum()) / 1e6, 3),
            "carbon_mt": round(float(a["carbon_tco2"].sum()) / 1e6, 3),
            "water_phys_mm3": round(float(a["water_phys_l"].sum()) / 1e9, 3),
            "water_onsite_mm3": round(float(a["water_onsite_l"].sum()) / 1e9, 3),
            "water_grid_mm3": round(float(a["water_grid_l"].sum()) / 1e9, 3),
            "scarcity_mm3eq": round(float(a["water_scarcity_l_eq"].sum()) / 1e9, 2),
            "scarcity_onsite_mm3eq": round(float(a["water_scarcity_onsite_l_eq"].sum()) / 1e9, 2),
            "offsite_share_of_scarcity_pct": round(100 * float(a["water_scarcity_grid_l_eq"].sum() / a["water_scarcity_l_eq"].sum()), 1)}


def _routing(a: pd.DataFrame) -> pd.DataFrame:
    from dcfootprint.routing import lyapunov
    t = pd.concat([lyapunov.compare(a, legal=True), lyapunov.compare(a, legal=False)], ignore_index=True)
    return t


def _headline(tab: pd.DataFrame, region: str, step: str) -> list[dict]:
    t = tab[tab["legal"].astype(bool)].set_index("policy")
    rows = []
    for p in ("greedy", "lyapunov", "oracle"):
        r = {"region": region, "step": step, "policy": p,
             "carbon_change_pct": -float(t.loc[p, "carbon_saving_pct_vs_static"]),
             "scarcity_change_pct": -float(t.loc[p, "scarcity_saving_pct_vs_static"])}
        if "carbon_saving_pct_vs_status_quo" in t.columns:
            r["carbon_change_vs_status_quo_pct"] = -float(t.loc[p, "carbon_saving_pct_vs_status_quo"])
            r["scarcity_change_vs_status_quo_pct"] = -float(t.loc[p, "scarcity_saving_pct_vs_status_quo"])
        rows.append(r)
    return rows


def _queue_diagnostics(tab: pd.DataFrame) -> dict:
    """Whether the basin queue changes anything: if lyapunov == greedy and lyapunov_pf == oracle, the
    water budgets never bind and the controller reduces to a per-month greedy LP."""
    t = tab[tab["legal"].astype(bool)].set_index("policy")
    same = lambda a, b: bool(np.isclose(t.loc[a, "carbon_tco2"], t.loc[b, "carbon_tco2"], rtol=1e-9)
                             and np.isclose(t.loc[a, "scarcity_m3eq"], t.loc[b, "scarcity_m3eq"], rtol=1e-9))
    return {"lyapunov_equals_greedy": same("lyapunov", "greedy"),
            "lyapunov_pf_equals_oracle": same("lyapunov_pf", "oracle")}


# ----------------------------------------------------------------------------- India
def india_account() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from dcfootprint.account.energy import correct_capacity_basis
    from dcfootprint.account.wue import facility_month_wue, recompute_onsite
    from dcfootprint.settings import params
    p = params()
    r5 = pd.read_parquet(R5 / "outputs" / "account_facility_month.parquet")
    fac = pd.read_parquet(R5 / "outputs" / "interim" / "facilities_geocoded.parquet")
    fac = fac[fac["facility_id"].isin(r5["facility_id"])]
    basis = fac.set_index("facility_id")["capacity_basis"]
    yr = int(p["region_config"]["India"]["account_year"])
    a = correct_capacity_basis(r5, basis, p)
    a = recompute_onsite(a, facility_month_wue(fac, p, yr))
    r6 = pd.read_parquet(R6 / "india" / "account_india.parquet")
    r6_cap = correct_capacity_basis(r6, basis, p)
    fp = fac[fac["capacity_basis"].isin(p["energy"]["facility_power_capacity_bases"])]
    meta = {"facility_power_rows": len(fp), "facility_power_mw_stated": round(float(fp["capacity_mw"].sum()), 1),
            "by_operator": fp.groupby("operator")["capacity_mw"].sum().round(1).to_dict()}
    totals = {"round5_published": _totals(r5), "round6_published": _totals(r6),
              "round6_plus_capacity_basis": _totals(r6_cap), "round7": _totals(a), "capacity_basis_fix": meta}
    return a, r6_cap, totals


def india_equity(a: pd.DataFrame) -> dict:
    """Aquifer shares from the round-5 CGWB matches (q2_scorecard.csv; data/cgwb is not in git)."""
    gw = pd.read_csv(R5 / "q2_scorecard.csv")[["facility_id", "gw_stage_pct", "gw_stage_source", "gw_category"]]
    ann = a.groupby("facility_id").agg(total=("water_scarcity_l_eq", "sum"),
                                       onsite=("water_scarcity_onsite_l_eq", "sum")).reset_index().merge(gw, on="facility_id")
    out = {}
    for col in ("onsite", "total"):
        by = ann.groupby("gw_category")[col].sum() / ann[col].sum() * 100
        out[col] = {"pct_over_exploited": round(float(by.get("over-exploited", 0)), 1),
                    "pct_semicritical_or_worse": round(float(sum(by.get(c, 0) for c in ("over-exploited", "critical", "semi-critical"))), 1)}
    out["note"] = ("onsite = the water a facility draws locally (the defensible aquifer claim); total also counts "
                   "power-plant water drawn in other basins (the round-5 63% / 81%)")
    return out


def india_siting(a: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Re-score the frozen round-5 India candidate cells with the fixes: carbon at the consumption CI
    the account charges, ZLD at 18.4% (Rajasthan), and basin pressure against the corrected existing
    on-site load. Same criteria and minimax regret as decisions/siting.py."""
    from dcfootprint.account.energy import month_hours
    from dcfootprint.counterfactual.levers import zld_saving_fraction
    from dcfootprint.decisions.siting import CRITERIA, _minimax
    from dcfootprint.io.electricitymaps import STATE_TO_EM
    from dcfootprint.project.recharge import basin_budgets
    from dcfootprint.settings import params
    p = params()
    sc = p["siting"]
    t = sc["new_facility_type"]
    S, yr = float(sc["new_facility_mw"]), int(p["region_config"]["India"]["account_year"])
    util = p["energy"]["utilisation"]["by_facility_type"][t]["default"]
    pue = p["energy"]["pue"]["by_facility_type"][t]["default"]
    wue = float(p["water_onsite"]["wue_default"]["default"])
    zld_old, zld_new = 0.92, zld_saving_fraction(p)
    e_it = month_hours(yr).assign(e=lambda d: S * util * d["hours"]).set_index("month")["e"]

    cells = pd.read_csv(R5 / "q1_siting.csv")
    cf = pd.read_parquet(R5 / "outputs" / "interim" / "basin_cf_monthly.parquet").pivot(index="basin_id", columns="month", values="cf")
    s1 = (cf.reindex(cells["basin_id"]).to_numpy() * (e_it.to_numpy() * 1000 * wue)).sum(axis=1)       # L
    zld = np.where(cells["legal_flags"].fillna("").str.contains("zld"), 1.0, 0.0)
    re = cells["legal_flags"].fillna("").str.extract(r"re>=(\d+)%")[0].astype(float).fillna(0).to_numpy() / 100
    s2_pue = cells["new_scarcity_m3eq"].to_numpy() * 1000 - s1 * (1 - zld * zld_old)
    cells["new_scarcity_m3eq"] = (s1 * (1 - zld * zld_new) + s2_pue) / 1000

    # consumption CI per regional grid and month, as the account charges it
    zci = (a[a["ci_source"] == "em_consumption"].assign(zone=lambda d: d["state"].map(STATE_TO_EM))
           .groupby(["zone", "month"])["ci_gco2_per_kwh"].first().unstack())
    zone = cells["state"].map(STATE_TO_EM)
    have = zone.isin(zci.index).to_numpy()
    c_cons = np.array([(zci.loc[z].to_numpy() * e_it.to_numpy()).sum() / 1000 if ok else np.nan
                       for z, ok in zip(zone, have)])
    cells["ci_mean_generation_r5"] = cells["ci_mean"]
    cells["ci_mean"] = np.where(have, [zci.loc[z].mean() if ok else np.nan for z, ok in zip(zone, have)], cells["ci_mean"])
    cells["new_carbon_tco2"] = np.where(have, c_cons * pue * (1 - re), cells["new_carbon_tco2"])
    cells["carbon_basis"] = np.where(have, "consumption (Electricity Maps regional)", "generation (zone not in account)")

    # basin pressure: peak-queue rise from the new facility, against the corrected existing load
    exist = (a.assign(basin_id=a["basin_id"].astype("int64")).groupby(["basin_id", "month"])["water_onsite_l"].sum())
    bud = basin_budgets(cells["basin_id"].unique(), 1.0).set_index(["basin_id", "month"])["budget_l"]
    new_m = e_it * 1000 * wue

    def peak(bid, extra):
        D, pk = 0.0, 0.0
        for y in range(2):
            for m in range(1, 13):
                D = max(D + exist.get((bid, m), 0.0) + extra * new_m[m] - bud.get((bid, m), 0.0), 0.0)
                if y == 1:
                    pk = max(pk, D)
        return pk
    cells["basin_pressure_m3"] = [(peak(b, 1 - z * zld_new) - peak(b, 0.0)) / 1000 for b, z in zip(cells["basin_id"], zld)]

    old = cells[["state", "basin_id", "rank", "max_regret"]].rename(columns={"rank": "rank_r5", "max_regret": "max_regret_r5"})
    keep = [c for c in cells.columns if c not in ("rank", "max_regret", "mean_rank", "rank_p10", "rank_p90",
                                                  "rank_nolegal", "recommendation")]
    n_w = int(sc["weight_samples"])
    # without legal effects: undo the renewable-share cut on carbon and the ZLD cut on water
    nl = cells[keep].copy()
    nl["new_carbon_tco2"] = nl["new_carbon_tco2"] / (1 - re)
    nl["new_scarcity_m3eq"] = nl["new_scarcity_m3eq"] + s1 * zld * zld_new / 1000
    nl["basin_pressure_m3"] = [(peak(b, 1.0) - peak(b, 0.0)) / 1000 for b in nl["basin_id"]]
    nl = _minimax(nl, n_w, 0).reset_index(drop=True)
    nl["rank_nolegal"] = np.arange(1, len(nl) + 1)
    r = _minimax(cells[keep], n_w, 0).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    r = r.merge(old, on=["state", "basin_id"], how="left").merge(nl[["state", "basin_id", "rank_nolegal"]],
                                                                 on=["state", "basin_id"], how="left")
    top = r.head(10)
    ex = r[r["n_existing_dc"] > 0]
    summ = {"n_cells": len(r), "criteria": CRITERIA,
            "top10_states_r5": pd.read_csv(R5 / "q1_siting.csv").head(10)["state"].value_counts().to_dict(),
            "top10_states_r7": top["state"].value_counts().to_dict(),
            "top10_states_r7_without_legal_effects": nl.head(10)["state"].value_counts().to_dict(),
            "top10_overlap_with_r5": len(set(zip(top["state"], top["basin_id"]))
                                             & set(zip(old.nsmallest(10, "rank_r5")["state"], old.nsmallest(10, "rank_r5")["basin_id"]))),
            "spearman_rank_r5_vs_r7": round(float(r[["rank", "rank_r5"]].corr("spearman").iloc[0, 1]), 3),
            "existing_dc_cells": ex[["state", "basin_id", "n_existing_dc", "rank", "rank_r5"]].to_dict("records"),
            "top_regret_spread": round(float(top["max_regret"].max() - top["max_regret"].min()), 3)}
    return r, summ


def _scorecard(a: pd.DataFrame, out: Path, gw: pd.DataFrame | None = None) -> dict:
    from dcfootprint.decisions.scorecard import scorecard
    sc = scorecard(a)
    if gw is not None:                                       # CGWB matches from round 5 (data/cgwb is not in git)
        sc = sc.drop(columns=[c for c in gw.columns if c != "facility_id" and c in sc.columns]).merge(gw, on="facility_id", how="left")
    sc.to_csv(out / "q2_scorecard.csv", index=False)
    f = sc[sc["harm_flag"]]
    key = "state" if "state" in f.columns else "facility_id"
    return {"n_flagged": len(f), "n_facilities": len(sc),
            "flagged_by_state": f[key].value_counts().to_dict(),
            "recommended_lever": f["recommended_lever"].value_counts().to_dict()}


def india(steps: list[dict], skip_mc: bool) -> dict:
    from dcfootprint.counterfactual.levers import rank_lever_savings
    from dcfootprint.uncertainty.monte_carlo import monte_carlo
    from dcfootprint.uncertainty.routing_mc import routing_monte_carlo
    from dcfootprint.validation import reported as rv
    out = OUT / "india"
    out.mkdir(parents=True, exist_ok=True)
    a, r6_cap, totals = india_account()
    a.to_parquet(out / "account_india.parquet")
    _json(out / "account_totals.json", totals)

    r6 = pd.read_csv(R6 / "headline_steps.csv")
    steps += _headline(pd.read_csv(R5 / "routing_comparison.csv").assign(legal=lambda d: d["legal"].astype(str) == "True"),
                       "India", "0 round 5 as published")
    steps += [{**r, "step": "1 round 6 (PR #1)"} for r in
              r6[(r6["region"] == "India") & r6["step"].str.startswith("2")].to_dict("records")]
    steps += _headline(_routing(r6_cap), "India", "2 + capacity basis")
    t = _routing(a)
    t.to_csv(out / "routing_comparison.csv", index=False)
    steps += _headline(t, "India", "3 + WUE clamp + ZLD physics (round 7)")
    gb = pd.read_csv(GB / "routing_comparison.csv").assign(legal=lambda d: d["legal"].astype(str) == "True")
    steps += _headline(gb, "India", "consistency: generation basis for carbon AND water (round 5 code)")

    rank_lever_savings(a).to_csv(out / "lever_savings.csv", index=False)
    _json(out / "uncertainty.json", monte_carlo(a))
    v = rv.compare(a)
    v.to_csv(out / "validation_reported.csv", index=False)
    _json(out / "validation_summary.json", rv.summary(v))
    _json(out / "groundwater_equity.json", india_equity(a))
    gw = pd.read_csv(R5 / "q2_scorecard.csv")[["facility_id", "gw_stage_pct", "gw_stage_source", "gw_category"]]
    _json(out / "q2_scorecard_summary.json", _scorecard(a, out, gw))
    sit, ss = india_siting(a)
    sit.to_csv(out / "q1_siting_rerank.csv", index=False)
    _json(out / "q1_siting_rerank_summary.json", ss)
    if not skip_mc:
        draws, summ = routing_monte_carlo(a)
        draws.to_csv(out / "routing_mc_draws.csv", index=False)
        _json(out / "routing_mc_summary.json", summ)
    return _queue_diagnostics(t)


# ----------------------------------------------------------------------------- US, EU
def us(steps: list[dict], skip_mc: bool) -> dict:
    from dcfootprint.account.build import facility_month_account
    from dcfootprint.counterfactual.levers import rank_lever_savings
    from dcfootprint.uncertainty.monte_carlo import monte_carlo
    from dcfootprint.uncertainty.routing_mc import routing_monte_carlo
    from dcfootprint.validation import reported as rv
    out = OUT / "us"
    out.mkdir(parents=True, exist_ok=True)
    acct = facility_month_account(region="US")              # eia_egrid + clamped climate WUE
    acct.to_parquet(out / "account_us.parquet")
    r6 = pd.read_parquet(R6 / "us" / "account_us.parquet")
    _json(out / "account_totals.json", {"round6_published": _totals(r6), "round7": _totals(acct)})
    r6h = pd.read_csv(R6 / "headline_steps.csv")
    steps += _headline(pd.read_csv(R5 / "us" / "routing_comparison.csv").assign(legal=lambda d: d["legal"].astype(str) == "True"),
                       "US", "0 round 5 as published")
    steps += [{**r, "step": "1 round 6 (PR #1)"} for r in
              r6h[(r6h["region"] == "US") & r6h["step"].str.startswith("2")].to_dict("records")]
    t = _routing(acct)
    t.to_csv(out / "routing_comparison.csv", index=False)
    steps += _headline(t, "US", "3 + WUE clamp + ZLD physics (round 7)")
    rank_lever_savings(acct).to_csv(out / "lever_savings.csv", index=False)
    _json(out / "q2_scorecard_summary.json", _scorecard(acct, out))
    _json(out / "uncertainty.json", monte_carlo(acct))
    v = rv.compare(acct)
    v.to_csv(out / "validation_reported.csv", index=False)
    _json(out / "validation_summary.json", rv.summary(v))
    if not skip_mc:
        draws, summ = routing_monte_carlo(acct)
        draws.to_csv(out / "routing_mc_draws.csv", index=False)
        _json(out / "routing_mc_summary.json", summ)
    return _queue_diagnostics(t)


def eu() -> None:
    from dcfootprint.counterfactual.levers import rank_lever_savings
    out = OUT / "eu"
    out.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(R5 / "outputs" / "account_eu.parquet")
    rank_lever_savings(a).to_csv(out / "lever_savings.csv", index=False)
    _json(out / "q2_scorecard_summary.json", _scorecard(a, out))


if __name__ == "__main__":
    from dcfootprint.settings import write_run_manifest
    skip_mc = "--skip-routing-mc" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    write_run_manifest(OUT, {"entry_point": "experiments/robustness_round7.py", "skip_routing_mc": skip_mc})
    steps: list[dict] = []
    diag = {}
    for name, fn in (("India", india), ("US", us)):
        t0 = time.time()
        diag[name] = fn(steps, skip_mc)
        print(f"{name} done in {time.time() - t0:.0f} s")
    eu()
    _json(OUT / "routing_queue_diagnostics.json", diag)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import forecast_significance
    forecast_significance.main()
    pd.DataFrame(steps).round(3).to_csv(OUT / "headline_steps.csv", index=False)
    print(pd.DataFrame(steps).round(2).to_string(index=False))
