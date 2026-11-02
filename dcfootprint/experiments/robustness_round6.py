"""Round 6 robustness run: validation, US plant-level grid data, routing Monte Carlo,
forecast value for routing, climate WUE and location sensitivity.

Writes dcfootprint/results/round6_robustness/ (the round-5 snapshot is read, never edited).

India starts from the frozen round-5 account (results/round5_final/outputs/), so it does not need
the GEM workbook or an Electricity Maps key; only scope-1 water is rebuilt (climate WUE), which
does not touch carbon or scope-2. The US account is rebuilt from raw data on eGRID + EIA.
Inputs needed in data/: ember (India + US subnational), aware, gadm (USA), compute_atlas,
egrid, eia (see data/README.md). NASA POWER wet-bulb is fetched once and cached in data/climate/.

Run from the repository root:
    PYTHONPATH=dcfootprint/src python dcfootprint/experiments/robustness_round6.py
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root

ROOT = _repo_root()
R5 = ROOT / "dcfootprint" / "results" / "round5_final"
OUT = ROOT / "dcfootprint" / "results" / "round6_robustness"


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n", encoding="utf-8")


def _totals(a: pd.DataFrame) -> dict:
    return {"n_facilities": int(a["facility_id"].nunique()),
            "electricity_twh": round(float(a["e_grid_mwh"].sum()) / 1e6, 3),
            "carbon_mt": round(float(a["carbon_tco2"].sum()) / 1e6, 3),
            "water_onsite_mm3": round(float(a["water_onsite_l"].sum()) / 1e9, 3),
            "water_grid_mm3": round(float(a["water_grid_l"].sum()) / 1e9, 3),
            "scarcity_mm3eq": round(float(a["water_scarcity_l_eq"].sum()) / 1e9, 2),
            "scarcity_onsite_mm3eq": round(float(a["water_scarcity_onsite_l_eq"].sum()) / 1e9, 2)}


def _routing(a: pd.DataFrame) -> pd.DataFrame:
    from dcfootprint.routing import lyapunov
    return pd.concat([lyapunov.compare(a, legal=True), lyapunov.compare(a, legal=False)], ignore_index=True)


def _headline(tab: pd.DataFrame, region: str, step: str) -> list[dict]:
    t = tab[tab["legal"]].set_index("policy")
    return [{"region": region, "step": step, "policy": p,
             "carbon_change_pct": -float(t.loc[p, "carbon_saving_pct_vs_static"]),
             "scarcity_change_pct": -float(t.loc[p, "scarcity_saving_pct_vs_static"])}
            for p in ("greedy", "lyapunov", "oracle")]


def india(steps: list[dict]) -> None:
    from dcfootprint.account.wue import facility_month_wue, recompute_onsite
    from dcfootprint.geo import join as gj
    from dcfootprint.geo.location_sensitivity import centroid_bounds
    from dcfootprint.routing.forecast_value import forecast_value
    from dcfootprint.settings import params
    from dcfootprint.uncertainty.routing_mc import routing_monte_carlo
    from dcfootprint.validation import reported as rv
    out = OUT / "india"
    out.mkdir(parents=True, exist_ok=True)
    r5 = pd.read_parquet(R5 / "outputs" / "account_facility_month.parquet")
    steps += _headline(pd.read_csv(R5 / "routing_comparison.csv"), "India", "0 round 5 as published")

    # 1. controller CI forecast on the charged (consumption) basis
    t1 = _routing(r5)
    t1.to_csv(out / "routing_comparison_r5_account.csv", index=False)
    steps += _headline(t1, "India", "1 basis-consistent CI forecast")

    # 2. climate-dependent on-site WUE (scope-1 only)
    fac = pd.read_parquet(R5 / "outputs" / "interim" / "facilities_geocoded.parquet")
    fac = fac[fac["facility_id"].isin(r5["facility_id"])]
    p = params()
    acct = recompute_onsite(r5, facility_month_wue(fac, p, 2024))
    _json(out / "account_totals.json", {"round5_constant_wue": _totals(r5), "round6_climate_wue": _totals(acct),
                                        "climate_wue_by_month": acct.groupby("month")["wue_l_per_kwh"].mean().round(3).to_dict()})
    acct.to_parquet(out / "account_india.parquet")
    t2 = _routing(acct)
    t2.to_csv(out / "routing_comparison.csv", index=False)
    steps += _headline(t2, "India", "2 + climate WUE (round 6 account)")

    # 3. routing Monte Carlo
    draws, summ = routing_monte_carlo(acct)
    draws.to_csv(out / "routing_mc_draws.csv", index=False)
    _json(out / "routing_mc_summary.json", summ)

    # 4. forecast value for routing
    forecast_value(acct).to_csv(out / "forecast_value.csv", index=False)

    # 5. validation against operator reports (both accounts, so the WUE change is visible)
    for name, a in (("round5", r5), ("round6", acct)):
        v = rv.compare(a)
        v.to_csv(out / f"validation_reported_{name}.csv", index=False)
        _json(out / f"validation_summary_{name}.json", rv.summary(v))

    # 6. centroid location sensitivity
    basins = gj.load_aware_basins()
    cf = gj.basin_monthly_cf(basins)
    bounds = {}
    for r in (10, 25, 50):
        tab, s = centroid_bounds(acct, fac, basins, cf, radius_km=r)
        bounds[f"{r}km"] = s
        if r == 25:
            tab.to_csv(out / "location_sensitivity_25km.csv", index=False)
    _json(out / "location_sensitivity.json", bounds)


def us(steps: list[dict]) -> None:
    from dcfootprint.account.build import facility_month_account
    from dcfootprint.account.wue import facility_month_wue, recompute_onsite
    from dcfootprint.io import us_facilities, us_grid
    from dcfootprint.settings import params
    from dcfootprint.uncertainty.routing_mc import routing_monte_carlo
    from dcfootprint.validation import reported as rv
    out = OUT / "us"
    out.mkdir(parents=True, exist_ok=True)
    us_grid.egrid_reproduction().round(4).to_csv(out / "egrid2023_reproduction.csv", index=False)
    steps += _headline(pd.read_csv(R5 / "us" / "routing_comparison.csv"), "US", "0 round 5 as published")

    acct = facility_month_account(region="US")                     # eia_egrid + climate WUE (config)
    fac = us_facilities.build()
    pc = copy.deepcopy(params())
    pc["water_onsite"]["wue_model"] = "constant"
    const = recompute_onsite(acct, facility_month_wue(fac[fac["facility_id"].isin(acct["facility_id"])], pc, 2024))
    r5 = pd.read_parquet(R5 / "outputs" / "account_us.parquet")
    _json(out / "account_totals.json", {"round5_ember_macknick": _totals(r5), "round6_eia_egrid_constant_wue": _totals(const),
                                        "round6_eia_egrid_climate_wue": _totals(acct),
                                        "meta": acct.attrs.get("meta", {})})
    acct.to_parquet(out / "account_us.parquet")

    t1 = _routing(const)
    steps += _headline(t1, "US", "1 eGRID + EIA plant data")
    t2 = _routing(acct)
    t2.to_csv(out / "routing_comparison.csv", index=False)
    steps += _headline(t2, "US", "2 + climate WUE (round 6 account)")

    draws, summ = routing_monte_carlo(acct)
    draws.to_csv(out / "routing_mc_draws.csv", index=False)
    _json(out / "routing_mc_summary.json", summ)

    for name, a in (("round5", r5), ("round6", acct)):
        v = rv.compare(a)
        v.to_csv(out / f"validation_reported_{name}.csv", index=False)
        _json(out / f"validation_summary_{name}.json", rv.summary(v))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []
    for name, fn in (("India", india), ("US", us)):
        t = time.time()
        fn(steps)
        print(f"{name} done in {time.time() - t:.0f} s")
    pd.DataFrame(steps).round(3).to_csv(OUT / "headline_steps.csv", index=False)
    print(pd.DataFrame(steps).round(2).to_string(index=False))
