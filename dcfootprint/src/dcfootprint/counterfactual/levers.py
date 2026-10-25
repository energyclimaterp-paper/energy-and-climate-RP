"""L7 levers — counterfactual policy levers (architecture v2 L7 "ranked levers").

(Named "counterfactual" in the package; it is NOT architecture L4, which is spatial
coupling — see geo/incidence.py.)

Each lever is a *parameter transform* on the present account; we recompute the affected
carbon/water and report the delta vs baseline. Deterministic scenario arithmetic — NOT
workload routing. "Which lever pays" is ranked by the primary metric: scarcity-weighted
water saved per year. Scope matters: a scope-1 saving is weighted by the facility-basin
CF, a scope-2 saving by the generation-basin CF (cf_grid_eff).

Levers (parameters from config/parameters.yaml):
  coastal_seawater_siting : WUE -> ~0.1 for coastal facilities (scope-1 only)
  zero_liquid_discharge   : W_onsite -> W_onsite*(1-recycle), recycle~0.92 (scope-1 ONLY:
                            on-site recycling cannot touch power-plant water)
  efficiency_standard     : PUE<=1.3 (carbon + scope-2 water down) & WUE<=0.7 (scope-1 down)
  mandatory_disclosure    : transparency lever — reveals the currently-unreported footprint
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

# coastal DC hubs (seawater/WSAC eligible) — city-level proxy, flagged
COASTAL = {"mumbai", "navi mumbai", "chennai", "visakhapatnam", "kolkata", "kochi", "mangalore", "surat"}
_BASELINE_DISCLOSURE = 0.30   # <1/3 of operators measure water (Mytton / water-governance)


def _load_account() -> pd.DataFrame:
    p = _repo_root() / "dcfootprint" / "outputs" / "account_facility_month.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from dcfootprint.account import build
    return build.facility_month_account()


def _backout(acct: pd.DataFrame) -> pd.DataFrame:
    """Recover per-row PUE, WUE, EWIF, CI from the account so levers need no re-load."""
    df = acct.copy()
    df["pue"] = df["e_grid_mwh"] / df["e_it_mwh"]
    df["wue_l_per_kwh"] = df["water_onsite_l"] / (df["e_it_mwh"] * 1000.0)
    df["ewif_l_per_mwh"] = np.where(df["e_grid_mwh"] > 0, df["water_grid_l"] / df["e_grid_mwh"], 0.0)
    df["ef_tco2_per_mwh"] = np.where(df["e_grid_mwh"] > 0, df["carbon_tco2"] / df["e_grid_mwh"], 0.0)
    return df


def lever_deltas(a: pd.DataFrame, lv: dict) -> dict[str, pd.DataFrame]:
    """Per facility-month savings for each reduction lever:
    {lever: DataFrame[d_onsite_l, d_grid_l, d_carbon_t]} (all >= 0, same index as `a`)."""
    zero = np.zeros(len(a))
    out = {}

    new_wue = float(lv["coastal_seawater_siting"]["new_wue"])
    coastal = a["city"].fillna("").str.lower().isin(COASTAL).values
    d = np.where(coastal & (a["wue_l_per_kwh"] > new_wue), (a["wue_l_per_kwh"] - new_wue) * a["e_it_mwh"] * 1000.0, 0.0)
    out["coastal_seawater_siting"] = pd.DataFrame({"d_onsite_l": d, "d_grid_l": zero, "d_carbon_t": zero}, index=a.index)

    rec = float(lv["zero_liquid_discharge"]["recycle_fraction"])
    out["zero_liquid_discharge"] = pd.DataFrame(
        {"d_onsite_l": a["water_onsite_l"].values * rec, "d_grid_l": zero, "d_carbon_t": zero}, index=a.index)

    pue_cap, wue_cap = float(lv["efficiency_standard"]["pue_cap"]), float(lv["efficiency_standard"]["wue_cap"])
    d_egrid = a["e_it_mwh"] * (a["pue"] - np.minimum(a["pue"], pue_cap))          # MWh grid avoided
    d_on = np.where(a["wue_l_per_kwh"] > wue_cap, (a["wue_l_per_kwh"] - wue_cap) * a["e_it_mwh"] * 1000.0, 0.0)
    out["efficiency_standard"] = pd.DataFrame(
        {"d_onsite_l": d_on, "d_grid_l": (d_egrid * a["ewif_l_per_mwh"]).values,
         "d_carbon_t": (d_egrid * a["ef_tco2_per_mwh"]).values}, index=a.index)
    return out


def _scarcity(d: pd.DataFrame, a: pd.DataFrame) -> pd.Series:
    """Scarcity-weighted litres-eq saved: scope-1 at the facility basin, scope-2 at the generation basins."""
    return d["d_onsite_l"] * a["cf"] + d["d_grid_l"] * a["cf_grid_eff"]


def rank_lever_savings(account: pd.DataFrame | None = None) -> pd.DataFrame:
    params = _cfg_params()
    lv = params["levers"]
    a = _backout(_load_account() if account is None else account)
    notes = {
        "coastal_seawater_siting": f"WUE->{lv['coastal_seawater_siting']['new_wue']} at coastal hubs (scope-1); "
                                   f"{int(a['city'].fillna('').str.lower().isin(COASTAL).sum() / 12)} facilities eligible",
        "zero_liquid_discharge": f"recycle {float(lv['zero_liquid_discharge']['recycle_fraction']):.0%} of ON-SITE (scope-1) water, all facilities",
        "efficiency_standard": f"PUE<={lv['efficiency_standard']['pue_cap']}, WUE<={lv['efficiency_standard']['wue_cap']} "
                               "(water+carbon; liquid-cooling tradeoff bounded by caps)",
    }
    rows = []
    for name, d in lever_deltas(a, lv).items():
        rows.append({"lever": name, "type": "reduction",
                     "water_phys_saved_m3_yr": float((d["d_onsite_l"] + d["d_grid_l"]).sum() / 1000.0),
                     "water_onsite_saved_m3_yr": float(d["d_onsite_l"].sum() / 1000.0),
                     "water_scarcity_saved_m3eq_yr": float(_scarcity(d, a).sum() / 1000.0),
                     "carbon_saved_tco2_yr": float(d["d_carbon_t"].sum()),
                     "note": notes[name]})

    revealed = float((a["water_phys_l"].sum() / 1000.0) * (1 - _BASELINE_DISCLOSURE))
    rows.append({"lever": "mandatory_disclosure", "type": "transparency",
                 "water_phys_saved_m3_yr": 0.0, "water_onsite_saved_m3_yr": 0.0,
                 "water_scarcity_saved_m3eq_yr": 0.0, "carbon_saved_tco2_yr": 0.0,
                 "note": f"reveals ~{revealed:,.0f} m3/yr currently unreported (baseline disclosure ~{_BASELINE_DISCLOSURE:.0%})"})

    out = pd.DataFrame(rows)
    base_scarcity = float(a["water_scarcity_l_eq"].sum() / 1000.0)
    base_carbon = float(a["carbon_tco2"].sum())
    out["pct_of_scarcity_baseline"] = (out["water_scarcity_saved_m3eq_yr"] / base_scarcity * 100).round(1)
    out["pct_of_carbon_baseline"] = (out["carbon_saved_tco2_yr"] / base_carbon * 100).round(1)
    return out.sort_values("water_scarcity_saved_m3eq_yr", ascending=False).reset_index(drop=True)


def facility_lever_savings(account: pd.DataFrame | None = None) -> pd.DataFrame:
    """[facility_id, lever, scarcity_saved_m3eq_yr, carbon_saved_tco2_yr] — per-facility
    savings, used by the Q2 scorecard to recommend the lever that pays most for that facility."""
    params = _cfg_params()
    a = _backout(_load_account() if account is None else account)
    rows = []
    for name, d in lever_deltas(a, params["levers"]).items():
        s = pd.DataFrame({"facility_id": a["facility_id"], "sc": _scarcity(d, a) / 1000.0, "c": d["d_carbon_t"]})
        g = s.groupby("facility_id", as_index=False).agg(scarcity_saved_m3eq_yr=("sc", "sum"), carbon_saved_tco2_yr=("c", "sum"))
        rows.append(g.assign(lever=name))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    res = rank_lever_savings()
    resdir = _repo_root() / "dcfootprint" / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(resdir / "lever_savings.csv", index=False)
    show = res[["lever", "type", "water_scarcity_saved_m3eq_yr", "water_phys_saved_m3_yr",
                "carbon_saved_tco2_yr", "pct_of_scarcity_baseline", "pct_of_carbon_baseline"]]
    print("WHICH LEVER PAYS (ranked by scarcity-weighted water saved):\n")
    print(show.to_string(index=False))
    print("\nnotes:")
    for _, r in res.iterrows():
        print(f"  - {r['lever']}: {r['note']}")
    print(f"\nwrote {resdir/'lever_savings.csv'}")
