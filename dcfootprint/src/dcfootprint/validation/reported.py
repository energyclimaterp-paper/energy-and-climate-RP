"""Validation of the modelled account against operator-REPORTED figures.

config/reported_footprints.csv holds figures that operators publish (sustainability / ESG
reports), each with its source URL and PDF page. The model is aggregated to the same scope:
  fleet  every account facility whose operator matches `operator_match` (case-insensitive)
  site   the one account facility `facility_id`
and compared metric by metric:

  electricity_mwh                 model E_grid (facility electricity)
  it_energy_mwh                   model E_IT
  water_withdrawal_m3 / _consumption_m3   model on-site (scope-1) water
  pue, wue_l_per_kwh              model energy-weighted PUE / on-site water per IT kWh
  scope2_location_tco2 / grid_electricity_mwh   reported grid emission factor vs model CI
  capacity_mw                     coverage: model MW of the matched facilities vs reported MW

Two kinds of ratio are reported, model / reported:
  total     the totals as they stand (sensitive to how many facilities the account holds)
  per_mw    totals per MW of capacity, where the operator reports its capacity (removes coverage)
and two capacity-free intensity checks: on-site water per facility kWh, and the grid emission
factor. A ratio of 1 = agreement. Reporting periods are operator fiscal years in India (April to
March) against the calendar account year: the overlap is stated, not corrected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root

REFERENCE = Path("dcfootprint") / "config" / "reported_footprints.csv"


def load_reported(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else _repo_root() / REFERENCE
    r = pd.read_csv(p, dtype={"facility_id": "string"})
    need = {"region", "operator_match", "facility_id", "scope", "period", "period_start", "period_end",
            "metric", "value", "unit", "source_url", "pdf_page"}
    missing = need - set(r.columns)
    if missing:
        raise ValueError(f"reported_footprints.csv is missing columns {sorted(missing)}")
    if not r["scope"].isin(["fleet", "site"]).all():
        raise ValueError("scope must be fleet or site")
    if (r["scope"].eq("site") & r["facility_id"].isna()).any():
        raise ValueError("site rows need a facility_id")
    return r


def _select(account: pd.DataFrame, row) -> pd.DataFrame:
    if row["scope"] == "site":
        return account[account["facility_id"] == row["facility_id"]]
    return account[account["operator"].fillna("").str.lower().str.contains(str(row["operator_match"]).lower(), regex=False)]


def _model_metrics(a: pd.DataFrame) -> dict:
    if a.empty:
        return {}
    e_grid, e_it = a["e_grid_mwh"].sum(), a["e_it_mwh"].sum()
    w = a["water_onsite_l"].sum() / 1000.0
    return {"electricity_mwh": e_grid, "it_energy_mwh": e_it,
            "water_withdrawal_m3": w, "water_consumption_m3": w,
            "pue": e_grid / e_it if e_it > 0 else np.nan,
            "wue_l_per_kwh": w * 1000.0 / (e_it * 1000.0) if e_it > 0 else np.nan,
            "capacity_mw": a.drop_duplicates("facility_id")["capacity_mw"].sum(),
            "carbon_t": a["carbon_tco2"].sum(),
            "n_facilities": a["facility_id"].nunique()}


def _overlap_months(start: str, end: str, year: int) -> int:
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return int(sum(m.year == year for m in months))


def compare(account: pd.DataFrame, reported: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per reported figure that can be compared (metric x scope), plus derived intensity
    checks per scope: water per facility kWh and the grid emission factor."""
    rep = load_reported() if reported is None else reported
    region = str(account["region"].iloc[0])
    year = int(pd.to_datetime(account["date"]).dt.year.mode()[0])
    rep = rep[rep["region"] == region]
    rows = []
    for key, g in rep.groupby(["scope", "operator_match", rep["facility_id"].fillna(""), "period"], sort=False):
        first = g.iloc[0]
        sel = _select(account, first)
        m = _model_metrics(sel)
        vals = dict(zip(g["metric"], g["value"].astype(float)))
        src = dict(zip(g["metric"], g["source_url"] + " p." + g["pdf_page"].astype(str)))
        base = {"region": region, "scope": first["scope"], "operator": first["operator_match"],
                "facility_id": first["facility_id"] if first["scope"] == "site" else "",
                "period": first["period"], "months_overlap_with_account_year":
                _overlap_months(first["period_start"], first["period_end"], year),
                "n_model_facilities": m.get("n_facilities", 0)}
        if not m:
            rows.append({**base, "metric": "no matching facility in the account"})
            continue
        rep_mw = vals.get("capacity_mw")
        for metric, v in vals.items():
            if metric in ("capacity_mw", "scope2_location_tco2", "grid_electricity_mwh"):
                continue
            mv = m[metric]
            per_mw = (mv / m["capacity_mw"]) / (v / rep_mw) if rep_mw and metric not in ("pue", "wue_l_per_kwh") else np.nan
            rows.append({**base, "metric": metric, "reported": v, "model": mv,
                         "ratio_total": mv / v if v else np.nan, "ratio_per_mw": per_mw,
                         "model_mw": m["capacity_mw"], "reported_mw": rep_mw, "source": src[metric]})
        e = vals.get("electricity_mwh")
        # model on-site water is evaporative (WUE x IT energy): compare with consumption where reported
        w = vals.get("water_consumption_m3", vals.get("water_withdrawal_m3"))
        if e and w:                                                  # capacity-free: water per facility kWh
            rw, mw_ = w * 1000 / (e * 1000), m["water_withdrawal_m3"] * 1000 / (m["electricity_mwh"] * 1000)
            rows.append({**base, "metric": "onsite_water_l_per_facility_kwh", "reported": rw, "model": mw_,
                         "ratio_total": mw_ / rw, "ratio_per_mw": np.nan, "model_mw": m["capacity_mw"],
                         "reported_mw": rep_mw, "source": "derived from the water and electricity rows"})
        if vals.get("scope2_location_tco2") and vals.get("grid_electricity_mwh"):  # grid emission factor
            ref = vals["scope2_location_tco2"] / vals["grid_electricity_mwh"]
            mod = m["carbon_t"] / m["electricity_mwh"]
            rows.append({**base, "metric": "grid_emission_factor_t_per_mwh", "reported": ref, "model": mod,
                         "ratio_total": mod / ref, "ratio_per_mw": np.nan, "model_mw": m["capacity_mw"],
                         "reported_mw": rep_mw, "source": src["scope2_location_tco2"]})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> dict:
    """Median model/reported ratio per metric family (geometric, so 2x and 0.5x balance)."""
    out = {}
    for metric, g in table.dropna(subset=["ratio_total"]).groupby("metric"):
        r = g["ratio_total"].to_numpy(float)
        out[metric] = {"n": len(r), "geomean_ratio": round(float(np.exp(np.mean(np.log(r)))), 3),
                       "min": round(float(r.min()), 3), "max": round(float(r.max()), 3)}
    return out
