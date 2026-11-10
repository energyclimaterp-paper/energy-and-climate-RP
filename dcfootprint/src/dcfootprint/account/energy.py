"""L2 — energy. E_IT = capacity_MW * utilisation * hours(month); E_grid = E_IT * PUE.
Utilisation & PUE from parameters.yaml (by facility type; PUE by operator where
disclosed). Facility type is inferred from the operator (hyperscaler vs colocation).

Capacity must be IT capacity. Rows whose `capacity_basis` is listed in
energy.facility_power_capacity_bases (the India list's `operational_total`: total facility
power, see its Notes & Method sheet) are converted with the facility's PUE,
C_IT = C_recorded / PUE, the same rule the US inventory applies (io/us_facilities.py).
`capacity_mw` is then the IT capacity and `capacity_recorded_mw` the stated figure.
Pure + vectorised; the caller validates the assembled account.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

# account columns that scale linearly with capacity (used to correct a frozen account)
EXTENSIVE_COLS = ("e_it_mwh", "e_grid_mwh", "carbon_tco2", "water_onsite_l", "water_grid_l", "water_phys_l",
                  "water_scarcity_onsite_l_eq", "water_scarcity_grid_l_eq", "water_scarcity_l_eq",
                  "water_grid_hydro_l", "water_scarcity_grid_hydro_l_eq", "capacity_mw")


def facility_power_mask(basis: pd.Series, params: dict) -> np.ndarray:
    """True where the stated capacity is total facility power rather than IT capacity."""
    bases = {str(b) for b in params["energy"].get("facility_power_capacity_bases", [])}
    return basis.astype("string").fillna("").isin(bases).to_numpy()

_HYPERSCALERS = ("microsoft", "google", "aws", "amazon", "meta", "facebook", "oracle")


def infer_facility_type(operator: str, operator_family: str) -> str:
    text = f"{operator} {operator_family}".lower()
    if any(h in text for h in _HYPERSCALERS):
        return "hyperscale"
    return "colocation"   # India market is colocation-dominant (fallback for 'unknown')


def month_hours(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"month": m, "hours": calendar.monthrange(year, m)[1] * 24} for m in range(1, 13)]
    )


def energy_account(fac: pd.DataFrame, params: dict, year: int) -> pd.DataFrame:
    """Facility x month energy. `fac` must have capacity_mw, operator[, operator_family]."""
    util_cfg = params["energy"]["utilisation"]["by_facility_type"]
    pue_cfg = params["energy"]["pue"]["by_facility_type"]
    pue_op = {k.lower(): v for k, v in params["energy"]["pue"].get("by_operator", {}).items()}

    f = fac.copy()
    fam = f["operator_family"] if "operator_family" in f.columns else pd.Series([""] * len(f), index=f.index)
    f["facility_type"] = [infer_facility_type(o, of) for o, of in zip(f["operator"].fillna(""), fam.fillna(""))]
    f["utilisation"] = f["facility_type"].map(lambda t: util_cfg.get(t, util_cfg["unknown"])["default"])
    f["pue"] = f["facility_type"].map(lambda t: pue_cfg.get(t, pue_cfg["unknown"])["default"])

    opl = f["operator"].fillna("").str.lower()
    for op, val in pue_op.items():                       # disclosed fleet PUE overrides type default
        f.loc[opl.str.contains(op, regex=False), "pue"] = val

    if "capacity_basis" in f.columns:                    # total facility power -> IT capacity
        f["capacity_recorded_mw"] = f.get("capacity_recorded_mw", f["capacity_mw"])
        fp = facility_power_mask(f["capacity_basis"], params)
        f.loc[fp, "capacity_mw"] = f.loc[fp, "capacity_mw"] / f.loc[fp, "pue"]

    hrs = month_hours(year)
    fm = f.merge(hrs, how="cross")
    fm["e_it_mwh"] = fm["capacity_mw"] * fm["utilisation"] * fm["hours"]
    fm["e_grid_mwh"] = fm["e_it_mwh"] * fm["pue"]
    return fm


def correct_capacity_basis(account: pd.DataFrame, basis_by_facility: pd.Series, params: dict) -> pd.DataFrame:
    """Apply the facility-power -> IT conversion to an account built without it (the frozen
    round-5 India account). Every capacity-driven column scales by 1/PUE on the affected rows,
    which is exact because the account is linear in capacity. `basis_by_facility` maps
    facility_id -> capacity_basis."""
    a = account.copy()
    basis = a["facility_id"].map(basis_by_facility)
    fp = facility_power_mask(basis, params)
    pue = a["e_grid_mwh"] / a["e_it_mwh"]
    scale = np.where(fp, 1.0 / pue, 1.0)
    a["capacity_recorded_mw"] = a["capacity_mw"]
    for c in EXTENSIVE_COLS:
        if c in a.columns:
            a[c] = a[c] * scale
    a["capacity_basis"] = basis
    return a
