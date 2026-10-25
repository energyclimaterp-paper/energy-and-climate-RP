"""L2 — energy. E_IT = capacity_MW * utilisation * hours(month); E_grid = E_IT * PUE.
Utilisation & PUE from parameters.yaml (by facility type; PUE by operator where
disclosed). Facility type is inferred from the operator (hyperscaler vs colocation).
Pure + vectorised; the caller validates the assembled account.
"""
from __future__ import annotations

import calendar

import pandas as pd

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

    hrs = month_hours(year)
    fm = f.merge(hrs, how="cross")
    fm["e_it_mwh"] = fm["capacity_mw"] * fm["utilisation"] * fm["hours"]
    fm["e_grid_mwh"] = fm["e_it_mwh"] * fm["pue"]
    return fm
