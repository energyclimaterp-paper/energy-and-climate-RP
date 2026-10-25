"""L2 — carbon. Carbon = E_grid * CI(zone, month).
India: Ember zone-month CO2 intensity (grid.zone = state -> state generation mix, which is
generation-based, not consumption-based: stated caveat R2; grid.zone = national -> the pooled
India-total mix). CEA national-annual EF is a cross-check only (io/cea.py).
"""
from __future__ import annotations

import pandas as pd


def carbon_account(energy_df: pd.DataFrame, ci_zone_month: pd.DataFrame) -> pd.DataFrame:
    """Add ci_gco2_per_kwh + carbon_tco2. `energy_df` needs zone_id + month;
    `ci_zone_month` is [zone_id, month, ci_gco2_per_kwh, ci_source] (io/ember.zone_month_ci).
    gCO2/kWh == kgCO2/MWh, so tCO2 = MWh * ci / 1000."""
    df = energy_df.merge(ci_zone_month, on=["zone_id", "month"], how="left")
    if df["ci_gco2_per_kwh"].isna().any():
        bad = sorted(df.loc[df["ci_gco2_per_kwh"].isna(), "zone_id"].astype(str).unique())
        raise KeyError(f"no grid CI for zone(s) {bad}; add an alias or fall back to grid.zone: national")
    df["carbon_tco2"] = df["e_grid_mwh"] * df["ci_gco2_per_kwh"] / 1000.0
    return df
