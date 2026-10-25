"""Every boundary contract must accept the canonical data and reject a deliberately broken copy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors as pe
import pytest

from dcfootprint.validation import schemas

R3 = Path(__file__).resolve().parents[2] / "dcfootprint" / "results" / "round5_final"


def _raises(schema, df):
    with pytest.raises((pe.SchemaError, pe.SchemaErrors)):
        schema.validate(df)


def test_ember_zone_month():
    ok = pd.DataFrame({"zone_id": ["A", "A"], "month": [1, 2], "ci_gco2_per_kwh": [500.0, 510.0],
                       "ci_source": ["ember_state", "ember_state"]})
    schemas.EmberZoneMonth.validate(ok)
    _raises(schemas.EmberZoneMonth, pd.concat([ok, ok.iloc[[0]]]))             # duplicate zone-month
    _raises(schemas.EmberZoneMonth, ok.assign(ci_gco2_per_kwh=[-1.0, 5.0]))     # negative CI


def test_fuel_shares_sum_to_one():
    ok = pd.DataFrame({"zone_id": ["A"] * 2, "month": [1, 1], "fuel": ["Coal", "Solar"], "share": [0.7, 0.3]})
    schemas.FuelShares.validate(ok)
    _raises(schemas.FuelShares, ok.assign(share=[0.7, 0.2]))


def test_basin_budget_needs_twelve_months():
    ok = pd.DataFrame({"basin_id": [1] * 12, "month": range(1, 13), "budget_l": [1.0] * 12})
    schemas.BasinBudget.validate(ok)
    _raises(schemas.BasinBudget, ok.iloc[:11])
    _raises(schemas.BasinBudget, ok.assign(budget_l=-1.0))


def test_account_contract_on_canonical_and_broken():
    a = pd.read_parquet(R3 / "outputs" / "account_facility_month.parquet")
    cols = [c for c in schemas.FacilityMonthAccount.to_schema().columns]
    schemas.FacilityMonthAccount.validate(a[cols])
    _raises(schemas.FacilityMonthAccount, pd.concat([a[cols], a[cols].iloc[[0]]]))           # not facility-month unique
    b = a[cols].copy(); b.loc[0, "water_scarcity_l_eq"] *= 2
    _raises(schemas.FacilityMonthAccount, b)                                                   # scopes no longer sum


@pytest.mark.parametrize("schema,file", [("LeverSavings", "lever_savings.csv"),
                                         ("RoutingComparison", "routing_comparison.csv"),
                                         ("Q2Scorecard", "q2_scorecard.csv"),
                                         ("Q1Siting", "q1_siting.csv"),
                                         ("ForecastCI", "forecast_ci.csv")])
def test_output_contracts_accept_canonical(schema, file):
    getattr(schemas, schema).validate(pd.read_csv(R3 / file))


def test_routing_contract_rejects_oracle_above_lyapunov():
    r = pd.read_csv(R3 / "routing_comparison.csv")
    r.loc[r["policy"] == "lyapunov", "penalty_gap_pct_vs_oracle"] = -5.0
    _raises(schemas.RoutingComparison, r)


def test_siting_contract_rejects_duplicate_cell():
    q = pd.read_csv(R3 / "q1_siting.csv")
    q.loc[1, ["state", "basin_id"]] = q.loc[0, ["state", "basin_id"]].values
    _raises(schemas.Q1Siting, q)
