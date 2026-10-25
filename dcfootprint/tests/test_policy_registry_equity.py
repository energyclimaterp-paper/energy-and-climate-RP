"""L5 policy registry (policy/registry.py, policy/gap.py) and the C7 groundwater-equity overlay
(decisions/equity.py). The equity tests use a synthetic account and a stubbed CGWB table, so they run
without data/cgwb/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ registry / gap
def test_binding_rules_match_legal_constraints_including_citation():
    # every binding rule in the registry must equal its row in legal_constraints.csv -- including the
    # source, which the first registry version dropped for the 5 rules attached to corpus documents
    from dcfootprint.policy import gap
    legal = pd.read_csv(ROOT / "dcfootprint" / "config" / "legal_constraints.csv").fillna("")
    hard = gap.HARD_CONSTRAINTS.fillna("")
    m = legal.merge(hard, on=["jurisdiction", "region", "rule"], how="outer", suffixes=("_cfg", "_reg"), indicator=True)
    assert (m["_merge"] == "both").all(), m[m["_merge"] != "both"][["region", "rule", "_merge"]]
    for c in ("type", "effect", "param", "source"):
        assert (m[f"{c}_cfg"].astype(str) == m[f"{c}_reg"].astype(str)).all(), c
    assert (hard["source"].astype(str).str.strip() != "").all()


def test_region_effects_equal_a_direct_reading_of_the_config():
    from dcfootprint.policy import gap
    legal = pd.read_csv(ROOT / "dcfootprint" / "config" / "legal_constraints.csv")
    for region in legal["region"].unique():
        rows = legal[(legal["region"] == region) & (legal["type"] == "hard") & legal["param"].notna()]
        assert gap.region_effects(region) == {r["effect"]: float(r["param"]) for _, r in rows.iterrows()}, region
    assert gap.region_effects("Nowhere") == {}


def test_four_axis_matrix_is_derived_from_in_force_mandates():
    from dcfootprint.policy import gap
    from dcfootprint.policy.registry import AXES, JURISDICTIONS
    m, reg = gap.four_axis_matrix(), gap.registry()
    assert list(m["axis"]) == AXES and set(JURISDICTIONS) <= set(m.columns)
    assert m[JURISDICTIONS].isin(["Yes", "No"]).all().all()
    for _, row in m.iterrows():
        for j in JURISDICTIONS:
            backed = ((reg["jurisdiction"] == j) & reg["in_force"].astype(bool)
                      & reg["mandates_axis"].fillna("").str.contains(row["axis"])).any()
            assert (row[j] == "Yes") == bool(backed), (row["axis"], j)


def test_gap_overlay_blind_spot_follows_the_matrix():
    from dcfootprint.policy import gap
    acct = pd.DataFrame({"facility_id": ["a", "a", "b"], "region": "India",
                         "water_scarcity_l_eq": [1000.0, 2000.0, 3000.0], "carbon_tco2": [1.0, 2.0, 3.0]})
    ov = gap.gap_overlay(acct)
    assert ov["n_facilities"] == 2 and ov["total_scarcity_m3eq"] == 6.0
    mandated = (gap.four_axis_matrix()["India"] == "Yes").any()
    assert ov["pct_burden_in_blind_spot"] == (0.0 if mandated else 100.0)


# ------------------------------------------------------------------ equity (C7)
@pytest.mark.parametrize("stage,expected", [(100.1, "over-exploited"), (100.0, "critical"), (90.0, "critical"),
                                            (89.9, "semi-critical"), (70.0, "semi-critical"), (69.9, "safe"),
                                            (float("nan"), "unknown")])
def test_cgwb_categories(stage, expected):
    from dcfootprint.decisions.equity import categorise
    assert categorise(stage) == expected


def test_city_names_normalise_to_cgwb_keys():
    from dcfootprint.decisions.equity import _CITY_ALIASES, _norm
    got = _norm(pd.Series(["Bengaluru (Urban)", "Navi-Mumbai", "Chennai District"])).replace(_CITY_ALIASES)
    assert list(got) == ["bangalore", "thane", "chennai"]


def test_equity_overlay_shares_on_a_synthetic_account(monkeypatch):
    from dcfootprint.decisions import equity
    monkeypatch.setattr(equity, "load_cgwb_stage",
                        lambda: ({"bangalore": 187.0, "chennai": 95.0}, {"maharashtra": 60.0, "telangana": 75.0}))
    acct = pd.DataFrame({  # litres; 3 facilities x 2 months
        "facility_id": ["f1", "f1", "f2", "f2", "f3", "f3"], "operator": ["A", "A", "B", "B", "C", "C"],
        "city": ["Bengaluru", "Bengaluru", "Chennai", "Chennai", "Pune", "Pune"],
        "state": ["Karnataka", "Karnataka", "Tamil Nadu", "Tamil Nadu", "Maharashtra", "Maharashtra"],
        "water_scarcity_l_eq": [300e3, 300e3, 100e3, 100e3, 100e3, 100e3]})
    df, s = equity.equity_overlay(acct)
    by = df.set_index("facility_id")
    assert by.loc["f1", "gw_category"] == "over-exploited" and by.loc["f1", "gw_stage_source"] == "city"
    assert by.loc["f2", "gw_category"] == "critical"
    assert by.loc["f3", "gw_category"] == "safe" and by.loc["f3", "gw_stage_source"] == "state"   # state fallback
    assert s["total_scarcity_m3eq_yr"] == 1000.0                                                    # m3-eq
    assert s["pct_burden_over_exploited"] == 60.0 and s["pct_burden_critical_or_worse"] == 80.0
    assert s["pct_burden_semicritical_or_worse"] == 80.0


@pytest.mark.skipif(not (ROOT / "data" / "cgwb").exists(), reason="CGWB files not on disk (data/cgwb/)")
def test_committed_equity_exhibit_reproduces():
    from dcfootprint.decisions import equity
    acct = pd.read_parquet(ROOT / "dcfootprint" / "results" / "round5_final" / "outputs" / "account_facility_month.parquet")
    df, _ = equity.equity_overlay(acct)
    ref = pd.read_csv(ROOT / "dcfootprint" / "results" / "round5_final" / "q2_equity_groundwater.csv")
    assert np.isclose(df["scarcity_m3eq_yr"].sum(), ref["scarcity_m3eq_yr"].sum(), rtol=1e-9)


def test_q2_groundwater_columns_degrade_without_cgwb(monkeypatch, tmp_path):
    # data/cgwb/ is not in git: Q2 must still run, with the groundwater columns empty
    from dcfootprint.decisions import equity
    def missing():
        raise FileNotFoundError(2, "No such file", str(tmp_path / "cgwb" / "x.csv"))
    monkeypatch.setattr(equity, "load_cgwb_stage", missing)
    df = pd.DataFrame({"city": ["Bengaluru"], "state": ["Karnataka"], "scarcity_m3eq_yr": [1.0]})
    out = equity.attach_gw_stage(df)
    assert out["gw_stage_pct"].isna().all() and (out["gw_category"] == "unknown").all()
    assert list(df.columns) == ["city", "state", "scarcity_m3eq_yr"]      # input not modified
