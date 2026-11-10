"""Round 7 fixes: capacity basis, ZLD physics, WUE clamp, Sobol estimators, groundwater scope,
registry kinds, the status-quo routing baseline, forecast significance helpers and the run manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = ROOT / "dcfootprint" / "results" / "round5_final" / "outputs" / "account_facility_month.parquet"


# ---------------------------------------------------------------- capacity basis
def _params(bases=("operational_total",)):
    return {"energy": {"facility_power_capacity_bases": list(bases),
                       "utilisation": {"by_facility_type": {"colocation": {"default": 0.5}, "unknown": {"default": 0.5}}},
                       "pue": {"by_facility_type": {"colocation": {"default": 1.6}, "unknown": {"default": 1.6}},
                               "by_operator": {}}}}


def test_facility_power_capacity_is_converted_to_it():
    from dcfootprint.account.energy import energy_account
    fac = pd.DataFrame({"facility_id": ["it", "tot", "crit"], "operator": ["A", "B", "C"],
                        "capacity_mw": [16.0, 16.0, 16.0],
                        "capacity_basis": ["operational_IT", "operational_total", "operational_critical"]})
    e = energy_account(fac, _params(), 2024).groupby("facility_id")[["e_it_mwh", "e_grid_mwh", "capacity_mw",
                                                                    "capacity_recorded_mw"]].agg("first")
    assert e.loc["tot", "capacity_mw"] == pytest.approx(10.0)              # 16 MW facility power / PUE 1.6
    assert e.loc["it", "capacity_mw"] == e.loc["crit", "capacity_mw"] == 16.0
    assert (e["capacity_recorded_mw"] == 16.0).all()
    # facility-power row: grid draw at full utilisation equals the stated facility power
    assert e.loc["tot", "e_grid_mwh"] == pytest.approx(16.0 * 0.5 * 744)


def test_correct_capacity_basis_scales_every_extensive_column():
    from dcfootprint.account.energy import EXTENSIVE_COLS, correct_capacity_basis
    a = pd.DataFrame({"facility_id": ["x", "y"], "e_it_mwh": [10.0, 10.0], "e_grid_mwh": [16.0, 16.0]})
    for c in EXTENSIVE_COLS:
        if c not in a:
            a[c] = 5.0
    out = correct_capacity_basis(a, pd.Series({"x": "operational_total", "y": "operational_IT"}), _params())
    for c in EXTENSIVE_COLS:
        assert out.loc[0, c] == pytest.approx(a.loc[0, c] / 1.6), c
        assert out.loc[1, c] == a.loc[1, c], c
    assert (out["capacity_recorded_mw"] == 5.0).all()


# ---------------------------------------------------------------- ZLD
def test_zld_saves_recovered_blowdown_only():
    from dcfootprint.counterfactual.levers import lever_deltas, zld_saving_fraction
    assert zld_saving_fraction(cycles=5, recovery=0.92) == pytest.approx(0.184)
    assert zld_saving_fraction(cycles=3, recovery=1.0) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        zld_saving_fraction(cycles=1, recovery=0.9)
    # mass balance: make-up = evaporation + blowdown, blowdown = E / (N - 1) -> blowdown / make-up = 1 / N
    E, N = 100.0, 5.0
    assert (E / (N - 1)) / (E + E / (N - 1)) == pytest.approx(1 / N)
    a = pd.DataFrame({"city": ["Pune"], "wue_l_per_kwh": [1.9], "e_it_mwh": [1.0], "water_onsite_l": [1900.0],
                      "pue": [1.6], "ewif_l_per_mwh": [0.0], "ef_tco2_per_mwh": [0.0]})
    lv = {"coastal_seawater_siting": {"new_wue": 0.1}, "efficiency_standard": {"pue_cap": 1.3, "wue_cap": 0.7},
          "zero_liquid_discharge": {"cycles_of_concentration": 5, "blowdown_recovery": 0.92}}
    d = lever_deltas(a, lv)["zero_liquid_discharge"]
    assert d["d_onsite_l"].iloc[0] == pytest.approx(1900 * 0.184)


def test_retired_recycle_fraction_is_rejected():
    from dcfootprint.settings import Levers
    with pytest.raises(ValueError, match="retired"):
        Levers.model_validate({"coastal_seawater_siting": {"new_wue": 0.1},
                               "efficiency_standard": {"pue_cap": 1.3, "wue_cap": 0.7},
                               "zero_liquid_discharge": {"recycle_fraction": 0.92, "applies_to": "scope1",
                                                         "cycles_of_concentration": 5, "blowdown_recovery": 0.92}})


# ---------------------------------------------------------------- WUE clamp
def test_gupta_wue_does_not_rise_in_cold_weather():
    from dcfootprint.account.wue import GUPTA_TW_MIN_F, gupta_cold_water_wue
    tmin_c = (GUPTA_TW_MIN_F - 32) * 5 / 9
    assert tmin_c == pytest.approx(9.3, abs=0.05)
    cold = gupta_cold_water_wue(np.linspace(-30, tmin_c, 50))
    np.testing.assert_allclose(cold, cold[-1])                              # flat at the vertex value
    warm = gupta_cold_water_wue(np.linspace(tmin_c, 35, 50))
    assert np.all(np.diff(warm) >= -1e-12)                                  # non-decreasing with wet-bulb


# ---------------------------------------------------------------- Sobol
def test_binned_first_order_is_bias_corrected():
    # y = x1 + x2: true S1 = 0.5, 0.5 and 0 for an irrelevant input. The raw between-bin share is
    # inflated by ~ (bins - 1) / N = 0.01 per input; averaged over seeds the corrected one is unbiased.
    from dcfootprint.uncertainty.monte_carlo import _first_order
    n, s_noise, s_x1 = 4000, [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        x1, x2, noise = rng.uniform(size=n), rng.uniform(size=n), rng.uniform(size=n)
        y = x1 + x2
        s_noise.append(_first_order(noise, y))
        s_x1.append(_first_order(x1, y))
    assert np.mean(s_noise) < 0.003                                         # raw estimator gives ~0.010
    assert np.mean(s_x1) == pytest.approx(0.5, abs=0.01)


def test_saltelli_recovers_known_indices():
    from dcfootprint.uncertainty.monte_carlo import sobol_saltelli
    # y = x1 + 2 x2 + x1 x3 with U(0,1) inputs: closed-form variances
    f = lambda x: x["x1"] + 2 * x["x2"] + x["x1"] * x["x3"]
    draw = lambda r, n: {k: r.uniform(size=n) for k in ("x1", "x2", "x3")}
    v1 = (1.5 ** 2) / 12                      # E[y|x1] = x1 (1 + 0.5)
    v2 = 4 / 12
    v3 = (0.5 ** 2) / 12                      # E[y|x3] = 0.5 x3 + const
    v13 = (1 / 12) * (1 / 12)                 # interaction (x1 - 1/2)(x3 - 1/2)
    V = v1 + v2 + v3 + v13
    s = sobol_saltelli(f, draw, ["x1", "x2", "x3"], 2 ** 15, seed=1, n_boot=50)
    assert s["x1"]["S1"] == pytest.approx(v1 / V, abs=0.03)
    assert s["x2"]["S1"] == pytest.approx(v2 / V, abs=0.03)
    assert s["x3"]["ST"] == pytest.approx((v3 + v13) / V, abs=0.01)
    assert s["x2"]["ST"] == pytest.approx(v2 / V, abs=0.01)


@pytest.mark.skipif(not ACCOUNT.exists(), reason="frozen account not on disk")
def test_account_sobol_first_order_sums_to_at_most_one():
    from dcfootprint.uncertainty.monte_carlo import monte_carlo
    u = monte_carlo(pd.read_parquet(ACCOUNT))
    s = u["scarcity_sobol"]
    assert s["target"] == "scarcity_m3eq_yr"
    assert s["sum_S1"] <= 1.0 + 0.03
    for v in s["indices"].values():
        assert v["S1_p05"] <= v["S1"] <= v["S1_p95"] and v["ST_p05"] <= v["ST"] <= v["ST_p95"]


# ---------------------------------------------------------------- groundwater scope
def test_aquifer_shares_count_onsite_water_only(monkeypatch):
    from dcfootprint.decisions import equity
    monkeypatch.setattr(equity, "load_cgwb_stage", lambda: ({"bangalore": 187.0}, {"maharashtra": 60.0}))
    acct = pd.DataFrame({"facility_id": ["f1", "f2"], "operator": ["A", "B"], "city": ["Bengaluru", "Pune"],
                         "state": ["Karnataka", "Maharashtra"],
                         "water_scarcity_l_eq": [900e3, 100e3],             # f1 mostly power-plant water
                         "water_scarcity_onsite_l_eq": [100e3, 100e3]})
    _, s = equity.equity_overlay(acct)
    assert s["pct_burden_over_exploited"] == 90.0                           # the round-5 reading
    assert s["pct_onsite_over_exploited"] == 50.0                           # the water drawn locally


# ---------------------------------------------------------------- registry kinds
def test_reference_documents_are_not_instruments():
    from dcfootprint.policy import gap
    from dcfootprint.policy.registry import REFERENCE_DOC_TYPES, build_registry, instruments
    reg = build_registry()
    ref = reg[reg["kind"] == "reference"]
    assert set(ref["doc_type"].str.lower()) <= REFERENCE_DOC_TYPES
    assert len(ref) == 4 and not ref["in_force"].any()
    assert len(instruments(reg)) == len(reg) - len(ref)
    assert "watttime_moer_methodology_2022" in set(ref["id"])
    assert set(gap.four_axis_matrix().columns) >= {"axis", "US", "EU", "India"}


# ---------------------------------------------------------------- status-quo routing baseline
@pytest.fixture(scope="module")
def prepared_nolegal():
    needed = [ROOT / "data" / "ember" / "india_monthly_full_release_long_format.csv",
              ROOT / "data" / "aware" / "AWARE20_Intermediate_Variables.xlsx"]
    if not (ACCOUNT.exists() and all(p.exists() for p in needed)):
        pytest.skip("needs the frozen account, Ember India and the AWARE intermediate variables")
    from dcfootprint.routing import lyapunov
    acct = pd.read_parquet(ACCOUNT)
    return acct, lyapunov.prepare(acct, legal=False)


def test_status_quo_moves_no_load(prepared_nolegal, monkeypatch):
    from dcfootprint.routing import lyapunov
    acct, P = prepared_nolegal
    base = lyapunov._params()
    monkeypatch.setattr(lyapunov, "_params", lambda: {**base, "demand_noise": 0.0})
    r = lyapunov.simulate(P, "status_quo", seed=0)
    assert r["carbon_tco2"] == pytest.approx(acct["carbon_tco2"].sum(), rel=1e-9)
    assert r["scarcity_m3eq"] == pytest.approx(acct["water_scarcity_l_eq"].sum() / 1000, rel=1e-9)
    assert r["unserved_mwh"] == 0.0


# ---------------------------------------------------------------- forecast significance helpers
def test_holm_and_sarima_fallback():
    sys.path.insert(0, str(ROOT / "dcfootprint" / "experiments"))
    import forecast_significance as fs
    np.testing.assert_allclose(fs._holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    key = {"resource": "electricity", "window": "holdout", "origin": "last12", "actual": 1.0}
    F = pd.DataFrame([{**key, "model": "seasonal_naive", "zone_id": z, "h": h, "mean": 2.0}
                      for z in ("A", "B") for h in (1, 2)] +
                     [{**key, "model": "sarima_dcf", "zone_id": "A", "h": h, "mean": 1.5} for h in (1, 2)])
    se = fs.squared_errors(F)
    fb = se[se["model"] == "sarima_dcf_fallback"].set_index(["zone_id", "h"])["se"]
    assert fb.loc[("A", 1)] == pytest.approx(0.25) and fb.loc[("B", 1)] == pytest.approx(1.0)   # B: naive
    r = fs.series_ratios(se)
    assert set(r[r["model"] == "sarima_dcf"]["zone_id"]) == {"A"}
    assert set(r[r["model"] == "sarima_dcf_fallback"]["zone_id"]) == {"A", "B"}


# ---------------------------------------------------------------- run manifest
def test_run_manifest_records_the_exact_config(tmp_path):
    import hashlib

    from dcfootprint.settings import config_path, write_run_manifest
    m = json.loads(write_run_manifest(tmp_path, {"entry_point": "test"}).read_text())
    cfg = config_path().read_text(encoding="utf-8")
    assert m["parameters_yaml"] == cfg
    assert m["parameters_sha256"] == hashlib.sha256(cfg.encode("utf-8")).hexdigest()
    assert m["entry_point"] == "test" and "git_commit" in m
