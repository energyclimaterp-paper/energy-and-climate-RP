"""Round 6 additions: routing Monte Carlo scaling, basis-consistent CI forecast, climate WUE,
reported-figure validation, Ember US subnational adapter, site geocodes, eGRID reproduction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _account(n_fac: int = 3) -> pd.DataFrame:
    """Small synthetic facility-month account with every column the new code reads."""
    rows = []
    zones = ["A", "B", "B"]
    for f in range(n_fac):
        for m in range(1, 13):
            e_it = 100.0 + 10 * f + m
            pue = 1.5
            ci = 500.0 if zones[f] == "A" else 300.0 + m
            onsite = 1.9 * e_it * 1000
            grid_h, grid = 200.0 * e_it, 900.0 * e_it
            cf = 10.0 + f
            rows.append({"facility_id": f"f{f}", "month": m, "date": pd.Timestamp(2024, m, 1), "region": "India",
                         "operator": ["Nxtra by Airtel", "CtrlS", "CtrlS"][f], "city": "X", "state": zones[f],
                         "zone_id": zones[f], "basin_id": 100 + f, "facility_type": "colocation",
                         "capacity_mw": 10.0 + f, "e_it_mwh": e_it, "e_grid_mwh": e_it * pue, "pue": pue,
                         "ci_gco2_per_kwh": ci, "ci_source": "em_consumption", "carbon_tco2": e_it * pue * ci / 1000,
                         "cf": cf, "water_onsite_l": onsite, "water_scarcity_onsite_l_eq": onsite * cf,
                         "water_grid_l": grid, "water_grid_hydro_l": grid_h,
                         "water_scarcity_grid_l_eq": grid * 5, "water_scarcity_grid_hydro_l_eq": grid_h * 5,
                         "water_phys_l": onsite + grid, "water_scarcity_l_eq": onsite * cf + grid * 5,
                         "inference_share": 0.85})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- routing Monte Carlo
def test_scale_account_identity_and_rules():
    from dcfootprint.uncertainty.routing_mc import scale_account
    a = _account()
    same = scale_account(a, {})
    cols = ["e_it_mwh", "e_grid_mwh", "carbon_tco2", "water_phys_l", "water_scarcity_l_eq"]
    np.testing.assert_allclose(same[cols].to_numpy(), a[cols].to_numpy(), rtol=1e-12)
    x = {"util_colocation": 0.5, "pue_colocation": 1.2, "wue": 2.0, "ewif": 0.8, "hydro": 0.0, "ci_A": 1.1}
    s = scale_account(a, x)
    np.testing.assert_allclose(s["e_grid_mwh"], a["e_grid_mwh"] * 0.5 * 1.2)
    k = np.where(a["zone_id"] == "A", 1.1, 1.0)
    np.testing.assert_allclose(s["carbon_tco2"], a["carbon_tco2"] * 0.5 * 1.2 * k)
    np.testing.assert_allclose(s["water_onsite_l"], a["water_onsite_l"] * 0.5 * 2.0)
    nh = a["water_grid_l"] - a["water_grid_hydro_l"]
    np.testing.assert_allclose(s["water_grid_l"], nh * 0.5 * 1.2 * 0.8)        # hydro multiplier 0
    np.testing.assert_allclose(s["water_phys_l"], s["water_onsite_l"] + s["water_grid_l"])


def test_scale_account_matches_analytic_monte_carlo_totals():
    """The routing MC rescales the account with the same rules as monte_carlo._totals."""
    from dcfootprint.uncertainty.monte_carlo import _components, _totals
    from dcfootprint.uncertainty.routing_mc import scale_account
    a = _account()
    x = {"util_colocation": 0.8, "pue_colocation": 1.1, "wue": 0.6, "ewif": 1.15, "hydro": 0.3, "inference": 0.85}
    s = scale_account(a, x)
    t = _totals(_components(a), x)
    assert s["carbon_tco2"].sum() == pytest.approx(t["carbon"], rel=1e-12)
    assert s["water_phys_l"].sum() / 1000 == pytest.approx(t["phys"], rel=1e-12)
    assert s["water_scarcity_l_eq"].sum() / 1000 == pytest.approx(t["scarcity"], rel=1e-12)


def test_ci_groups_share_one_draw_per_grid():
    from dcfootprint.uncertainty.routing_mc import ci_groups
    a = _account()
    a.loc[a["facility_id"] == "f1", "zone_id"] = "B1"         # B and B1 carry the identical CI series
    g = ci_groups(a)
    assert g["B"] == g["B1"] == "B+B1" and g["A"] == "A"


# ---------------------------------------------------------------- basis-consistent CI forecast
def test_ci_forecast_carries_relative_error_onto_charged_basis(monkeypatch):
    from dcfootprint.io import ember
    from dcfootprint.project import hierarchy
    from dcfootprint.routing import lyapunov
    a = _account().sort_values(["facility_id", "month"]).reset_index(drop=True)
    gen_true = pd.DataFrame([{"zone_id": z, "month": m, "ci_gco2_per_kwh": 400.0, "ci_source": "ember_state"}
                             for z in ("A", "B") for m in range(1, 13)])
    hat = gen_true.rename(columns={"ci_gco2_per_kwh": "ci_hat"}).assign(ci_hat=440.0)   # +10% forecast error
    monkeypatch.setattr(hierarchy, "chosen_onestep_ci", lambda *a, **k: (hat, None, "x"))
    monkeypatch.setattr(ember, "load_raw", lambda region: None)
    monkeypatch.setattr(ember, "zone_month_ci", lambda raw, y, z: gen_true)
    out = lyapunov._ci_forecast(a)
    np.testing.assert_allclose(out, a["ci_gco2_per_kwh"] * 1.1)               # charged CI x (1 + error)
    b = a.assign(ci_source="ember_state")                                       # same basis: forecast as is
    np.testing.assert_allclose(lyapunov._ci_forecast(b), 440.0)


# ---------------------------------------------------------------- climate WUE
def test_gupta_cold_water_wue_values():
    from dcfootprint.account.wue import gupta_cold_water_wue
    tw_c = (np.array([60.0, 75.0, 80.0]) - 32) * 5 / 9
    expect = [0.0005112 * t ** 2 - 0.04982 * t + 2.387 for t in (60.0, 75.0, 80.0)]
    np.testing.assert_allclose(gupta_cold_water_wue(tw_c), expect, rtol=1e-12)
    assert (gupta_cold_water_wue(np.linspace(-30, 40, 200)) >= 0).all()


def test_recompute_onsite_only_touches_scope1():
    from dcfootprint.account.wue import recompute_onsite
    a = _account()
    w = a[["facility_id", "month"]].drop_duplicates().assign(wue_l_per_kwh=1.0, wue_source="t", tw_c=np.nan)
    b = recompute_onsite(a, w)
    np.testing.assert_allclose(b["water_onsite_l"], a["e_it_mwh"] * 1000.0)
    np.testing.assert_allclose(b["water_scarcity_onsite_l_eq"], b["water_onsite_l"] * a["cf"])
    np.testing.assert_allclose(b["water_grid_l"], a["water_grid_l"])
    np.testing.assert_allclose(b["carbon_tco2"], a["carbon_tco2"])
    np.testing.assert_allclose(b["water_scarcity_l_eq"], b["water_scarcity_onsite_l_eq"] + a["water_scarcity_grid_l_eq"])


def test_facility_month_wue_constant_keeps_operator_values():
    from dcfootprint.account.wue import facility_month_wue
    fac = pd.DataFrame({"facility_id": ["a", "b"], "operator": ["Meta", "Other"], "latitude": [1.0, 2.0],
                        "longitude": [1.0, 2.0]})
    p = {"water_onsite": {"wue_model": "constant", "wue_default": {"default": 1.9}, "wue_by_operator": {"meta": 0.19}}}
    w = facility_month_wue(fac, p, 2024)
    assert len(w) == 24
    assert set(w.loc[w["facility_id"] == "a", "wue_l_per_kwh"]) == {0.19}
    assert set(w.loc[w["facility_id"] == "b", "wue_l_per_kwh"]) == {1.9}


# ---------------------------------------------------------------- validation against reports
def test_reported_reference_file_is_complete():
    from dcfootprint.validation.reported import load_reported
    r = load_reported()
    assert r["source_url"].str.startswith("http").all() and r["pdf_page"].notna().all()
    assert (r["value"] > 0).all()


def test_compare_ratios_on_synthetic_account():
    from dcfootprint.validation.reported import compare
    a = _account()
    nx = a[a["operator"].str.startswith("Nxtra")]
    rep = pd.DataFrame([
        {"region": "India", "operator_match": "nxtra", "facility_id": pd.NA, "scope": "fleet", "period": "FY",
         "period_start": "2024-04-01", "period_end": "2025-03-31", "metric": m, "value": v, "unit": "",
         "source_url": "http://x", "pdf_page": 1}
        for m, v in (("electricity_mwh", nx["e_grid_mwh"].sum() / 2), ("capacity_mw", 20.0),
                     ("water_withdrawal_m3", nx["water_onsite_l"].sum() / 1000 / 4))])
    t = compare(a, rep).set_index("metric")
    assert t.loc["electricity_mwh", "ratio_total"] == pytest.approx(2.0)
    assert t.loc["electricity_mwh", "ratio_per_mw"] == pytest.approx(2.0 * 20.0 / 10.0)
    assert t.loc["onsite_water_l_per_facility_kwh", "ratio_total"] == pytest.approx(2.0)
    assert t.loc["electricity_mwh", "months_overlap_with_account_year"] == 9


# ---------------------------------------------------------------- Ember US subnational adapter
def test_us_subnational_adapter():
    from dcfootprint.io.ember import _us_subnational_to_long
    raw = pd.DataFrame([
        {"State": s, "State code": c, "Date": "2024-01-01", "Electricity source": src,
         "Generation (TWh)": g, "Emissions (MtCO2e)": e}
        for s, c in (("Texas", "TX"), ("US (total)", "US"), ("Puerto Rico", "PR"))
        for src, g, e in (("Gas", 2.0, 1.0), ("Total generation", 2.0, 1.0))])
    d = _us_subnational_to_long(raw)
    assert set(d["State"]) == {"Texas", "US Total"}                       # PR dropped, total renamed
    ci = d[(d["Variable"] == "CO2 intensity") & (d["State"] == "Texas")]["Value"].iloc[0]
    assert ci == pytest.approx(500.0)                                     # 1000 kt / 2000 GWh = 500 g/kWh


# ---------------------------------------------------------------- site geocodes
def test_site_geocodes_are_checked(tmp_path):
    from dcfootprint.io.facilities import attach_site_geocodes, load_site_geocodes
    p = tmp_path / "g.csv"
    pd.DataFrame({"facility_name": ["X"], "latitude": [19.1], "longitude": [72.9], "precision": ["address"],
                  "source_url": ["https://example.org"]}).to_csv(p, index=False)
    g = load_site_geocodes(p)
    raw = pd.DataFrame({"facility_name": ["X", "Y"]})
    fac = pd.DataFrame({"latitude": [0.0, 0.0], "longitude": [0.0, 0.0], "geocode_method": ["atlas_city_centroid"] * 2})
    out = attach_site_geocodes(fac, raw, g)
    assert out.loc[0, "geocode_method"] == "site_address" and out.loc[0, "latitude"] == 19.1
    assert out.loc[1, "geocode_method"] == "atlas_city_centroid"
    pd.DataFrame({"facility_name": ["X"], "latitude": [51.5], "longitude": [0.0], "precision": ["address"],
                  "source_url": ["https://example.org"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_site_geocodes(p)                                             # outside India
    pd.DataFrame({"facility_name": ["X"], "latitude": [19.1], "longitude": [72.9], "precision": ["guess"],
                  "source_url": [""]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_site_geocodes(p)                                             # no source / bad precision


# ---------------------------------------------------------------- location sensitivity
def test_centroid_bounds_take_extreme_basins():
    import geopandas as gpd
    from shapely.geometry import box

    from dcfootprint.geo.location_sensitivity import centroid_bounds
    basins = gpd.GeoDataFrame({"basin_id": [1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    cf = pd.DataFrame([{"basin_id": b, "month": m, "cf": c} for b, c in ((1, 10.0), (2, 2.0)) for m in range(1, 13)])
    a = pd.DataFrame([{"facility_id": "f", "month": m, "basin_id": 1, "city": "X", "water_onsite_l": 100.0,
                       "water_scarcity_onsite_l_eq": 1000.0, "water_scarcity_l_eq": 1500.0} for m in range(1, 13)])
    fac = pd.DataFrame({"facility_id": ["f"], "latitude": [0.5], "longitude": [0.95],
                        "geocode_method": ["atlas_city_centroid"]})
    t, s = centroid_bounds(a, fac, basins, cf, radius_km=20)
    assert t.loc[0, "n_basins"] == 2
    assert s["scope1_scarcity_range_m3eq"] == [round(100 * 2 * 12 / 1000), round(100 * 10 * 12 / 1000)]


# ---------------------------------------------------------------- eGRID reproduction (raw data)
_EGRID = ROOT / "data" / "egrid" / "egrid2023_data_rev2.xlsx"
_EIA = ROOT / "data" / "eia"


@pytest.mark.skipif(not (_EGRID.exists() and (_EIA / "Cooling_Boiler_Generator_Data_Summary_2023.xlsx").exists()),
                    reason="eGRID / EIA raw files not in data/ (see data/README.md)")
def test_egrid_subregion_rates_reproduced():
    from dcfootprint.io.us_grid import egrid_reproduction
    d = egrid_reproduction()
    assert len(d) >= 25
    assert d["rel_diff"].abs().median() < 0.01 and d["rel_diff"].abs().max() < 0.08
