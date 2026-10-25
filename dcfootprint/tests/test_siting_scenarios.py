"""Q1 2030/2050 scenarios: minimax regret over weights x Aqueduct water scenarios, with carbon on the
official path (decisions/siting.py, project/carbon_future.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcfootprint.decisions.siting import CRITERIA, _minimax_scenarios
from dcfootprint.project.scarcity_future import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]


def _cells(base: list[list[float]], ws: list[list[float]], f2030=None, f2050=None) -> pd.DataFrame:
    d = pd.DataFrame(base, columns=CRITERIA)
    for j, s in enumerate(SCENARIOS):
        d[f"ws_{s}"] = [row[j] for row in ws]
    d["state"], d["basin_id"], d["excluded"] = "S", range(len(d)), False
    for y, f in ((2030, f2030), (2050, f2050)):                     # carbon path factor (1 = unchanged)
        d[f"new_carbon_tco2_{y}"] = d["new_carbon_tco2"] * (np.ones(len(d)) if f is None else np.asarray(f))
    return d


def test_dominating_cell_has_zero_regret():
    flat = [1.0] * len(SCENARIOS)
    c = _cells([[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]], [flat, [5.0] * len(SCENARIOS), [3.0] * len(SCENARIOS)])
    r = _minimax_scenarios(c, 200, 0).set_index("basin_id")
    assert r.loc[0, "max_regret"] == pytest.approx(0.0)
    assert r.loc[0, "max_regret"] < r.loc[2, "max_regret"] < r.loc[1, "max_regret"]


def test_max_regret_is_worst_scenario_and_future_stress_counts():
    # identical today; cell 1's basin worsens to 5 under pes50 only -> its worst case is pes50
    same = [0.5, 0.5, 0.5, 0.5]
    ws0 = [2.0] * len(SCENARIOS)
    ws1 = [2.0 if s != "pes50" else 5.0 for s in SCENARIOS]
    r = _minimax_scenarios(_cells([same, same], [ws0, ws1]), 200, 0).set_index("basin_id")
    per = r[[f"max_regret_{s}" for s in SCENARIOS]]
    assert np.allclose(r["max_regret"], per.max(axis=1))
    assert r.loc[1, "worst_scenario"] == "pes50"
    assert r.loc[1, "max_regret"] > r.loc[0, "max_regret"] == pytest.approx(0.0)
    assert r.loc[1, "max_regret_baseline"] == pytest.approx(0.0)


def test_excluded_cells_are_not_ranked():
    flat = [1.0] * len(SCENARIOS)
    c = _cells([[0, 0, 0, 0], [1, 1, 1, 1]], [flat, flat])
    c.loc[0, "excluded"] = True
    assert list(_minimax_scenarios(c, 50, 0)["basin_id"]) == [1]


@pytest.mark.skipif(not (ROOT / "data" / "aqueduct" / "Aq40_Y2023D07M05.gdb").exists(), reason="Aqueduct GDB not on disk")
def test_aqueduct_basin_scores_contract():
    from dcfootprint.project.scarcity_future import basin_scores
    s = basin_scores([47731, 50051])                      # Karnataka / Tamil Nadu cells in the India Q1 table
    assert set(s["scenario"]) == set(SCENARIOS) and len(s) == 2 * len(SCENARIOS)
    assert s["ws_score"].between(0, 5).all() and s["scored_share"].between(0, 1.01).all()


def test_carbon_path_changes_the_future_scenarios_only():
    # cell 1 is dirtier today but its grid is carbon-free by 2050 (factor 0); cell 0 keeps its grid.
    # Today cell 0 dominates; in the 2050 scenarios cell 1 has the lower carbon, so it has zero regret there.
    flat = [2.0] * len(SCENARIOS)
    c = _cells([[0.5, 0.2, 0.5, 0.5], [0.5, 0.8, 0.5, 0.5]], [flat, flat], f2030=[1.0, 1.0], f2050=[1.0, 0.0])
    r = _minimax_scenarios(c, 200, 0).set_index("basin_id")
    assert r.loc[0, "max_regret_baseline"] == pytest.approx(0.0) and r.loc[1, "max_regret_baseline"] > 0
    assert r.loc[1, "max_regret_bau50"] == pytest.approx(0.0) and r.loc[0, "max_regret_bau50"] > 0


def test_uniform_path_keeps_order_but_narrows_carbon_spread():
    # same factor everywhere (India: national path) -> carbon order unchanged, regret from carbon shrinks
    flat = [2.0] * len(SCENARIOS)
    c = _cells([[0.5, 0.2, 0.5, 0.5], [0.5, 0.8, 0.5, 0.5]], [flat, flat], f2030=[0.6, 0.6], f2050=[0.6, 0.6])
    r = _minimax_scenarios(c, 200, 0).set_index("basin_id")
    assert r.loc[0, "max_regret"] == pytest.approx(0.0)
    assert 0 < r.loc[1, "max_regret_bau50"] < r.loc[1, "max_regret_baseline"]


def test_zone_factors_fallback_and_uniform(monkeypatch):
    from dcfootprint.project import carbon_future as cf
    t = pd.DataFrame({"zone_id": ["A", "A", "B", "B", "C", "C"], "year": [2030, 2050] * 3,
                      "path_base": 1.0, "path_year": [0.5, 0.2, 0.7, 0.4, 0.9, 0.6],
                      "factor": [0.5, 0.2, 0.7, 0.4, 0.9, 0.6], "source": "src"})
    monkeypatch.setattr(cf, "path_factors", lambda region: t)
    z = cf.zone_factors("X", ["A", "Z"]).set_index("zone_id")
    assert z.loc["A", "carbon_f2030"] == 0.5 and z.loc["Z", "carbon_f2030"] == pytest.approx(0.7)   # median
    assert z.loc["Z", "carbon_path_source"].startswith("fallback")
    monkeypatch.setattr(cf, "path_factors", lambda region: t.iloc[0:0])
    assert (cf.zone_factors("X", ["A"])[["carbon_f2030", "carbon_f2050"]] == 1.0).all(axis=None)


@pytest.mark.skipif(not (ROOT / "data" / "carbon_future" / "Cambium22_LRMER_states.xlsx").exists(), reason="Cambium not on disk")
def test_us_path_reads_cambium_mid_case():
    from dcfootprint.project.carbon_future import path_factors
    t = path_factors("US").set_index(["zone_id", "year"])
    assert t.loc[("Alabama", 2030), "factor"] == pytest.approx(256.9 / 315.4, rel=1e-3)   # workbook Mid-case values
    assert t.loc[("Washington", 2050), "factor"] == pytest.approx(0.0)
    assert len(t) == 48 * 2


@pytest.mark.skipif(not (ROOT / "data" / "cea" / "CEA_Database_V22.xlsx").exists(), reason="CEA database not on disk")
def test_india_path_is_national_nep_over_cea_base():
    from dcfootprint.project.carbon_future import path_factors
    t = path_factors("India").set_index("year")
    base = t.loc[2030, "path_base"]
    assert t.loc[2030, "path_year"] == pytest.approx(0.548 + 0.8 * (0.430 - 0.548))   # FY 2030-31 on the NEP line
    assert t.loc[2050, "path_year"] == pytest.approx(0.430) and t.loc[2050, "factor"] == pytest.approx(0.430 / base)


@pytest.mark.skipif(not (ROOT / "data" / "carbon_future" / "ref2020_energy-transport-ghg.xlsx").exists(), reason="EU REF2020 not on disk")
def test_eu_path_covers_member_states():
    from dcfootprint.project.carbon_future import path_factors
    t = path_factors("EU")
    assert t["zone_id"].nunique() == 27 and (t["factor"] >= 0).all()
