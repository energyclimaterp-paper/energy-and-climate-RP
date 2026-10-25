"""L3 one-step SARIMA validity checks (project/hierarchy.py), ported from experiments/forecast_run.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
US_EMBER = ROOT / "data" / "ember" / "us_monthly_full_release_long_format.csv"


@pytest.mark.skipif(not US_EMBER.exists(), reason="Ember US file not on disk")
def test_degenerate_new_york_generation_fit_is_rejected():
    # SARIMA on US New York generation to Dec 2021 "converges" with every coefficient at +-1,
    # log-likelihood 0 and a one-step forecast of exactly 0 GWh; before the check it fed Q3 and
    # made MinT's reconciled New York CI ~1e5 g/kWh
    from dcfootprint.io import ember
    from dcfootprint.project import hierarchy as H
    y = H._panel(ember.load_raw("US"))["generation"]["New York"]
    with pytest.raises(RuntimeError, match="degenerate"):
        H._sarima_onestep(y.astype(float), *map(__import__("pandas").Timestamp, ["2021-12-01", "2022-01-01", "2022-12-01"]))
    r = H._choose_and_forecast(y, 2023)          # selection year 2022 uses the degenerate fit
    assert r["model"] == "seasonal_naive" and "degenerate" in r["sarima_fail"]
    assert np.all(r["pred"] > 1000)               # New York generates ~10 TWh a month; never 0


def test_mint_spreads_discrepancy_in_proportion_to_weights():
    # one-level MinT, W = diag(w): bottom_i moves by w_i / sum(w) x (total - sum of bottoms). A zero base
    # forecast (the degenerate New York fit) therefore gets a sliver, and E/G of slivers explodes.
    from dcfootprint.project.hierarchy import _mint
    bottom = np.array([100.0, 50.0, 10.0])
    out = _mint(170.0, bottom, np.array([3.0, 1.0, 1.0, 1.0]))
    assert np.allclose(out - bottom, (170.0 - bottom.sum()) / 6.0)
