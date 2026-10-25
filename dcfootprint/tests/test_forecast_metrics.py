"""Forecast metrics checked against hand-computed values."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from forecast_metrics import crps_gaussian, mase_scale, metrics  # noqa: E402


def test_point_metrics_by_hand():
    df = pd.DataFrame({"actual": [10.0, 20.0, 30.0, 40.0], "mean": [12.0, 18.0, 33.0, 40.0]})
    m = metrics(df)                          # errors f - y = [2, -2, 3, 0]
    assert m["MAE"] == pytest.approx(7 / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt(17 / 4))
    assert m["WMAPE_pct"] == pytest.approx(7 / 100 * 100)
    assert m["Bias"] == pytest.approx(3 / 4)
    assert m["MedAE"] == pytest.approx(2.0)
    assert m["sMAPE_pct"] == pytest.approx(np.mean([4 / 22, 4 / 38, 6 / 63, 0]) * 100)
    assert m["Pearson_r"] == pytest.approx(np.corrcoef([12, 18, 33, 40], [10, 20, 30, 40])[0, 1])
    assert np.isnan(m["PICP"]) and np.isnan(m["CRPS"]) and np.isnan(m["MASE"])   # no interval / scale given


def test_perfect_forecast():
    y = np.arange(1, 25, dtype=float)
    m = metrics(pd.DataFrame({"actual": y, "mean": y}))
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["Bias"] == 0 and m["Pearson_r"] == pytest.approx(1.0)


def test_mase_scale_and_mase():
    train = np.r_[np.arange(12, dtype=float), np.arange(12, dtype=float) + 3]   # lag-12 diffs all = 3
    assert mase_scale(train) == pytest.approx(3.0)
    m = metrics(pd.DataFrame({"actual": [5.0, 5.0], "mean": [8.0, 2.0], "mase_scale": [3.0, 3.0]}))
    assert m["MASE"] == pytest.approx(1.0)


def test_picp_and_crps():
    assert crps_gaussian(0.0, 0.0, 1.0) == pytest.approx(2 * 0.3989422804 - 1 / np.sqrt(np.pi))   # 0.2337
    assert crps_gaussian(3.0, 1.0, 2.0) == pytest.approx(2 * crps_gaussian(1.0, 0.0, 1.0))        # scale invariance
    df = pd.DataFrame({"actual": [0.0, 5.0, -5.0, 1.0], "mean": [0.0] * 4, "lo": [-2.0] * 4, "hi": [2.0] * 4,
                       "sigma": [1.0] * 4})
    m = metrics(df)
    assert m["PICP"] == pytest.approx(0.5)
    assert m["CRPS"] == pytest.approx(np.mean(crps_gaussian(df["actual"], 0.0, 1.0)))
