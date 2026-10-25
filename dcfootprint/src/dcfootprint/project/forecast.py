"""L3 — grid-CI forecasting (real time series).

Rolling-origin backtest (several 12-month folds, not one holdout) of seasonal-naive vs
SARIMA on the Ember India national CO2-intensity series; a model replaces seasonal naive
only if it beats it on the backtest (the skill-floor rule). Forecast bands are the
empirical 10th/90th percentiles of the winner's backtest errors, not a fixed +-10%.

Also loads the reused earlier multi-model benchmark. LABELLING: in that benchmark the model
column "Nexus" holds **Chronos-2 running in the Nexus slot**, NOT the five-agent LLM Nexus
framework (see gat-sarima-nexus/docs/AUDIT_REPORT.md §1 and NEXUS_FRAMEWORK.md). It is
reported as "Chronos-2 (Nexus slot)"; the multi-agent Nexus has not been benchmarked here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.io import ember

NEXUS_SLOT_LABEL = "Chronos-2 (Nexus slot)"


def _gat_dir() -> Path | None:
    """preliminary-experiment repository location: $DCF_GAT_DIR, else <repo>/gat-based-forecasting/diff/gat-sarima-nexus,
    else the sibling ../diff/gat-sarima-nexus (this repo checked out inside the preliminary-experiment repository)."""
    root = _repo_root()
    for p in [os.environ.get("DCF_GAT_DIR"), root / "gat-based-forecasting" / "diff" / "gat-sarima-nexus",
              root.parent / "diff" / "gat-sarima-nexus"]:
        if p and Path(p).is_dir():
            return Path(p)
    return None


def _series(ci: pd.DataFrame) -> pd.Series:
    s = ci.set_index("date")["ci_gco2_per_kwh"].sort_index().asfreq("MS")
    return s.interpolate(limit_direction="both")


def _seasonal_naive(train: pd.Series, h: int) -> np.ndarray:
    last = train.iloc[-12:].values
    return np.array([last[i % 12] for i in range(h)])


def _sarima_forecast(train: pd.Series, h: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    m = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    f = m.get_forecast(h)
    return f.predicted_mean.values, f.conf_int(alpha=0.2).values


def backtest_ci(h: int = 12, n_folds: int = 3) -> dict:
    """Rolling-origin backtest: origins at len-h*n_folds, ..., len-h; each fold forecasts h months."""
    s = _series(ember.national_monthly_ci(ember.load_india_raw()))
    errs = {"seasonal_naive": [], "SARIMA": []}
    folds = []
    for k in range(n_folds, 0, -1):
        cut = len(s) - h * k
        train, test = s.iloc[:cut], s.iloc[cut:cut + h]
        errs["seasonal_naive"].append(_seasonal_naive(train, len(test)) - test.values)
        try:
            pred, _ = _sarima_forecast(train, len(test))
            errs["SARIMA"].append(pred - test.values)
        except Exception:
            errs["SARIMA"].append(None)
        folds.append(str(test.index[0].date()))
    rmse, fold_rmse = {}, {}
    for m, e in errs.items():
        if any(x is None for x in e):
            rmse[m] = None
            continue
        fold_rmse[m] = [round(float(np.sqrt(np.mean(x ** 2))), 2) for x in e]
        rmse[m] = float(np.sqrt(np.mean(np.concatenate(e) ** 2)))
    cand = {k: v for k, v in rmse.items() if v is not None}
    winner = min(cand, key=cand.get)
    skill = {m: round(1 - v / rmse["seasonal_naive"], 4) for m, v in cand.items()}
    resid = np.concatenate(errs[winner])
    return {"rmse": rmse, "fold_rmse": fold_rmse, "fold_origins": folds, "winner": winner,
            "skill_vs_naive": skill, "resid_q10": float(np.percentile(resid, 10)),
            "resid_q90": float(np.percentile(resid, 90))}


def forecast_ci(year_end: int = 2030) -> dict:
    """Backtest, then forecast national CI monthly to Dec `year_end`."""
    bt = backtest_ci()
    s = _series(ember.national_monthly_ci(ember.load_india_raw()))
    last = s.index[-1]
    h = (year_end - last.year) * 12 + (12 - last.month)
    dates = pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    if bt["winner"] == "SARIMA":
        try:
            mean, ci = _sarima_forecast(s, h)
            fc = pd.DataFrame({"date": dates, "ci_gco2_per_kwh": mean, "lo": ci[:, 0], "hi": ci[:, 1]})
        except Exception:
            bt["winner"] = "seasonal_naive"
    if bt["winner"] != "SARIMA":
        mean = _seasonal_naive(s, h)
        # error = forecast - actual  ->  actual in [mean - q90, mean - q10]
        fc = pd.DataFrame({"date": dates, "ci_gco2_per_kwh": mean,
                           "lo": mean - bt["resid_q90"], "hi": mean - bt["resid_q10"]})
    fc["model"] = bt["winner"]
    return {"backtest": bt, "forecast": fc}


def core_benchmark() -> dict:
    """Reuse the earlier cached benchmark: mean RMSE by model + the seasonal-naive skill
    audit. The CSV's "Nexus" rows are Chronos-2 in the Nexus slot -> relabelled explicitly."""
    d_gat = _gat_dir()
    if d_gat is None:
        return {}
    out = {"source": str(d_gat), "nexus_note": "'Nexus' in the earlier benchmark CSVs = Chronos-2 in the Nexus slot, "
                                              "not the multi-agent LLM framework (AUDIT_REPORT.md §1)"}
    mc = d_gat / "outputs_v2" / "partB" / "model_comparison_region.csv"
    if mc.exists():
        d = pd.read_csv(mc)
        d["model"] = d["model"].replace({"Nexus": NEXUS_SLOT_LABEL})
        out["rmse_by_model"] = (d.groupby(["resource", "model"])["RMSE"].mean()
                                .round(2).reset_index().to_dict("records"))
    sk = d_gat / "outputs_gapfill_v5" / "skill_vs_seasonal_naive.csv"
    if sk.exists():
        s = pd.read_csv(sk)
        out["skill_vs_naive"] = s[["resource", "method", "mean_skill", "pct_regions_beating_sn"]].round(3).to_dict("records")
    return out


if __name__ == "__main__":
    r = forecast_ci()
    print("backtest:", {k: v for k, v in r["backtest"].items()})
    fc = r["forecast"]
    print(f"forecast {fc['date'].min().date()}->{fc['date'].max().date()} ({len(fc)} mo), model={fc['model'].iloc[0]}")
    print(fc.head(3).to_string(index=False))
    cb = core_benchmark()
    print("\nearlier cached benchmark reused:", cb.get("source", "absent"))
    for rec in cb.get("rmse_by_model", []):
        if rec["resource"] == "carbon":
            print("  ", rec)
