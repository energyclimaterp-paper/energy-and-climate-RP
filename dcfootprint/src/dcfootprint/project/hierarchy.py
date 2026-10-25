"""L3 — one-step grid-CI forecasts for Q3: candidate methods + a pre-registered choice.

Series: Ember India monthly power-sector emissions (ktCO2) and generation (GWh) for every
state and the India total (states sum to the total within 1e-6, and CI = emissions /
generation within 0.005 g/kWh — checked when this was built).

Candidate methods for the one-step state CI in year Y (each uses data up to t-1 only):
  seasonal_naive   CI(z, t-12)  — the architecture's default
  bottom_up        per series (emissions, generation; each state), pick seasonal naive or
                   SARIMA(1,1,1)(1,0,1,12) on one-step RMSE in Y-1; SARIMA parameters fitted on
                   data up to Dec Y-1, state updated monthly (apply, refit=False); CI = E/G.
                   No reconciliation (national = sum of states, coherent by construction).
  mint_wls_var     bottom_up base forecasts reconciled with MinT, W = diag(one-step error
                   variance of each series' chosen model in Y-1)  (Wickramasuriya et al. 2019)
  mint_wls_struct  same with structural weights W = diag(S 1)

Pre-registered choice (fixed before any comparison was run): the method with the lowest mean
one-step CI RMSE over the datacenter-hosting states in the backtest years 2022 and 2023 (each
year selected/fitted on earlier data only) is used for the account year; a method replaces
seasonal_naive only if it beats it (skill floor). The account year itself is reported as an
out-of-sample check, never used for the choice.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from dcfootprint.io import ember

METHODS = ("seasonal_naive", "bottom_up", "mint_wls_var", "mint_wls_struct")
BACKTEST_YEARS = (2022, 2023)


def _panel(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def wide(cat, var, unit):
        d = raw[(raw["Category"] == cat) & (raw["Variable"] == var) & (raw["Unit"] == unit)]
        return d.pivot_table(index="date", columns="State", values="Value", aggfunc="sum").sort_index().asfreq("MS")
    return {"emissions": wide("Power sector emissions", "Total emissions", "ktCO2"),
            "generation": wide("Electricity generation", "Total Generation", "GWh")}


MAXITER = 500      # L-BFGS budget; a fit not converged by then is a failed fit (same rule as experiments/forecast_run.py)


def _sarima_onestep(y: pd.Series, fit_end: pd.Timestamp, pred_start: pd.Timestamp, pred_end: pd.Timestamp):
    """One-step predictions over [pred_start, pred_end] with parameters fitted on y[:fit_end].
    Raises on an invalid fit; the caller then falls back to seasonal naive."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # stationarity/invertibility ENFORCED (statsmodels defaults): unconstrained fits produced
    # explosive roots and one-step forecasts up to 1e88 on 2019-2022 data
    kw = dict(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y[:fit_end], **kw).fit(disp=False, maxiter=MAXITER)
        full = res.apply(y[:pred_end], refit=False)
        p = full.get_prediction(start=pred_start, end=pred_end, dynamic=False)
        pred, se = p.predicted_mean.values, p.se_mean.values
    # validity checks ported from the forecasting run: a "converged" fit with every coefficient
    # at +-1 and log-likelihood 0 forecast exactly 0 GWh for US New York generation (2022) and Texas
    if not res.mle_retvals.get("converged", False):
        raise RuntimeError(f"MLE did not converge in {MAXITER} iterations")
    coefs = res.params.drop("sigma2", errors="ignore").to_numpy(float)
    if np.any(np.abs(coefs) >= 0.9999) or not np.all(np.isfinite(se)):
        raise RuntimeError("degenerate fit (coefficient on the unit boundary or non-finite SE)")
    if not np.all(np.isfinite(pred)) or np.nanmax(np.abs(pred)) > 10 * max(float(np.nanmax(np.abs(y[:fit_end]))), 1e-9):
        raise FloatingPointError("SARIMA forecast diverged")
    return pred


def _snaive_onestep(y: pd.Series, pred_start, pred_end):
    idx = pd.date_range(pred_start, pred_end, freq="MS")
    return y.shift(12).reindex(idx).values


def _choose_and_forecast(y: pd.Series, year: int) -> dict:
    """Pick the model on one-step errors in year-1, then forecast `year` one step at a time.
    Also returns the chosen model's one-step error variance in year-1 (MinT-var weight)."""
    y = y.astype(float).interpolate(limit_direction="both")
    b0, b1 = pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year - 1, 12, 1)
    f0, f1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 1)
    truth_b = y[b0:b1].values
    err = {"seasonal_naive": _snaive_onestep(y, b0, b1) - truth_b}
    sarima_fail = None
    if y[:b0].std() > 0:
        try:
            err["SARIMA"] = _sarima_onestep(y, b0 - pd.offsets.MonthBegin(1), b0, b1) - truth_b
        except Exception as e:                                    # SARIMA not eligible for this series
            sarima_fail = f"selection: {type(e).__name__}: {e}"
    rmse = {k: float(np.sqrt(np.nanmean(v ** 2))) for k, v in err.items()}
    model = min(rmse, key=rmse.get)
    if model == "SARIMA":
        try:
            pred = _sarima_onestep(y, f0 - pd.offsets.MonthBegin(1), f0, f1)
        except Exception as e:                                    # diverged/failed in the forecast year
            model, pred = "seasonal_naive(sarima_failed)", _snaive_onestep(y, f0, f1)
            sarima_fail = f"forecast: {type(e).__name__}: {e}"
    else:
        pred = _snaive_onestep(y, f0, f1)
    return {"model": model, "backtest_rmse": rmse, "pred": np.maximum(pred, 0.0), "sarima_fail": sarima_fail,
            "err_var": float(np.nanmean(err["SARIMA" if model == "SARIMA" else "seasonal_naive"] ** 2))}


def _mint(base_total: float, base_bottom: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reconciled bottom forecasts for a one-level hierarchy with W = diag(w) (w[0] = total)."""
    n = len(base_bottom)
    S = np.vstack([np.ones((1, n)), np.eye(n)])
    Winv = np.diag(1.0 / np.maximum(w, 1e-12))
    G = np.linalg.solve(S.T @ Winv @ S, S.T @ Winv)
    return G @ np.concatenate([[base_total], base_bottom])


@lru_cache(maxsize=32)
def onestep_ci(year: int, zone: str = "state", method: str = "bottom_up", region: str = "India") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ci_hat [zone_id, month, ci_hat, ci_hat_source], diagnostics per series)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    raw = ember.load_raw(region)
    NATIONAL = ember.national_of(raw)
    panel = _panel(raw)
    states = [c for c in panel["generation"].columns if c != NATIONAL]
    if method == "seasonal_naive":
        prev = ember.zone_month_ci(raw, year - 1, zone)
        out = prev.rename(columns={"ci_gco2_per_kwh": "ci_hat"})[["zone_id", "month", "ci_hat"]]
        return out.assign(ci_hat_source="seasonal_naive"), pd.DataFrame()
    fc, var, diag = {}, {}, []
    for v, wide in panel.items():
        fc[v], var[v] = {}, {}
        for series in [NATIONAL] + states:
            r = _choose_and_forecast(wide[series], year)
            fc[v][series], var[v][series] = r["pred"], r["err_var"]
            diag.append({"series": series, "variable": v, "model": r["model"],
                         "rmse_snaive": r["backtest_rmse"]["seasonal_naive"],
                         "rmse_sarima": r["backtest_rmse"].get("SARIMA"), "sarima_fail": r["sarima_fail"]})
    rows = []
    for t in range(12):
        rec = {}
        for v in fc:
            bottom = np.array([fc[v][s][t] for s in states])
            if method == "bottom_up":
                rec[v] = bottom
            else:
                w = (np.array([var[v][NATIONAL]] + [var[v][s] for s in states]) if method == "mint_wls_var"
                     else np.array([len(states)] + [1.0] * len(states)))
                rec[v] = _mint(fc[v][NATIONAL][t], bottom, w)
        ci_nat = rec["emissions"].sum() / rec["generation"].sum() * 1000.0
        for i, s in enumerate(states):
            e, g = rec["emissions"][i], rec["generation"][i]
            ok = g > 0 and e >= 0
            rows.append({"zone_id": s, "month": t + 1,
                         "ci_hat": (e / g * 1000.0) if (zone == "state" and ok) else ci_nat,
                         "ci_hat_source": f"{method}_state" if (zone == "state" and ok) else f"{method}_national_fill"})
    return pd.DataFrame(rows), pd.DataFrame(diag)


def _rmse(year: int, method: str, zones, zone: str = "state", region: str = "India") -> float:
    ci_hat, _ = onestep_ci(year, zone, method, region)
    truth = ember.zone_month_ci(ember.load_raw(region), year, zone)
    d = truth.merge(ci_hat, on=["zone_id", "month"])
    d = d[d["zone_id"].isin(zones)]
    return float(np.sqrt(np.mean((d["ci_hat"] - d["ci_gco2_per_kwh"]) ** 2)))


def backtest_methods(zones, account_year: int, zone: str = "state", region: str = "India") -> tuple[pd.DataFrame, str]:
    """RMSE per method in each backtest year (selection) and in the account year (out-of-sample
    check only). Returns (table, chosen_method) under the pre-registered rule."""
    rows = []
    for m in METHODS:
        r = {"method": m}
        for y in BACKTEST_YEARS:
            r[f"rmse_{y}"] = _rmse(y, m, zones, zone, region)
        r["rmse_backtest_mean"] = float(np.mean([r[f"rmse_{y}"] for y in BACKTEST_YEARS]))
        r[f"rmse_{account_year}_out_of_sample"] = _rmse(account_year, m, zones, zone, region)
        rows.append(r)
    tab = pd.DataFrame(rows)
    best = tab.loc[tab["rmse_backtest_mean"].idxmin(), "method"]
    naive = float(tab.loc[tab["method"] == "seasonal_naive", "rmse_backtest_mean"].iloc[0])
    chosen = best if float(tab.loc[tab["method"] == best, "rmse_backtest_mean"].iloc[0]) < naive else "seasonal_naive"
    tab["chosen"] = tab["method"] == chosen
    return tab, chosen


CACHE_VERSION = "v3-validfit"        # bump when the forecasting code changes


def _cache_dir():
    from dcfootprint.io.facilities import _repo_root
    return _repo_root() / "data" / "ember"


def _fresh(path, region: str = "India") -> bool:
    src = _cache_dir() / ember.EMBER_FILES[region]          # each region's cache depends on its own Ember file
    return path.exists() and path.stat().st_mtime >= src.stat().st_mtime


def chosen_onestep_ci(zones: tuple, account_year: int, zone: str = "state",
                      region: str = "India") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(ci_hat for the account year from the pre-registered method, backtest table, method).
    Both are cached next to the Ember file (derived data, rebuilt when Ember or CACHE_VERSION changes)."""
    d = _cache_dir()
    tag = "" if region == "India" else f"{region.lower()}_"            # India keeps its existing cache names
    tab_p = d / f"ci_method_backtest_{tag}{account_year}_{zone}_{CACHE_VERSION}.csv"
    if _fresh(tab_p, region):
        tab = pd.read_csv(tab_p)
    else:
        tab, _ = backtest_methods(list(zones), account_year, zone, region)
        tab.to_csv(tab_p, index=False)
        tab = pd.read_csv(tab_p)              # same values whether computed now or read from the cache (csv round trip)
    method = str(tab.loc[tab["chosen"], "method"].iloc[0])
    ci_p = d / f"ci_onestep_{tag}{method}_{account_year}_{zone}_{CACHE_VERSION}.parquet"
    if _fresh(ci_p, region):
        ci = pd.read_parquet(ci_p)
    else:
        ci = onestep_ci(account_year, zone, method, region)[0]
        ci.to_parquet(ci_p)
    return ci, tab, method
