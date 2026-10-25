"""Forecast-evaluation metrics used by the fresh forecasting run.

Point: MAE, RMSE, WMAPE, Bias (+ Bias %), MASE (seasonal-naive m=12 in-sample scale, per series),
       sMAPE, MedAE, Pearson r.
Probabilistic (only when the forecast carries a predictive distribution):
       PICP  share of actuals inside the stated central interval (nominal level reported alongside)
       CRPS  closed form for a Gaussian predictive N(mean, sigma^2) (Gneiting & Raftery 2007):
             CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (y - mean) / sigma
Forecasts without intervals get NaN for PICP/CRPS — never a made-up value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def mase_scale(train: np.ndarray, m: int = 12) -> float:
    """In-sample MAE of the seasonal-naive (lag m) forecast on the training series."""
    t = np.asarray(train, float)
    t = t[~np.isnan(t)]
    if len(t) <= m:
        return np.nan
    d = np.abs(t[m:] - t[:-m])
    s = float(np.mean(d))
    return s if s > 0 else np.nan


def crps_gaussian(y, mu, sigma):
    y, mu, sigma = map(lambda a: np.asarray(a, float), (y, mu, sigma))
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def metrics(df: pd.DataFrame) -> dict:
    """df columns: actual, mean [, lo, hi, sigma, mase_scale]. Rows with NaN actual are dropped."""
    d = df.dropna(subset=["actual", "mean"])
    if d.empty:
        return {}
    y, f = d["actual"].to_numpy(float), d["mean"].to_numpy(float)
    e = f - y
    out = {"n_points": int(len(d)),
           "MAE": float(np.mean(np.abs(e))),
           "RMSE": float(np.sqrt(np.mean(e ** 2))),
           "WMAPE_pct": float(np.sum(np.abs(e)) / np.sum(np.abs(y)) * 100) if np.sum(np.abs(y)) > 0 else np.nan,
           "Bias": float(np.mean(e)),
           "Bias_pct": float(np.sum(e) / np.sum(np.abs(y)) * 100) if np.sum(np.abs(y)) > 0 else np.nan,
           "sMAPE_pct": float(np.mean(np.where((np.abs(f) + np.abs(y)) > 0, 2 * np.abs(e) / (np.abs(f) + np.abs(y)), 0.0)) * 100),
           "MedAE": float(np.median(np.abs(e))),
           "Pearson_r": float(np.corrcoef(f, y)[0, 1]) if len(d) > 2 and np.std(f) > 0 and np.std(y) > 0 else np.nan}
    if "mase_scale" in d and d["mase_scale"].notna().any():
        sc = d["mase_scale"].to_numpy(float)
        ok = np.isfinite(sc) & (sc > 0)
        out["MASE"] = float(np.mean(np.abs(e[ok]) / sc[ok])) if ok.any() else np.nan
    else:
        out["MASE"] = np.nan
    # probabilistic metrics over the rows that carry a valid predictive distribution; the count is reported
    if {"lo", "hi", "sigma"} <= set(d.columns):
        ok = (d["lo"].notna() & d["hi"].notna() & d["sigma"].notna() & (d["sigma"] > 0)).to_numpy()
        out["n_prob"] = int(ok.sum())
        out["PICP"] = float(np.mean((y[ok] >= d["lo"].to_numpy()[ok]) & (y[ok] <= d["hi"].to_numpy()[ok]))) if ok.any() else np.nan
        out["CRPS"] = float(np.mean(crps_gaussian(y[ok], f[ok], d["sigma"].to_numpy()[ok]))) if ok.any() else np.nan
    else:
        out["n_prob"], out["PICP"], out["CRPS"] = 0, np.nan, np.nan
    return out
