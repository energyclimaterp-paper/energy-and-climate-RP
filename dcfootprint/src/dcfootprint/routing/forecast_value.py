"""L6 x L3 - the value of each forecasting model FOR ROUTING (decision-focused evaluation).

The forecasting benchmark (runs/forecast_benchmark_*) scores models on forecast error alone.
Here each model's grid-CI forecast drives the Q3 controller, and the outcome is scored as a
decision: the carbon and scarcity the routed load actually causes, and the penalty gap to the
same controller with perfect CI foresight. This links the forecasting section to the routing
result: a model is only worth its cost if it moves this gap.

Forecasts: the benchmark's rolling window with origin December of (account year - 1), horizons
1-12 = the account year's months (issued once, before the year starts). The benchmark forecasts
Ember generation CI; each forecast's relative error (mean / actual) is carried onto the CI the
account charges, exactly as routing/lyapunov._ci_forecast does for the pipeline's own forecast.
Rows:  perfect     ratio 1 (the lyapunov_pf reference)
       pipeline    the pre-registered one-step method of project/hierarchy.py (the default)
       <model>     each benchmark model with a forecast for the account year (xLSTM: mean of seeds)
The LLM Nexus run covers only the benchmark's holdout window (not the account year) and is
therefore not scored here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

BENCHMARK_RUN = "runs/forecast_benchmark_2026-09-28"
MODELS = ("seasonal_naive", "sarima_dcf", "sarima_core", "timesfm_2p5", "chronos_2", "xlstm")
_REGION = {"India": "India", "US": "US"}


def _zone_key(z: str) -> str:
    return str(z).split("|", 1)[-1].strip().lower()


def benchmark_ratios(model: str, region: str, year: int, run: str | Path = BENCHMARK_RUN) -> pd.DataFrame:
    """[zone_key, month, ratio = forecast / actual] for the forecast issued Dec (year-1)."""
    d = Path(run) if Path(run).is_absolute() else _repo_root() / run
    files = sorted((d / "forecasts" / model).glob("carbon*.csv"))
    if not files:
        raise FileNotFoundError(f"no carbon forecasts for {model} in {d}")
    f = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    f = f[(f["region"] == _REGION[region]) & (f["window"] == "rolling") & (f["origin"] == f"{year - 1}-12-01")]
    f = f[pd.to_datetime(f["timestamp"]).dt.year == year]
    if f.empty:
        raise ValueError(f"{model}: no rolling forecast issued {year - 1}-12 for {region}")
    f = f.groupby(["zone_id", "timestamp"], as_index=False).agg(mean=("mean", "mean"), actual=("actual", "first"))
    f["zone_key"] = f["zone_id"].map(_zone_key)
    f["month"] = pd.to_datetime(f["timestamp"]).dt.month
    f = f[(f["actual"] > 0) & np.isfinite(f["mean"])]
    f["ratio"] = f["mean"].clip(lower=0) / f["actual"]
    return f[["zone_key", "month", "ratio"]]


def _with_ci_hat(account: pd.DataFrame, ratios: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    """Account with a ci_hat column (charged CI x ratio); returns (account, coverage share)."""
    a = account.sort_values(["facility_id", "month"]).reset_index(drop=True).copy()
    if ratios is None:
        a["ci_hat"] = a["ci_gco2_per_kwh"]
        return a, 1.0
    key = a["zone_id"].map(_zone_key)
    m = pd.DataFrame({"zone_key": key, "month": a["month"]}).merge(ratios, on=["zone_key", "month"], how="left")
    cov = float(m["ratio"].notna().mean())
    a["ci_hat"] = a["ci_gco2_per_kwh"].to_numpy() * m["ratio"].fillna(1.0).to_numpy()
    return a, cov


def forecast_value(account: pd.DataFrame, models=MODELS, seeds: int | None = None, legal: bool = True,
                   A_basin: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per forecast source: CI forecast error on the account's zone-months (energy-weighted
    MAPE of the ratio) and the routing outcome (lyapunov and greedy % change vs static; lyapunov
    penalty gap to perfect foresight). Mean over `seeds` demand draws."""
    from dcfootprint.routing import lyapunov
    region = str(account["region"].iloc[0])
    year = int(_cfg_params()["region_config"][region]["account_year"])
    seeds = int(_cfg_params()["routing"]["seeds"]) if seeds is None else seeds

    sources: dict[str, tuple[pd.DataFrame, float]] = {"perfect": _with_ci_hat(account, None)}
    base = account.sort_values(["facility_id", "month"]).reset_index(drop=True).copy()
    base["ci_hat"] = lyapunov._ci_forecast(base)
    sources["pipeline"] = (base, 1.0)
    for m in models:
        try:
            sources[m] = _with_ci_hat(account, benchmark_ratios(m, region, year))
        except (FileNotFoundError, ValueError) as e:
            print(f"forecast_value: {m} skipped ({e})")

    rows = []
    for name, (a, cov) in sources.items():
        P = lyapunov.prepare(a, legal=legal, A_basin=A_basin)
        w = a["e_grid_mwh"].to_numpy()
        err = np.abs(a["ci_hat"].to_numpy() / a["ci_gco2_per_kwh"].to_numpy() - 1)
        res = {pol: [lyapunov.simulate(P, pol, seed=s) for s in range(seeds)] for pol in ("static", "greedy", "lyapunov")}
        def mean(pol, k, res=res):
            return float(np.mean([r[k] for r in res[pol]]))
        st_c, st_s = mean("static", "carbon_tco2"), mean("static", "scarcity_m3eq")
        rows.append({"forecast": name, "zone_month_coverage": round(cov, 3),
                     "ci_mape_pct": round(float(np.average(err, weights=w)) * 100, 2),
                     "greedy_carbon_change_pct": round((mean("greedy", "carbon_tco2") / st_c - 1) * 100, 3),
                     "lyapunov_carbon_change_pct": round((mean("lyapunov", "carbon_tco2") / st_c - 1) * 100, 3),
                     "lyapunov_scarcity_change_pct": round((mean("lyapunov", "scarcity_m3eq") / st_s - 1) * 100, 3),
                     "lyapunov_penalty": mean("lyapunov", "penalty")})
    out = pd.DataFrame(rows)
    ref = float(out.loc[out["forecast"] == "perfect", "lyapunov_penalty"].iloc[0])
    out["lyapunov_penalty_gap_vs_perfect_pct"] = ((out["lyapunov_penalty"] / ref - 1) * 100).round(3)
    return out
