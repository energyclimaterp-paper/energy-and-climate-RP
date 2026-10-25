"""Score real (LLM) Nexus forecasts with the run's full metric set, side by side with the earlier cached benchmark.

Inputs: one or more nexus_<resource>_ollama.csv files written by the Kaggle / local Nexus runner
(columns region_id, resource, arm, tag, timestamp, value, actual, seconds), plus the preliminary-experiment series dump
and cached forecasts (read-only). Scored exactly like experiments/score_core_cached.py:
  - actuals from diff/gat-sarima-nexus/work/region_series_dump.json (the runner's own `actual` column
    is checked against them, and the run stops if they disagree);
  - MASE scale = in-sample lag-12 naive MAE of the series before its first forecast month;
  - metrics from forecast_metrics.metrics(). Nexus emits point forecasts only, so PICP and CRPS are NaN
    (as for the cached models), never estimated.
Series whose tag is not "ollama" (a fallback fired) are reported and excluded from the LLM-Nexus scores.

Repeats (column `rep`): LLM Nexus is not repeatable on GPU even at temperature 0 (two runs of the same
3 basins differed by up to 88, also with parallelism and flash attention off), so each series is run
several times. Pre-registered: the Nexus forecast is the per-month MEDIAN over repeats; each repeat
is also scored alone, and the RMSE/MASE range across repeats and the forecast spread (max - min) are
written to <label>_repeat_variability.csv. Files without `rep` are one repeat.

The comparison table uses only the series that Nexus actually completed, for every model, so a
partial run is still compared like for like.

    python dcfootprint/experiments/score_nexus_llm.py <run_dir> <label> <nexus_csv> [<nexus_csv> ...]
    e.g. ... runs/nexus_llm_2026-09-28 "nexus-llm-gemma4-kaggle-gpu" nexus_electricity_ollama.csv nexus_carbon_ollama.csv
Use a NEW run folder: the folders in runs/ are frozen snapshots and must not be written to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_metrics import mase_scale, metrics  # noqa: E402
from score_core_cached import CORE, LABEL, region_of  # noqa: E402

HOLDOUT = 12


def _records(fc: pd.DataFrame, dump: dict, model_col: str | None = None) -> pd.DataFrame:
    recs = []
    for (res, rid), g in fc.groupby(["resource", "region_id"]):
        s = pd.Series(dump[res][rid]["values"], index=pd.to_datetime(dump[res][rid]["index"]), dtype=float)
        first = g["timestamp"].min()
        sc = mase_scale(s[s.index < first].to_numpy(float))
        for _, r in g.sort_values("timestamp").iterrows():
            recs.append({"model": r[model_col] if model_col else None, "resource": res, "zone_id": rid,
                         "region": region_of(rid, res), "timestamp": r["timestamp"],
                         "actual": s.get(r["timestamp"], np.nan), "mean": r["value"], "mase_scale": sc,
                         "lo": np.nan, "hi": np.nan, "sigma": np.nan})
    return pd.DataFrame(recs)


def run(run_dir: Path, label: str, files: list[Path]) -> pd.DataFrame:
    dump = json.load(open(CORE / "work" / "region_series_dump.json"))
    nx = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in files], ignore_index=True)
    if "rep" not in nx.columns:
        nx["rep"] = 1                                       # single-run files (before repeats existed)
    nx = nx.drop_duplicates(["resource", "region_id", "rep", "timestamp"])
    complete = nx.groupby(["resource", "region_id", "rep"])["timestamp"].transform("size") == HOLDOUT
    nx = nx[complete]
    fell_back = nx[nx["tag"] != "ollama"][["resource", "region_id", "rep", "tag"]].drop_duplicates()
    if len(fell_back):
        print(f"{len(fell_back)} series x repeat units fell back (excluded):\n{fell_back.to_string(index=False)}")
    nx = nx[nx["tag"] == "ollama"]
    reps = nx.groupby(["resource", "region_id"])["rep"].nunique()
    if reps.nunique() > 1:
        print(f"unequal repeat counts per series (scored on what exists): {reps.value_counts().to_dict()}")
    per_rep = nx.copy()
    # the Nexus forecast = per-month MEDIAN over repeats (pre-registered: LLM Nexus is not repeatable on GPU)
    nx = (nx.groupby(["resource", "region_id", "timestamp"], as_index=False)
          .agg(value=("value", "median"), actual=("actual", "first"), seconds=("seconds", "sum"),
               n_reps=("rep", "nunique"), spread=("value", lambda v: float(v.max() - v.min()))))
    nx["tag"] = "ollama"

    # the runner's actuals must equal the dump's (same series, same holdout) -- stop if not
    for (res, rid), g in nx.groupby(["resource", "region_id"]):
        s = pd.Series(dump[res][rid]["values"], index=pd.to_datetime(dump[res][rid]["index"]))
        idx = s.index[-HOLDOUT:]
        if sorted(g["timestamp"]) != list(idx) or not np.allclose(g.sort_values("timestamp")["actual"], s[idx].values):
            raise SystemExit(f"holdout/actuals mismatch for {res}/{rid}: not the preliminary-experiment holdout")

    d_nx = _records(nx.assign(model=label), dump, "model")
    cache = pd.read_csv(CORE / "outputs_v2" / "partB" / "per_region_base_forecasts.csv", parse_dates=["timestamp"])
    cache["model"] = cache["model"].map(LABEL)
    keys = set(zip(nx["resource"], nx["region_id"]))
    cache = cache[[k in keys for k in zip(cache["resource"], cache["region_id"])]]      # same series only
    d_all = pd.concat([d_nx, _records(cache, dump, "model")], ignore_index=True)

    # save Nexus forecasts in the run's format
    for (res, reg), d in d_nx.groupby(["resource", "region"]):
        out = run_dir / "forecasts" / label / res
        out.mkdir(parents=True, exist_ok=True)
        d.assign(h=d.groupby("zone_id").cumcount() + 1).drop(columns=["model", "resource", "region"]) \
            .to_csv(out / f"{reg.lower()}.csv", index=False)

    rows, zrows = [], []
    for (model, res, reg), d in d_all.groupby(["model", "resource", "region"]):
        rows.append({"model": model, "resource": res, "region": reg, "n_series": int(d["zone_id"].nunique()), **metrics(d)})
        for z, zz in d.groupby("zone_id"):
            zrows.append({"model": model, "resource": res, "region": reg, "zone_id": z, **metrics(zz)})
    m = run_dir / "metrics"
    m.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows).sort_values(["resource", "region", "RMSE"])
    tab.to_csv(m / f"{label}_vs_core_cached_per_region.csv", index=False)
    pd.DataFrame(zrows).to_csv(m / f"{label}_vs_core_cached_per_zone.csv", index=False)

    # run-to-run variability: each repeat scored on its own, and the spread between repeats
    d_rep = _records(per_rep.assign(model=[f"{label}__rep{int(k)}" for k in per_rep["rep"]]), dump, "model") \
        if per_rep["rep"].nunique() > 1 else pd.DataFrame()
    if len(d_rep):
        rr = pd.DataFrame([{"resource": res, "region": reg, "model": mdl, **metrics(d)}
                           for (mdl, res, reg), d in d_rep.groupby(["model", "resource", "region"])])
        var = (rr.groupby(["resource", "region"])
               .agg(n_repeats=("model", "nunique"), RMSE_min=("RMSE", "min"), RMSE_max=("RMSE", "max"),
                    MASE_min=("MASE", "min"), MASE_max=("MASE", "max")).reset_index())
        spread = nx.assign(region=[region_of(r, s) for r, s in zip(nx["region_id"], nx["resource"])]) \
            .groupby(["resource", "region"])["spread"].agg(spread_median="median", spread_max="max").reset_index()
        var = var.merge(spread, on=["resource", "region"])
        var.to_csv(m / f"{label}_repeat_variability.csv", index=False)
        rr.to_csv(m / f"{label}_per_repeat_per_region.csv", index=False)
        print("run-to-run variability (each repeat scored alone; spread = max-min forecast across repeats):\n"
              + var.round(3).to_string(index=False) + "\n")
    runtime = nx.groupby(["resource", "region_id"])["seconds"].first()
    print(f"\n{label}: {len(keys)} series scored ({nx.groupby('resource').region_id.nunique().to_dict()}); "
          f"median {runtime.median():.0f} s per series\n")
    return tab


if __name__ == "__main__":
    t = run(Path(sys.argv[1]), sys.argv[2], [Path(f) for f in sys.argv[3:]])
    cols = ["resource", "region", "model", "n_series", "RMSE", "MAE", "MASE", "WMAPE_pct", "sMAPE_pct",
            "Bias", "MedAE", "Pearson_r", "PICP", "CRPS"]
    print(t[cols].round(3).to_string(index=False))
