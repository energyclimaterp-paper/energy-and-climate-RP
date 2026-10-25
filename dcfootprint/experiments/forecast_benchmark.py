"""Fresh forecasting run on the benchmark series grid: every model, one grid, all 10 metrics.

Why this grid: it is the grid the real (LLM) Nexus runs on (Kaggle, run_nexus_kaggle.py), and its 95
zones contain every zone of the account (India 9 states, US 36 states, EU 21 countries). Per
docs/methodology.md (forecasting layer), the research questions need grid carbon intensity
(and electricity as its component) forecast per account zone and backtested at a 12-month horizon
against seasonal naive. Water is NOT forecast (future water = Aqueduct scenarios), so the G3P
basins are left out; 5-year "forecasts" to 2030 are left out too (scenario, risk register R5/R11).

Second grid (SERIES_DUMP=data/ember/dcfootprint_grid_series_dump.json, RUN_DIR=runs/fresh_run_dcf_grid_2026-09-28):
every Ember zone the pipeline reads -- 36 India states, 52 US states, 27 EU countries -- plus the national
/ EU aggregates (scored as level "aggregate"), same models, windows and metrics; no cache check, no Nexus.

Design (fixed before running; do not change after seeing results)
  series    preliminary-experiment work/region_series_dump.json (read-only), electricity + carbon, 95 zones each
  windows   holdout : the last 12 observations of each series (= earlier cached benchmark = Nexus on Kaggle)
            rolling : origins Dec 2022, Dec 2023, Dec 2024, horizon 12 (robustness; no Nexus)
  models    seasonal_naive  y(t-12); sigma = SD of its errors over the 12 months up to the origin
            sarima_dcf      dcfootprint SARIMA(1,1,1)(1,0,1,12) with the validity checks
                            (experiments/forecast_run._sarima: converged in 500 iters, no unit-boundary
                            coefficient, finite SE); a rejected fit is a recorded failure
            sarima_core     the preliminary-experiment notebook's fit_sarima VERBATIM (cell 19 of partB notebook),
                            (1,1,1)(1,1,0,12), unconstrained: reproduces the cache (sanity check)
            timesfm_2p5     google/timesfm-2.5-200m-pytorch, notebook config; point = q0.5,
                            80% interval = q0.1..q0.9
            chronos_2       amazon/chronos-2 via preliminary-experiment src/nexus_local; point = q0.5, 80% = q0.1..q0.9
            xlstm           the notebook's xLSTM VERBATIM, seeds 42-46, seed set per series;
                            no predictive distribution -> PICP/CRPS not computed; seed spread reported
            nexus_llm       added later from the Kaggle CSVs (score-nexus), holdout window only
  intervals 80% central; sigma for quantile models = (q90 - q10) / (2 * 1.2816) for Gaussian CRPS
  metrics   experiments/forecast_metrics.metrics (MAE RMSE WMAPE Bias MASE sMAPE MedAE Pearson PICP CRPS)
  levels    region (India / US / Europe) x {all zones, account zones}

    python forecast_benchmark.py run            # everything (resumable)
    python forecast_benchmark.py score          # recompute metrics from the forecast files
    python forecast_benchmark.py score-nexus nexus_electricity_ollama.csv nexus_carbon_ollama.csv
Run with a Python environment that has torch, chronos and timesfm installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from forecast_metrics import mase_scale, metrics  # noqa: E402

REPO = HERE.parents[1]
CORE = Path(os.environ.get("DCF_GAT_DIR", REPO.parent / "diff" / "gat-sarima-nexus"))
CORE_DUMP = CORE / "work" / "region_series_dump.json"
# SERIES_DUMP selects the grid: the preliminary-experiment series dump (default) or data/ember/dcfootprint_grid_series_dump.json
# (every Ember state/country the pipeline reads + the national/EU aggregates; written from io/ember)
DUMP = Path(os.environ.get("SERIES_DUMP", CORE_DUMP))
GRID = "core" if DUMP.resolve() == CORE_DUMP.resolve() else "dcfootprint"
CORE_NB = CORE / "notebooks" / "v2" / "partB_gat_weighted_forecasting.ipynb"
CACHE = CORE / "outputs_v2" / "partB" / "per_region_base_forecasts.csv"
RUN = Path(os.environ.get("RUN_DIR", REPO / "runs" / "forecast_benchmark_2026-09-28"))

RESOURCES = ("electricity", "carbon")
H = 12
ORIGINS = ("2022-12-01", "2023-12-01", "2024-12-01")
SEEDS = (42, 43, 44, 45, 46)
Z80 = 1.2815515655446004
MIN_TRAIN = 36
XLSTM_WORKERS = int(os.environ.get("XLSTM_WORKERS", "16"))
ACCOUNT = {  # the zones the account uses (dcfootprint/outputs/account_*.parquet), in benchmark zone ids
    "India": ["delhi", "gujarat", "karnataka", "maharashtra", "odisha", "tamil nadu", "telangana",
              "uttar pradesh", "west bengal"],
    "US": ["alabama", "arizona", "arkansas", "california", "colorado", "florida", "georgia", "illinois",
           "indiana", "iowa", "kansas", "kentucky", "louisiana", "maryland", "massachusetts", "michigan",
           "minnesota", "mississippi", "nebraska", "nevada", "new jersey", "new mexico", "new york",
           "north carolina", "ohio", "oklahoma", "oregon", "pennsylvania", "south carolina", "tennessee",
           "texas", "utah", "virginia", "washington", "wisconsin", "wyoming"],
    "Europe": ["AUT", "BEL", "BGR", "HRV", "DNK", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
               "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ESP", "SWE"],
}


def region_of(zid: str) -> str:
    return "India" if zid.startswith("IND|") else "US" if zid.startswith("USA|") else "Europe"


_META = {k: set(v) for k, v in json.load(open(DUMP)).items() if k in ("_account_zones", "_aggregate_zones")}


def is_account(zid: str) -> bool:
    if "_account_zones" in _META:                               # the dump names its own account zones
        return zid in _META["_account_zones"]
    r = region_of(zid)
    return (zid.split("|", 1)[1] if "|" in zid else zid) in ACCOUNT[r]


def is_aggregate(zid: str) -> bool:
    return zid in _META.get("_aggregate_zones", set())


def load_series() -> dict[str, dict[str, pd.Series]]:
    d = json.load(open(DUMP))
    lim = int(os.environ.get("LIMIT_ZONES", "0"))            # smoke tests only
    return {res: {z: pd.Series(p["values"], index=pd.to_datetime(p["index"]), dtype=float).sort_index()
                  for z, p in (sorted(d[res].items())[:lim] if lim else d[res].items())} for res in RESOURCES}


def windows(s: pd.Series):
    """(window, origin label, train, test) -- positional holdout + calendar rolling origins."""
    yield "holdout", "last12", s.iloc[:-H], s.iloc[-H:]
    for o in ORIGINS:
        o = pd.Timestamp(o)
        tr = s[s.index <= o]
        te = s[s.index > o].iloc[:H]
        if len(tr) >= MIN_TRAIN and len(te) == H:
            yield "rolling", o.date().isoformat(), tr, te


# ------------------------------------------------------------------ preliminary-experiment notebook forecasters
def core_defs() -> dict:
    """exec fit_sarima / train_xlstm / predict_xlstm from notebook cell 19 (lines 1-106), verbatim."""
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    nb = json.load(open(CORE_NB, encoding="utf-8"))
    src = "\n".join("".join(nb["cells"][19]["source"]).split("\n")[1:107])
    ns = dict(np=np, pd=pd, torch=torch, nn=nn, Dataset=Dataset, DataLoader=DataLoader, SARIMAX=SARIMAX,
              DEVICE=torch.device("cpu"), LOOKBACK=12, HIDDEN_SIZE=64, NUM_LAYERS=2, DROPOUT=0.2,
              EPOCHS=100, LR=1e-3, BATCH_SIZE=16)       # the notebook's cell-2 hyper-parameters
    exec(src, ns)
    return ns


def rows(model, res, zid, window, origin, train, test, mean, sigma=None, lo=None, hi=None, seed=None):
    mean = np.asarray(mean, float)
    sig = np.full(H, np.nan) if sigma is None else np.asarray(sigma, float)
    lo = mean - Z80 * sig if lo is None else np.asarray(lo, float)
    hi = mean + Z80 * sig if hi is None else np.asarray(hi, float)
    return pd.DataFrame({"model": model, "resource": res, "zone_id": zid, "region": region_of(zid),
                         "account_zone": is_account(zid), "aggregate_zone": is_aggregate(zid),
                         "window": window, "origin": origin,
                         "h": np.arange(1, H + 1), "timestamp": test.index, "actual": test.values,
                         "mean": mean, "sigma": sig, "lo": lo, "hi": hi, "interval_level": 0.80,
                         "mase_scale": mase_scale(train.to_numpy(float)), "seed": seed})


def fc_path(model: str, res: str, seed=None) -> Path:
    return RUN / "forecasts" / model / (f"{res}.csv" if seed is None else f"{res}_seed{seed}.csv")


def _save(df: pd.DataFrame, fails: list, model: str, res: str, seed=None):
    p = fc_path(model, res, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    fp = RUN / "metrics" / "failures.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fails:
        pd.DataFrame(fails).to_csv(fp, mode="a", header=not fp.exists(), index=False)
    print(f"  {model:<15} {res:<11}{'' if seed is None else f' seed {seed}'}: {df.groupby(['window','origin']).zone_id.nunique().to_dict()}"
          f"  failures={len(fails)}", flush=True)


def run_stat(S):
    """seasonal_naive, sarima_dcf, sarima_core."""
    import forecast_run as FR                                 # dcfootprint SARIMA with validity checks
    ns = core_defs()
    for model in ("seasonal_naive", "sarima_dcf", "sarima_core"):
        for res in RESOURCES:
            if fc_path(model, res).exists():
                continue
            out, fails = [], []
            for zid, s in S[res].items():
                y = s.asfreq("MS")                            # calendar index; gaps stay NaN (Kalman handles them)
                for win, org, tr, te in windows(s):
                    try:
                        if model in ("seasonal_naive", "sarima_dcf"):
                            # calendar forecasts for the 12 months after the origin, aligned to the test
                            # stamps (identical except for a gappy series, where months beyond h=12 are NaN)
                            idx, mu, sg = (FR._snaive if model == "seasonal_naive" else FR._sarima)(y, tr.index[-1])
                            mu = pd.Series(mu, index=idx).reindex(te.index).to_numpy(float)
                            sg = pd.Series(sg, index=idx).reindex(te.index).to_numpy(float)
                        else:
                            mu, r = ns["fit_sarima"](tr, H)
                            sg = np.asarray(r.get_forecast(H).se_mean, float)
                            if not np.all(np.isfinite(mu)):
                                raise FloatingPointError("non-finite forecast")
                        out.append(rows(model, res, zid, win, org, tr, te, mu, sg))
                    except Exception as e:
                        fails.append({"model": model, "resource": res, "zone_id": zid, "window": win,
                                      "origin": org, "reason": f"{type(e).__name__}: {e}"[:200]})
            _save(pd.concat(out, ignore_index=True), fails, model, res)


def run_foundation(S):
    """timesfm_2p5 and chronos_2 (quantile models)."""
    sys.path.insert(0, str(CORE / "src"))
    import torch
    if not fc_path("timesfm_2p5", RESOURCES[-1]).exists():
        import timesfm
        from timesfm import ForecastConfig
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                                 use_continuous_quantile_head=True))
        for res in RESOURCES:
            if fc_path("timesfm_2p5", res).exists():
                continue
            out, fails = [], []
            for zid, s in S[res].items():
                for win, org, tr, te in windows(s):
                    try:
                        _, q = m.forecast(horizon=H, inputs=[np.asarray(tr.values, dtype=np.float32)])
                        q = np.asarray(q)[0]                  # (H, 10): [mean, q0.1 .. q0.9]
                        mu, lo, hi = q[:, 5], q[:, 1], q[:, 9]
                        out.append(rows("timesfm_2p5", res, zid, win, org, tr, te, mu, (hi - lo) / (2 * Z80), lo, hi))
                    except Exception as e:
                        fails.append({"model": "timesfm_2p5", "resource": res, "zone_id": zid, "window": win,
                                      "origin": org, "reason": f"{type(e).__name__}: {e}"[:200]})
            _save(pd.concat(out, ignore_index=True), fails, "timesfm_2p5", res)
    from nexus_local import load_chronos
    pipe = load_chronos()
    for res in RESOURCES:
        if fc_path("chronos_2", res).exists():
            continue
        out, fails = [], []
        for zid, s in S[res].items():
            for win, org, tr, te in windows(s):
                try:
                    q, _ = pipe.predict_quantiles([torch.tensor(tr.values, dtype=torch.float32)],
                                                  prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
                    q = np.asarray(q[0][0], float)          # (H, 3)
                    mu, lo, hi = q[:, 1], q[:, 0], q[:, 2]
                    out.append(rows("chronos_2", res, zid, win, org, tr, te, mu, (hi - lo) / (2 * Z80), lo, hi))
                except Exception as e:
                    fails.append({"model": "chronos_2", "resource": res, "zone_id": zid, "window": win,
                                  "origin": org, "reason": f"{type(e).__name__}: {e}"[:200]})
        _save(pd.concat(out, ignore_index=True), fails, "chronos_2", res)


_NS = None


def _xlstm_init():
    global _NS
    import torch
    torch.set_num_threads(1)
    _NS = core_defs()


def _xlstm_task(args):
    res, zid, win, org, tr_vals, tr_idx, te_vals, te_idx, seed = args
    import torch
    from sklearn.preprocessing import MinMaxScaler
    tr = pd.Series(tr_vals, index=pd.to_datetime(tr_idx)); te = pd.Series(te_vals, index=pd.to_datetime(te_idx))
    try:
        np.random.seed(seed); torch.manual_seed(seed)          # per series: independent of scheduling order
        sc = MinMaxScaler(); trs = sc.fit_transform(tr.values.reshape(-1, 1))
        mdl = _NS["train_xlstm"](trs, input_size=1)
        mu = _NS["predict_xlstm"](mdl, sc, tr.values, H)
        if not np.all(np.isfinite(mu)):
            raise FloatingPointError("non-finite forecast")
        return rows("xlstm", res, zid, win, org, tr, te, mu, seed=seed), None
    except Exception as e:
        return None, {"model": "xlstm", "resource": res, "zone_id": zid, "window": win, "origin": org,
                      "seed": seed, "reason": f"{type(e).__name__}: {e}"[:200]}


def run_xlstm(S):
    from multiprocessing import Pool
    for seed in SEEDS:
        for res in RESOURCES:
            if fc_path("xlstm", res, seed).exists():
                continue
            tasks = [(res, zid, win, org, tr.values, tr.index.astype(str).tolist(), te.values,
                      te.index.astype(str).tolist(), seed)
                     for zid, s in S[res].items() for win, org, tr, te in windows(s)]
            t0 = time.time()
            with Pool(XLSTM_WORKERS, initializer=_xlstm_init) as pool:
                got = pool.map(_xlstm_task, tasks, chunksize=2)
            out = [g for g, _ in got if g is not None]
            fails = [f for _, f in got if f is not None]
            _save(pd.concat(out, ignore_index=True), fails, "xlstm", res, seed)
            print(f"    ({len(tasks)} fits in {time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ scoring
def all_forecasts() -> pd.DataFrame:
    fs = sorted((RUN / "forecasts").glob("*/*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def score():
    F = all_forecasts()
    rows_, zone_rows = [], []
    for (model, res, win, seed), g in F.groupby(["model", "resource", "window", F["seed"].fillna(-1)]):
        for region in ("India", "US", "Europe", "All"):
            gr = g if region == "All" else g[g["region"] == region]
            agg = gr["aggregate_zone"].astype(bool) if "aggregate_zone" in gr else pd.Series(False, index=gr.index)
            for level, gl in (("all_zones", gr[~agg]), ("account_zones", gr[gr["account_zone"].astype(bool)]),
                              ("aggregate", gr[agg])):         # national / EU totals scored apart
                if len(gl):
                    rows_.append({"model": model, "resource": res, "window": win,
                                  "seed": None if seed == -1 else int(seed), "region": region, "level": level,
                                  "n_series": gl.zone_id.nunique(), "n_windows": gl.groupby(["zone_id", "origin"]).ngroups,
                                  **metrics(gl)})
        for (zid, org), gz in g.groupby(["zone_id", "origin"]):
            zone_rows.append({"model": model, "resource": res, "window": win, "origin": org,
                              "seed": None if seed == -1 else int(seed), "zone_id": zid, **metrics(gz)})
    M = pd.DataFrame(rows_)
    out = RUN / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "per_model_per_region.csv", index=False)
    pd.DataFrame(zone_rows).to_csv(out / "per_model_per_zone.csv", index=False)
    # xLSTM: mean and spread across the 5 seeds (the per-seed rows stay in per_model_per_region.csv)
    x = M[M["model"] == "xlstm"]
    if len(x):
        num = [c for c in x.columns if c not in ("model", "resource", "window", "seed", "region", "level")
               and pd.api.types.is_numeric_dtype(x[c])]
        agg = x.groupby(["resource", "window", "region", "level"])[num].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.reset_index().assign(model="xlstm", n_seeds=x.seed.nunique()).to_csv(out / "xlstm_seed_spread.csv", index=False)
    return M


def headline(M: pd.DataFrame) -> pd.DataFrame:
    """One row per model x resource x window x region (account zones); xLSTM = mean over seeds."""
    a = M[M["level"] == "account_zones"].copy()
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c != "seed"]
    return a.groupby(["model", "resource", "window", "region"])[num].mean().reset_index()


def score_nexus(paths):
    """Kaggle CSVs (region_id, resource, timestamp, value, actual[, rep, tag]) -> forecasts/nexus_llm/<res>.csv.
    Same pre-registered rules as experiments/score_nexus_llm.py: the Nexus forecast is the per-month
    MEDIAN over repeats (`rep`); series whose tag is not "ollama" (a fallback fired) are excluded."""
    S = load_series()
    for p in paths:
        k = pd.read_csv(p)
        if "tag" not in k:
            k = k.assign(tag="untagged")
        bad = k.loc[~k["tag"].isin(["ollama", "untagged"]), "region_id"].unique()
        if len(bad):
            print(f"  nexus: {len(bad)} series excluded (fallback fired): {list(bad)[:8]}")
            k = k[~k.region_id.isin(bad)]
        k = (k.groupby(["resource", "region_id", "timestamp"], as_index=False)
              .agg(value=("value", "median"), actual=("actual", "first"), tag=("tag", "first"),
                   n_rep=("value", "size")))
        for res, g in k.groupby("resource"):
            out = []
            for zid, gz in g.groupby("region_id"):
                s = S[res][zid]
                tr, te = s.iloc[:-H], s.iloc[-H:]
                gz = gz.assign(timestamp=pd.to_datetime(gz["timestamp"])).set_index("timestamp").reindex(te.index)
                if gz["value"].isna().any():
                    print(f"  nexus {res}/{zid}: timestamps do not match the holdout -- skipped")
                    continue
                if not np.allclose(gz["actual"].to_numpy(float), te.to_numpy(float)):
                    raise ValueError(f"nexus {res}/{zid}: actuals differ from the dump -- different series")
                d = rows("nexus_llm", res, zid, "holdout", "last12", tr, te, gz["value"].to_numpy(float))
                d["tag"], d["n_rep"] = gz["tag"].to_numpy(), gz["n_rep"].to_numpy()
                out.append(d)
            _save(pd.concat(out, ignore_index=True), [], "nexus_llm", res)
    score()


def cache_check():
    """Fresh holdout forecasts vs the earlier cached benchmark (same code, same data -> should agree)."""
    if GRID != "core" or not CACHE.exists():                   # the cache exists for the benchmark grid only
        return
    c = pd.read_csv(CACHE)
    c = c[c["resource"].isin(RESOURCES)].assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    F = all_forecasts()
    F = F[F["window"] == "holdout"].assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    pairs = {"SARIMA": "sarima_core", "TimesFM": "timesfm_2p5", "Nexus": "chronos_2", "xLSTM": "xlstm"}
    rows_ = []
    for cm, fm in pairs.items():
        f = F[F["model"] == fm]
        if fm == "xlstm":
            f = f[f["seed"] == 42]
        m = c[c["model"] == cm].merge(f, left_on=["region_id", "resource", "timestamp"],
                                      right_on=["zone_id", "resource", "timestamp"])
        if not len(m):
            continue
        rel = (m["mean"] - m["value"]).abs() / m["value"].abs().clip(lower=1e-9)
        rm = lambda v: m.assign(e=(v - m["actual"]) ** 2).groupby(["resource", "zone_id"])["e"].mean() ** 0.5
        rows_.append({"cache_model": cm, "fresh_model": fm, "n_points": len(m),
                      "median_rel_diff": float(rel.median()), "p95_rel_diff": float(rel.quantile(0.95)),
                      "share_within_1pct": float((rel < 0.01).mean()),
                      "mean_zone_rmse_cache": float(rm(m["value"]).mean()), "mean_zone_rmse_fresh": float(rm(m["mean"]).mean())})
    t = pd.DataFrame(rows_)
    t.to_csv(RUN / "metrics" / "cache_check.csv", index=False)
    print("\nCACHE CHECK (fresh holdout vs earlier cached benchmark):\n" + t.round(4).to_string(index=False))


def manifest(t_start: float):
    m = RUN / "manifest"
    m.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    json.dump({
        "run": RUN.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_clock_min": round((time.time() - t_start) / 60, 1),
        "script": str(Path(__file__).relative_to(REPO)), "dcfootprint_commit": git("-C", str(REPO), "rev-parse", "HEAD"),
        "core_repo_commit": git("-C", str(CORE), "rev-parse", "HEAD"),
        "core_repo_dirty": bool(git("-C", str(CORE), "status", "--porcelain", "--", ".")),
        "series_file": str(DUMP), "series_sha256": hashlib.sha256(DUMP.read_bytes()).hexdigest(),
        "notebook_file": str(CORE_NB), "notebook_sha256": hashlib.sha256(CORE_NB.read_bytes()).hexdigest(),
        "python": sys.version, "resources": RESOURCES, "horizon": H, "rolling_origins": ORIGINS,
        "xlstm_seeds": SEEDS, "xlstm_hparams": {"lookback": 12, "hidden": 64, "layers": 2, "dropout": 0.2,
                                                "epochs": 100, "lr": 1e-3, "batch": 16},
        "interval": "80% central; quantile models q0.1..q0.9; sigma=(q90-q10)/2.563 for CRPS",
        "notes": ["holdout = last 12 observations (positional), as in the earlier cached benchmark and the Kaggle Nexus run",
                  "USA|washington dc carbon has missing months: positional holdout spans more calendar months",
                  "xLSTM has no predictive distribution: PICP/CRPS are NaN for it by design",
                  "water not forecast: future water = Aqueduct scenarios (see docs/methodology.md)"],
    }, open(m / "run_manifest.json", "w"), indent=2)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (m / "env_freeze_core_python311.txt").write_text(freeze, encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    fmt = lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def summary():
    M = pd.read_csv(RUN / "metrics" / "per_model_per_region.csv")
    h = headline(M)
    cols = ["model", "resource", "window", "region", "n_series", "RMSE", "MASE", "WMAPE_pct", "PICP", "CRPS"]
    cols = [c for c in cols if c in h.columns]
    L = [f"# {RUN.name}", "", "Account zones only (India 9, US 36, Europe 21); xLSTM = mean of 5 seeds. "
         "Generated from metrics/per_model_per_region.csv.", ""]
    for (res, win), g in h.groupby(["resource", "window"]):
        L += [f"## {res}, {win}", "", md_table(g[cols].sort_values(["region", "RMSE"])), ""]
    cc = RUN / "metrics" / "cache_check.csv"
    if cc.exists():
        L += ["## Check against the earlier cached benchmark (holdout)", "", md_table(pd.read_csv(cc)), ""]
    fp = RUN / "metrics" / "failures.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        L += ["## Failures", "", md_table(f.groupby(["model", "resource"]).size().rename("n").reset_index()), ""]
    (RUN / "report").mkdir(parents=True, exist_ok=True)
    (RUN / "report" / "RUN_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    RUN.mkdir(parents=True, exist_ok=True)
    if cmd == "run":
        t0 = time.time()
        S = load_series()
        print(f"series: { {r: len(v) for r, v in S.items()} } -> {RUN}", flush=True)
        run_stat(S); run_foundation(S); run_xlstm(S)
        score(); cache_check(); manifest(t0); summary()
        print("DONE", flush=True)
    elif cmd == "score":
        score(); cache_check(); summary()
    elif cmd == "score-nexus":
        score_nexus(sys.argv[2:]); cache_check(); summary()
