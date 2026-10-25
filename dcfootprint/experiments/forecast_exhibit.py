"""Forecasting exhibit "does complexity pay?" -> results/forecast_simple_vs_complex.{md,csv}.

Generated from files, nothing typed by hand (replaces the hand-assembled first version of
2026-10-03, which reported the holdout window only while calling it a rolling backtest, labelled
the per-zone SARIMA as hierarchical/MinT, read MASE < 1 as "beats seasonal naive", and dropped
sarima_dcf from the electricity table).

Inputs
  runs/forecast_benchmark_2026-09-28/forecasts/   every model, both windows (66 account zones used)
  results/forecast_ci_method_backtest.csv          India one-step hierarchical backtest (Q3 forecast)
  data/ember/ci_method_backtest_us_2024_state_v3-validfit.csv   the same for the US (cache file)
Method (fixed before generating)
  - account zones only (India 9, US 36, EU 21), windows: holdout (last 12) and rolling (3 origins)
  - PAIRED skill vs seasonal naive: each model is compared with seasonal naive on exactly the
    (zone, window, origin, month) points that model forecast, so a model with failed fits is not
    flattered or penalised by a different series set. skill = 1 - RMSE_model / RMSE_naive (same points)
  - xLSTM: 5 seeds scored separately; mean and SD reported
  - MASE is scaled by each series' in-sample seasonal-naive error, so MASE < 1 is NOT "beats seasonal
    naive out of sample"; the out-of-sample comparison is the skill column

    .venv/Scripts/python.exe dcfootprint/experiments/forecast_exhibit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from forecast_metrics import metrics  # noqa: E402

REPO = HERE.parents[1]
RUN = REPO / "runs" / "forecast_benchmark_2026-09-28"
RES = REPO / "dcfootprint" / "results"
HIER = {"India": RES / "forecast_ci_method_backtest.csv",
        "US": REPO / "data" / "ember" / "ci_method_backtest_us_2024_state_v3-validfit.csv"}
KEY = ["resource", "zone_id", "window", "origin", "h"]
CLASS = {"seasonal_naive": "simple: seasonal naive",
         "sarima_dcf": "classical: per-zone SARIMA(1,1,1)(1,0,1,12) with fit checks",
         "sarima_core": "classical: per-zone SARIMA(1,1,1)(1,1,0,12), preliminary-experiment notebook",
         "timesfm_2p5": "complex: foundation model (TimesFM 2.5)",
         "chronos_2": "complex: foundation model (Chronos-2)",
         "xlstm": "complex: deep net (xLSTM, 5 seeds)",
         "nexus_llm": "complex: LLM multi-agent Nexus (gemma4:26b, median of 3 repeats; holdout only)"}


def load() -> pd.DataFrame:
    F = pd.concat([pd.read_csv(f) for f in sorted((RUN / "forecasts").glob("*/*.csv"))], ignore_index=True)
    F = F[F["account_zone"].astype(bool) & F["model"].isin(CLASS)]
    return F.dropna(subset=["actual", "mean"])


def table(F: pd.DataFrame) -> pd.DataFrame:
    naive = F[F["model"] == "seasonal_naive"].set_index(KEY)
    rows = []
    for (res, win, model), g in F.groupby(["resource", "window", "model"]):
        per_seed = []
        for seed, gs in (g.groupby("seed") if model == "xlstm" else [(None, g)]):
            m = metrics(gs)
            pts = gs.set_index(KEY).index.intersection(naive.index)
            nv = naive.loc[pts]
            rm_model = float(np.sqrt(np.mean((gs.set_index(KEY).loc[pts, "mean"] - gs.set_index(KEY).loc[pts, "actual"]) ** 2)))
            rm_naive = float(np.sqrt(np.mean((nv["mean"] - nv["actual"]) ** 2)))
            per_seed.append({**m, "n_series": gs["zone_id"].nunique(), "n_windows": gs.groupby(["zone_id", "origin"]).ngroups,
                             "RMSE_naive_same_points": rm_naive, "skill_vs_naive_pct": 100 * (1 - rm_model / rm_naive)})
        d = pd.DataFrame(per_seed)
        row = {"resource": res, "window": win, "model": model, "class": CLASS[model],
               "n_series": int(d["n_series"].iloc[0]), "n_windows": int(d["n_windows"].iloc[0])}
        for c in ("RMSE", "MASE", "WMAPE_pct", "PICP", "CRPS", "RMSE_naive_same_points", "skill_vs_naive_pct"):
            row[c] = float(d[c].mean()) if c in d else np.nan
        row["RMSE_sd_over_seeds"] = float(d["RMSE"].std()) if len(d) > 1 else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    # rank by PAIRED skill (same points as seasonal naive): raw RMSE is not comparable across models
    # that completed different windows (sarima_dcf loses electricity windows to its fit checks)
    return t.sort_values(["resource", "window", "skill_vs_naive_pct"], ascending=[True, True, False]).reset_index(drop=True)


def hierarchical() -> pd.DataFrame:
    out = []
    for region, p in HIER.items():
        if p.exists():
            out.append(pd.read_csv(p).assign(region=region, source=str(p.relative_to(REPO)).replace("\\", "/")))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def md(t: pd.DataFrame) -> str:
    def f(v, nd=1):
        return "" if pd.isna(v) else f"{v:.{nd}f}"
    L = ["| rank (by skill) | model | class | series / windows | RMSE | skill vs naive (same points) | MASE | PICP |", "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(t.itertuples(), 1):
        rmse = f(r.RMSE) + (f" ± {f(r.RMSE_sd_over_seeds)}" if pd.notna(r.RMSE_sd_over_seeds) else "")
        L.append(f"| {i} | `{r.model}` | {r._4} | {r.n_series} / {r.n_windows} | {rmse} | {f(r.skill_vs_naive_pct)}% | "
                 f"{f(r.MASE, 2)} | {f(r.PICP, 2)} |")
    return "\n".join(L)


def main():
    t = table(load())
    h = hierarchical()
    t.to_csv(RES / "forecast_simple_vs_complex.csv", index=False)
    if len(h):
        h.to_csv(RES / "forecast_hierarchical_onestep.csv", index=False)
    g = lambda res, win, model, c: float(t[(t.resource == res) & (t.window == win) & (t.model == model)][c].iloc[0])
    L = ["# Forecasting: does complexity pay?", "",
         "Answers two questions: do simple forecasters match compute-heavy deep networks, and does a",
         "non-deep-learning hierarchical method help? Generated by `dcfootprint/experiments/forecast_exhibit.py`",
         "from `runs/forecast_benchmark_2026-09-28` (66 account-zone series: India 9 · US 36 · EU 21; 10 metrics,",
         "80% intervals). **Holdout** = the last 12 months of each series (66 windows); **rolling** = origins",
         "Dec 2022 / Dec 2023 / Dec 2024, 12 months each (189 windows, the sturdier test).", "",
         "**How to read it.** *Skill* = how much lower the model's RMSE is than seasonal naive's on exactly the same",
         "points (positive = better than naive). MASE is scaled by each series' *in-sample* naive error, so MASE < 1",
         "does not mean \"beats seasonal naive\" here; the out-of-sample seasonal naive itself scores MASE "
         f"{g('carbon', 'holdout', 'seasonal_naive', 'MASE'):.2f} (carbon) and {g('electricity', 'holdout', 'seasonal_naive', 'MASE'):.2f} (electricity).", ""]
    for res, unit in (("carbon", "gCO2/kWh"), ("electricity", "GWh/month")):
        for win in ("rolling", "holdout"):
            L += [f"## {res.capitalize()} ({unit}), {win}", "", md(t[(t.resource == res) & (t.window == win)]), ""]
    if len(h):
        L += ["## The non-DL hierarchical methods R2 asked for (one step ahead, carbon intensity)", "",
              "Bottom-up and MinT reconciliation are benchmarked in the Q3 forecast layer (`project/hierarchy.py`):",
              "one-step state carbon intensity, chosen on 2022-2023 and checked out of sample in 2024 (RMSE, g/kWh).", "",
              "| region | method | backtest 2022-23 | 2024 out of sample | chosen |", "|---|---|---|---|---|"]
        for r in h.itertuples():
            L.append(f"| {r.region} | `{r.method}` | {r.rmse_backtest_mean:.1f} | {r.rmse_2024_out_of_sample:.1f} | {'yes' if r.chosen else ''} |")
        L += ["", f"Sources: {', '.join(sorted(h['source'].unique()))}.", ""]
    cr, ch = g("carbon", "rolling", "sarima_dcf", "skill_vs_naive_pct"), g("carbon", "holdout", "sarima_dcf", "skill_vs_naive_pct")
    er_c, er_t = g("electricity", "rolling", "chronos_2", "skill_vs_naive_pct"), g("electricity", "rolling", "timesfm_2p5", "skill_vs_naive_pct")
    eh_c = g("electricity", "holdout", "chronos_2", "skill_vs_naive_pct")
    xr, xe = g("carbon", "rolling", "xlstm", "skill_vs_naive_pct"), g("electricity", "rolling", "xlstm", "skill_vs_naive_pct")
    L += ["## Findings", "",
          f"1. **The deep net is the worst model** in both windows and both resources (rolling skill: carbon {xr:+.0f}%, "
          f"electricity {xe:+.0f}%): compute-heavy deep learning does not pay here.",
          f"2. **Carbon intensity, the quantity the account uses, is best forecast by a classical per-zone SARIMA** "
          f"(`sarima_dcf`): {cr:.0f}% below seasonal naive on the rolling windows, {ch:.0f}% on the holdout. Forecasting does "
          "beat seasonal naive for carbon, by a modest but consistent margin.",
          f"3. **Electricity favours the foundation models on the sturdier test:** on the rolling windows Chronos-2 "
          f"({er_c:.0f}%) and TimesFM ({er_t:.0f}%) beat seasonal naive, while on the single holdout seasonal naive is "
          f"best (Chronos-2 {eh_c:+.0f}%). One 12-month window is not enough to call a winner.",
          "4. **The hierarchical (bottom-up / MinT) methods give no robust gain at one step:** in India seasonal naive",
          "   is chosen (bottom-up and MinT are worse on 2022-23), in the US bottom-up is chosen by a small margin and",
          "   seasonal naive is best out of sample in 2024. They answer R2 at the horizon Q3 uses.",
          f"5. **The real LLM agent (Nexus, gemma4:26b) is no better:** holdout skill vs seasonal naive "
          f"{g('carbon', 'holdout', 'nexus_llm', 'skill_vs_naive_pct'):+.0f}% for carbon (last of all models) and "
          f"{g('electricity', 'holdout', 'nexus_llm', 'skill_vs_naive_pct'):+.0f}% for electricity, and its repeats disagree "
          "(`runs/nexus_llm_2026-09-28/`).",
          "6. **The GNN adds nothing** (preliminary experiment: graph structure does not beat degree-matched random controls; Appendix B of the paper).", "",
          "## What this means for the paper",
          "- Forecasting is a **supporting projection layer, not a contribution.** Carbon: classical SARIMA; electricity:",
          "  a foundation model or seasonal naive; the deep net is reported as the negative result.",
          "- The resource/geography **coupling lives in the account and the optimisation** (incidence matrices, basin",
          "  queues, routing), as R1 suggested, not in a joint or graph forecaster.",
          "- The two most compute-heavy models (xLSTM and the LLM agent) are the two weakest.", "",
          "*Data: `results/forecast_simple_vs_complex.csv` (this table), `results/forecast_hierarchical_onestep.csv` (hierarchical).*"]
    (RES / "forecast_simple_vs_complex.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
