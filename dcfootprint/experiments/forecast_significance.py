"""Forecast benchmark: scale-free skill and significance against seasonal naive.

Answers three reviewer questions about Table 8 (experiments/forecast_exhibit.py):

1. Pooled RMSE is dominated by the largest series (US national electricity is orders of magnitude
   above a small state), so "skill" there mostly measures a few series. Here skill is scale-free:
   per series, relative RMSE r_s = RMSE_model / RMSE_naive on the same points, and
       skill_gm = 1 - geometric_mean_s(r_s)
   with a 90% bootstrap interval over series.
2. Is the difference significant? A per-series Diebold-Mariano test has little power with 12 to 36
   points per series, and the 12 horizons from one origin are serially dependent. The test is
   therefore ACROSS series: a paired Wilcoxon signed-rank test of log(r_s^2) = log(MSE ratio)
   against 0, Holm-corrected across the models of each resource x window.
3. The methodology replaces failed SARIMA fits with seasonal naive, but the benchmark dropped them
   (sarima_dcf completes 57 and 47 of 66 electricity series). `sarima_dcf_fallback` applies the stated
   rule: every missing point takes the seasonal-naive forecast.

xLSTM: the squared error of each point is averaged over its five seeds (the expected loss of one
trained network). Reads runs/forecast_benchmark_2026-09-28/forecasts; writes
dcfootprint/results/round7_fixes/forecast_significance.{csv,md}.

    PYTHONPATH=dcfootprint/src python dcfootprint/experiments/forecast_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_exhibit import KEY, load

from dcfootprint.io.facilities import _repo_root

OUT = _repo_root() / "dcfootprint" / "results" / "round7_fixes"
SERIES = ["resource", "window", "zone_id"]


def squared_errors(F: pd.DataFrame) -> pd.DataFrame:
    """[model, *KEY, se] with xLSTM averaged over seeds and the SARIMA fallback variant added."""
    F = F.assign(se=(F["mean"] - F["actual"]) ** 2)
    se = F.groupby(["model", *KEY], as_index=False)["se"].mean()          # xLSTM: mean over seeds
    naive = se[se["model"] == "seasonal_naive"].set_index(KEY)["se"]
    sar = se[se["model"] == "sarima_dcf"].set_index(KEY)["se"]
    fb = naive.copy()
    fb.loc[sar.index.intersection(naive.index)] = sar
    fallback = fb.rename("se").reset_index().assign(model="sarima_dcf_fallback")
    return pd.concat([se, fallback], ignore_index=True)


def series_ratios(se: pd.DataFrame) -> pd.DataFrame:
    """Per model x series: MSE of the model and of naive on the SAME points, and their ratio."""
    naive = se[se["model"] == "seasonal_naive"].set_index(KEY)["se"].rename("se_naive")
    m = se[se["model"] != "seasonal_naive"].join(naive, on=KEY, how="inner")
    g = m.groupby(["model", *SERIES]).agg(mse=("se", "mean"), mse_naive=("se_naive", "mean"), n=("se", "size"))
    g = g[(g["mse_naive"] > 0) & (g["mse"] > 0)]
    g["log_mse_ratio"] = np.log(g["mse"] / g["mse_naive"])
    return g.reset_index()


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def summarise(r: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (res, win, model), g in r.groupby(["resource", "window", "model"]):
        x = g["log_mse_ratio"].to_numpy()
        # geometric mean of relative RMSE = exp(mean(log MSE ratio) / 2)
        skill = 100 * (1 - np.exp(x.mean() / 2))
        boot = [100 * (1 - np.exp(rng.choice(x, len(x)).mean() / 2)) for _ in range(n_boot)]
        p = wilcoxon(x).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan
        rows.append({"resource": res, "window": win, "model": model, "n_series": len(x),
                     "skill_gm_pct": round(skill, 1),
                     "skill_gm_p05": round(float(np.percentile(boot, 5)), 1),
                     "skill_gm_p95": round(float(np.percentile(boot, 95)), 1),
                     "series_better_than_naive": int((x < 0).sum()),
                     "wilcoxon_p": p})
    t = pd.DataFrame(rows)
    t["wilcoxon_p_holm"] = np.nan
    for idx in t.groupby(["resource", "window"]).groups.values():
        p = t.loc[idx, "wilcoxon_p"].to_numpy(float)
        ok = np.isfinite(p)
        adj = np.full_like(p, np.nan)
        adj[ok] = _holm(p[ok])
        t.loc[idx, "wilcoxon_p_holm"] = adj
    t["significant_5pct"] = t["wilcoxon_p_holm"] < 0.05
    t["wilcoxon_p"] = t["wilcoxon_p"].round(4)
    t["wilcoxon_p_holm"] = t["wilcoxon_p_holm"].round(4)
    return t.sort_values(["resource", "window", "skill_gm_pct"], ascending=[True, True, False]).reset_index(drop=True)


def md(t: pd.DataFrame) -> str:
    L = ["# Forecast skill: scale-free and tested across series", "",
         ("Skill = 1 - geometric mean over series of RMSE_model / RMSE_naive (same points), in %. "
         "90% interval: bootstrap over series. p: paired Wilcoxon signed-rank test of the per-series log MSE ratio, "
         "Holm-corrected within each resource x window. `sarima_dcf_fallback` replaces failed fits with seasonal naive, "
         "the rule the methodology states. Generated by `dcfootprint/experiments/forecast_significance.py`."), ""]
    for (res, win), g in t.groupby(["resource", "window"], sort=False):
        L += [f"## {res}, {win}", "", "| model | series | skill (gm) | 90% interval | series better than naive | p (Holm) |",
              "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            star = " *" if r.significant_5pct else ""
            p = "n/a" if pd.isna(r.wilcoxon_p_holm) else f"{r.wilcoxon_p_holm:.3g}{star}"
            L.append(f"| `{r.model}` | {r.n_series} | {r.skill_gm_pct:+.1f}% | {r.skill_gm_p05:+.1f} to {r.skill_gm_p95:+.1f} | "
                     f"{r.series_better_than_naive}/{r.n_series} | {p} |")
        L.append("")
    L.append("\\* significant at 5% after Holm correction.")
    return "\n".join(L) + "\n"


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    t = summarise(series_ratios(squared_errors(load())))
    t.to_csv(OUT / "forecast_significance.csv", index=False)
    (OUT / "forecast_significance.md").write_text(md(t), encoding="utf-8")
    return t


if __name__ == "__main__":
    print(main().to_string(index=False))
