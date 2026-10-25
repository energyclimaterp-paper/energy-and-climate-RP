# Fresh forecasting run — benchmark grid (2026-09-28)

Every forecasting model run fresh on **one grid, one set of windows, one metric set**, with prediction
intervals, so the models can be compared like for like. This is the grid the real (LLM) Nexus runs on
(Kaggle); its scored results are in [`../nexus_llm_2026-09-28/`](../nexus_llm_2026-09-28/). Numbers: [`report/RUN_SUMMARY.md`](report/RUN_SUMMARY.md).

## Why this run, and why this grid

The research questions (`docs/methodology.md`) need **grid carbon
intensity** — and electricity as its component — forecast per account zone and backtested at a **12-month
horizon against seasonal naive**. The benchmark grid's 95 zones contain every zone of the account (India 9 states,
US 36 states, EU 21 countries). Left out on purpose: **water** (the paper uses Aqueduct 2030/2050 scenarios,
not forecasts) and **5-year "forecasts" to 2030** (a scenario, per the architecture's risk register).

## Design (fixed before running)

| | |
|---|---|
| Series | preliminary-experiment `work/region_series_dump.json` (read-only): electricity (GWh) and carbon intensity (gCO2/kWh), 95 zones each |
| Windows | **holdout**: the last 12 observations of each series (same as the earlier cached benchmark and the Kaggle Nexus run) · **rolling**: origins Dec 2022, Dec 2023, Dec 2024, horizon 12 (robustness) |
| Intervals | 80% central. TimesFM / Chronos-2: their q0.1..q0.9; sigma = (q90 − q10) / 2.563 for CRPS |
| Metrics | MAE, RMSE, WMAPE, Bias, MASE, sMAPE, MedAE, Pearson r, PICP, CRPS (`dcfootprint/experiments/forecast_metrics.py`) |
| Levels | region (India / US / Europe / All) × all zones / account zones |

| Model | What it is | Intervals |
|---|---|---|
| `seasonal_naive` | y(t−12); sigma from its own errors over the 12 months before the origin | yes |
| `sarima_dcf` | the pipeline's SARIMA(1,1,1)(1,0,1,12) **with the fit checks** (converged in 500 iterations, no coefficient on the unit boundary, finite SE); a rejected fit is a recorded failure, not replaced | yes |
| `sarima_core` | the preliminary-experiment notebook's `fit_sarima`, verbatim ((1,1,1)(1,1,0,12), unconstrained): reproduces the cache | yes |
| `timesfm_2p5` | google/timesfm-2.5-200m-pytorch, notebook configuration; point = median | yes |
| `chronos_2` | amazon/chronos-2 via preliminary-experiment `src/nexus_local.py`; point = median | yes |
| `xlstm` | the notebook's xLSTM, verbatim (lookback 12, hidden 64, 2 layers, 100 epochs), **5 seeds (42–46)**, seed set per series | no (no predictive distribution): PICP/CRPS blank; seed spread in `metrics/xlstm_seed_spread.csv` |
| `nexus_llm` | real multi-agent Nexus (gemma4:26b, Kaggle), **added 2026-10-12** from `../nexus_llm_2026-09-28/inputs/`; holdout only; median over 3 repeats (no electricity/carbon fallbacks) | no |

## Checks

- **Reproduces the earlier cached benchmark:** fresh `sarima_core`, `timesfm_2p5` and `chronos_2` holdout forecasts equal the
  cache point for point (`metrics/cache_check.csv`). xLSTM differs by design: the cache set seed 42 once for the
  whole run (order-dependent); here each series gets its own seed.
- **Script version:** the committed `forecast_benchmark.py` adds, after this run, the second-grid option
  (`SERIES_DUMP`) and the Nexus repeat rules. The forecasting code is unchanged: re-scoring this run with the
  committed script reproduces every file in `metrics/` and `report/` byte for byte.
- Failures are listed, never silently filled: `metrics/failures.csv`. Most are `sarima_dcf` on electricity, where
  strongly seasonal generation series push the seasonal coefficient onto the unit boundary; the same pattern was in
  the 2026-09-27 run (71 electricity vs 5 carbon rejections). Because failed windows are not replaced, compare
  models on `n_series` / `n_windows` as well as the scores.

## Caveats

- Holdout = last 12 **observations**. `USA|washington dc` carbon has missing months, so its holdout spans more
  than 12 calendar months; the calendar-based models (`seasonal_naive`, `sarima_dcf`) leave the months beyond
  h = 12 empty there.
- Rolling windows need 12 full months after the origin, so zones whose data ends before Dec 2025 drop out of the
  Dec 2024 origin (82 of 95).
- The cache's "Nexus" column is **Chronos-2 in the Nexus slot**; here it is labelled `chronos_2`. Real Nexus is `nexus_llm`.

## Layout

```
forecasts/<model>/<resource>[_seed<N>].csv   one row per zone x window x horizon month:
                                              actual, mean, sigma, lo, hi, mase_scale, seed
metrics/per_model_per_region.csv              all metrics per model x resource x window x region x level (per seed for xLSTM)
metrics/per_model_per_zone.csv                the same per zone and origin
metrics/xlstm_seed_spread.csv                 xLSTM mean and SD across the 5 seeds
metrics/cache_check.csv                       fresh vs earlier cached benchmark
metrics/failures.csv                          every failed fit, with the reason
manifest/run_manifest.json                    commits, file hashes, settings, wall clock
manifest/env_freeze_core_python311.txt        pip freeze of the Python used
report/RUN_SUMMARY.md                         headline tables (account zones), generated from metrics/
run.log                                       console log of the run
```

## Reproduce

```bash
cd dcfootprint/experiments
<python with torch, chronos and timesfm> forecast_benchmark.py run
```
Needs the preliminary-experiment repository next to this one (`../diff/gat-sarima-nexus`, or set `DCF_GAT_DIR`), and a Python with
torch, chronos-forecasting, timesfm, statsmodels, scikit-learn (see the environment freeze). The run resumes:
finished model files are skipped. Add Nexus with
`forecast_benchmark.py score-nexus nexus_electricity_ollama.csv nexus_carbon_ollama.csv`.
