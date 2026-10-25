# Real (LLM) Nexus — Kaggle GPU run (2026-09-28), scored 2026-10-12

The multi-agent Nexus forecaster (gemma4:26b via Ollama, the pipeline in preliminary-experiment `src/nexus_ollama.py`) run for
real for the first time on the full benchmark grid. Until now the cache's "Nexus" column was Chronos-2 in the Nexus
slot; this is the LLM version.

## What was run

| | |
|---|---|
| Where | Kaggle, 2 × Tesla T4, Ollama 0.34.4; notebook = `kaggle_notebook_nexus-llm-ollama-gemma4-kaggle-gpu_v2.ipynb` (version 2, executed; also on branch `nexus`) |
| Model | `gemma4:26b`, digest `08ae7ec1744b` (checked equal to the local CPU runs), temperature 0, seed 42, thinking off, 48 history points |
| Series | preliminary-experiment `work/region_series_dump.json`: electricity 95 zones, carbon 95 zones, water 24 basins; holdout = last 12 observations (the same as the earlier cached benchmark and the fresh runs) |
| Repeats | **3 per series.** The GPU output is not deterministic even at temperature 0 (the same 3 basins differed by up to 88 between two runs on one GPU), so the forecast is the per-month **median** of the repeats; the spread is reported |
| Wall clock | about 32 s per series per repeat, 5.75 h in all (one 10 h budget, checkpoints to a private Kaggle dataset) |
| Completeness | 95 / 95 / 24 series, 3 repeats each, 12 months each, no missing values |
| Fallbacks | 2 water repeats fell back to the non-LLM forecast (Columbia River repeat 1, Godavari repeat 3); those repeats are dropped and the median is taken over the other two. None for electricity or carbon |

## Results

**Real LLM Nexus never beats Chronos-2 or TimesFM, in any region or resource, and is last for carbon on the account zones and last for water.**

On the 66 account zones, against every other model run fresh on the same points
(`runs/forecast_benchmark_2026-09-28/report/RUN_SUMMARY.md`, holdout window):

| | best | seasonal naive | xLSTM | **Nexus (LLM)** | Nexus skill vs naive (same points) |
|---|---|---|---|---|---|
| carbon RMSE (g/kWh) | 37.2 (`sarima_dcf`) | 41.7 | 46.3 | **46.8** | −12% (last of 7) |
| electricity RMSE (GWh/month) | 821.9 (seasonal naive) | 821.9 | 1,089.3 | **1,015.5** | −24% (5th of 7; ahead of xLSTM and of `sarima_dcf` on its 47 zones) |

On all zones, against the earlier cached benchmark's four models (`metrics/*_vs_core_cached_per_region.csv`; rank of 5): carbon Europe
58.8 (4th; Chronos-2 48.5), India 59.4 (3rd; Chronos-2 55.1), US 36.4 (3rd; Chronos-2 31.0); electricity Europe 1,235
(4th; TimesFM 915), India 1,005 (4th), US 724 (4th); water 69.3 (5th; SARIMA 40.8).

**Repeats** (`metrics/*_repeat_variability.csv`): the median of the 3 repeats scores better than any single repeat
(e.g. carbon Europe 58.8 vs 59.4–62.7 per repeat); single forecasts can be wild (largest spread between repeats:
21,250 GWh in one European electricity month). Nexus is among the weakest models here and the only one whose forecasts change from run to run.

**Consistency check:** the same 3 water basins re-run on the GPU gave RMSE 38.0 / 38.3 / 60.5 against 78.7 / 41.8 /
57.6 on the local CPU (`inputs/check_summary.csv`), in line with the non-determinism above.

## Files

```
inputs/nexus_{electricity,carbon,water}_ollama.csv   the Kaggle output, unmodified (region_id, rep, tag, timestamp, value, actual, seconds)
inputs/check_*.csv                                    the GPU-vs-CPU consistency check (two runs)
forecasts/nexus-llm-gemma4-kaggle-gpu/<resource>/     median-over-repeats forecasts, as scored
metrics/*_vs_core_cached_per_region.csv | _per_zone.csv   Nexus vs the earlier cached benchmark (xLSTM, TimesFM 2.5, Chronos-2, SARIMA)
metrics/*_per_repeat_per_region.csv                  each repeat scored alone
metrics/*_repeat_variability.csv                     RMSE / MASE range across repeats, forecast spread
kaggle_notebook_..._v2.ipynb                         the executed Kaggle notebook (settings, checks, logs)
```
The Ollama server log (18 MB, in the Kaggle output) is not kept here.

## Reproduce the scoring

```bash
.venv/Scripts/python.exe dcfootprint/experiments/score_nexus_llm.py runs/nexus_llm_2026-09-28 nexus-llm-gemma4-kaggle-gpu \
    runs/nexus_llm_2026-09-28/inputs/nexus_electricity_ollama.csv runs/nexus_llm_2026-09-28/inputs/nexus_carbon_ollama.csv \
    runs/nexus_llm_2026-09-28/inputs/nexus_water_ollama.csv
cd dcfootprint/experiments && <python with torch, chronos and timesfm> forecast_benchmark.py score-nexus \
    ../../runs/nexus_llm_2026-09-28/inputs/nexus_electricity_ollama.csv ../../runs/nexus_llm_2026-09-28/inputs/nexus_carbon_ollama.csv
```
Note on fallbacks: `score_nexus_llm.py` drops the fallback **repeats** (its docstring says series); `forecast_benchmark.py
score-nexus` would drop the whole series. Only water had fallbacks, and only `score_nexus_llm.py` scores water, so the
two agree on every number above.
