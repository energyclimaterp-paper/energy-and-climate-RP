# experiments/

Scripts outside the main pipeline: verification checks, tables for the paper and the forecasting benchmark.
None is needed to produce the account. Run them from the repository root, for example
`PYTHONPATH=dcfootprint/src python dcfootprint/experiments/method_rerank.py`.

## Verification

| Script | What it does |
|---|---|
| `determinism_check.py` | compares two clean runs file by file (writes `DETERMINISM.md`) |
| `spotcheck_trace.py` | recomputes three Indian facilities from the raw files without the pipeline's functions (writes `SPOTCHECK.md`) |
| `compare_results.py <dir_a> <dir_b>` | compares two results folders file by file, CSVs numerically |
| `gadm_gem_state_check.py` | checks GEM power-plant state labels against GADM boundaries |
| `epoch_crosscheck.py` | US coverage of the Epoch AI frontier datacenter list (`us/epoch_crosscheck.md`) |

## Paper tables and inputs

| Script | What it does |
|---|---|
| `method_rerank.py` | physical vs scarcity-weighted ranking (paper Table 5) → `tables/method_rerank.csv` |
| `extract_eu_tables.py` | rebuilds `config/eu_member_state_2023.csv` from the EED report PDF and checks it against the report's totals |

## Forecasting benchmark (paper Section 4.9, Table 8, Appendix C)

| Script | What it does |
|---|---|
| `forecast_metrics.py` | the 10 metrics (MAE, RMSE, WMAPE, bias, MASE, sMAPE, MedAE, Pearson r, PICP, CRPS) |
| `forecast_benchmark.py` | runs every model on one grid and one set of windows → `runs/forecast_benchmark_2026-09-28/` |
| `score_nexus_llm.py` | scores the LLM forecaster's runs → `runs/nexus_llm_2026-09-28/` |
| `forecast_exhibit.py` | builds `results/forecast_simple_vs_complex.{md,csv}` from the benchmark run |

The benchmark needs a Python environment with torch, chronos-forecasting and timesfm, and reads its input series
from the preliminary experiment's series file (`DCF_GAT_DIR`), which is not part of this release. The scored
outputs are included in `runs/`.
