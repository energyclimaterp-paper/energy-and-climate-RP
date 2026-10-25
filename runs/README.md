# runs/: forecasting benchmark (frozen)

Forecasting is a supporting layer in the paper (Section 4.9, Table 8, Appendix C), not a contribution. These
folders hold the scored runs behind it. They are frozen: a new run gets a new dated folder.

| Folder | What it is |
|---|---|
| [`forecast_benchmark_2026-09-28/`](forecast_benchmark_2026-09-28/) | seasonal naive, two SARIMA specifications, Chronos-2, TimesFM 2.5 and xLSTM (5 seeds), run fresh on one grid with one set of windows and 10 metrics |
| [`nexus_llm_2026-09-28/`](nexus_llm_2026-09-28/) | the LLM multi-agent forecaster (gemma4:26b, 3 repeats, run on a Kaggle GPU), scored with the same rules |

The paper's comparison (66 account-zone series: 9 India, 36 US, 21 EU; 12-month holdout and rolling origins in
December 2022, 2023 and 2024, 189 windows) is summarised in
`dcfootprint/results/round5_final/forecast_simple_vs_complex.md`.

**Main result:** a classical per-zone SARIMA forecasts carbon intensity best (about 11% lower RMSE than seasonal
naive). The two most compute-intensive models, xLSTM and the LLM forecaster, are the two weakest.

**Inputs:** the series come from the preliminary graph-based experiment's series file, which is not part of this
release. The scripts are in `dcfootprint/experiments/` (`forecast_benchmark.py`, `score_nexus_llm.py`,
`forecast_exhibit.py`).
