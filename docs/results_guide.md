# Results guide: paper → files

Every number in the paper comes from the canonical snapshot
[`dcfootprint/results/round5_final/`](../dcfootprint/results/round5_final/) (below: `R5/`), unless another file is
named. India files sit at the top of `R5/`; US and EU files are in `R5/us/` and `R5/eu/`.

## Tables

| Paper | Content | Source |
|---|---|---|
| Table 1 | Datasets | [`docs/data_sources.md`](data_sources.md) |
| Table 2 | Account parameters | `dcfootprint/config/parameters.yaml` |
| Table 3 | Routing results against a static allocation | `R5/routing_comparison.csv`, `R5/us/routing_comparison.csv` |
| Table 4 | The account by region | `R5/RESULTS_FULL.md`, `R5/us/RESULTS_FULL.md`, `R5/eu/RESULTS_FULL.md`, `R5/india_account_summary.csv`, `R5/us/us_account_summary.csv`, `R5/eu/eu_account_summary.csv` |
| Table 5 | Physical vs scarcity-weighted ranking | [`tables/method_rerank.csv`](../tables/method_rerank.csv) (`experiments/method_rerank.py`) |
| Table 6 | Mandates by axis and region | `R5/regulation_matrix.csv`, `R5/policy_registry.csv` |
| Table 7 | Lever savings | `R5/lever_savings.csv`, `R5/us/lever_savings.csv`, `R5/eu/lever_savings.csv` |
| Table 8 | Forecast skill vs seasonal naive | `R5/forecast_simple_vs_complex.md` / `.csv` |
| Appendix A | Ember, Electricity Maps, Aqueduct and ATLAS coverage | raw inputs in `data/` (see `data/README.md`) |
| Appendix B | Preliminary graph-based forecasting experiment | reported in the paper only; not part of this release |
| Appendix C | LLM forecaster | [`runs/nexus_llm_2026-09-28/`](../runs/nexus_llm_2026-09-28/) |

## Figures

All figures are in [`figures/`](../figures/) and are redrawn by `figures/scripts/`.

| Paper | File | Data |
|---|---|---|
| Figure 1 | `fig1_framework.png` | diagram (`fig1_framework.py`) |
| Figure 2 | `fig2a_india_routing.png`, `fig2b_us_routing.png` | `R5/routing_*.csv`, `R5/us/routing_*.csv` |
| Figure 3 | `fig3_india_scarcity_map.png` | `R5/outputs/account_facility_month.parquet`, AWARE |
| Figure 4 | `fig4a_us_scarcity_map.png`, `fig4b_eu_scarcity_map.png` | `R5/outputs/account_us.parquet`, `account_eu.parquet` |
| Figure 5 | `fig5_levers.png` | `lever_savings.csv` (three regions) |
| Figure 6 | `fig6a_india_siting_map.png`, `fig6b_us_siting_map.png`, `fig6c_eu_siting_map.png` | `q1_siting.csv` (three regions) |

## Headline numbers

| Claim | Value | Source |
|---|---|---|
| Account totals | India 3.63 Mt CO₂, 847 M m³-eq; US 16.0 Mt, 2.51 bn m³-eq; EU 3.35 Mt CO₂e, 137 M m³-eq | `R5/cross_region_summary.md`, `*_account_summary.csv` |
| Off-site share of scarcity-weighted water | 60%–82% | `R5/cross_region_summary.md` |
| Routing, US | water −29.3%, carbon −10.2% | `R5/us/routing_comparison.csv`; `R5/ROUND5_REPORT.md` |
| Routing, India | water −21.7%, carbon +1.9%, peak overdraft 545,700 m³ | `R5/routing_comparison.csv`; `R5/ROUND5_REPORT.md` |
| India burden on over-exploited aquifers | 63% (536 of 846.6 M m³-eq) | `R5/q2_equity_groundwater.md` |
| Concentration | Tamil Nadu 36% of India's burden on 21% of capacity; Arizona 38% of the US burden on 6% | `R5/india_account_summary.csv`, `R5/us/us_account_summary.csv` |
| Mandates | 0 of 4 axes in every region | `R5/regulation_matrix.csv` |
| Levers, India | efficiency standard −36.6% water, −18.8% carbon; zero liquid discharge −36.9% water; coastal cooling −17.3% water | `R5/lever_savings.csv` |
| Levers, US | efficiency standard −31.4% water, −12.9% carbon; zero liquid discharge −29.3% water | `R5/us/lever_savings.csv` |
| Levers, EU | zero liquid discharge −16.4%; efficiency standard −6.9% | `R5/eu/lever_savings.csv` |
| Calibration and coverage | India vs CEEW/JLL, US vs LBNL, EU coverage | `R5/RESULTS_FULL.md`, `R5/us/calibration.json`, `R5/eu/calibration.json` |
| Uncertainty | Monte Carlo bands, Sobol indices, no-hydro variant | `R5/uncertainty.json`, `R5/us/uncertainty.json`, `R5/eu/uncertainty.json` |
| Seasonality | monthly profile of carbon and water | `R5/seasonal_profile.md` / `.csv` (and `us/`, `eu/`) |
| Siting | ranks and scenario ranks | `R5/q1_siting.csv`, `q1_siting_scenarios.csv`, `q1_siting_small_grids.csv` (each region) |
| Harm scorecard | flagged facilities and recommended lever | `R5/q2_scorecard.csv` (each region) |
| India carbon, generation basis | sensitivity to the carbon basis | [`results/generation_basis_check/`](../dcfootprint/results/generation_basis_check/) |
| US coverage of frontier AI campuses | Epoch AI cross-check | `R5/us/epoch_crosscheck.md` |
| Forecasting benchmark | all models, 66 series | `R5/forecast_simple_vs_complex.md`, [`runs/forecast_benchmark_2026-09-28/`](../runs/forecast_benchmark_2026-09-28/) |
| Retrieval check of the gap matrix | 15 documents, 1,056 passages | `R5/rag_axis_audit.md`, `R5/policy_corpus.csv` |

## Run evidence

| File | What it shows |
|---|---|
| `R5/DETERMINISM.md` | two clean runs identical file by file |
| `R5/SPOTCHECK.md` | three Indian facilities recomputed independently, 0 discrepancies |
| `R5/ROUND5_REPORT.md` | every result file compared with the previous snapshot under a rule fixed before the run |
| `R5/logs/` | console logs of both runs |

## Note on the policy registry

`dcfootprint/config/legal_constraints.csv` and `policy/registry.py` now carry a source link for every registry
entry, and the Minnesota entry describes the law as enacted. The frozen `R5/policy_registry.csv` predates these
edits. They change no number, legal limit or matrix cell.
