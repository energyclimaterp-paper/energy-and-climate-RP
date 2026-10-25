# round5_final: canonical snapshot (India, US, EU)

**Frozen. Every number in the paper comes from here.** Never edit these files. A later canonical run gets a new
folder. The paper's tables, figures and numbers are mapped to these files in
[`docs/results_guide.md`](../../../docs/results_guide.md).

## How it was produced (2026-10-15)

1. Inputs: the datasets in [`docs/data_sources.md`](../../../docs/data_sources.md), including the CGWB groundwater
   files, the CEA database, the Electricity Maps cache (India consumption-based carbon) and the official 2030/2050
   grid-carbon paths.
2. Test suite: all tests passed at the run commit.
3. Two clean runs of `python -m dcfootprint.pipeline` (India) and `python -m dcfootprint.regions US EU`, each
   started with no cached outputs or forecasts. India completed 14/14 stages, US and EU 18/18, in both runs.
   Logs: `logs/`.
4. Determinism: the two runs are identical (60 text files byte for byte, 6 parquet tables):
   [`DETERMINISM.md`](DETERMINISM.md).
5. Independent trace: three Indian facilities recomputed from the source files without the pipeline's own
   functions, with 0 discrepancies: [`SPOTCHECK.md`](SPOTCHECK.md).
6. Comparison with the previous snapshot: [`ROUND5_REPORT.md`](ROUND5_REPORT.md). The only method change was
   placing the 2030/2050 siting scenarios on official grid-carbon paths (India CEA National Electricity Plan, US
   NREL Cambium, EU Reference Scenario 2020). Every headline number equals the previous snapshot. The previous
   snapshot itself is not part of this release.

## Layout

| Path | Content |
|---|---|
| `cross_region_summary.md` | three-region synthesis, every number mapped to a research question (start here) |
| `RESULTS.md`, `RESULTS_FULL.md` | India run summary and full report |
| `*.csv`, `*.json`, `*.md` (top level) | India results and the cross-region exhibits |
| `us/`, `eu/` | US (facility level, 2024) and EU (country level, 2023) results |
| `outputs/` | account tables (India facility-month, US, EU) and interim files, as parquet |
| `logs/` | console logs of both runs |
| `DETERMINISM.md`, `SPOTCHECK.md`, `ROUND5_REPORT.md` | run evidence |

Figures are not part of the snapshot, since PNG output is not byte-stable. Every number in them is in a table
here, and `figures/scripts/` redraws them.

The carbon basis differs by region: India is consumption-based and life-cycle (Electricity Maps), while the US
and EU are generation-based and direct (Ember). Read `cross_region_summary.md` before comparing regions. The
India generation-basis run is in [`../generation_basis_check/`](../generation_basis_check/).
