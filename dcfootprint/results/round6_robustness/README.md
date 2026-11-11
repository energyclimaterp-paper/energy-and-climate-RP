# Round 6 robustness run

Five checks on the round-5 results, produced by `dcfootprint/experiments/robustness_round6.py`.
The round-5 snapshot is read, never edited.
Two clean runs of the script give byte-identical files.

India starts from the frozen round-5 account (`round5_final/outputs/`), so it needs neither the GEM workbook nor an Electricity Maps key.
Only scope-1 water is rebuilt there, which does not touch carbon or scope-2.
The US account is rebuilt from raw data.

## Headline in one table

`headline_steps.csv`: change vs the static allocation, with legal limits on (negative carbon = less carbon).

| Region | Step | Lyapunov carbon | Oracle carbon | Lyapunov scarcity water |
|---|---|---|---|---|
| India | 0 round 5 as published | +1.87% | +0.35% | -21.7% |
| India | 1 basis-consistent CI forecast | +1.08% | +0.35% | -22.6% |
| India | 2 + climate WUE (round 6 account) | +1.06% | +0.45% | -22.5% |
| US | 0 round 5 as published | -10.2% | -10.6% | -29.3% |
| US | 1 eGRID + EIA plant data | -2.9% | -3.7% | -30.1% |
| US | 2 + climate WUE (round 6 account) | -2.7% | -3.4% | -30.0% |

The India trade-off survives every check but is about half the published size.
The US co-benefit survives but is about a quarter of the published size.

## 1. Routing Monte Carlo (`*/routing_mc_*`)

The whole routing comparison is re-run for 1,000 draws of utilisation, PUE, WUE, EWIF, the hydro multiplier, a grid-CI multiplier and the demand seed (`uncertainty/routing_mc.py`).
The CI multiplier is drawn once per grid: zones with the identical CI series share one draw (India: the five Electricity Maps regional grids; US: eGRID subregions).
Its band (plus or minus 10%) is an assumption, set in `uncertainty.ci_zone_band`.

| | India lyapunov | India oracle | US lyapunov | US oracle |
|---|---|---|---|---|
| carbon change, mean | +0.98% | +0.45% | -2.83% | -3.56% |
| carbon change, 90% interval | +0.07 to +1.60% | -0.57 to +1.28% | -4.09 to -1.65% | -5.38 to -2.14% |
| draws with more carbon | 96% | 84% | 0% | 0% |
| scarcity water change, mean | -22.0% | -22.4% | -29.4% | -28.9% |

The largest first-order driver of the India carbon change is colocation utilisation (0.70): it sets how much headroom there is to move load into.

## 2. Validation against operator reports (`*/validation_*`)

Reported figures, each with its URL and PDF page, are in `dcfootprint/config/reported_footprints.csv`; the comparison is `validation/reported.py`.
Ratios are model / reported (1 = agreement).

| Operator | Scope | Electricity | On-site water | Water per facility kWh |
|---|---|---|---|---|
| Nxtra (FY2024-25) | fleet, 196 of 220 MW | 1.60 (1.80 per MW) | 5.65 | 3.52 |
| CtrlS (FY2023-24) | fleet, 154 vs 140 MW | 2.51 (2.27 per MW) | 2.25 | 0.90 |
| STT GDC India (FY2024-25) | fleet | 1.21 | 3.32 | 2.73 |
| Meta Altoona, Prineville (2024) | site | 1.47, 1.54 | 1.69, 1.42 | 1.15, 0.93 |
| Meta Gallatin, Mesa (2024) | site, ramping up | 5.5, 13.5 | 1.71, 1.02 | not meaningful |
| Google The Dalles (2024) | site | not reported | 1.31 (withdrawal), 1.67 (consumption) | |

What this says:

- Modelled electricity is 1.2 to 2.5 times what operators report.
  CtrlS reports 259 GWh of IT energy on 140+ MW, an IT load factor near 0.21, against the assumed 0.45.
  The utilisation assumption is the main bias in the absolute numbers.
- On-site water per kWh matches where cooling is evaporative (CtrlS 0.90, Meta 0.93 to 1.15) and is 2.7 to 3.5 times too high for fleets that report much lower WUE (Nxtra, STT GDC India).
  No climate curve can fix this: it is a cooling-technology difference.
- Nxtra's reported location-based grid factor (0.727 t/MWh) is 24% above the model's consumption-basis CI (0.586).
- Indian figures are fiscal years (April to March); the overlap with calendar 2024 is in `months_overlap_with_account_year`.

## 3. US on eGRID and EIA plant data (`us/`)

`region_config.US.grid_source: eia_egrid` (`io/us_grid.py`) replaces Ember state mix x Macknick coefficients.

- Zone: eGRID subregion (plants: eGRID's assignment; datacenters: the eGRID2023 subregion map).
- Carbon: EIA-923 2024 monthly plant generation x each plant's eGRID2023 CO2 output rate.
  On 2023 this reproduces eGRID's own subregion rates to a median 0.3% (largest gap 6.8%, AKMS): `us/egrid2023_reproduction.csv`.
- Power-plant water: EIA reported cooling-water consumption per plant-month; reporting gaps filled with the same technology x cooling-type intensity; Macknick only for plants that do not report.
- Scarcity: each plant's water at the AWARE CF of its own basin.

| US account (2024) | Round 5 | eGRID + EIA | + climate WUE |
|---|---|---|---|
| Carbon (Mt) | 16.04 | 16.20 | 16.20 |
| Power-plant water (Mm3) | 140.1 | 152.2 | 152.2 |
| On-site water (Mm3) | 56.1 | 56.1 | 38.1 |
| Scarcity-weighted water (Mm3-eq) | 2,513 | 2,525 | 2,259 |

The totals barely move, but the routing carbon co-benefit falls from about 10% to about 3%.
Subregions pool several states, so there is less carbon contrast for a router to exploit than state-level Ember data suggested.

## 4. Forecast value for routing (`india/forecast_value.csv`)

Each benchmark model's 2024 CI forecast (issued December 2023) drives the controller (`routing/forecast_value.py`).

| Forecast | CI error (MAPE) | Lyapunov carbon | Penalty gap to perfect foresight |
|---|---|---|---|
| perfect | 0% | +0.81% | 0 |
| pipeline (= seasonal naive) | 4.9% | +1.06% | 0.06% |
| SARIMA (pipeline's checked fit) | 4.1% | +0.86% | 0.02% |
| TimesFM 2.5 | 4.4% | +0.87% | 0.02% |
| xLSTM | 6.8% | +0.83% | 0.02% |
| Chronos-2 | 5.3% | +1.17% | 0.10% |
| SARIMA (notebook fit) | 5.4% | +1.47% | 0.14% |

Every model is within 0.14% of perfect foresight on the routing objective.
Forecast quality explains up to about 0.7 points of the India carbon increase and nothing else of the routing result.
The LLM Nexus run only covers the benchmark's holdout window, not 2024, so it is not scored.

## 5. Climate WUE and location sensitivity (`india/account_totals.json`, `india/location_sensitivity*`)

`water_onsite.wue_model: climate` (`account/wue.py`) gives facilities without a disclosed operator WUE the cooling-tower WUE of their monthly wet-bulb temperature (Gupta et al. 2024, ACM e-Energy; NASA POWER T2MWET).
India on-site water falls from 7.22 to 5.67 Mm3, and scarcity-weighted water from 847 to 767 Mm3-eq.

India facilities sit at city centroids.
Allowing each one to sit anywhere within a radius changes total scarcity-weighted water by:

| Radius | Facilities with another basin in reach | Total scarcity change |
|---|---|---|
| 10 km | 27 of 71 | -7.2% to 0.0% |
| 25 km | 62 of 71 | -8.6% to +0.1% |
| 50 km | 71 of 71 | -22.0% to +0.2% |

The centroids sit near the top of the range: the account's scarcity is close to an upper bound for location.
Researched site coordinates can be added in `datasets/india_dc_facilities/geocodes.csv`.

## Root-cause fix behind step 1

On the consumption basis, the routing controller planned on Ember generation-CI forecasts while it was scored on Electricity Maps consumption CI (e.g. Karnataka about 400 g/kWh planned vs 543 charged; Uttar Pradesh about 766 vs 547).
`routing/lyapunov._ci_forecast` now carries the pre-registered forecast's relative error onto the charged basis.
The generation-basis runs are unchanged by this fix.

## Files

| File | Content |
|---|---|
| `headline_steps.csv` | the step table above, greedy, lyapunov and oracle |
| `india/account_india.parquet`, `us/account_us.parquet` | round-6 accounts |
| `*/account_totals.json` | account totals per variant |
| `*/routing_comparison.csv` | routing on the round-6 account, legal on and off |
| `india/routing_comparison_r5_account.csv` | routing on the round-5 account with the forecast fix only |
| `*/routing_mc_draws.csv`, `*/routing_mc_summary.json` | Monte Carlo draws and summary |
| `*/validation_reported_round{5,6}.csv`, `*/validation_summary_round{5,6}.json` | validation of both accounts |
| `india/forecast_value.csv` | forecast value for routing |
| `india/location_sensitivity.json`, `india/location_sensitivity_25km.csv` | centroid bounds |
| `us/egrid2023_reproduction.csv` | eGRID2023 subregion rates reproduced |
