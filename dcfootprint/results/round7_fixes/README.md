# Round 7: bug fixes and consistency checks

Six fixes and four consistency checks on the round-5 and round-6 results, produced by `dcfootprint/experiments/robustness_round7.py`.
`round5_final/` and `round6_robustness/` are read, never edited.
Two clean runs of the script give byte-identical files (`run_manifest.json` aside, which records the commit).

India starts from the frozen round-5 account, as in round 6, so it needs neither the GEM workbook nor an Electricity Maps key.
The US account is rebuilt from raw data on eGRID and EIA.
`run_manifest.json` holds the exact `parameters.yaml`, its SHA-256, the commit and the package versions.

## Headline in one table

`headline_steps.csv`: change against the static allocation, legal limits on (positive carbon = more carbon).

| Region | Step | Lyapunov carbon | Oracle carbon | Lyapunov scarcity water |
|---|---|---|---|---|
| India | 0 round 5 as published | +1.87% | +0.35% | -21.7% |
| India | 1 round 6 (PR #1) | +1.06% | +0.45% | -22.5% |
| India | 2 + capacity basis | +1.30% | +0.67% | -22.6% |
| India | 3 + WUE clamp + ZLD physics (round 7) | +1.30% | +0.67% | -22.6% |
| India | consistency: generation basis for carbon and water | +1.44% | +1.28% | -21.7% |
| US | 0 round 5 as published | -10.2% | -10.6% | -29.3% |
| US | 1 round 6 (PR #1) | -2.7% | -3.4% | -30.0% |
| US | 3 round 7, against static | -2.7% | -3.4% | -30.0% |
| US | 3 round 7, against status quo (no load moves) | -1.9% | -2.5% | -26.9% |

The India trade-off is larger after the fixes, and it is larger again when carbon and water are put on the same basis.
The US co-benefit survives but is smaller against a status-quo baseline than against the static allocation.
In India the two baselines coincide, because every account facility is colocation with the same utilisation.

## Routing Monte Carlo (1,000 draws, `*/routing_mc_summary.json`)

| | India greedy | India Lyapunov | India oracle | US Lyapunov | US oracle |
|---|---|---|---|---|---|
| carbon change, mean | +0.62% | +1.19% | +0.63% | -2.82% | -3.54% |
| carbon change, 90% interval | -0.39 to +1.37% | +0.36 to +1.77% | -0.32 to +1.42% | -4.07 to -1.63% | -5.34 to -2.12% |
| draws with more carbon | 88% | 98% | 90% | 0% | 0% |
| scarcity water change, mean | -22.5% | -22.1% | -22.5% | -29.4% | -29.0% |

"Even the oracle raises carbon in India" holds in 90% of draws, not in all of them.
The largest first-order driver of the India Lyapunov carbon change is colocation utilisation (0.59, bias-corrected; round 6 reported 0.70 with the uncorrected estimator), then the southern grid's CI (0.17).

In the US the basin queue never binds: Lyapunov equals greedy and Lyapunov with perfect carbon equals the oracle, in the point run and in every Monte Carlo draw (`routing_queue_diagnostics.json`).
The US result is therefore a per-month greedy reallocation, and the paper should not credit the controller there.

## 1. Capacity basis (India)

The India list states each MW figure on its own basis, and its Notes & Method sheet says the operational MW "mixes IT, critical-IT and total-facility bases".
Rounds 5 and 6 treated every figure as IT capacity.
16 of the 71 account facilities (239.8 MW stated: CtrlS 114.4, Techno Digital 36, L&T Cloudfiniti 30, Yotta 30, Iron Mountain 26.4, Cyfuture 3) are stated as total facility power (`operational_total`).
They are now converted with C_IT = C / PUE (`account/energy.py`), the rule the US inventory already used.

| India account (2024) | Round 5 | Round 6 | Round 7 |
|---|---|---|---|
| IT capacity (MW) | 962 | 962 | 872 |
| Electricity (TWh) | 6.08 | 6.08 | 5.51 |
| Carbon (Mt CO2) | 3.63 | 3.63 | 3.29 |
| Physical water (Mm3) | 24.8 | 23.3 | 21.2 |
| On-site water (Mm3) | 7.22 | 5.67 | 5.14 |
| Scarcity-weighted water (Mm3-eq) | 847 | 767 | 699 |
| Off-site share of scarcity-weighted water | 60% | 66% | 66% |
| Monte Carlo 90% band, scarcity-weighted water (Mm3-eq) | 573 to 1,074 | | 477 to 886 |

The fix also closes part of the validation gap that round 6 blamed on utilisation.
CtrlS modelled electricity falls from 2.51 to 1.81 times the reported value, IT energy from 2.35 to 1.70, and on-site water from 2.25 to 1.63 (`india/validation_reported.csv`).

## 2. Zero liquid discharge

ZLD treats and reuses cooling-tower blowdown; it does not touch evaporation, which is most of the water a tower consumes.
With make-up M = E + B and blowdown B = E / (N - 1) at N cycles of concentration, B / M = 1 / N, so ZLD saves r / N of on-site water for a blowdown recovery r.
At N = 5 and r = 0.92 that is 18.4% (`counterfactual/levers.zld_saving_fraction`, parameters in `parameters.yaml`, cycles band 3 to 7).
Round 5 removed 92% of all on-site water, which treated evaporation as recoverable.
Under a strictly consumptive reading the discharged blowdown is returned water and ZLD saves almost no consumption, so 18.4% is an upper bound.
The same function now sets the Rajasthan legal rule in routing and siting (`policy/gap.region_effects`; `legal_constraints.csv` holds a flag).

| Share of scarcity-weighted water saved (carbon) | India | US | EU |
|---|---|---|---|
| ZLD, round 5 | 36.9% (0%) | 29.3% (0%) | 16.4% (0%) |
| ZLD, round 7 | 6.3% (0%) | 4.3% (0%) | 3.3% (0%) |
| Efficiency standard, round 5 | 36.6% (18.8%) | 31.4% (12.9%) | 6.9% (4.4%) |
| Efficiency standard, round 7 | 30.0% (18.7%) | 21.8% (12.7%) | 6.9% (4.4%) |
| Coastal seawater cooling, round 7 | 15.2% (0%) | n.a. | n.a. |

The efficiency standard is now the recommended lever for every flagged facility in India (16 of 16) and the EU (4 of 4), and for 50 of 53 in the US (`*/q2_scorecard_summary.json`).
The efficiency standard still caps PUE and WUE together without the PUE penalty that dry cooling brings; that trade-off is not modelled.

## 3. Siting on the carbon basis the account charges

Siting charged a new facility Ember's own-generation CI while the account charges India's facilities the consumption CI of their regional grid: Karnataka was 396 g/kWh in siting and 543 in the account.
`decisions/siting.py` now uses `account.build.charged_ci`, the function the account uses.
A fresh India run needs the GEM workbook, so India is re-scored here from the frozen round-5 candidate cells with the same criteria and minimax regret (`india/q1_siting_rerank.csv`).

| India siting | Round 5 | Round 7 |
|---|---|---|
| Top 10 cells, legal limits on | Karnataka 10 | Gujarat 10 |
| Top 10 cells, legal limits off | | Rajasthan 5, Karnataka 5 |
| Spearman, round 5 rank vs round 7 rank | | 0.944 |
| Spread of max regret across the top 10 | | 0.018 |
| Ranks of the 9 cells that hold datacenters (of 117) | 11, 34, 45, 82, 84, 86, 99, 102, 103 | 16, 38, 45, 74, 90, 92, 94, 105, 106 |

The top of the ranking is not robust: it flips with the carbon basis, and with the basis fixed it flips again with Gujarat's 51% renewable-supply rule.
The top cells are near ties.
The finding that does survive is the other one: most cells that hold today's datacenters rank in the bottom half (6 of 9 below rank 70).

## 4. Cold-weather WUE

The Gupta et al. cooling-tower quadratic has its minimum at 9.3 C wet-bulb and rises again below it (1.69 L/kWh at -8 C).
Below 9.3 C the WUE is now held at the minimum, 1.17 L/kWh (`account/wue.py`).
India is unaffected (no account month is that cold).
US on-site water falls from 38.1 to 36.9 Mm3 and scarcity-weighted water from 2,259 to 2,249 Mm3-eq.
The clamp credits no free-cooling hours, so winter WUE is still an upper estimate.

## 5. Sobol indices

The round-5 first-order indices came from 40 quantile bins of the Monte Carlo sample, were biased upward by the noise of the bin means, and summed to 1.026 for India.
They were also computed on inference-attributed water, not on the headline total.
`uncertainty/monte_carlo.sobol_saltelli` now gives first-order (Saltelli 2010) and total (Jansen 1999) indices of the total scarcity-weighted water from 2^15 base samples, with bootstrap 90% intervals.

| Scarcity-weighted water | India S1 | India ST | US S1 | US ST |
|---|---|---|---|---|
| Colocation utilisation | 0.66 (0.60 to 0.72) | 0.70 | 0.53 (0.48 to 0.58) | 0.53 |
| WUE | 0.16 (0.13 to 0.18) | 0.16 | 0.10 (0.08 to 0.12) | 0.09 |
| Hydro multiplier | 0.06 (0.04 to 0.07) | 0.05 | 0.29 (0.26 to 0.33) | 0.28 |
| EWIF | 0.04 (0.03 to 0.06) | 0.06 | 0.05 (0.04 to 0.07) | 0.05 |
| Colocation PUE | 0.04 (0.02 to 0.05) | 0.05 | 0.03 (0.01 to 0.05) | 0.05 |

The paper's reading (utilisation about two thirds, WUE about a fifth for India) holds, now with intervals.
In the US the hydro attribution matters far more than in India (0.29), which the paper does not say.
The routing Monte Carlo has one sample per draw and keeps the binned estimator with the ANOVA bias correction.

## 6. Groundwater overlay

Round 5 said 63% of India's scarcity-weighted water falls on over-exploited aquifers.
That share counted power-plant water, which is drawn in the generation basins, against the facility's own aquifer.
On the water a facility draws locally (on-site scarcity-weighted water) the share is 59.0% over-exploited and 70.6% semi-critical or worse (`india/groundwater_equity.json`; the total-burden shares are 66.2% and 82.4% on the round-7 account).
`decisions/equity.py` now reports both and leads with on-site.

## 7. Policy registry

Four of the 26 registry rows are reference documents, not instruments: the WattTime marginal-emissions methodology, the WRI Aqueduct framework paper, the Environmental Law Institute fact sheet and the PNNL atlas article.
They are now `kind = reference`, are never read as in force, and are left out of the gap matrix (`policy/registry.py`).
The registry holds 22 instruments; the four-axis result (no instrument mandates any axis) is unchanged.

## 8. Forecast benchmark, scale-free and tested (`forecast_significance.md`)

Pooled RMSE across series of very different size is dominated by the largest series.
Skill is now the geometric mean over series of RMSE_model / RMSE_naive, with a bootstrap interval, and a paired Wilcoxon test across series, Holm-corrected.
`sarima_dcf_fallback` replaces failed fits with seasonal naive, the rule the methodology states (the benchmark had dropped them).

| Skill vs seasonal naive | Carbon, rolling | Carbon, holdout | Electricity, rolling | Electricity, holdout |
|---|---|---|---|---|
| SARIMA (fit checks, naive fallback) | +11.3% * | +7.9% | +9.6% * | -0.7% |
| Chronos-2 | +11.1% * | +9.3% | +13.6% * | +9.6% * |
| TimesFM 2.5 | +6.0% * | +3.8% | +12.9% * | +6.1% |
| xLSTM (5 seeds) | -9.5% * | -16.6% * | -9.9% * | -17.0% * |
| Nexus LLM agents | n.a. | -8.2% | n.a. | -13.8% * |

\* significantly different from seasonal naive at 5% after Holm correction.

SARIMA and Chronos-2 are tied for carbon, and neither is significantly better than naive on the single holdout.
On the electricity holdout Chronos-2 is significantly better than naive (+9.6%), where the pooled RMSE in the paper showed -6.9%.
What survives is narrower than "simple methods are sufficient": pretrained foundation models match or beat classical models, and training a deep network or using an LLM agent does not pay.

## What the paper should change

- Table 3, Section 5.1 and the abstract: the India trade-off is +1.3% (Lyapunov) and +0.7% (oracle), up in 98% and 90% of draws; the US co-benefit is about -2 to -3% depending on the baseline.
- Report the consistent generation-basis India result (+1.4% Lyapunov, +1.3% oracle) as the robustness row for the carbon-water basis.
- Table 4: India totals as in section 1 above; off-site share 66%.
- Table 7: ZLD at 6.3% / 4.3% / 3.3%, with the blowdown explanation; ZLD is no longer a recommended lever.
- Section 5.5.3: do not name a best Indian state; report that most existing hub cells rank in the bottom half and that the top flips with the carbon basis and with Gujarat's renewable rule.
- Section 5.3: groundwater 59% / 71% on on-site water.
- Section 4.6 and Table 2 notes: Saltelli/Jansen indices with intervals; mention the US hydro sensitivity.
- Section 4.7: 22 instruments plus 4 reference documents.
- Section 5.6, Table 8 and Section 6.4: the scale-free, tested skill and the narrower conclusion.
- Section 4.8.4: say the US queue never binds.

## Not done here (needs the authors or more data)

- US inventory vintage: Project Rainier (590 MW), Crusoe Abilene (200 MW) and xAI Colossus (150 MW) are counted for all of 2024; check commissioning dates, and type AI-training campuses as hyperscale.
- US siting still charges Ember state generation CI while the US account uses eGRID subregions; a state-by-basin cell has no single subregion.
- Siting uses the constant WUE (1.9 L/kWh) while the account uses the climate WUE.
- India water is still on each state's own generation mix while India carbon is on the regional consumption mix; the generation-basis row above is the consistent check, and a consumption-basis water account needs the Electricity Maps power breakdown.
- Policy registry coverage: no instrument rows for Tamil Nadu, Karnataka, Maharashtra, Uttar Pradesh or Telangana (where India's burden is), Arizona (38% of the US burden) or Spain (largest in the EU).
- Marginal emissions and hourly routing (planned for the next PR).
