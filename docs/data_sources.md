# Data sources

The fifteen datasets of the paper (Table 1), plus the supporting inputs the code reads. Raw third-party files
are not redistributed here: download them into `data/` as listed in [`data/README.md`](../data/README.md).
Each provider's licence applies to its data and to values derived from it.

## Datasets in the paper (Table 1)

| Category | Dataset | Coverage | Spatial resolution | Source |
|---|---|---|---|---|
| Electricity grid | Ember US Monthly | 2001–2025, monthly | state (50 states, DC, Puerto Rico) | https://ember-energy.org/data/monthly-electricity-data/ |
| Electricity grid | Ember Europe Monthly | 2010–2026, monthly | country (38) | https://ember-energy.org/data/monthly-electricity-data/ |
| Electricity grid | Ember India Monthly | 2019–2025, monthly | state/UT (29 states, 8 UTs) | https://ember-energy.org/data/monthly-electricity-data/ |
| Electricity grid | GEM Global Integrated Power | March 2026 release | plant | https://globalenergymonitor.org/projects/global-integrated-power-tracker/ |
| Carbon intensity | Electricity Maps API | 2024, hourly | five Indian regional grids | https://www.electricitymaps.com |
| Carbon intensity | CEA National Electricity Plan 2022–32 | projections to 2031–32 | India | https://cea.nic.in/national-electricity-plan/ |
| Carbon intensity | NREL Cambium 2022 | 2030 and 2050 | US state | https://doi.org/10.2172/1915250 |
| Carbon intensity | EU Reference Scenario 2020 | five-yearly | EU country | https://doi.org/10.2833/35750 |
| Water | AWARE 2.0 | monthly factors | basin (11,661) | https://doi.org/10.1111/jiec.70023 |
| Water | WRI Aqueduct 4.0 | baseline, 2030, 2050 | sub-basin (68,506) | https://doi.org/10.46830/writn.23.00061 |
| Water | CGWB groundwater resources | 2024 assessment | city, district or state | https://cgwb.gov.in |
| Datacenter | India datacenter list (this study) | 2026 snapshot | facility (194 rows) | [`datasets/india_dc_facilities/`](../datasets/india_dc_facilities/) |
| Datacenter | Global Datacenter Atlas | snapshot | point (18,110 records) | https://github.com/Ringmast4r/Global-Data-Center-Map |
| Datacenter | Compute Atlas v1.34.0 | snapshot | US facility (2,025 records) | https://github.com/ek33450505/compute-atlas |
| Datacenter | EU datacenter assessment (EED) | 2023, annual | country (21 member states) | https://doi.org/10.2833/3168794 |

## Supporting inputs

| Input | Used for | Where |
|---|---|---|
| GADM 4.1 boundaries (India, US) | state of each facility and power plant; maps | https://gadm.org |
| Macknick et al. (2012) water coefficients | EWIF per fuel | values in `dcfootprint/config/parameters.yaml` |
| EU member-state energy and water table | EU account | `dcfootprint/config/eu_member_state_2023.csv`, rebuilt from the EED report by `experiments/extract_eu_tables.py` |
| CEA CO₂ Baseline Database v22 | India national anchor | https://cea.nic.in |
| G3P terrestrial water storage | groundwater context | GFZ G3P v1.12 |
| Epoch AI frontier datacenters | US coverage cross-check only | https://epoch.ai/data |
| Policy documents (15) and legal rules (7) | regulatory gap and legal limits | `dcfootprint/config/policy_sources.csv`, `dcfootprint/config/legal_constraints.csv` (each with its link) |
| EPA eGRID2023 (rev2, June 2025): plant table, subregion table, subregion map | US grid zone (eGRID subregion), plant CO₂ output rates | https://www.epa.gov/egrid/download-data, https://www.epa.gov/egrid/egrid-mapping-files |
| EIA-923 monthly generation and fuel data, 2023 and 2024 | US plant-month net generation (carbon weights, EWIF denominator) | https://www.eia.gov/electricity/data/eia923/ |
| EIA thermoelectric cooling water data, 2023 and 2024 | US plant-month reported cooling-water consumption | https://www.eia.gov/electricity/data/water/ |
| EIA-860 schedule 2 (2024) | locations of US plants not in eGRID2023 | https://www.eia.gov/electricity/data/eia860/ |
| NASA POWER monthly T2MWET (MERRA-2) | facility-month wet-bulb temperature for the climate WUE | https://power.larc.nasa.gov (fetched by `account/wue.py`, cached in `data/climate/`) |
| Operator sustainability and ESG reports (Nxtra, CtrlS, STT GDC India, Meta, Google) | validation of energy and water against reported figures | `dcfootprint/config/reported_footprints.csv` (each figure with its link and PDF page) |

## Notes

- **Electricity Maps** needs an API key: put it in `data/electricitymaps.key` or the `ELECTRICITYMAPS_API_KEY`
  environment variable. The key is never versioned. Downloaded intensities are cached in `data/em/`.
- **Aqueduct 4.0** is joined spatially from its geodatabase; only its water-stress scores enter the 2030/2050
  siting scenarios.
- Third-party terms apply to derived values too. Check the Electricity Maps terms before redistributing the India
  carbon values in `round5_final/outputs/`, and credit Ember and GEM (CC BY 4.0).
- **Ember US.** Ember retired `us_monthly_full_release_long_format.csv` (the file now holds only a notice) in
  favour of `release_us_subnational_generation_monthly.csv`, which reports lifecycle CO₂e rather than direct CO₂.
  `io/ember.py` reads the new file when the old one is absent (Puerto Rico dropped: it is not in Ember's US
  total). Since round 6 the US account takes its carbon and power-plant water from eGRID and EIA
  (`io/us_grid.py`), so Ember US is used only for state names and the Ember-based sensitivity.
