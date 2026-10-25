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

## Notes

- **Electricity Maps** needs an API key: put it in `data/electricitymaps.key` or the `ELECTRICITYMAPS_API_KEY`
  environment variable. The key is never versioned. Downloaded intensities are cached in `data/em/`.
- **Aqueduct 4.0** is joined spatially from its geodatabase; only its water-stress scores enter the 2030/2050
  siting scenarios.
- Third-party terms apply to derived values too. Check the Electricity Maps terms before redistributing the India
  carbon values in `round5_final/outputs/`, and credit Ember and GEM (CC BY 4.0).
