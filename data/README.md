# data/ (raw inputs, not included)

The raw third-party files are not redistributed. Download them from the sources in
[`docs/data_sources.md`](../docs/data_sources.md) and place them as below. Everything in this folder except this
README is git-ignored. Derived caches (parquet copies, forecast caches) are written next to the raw files and
rebuilt automatically.

```
data/
├── ember/          india_monthly_full_release_long_format.csv
│                   us_monthly_full_release_long_format.csv
│                   europe_monthly_full_release_long_format.csv
├── gem/            Global-Integrated-Power-March-2026-II.xlsx
├── aware/          AWARE20_Native_CFs_geospatial.gpkg
│                   AWARE20_Intermediate_Variables.xlsx
│                   AWARE20_Countries_and_Regions.xlsx
├── aqueduct/       Aq40_Y2023D07M05.gdb                      (2030/2050 siting scenarios)
├── carbon_future/  Cambium22_LRMER_states.xlsx               (US 2030/2050 grid carbon)
│                   ref2020_energy-transport-ghg.xlsx         (EU 2030/2050 grid carbon)
├── em/             ci_IN-{NO,WE,SO,EA,NE}_2024.parquet       (Electricity Maps cache, written on first run)
├── cea/            CEA_Database_V22.xlsx                     (India national anchor)
├── cgwb/           India_Groundwater_Availability_Utilization_and_E.csv
│                   Major_Cities_Groundwater_Availability_Utilizatio.csv
├── gadm/           gadm41_IND.gpkg, gadm41_USA.gpkg
├── compute_atlas/  facilities_v1.34.0.json                   (US facilities)
├── atlas/          datacenters.parquet                       (Global Datacenter Atlas, coordinates)
├── g3p/            G3P_v1.12_tws_rivbas.csv                  (groundwater context)
├── eu_eed/         EU datacenter assessment report (PDF)     (source of config/eu_member_state_2023.csv)
├── epoch/          data_centers.csv                          (optional US cross-check)
└── electricitymaps.key                                       (your API key; never commit it)
```

The India facility list is not here: it is part of this release, in
[`datasets/india_dc_facilities/`](../datasets/india_dc_facilities/).
