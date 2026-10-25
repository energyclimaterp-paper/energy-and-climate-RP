# Data licence: CC BY 4.0

The data in this repository listed below are licensed under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/ (full legal code:
https://creativecommons.org/licenses/by/4.0/legalcode).

Copyright (c) 2026 The authors.

## Covered

| Path | Content |
|---|---|
| `datasets/india_dc_facilities/` | the India datacenter facility list |
| `dcfootprint/results/` | result tables, reports and account outputs of the canonical snapshot and the sensitivity run |
| `runs/` | forecasting benchmark outputs and metrics |
| `tables/` | tables built for the paper |
| `figures/*.png` | the paper's figures |

You may share and adapt this material for any purpose, including commercially, provided you give appropriate
credit (cite the paper), link to the licence and indicate if changes were made.

## Not covered

- **Code** (everything under `dcfootprint/src/`, `dcfootprint/tests/`, `dcfootprint/experiments/`,
  `dcfootprint/workflow/` and `figures/scripts/`) is licensed under the MIT licence in [`LICENSE`](LICENSE).
- **Third-party data** keep their own licences. This covers the raw inputs listed in
  [`docs/data_sources.md`](docs/data_sources.md), and the policy documents linked from `dcfootprint/config/`.
  Values derived from them can carry the provider's terms. In particular, the India carbon intensities in
  `dcfootprint/results/round5_final/outputs/` derive from Electricity Maps data, and Ember and Global Energy
  Monitor data require attribution under CC BY 4.0.
- **Links** in the facility list (`source_url`) point to third-party pages that remain under their owners' terms.
