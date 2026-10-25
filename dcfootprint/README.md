# dcfootprint

The Python package behind the paper: a facility-month account of datacenter carbon and scarcity-weighted water,
with decision analyses on top. Start with the [repository README](../README.md) and
[`docs/methodology.md`](../docs/methodology.md).

```
dcfootprint/
├── pyproject.toml
├── config/
│   ├── parameters.yaml           every numeric assumption, with its source and Monte Carlo band
│   ├── datasets.yaml             dataset registry: path, granularity, join key, role
│   ├── eu_member_state_2023.csv  EU per-country energy and water (from the EED report)
│   ├── legal_constraints.csv     the 7 legal rules applied in siting and routing, with links
│   └── policy_sources.csv        the 15 collected policy documents, with links
├── src/dcfootprint/
│   ├── io/              L0  ingest (Ember, GEM, AWARE, Electricity Maps, CEA, facility lists)
│   ├── geo/             L1  facility and power-plant linkage to grid zones and basins; L4 incidence
│   ├── account/         L2  energy, carbon, water account; L2.5 calibration; EU account; seasonality
│   ├── project/         L3  carbon forecasts, 2030/2050 water and carbon scenarios, basin budgets
│   ├── policy/          L5  policy registry and four-axis gap matrix
│   ├── routing/         L6  Lyapunov controller, oracle, price decomposition
│   ├── counterfactual/  L7  mitigation levers
│   ├── decisions/       L7  harm scorecard, groundwater overlay, siting
│   ├── uncertainty/     L8  Monte Carlo and Sobol
│   ├── viz/             L9  figures and maps
│   ├── validation/          pandera data contracts
│   ├── settings.py          loads and validates parameters.yaml
│   ├── pipeline.py          India: all layers
│   └── regions.py           US and EU
├── tests/               pytest suite
├── experiments/         checks and the forecasting benchmark (see experiments/README.md)
├── workflow/Snakefile   optional Snakemake wrapper around pipeline.py
└── results/             canonical snapshot and sensitivity runs (see results/README.md)
```

## Run

From the repository root:

```bash
PYTHONPATH=dcfootprint/src python -m dcfootprint.pipeline         # India
PYTHONPATH=dcfootprint/src python -m dcfootprint.regions US EU    # US and EU
PYTHONPATH=dcfootprint/src python -m pytest dcfootprint/tests -q  # tests
```

Outputs go to `dcfootprint/results/` (India), `results/us/` and `results/eu/`, with account tables in
`dcfootprint/outputs/` (git-ignored). The optional `forecast-tsfm` extra installs Chronos-2 for the forecasting
benchmark only.
