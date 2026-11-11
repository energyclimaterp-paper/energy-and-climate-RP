# Datacenter Carbon and Scarcity-Weighted Water: India, United States, European Union

Code, data, and results for the paper **"Co-benefit or trade-off: the limits of shifting datacenter load for carbon and scarcity-weighted water in India, the United States and the European Union"** (*Energy and Climate Change*, special issue on computing and digitalization).

The pipeline builds a monthly account of energy, carbon, and water for every datacenter, weights water consumption by scarcity in the river basin where it is used (both on-site and at the power plants supplying the grid), and uses that account to evaluate load shifting, mitigation levers, siting, and the reach of current regulation.

| Region | Unit | Coverage | Year |
|---|---|---|---|
| India | Facility | 71 colocation facilities (from the 194-row list released here) | 2024 |
| United States | Facility | 235 facilities (Compute Atlas) | 2024 |
| European Union | Country | 21 reporting member states (Energy Efficiency Directive reports) | 2023 |

## Headline Results

All figures below come from the canonical snapshot [`dcfootprint/results/round5_final/`](dcfootprint/results/round5_final/).

| Result | India | United States | European Union |
|---|---:|---:|---:|
| Annual carbon emissions | 3.63 Mt CO₂ | 16.0 Mt CO₂ | 3.35 Mt CO₂e |
| Annual scarcity-weighted water | 847 million m³-eq | 2.51 billion m³-eq | 137 million m³-eq |
| Load shifting: water change (30% of load movable) | −21.7% | −29.3% | Not run (no site locations) |
| Load shifting: carbon change (30% of load movable) | +1.9% | −10.2% | Not run (no site locations) |

Key findings:

- **60–82%** of scarcity-weighted water use occurs off-site, at power plants.
- **59%** of the water India's datacenters draw on site (scarcity-weighted) falls on over-exploited aquifers (round 7; round 5 reported 63% by also counting power-plant water drawn in other basins).
- None of the 22 instruments in the policy registry mandates carbon intensity, marginal carbon, scarcity-weighted water, or AI-inference attribution (the registry's other 4 rows are reference documents).

The paper's tables, figures, and reported values are mapped to their source files in [`docs/results_guide.md`](docs/results_guide.md).

### Corrections since the canonical snapshot

Two later rounds revise the numbers above; `round5_final/` itself is frozen.
[`round6_robustness/`](dcfootprint/results/round6_robustness/) adds validation against operator reports, US eGRID/EIA plant data, a routing Monte Carlo and climate-dependent WUE.
[`round7_fixes/`](dcfootprint/results/round7_fixes/) fixes six bugs: India capacities stated as total facility power were treated as IT capacity, zero liquid discharge was credited with 92% of on-site water instead of the recovered blowdown (18%), siting used a different carbon basis from the account, the climate WUE rose in cold weather, the Sobol estimator was biased, and the groundwater overlay counted power-plant water against local aquifers.

| Result (round 7) | India | United States |
|---|---:|---:|
| Annual carbon emissions | 3.29 Mt CO₂ | 16.2 Mt CO₂ |
| Annual scarcity-weighted water | 699 million m³-eq | 2.25 billion m³-eq |
| Load shifting, Lyapunov: carbon change (90% Monte Carlo interval) | +1.3% (+0.4 to +1.8%) | −2.7% (−4.1 to −1.6%) |
| Load shifting, Lyapunov: scarcity-weighted water change | −22.6% | −30.0% |
| Load shifting, offline oracle: carbon change | +0.7% (carbon up in 90% of draws) | −3.4% |

## Repository Layout

```text
.
├── README.md                         # This file
├── LICENSE                           # MIT licence (code)
├── LICENSE-DATA.md                   # CC BY 4.0 (India list, results, figures)
├── docs/
│   ├── methodology.md                # Pipeline layers and equations
│   ├── data_sources.md               # Input datasets, versions, and licence notes
│   ├── results_guide.md              # Paper table/figure/value -> source file
│   └── architecture_flow.svg         # Architecture diagram
├── data/                             # Raw third-party inputs (not included)
│   └── README.md                     # Data acquisition instructions
├── datasets/
│   └── india_dc_facilities/          # Open India datacenter list
├── dcfootprint/                      # Python package
│   ├── pyproject.toml                # Dependencies and project configuration
│   ├── config/                       # Assumptions, parameters, legal limits, policy sources
│   ├── src/dcfootprint/              # Pipeline implementation, layers L0-L9
│   ├── tests/                        # pytest suite
│   ├── experiments/                  # Checks and forecasting benchmark
│   ├── workflow/
│   │   └── Snakefile                 # Optional Snakemake wrapper
│   └── results/
│       ├── round5_final/             # Frozen canonical snapshot used in the paper
│       └── generation_basis_check/   # India generation-carbon-basis sensitivity
├── runs/
│   ├── forecast_benchmark_2026-09-28/
│   │                                   # Seven models, 66 account-zone series
│   └── nexus_llm_2026-09-28/
│                                       # LLM multi-agent forecaster (Appendix C)
├── figures/                          # Paper figures (PNG, 300 dpi) and scripts
└── tables/                           # Tables generated from the canonical snapshot
```

## Quick Start

### Requirements

- Python 3.11
- Git
- Raw input datasets, downloaded separately as described in [`data/README.md`](data/README.md)

### 1. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e "dcfootprint[dev]"
```

**macOS/Linux:**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e "dcfootprint[dev]"
```

### 2. Run the Tests

The test suite does not require the raw input datasets.

**Windows (PowerShell):**

```powershell
.venv\Scripts\python -m pytest dcfootprint/tests -q
```

**macOS/Linux:**

```bash
.venv/bin/python -m pytest dcfootprint/tests -q
```

### 3. Download the Raw Data

Follow the instructions in [`data/README.md`](data/README.md) and place the required input files in `data/`.

### 4. Run the Pipeline

Execute these commands from the repository root.

**India:**

```bash
PYTHONPATH=dcfootprint/src .venv/bin/python -m dcfootprint.pipeline
```

**United States and European Union:**

```bash
PYTHONPATH=dcfootprint/src .venv/bin/python -m dcfootprint.regions US EU
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`. If `PYTHONPATH` is not set in your shell, use the equivalent environment-variable syntax for your platform.

Fresh runs write results to `dcfootprint/results/`. The frozen `round5_final/` snapshot is not overwritten.

### 5. Generate the Figures

```bash
.venv/bin/python figures/scripts/fig1_framework.py
.venv/bin/python figures/scripts/make_figures.py
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.

Each run prints a stage-by-stage status table. Every hand-off between pipeline stages is validated against a data contract using Pandera. A failed contract causes the relevant stage to fail.

## Validation and Reproducibility

The project includes the following checks and supporting records:

- **Determinism:** Two clean runs produced identical outputs across 60 text files and six Parquet tables. See [`DETERMINISM.md`](dcfootprint/results/round5_final/DETERMINISM.md).
- **Independent traceability check:** Three Indian facilities were recomputed directly from source files without using the pipeline's own functions, with zero discrepancies. See [`SPOTCHECK.md`](dcfootprint/results/round5_final/SPOTCHECK.md).
- **Calibration:** Bottom-up totals were compared with CEEW/JLL for Indian capacity, LBNL for US energy, and EU reporting coverage. See [`docs/results_guide.md`](docs/results_guide.md).
- **Uncertainty analysis:** 4,000 Monte Carlo draws per region, with first-order and total Sobol indices from a Saltelli design and bootstrap intervals (round 7). See `uncertainty.json` in the relevant results.
- **Run manifest:** every pipeline and robustness run writes `run_manifest.json` with the exact `parameters.yaml`, its SHA-256, the commit and the package versions.

## Scope and Limitations

- **Load shifting:** The routing analysis is a diagnostic bound based on stylised demand, not a deployable scheduler.
- **Coverage:** Absolute totals are lower bounds because facility and reporting coverage is partial in every region.
- **Carbon accounting:** India uses a consumption-based, life-cycle carbon basis from Electricity Maps. The United States and EU use a generation-based, direct-emissions basis from Ember. Cross-region comparisons should therefore be interpreted with care. See `cross_region_summary.md`.
- **Policy review:** The policy registry reflects English-language sources accessible in September 2026.
- **EU load shifting:** The analysis was not run because site-level locations were unavailable.

## Licence and Citation

- **Code:** MIT licence. See [`LICENSE`](LICENSE).
- **India facility list, result tables, figures, and benchmark outputs:** CC BY 4.0. See [`LICENSE-DATA.md`](LICENSE-DATA.md).
- **Third-party datasets:** Retain their respective licences; restrictions may also apply to derived values. See [`docs/data_sources.md`](docs/data_sources.md).

Citation details will be added upon publication.
