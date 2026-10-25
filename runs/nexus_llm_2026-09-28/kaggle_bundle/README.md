# Kaggle bundle used for the real LLM Nexus run

`run_nexus_kaggle.py` is the runner the Kaggle notebook (`../kaggle_notebook_nexus-llm-ollama-gemma4-kaggle-gpu_v2.ipynb`)
executed, taken unchanged from the uploaded dataset version (`nexus-kaggle-bundle2.zip`, the version with repeats).

The rest of the bundle is not duplicated here because it is byte-identical to files already versioned in the preliminary-experiment
repository (`diff/gat-sarima-nexus/`), checked by SHA-256 (first 16 hex digits):

| bundle file | preliminary-experiment file | sha256 |
|---|---|---|
| `src/nexus_ollama.py` | `src/nexus_ollama.py` | `e975bd4f4fb078e1` (also listed in preliminary-experiment `docs/AUDIT_REPORT.md`) |
| `src/nexus_local.py` | `src/nexus_local.py` | `1b16fc7a15f95d4c` |
| `work/region_series_dump.json` | `work/region_series_dump.json` | `c717b7cd976d8b4c` |
| `work/local_cpu_water_ollama.csv` | `work/nexus_water_ollama.csv` (renamed so resume logic never mistakes it for Kaggle output) | — |
| `run_nexus_kaggle.py` | this folder | `5088e2d3b7d025be` |

What the runner adds on top of the preliminary experiment's `scripts/run_water_nexus.py`: every resource in the series dump, a time
budget (`NEXUS_MAX_HOURS`), checkpoints every `NEXUS_CKPT_MIN` minutes to a private Kaggle dataset
(`nexus-results-checkpoint`), safe resume (a series x repeat counts as done only with all 12 rows), `NEXUS_REPEATS`
repeats per series (3 in the run, because the GPU output is not deterministic at temperature 0), and a
CPU-vs-GPU consistency check (`python run_nexus_kaggle.py check`).

To rebuild the bundle: copy this runner plus the three preliminary-experiment files above into `run_nexus_kaggle.py`, `src/`,
`work/`, and rename preliminary-experiment `work/nexus_water_ollama.csv` to `work/local_cpu_water_ollama.csv`.
