# results/

| Folder | What it is |
|---|---|
| [`round5_final/`](round5_final/) | **The canonical snapshot quoted in the paper.** India, US and EU. Frozen: never edited. |
| [`generation_basis_check/`](generation_basis_check/) | India rerun with carbon on the generation basis (Ember) instead of the consumption basis, used for the carbon-basis sensitivity |

A new run of the pipeline writes its outputs directly into `results/` (India), `results/us/` and `results/eu/`.
It does not touch the frozen folders. To compare a new run with the snapshot, use
`experiments/compare_results.py`.

The paper's tables, figures and numbers are mapped to these files in
[`docs/results_guide.md`](../../docs/results_guide.md).
