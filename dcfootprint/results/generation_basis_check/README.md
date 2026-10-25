# generation_basis_check: India on the generation carbon basis

A full India run in which carbon uses each state's **generation-based** intensity (Ember) instead of the
**consumption-based**, import-adjusted intensity of the regional grids (Electricity Maps) used in the canonical
snapshot. Everything else is the same. It supports the carbon-basis sensitivity in the paper:

- The national India total moves by only about 0.2% between the two bases. Individual states move much more,
  since import adjustment shifts carbon between states that share a regional grid.
- In the routing bound, India's carbon change is +1.9% on the consumption basis and about +1.4% on the
  generation basis. The trade-off holds on both.

Compare `india_account_summary.csv` and `routing_comparison.csv` here with the same files in
[`../round5_final/`](../round5_final/). Water results do not depend on the carbon basis. The run was made with
the code of the previous snapshot. That code differs from `round5_final` only in the 2030/2050 siting scenarios
(`q1_siting_scenarios.csv`), which were later placed on official grid-carbon paths, so do not quote the scenario
file from this folder. Run log: `gen_india.log`.
