# figures/

The paper's figures (PNG, 300 dpi), drawn from the canonical snapshot `dcfootprint/results/round5_final/`.

| Figure | File(s) |
|---|---|
| 1 Framework data flow | `fig1_framework.png` |
| 2 Routing bound, India and US | `fig2a_india_routing.png`, `fig2b_us_routing.png` |
| 3 India scarcity map | `fig3_india_scarcity_map.png` |
| 4 US and EU scarcity maps | `fig4a_us_scarcity_map.png`, `fig4b_eu_scarcity_map.png` |
| 5 Mitigation levers, three regions | `fig5_levers.png` |
| 6 Siting by minimax regret | `fig6a_india_siting_map.png`, `fig6b_us_siting_map.png`, `fig6c_eu_siting_map.png` |

Redraw from the repository root:

```bash
python figures/scripts/fig1_framework.py    # Figure 1
python figures/scripts/make_figures.py      # Figures 2-6 (the maps need data/gadm/)
```

PNG output is not byte-stable across machines, but every value plotted is in a table in `round5_final/`.
