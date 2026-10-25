"""Figures 2-6 of the paper, rendered from the canonical snapshot dcfootprint/results/round5_final.

Writes into figures/ (300 dpi, no in-figure titles):
  fig2a_india_routing.png, fig2b_us_routing.png       routing bound, India and US
  fig3_india_scarcity_map.png                         India scarcity-weighted water by facility
  fig4a_us_scarcity_map.png, fig4b_eu_scarcity_map.png US and EU maps
  fig5_levers.png                                     mitigation levers, three regions
  fig6a/b/c_<region>_siting_map.png                   siting by minimax regret, India, US, EU

Run from the repository root:  python figures/scripts/make_figures.py
Needs data/gadm/ (state boundaries) for the maps. Figure 1 is drawn by fig1_framework.py.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dcfootprint" / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

_orig_save = Figure.savefig


def _clean_save(self, fname, *a, **k):
    # drop the figure-level title and, for single-map figures, the axes title; panel titles of
    # multi-panel charts stay, since they name the panel's measure
    if self._suptitle is not None:
        self._suptitle.set_text("")
    data_axes = [ax for ax in self.axes if ax.get_label() != "<colorbar>"]
    if len(data_axes) == 1:
        data_axes[0].set_title("")
    k["dpi"] = 300
    return _orig_save(self, fname, *a, **k)


Figure.savefig = _clean_save

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dcfootprint.viz import figures as F  # noqa: E402

R5 = ROOT / "dcfootprint" / "results" / "round5_final"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp())

# Figure 2: routing bound, India and US
states = F._india_states()
F.fig_routing(R5, tmp)
shutil.copy(tmp / "fig_routing.png", OUT / "fig2a_india_routing.png")
(tmp / "us").mkdir()
F.fig_routing(R5 / "us", tmp / "us")
shutil.copy(tmp / "us" / "fig_routing.png", OUT / "fig2b_us_routing.png")

# Figures 3 and 4: scarcity-weighted water maps
F.fig_mismatch(R5, tmp, states)
shutil.copy(tmp / "fig_mismatch_map.png", OUT / "fig3_india_scarcity_map.png")
for region, name in (("US", "fig4a_us_scarcity_map.png"), ("EU", "fig4b_eu_scarcity_map.png")):
    d = tmp / f"map_{region.lower()}"
    d.mkdir()
    F.fig_mismatch_region(region, R5 / region.lower(), d, F._region_states(region), F._region_sites(region))
    shutil.copy(d / "fig_mismatch_map.png", OUT / name)

# Figure 6: siting maps, India, US, EU
F.fig_q1(R5, tmp, states)
shutil.copy(tmp / "fig_q1_siting_map.png", OUT / "fig6a_india_siting_map.png")
for region, name in (("US", "fig6b_us_siting_map.png"), ("EU", "fig6c_eu_siting_map.png")):
    d = tmp / f"q1_{region.lower()}"
    d.mkdir()
    F.fig_q1_region(region, R5 / region.lower(), d, F._region_states(region), F._region_sites(region))
    shutil.copy(d / "fig_q1_siting_map.png", OUT / name)

# Figure 5: levers, three regions in one chart (coastal cooling applies to India only)
rows = {reg: pd.read_csv(p / "lever_savings.csv").set_index("lever")
        for reg, p in (("India", R5), ("US", R5 / "us"), ("EU", R5 / "eu"))}
levers = [("zero_liquid_discharge", "Zero liquid\ndischarge"), ("efficiency_standard", "Efficiency\nstandard"),
          ("coastal_seawater_siting", "Coastal seawater\ncooling")]
colors = {"India": F.BLUE, "US": F.ORANGE, "EU": F.AQUA}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
for ax, col, title in [(axes[0], "pct_of_scarcity_baseline", "scarcity-weighted water saved (%)"),
                       (axes[1], "pct_of_carbon_baseline", "carbon saved (%)")]:
    for j, reg in enumerate(rows):
        for i, (key, _) in enumerate(levers):
            if key == "coastal_seawater_siting" and reg != "India":
                continue  # no eligible coastal facilities outside India
            v = float(rows[reg].loc[key, col])
            y = i + (j - 1) * 0.26
            ax.barh(y, v, height=0.24, color=colors[reg], label=reg if i == 0 else None)
            ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=6.8, color=F.INK2)
    ax.set_xlim(0, 45)
    ax.set_title(title, fontweight="normal", color=F.INK2, fontsize=9)
    ax.grid(axis="x", color=F.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(levers)), [n for _, n in levers])
axes[0].set_ylim(2.55, -0.5)
axes[1].legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig5_levers.png")

print(sorted(p.name for p in OUT.glob("*.png")))
