"""Figure 1: data flow of the framework (paper Section 4.1). Writes figures/fig1_framework.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, BLUE, ORANGE, LIGHT = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e8f1fc"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")


def box(x, y, w, h, title, body, fill=LIGHT, edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontweight="bold", color=INK, fontsize=8.8)
    nl = title.count("\n")
    ax.text(x + w / 2, y + h - 5.2 - 2.4 * nl, body, ha="center", va="top", color=INK2, fontsize=7.2,
            linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.0))


box(1, 36, 22, 18, "Public data",
    "Ember, Electricity Maps, GEM\nAWARE 2.0, Aqueduct 4.0\nCGWB groundwater\nfacility inventories (India list,\nCompute Atlas, EU EED)")
box(27, 36, 21, 18, "Cleaning and\nspatial linkage",
    "facilities and power plants\nplaced in grid zones (states)\nand river basins (AWARE)")
box(52, 36, 23, 18, "Monthly facility\naccount",
    "energy, carbon, physical and\nscarcity-weighted water,\non-site and off-site,\nper facility and month")
box(79, 36, 20, 18, "Calibration",
    "checked against\nCEEW, LBNL and\nEU reporting totals")
arrow(23.6, 45, 26.4, 45); arrow(48.6, 45, 51.4, 45); arrow(75.6, 45, 78.4, 45)

box(1, 17, 23, 14, "Forecasts and scenarios",
    "one-month carbon forecast\n2030 and 2050 water and\ngrid-carbon paths")
box(28, 17, 21, 14, "Incidence matrices",
    "facility-to-grid and\nfacility-to-basin links\nbasin budgets and queues")
box(53, 17, 22, 14, "Regulatory gap",
    "26-instrument registry\nfour-axis mandate matrix\nbinding legal limits")
box(79, 17, 20, 14, "Uncertainty",
    "4,000 Monte Carlo draws\nfirst-order Sobol\nindices")

box(12, 1, 76, 11, "Decision analyses",
    "routing bound (Lyapunov controller against an offline oracle)\n"
    "mitigation levers   |   harm scorecard   |   siting by minimax regret",
    fill="#fdeee6", edge=ORANGE)
arrow(63.5, 35.4, 64, 31.6)
arrow(63.5, 35.4, 38.5, 31.6)
arrow(63.5, 35.4, 12.5, 31.6)
arrow(63.5, 35.4, 89, 31.6)
arrow(12.5, 16.4, 20, 12.6)
arrow(38.5, 16.4, 38.5, 12.6)
arrow(64, 16.4, 64, 12.6)

fig.savefig(Path(__file__).resolve().parents[1] / "fig1_framework.png",
            dpi=300, bbox_inches="tight", facecolor=SURFACE)
print("ok")
