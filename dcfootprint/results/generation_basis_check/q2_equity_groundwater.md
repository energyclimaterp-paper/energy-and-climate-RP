# Groundwater-equity overlay (C7) — India

Each facility's **scarcity-weighted water burden** overlaid on local **CGWB groundwater stage of extraction**
(city/district where available, else state). Standalone from the canonical `round3-final` account; no re-run.
CGWB categories: >100% = over-exploited (drawn beyond recharge) · 90–100 critical · 70–90 semi-critical · <70 safe.

## Finding
- **63.3% of India's DC scarcity-weighted water burden lands on OVER-EXPLOITED aquifers (>100% extraction).**
- **81.2% lands on semi-critical-or-worse groundwater.**
- The top hubs draw from grossly over-extracted groundwater: **Bengaluru 187%**, **Chennai 125%**.
- By category (of 846.6 M m³-eq/yr): over-exploited **536 M**, semi-critical **152 M**, safe **159 M**.
- Coverage: 41 facilities matched at city/district level, 30 at state level (71 total).

## Why it matters
The burden is concentrated exactly where groundwater is **already drawn beyond recharge** — a distributional /
environmental-justice concern, and it **reinforces the India routing result**: load-shifting can't route out of the
overdraft because the hubs' own aquifers are over-exploited. Diagnostic, **not prescriptive** (red-team guidance).

## Caveats
- 30/71 facilities use the **state** stage (coarser) where the city/district isn't in the CGWB major-cities file.
- CGWB stage is **annual, district/state** — not basin-level or monthly; it complements (not replaces) the
  AWARE basin CF used in the account.
- Datacenter coordinates are city-centroids.

*Source: `data/cgwb/` (Major-Cities + India availability) × the canonical India account →
`results/q2_equity_groundwater.csv`. Module: `decisions/equity.py`.*
