"""L2.5 — calibration: bottom-up totals vs independent aggregates.

Validates magnitude (not distribution) against CEEW/JLL India (~1.5-1.8 GW
operational). Absolutes are then reported as "consistent-with" ranges, never point
estimates (red-team R1). This is Gate 1, as a module.
"""
from __future__ import annotations

import pandas as pd

CEEW_LOW_MW, CEEW_HIGH_MW = 1500.0, 1800.0   # CEEW/JLL India operational range


def calibrate_capacity(fac: pd.DataFrame) -> dict:
    op = fac[(fac["status"] == "Operational") & fac["capacity_mw"].notna()]
    total = float(op["capacity_mw"].sum())
    ratio = total / CEEW_LOW_MW
    return {
        "bottom_up_operational_mw": round(total, 1),
        "n_operational_costed": int(len(op)),
        "ceew_jll_range_mw": [CEEW_LOW_MW, CEEW_HIGH_MW],
        "ratio_vs_low": round(ratio, 2),
        "verdict": "PASS (consistent)" if 0.7 <= total / CEEW_LOW_MW <= 1.2 else "REVIEW",
    }
