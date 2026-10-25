"""Does scarcity weighting change the ranking? Physical vs scarcity-weighted water, per region.

Reads a canonical snapshot (default results/round5_final) and compares, for facilities and for states
(EU: countries), the ranking by physical water (m3/yr) with the ranking by scarcity-weighted water
(m3-eq/yr): Spearman rank correlation, overlap of the top 5, and the top unit under each. Descriptive only;
nothing in the account changes.

Run: PYTHONPATH=dcfootprint/src python dcfootprint/experiments/method_rerank.py [results/round5_final]
Writes tables/method_rerank.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]


def main(snap: Path) -> pd.DataFrame:
    rows = []
    for region, f in [("India", "india_account_summary.csv"), ("US", "us/us_account_summary.csv"),
                      ("EU", "eu/eu_account_summary.csv")]:
        s = pd.read_csv(snap / f)
        for level in (["facility_id", "state"] if region != "EU" else ["state"]):
            g = s.groupby(level)[["water_phys_m3_yr", "water_scarcity_m3eq_yr"]].sum()
            k = min(5, len(g))
            top_p = set(g["water_phys_m3_yr"].nlargest(k).index)
            top_s = set(g["water_scarcity_m3eq_yr"].nlargest(k).index)
            rows.append({"region": region, "level": "facility" if level == "facility_id" else ("country" if region == "EU" else "state"),
                         "n": len(g),
                         "spearman_physical_vs_scarcity": round(float(spearmanr(g["water_phys_m3_yr"], g["water_scarcity_m3eq_yr"]).statistic), 3),
                         f"top{k}_overlap": f"{len(top_p & top_s)}/{k}",
                         "top_by_physical": g["water_phys_m3_yr"].idxmax(),
                         "top_by_scarcity": g["water_scarcity_m3eq_yr"].idxmax()})
    out = pd.DataFrame(rows)
    dest = ROOT / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "method_rerank.csv", index=False)
    return out


if __name__ == "__main__":
    snap = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dcfootprint" / "results" / "round5_final"
    print(main(snap).to_string(index=False))
