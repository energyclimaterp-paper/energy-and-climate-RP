"""L2b / C3 — SEASONAL "when": the account is a monthly panel, so surface WHEN the joint
carbon + scarcity-water burden peaks and whether the two co-peak (dry-season concentration).

The "where" is RQ1's spatial account; this is the "when" half of contribution C3, which the
annual summaries leave latent. Standalone from the account parquet (like decisions/equity.py);
writes a per-region exhibit. Scarcity seasonality is driven by the AWARE monthly CF (dry-season
high), carbon by the monthly grid CI.
"""
from __future__ import annotations

import pandas as pd

from dcfootprint.io.facilities import _repo_root

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def seasonal_profile(account: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """[month, carbon_tco2, scarcity_m3eq, carbon_share_pct, scarcity_share_pct] + a summary of
    the peaks, whether they coincide, and how concentrated the year is."""
    m = account.groupby("month", as_index=False).agg(
        carbon_tco2=("carbon_tco2", "sum"),
        scarcity_m3eq=("water_scarcity_l_eq", lambda s: s.sum() / 1000.0)).sort_values("month").reset_index(drop=True)
    ctot, stot = m["carbon_tco2"].sum(), m["scarcity_m3eq"].sum()
    m["carbon_share_pct"] = (100 * m["carbon_tco2"] / ctot).round(1)
    m["scarcity_share_pct"] = (100 * m["scarcity_m3eq"] / stot).round(1)
    pc = int(m.loc[m["carbon_tco2"].idxmax(), "month"])
    ps = int(m.loc[m["scarcity_m3eq"].idxmax(), "month"])
    top3 = m.nlargest(3, "scarcity_m3eq")
    summary = {
        "peak_carbon_month": _MONTHS[pc - 1],
        "peak_scarcity_month": _MONTHS[ps - 1],
        "peaks_coincide_within_1mo": bool(min(abs(pc - ps), 12 - abs(pc - ps)) <= 1),
        "top3_scarcity_months": [_MONTHS[x - 1] for x in sorted(top3["month"])],
        "top3_scarcity_share_pct": round(float(top3["scarcity_share_pct"].sum()), 1),
        "flat_would_be_pct": 25.0,                       # 3 of 12 months if perfectly flat
        "scarcity_corr_carbon": round(float(m["scarcity_m3eq"].corr(m["carbon_tco2"])), 2),
        "scarcity_seasonality_cv": round(float(m["scarcity_m3eq"].std() / m["scarcity_m3eq"].mean()), 3),
    }
    return m, summary


def _load(region: str) -> pd.DataFrame | None:
    root = _repo_root() / "dcfootprint" / "outputs"
    p = root / ("account_facility_month.parquet" if region == "India" else f"account_{region.lower()}.parquet")
    return pd.read_parquet(p) if p.exists() else None


if __name__ == "__main__":
    res = _repo_root() / "dcfootprint" / "results"
    lines = ["# Seasonal profile (C3 - 'when') - joint carbon + scarcity-water by month\n",
             "*When the burden peaks, and whether carbon and scarcity co-peak. From the monthly account; "
             "the 'where' is RQ1's spatial result, this is the 'when'. Scarcity seasonality = AWARE monthly CF.*\n"]
    for region in ["India", "US", "EU"]:
        acct = _load(region)
        if acct is None or "month" not in acct.columns:
            lines.append(f"## {region}\n- (no monthly account on disk)\n"); continue
        m, s = seasonal_profile(acct)
        outdir = res if region == "India" else res / region.lower()
        outdir.mkdir(parents=True, exist_ok=True)
        m.to_csv(outdir / "seasonal_profile.csv", index=False)
        lines.append(
            f"## {region}\n"
            f"- Scarcity-water peaks in **{s['peak_scarcity_month']}**, carbon in **{s['peak_carbon_month']}** "
            f"(co-peak within 1 month: **{s['peaks_coincide_within_1mo']}**).\n"
            f"- Top-3 months hold **{s['top3_scarcity_share_pct']}%** of annual scarcity water "
            f"({', '.join(s['top3_scarcity_months'])}) vs 25% if flat.\n"
            f"- Monthly carbon-vs-scarcity correlation **{s['scarcity_corr_carbon']}**; "
            f"scarcity seasonality CV **{s['scarcity_seasonality_cv']}**.\n")
        print(f"{region}: scarcity peak {s['peak_scarcity_month']}, carbon peak {s['peak_carbon_month']}, "
              f"top3 {s['top3_scarcity_share_pct']}%, corr {s['scarcity_corr_carbon']}")
    (res / "seasonal_profile.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote results/seasonal_profile.md + per-region seasonal_profile.csv")
