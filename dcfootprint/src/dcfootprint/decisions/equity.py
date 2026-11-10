"""L7 / Q2 extension — C7 groundwater-equity overlay (distributional diagnostic).

Overlays each India facility's scarcity-weighted water burden on the local CGWB
groundwater **stage of extraction** (city/district where available, else state), to
show how much of the burden lands on already over-exploited aquifers. Diagnostic, never
prescriptive. Standalone from the committed account — no pipeline re-run.

CGWB "Stage of GW extraction (%)": >100 = over-exploited (drawing beyond recharge),
90-100 critical, 70-90 semi-critical, <70 safe (CGWB/CGWA categories).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root

# account city -> CGWB district key AFTER _norm() (which strips "urban"/"rural"/parens)
_CITY_ALIASES = {
    "bengaluru": "bangalore",
    "new delhi": "delhi", "gurugram": "gurgaon",
    "greater noida": "gautam buddha nagar", "noida": "gautam buddha nagar",
    "navi mumbai": "thane",
}


def _norm(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.lower()
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\b(urban|rural|district|total)\b", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def categorise(stage: float) -> str:
    if pd.isna(stage):
        return "unknown"
    if stage > 100:
        return "over-exploited"
    if stage >= 90:
        return "critical"
    if stage >= 70:
        return "semi-critical"
    return "safe"


def load_cgwb_stage() -> tuple[dict, dict]:
    root = _repo_root() / "data" / "cgwb"
    city = pd.read_csv(root / "Major_Cities_Groundwater_Availability_Utilizatio.csv")
    state = pd.read_csv(root / "India_Groundwater_Availability_Utilization_and_E.csv")
    cs = next(c for c in city.columns if "Stage of GW" in c)
    ss = next(c for c in state.columns if "Stage of GW" in c)
    cn = next(c for c in city.columns if "District" in c or "City" in c)
    sn = next(c for c in state.columns if "State" in c)
    city_map = dict(zip(_norm(city[cn]), pd.to_numeric(city[cs], errors="coerce")))
    state_map = dict(zip(_norm(state[sn]), pd.to_numeric(state[ss], errors="coerce")))
    return {k: v for k, v in city_map.items() if k}, {k: v for k, v in state_map.items() if k}


def attach_gw_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Add gw_stage_pct / gw_stage_source / gw_category by matching each row's city
    (alias-normalised, CGWB major-cities) else its state to the CGWB stage of extraction.
    India only. Shared with the Q2 scorecard so Q2 carries the groundwater dimension.
    Without the CGWB files (data/cgwb/ is not in git) the columns are added empty (category
    "unknown") and a message is printed, so Q2 still runs -- the same fallback idea as the
    consumption-carbon basis when the Electricity Maps cache is absent."""
    try:
        city_map, state_map = load_cgwb_stage()
    except FileNotFoundError as exc:
        print(f"[equity] CGWB files not found ({exc.filename}); groundwater columns left empty")
        return df.assign(gw_stage_pct=np.nan, gw_stage_source="", gw_category="unknown")
    out = df.copy()
    cnorm = _norm(out["city"]).replace(_CITY_ALIASES)
    snorm = _norm(out["state"])
    out["gw_stage_pct"] = [city_map.get(c, np.nan) for c in cnorm]
    out["gw_stage_source"] = np.where(out["gw_stage_pct"].notna(), "city", "")
    miss = out["gw_stage_pct"].isna()
    out.loc[miss, "gw_stage_pct"] = [state_map.get(s, np.nan) for s in snorm[miss]]
    out.loc[miss & out["gw_stage_pct"].notna(), "gw_stage_source"] = "state"
    out["gw_category"] = out["gw_stage_pct"].map(categorise)
    return out


def _category_shares(ann: pd.DataFrame, col: str, prefix: str) -> dict:
    tot = ann[col].sum()
    by_cat = ann.groupby("gw_category")[col].sum()
    oe, cr = by_cat.get("over-exploited", 0), by_cat.get("critical", 0)
    pct = lambda v: round(100 * v / tot, 1) if tot else 0
    return {f"pct_{prefix}_over_exploited": pct(oe),
            f"pct_{prefix}_critical_or_worse": pct(oe + cr),
            f"pct_{prefix}_semicritical_or_worse": pct(sum(by_cat.get(c, 0) for c in ("over-exploited", "critical", "semi-critical"))),
            f"{prefix}_by_category_m3eq_yr": {k: round(float(v), 0) for k, v in by_cat.sort_values(ascending=False).items()}}


def equity_overlay(account: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Facility burden x local groundwater stage.

    The aquifer claim uses ON-SITE (scope-1) scarcity-weighted water only: that is the only water a
    facility draws locally. Power-plant (scope-2) water is drawn in the generation basins, so counting
    it against the facility's own aquifer (the round-5 headline, 63% over-exploited) mixes places;
    the total-burden shares are kept as `pct_burden_*` for comparison."""
    agg = {"scarcity_m3eq_yr": ("water_scarcity_l_eq", lambda s: s.sum() / 1000.0)}
    has_onsite = "water_scarcity_onsite_l_eq" in account.columns
    if has_onsite:
        agg["scarcity_onsite_m3eq_yr"] = ("water_scarcity_onsite_l_eq", lambda s: s.sum() / 1000.0)
    ann = account.groupby(["facility_id", "operator", "city", "state"], as_index=False).agg(**agg)
    ann = attach_gw_stage(ann)

    summary = {"total_scarcity_m3eq_yr": round(float(ann["scarcity_m3eq_yr"].sum()), 0),
               "coverage": ann["gw_stage_source"].value_counts().to_dict()}
    if has_onsite:
        summary["total_scarcity_onsite_m3eq_yr"] = round(float(ann["scarcity_onsite_m3eq_yr"].sum()), 0)
        summary |= _category_shares(ann, "scarcity_onsite_m3eq_yr", "onsite")
    s = _category_shares(ann, "scarcity_m3eq_yr", "burden")
    summary |= {k: v for k, v in s.items() if k.startswith("pct_")}
    summary["scarcity_by_category_m3eq_yr"] = s["burden_by_category_m3eq_yr"]
    return ann.sort_values("scarcity_m3eq_yr", ascending=False).reset_index(drop=True), summary


if __name__ == "__main__":
    root = _repo_root()
    acct = None
    for p in [root / "dcfootprint" / "results" / "round5_final" / "outputs" / "account_facility_month.parquet",
              root / "dcfootprint" / "outputs" / "account_facility_month.parquet"]:
        if p.exists():
            acct = pd.read_parquet(p); break
    df, s = equity_overlay(acct)
    out = root / "dcfootprint" / "results" / "q2_equity_groundwater.csv"
    df.to_csv(out, index=False)
    print("GROUNDWATER-EQUITY OVERLAY (C7):")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("\ntop 10 facilities by burden x local groundwater stress:")
    print(df.head(10)[["operator", "city", "state", "scarcity_m3eq_yr", "gw_stage_pct", "gw_stage_source", "gw_category"]].to_string(index=False))
    print(f"\nwrote {out}")
