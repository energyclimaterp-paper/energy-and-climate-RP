"""L7 / Q2 — existing-DC environmental scorecard.

Per facility: annual carbon, physical water, scarcity-weighted water, each as a
RELATIVE PERCENTILE within India (robust to the util/PUE scalar, red-team R1) plus
the absolute value. A harm flag fires when scarcity-weighted water is in the top
quartile AND the facility's basin is high-scarcity. Flagged facilities get the lever
that saves the most scarcity-weighted water AT THAT FACILITY (counterfactual/levers.py),
and every facility carries its jurisdiction's legal obligations (policy/gap.py).
"""
from __future__ import annotations

import pandas as pd


def scorecard(account: pd.DataFrame) -> pd.DataFrame:
    from dcfootprint.counterfactual.levers import facility_lever_savings
    from dcfootprint.policy.gap import HARD_CONSTRAINTS
    ann = account.groupby(["facility_id", "operator", "city", "state", "basin_id"], as_index=False, dropna=False).agg(  # EU: no basin/city
        carbon_tco2_yr=("carbon_tco2", "sum"),
        water_phys_m3_yr=("water_phys_l", lambda s: s.sum() / 1000.0),
        scarcity_m3eq_yr=("water_scarcity_l_eq", lambda s: s.sum() / 1000.0),
        scarcity_onsite_m3eq_yr=("water_scarcity_onsite_l_eq", lambda s: s.sum() / 1000.0),
        mean_cf=("cf", "mean"),
        mean_ci_gco2_per_kwh=("ci_gco2_per_kwh", "mean"),
        capacity_mw=("capacity_mw", "first"),
    )
    ann["carbon_per_mw"] = ann["carbon_tco2_yr"] / ann["capacity_mw"]
    ann["scarcity_per_mw"] = ann["scarcity_m3eq_yr"] / ann["capacity_mw"]
    for col, name in [("carbon_tco2_yr", "carbon_pctile"),
                      ("water_phys_m3_yr", "water_pctile"),
                      ("scarcity_m3eq_yr", "scarcity_pctile"),
                      ("carbon_per_mw", "carbon_intensity_pctile"),
                      ("scarcity_per_mw", "scarcity_intensity_pctile")]:
        ann[name] = (ann[col].rank(pct=True) * 100).round(0)

    top_q = ann["scarcity_m3eq_yr"].quantile(0.75)
    high_cf = ann["mean_cf"].quantile(0.60)
    ann["harm_flag"] = (ann["scarcity_m3eq_yr"] >= top_q) & (ann["mean_cf"] >= high_cf)
    ann["verdict"] = pd.cut(ann["scarcity_pctile"], [-1, 50, 75, 101],
                            labels=["better than peers", "typical", "worse than peers"])

    lev = facility_lever_savings(account)
    best = lev.sort_values("scarcity_saved_m3eq_yr", ascending=False).drop_duplicates("facility_id")
    ann = ann.merge(best[["facility_id", "lever", "scarcity_saved_m3eq_yr"]]
                    .rename(columns={"lever": "best_lever", "scarcity_saved_m3eq_yr": "best_lever_saving_m3eq_yr"}),
                    on="facility_id", how="left")
    ann["recommended_lever"] = ann["best_lever"].where(ann["harm_flag"], "none")

    hard = HARD_CONSTRAINTS[HARD_CONSTRAINTS["type"] == "hard"].groupby("region")["rule"].apply("; ".join)
    ann["legal_obligations"] = ann["state"].map(hard).fillna("none (no hard rule in the regulation matrix)")

    region = str(account["region"].iloc[0]) if "region" in account.columns and len(account) else ""
    if region == "India":                               # fuse the C7 groundwater-equity dimension into Q2
        from dcfootprint.decisions.equity import attach_gw_stage
        ann = attach_gw_stage(ann)
    return ann.sort_values("scarcity_m3eq_yr", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from dcfootprint.io.facilities import _repo_root
    acct = pd.read_parquet(_repo_root() / "dcfootprint" / "outputs" / "account_facility_month.parquet")
    sc = scorecard(acct)
    print(f"facilities scored: {len(sc)} | harm-flagged: {int(sc['harm_flag'].sum())}")
    print(sc.head(8)[["operator", "city", "state", "carbon_pctile", "scarcity_pctile", "harm_flag", "recommended_lever"]].to_string(index=False))
