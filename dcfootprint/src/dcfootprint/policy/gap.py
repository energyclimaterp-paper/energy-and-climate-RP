"""L5 — regulatory four-axis gap map (C5 / RQ3), DERIVED from the policy registry.

The thesis is now computed, not asserted: for each (axis x jurisdiction) we check whether
ANY in-force instrument in the registry mandates that axis. None do, so every cell is "No"
and 100% of the measured burden falls in a blind spot — but if an instrument ever mandated
an axis, the matrix and the overlay would change automatically.

Everything reads one source (`policy/registry.py`). The binding-constraints interface used
by routing (Q3) and siting (Q1) is preserved: `HARD_CONSTRAINTS` (columns type/region/rule/
effect/param) and `region_effects(region) -> {effect: param}`.
"""
from __future__ import annotations

import pandas as pd

from dcfootprint.policy.registry import load_registry, AXES, JURISDICTIONS

_REGISTRY = load_registry()
# binding-constraints view (what regulation DOES require) — preserves the old schema
HARD_CONSTRAINTS = (_REGISTRY[_REGISTRY["type"].isin(["hard", "soft"])]
                    [["jurisdiction", "region", "rule", "type", "effect", "param", "source"]]
                    .reset_index(drop=True))


def registry() -> pd.DataFrame:
    return _REGISTRY


def four_axis_matrix() -> pd.DataFrame:
    """Derived: per (axis x jurisdiction), does any in-force instrument mandate it? + the
    nearest ('closest') instrument that touches the axis without mandating it."""
    reg = _REGISTRY
    rows = []
    for ax in AXES:
        cell = {"axis": ax}
        for j in JURISDICTIONS:
            mandated = bool(((reg["jurisdiction"] == j) & reg["in_force"].astype(bool)
                             & reg["mandates_axis"].fillna("").str.contains(ax)).any())
            cell[j] = "Yes" if mandated else "No"
        cl = reg[reg["closest_for"] == ax]
        cell["closest_instrument"] = (f"{cl['instrument'].iloc[0]} - {cl['closest_note'].iloc[0]}"
                                      if len(cl) else "(none found)")
        rows.append(cell)
    return pd.DataFrame(rows)


def gap_overlay(account: pd.DataFrame) -> dict:
    """Computed overlay: the share of measured burden whose axes are ALL unmandated in its
    jurisdiction. With 0/12 cells mandated this is 100%, but it is derived, not hardcoded."""
    m = four_axis_matrix()
    region = str(account["region"].iloc[0]) if "region" in account.columns else "India"
    mandated_here = [ax for ax in AXES
                     if region in JURISDICTIONS and m.loc[m["axis"] == ax, region].iloc[0] == "Yes"]
    axes_mandated_anywhere = int((m[JURISDICTIONS] == "Yes").to_numpy().sum())

    ann = account.groupby("facility_id").agg(
        scarcity=("water_scarcity_l_eq", lambda s: s.sum() / 1000.0),
        carbon=("carbon_tco2", "sum")).reset_index()
    total_scarcity = float(ann["scarcity"].sum())
    blind = total_scarcity if not mandated_here else 0.0        # all axes unmandated -> all burden blind
    return {
        "region": region,
        "n_facilities": int(len(ann)),
        "total_scarcity_m3eq": round(total_scarcity, 0),
        "total_carbon_tco2": round(float(ann["carbon"].sum()), 0),
        "axes_mandated_in_region": mandated_here,
        "axes_mandated_anywhere": axes_mandated_anywhere,
        "pct_burden_in_blind_spot": round(100.0 * blind / total_scarcity, 1) if total_scarcity else 0.0,
    }


def region_effects(region: str, constraints: pd.DataFrame | None = None) -> dict:
    """{effect: param} for the HARD rules that apply in `region` (state / country name).
    Effects understood downstream (unchanged):
      zld_mandate · re_share_min · ban_new_above_mw · pue_cap_new · water_permit_mgal."""
    c = HARD_CONSTRAINTS if constraints is None else constraints
    rows = c[(c["type"] == "hard") & (c["region"] == region)]
    return {r["effect"]: float(r["param"]) for _, r in rows.iterrows()
            if str(r["effect"]) != "" and pd.notna(r["param"]) and str(r["param"]) != ""}


if __name__ == "__main__":
    print("FOUR-AXIS GAP MATRIX (derived):")
    print(four_axis_matrix().to_string(index=False))
    print(f"\naxes mandated anywhere (of {len(AXES)*len(JURISDICTIONS)} cells):",
          int((four_axis_matrix()[JURISDICTIONS] == "Yes").to_numpy().sum()))
    print("\nbinding constraints (feed Q1 siting / Q3 routing):")
    print(HARD_CONSTRAINTS.to_string(index=False))
    print("\nregion_effects examples:",
          {r: region_effects(r) for r in ["Rajasthan", "Germany", "Minnesota", "Gujarat"]})
