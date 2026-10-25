"""L7 / Q1 — siting: rank candidate (state x basin) regions for a NEW datacenter.

Candidate grid = every (state x AWARE basin) cell holding >=1 operating GEM power plant
(existing grid infrastructure), plus every cell that already holds a datacenter — so the
tool can point to unoccupied regions, not only "least bad among where we already are".

A hypothetical facility of size S (parameters.yaml siting.*) is scored per cell on four
criteria, all lower-is-better:
  new_scarcity_m3eq   its scarcity-weighted water: scope-1 x CF(basin, m) +
                      scope-2 x generation-basin CF of the state's grid (sewif)
  new_carbon_tco2     its carbon at the state's 2024 monthly CI
  basin_pressure_m3   the rise in the basin's peak water queue that the new facility causes
                      (Delta D_b), with the Q3 budget: water left after human use and
                      environmental flows (AWARE AMD x area); months with AMD <= 0 are overdrafts
  grid_fossil_share   operating fossil share of the state's GEM capacity
Legal screen (policy/gap.region_effects): bans exclude a cell; ZLD / renewable-share /
PUE mandates change the new facility's footprint. Ranked with AND without legal effects.

Weights are Dirichlet-sampled; cells are ranked by MINIMAX REGRET (worst-case gap to the
best cell across weightings), with mean rank and a p10-p90 rank band as the stability
check. Output = regions with reasons, not plots (land, fibre and interconnection queues
are not in the data).

2030/2050 scenarios (a separate table; the headline ranking is unchanged): a fifth criterion,
the basin's Aqueduct 4.0 0-5 water-stress score (project/scarcity_future.py), takes seven values
(baseline, bau/opt/pes x 2030/2050); regret is taken over weights x scenarios and the worst case
kept. Carbon follows the region's official path (project/carbon_future.py): in a 2030 or 2050 scenario
the new facility's carbon is its account-year carbon x the zone's path factor for that year, scaled
jointly over all scenarios like the water score (a cleaner future grid narrows the carbon differences).

States generating < siting.small_grid_twh a year are ranked in a SEPARATE table: their
Ember CI is their own (often hydro-only) generation, not what a new 50 MW load would draw
on the grid it imports from (R2), so they are not part of the headline recommendation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.account.energy import month_hours
from dcfootprint.settings import params as _cfg_params

CRITERIA = ["new_scarcity_m3eq", "new_carbon_tco2", "basin_pressure_m3", "grid_fossil_share"]


def _cfg() -> dict:
    return _cfg_params()


def candidate_cells(account: pd.DataFrame, g: dict) -> pd.DataFrame:
    p = g["plants"].dropna(subset=["state"])
    grid_cells = p.groupby(["state", "basin_id"], as_index=False).agg(n_plants=("plant", "size"))
    dc = (account.dropna(subset=["basin_id"])                      # EU (country tier) has no facility basins
          .groupby(["state", "basin_id"], as_index=False).agg(n_existing_dc=("facility_id", "nunique")))
    dc["basin_id"] = dc["basin_id"].astype("int64")
    cells = grid_cells.merge(dc, on=["state", "basin_id"], how="outer").fillna({"n_plants": 0, "n_existing_dc": 0})
    # GEM plant states are GADM-verified (io/gem.assign_state_of_record); no plant-count guard
    cells = cells[cells["state"].isin(set(g["ci"]["zone_id"]))]             # needs a grid CI
    return cells.astype({"basin_id": "int64", "n_plants": int, "n_existing_dc": int})


def score_cells(account: pd.DataFrame, legal: bool = True) -> pd.DataFrame:
    from dcfootprint.account.build import grid_tables
    from dcfootprint.io.gem import state_fossil_share
    from dcfootprint.policy.gap import region_effects
    from dcfootprint.project.recharge import basin_budgets
    cfg = _cfg()
    sc, gcfg = cfg["siting"], cfg["grid"]
    region = str(account["region"].iloc[0])
    year = int(cfg["region_config"][region]["account_year"])
    g = grid_tables(gcfg["zone"], year, region)
    cells = candidate_cells(account, g)

    t = sc["new_facility_type"]
    S = float(sc["new_facility_mw"])
    util = cfg["energy"]["utilisation"]["by_facility_type"][t]["default"]
    pue0 = cfg["energy"]["pue"]["by_facility_type"][t]["default"]
    wue = float(cfg["water_onsite"]["wue_default"]["default"])
    hrs = month_hours(year)
    e_it = hrs.assign(e_it_mwh=S * util * hrs["hours"])[["month", "e_it_mwh"]]          # MWh-IT per month

    cf = g["basin_cf"][g["basin_cf"]["basin_id"].isin(cells["basin_id"])]
    s1 = (cf.merge(e_it, on="month").assign(v=lambda d: d["e_it_mwh"] * 1000 * wue * d["cf"])
          .groupby("basin_id")["v"].sum().rename("s1_l"))                              # before legal
    cfm = cf.groupby("basin_id")["cf"].mean().rename("cf_mean")
    zone = (g["grid_water"].merge(g["ci"], on=["zone_id", "month"]).merge(e_it, on="month")
            .assign(s2=lambda d: d["e_it_mwh"] * d["sewif_l_eq_per_mwh"],
                    c=lambda d: d["e_it_mwh"] * d["ci_gco2_per_kwh"] / 1000)
            .groupby("zone_id").agg(s2_per_pue=("s2", "sum"), c_per_pue=("c", "sum"),
                                    ci_mean=("ci_gco2_per_kwh", "mean")))
    cells = (cells.merge(s1, left_on="basin_id", right_index=True, how="left")
             .merge(cfm, left_on="basin_id", right_index=True, how="left")
             .merge(zone, left_on="state", right_index=True, how="left")
             .merge(state_fossil_share(g["plants"])[["state", "fossil_share"]], on="state", how="left")
             .dropna(subset=["s1_l", "s2_per_pue"]))
    cells = cells.rename(columns={"fossil_share": "grid_fossil_share"})

    # legal effects on the NEW facility (or exclusion)
    cells["legal_flags"], cells["excluded"] = "", False
    zld = np.zeros(len(cells)); re = np.zeros(len(cells)); pue = np.full(len(cells), pue0)
    if legal:
        for i, st in enumerate(cells["state"].values):
            eff = region_effects(st)
            flags = []
            if "ban_new_above_mw" in eff and S >= eff["ban_new_above_mw"]:
                cells.iloc[i, cells.columns.get_loc("excluded")] = True; flags.append("ban")
            if "zld_mandate" in eff:
                zld[i] = eff["zld_mandate"]; flags.append("zld")
            if "re_share_min" in eff:
                re[i] = eff["re_share_min"]; flags.append(f"re>={eff['re_share_min']:.0%}")
            if "pue_cap_new" in eff:
                pue[i] = min(pue0, eff["pue_cap_new"]); flags.append("pue_cap")
            cells.iloc[i, cells.columns.get_loc("legal_flags")] = ",".join(flags)
    cells["new_scarcity_m3eq"] = (cells["s1_l"] * (1 - zld) + cells["s2_per_pue"] * pue) / 1000.0
    cells["new_carbon_tco2"] = cells["c_per_pue"] * pue * (1 - re)
    new_onsite_l = float(e_it["e_it_mwh"].sum() * 1000 * wue) * (1 - zld)

    # marginal basin pressure: Delta D_b = rise in the peak queue D(t+1) = (D + W - R)^+ over a
    # steady year (one spin-up year), with vs without the new facility's scope-1 draw
    cyc = []
    exist_m = (account.dropna(subset=["basin_id"]).pipe(lambda d: d.assign(basin_id=d["basin_id"].astype("int64")))
               .groupby(["basin_id", "month"])["water_onsite_l"].sum())
    bud = basin_budgets(cells["basin_id"].unique(), 1.0).set_index(["basin_id", "month"])["budget_l"]
    new_m = (e_it.set_index("month")["e_it_mwh"] * 1000 * wue)                        # L per month
    def peak(bid, extra):
        D, pk = 0.0, 0.0
        for yr in range(2):
            for m in range(1, 13):
                D = max(D + exist_m.get((bid, m), 0.0) + extra * new_m[m] - bud.get((bid, m), 0.0), 0.0)
                if yr == 1:
                    pk = max(pk, D)
        return pk
    for bid, z in zip(cells["basin_id"], zld):
        cyc.append((peak(bid, 1 - z) - peak(bid, 0.0)) / 1000.0)
    cells["basin_pressure_m3"] = cyc
    cells["overdraft_months"] = [int(sum(bud.get((b, m), 0.0) <= 0 for m in range(1, 13))) for b in cells["basin_id"]]

    # R2 guard: a state's Ember CI is its OWN generation; a small grid that imports most of its
    # power (e.g. hydro-only NE states) shows a CI the datacenter would not actually see.
    from dcfootprint.io import ember
    raw = ember.load_raw(region)
    gen = raw[(raw["Category"] == "Electricity generation") & (raw["Unit"] == "GWh")
              & (raw["Variable"].isin(ember._FUELS)) & (raw["date"].dt.year == year)]
    twh = gen.groupby("State")["Value"].sum() / 1000.0
    cells["state_generation_twh"] = cells["state"].map(twh).round(2)
    cells["small_grid"] = cells["state_generation_twh"] < float(sc.get("small_grid_twh", 10))
    return cells.drop(columns=["s1_l", "s2_per_pue", "c_per_pue"]).reset_index(drop=True)


def _minimax(cells: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    c = cells[~cells["excluded"]].copy()
    X = c[CRITERIA].values.astype(float)
    rng_ = X.max(0) - X.min(0)
    Xn = np.where(rng_ > 0, (X - X.min(0)) / np.where(rng_ > 0, rng_, 1), 0.0)
    W = np.random.default_rng(seed).dirichlet(np.ones(len(CRITERIA)), n)             # n x k
    S = Xn @ W.T                                                                       # cells x n
    regret = S - S.min(axis=0, keepdims=True)
    ranks = S.argsort(axis=0).argsort(axis=0) + 1
    c["max_regret"] = regret.max(axis=1)
    c["mean_rank"] = ranks.mean(axis=1)
    c["rank_p10"] = np.percentile(ranks, 10, axis=1)
    c["rank_p90"] = np.percentile(ranks, 90, axis=1)
    return c.sort_values(["max_regret", "mean_rank"])


def _with_ws(cells: pd.DataFrame, region: str) -> pd.DataFrame:
    """Attach the Aqueduct 0-5 score per scenario (ws_<scenario>), the lowest scored share and the zone's
    carbon path factors (carbon_f2030/2050, new_carbon_tco2_2030/2050); cells whose basin has no score in
    some scenario are dropped (counted by the caller)."""
    from dcfootprint.project.scarcity_future import SCENARIOS, basin_scores
    s = basin_scores(cells["basin_id"].unique())
    wide = s.pivot(index="basin_id", columns="scenario", values="ws_score").add_prefix("ws_")[[f"ws_{x}" for x in SCENARIOS]]
    share = s.groupby("basin_id")["scored_share"].min().rename("ws_scored_share_min")
    out = cells.merge(wide, left_on="basin_id", right_index=True, how="left").merge(share, left_on="basin_id", right_index=True, how="left")
    out = out.dropna(subset=list(wide.columns))
    from dcfootprint.project.carbon_future import zone_factors
    cf = zone_factors(region, out["state"]).rename(columns={"zone_id": "state"})
    out = out.merge(cf, on="state", how="left")
    for y in (2030, 2050):
        out[f"new_carbon_tco2_{y}"] = out["new_carbon_tco2"] * out[f"carbon_f{y}"]
    return out


def _minimax_scenarios(cells: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Minimax regret over weights x scenarios. Criteria = the four baseline criteria plus the basin's
    Aqueduct score under the scenario; carbon takes the scenario year's value (account year x path
    factor). Score and carbon are min-max scaled over ALL scenarios jointly, so a basin that worsens
    in 2050, or a grid that gets cleaner, moves on one common scale."""
    from dcfootprint.project.scarcity_future import SCENARIOS
    c = cells[~cells["excluded"]].copy()
    X = c[CRITERIA].values.astype(float)
    rng_ = X.max(0) - X.min(0)
    Xn = np.where(rng_ > 0, (X - X.min(0)) / np.where(rng_ > 0, rng_, 1), 0.0)
    WS = c[[f"ws_{s}" for s in SCENARIOS]].values.astype(float)
    span = WS.max() - WS.min()
    WSn = (WS - WS.min()) / span if span > 0 else np.zeros_like(WS)
    ic = CRITERIA.index("new_carbon_tco2")
    C = np.column_stack([c["new_carbon_tco2"].values if s == "baseline" else c[f"new_carbon_tco2_20{s[-2:]}"].values
                         for s in SCENARIOS]).astype(float)                           # cells x scenarios
    cspan = C.max() - C.min()
    Cn = (C - C.min()) / cspan if cspan > 0 else np.zeros_like(C)
    W = np.random.default_rng(seed).dirichlet(np.ones(len(CRITERIA) + 1), n)         # n x (k+1)
    regret_s, ranks = [], []
    for j in range(len(SCENARIOS)):
        Xj = Xn.copy()
        Xj[:, ic] = Cn[:, j]
        S = np.hstack([Xj, WSn[:, [j]]]) @ W.T                                        # cells x n
        regret_s.append((S - S.min(axis=0, keepdims=True)).max(axis=1))
        ranks.append(S.argsort(axis=0).argsort(axis=0) + 1)
    R = np.column_stack(regret_s)                                                     # cells x scenarios
    ranks = np.hstack(ranks)
    c["max_regret"] = R.max(axis=1)
    c["worst_scenario"] = np.array(SCENARIOS)[R.argmax(axis=1)]
    for j, s in enumerate(SCENARIOS):
        c[f"max_regret_{s}"] = R[:, j]
    c["mean_rank"] = ranks.mean(axis=1)
    c["rank_p10"] = np.percentile(ranks, 10, axis=1)
    c["rank_p90"] = np.percentile(ranks, 90, axis=1)
    return c.sort_values(["max_regret", "mean_rank"])


def _rank(legal_cells: pd.DataFrame, nolegal_cells: pd.DataFrame, n: int, seed: int, minimax=_minimax) -> pd.DataFrame:
    with_legal = minimax(legal_cells, n, seed).reset_index(drop=True)
    no_legal = minimax(nolegal_cells, n, seed)
    no_legal["rank_nolegal"] = np.arange(1, len(no_legal) + 1)
    out = with_legal.merge(no_legal[["state", "basin_id", "rank_nolegal"]], on=["state", "basin_id"], how="left")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    q = len(out)
    out["recommendation"] = np.select([out["rank"] <= q * 0.25, out["rank"] > q * 0.75],
                                      ["preferable", "avoid"], "middle")
    return out


def rank_sites(account: pd.DataFrame, n_weight_samples: int | None = None, seed: int = 0) -> dict:
    """{'headline': cells on grids >= small_grid_twh, ranked among themselves,
        'small_grids': the excluded small-grid cells, ranked among themselves (caveat: own-
        generation CI is not what a new datacenter would draw),
        'scenarios': headline-grid cells ranked by minimax regret over weights x Aqueduct
        2030/2050 water scenarios, with the headline rank alongside (rank_headline);
        'scenario_unscored': headline cells dropped for lacking an Aqueduct score}."""
    n = int(_cfg()["siting"]["weight_samples"]) if n_weight_samples is None else n_weight_samples
    lc, nc = score_cells(account, legal=True), score_cells(account, legal=False)
    big = lambda d: d[~d["small_grid"]]
    small = lambda d: d[d["small_grid"]]
    headline = _rank(big(lc), big(nc), n, seed)
    region = str(account["region"].iloc[0])
    lw, nw = _with_ws(big(lc), region), _with_ws(big(nc), region)
    scen = _rank(lw, nw, n, seed, minimax=_minimax_scenarios)
    scen = scen.merge(headline[["state", "basin_id", "rank"]].rename(columns={"rank": "rank_headline"}),
                      on=["state", "basin_id"], how="left")
    unscored = big(lc).merge(lw[["state", "basin_id"]], on=["state", "basin_id"], how="left", indicator=True)
    return {"headline": headline,
            "small_grids": _rank(small(lc), small(nc), n, seed),
            "scenarios": scen,
            "scenario_unscored": unscored[unscored["_merge"] == "left_only"].drop(columns="_merge")}


def scenario_summary(both: dict, top: int = 5) -> str:
    """Report text for the 2030/2050 scenario table (shared by pipeline.py and regions.py)."""
    s, h = both["scenarios"], both["headline"]
    if not len(s):
        return ""
    top_h = set(zip(h.head(top)["state"], h.head(top)["basin_id"]))
    kept = sum((a, b) in top_h for a, b in zip(s.head(top)["state"], s.head(top)["basin_id"]))
    lines = [f"  {int(r['rank'])}. {r['state']} basin {int(r['basin_id'])} (headline #{int(r['rank_headline'])}): "
             f"Aqueduct score {r['ws_baseline']:.2f} now -> {r['ws_bau50']:.2f} bau 2050 "
             f"(pes 2050 {r['ws_pes50']:.2f}); carbon x{r['carbon_f2030']:.2f} in 2030, x{r['carbon_f2050']:.2f} in 2050; "
             f"worst case in {r['worst_scenario']}"
             for _, r in s.head(top).iterrows()]
    return (f"- **2030/2050 water scenarios** (q1_siting_scenarios.csv): minimax regret over weights x 7 Aqueduct 4.0 "
            f"scenarios (baseline, bau/opt/pes x 2030/2050; 0-5 scores as a fifth criterion; carbon on the official "
            f"path, {s['carbon_path_source'].iloc[0]}). {len(s)} cells scored, {len(both['scenario_unscored'])} without a score; {kept} of the headline "
            f"top {top} stay in the scenario top {top}:\n" + "\n".join(lines))


if __name__ == "__main__":
    acct = pd.read_parquet(_repo_root() / "dcfootprint" / "outputs" / "account_facility_month.parquet")
    both = rank_sites(acct)
    r = both["headline"]
    print(f"small-grid cells ranked separately: {len(both['small_grids'])}")
    print(f"Q1 SITING — {len(r)} candidate cells ({int((r['n_existing_dc'] > 0).sum())} already hold a DC); minimax regret:\n")
    cols = ["rank", "state", "basin_id", "n_existing_dc", "cf_mean", "ci_mean", "grid_fossil_share",
            "new_scarcity_m3eq", "basin_pressure_m3", "overdraft_months", "max_regret", "rank_p10", "rank_p90", "rank_nolegal", "legal_flags"]
    print(r.head(12)[cols].round(3).to_string(index=False))
    print("\nexisting-DC cells:\n", r[r["n_existing_dc"] > 0][cols].round(3).to_string(index=False))
