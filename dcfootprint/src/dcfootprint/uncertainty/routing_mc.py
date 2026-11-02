"""L8 - Monte Carlo over the Q3 routing comparison: does the carbon/water trade-off survive?

uncertainty/monte_carlo.py puts intervals on the account totals. The routing headline (the
carbon change when load is shifted to cut scarcity-weighted water) had no interval: it was one
run at the point estimate. Here the whole comparison is re-run per draw:

  1. draw the same parameter bands as monte_carlo.py (util and PUE per facility type, WUE,
     EWIF, hydro multiplier), plus a grid-CI multiplier per CI GROUP (uncertainty.ci_zone_band)
     and a fresh demand-noise seed. A CI group is the set of zones that carry the identical
     monthly CI series (e.g. Karnataka, Tamil Nadu and Telangana all take Electricity Maps'
     IN-SO grid): one grid, one draw, so the MC never invents CI gaps between them;
  2. rescale the account with the same scaling rules monte_carlo.py uses (scale_account);
  3. run static / greedy / lyapunov / lyapunov_pf / oracle on it (routing/lyapunov.py);
  4. record each policy's carbon and scarcity change vs static.

Reported: the distribution of each change, the share of draws in which carbon rises, and
first-order sensitivity indices of the lyapunov carbon change. The controller's CI forecast
keeps its base-run relative error in every draw (ci_hat scales with the zone's CI multiplier).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dcfootprint.settings import params as _cfg_params
from dcfootprint.uncertainty.monte_carlo import _bands, _draw, _first_order

POLICIES = ("static", "greedy", "lyapunov", "lyapunov_pf", "oracle")


def scale_account(account: pd.DataFrame, x: dict) -> pd.DataFrame:
    """Account rescaled by one draw `x` (multipliers; 1.0 = point estimate):
      util_<type>, pue_<type>  per facility type    wue, ewif, hydro    ci_<zone_id> per zone
    E_IT ~ util; E_grid ~ util*pue; carbon ~ util*pue*ci; on-site water ~ util*wue;
    grid water ~ util*pue*ewif (non-hydro) and util*pue*hydro (hydro). Scarcity columns scale
    with their physical counterparts (CFs are not drawn). Totals are rebuilt from the parts."""
    a = account.copy()
    t = a["facility_type"].astype(str)
    u = t.map(lambda k: x.get(f"util_{k}", 1.0)).to_numpy(float)
    p = t.map(lambda k: x.get(f"pue_{k}", 1.0)).to_numpy(float)
    groups = ci_groups(a)
    k = a["zone_id"].astype(str).map(lambda z: x.get(f"ci_{groups[z]}", 1.0)).to_numpy(float)
    wue, ewif, hydro = x.get("wue", 1.0), x.get("ewif", 1.0), x.get("hydro", 1.0)

    nh = (a["water_grid_l"] - a["water_grid_hydro_l"]).to_numpy(float)
    snh = (a["water_scarcity_grid_l_eq"] - a["water_scarcity_grid_hydro_l_eq"]).to_numpy(float)
    a["e_it_mwh"] = a["e_it_mwh"] * u
    a["e_grid_mwh"] = a["e_grid_mwh"] * u * p
    a["carbon_tco2"] = a["carbon_tco2"] * u * p * k
    a["ci_gco2_per_kwh"] = a["ci_gco2_per_kwh"] * k
    if "ci_hat" in a.columns:
        a["ci_hat"] = a["ci_hat"] * k
    a["water_onsite_l"] = a["water_onsite_l"] * u * wue
    a["water_scarcity_onsite_l_eq"] = a["water_scarcity_onsite_l_eq"] * u * wue
    a["water_grid_hydro_l"] = a["water_grid_hydro_l"] * u * p * hydro
    a["water_scarcity_grid_hydro_l_eq"] = a["water_scarcity_grid_hydro_l_eq"] * u * p * hydro
    a["water_grid_l"] = nh * u * p * ewif + a["water_grid_hydro_l"]
    a["water_scarcity_grid_l_eq"] = snh * u * p * ewif + a["water_scarcity_grid_hydro_l_eq"]
    a["water_phys_l"] = a["water_onsite_l"] + a["water_grid_l"]
    a["water_scarcity_l_eq"] = a["water_scarcity_onsite_l_eq"] + a["water_scarcity_grid_l_eq"]
    return a


def ci_groups(account: pd.DataFrame) -> dict[str, str]:
    """zone_id -> CI group label: zones with the identical 12-month CI series share a group,
    named after its zones joined by '+'."""
    z = account.groupby(["zone_id", "month"])["ci_gco2_per_kwh"].first().unstack().round(6)
    key = z.apply(lambda r: tuple(r.to_numpy()), axis=1)
    label = {k: "+".join(sorted(map(str, idx))) for k, idx in key.groupby(key).groups.items()}
    return {str(zone): label[key[zone]] for zone in z.index}


def _ranges(account: pd.DataFrame) -> dict:
    b = _bands()
    ci_band = tuple(_cfg_params()["uncertainty"].get("ci_zone_band", (1.0, 1.0)))
    r = {}
    for t in sorted(account["facility_type"].astype(str).unique()):
        r[f"util_{t}"] = b["util"].get(t, b["util"]["unknown"])
        r[f"pue_{t}"] = b["pue"].get(t, b["pue"]["unknown"])
    r.update({"wue": b["wue"], "ewif": b["ewif"], "hydro": b["hydro"]})
    for g in sorted(set(ci_groups(account).values())):
        r[f"ci_{g}"] = ci_band
    return r


def _one_run(account: pd.DataFrame, A_basin, seed: int, legal: bool) -> dict:
    from dcfootprint.routing import lyapunov
    P = lyapunov.prepare(account, legal=legal, A_basin=A_basin)
    out = {}
    for pol in ("static", "greedy", "lyapunov", "lyapunov_pf"):
        out[pol] = lyapunov.simulate(P, pol, seed=seed)
    out["oracle"] = lyapunov.oracle(P, out["lyapunov"]["_qpeak_by_basin"], seed=seed)
    base = out["static"]
    row = {}
    for pol in POLICIES:
        row[f"{pol}_carbon_change_pct"] = (out[pol]["carbon_tco2"] / base["carbon_tco2"] - 1) * 100
        row[f"{pol}_scarcity_change_pct"] = (out[pol]["scarcity_m3eq"] / base["scarcity_m3eq"] - 1) * 100
    return row


def routing_monte_carlo(account: pd.DataFrame, n: int | None = None, seed: int = 0, legal: bool = True,
                        A_basin: pd.DataFrame | None = None,
                        distribution: str | None = None) -> tuple[pd.DataFrame, dict]:
    """(draws: one row per draw with every multiplier and each policy's % change vs static,
    summary dict). Change = (policy - static) / static * 100: positive carbon = MORE carbon."""
    from dcfootprint.routing import lyapunov
    cfg = _cfg_params()["uncertainty"]
    n = int(cfg.get("routing_n_samples", 1000)) if n is None else n
    dist = cfg["distribution"] if distribution is None else distribution
    base = account.sort_values(["facility_id", "month"]).reset_index(drop=True).copy()
    base["ci_hat"] = lyapunov._ci_forecast(base)            # base-run forecast, rescaled per draw
    ranges = _ranges(base)
    rng = np.random.default_rng(seed)
    draws = {k: _draw(rng, lo, 1.0, hi, n, dist) for k, (lo, hi) in ranges.items()}

    rows = []
    for i in range(n):
        x = {k: float(v[i]) for k, v in draws.items()}
        rows.append({**x, **_one_run(scale_account(base, x), A_basin, seed=i, legal=legal)})
    df = pd.DataFrame(rows)
    return df, summarise(df, list(ranges), dist, legal)


def summarise(df: pd.DataFrame, params: list[str], dist: str, legal: bool) -> dict:
    def q(v):
        v = np.asarray(v, float)
        return {"mean": round(float(v.mean()), 3), "p05": round(float(np.percentile(v, 5)), 3),
                "p50": round(float(np.percentile(v, 50)), 3), "p95": round(float(np.percentile(v, 95)), 3)}

    out = {"n_draws": len(df), "distribution": dist, "legal": legal,
           "sign_convention": "change vs static in %; positive carbon = more carbon, negative scarcity = water saved"}
    for pol in POLICIES[1:]:
        c, s = df[f"{pol}_carbon_change_pct"], df[f"{pol}_scarcity_change_pct"]
        out[pol] = {"carbon_change_pct": q(c), "scarcity_change_pct": q(s),
                    "share_draws_carbon_up": round(float((c > 0).mean()), 3),
                    "share_draws_scarcity_down": round(float((s < 0).mean()), 3)}
    y = df["lyapunov_carbon_change_pct"].to_numpy(float)
    bins = int(np.clip(len(df) // 25, 2, 40))       # the binned estimator needs many draws per bin
    sens = {k: round(_first_order(df[k].to_numpy(float), y, bins=bins), 3) for k in params if df[k].std() > 0}
    out["lyapunov_carbon_first_order"] = dict(sorted(sens.items(), key=lambda kv: -kv[1]))
    return out
