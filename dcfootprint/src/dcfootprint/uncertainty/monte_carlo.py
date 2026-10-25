"""L8 — Monte-Carlo uncertainty over the modelled parameter bands.

Absolute footprints rest on util/PUE/WUE/EWIF assumptions (red-team R1), so we propagate
their bands (parameters.yaml) to intervals on the totals. Analytic scaling of the account
components per facility type (fast, exact) rather than re-running the pipeline:
  carbon         ~ util_t * pue_t
  onsite W       ~ util_t * wue      (band 0.7-2.5 L/kWh around the 1.9 default)
  grid W (non-hydro) ~ util_t * pue_t * ewif
  grid W (hydro) ~ util_t * pue_t * hydro     (hydro multiplier on Macknick's 17,000 L/MWh)
  inference-attributed = total * inference_share
Util and PUE are drawn per facility type from that type's band (hyperscale / colocation /
...), independently across types. Draws are triangular with the mode at the point estimate
(uncertainty.distribution; uniform is available as a sensitivity). Sensitivity: first-order Sobol indices estimated from the
same samples (variance of the binned conditional mean; SALib not required), plus the
one-at-a-time swing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params


def _bands() -> dict:
    p = _cfg_params()
    rel = lambda d: (d["band"][0] / d["default"], d["band"][1] / d["default"])
    return {
        "util": {t: rel(v) for t, v in p["energy"]["utilisation"]["by_facility_type"].items()},
        "pue": {t: rel(v) for t, v in p["energy"]["pue"]["by_facility_type"].items()},
        "wue": rel(p["water_onsite"]["wue_default"]),
        "ewif": (0.8, 1.2),
        "hydro": tuple(p["water_grid"]["hydro_multiplier_band"]),
        "inference": tuple(p["inference"]["sectoral_share"]["band"]),
        "inference_default": p["inference"]["sectoral_share"]["default"],
        "distribution": p.get("uncertainty", {}).get("distribution", "triangular"),
        "n_samples": int(p.get("uncertainty", {}).get("n_samples", 4000)),
    }


def _components(account: pd.DataFrame) -> pd.DataFrame:
    a = account.assign(
        grid_nh=account["water_grid_l"] - account["water_grid_hydro_l"],
        sc_grid_nh=account["water_scarcity_grid_l_eq"] - account["water_scarcity_grid_hydro_l_eq"])
    return a.groupby("facility_type").agg(
        carbon=("carbon_tco2", "sum"),
        onsite=("water_onsite_l", lambda s: s.sum() / 1000), sc_onsite=("water_scarcity_onsite_l_eq", lambda s: s.sum() / 1000),
        grid_nh=("grid_nh", lambda s: s.sum() / 1000), sc_grid_nh=("sc_grid_nh", lambda s: s.sum() / 1000),
        grid_h=("water_grid_hydro_l", lambda s: s.sum() / 1000),
        sc_grid_h=("water_scarcity_grid_hydro_l_eq", lambda s: s.sum() / 1000))


def _totals(comp: pd.DataFrame, x: dict) -> dict:
    carbon = phys = scar = 0.0
    for t, r in comp.iterrows():
        u, p = x[f"util_{t}"], x[f"pue_{t}"]
        carbon = carbon + r["carbon"] * u * p
        phys = phys + r["onsite"] * u * x["wue"] + (r["grid_nh"] * x["ewif"] + r["grid_h"] * x["hydro"]) * u * p
        scar = scar + r["sc_onsite"] * u * x["wue"] + (r["sc_grid_nh"] * x["ewif"] + r["sc_grid_h"] * x["hydro"]) * u * p
    return {"carbon": carbon, "phys": phys, "scarcity": scar,
            "carbon_inference": carbon * x["inference"], "scarcity_inference": scar * x["inference"]}


def _first_order(x: np.ndarray, y: np.ndarray, bins: int = 40) -> float:
    q = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, bins - 1)
    means = np.array([y[idx == k].mean() for k in range(bins) if (idx == k).any()])
    counts = np.array([(idx == k).sum() for k in range(bins) if (idx == k).any()])
    return float(np.sum(counts * (means - y.mean()) ** 2) / len(y) / y.var()) if y.var() > 0 else 0.0


def _draw(rng, lo, mode, hi, n, dist):
    if dist == "uniform" or not (lo < hi):
        return rng.uniform(lo, hi, n)
    return rng.triangular(lo, min(max(mode, lo), hi), hi, n)


def monte_carlo(account: pd.DataFrame, n: int | None = None, seed: int = 0, distribution: str | None = None,
                measured: bool = False) -> dict:
    """measured=True (EU): energy and on-site water are REPORTED, so util/PUE/WUE are not
    sampled (held at 1); only EWIF, hydro and the inference share vary."""
    comp = _components(account)
    b = _bands()
    n = b["n_samples"] if n is None else n
    dist = b["distribution"] if distribution is None else distribution
    rng = np.random.default_rng(seed)
    ranges = {}
    for t in comp.index:
        ranges[f"util_{t}"] = b["util"].get(t, b["util"]["unknown"])
        ranges[f"pue_{t}"] = b["pue"].get(t, b["pue"]["unknown"])
    ranges.update({"wue": b["wue"], "ewif": b["ewif"], "hydro": b["hydro"], "inference": b["inference"]})
    if measured:
        ranges = {k: ((1.0, 1.0) if k.startswith(("util_", "pue_")) or k == "wue" else v) for k, v in ranges.items()}
    base = {k: 1.0 for k in ranges} | {"inference": b["inference_default"]}   # point-estimate values
    s = {k: _draw(rng, lo, base[k], hi, n, dist) for k, (lo, hi) in ranges.items()}
    tot = _totals(comp, s)

    def ci(v):
        return {"mean": round(float(np.mean(v)), 0), "p05": round(float(np.percentile(v, 5)), 0),
                "p95": round(float(np.percentile(v, 95)), 0)}

    oat = {}
    for k, (lo, hi) in ranges.items():
        oat[k] = round(float(abs(_totals(comp, base | {k: hi})["scarcity_inference"]
                           - _totals(comp, base | {k: lo})["scarcity_inference"])) / 1e6, 1)
    sobol = {k: round(_first_order(s[k], tot["scarcity_inference"]), 3) for k in ranges}
    point = _totals(comp, base)
    return {
        "n_samples": n, "distribution": dist, "measured_energy_and_water": measured,
        "point_scarcity_m3eq_yr": round(float(point["scarcity"]), 0),
        "facility_types": {t: int(account.loc[account["facility_type"] == t, "facility_id"].nunique()) for t in comp.index},
        "carbon_tco2_yr": ci(tot["carbon"]),
        "carbon_inference_tco2_yr": ci(tot["carbon_inference"]),
        "water_phys_m3_yr": ci(tot["phys"]),
        "scarcity_m3eq_yr": ci(tot["scarcity"]),
        "scarcity_inference_m3eq_yr": ci(tot["scarcity_inference"]),
        "scarcity_mean_over_point": round(float(np.mean(tot["scarcity"]) / point["scarcity"]), 3),
        "scarcity_at_hydro_0_m3eq_yr": round(float(_totals(comp, base | {"hydro": 0.0})["scarcity"]), 0),
        "scarcity_sobol_first_order": dict(sorted(sobol.items(), key=lambda kv: -kv[1])),
        "scarcity_oat_swing_Mm3eq": dict(sorted(oat.items(), key=lambda kv: -kv[1])),
    }


if __name__ == "__main__":
    acct = pd.read_parquet(_repo_root() / "dcfootprint" / "outputs" / "account_facility_month.parquet")
    for dist in ("triangular", "uniform"):
        print(f"--- {dist}")
        for k, v in monte_carlo(acct, distribution=dist).items():
            print(k, "=", v)
