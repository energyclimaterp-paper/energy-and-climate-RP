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
(uncertainty.distribution; uniform is available as a sensitivity). Sensitivity: first-order and
total Sobol indices of the total scarcity-weighted water from a Saltelli design with bootstrap
intervals (sobol_saltelli; the model is multiplicative, so interactions are reported through
ST - S1), plus the one-at-a-time swing.
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
        "sobol_n_samples": int(p.get("uncertainty", {}).get("sobol_n_samples", 32768)),
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
    """Given-data first-order index from quantile bins, bias-corrected.

    The raw between-bin share SSB/SST is biased upward by the sampling noise of the bin means,
    about (bins - 1) / N of the residual variance per input, so the raw indices of independent
    inputs could sum above 1 (round 5 India: 1.026). The ANOVA epsilon-squared correction removes it:
        S = (SSB - (k - 1) * SSW / (N - k)) / SST, clipped at 0.
    Used where only one sample exists (the routing Monte Carlo); the account Monte Carlo uses the
    Saltelli design below."""
    if y.var() <= 0:
        return 0.0
    q = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, bins - 1)
    groups = [y[idx == k] for k in range(bins) if (idx == k).any()]
    k, n = len(groups), len(y)
    ssb = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ((y - y.mean()) ** 2).sum()
    return float(max(ssb - (k - 1) * ssw / max(n - k, 1), 0.0) / sst)


def sobol_saltelli(f, draw, names: list[str], n: int, seed: int = 0, n_boot: int = 200) -> dict:
    """First-order and total Sobol indices from a Saltelli design (n * (k + 2) model runs).

    `draw(rng, n)` returns {name: array(n)} of independent input draws; `f(x)` maps such a dict to
    an output array. With A, B two independent samples and AB_i = A with column i taken from B:
        S_i  = mean(f(B) * (f(AB_i) - f(A))) / V        (Saltelli et al. 2010, eq. b)
        ST_i = mean((f(A) - f(AB_i))^2) / (2 V)          (Jansen 1999)
    V is the variance of f over A and B pooled. 90% intervals come from a bootstrap over the n rows.
    Saltelli, Annoni, Azzini, Campolongo, Ratto, Tarantola, Comput. Phys. Commun. 181 (2010) 259-270."""
    rng = np.random.default_rng(seed)
    A, B = draw(rng, n), draw(rng, n)
    fA, fB = np.asarray(f(A), float), np.asarray(f(B), float)
    fAB = {k: np.asarray(f(A | {k: B[k]}), float) for k in names}

    def est(ix):
        V = np.var(np.concatenate([fA[ix], fB[ix]]))
        if V <= 0:
            return {k: (0.0, 0.0) for k in names}
        return {k: (float(np.mean(fB[ix] * (fAB[k][ix] - fA[ix])) / V),
                    float(np.mean((fA[ix] - fAB[k][ix]) ** 2) / (2 * V))) for k in names}

    point = est(np.arange(n))
    boots = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    out = {}
    for k in names:
        s1 = np.array([b[k][0] for b in boots])
        st = np.array([b[k][1] for b in boots])
        out[k] = {"S1": round(point[k][0], 3), "S1_p05": round(float(np.percentile(s1, 5)), 3),
                  "S1_p95": round(float(np.percentile(s1, 95)), 3),
                  "ST": round(point[k][1], 3), "ST_p05": round(float(np.percentile(st, 5)), 3),
                  "ST_p95": round(float(np.percentile(st, 95)), 3)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["ST"]))


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
    # Sobol indices of the HEADLINE total (scarcity-weighted water, not the inference-attributed
    # share, which added the inference parameter's variance to the round-5 indices)
    varying = [k for k, (lo, hi) in ranges.items() if lo < hi and k != "inference"]
    sobol_full = sobol_saltelli(
        lambda x: _totals(comp, base | x)["scarcity"],
        lambda r, m: {k: _draw(r, ranges[k][0], base[k], ranges[k][1], m, dist) for k in varying},
        varying, b["sobol_n_samples"], seed=seed + 1) if varying else {}
    sobol = {k: v["S1"] for k, v in sobol_full.items()}
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
        "scarcity_sobol": {"estimator": "Saltelli 2010 (S1) / Jansen 1999 (ST), bootstrap 90% intervals",
                           "target": "scarcity_m3eq_yr", "indices": sobol_full,
                           "sum_S1": round(sum(sobol.values()), 3)},
        "scarcity_oat_swing_Mm3eq": dict(sorted(oat.items(), key=lambda kv: -kv[1])),
    }


if __name__ == "__main__":
    acct = pd.read_parquet(_repo_root() / "dcfootprint" / "outputs" / "account_facility_month.parquet")
    for dist in ("triangular", "uniform"):
        print(f"--- {dist}")
        for k, v in monte_carlo(acct, distribution=dist).items():
            print(k, "=", v)
