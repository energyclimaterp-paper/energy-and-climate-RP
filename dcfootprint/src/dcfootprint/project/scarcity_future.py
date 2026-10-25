"""L3 — future water stress for Q1 siting: WRI Aqueduct 4.0 scores, baseline and 2030/2050.

Source: Aqueduct 4.0 (Kuzma et al. 2023, technical note; GDB Y2023M07D05).
  baseline   baseline_annual.bws_score        (constant within a pfaf_id: checked, 0 of 15,833 differ)
  future     future_annual.<sc><yy>_ws_x_s    sc in {bau, opt, pes}, yy in {30, 50}
Why SCORES, not raw ratios (decided from the technical note): raw water stress is capped to [0, 1]
(p. 11), so a future/baseline ratio understates change where stress is already at the cap; future
projections are bias-corrected to the baseline (p. 10), so the harmonised 0-5 scores are the
comparable measure. "Arid and low water use" scores 5 in both layers (p. 14; verified in the data).
2080 is not used (beyond any siting horizon). Carbon has no scenario (no sourced CI path).

Aqueduct pfaf (HydroBASINS L6) polygons do not coincide with AWARE basins, so each AWARE basin gets
the AREA-WEIGHTED mean score of the pfaf polygons it overlaps (equal-area EPSG:6933), per scenario,
over the part of the basin that has a score; `scored_share` records that part.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

SCENARIOS = ("baseline", "bau30", "opt30", "pes30", "bau50", "opt50", "pes50")
EQUAL_AREA = "EPSG:6933"


def _gdb() -> Path:
    return _repo_root() / _cfg_params()["siting"]["aqueduct_gdb_path"]


def pfaf_scores() -> "gpd.GeoDataFrame":
    """[pfaf_id, ws_<scenario> for SCENARIOS, geometry]; no-data (-9999 / NaN) -> NaN."""
    import pyogrio
    p = _gdb()
    fut_cols = [f"{s}_ws_x_s" for s in SCENARIOS if s != "baseline"]
    fut = pyogrio.read_dataframe(p, layer="future_annual", columns=["pfaf_id", *fut_cols])
    base = pyogrio.read_dataframe(p, layer="baseline_annual", columns=["pfaf_id", "bws_score"], read_geometry=False)
    base = base[base["bws_score"] > -9000].groupby("pfaf_id", as_index=False)["bws_score"].first()
    g = fut.merge(base, on="pfaf_id", how="left").rename(columns={"bws_score": "ws_baseline"})
    g = g.rename(columns={f"{s}_ws_x_s": f"ws_{s}" for s in SCENARIOS if s != "baseline"})
    for s in SCENARIOS:
        g[f"ws_{s}"] = g[f"ws_{s}"].where(g[f"ws_{s}"].between(0, 5))
    return g[["pfaf_id", *[f"ws_{s}" for s in SCENARIOS], "geometry"]]


def basin_scores(basin_ids) -> pd.DataFrame:
    """[basin_id, scenario, ws_score, scored_share] for the given AWARE basin ids (long format).
    Cached next to the GDB; basins missing from the cache are computed and appended."""
    import geopandas as gpd
    from dcfootprint.geo.join import load_aware_basins
    ids = sorted({int(b) for b in basin_ids})
    cache = _gdb().parent / "aware_basin_ws_scores.parquet"
    have = pd.read_parquet(cache) if cache.exists() and cache.stat().st_mtime >= _gdb().stat().st_mtime else None
    todo = ids if have is None else sorted(set(ids) - set(have["basin_id"]))
    if todo:
        aw = load_aware_basins()
        aw = aw[aw["basin_id"].isin(todo)][["basin_id", "geometry"]]
        aw = gpd.GeoDataFrame(aw, geometry="geometry", crs="EPSG:4326")
        aq = pfaf_scores()
        aq = aq[aq.intersects(aw.union_all().envelope)]
        aw, aq = aw.to_crs(EQUAL_AREA), aq.to_crs(EQUAL_AREA)
        aw["geometry"], aq["geometry"] = aw.make_valid(), aq.make_valid()
        ov = gpd.overlay(aw, aq, how="intersection", keep_geom_type=True)
        ov["a"] = ov.area
        basin_area = aw.set_index("basin_id").area
        rows = []
        for s in SCENARIOS:
            c = f"ws_{s}"
            v = ov.dropna(subset=[c])
            agg = v.groupby("basin_id").apply(lambda d: np.average(d[c], weights=d["a"]), include_groups=False)
            agg = agg.clip(0.0, 5.0)              # float round-off: a mean of 5s can come out as 5.000000000000001
            share = (v.groupby("basin_id")["a"].sum() / basin_area).fillna(0.0)   # no scored overlap -> 0
            for b in todo:
                rows.append({"basin_id": b, "scenario": s, "ws_score": float(agg.get(b, np.nan)),
                             "scored_share": float(share.get(b, 0.0))})
        new = pd.DataFrame(rows)
        have = new if have is None else pd.concat([have, new], ignore_index=True)
        from dcfootprint.validation import schemas
        have = schemas.AqueductBasinScores.validate(have)
        have.to_parquet(cache)
    return have[have["basin_id"].isin(ids)].reset_index(drop=True)
