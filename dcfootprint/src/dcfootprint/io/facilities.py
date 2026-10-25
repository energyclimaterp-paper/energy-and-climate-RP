"""L0/L1 — the facility spine (the make-or-break layer).

Loads the hand-curated India datacenter master, normalises it to the
``schemas.Facilities`` contract, and attaches ATLAS **city-centroid** coordinates
as a first, honest geocode (precise siting is a later, per-facility step in
``geo/geocode.py``). One row per facility; the facility is the join key for the
whole account.

Design (same pattern as the account/ modules):
  * pure-ish: reads declared inputs, returns a DataFrame; the DAG writes it.
  * config-driven paths, resolved from the repo root (runnable standalone too).
  * output validated against schemas.Facilities before it leaves this module.

Correction vs the scaffolding note ("merge our-list + ATLAS coords + CEA plant"):
CEA V22 is a *power-generation* database (its `Data` sheet is generating plants),
not a datacenter inventory — it is the carbon anchor used in L2, NOT a source of
DC capacity. So the facility layer = our master (normalised) + ATLAS coords only.
Uncosted operational rows are filled later from JLL/trade-press, not from CEA.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- India city aliases: our master's spelling -> the spelling ATLAS tends to use.
# Kept small and explicit; extend as match-rate analysis warrants.
_CITY_ALIASES = {
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
    "navi mumbai": "navi mumbai",
    "greater noida": "noida",
}


def _repo_root() -> Path:
    """Locate the repository root (the dir containing both `dcfootprint/` and `datasets/`),
    walking up from this file. Falls back to the fixed package depth."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "dcfootprint").is_dir() and (parent / "datasets").is_dir():
            return parent
    return here.parents[4]  # dcfootprint/src/dcfootprint/io/facilities.py -> repo root


def _norm_city(s: pd.Series) -> pd.Series:
    """Lowercase, strip punctuation, collapse whitespace, apply India aliases."""
    out = (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out.replace(_CITY_ALIASES)


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s or "facility"


def _make_facility_ids(names: pd.Series, prefix: str = "in") -> pd.Series:
    """Deterministic, unique, human-readable ids: `in-<name-slug>` with a numeric
    suffix only where slugs collide (stable under row order)."""
    ids, seen = [], {}
    for nm in names:
        base = f"{prefix}-{_slugify(nm)}"
        if base in seen:
            seen[base] += 1
            ids.append(f"{base}-{seen[base]}")
        else:
            seen[base] = 1
            ids.append(base)
    return pd.Series(ids, index=names.index)


# --------------------------------------------------------------------------- #
# Load + normalise
# --------------------------------------------------------------------------- #
def load_india_master(path: str | Path | None = None) -> pd.DataFrame:
    """Raw read of the hand-curated India facility master (194 rows)."""
    path = Path(path) if path else _repo_root() / "datasets" / "india_dc_facilities" / "India_DC_Facilities_v0.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"India facility master not found: {path}")
    return pd.read_excel(path)


def normalize_india(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the master's columns onto the Facilities contract (+ useful extras).
    Coordinates, zone_id and basin_id are left null here — filled downstream
    (ATLAS centroids below; precise geocode + point-in-polygon in geo/)."""
    df = pd.DataFrame(index=raw.index)
    df["facility_id"] = _make_facility_ids(raw["facility_name"])
    df["operator"] = raw["operator"].fillna(raw.get("operator_family"))
    df["region"] = "India"
    df["state"] = raw["state"]
    df["latitude"] = np.nan
    df["longitude"] = np.nan

    cap = pd.to_numeric(raw["capacity_mw"], errors="coerce")
    cap = cap.where(cap > 0)  # schema requires >0; blanks/zeros -> NaN (nullable)
    df["capacity_mw"] = cap

    df["zone_id"] = pd.NA            # assigned in geo/join.py (GADM state)
    df["basin_id"] = pd.array([pd.NA] * len(raw), dtype="Int64")  # AWARE point-in-polygon

    # --- extras carried for L1/L2/coverage/calibration (schema is strict=False) ---
    df["city"] = raw["city"]
    df["cea_grid_region"] = raw["cea_grid_region"]
    df["status"] = raw["status"]
    df["capacity_basis"] = raw.get("capacity_basis")
    df["op_capacity_mw"] = pd.to_numeric(raw.get("op_capacity_mw"), errors="coerce")
    df["confidence"] = raw.get("confidence")
    df["area_sqft"] = pd.to_numeric(raw.get("area_sqft"), errors="coerce")
    df["operator_family"] = raw.get("operator_family")
    df["source_url"] = raw.get("source_url")   # provenance for the open-dataset contribution
    df["geocode_method"] = pd.NA
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# ATLAS city-centroid geocode (first pass)
# --------------------------------------------------------------------------- #
def atlas_city_centroids(atlas_path: str | Path | None = None) -> pd.DataFrame:
    """India city -> (lat, lon) centroid from ATLAS, with the row count behind it.
    Returns [city_norm, latitude, longitude, atlas_n]."""
    atlas_path = Path(atlas_path) if atlas_path else _repo_root() / "data" / "atlas" / "datacenters.parquet"
    atlas = pd.read_parquet(atlas_path)
    ind = atlas[atlas["country"].astype(str).str.strip().str.lower() == "india"].copy()
    ind = ind.dropna(subset=["latitude", "longitude"])
    ind["city_norm"] = _norm_city(ind["city"])
    ind = ind[ind["city_norm"] != ""]
    return (
        ind.groupby("city_norm", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), atlas_n=("name", "size"))
    )


def attach_atlas_city_centroids(fac: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Fill lat/lon from the ATLAS city centroid where the facility's city matches.
    Marks geocode_method='atlas_city_centroid'. Non-matches stay null for geo/."""
    df = fac.copy()
    df["city_norm"] = _norm_city(df["city"])
    m = df.merge(centroids, on="city_norm", how="left", suffixes=("", "_atlas"))
    hit = m["latitude_atlas"].notna()
    df.loc[hit, "latitude"] = m.loc[hit, "latitude_atlas"].values
    df.loc[hit, "longitude"] = m.loc[hit, "longitude_atlas"].values
    df.loc[hit, "geocode_method"] = "atlas_city_centroid"
    return df.drop(columns=["city_norm"])


# --------------------------------------------------------------------------- #
# Orchestration + validation
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the Facilities contract. Loud warning (not silent skip) if pandera
    is absent, so a partial env still runs but the gap is visible."""
    try:
        from dcfootprint.validation import schemas
    except Exception as e:  # pragma: no cover - env-dependent
        print(f"[facilities] WARNING: schema validation skipped ({e!r}); install pandera to enforce.")
        return df
    return schemas.Facilities.validate(df)


def build(config: dict | None = None) -> pd.DataFrame:
    """Build the validated facility spine. `config` is accepted for DAG parity;
    paths currently resolve from the repo root."""
    raw = load_india_master()
    fac = normalize_india(raw)
    try:
        fac = attach_atlas_city_centroids(fac, atlas_city_centroids())
    except Exception as e:  # ATLAS coords are a gap-filler, not a hard dependency
        print(f"[facilities] WARNING: ATLAS geocode skipped ({e!r}); coordinates left null.")
    return validate(fac)


if __name__ == "__main__":  # smoke test on real data
    fac = build()
    op = fac[fac["status"] == "Operational"]
    print(f"rows: {len(fac)}  | region: {fac['region'].unique().tolist()}  | states: {fac['state'].nunique()}")
    print("status:", fac["status"].value_counts(dropna=False).to_dict())
    print(f"capacity_mw   non-null: {fac['capacity_mw'].notna().sum():>3}/{len(fac)}  "
          f"sum={fac['capacity_mw'].sum():.1f} MW  (operational sum={op['capacity_mw'].sum():.1f} MW)")
    print(f"op_capacity_mw non-null: {fac['op_capacity_mw'].notna().sum():>3}/{len(fac)}  "
          f"sum={fac['op_capacity_mw'].sum():.1f} MW")
    geo = fac["latitude"].notna()
    print(f"geocoded (ATLAS city-centroid): {geo.sum()}/{len(fac)}  "
          f"({geo.mean()*100:.0f}%); unique ids: {fac['facility_id'].is_unique}")
    print("\nsample:\n", fac.loc[:4, ["facility_id", "operator", "city", "state", "capacity_mw", "latitude", "longitude"]].to_string(index=False))

    out = _repo_root() / "dcfootprint" / "outputs" / "interim" / "facilities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    fac.to_parquet(out)
    print(f"\nwrote {out}")
