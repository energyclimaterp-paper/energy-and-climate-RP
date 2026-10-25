"""Data contracts (pandera) — the anti-granularity-mistake core.

The GAT project failed by forcing region-level data to a facility-level task.
Here that error becomes a *test failure*: every dataframe crossing a layer
boundary must pass a schema that pins its granularity, units, keys and ranges.

Rule (from DATA.md §0): the atomic unit is the **facility-month**; each resource
keeps its native partition; the facility is the join key; you aggregate UP to a
resource's native grain, never down-force one partition onto another. These
schemas make that rule executable.

Usage:
    from dcfootprint.validation import schemas
    facilities = schemas.Facilities.validate(df)          # raises on violation
    account    = schemas.FacilityMonthAccount.validate(df)
"""
from __future__ import annotations
import pandera.pandas as pa
from pandera.typing import Series
import pandas as pd

VALID_REGIONS = ("India", "US", "EU")


class Facilities(pa.DataFrameModel):
    """L0/L1 — one row per facility (the spine)."""
    facility_id: Series[str] = pa.Field(unique=True)
    operator: Series[str] = pa.Field(nullable=True)
    region: Series[str] = pa.Field(isin=VALID_REGIONS)
    state: Series[str] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(ge=-90, le=90, nullable=True)   # nullable until geocoded
    longitude: Series[float] = pa.Field(ge=-180, le=180, nullable=True)
    capacity_mw: Series[float] = pa.Field(gt=0, le=5000, nullable=True)  # disclosed MW; large values = verified announced campus/park totals (e.g. Lodha Palava 2500), not single operational buildings
    zone_id: Series[str] = pa.Field(nullable=True)   # carbon join key (ISO|state or ISO3)
    basin_id: Series["Int64"] = pa.Field(nullable=True)  # AWARE Basin_ID (post point-in-polygon)

    class Config:
        strict = False
        coerce = True


class EmberZoneMonth(pa.DataFrameModel):
    """L0 carbon panel — zone x month of the account year (io/ember.zone_month_ci). NOT
    facility-level: the native partition that gets *applied* to facilities."""
    zone_id: Series[str]
    month: Series[int] = pa.Field(ge=1, le=12)
    ci_gco2_per_kwh: Series[float] = pa.Field(ge=0, le=1500)
    ci_source: Series[str] = pa.Field(isin=["ember_state", "ember_national_fill", "ember_national"])

    @pa.dataframe_check
    def unique_zone_month(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(subset=["zone_id", "month"]).any()

    class Config:
        coerce = True


class FuelShares(pa.DataFrameModel):
    """L0 generation shares — zone x month x fuel (io/ember.zone_month_fuel_shares)."""
    zone_id: Series[str]
    month: Series[int] = pa.Field(ge=1, le=12)
    fuel: Series[str]
    share: Series[float] = pa.Field(ge=0, le=1)

    @pa.dataframe_check
    def shares_sum_to_one(cls, df: pd.DataFrame) -> bool:
        import numpy as np
        return bool(np.allclose(df.groupby(["zone_id", "month"])["share"].sum(), 1.0, atol=1e-9))

    class Config:
        strict = False
        coerce = True


class GemPlants(pa.DataFrameModel):
    """L0/L1 — GEM operating plants with basin and state of record (io/gem + geo/generation_basins)."""
    plant_id: Series[str] = pa.Field(unique=True)
    type: Series[str]
    capacity_mw: Series[float] = pa.Field(gt=0)
    latitude: Series[float] = pa.Field(ge=-90, le=90)
    longitude: Series[float] = pa.Field(ge=-180, le=180)
    basin_id: Series["Int64"]
    state: Series[str] = pa.Field(nullable=True)          # NaN only for 'ambiguous'
    state_source: Series[str] = pa.Field(isin=["gadm_agrees", "gadm", "override", "ambiguous",
                                               "gem_label_outside_gadm", "country"])

    @pa.dataframe_check
    def only_ambiguous_has_no_state(cls, df: pd.DataFrame) -> bool:
        return bool((df["state"].isna() == (df["state_source"] == "ambiguous")).all())

    class Config:
        strict = False
        coerce = True


class GridWater(pa.DataFrameModel):
    """L1 — scope-2 water intensity per zone-month, physical and scarcity-weighted."""
    zone_id: Series[str]
    month: Series[int] = pa.Field(ge=1, le=12)
    ewif_l_per_mwh: Series[float] = pa.Field(ge=0)
    sewif_l_eq_per_mwh: Series[float] = pa.Field(ge=0)
    ewif_hydro_l_per_mwh: Series[float] = pa.Field(ge=0)
    sewif_hydro_l_eq_per_mwh: Series[float] = pa.Field(ge=0)

    @pa.dataframe_check
    def hydro_is_part_of_total(cls, df: pd.DataFrame) -> bool:
        return bool(((df["ewif_hydro_l_per_mwh"] <= df["ewif_l_per_mwh"] + 1e-9)
                     & (df["sewif_hydro_l_eq_per_mwh"] <= df["sewif_l_eq_per_mwh"] + 1e-6)).all())

    @pa.dataframe_check
    def unique_zone_month(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(subset=["zone_id", "month"]).any()

    class Config:
        coerce = True


class AwareRemaining(pa.DataFrameModel):
    """L0 — AWARE 2.0 water remaining per basin-month (negative = over-committed)."""
    basin_id: Series["Int64"]
    month: Series[int] = pa.Field(ge=1, le=12)
    amd_m3_per_m2: Series[float] = pa.Field(nullable=True)
    area_m2: Series[float] = pa.Field(gt=0, nullable=True)
    remaining_m3: Series[float] = pa.Field(nullable=True)

    @pa.dataframe_check
    def unique_basin_month(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(subset=["basin_id", "month"]).any()

    class Config:
        coerce = True


class BasinBudget(pa.DataFrameModel):
    """L3/L6 — datacenter water budget per basin-month (project/recharge.basin_budgets)."""
    basin_id: Series["Int64"]
    month: Series[int] = pa.Field(ge=1, le=12)
    budget_l: Series[float] = pa.Field(ge=0)

    @pa.dataframe_check
    def twelve_months_per_basin(cls, df: pd.DataFrame) -> bool:
        return bool((df.groupby("basin_id")["month"].nunique() == 12).all())

    class Config:
        strict = False
        coerce = True


class BasinMonthlyCF(pa.DataFrameModel):
    """AWARE native CFs — basin x 12 climatological months (NOT year-specific).
    Join to facilities is SPATIAL (point-in-polygon on the gpkg), by Basin_ID."""
    basin_id: Series["Int64"]
    month: Series[int] = pa.Field(ge=1, le=12)
    cf: Series[float] = pa.Field(ge=0, le=100)   # AWARE capped at 100

    @pa.dataframe_check
    def unique_basin_month(cls, df: pd.DataFrame) -> bool:
        """Long table: one row per (basin, month) — basin_id repeats across months."""
        return not df.duplicated(subset=["basin_id", "month"]).any()

    class Config:
        coerce = True


class FacilityMonthAccount(pa.DataFrameModel):
    """L2 OUTPUT — the atomic unit. One row per (facility, month).
    Physical water and scarcity-weighted water are SEPARATE columns (no frankenmetric).
    """
    facility_id: Series[str]
    date: Series[pa.DateTime]
    region: Series[str] = pa.Field(isin=VALID_REGIONS)
    e_it_mwh: Series[float] = pa.Field(ge=0)
    e_grid_mwh: Series[float] = pa.Field(ge=0)
    carbon_tco2: Series[float] = pa.Field(ge=0)
    water_onsite_l: Series[float] = pa.Field(ge=0)       # scope-1
    water_grid_l: Series[float] = pa.Field(ge=0)         # scope-2
    water_phys_l: Series[float] = pa.Field(ge=0)         # physical total (reported FIRST)
    water_scarcity_onsite_l_eq: Series[float] = pa.Field(ge=0)  # scope-1 x CF(facility basin)
    water_scarcity_grid_l_eq: Series[float] = pa.Field(ge=0)    # scope-2 x CF(generation basins)
    water_scarcity_l_eq: Series[float] = pa.Field(ge=0)  # AWARE-weighted (SEPARATE)
    inference_share: Series[float] = pa.Field(ge=0, le=1)

    @pa.dataframe_check
    def scarcity_is_sum_of_scopes(cls, df: pd.DataFrame) -> bool:
        import numpy as np
        return np.allclose(df["water_scarcity_l_eq"],
                           df["water_scarcity_onsite_l_eq"] + df["water_scarcity_grid_l_eq"], rtol=1e-6)

    @pa.dataframe_check
    def atomic_unit_is_facility_month(cls, df: pd.DataFrame) -> bool:
        """The load-bearing contract: exactly one row per facility-month."""
        return not df.duplicated(subset=["facility_id", "date"]).any()

    @pa.dataframe_check
    def water_phys_is_sum_of_scopes(cls, df: pd.DataFrame) -> bool:
        import numpy as np
        return np.allclose(df["water_phys_l"], df["water_onsite_l"] + df["water_grid_l"], rtol=1e-6)

    class Config:
        coerce = True


# --------------------------------------------------------------------------- #
# Decision-layer outputs (validated by pipeline.py before they are written)
# --------------------------------------------------------------------------- #
class LeverSavings(pa.DataFrameModel):
    lever: Series[str] = pa.Field(unique=True)
    type: Series[str] = pa.Field(isin=["reduction", "transparency"])
    water_phys_saved_m3_yr: Series[float] = pa.Field(ge=0)
    water_scarcity_saved_m3eq_yr: Series[float] = pa.Field(ge=0)
    carbon_saved_tco2_yr: Series[float] = pa.Field(ge=0)
    pct_of_scarcity_baseline: Series[float] = pa.Field(ge=0, le=100)

    class Config:
        strict = False


class RoutingComparison(pa.DataFrameModel):
    policy: Series[str] = pa.Field(isin=["static", "greedy", "lyapunov", "oracle", "lyapunov_pf"])
    legal: Series[bool]
    carbon_tco2: Series[float] = pa.Field(gt=0)
    scarcity_m3eq: Series[float] = pa.Field(gt=0)
    peak_basin_queue_m3: Series[float] = pa.Field(ge=0)
    unserved_mwh: Series[float] = pa.Field(ge=0)
    penalty_gap_pct_vs_oracle: Series[float]

    @pa.dataframe_check
    def oracle_bounds_lyapunov(cls, df: pd.DataFrame) -> bool:
        """The offline oracle is a lower bound on lyapunov's penalty (same queue peaks)."""
        return bool((df.loc[df["policy"] == "lyapunov", "penalty_gap_pct_vs_oracle"] >= -1e-6).all())

    class Config:
        strict = False


class Q2Scorecard(pa.DataFrameModel):
    facility_id: Series[str] = pa.Field(unique=True)
    scarcity_pctile: Series[float] = pa.Field(ge=0, le=100)
    carbon_pctile: Series[float] = pa.Field(ge=0, le=100)
    harm_flag: Series[bool]
    recommended_lever: Series[str]

    class Config:
        strict = False


class Q1Siting(pa.DataFrameModel):
    rank: Series[int] = pa.Field(ge=1, unique=True)
    state: Series[str]
    basin_id: Series[int]
    max_regret: Series[float] = pa.Field(ge=0)
    rank_p10: Series[float] = pa.Field(ge=1)
    rank_p90: Series[float] = pa.Field(ge=1)
    small_grid: Series[bool]

    @pa.dataframe_check
    def unique_cell(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(subset=["state", "basin_id"]).any()

    class Config:
        strict = False


class AqueductBasinScores(pa.DataFrameModel):
    """Aqueduct 4.0 water-stress score per AWARE basin and scenario (project/scarcity_future.py)."""
    basin_id: Series[int]
    scenario: Series[str] = pa.Field(isin=["baseline", "bau30", "opt30", "pes30", "bau50", "opt50", "pes50"])
    ws_score: Series[float] = pa.Field(ge=0, le=5, nullable=True)          # NaN = no Aqueduct score over the basin
    scored_share: Series[float] = pa.Field(ge=0, le=1.01)                   # share of basin area with a score

    @pa.dataframe_check
    def unique_basin_scenario(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(subset=["basin_id", "scenario"]).any()

    class Config:
        strict = False


class ForecastCI(pa.DataFrameModel):
    date: Series[pa.DateTime]
    ci_gco2_per_kwh: Series[float] = pa.Field(ge=0, le=1500)
    lo: Series[float]
    hi: Series[float]

    @pa.dataframe_check
    def band_brackets_mean(cls, df: pd.DataFrame) -> bool:
        return bool(((df["lo"] <= df["ci_gco2_per_kwh"] + 1e-9) & (df["ci_gco2_per_kwh"] <= df["hi"] + 1e-9)).all())

    class Config:
        strict = False
        coerce = True
