"""Validated configuration: config/parameters.yaml checked by pydantic before any module uses it.

`params()` parses the YAML, validates it against the models below (types, ranges, every
default inside its band, paths that must exist, cross-references between sections) and
returns the ORIGINAL dict, so call sites keep their dict access. A bad config fails loudly
at load time instead of producing numbers. Documentation-only sections (scarcity, outputs,
regions) and free-text keys (source, note, ...) are allowed through unvalidated.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dcfootprint.io.facilities import _repo_root

EMBER_FUELS = {"Coal", "Gas", "Nuclear", "Bioenergy", "Hydro", "Solar", "Wind", "Other Fossil", "Other Renewables"}


class _M(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Band(_M):
    default: float
    band: tuple[float, float]

    @model_validator(mode="after")
    def _inside(self):
        lo, hi = self.band
        if not (lo <= self.default <= hi and lo < hi):
            raise ValueError(f"default {self.default} not inside band {self.band}")
        return self


class ByType(_M):
    by_facility_type: dict[str, Band]
    by_operator: dict[str, float] = {}

    @field_validator("by_facility_type")
    @classmethod
    def _has_unknown(cls, v):
        if "unknown" not in v:
            raise ValueError("by_facility_type needs an 'unknown' fallback")
        return v


class Energy(_M):
    utilisation: ByType
    pue: ByType

    @model_validator(mode="after")
    def _ranges(self):
        for t, b in self.utilisation.by_facility_type.items():
            if not (0 < b.band[0] and b.band[1] <= 1):
                raise ValueError(f"utilisation band for {t} must be within (0, 1]")
        for t, b in self.pue.by_facility_type.items():
            if b.band[0] < 1:
                raise ValueError(f"PUE band for {t} must be >= 1")
        if any(v < 1 for v in self.pue.by_operator.values()):
            raise ValueError("operator PUE must be >= 1")
        return self


class WaterOnsite(_M):
    wue_default: Band
    wue_by_operator: dict[str, float] = {}

    @model_validator(mode="after")
    def _nonneg(self):
        if self.wue_default.band[0] < 0 or any(v < 0 for v in self.wue_by_operator.values()):
            raise ValueError("WUE must be >= 0")
        return self


class WaterGrid(_M):
    ewif_coeff_L_per_MWh: dict[str, float]
    hydro_multiplier_band: tuple[float, float]

    @field_validator("ewif_coeff_L_per_MWh")
    @classmethod
    def _all_fuels(cls, v):
        missing = EMBER_FUELS - set(v)
        if missing or any(c < 0 for c in v.values()):
            raise ValueError(f"EWIF needs every Ember fuel, >= 0 (missing {sorted(missing)})")
        return v


class Override(_M):
    state: str
    note: str = ""


class Grid(_M):
    zone: Literal["state", "national"]
    carbon_basis: Literal["generation", "consumption"] = "generation"
    account_year: int = Field(ge=2019, le=2025)
    gem_path: str
    gadm_path: str
    gem_type_to_fuel: dict[str, str]
    gem_state_overrides: dict[str, Override] = {}
    gem_state_ambiguous: dict[str, str] = {}


class Inference(_M):
    sectoral_share: Band


class Levers(_M):
    coastal_seawater_siting: dict
    zero_liquid_discharge: dict
    efficiency_standard: dict

    @model_validator(mode="after")
    def _vals(self):
        r = float(self.zero_liquid_discharge["recycle_fraction"])
        if not 0 <= r <= 1 or self.zero_liquid_discharge.get("applies_to") != "scope1":
            raise ValueError("ZLD recycle_fraction in [0,1] and applies_to: scope1")
        if float(self.efficiency_standard["pue_cap"]) < 1 or float(self.efficiency_standard["wue_cap"]) < 0:
            raise ValueError("efficiency caps: pue_cap >= 1, wue_cap >= 0")
        if float(self.coastal_seawater_siting["new_wue"]) < 0:
            raise ValueError("coastal new_wue >= 0")
        return self


class Routing(_M):
    solver: Literal["price_decomposition", "central_lp"]
    flexible_share: float = Field(gt=0, le=1)
    max_util: float = Field(gt=0, le=1)
    V: float = Field(gt=0)
    v_sweep: list[float]
    lambda_: float = Field(alias="lambda", ge=0)
    demand_noise: float = Field(ge=0)
    seeds: int = Field(ge=1)
    budget_alpha: float = Field(gt=0)
    budget_sweep_alpha: list[float]
    aware_intermediate_path: str


class Siting(_M):
    new_facility_mw: float = Field(gt=0)
    new_facility_type: str
    weight_samples: int = Field(ge=10)
    small_grid_twh: float = Field(ge=0)
    aqueduct_gdb_path: str = "data/aqueduct/Aq40_Y2023D07M05.gdb"


class Uncertainty(_M):
    distribution: Literal["triangular", "uniform"]
    n_samples: int = Field(ge=100)


class Parameters(_M):
    energy: Energy
    water_onsite: WaterOnsite
    water_grid: WaterGrid
    grid: Grid
    inference: Inference
    levers: Levers
    routing: Routing
    siting: Siting
    uncertainty: Uncertainty

    @model_validator(mode="after")
    def _cross(self):
        root = _repo_root()
        for p in [self.grid.gem_path, self.grid.gadm_path, self.routing.aware_intermediate_path]:
            if not (root / p).exists():
                raise ValueError(f"configured path does not exist: {p}")
        if self.siting.new_facility_type not in self.energy.utilisation.by_facility_type:
            raise ValueError("siting.new_facility_type must be an energy facility type")
        bad = set(self.grid.gem_type_to_fuel.values()) - set(self.water_grid.ewif_coeff_L_per_MWh)
        if bad:
            raise ValueError(f"gem_type_to_fuel maps to fuels without EWIF: {sorted(bad)}")
        if self.inference.sectoral_share.band[1] > 1:
            raise ValueError("inference share band must be within [0, 1]")
        return self


def config_path() -> Path:
    return _repo_root() / "dcfootprint" / "config" / "parameters.yaml"


@lru_cache(maxsize=1)
def _load(path: str, mtime: float) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    Parameters.model_validate(raw)
    return raw


def params() -> dict:
    """The validated parameters.yaml as a dict (re-read and re-validated if the file changes)."""
    p = config_path()
    return _load(str(p), p.stat().st_mtime)
