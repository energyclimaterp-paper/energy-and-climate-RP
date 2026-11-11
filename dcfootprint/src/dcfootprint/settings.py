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
    wue_model: Literal["constant", "climate"] = "constant"
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
        z = self.zero_liquid_discharge
        if "recycle_fraction" in z:
            raise ValueError("ZLD recycle_fraction is retired: ZLD recovers blowdown only; set "
                             "cycles_of_concentration and blowdown_recovery (parameters.yaml levers)")
        n, rec = float(z["cycles_of_concentration"]), float(z["blowdown_recovery"])
        if n <= 1 or not 0 <= rec <= 1 or z.get("applies_to") != "scope1":
            raise ValueError("ZLD: cycles_of_concentration > 1, blowdown_recovery in [0,1], applies_to: scope1")
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
    sobol_n_samples: int = Field(default=32768, ge=1000)
    routing_n_samples: int = Field(default=1000, ge=10)
    ci_zone_band: tuple[float, float] = (1.0, 1.0)

    @model_validator(mode="after")
    def _ci_band(self):
        lo, hi = self.ci_zone_band
        if not (0 < lo <= 1 <= hi):
            raise ValueError(f"ci_zone_band {self.ci_zone_band} must bracket 1 and be > 0")
        return self


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
        # Raw-data paths are checked where they are read (data_path), not here: a stage that does
        # not use GEM (e.g. routing on a frozen account, or the US on EIA plant data) must not
        # need the GEM workbook on disk.
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


def data_path(rel: str) -> Path:
    """Repo-relative raw-data path, checked at the point of use: a missing file fails the stage
    that needs it with a pointer to data/README.md."""
    path = _repo_root() / rel
    if not path.exists():
        raise FileNotFoundError(f"raw input not on disk: {rel} (download as in data/README.md)")
    return path


def write_run_manifest(outdir: Path | str, extra: dict | None = None) -> Path:
    """Write run_manifest.json into a results folder: the exact parameters.yaml used (verbatim text
    and its SHA-256), the git commit and whether the tree was dirty, and package versions. Without
    it a results folder cannot be tied to the configuration that produced it once defaults change."""
    import hashlib
    import json
    import platform
    import subprocess
    from importlib import metadata

    root = _repo_root()

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    cfg = config_path().read_text(encoding="utf-8")
    versions = {}
    for pkg in ("pandas", "numpy", "scipy", "statsmodels", "geopandas", "pandera", "pydantic"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    manifest = {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "python": platform.python_version(), "packages": versions,
                "parameters_sha256": hashlib.sha256(cfg.encode("utf-8")).hexdigest(),
                "parameters_yaml": cfg, **(extra or {})}
    out = Path(outdir) / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out
