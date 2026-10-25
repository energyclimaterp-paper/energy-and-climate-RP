"""L0 — Ember monthly (India / US / EU, load_raw). Two products for the account:
  * national monthly FUEL SHARES -> scope-2 EWIF (via Macknick coeffs).
  * national monthly CO2 INTENSITY (generation-based) -> the *flagged sensitivity*
    carbon source (the primary India carbon is CEA national-annual; see io/cea.py).

Why national, not per-state: India is a synchronous national grid — a datacentre
consumes the pooled national mix, not its own state's generation (config R2). Shares
are ratio-based, so aggregating across whatever rows Ember provides is scale-safe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dcfootprint.io.facilities import _repo_root
from dcfootprint.settings import params as _cfg_params

# Ember atomic-fuel Variable names that map 1:1 to parameters.yaml ewif_coeff keys.
_FUELS = ["Coal", "Gas", "Nuclear", "Bioenergy", "Hydro", "Solar", "Wind",
          "Other Fossil", "Other Renewables"]


EMBER_FILES = {"India": "india_monthly_full_release_long_format.csv",
               "US": "us_monthly_full_release_long_format.csv",
               "EU": "europe_monthly_full_release_long_format.csv"}


def load_raw(region: str = "India", path: str | Path | None = None) -> pd.DataFrame:
    """Ember monthly release for a region, normalised to ONE long format and unit set:
    columns State, date, Category, Variable, Unit, Value; generation in GWh, emissions in
    ktCO2, intensity labelled gCO2/kWh; the aggregate row named '<X> Total'
      India  as released (states + 'India Total'; 'Others' is not a zone)
      US     as released (states incl. 'Washington, D.C.' + 'US Total')
      EU     country-level release filtered to the EU-27 members; the Ember 'EU' region row
             becomes 'EU Total'. Ember Europe reports TWh / MtCO2e / gCO2e per kWh: converted
             (x1000) and relabelled — note the EU emissions are CO2-EQUIVALENT (flagged)."""
    path = Path(path) if path else _repo_root() / "data" / "ember" / EMBER_FILES[region]
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"])
    if region == "EU":
        member = df["EU"].astype(str).str.lower().isin(["true", "1", "1.0"])
        countries = df["Area type"].eq("Country or economy") & member
        agg = df["Area"].eq("EU") & df["Area type"].eq("Region")
        df = df[countries | agg].copy()
        df["State"] = df["Area"].where(~agg, "EU Total")
        conv = {"TWh": ("GWh", 1000.0), "MtCO2e": ("ktCO2", 1000.0), "gCO2e per kWh": ("gCO2/kWh", 1.0)}
        for u, (new, k) in conv.items():
            m = df["Unit"].eq(u)
            df.loc[m, "Value"] = df.loc[m, "Value"] * k
            df.loc[m, "Unit"] = new
        # Ember Europe splits fuels finer than India/US: fold onto the 9-fuel set (lignite takes
        # the coal EWIF coefficient — flagged), and build 'Total emissions' as the fuel sum.
        fuel_map = {"Hard coal": "Coal", "Lignite": "Coal", "Onshore wind": "Wind", "Offshore wind": "Wind",
                    "Other fossil": "Other Fossil", "Other renewables": "Other Renewables"}
        df["Variable"] = df["Variable"].replace({"Total generation": "Total Generation"})
        fuel = df["Subcategory"].eq("Fuel") & df["Unit"].isin(["GWh", "ktCO2"])
        f = df[fuel].assign(Variable=lambda d: d["Variable"].replace(fuel_map))
        keys = ["State", "date", "Date", "Category", "Subcategory", "Variable", "Unit"]
        f = f.groupby(keys, as_index=False)["Value"].sum()
        tot_em = (f[f["Category"].eq("Power sector emissions")].groupby(["State", "date", "Date"], as_index=False)["Value"].sum()
                  .assign(Category="Power sector emissions", Subcategory="Total", Variable="Total emissions", Unit="ktCO2"))
        df = pd.concat([df[~fuel & ~df["Unit"].eq("%")], f, tot_em], ignore_index=True)
    return _drop_impossible_zero_ci(df)


def _drop_impossible_zero_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Drop CO2-intensity rows that read exactly 0 in a zone-month with positive FOSSIL generation:
    fossil generation always emits, so these are data errors, not clean months. Evidence (raw files):
    Ember Europe lists EU CO2 intensity 0.00 for Jan-Apr 2026 next to ~94 TWh/month of fossil
    generation (emissions not yet published), and Luxembourg 0.00 in 4 months (2 in 2023) with fossil
    generation. Dropped rows fall back to the documented national/EU fill in zone_month_ci(). Zero
    intensity with zero fossil generation (Goa, Lakshadweep, Luxembourg Oct 2025) is kept as genuine."""
    fos = df[df["Variable"].eq("Fossil") & df["Unit"].eq("GWh")]
    if fos.empty:
        fos = df[df["Variable"].isin(["Coal", "Gas", "Other Fossil"]) & df["Unit"].eq("GWh")]
    fos = fos.groupby(["State", "date"])["Value"].sum()
    zero_ci = df["Variable"].eq("CO2 intensity") & df["Value"].eq(0)
    if not zero_ci.any():
        return df
    key = pd.MultiIndex.from_frame(df.loc[zero_ci, ["State", "date"]])
    bad = zero_ci.copy()
    bad[zero_ci] = fos.reindex(key).fillna(0).to_numpy() > 0
    return df[~bad].reset_index(drop=True)


def load_india_raw(path: str | Path | None = None) -> pd.DataFrame:
    return load_raw("India", path)


def national_of(raw: pd.DataFrame) -> str:
    """The aggregate row's name ('India Total', 'US Total', 'EU Total')."""
    tot = [x for x in raw["State"].dropna().unique() if str(x).endswith(" Total")]
    if len(tot) != 1:
        raise ValueError(f"expected one '<X> Total' row, found {tot}")
    return tot[0]


def zones_of(raw: pd.DataFrame) -> list[str]:
    nat = national_of(raw)
    return sorted(z for z in raw["State"].dropna().unique() if z not in {nat, "Others"})


def national_monthly_fuel_shares(raw: pd.DataFrame) -> pd.DataFrame:
    """[date, fuel, share] — national generation share per fuel per month.
    Shares are ratios, so summing across state (and any national) rows is safe."""
    gen = raw[(raw["Category"] == "Electricity generation")
              & (raw["Variable"].isin(_FUELS))
              & (raw["Unit"] == "GWh")].copy()
    national = gen[gen["State"] == national_of(raw)]        # exact national row
    if national.empty:                                       # fallback: sum states (skip 'total' to avoid double-count)
        national = gen[gen["State type"].isin(["state", "Union territory"])]
    nat = (national.groupby(["date", "Variable"], as_index=False)["Value"].sum()
              .rename(columns={"Variable": "fuel", "Value": "gwh"}))
    nat["gwh"] = nat["gwh"].clip(lower=0)
    tot = nat.groupby("date")["gwh"].transform("sum")
    nat["share"] = (nat["gwh"] / tot).fillna(0.0)
    return nat[["date", "fuel", "share"]]


def national_monthly_ci(raw: pd.DataFrame) -> pd.DataFrame:
    """[date, ci_gco2_per_kwh] — generation-based national CI (flagged sensitivity)."""
    ci = raw[(raw["Variable"] == "CO2 intensity") & (raw["Unit"] == "gCO2/kWh")].copy()
    national = ci[ci["State"] == national_of(raw)]
    if national.empty:
        national = ci
    return (national.groupby("date", as_index=False)["Value"].mean()
              .rename(columns={"Value": "ci_gco2_per_kwh"}))


def monthly_ewif_l_per_mwh(shares: pd.DataFrame, ewif_coeff: dict[str, float]) -> pd.DataFrame:
    """[month(1-12), ewif_l_per_mwh] — climatological monthly scope-2 water intensity,
    averaged across available years (the account is a representative-year panel)."""
    s = shares.copy()
    s["coeff"] = s["fuel"].map(ewif_coeff)
    missing = sorted(s.loc[s["coeff"].isna(), "fuel"].unique())
    if missing:
        raise KeyError(f"Fuels with no EWIF coefficient (add to parameters.yaml): {missing}")
    s["contrib"] = s["share"] * s["coeff"]
    per_date = s.groupby("date", as_index=False)["contrib"].sum().rename(columns={"contrib": "ewif"})
    per_date["month"] = per_date["date"].dt.month
    return (per_date.groupby("month", as_index=False)["ewif"].mean()
                    .rename(columns={"ewif": "ewif_l_per_mwh"}))


# --------------------------------------------------------------------------- #
# Zone-month tables (zone = Ember state/country, or the aggregate for every zone)
# --------------------------------------------------------------------------- #


def zone_month_ci(raw: pd.DataFrame, year: int, zone: str = "state") -> pd.DataFrame:
    """[zone_id, month, ci_gco2_per_kwh, ci_source] for `year`.
    zone='state': Ember state generation CI (generation-based; R2 caveat), with the national
    value filling any state-month Ember leaves blank. zone='national': India Total for all."""
    ci = raw[(raw["Variable"] == "CO2 intensity") & (raw["Unit"] == "gCO2/kWh")
             & (raw["date"].dt.year == year)].copy()
    ci["month"] = ci["date"].dt.month
    nat = (ci[ci["State"] == national_of(raw)].groupby("month")["Value"].mean()
           .rename("ci_national"))
    states = zones_of(raw)
    grid = pd.MultiIndex.from_product([states, range(1, 13)], names=["zone_id", "month"]).to_frame(index=False)
    grid = grid.merge(nat, on="month", how="left")
    if zone == "state":
        st = (ci[ci["State"].isin(states)].groupby(["State", "month"])["Value"].mean()
              .rename("ci_state").reset_index().rename(columns={"State": "zone_id"}))
        grid = grid.merge(st, on=["zone_id", "month"], how="left")
        grid["ci_gco2_per_kwh"] = grid["ci_state"].fillna(grid["ci_national"])
        grid["ci_source"] = grid["ci_state"].notna().map({True: "ember_state", False: "ember_national_fill"})
    elif zone == "national":
        grid["ci_gco2_per_kwh"] = grid["ci_national"]
        grid["ci_source"] = "ember_national"
    else:
        raise ValueError(f"unknown grid zone {zone!r} (state | national)")
    from dcfootprint.validation import schemas
    return schemas.EmberZoneMonth.validate(grid[["zone_id", "month", "ci_gco2_per_kwh", "ci_source"]])


def zone_month_fuel_shares(raw: pd.DataFrame, year: int, zone: str = "state") -> pd.DataFrame:
    """[zone_id, month, fuel, share] generation shares for `year`; a state-month with no
    generation in Ember takes the national shares (flagged via share_source)."""
    gen = raw[(raw["Category"] == "Electricity generation") & (raw["Variable"].isin(_FUELS))
              & (raw["Unit"] == "GWh") & (raw["date"].dt.year == year)].copy()
    gen["month"] = gen["date"].dt.month
    gen["Value"] = gen["Value"].clip(lower=0)

    def _shares(df, key):
        g = df.groupby([key, "month", "Variable"], as_index=False)["Value"].sum()
        g["share"] = g["Value"] / g.groupby([key, "month"])["Value"].transform("sum")
        return g.rename(columns={key: "zone_id", "Variable": "fuel"})[["zone_id", "month", "fuel", "share"]]

    nat = _shares(gen[gen["State"] == national_of(raw)], "State").drop(columns="zone_id")
    states = zones_of(raw)
    if zone == "national":
        out = pd.concat([nat.assign(zone_id=s) for s in states], ignore_index=True)
        out["share_source"] = "ember_national"
        from dcfootprint.validation import schemas
        return schemas.FuelShares.validate(out)
    st = _shares(gen[gen["State"].isin(states)], "State").dropna(subset=["share"])
    have = set(map(tuple, st[["zone_id", "month"]].drop_duplicates().values))
    missing = [(s, m) for s in states for m in range(1, 13) if (s, m) not in have]
    fill = pd.concat([nat[nat["month"] == m].assign(zone_id=s) for s, m in missing]
                     or [nat.iloc[0:0].assign(zone_id="")], ignore_index=True)
    st["share_source"] = "ember_state"
    fill["share_source"] = "ember_national_fill"
    from dcfootprint.validation import schemas
    return schemas.FuelShares.validate(
        pd.concat([st, fill], ignore_index=True)[["zone_id", "month", "fuel", "share", "share_source"]])


if __name__ == "__main__":
    raw = load_india_raw()
    print("States:", sorted(map(str, raw["State"].dropna().unique()))[:30])
    print("State type:", raw["State type"].dropna().unique().tolist())
    shares = national_monthly_fuel_shares(raw)
    latest = shares[shares["date"] == shares["date"].max()].set_index("fuel")["share"].round(3)
    print(f"\nfuel shares @ {shares['date'].max().date()} (sum={latest.sum():.2f}):\n{latest.to_dict()}")
    params = _cfg_params()
    ewif = monthly_ewif_l_per_mwh(shares, params["water_grid"]["ewif_coeff_L_per_MWh"])
    print("\nmonthly EWIF (L/MWh):\n", ewif.to_string(index=False))
    ci = national_monthly_ci(raw)
    print(f"\nnational CI (gen-proxy) range: {ci['ci_gco2_per_kwh'].min():.0f}-{ci['ci_gco2_per_kwh'].max():.0f} gCO2/kWh")
