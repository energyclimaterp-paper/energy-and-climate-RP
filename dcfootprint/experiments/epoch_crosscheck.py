"""Cross-check the US account against Epoch AI's frontier AI datacenter list (REPORT ONLY, corrects nothing).

Epoch AI "Frontier data centers" (data/epoch/data_centers.csv): named AI campuses with current power (MW),
owner and address. Question: which of the large US AI campuses does the US account (Compute Atlas spine,
io/us_facilities.py) contain, and with what capacity? A campus that is missing is not counted in the
account; one present with a much smaller capacity is undercounted. Matching (fixed before running):
  same state (from the address's 2-letter code) AND same city (normalised)  -> candidate
  candidate whose operator family shares a word with the Epoch owner        -> "matched"
  candidate with another operator                                           -> "same city, other operator"
  no candidate                                                              -> "not in account"
Capacities: Epoch "Current power (MW)" is facility power; the account's capacity_mw is IT (after the
record-by-record basis classification), so the account's facility power is shown as capacity_mw x PUE.

    .venv/Scripts/python.exe dcfootprint/experiments/epoch_crosscheck.py
Writes dcfootprint/results/us/epoch_crosscheck.{csv,md}.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dcfootprint" / "src"))
from dcfootprint.io import us_facilities  # noqa: E402
from dcfootprint.settings import params  # noqa: E402

STATES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
          "CT": "Connecticut", "DE": "Delaware", "DC": "Washington, D.C.", "FL": "Florida", "GA": "Georgia",
          "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
          "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
          "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
          "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
          "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
          "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
          "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
          "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
NAME_TO_CODE = {v.lower(): k for k, v in STATES.items()}
_GENERIC = {"data", "center", "centers", "campus", "inc", "llc", "the", "and", "site", "nan", "infrastructure",
            "services", "technologies", "company", "group", "east", "west", "north", "south", "expansion"}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def parse_address(addr: str) -> tuple[str | None, str | None]:
    """(state name, city) from 'street, City, ST 12345' or a free-text address naming the state."""
    a = str(addr)
    for m in re.finditer(r"(?:^|,)\s*([A-Za-z .'-]+?),\s*([A-Z]{2})(?=\s*\d{5}|\s*,|\s*$)", a):
        if m.group(2) in STATES:                             # 'City, ST 12345' / 'street, City, ST' / no ZIP
            return STATES[m.group(2)], _norm(m.group(1))
    for name, code in sorted(NAME_TO_CODE.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", a.lower()):
            pre = a.lower().split(name)[0]
            city = _norm(pre.split(",")[-2]) if pre.count(",") >= 1 else None
            return STATES[code], city
    return None, None


def main():
    ep = pd.read_csv(REPO / "data" / "epoch" / "data_centers.csv")
    ep = ep[ep["Country"].astype(str).str.strip().eq("United States")].copy()
    ep["owner"] = ep["Owner"].fillna("").str.replace(r"#\w+", "", regex=True).str.strip()
    ep[["state", "city"]] = ep["Address"].apply(lambda a: pd.Series(parse_address(a)))
    us = us_facilities.build()
    us["city_n"] = us["city"].map(_norm)
    pue = params()["energy"]["pue"]["by_facility_type"]
    rows = []
    for _, e in ep.iterrows():
        cand = (us[(us["state"] == e["state"]) & (us["city_n"] == e["city"])]
                if isinstance(e["state"], str) and e["state"] else us.iloc[0:0])
        # operator evidence = owner + users + construction / energy companies + investors + project + name
        # (e.g. Stargate Abilene is built by Crusoe; Colossus is owned by xAI, listed as "SpaceXAI")
        text = " ".join(str(e.get(c, "")) for c in ("owner", "Users", "Construction companies", "Energy companies",
                                                   "Investors", "Project", "Name"))
        owner_words = {w for w in _norm(re.sub(r"#\w+", "", text)).split() if len(w) > 2 and w not in _GENERIC}
        owner_words |= {"xai"} if "spacexai" in owner_words else set()
        op = cand[cand.apply(lambda r: bool(owner_words & set(_norm(f"{r['operator']} {r.get('operator_family', '')}").split())), axis=1)] \
            if len(cand) and owner_words else cand.iloc[0:0]
        hit = op if len(op) else cand
        no_state = not isinstance(e["state"], str) or not e["state"]
        status = ("matched" if len(op) else "possible match (same city, operator not confirmed)" if len(cand)
                  else "no address in Epoch" if no_state else "not in account")
        fac_mw = sum(float(r["capacity_mw"]) * float(pue.get(r.get("facility_type_hint", "hyperscale"), pue["hyperscale"])["default"])
                     for _, r in hit.iterrows()) if len(hit) else 0.0
        rows.append({"epoch_name": e["Name"], "owner": e["owner"], "state": e["state"], "city": e["city"],
                     "epoch_power_mw": float(e["Current power (MW)"]) if pd.notna(e["Current power (MW)"]) else None,
                     "status": status, "account_facilities": "; ".join(hit["facility_id"].head(4)),
                     "account_operators": "; ".join(sorted(set(hit["operator"].astype(str)))[:3]),
                     "account_it_mw": float(hit["capacity_mw"].sum()) if len(hit) else 0.0,
                     "account_facility_mw_est": round(fac_mw, 1)})
    t = pd.DataFrame(rows).sort_values("epoch_power_mw", ascending=False)
    out = REPO / "dcfootprint" / "results" / "us"
    t.to_csv(out / "epoch_crosscheck.csv", index=False)

    tot = t["epoch_power_mw"].sum()
    by = t.groupby("status")["epoch_power_mw"].agg(["size", "sum"])
    m = t[t["status"] == "matched"]
    mp = m[m["epoch_power_mw"] > 0]                               # Epoch lists some campuses at 0 MW (not yet built)
    ratio = (mp["account_facility_mw_est"] / mp["epoch_power_mw"]).describe()
    big = t[t["epoch_power_mw"] >= 200]
    L = ["# US account vs Epoch AI frontier datacenters (report only)", "",
         "Generated by `dcfootprint/experiments/epoch_crosscheck.py` (method in its docstring). Epoch AI lists named",
         f"frontier AI campuses; **{len(t)} are in the US, {tot:,.0f} MW of current power.** The US account (Compute Atlas",
         "spine) has 235 facilities. Matching is by state + city, then operator.", "",
         "| status | campuses | Epoch MW | share of Epoch US MW |", "|---|---|---|---|"]
    for s_ in ("matched", "possible match (same city, operator not confirmed)", "not in account", "no address in Epoch"):
        if s_ in by.index:
            L.append(f"| {s_} | {int(by.loc[s_, 'size'])} | {by.loc[s_, 'sum']:,.0f} | {100 * by.loc[s_, 'sum'] / tot:.0f}% |")
    L += ["", f"For matched campuses, account facility power / Epoch power: median {ratio['50%']:.2f} "
          f"(range {ratio['min']:.2f}-{ratio['max']:.2f}, n = {int(ratio['count'])}).", "",
          "## Campuses of 200 MW or more", "", "| campus | owner | state | Epoch MW | status | account facility MW (est.) |",
          "|---|---|---|---|---|---|"]
    for r in big.itertuples():
        L.append(f"| {r.epoch_name} | {r.owner} | {r.state} | {r.epoch_power_mw:,.0f} | {r.status} | {r.account_facility_mw_est:,.0f} |")
    L += ["", "## What it means", "",
          "- Campuses **not in the account** are AI load the US numbers do not include, so the US account is a lower",
          "  bound for frontier AI (it already covers only ~30% of LBNL's national datacenter energy).",
          "- This is a coverage check, not a correction: nothing in the account was changed (the pipeline's rule is",
          "  to fix parameters from evidence before a run, and Epoch's MW are estimates for facilities the Compute",
          "  Atlas release may not yet list).",
          "- *Possible match* = the account has a facility in the same city but no operator evidence in Epoch links them",
          "  (e.g. Stargate Abilene vs Crusoe's Abilene site); these need a manual check before being counted either way.",
          "- One account record can stand for several Epoch campuses (Colossus 1 and 2 both match the xAI Colossus record)."]
    (out / "epoch_crosscheck.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
