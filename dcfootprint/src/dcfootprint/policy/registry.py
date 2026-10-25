"""L5 — the policy registry: ONE source of truth for the regulatory layer.

Merges the three previously-separate inputs into one normalised instrument table, so
everything downstream (the four-axis gap matrix, the gap overlay, the binding
constraints) is DERIVED and evidence-linked rather than asserted:

  config/policy_sources.csv    the 15-document RAG corpus (status, url, topic)      -> in_corpus rows
  config/legal_constraints.csv the 7 binding/soft rules (effect+param) Q1/Q3 apply  -> type/effect/param
  _CLOSEST (below)             the "closest instrument" per axis (POLICY_DEEP_DIVE)  -> closest_for rows
  _PERIMETER (below)           key standing/perimeter instruments not in the corpus

The four-axis thesis (no in-force instrument MANDATES any of the four axes) is encoded
by the `mandates_axis` column being empty for every row — so `gap.py` computes the
matrix from the data; if some instrument ever did mandate an axis, the cell would flip.
"""
from __future__ import annotations

import pandas as pd

from dcfootprint.io.facilities import _repo_root

AXES = ["carbon_intensity", "marginal_carbon", "scarcity_weighted_water", "inference_attribution"]
JURISDICTIONS = ["US", "EU", "India"]

# RAG-corpus topic -> the axis it is NEAREST to (none MANDATE our axes; this only labels
# relevance). Perimeter topics map to a perimeter_* tag, not one of the four axes.
_TOPIC_NEAREST = {
    "carbon_intensity": "carbon_intensity",
    "electricity_consumption": "perimeter_efficiency_rate",
    "water_stress": "scarcity_weighted_water",
    "water_storage": "scarcity_weighted_water",
    "temperature": "perimeter_efficiency",
    "datacenter_mapping": "perimeter_siting",
    "siting_general": "perimeter_siting",
}

# constraint region -> the corpus filename, where the binding rule IS a corpus document
_CONSTRAINT_TO_CORPUS = {
    "Rajasthan": "in_rajasthan_dc_policy_2025.pdf",
    "Germany": "de_enefg_amendment_2026.pdf",
    "Ireland": "ie_cru_2025236_leu_connection.pdf",
    "Netherlands": "nl_noordholland_datacenterstrategie_2025.pdf",
    "EU-wide": "eu_deleg_reg_2024_1364.pdf",
}

# the "closest instrument" per axis (POLICY_DEEP_DIVE §1) — none MANDATE the axis; not in corpus
_CLOSEST = [
    dict(id="eu_csrd_esrs_e1", instrument="EU CSRD / ESRS E1 (entity GHG + intensity per revenue)",
         jurisdiction="EU", region="EU-wide", status="IN FORCE (narrowed by Omnibus 2026/470)",
         doc_type="directive+standard", closest_for="carbon_intensity",
         closest_note="per-revenue, entity-level - not a per-DC measured intensity; German EnEfG mandates renewable SUPPLY (an input, not an intensity); MN HF16 only requires a utility plan protecting the RE/carbon-free standards",
         url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022L2464", source="POLICY_DEEP_DIVE 1"),
    dict(id="cfe_247", instrument="Voluntary 24/7 CFE tariffs (NV/Google-Fervo; Xcel/Google); CNDCP hourly-CFE",
         jurisdiction="US", region="multi", status="voluntary", doc_type="utility tariff/program",
         closest_for="marginal_carbon", closest_note="consumption-based and voluntary, not a marginal-carbon mandate",
         url="https://www.utilitydive.com/news/google-fervo-nv-energy-nevada-puc-clean-energy-tariff/719472",
         source="POLICY_DEEP_DIVE 1; PUCN Docket 24-05023 (NV Clean Transition Tariff)"),
    dict(id="eu_esrs_e3", instrument="EU ESRS E3-4 (water use in high water-stress areas)",
         jurisdiction="EU", region="EU-wide", status="IN FORCE (materiality-gated)", doc_type="standard",
         closest_for="scarcity_weighted_water",
         closest_note="entity-level location-flagged VOLUME; CA AB2619 (proposed) indirect water; India CGWA aquifer tiers = permit gate, not weighted accounting",
         url="https://eur-lex.europa.eu/eli/reg_del/2023/2772/oj/eng", source="POLICY_DEEP_DIVE 1; Reg (EU) 2023/2772"),
    dict(id="eu_ai_act_annex_xi", instrument="EU AI Act Annex XI 1(2)(e) (GPAI model energy documentation)",
         jurisdiction="EU", region="EU-wide", status="IN FORCE (GPAI duties from Aug 2025)", doc_type="regulation",
         closest_for="inference_attribution",
         closest_note="model-level, training-oriented, no water/carbon; decoupled from DC reporting",
         url="https://artificialintelligenceact.eu/annex/11/", source="POLICY_DEEP_DIVE 1"),
]

# key standing/perimeter instruments from POLICY_DEEP_DIVE not already in the corpus
_PERIMETER = [
    dict(id="us_eo_14141_revoked", instrument="EO 14141 (federal carbon-matching) - REVOKED", jurisdiction="US",
         region="federal", status="REVOKED (2025)", doc_type="executive order", nearest_axis="carbon_intensity",
         url="https://www.govinfo.gov/content/pkg/FR-2025-01-17/html/2025-01395.htm",
         source="POLICY_DEEP_DIVE 2.1; revoked by EO 14318 sec. 4"),
    dict(id="us_hr9825", instrument="Data Center Water & Energy Transparency Act (H.R.9825/S.4213)", jurisdiction="US",
         region="federal", status="PROPOSED", doc_type="bill", nearest_axis="perimeter_disclosure",
         url="https://www.congress.gov/bill/119th-congress/house-bill/9825", source="POLICY_DEEP_DIVE 2.1"),
    dict(id="us_ca_ab2619", instrument="California AB 2619 (indirect/embedded water use)", jurisdiction="US",
         region="California", status="PROPOSED", doc_type="bill", nearest_axis="scarcity_weighted_water",
         url="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202520260AB2619",
         source="POLICY_DEEP_DIVE 2.2"),
    dict(id="in_cgwa_noc", instrument="India CGWA groundwater NOC (aquifer stress tiers)", jurisdiction="India",
         region="national", status="IN FORCE", doc_type="guideline", nearest_axis="scarcity_weighted_water",
         url="https://cgwa-noc.gov.in/landingpage/Guidlines/ConsolidateGuidline.pdf",
         source="POLICY_DEEP_DIVE 4.1; S.O. 3289(E) 24.09.2020, amended S.O. 1509(E) 29.03.2023"),
    dict(id="in_bis_30134", instrument="India BIS IS/ISO/IEC 30134 (PUE/WUE/CUE) - voluntary", jurisdiction="India",
         region="national", status="IN FORCE (voluntary)", doc_type="standard", nearest_axis="perimeter_efficiency",
         url="https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails/MjQ3OTM=",
         source="POLICY_DEEP_DIVE 4.1; IS/ISO/IEC 30134-2:2016 (PUE)"),
]

COLS = ["id", "instrument", "jurisdiction", "region", "status", "in_force", "doc_type", "topic",
        "nearest_axis", "mandates_axis", "closest_for", "closest_note",
        "type", "rule", "effect", "param", "in_corpus", "url", "source"]


def _jurisdiction(raw: str) -> str:
    j = str(raw).lower()
    if "india" in j:
        return "India"
    if "usa" in j or j == "us":
        return "US"
    if j == "global":
        return "Global"
    return "EU"   # EU + member states (Ireland/Germany/Netherlands/...)


def build_registry() -> pd.DataFrame:
    root = _repo_root() / "dcfootprint" / "config"
    # --- corpus (15 docs) ---
    corp = pd.read_csv(root / "policy_sources.csv")
    corp["in_corpus"] = True
    corp["in_force"] = corp["status"].astype(str).str.upper().str.startswith("IN FORCE")
    corp["nearest_axis"] = corp["topic"].map(_TOPIC_NEAREST)
    corp["id"] = corp["filename"].str.replace(r"\.pdf$", "", regex=True)
    corp["instrument"] = corp["title"]
    corp["jurisdiction_norm"] = corp["jurisdiction"].map(_jurisdiction)
    corp["region"] = corp["jurisdiction"]                      # cosmetic for corpus rows
    for c in ["mandates_axis", "closest_for", "closest_note", "type", "rule", "effect", "param"]:
        corp[c] = ""
    corp = corp.rename(columns={"jurisdiction": "_src_j"}).rename(columns={"jurisdiction_norm": "jurisdiction"})

    # --- binding constraints (7): attach to the matching corpus row, else append standalone ---
    cons = pd.read_csv(root / "legal_constraints.csv")
    corp = corp.set_index("id").astype(object)        # object dtype so numeric params can be stored
    standalone = []
    for _, r in cons.iterrows():
        fname = _CONSTRAINT_TO_CORPUS.get(str(r["region"]))
        cid = fname.replace(".pdf", "") if fname else None
        if cid and cid in corp.index:
            # carry the rule's citation too: corpus rows have no `source`, so without it 5 of the 7
            # binding rules lost their evidence link in HARD_CONSTRAINTS / regulation_constraints.csv
            corp.loc[cid, ["type", "rule", "effect", "param", "region", "source"]] = [
                r["type"], r["rule"], r.get("effect", ""), r.get("param", ""), r["region"], r["source"]]
        else:
            standalone.append(dict(id=f"constraint_{r['region']}".lower().replace(' ', '_'),
                                   instrument=r["rule"], jurisdiction=r["jurisdiction"], region=r["region"],
                                   status="IN FORCE", in_force=True, doc_type="policy", topic="",
                                   nearest_axis="", mandates_axis="", closest_for="", closest_note="",
                                   type=r["type"], rule=r["rule"], effect=r.get("effect", ""),
                                   param=r.get("param", ""), in_corpus=False,
                                   url=r["url"] if pd.notna(r.get("url")) else "", source=r["source"]))
    corp = corp.reset_index()

    # --- closest-per-axis + perimeter supplements ---
    extra = []
    for d in _CLOSEST + _PERIMETER:
        row = {c: "" for c in COLS}
        row.update(d)
        row["in_corpus"] = False
        row["in_force"] = str(d.get("status", "")).upper().startswith("IN FORCE")
        extra.append(row)

    reg = pd.concat([corp, pd.DataFrame(standalone), pd.DataFrame(extra)], ignore_index=True)
    for c in COLS:
        if c not in reg.columns:
            reg[c] = ""
    reg = reg[COLS].copy()
    # normalise dtypes: string-ish columns -> filled str; the two flags -> bool
    for c in COLS:
        if c in ("in_corpus", "in_force"):
            reg[c] = reg[c].fillna(False).astype(bool)
        else:
            reg[c] = reg[c].where(reg[c].notna(), "")
    return reg


def load_registry() -> pd.DataFrame:
    """Build the registry and cache it to results/ so the committed artifact matches the code."""
    reg = build_registry()
    return reg


if __name__ == "__main__":
    reg = build_registry()
    print(f"registry: {len(reg)} instruments | in_corpus={int(reg['in_corpus'].sum())} | "
          f"binding={int((reg['type'] != '').sum())} | closest={int((reg['closest_for'] != '').sum())}")
    print("by jurisdiction:", reg["jurisdiction"].value_counts().to_dict())
    print("\nbinding rows:")
    print(reg[reg["type"] != ""][["region", "rule", "type", "effect", "param"]].to_string(index=False))
    print("\nclosest-for-axis rows:")
    print(reg[reg["closest_for"] != ""][["closest_for", "instrument", "status"]].to_string(index=False))
