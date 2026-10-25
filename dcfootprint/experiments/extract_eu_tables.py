"""Rebuild the EU Member State table from the Commission report PDF (Tables 13 and 24) and check it
against the report's own EU totals (Table 22 and the Table 13 TOTAL row).

Source PDF: data/eu_eed/EU_DC_assessment_first_technical_report_2025-07.pdf
  (Commission, "Assessment of the energy performance and sustainability of data centres in EU —
   First Technical Report", EY/AIT/Borderstep, July 2025; op.europa.eu identifier
   83be4c3e-5c79-11f0-a9d0-01aa75ed71a1)
Output: dcfootprint/config/eu_member_state_2023.csv (versioned; region_config.EU.eu_eed_path)

Run: python dcfootprint/experiments/extract_eu_tables.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "eu_eed" / "EU_DC_assessment_first_technical_report_2025-07.pdf"
OUT = ROOT / "dcfootprint" / "config" / "eu_member_state_2023.csv"
ISO2 = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "DE": "Germany", "DK": "Denmark", "EL": "Greece",
        "ES": "Spain", "FI": "Finland", "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
        "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
        "PL": "Poland", "PT": "Portugal", "SE": "Sweden"}
EU27 = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France",
        "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
        "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]


def main() -> pd.DataFrame:
    pages = [p.extract_text() or "" for p in PdfReader(PDF).pages]
    t13_page = next(i for i, t in enumerate(pages) if "Table 13 - Estimated vs. reported number" in t)
    lines = "\n".join(pages[t13_page:t13_page + 2]).split("\n")
    t13 = {}
    for ln in lines:
        m = re.match(r"^([A-Z][A-Za-z ]+?)\*?\s+(\d[\d ]*?)\s+(\d+)\s+(\d+)%\s*$", ln.strip())
        if m and m.group(1).strip() in EU27 + ["TOTAL"]:
            t13[m.group(1).strip()] = (int(m.group(2).replace(" ", "")), int(m.group(3)), int(m.group(4)))
            if m.group(1).strip() == "TOTAL":
                break                                    # Table 14 on the next page has its own TOTAL row
    t24_page = next(i for i, t in enumerate(pages) if "Table 24 - Member State data on total IT power" in t)
    t24 = {}
    for ln in "\n".join(pages[t24_page:t24_page + 2]).split("\n"):
        m = re.match(r"^([A-Z]{2})\s+([\d ]+\.\d+)\s+([\d ]+\.\d+)\s+([\d ]+)\s*$", ln.strip())
        if m and m.group(1) in ISO2:
            f = lambda s: float(s.replace(" ", ""))
            t24[ISO2[m.group(1)]] = (m.group(1), f(m.group(2)), f(m.group(3)), f(m.group(4)))
    rows = []
    for c in EU27:
        est, rep, pct = t13[c]
        iso, mw, gwh, w = t24.get(c, (None, None, None, None))
        rows.append({"country": c, "est_n_dc": est, "reporting_n_dc": rep, "reporting_share_pct": pct,
                     "iso2_eu": iso, "reported_it_mw": mw, "reported_energy_gwh": gwh, "reported_water_input_m3": w,
                     "source_table13": f"Table 13 (pdf p.{t13_page + 1})",
                     "source_table24": f"Table 24 (pdf pp.{t24_page + 1}-{t24_page + 2})" if iso else "no reporting DCs"})
    df = pd.DataFrame(rows)
    # checks against the report's own totals
    assert abs(df["reported_it_mw"].sum() - 3738.86) < 0.02, df["reported_it_mw"].sum()
    assert abs(df["reported_energy_gwh"].sum() - 14088.0) < 0.02
    assert abs(df["reported_water_input_m3"].sum() - 6223391) < 0.5
    assert (df["est_n_dc"].sum(), df["reporting_n_dc"].sum()) == (t13["TOTAL"][0], t13["TOTAL"][1]) == (2161, 770)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} Member States, {df['reported_energy_gwh'].notna().sum()} reporting); totals match the report")
    return df


if __name__ == "__main__":
    main()
