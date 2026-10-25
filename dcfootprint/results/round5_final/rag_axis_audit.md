# L5 — RAG retrieval audit (frozen build-time output, 2026-10-03)

How the external legal-RAG (`<RAG_DIR>`; LangGraph, hybrid dense bge-m3 +
BM25 + RRF + cross-encoder rerank) is used in the policy layer, with the evidence.

## Decision
**The RAG is NOT the policy backbone.** The hand-verified registry (`policy/registry.py`,
26 instruments) is more complete and India-correct than the RAG corpus. The RAG is used as
(1) this **frozen corroboration exhibit**, and (2) an **optional live audit** of the EU/US
perimeter it covers well (`rag_bridge.answer_live`). The corpus is **not expanded** — a
retrieval+LLM system to produce a 12-cell gap map is over-engineering. India depth + the four "closest instruments" stay hand-verified in the registry.

## Corpus skew (1,056 chunks across 15 docs)
**86% EU:** CADA 271 (a *proposed*, tangential act), CRU 259, EnEfG 184, JRC 121, 2024/1364 37,
Noord-Holland 35. **India = 3 chunks** (Rajasthan — scanned PDF, text extraction failed) **+ a
lapsed MeitY-2020 draft (22).** US thin (TX SB6 25, EO 14318 9, VA 3, PNNL 3, ELI 7).
→ the corpus **cannot support India depth** and **cannot retrieve the Rajasthan ZLD mandate.**

## Retrieval results (run 2026-10-03; retrieval only, no LLM)
| Our axis / query | Top retrievals | Relevant to our axis? |
|---|---|---|
| carbon_intensity | Texas SB6, CRU (grid-access / rate) | **No** — perimeter; no intensity mandate exists to retrieve |
| marginal_carbon | WattTime MOER (method) + CRU/ELI noise | Partial — 1 method hit, no mandate |
| scarcity_weighted_water | ELI fact sheet, Aqueduct (method), NL siting | Water-adjacent context/method; **no weighting mandate, no India** |
| inference_attribution | ELI water fact sheet ×3, JRC efficiency | **No** — noise; no inference doc in the corpus |
| ZLD mandate (India) | MeitY-2020 draft (lapsed) ×3 | **No** — the real Rajasthan ZLD is unretrievable (scanned PDF) |
| India policy (general) | MeitY-2020 draft + EU JRC guidelines | Weak — a lapsed draft + EU documents |

## The one useful finding
For **all four axes**, the top retrievals are grid-access / efficiency / water-volume /
methodology documents — **never a mandate.** This **corroborates the regulatory-gap thesis**
(0/4 axes mandated → 100% blind spot), independently of the hand-coded registry. Usable paper
sentence: *"hybrid retrieval over 15 real policy documents returns, for each of the four axes,
only grid-access, efficiency, water-volume and methodology instruments — never a mandate —
corroborating the gap."*

## Why the registry wins for the map
The four "closest instruments" the matrix needs — **CSRD/ESRS E1, ESRS E3, AI Act Annex XI,
24/7 CFE tariffs** — are **not in the corpus**. The registry has them; the RAG cannot retrieve them.

## If ever expanded (not recommended for a 12-cell map)
OCR the Rajasthan PDF; add CSRD/ESRS, AI Act Annex XI, CFE examples, CGWA, BIS, Gujarat;
rebalance away from CADA's 271 chunks; populate `goldset.csv` and run `eval.py` (recall@8 ≥ 0.85).

## Reproduce
`cd <RAG_DIR> && venv/Scripts/python.exe search.py "<query>"`
(retrieval only, no LLM; BGE models are cached). Query set as in the table above.
