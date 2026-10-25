"""L5 — bridge to an external legal-RAG project (location set by $DCF_RAG_DIR; not included here).

The policy corpus now lives in the single registry (`policy/registry.py`, built from
`config/policy_sources.csv`); this module exposes the corpus view + the optional live-RAG
hook. The external project is a LangGraph RAG (hybrid dense bge-m3 + BM25 + RRF +
cross-encoder rerank; LLM answers with inline citations and IN FORCE / PROPOSED /
DRAFT / REVOKED status) over 15 real policy PDFs. Its retrieval + LLM steps run in its own
venv (Ollama or an API key) and are NOT invoked by the pipeline — `answer_live()` is optional.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from dcfootprint.policy.registry import load_registry

RAG_DIR = Path(os.environ.get("DCF_RAG_DIR", "rag"))


def load_manifest() -> pd.DataFrame:
    """The corpus view of the registry (the 15 cited, status-flagged policy documents)."""
    reg = load_registry()
    return reg[reg["in_corpus"]].copy()


def cited_constraints() -> pd.DataFrame:
    """In-force corpus instruments, cited (the regulated perimeter that actually bites)."""
    corp = load_manifest()
    keep = corp[corp["in_force"]].copy().rename(columns={"instrument": "title"})
    return keep[["jurisdiction", "region", "title", "doc_type", "topic", "status", "url"]].reset_index(drop=True)


def corpus_summary() -> dict:
    corp = load_manifest()
    return {
        "n_documents": int(len(corp)),
        "n_in_force": int(corp["in_force"].sum()),
        "n_proposed_or_draft": int((~corp["in_force"]).sum()),
        "jurisdictions": sorted(corp["region"].astype(str).unique().tolist()),   # fine-grained (region)
        "topics": sorted(corp["topic"].astype(str).unique().tolist()),
        "axes_directly_mandated": 0,     # corpus is the regulated *perimeter* — confirms the 4-axis gap
        "source": "policy registry (config/policy_sources.csv), human-verified + status-flagged",
    }


def answer_live(question: str, timeout: int = 300) -> str | None:
    """OPTIONAL: run the external RAG in its own venv to (re)generate a cited answer.
    Needs Ollama running or ANTHROPIC_API_KEY. Not called by the pipeline; returns None
    if the venv/entrypoint isn't available."""
    py = RAG_DIR / "venv" / "Scripts" / "python.exe"
    if not (py.exists() and (RAG_DIR / "ask.py").exists()):
        return None
    try:
        r = subprocess.run([str(py), "ask.py", question], cwd=str(RAG_DIR),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or None
    except Exception:
        return None


if __name__ == "__main__":
    print("corpus:", corpus_summary())
    print("\nin-force cited constraints:")
    print(cited_constraints().to_string(index=False))
