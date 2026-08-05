"""Pecos Credit Intelligence -- document intelligence for middle-market lending.

Package layout (modules land here as milestones ship):

    config.py      M1  -- typed settings, single source of env truth
    corpus.py      M2  -- seeded synthetic loan-package generator
    ingest.py      M3  -- OCR, layout, table extraction, page provenance
    chunking.py    M4  -- structure-aware chunking with source anchors
    retrieval.py   M5  -- BM25 + dense + cross-encoder rerank
    evals.py       M6  -- retrieval and generation evaluation harness
    agent.py       M7  -- LangGraph credit-memo agent with a calculator tool
    guardrails.py  M8  -- injection defence, numeric grounding, refusal paths
    routing.py     M9  -- model routing and per-memo cost accounting
    review.py      M10 -- human-in-the-loop review queue and governance
"""

__version__ = "0.1.0"
