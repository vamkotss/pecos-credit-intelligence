# Pecos Credit Intelligence

**A credit-memo drafting system that refuses to state a number it cannot trace
to a page.**

Middle-market commercial lenders receive loan packages as 200–600 page folders
of scanned PDFs — financial statements, tax returns, rent rolls, appraisals,
loan agreements. A credit analyst spends 6–8 hours reading one to produce a
credit memo for committee.

The obvious thing to build is a summariser. The obvious thing is wrong, because
in this domain a *confidently invented figure is worse than no output at all*.
A memo that misstates EBITDA by transposing two digits goes to committee as
fact.

So the design constraint here is inverted from a typical RAG project. The
system's job is not to always answer. It is to answer only when every figure it
states resolves to an extracted value on a cited page — and to refuse and
escalate to a human when it cannot.

---

## Status

🚧 **In progress — Milestone 1 of 10 complete.**

| # | Milestone | Status |
|---|---|---|
| M1 | Scaffold, business brief, ADRs, config, CI | ✅ |
| M2 | Seeded loan-package generator + ground-truth manifest | ⬜ |
| M3 | Ingestion: OCR, deskew, table extraction, page provenance | ⬜ |
| M4 | Structure-aware chunking with source anchors | ⬜ |
| M5 | Hybrid retrieval: BM25 + dense + cross-encoder rerank | ⬜ |
| M6 | Evaluation harness + CI regression gate | ⬜ |
| M7 | LangGraph credit-memo agent + calculator tool | ⬜ |
| M8 | Guardrails, grounding verifier, red-team suite | ⬜ |
| M9 | Model routing, cost accounting, caching | ⬜ |
| M10 | Human review queue, model card, runbook | ⬜ |

---

## The three decisions that shape this repo

**1. The corpus is generated, so ground truth is knowable.**
Real loan packages carry PII and cannot be distributed; public filings are too
clean to be interesting. A seeded generator produces synthetic packages *and* a
manifest recording every planted figure, its page, its fiscal year, and its
defect class. The documents render to real PDFs, get rasterised, degraded, and
rotated — the ingestion pipeline sees genuine scan noise. The documents are
synthetic; the difficulty is not. → [ADR 0001](docs/decisions/0001-synthetic-corpus.md)

**2. Retrieval is hybrid, because dense-only fails predictably here.**
Analyst questions are dense with exact tokens: account labels, entity names,
defined terms, fiscal-year markers. Embeddings compress exactly that detail
away, so a dense-only retriever returns the FY2023 revenue table when asked
about FY2024. BM25 plus dense, fused, then cross-encoder reranked — with an
ablation table proving each stage earns its place. → [ADR 0002](docs/decisions/0002-hybrid-retrieval.md)

**3. The evaluation harness ships before the agent.**
Evaluation built after a pipeline exists is not evaluation, it is
justification — the question set ends up written by looking at what the
pipeline already answers well. Here the question set is derived from the M2
ground-truth manifest before any generation code is written, and a committed
baseline gates CI. → [ADR 0003](docs/decisions/0003-evals-before-agent.md)

---

## The planted difficulty

Ten defect classes, seeded at controlled rates so evaluation can report recall
on rotated pages separately from recall on clean ones:

scanned pages with no text layer · 90° and 180° rotation · tables spanning page
breaks · multi-column layouts · three fiscal-year columns side by side ·
restated prior-year figures · handwritten annotations · inconsistent entity
naming · negatives in parentheses · units drift between thousands and units ·
and a prompt-injection payload embedded in borrower content.

Full catalogue with target rates: [data dictionary](docs/data-dictionary.md).

---

## Quick start

```powershell
# 1. Clone and enter
git clone https://github.com/vamkotss/pecos-credit-intelligence.git
cd pecos-credit-intelligence

# 2. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Dependencies
pip install -r requirements.txt

# 4. Configuration
Copy-Item .env.example .env      # then add your OpenAI key

# 5. Verify
$env:PYTHONPATH = "src"
python -m pytest
```

Local Postgres with pgvector (needed from M5):

```powershell
docker compose up -d
```

---

## Architecture

Full diagrams and the component trade-off table:
[docs/architecture.md](docs/architecture.md).
Business framing, stakeholders, and success metrics:
[docs/business-brief.md](docs/business-brief.md).

---

## Stack

Python 3.12 · Tesseract OCR · Poppler · Postgres + pgvector · rank-bm25 ·
sentence-transformers cross-encoder · LangGraph · OpenAI API · RAGAS ·
Langfuse · pytest · ruff · GitHub Actions
