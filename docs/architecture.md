# Architecture

## System overview

```mermaid
flowchart TD
    subgraph GEN["M2 · Corpus generation"]
        A1["Seeded loan-package generator<br/>40 borrowers, planted defects"]
        A2["Ground-truth manifest<br/>every figure, its page, its year"]
        A1 --> A2
    end

    subgraph ING["M3 · Ingestion"]
        B1["Page rasterise + deskew"]
        B2["Text layer OR Tesseract OCR"]
        B3["Table detection + extraction"]
        B4["Provenance record<br/>doc / page / bbox"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph IDX["M4-M5 · Index & retrieve"]
        C1["Structure-aware chunking<br/>tables kept whole, headers carried"]
        C2["BM25 sparse index"]
        C3["Dense index · pgvector"]
        C4["Cross-encoder rerank"]
        C1 --> C2
        C1 --> C3
        C2 --> C4
        C3 --> C4
    end

    subgraph AGT["M7-M9 · Agent"]
        D1["LangGraph credit-memo agent"]
        D2["Calculator tool<br/>ratios computed, never generated"]
        D3["Grounding verifier"]
        D4["Model router + cost meter"]
        D1 --> D2 --> D3
        D4 -.-> D1
    end

    subgraph OUT["M6, M8, M10 · Trust layer"]
        E1["Eval harness<br/>recall@k, faithfulness, year-binding"]
        E2["Guardrails + red team"]
        E3["Human review queue"]
        E4["Langfuse traces"]
    end

    A2 --> B1
    B4 --> C1
    C4 --> D1
    D3 --> E3
    D1 -.trace.-> E4
    A2 -.ground truth.-> E1
    C4 -.retrieval eval.-> E1
    D3 -.generation eval.-> E1
    E2 -.blocks.-> D1
    E1 -.CI gate.-> CI["GitHub Actions<br/>fails build on eval regression"]
```

## The retrieval path in detail

```mermaid
flowchart LR
    Q["Analyst question<br/>'What was FY2024 EBITDA?'"] --> R1["Query expansion<br/>synonyms + fiscal-year hint"]
    R1 --> S["BM25 top-20"]
    R1 --> D["Dense top-20"]
    S --> F["Reciprocal rank fusion"]
    D --> F
    F --> X["Cross-encoder rerank<br/>top-6"]
    X --> G["Answer with citations<br/>doc + page + bbox"]
    G --> V{"Grounding<br/>verifier"}
    V -->|"every figure matches<br/>an extracted value"| OK["Return"]
    V -->|"any figure unmatched"| REJ["Refuse + queue<br/>for human review"]
```

The refusal branch is the point of the diagram. Most RAG demos have no arrow
that returns nothing. This one does, and the eval suite specifically measures
how often it fires correctly versus spuriously.

## Why these components

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Vector store | pgvector in Postgres | Pinecone, Chroma, FAISS | Chunks need relational neighbours (doc, page, entity, fiscal year); a metadata-filtered hybrid query is one SQL statement. No new managed service, no bill. |
| Sparse retrieval | BM25 (rank_bm25) | Postgres full-text only | Financial queries are full of exact tokens — account labels, entity names, "EBITDA". Dense-only retrieval reliably misses them. This is the single highest-value non-obvious component. |
| Reranking | Local cross-encoder | LLM-as-reranker | 40× cheaper, ~10 ms, no API dependency in the hot path. Quality difference is measured in M6, not asserted. |
| OCR | Tesseract via pytesseract | Cloud OCR (Textract, Document AI) | Free, local, reproducible in CI. Cloud OCR is better; the ADR records that trade honestly, and the AWS billing incident from earlier this year is a live reason to keep this project off metered cloud services. |
| Agent framework | LangGraph | Bare function-calling loop, CrewAI | Explicit state graph means the control flow is inspectable and testable. An agent you cannot unit-test is a demo. |
| Tracing | Langfuse (self-hosted via Compose) | LangSmith | Runs offline, no seat cost, and it is one more container in the same Compose file. |
| Numbers | Calculator tool, always | Let the LLM compute ratios | LLM arithmetic is the loss event described in the brief. Ratios are computed in Python from extracted values, full stop. |

## Milestones

| # | Milestone | Ships | Est. hrs |
|---|---|---|---|
| M1 | Scaffold, brief, ADRs, config, CI | This delivery | 6 |
| M2 | Seeded loan-package generator + ground-truth manifest | Corpus with planted defects | 12 |
| M3 | Ingestion: OCR, deskew, tables, page provenance | Extraction quality report | 12 |
| M4 | Structure-aware chunking with source anchors | Chunk store + anchor tests | 8 |
| M5 | Hybrid retrieval: BM25 + dense + rerank | Retriever with tuned fusion | 10 |
| M6 | Eval harness + CI regression gate | recall@k, faithfulness, baseline JSON | 12 |
| M7 | LangGraph credit-memo agent + calculator tool | End-to-end memo generation | 12 |
| M8 | Guardrails, grounding verifier, red-team suite | Injection defence report | 8 |
| M9 | Model routing, cost accounting, caching | Cost-per-memo report | 6 |
| M10 | HITL review queue, model card, runbook, README | Ship ritual | 8 |

Total ≈ 94 hours. Build order is deliberate: **the eval harness (M6) lands
before the agent (M7)**, so the agent is developed against a scoreboard rather
than against vibes. That inversion is the thing most portfolio RAG projects get
wrong, and it is the thing an interviewer will notice.
