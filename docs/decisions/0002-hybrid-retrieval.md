# ADR 0002 — Hybrid retrieval (BM25 + dense + cross-encoder rerank)

**Status:** Accepted · **Date:** 2026-08-04

## Context

The default portfolio-project retrieval stack is: embed chunks, cosine
similarity, top-5, done. It is one function call and it demos well.

It also fails on this corpus in a specific, predictable way. Analyst questions
are dense with **exact tokens**: account labels ("Accounts receivable, net"),
entity names ("Big Bend Fabrication LLC"), defined terms from the loan agreement
("Fixed Charge Coverage Ratio"), and fiscal-year markers. Embedding models
compress exactly this kind of surface detail away — that is what they are for.
A dense-only retriever will happily return the FY2023 revenue table when asked
about FY2024, because the two are near-identical in embedding space.

## Decision

Three stages:

1. **BM25** over chunk text — top-20. Catches exact tokens.
2. **Dense retrieval** over pgvector — top-20. Catches paraphrase.
3. **Reciprocal rank fusion**, then a **local cross-encoder rerank** to top-6.

## Rationale

BM25 and dense retrieval fail on *different* queries, which is the entire
argument for fusing them — the union recovers a materially higher recall ceiling
than either alone. Reciprocal rank fusion is chosen over score normalisation
because BM25 and cosine scores are not on comparable scales and any
normalisation scheme is an unprincipled fudge that needs retuning per corpus.

The cross-encoder is where the actual precision comes from. Bi-encoders score
query and document independently; a cross-encoder reads them together and can
tell that a chunk mentioning FY2023 does not answer an FY2024 question. Running
it locally keeps it out of the API bill and out of the latency budget.

## Consequences

- More moving parts. Justified only because M6 will **measure** each stage's
  contribution — a recall@k ablation table (dense only / BM25 only / fused /
  fused+reranked) is a deliverable, not a footnote.
- If the ablation shows the cross-encoder adds nothing, it gets cut and the ADR
  gets a superseding note. Being willing to delete a component because the
  numbers said so is worth more in an interview than the component itself.
- The cross-encoder model (~90 MB) downloads to the Hugging Face cache on D:.
