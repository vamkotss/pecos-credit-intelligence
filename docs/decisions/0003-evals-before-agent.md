# ADR 0003 — The evaluation harness ships before the agent

**Status:** Accepted · **Date:** 2026-08-04

## Context

Natural build order for a RAG-and-agents project is: ingest, chunk, retrieve,
generate, agent, and then — once there is something impressive to show —
evaluate. Nearly every portfolio project in this category is built that way.

## Decision

M6 (evaluation harness, labelled question set, CI regression gate) ships
**before** M7 (the agent). No agent code is written until there is a scoreboard.

## Rationale

Evaluation built after the fact is not evaluation, it is justification. Once a
pipeline exists and appears to work, the question set gets written by looking at
what the pipeline already answers well. The resulting numbers are high and
meaningless, and the author usually cannot tell.

Building the question set from the M2 ground-truth manifest — before any
generation code exists — makes that failure impossible. The questions are
derived from what is *in the documents*, not from what the system happens to
handle.

Second reason: every subsequent milestone becomes a measurable experiment
instead of a guess. "Should chunks be 512 or 1024 tokens?" has an answer.
"Does the reranker earn its latency?" has an answer. Without M6 those are
opinions, and an interviewer asking "how do you know?" gets nothing.

Third: the CI gate. A committed baseline JSON plus a workflow step that fails
the build when faithfulness drops more than a stated tolerance turns evaluation
from a one-time report into a standing constraint. This is the single strongest
signal in the whole repo that the work is production-minded — it is what
separates an AI engineer from someone who has used the OpenAI SDK.

## Consequences

- The repo has no demo-able output until roughly hour 60. Accepted.
- The question set is a first-class, version-controlled artefact under
  `evals/datasets/`, reviewed as carefully as code.
- Evaluating evaluation: a small adjudicated subset checks that the
  LLM-as-judge agrees with human labels. An unvalidated judge is just another
  ungrounded model.
