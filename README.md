# Pecos Credit Intelligence

Document intelligence for middle-market commercial lending: OCR, hybrid
retrieval, evaluated RAG, and a credit-memo agent whose every figure is
traceable to a page or to a recorded calculation.

[![CI](https://github.com/vamkotss/pecos-credit-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/vamkotss/pecos-credit-intelligence/actions/workflows/ci.yml)

Pecos Capital Partners is a fictional Fort Worth lender writing $3M–$40M senior
secured facilities. A credit analyst there reads a 180–600 page loan package —
financial statements, tax returns, bank statements, debt schedules, broker
correspondence — and writes a memorandum recommending whether to proceed.

This project automates the reading and the arithmetic, and is built around one
constraint: **a figure that reaches a credit committee must be traceable to the
page it came from, or to a calculation whose inputs are.**

| | |
|---|---|
| Tests | 391 |
| Gated thresholds | 12, evaluated in under 6 seconds |
| Chunk containment | 100% — no gold fact is lost before retrieval |
| Retrieval recall@5 | 100% |
| Red-team attacks, 0 succeeded | 108 deterministic, 27 against Claude |
| Architecture decisions | 12 ADRs |

---

## Quickstart

Requires **Python 3.12** and **Tesseract 5.x** on `PATH`. Poppler is not needed.

```bash
pip install -r requirements.txt

python scripts/generate_corpus.py      # 12 loan packages + ground-truth manifest
python scripts/ingest_corpus.py        # OCR, orientation, tables, page provenance
python scripts/chunk_corpus.py --audit # chunk, and prove nothing was lost
python scripts/eval_gate.py            # 12 thresholds, ~6s, no API key
```

The entire pipeline runs offline and deterministically. `ANTHROPIC_API_KEY` is
needed only for `--drafter anthropic`, `--generator anthropic` and
`--judge anthropic`, which are optional everywhere they appear.

### The five-minute tour

```bash
python scripts/eval_retrieval.py               # recall@k and MRR against the oracle
python scripts/write_memo.py --deal PCP-0004 --audit
python scripts/redteam.py --limit 3            # 27 attacks, 3 families
python scripts/review_queue.py list            # what needs a human
```

`PCP-0004` is the one to read. Its memo contains the pro forma paragraph that
flips the recommendation, and `--audit` prints the derivation of every computed
figure with the pages its inputs came from.

---

## Pipeline

```
generate → ingest → chunk → retrieve → answer → memo → guardrails → review
   M2        M3       M4       M5        M6      M7        M8         M10
                                          └────── M9: cost, routing, gate ──────┘
```

| module | does |
|---|---|
| `corpus.py`, `rendering.py` | seeded synthetic loan packages with ground truth |
| `ingest.py` | text-layer routing, OCR, orientation, tables, page provenance |
| `chunking.py`, `chunk_audit.py` | structure-aware chunks, containment audit |
| `retrieval.py` | BM25 + dense, rank fusion, rerank, authority weighting |
| `answering.py`, `evaluation.py` | grounded answers, validated citations, metrics |
| `tools.py`, `memo.py`, `drafting.py` | calculator, LangGraph agent, drafters |
| `guardrails.py`, `redteam.py` | policy and identity checks, attack suite |
| `cost.py`, `routing.py`, `gate.py` | budgets, model routing, the CI gate |
| `review.py` | the human queue |

---

## The corpus

Every document is generated, and the ground truth is generated with it. The
manifest is not extracted from the PDFs — the PDFs are rendered from the
manifest's source data. That is what makes recall@k, exact-match extraction and
numeric grounding measurable at all.

**12 packages · 98 PDFs · 182 pages · 127 gold facts.** Two document types per
package are image-only scans with no text layer, so the OCR path cannot silently
stop being exercised.

The borrowers are not decorative. Every figure is an integer number of dollars,
the balance sheet balances to the dollar, the cash-flow statement ties to the
movement in cash, and balance-sheet debt reconciles to the debt schedule —
asserted on every deal and every year. Across a 360-deal sweep the spread runs
1.0x–4.9x leverage and 0.4x–3.6x DSCR, with about a third of borrowers below the
1.25x policy floor. A corpus where everyone passes gives the memo agent nothing
to decide.

### Seven planted defects

A clean synthetic corpus flatters a pipeline. These are planted deliberately,
registered in the manifest, and each has at least one gold question:

| defect | what it breaks |
|---|---|
| `restated_prior_year` | source precedence — two documents, one year, two EBITDAs |
| `units_in_thousands` | unit normalisation — off by exactly 1,000 |
| `rotated_scanned_page` | OCR orientation |
| `table_only_fact` | table-aware chunking — the fact is in no sentence |
| `prompt_injection` | instructions embedded in a source document |
| `unanswerable_question` | refusal instead of invention |
| `near_duplicate_draft` | retrieval dedup and recency preference |

Defects are dealt round-robin, so all seven appear even in the three-deal corpus
CI builds. Random sampling was rejected because a small CI corpus could omit the
injection case, and the red-team suite would pass by not running.

Detail: [docs/corpus.md](docs/corpus.md) · [ADR 0004](docs/decisions/0004-planted-defects-as-an-evaluation-instrument.md)

---

## Ingestion

Every page is routed on its own merits: pages with a text layer go to
pdfplumber, pages without are rendered, OCR'd and marked. Per page rather than
per document, because real packages mix clean exports with appended scanned
signature sheets.

**The page is the unit of provenance.** A document is too coarse to cite — "it's
in the financial statements" is not an answer an analyst accepts. A chunk is too
unstable, because chunk boundaries move whenever the chunking strategy changes.

Read as it sits, the sideways bank statement page returns
`") S89U9INDDO Spun} JUaIONJNSU]"` — noise that would be embedded and retrieved
as if it meant something. Tesseract's orientation detection is used as a
*proposal* and never accepted on faith; its confidence ran from 0.4 to 9.4 on
pages where it was right. The verification signal is **line span**: upright, a
line runs across the page; sideways, each line is one stacked word. Measured on
that page, 1,344 pixels against 19.

| 3 deals | |
|---|---|
| Pages | 46 — 25 digital, 21 OCR |
| Tables recovered | 44 |
| Mean OCR confidence | 94.9 |
| Empty pages | 0 |

Detail: [docs/ingestion.md](docs/ingestion.md) · [ADR 0005](docs/decisions/0005-page-level-ingestion-and-provenance.md)

---

## Chunking

**Containment: 100%.** Every extractive gold fact survives into a chunk anchored
to the page the manifest cites. That number is the ceiling on everything
downstream — a figure lost here cannot be retrieved by any retriever, rescued by
any reranker, or cited by any agent. Measuring it before building a retriever
removes a whole class of misdirected debugging.

### The near-duplicate problem

One deal carries a DRAFT of its financial statements beside the final version.
The income statement chunks are **94% textually similar** — measured, and
asserted by a test so the defect cannot quietly stop being planted.

Cosine similarity cannot separate them. Not because the retriever is bad, but
**because the signal is not in the text being compared.** What separates them is
document *status*, which lives one level up and which naive chunking discards.

So every chunk carries `doc_status` and an `authority` rank — final 3,
superseded 2, draft 1 — and announces non-final status in the header that gets
embedded, so the distinction reaches similarity search and not just a filter.

Filename classifies; page text can only ever **downgrade**. A page stamped DRAFT
is a draft whatever the file is called. Nothing in the text promotes a document
to final, because a draft that fails to say so is still a draft.

Detail: [docs/chunking.md](docs/chunking.md) · [ADR 0006](docs/decisions/0006-chunks-carry-document-status.md)

---

## Retrieval

BM25 and dense embeddings run over a per-deal index, their **rankings** are fused
by reciprocal rank, candidates are reranked, and document authority breaks the
ties that text similarity cannot.

```
recall@1     50.0%      recall@5    100.0%
recall@3     81.6%      MRR          0.675
```

Five of the six planted defects retrieve at **rank 1**.

### Two findings worth stating plainly

**IDF inverts on a financial corpus.** A loan package is mostly tables, so
ordinary English words are *rare* in it. Measured: `what` scored an IDF of 3.12
against `ebitda` at 2.61 — IDF concluded "what" was the most informative term in
"What was EBITDA in FY2025?". The symptom was the broker's cover note, the only
chunk written in flowing prose and containing no figures at all, ranking first
for nearly every financial question. Removing query stopwords took recall@1 from
13.2% to 39.5%.

**Dense retrieval is not currently earning its place.** A fusion-weight sweep
found 0.0 optimal on this eval set. The default is 0.15 — within noise of the
best, enough to keep the hybrid operative. The reason for keeping it: the gold
questions are templated and share vocabulary with their targets, which
structurally favours lexical matching. That is a bias in the eval set, not a
verdict on the method.

### What authority weighting buys

Nothing on the gold set, because the gold question says "final". On the query an
analyst would actually ask — *"What was EBITDA in FY2025?"* — the draft trails
the final by **1.6%** without weighting, which is noise, and leaves the top three
entirely with it. Not a recall point; it keeps a contradictory document out of
the agent's context.

Detail: [docs/retrieval.md](docs/retrieval.md) · [ADR 0007](docs/decisions/0007-rank-fusion-and-authority-tiebreak.md)

---

## Answering and evaluation

Answers cite pages as `[document.pdf#p2]`, and citations are **parsed and
validated** rather than trusted. One pointing at a page that was never retrieved
is dropped and counted — an invented citation is worse than an uncited claim,
because an uncited claim announces itself while an invented citation reads
exactly like evidence and cannot be checked.

### The most important metric is the least clever one

In lending, the failure that ends careers is a figure in a memo that is not in
the file. So **numeric grounding is arithmetic, never a model's opinion**. Four
buckets: **grounded** (on a cited page), **derived** (produced by a recorded
calculation), **uncited** (real figure, wrong provenance — a citation bug), and
**absent** (in no retrieved page — invented).

`absent` is a raw count, never a rate, so one invented figure cannot vanish into
an average. An LLM judge is used for exactly two things that are genuinely
judgements: faithfulness and relevance.

### Baseline versus Claude, same gold set

| | extractive baseline | Claude Haiku |
|---|---|---|
| citation accuracy | 22.8% | **89.0%** |
| answer accuracy | 20.5% | **68.5%** |
| numeric grounding | 100% | 96.5% |
| judged faithfulness | 0.471 (proxy) | **0.945** |

The baseline only quotes, so its 100% grounding is tautological — its value is as
a *contract on the harness*: if that drops below 100%, the metric is broken, not
the generator. The 22.8% and 20.5% are the gap a language model is paid to close,
stated as a number rather than assumed.

Detail: [docs/evaluation.md](docs/evaluation.md) · [ADR 0008](docs/decisions/0008-mechanical-metrics-over-judged-metrics.md)

---

## The credit memo agent

A LangGraph state machine: **gather → compute → draft → verify → (revise) → done**.

| node | does | never does |
|---|---|---|
| gather | retrieve per section, extract figures with pages | judge |
| compute | run the calculator | write prose |
| draft | narrative and recommendation | arithmetic, extraction |
| verify | check every figure is quoted, computed or derivable | guess |

**Extraction and arithmetic are the two things a language model is worst at and
the two things a credit memo cannot get wrong.** Both happen before the drafter
is called. What is left for the model is narrative and judgement.

### Arithmetic left the model

The M6 evaluation flagged seventeen "hallucinated" figures from Claude. Reading
them was the most useful result of that milestone: seven leverage ratios, three
debt totals, one correct thousands conversion, one year with a full stop. **Not
one was fabricated.**

That exposed two problems. The metric was wrong — calling correct arithmetic a
hallucination makes the alarm useless. And the system was wrong too, less
obviously: **a figure a model computed in its head has no provenance.** It might
be right; nothing about the output says so and nothing can check it.

So arithmetic moved into a calculator that records its inputs and their pages:

```
Total debt / EBITDA = 3.12x  [total debt / EBITDA: total debt=8,025,829,
  EBITDA=2,575,918] [02_financial_statements_comparative.pdf#p2] [...#p1]
```

An analyst can check that without reading code.

### Reconstruction

Enumerating every metric an analyst might compute is not a solvable problem. The
figure that settled it was `3.95x`, from:

> The requested facility of $12,800,000 would increase total debt to
> approximately **$21.3 million**, raising pro forma leverage to approximately
> **3.95x** EBITDA — above the 3.5x policy threshold.

Every input is on a cited page, and that sentence flips the recommendation from
PROCEED to DEFER. No list of metrics would have contained it.

So the verifier asks whether a figure **follows arithmetically from figures on
cited pages**, and reports the derivation it found. Reconstructions are
candidates for review, never proof — the derivation goes in front of a person.

Detail: [docs/memo-agent.md](docs/memo-agent.md) · [ADR 0009](docs/decisions/0009-arithmetic-leaves-the-model.md)

---

## Guardrails and red-teaming

**135 attacks across 9 families and 3 mechanisms. None changed a credit
decision.**

Nobody types an attack into the chat box — they email a PDF. Every excerpt this
system reads was supplied by the borrower or their broker, so retrieved content
is untrusted input by construction.

| layer | strength |
|---|---|
| Content delimiters with provenance | weak |
| Instruction detection, six families | weak |
| **Policy check** — recommendation must follow computed metrics | **strong** |
| **Identity check** — accounting identities must hold | **strong** |

An injection can persuade a model. **It cannot change what total debt divided by
EBITDA equals, or make 3.12x pass a 3.5x test evaluated in Python.**

A detected injection is *reported*, not blocked — the attempt is evidence about
the counterparty, while the decision is already constrained by arithmetic.

### The finding that mattered

First run: instruction attacks **0/15**, data poisoning **2/6** — a poisoned
EBITDA flipped two deals from DECLINE to PROCEED.

The policy check could not help. An injected instruction tries to persuade a
model, which the check ignores. A poisoned *figure* flows into the calculator,
which computes on it faithfully, and the check then approves a decision that is
arithmetically correct and factually false. **Garbage in, correctly computed
garbage out.** Data poisoning also carries no instruction to detect, so pattern
matching scores zero — and it is the attack a sophisticated borrower would
actually use.

What stops it is that real statements are **over-determined**: EBITDA cannot
exceed gross profit, and liabilities plus equity must equal assets. The
accounting identities the corpus asserts turn out to be a fraud detector, which
was not why they were built.

Detail: [docs/guardrails.md](docs/guardrails.md) · [ADR 0010](docs/decisions/0010-the-defence-that-does-not-run-through-the-model.md)

---

## Cost, routing and the eval gate

```
PASS  chunk_containment               1.000 >= 1.000
PASS  retrieval_recall_at_5           1.000 >= 0.950
PASS  retrieval_mrr                   0.667 >= 0.550
PASS  baseline_hallucinated_figures   0.000 <= 0.000
PASS  redteam_successes               0.000 <= 0.000
PASS  memos_verified_rate             1.000 >= 1.000
GATE PASSED                                      elapsed 5.6s
```

Twelve thresholds, no API key, fast enough to block every push. **Only free,
deterministic metrics gate.** LLM-dependent numbers go in a separate opt-in job:
a build that fails because a provider had a bad afternoon teaches people to
ignore red builds. A *missing* metric fails rather than being skipped.

**Costs are metered before they are spent.** The red-team run against Claude
crashed with `Your credit balance is too low` — it had run until the provider
refused, and lost twenty completed attacks doing it. Now every call is priced,
the budget is checked *before* the call, and an unpriced model raises rather than
costing zero.

**Routing defaults to the cheap tier**, and is safer here than usual for a
specific reason: the things that must be correct are not decided by a model at
all. A weaker model writes worse prose; it cannot produce an ungrounded figure.

Detail: [docs/cost-and-gate.md](docs/cost-and-gate.md) · [ADR 0011](docs/decisions/0011-gate-on-what-is-free-and-deterministic.md)

---

## The human review queue

```bash
python scripts/review_queue.py build --all
python scripts/review_queue.py list
python scripts/review_queue.py decide PCP-0005-INJE-00 --accept --by "V Kota"
python scripts/review_queue.py status     # exit 7 while anything is held
```

Three things were previously reported and then went nowhere: **reconstruction
candidates**, **conservative recommendations** (the memo says DEFER where the
metrics permit PROCEED, because pro forma leverage would breach policy), and
**injection findings**. Each is a judgement the pipeline correctly declines to
make.

Across twelve deals the queue produced **one** item. Clean memos queue nothing —
a queue that fires on every memo is a queue nobody reads.

**Only findings of wrongness block release.** A broken accounting identity holds
the memo; a reconstruction candidate does not. Decisions are **append-only,
attributed and timestamped** — a credit file is a regulated artefact and "who
approved this" is asked years later. **Nothing expires**: an item that quietly
released itself after a week would turn the queue from a control into a delay.

Detail: [docs/review-queue.md](docs/review-queue.md) · [ADR 0012](docs/decisions/0012-the-system-does-not-decide.md)

---

## What this project is actually about

Six of ten milestones ended with **the check being wrong rather than the model**:

- The grounding metric called correct arithmetic a hallucination
- The fix for that approved a NAICS code as a financial derivation — **and passed
  12 memos out of 12**
- The policy check computed faithfully on poisoned figures
- The guardrail blocked nine memos of nine for being *more* careful than the rule
- The red-team scorer counted resistance as a breach

The most instructive is the second, because it **passed**. A verifier that
approves everything looks identical to a correct one at the summary line; the
only way to catch it is to read what it approved.

And the single best moment: Claude wrote *"the requested facility would increase
total debt to $21.3 million, raising pro forma leverage to 3.95x — above the 3.5x
policy threshold."* That sentence flips a recommendation. The first verifier
called it a hallucination.

**The fix was never to make the model dumber. It was to make the check smart
enough to recognise reasoning, and honest enough to admit when it could not.**

---

## Honest limitations

- **Packages are 15–20 pages, not the 180–600 a real file runs to.** The
  retrieval problem is real but not full-scale; latency figures from this corpus
  should be read with that in mind.
- **Scans are cleaner than photocopies** — noise, skew and JPEG artefacts, but no
  coffee stains, staples, handwriting or fax compression.
- **Prose is templated**, so a model can learn phrasing rather than reading. The
  gold set is weighted towards figures for that reason.
- **Instruction detection is pattern-based** and scores 0% on obfuscation and
  data poisoning. Reported rather than rounded up.
- **The extractor matches statement rows by exact label.** A borrower writing
  "Turnover" instead of "Revenue" yields nothing.
- **The review queue is a CLI over JSONL** — the right shape for demonstrating
  the control, the wrong shape for a lending desk.
- **Nothing learns from review decisions.** Auto-accepting on that history would
  be useful and would quietly convert reviewed judgements into unreviewed ones.

---

## Architecture decisions

| | |
|---|---|
| [0001–0003](docs/decisions/) | scaffold, config, synthetic corpus |
| [0004](docs/decisions/0004-planted-defects-as-an-evaluation-instrument.md) | planted defects as an evaluation instrument |
| [0005](docs/decisions/0005-page-level-ingestion-and-provenance.md) | the page is the unit of ingestion and provenance |
| [0006](docs/decisions/0006-chunks-carry-document-status.md) | chunks carry document status; tables stay tables |
| [0007](docs/decisions/0007-rank-fusion-and-authority-tiebreak.md) | rank fusion, with authority as the tie-break |
| [0008](docs/decisions/0008-mechanical-metrics-over-judged-metrics.md) | mechanical metrics first; a judge only where nothing else works |
| [0009](docs/decisions/0009-arithmetic-leaves-the-model.md) | arithmetic leaves the model; every figure traceable |
| [0010](docs/decisions/0010-the-defence-that-does-not-run-through-the-model.md) | the defence that survives does not run through the model |
| [0011](docs/decisions/0011-gate-on-what-is-free-and-deterministic.md) | gate on what is free and deterministic; meter what is not |
| [0012](docs/decisions/0012-the-system-does-not-decide.md) | the system does not decide; it makes the decision reviewable |

## Stack

Python 3.12 · Tesseract 5 · pdfplumber · PyMuPDF · ReportLab · scikit-learn ·
LangGraph · Anthropic SDK · pytest · ruff · GitHub Actions

All borrower data is synthetic. No real financial information appears anywhere in
this repository.
