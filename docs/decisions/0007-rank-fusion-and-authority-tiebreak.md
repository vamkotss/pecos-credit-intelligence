# ADR 0007 — Rank fusion over lexical and dense, with authority as the tie-break

Status: Accepted
Date: 2026-08-05
Relates to: ADR 0006 (chunks carry document status)

## Context

M4 established the retrieval ceiling: every extractive gold fact survives into a
chunk anchored to the right page, so 100% recall is achievable in principle.
This milestone is about actually reaching it, and about the ranking quality that
determines what the M7 agent will have in its context window.

Three things had to be settled.

**Which retrieval method.** The queries split into two kinds that no single
method handles. *Exact-figure questions* — "what was total interest-bearing debt
at FY2025 year end" — have an answer that is a specific string on a specific
page. *Paraphrased questions* — "how exposed is the borrower to a single
customer" — hit a page that says "concentration" and "% of AR" and never uses
the word "exposed".

**How to combine methods** whose scores are not comparable.

**How to resolve the near-duplicate defect**, where the DRAFT and final
statements are 94% textually identical and the wrong one sits directly behind
the right one at a margin of 1.6%.

## Decision

### BM25 and dense both run; ranks are fused, not scores

Neither is a fallback for the other. They fail on disjoint query types, so both
run over the same per-deal index and their **rankings** are combined by
reciprocal rank fusion.

Fusing ranks rather than scores is the substantive choice. BM25 is unbounded and
corpus-dependent; cosine similarity is bounded in [-1, 1]. Normalising them into
a shared range means picking a normalisation, and every such choice is a hidden
weighting decision that then needs tuning and defending. Rank position sidesteps
it: a document both methods like outranks one only a single method loves, and
that is the property fusion exists to produce.

RRF's constant is left at 60, the value from the original TREC work. It is
deliberately large, which flattens the gap between ranks 1 and 2 so that
agreement between retrievers matters more than either one's internal confidence.

### Indexes are built per deal

A lending question is always about one borrower; there is no such thing as "what
was revenue" across a portfolio. Cross-deal results are never useful and would
be a confidentiality failure in a real system.

Scoping at index time rather than filtering afterwards also means IDF statistics
reflect one package's vocabulary, which is what makes a term like a specific
lender's name discriminating rather than common.

### The tokeniser preserves whole financial figures

`\w+` turns `32,041,248` into `32`, `041`, `248`. Every large figure then
collides with every other figure sharing a fragment, and BM25 — whose entire
value here is matching exact amounts — silently loses its strongest signal.
Numbers are matched first as whole units including separators, and each numeric
token is also emitted stripped to bare digits so a query written either way
matches a page printed either way.

### Query stopwords are removed, and the reason is an inversion

Inverse document frequency assumes a natural-language corpus, where function
words are common and therefore uninformative. **A loan package is not that
corpus.** It is mostly tables of figures, so ordinary English words are *rare*
in it. Measured on one deal, `what` and `was` scored an IDF of 3.12 while
`ebitda` scored 2.61 — IDF concluded that "what" was the most informative term
in "What was EBITDA in FY2025?".

The symptom was the broker's cover note, the only chunk written in flowing
prose, ranking first for almost every financial question. It contains no figures
at all; it simply contains English. Stripping stopwords from queries took
recall@1 from 13.2% to 39.5%.

Stopwords are removed from queries only. Leaving them in the document index
costs nothing — an unmatched term contributes zero — while removing them from
documents would change the length normalisation BM25 depends on.

### Ranks are counted over distinct pages

A hit is a page, so recall@5 means "the answer's page is among the first five
*pages* retrieved". Counting chunks would let a page contributing both a table
chunk and a prose chunk burn two slots and make a good retriever look worse. It
is also the quantity that matters downstream: at M7 the question is how many
distinct pages the agent must read, because pages consume its context budget.
This change alone took recall@5 from 98.7% to 100%.

### Authority is applied after reranking, as a multiplicative demotion

Not a filter. Excluding non-final documents would make "what did the draft say"
and "what was originally reported before the restatement" unanswerable, and both
are legitimate analyst questions. A multiplier lets a strong text match on the
draft still win when the query is genuinely about the draft, while the final
version wins every tie — and with a near-duplicate, every comparison is a tie.

Weights: final 1.0, superseded 0.82, draft 0.75.

### Embedding and reranking sit behind protocols

Defaults (`LsaEmbedder`, `LexicalOverlapReranker`) need no model download and no
network, which keeps the test suite hermetic and CI free of a torch install.
`SentenceTransformerEmbedder` and `CrossEncoderReranker` import lazily and are
selected by flag, so the stronger choice is measurable on the same gold set
rather than assumed.

## Measured results

Full 7-deal corpus, 221 chunks, 76 scored queries:

| configuration | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|
| BM25 only | 46.1% | 80.3% | 97.4% | 0.669 |
| Dense (LSA) only | 15.8% | 38.2% | 72.4% | 0.366 |
| BM25 + dense | 46.1% | 80.3% | 97.4% | 0.667 |
| BM25 + rerank | 50.0% | 81.6% | 100% | **0.687** |
| Dense + rerank | 34.2% | 72.4% | 100% | 0.567 |
| **Full stack** | **50.0%** | **81.6%** | **100%** | **0.675** |

Five of six defects retrieve at rank 1. `units_in_thousands` sits at rank 3.

## Consequences

**Good.** Recall@5 is total, so the M7 agent will always have the answer's page
in context if it reads five pages.

**Good.** The per-stage scores on every result answer "why did this rank here",
which is the question that comes up every time retrieval underperforms.

**Honest cost — dense retrieval is not currently earning its place.** A sweep of
the fusion weight showed 0.0 is optimal on this eval set; every increment hurt.
The default is 0.15: within noise of the best, and enough to keep the hybrid
genuinely operative.

The reason it underperforms is a property of the *eval set*, not necessarily of
the method. The gold questions are generated from templates and share vocabulary
with the documents they target, which structurally favours lexical matching.
Real analyst questions paraphrase far more. Rather than delete the dense path on
evidence that is known to be biased, it is kept at a low weight and the flag
exists to re-measure with a trained encoder.

**Honest cost — the authority weighting does not move recall on the gold set.**
The gold question for the near-duplicate defect says "final", which does the work
lexically. Its value shows on a neutral query: for "What was EBITDA in FY2025?"
the draft trails the final by 1.6% without weighting — noise — and leaves the
top three entirely with it. What that buys is not a recall point; it is keeping a
contradictory document out of the agent's context at M7.

**Accepted limitation.** Remaining rank-1 failures are almost all balance-sheet
questions losing to the cash flow statement, which legitimately mentions cash,
debt and year-end. A lexical reranker cannot resolve that; it is the case a
cross-encoder exists for, and it is unmeasured here.

**Accepted limitation.** Derived-metric facts (leverage, DSCR) are scored against
the page whose figures you would compute from. Other statement pages are
defensible answers, so their recall@1 understates real performance.

**Accepted limitation.** Indexes are rebuilt in memory on every run — about 1.4
seconds for the whole corpus. Persistence is not worth building until the corpus
is large enough for the rebuild to be felt.

## Alternatives considered

**A vector database.** Rejected as premature. At a few hundred chunks per deal a
brute-force dot product takes microseconds, and an ANN index would add a
dependency, a build step and approximation error to buy nothing. That trade
flips somewhere near a hundred thousand chunks, which this corpus is three orders
of magnitude away from.

**`rank_bm25` from PyPI.** BM25 is forty lines. Writing it removes a dependency
and makes the scoring the part of this milestone most worth being able to explain
line by line.

**Score normalisation instead of rank fusion.** Requires choosing a
normalisation, which is a hidden weighting decision. See above.

**Hard-filtering non-final documents.** Would work on the near-duplicate defect
and break legitimate questions about what a draft or a superseded statement said.
It also cannot express partial trust, which is exactly what is wanted.

**Query expansion with an LLM.** Would likely help the paraphrase cases and adds
a paid API call plus latency to every query, before there is evidence that
paraphrase is the bottleneck. The defect breakdown at M6 is the right place to
decide that.
