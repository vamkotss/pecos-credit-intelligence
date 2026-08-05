# ADR 0006 — Chunks carry document status, and tables stay tables

Status: Accepted
Date: 2026-08-05
Relates to: ADR 0005 (page-level ingestion and provenance)

## Context

M3 produced page records. Retrieval needs smaller units. The obvious approach —
split page text every N characters with some overlap — is what most tutorials
do, and it fails this corpus in two specific, predictable ways.

**The near-duplicate problem.** One deal carries a DRAFT of its financial
statements sitting beside the final version. The two documents are near-identical:
same borrower, same layout, same wording, one figure changed. Measured on the
generated corpus, the income statement chunks are **94% textually similar**.

Chunks that similar are indistinguishable to an embedding model, because they
genuinely are almost the same text. Cosine similarity cannot separate them. No
reranker tuning helps, no query rewriting helps, and no amount of prompt
engineering helps, because the signal is not in the text being compared.

What separates them is **status**: one document is final, one is a draft. That
fact lives at the document level. Naive chunking emits bare strings and drops it
at the boundary, and once dropped it is unrecoverable.

The same problem, in a different costume, is the restatement defect: superseded
statements that disagree with the comparative set about the same fiscal year.

**The table problem.** The customer concentration percentage appears in no
sentence anywhere in the corpus — it exists only inside a table. Flatten that
table into running prose and the figure loses its row label. `34.2` sitting in a
stream of other numbers is not retrievable by a question asking which customer
owes the most, however good the retriever is.

## Decision

### The chunk carries document status and an authority rank

Every chunk records `doc_kind`, `doc_status`, `authority` and `source_trust`
alongside its text.

Authority is derived from status: **final (3) > superseded (2) > draft (1)**. A
superseded document outranks a draft because it was at least issued to a third
party at the time; a draft never was.

`source_trust` answers a different question — who produced this — because a
lender weighs a bank-issued statement differently from a borrower's own
spreadsheet, and differently again from a broker's cover note. Values are
`bank_issued`, `tax_filing`, `borrower_prepared` and `third_party`. The last of
these is groundwork for M8: the prompt-injection payload arrives inside a
document the broker wrote, and marking that before it is needed is cheaper than
retrofitting it.

Status is also written into the **context header** prepended before embedding,
not only into the metadata. That makes the distinction available to similarity
search as well as to a hard filter, so a query mentioning "the draft" can find
it rather than depending entirely on a downstream filter clause.

### Filename classifies, page text can downgrade

Filenames give the first guess. They are perfectly regular in this generated
corpus and will not be in a real one, where a package arrives as whatever the
accountant happened to save — `Statements FINAL v3 (2).pdf` is not a schema.

So the page text votes, and it can only ever **downgrade** status. A page stamped
DRAFT is a draft regardless of the filename. Nothing in the text can promote a
document to final, because a draft that merely fails to say so is still a draft.
The safe error is to under-trust.

Status is resolved once per document by pooling all its pages, then applied to
every chunk from it. A DRAFT stamp usually appears only on page 1, and page 2 of
a draft must not be treated as final because the stamp was not repeated.

### Tables are chunked as tables

Tables are emitted as their own chunks, serialised as pipe rows. A row is never
split. When a table is too large for one chunk, it is split between rows and the
header row is repeated at the top of every group — otherwise the second half is a
wall of unlabelled numbers, which retrieves on the figures and then cannot say
what they measure.

Pipe delimiters rather than a bare join, because the delimiter is what preserves
the label-to-figure association after the text has passed through an embedding
model and come back out in a prompt.

### Prose splits only on line boundaries, and table content is removed from it

A financial statement line is an atomic fact — `EBITDA 2,418,000`. A chunker that
splits mid-line to hit a character target can leave the label in one chunk and
the figure in another; both then retrieve badly and neither answers anything.

Lines that reproduce a table row are dropped from the prose stream. The extracted
page text already contains every figure the tables hold, so emitting both
unfiltered would index the same numbers twice — once with structure, once
without — and the unstructured copy would compete with the good one in retrieval.
What remains is headings, notes and narrative: the content tables genuinely do
not carry.

### Sizes are in characters, not tokens

Target 900, ceiling 1,400, overlap 150. A tokeniser would pin this module to one
model family and add a dependency for a value that only has to be roughly right.
English financial prose runs about four characters per token, so the target is
broadly 220 tokens.

The minimum-length rule applies **to prose only**. An early version applied it to
both and silently dropped a two-row table rendering to 39 characters — precisely
the content the `table_only_fact` defect exists to punish losing. A short table
is a structured fact; a short line of prose usually is not. There is now a
regression test.

### Chunking never reads the manifest

`chunking.py` sees page records and nothing else. A chunker that consulted the
answer key would measure nothing.

Scoring lives in a separate module, `chunk_audit.py`, which reports the
**containment rate**: the fraction of extractive gold facts that survive into a
chunk anchored to the page the manifest cites. That number is the ceiling on
everything downstream — a figure that does not survive chunking cannot be
retrieved by any retriever, rescued by any reranker, or cited by any agent.
Currently **100%**.

Three categories are excluded from the denominator, and each exclusion is
counted rather than silently dropped, because unprincipled exclusions would make
a 100% score meaningless:

- **Derived metrics.** Leverage and DSCR are never printed on any page; they are
  computed at M7 from figures that are.
- **Behavioural facts.** The injection and unanswerable cases are scored on
  refusal, not on a string appearing anywhere.
- **Rescaled pages.** The tax return prints 32,041 where the true figure is
  32,041,248. The preparer rounded, so the exact figure is not on the page and
  never could be. The audit compares against the printed form.

## Consequences

**Good.** M5 can rank on authority when text similarity ties, which is the only
mechanism that resolves the draft and restatement defects.

**Good.** The retrieval ceiling is a measured number rather than an assumption.
When retrieval underperforms at M5, containment says immediately whether the
fault is upstream, which removes an entire class of misdirected debugging.

**Good.** Citations survive every downstream design change, because chunks anchor
to pages and the page is stable in a way chunk boundaries are not.

**Cost.** Metadata makes chunks larger on disk and duplicates document-level
fields across every chunk from a document. At this corpus size that is
irrelevant; at a million chunks it would argue for a separate document table and
a join.

**Cost.** Section headings come from a curated list of the headings a financial
package actually uses. A general "looks like a title" heuristic was rejected
because it fires on every short line, and a table of figures is full of short
lines — but the curated list will miss headings in document types not yet seen.

**Accepted limitation.** Table-to-heading association is approximate: a table is
assigned the last heading appearing before its first cell in the page text.
Table bounding boxes were not carried through M3, so exact positional matching is
not available. Correct on this corpus, and it would misattribute a table that
appears above the heading it belongs to.

**Accepted limitation.** Deduplication between prose and tables matches on
normalised row text. A prose line that differs slightly from its table row will
survive in both streams. This produces duplication, not loss, which is the right
direction to fail.

## Alternatives considered

**Semantic chunking** — split on embedding-similarity troughs rather than
structure. Rejected as the wrong tool here. Financial statements are already
explicitly structured, with sections and tables that mark their own boundaries
far more reliably than a similarity curve would infer them. Semantic chunking
earns its cost on unstructured narrative, which this corpus has little of.

**One chunk per page.** Simplest possible, and appealing given that the page is
already the provenance unit. Rejected because a statements page runs to roughly
2,500 characters of dense figures; retrieving it returns a large block where
most of the content is irrelevant to the query, which wastes context budget and
dilutes reranking signal.

**Sentence-window retrieval** — index sentences, return neighbours. Rejected
because financial statements are not sentences. A line of a balance sheet has no
sentence structure at all, and the technique would fragment exactly the content
that matters most.

**Dropping status and filtering at query time by filename.** Would work on this
corpus and nowhere else, since it depends on the filename convention holding.
It also cannot express partial trust — a superseded document should be ranked
down, not excluded, because it is still the right source for questions about
what was originally reported.
