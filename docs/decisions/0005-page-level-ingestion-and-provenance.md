# ADR 0005 — The page is the unit of ingestion and of provenance

Status: Accepted
Date: 2026-08-05
Relates to: ADR 0004 (planted defects as an evaluation instrument)

## Context

A loan package arrives as a folder of PDFs of wildly uneven quality. Some were
exported from accounting software and carry a clean text layer. Some were
printed, signed, scanned and emailed back, and contain no text at all — only
pixels. Everything downstream of ingestion assumes it is working with text, and
this is the boundary where that assumption is made true.

Three questions had to be settled before writing any of it.

**What granularity gets classified?** Document, page, or region.

**What granularity carries provenance?** A credit memo that states an EBITDA
figure must be able to say where the figure came from, and a retrieval eval
scored on recall@k needs a target to score against.

**How much do we trust the OCR engine's own judgement?**

## Decision

### Route per page, not per document

Each page is inspected for a text layer independently and routed on its own
merits. A page with fewer than 40 characters of embedded text is treated as
scanned.

Document-level classification was rejected because real packages are mixed: an
accountant exports ten clean pages and then appends two scanned signature
sheets. Classifying the file would send every page of it down one wrong path.

The threshold is 40 rather than zero because image-only PDFs sometimes carry a
stray character from a stamp or a header, and a strict `== 0` test would route
those pages to the digital extractor, which would return almost nothing and
report success.

### The page is the unit of provenance

Not the document, not the chunk.

A document is too coarse to cite. "It's in the financial statements" is not an
answer a credit analyst accepts.

A chunk is too fine and too unstable. Chunk boundaries change whenever the
chunking strategy changes, which it will at M4, and every stored citation would
break with them.

The page is what a human points at, it is stable across every downstream design
change, and the ground-truth manifest already records it — so extraction and
evaluation speak the same language without a translation layer.

Every page record carries the deal, the document, the page number, the method
that read it, the rotation applied, the OCR confidence, and the extractor
version. The version field exists so a half-reingested interim directory is
detectable rather than silently producing inconsistent evals.

### One layout algorithm, two sources of word boxes

Tables are recovered by clustering word boxes on whitespace: words on the same
line are grouped, a horizontal gap beyond a threshold is a column boundary, and
a run of consecutive multi-column lines is a table. The identical algorithm runs
on Tesseract's word boxes and on pdfplumber's.

Ruling-line extraction is tried first on digital pages and is close to exact
when a table has real borders. Borrower-prepared statements usually have none —
columns are separated by whitespace and nothing else — and the ruling-line
strategy finds zero tables on them, which is what made the fallback necessary.

pdfplumber's own text-based table strategy was tried and rejected: on this
corpus it split words mid-token, turning `TRINITY BEND` into cells reading
`RINIT` and `Y BEND`.

Output records which method found each table (`pdfplumber_lines` versus
`layout_clustering`), because inferred structure is not as reliable as read
structure and downstream code that treats them as equivalent will be wrong about
scanned tables.

### Orientation: propose, then compare — never merely accept

Tesseract's orientation detection is used as a **proposal**, evaluated against a
half-scale render because a four-way choice needs no detail. It is not trusted
on its own. Its confidence score ran from 0.4 to 9.4 on pages of this corpus
where the answer was correct, so the score carries almost no information, and
detection proved sensitive to scan noise — the same page with a different noise
pattern was sometimes detected and sometimes not.

The proposal is verified by OCR-ing at that orientation and scoring the result.
The upright reading is **always** computed alongside it, and the better of the
two wins. Only if both score poorly are the remaining orientations tried.

The scoring function is the substantive part. Mean OCR confidence is not
sufficient: Tesseract reads a sideways page as a column of single words and
reports roughly 95% confidence on each, because it really did recognise those
glyphs — it simply has no idea they were meant to be read across the page rather
than down it. Confidence answers "did I read these characters correctly", not
"was this the right way to read the page".

The signal that separates the cases is **how far a line runs**. Upright, a line
spans most of the page width. Sideways, each line is one stacked word spanning
almost nothing. Measured on the rotated bank statement page: a median line span
of 1,344 pixels upright against 19 sideways. Score is mean confidence multiplied
by median line span as a fraction of page width, giving roughly 50 for a good
page and under 1 for a bad one.

## Consequences

**Good.** The `rotated_scanned_page` defect is defused, and the fix is tested by
OCR-ing the same page with and without correction and asserting the uncorrected
read is unusable. If the correction ever silently stops mattering, that test
fails.

**Good.** The `units_in_thousands` defect is detected at ingestion rather than
being left for a model to notice. The multiplier and the matched note both
travel with the page, so a reviewer can check the call instead of trusting it.

**Good.** The `table_only_fact` defect survives, because tables are extracted as
structure rather than flattened into prose.

**Cost.** OCR is slow — roughly two seconds a scanned page, so a full pass over
twelve deals takes several minutes. Ingestion is not incremental; re-runs redo
the work. That is a deliberate trade for now: a stale extraction cache is a
nastier bug than a slow rebuild, and the corpus is small enough to afford it.
Incremental ingestion keyed on a content hash is the obvious later improvement.

**Cost.** Layout-clustered tables are approximate. The threshold for a column
gap is a single global fraction of page width, which will misread a table whose
columns are unusually tight. This is labelled in the output rather than hidden.

**Accepted limitation.** Orientation is only ever corrected in 90-degree steps.
Skew of a degree or two is tolerated by the line grouping rather than removed;
no deskew is applied. That was sufficient for this corpus and would not be for
genuinely poor photocopies.

## Alternatives considered

**A hosted document-AI service** (Textract, Document AI, Azure Document
Intelligence). Better OCR and far better table recovery than this. Rejected for
three reasons: it puts a paid API in the path of every test run, it makes the
project unreproducible for anyone without an account, and it moves the
interesting engineering — routing, orientation, units, provenance — inside a
vendor's black box, where it cannot be discussed or defended.

**A layout model** such as LayoutLM or a table-transformer. Substantially better
on complex tables. Rejected as premature: it adds a model dependency and a GPU
expectation before there is any evidence that whitespace clustering is the
bottleneck. The defect-level eval breakdown at M6 will say whether it is, and
that is the right point to decide.

**OCR everything, skip the router.** Simpler, and strictly worse. OCR introduces
recognition errors that a clean text layer does not have, and it would be roughly
twenty times slower on the digital pages that make up more than half the corpus.
