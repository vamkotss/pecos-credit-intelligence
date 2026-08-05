# Chunking

Turns page extractions into the units a retriever indexes. Reasoning:
[ADR 0006](decisions/0006-chunks-carry-document-status.md).

## Running it

```bash
python scripts/chunk_corpus.py            # chunk everything
python scripts/chunk_corpus.py --audit    # and score against ground truth
```

Reads `data/interim/extractions/*.jsonl`, writes
`data/interim/chunks/<deal_id>.jsonl` plus a `chunk_summary.json`.

Chunking itself is instant — under a second for the whole corpus. All the time
in the pipeline is OCR, upstream in M3.

## The containment audit

`--audit` reports the number that matters most: the **retrieval ceiling**.

A figure that does not survive into a chunk anchored to the right page cannot be
retrieved by any retriever, rescued by any reranker, or cited by any agent.
Retrieval work spent chasing such a fact is wasted, and the failure looks
identical to a retrieval bug — so it gets ruled out here, before any retriever
exists to blame.

Measured over three deals:

```
extractive facts   24
found in a chunk   24
containment        100.0%
excluded, derived  6   (never printed on a page)
excluded, refusal  3
defect coverage
  near_duplicate_draft     1
  table_only_fact          1
  units_in_thousands       1
```

The exclusions are counted, not silently dropped, because unprincipled
exclusions would make a 100% score meaningless. Derived metrics like leverage
are never printed anywhere; behavioural facts are scored on refusal; rescaled
pages print rounded thousands, so the audit compares against the printed form.

## Chunk schema

| Field | Meaning |
|---|---|
| `chunk_id` | `PCP-0007::02_financial_statements_comparative.pdf::p001::c000` |
| `deal_id`, `document`, `page_number` | The anchor. Same triple the manifest uses |
| `chunk_index`, `chunk_type` | Order on the page; `prose` or `table` |
| `text` | The content |
| `section` | `Statements of Income`, `Balance Sheets`, … |
| `doc_kind` | `financial_statements`, `tax_return`, `bank_statements`, … |
| `doc_status` | `final`, `superseded` or `draft` |
| `authority` | 3, 2 or 1, derived from status |
| `source_trust` | `borrower_prepared`, `bank_issued`, `tax_filing`, `third_party` |
| `extraction_method`, `mean_word_confidence` | Inherited from M3 |
| `scale_factor`, `scale_evidence` | The units multiplier and the note it came from |
| `table_source` | `pdfplumber_lines` or `layout_clustering`, tables only |
| `context_header` | One-line preamble prepended before embedding |
| `chunker_version` | Bumped when boundaries or metadata change |

## The context header

A bare chunk of figures is close to meaningless to an embedding model.
`Revenue 32,041,248` could come from any document of any vintage. The header
restores the context a human gets for free from looking at the page:

```
02_financial_statements_comparative.pdf | page 1 | financial statements | Statements of Income
10_financial_statements_draft.pdf | page 1 | financial statements | Statements of Income | DRAFT
09_tax_return_extract.pdf | page 1 | tax return | U.S. CORPORATION INCOME TAX RETURN | figures in units of 1,000
```

Non-final status and a non-unit scale factor are announced explicitly, so both
are visible to similarity search and not only to a metadata filter.

## How the defects are handled

**`near_duplicate_draft`.** The draft and the final are **94% textually
similar** — measured, not assumed, and asserted by a test so the defect cannot
quietly stop being planted. Cosine similarity cannot separate them because the
signal is not in the text. Status can: authority 3 against 1, on every chunk.

**`restated_prior_year`.** Same mechanism. The superseded issued copy is
detected from its own text (`refer to the comparative statements for restated
figures`) and ranked at authority 2, below the comparative set at 3.

**`table_only_fact`.** Tables are chunked as tables, rows never split, header
repeated when one has to be. The concentration percentage comes back in the same
row as the customer it belongs to.

**`units_in_thousands`.** The multiplier and its evidence travel onto every
chunk from the page, and into the context header. A chunk reading
`Gross receipts or sales | 32,041` with no units attached is off by three orders
of magnitude and nothing about it looks wrong.

**`prompt_injection`.** Not defended here — that is M8 — but the broker note is
marked `source_trust: third_party` now, because retrofitting provenance after
you need it is harder than carrying it forward.

## Sizes

Target 900 characters, ceiling 1,400, overlap 150. Characters rather than tokens:
a tokeniser would pin the module to one model family for a value that only has to
be roughly right. English financial prose runs about four characters per token,
so the target is broadly 220 tokens.

Observed on three deals: 93 chunks from 46 pages, 49 prose and 44 table, mean
332 characters, max 891.

## Limitations

- **Table-to-heading association is approximate.** A table is assigned the last
  heading appearing before its first cell in the page text. Bounding boxes were
  not carried through M3, so exact positional matching is unavailable. Correct on
  this corpus; would misattribute a table sitting above its heading.
- **Headings come from a curated list.** A general "looks like a title"
  heuristic fires on every short line, and a table of figures is nothing but
  short lines. The curated list will miss headings in unseen document types.
- **Prose/table deduplication matches on normalised row text.** A prose line
  differing slightly from its table row survives in both. That is duplication,
  not loss — the right direction to fail.
- **No cross-page chunks.** A table continuing across a page break becomes two
  chunks with two anchors. Correct for citation, imperfect for reading.
