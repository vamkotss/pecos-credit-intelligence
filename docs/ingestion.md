# Ingestion

Turns the loan packages into page-level extractions that everything downstream
reads. Reasoning behind the design: [ADR 0005](decisions/0005-page-level-ingestion-and-provenance.md).

## Running it

```bash
python scripts/ingest_corpus.py                  # everything
python scripts/ingest_corpus.py --limit 3        # first three deals
python scripts/ingest_corpus.py --deals PCP-0002 # one deal
```

Reads `data/raw/packages/<deal_id>/*.pdf`, writes
`data/interim/extractions/<deal_id>.jsonl` plus an `ingest_summary.json`.

Roughly 20 seconds a deal, almost all of it OCR. A full twelve-deal pass takes
about four minutes.

## Requirements

**Tesseract 5.x must be on `PATH`.** It is the only external binary the project
needs. Poppler is not required — page rendering is done by PyMuPDF and digital
text extraction by pdfplumber, neither of which shells out to it.

## What happens to a page

```
                    ┌─────────────────────────┐
                    │  does the page have a   │
                    │  text layer? (≥40 chars)│
                    └───────────┬─────────────┘
                   yes │                 │ no
        ┌──────────────▼──────┐   ┌──────▼──────────────────┐
        │ pdfplumber          │   │ render at 200 dpi       │
        │  · extract_text     │   │ propose rotation (OSD)  │
        │  · ruling-line      │   │ OCR proposal AND 0°     │
        │    tables, else     │   │ keep the better score   │
        │    layout clustering│   │ layout-cluster tables   │
        └──────────────┬──────┘   └──────┬──────────────────┘
                       └──────┬──────────┘
                       ┌──────▼──────────┐
                       │ detect units    │
                       │ note → scale    │
                       │ write page rec  │
                       └─────────────────┘
```

Routing is **per page, not per document**. A real package mixes clean exports
with appended scanned signature sheets, and classifying the whole file would
send every page of it down one wrong path.

## Record schema

One JSON object per page:

| Field | Meaning |
|---|---|
| `deal_id`, `document`, `page_number` | The citation. Page numbers are 1-based and match the M2 manifest |
| `method` | `digital` or `ocr` |
| `rotation_applied` | Degrees clockwise: 0, 90, 180 or 270 |
| `text` | Extracted text, reassembled into reading-order lines |
| `word_count` | Words recovered |
| `mean_word_confidence` | OCR confidence, `null` on digital pages |
| `tables[]` | `{rows, source}` where source is `pdfplumber_lines` or `layout_clustering` |
| `scale_factor` | 1, 1,000 or 1,000,000 |
| `scale_evidence` | The units note that was matched, or `null` |
| `extractor_version` | Bumped when behaviour changes, so a half-reingested directory is detectable |

## How the three defects are handled

**`rotated_scanned_page`.** Orientation is corrected before reading. This is not
optional cosmetic work: read as it sits, the page returns strings like
`") S89U9INDDO Spun} JUaIONJNSU]"` — noise that would be embedded and retrieved
as though it meant something. A test OCRs the same page with and without
correction and asserts the uncorrected read is unusable, so the step cannot
silently stop mattering.

Tesseract's own orientation detection is used as a proposal, never accepted on
faith. Its confidence ran from 0.4 to 9.4 on pages where the answer was correct.
The verification signal is **line span**: upright, a line runs across the page;
sideways, each line is one stacked word. Measured on the rotated page, a median
span of 1,344 pixels upright against 19 sideways.

**`units_in_thousands`.** Detected at ingestion by pattern-matching the units
note, with both the multiplier and the matched text stored on the page. Of all
the failure modes in this corpus this is the one most likely to reach a credit
committee undetected — no confidence score flags it, the figure does not look
wrong, and it is off by three orders of magnitude.

One honest caveat, recorded in the test that proves it works: the document
prints 32,041 where the true figure is 32,041,248. The preparer rounded to
thousands, so the original precision is not on the page and no extraction method
could return it. What is recoverable is the magnitude, and that is what the test
asserts — within 0.1%, against a naive read that is wrong by 99.9%.

**`table_only_fact`.** Tables are extracted as structure rather than flattened
into running text, so the customer concentration percentage comes back as a cell
in the same row as the customer it belongs to. It appears in no sentence
anywhere in the corpus.

## Measured behaviour

On the 7-deal corpus, ingesting three deals:

| | |
|---|---|
| Pages | 46 (25 digital, 21 OCR) |
| Tables recovered | 44 |
| Mean OCR confidence | 94.9 |
| Rotated pages corrected | 1 |
| Pages rescaled | 1 |
| Empty pages | 0 |

Every gold fact whose source is a digital page is asserted to appear, exactly as
formatted, in the text extracted from the page the manifest cites.

## Limitations

- **Not incremental.** Re-running redoes all the OCR. Deliberate for now: a
  stale extraction cache is a nastier bug than a slow rebuild. Content-hash
  keying is the obvious later improvement.
- **Rotation is 90-degree steps only.** Skew of a degree or two is tolerated by
  the line grouping rather than removed. No deskew is applied.
- **Column gaps use one global threshold**, a fixed fraction of page width. A
  table with unusually tight columns will be misread. Such tables are labelled
  `layout_clustering` so the uncertainty is visible.
- **No handwriting, no stamps, no signatures.** Nothing in the corpus has them,
  and nothing here would read them.
- **Reading order is single-column.** A genuine two-column page would interleave.
