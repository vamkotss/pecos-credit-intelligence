# The Pecos synthetic corpus

Every document in this project is generated. Nothing here is real borrower data.

## Why generated

Commercial credit files are confidential, and no public corpus of borrower
financial packages exists with page-level answer keys. Without an answer key
there is no recall@k, no exact-match extraction score, and no way to prove a
credit memo's figures came from the documents rather than from the model.

So the documents and their labels are produced together, from the same numbers.
The manifest is not extracted from the PDFs — the PDFs are rendered from the
manifest's source data. That is what makes the ground truth trustworthy.

## Regenerating

```bash
python scripts/generate_corpus.py              # 12 deals, full render
python scripts/generate_corpus.py --deals 3    # quick
python scripts/generate_corpus.py --no-pdfs    # manifest and gold set only
```

The PDFs are gitignored. They are fully reproducible from `PC_SEED`, and
committing hundreds of megabytes of synthetic scans would be a poor use of a
portfolio repository. The manifest and the gold set **are** committed, because
they are small and they are the contract every later milestone codes against.

## What a package contains

| File | Kind | Notes |
|---|---|---|
| `01_loan_application.pdf` | digital | Request amount, use of proceeds, borrower profile |
| `02_financial_statements_comparative.pdf` | digital | 3 pages: income statement, balance sheet, cash flow |
| `03_financial_statements_superseded.pdf` | digital | Only when the restatement defect is assigned |
| `04_debt_schedule.pdf` | digital | Facility, lender, rate, maturity, collateral |
| `05_ar_aging_and_concentration.pdf` | digital | Ageing buckets and named customers |
| `06_borrower_questionnaire.pdf` | digital | Prose answers: ownership, litigation, related parties |
| `07_broker_email_thread.pdf` | digital | Cover note; carries the injection when assigned |
| `08_bank_statements.pdf` | **scanned** | 6 pages, image-only, no text layer |
| `09_tax_return_extract.pdf` | **scanned** | Form 1120 extract, image-only |
| `10_financial_statements_draft.pdf` | digital | Only when the near-duplicate defect is assigned |

**Digital** documents carry a real text layer and can be parsed directly.
**Scanned** documents are rasterised at 165 dpi, converted to greyscale, given
Gaussian sensor noise and a fraction of a degree of skew, re-encoded through
JPEG, and re-wrapped as image-only PDFs. Nothing but OCR will read them. A test
asserts they contain zero extractable text, so the OCR path at M3 cannot
silently stop being exercised.

## The financial model

Every figure is an integer number of whole dollars. Floats are never used for
money, so the accounting identities can be asserted exactly rather than to a
tolerance.

Construction order, which is the part that makes it hold together:

1. Operating performance is chosen first — revenue, gross margin, EBITDA margin.
   These do not depend on how the business is financed.
2. The debt schedule is sized as a multiple of year-0 EBITDA, which yields
   interest expense. Revolving lines do not amortise; term loans and mortgages
   do.
3. The income statement is completed using that interest expense.
4. The balance sheet is built with **cash as the balancing plug**. Receivables
   come from days sales outstanding, inventory from days inventory outstanding,
   payables from days payable outstanding, PP&E from a capex-and-depreciation
   roll-forward, retained earnings from a profit-and-distributions roll-forward.
   Cash is whatever makes assets equal liabilities plus equity.
5. The cash-flow statement is derived. Because cash was the plug, the change in
   cash provably equals CFO + CFI + CFF.

Two guards keep the borrowers underwritable rather than merely arithmetically
valid. Distributions are trimmed if paying them would drive cash below roughly
two weeks of revenue. If a year still runs short, the whole path is rebuilt with
opening liquidity raised by exactly the shortfall — economically, "this borrower
started with a bigger cushion", which is honest, rather than flooring a negative
balance at zero, which is not.

Tests assert, on every deal and every year: assets equal liabilities plus equity
to the dollar; cash flow ties to the movement in cash to the dollar; balance
sheet debt reconciles to the debt schedule; equity and cash are positive;
leverage is inside a believable band.

Across a 360-deal sweep the resulting spread is roughly 1.0x to 4.9x leverage
and 0.4x to 3.6x debt service coverage, with about a third of deals falling
below Pecos's 1.25x DSCR requirement. That mix matters: a corpus where every
borrower passes gives the M7 credit-memo agent no decision to make.

## The manifest

`data/raw/corpus_manifest.json`

```
seed          the PC_SEED the corpus was built from
n_deals       how many packages
years         fiscal years per borrower
deals[]       one summary record per deal, including the derived credit metrics
facts[]       the ground-truth question set
defect_index  defect id -> the deal ids carrying it
```

Each fact:

| Field | Meaning |
|---|---|
| `fact_id` | Stable id, e.g. `PCP-0003-F007` |
| `question` | The question an analyst would ask |
| `answer_value` | Machine-comparable answer, or null for behavioural cases |
| `answer_unit` | `USD`, `x`, `percent`, `text`, `none` |
| `answer_text` | Human-readable answer |
| `fact_type` | `income_statement`, `balance_sheet`, `derived_metric`, `behavioural`, … |
| `source_document` | The retrieval target filename, or null |
| `source_page` | 1-based page, or null |
| `answerable` | **False for the planted unanswerable question** |
| `defect_tag` | Which defect this fact exercises, if any |
| `notes` | Why this fact is interesting |

`answerable` deserves attention. Scoring a correct refusal as a failure is a
common evaluation bug, and it punishes exactly the behaviour a lender wants. The
flag exists so no harness can make that mistake by accident.

`evals/datasets/qa_gold.jsonl` carries the same facts one per line, so an eval
harness can stream it and a changed question produces a one-line diff.

## Planted defects

See `docs/decisions/0004-planted-defects-as-an-evaluation-instrument.md` for the
reasoning. Seven defects, dealt round-robin so all of them appear even in the
three-deal corpus CI builds.

## Honest limitations

- **Packages are smaller than real ones.** A genuine Pecos file runs 180–600
  pages. A generated package is roughly 15–20. The retrieval problem is real but
  it is not the full-scale problem, and any latency figure quoted from this
  corpus should be read with that in mind.
- **Scans are cleaner than photocopies.** There is noise, skew and JPEG
  artefacting, but no coffee stains, staple shadows, handwriting or fax
  compression.
- **Prose is templated.** Sentence structure repeats across deals, so a model
  can learn phrasing rather than reading. Numeric facts do not have this
  problem, which is why the gold set is weighted towards figures.
- **Single fiscal-year-end.** Every borrower closes 31 December. Real portfolios
  have stub periods and off-calendar year ends.
- **The tax return is an extract, not a full return.** One page of Form 1120,
  not the schedules that come with it.

These are limitations, not defects — they are places the corpus is easier than
reality, and they are listed here so nobody has to discover them by being
surprised at M6.
