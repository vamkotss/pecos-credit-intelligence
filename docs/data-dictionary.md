# Data dictionary

Populated in full at M2, once the generator defines the corpus schema. The
tables below are the contract the generator must satisfy.

## `loan_packages` — one row per credit request

| Field | Type | Description |
|---|---|---|
| `package_id` | str | Stable ID, e.g. `PKG-0007` |
| `borrower_name` | str | Legal entity name as it appears on the primary statement |
| `borrower_aliases` | list[str] | Variant spellings planted across documents |
| `industry` | str | NAICS-style sector label |
| `request_amount_usd` | int | Loan amount requested |
| `collateral_type` | str | `CRE`, `ABL`, or `CASHFLOW` |
| `fiscal_year_end` | date | Drives which year a figure belongs to |
| `n_documents` | int | Documents in the package |
| `n_pages` | int | Total pages |

## `documents` — one row per PDF

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Stable ID |
| `package_id` | str | FK to `loan_packages` |
| `doc_type` | str | `INCOME_STMT`, `BALANCE_SHEET`, `TAX_RETURN`, `RENT_ROLL`, `LOAN_AGREEMENT`, `APPRAISAL`, `DEBT_SCHEDULE`, `PFS` |
| `as_of_date` | date | Statement date |
| `fiscal_year` | int | Year the document primarily covers |
| `is_restatement` | bool | Whether it restates a prior year |
| `layout_family` | str | Which of the synthetic layout templates was used |
| `scan_dpi` | int | 200, 240, or 300 |
| `has_text_layer` | bool | False for scanned documents — these require OCR |

## `ground_truth_figures` — the manifest that makes evaluation possible

One row per financial figure planted in the corpus. This is the artefact the
entire evaluation strategy rests on.

| Field | Type | Description |
|---|---|---|
| `figure_id` | str | Stable ID |
| `doc_id` | str | FK to `documents` |
| `page_number` | int | 1-indexed page the figure appears on |
| `bbox` | tuple | Pixel bounding box at the rendered DPI |
| `label` | str | Canonical account label, e.g. `TOTAL_REVENUE` |
| `label_as_printed` | str | The wording actually on the page |
| `fiscal_year` | int | Which year column the figure sits in |
| `value` | Decimal | The true value, in units |
| `printed_units` | str | `UNITS` or `THOUSANDS` — the planted units-drift trap |
| `printed_sign_style` | str | `MINUS` or `PARENTHESES` |
| `defect_classes` | list[str] | e.g. `["ROTATED_180", "SPANS_PAGES"]` |

## `defect_classes` — the planted difficulty catalogue

| Code | Description | Target rate |
|---|---|---|
| `NO_TEXT_LAYER` | Pure scan, OCR required | 0.60 |
| `ROTATED_90` / `ROTATED_180` | Page orientation wrong | 0.08 |
| `SPANS_PAGES` | Table continues across a page break | 0.15 |
| `MULTI_COLUMN` | Two-column layout | 0.12 |
| `RESTATED` | Prior-year figure restated elsewhere | 0.10 |
| `UNITS_THOUSANDS` | Page reports in thousands | 0.20 |
| `PAREN_NEGATIVE` | Negatives in parentheses | 0.25 |
| `HANDWRITTEN_MARK` | Annotation or stamp over text | 0.10 |
| `ALIAS_ENTITY` | Borrower named inconsistently | 0.30 |
| `INJECTION_PAYLOAD` | Instruction text embedded in the document | 0.02 |
