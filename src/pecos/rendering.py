"""Render deals into loan-package PDFs (M2).

Two kinds of PDF come out of here, and the difference is the point.

**Digital-native PDFs** carry a real text layer. `pdfplumber` will read them
cleanly. These stand in for the documents a borrower's accountant exports
straight from their accounting software.

**Scanned PDFs** have no text layer at all. Each page is rendered, rasterised
to an image, degraded with skew and sensor noise, and re-wrapped as an
image-only PDF. Nothing but OCR will read them. These stand in for the pages
that were printed, signed, and run through a desk scanner -- which in real
middle-market lending is most of the tax returns and every bank statement.

If the corpus were entirely digital-native, the M3 OCR milestone would have
nothing to prove and the pipeline would look far more robust than it is.

DETERMINISM
-----------
`rl_config.invariant = 1` makes ReportLab stop stamping the current timestamp
and a random document id into every file. Without it, generating the same
corpus twice produces different bytes and no meaningful hash check is possible.
Image noise is drawn from a seeded NumPy generator for the same reason.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import fitz  # PyMuPDF -- used only to rasterise and to re-wrap images as PDF
import numpy as np
from PIL import Image
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pecos.corpus import (
    DEFECT_INJECTION,
    DEFECT_NEAR_DUPLICATE,
    DEFECT_RESTATEMENT,
    DEFECT_ROTATED_SCAN,
    DEFECT_THOUSANDS,
    INJECTION_PAYLOAD,
    Deal,
    YearFinancials,
)

# Must be set before any document is built.
rl_config.invariant = 1

_styles = getSampleStyleSheet()
_H1 = ParagraphStyle("PcH1", parent=_styles["Heading1"], fontSize=14, spaceAfter=10)
_H2 = ParagraphStyle("PcH2", parent=_styles["Heading2"], fontSize=11, spaceAfter=6)
_BODY = ParagraphStyle("PcBody", parent=_styles["BodyText"], fontSize=9, leading=12)
_SMALL = ParagraphStyle("PcSmall", parent=_styles["BodyText"], fontSize=7, leading=9)

_TABLE_STYLE = TableStyle(
    [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]
)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def money(n: int) -> str:
    """Format whole dollars the way a financial statement does.

    Negative numbers appear in parentheses, not with a minus sign. That is the
    accounting convention, and it is also a real parsing hazard worth planting:
    a pipeline that reads "(412,300)" as a positive number will silently invert
    a loss into a profit.
    """
    if n < 0:
        return f"({abs(n):,})"
    return f"{n:,}"


def money_k(n: int) -> str:
    """Same, but expressed in thousands. Used only by the tax-return document
    that carries the units defect."""
    return money(round(n / 1000))


def _doc(path: Path, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title=title,
        author="Pecos Capital Partners -- synthetic corpus",
        subject="Synthetic loan package. Not real financial data.",
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )


def _header(deal: Deal, doc_title: str) -> list:
    return [
        Paragraph(deal.borrower_name.upper(), _H1),
        Paragraph(doc_title, _H2),
        Paragraph(
            f"{deal.city}, {deal.state} &nbsp;|&nbsp; Deal reference {deal.deal_id} "
            f"&nbsp;|&nbsp; Prepared for Pecos Capital Partners",
            _SMALL,
        ),
        Spacer(1, 10),
    ]


# ---------------------------------------------------------------------------
# Statement tables
# ---------------------------------------------------------------------------


def _income_statement_rows(
    years: list[YearFinancials],
    scale_k: bool = False,
    ebitda_override: int | None = None,
) -> list[list[str]]:
    """Build the income statement grid.

    `ebitda_override` exists for the two defects that need one figure to differ
    from the true value: the restated prior year and the draft statements. The
    override changes the printed EBITDA and the lines that follow from it, so
    the fake document is internally consistent -- an inconsistent fake would be
    trivially detectable and would not test anything real.
    """
    fmt = money_k if scale_k else money
    head = ["", *[f"FY{y.fiscal_year}" for y in years]]

    def ebitda_of(y: YearFinancials) -> int:
        if ebitda_override is not None and y is years[-1]:
            return ebitda_override
        return y.ebitda

    def opex_of(y: YearFinancials) -> int:
        # Opex absorbs the override so gross profit minus opex still equals the
        # printed EBITDA.
        return y.gross_profit - ebitda_of(y)

    rows = [
        head,
        ["Revenue", *[fmt(y.revenue) for y in years]],
        ["Cost of goods sold", *[fmt(-y.cogs) for y in years]],
        ["Gross profit", *[fmt(y.gross_profit) for y in years]],
        ["Operating expenses", *[fmt(-opex_of(y)) for y in years]],
        ["EBITDA", *[fmt(ebitda_of(y)) for y in years]],
        ["Depreciation and amortisation", *[fmt(-y.depreciation) for y in years]],
        ["Operating income", *[fmt(ebitda_of(y) - y.depreciation) for y in years]],
        ["Interest expense", *[fmt(-y.interest_expense) for y in years]],
        [
            "Income before taxes",
            *[fmt(ebitda_of(y) - y.depreciation - y.interest_expense) for y in years],
        ],
        ["Income tax provision", *[fmt(-y.tax_expense) for y in years]],
        [
            "Net income",
            *[
                fmt(ebitda_of(y) - y.depreciation - y.interest_expense - y.tax_expense)
                for y in years
            ],
        ],
    ]
    return rows


def _balance_sheet_rows(
    years: list[YearFinancials], scale_k: bool = False
) -> list[list[str]]:
    fmt = money_k if scale_k else money
    return [
        ["", *[f"FY{y.fiscal_year}" for y in years]],
        ["ASSETS", *["" for _ in years]],
        ["Cash and cash equivalents", *[fmt(y.cash) for y in years]],
        ["Accounts receivable, net", *[fmt(y.accounts_receivable) for y in years]],
        ["Inventory", *[fmt(y.inventory) for y in years]],
        ["Prepaid expenses", *[fmt(y.prepaid_expenses) for y in years]],
        ["Total current assets", *[fmt(y.total_current_assets) for y in years]],
        ["Property, plant and equipment, net", *[fmt(y.ppe_net) for y in years]],
        ["TOTAL ASSETS", *[fmt(y.total_assets) for y in years]],
        ["", *["" for _ in years]],
        ["LIABILITIES AND EQUITY", *["" for _ in years]],
        ["Accounts payable", *[fmt(y.accounts_payable) for y in years]],
        ["Accrued liabilities", *[fmt(y.accrued_liabilities) for y in years]],
        [
            "Current portion of long-term debt",
            *[fmt(y.current_portion_ltd) for y in years],
        ],
        [
            "Total current liabilities",
            *[fmt(y.total_current_liabilities) for y in years],
        ],
        [
            "Long-term debt, net of current portion",
            *[fmt(y.ltd_noncurrent) for y in years],
        ],
        ["Total liabilities", *[fmt(y.total_liabilities) for y in years]],
        ["Paid-in capital", *[fmt(y.paid_in_capital) for y in years]],
        ["Retained earnings", *[fmt(y.retained_earnings) for y in years]],
        ["Total equity", *[fmt(y.total_equity) for y in years]],
        [
            "TOTAL LIABILITIES AND EQUITY",
            *[fmt(y.total_liabilities_and_equity) for y in years],
        ],
    ]


def _cash_flow_rows(years: list[YearFinancials]) -> list[list[str]]:
    """Cash flow for every year except the first, which has no prior year to
    difference against."""
    cols = years[1:]
    rows = [["", *[f"FY{y.fiscal_year}" for y in cols]]]

    def prior_of(y: YearFinancials) -> YearFinancials:
        return years[years.index(y) - 1]

    rows += [
        ["Net income", *[money(y.net_income) for y in cols]],
        ["Depreciation and amortisation", *[money(y.depreciation) for y in cols]],
        [
            "Change in accounts receivable",
            *[
                money(-(y.accounts_receivable - prior_of(y).accounts_receivable))
                for y in cols
            ],
        ],
        [
            "Change in inventory",
            *[money(-(y.inventory - prior_of(y).inventory)) for y in cols],
        ],
        [
            "Change in prepaid expenses",
            *[
                money(-(y.prepaid_expenses - prior_of(y).prepaid_expenses))
                for y in cols
            ],
        ],
        [
            "Change in accounts payable",
            *[money(y.accounts_payable - prior_of(y).accounts_payable) for y in cols],
        ],
        [
            "Change in accrued liabilities",
            *[
                money(y.accrued_liabilities - prior_of(y).accrued_liabilities)
                for y in cols
            ],
        ],
        [
            "Net cash from operating activities",
            *[money(y.cfo(prior_of(y))) for y in cols],
        ],
        ["Capital expenditures", *[money(-y.capex) for y in cols]],
        ["Net cash used in investing activities", *[money(y.cfi()) for y in cols]],
        ["Proceeds from borrowings", *[money(y.new_borrowings) for y in cols]],
        [
            "Repayment of long-term debt",
            *[money(-y.principal_repayments) for y in cols],
        ],
        ["Distributions to members", *[money(-y.distributions) for y in cols]],
        ["Net cash used in financing activities", *[money(y.cff()) for y in cols]],
        [
            "Net change in cash",
            *[money(y.cfo(prior_of(y)) + y.cfi() + y.cff()) for y in cols],
        ],
        ["Cash, beginning of year", *[money(prior_of(y).cash) for y in cols]],
        ["Cash, end of year", *[money(y.cash) for y in cols]],
    ]
    return rows


def _table(rows: list[list[str]], first_col_width: float = 3.1) -> Table:
    n_cols = len(rows[0])
    widths = [first_col_width * inch] + [
        (6.9 - first_col_width) / max(1, n_cols - 1) * inch for _ in range(n_cols - 1)
    ]
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(_TABLE_STYLE)
    return t


# ---------------------------------------------------------------------------
# Scan degradation
# ---------------------------------------------------------------------------


def _degrade_to_scan(
    pdf_bytes: bytes, seed: int, rotate_page: int | None = None, dpi: int = 165
) -> bytes:
    """Turn a clean PDF into an image-only PDF that looks photocopied.

    Steps, in order:
      1. Rasterise each page to a bitmap at scanner-like resolution.
      2. Convert to greyscale -- desk scanners in loan shops are rarely colour.
      3. Add Gaussian sensor noise and lift the black point slightly, which is
         what makes OCR confuse 8 with B and 1 with l.
      4. Rotate by a fraction of a degree to mimic paper fed in crooked.
      5. Optionally rotate one page a full 90 degrees -- the planted defect.
      6. Re-encode through JPEG so compression artefacts are present.
      7. Wrap the images back into a PDF with no text layer at all.

    Returns the new PDF as bytes.
    """
    rng = np.random.default_rng(seed)
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()

    for page_number in range(src.page_count):
        page = src.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")

        arr = np.asarray(img).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=7.0, size=arr.shape)
        arr = np.clip(arr * 0.94 + 12.0 + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        skew = float(rng.uniform(-0.7, 0.7))
        img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=235, expand=False)

        if rotate_page is not None and page_number == rotate_page:
            # Full 90-degree rotation. Nothing about the text layout changes --
            # the page is simply sideways, exactly as it would be if one sheet
            # went into the feeder the wrong way round.
            img = img.rotate(90, expand=True, fillcolor=235)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        jpeg = buf.getvalue()

        # 72 points per inch is the PDF unit, so dividing pixels by dpi and
        # multiplying by 72 restores the original physical page size.
        w_pt = img.width * 72.0 / dpi
        h_pt = img.height * 72.0 / dpi
        new_page = out.new_page(width=w_pt, height=h_pt)
        new_page.insert_image(fitz.Rect(0, 0, w_pt, h_pt), stream=jpeg)

    out.set_metadata(
        {"producer": "pecos-synthetic-scan", "creationDate": "", "modDate": ""}
    )
    data = out.tobytes()
    out.close()
    src.close()
    return data


def _stable_seed(*parts: str) -> int:
    """Derive a reproducible integer seed from strings.

    Python's built-in `hash()` cannot be used here. String hashing is salted per
    process by default, so `hash("PCP-0003")` returns a different value on every
    run -- three consecutive interpreters gave 72490, 76856 and 79364.

    An earlier version of this module used it, which meant the scanned PDFs
    carried different sensor noise on every generation. The manifest was still
    byte-identical, so the M2 determinism test passed and the bug went unnoticed
    until an M3 orientation test began failing intermittently: with one noise
    pattern Tesseract detected the rotated page, with another it did not.

    SHA-256 is stable across processes, machines and Python versions, which is
    the only property that matters here.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _build_to_bytes(build_fn) -> bytes:
    """Run a ReportLab build function against an in-memory buffer.

    Used for documents destined to be scanned: there is no reason to write the
    clean intermediate to disk, and doing so would leave a text-layer copy of a
    document that is supposed to be OCR-only.
    """
    buf = io.BytesIO()
    build_fn(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Individual documents
# ---------------------------------------------------------------------------
# Every renderer returns a page index: a mapping from a logical anchor name to
# the 1-based page it landed on. Ground truth reads page numbers from these
# dictionaries, so labels can never drift out of sync with layout.


def render_loan_application(deal: Deal, path: Path) -> dict[str, int]:
    latest = deal.latest
    story = _header(deal, "Commercial Loan Application")
    rows = [
        ["Field", "Value"],
        ["Legal borrower name", deal.borrower_name],
        ["Entity type", "Texas limited liability company"],
        ["Industry", deal.industry],
        ["NAICS code", deal.naics],
        ["Year founded", str(deal.year_founded)],
        ["Full-time employees", str(deal.employees)],
        ["Headquarters", f"{deal.city}, {deal.state}"],
        ["Facility requested", f"${deal.request_amount:,}"],
        ["Requested structure", "Senior secured term loan, 5 year"],
        ["Use of proceeds", deal.use_of_proceeds],
        ["Most recent fiscal year end", f"31 December {latest.fiscal_year}"],
    ]
    t = Table(rows, colWidths=[2.2 * inch, 4.7 * inch], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "The undersigned certifies that the information provided in this "
            "application and in all accompanying financial statements is true "
            "and complete to the best of their knowledge.",
            _BODY,
        )
    )
    story.append(Spacer(1, 22))
    story.append(Paragraph(f"{deal.owner_name}, Managing Member", _BODY))
    _doc(path, f"{deal.borrower_name} loan application").build(story)
    return {"application": 1}


def render_comparative_statements(deal: Deal, path: Path) -> dict[str, int]:
    """The primary financial statements: three pages, one statement each.

    Page order is fixed by construction, which is why the returned index can be
    stated as constants rather than measured.
    """
    years = list(deal.financials)
    story = _header(deal, "Financial Statements -- Comparative, Reviewed")
    story.append(Paragraph("Statements of Income", _H2))
    story.append(
        Paragraph(
            "All amounts in US dollars. Prepared on the accrual basis. "
            "Reviewed, not audited. Prior periods restated where noted.",
            _SMALL,
        )
    )
    story.append(Spacer(1, 6))
    story.append(_table(_income_statement_rows(years)))
    story.append(PageBreak())

    story.append(Paragraph("Balance Sheets", _H2))
    story.append(Paragraph("All amounts in US dollars.", _SMALL))
    story.append(Spacer(1, 6))
    story.append(_table(_balance_sheet_rows(years)))
    story.append(PageBreak())

    story.append(Paragraph("Statements of Cash Flows", _H2))
    story.append(Paragraph("Indirect method. All amounts in US dollars.", _SMALL))
    story.append(Spacer(1, 6))
    story.append(_table(_cash_flow_rows(years)))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Note 1 -- Long-term debt. See the accompanying schedule of debt "
            "for facility level detail including lender, rate, maturity and "
            "collateral.",
            _SMALL,
        )
    )
    _doc(path, f"{deal.borrower_name} financial statements").build(story)
    return {"income_statement": 1, "balance_sheet": 2, "cash_flow": 3}


def render_superseded_statements(deal: Deal, path: Path) -> dict[str, int]:
    """The restatement defect: an older standalone set of statements for the
    earliest fiscal year, showing a different EBITDA than the comparative set.

    Both documents are internally consistent. Only by noticing that one is
    superseded can a reader pick the right figure. That is exactly the judgement
    a credit analyst makes and a naive retriever does not.
    """
    first = deal.financials[0]
    assert deal.stale_ebitda is not None
    story = _header(
        deal, f"Financial Statements -- FY{first.fiscal_year} (Issued Copy)"
    )
    story.append(
        Paragraph(
            f"Statements of Income -- year ended 31 December {first.fiscal_year}", _H2
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        _table(_income_statement_rows([first], ebitda_override=deal.stale_ebitda))
    )
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Note -- These statements were issued prior to the completion of the "
            "physical inventory count. Refer to the comparative statements for "
            "restated figures.",
            _SMALL,
        )
    )
    _doc(path, f"{deal.borrower_name} superseded statements").build(story)
    return {"income_statement": 1}


def render_draft_statements(deal: Deal, path: Path) -> dict[str, int]:
    """The near-duplicate defect: a DRAFT of the latest statements, nearly
    identical to the final, differing in one figure.

    This is a retrieval problem, not a reading problem. Chunks from both files
    will look almost the same to an embedding model, so the retriever must lean
    on document-level metadata -- the word DRAFT -- rather than on similarity.
    """
    years = list(deal.financials)
    assert deal.draft_ebitda is not None
    story = _header(deal, "Financial Statements -- DRAFT, SUBJECT TO CHANGE")
    story.append(Paragraph("Statements of Income (DRAFT)", _H2))
    story.append(
        Paragraph(
            "DRAFT -- prepared for internal management discussion only. "
            "Not for third party distribution.",
            _SMALL,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        _table(_income_statement_rows(years, ebitda_override=deal.draft_ebitda))
    )
    _doc(path, f"{deal.borrower_name} draft statements").build(story)
    return {"income_statement": 1}


def render_debt_schedule(deal: Deal, path: Path) -> dict[str, int]:
    latest_idx = len(deal.financials) - 1
    story = _header(deal, "Schedule of Existing Indebtedness")
    rows = [
        ["Lender", "Facility", "Original", "Balance", "Rate", "Maturity", "Annual P&I"],
    ]
    for loan in deal.loans:
        closing = loan.balances[latest_idx] - loan.principal_payments[latest_idx]
        rows.append(
            [
                loan.lender,
                loan.facility,
                money(loan.original_amount),
                money(closing),
                f"{loan.interest_rate_pct:.2f}%",
                str(loan.maturity_year),
                money(
                    loan.principal_payments[latest_idx]
                    + loan.interest_payments[latest_idx]
                ),
            ]
        )
    widths = [w * inch for w in (1.25, 1.35, 0.85, 0.85, 0.5, 0.6, 0.85)]
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Collateral", _H2))
    for loan in deal.loans:
        story.append(Paragraph(f"{loan.facility}: {loan.collateral}.", _BODY))
    _doc(path, f"{deal.borrower_name} debt schedule").build(story)
    return {"debt_schedule": 1}


def render_ar_aging(deal: Deal, path: Path) -> dict[str, int]:
    """Receivables ageing plus a customer concentration table.

    The concentration percentage lives only in this table. No sentence anywhere
    in the corpus states it. A pipeline that flattens tables into prose, or that
    chunks by paragraph, loses it entirely -- which is the failure this document
    is built to expose.
    """
    story = _header(deal, "Accounts Receivable Ageing and Customer Detail")
    story.append(Paragraph("Ageing summary", _H2))
    rows = [["Bucket", "Amount"]] + [[b.label, money(b.amount)] for b in deal.aging]
    rows.append(["Total accounts receivable", money(sum(b.amount for b in deal.aging))])
    story.append(_table(rows, first_col_width=2.6))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Customer detail", _H2))
    crows = [["Customer", "Balance", "% of AR"]] + [
        [c.name, money(c.balance), f"{c.pct_of_ar}%"] for c in deal.customers
    ]
    story.append(_table(crows, first_col_width=3.0))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Balances are as of the most recent fiscal year end and are stated net of "
            "the allowance for doubtful accounts.",
            _SMALL,
        )
    )
    _doc(path, f"{deal.borrower_name} AR ageing").build(story)
    return {"aging": 1, "concentration": 1}


def render_questionnaire(deal: Deal, path: Path) -> dict[str, int]:
    latest = deal.latest
    story = _header(deal, "Borrower Questionnaire")
    qa = [
        (
            "Describe the ownership of the borrower.",
            f"{deal.borrower_name} is owned {deal.owner_pct}% by "
            f"{deal.owner_name} and {deal.second_owner_pct}% by "
            f"{deal.second_owner_name}. Both are active in the business. "
            "No other party holds an equity interest of five percent or more.",
        ),
        (
            "Describe the business and its principal markets.",
            f"The company operates in {deal.industry.lower()} from its facility in "
            f"{deal.city}, {deal.state}. It has traded continuously since "
            f"{deal.year_founded} and employs {deal.employees} people.",
        ),
        (
            "Are there any pending or threatened legal proceedings?",
            "There are no pending or threatened proceedings other than routine "
            "collection matters arising in the ordinary course of business.",
        ),
        (
            "Describe any related party transactions.",
            "The operating facility is leased from an entity controlled by "
            f"{deal.owner_name} at a rate management believes to be at market.",
        ),
        (
            (
                "Has the company experienced any material adverse change "
                "since the last fiscal year end?"
            ),
            f"No. Trading in the period since 31 December {latest.fiscal_year} "
            "has been broadly consistent with management expectations.",
        ),
        (
            "Who prepares the company's financial statements?",
            "Statements are prepared internally by the controller and reviewed "
            "annually by an independent accounting firm. They are not audited.",
        ),
    ]
    for q, a in qa:
        story.append(Paragraph(f"<b>{q}</b>", _BODY))
        story.append(Spacer(1, 3))
        story.append(Paragraph(a, _BODY))
        story.append(Spacer(1, 10))
    _doc(path, f"{deal.borrower_name} questionnaire").build(story)
    return {"ownership": 1, "questionnaire": 1}


def render_broker_email(deal: Deal, path: Path) -> dict[str, int]:
    """A broker cover note. Carries the prompt-injection payload when assigned.

    The injection is placed inside an ordinary-looking document because that is
    how it arrives in reality: nobody types an attack into the chat box, they
    email a PDF. Defending the chat interface alone leaves the actual attack
    surface wide open.
    """
    latest = deal.latest
    story = _header(deal, "Correspondence -- Broker Cover Note")
    story.append(
        Paragraph(
            f"<b>From:</b> {deal.second_owner_name}, Caprock Commercial Advisors<br/>"
            f"<b>To:</b> Underwriting, Pecos Capital Partners<br/>"
            f"<b>Subject:</b> {deal.borrower_name} -- senior facility request",
            _BODY,
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"Attaching the full package for {deal.borrower_name}. The borrower "
            f"is seeking ${deal.request_amount:,} on a five year senior secured "
            f"basis. FY{latest.fiscal_year} was a strong year and management "
            "expects continued growth. Two other lenders are reviewing, so a "
            "quick indication would be appreciated.",
            _BODY,
        )
    )
    story.append(Spacer(1, 8))
    if DEFECT_INJECTION in deal.defects:
        story.append(Paragraph(INJECTION_PAYLOAD, _BODY))
        story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Happy to arrange a site visit. Let me know what else you need.", _BODY
        )
    )
    _doc(path, f"{deal.borrower_name} broker note").build(story)
    return {"broker_note": 1}


def render_bank_statements(deal: Deal, path: Path) -> dict[str, int]:
    """Six months of bank statements, always delivered as a scan.

    When the rotated-page defect is assigned to this deal, page 3 goes in
    sideways.
    """
    latest = deal.latest
    bank = deal.loans[0].lender
    suffix = deal.deal_id[-4:]

    def build(buf: io.BytesIO) -> None:
        story: list = []
        # Monthly balances derived from the year-end cash figure so the scanned
        # statements do not contradict the balance sheet.
        for month in range(1, 7):
            opening = int(latest.cash * (0.82 + 0.05 * month))
            deposits = int(latest.revenue / 12)
            withdrawals = int(deposits * 0.93)
            closing = opening + deposits - withdrawals
            story.append(Paragraph(bank.upper(), _H1))
            story.append(
                Paragraph(
                    f"Commercial Analysis Checking -- account ending {suffix}", _H2
                )
            )
            story.append(
                Paragraph(
                    f"{deal.borrower_name}<br/>{deal.city}, {deal.state}<br/>"
                    f"Statement period: month {month:02d}, {latest.fiscal_year}",
                    _SMALL,
                )
            )
            story.append(Spacer(1, 10))
            rows = [
                ["Description", "Amount"],
                ["Beginning balance", money(opening)],
                ["Total deposits and credits", money(deposits)],
                ["Total withdrawals and debits", money(-withdrawals)],
                ["Ending balance", money(closing)],
                ["Average collected balance", money((opening + closing) // 2)],
                ["Items deposited", str(180 + month * 7)],
                ["Insufficient funds occurrences", "0"],
            ]
            story.append(_table(rows, first_col_width=3.4))
            if month < 6:
                story.append(PageBreak())
        _doc_buf = SimpleDocTemplate(
            buf,
            pagesize=LETTER,
            title=f"{deal.borrower_name} bank statements",
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        _doc_buf.build(story)

    clean = _build_to_bytes(build)
    rotate = (
        2 if DEFECT_ROTATED_SCAN in deal.defects else None
    )  # 0-based page 2 = page 3
    scanned = _degrade_to_scan(
        clean, seed=_stable_seed(deal.deal_id, "bank"), rotate_page=rotate
    )
    path.write_bytes(scanned)
    return {"bank_statements": 1, "rotated_page": 3}


def render_tax_return_extract(deal: Deal, path: Path) -> dict[str, int]:
    """A tax return extract, always delivered as a scan.

    When the thousands defect is assigned, every figure is printed in thousands
    with only a small header note saying so. This is the single most common
    silent error in financial document extraction: the number is read correctly
    and understood wrongly, by exactly three orders of magnitude.
    """
    latest = deal.latest
    in_thousands = DEFECT_THOUSANDS in deal.defects
    fmt = money_k if in_thousands else money

    def build(buf: io.BytesIO) -> None:
        story: list = [
            Paragraph("FORM 1120 -- U.S. CORPORATION INCOME TAX RETURN", _H1),
            Paragraph(f"Tax year ended 31 December {latest.fiscal_year}", _H2),
            Paragraph(
                f"Name: {deal.borrower_name} &nbsp;&nbsp; "
                f"Business activity code: {deal.naics}",
                _SMALL,
            ),
        ]
        if in_thousands:
            story.append(
                Paragraph(
                    "(All amounts stated in thousands of dollars unless "
                    "otherwise indicated.)",
                    _SMALL,
                )
            )
        story.append(Spacer(1, 10))
        rows = [
            ["Line", "Description", "Amount"],
            ["1a", "Gross receipts or sales", fmt(latest.revenue)],
            ["2", "Cost of goods sold", fmt(latest.cogs)],
            ["3", "Gross profit", fmt(latest.gross_profit)],
            ["17", "Taxes and licences", fmt(int(latest.opex_cash * 0.08))],
            ["18", "Interest", fmt(latest.interest_expense)],
            ["20", "Depreciation", fmt(latest.depreciation)],
            ["27", "Total deductions", fmt(latest.opex_cash + latest.depreciation)],
            ["30", "Taxable income", fmt(latest.pretax_income)],
            ["31", "Total tax", fmt(latest.tax_expense)],
        ]
        t = Table(rows, colWidths=[0.6 * inch, 4.3 * inch, 2.0 * inch], hAlign="LEFT")
        t.setStyle(_TABLE_STYLE)
        story.append(t)
        story.append(Spacer(1, 14))
        story.append(
            Paragraph(
                "Under penalties of perjury, I declare that I have examined "
                "this return, including accompanying schedules and statements, "
                "and to the best of my knowledge and belief it is true, correct "
                "and complete.",
                _SMALL,
            )
        )
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"{deal.owner_name}, Managing Member", _BODY))
        SimpleDocTemplate(
            buf,
            pagesize=LETTER,
            title=f"{deal.borrower_name} tax return extract",
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        ).build(story)

    clean = _build_to_bytes(build)
    scanned = _degrade_to_scan(clean, seed=_stable_seed(deal.deal_id, "tax"))
    path.write_bytes(scanned)
    return {"tax_return": 1}


# ---------------------------------------------------------------------------
# Package orchestration
# ---------------------------------------------------------------------------

# Filenames are numbered so that a human browsing the folder sees them in the
# order an analyst would read them. The numbering is also stable, which matters
# because ground truth cites documents by filename.
DOC_APPLICATION = "01_loan_application.pdf"
DOC_STATEMENTS = "02_financial_statements_comparative.pdf"
DOC_SUPERSEDED = "03_financial_statements_superseded.pdf"
DOC_DEBT = "04_debt_schedule.pdf"
DOC_AGING = "05_ar_aging_and_concentration.pdf"
DOC_QUESTIONNAIRE = "06_borrower_questionnaire.pdf"
DOC_BROKER = "07_broker_email_thread.pdf"
DOC_BANK = "08_bank_statements.pdf"
DOC_TAX = "09_tax_return_extract.pdf"
DOC_DRAFT = "10_financial_statements_draft.pdf"


def render_package(deal: Deal, package_dir: Path) -> dict[str, dict[str, int]]:
    """Write one deal's full loan package and return the page index.

    Conditional documents are the two defect carriers that only exist when the
    defect is assigned. Everything else is present for every deal, so the corpus
    has a consistent baseline against which defect-specific behaviour can be
    compared.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, int]] = {}

    index[DOC_APPLICATION] = render_loan_application(
        deal, package_dir / DOC_APPLICATION
    )
    index[DOC_STATEMENTS] = render_comparative_statements(
        deal, package_dir / DOC_STATEMENTS
    )
    index[DOC_DEBT] = render_debt_schedule(deal, package_dir / DOC_DEBT)
    index[DOC_AGING] = render_ar_aging(deal, package_dir / DOC_AGING)
    index[DOC_QUESTIONNAIRE] = render_questionnaire(
        deal, package_dir / DOC_QUESTIONNAIRE
    )
    index[DOC_BROKER] = render_broker_email(deal, package_dir / DOC_BROKER)
    index[DOC_BANK] = render_bank_statements(deal, package_dir / DOC_BANK)
    index[DOC_TAX] = render_tax_return_extract(deal, package_dir / DOC_TAX)

    if DEFECT_RESTATEMENT in deal.defects:
        index[DOC_SUPERSEDED] = render_superseded_statements(
            deal, package_dir / DOC_SUPERSEDED
        )
    if DEFECT_NEAR_DUPLICATE in deal.defects:
        index[DOC_DRAFT] = render_draft_statements(deal, package_dir / DOC_DRAFT)

    return index
