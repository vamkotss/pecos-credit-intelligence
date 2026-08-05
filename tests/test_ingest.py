"""End-to-end ingestion tests (M3).

These render a corpus and run real OCR, so they take about a minute. They are
marked `slow`; CI runs them, `pytest -m "not slow"` skips them.

The design principle throughout: **score extraction against the M2 manifest, not
against itself.** A test that checks the ingester produced *something* proves
nothing. A test that checks the ingester recovered the exact figure the renderer
printed, on the exact page the manifest cites, closes the loop between the two
milestones and is the reason the ground truth was built first.
"""

from __future__ import annotations

import re

import fitz
import pytest

from pecos.corpus import CorpusSpec, build_deal
from pecos.corpus_build import generate_corpus
from pecos.ingest import (
    OCR_DPI,
    _ocr_once,
    ingest_corpus,
    load_pages,
    render_page,
)

pytestmark = pytest.mark.slow

SEED = 20260804
N_DEALS = 7  # smallest size at which each defect lands on its own deal

SCANNED_DOCS = ("08_bank_statements.pdf", "09_tax_return_extract.pdf")


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    """Generate a corpus, then ingest the three deals the defect tests need.

    Ingesting all seven would triple the runtime for no extra coverage: the
    remaining deals are clean, and cleanliness is already exercised by the
    control assertions inside each defect test.
    """
    root = tmp_path_factory.mktemp("m3")
    spec = CorpusSpec(seed=SEED, n_deals=N_DEALS, out_dir=root / "raw", years=3)
    manifest = generate_corpus(spec, gold_dir=root / "evals", write_pdfs=True)

    defect = manifest.defect_index
    wanted = sorted(
        {
            defect["rotated_scanned_page"][0],
            defect["units_in_thousands"][0],
            defect["table_only_fact"][0],
        }
    )
    out_dir = root / "interim"
    summary = ingest_corpus(root / "raw" / "packages", out_dir, deal_ids=wanted)

    pages = {deal: load_pages(out_dir / f"{deal}.jsonl") for deal in wanted}
    return {
        "manifest": manifest,
        "packages": root / "raw" / "packages",
        "out_dir": out_dir,
        "pages": pages,
        "summary": summary,
    }


def _all_pages(ingested) -> list[dict]:
    return [p for records in ingested["pages"].values() for p in records]


def _page(ingested, deal: str, document: str, number: int) -> dict:
    for record in ingested["pages"][deal]:
        if record["document"] == document and record["page_number"] == number:
            return record
    raise AssertionError(f"no record for {deal}/{document}#{number}")


def _digits(text: str) -> int | None:
    """Pull a whole-dollar integer out of an OCR'd cell.

    Accounting parentheses are treated as a negative sign, which is the
    convention the statements are printed in.
    """
    cleaned = text.strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    digits = re.sub(r"[^0-9]", "", cleaned)
    if not digits:
        return None
    value = int(digits)
    return -value if negative else value


# ---------------------------------------------------------------------------
# Coverage and routing
# ---------------------------------------------------------------------------


def test_every_pdf_page_produces_exactly_one_record(ingested):
    """Page counts must match the source PDFs exactly. A dropped page is a hole
    in the index that only surfaces as an unexplained retrieval miss."""
    for deal, records in ingested["pages"].items():
        by_document: dict[str, list[int]] = {}
        for record in records:
            by_document.setdefault(record["document"], []).append(record["page_number"])

        for pdf_path in sorted((ingested["packages"] / deal).glob("*.pdf")):
            doc = fitz.open(pdf_path)
            expected = doc.page_count
            doc.close()
            got = sorted(by_document.get(pdf_path.name, []))
            assert got == list(range(1, expected + 1)), (
                f"{deal}/{pdf_path.name}: expected pages 1..{expected}, got {got}"
            )


def test_pages_are_routed_by_text_layer_not_by_filename(ingested):
    """The router inspects the page, it does not pattern-match the name.

    The assertion is written against the known-scanned filenames only because
    that is what the corpus happens to contain; the code path being tested never
    sees them.
    """
    for record in _all_pages(ingested):
        expected = "ocr" if record["document"] in SCANNED_DOCS else "digital"
        assert record["method"] == expected, (
            f"{record['deal_id']}/{record['document']}#{record['page_number']} "
            f"went down the {record['method']} path"
        )


def test_no_page_comes_back_empty(ingested):
    """A page with no text is a page the pipeline cannot answer from."""
    for record in _all_pages(ingested):
        assert record["text"].strip(), (
            f"{record['deal_id']}/{record['document']}#{record['page_number']} is empty"
        )
        assert record["word_count"] > 0


def test_provenance_is_complete_on_every_record(ingested):
    """Every field a citation needs, on every page, with no nulls.

    Provenance that is present most of the time is not provenance -- the memo
    generator cannot know in advance which page it will need to cite.
    """
    for record in _all_pages(ingested):
        assert record["deal_id"]
        assert record["document"].endswith(".pdf")
        assert record["page_number"] >= 1
        assert record["method"] in ("digital", "ocr")
        assert record["rotation_applied"] in (0, 90, 180, 270)
        assert record["extractor_version"]
        assert record["scale_factor"] in (1, 1_000, 1_000_000)
        if record["method"] == "ocr":
            assert record["mean_word_confidence"] is not None
        else:
            assert record["mean_word_confidence"] is None


def test_ocr_confidence_clears_the_quality_bar(ingested):
    """A quality floor, not a vanity metric.

    If a future change to scan degradation pushes OCR below this, every
    downstream number becomes suspect, and it should fail here rather than show
    up as a mysterious drop in extraction accuracy at M6.
    """
    ocr_pages = [r for r in _all_pages(ingested) if r["method"] == "ocr"]
    assert ocr_pages
    for record in ocr_pages:
        assert record["mean_word_confidence"] >= 85.0, (
            f"{record['deal_id']}/{record['document']}#{record['page_number']} "
            f"scored {record['mean_word_confidence']}"
        )


# ---------------------------------------------------------------------------
# Rotated page defect
# ---------------------------------------------------------------------------


def test_the_rotated_page_is_detected_and_corrected(ingested):
    """Exactly one page rotated, on the deal the manifest says carries it."""
    deal = ingested["manifest"].defect_index["rotated_scanned_page"][0]
    rotated = [r for r in ingested["pages"][deal] if r["rotation_applied"] != 0]
    assert len(rotated) == 1, f"expected 1 rotated page, found {len(rotated)}"

    page = rotated[0]
    assert page["document"] == "08_bank_statements.pdf"
    assert page["page_number"] == 3
    assert page["rotation_applied"] == 90


def test_clean_deals_have_no_rotated_pages(ingested):
    """The control. Rotation detection that fires everywhere is not detection."""
    carrier = ingested["manifest"].defect_index["rotated_scanned_page"][0]
    for deal, records in ingested["pages"].items():
        if deal == carrier:
            continue
        assert all(r["rotation_applied"] == 0 for r in records), (
            f"{deal} has a spurious rotation"
        )


def test_orientation_correction_is_load_bearing(ingested):
    """Proves the correction step earns its cost.

    The same page is OCR'd twice: once as it sits in the PDF, once after
    correction. Uncorrected, Tesseract returns confident nonsense -- strings
    like `") S89U9INDDO Spun} JUaIONJNSU]"` -- which would be embedded, indexed
    and retrieved as though it meant something. This test fails if the
    correction ever silently stops mattering.
    """
    deal = ingested["manifest"].defect_index["rotated_scanned_page"][0]
    corrected = _page(ingested, deal, "08_bank_statements.pdf", 3)

    doc = fitz.open(ingested["packages"] / deal / "08_bank_statements.pdf")
    try:
        image = render_page(doc.load_page(2), OCR_DPI)
    finally:
        doc.close()
    raw_text, _, _ = _ocr_once(image)

    marker = "Commercial Analysis Checking"
    assert marker.lower() in corrected["text"].lower(), (
        "corrected page lost its heading"
    )
    assert marker.lower() not in raw_text.lower(), (
        "the uncorrected page was already readable -- the rotation defect is "
        "no longer being planted, or the test is checking the wrong page"
    )


def test_the_rotated_page_still_yields_its_table(ingested):
    """Structure has to survive rotation, not just text.

    Skew runs along the long axis once a page is turned, so this is the hardest
    page in the corpus for line grouping and the reason the algorithm walks
    words left to right instead of sorting them by height.
    """
    deal = ingested["manifest"].defect_index["rotated_scanned_page"][0]
    page = _page(ingested, deal, "08_bank_statements.pdf", 3)
    assert page["tables"], "the rotated page produced no table"

    cells = [c for t in page["tables"] for row in t["rows"] for c in row]
    joined = " ".join(cells).lower()
    assert "beginning balance" in joined
    assert "ending balance" in joined


# ---------------------------------------------------------------------------
# Units-in-thousands defect
# ---------------------------------------------------------------------------


def test_the_units_note_is_detected_only_where_it_is_planted(ingested):
    carrier = ingested["manifest"].defect_index["units_in_thousands"][0]

    page = _page(ingested, carrier, "09_tax_return_extract.pdf", 1)
    assert page["scale_factor"] == 1_000
    assert "thousand" in page["scale_evidence"].lower()

    for deal, records in ingested["pages"].items():
        if deal == carrier:
            continue
        for record in records:
            assert record["scale_factor"] == 1, (
                f"{deal}/{record['document']}#{record['page_number']} was rescaled "
                f"without a units note"
            )


def test_rescaling_recovers_the_true_dollar_figure(ingested):
    """The end-to-end proof that the units defect is defused.

    A tolerance is used rather than exact equality, and the reason is worth
    stating: the document genuinely prints 32,041 where the true figure is
    32,041,248. The thousands were rounded by the preparer, so the original
    precision is not recoverable from the page and no extraction method could
    return it. What *is* recoverable is the magnitude, and getting that wrong is
    the failure that matters -- it is the difference between a $32M borrower and
    a $32K one.
    """
    carrier = ingested["manifest"].defect_index["units_in_thousands"][0]
    record = next(d for d in ingested["manifest"].deals if d["deal_id"] == carrier)
    true_revenue = record["latest_revenue"]

    page = _page(ingested, carrier, "09_tax_return_extract.pdf", 1)
    assert page["scale_factor"] == 1_000

    printed: int | None = None
    for table in page["tables"]:
        for row in table["rows"]:
            if any("gross receipts" in cell.lower() for cell in row):
                printed = _digits(row[-1])
                break
    assert printed is not None, "could not find the gross receipts row"

    normalised = printed * page["scale_factor"]
    error = abs(normalised - true_revenue) / true_revenue
    assert error < 0.001, (
        f"normalised {normalised:,} vs true {true_revenue:,} "
        f"({error:.4%} error) -- rescaling did not recover the magnitude"
    )

    # And confirm the naive read really would have been catastrophic.
    naive_error = abs(printed - true_revenue) / true_revenue
    assert naive_error > 0.9


# ---------------------------------------------------------------------------
# Table-only fact defect
# ---------------------------------------------------------------------------


def test_the_table_only_fact_survives_as_a_table_cell(ingested):
    """The concentration figure exists in no sentence anywhere in the corpus.

    If tables flatten into prose, the number loses its row label and becomes
    unretrievable. This asserts it comes back as a cell sitting in the same row
    as the customer it belongs to.
    """
    carrier = ingested["manifest"].defect_index["table_only_fact"][0]
    fact = next(
        f
        for f in ingested["manifest"].facts
        if f["deal_id"] == carrier and f["defect_tag"] == "table_only_fact"
    )
    expected_pct = f"{fact['answer_value']}%"
    customer = fact["answer_text"].split("(")[-1].rstrip(")")

    page = _page(ingested, carrier, fact["source_document"], fact["source_page"])
    assert page["tables"], "the concentration table was not recovered"

    matching_rows = [
        row
        for table in page["tables"]
        for row in table["rows"]
        if any(expected_pct in cell for cell in row)
    ]
    assert matching_rows, f"{expected_pct} not found in any table cell"
    assert any(
        any(customer.split()[0] in cell for cell in row) for row in matching_rows
    ), f"{expected_pct} was recovered but not alongside {customer}"


# ---------------------------------------------------------------------------
# Fidelity against ground truth
# ---------------------------------------------------------------------------


def test_digital_pages_reproduce_the_gold_figures_on_the_cited_page(ingested):
    """The strongest check in the milestone.

    For every gold fact whose source is a digital document, the exact formatted
    figure the renderer printed must appear in the text extracted from the exact
    page the manifest cites. This is what makes recall@k and exact-match
    extraction measurable at M5 and M6 -- if it fails, the oracle and the
    pipeline are looking at different documents.
    """
    ingested_deals = set(ingested["pages"])
    checked = 0

    for fact in ingested["manifest"].facts:
        if fact["deal_id"] not in ingested_deals:
            continue
        if not fact["answerable"] or fact["answer_unit"] != "USD":
            continue
        if fact["source_document"] in SCANNED_DOCS:
            continue

        page = _page(
            ingested, fact["deal_id"], fact["source_document"], fact["source_page"]
        )
        formatted = f"{fact['answer_value']:,}"
        assert formatted in page["text"], (
            f"{fact['fact_id']}: {formatted} not found on "
            f"{fact['source_document']}#{fact['source_page']}"
        )
        checked += 1

    assert checked >= 10, f"only {checked} facts were checked"


def test_the_inputs_to_a_derived_metric_are_all_extractable(ingested):
    """Leverage is never printed anywhere -- it is total debt over EBITDA, and
    neither the ratio nor the debt total appears on any page.

    EBITDA is a printed line. Total debt has to be assembled from two separate
    balance-sheet lines. Both components must therefore come back from
    extraction, or M7's calculator tool has nothing to compute from.
    """
    ingested_deals = set(ingested["pages"])
    checked = 0
    for i, record in enumerate(ingested["manifest"].deals):
        if record["deal_id"] not in ingested_deals:
            continue
        deal = build_deal(
            CorpusSpec(
                seed=SEED, n_deals=N_DEALS, out_dir=ingested["packages"], years=3
            ),
            i,
        )
        latest = deal.latest

        page_one = _page(
            ingested, record["deal_id"], "02_financial_statements_comparative.pdf", 1
        )
        page_two = _page(
            ingested, record["deal_id"], "02_financial_statements_comparative.pdf", 2
        )
        assert f"{record['latest_ebitda']:,}" in page_one["text"]
        assert f"{latest.current_portion_ltd:,}" in page_two["text"]
        assert f"{latest.ltd_noncurrent:,}" in page_two["text"]
        assert (
            latest.current_portion_ltd + latest.ltd_noncurrent
            == record["latest_total_debt"]
        )
        checked += 1
    assert checked >= 1


# ---------------------------------------------------------------------------
# Determinism and summary
# ---------------------------------------------------------------------------


def test_ingestion_is_deterministic(ingested, tmp_path):
    """Same PDFs in, same extractions out.

    OCR is not obviously deterministic to anyone who has not checked, and if it
    were not, every eval score in the project would be unreproducible.
    """
    deal = sorted(ingested["pages"])[0]
    second = ingest_corpus(ingested["packages"], tmp_path, deal_ids=[deal])
    assert second.pages > 0

    original = ingested["pages"][deal]
    repeat = load_pages(tmp_path / f"{deal}.jsonl")
    assert len(original) == len(repeat)
    for a, b in zip(original, repeat, strict=True):
        assert a["text"] == b["text"]
        assert a["rotation_applied"] == b["rotation_applied"]
        assert a["scale_factor"] == b["scale_factor"]
        assert a["tables"] == b["tables"]


def test_the_summary_agrees_with_the_records(ingested):
    """The summary is what a reviewer reads instead of 45 JSON files, so it must
    not be able to drift from them."""
    summary = ingested["summary"]
    records = _all_pages(ingested)

    assert summary.pages == len(records)
    assert summary.digital_pages == sum(1 for r in records if r["method"] == "digital")
    assert summary.ocr_pages == sum(1 for r in records if r["method"] == "ocr")
    assert summary.rotated_pages == sum(1 for r in records if r["rotation_applied"])
    assert summary.scaled_pages == sum(1 for r in records if r["scale_factor"] != 1)
    assert summary.tables == sum(len(r["tables"]) for r in records)
    assert summary.empty_pages == []
    assert summary.mean_ocr_confidence >= 85.0
