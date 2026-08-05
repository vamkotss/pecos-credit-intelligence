"""End-to-end chunking tests (M4).

These generate a real corpus, ingest it, and chunk the result, so they run real
OCR and take about half a minute. Marked `slow`.

Only the documents the assertions actually need are ingested. Bank statements
are six OCR pages each and nothing here depends on them, so skipping them cuts
the fixture from roughly two minutes to thirty seconds -- and a test that takes
two minutes is a test that gets skipped.

The headline assertion is `test_containment_is_total`. It measures the ceiling
chunking imposes on the whole rest of the project: a figure that does not
survive into a chunk cannot be retrieved by any retriever, rescued by any
reranker, or cited by any agent.
"""

from __future__ import annotations

import pytest

from pecos.chunk_audit import audit_containment, audit_near_duplicates
from pecos.chunking import chunk_deal
from pecos.corpus import CorpusSpec
from pecos.corpus_build import generate_corpus
from pecos.ingest import ingest_document

pytestmark = pytest.mark.slow

SEED = 20260804
N_DEALS = 7

# Everything except the bank statements, which are six OCR pages per deal and
# carry nothing any assertion here depends on.
DOCUMENTS = (
    "01_loan_application.pdf",
    "02_financial_statements_comparative.pdf",
    "03_financial_statements_superseded.pdf",
    "04_debt_schedule.pdf",
    "05_ar_aging_and_concentration.pdf",
    "06_borrower_questionnaire.pdf",
    "07_broker_email_thread.pdf",
    "09_tax_return_extract.pdf",
    "10_financial_statements_draft.pdf",
)


@pytest.fixture(scope="module")
def chunked(tmp_path_factory):
    root = tmp_path_factory.mktemp("m4")
    spec = CorpusSpec(seed=SEED, n_deals=N_DEALS, out_dir=root / "raw", years=3)
    manifest = generate_corpus(spec, gold_dir=root / "evals", write_pdfs=True)

    defect = manifest.defect_index
    wanted = sorted(
        {
            defect["restated_prior_year"][0],
            defect["units_in_thousands"][0],
            defect["table_only_fact"][0],
            defect["near_duplicate_draft"][0],
            defect["prompt_injection"][0],
        }
    )

    packages = root / "raw" / "packages"
    chunks: list[dict] = []
    for deal_id in wanted:
        pages = []
        for name in DOCUMENTS:
            path = packages / deal_id / name
            if path.exists():
                pages.extend(ingest_document(path, deal_id))
        chunks.extend(c.to_record() for c in chunk_deal(pages))

    return {"manifest": manifest, "chunks": chunks, "deals": set(wanted)}


def _for(chunked, deal: str, document: str | None = None) -> list[dict]:
    return [
        c
        for c in chunked["chunks"]
        if c["deal_id"] == deal and (document is None or c["document"] == document)
    ]


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------


def test_containment_is_total(chunked):
    """Every extractive gold fact survives into a chunk anchored to its page.

    This is the single most important number in the milestone. Retrieval work
    spent chasing a fact that chunking already destroyed is wasted, and the
    failure looks identical to a retrieval bug -- so it has to be ruled out
    here, before any retriever exists to blame.
    """
    result = audit_containment(
        chunked["chunks"], chunked["manifest"].facts, chunked["deals"]
    )
    assert result.extractive >= 20, f"only {result.extractive} facts were checked"
    assert result.rate == 1.0, (
        f"containment {result.rate:.1%}; missing: "
        f"{[(m.fact_id, m.needle) for m in result.misses]}"
    )


def test_the_audit_excludes_the_right_things(chunked):
    """The exclusions have to be principled, or a 100% score means nothing.

    Derived metrics are never printed on any page; behavioural facts are scored
    on refusal. Counting either as extractive would report a false failure rate,
    and quietly dropping them would let a real regression hide.
    """
    result = audit_containment(
        chunked["chunks"], chunked["manifest"].facts, chunked["deals"]
    )
    assert result.excluded_derived >= 4
    assert result.excluded_behavioural >= 1
    assert result.extractive + result.excluded_derived + result.excluded_behavioural > 0


def test_defect_facts_are_inside_the_audited_set(chunked):
    """The audit must actually be exercising the hard cases, not just the easy
    ones. A 100% containment rate over only clean facts would be worthless."""
    result = audit_containment(
        chunked["chunks"], chunked["manifest"].facts, chunked["deals"]
    )
    for defect in ("table_only_fact", "units_in_thousands", "near_duplicate_draft"):
        assert result.by_defect[defect] >= 1, f"{defect} was not audited"


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def test_every_chunk_anchors_to_a_real_page(chunked):
    for chunk in chunked["chunks"]:
        assert chunk["deal_id"].startswith("PCP-")
        assert chunk["document"].endswith(".pdf")
        assert chunk["page_number"] >= 1
        assert chunk["chunk_id"].startswith(
            f"{chunk['deal_id']}::{chunk['document']}::"
        )
        assert chunk["chunker_version"]


def test_chunk_ids_are_unique_across_the_corpus(chunked):
    ids = [c["chunk_id"] for c in chunked["chunks"]]
    assert len(ids) == len(set(ids))


def test_no_ingested_page_vanishes(chunked):
    """Every page that produced text must produce at least one chunk.

    Pages routinely produce two -- a table and the prose around it -- but zero
    means content was dropped between M3 and M4, which is invisible until
    someone asks a question only that page could answer.
    """
    pages = {(c["deal_id"], c["document"], c["page_number"]) for c in chunked["chunks"]}
    for deal in chunked["deals"]:
        statements = _for(chunked, deal, "02_financial_statements_comparative.pdf")
        assert statements
        assert {c["page_number"] for c in statements} == {1, 2, 3}
    assert len(pages) >= 25


def test_chunks_respect_the_size_ceiling(chunked):
    for chunk in chunked["chunks"]:
        assert chunk["char_count"] <= 1_400, chunk["chunk_id"]
        assert chunk["text"].strip()


# ---------------------------------------------------------------------------
# The near-duplicate defect
# ---------------------------------------------------------------------------


def test_the_draft_and_the_final_are_nearly_identical_in_text(chunked):
    """Establishes that the problem is real before testing the defence.

    If these two documents were easy to tell apart by content, carrying document
    status through chunking would be unnecessary ceremony. They are not: the
    text overlaps almost entirely, which is exactly why cosine similarity cannot
    separate them.
    """
    import difflib

    deal = chunked["manifest"].defect_index["near_duplicate_draft"][0]
    final = _for(chunked, deal, "02_financial_statements_comparative.pdf")
    draft = _for(chunked, deal, "10_financial_statements_draft.pdf")
    assert final and draft

    final_income = next(
        c for c in final if c["page_number"] == 1 and c["chunk_type"] == "table"
    )
    draft_income = next(c for c in draft if c["chunk_type"] == "table")

    ratio = difflib.SequenceMatcher(
        None, final_income["text"], draft_income["text"]
    ).ratio()
    assert ratio > 0.85, f"the draft is only {ratio:.0%} similar; defect not planted?"


def test_status_metadata_separates_what_text_cannot(chunked):
    """The defence itself. Identical text, different authority."""
    deal = chunked["manifest"].defect_index["near_duplicate_draft"][0]
    final = _for(chunked, deal, "02_financial_statements_comparative.pdf")
    draft = _for(chunked, deal, "10_financial_statements_draft.pdf")

    assert all(c["doc_status"] == "final" and c["authority"] == 3 for c in final)
    assert all(c["doc_status"] == "draft" and c["authority"] == 1 for c in draft)


def test_the_draft_announces_itself_in_the_embedded_text(chunked):
    """Status is put in the context header as well as the metadata, so it is
    available to similarity search and not only to a hard filter. A query asking
    for the draft should be able to find it."""
    deal = chunked["manifest"].defect_index["near_duplicate_draft"][0]
    for chunk in _for(chunked, deal, "10_financial_statements_draft.pdf"):
        assert "DRAFT" in chunk["context_header"]


def test_the_superseded_statements_are_ranked_below_the_comparative(chunked):
    """Same mechanism, different defect. The restatement case turns on
    preferring the comparative statements over the superseded issued copy."""
    deal = chunked["manifest"].defect_index["restated_prior_year"][0]
    superseded = _for(chunked, deal, "03_financial_statements_superseded.pdf")
    comparative = _for(chunked, deal, "02_financial_statements_comparative.pdf")

    assert superseded
    assert all(c["doc_status"] == "superseded" for c in superseded)
    assert all(c["authority"] == 2 for c in superseded)
    assert all(c["authority"] == 3 for c in comparative)


def test_the_audit_reports_every_non_final_document(chunked):
    near = audit_near_duplicates(chunked["chunks"])
    documents = {d for entry in near.values() for d in entry["documents"]}
    assert "10_financial_statements_draft.pdf" in documents
    assert "03_financial_statements_superseded.pdf" in documents


# ---------------------------------------------------------------------------
# The table-only defect
# ---------------------------------------------------------------------------


def test_the_concentration_figure_survives_with_its_row_label(chunked):
    """The figure appears in no sentence anywhere in the corpus. If tables
    flatten into prose it loses its label and becomes unretrievable no matter
    how good the retriever is."""
    deal = chunked["manifest"].defect_index["table_only_fact"][0]
    fact = next(
        f
        for f in chunked["manifest"].facts
        if f["deal_id"] == deal and f["defect_tag"] == "table_only_fact"
    )
    needle = f"{fact['answer_value']}%"
    customer = fact["answer_text"].split("(")[-1].rstrip(")").split()[0]

    matching = [
        c
        for c in _for(chunked, deal, fact["source_document"])
        if c["chunk_type"] == "table" and needle in c["text"]
    ]
    assert matching, f"{needle} is in no table chunk"
    assert any(
        customer in c["text"] for c in matching
    ), f"{needle} survived but lost its association with {customer}"


def test_table_chunks_record_how_the_table_was_found(chunked):
    """Structure read from ruling lines is more reliable than structure inferred
    from whitespace. Downstream code that cannot tell them apart will be wrong
    about scanned tables."""
    table_chunks = [c for c in chunked["chunks"] if c["chunk_type"] == "table"]
    assert table_chunks
    for chunk in table_chunks:
        assert chunk["table_source"] in ("pdfplumber_lines", "layout_clustering")


# ---------------------------------------------------------------------------
# The units defect
# ---------------------------------------------------------------------------


def test_rescaled_pages_carry_their_units_into_every_chunk(chunked):
    """The multiplier has to travel with the chunk. A chunk reading
    `Gross receipts or sales | 32,041` with no units note attached is off by
    three orders of magnitude and nothing about it looks wrong."""
    deal = chunked["manifest"].defect_index["units_in_thousands"][0]
    tax_chunks = _for(chunked, deal, "09_tax_return_extract.pdf")
    assert tax_chunks
    for chunk in tax_chunks:
        assert chunk["scale_factor"] == 1_000
        assert chunk["scale_evidence"]
        assert "units of 1,000" in chunk["context_header"]


def test_other_deals_tax_returns_are_not_rescaled(chunked):
    """The control. A rescaler that fires everywhere is not a rescaler."""
    carrier = chunked["manifest"].defect_index["units_in_thousands"][0]
    for deal in chunked["deals"]:
        if deal == carrier:
            continue
        for chunk in _for(chunked, deal, "09_tax_return_extract.pdf"):
            assert chunk["scale_factor"] == 1


# ---------------------------------------------------------------------------
# Sections and provenance
# ---------------------------------------------------------------------------


def test_statement_chunks_are_labelled_with_their_section(chunked):
    """`Statements of Income` versus `Balance Sheets` is the difference between
    a revenue figure and a cash figure. Losing it makes two chunks of similar
    numbers indistinguishable."""
    deal = sorted(chunked["deals"])[0]
    statements = _for(chunked, deal, "02_financial_statements_comparative.pdf")
    sections = {c["section"] for c in statements if c["section"]}
    assert "Statements of Income" in sections
    assert "Balance Sheets" in sections
    assert "Statements of Cash Flows" in sections


def test_the_broker_note_is_marked_as_third_party(chunked):
    """Groundwork for M8. The prompt-injection payload arrives inside a document
    the borrower's broker wrote, not one the lender or the borrower prepared,
    and that distinction should be on the chunk before it is needed."""
    deal = chunked["manifest"].defect_index["prompt_injection"][0]
    chunks = _for(chunked, deal, "07_broker_email_thread.pdf")
    assert chunks
    for chunk in chunks:
        assert chunk["source_trust"] == "third_party"


def test_bank_and_tax_documents_are_trusted_differently_from_borrower_prepared(chunked):
    deal = sorted(chunked["deals"])[0]
    tax = _for(chunked, deal, "09_tax_return_extract.pdf")
    statements = _for(chunked, deal, "02_financial_statements_comparative.pdf")
    assert all(c["source_trust"] == "tax_filing" for c in tax)
    assert all(c["source_trust"] == "borrower_prepared" for c in statements)


def test_ocr_confidence_travels_with_ocr_chunks(chunked):
    deal = sorted(chunked["deals"])[0]
    tax = _for(chunked, deal, "09_tax_return_extract.pdf")
    assert tax
    for chunk in tax:
        assert chunk["extraction_method"] == "ocr"
        assert chunk["mean_word_confidence"] is not None
        assert chunk["mean_word_confidence"] > 80
