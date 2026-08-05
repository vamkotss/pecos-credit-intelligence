"""Fast unit tests for chunking (M4).

Nothing here opens a PDF or calls Tesseract. Page records are constructed by
hand, which is the only way to test boundary behaviour deterministically -- a
real page's line positions depend on ReportLab's layout and would make these
tests fail for reasons unrelated to chunking.
"""

from __future__ import annotations

from pecos.chunking import (
    MAX_CHARS,
    MIN_PROSE_CHARS,
    STATUS_DRAFT,
    STATUS_FINAL,
    STATUS_SUPERSEDED,
    TARGET_CHARS,
    chunk_deal,
    chunk_page,
    chunk_table,
    find_heading,
    profile_document,
    render_table_rows,
)


def _page(text: str, tables: list[dict] | None = None, **overrides) -> dict:
    record = {
        "deal_id": "PCP-0001",
        "document": "02_financial_statements_comparative.pdf",
        "page_number": 1,
        "method": "digital",
        "rotation_applied": 0,
        "text": text,
        "word_count": len(text.split()),
        "mean_word_confidence": None,
        "tables": tables or [],
        "scale_factor": 1,
        "scale_evidence": None,
        "extractor_version": "m3.1",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Document profiling
# ---------------------------------------------------------------------------


def test_known_filenames_map_to_a_profile():
    profile = profile_document("02_financial_statements_comparative.pdf")
    assert profile.kind == "financial_statements"
    assert profile.status == STATUS_FINAL
    assert profile.authority == 3

    bank = profile_document("08_bank_statements.pdf")
    assert bank.trust == "bank_issued"

    broker = profile_document("07_broker_email_thread.pdf")
    assert broker.trust == "third_party"


def test_an_unknown_filename_still_produces_a_usable_profile():
    """Real packages arrive with whatever the accountant saved the file as.
    An unrecognised name must not crash the chunker or lose the page."""
    profile = profile_document("Statements FINAL v3 (2).pdf")
    assert profile.kind == "unknown"
    assert profile.status == STATUS_FINAL


def test_a_draft_stamp_in_the_text_downgrades_the_status():
    """The near-duplicate defence, at its narrowest point.

    A document stamped DRAFT is a draft whatever the file is called, because
    filename conventions are not a schema outside this generated corpus.
    """
    profile = profile_document(
        "02_financial_statements_comparative.pdf",
        "Statements of Income (DRAFT)\nDRAFT -- subject to change",
    )
    assert profile.status == STATUS_DRAFT
    assert profile.authority == 1


def test_superseded_language_downgrades_the_status():
    profile = profile_document(
        "02_financial_statements_comparative.pdf",
        "Refer to the comparative statements for restated figures.",
    )
    assert profile.status == STATUS_SUPERSEDED
    assert profile.authority == 2


def test_text_can_only_downgrade_never_promote():
    """A draft that fails to say so is still a draft.

    The safe error is to under-trust a document, so nothing in the page text is
    allowed to raise a status. If this ever inverted, a draft whose filename was
    misread would outrank the final version it contradicts.
    """
    profile = profile_document(
        "10_financial_statements_draft.pdf",
        "These are the final audited statements, approved by the board.",
    )
    assert profile.status == STATUS_DRAFT


def test_authority_ranks_final_above_superseded_above_draft():
    final = profile_document("02_financial_statements_comparative.pdf")
    superseded = profile_document("03_financial_statements_superseded.pdf")
    draft = profile_document("10_financial_statements_draft.pdf")
    assert final.authority > superseded.authority > draft.authority


# ---------------------------------------------------------------------------
# Table chunking
# ---------------------------------------------------------------------------


def test_a_small_table_stays_in_one_piece():
    rows = [["Bucket", "Amount"], ["Current", "3,506,219"], ["Over 90 days", "267,218"]]
    groups = chunk_table(rows)
    assert len(groups) == 1
    assert groups[0] == rows


def test_a_large_table_splits_without_breaking_a_row():
    rows = [["Customer", "Balance", "% of AR"]]
    rows += [
        [f"Customer {i} Holdings", f"{i * 11_111:,}", f"{i}.0%"] for i in range(60)
    ]
    groups = chunk_table(rows)

    assert len(groups) > 1
    for group in groups:
        assert len(render_table_rows(group)) <= MAX_CHARS
        for row in group:
            assert len(row) == 3  # every row intact, never sliced


def test_every_split_group_repeats_the_header():
    """Without the header, the second half of a split table is a wall of
    unlabelled numbers -- which retrieves on the figures and then cannot say
    what they measure."""
    rows = [["Customer", "Balance", "% of AR"]]
    rows += [[f"Customer {i}", f"{i * 9_999:,}", f"{i}.0%"] for i in range(60)]
    groups = chunk_table(rows)
    assert len(groups) > 1
    for group in groups:
        assert group[0] == ["Customer", "Balance", "% of AR"]


def test_no_row_is_lost_when_a_table_splits():
    rows = [["h1", "h2"]] + [[f"label {i}", str(i)] for i in range(80)]
    groups = chunk_table(rows)
    recovered = [row for group in groups for row in group[1:]]
    assert recovered == rows[1:]


def test_an_oversized_single_row_still_gets_emitted():
    """Rows are never split, so the size ceiling yields to the structure rather
    than the other way round."""
    rows = [["h1", "h2"], ["label", "x" * (MAX_CHARS * 2)]]
    groups = chunk_table(rows)
    assert len(groups) == 1
    assert groups[0][1][1].startswith("x")


def test_pipe_rendering_keeps_fields_separable():
    text = render_table_rows([["Red River Distribution", "1,803,387", "34.2%"]])
    assert text == "Red River Distribution | 1,803,387 | 34.2%"
    assert text.count("|") == 2


# ---------------------------------------------------------------------------
# Prose chunking
# ---------------------------------------------------------------------------


def _long_prose(lines: int = 60) -> str:
    return "\n".join(
        f"Line {i}: the borrower reported steady trading across the period."
        for i in range(lines)
    )


def test_long_prose_is_split_into_several_chunks():
    chunks = chunk_page(_page(_long_prose()))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.char_count <= MAX_CHARS


def test_chunks_never_split_mid_line():
    """A financial statement line is an atomic fact. Splitting `EBITDA` from
    `2,418,000` leaves two chunks that each retrieve badly and neither answers
    anything."""
    text = _long_prose()
    source_lines = set(text.splitlines())
    for chunk in chunk_page(_page(text)):
        for line in chunk.text.splitlines():
            assert line in source_lines


def test_consecutive_chunks_overlap():
    """A fact sitting on a boundary should appear in both neighbours."""
    chunks = chunk_page(_page(_long_prose()))
    assert len(chunks) >= 2
    first_lines = chunks[0].text.splitlines()
    second_lines = chunks[1].text.splitlines()
    assert set(first_lines) & set(second_lines)


def test_chunks_are_close_to_the_target_size():
    chunks = chunk_page(_page(_long_prose(120)))
    # The last chunk is whatever is left over, so it is excluded.
    for chunk in chunks[:-1]:
        assert TARGET_CHARS * 0.5 <= chunk.char_count <= MAX_CHARS


def test_trivially_short_prose_is_not_indexed():
    """Stray headings and page furniture add noise without adding anything
    retrievable."""
    assert chunk_page(_page("Page 3")) == []
    assert len("Page 3") < MIN_PROSE_CHARS


def test_a_short_table_is_still_indexed():
    """Regression test. The minimum-length rule applies to prose only.

    An early version applied it to tables too and silently dropped a two-row
    table that rendered to 39 characters -- exactly the content the
    `table_only_fact` defect exists to punish losing. A short table is a
    structured fact; a short line of prose usually is not.
    """
    tables = [{"rows": [["Top customer", "34.2%"]], "source": "pdfplumber_lines"}]
    chunks = chunk_page(_page("Customer detail", tables))
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert table_chunks
    assert len(table_chunks[0].text) < MIN_PROSE_CHARS
    assert "34.2%" in table_chunks[0].text


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def test_table_content_is_not_duplicated_into_prose():
    """The extracted page text already contains every figure the tables hold.
    Emitting both unfiltered would index the same numbers twice, once with
    structure and once without, and the unstructured copy would compete with the
    good one in retrieval."""
    text = (
        "Statements of Income\n"
        "Revenue 32,041,248\n"
        "EBITDA 2,418,000\n"
        "Note: reviewed, not audited, prepared on the accrual basis."
    )
    tables = [
        {
            "rows": [["Revenue", "32,041,248"], ["EBITDA", "2,418,000"]],
            "source": "layout_clustering",
        }
    ]
    chunks = chunk_page(_page(text, tables))

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
    assert table_chunks
    assert "32,041,248" in table_chunks[0].text
    for chunk in prose_chunks:
        assert "32,041,248" not in chunk.text
        assert "accrual basis" in chunk.text or "Statements of Income" in chunk.text


def test_chunks_carry_the_anchor_back_to_the_page():
    chunks = chunk_page(_page(_long_prose(), page_number=7))
    for chunk in chunks:
        assert chunk.deal_id == "PCP-0001"
        assert chunk.document == "02_financial_statements_comparative.pdf"
        assert chunk.page_number == 7
        assert chunk.chunk_id.startswith("PCP-0001::")
        assert "p007" in chunk.chunk_id


def test_chunk_ids_are_unique_within_a_page():
    chunks = chunk_page(_page(_long_prose(120)))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_extraction_provenance_is_inherited_from_the_page():
    page = _page(
        _long_prose(),
        method="ocr",
        mean_word_confidence=94.5,
        scale_factor=1_000,
        scale_evidence="All amounts stated in thousands",
    )
    for chunk in chunk_page(page):
        assert chunk.extraction_method == "ocr"
        assert chunk.mean_word_confidence == 94.5
        assert chunk.scale_factor == 1_000


def test_headings_are_recognised_and_attached():
    assert find_heading("Statements of Income") == "Statements of Income"
    assert find_heading("Balance Sheets") == "Balance Sheets"
    assert find_heading("Revenue 32,041,248") is None
    assert find_heading("") is None


def test_chunking_is_deterministic():
    page = _page(_long_prose(90))
    first = [c.to_record() for c in chunk_page(page)]
    second = [c.to_record() for c in chunk_page(page)]
    assert first == second


# ---------------------------------------------------------------------------
# Context header
# ---------------------------------------------------------------------------


def test_the_context_header_restores_what_a_bare_chunk_loses():
    """`Revenue 32,041,248` could come from any document of any vintage. The
    header is what lets a query mentioning the tax return or the draft reach the
    right chunks at all."""
    chunk = chunk_page(_page(_long_prose()))[0]
    header = chunk.context_header
    assert "02_financial_statements_comparative.pdf" in header
    assert "page 1" in header
    assert "financial statements" in header
    assert chunk.embedding_text.startswith(header)


def test_a_non_final_status_is_visible_in_the_header():
    """Marked in the embedded text as well as in the metadata, so the
    distinction is available to similarity search and not only to a filter."""
    page = _page(_long_prose(), document="10_financial_statements_draft.pdf")
    chunk = chunk_page(page)[0]
    assert "DRAFT" in chunk.context_header
    assert chunk.authority == 1


def test_a_rescaled_page_announces_its_units_in_the_header():
    page = _page(
        _long_prose(),
        document="09_tax_return_extract.pdf",
        scale_factor=1_000,
        scale_evidence="in thousands",
    )
    chunk = chunk_page(page)[0]
    assert "units of 1,000" in chunk.context_header


# ---------------------------------------------------------------------------
# Deal assembly
# ---------------------------------------------------------------------------


def test_document_status_is_pooled_across_all_its_pages():
    """A DRAFT stamp usually appears only on page 1. Page 2 of a draft must not
    be treated as final merely because the stamp was not repeated on it."""
    pages = [
        _page("Statements of Income (DRAFT)\n" + _long_prose(20), page_number=1),
        _page(_long_prose(20), page_number=2),
    ]
    chunks = chunk_deal(pages)
    assert chunks
    assert all(c.doc_status == STATUS_DRAFT for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}
