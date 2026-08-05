"""Fast unit tests for the ingestion logic (M3).

Nothing here touches a PDF, renders an image, or calls Tesseract, so the whole
file runs in milliseconds and stays in the default test path. The slow
end-to-end checks live in `test_ingest.py`.

The line-grouping tests are the interesting ones. They construct word boxes by
hand with a known skew, which is the only way to test the skew behaviour
deterministically -- reproducing it through a real scan would depend on
Tesseract's exact output and would fail for reasons unrelated to the algorithm.
"""

from __future__ import annotations

from pecos.ingest import (
    ExtractedTable,
    Word,
    _clean_cell,
    _group_into_lines,
    _orientation_score,
    _overlap_ratio,
    cluster_words_into_tables,
    detect_scale,
)


def _word(text: str, left: int, top: int, width: int = 40, height: int = 20) -> Word:
    return Word(
        text=text, left=left, top=top, width=width, height=height, confidence=95.0
    )


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------


def test_plain_text_has_no_scale_factor():
    factor, evidence = detect_scale("Revenue 32,041,248\nCost of goods sold 19,565,501")
    assert factor == 1
    assert evidence is None


def test_thousands_note_is_detected_in_its_usual_wording():
    """The exact phrasing used by the corpus, plus the variants that show up on
    real statements. All must be caught: this is the failure mode that changes a
    figure by three orders of magnitude without looking wrong."""
    for phrasing in (
        "(All amounts stated in thousands of dollars unless otherwise indicated.)",
        "Balance Sheet (in thousands)",
        "Revenue ($000s)",
        "Summary (000s)",
        "AMOUNTS STATED IN THOUSANDS",
    ):
        factor, evidence = detect_scale(phrasing)
        assert factor == 1_000, phrasing
        assert evidence


def test_millions_note_is_detected_and_distinguished():
    factor, _ = detect_scale("Consolidated results in millions")
    assert factor == 1_000_000


def test_scale_evidence_quotes_the_surrounding_text():
    """The matched note travels with the page so a reviewer can check the call
    rather than trust it."""
    text = "Statements of Income\n(All amounts stated in thousands.)\nRevenue 32,041"
    _, evidence = detect_scale(text)
    assert "thousands" in evidence.lower()
    assert "Revenue" in evidence or "Income" in evidence


def test_ocr_noise_around_the_note_does_not_defeat_detection():
    """Units notes are set in small type on scanned pages, which is exactly
    where OCR is least reliable. The patterns are loose on purpose."""
    factor, _ = detect_scale("(All amounts stated 1n thousands of dollars)")
    assert factor == 1_000


def test_ordinary_prose_about_thousands_does_not_trigger_a_rescale():
    """The catch-all pattern must not fire on narrative text. A false positive
    here divides a correct figure by a thousand, which is just as damaging as
    the miss it was added to prevent."""
    for prose in (
        "The company serves thousands of customers across three states.",
        "Thousands of units shipped in the fourth quarter.",
    ):
        factor, _ = detect_scale(prose)
        assert factor == 1, prose


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------


def test_words_at_the_same_height_form_one_line():
    words = [_word("Revenue", 100, 200), _word("32,041,248", 400, 201)]
    lines = _group_into_lines(words)
    assert len(lines) == 1
    assert [w.text for w in lines[0]] == ["Revenue", "32,041,248"]


def test_words_at_different_heights_form_separate_lines():
    words = [_word("Revenue", 100, 200), _word("Cost of goods", 100, 240)]
    assert len(_group_into_lines(words)) == 2


def test_a_skewed_line_stays_a_single_line():
    """The regression that motivated the whole algorithm.

    A line drifting 24 pixels across the page -- roughly 0.7 degrees of skew
    over 2,000 pixels, which is what the scans in this corpus carry -- must not
    fragment. The earlier implementation compared each word's top coordinate
    against a fixed tolerance and split this into six lines, which destroyed
    every table on the rotated bank statement page.
    """
    words = [_word(f"w{i}", 100 + i * 350, 200 + i * 4) for i in range(7)]
    lines = _group_into_lines(words)
    assert len(lines) == 1, f"skewed line fragmented into {len(lines)} lines"
    assert [w.text for w in lines[0]] == [f"w{i}" for i in range(7)]


def test_skew_does_not_merge_genuinely_separate_lines():
    """The other side of the same coin. Tolerating drift must not become
    tolerating everything, or the whole page collapses into one line."""
    top_line = [_word(f"a{i}", 100 + i * 350, 200 + i * 4) for i in range(5)]
    next_line = [_word(f"b{i}", 100 + i * 350, 245 + i * 4) for i in range(5)]
    lines = _group_into_lines(top_line + next_line)
    assert len(lines) == 2


def test_lines_come_back_in_reading_order():
    words = [_word("second", 100, 300), _word("first", 100, 200)]
    lines = _group_into_lines(words)
    assert [line[0].text for line in lines] == ["first", "second"]


def test_overlap_ratio_is_scale_free():
    small = _word("x", 0, 100, height=10)
    large = _word("X", 50, 95, height=30)
    # The small box sits entirely inside the large one's vertical span.
    assert _overlap_ratio(small, large) == 1.0
    apart = _word("y", 50, 200, height=10)
    assert _overlap_ratio(small, apart) == 0.0


# ---------------------------------------------------------------------------
# Table clustering
# ---------------------------------------------------------------------------


def _statement_words() -> list[Word]:
    """Three label/value rows laid out like a financial statement."""
    rows = [
        ("Revenue", "32,041,248"),
        ("Cost of goods sold", "19,565,501"),
        ("EBITDA", "2,418,000"),
    ]
    words: list[Word] = []
    for r, (label, value) in enumerate(rows):
        top = 200 + r * 40
        for c, token in enumerate(label.split()):
            words.append(_word(token, 100 + c * 45, top))
        words.append(_word(value, 700, top))
    return words


def test_a_whitespace_separated_table_is_recovered():
    tables = cluster_words_into_tables(_statement_words(), page_width=1000)
    assert len(tables) == 1
    table = tables[0]
    assert table.n_rows == 3
    assert table.n_cols == 2
    assert table.rows[0] == ("Revenue", "32,041,248")
    assert table.rows[1][0] == "Cost of goods sold"


def test_clustered_tables_are_labelled_as_inferred():
    """Structure inferred from whitespace is not the same as structure read from
    ruling lines. Downstream code that treats them as equally reliable will be
    wrong about scanned tables, so the source is recorded."""
    tables = cluster_words_into_tables(_statement_words(), page_width=1000)
    assert tables[0].source == "layout_clustering"


def test_short_runs_are_not_treated_as_tables():
    """Two label/value pairs are a heading with a figure beside it, not a table.
    Promoting them would fill the index with noise."""
    words = [_word("Total", 100, 200), _word("5,000", 700, 200)]
    assert cluster_words_into_tables(words, page_width=1000) == []


def test_prose_produces_no_table():
    words = [_word(f"word{i}", 100 + i * 60, 200) for i in range(8)]
    assert cluster_words_into_tables(words, page_width=1000) == []


def test_empty_input_is_handled():
    assert cluster_words_into_tables([], page_width=1000) == []
    assert _group_into_lines([]) == []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_cell_cleaning_normalises_whitespace_and_none():
    assert _clean_cell(None) == ""
    assert (
        _clean_cell("  Total   accounts \n receivable ") == "Total accounts receivable"
    )


def test_table_cells_flattens_and_drops_blanks():
    table = ExtractedTable(rows=(("A", ""), ("B", "C")), source="layout_clustering")
    assert table.cells() == ["A", "B", "C"]
    assert table.n_rows == 2
    assert table.n_cols == 2


# ---------------------------------------------------------------------------
# Orientation scoring
# ---------------------------------------------------------------------------


def test_upright_text_scores_far_higher_than_sideways_text():
    """The signal that distinguishes a page read correctly from one read
    sideways, tested on synthetic boxes so it cannot depend on Tesseract.

    Upright: lines run across the page. Sideways: each "line" is one stacked
    word spanning almost nothing. Confidence is held identical at 95 in both
    cases, which is the whole point -- Tesseract really is confident about the
    glyphs on a sideways page, so confidence alone cannot tell them apart.
    """
    upright = [
        _word(f"w{r}{c}", 50 + c * 200, 100 + r * 40)
        for r in range(5)
        for c in range(4)
    ]
    sideways = [_word(f"w{i}", 50, 100 + i * 40) for i in range(20)]

    upright_score = _orientation_score(upright, page_width=1000)
    sideways_score = _orientation_score(sideways, page_width=1000)

    assert upright_score > 4 * sideways_score
    assert upright_score >= 20.0
    assert sideways_score < 20.0


def test_orientation_score_is_zero_on_a_nearly_blank_page():
    """Too few words to judge. Returning a confident score here would let a
    blank page silently drive a rotation decision."""
    assert _orientation_score([_word("x", 10, 10)], page_width=1000) == 0.0
    assert _orientation_score([], page_width=1000) == 0.0
