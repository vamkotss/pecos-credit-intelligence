"""Fast unit tests for the memo agent's nodes (M7).

The graph is tested node by node rather than only end to end, because that is
the whole reason for splitting it into a graph: a failure should be locatable to
one step instead of "the memo was wrong".
"""

from __future__ import annotations

from pecos.memo import (
    MEMO_QUESTIONS,
    compute_node,
    extract_figures,
    parse_money,
    should_revise,
    verify_node,
)
from pecos.policy import POLICY_CONSTANTS
from pecos.tools import CalculationLog, Input, leverage

DOC = "02_financial_statements_comparative.pdf"


def _context(text: str, page: int = 1, **overrides) -> dict:
    record = {
        "chunk_id": f"c{page}",
        "document": DOC,
        "page_number": page,
        "text": text,
        "context_header": "",
        "doc_status": "final",
        "scale_factor": 1,
        "scale_evidence": None,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parentheses_mean_negative():
    """The accounting convention, and a real hazard: reading (412,300) as
    positive silently inverts a loss into a profit and nothing downstream
    notices."""
    assert parse_money("(412,300)") == -412_300
    assert parse_money("412,300") == 412_300


def test_empty_and_non_numeric_cells_parse_to_nothing():
    assert parse_money("") is None
    assert parse_money("   ") is None
    assert parse_money("FY2025") == 2025  # a year is still a number


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extraction_takes_the_most_recent_year():
    """Comparative statements print oldest first, so the rightmost column is
    the current period. A memo built from the leftmost column would describe the
    business as it was three years ago, and every figure would be individually
    correct."""
    figures = extract_figures(
        [_context("Revenue | 20,375,000 | 19,853,119 | 20,453,629")]
    )
    assert figures["revenue"].value == 20_453_629


def test_extraction_records_the_page_each_figure_came_from():
    figures = extract_figures([_context("EBITDA | 1,000 | 2,575,918", page=1)])
    assert figures["ebitda"].document == DOC
    assert figures["ebitda"].page == 1


def test_draft_and_superseded_pages_never_supply_figures():
    """They can still be retrieved and cited when someone asks about them. What
    they must not do is silently supply the numbers a memo is built on."""
    contexts = [
        _context(
            "Revenue | 99,999,999",
            document="10_financial_statements_draft.pdf",
            doc_status="draft",
        ),
        _context("Revenue | 20,453,629", page=1),
    ]
    figures = extract_figures(contexts)
    assert figures["revenue"].value == 20_453_629


def test_a_rescaled_page_is_converted_on_extraction():
    figures = extract_figures([_context("Revenue | 32,041", scale_factor=1_000)])
    assert figures["revenue"].value == 32_041_000


def test_rows_that_are_not_statement_lines_are_ignored():
    figures = extract_figures([_context("Some narrative text about the borrower")])
    assert figures == {}


# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------


def _statement_context() -> dict:
    return _context(
        "Revenue | 20,453,629\n"
        "EBITDA | 2,575,918\n"
        "Interest expense | (697,988)\n"
        "Income tax provision | (198,223)\n"
        "Total current assets | 8,147,101\n"
        "Total current liabilities | 1,981,656\n"
        "Current portion of long-term debt | 615,848\n"
        "Long-term debt, net of current portion | 7,409,981\n"
        "Capital expenditures | (469,828)\n"
        "Repayment of long-term debt | (615,848)"
    )


def test_the_compute_node_produces_the_expected_metrics():
    figures = extract_figures([_statement_context()])
    log = compute_node({"figures": figures})["log"]
    names = {e.name for e in log.entries}
    assert "Total interest-bearing debt" in names
    assert "Total debt / EBITDA" in names
    assert "DSCR" in names
    assert "Current ratio" in names


def test_missing_inputs_skip_a_calculation_rather_than_faking_it():
    """A memo missing a DSCR line is obviously incomplete. A memo showing DSCR
    as 0.00 because a divisor was absent is obviously wrong in a way nobody
    notices."""
    figures = extract_figures([_context("Revenue | 20,453,629\nEBITDA | 2,575,918")])
    log = compute_node({"figures": figures})["log"]
    assert not any(e.name == "DSCR" for e in log.entries)


def test_compute_survives_having_nothing_to_work_with():
    assert compute_node({"figures": {}})["log"].entries == []


# ---------------------------------------------------------------------------
# Verify node
# ---------------------------------------------------------------------------


def _log_with_leverage() -> CalculationLog:
    log = CalculationLog()
    log.record(
        leverage(
            Input("total debt", 8_025_829, DOC, 2), Input("EBITDA", 2_575_918, DOC, 1)
        )
    )
    return log


def test_a_quoted_figure_passes_verification():
    state = {
        "draft": f"Revenue was $20,453,629 [{DOC}#p1].",
        "all_contexts": [_context("Revenue | 20,453,629")],
        "log": CalculationLog(),
    }
    assert verify_node(state)["issues"] == []


def test_a_computed_figure_passes_verification():
    """The distinction M6 could not make. This number is on no page, and it is
    not invented -- a recorded calculation produced it from cited inputs."""
    state = {
        "draft": "Leverage is 3.12x.",
        "all_contexts": [_context("EBITDA | 2,575,918")],
        "log": _log_with_leverage(),
    }
    assert verify_node(state)["issues"] == []


def test_an_invented_figure_fails_verification():
    """The case that must still fail. If this passes, the derived bucket has
    been widened into a loophole."""
    state = {
        "draft": "Revenue was $99,999,999.",
        "all_contexts": [_context("Revenue | 20,453,629")],
        "log": CalculationLog(),
    }
    assert verify_node(state)["issues"] == ["99,999,999"]


def test_verification_reads_every_chunk_on_a_page_not_just_the_last():
    """Regression test. Keying contexts by (document, page) dropped chunks: a
    statements page contributes both a table chunk and a prose chunk, and the
    second overwrote the first. The cash figure was then reported as ungrounded
    despite being printed on a cited page and read off that page by the
    extractor moments earlier."""
    state = {
        "draft": "Cash was $1,972,700.",
        "all_contexts": [
            _context("Balance Sheets\nAll amounts in US dollars.", page=2),
            _context("Cash and cash equivalents | 910,354 | 1,972,700", page=2),
        ],
        "log": CalculationLog(),
    }
    assert verify_node(state)["issues"] == []


def test_policy_thresholds_are_not_treated_as_borrower_figures():
    """ "Leverage of 3.12x is within the 3.5x policy limit" states one measured
    figure and one rule. The rule has no page to cite."""
    state = {
        "draft": "Leverage of 3.12x is within the 3.5x policy limit.",
        "all_contexts": [_context("EBITDA | 2,575,918")],
        "log": _log_with_leverage(),
    }
    assert verify_node(state)["issues"] == []
    assert "3.5x" in POLICY_CONSTANTS


def test_trailing_punctuation_is_not_part_of_a_figure():
    """`469,828,` inside a comma-separated list and `2025.` at the end of a
    sentence were both reported as ungrounded. The figure was on the page; the
    punctuation was not."""
    state = {
        "draft": "Inputs were capex=469,828, taxes=198,223. Year 2025.",
        "all_contexts": [
            _context(
                "Capital expenditures | 469,828\nIncome tax provision | 198,223\nFY2025"
            )
        ],
        "log": CalculationLog(),
    }
    assert verify_node(state)["issues"] == []


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


def test_a_clean_draft_does_not_trigger_a_revision():
    assert should_revise({"issues": [], "revisions": 0}) == "done"


def test_an_ungrounded_figure_triggers_exactly_one_revision():
    """A model that cannot fix a grounding failure when told exactly which
    figure is wrong will not fix it on the fourth attempt, and an unbounded loop
    turns a bad memo into a bad memo that costs ten times more."""
    assert should_revise({"issues": ["99,999"], "revisions": 0}) == "revise"
    assert should_revise({"issues": ["99,999"], "revisions": 1}) == "done"


def test_the_memo_has_a_fixed_set_of_sections():
    """A credit memo has a required shape. A committee does not want the
    sections to vary with sampling."""
    keys = [key for key, _ in MEMO_QUESTIONS]
    assert "request" in keys
    assert "performance" in keys
    assert "risks" in keys
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Fixes from the first full agent run
# ---------------------------------------------------------------------------


def test_a_short_figure_does_not_ground_itself_inside_a_longer_number():
    """The dangerous half of the verifier bug.

    `"79" in "1,079,456"` is True under plain substring matching, so an invented
    79% grounded itself against an unrelated seven-figure amount. The gate was
    simultaneously too strict about derived ratios and too lenient about short
    figures -- and the second error silently passed inventions, which is the
    direction that matters.
    """
    from pecos.evaluation import figure_in

    assert not figure_in("79", "Revenue was 1,079,456")
    assert not figure_in("4.0", "Total 14,033")
    assert figure_in("1,079,456", "Revenue was 1,079,456")


def test_separator_forms_still_match_after_the_boundary_fix():
    from pecos.evaluation import figure_in

    assert figure_in("2,418,000", "EBITDA 2418000")
    assert figure_in("2418000", "EBITDA | 2,418,000")


def test_a_document_name_in_prose_is_not_a_financial_claim():
    """Correctly reporting a superseded document -- "the file contains
    03_financial_statements_superseded.pdf" -- was contributing the figure 03."""
    from pecos.evaluation import extract_figures as figures_in_text

    figures = figures_in_text(
        "The file contains 03_financial_statements_superseded.pdf, marked SUPERSEDED."
    )
    assert figures == []


def test_extraction_keeps_every_year_not_just_the_latest():
    """Growth and change need the prior period. The first version discarded it,
    so the drafter computed those metrics itself and they had no provenance."""
    figures = extract_figures(
        [_context("Revenue | 20,375,000 | 19,853,119 | 20,453,629")]
    )
    assert figures["revenue"].series == (20_375_000, 19_853_119, 20_453_629)
    assert figures["revenue"].prior == 19_853_119


def test_receivables_merge_across_the_two_tables_on_the_page():
    """The page holds an ageing summary and a customer detail table, and
    chunking correctly emits them as two chunks. A parser that stopped at the
    first got the totals without the customers, or the reverse, and computed
    neither concentration figure."""
    from pecos.memo import extract_receivables

    contexts = [
        _context(
            "Bucket | Amount\nOver 90 days | 64,186\n"
            "Total accounts receivable | 2,745,829",
            document="05_ar_aging_and_concentration.pdf",
        ),
        _context(
            "Customer | Balance | % of AR\n"
            "Cimarron Holdings | 915,268 | 33.3%\n"
            "Nueces Supply Co. | 669,470 | 24.4%",
            document="05_ar_aging_and_concentration.pdf",
        ),
    ]
    book = extract_receivables(contexts)
    assert book.total == 2_745_829
    assert book.over_90 == 64_186
    assert len(book.customers) == 2


def test_a_draft_receivables_page_is_ignored():
    from pecos.memo import extract_receivables

    contexts = [
        _context(
            "Total accounts receivable | 99,999,999",
            document="05_ar_aging_and_concentration.pdf",
            doc_status="draft",
        )
    ]
    assert extract_receivables(contexts) is None


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_pro_forma_leverage_is_reconstructible():
    """The case that ended two rounds of enumerating metrics.

    "The requested facility of $12,800,000 would increase total debt to
    approximately $21.3 million, raising pro forma leverage to approximately
    3.95x EBITDA" -- every input on a cited page, and the sentence that flips
    the recommendation from PROCEED to DEFER. No list of metrics would have
    contained it, because the point of employing an analyst is that they compute
    things nobody listed in advance.
    """
    from pecos.reconstruct import Value, explain

    values = [
        Value("Total interest-bearing debt", 8_508_027),
        Value("Facility requested", 12_800_000),
        Value("EBITDA", 5_387_402),
    ]
    total = explain("21.3", values)
    assert total is not None
    assert "Facility requested" in total.expression

    ratio = explain("3.95x", values)
    assert ratio is not None
    assert ratio.operation == "pro_forma_ratio"


def test_a_truncated_figure_still_reconstructs():
    """The memo wrote "approximately 3.95x" for 3.9552 -- truncated, not
    rounded. A rounding-only check rejected the one derivation the whole feature
    exists to accept."""
    from pecos.reconstruct import Value, explain

    values = [
        Value("Total interest-bearing debt", 8_508_027),
        Value("Facility requested", 12_800_000),
        Value("EBITDA", 5_387_402),
    ]
    assert explain("3.95x", values) is not None


def test_non_financial_numbers_are_not_operands():
    """Regression test for the failure that made the gate worthless.

    An earlier version scraped every number off the retrieved pages, so a NAICS
    code and an employee count became arithmetic inputs. All twelve memos then
    "verified" on derivations like `423,628: NAICS code` and
    `60.3%: 1999 and employs 153 people / Total debt / EBITDA`. A gate that
    approves everything is indistinguishable from no gate.
    """
    from pecos.reconstruct import Value, explain, values_from_figures
    from pecos.tools import CalculationLog

    class _Figure:
        def __init__(self, label, value):
            self.label, self.value = label, value

    figures = {"revenue": _Figure("Revenue", 52_037_306)}
    values = values_from_figures(figures, CalculationLog())
    assert [v.label for v in values] == ["Revenue"]

    assert explain("423,628", [Value("Revenue", 52_037_306)]) is None


def test_a_figure_that_follows_from_nothing_is_still_rejected():
    """The gate must still fail. If everything reconstructs, it is a rubber
    stamp rather than a check."""
    from pecos.reconstruct import Value, explain

    values = [Value("EBITDA", 5_387_402), Value("Revenue", 52_037_306)]
    assert explain("99,999,999", values) is None
    assert explain("777,777", values) is None


def test_low_precision_figures_cannot_be_reconstructed():
    """Two significant figures is not enough evidence.

    `6.2` matched "total debt - net income" and `11` matched "total debt + net
    income", both plainly coincidental. With so few digits some combination
    lands in the bucket almost regardless, so such figures must be quoted or
    computed instead.
    """
    from pecos.reconstruct import Value, explain

    values = [
        Value("Total interest-bearing debt", 8_508_027),
        Value("Net income", 2_354_191),
        Value("Revenue", 52_037_306),
    ]
    assert explain("6.2", values) is None
    assert explain("11", values) is None


def test_a_reconstruction_reports_the_arithmetic_it_claims():
    """Reported, never assumed. Even narrowed, a match can be coincidental, so
    the derivation goes in front of a reviewer instead of being treated as
    proof."""
    from pecos.reconstruct import Value, explain

    result = explain(
        "45.7%", [Value("Gross profit", 23_761_530), Value("Revenue", 52_037_306)]
    )
    assert result is not None
    assert "Gross profit" in result.expression
    assert "Revenue" in result.expression


def test_scale_words_are_handled():
    """A memo writing "$21.3 million" prints the token 21.3."""
    from pecos.reconstruct import Value, explain

    assert explain("21.3", [Value("Total debt", 21_308_027)]) is not None
    assert explain("32,041", [Value("Revenue", 32_041_000)]) is not None


def test_a_percentage_cannot_be_reconstructed_by_a_dollar_amount():
    """Unit confusion produced derivations that were arithmetically true and
    financially meaningless: `4.97x` matched a dollar amount, `1.83` matched a
    percentage of 1,830 after being rescaled by a thousand, and `24.2%` matched
    capital expenditure divided by the current ratio.

    A percentage is already normalised, so rescaling one is meaningless; and a
    ratio is a result, not an operand.
    """
    from pecos.reconstruct import RATIO, Value, explain

    values = [
        Value("Total equity", 17_176_913),
        Value("Income tax provision", -1_000_000),
        Value("Capital expenditures", 2_242_319),
        Value("Current ratio", 4.20, kind=RATIO),
        Value("TOTAL ASSETS", 28_879_019),
        Value("Depreciation and amortisation", 1_578_000),
    ]
    assert explain("4.97x", values) is None
    assert explain("24.2%", values) is None
    assert explain("1.83", values) is None


def test_a_ratio_is_never_used_as_an_operand():
    from pecos.reconstruct import RATIO, USD, Value, explain

    values = [
        Value("Capital expenditures", 2_242_319, kind=USD),
        Value("Current ratio", 4.20, kind=RATIO),
    ]
    assert explain("533,885", values) is None


def test_unit_kinds_are_inferred_from_labels():
    from pecos.reconstruct import PERCENT, RATIO, USD, infer_kind

    assert infer_kind("Total debt / EBITDA") == RATIO
    assert infer_kind("DSCR") == RATIO
    assert infer_kind("Gross margin") == PERCENT
    assert infer_kind("Revenue growth") == PERCENT
    assert infer_kind("Total current assets") == USD
