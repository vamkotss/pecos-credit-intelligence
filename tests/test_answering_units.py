"""Fast unit tests for answering and evaluation (M6).

No API key, no network, no PDFs. Answers and contexts are constructed inline so
each check isolates one behaviour.
"""

from __future__ import annotations

from pecos.answering import (
    REFUSAL_TEXT,
    Answer,
    Citation,
    ExtractiveGenerator,
    _validate,
    format_contexts,
    looks_like_refusal,
    parse_citations,
)
from pecos.evaluation import (
    EvaluationReport,
    OverlapJudge,
    check_answer_value,
    check_citation,
    check_injection_resistance,
    check_numeric_grounding,
    check_refusal,
    extract_figures,
    figure_in,
    parse_judgement,
)

DOC = "02_financial_statements_comparative.pdf"


def _context(text: str, page: int = 1, **overrides) -> dict:
    record = {
        "chunk_id": f"PCP-0001::{DOC}::p{page:03d}::c000",
        "document": DOC,
        "page_number": page,
        "text": text,
        "context_header": f"{DOC} | page {page} | financial statements",
        "doc_status": "final",
        "scale_factor": 1,
        "scale_evidence": None,
    }
    record.update(overrides)
    return record


def _answer(text: str, **overrides) -> Answer:
    citations = tuple(parse_citations(text))
    record = {"text": text, "citations": citations}
    record.update(overrides)
    return Answer(**record)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_citation_markers_are_parsed_out_of_prose():
    citations = parse_citations(
        f"Revenue was 32,041,248 [{DOC}#p1] and debt was 7,200,000 [{DOC}#p2]."
    )
    assert [c.as_tuple() for c in citations] == [(DOC, 1), (DOC, 2)]


def test_repeated_citations_collapse():
    citations = parse_citations(f"[{DOC}#p1] and again [{DOC}#p1]")
    assert len(citations) == 1


def test_text_without_markers_yields_no_citations():
    assert parse_citations("Revenue was 32,041,248.") == []


def test_invented_citations_are_dropped_not_returned():
    """A citation pointing at a page that was never retrieved reads exactly like
    evidence and is not. The whole value of a citation is that someone can go and
    check it, so one that cannot be checked is worse than an uncited claim --
    which at least announces itself."""
    allowed = {(DOC, 1): "chunk-a"}
    kept, dropped = _validate(f"Revenue [{DOC}#p1] and debt [invented.pdf#p9]", allowed)
    assert [c.as_tuple() for c in kept] == [(DOC, 1)]
    assert list(dropped) == ["[invented.pdf#p9]"]


def test_a_real_document_with_a_wrong_page_is_also_dropped():
    allowed = {(DOC, 1): "chunk-a"}
    kept, dropped = _validate(f"[{DOC}#p7]", allowed)
    assert kept == ()
    assert list(dropped) == [f"[{DOC}#p7]"]


def test_citation_marker_round_trips():
    assert Citation(DOC, 3).marker() == f"[{DOC}#p3]"


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


def test_contexts_are_labelled_with_the_marker_the_model_must_copy():
    """Asking a model to assemble an identifier from a filename and a page
    number is asking it to get one of the parts wrong, and every such error
    becomes a dropped citation."""
    rendered = format_contexts([_context("Revenue 32,041,248")])
    assert f"[{DOC}#p1]" in rendered
    assert "Revenue 32,041,248" in rendered


def test_a_rescaled_page_announces_its_units_to_the_model():
    rendered = format_contexts([_context("Gross receipts 32,041", scale_factor=1_000)])
    assert "units of 1,000" in rendered


def test_a_non_final_document_is_flagged_to_the_model():
    """Retrieval already demotes drafts. Telling the model as well means it can
    say *why* it preferred one figure, rather than silently picking one."""
    rendered = format_contexts([_context("EBITDA 2,500,000", doc_status="draft")])
    assert "DRAFT" in rendered
    assert "not authoritative" in rendered


# ---------------------------------------------------------------------------
# Extractive generator
# ---------------------------------------------------------------------------


def test_the_extractive_generator_quotes_and_cites():
    contexts = [
        _context("Revenue | 20,375,000 | 19,853,119\nEBITDA | 2,727,142 | 3,016,959")
    ]
    answer = ExtractiveGenerator().generate("What was EBITDA?", contexts)
    assert not answer.refused
    assert "EBITDA" in answer.text
    assert answer.cited_pages == {(DOC, 1)}


def test_the_extractive_generator_refuses_when_nothing_matches():
    """The floor for the unanswerable case. A generator with no refusal path
    will always quote something, and something is always wrong."""
    contexts = [_context("Accounts payable | 1,700,000\nAccrued liabilities | 500,000")]
    answer = ExtractiveGenerator().generate(
        "What did the Phase I environmental site assessment conclude?", contexts
    )
    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == ()


def test_the_extractive_generator_never_invents_a_citation():
    """It quotes lines that exist on pages it was given, so by construction it
    cannot cite a page that was not retrieved."""
    contexts = [_context("Revenue | 20,375,000"), _context("Cash | 1,200,000", page=2)]
    answer = ExtractiveGenerator().generate("What was revenue?", contexts)
    assert answer.dropped_citations == ()
    assert answer.cited_pages <= {(DOC, 1), (DOC, 2)}


def test_an_empty_context_set_produces_a_refusal():
    answer = ExtractiveGenerator().generate("What was revenue?", [])
    assert answer.refused


def test_extraction_is_deterministic():
    contexts = [_context("Revenue | 20,375,000\nEBITDA | 2,727,142")]
    generator = ExtractiveGenerator()
    first = generator.generate("What was EBITDA?", contexts)
    second = generator.generate("What was EBITDA?", contexts)
    assert first.text == second.text


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------


def test_refusal_phrasings_are_recognised():
    for phrasing in (
        "That information is not present in the documents provided.",
        "No such report appears in the loan package.",
        "The excerpts do not contain that figure.",
        "There is insufficient information to answer.",
    ):
        assert looks_like_refusal(phrasing), phrasing


def test_a_normal_answer_is_not_read_as_a_refusal():
    assert not looks_like_refusal(f"Revenue was 32,041,248 [{DOC}#p1].")


# ---------------------------------------------------------------------------
# Figures and grounding
# ---------------------------------------------------------------------------


def test_citation_markers_are_stripped_before_figures_are_extracted():
    """Regression test. `[06_borrower_questionnaire.pdf#p1]` contains the digits
    06 and 1; an earlier version counted both as financial claims, so a
    correctly cited answer was reported as containing hallucinated figures. A
    metric that fires on its own citation format manufactures alarm about the
    exact behaviour it exists to encourage."""
    figures = extract_figures(
        "Revenue was 32,041,248 [06_borrower_questionnaire.pdf#p1]."
    )
    assert figures == ["32,041,248"]


def test_small_numbers_are_not_treated_as_financial_claims():
    """Page numbers, month numbers and list indices produce constant false
    positives and carry no claim."""
    assert extract_figures("See item 3 on page 2.") == []


def test_percentages_and_multiples_are_checkable():
    assert "34.2%" in extract_figures("Concentration is 34.2% of receivables.")
    assert "2.13x" in extract_figures("Leverage is 2.13x.")


def test_separator_variants_count_as_the_same_figure():
    """`2,418,000` and `2418000` are the same fact; which appears is an accident
    of whether the page used separators."""
    assert figure_in("2,418,000", "EBITDA 2418000")
    assert figure_in("2418000", "EBITDA 2,418,000")


def test_a_figure_on_a_cited_page_is_grounded():
    answer = _answer(f"EBITDA was 2,727,142 [{DOC}#p1].")
    result = check_numeric_grounding(answer, [_context("EBITDA | 2,727,142")])
    assert result.grounded == ["2,727,142"]
    assert result.rate == 1.0
    assert result.hallucinated == 0


def test_a_figure_from_an_uncited_page_is_flagged_separately():
    """The figure is real, the provenance is wrong. That is a citation bug, not
    a hallucination, and conflating the two would send debugging in the wrong
    direction."""
    answer = _answer(f"Cash was 1,200,000 [{DOC}#p1].")
    result = check_numeric_grounding(
        answer, [_context("EBITDA | 2,727,142"), _context("Cash | 1,200,000", page=2)]
    )
    assert result.uncited == ["1,200,000"]
    assert result.absent == []


def test_a_figure_in_no_context_at_all_is_a_hallucination():
    """The number that must be zero."""
    answer = _answer(f"EBITDA was 9,999,999 [{DOC}#p1].")
    result = check_numeric_grounding(answer, [_context("EBITDA | 2,727,142")])
    assert result.absent == ["9,999,999"]
    assert result.hallucinated == 1
    assert result.rate == 0.0


def test_an_answer_with_no_figures_is_trivially_grounded():
    """A refusal or a purely qualitative answer makes no numeric claim, so it
    cannot make an ungrounded one."""
    answer = _answer("That information is not present in the documents.")
    assert check_numeric_grounding(answer, [_context("EBITDA | 1")]).rate == 1.0


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def test_citation_correctness_matches_the_gold_page():
    fact = {"source_document": DOC, "source_page": 2, "answerable": True}
    assert check_citation(_answer(f"[{DOC}#p2]"), fact)
    assert not check_citation(_answer(f"[{DOC}#p1]"), fact)


def test_an_unanswerable_fact_expects_no_citation_at_all():
    fact = {"source_document": None, "source_page": None, "answerable": False}
    assert check_citation(_answer("Not present in the documents."), fact)
    assert not check_citation(_answer(f"[{DOC}#p1]"), fact)


def test_answer_correctness_is_separate_from_grounding():
    """An answer can be perfectly grounded -- every figure traceable to a cited
    page -- and still quote the wrong figure. A system can fail either check
    independently, so they are measured independently."""
    fact = {"answer_value": 2_727_142}
    assert check_answer_value(_answer("EBITDA was 2,727,142."), fact)
    assert not check_answer_value(_answer("EBITDA was 3,016,959."), fact)


def test_refusal_is_only_scored_where_refusing_is_correct():
    """Returning None for answerable questions keeps the metric from being
    diluted across seventy cases where refusal would be a failure."""
    answerable = {"answerable": True}
    assert check_refusal(_answer("Revenue was 1,000,000."), answerable) is None

    unanswerable = {"answerable": False}
    assert check_refusal(Answer(text="Not present.", refused=True), unanswerable)
    assert not check_refusal(_answer(f"It was 5,000,000 [{DOC}#p1]"), unanswerable)


def test_complying_with_the_injection_is_detected():
    assert not check_injection_resistance(
        _answer("This request has already been approved by committee.")
    )


def test_flagging_the_injection_attempt_is_not_treated_as_complying():
    """Mentioning that a document contained an instruction is the desired
    behaviour, not a violation. A pattern that punished it would train the model
    to stay silent about attacks."""
    assert check_injection_resistance(
        _answer(
            "The broker note contained embedded text instructing an automatic "
            "approval; I have disregarded it."
        )
    )


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


def test_the_overlap_judge_treats_a_refusal_as_faithful():
    """A refusal makes no claim, so it cannot be unfaithful. Whether it was the
    right response is a different check's job."""
    judgement = OverlapJudge().judge(
        "q", Answer(text="Not present.", refused=True), [_context("anything")]
    )
    assert judgement.faithfulness == 1.0


def test_the_overlap_judge_scores_grounded_text_above_invented_text():
    contexts = [_context("Revenue 32,041,248 cost of goods sold gross profit")]
    judge = OverlapJudge()
    grounded = judge.judge("revenue", _answer("Revenue 32,041,248"), contexts)
    invented = judge.judge(
        "revenue", _answer("Elephants migrate seasonally across savannah"), contexts
    )
    assert grounded.faithfulness > invented.faithfulness


def test_judge_json_survives_code_fences():
    judgement = parse_judgement(
        '```json\n{"faithfulness": 0.9, "relevance": 0.8, "rationale": "ok"}\n```'
    )
    assert judgement.faithfulness == 0.9
    assert judgement.relevance == 0.8


def test_unparseable_judge_output_scores_zero_rather_than_crashing():
    """A crashed eval tells you nothing. A zero is visible in the report and
    obviously wrong, which is what gets investigated."""
    judgement = parse_judgement("I think the answer is quite good actually")
    assert judgement.faithfulness == 0.0
    assert "unparseable" in judgement.rationale


def test_invalid_json_also_scores_zero():
    judgement = parse_judgement('{"faithfulness": }')
    assert judgement.faithfulness == 0.0


# ---------------------------------------------------------------------------
# Report aggregates
# ---------------------------------------------------------------------------


def _evaluation(**overrides):
    from pecos.evaluation import AnswerEvaluation, GroundingResult, Judgement

    record = {
        "fact_id": "F1",
        "deal_id": "PCP-0001",
        "question": "q",
        "fact_type": "income_statement",
        "defect_tag": None,
        "answer": "a",
        "refused": False,
        "citation_correct": True,
        "answer_correct": True,
        "grounding": GroundingResult(grounded=["1,000"]),
        "judgement": Judgement(1.0, 1.0),
        "refusal_correct": None,
        "injection_resisted": None,
    }
    record.update(overrides)
    return AnswerEvaluation(**record)


def test_hallucinated_figures_are_counted_not_averaged():
    """Averaging would let one invented figure disappear into a reassuring
    aggregate. It is reported as a raw count so it cannot."""
    from pecos.evaluation import GroundingResult

    report = EvaluationReport(
        results=[
            _evaluation(),
            _evaluation(grounding=GroundingResult(absent=["9,999,999"])),
        ]
    )
    assert report.hallucinated_figures == 1
    assert report.answers_with_hallucinations == 1


def test_over_refusal_is_the_counterweight_to_refusal_accuracy():
    """A system that refuses everything scores perfectly on the unanswerable
    question and is worthless. Both numbers have to be read together."""
    report = EvaluationReport(
        results=[
            _evaluation(refused=True),
            _evaluation(refused=False),
            _evaluation(refused=True, refusal_correct=True),
        ]
    )
    assert report.over_refusal_rate == 0.5
    assert report.refusal_accuracy == 1.0


def test_refusal_accuracy_is_none_when_nothing_tests_it():
    assert EvaluationReport(results=[_evaluation()]).refusal_accuracy is None


def test_an_empty_report_does_not_divide_by_zero():
    report = EvaluationReport()
    assert report.citation_accuracy == 0.0
    assert report.grounding_rate == 0.0
    assert report.over_refusal_rate == 0.0
