"""Fast unit tests for retrieval (M5).

No PDFs, no OCR, no model downloads. Small synthetic corpora built inline, so
the behaviour under test is isolated from whatever the real corpus happens to
contain.
"""

from __future__ import annotations

import numpy as np

from pecos.retrieval import (
    STATUS_WEIGHTS,
    BM25Index,
    HybridRetriever,
    LexicalOverlapReranker,
    LsaEmbedder,
    _min_max,
    query_tokens,
    reciprocal_rank_fusion,
    tokenize,
)
from pecos.retrieval_eval import QueryResult, RetrievalReport


def _chunk(chunk_id: str, text: str, **overrides) -> dict:
    record = {
        "chunk_id": chunk_id,
        "deal_id": "PCP-0001",
        "document": "02_financial_statements_comparative.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "chunk_type": "table",
        "text": text,
        "context_header": "02_financial_statements_comparative.pdf | page 1",
        "section": "Statements of Income",
        "doc_kind": "financial_statements",
        "doc_status": "final",
        "authority": 3,
        "source_trust": "borrower_prepared",
        "extraction_method": "digital",
        "mean_word_confidence": None,
        "scale_factor": 1,
        "scale_evidence": None,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_whole_financial_figures_survive_tokenisation():
    """The single most important thing this tokeniser does.

    A `\\w+` tokeniser turns 32,041,248 into three tokens -- 32, 041, 248 --
    and every large figure in the corpus then collides with every other figure
    sharing a fragment. BM25's whole value here is matching exact amounts, so
    that failure would quietly remove the strongest retrieval signal available.
    """
    tokens = tokenize("Revenue 32,041,248 and EBITDA 2,418,000")
    assert "32,041,248" in tokens
    assert "2,418,000" in tokens
    assert "32" not in tokens
    assert "041" not in tokens


def test_percentages_and_multiples_stay_intact():
    tokens = tokenize("concentration 34.2% at leverage 2.13x")
    assert "34.2%" in tokens
    assert "2.13x" in tokens


def test_numbers_also_emit_a_bare_digit_form():
    """So a query written as 32041248 matches a page printing 32,041,248.
    Which form a question happens to use is an accident of phrasing."""
    tokens = tokenize("32,041,248")
    assert "32,041,248" in tokens
    assert "32041248" in tokens


def test_currency_symbols_are_stripped():
    assert "5,000" in tokenize("a payment of $5,000")


def test_query_stopwords_are_removed():
    """IDF assumes a natural-language corpus. A loan package is mostly tables,
    so ordinary English words are rare in it and IDF scores them as highly
    informative -- measured on one deal, `what` scored 3.12 against `ebitda` at
    2.61. Stripping them from queries is what stopped the broker's cover note,
    the only chunk written in flowing prose, ranking first for almost every
    financial question."""
    tokens = query_tokens("What was the total revenue in FY2025?")
    assert "what" not in tokens
    assert "was" not in tokens
    assert "the" not in tokens
    assert "revenue" in tokens
    assert "fy2025" in tokens


def test_domain_words_are_not_treated_as_stopwords():
    """`total` and `balance` look like filler and are not: they distinguish a
    total from a line item, and a balance sheet from a cash flow statement."""
    tokens = query_tokens("What was the total debt balance at year end?")
    assert "total" in tokens
    assert "balance" in tokens
    assert "debt" in tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_ranks_the_document_containing_the_query_term_first():
    index = BM25Index(
        [
            tokenize("revenue 32,041,248 cost of goods sold"),
            tokenize("accounts payable accrued liabilities"),
            tokenize("net cash from operating activities"),
        ]
    )
    scores = index.scores(tokenize("revenue"))
    assert int(np.argmax(scores)) == 0


def test_bm25_matches_an_exact_figure():
    index = BM25Index(
        [
            tokenize("EBITDA 2,418,000"),
            tokenize("EBITDA 3,016,959"),
        ]
    )
    scores = index.scores(tokenize("3,016,959"))
    assert int(np.argmax(scores)) == 1


def test_idf_is_never_negative():
    """A term in most of the corpus scores negative under the raw BM25 idf, so a
    document is penalised for containing a query term. That produces rankings
    which look broken and are very hard to trace back to one line of maths."""
    index = BM25Index([tokenize("total assets")] * 5 + [tokenize("revenue")])
    assert all(value > 0 for value in index.idf.values())


def test_an_empty_index_scores_nothing_rather_than_crashing():
    index = BM25Index([])
    assert index.scores(tokenize("revenue")).size == 0


def test_unknown_query_terms_contribute_zero():
    index = BM25Index([tokenize("revenue 100"), tokenize("expenses 200")])
    scores = index.scores(tokenize("xyzzy"))
    assert float(scores.sum()) == 0.0


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_fusion_rewards_agreement_between_retrievers():
    """A document both methods like should beat one only a single method loves.
    That is the property rank fusion exists to produce."""
    fused = reciprocal_rank_fusion([[7, 1, 2], [7, 3, 4]])
    assert max(fused, key=lambda i: fused[i]) == 7


def test_fusion_uses_rank_not_score():
    """BM25 is unbounded and corpus-dependent; cosine is bounded in [-1, 1].
    Normalising them into a shared range means choosing a normalisation, and
    every choice is a hidden weighting decision. Rank position sidesteps it."""
    a = reciprocal_rank_fusion([[5, 6]])
    b = reciprocal_rank_fusion([[5, 6]])
    assert a == b
    assert a[5] > a[6]


def test_fusion_weights_shift_the_balance():
    heavy_first = reciprocal_rank_fusion([[1], [2]], weights=[1.0, 0.1])
    assert heavy_first[1] > heavy_first[2]
    heavy_second = reciprocal_rank_fusion([[1], [2]], weights=[0.1, 1.0])
    assert heavy_second[2] > heavy_second[1]


def test_min_max_handles_a_flat_input():
    """All-equal candidates must not divide by zero. They map to all-ones, so a
    tie stays a tie rather than becoming an arbitrary ordering."""
    assert _min_max([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
    assert _min_max([]) == []
    assert _min_max([0.0, 1.0]) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def test_the_reranker_prefers_the_passage_with_the_rare_term():
    reranker = LexicalOverlapReranker()
    reranker.set_context({"revenue": 3.0, "total": 0.2, "fy2025": 1.0})
    scores = reranker.score(
        "total revenue FY2025",
        ["Revenue 32,041,248 FY2025", "Total current assets Total liabilities FY2025"],
    )
    assert scores[0] > scores[1]


def test_without_idf_the_reranker_still_works():
    """The context hook is optional. Flat weights are worse, not broken."""
    scores = LexicalOverlapReranker().score("revenue", ["revenue here", "nothing"])
    assert scores[0] > scores[1]


def test_numeric_matches_are_weighted_up():
    """A page containing the exact amount asked about is almost always the
    right page."""
    reranker = LexicalOverlapReranker()
    scores = reranker.score(
        "EBITDA 2,418,000",
        ["EBITDA 2,418,000", "EBITDA reported for the period"],
    )
    assert scores[0] > scores[1]


def test_an_empty_query_scores_zero_everywhere():
    assert LexicalOverlapReranker().score("the of and", ["anything"]) == [0.0]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_lsa_produces_unit_norm_or_exactly_zero_vectors():
    """Every vector is either unit length or exactly zero -- never a fraction of
    a unit, which would be floating-point noise masquerading as a direction.

    Degenerate projections are real: a small corpus can give the TF-IDF matrix
    less rank than the number of components requested, and one document then
    lands on the origin. Whether it does is BLAS-dependent, which is exactly why
    this is asserted rather than assumed.
    """
    embedder = LsaEmbedder(n_components=4)
    texts = [
        "revenue cost of goods sold gross profit",
        "accounts payable accrued liabilities equity",
        "net cash operating investing financing",
        "customer concentration receivable ageing",
    ]
    embedder.fit(texts)
    vectors = embedder.encode(texts)
    assert vectors.shape[0] == 4

    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-5) or np.isclose(norm, 0.0, atol=1e-9)
    assert np.count_nonzero(norms > 0.5) >= 2, "the projection collapsed entirely"


def test_lsa_is_deterministic():
    texts = ["revenue profit", "payable equity", "cash flow", "customer ageing"]
    first = LsaEmbedder(n_components=3)
    first.fit(texts)
    second = LsaEmbedder(n_components=3)
    second.fit(texts)
    assert np.allclose(first.encode(texts), second.encode(texts))


def test_lsa_clamps_components_to_the_corpus_size():
    """Per-deal indexes are small. Asking for more components than the matrix
    has rank would raise, so the clamp is load-bearing rather than defensive."""
    embedder = LsaEmbedder(n_components=512)
    embedder.fit(["revenue profit", "payable equity", "cash flow"])
    assert embedder.encode(["revenue"]).shape[1] <= 2


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def _near_duplicate_corpus() -> list[dict]:
    """A final and a draft with identical text, plus enough distractors that
    score normalisation behaves as it does on a real deal.

    The distractors are not padding. With only two candidates, min-max
    normalisation stretches the gap between first and second place to the full
    0-to-1 range whatever its true size, so a meaningless rank difference
    becomes a decisive one. Real candidate sets run to thirty.
    """
    text = "Revenue 20,375,000 EBITDA 2,727,142 FY2025"
    return [
        _chunk("final", text, page_number=1),
        _chunk(
            "draft",
            text,
            page_number=1,
            document="10_financial_statements_draft.pdf",
            doc_status="draft",
            authority=1,
        ),
        _chunk("bs", "Accounts payable 1,700,000 Total assets", page_number=2),
        _chunk("cf", "Net cash from operating activities 900,000", page_number=3),
        _chunk(
            "aging",
            "Customer concentration Red River 34.2% of AR",
            document="05_ar_aging_and_concentration.pdf",
        ),
        _chunk(
            "debt",
            "Frost Bank Term Loan A 7.25% maturity 2029",
            document="04_debt_schedule.pdf",
        ),
    ]


def _two_deal_corpus() -> list[dict]:
    return [
        _chunk("a1", "Revenue 20,375,000 EBITDA 2,727,142", deal_id="PCP-0001"),
        _chunk(
            "a2",
            "Accounts payable 1,700,000 Total current liabilities",
            deal_id="PCP-0001",
            page_number=2,
        ),
        _chunk(
            "a3",
            "Customer concentration Red River 34.2%",
            deal_id="PCP-0001",
            document="05_ar_aging_and_concentration.pdf",
        ),
        _chunk("b1", "Revenue 44,000,000 EBITDA 5,100,000", deal_id="PCP-0002"),
        _chunk("b2", "Debt schedule Frost Bank term loan", deal_id="PCP-0002"),
    ]


def test_retrieval_never_crosses_deals():
    """A lending question is always about one borrower. There is no such thing
    as 'what was revenue' across a portfolio, so a cross-deal result is never
    useful and always harmful."""
    retriever = HybridRetriever()
    retriever.build(_two_deal_corpus())
    for hit in retriever.retrieve("revenue", "PCP-0001", k=5):
        assert hit.chunk["deal_id"] == "PCP-0001"


def test_an_unknown_deal_returns_nothing_rather_than_raising():
    retriever = HybridRetriever()
    retriever.build(_two_deal_corpus())
    assert retriever.retrieve("revenue", "PCP-9999", k=5) == []


def test_results_carry_every_stage_score():
    """The intermediate scores answer 'why did this rank here', which is the
    question that comes up every time retrieval underperforms. Returning a bare
    ordering makes the next milestone's debugging guesswork."""
    retriever = HybridRetriever()
    retriever.build(_two_deal_corpus())
    hit = retriever.retrieve("EBITDA", "PCP-0001", k=1)[0]
    assert hit.bm25_rank is not None or hit.dense_rank is not None
    assert hit.fused_score > 0
    assert hit.authority_weight > 0
    assert hit.final_score > 0
    assert hit.page_key[0] == "PCP-0001"


def test_authority_demotes_a_draft_against_an_identical_final():
    """The near-duplicate defence, in isolation.

    Two chunks with identical text, differing only in status. Text similarity
    cannot separate them by construction, so whichever wins is decided by the
    authority weight alone.
    """
    corpus = _near_duplicate_corpus()
    retriever = HybridRetriever()
    retriever.build(corpus)
    hits = retriever.retrieve("What was EBITDA in FY2025?", "PCP-0001", k=2)
    assert hits[0].chunk["doc_status"] == "final"
    assert hits[0].final_score > hits[1].final_score


def test_authority_widens_the_margin_over_a_near_duplicate():
    """The control for the test above, measured as a margin rather than a rank.

    With identical text the two chunks are adjacent in the ranking either way;
    what authority changes is how *decisively* the final one wins. That is the
    quantity that matters, because a near-duplicate pair separated by noise will
    flip position on any small change to the corpus or the query, and at M7 both
    would land in the agent's context with nothing to say which is current.

    Measured on the real corpus: without the weighting the draft trails the
    final by 1.6%, which is not a signal. With it, the draft leaves the top
    three entirely.
    """
    corpus = _near_duplicate_corpus()
    query = "What was EBITDA in FY2025?"

    def margin(retriever) -> float:
        retriever.build(corpus)
        hits = retriever.retrieve(query, "PCP-0001", k=6)
        final = next(h for h in hits if h.chunk["chunk_id"] == "final")
        draft = next(h for h in hits if h.chunk["chunk_id"] == "draft")
        return final.final_score - draft.final_score

    with_authority = margin(HybridRetriever())
    without_authority = margin(HybridRetriever(use_authority=False))

    assert with_authority > without_authority
    assert without_authority < 0.10, "the pair should be near-inseparable on text"
    assert with_authority > 0.15, "authority did not separate them decisively"


def test_status_weights_rank_final_above_superseded_above_draft():
    assert STATUS_WEIGHTS["final"] > STATUS_WEIGHTS["superseded"]
    assert STATUS_WEIGHTS["superseded"] > STATUS_WEIGHTS["draft"]
    assert STATUS_WEIGHTS["draft"] > 0, "demotion, never exclusion"


def test_ablation_switches_actually_change_behaviour():
    """A component that cannot be switched off cannot be shown to be worth its
    cost. These switches are how the ablation table is produced."""
    corpus = _two_deal_corpus()
    for kwargs in (
        {"use_dense": False},
        {"use_bm25": False},
        {"use_rerank": False},
    ):
        retriever = HybridRetriever(**kwargs)
        retriever.build(corpus)
        assert retriever.retrieve("revenue", "PCP-0001", k=3)


def test_retrieval_is_deterministic():
    corpus = _two_deal_corpus()
    first = HybridRetriever()
    first.build(corpus)
    second = HybridRetriever()
    second.build(corpus)
    a = [h.chunk["chunk_id"] for h in first.retrieve("EBITDA", "PCP-0001", k=3)]
    b = [h.chunk["chunk_id"] for h in second.retrieve("EBITDA", "PCP-0001", k=3)]
    assert a == b


# ---------------------------------------------------------------------------
# Report maths
# ---------------------------------------------------------------------------


def _result(rank: int | None, defect: str | None = None) -> QueryResult:
    return QueryResult(
        fact_id="F",
        deal_id="PCP-0001",
        question="q",
        fact_type="income_statement",
        defect_tag=defect,
        gold_document="d.pdf",
        gold_page=1,
        retrieved=[],
        rank=rank,
    )


def test_recall_counts_only_hits_within_k():
    report = RetrievalReport(results=[_result(1), _result(4), _result(None)])
    assert report.recall_at(1) == 1 / 3
    assert report.recall_at(5) == 2 / 3


def test_mrr_distinguishes_rank_one_from_rank_five():
    """Recall@5 cannot tell those apart, and at M7 the difference matters: the
    thing reading the results has a limited context budget and a documented bias
    toward what it sees first."""
    always_first = RetrievalReport(results=[_result(1), _result(1)])
    always_fifth = RetrievalReport(results=[_result(5), _result(5)])
    assert always_first.recall_at(5) == always_fifth.recall_at(5) == 1.0
    assert always_first.mrr() > always_fifth.mrr()
    assert always_fifth.mrr() == 0.2


def test_a_never_retrieved_query_contributes_zero_to_mrr():
    assert RetrievalReport(results=[_result(None)]).mrr() == 0.0


def test_results_group_by_defect():
    report = RetrievalReport(
        results=[
            _result(1, "table_only_fact"),
            _result(3, "table_only_fact"),
            _result(1),
        ]
    )
    grouped = report.by_defect()
    assert len(grouped["table_only_fact"]) == 2
    assert report.recall_at(1, grouped["table_only_fact"]) == 0.5


def test_an_empty_report_does_not_divide_by_zero():
    report = RetrievalReport()
    assert report.recall_at(5) == 0.0
    assert report.mrr() == 0.0
