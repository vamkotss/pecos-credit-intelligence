"""End-to-end credit memo tests (M7).

Real corpus, real retrieval, real graph, deterministic drafter. Marked `slow`.

The template drafter cannot invent a figure -- it prints only what the extractor
and the calculator produced. That makes it the control which proves the verify
gate is measuring the drafter rather than waving everything through: if these
memos verify clean and the deliberately-broken draft in the unit tests still
fails, the gate is doing real work.
"""

from __future__ import annotations

import pytest

from pecos.chunking import chunk_deal
from pecos.corpus import CorpusSpec
from pecos.corpus_build import generate_corpus
from pecos.drafting import TemplateDrafter
from pecos.ingest import ingest_document
from pecos.memo import MemoWriter
from pecos.policy import MAX_LEVERAGE, MIN_DSCR
from pecos.retrieval import HybridRetriever

pytestmark = pytest.mark.slow

SEED = 20260804
N_DEALS = 7
DOCUMENTS = (
    "01_loan_application.pdf",
    "02_financial_statements_comparative.pdf",
    "03_financial_statements_superseded.pdf",
    "04_debt_schedule.pdf",
    "05_ar_aging_and_concentration.pdf",
    "06_borrower_questionnaire.pdf",
    "07_broker_email_thread.pdf",
    "10_financial_statements_draft.pdf",
)


@pytest.fixture(scope="module")
def memos(tmp_path_factory):
    root = tmp_path_factory.mktemp("m7")
    spec = CorpusSpec(seed=SEED, n_deals=N_DEALS, out_dir=root / "raw", years=3)
    manifest = generate_corpus(spec, gold_dir=root / "evals", write_pdfs=True)

    defect = manifest.defect_index
    deals = sorted(
        {
            defect["near_duplicate_draft"][0],
            defect["restated_prior_year"][0],
            defect["table_only_fact"][0],
        }
    )
    packages = root / "raw" / "packages"
    chunks: list[dict] = []
    for deal_id in deals:
        pages = []
        for name in DOCUMENTS:
            path = packages / deal_id / name
            if path.exists():
                pages.extend(ingest_document(path, deal_id))
        chunks.extend(c.to_record() for c in chunk_deal(pages))

    retriever = HybridRetriever()
    retriever.build(chunks)
    writer = MemoWriter(retriever=retriever, drafter=TemplateDrafter())
    results = {deal: writer.write(deal) for deal in deals}
    return {"manifest": manifest, "results": results, "writer": writer, "deals": deals}


def test_every_memo_verifies(memos):
    """The headline. No figure in any memo is unaccounted for."""
    for deal, result in memos["results"].items():
        assert result.verified, f"{deal} has ungrounded figures: {result.ungrounded}"


def test_no_memo_needed_a_revision(memos):
    """The template drafter prints only extracted and calculated figures, so a
    revision would mean the verifier and the drafter disagree about what counts
    as grounded -- which would make the gate untrustworthy in both directions."""
    for result in memos["results"].values():
        assert result.revisions == 0


def test_figures_are_extracted_from_the_statements(memos):
    for result in memos["results"].values():
        assert result.figures_extracted >= 12


def test_the_core_credit_metrics_are_computed(memos):
    for deal, result in memos["results"].items():
        names = {entry.name for entry in result.computations.entries}
        assert "Total debt / EBITDA" in names, deal
        assert "DSCR" in names, deal


def test_every_calculation_carries_its_inputs_and_pages(memos):
    """A ratio without recorded inputs is right or wrong with no way to tell
    which. That is the failure mode this milestone exists to close."""
    for result in memos["results"].values():
        for entry in result.computations.entries:
            assert entry.inputs
            assert entry.citations
            for item in entry.inputs:
                assert item.document and item.page


def test_memos_cite_pages(memos):
    for result in memos["results"].values():
        assert result.citations
        assert all(marker.endswith(("]",)) for marker in result.citations)


def test_the_memo_states_a_recommendation(memos):
    """A credit memo that does not conclude is not a credit memo."""
    for result in memos["results"].values():
        assert "RECOMMENDATION" in result.text
        assert any(word in result.text for word in ("PROCEED", "DECLINE", "DEFER"))


def test_the_recommendation_follows_the_policy_thresholds(memos):
    """The conclusion has to be derivable from the numbers above it. A memo
    whose recommendation contradicts its own metrics is worse than one with no
    recommendation, because it looks reasoned."""
    for deal, result in memos["results"].items():
        metrics = {e.name: e.result for e in result.computations.entries}
        leverage_value = metrics.get("Total debt / EBITDA")
        dscr_value = metrics.get("DSCR")
        if leverage_value is None or dscr_value is None:
            assert "DEFER" in result.text
            continue
        within_policy = leverage_value <= MAX_LEVERAGE and dscr_value >= MIN_DSCR
        if within_policy:
            assert "PROCEED" in result.text, deal
        else:
            assert "DECLINE" in result.text, deal


def test_a_non_final_document_in_the_file_is_reported(memos):
    """A memo that quietly ignores a superseded statement gives the committee no
    way to know the file contained a contradiction."""
    manifest = memos["manifest"]
    for defect, marker in (
        ("restated_prior_year", "SUPERSEDED"),
        ("near_duplicate_draft", "DRAFT"),
    ):
        deal = manifest.defect_index[defect][0]
        if deal not in memos["results"]:
            continue
        text = memos["results"][deal].text
        assert marker in text, f"{deal} did not flag its {marker} document"


def test_the_audit_trail_shows_how_each_metric_was_derived(memos):
    for result in memos["results"].values():
        trail = result.audit_trail()
        assert "total debt / EBITDA" in trail
        assert "#p" in trail


def test_memo_generation_is_deterministic(memos):
    """Every claim in this milestone is meaningless if the same corpus produces
    a different memo on the next run."""
    deal = memos["deals"][0]
    again = memos["writer"].write(deal)
    assert again.text == memos["results"][deal].text
