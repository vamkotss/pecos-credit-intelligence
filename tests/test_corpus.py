"""Tests for the synthetic corpus generator (M2).

These are fast: they never render a PDF. `write_pdfs=False` builds the deals and
the ground truth only, so the whole file runs in well under a second and can be
left in the default CI path.

The most important tests here are the two accounting identities. They are what
turn "plausible-looking numbers" into "numbers a credit analyst would accept",
and they are asserted to the exact dollar rather than to a tolerance.
"""

from __future__ import annotations

import json

import pytest

from pecos.corpus import (
    ALL_DEFECTS,
    DEFECT_INJECTION,
    DEFECT_NEAR_DUPLICATE,
    DEFECT_RESTATEMENT,
    DEFECT_THOUSANDS,
    DEFECT_UNANSWERABLE,
    CorpusSpec,
    build_deal,
)
from pecos.corpus_build import generate_corpus

SEED = 20260804


def _spec(tmp_path, n_deals: int = 4) -> CorpusSpec:
    return CorpusSpec(seed=SEED, n_deals=n_deals, out_dir=tmp_path / "raw", years=3)


@pytest.fixture(scope="module")
def deals():
    """Six deals built once and shared. No files are written."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        spec = CorpusSpec(seed=SEED, n_deals=6, out_dir=Path(td), years=3)
        yield [build_deal(spec, i) for i in range(6)]


# ---------------------------------------------------------------------------
# Accounting identities
# ---------------------------------------------------------------------------


def test_balance_sheet_balances_exactly(deals):
    """Assets must equal liabilities plus equity, to the dollar, every year.

    If this ever fails, the generator has introduced a figure that does not
    come from the same roll-forward as everything else, and every downstream
    eval built on those numbers is measuring against a broken oracle.
    """
    for deal in deals:
        for year in deal.financials:
            assert year.total_assets == year.total_liabilities_and_equity, (
                f"{deal.deal_id} FY{year.fiscal_year}: assets "
                f"{year.total_assets:,} != L+E {year.total_liabilities_and_equity:,}"
            )


def test_cash_flow_ties_to_change_in_cash(deals):
    """CFO + CFI + CFF must equal the actual movement in the cash balance."""
    for deal in deals:
        for i in range(1, len(deal.financials)):
            prior = deal.financials[i - 1]
            current = deal.financials[i]
            computed = current.cfo(prior) + current.cfi() + current.cff()
            actual = current.cash - prior.cash
            assert computed == actual, (
                f"{deal.deal_id} FY{current.fiscal_year}: cash flow {computed:,} "
                f"!= change in cash {actual:,}"
            )


def test_income_statement_is_internally_consistent(deals):
    for deal in deals:
        for y in deal.financials:
            assert y.gross_profit == y.revenue - y.cogs
            assert y.ebitda == y.gross_profit - y.opex_cash
            assert y.ebit == y.ebitda - y.depreciation
            assert y.net_income == y.pretax_income - y.tax_expense


def test_debt_schedule_reconciles_to_balance_sheet(deals):
    """Interest-bearing debt on the balance sheet must equal the sum of the
    facility balances on the debt schedule, after that year's repayments."""
    for deal in deals:
        for i, year in enumerate(deal.financials):
            schedule_total = sum(
                loan.balances[i] - loan.principal_payments[i] for loan in deal.loans
            )
            assert year.total_debt == schedule_total, (
                f"{deal.deal_id} FY{year.fiscal_year}: balance sheet debt "
                f"{year.total_debt:,} != schedule {schedule_total:,}"
            )
            assert year.interest_expense == sum(
                loan.interest_payments[i] for loan in deal.loans
            )


# ---------------------------------------------------------------------------
# Credit plausibility
# ---------------------------------------------------------------------------


def test_borrowers_are_creditworthy_enough_to_be_realistic(deals):
    """No negative equity, no negative cash, no absurd leverage.

    A corpus full of insolvent borrowers would make every generated credit memo
    a decline, and the agent's decision logic would never be exercised.
    """
    for deal in deals:
        for y in deal.financials:
            assert y.cash > 0, f"{deal.deal_id} FY{y.fiscal_year} has negative cash"
            assert (
                y.total_equity > 0
            ), f"{deal.deal_id} FY{y.fiscal_year} negative equity"
            assert y.ebitda > 0
            assert 0.3 <= y.leverage <= 8.0, f"{deal.deal_id} leverage {y.leverage}"


def test_request_amount_is_inside_pecos_credit_box(deals):
    """Pecos writes $3M to $40M. A generated deal outside that band would be a
    request the lender's own policy forbids it from considering."""
    for deal in deals:
        assert 3_000_000 <= deal.request_amount <= 40_000_000


def test_receivable_buckets_sum_to_the_balance_sheet(deals):
    for deal in deals:
        total = sum(b.amount for b in deal.aging)
        assert total == deal.latest.accounts_receivable


def test_customer_shares_sum_to_roughly_one_hundred_percent(deals):
    """Shares are rounded to one decimal for display, so a small drift is
    expected. A large one means the split logic is wrong."""
    for deal in deals:
        total = sum(c.pct_of_ar for c in deal.customers)
        assert 99.0 <= total <= 101.0, f"{deal.deal_id} concentration sums to {total}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_an_identical_manifest(tmp_path):
    """The manifest hash is the corpus's fingerprint.

    Every eval score recorded later is only meaningful if the corpus it was
    measured on can be reproduced exactly. This test is what makes that claim
    defensible rather than assumed.
    """
    a = generate_corpus(_spec(tmp_path / "a"), write_pdfs=False)
    b = generate_corpus(_spec(tmp_path / "b"), write_pdfs=False)
    assert a.sha256() == b.sha256()


def test_different_seed_produces_a_different_manifest(tmp_path):
    a = generate_corpus(_spec(tmp_path / "a"), write_pdfs=False)
    other = CorpusSpec(seed=SEED + 1, n_deals=4, out_dir=tmp_path / "b", years=3)
    b = generate_corpus(other, write_pdfs=False)
    assert a.sha256() != b.sha256()


def test_a_deal_is_stable_across_corpus_sizes(tmp_path):
    """Deal 1's borrower and financials must not change when the corpus grows.

    This is what lets a two-deal CI run be evidence about the full corpus.
    Defect assignment is excluded, since defects are dealt round-robin so that
    every failure mode appears at any corpus size.
    """
    small = build_deal(CorpusSpec(seed=SEED, n_deals=2, out_dir=tmp_path), 1)
    large = build_deal(CorpusSpec(seed=SEED, n_deals=12, out_dir=tmp_path), 1)

    assert small.borrower_name == large.borrower_name
    assert small.request_amount == large.request_amount
    assert [f.revenue for f in small.financials] == [
        f.revenue for f in large.financials
    ]
    assert [f.ebitda for f in small.financials] == [f.ebitda for f in large.financials]
    assert [f.cash for f in small.financials] == [f.cash for f in large.financials]


# ---------------------------------------------------------------------------
# Planted defects
# ---------------------------------------------------------------------------


def test_every_defect_appears_even_in_a_tiny_ci_corpus(tmp_path):
    """The whole point of round-robin assignment.

    If defects were sampled randomly, a two-deal CI run could quietly skip the
    injection case, and the red-team suite would pass by not running.
    """
    for n in (1, 2, 3, 7, 12):
        manifest = generate_corpus(
            CorpusSpec(seed=SEED, n_deals=n, out_dir=tmp_path / f"n{n}"),
            write_pdfs=False,
        )
        for defect in ALL_DEFECTS:
            assert manifest.defect_index[
                defect
            ], f"defect {defect} missing from a {n}-deal corpus"


def test_defect_payloads_exist_when_the_defect_is_assigned(deals):
    """A defect flag with no payload behind it would be a silent no-op."""
    for deal in deals:
        if DEFECT_RESTATEMENT in deal.defects:
            assert deal.stale_ebitda is not None
            # The restated figure must differ enough to matter. A one-dollar
            # difference would not be a restatement, it would be a rounding bug.
            drift = abs(deal.stale_ebitda - deal.financials[0].ebitda)
            assert drift > deal.financials[0].ebitda * 0.05
        else:
            assert deal.stale_ebitda is None

        if DEFECT_NEAR_DUPLICATE in deal.defects:
            assert deal.draft_ebitda is not None
            assert deal.draft_ebitda != deal.latest.ebitda
        else:
            assert deal.draft_ebitda is None


def test_no_defect_is_assigned_to_every_deal(tmp_path):
    """There must be clean deals to compare against, otherwise a pipeline that
    always assumes a defect is present scores well for the wrong reason."""
    manifest = generate_corpus(
        CorpusSpec(seed=SEED, n_deals=12, out_dir=tmp_path), write_pdfs=False
    )
    for defect, deal_ids in manifest.defect_index.items():
        assert len(deal_ids) < manifest.n_deals, f"{defect} is on every deal"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def test_fact_ids_are_unique_across_the_corpus(tmp_path):
    manifest = generate_corpus(
        CorpusSpec(seed=SEED, n_deals=8, out_dir=tmp_path), write_pdfs=False
    )
    ids = [f["fact_id"] for f in manifest.facts]
    assert len(ids) == len(set(ids))


def test_answerable_facts_cite_a_document_and_page(tmp_path):
    """A retrieval target with no page number cannot be scored on recall@k."""
    manifest = generate_corpus(
        CorpusSpec(seed=SEED, n_deals=8, out_dir=tmp_path), write_pdfs=False
    )
    for fact in manifest.facts:
        if fact["answerable"]:
            assert fact["source_document"], fact["fact_id"]
            assert isinstance(fact["source_page"], int), fact["fact_id"]
            assert fact["source_page"] >= 1


def test_the_unanswerable_question_cites_nothing(tmp_path):
    """Marked unanswerable and pointing at no source. The correct model output
    is a refusal, and an eval harness that scored refusal as failure would
    punish exactly the behaviour we want."""
    manifest = generate_corpus(
        CorpusSpec(seed=SEED, n_deals=8, out_dir=tmp_path), write_pdfs=False
    )
    unanswerable = [f for f in manifest.facts if f["defect_tag"] == DEFECT_UNANSWERABLE]
    assert unanswerable
    for fact in unanswerable:
        assert fact["answerable"] is False
        assert fact["source_document"] is None
        assert fact["source_page"] is None


def test_derived_metrics_match_a_recomputation(tmp_path):
    """Leverage and DSCR in the manifest must be reproducible from the raw
    figures, not stored values that could drift."""
    spec = CorpusSpec(seed=SEED, n_deals=6, out_dir=tmp_path)
    manifest = generate_corpus(spec, write_pdfs=False)
    for i, record in enumerate(manifest.deals):
        deal = build_deal(spec, i)
        latest = deal.latest
        assert record["leverage"] == round(latest.total_debt / latest.ebitda, 2)
        expected_dscr = round(
            (latest.ebitda - latest.capex - latest.tax_expense)
            / (latest.interest_expense + latest.principal_repayments),
            2,
        )
        assert record["dscr"] == expected_dscr


def test_gold_jsonl_is_written_and_parseable(tmp_path):
    gold_dir = tmp_path / "evals"
    generate_corpus(
        CorpusSpec(seed=SEED, n_deals=3, out_dir=tmp_path / "raw"),
        gold_dir=gold_dir,
        write_pdfs=False,
    )
    path = gold_dir / "qa_gold.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) > 20
    for line in lines:
        record = json.loads(line)
        assert record["question"]
        assert record["fact_id"].startswith("PCP-")


def test_manifest_covers_every_defect_with_at_least_one_gold_fact(tmp_path):
    """Every planted defect must have a question that exercises it. A defect
    with no question is scenery, not an evaluation instrument."""
    manifest = generate_corpus(
        CorpusSpec(seed=SEED, n_deals=7, out_dir=tmp_path), write_pdfs=False
    )
    tagged = {f["defect_tag"] for f in manifest.facts if f["defect_tag"]}
    # The rotated-scan defect is exercised through OCR at M3 and is tagged too.
    for defect in ALL_DEFECTS:
        assert defect in tagged, f"{defect} has no gold question"


def test_thousands_defect_records_the_true_dollar_value(tmp_path):
    """The gold answer must be in dollars even though the document prints
    thousands. Storing the printed figure would bake the bug into the oracle."""
    spec = CorpusSpec(seed=SEED, n_deals=7, out_dir=tmp_path)
    manifest = generate_corpus(spec, write_pdfs=False)
    facts = [f for f in manifest.facts if f["defect_tag"] == DEFECT_THOUSANDS]
    assert facts
    for fact in facts:
        deal_record = next(d for d in manifest.deals if d["deal_id"] == fact["deal_id"])
        assert fact["answer_value"] == deal_record["latest_revenue"]
        assert fact["answer_value"] > 1_000_000


def test_injection_fact_is_behavioural_not_extractive(tmp_path):
    """The injection case is scored on what the model refuses to do, so it must
    not carry a numeric answer that a naive harness would try to match."""
    manifest = generate_corpus(
        CorpusSpec(seed=SEED, n_deals=7, out_dir=tmp_path), write_pdfs=False
    )
    facts = [f for f in manifest.facts if f["defect_tag"] == DEFECT_INJECTION]
    assert facts
    for fact in facts:
        assert fact["fact_type"] == "behavioural"
        assert fact["answer_value"] is None
