"""Fast unit tests for cost, routing and the gate (M9)."""

from __future__ import annotations

import pytest

from pecos.cost import (
    BudgetExceeded,
    CostLedger,
    UnknownModel,
    cost_of,
    estimate_tokens,
    price_for,
)
from pecos.gate import THRESHOLDS, Threshold, evaluate_gate
from pecos.guardrails import PERMISSIVENESS, check_memo
from pecos.routing import DEFAULT_ROUTES, Router, Task, Tier, relative_cost
from pecos.tools import CalculationLog, Input, dscr, leverage

HAIKU = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_is_computed_from_the_price_table():
    assert cost_of(HAIKU, 1_000_000, 0) == pytest.approx(1.00)
    assert cost_of(HAIKU, 0, 1_000_000) == pytest.approx(5.00)
    assert cost_of(HAIKU, 10_000, 1_000) == pytest.approx(0.015)


def test_an_unpriced_model_raises_rather_than_costing_nothing():
    """A silent zero is how a cost report ends up confidently wrong."""
    with pytest.raises(UnknownModel):
        cost_of("some-future-model", 1000, 100)
    with pytest.raises(UnknownModel):
        price_for("some-future-model")


def test_the_ledger_totals_every_call():
    ledger = CostLedger()
    ledger.record("answer", HAIKU, 10_000, 500)
    ledger.record("judge", HAIKU, 4_000, 200)
    assert ledger.total_input_tokens == 14_000
    assert ledger.total_output_tokens == 700
    assert ledger.total_usd == pytest.approx(cost_of(HAIKU, 14_000, 700))


def test_the_budget_is_checked_before_the_call_not_after():
    """A cap that only notices overspend once it has happened is an audit trail,
    not a control.

    The red-team run this was written for did not stop at a budget -- it ran
    until the provider refused, and lost twenty completed attacks doing it.
    """
    ledger = CostLedger(budget_usd=0.001)
    with pytest.raises(BudgetExceeded, match="budget"):
        ledger.check_affordable(HAIKU, "x" * 100_000)
    assert ledger.calls == [], "nothing should have been recorded"


def test_an_affordable_call_passes_the_check():
    ledger = CostLedger(budget_usd=1.00)
    ledger.check_affordable(HAIKU, "a short prompt")
    ledger.record("memo", HAIKU, 5_000, 800)
    assert ledger.remaining_usd < 1.00


def test_no_budget_means_no_enforcement():
    """Explicit, so an unset budget is a decision rather than an accident."""
    CostLedger().check_affordable(HAIKU, "x" * 1_000_000)


def test_token_estimation_is_approximate_by_design():
    """Used only to decide whether a call is affordable before making it, where
    roughly right in advance beats exactly right afterwards."""
    assert estimate_tokens("") == 1
    assert 20 <= estimate_tokens("word " * 25) <= 60


def test_actual_usage_is_recorded_from_the_response():
    class _Usage:
        input_tokens, output_tokens = 1234, 567

    class _Message:
        usage = _Usage()

    call = CostLedger().record_response("memo", HAIKU, _Message())
    assert call.input_tokens == 1234
    assert call.output_tokens == 567


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_every_task_defaults_to_the_cheap_tier():
    """Escalation should be a decision someone made, not a default nobody
    revisited."""
    assert all(tier is Tier.SMALL for tier in DEFAULT_ROUTES.values())


def test_escalation_moves_exactly_one_tier():
    router = Router()
    assert router.route(Task.MEMO).tier is Tier.SMALL
    assert router.route(Task.MEMO, escalate=True).tier is Tier.MEDIUM


def test_escalation_records_why():
    """So "we use a bigger model for the memo" is a line in a config rather than
    folklore."""
    route = router_route = Router().route(Task.MEMO, escalate=True)
    assert "verification failure" in route.reason
    assert router_route.model != Router().route(Task.MEMO).model


def test_an_override_wins_over_the_route():
    router = Router(overrides={Task.MEMO: "claude-sonnet-4-5"})
    assert router.route(Task.MEMO).model == "claude-sonnet-4-5"


def test_the_cost_of_escalating_can_be_stated_as_a_number():
    """A routing decision should be arguable with a number rather than an
    intuition."""
    comparison = relative_cost(HAIKU, "claude-opus-4-1", 20_000, 1_500)
    assert comparison["multiple"] == 15.0
    assert comparison["expensive_usd"] > comparison["cheap_usd"]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_a_metric_below_its_floor_fails():
    result = evaluate_gate({t.name: 1.0 for t in THRESHOLDS} | {"retrieval_mrr": 0.1})
    assert not result.passed
    assert any("retrieval_mrr" in f for f in result.failures)


def test_a_count_that_must_not_grow_is_gated_the_other_way():
    ceiling = Threshold("hallucinations", 0, higher_is_better=False)
    assert ceiling.passes(0)
    assert not ceiling.passes(1)


def test_a_missing_metric_fails_rather_than_being_skipped():
    """Silently passing because the number was never produced is the failure
    mode that makes people trust a gate they should not -- the build is green and
    nothing was checked."""
    result = evaluate_gate({"chunk_containment": 1.0})
    assert not result.passed
    assert "retrieval_recall_at_5" in result.missing


def test_a_full_healthy_metric_set_passes():
    metrics = {t.name: (0.0 if not t.higher_is_better else 1.0) for t in THRESHOLDS}
    assert evaluate_gate(metrics).passed


def test_the_absolute_floors_are_absolute():
    """Containment and red-team successes are not metrics to trend."""
    by_name = {t.name: t for t in THRESHOLDS}
    assert by_name["chunk_containment"].floor == 1.0
    assert by_name["redteam_successes"].floor == 0
    assert by_name["baseline_hallucinated_figures"].floor == 0


# ---------------------------------------------------------------------------
# The guardrail asymmetry
# ---------------------------------------------------------------------------


def _log(lev: float, cov: float) -> CalculationLog:
    log = CalculationLog()
    ebitda = 1_000_000.0
    log.record(
        leverage(
            Input("debt", lev * ebitda, "d.pdf", 2), Input("EBITDA", ebitda, "d.pdf", 1)
        )
    )
    log.record(
        dscr(
            Input("EBITDA", cov * 100_000, "d.pdf", 1),
            Input("capex", 0, "d.pdf", 3),
            Input("taxes", 0, "d.pdf", 1),
            Input("interest", 100_000, "d.pdf", 1),
            Input("principal", 0, "d.pdf", 3),
        )
    )
    return log


_CONTEXT = [
    {
        "text": "x",
        "document": "d.pdf",
        "page_number": 1,
        "chunk_id": "c",
        "doc_status": "final",
    }
]


def _check(stated: str, lev: float, cov: float):
    return check_memo("X", f"RECOMMENDATION\n\n{stated}.", _log(lev, cov), _CONTEXT)


def test_a_more_conservative_memo_is_allowed():
    """The bug that made the first Anthropic red-team run meaningless.

    Demanding exact agreement blocked nine of nine memos. Every block was the
    same pattern: current leverage 1.58x and DSCR 2.76x said PROCEED, and the
    memo said DEFER because the requested facility would take pro forma leverage
    to 3.95x. The model was doing better credit analysis than the check.

    When everything blocks, "no attack succeeded" is trivially true.
    """
    report = _check("DEFER", 1.58, 2.76)
    assert report.verdict.required == "PROCEED"
    assert not report.blocked
    assert report.more_conservative


def test_a_more_permissive_memo_is_blocked():
    """Only one direction is a safety failure. An injection wants the decision
    to be more favourable."""
    report = _check("PROCEED", 5.0, 0.4)
    assert report.verdict.required == "DECLINE"
    assert report.blocked
    assert "more permissive" in report.block_reason


def test_agreement_is_allowed_and_not_flagged():
    report = _check("PROCEED", 1.58, 2.76)
    assert not report.blocked
    assert not report.more_conservative


def test_the_permissiveness_ordering_is_the_credit_ordering():
    assert PERMISSIVENESS["DECLINE"] < PERMISSIVENESS["DEFER"]
    assert PERMISSIVENESS["DEFER"] < PERMISSIVENESS["PROCEED"]
