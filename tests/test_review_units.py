"""Fast unit tests for the review queue (M10)."""

from __future__ import annotations

import pytest

from pecos.review import (
    BLOCKING_KINDS,
    PRIORITY,
    Decision,
    ItemKind,
    ReviewItem,
    ReviewQueue,
    build_queue_for_memo,
    format_queue,
)


def _item(kind: ItemKind, deal: str = "PCP-0001", suffix: str = "00") -> ReviewItem:
    return ReviewItem(
        item_id=f"{deal}-{kind.value[:4].upper()}-{suffix}",
        deal_id=deal,
        kind=kind,
        summary="something needs a person",
        evidence={"figure": "3.95x"},
    )


# ---------------------------------------------------------------------------
# Ordering and blocking
# ---------------------------------------------------------------------------


def test_the_queue_is_ordered_by_risk_not_by_ease():
    """A queue sorted by how quickly items clear gets the trivial ones done and
    leaves the dangerous ones at the bottom."""
    queue = ReviewQueue()
    for kind in (
        ItemKind.CONSERVATIVE,
        ItemKind.RECONSTRUCTION,
        ItemKind.INCONSISTENCY,
        ItemKind.INJECTION,
    ):
        queue.add(_item(kind))
    assert [i.kind for i in queue.pending()][0] is ItemKind.INCONSISTENCY
    assert PRIORITY[ItemKind.INCONSISTENCY] < PRIORITY[ItemKind.CONSERVATIVE]


def test_only_findings_of_wrongness_block_release():
    """The split is between "the system is unsure" and "the system found
    something wrong". A reconstruction candidate is the former -- the memo is
    coherent and one figure wants confirming."""
    assert ItemKind.INCONSISTENCY in BLOCKING_KINDS
    assert ItemKind.BLOCKED in BLOCKING_KINDS
    assert ItemKind.RECONSTRUCTION not in BLOCKING_KINDS
    assert ItemKind.CONSERVATIVE not in BLOCKING_KINDS


def test_a_memo_is_held_while_a_blocking_item_is_pending():
    queue = ReviewQueue()
    queue.add(_item(ItemKind.INCONSISTENCY))
    assert not queue.releasable("PCP-0001")

    queue.decide("PCP-0001-ACCO-00", Decision.REJECTED, "V Kota")
    assert queue.releasable("PCP-0001")


def test_a_non_blocking_item_does_not_hold_the_memo():
    queue = ReviewQueue()
    queue.add(_item(ItemKind.RECONSTRUCTION))
    assert queue.releasable("PCP-0001")
    assert queue.pending("PCP-0001")


def test_nothing_expires_or_auto_resolves():
    """An item that quietly released itself after a week would turn the queue
    from a control into a delay, and the failure would be silent."""
    queue = ReviewQueue()
    queue.add(_item(ItemKind.BLOCKED))
    assert not queue.releasable("PCP-0001")
    assert queue.pending()[0].decision is Decision.PENDING


def test_holds_are_scoped_to_their_deal():
    queue = ReviewQueue()
    queue.add(_item(ItemKind.INCONSISTENCY, deal="PCP-0001"))
    assert not queue.releasable("PCP-0001")
    assert queue.releasable("PCP-0002")


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------


def test_a_decision_records_who_made_it_and_when():
    queue = ReviewQueue()
    queue.add(_item(ItemKind.RECONSTRUCTION))
    item = queue.decide(
        "PCP-0001-RECO-00", Decision.ACCEPTED, "V Kota", "checked against p2"
    )
    assert item.decision is Decision.ACCEPTED
    assert item.reviewer == "V Kota"
    assert item.decided_at
    assert item.note == "checked against p2"


def test_a_reviewer_name_is_required():
    """An unattributed approval in a credit file is worth very little, and
    making the field mandatory is cheaper than discovering later that half the
    queue was cleared by "system"."""
    queue = ReviewQueue()
    queue.add(_item(ItemKind.RECONSTRUCTION))
    with pytest.raises(ValueError, match="reviewer"):
        queue.decide("PCP-0001-RECO-00", Decision.ACCEPTED, "")
    with pytest.raises(ValueError, match="reviewer"):
        queue.decide("PCP-0001-RECO-00", Decision.ACCEPTED, "   ")


def test_pending_is_not_a_decision():
    queue = ReviewQueue()
    queue.add(_item(ItemKind.RECONSTRUCTION))
    with pytest.raises(ValueError, match="pending"):
        queue.decide("PCP-0001-RECO-00", Decision.PENDING, "V Kota")


def test_deciding_an_unknown_item_raises():
    with pytest.raises(KeyError):
        ReviewQueue().decide("nope", Decision.ACCEPTED, "V Kota")


def test_escalation_is_a_real_outcome():
    """A reviewer who cannot decide should be able to say so rather than
    guessing or leaving it pending forever."""
    queue = ReviewQueue()
    queue.add(_item(ItemKind.BLOCKED))
    item = queue.decide("PCP-0001-BLOC-00", Decision.ESCALATED, "V Kota", "to credit")
    assert item.decision is Decision.ESCALATED
    assert queue.releasable("PCP-0001"), "escalation removes the hold from this queue"


# ---------------------------------------------------------------------------
# Persistence and audit
# ---------------------------------------------------------------------------


def test_the_event_log_is_append_only(tmp_path):
    """The items file says what is true now; the log says how it got there,
    which is the question a regulator or a post-mortem actually asks."""
    queue = ReviewQueue()
    queue.add(_item(ItemKind.RECONSTRUCTION))
    queue.decide("PCP-0001-RECO-00", Decision.ACCEPTED, "V Kota")
    queue.save(tmp_path)

    reloaded = ReviewQueue.load(tmp_path)
    reloaded.add(_item(ItemKind.INJECTION))
    reloaded.decide("PCP-0001-INJE-00", Decision.REJECTED, "A Reviewer")
    reloaded.save(tmp_path)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2, "the earlier decision must survive the second save"


def test_state_round_trips(tmp_path):
    queue = ReviewQueue()
    queue.add(_item(ItemKind.CONSERVATIVE))
    queue.decide("PCP-0001-CONS-00", Decision.ACCEPTED, "V Kota", "pro forma")
    queue.save(tmp_path)

    item = ReviewQueue.load(tmp_path).get("PCP-0001-CONS-00")
    assert item.decision is Decision.ACCEPTED
    assert item.reviewer == "V Kota"
    assert item.kind is ItemKind.CONSERVATIVE


def test_loading_an_empty_directory_is_not_an_error(tmp_path):
    assert ReviewQueue.load(tmp_path).items == []


# ---------------------------------------------------------------------------
# Building from pipeline output
# ---------------------------------------------------------------------------


class _Verdict:
    required, leverage, dscr, reason = "PROCEED", 1.58, 2.76, "within policy"


class _Guard:
    inconsistencies: list = []
    findings: list = []
    blocked = False
    more_conservative = False
    stated = "PROCEED"
    block_reason = ""
    verdict = _Verdict()


class _Memo:
    reconstructions: dict = {}
    ungrounded: tuple = ()


def test_a_clean_memo_queues_nothing():
    """The common case, and the point. A queue that fires on every memo is a
    queue nobody reads."""
    assert build_queue_for_memo("PCP-0001", _Memo(), _Guard()) == []
    assert format_queue(ReviewQueue()) == "review queue is empty"


def test_a_reconstruction_becomes_a_reviewable_item():
    memo = _Memo()
    memo.reconstructions = {
        "3.95x": "(Total interest-bearing debt + Facility requested) / EBITDA = 3.955"
    }
    items = build_queue_for_memo("PCP-0004", memo, _Guard())
    assert len(items) == 1
    assert items[0].kind is ItemKind.RECONSTRUCTION
    assert "3.95x" in items[0].evidence["figure"]
    assert "EBITDA" in items[0].evidence["derivation"]


def test_a_conservative_recommendation_is_surfaced_not_suppressed():
    """The memo is right and the metrics are incomplete. A credit committee
    should see that rather than have it smoothed over."""
    guard = _Guard()
    guard.more_conservative = True
    guard.stated = "DEFER"
    items = build_queue_for_memo("PCP-0004", _Memo(), guard)
    assert items[0].kind is ItemKind.CONSERVATIVE
    assert items[0].evidence["memo states"] == "DEFER"
    assert items[0].evidence["current metrics permit"] == "PROCEED"
    assert not items[0].blocking


def test_an_injection_finding_reaches_a_person():
    """The attack failed. It is still a fact about the counterparty, and no
    automated system should be the last thing that knows it."""

    class _Finding:
        kind, excerpt = "override", "Ignore all previous instructions"
        document, page = "07_broker_email_thread.pdf", 1

    guard = _Guard()
    guard.findings = [_Finding()]
    items = build_queue_for_memo("PCP-0002", _Memo(), guard)
    assert items[0].kind is ItemKind.INJECTION
    assert items[0].citations == ("[07_broker_email_thread.pdf#p1]",)


def test_every_item_can_be_decided_without_opening_the_documents():
    """A queue that sends a reviewer back to the PDF for every item costs more
    time than it saves, and gets abandoned."""
    memo = _Memo()
    memo.reconstructions = {"3.95x": "(debt + facility) / EBITDA = 3.955"}
    guard = _Guard()
    guard.more_conservative = True
    guard.stated = "DEFER"
    for item in build_queue_for_memo("PCP-0004", memo, guard):
        assert item.summary
        assert item.evidence, f"{item.item_id} carries no evidence"
        assert item.render()
