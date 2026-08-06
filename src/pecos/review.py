"""The human review queue (M10).

WHAT THIS MILESTONE IS ACTUALLY FOR
-----------------------------------
Nine milestones produced a pipeline that reads loan documents, computes credit
metrics, writes a memo, and refuses to release one that contradicts its own
arithmetic. What it never did was decide anything a person should decide.

Three things have been *reported* and then gone nowhere:

**Reconstruction candidates (M7).** "3.95x follows from (total debt + facility
requested) / EBITDA." Almost certainly right, and coincidental matches are
possible with three-significant-figure numbers. The system found the arithmetic;
it cannot know whether that arithmetic is what the writer meant.

**Conservative recommendations (M9).** The memo says DEFER where the computed
metrics permit PROCEED — usually because pro forma leverage after the requested
facility would breach policy. The memo is right and the metrics are incomplete.
Allowed, and a credit committee should see it rather than have it smoothed over.

**Injection findings (M8).** A broker's cover note contained text addressed to an
automated reader. The attack failed. It is still a fact about the counterparty,
and no automated system should be the last thing that knows it.

Each is a judgement the pipeline correctly declines to make. This module gives
them a queue, a decision, and a record of who made it.

THE DESIGN CONSTRAINT THAT MATTERS
----------------------------------
**An item must be decidable without opening the source documents.** A queue that
sends a reviewer back to the PDF for every item costs more time than it saves,
and gets abandoned. So every item carries its evidence inline: the figure, the
derivation, the citation, the metrics, the excerpt.

**Decisions are append-only, with reviewer and timestamp.** A credit file is a
regulated artefact and "who approved this" is a question that gets asked years
later. Mutating a decision in place destroys the only record of what was known
at the time.

**Nothing auto-resolves.** A blocking item left unreviewed keeps the memo
unreleased. An expiry that quietly released stale items would convert the queue
from a control into a delay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class ItemKind(StrEnum):
    """Why a person is being asked."""

    RECONSTRUCTION = "reconstruction"  # a figure derived, not quoted
    CONSERVATIVE = "conservative_recommendation"  # memo stricter than the metrics
    INJECTION = "injection_finding"  # instruction-like text in a document
    BLOCKED = "blocked_memo"  # decision contradicts the arithmetic
    INCONSISTENCY = "accounting_inconsistency"  # figures cannot all be true
    LOW_CONFIDENCE = "low_confidence_extraction"  # OCR the pipeline half-trusts


class Decision(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ESCALATED = "escalated"


# Whether an unresolved item of this kind should stop the memo being released.
#
# The split is between "the system is unsure" and "the system found something
# wrong". A reconstruction candidate is the former: the memo is coherent and one
# figure wants confirming. A blocked memo or a broken accounting identity is the
# latter, and releasing it on the grounds that nobody got round to reviewing it
# would defeat every check that produced it.
BLOCKING_KINDS: frozenset[ItemKind] = frozenset(
    {ItemKind.BLOCKED, ItemKind.INCONSISTENCY}
)

# Review order. Risk first, not ease first -- a queue sorted by how quickly items
# can be cleared gets the trivial ones done and leaves the dangerous ones at the
# bottom.
PRIORITY: dict[ItemKind, int] = {
    ItemKind.INCONSISTENCY: 0,
    ItemKind.BLOCKED: 1,
    ItemKind.INJECTION: 2,
    ItemKind.RECONSTRUCTION: 3,
    ItemKind.CONSERVATIVE: 4,
    ItemKind.LOW_CONFIDENCE: 5,
}


@dataclass
class ReviewItem:
    """One thing a person must decide.

    `evidence` is what makes the queue usable. A reviewer should be able to
    accept or reject from this record alone; if they have to open the PDF, the
    queue is costing more time than it saves and it will be abandoned.
    """

    item_id: str
    deal_id: str
    kind: ItemKind
    summary: str
    evidence: dict = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    decision: Decision = Decision.PENDING
    reviewer: str | None = None
    decided_at: str | None = None
    note: str = ""

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING_KINDS and self.decision is Decision.PENDING

    @property
    def priority(self) -> int:
        return PRIORITY[self.kind]

    def to_record(self) -> dict:
        record = asdict(self)
        record["kind"] = str(self.kind)
        record["decision"] = str(self.decision)
        record["citations"] = list(self.citations)
        return record

    @classmethod
    def from_record(cls, record: dict) -> ReviewItem:
        return cls(
            item_id=record["item_id"],
            deal_id=record["deal_id"],
            kind=ItemKind(record["kind"]),
            summary=record["summary"],
            evidence=record.get("evidence", {}),
            citations=tuple(record.get("citations", [])),
            decision=Decision(record.get("decision", "pending")),
            reviewer=record.get("reviewer"),
            decided_at=record.get("decided_at"),
            note=record.get("note", ""),
        )

    def render(self) -> str:
        lines = [
            f"[{self.item_id}] {self.kind} -- {self.deal_id}  ({self.decision})",
            f"  {self.summary}",
        ]
        for key, value in self.evidence.items():
            lines.append(f"    {key}: {value}")
        if self.citations:
            lines.append(f"    cited: {' '.join(self.citations)}")
        if self.decision is not Decision.PENDING:
            lines.append(
                f"    decided by {self.reviewer} at {self.decided_at}"
                + (f" -- {self.note}" if self.note else "")
            )
        return "\n".join(lines)


@dataclass
class ReviewEvent:
    """An append-only record of one decision.

    Kept separately from the item's current state on purpose. The item says what
    is true now; the log says how it got there, which is the question a regulator
    or a post-mortem actually asks. Mutating a decision in place destroys the
    only record of what was known at the time.
    """

    item_id: str
    decision: Decision
    reviewer: str
    at: str
    note: str = ""

    def to_record(self) -> dict:
        return {
            "item_id": self.item_id,
            "decision": str(self.decision),
            "reviewer": self.reviewer,
            "at": self.at,
            "note": self.note,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ReviewQueue:
    """Items awaiting a person, and the log of what was decided."""

    items: list[ReviewItem] = field(default_factory=list)
    events: list[ReviewEvent] = field(default_factory=list)

    # --- construction ----------------------------------------------------

    def add(self, item: ReviewItem) -> ReviewItem:
        self.items.append(item)
        return item

    def get(self, item_id: str) -> ReviewItem:
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(f"no review item {item_id!r}")

    # --- reading ---------------------------------------------------------

    def pending(self, deal_id: str | None = None) -> list[ReviewItem]:
        """Pending items, most dangerous first."""
        rows = [i for i in self.items if i.decision is Decision.PENDING]
        if deal_id:
            rows = [i for i in rows if i.deal_id == deal_id]
        return sorted(rows, key=lambda i: (i.priority, i.item_id))

    def blocking(self, deal_id: str | None = None) -> list[ReviewItem]:
        return [i for i in self.pending(deal_id) if i.blocking]

    def releasable(self, deal_id: str) -> bool:
        """Can this memo go to committee?

        False while any blocking item is unreviewed. There is deliberately no
        expiry: an item that quietly released itself after a week would turn the
        queue from a control into a delay, and the failure would be silent.
        """
        return not self.blocking(deal_id)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            key = f"{item.kind}:{item.decision}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    # --- deciding --------------------------------------------------------

    def decide(
        self, item_id: str, decision: Decision, reviewer: str, note: str = ""
    ) -> ReviewItem:
        """Record a decision.

        A reviewer name is required rather than optional. An unattributed
        approval in a credit file is worth very little, and making the field
        mandatory is cheaper than discovering later that half the queue was
        cleared by "system".
        """
        if not reviewer or not reviewer.strip():
            raise ValueError("a reviewer name is required to decide an item")
        if decision is Decision.PENDING:
            raise ValueError("pending is not a decision")

        item = self.get(item_id)
        item.decision = decision
        item.reviewer = reviewer.strip()
        item.decided_at = _now()
        item.note = note
        self.events.append(
            ReviewEvent(
                item_id=item_id,
                decision=decision,
                reviewer=item.reviewer,
                at=item.decided_at,
                note=note,
            )
        )
        return item

    # --- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "items.jsonl").write_text(
            "\n".join(json.dumps(i.to_record(), sort_keys=True) for i in self.items)
            + ("\n" if self.items else ""),
            encoding="utf-8",
        )
        # Appended, never rewritten. The log is the audit trail; rewriting it
        # would make it a summary of the current state, which the items file
        # already is.
        log = directory / "events.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_record(), sort_keys=True) + "\n")
        self.events = []

    @classmethod
    def load(cls, directory: Path) -> ReviewQueue:
        path = directory / "items.jsonl"
        if not path.exists():
            return cls()
        items = [
            ReviewItem.from_record(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(items=items)


# ---------------------------------------------------------------------------
# Building the queue from pipeline output
# ---------------------------------------------------------------------------


def _item_id(deal_id: str, kind: ItemKind, index: int) -> str:
    return f"{deal_id}-{kind.value[:4].upper()}-{index:02d}"


def build_queue_for_memo(
    deal_id: str, memo_result, guardrail_report
) -> list[ReviewItem]:
    """Turn one memo's open questions into review items.

    Only things the pipeline genuinely could not settle end up here. A memo where
    every figure is quoted or computed, the recommendation matches the metrics,
    and no instruction-like text was found produces **no items at all** -- which
    is the common case and the point. A queue that fires on every memo is a queue
    nobody reads.
    """
    items: list[ReviewItem] = []

    for index, finding in enumerate(getattr(guardrail_report, "inconsistencies", [])):
        items.append(
            ReviewItem(
                item_id=_item_id(deal_id, ItemKind.INCONSISTENCY, index),
                deal_id=deal_id,
                kind=ItemKind.INCONSISTENCY,
                summary=("Figures in the file cannot all be true; the memo is held."),
                evidence={
                    "identity": finding.identity,
                    "expected": f"{finding.expected:,.0f}",
                    "statements show": f"{finding.actual:,.0f}",
                    "gap": f"{finding.gap:.1%}",
                },
            )
        )

    if getattr(guardrail_report, "blocked", False) and not getattr(
        guardrail_report, "inconsistencies", []
    ):
        verdict = guardrail_report.verdict
        items.append(
            ReviewItem(
                item_id=_item_id(deal_id, ItemKind.BLOCKED, 0),
                deal_id=deal_id,
                kind=ItemKind.BLOCKED,
                summary="The memo's recommendation is not supported by its metrics.",
                evidence={
                    "memo states": guardrail_report.stated,
                    "metrics permit at most": verdict.required if verdict else "?",
                    "reason": guardrail_report.block_reason,
                },
            )
        )

    if getattr(guardrail_report, "more_conservative", False):
        verdict = guardrail_report.verdict
        items.append(
            ReviewItem(
                item_id=_item_id(deal_id, ItemKind.CONSERVATIVE, 0),
                deal_id=deal_id,
                kind=ItemKind.CONSERVATIVE,
                summary=(
                    "The memo is more cautious than the current metrics require. "
                    "Usually pro forma leverage after the requested facility."
                ),
                evidence={
                    "memo states": guardrail_report.stated,
                    "current metrics permit": verdict.required if verdict else "?",
                    "leverage": f"{verdict.leverage:.2f}x" if verdict else "?",
                    "dscr": f"{verdict.dscr:.2f}x" if verdict else "?",
                },
            )
        )

    for index, finding in enumerate(getattr(guardrail_report, "findings", [])):
        items.append(
            ReviewItem(
                item_id=_item_id(deal_id, ItemKind.INJECTION, index),
                deal_id=deal_id,
                kind=ItemKind.INJECTION,
                summary=(
                    "A document contains text addressed to an automated reader. "
                    "The memo was unaffected; this is a fact about the counterparty."
                ),
                evidence={"pattern": finding.kind, "excerpt": finding.excerpt},
                citations=(f"[{finding.document}#p{finding.page}]",),
            )
        )

    for index, (figure, derivation) in enumerate(
        sorted(getattr(memo_result, "reconstructions", {}).items())
    ):
        items.append(
            ReviewItem(
                item_id=_item_id(deal_id, ItemKind.RECONSTRUCTION, index),
                deal_id=deal_id,
                kind=ItemKind.RECONSTRUCTION,
                summary=(
                    f"{figure} appears on no page; it follows arithmetically from "
                    f"figures that do. Confirm the derivation is the intended one."
                ),
                evidence={"figure": figure, "derivation": derivation},
            )
        )

    for index, figure in enumerate(getattr(memo_result, "ungrounded", ())):
        items.append(
            ReviewItem(
                item_id=_item_id(deal_id, ItemKind.LOW_CONFIDENCE, index),
                deal_id=deal_id,
                kind=ItemKind.LOW_CONFIDENCE,
                summary=(
                    f"{figure} is neither quoted from a cited page nor derivable "
                    f"from figures that are."
                ),
                evidence={"figure": figure},
            )
        )

    return items


def format_queue(queue: ReviewQueue, deal_id: str | None = None) -> str:
    pending = queue.pending(deal_id)
    if not pending:
        return "review queue is empty"
    lines = [f"{len(pending)} item(s) pending, most consequential first", ""]
    lines += [item.render() for item in pending]
    blocking = queue.blocking(deal_id)
    if blocking:
        lines += [
            "",
            f"{len(blocking)} blocking item(s): the memo cannot go to committee "
            f"until these are decided.",
        ]
    return "\n".join(lines)
