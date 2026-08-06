# ADR 0012 — The system does not decide; it makes the decision reviewable

Status: Accepted
Date: 2026-08-06
Relates to: ADR 0011 (gate on what is free and deterministic)

## Context

Nine milestones built a pipeline that reads loan documents, computes credit
metrics, writes a memo, and refuses to release one contradicting its own
arithmetic. Three things it produced were *reported* and then went nowhere:

**Reconstruction candidates (M7).** "3.95x follows from (total debt + facility
requested) / EBITDA." Almost certainly right; coincidental matches are possible
at three significant figures. The system found the arithmetic and cannot know
whether that arithmetic is what the writer meant.

**Conservative recommendations (M9).** The memo says DEFER where the computed
metrics permit PROCEED, because pro forma leverage would breach policy. The memo
is right and the metrics are incomplete.

**Injection findings (M8).** A broker's note contained text addressed to an
automated reader. The attack failed. It remains a fact about the counterparty.

Each is a judgement the pipeline correctly declines to make.

## Decision

A review queue with four properties.

### An item must be decidable without opening the source

Every item carries its evidence inline: the figure, the derivation, the metrics,
the excerpt, the citation. A queue that sends a reviewer back to the PDF for
each item costs more time than it saves and gets abandoned — and an abandoned
queue is worse than none, because the pipeline still behaves as though someone
is reading it.

### Decisions are append-only, attributed and timestamped

`items.jsonl` holds current state; `events.jsonl` is the log. Both exist because
they answer different questions: what is true now, and how it got there. A
credit file is a regulated artefact and "who approved this" is asked years later,
so mutating a decision in place destroys the only record of what was known at
the time.

A reviewer name is **required**, not optional. An unattributed approval is worth
little, and making the field mandatory is cheaper than discovering later that
half the queue was cleared by "system".

### Only findings of wrongness block release

The split is between *the system is unsure* and *the system found something
wrong*.

Blocking: a broken accounting identity, a recommendation contradicting the
metrics. Non-blocking: reconstruction candidates, conservative recommendations,
injection findings — the memo is coherent and something wants confirming.

`ESCALATE` is a real outcome. A reviewer who cannot decide should be able to say
so rather than guessing or leaving an item pending forever.

### Nothing expires

An item that quietly released itself after a week would turn the queue from a
control into a delay, and the failure would be silent. A blocking item left
unreviewed keeps the memo held, indefinitely, visibly.

## Consequences

**Good.** Across three deals the queue produced **one** item — the planted
injection finding. Clean memos queue nothing, which is the common case and the
point. A queue that fires on every memo is a queue nobody reads.

**Good.** `review_queue.py status` exits 7 while anything is held, so "is
anything waiting on a person" is answerable by a script rather than by asking.

**Good.** Rebuilding is idempotent. Re-running `build` will not duplicate an
item or overwrite a decision someone already made.

**Cost.** The queue is a CLI over JSONL. That is the right shape for
demonstrating the control and the wrong shape for a lending desk, which wants a
web queue with assignment and SLAs. The data model would survive that change;
the interface would not.

**Cost.** Items are per memo. A reviewer seeing the same reconstruction pattern
across twelve deals must decide it twelve times. Grouping by pattern is the
obvious improvement and is not built.

**Accepted limitation.** Nothing learns from decisions. If a reviewer accepts the
same derivation shape forty times, the system keeps asking. Using that history to
auto-accept would be genuinely useful and would also quietly convert reviewed
judgements into unreviewed ones, which is the thing this milestone exists to
prevent. It needs its own design, not a shortcut.

## Alternatives considered

**Auto-accepting high-confidence reconstructions.** A confidence score on a
coincidental arithmetic match is a guess about a guess. The whole reason
reconstructions reach a human is that the system cannot tell a real derivation
from a coincidence.

**Blocking on everything pending.** Would make the queue a bottleneck rather than
a control, and would train reviewers to clear items without reading them —
converting a safety mechanism into a rubber stamp, exactly as happened to the
M7.2 verifier.

**Storing decisions in the memo file.** Simpler, and it puts mutable review state
inside an artefact that ought to be immutable once written. The memo is what was
concluded; the queue is what was questioned about it.
