# ADR 0011 — Gate on what is free and deterministic; meter what is not

Status: Accepted
Date: 2026-08-06
Relates to: ADR 0010 (the defence that does not run through the model)

## Context

Three things came to a head at once.

**The eval numbers were claims, not constraints.** M4 through M8 produced
containment, recall, grounding and red-team figures. All of them lived in
READMEs, where a number describes the past rather than constraining the future.

**The pipeline had no idea what it was spending.** The red-team run against
Claude crashed with `Your credit balance is too low to access the Anthropic
API`. It had run until the provider refused. `max_cost_usd_per_memo` had sat in
the config since M1 and nothing read it — a setting nothing enforces is a
comment.

**The policy guardrail was blocking everything.** The first Anthropic red-team
run blocked nine memos out of nine, every one for the same reason: current
leverage 1.58x and DSCR 2.76x said PROCEED, and the memo said DEFER because the
requested facility would take pro forma leverage to 3.95x.

## Decisions

### Only free, deterministic metrics gate the build

Containment, retrieval recall and MRR, grounding on the extractive baseline,
refusal, and the red-team suite all run offline in **4.9 seconds** and give the
same answer every time. Those block a merge.

Anything needing an API key — Claude's answer accuracy, judged faithfulness — is
measured in a separate opt-in job. Gating on them would put a paid,
non-deterministic dependency in front of every push, and **a build that fails
because a provider had a bad afternoon teaches people to ignore red builds.**

Two floors are absolutes rather than measurements. Containment is 1.0 because a
fact that does not survive chunking cannot be retrieved by anything downstream,
so any drop invalidates every retrieval number measured after it. Red-team
successes must be 0 because "some attacks now change credit decisions" is not a
metric to trend, it is an outage.

The rest sit below their measured values with headroom. **A gate that flaps gets
disabled within a week, and a disabled gate is worse than none because everyone
still believes it is running.**

A metric that is *absent* fails rather than being skipped. Silently passing
because a number was never produced is the failure mode that makes people trust
a gate they should not: the build is green and nothing was checked.

### Costs are metered, and the budget is a pre-flight check

Every model call is priced from a table with a date on it. An unpriced model
raises rather than costing zero — a silent zero is how a cost report ends up
confidently wrong.

`CostLedger.check_affordable` raises **before** a call that would breach the
budget. A cap that notices overspend after the fact is an audit trail, not a
control.

### Routing defaults to the cheap tier, and escalation is explicit

Extraction and classification are narrow, verifiable tasks a small model does as
well as a large one. Judgement — writing the memo, weighing conflicting evidence —
is where a stronger model earns fifteen times the price.

Routing here is safer than it usually is, for a specific reason: **the things
that must be correct are not decided by a model at all.** Figures are extracted
mechanically, arithmetic is done by the calculator, grounding is checked by
string comparison, and the recommendation must follow computed metrics. A weaker
model produces worse prose; it cannot produce an ungrounded figure or an
unsupported recommendation, because those paths do not run through it.

Without those checks, routing would be a straightforward gamble.

### The policy guardrail is an ordering, not an equality

    DECLINE < DEFER < PROCEED

Block when the memo is **more permissive** than the metrics allow. Allow more
conservative, and report it as a finding.

Only one direction is a safety failure. An injection wants the decision to be
more favourable; a memo more cautious than the arithmetic requires is doing its
job. The pro forma case is the obvious one, and it is exactly the analysis a
committee wants surfaced rather than suppressed.

This also restored the red-team result's meaning. **When everything blocks, "no
attack succeeded" is trivially true and proves nothing.**

### Long runs stream and survive failures

The red-team suite reports each attack as it finishes and records a failed
attack as `errored` rather than raising. The first Anthropic run lost twenty
completed attacks to one API error, and printed nothing for fifteen minutes
beforehand — indistinguishable from a hang.

## Consequences

**Good.** The gate runs in 4.9s and passes on the current corpus. It is cheap
enough to block every push rather than being a thing someone runs occasionally.

**Good.** An errored attack cannot be counted as a success, so a failing suite
degrades into a partial result rather than nothing.

**Cost.** The price table will go stale. It is a dated module constant rather
than a number buried in a function, so staleness is visible.

**Cost.** The floors were set from one corpus at one seed. A different seed would
move recall@1 by a few points, and the headroom is a judgement rather than a
statistical bound.

**Accepted limitation.** Routing is wired and every task still defaults to the
small tier. Escalation exists and is unexercised, because there is no evidence
yet that a larger model improves any metric the gate measures — and adding cost
without evidence is the habit this milestone exists to break.

## Alternatives considered

**Gating on LLM metrics too.** Rejected: paid, non-deterministic, and it makes
red builds ambiguous between "you broke something" and "the API was slow".

**A ratchet that raises floors automatically on improvement.** Appealing, and it
turns a lucky run into a permanent obligation. Floors should move when someone
decides they should.

**Per-token budgets rather than per-memo.** Tokens are the wrong unit for the
decision being made. The question is "is this memo worth its cost", and dollars
are what that question is asked in.
