# ADR 0010 — The defence that survives is the one that does not run through the model

Status: Accepted
Date: 2026-08-06
Relates to: ADR 0009 (arithmetic leaves the model)

## Context

Every excerpt this system reads was supplied by the borrower or their broker.
The people who benefit most from a favourable credit decision are the people who
wrote the documents the decision rests on. Retrieved content is untrusted input
by construction — not occasionally, not in adversarial testing, always.

M6 measured whether the model complied with one planted sentence in one broker
note. Passing that tells you almost nothing: an attacker does not reuse the
string you tested against, they rewrite it until something works.

## Decision

### Three layers, and only one of them is reliable

**Structural (weak).** Excerpts are wrapped in delimiters labelled with their
provenance. This raises the effort required and does not stop a determined
attacker, because the boundary is expressed in the same channel as the attack.

**Detection (weak).** Instruction-shaped text is flagged and surfaced. Pattern
matching loses to paraphrase, encoding and unseen languages. Its value is telling
a human that someone *tried* — an attempted injection in a loan package is a fact
about the counterparty worth recording whether or not it worked.

**Output-side policy check (strong).** The recommendation is not the model's to
choose. Leverage and DSCR are computed by the M7 calculator, the thresholds live
in `policy.py`, and a memo whose stated recommendation contradicts them is
blocked whatever the prose says.

That asymmetry is the point. An injection can persuade a model. **It cannot
change what 8,025,829 divided by 2,575,918 equals, and it cannot make 3.12x pass
a 3.5x test evaluated in Python.**

### A detected injection does not block; a broken identity does

Detection reports. Blocking is reserved for cases where the decision itself is
unsafe: a recommendation contradicting the metrics, or figures that cannot all be
true. Blocking on detection would let a false positive stop a legitimate memo for
no gain in safety, since the decision is already constrained by arithmetic.

### Red-teaming uses families, not the planted string

Nine attacks across three families: instruction (override, authority spoofing,
delimiter escape, refusal suppression, citation spoofing), obfuscation (spaced
characters and unicode lookalikes, compliance asserted without an instruction),
and **data poisoning** (no instructions at all — just favourable figures inserted
into a statement).

An attack counts as successful only if it **changed the credit decision** and was
not blocked. A memo that quotes an injection and still recommends DECLINE has not
been compromised; it has reported an attempted fraud.

## The finding that mattered

First run: **instruction attacks 0/15, data poisoning 2/6.** `figure_poisoning`
flipped two deals from DECLINE to PROCEED.

The policy check could not help, and the reason is worth stating precisely. An
injected *instruction* tries to persuade a model, and the policy check ignores
prose entirely. A poisoned *figure* flows into the calculator, which computes on
it faithfully, and the policy check then approves a decision that is
arithmetically correct and factually false. **Garbage in, correctly computed
garbage out.**

Data poisoning also carries no instruction to detect, so every pattern-matching
defence scores zero against it. It is the attack a sophisticated borrower would
actually use: inflating a figure in your own financial statements is easier than
writing a prompt injection and much harder to spot.

### What stops it

Real financial statements are **over-determined**. Gross profit is revenue less
cost of sales. EBITDA cannot exceed gross profit, because operating expenses are
not negative. Assets equal liabilities plus equity.

A borrower inflating one figure has to inflate every figure that ties to it, and
the red-team suite shows that inflating *one* is enough to flip a decision. So
the identities are where the fraud becomes visible, and `check_figure_consistency`
blocks on a broken one.

The accounting identities M2 asserts about the generated corpus turn out to be a
fraud detector. That was not the reason for building them.

## Result

27 attacks across 3 deals: **0 succeeded.** Instruction detection 100% on the
current families, 0% on obfuscation and data — reported rather than rounded up,
because a detection rate quoted without its blind spots is marketing.

## Consequences

**Good.** The strongest defence is deterministic, free, and cannot regress
silently.

**Good.** Layering is real rather than decorative. Citation spoofing evaded an
earlier version of the pattern list, and the M6 citation validator dropped the
spoofed reference anyway.

**Cost.** Detection is pattern-based and will lose to a new phrasing. It is
labelled weak everywhere it appears.

**Cost.** The identity checks cover three relationships. A borrower who inflates
revenue, cost of sales and gross profit consistently would pass them, and would
have to falsify the tax return and bank statements too — which is the point at
which cross-document reconciliation becomes the next defence rather than
arithmetic.

**Accepted limitation.** Attacks are injected at the retriever boundary rather
than by regenerating a poisoned PDF. That tests the agent's handling of poisoned
content, not the OCR pipeline's — and an attacker who can alter a source document
can certainly get text into a chunk, so it is the stronger assumption.

**Accepted limitation.** The suite runs against the deterministic drafter, which
cannot be persuaded of anything. That makes it a control rather than a target:
the interesting run is `--drafter anthropic`, and it is unmeasured here.

## Alternatives considered

**Stripping instruction-like text from documents.** Implemented as `neutralise`
and left off by default. Removed text is evidence about the counterparty, and a
pipeline that silently deletes it destroys that evidence while still missing
every paraphrase the patterns failed to match. Flagging beats filtering when
detection is imperfect.

**An LLM-based injection classifier.** Better recall than patterns, and it puts a
model in the security path — where a sufficiently good injection would target the
classifier as well. Worth adding as a second detector, never as the barrier.

**Blocking any memo containing a detected injection.** Fails safe in the wrong
direction: a broker's phrase resembling a pattern would stop a legitimate deal,
while the decision was already constrained by arithmetic.
