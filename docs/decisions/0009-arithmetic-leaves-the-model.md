# ADR 0009 — Arithmetic leaves the model, and every figure must be traceable

Status: Accepted
Date: 2026-08-05
Relates to: ADR 0008 (mechanical metrics over judged metrics)

## Context

M6 ran Claude over the 127 gold questions and the grounding check reported
**seventeen hallucinated figures**. Reading them was the most useful thing that
happened in that milestone. They were:

- seven leverage ratios (`3.35x`, `2.26x`, `3.07x`, …)
- three debt totals (`16,672,651`, `8,507,027`, `18,842,975`)
- one units conversion (`32,041,000` — correctly multiplying a figure printed in
  thousands)
- a year with a full stop attached (`2025.`)

Not one was fabricated. Every ratio and total was correct arithmetic on figures
that were on cited pages, and the conversion was the exact behaviour the
`units_in_thousands` defect exists to test.

Two separate problems sat behind that.

**The metric was wrong.** Calling correct arithmetic a hallucination makes the
alarm useless: a number that must stay at zero was never zero, for an honest
reason, so nobody would look at it.

**The system was also wrong, less obviously.** A figure the model computed in its
head has no provenance. It might be right; nothing about the output says so, and
nothing can check it. In a credit memo that is the same problem as an invented
figure wearing better clothes.

## Decision

### Arithmetic moves out of the model into a calculator tool

`tools.py` computes leverage, DSCR, total debt, current ratio, concentration,
growth and unit conversions. Every calculation returns a `Computation` carrying
its **inputs, each tagged with the document and page it came from**, the formula,
and the result.

Grounding then checks the *inputs* — quoted figures on cited pages — instead of
guessing at the output. A derived figure becomes as traceable as a quoted one.

Calculations that cannot be performed honestly raise rather than returning a
sentinel. Zero EBITDA is not infinite leverage; it is a business with no
operating profit, which is a finding to state in words. A memo showing DSCR as
`0.00` because a divisor was missing is wrong in a way nobody notices, while a
memo with no DSCR line is obviously incomplete.

### Grounding gains a fourth bucket

`grounded` (on a cited page), **`derived`** (produced by a recorded calculation),
`uncited` (real figure, wrong provenance), `absent` (invented).

`derived` counts as traceable. This is not a loosening — a figure only lands
there when a recorded calculation produced it from inputs that carry their own
pages. It is the distinction between "computed, here is how" and "came from
nowhere", which raw text cannot express and a credit memo depends on.

### The memo is a graph, not a prompt

    plan → gather → compute → draft → verify → (revise) → done

One large prompt works often enough to demo and fails in ways that cannot be
diagnosed, because there is no point at which you can ask "was the leverage input
right" separately from "was the conclusion right".

**gather** retrieves and extracts labelled figures with provenance. **compute**
runs the calculator; no arithmetic happens anywhere else. **draft** writes prose
from figures that already exist. **verify** is a real gate — an ungrounded figure
sends the draft back with the offending numbers named.

Extraction and arithmetic are the two things a language model is worst at and
the two things a credit memo cannot get wrong, so both happen before the drafter
is called. What is left for the model is narrative and judgement, which is what
it is actually good at.

### One revision, not unlimited

A model that cannot fix a grounding failure when told exactly which figure is
wrong will not fix it on the fourth pass, and an unbounded loop turns a bad memo
into a bad memo that costs ten times more.

### Policy thresholds are constants in a module

`policy.py` holds 3.5x leverage, 1.25x DSCR, the facility band. A threshold
buried in a prompt cannot be tested and drifts silently whenever the prompt is
edited. It also lets the verifier recognise them: *"leverage of 3.12x is within
the 3.5x policy limit"* states one measured figure and one rule, and the rule has
no page to cite.

### Non-final documents never supply figures

A draft or superseded page can still be retrieved and cited when someone asks
about it. What it must not do is silently supply the numbers a memo is built on.
The extractor skips them, and their presence in the file is itself reported as a
finding — a memo that quietly ignores a superseded statement gives the committee
no way to know the file contained a contradiction.

## Consequences

**Good.** All memos verify with zero ungrounded figures and zero revisions on
the template drafter, which is the control proving the gate measures the drafter
rather than waving everything through.

**Good.** `write_memo.py` exits 4 on an unverifiable figure. That failure fails
the command rather than printing a warning.

**Good.** Every derived figure has an audit line an analyst can check without
reading code:
`Total debt / EBITDA = 3.12x [total debt / EBITDA: total debt=8,025,829, EBITDA=2,575,918] [...#p2] [...#p1]`

**Three bugs found while building this**, all in the verifier rather than the
generator:

1. Contexts were keyed by `(document, page)`, so a page contributing both a
   table chunk and a prose chunk lost one. The cash figure was reported as
   ungrounded despite being printed on a cited page and read off that page by the
   extractor moments earlier.
2. Trailing punctuation was captured as part of a figure — `469,828,` inside a
   list, `2025.` at the end of a sentence.
3. Policy thresholds were flagged as ungrounded borrower figures.

All three now have regression tests. The pattern is worth naming: **every one was
the metric misfiring, not the system misbehaving**, and each was only visible
because a drafter that provably cannot invent figures was there to compare
against.

**Cost.** The extractor matches statement rows by exact label. A borrower whose
accountant writes "Turnover" instead of "Revenue" yields nothing. That is
acceptable for a generated corpus and would need fuzzy matching or an extraction
model against real documents.

**Cost.** The template drafter cannot notice anything the template did not
anticipate. It is a floor, not a product.

**Accepted limitation.** The Anthropic drafter is wired and unmeasured here, as
no API key was available in the build environment.

## Alternatives considered

**Model-native tool calling.** Let Claude call the calculator itself. More
flexible, and it puts the choice of *which* calculation to run back inside the
model — so a memo could silently omit DSCR because the model did not think to ask
for it. A credit memo has a required shape; computing every supported metric and
letting the drafter narrate them guarantees the shape.

**No graph, just a pipeline of functions.** Would work today. The conditional
retry edge depends on state several nodes contribute to, and expressing that as a
graph keeps control flow visible instead of buried in a while loop with three
flags. It also makes the human-in-the-loop step at M10 an edge rather than a
rewrite.

**Verifying with an LLM.** Rejected for the same reason as M6: whether a figure
appears on a page is decidable by string comparison, and routing it through a
model makes the most important check the least reliable one.
