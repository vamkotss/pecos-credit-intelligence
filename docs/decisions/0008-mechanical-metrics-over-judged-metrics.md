# ADR 0008 — Mechanical metrics first, a judge only where nothing else works

Status: Accepted
Date: 2026-08-05
Relates to: ADR 0007 (rank fusion and authority)

## Context

M5 established that the right page reaches the top five every time. This
milestone is about what happens next: turning retrieved pages into an answer,
and deciding whether that answer is any good.

The default approach in RAG projects is LLM-as-judge for everything — faithfulness,
relevance, correctness, groundedness, all scored by a model reading the output of
another model. It is fast to build and produces a dashboard full of numbers.

It is also the wrong instrument for the failure that matters here. In commercial
lending the mistake that ends careers is **a figure in a credit memo that is not
in the file**. Not a clumsy sentence — a number that came from nowhere and got
approved. Asking a language model whether a number appears on a page, when the
page is right there and the check is a string comparison, is choosing the least
reliable available method for the most important question in the system.

## Decision

### Four things are decided mechanically, with no model involved

**Numeric grounding.** Every figure in an answer is looked up in the text of the
pages that answer cited. Reported in three buckets, because the difference
between them is the difference between a citation bug and a hallucination:

- *grounded* — on a cited page. Correct.
- *uncited* — in the retrieved context, but on a page the answer did not cite.
  The figure is real; the provenance is wrong. Recoverable.
- *absent* — nowhere in the retrieved context. Invented.

The last is reported as a **raw count, never a rate**, so a single invented
figure cannot disappear into a reassuring average.

**Citation correctness.** Does the answer cite the page the manifest says holds
the answer? Decidable against ground truth.

**Refusal.** Scored only on the question that has no answer in the corpus,
returning `None` elsewhere rather than `True`, so it is not diluted across
seventy cases where refusing would be a failure. Paired with an **over-refusal
rate** over answerable questions, because a system that refuses everything scores
perfectly on the first metric and is worthless.

**Injection resistance.** Heuristic, and labelled as one; M8 does the adversarial
work.

### Citations are validated, not trusted

Every citation is parsed out and checked against the pages actually retrieved.
One pointing at a page that was never retrieved is **dropped**, and the dropped
markers are counted.

An invented citation is worse than an uncited claim. An uncited claim announces
itself; an invented citation reads exactly like evidence and cannot be checked,
which is the property that makes evidence worth anything.

### A judge is used only for what is genuinely a judgement

Faithfulness and answer relevance are not decidable by string comparison. "Does
this sentence follow from that page" is a judgement, and those two sit behind a
`Judge` protocol.

`OverlapJudge` is the default: token overlap between answer and context. It is
**labelled a proxy everywhere it is reported**, because it measures overlap,
which correlates with faithfulness and is not faithfulness. An answer that
recombines context words into a false claim scores well; a correct paraphrase
scores badly. It exists so the metric shape runs in CI without an API key and so
a catastrophic regression is caught for free. A smoke alarm, not a measurement.

`AnthropicJudge` is selected by flag, imports lazily, and grades only those two
things.

### The offline generator is a floor, not a competitor

`ExtractiveGenerator` quotes the best-matching line from the retrieved pages and
cites it. Because it only quotes, it is grounded and citation-safe **by
construction** — measured: 100% grounding, 0 hallucinated figures, 0 invented
citations, 100% refusal accuracy, 0% over-refusal.

That is the point. Any language model scoring below those numbers is measurably
making things worse. What the baseline cannot do is combine two pages, compute a
ratio, or recognise that "how exposed is the borrower" asks about concentration —
and its **24.2% citation accuracy and 21.2% answer accuracy** are the gap the
model is being paid to close, stated as a number rather than assumed.

## Consequences

**Good.** The metric that matters most cannot regress silently, cannot cost
money, and cannot disagree with itself between runs.

**Good.** CI runs the whole harness free and hermetically, so the eval gate at
M9 has something to gate on that does not require a key on the runner.

**Good.** The baseline gives every future generator a floor to beat rather than a
vacuum to be praised against.

**Cost.** The overlap proxy is weak. Its faithfulness score of 0.475 on the
baseline is close to meaningless in absolute terms — useful only as a trend and
a crash detector.

**Cost.** Injection resistance is pattern-based. It correctly distinguishes
obeying an instruction from reporting one, and it would not survive a determined
adversary. That is M8's job.

**Bug found and fixed.** The first version of figure extraction ran over the raw
answer text including citation markers, so `[06_borrower_questionnaire.pdf#p1]`
contributed the "figures" 06 and 1 and a correctly cited answer was reported as
containing eight hallucinated figures. A metric that fires on its own citation
format manufactures alarm about the exact behaviour it exists to encourage. There
is now a regression test.

## Alternatives considered

**RAGAS as a library.** Well-designed and would have given faithfulness, answer
relevance and context precision out of the box. Rejected because it routes every
metric through an LLM, including ones that are decidable arithmetically here, and
because its context-precision metric assumes chunk-level relevance labels that
this corpus expresses at page level. The metric *shapes* are borrowed; the
implementation is not.

**LLM-as-judge for correctness.** Rejected outright. The manifest holds the exact
figure. Asking a model whether the answer matches, when `==` will answer it, adds
cost, latency and disagreement for nothing.

**Skipping the offline generator and testing only against Claude.** Would have
made every test need a key, cost money on every CI run, and made failures
ambiguous between a harness bug and a model regression.

**Self-consistency sampling.** Generating several answers and measuring
agreement. Real value against hallucination, and deferred: it multiplies cost by
the sample count and is worth adding once there is evidence that grounding
failures are sampling-driven rather than prompt-driven.
