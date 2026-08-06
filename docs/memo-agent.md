# The credit memo agent

Reasoning: [ADR 0009](decisions/0009-arithmetic-leaves-the-model.md).

## Running it

```bash
python scripts/write_memo.py --deal PCP-0001
python scripts/write_memo.py --all --out reports/memos
python scripts/write_memo.py --deal PCP-0001 --drafter anthropic --audit
```

The default drafter needs no API key. Exit code **4** if any memo contains a
figure that is neither quoted from a cited page nor produced by a recorded
calculation.

## The graph

```
gather ──► compute ──► draft ──► verify ──┬──► done
                        ▲                 │
                        └──── revise ◄────┘  (once)
```

| node | does | never does |
|---|---|---|
| gather | retrieve per section, extract labelled figures with pages | judge |
| compute | run the calculator | write prose |
| draft | narrative and recommendation | arithmetic, extraction |
| verify | check every figure is quoted or computed | guess |

**Extraction and arithmetic are the two things a language model is worst at and
the two things a credit memo cannot get wrong.** Both happen before the drafter
is called. What is left for the model is narrative and judgement.

## Why the calculator exists

M6 flagged seventeen "hallucinated" figures from Claude. Reading them was the
most useful result of that milestone — they were seven leverage ratios, three
debt totals, one correct thousands conversion, and a year with a full stop
attached. Not one was fabricated.

Two problems sat behind that. The metric was wrong: calling correct arithmetic a
hallucination makes the alarm useless. And the system was wrong too, less
obviously — **a figure a model computed in its head has no provenance.** It might
be right; nothing about the output says so and nothing can check it.

Now every calculation records its inputs and their pages:

```
Total debt / EBITDA = 3.12x  [total debt / EBITDA: total debt=8,025,829,
  EBITDA=2,575,918] [02_financial_statements_comparative.pdf#p2]
  [02_financial_statements_comparative.pdf#p1]
```

An analyst can check that without reading code.

## Grounding has four buckets

| bucket | meaning | verdict |
|---|---|---|
| grounded | on a cited page | pass |
| **derived** | produced by a recorded calculation from cited inputs | pass |
| uncited | in context, wrong page cited | citation bug |
| absent | nowhere | **invented** |

`derived` is not a loosening. A figure only lands there when a recorded
calculation produced it from inputs that carry their own pages.

## Results

Template drafter, all deals:

```
PCP-0001  VERIFIED  figures=19  calcs=4  citations=3  revisions=0
PCP-0002  VERIFIED  figures=19  calcs=4  citations=3  revisions=0
PCP-0004  VERIFIED  figures=19  calcs=4  citations=3  revisions=0
PCP-0007  VERIFIED  figures=19  calcs=4  citations=3  revisions=0
```

Zero ungrounded figures, zero revisions. The template drafter prints only what
the extractor and calculator produced, so it **cannot** invent a figure — which
makes it the control proving the verify gate measures the drafter rather than
waving everything through.

## Three bugs found, all in the verifier

1. **Contexts keyed by (document, page)** dropped chunks. A statements page
   contributes both a table chunk and a prose chunk; the second overwrote the
   first, so the cash figure was reported ungrounded despite being printed on a
   cited page and read off that page by the extractor moments earlier.
2. **Trailing punctuation** was captured as part of a figure — `469,828,` inside
   a list, `2025.` at a sentence end.
3. **Policy thresholds** were flagged as ungrounded borrower figures. "Leverage
   of 3.12x is within the 3.5x policy limit" states one measured figure and one
   rule; the rule has no page to cite.

All three have regression tests. Every one was the metric misfiring, not the
system misbehaving — visible only because a drafter that provably cannot invent
figures was there to compare against.

## Policy lives in code

`policy.py`: 3.5x leverage, 1.25x DSCR, 1.1x current ratio, the $3M–$40M band.
A threshold buried in a prompt cannot be tested and drifts silently whenever the
prompt is edited.

The recommendation follows from the thresholds and is asserted to: a memo whose
conclusion contradicts its own metrics is worse than one with no conclusion,
because it looks reasoned.

## Non-final documents

A draft or superseded page can be retrieved and cited when asked about. It never
supplies figures the memo is built on, and its presence is reported as a finding
— a memo that quietly ignores a superseded statement gives the committee no way
to know the file contained a contradiction.

## Limitations

- **The extractor matches statement rows by exact label.** A borrower writing
  "Turnover" instead of "Revenue" yields nothing. Fine for a generated corpus;
  real documents would need fuzzy matching or an extraction model.
- **The template drafter cannot notice anything the template did not
  anticipate.** It is a floor, not a product.
- **The Anthropic drafter is wired but unmeasured**, no key in the build
  environment.
- **One revision, then stop.** Deliberate, and it means a persistently
  ungrounded draft is reported rather than fixed.
