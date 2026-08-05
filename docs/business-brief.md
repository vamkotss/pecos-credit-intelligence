# Business brief — Pecos Capital Partners

## The company

Pecos Capital Partners is a middle-market commercial lender headquartered in
Fort Worth, Texas. It originates senior secured loans of \$3M to \$40M against
operating businesses and income-producing commercial real estate across Texas,
Oklahoma, and New Mexico. Roughly 340 new credit requests reach the underwriting
desk each year; about 90 close.

## The pain point

Every credit request arrives as a **loan package**: a folder of PDFs assembled by
a borrower's accountant, a broker, or the borrower themselves. A typical package
runs 180–600 pages and contains some mix of:

- Three years of financial statements (audited, reviewed, or company-prepared)
- Business and personal tax returns
- A rent roll and operating statement, if there is real estate collateral
- The existing loan agreement, if this is a refinance
- An appraisal or broker opinion of value
- A debt schedule, an A/R ageing, and a borrowing-base certificate
- A personal financial statement for each guarantor

A credit analyst reads that package and writes a **credit memo**: a 6–10 page
document that states the borrower's leverage, coverage, and liquidity, tests
them against Pecos's credit policy, identifies the risks, and recommends
approve, approve-with-conditions, or decline. The memo goes to a credit
committee that meets weekly.

Two things are true about this work simultaneously:

1. **It is slow.** Reading and tying out a package takes an analyst 6–8 hours.
   With three analysts, the desk clears about nine memos a week and carries a
   two-to-three-week backlog in busy quarters. Brokers route deals to whoever
   answers first, so cycle time is directly a revenue problem.

2. **It is unforgiving.** A memo that misstates EBITDA by transposing two
   digits from a scanned income statement goes to committee as fact. Pecos has
   had two such incidents in four years. Neither caused a loss, but both
   destroyed the committee's willingness to trust anything it did not verify
   itself, which added time back to the process.

That combination is what makes this a hard problem rather than a summarisation
demo. A system that is fast but occasionally invents a number is *worse than
useless here* — it is a liability, because it produces confident output that a
busy committee will not re-verify.

## Why the documents are genuinely hard

The corpus generator (M2) reproduces every one of these deliberately:

| Defect | Why it breaks naive RAG |
|---|---|
| Scanned pages, 200–300 DPI, some rotated 90°/180° | Text layer is absent; OCR quality gates the whole pipeline |
| Financial tables spanning 2–3 pages | A chunker that splits on token count severs a table from its header row, so "12,480" loses the label "Total Revenue" |
| Multi-column statement layouts | Naive text extraction interleaves columns and produces numeric nonsense |
| Three fiscal-year columns side by side | The model must bind a figure to the *right year*, the single most common silent error |
| Restated prior-year figures | Two different values for the same label are both correct; which one applies depends on the document date |
| Handwritten annotations and stamps | Noise that OCR converts into plausible-looking tokens |
| Inconsistent entity naming across documents | "Big Bend Fabrication LLC" / "Big Bend Fab." / "BBF Holdings" |
| Negative numbers in parentheses, "(1,204)" | Sign errors that flip coverage ratios |
| Units drift: thousands on one page, units on the next | Off-by-1000 errors that look like a real number |
| **A poisoned document** containing an instruction addressed to the reader | Prompt injection arriving inside legitimate borrower content |

## Stakeholders

| Stakeholder | What they need | How this system is judged by them |
|---|---|---|
| Credit analyst | To stop transcribing and start analysing | Hours saved per deal; does the draft need heavy rework? |
| Chief Credit Officer | Committee memos that are defensible | Zero ungrounded figures; every number clickable to a source page |
| Head of Originations | Faster turnaround to brokers | Median days from package receipt to committee-ready memo |
| Compliance / audit | An evidence trail | Can we reconstruct which page produced which number, months later? |
| Whoever runs this at 2 a.m. | It fails loudly, not silently | Refusal rate, alerting, review queue depth |

## Success metrics

**Primary (the thing the project exists to prove)**

- **Numeric grounding: zero tolerance.** Every figure in a generated memo must
  resolve to an extracted value on a cited page. The eval suite fails the build
  on a single ungrounded number, not on a percentage threshold.

**Secondary**

| Metric | Baseline | Target |
|---|---|---|
| Retrieval recall@10 on a labelled question set | — | ≥ 0.90 |
| Faithfulness (RAGAS + LLM-as-judge, adjudicated) | — | ≥ 0.85 |
| Correct fiscal-year binding on multi-year tables | — | ≥ 0.95 |
| Injection attempts that alter agent behaviour | — | 0 of the red-team set |
| Cost per memo | — | ≤ \$0.50 |
| Analyst hours per deal (modelled, with stated assumptions) | 6–8 hrs | ≤ 2.5 hrs |

## Assumptions

1. Synthetic corpus, seeded and version-controlled. Real loan packages contain
   PII and are not distributable; a generator that *plants* known defects is
   strictly better for evaluation anyway, because ground truth is knowable.
2. English-language documents only.
3. Credit policy thresholds are simplified to a documented, testable rule set.
4. The system **drafts**; a human approves. Full autonomy is out of scope by
   design, and the review queue (M10) is where that design shows up.

## Risks and how the build addresses them

| Risk | Mitigation | Milestone |
|---|---|---|
| The model invents a plausible figure | Calculator tool + grounding check that fails the memo | M7, M8 |
| Retrieval looks fine on cherry-picked questions | Labelled question set built *before* the retriever | M6 |
| Evals rot as the pipeline changes | Eval regression gate in CI with a committed baseline | M6 |
| Borrower document carries an injection payload | Red-team set; retrieved content never enters the instruction channel | M8 |
| An agent loop burns \$40 on one memo | Hard per-memo cost ceiling; abort with a partial result | M9 |
| "It works on my machine" | Docker Compose, seeded corpus, deterministic CI mode | M1, M2 |

## Explicit non-goals

- Not a lending decision engine. It drafts analysis for a human committee.
- Not a general document Q&A chatbot.
- Not fine-tuned. A LoRA experiment is a stretch goal only, and only if the
  eval harness can prove it beat prompting.
