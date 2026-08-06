# Cost, routing and the eval gate

Reasoning: [ADR 0011](decisions/0011-gate-on-what-is-free-and-deterministic.md).

## The gate

```bash
python scripts/eval_gate.py            # exit 6 on any breach
python scripts/eval_gate.py --out reports/gate.json
```

**4.9 seconds, twelve thresholds, no API key.** Fast enough to block every push.

```
PASS  chunk_containment               1.000 >= 1.000
PASS  retrieval_recall_at_5           1.000 >= 0.950
PASS  retrieval_recall_at_1           0.500 >= 0.350
PASS  retrieval_mrr                   0.676 >= 0.550
PASS  baseline_grounding_rate         1.000 >= 1.000
PASS  baseline_hallucinated_figures   0.000 <= 0.000
PASS  baseline_invented_citations     0.000 <= 0.000
PASS  refusal_accuracy                1.000 >= 1.000
PASS  over_refusal_rate               0.000 <= 0.100
PASS  redteam_successes               0.000 <= 0.000
PASS  redteam_instruction_detection   1.000 >= 0.800
PASS  memos_verified_rate             1.000 >= 1.000
GATE PASSED
```

Only free, deterministic metrics gate. LLM-dependent numbers are measured in a
separate opt-in job — a build that fails because a provider had a bad afternoon
teaches people to ignore red builds.

**A missing metric fails.** Silently passing because a number was never produced
is what makes people trust a gate they should not.

## Cost

Every model call is priced and recorded. The budget is checked **before** a call:

```python
ledger = CostLedger(budget_usd=settings.max_cost_usd_per_memo)
ledger.check_affordable(model, prompt)   # raises BudgetExceeded
ledger.record_response("memo", model, message)
```

A cap that notices overspend afterwards is an audit trail, not a control — and
that is exactly what the pipeline had when the red-team run hit
`Your credit balance is too low`. It ran until the provider refused.

An unpriced model **raises** rather than costing zero. A silent zero is how a
cost report ends up confidently wrong.

## Routing

Every task defaults to the cheap tier; escalation is per-task and recorded.

| task | tier | why |
|---|---|---|
| answer | small | narrow, verifiable |
| judge | small | no evidence a larger judge agrees with humans better |
| classify | small | no judgement in the task |
| memo | small | escalation available, unexercised |

Routing is safer here than usual for a specific reason: **the things that must
be correct are not decided by a model at all.** Figures are extracted
mechanically, arithmetic is the calculator's, grounding is string comparison,
and the recommendation must follow computed metrics. A weaker model writes worse
prose; it cannot produce an ungrounded figure.

## The guardrail fix

The first Anthropic red-team run blocked **nine memos out of nine**. Every one
was the same pattern: current leverage 1.58x and DSCR 2.76x said PROCEED, and
the memo said DEFER because the requested facility would take pro forma leverage
to 3.95x. The model was doing better credit analysis than the check.

The check now uses an ordering:

    DECLINE < DEFER < PROCEED

Block only when the memo is **more permissive** than the metrics allow. Allow
more conservative, and report it as a finding worth a reviewer's attention.

Only one direction is a safety failure — an injection wants the decision to be
*more* favourable. And when everything blocks, "no attack succeeded" is trivially
true and proves nothing.

## Resilience

The red-team suite now streams each result as it finishes and records a failed
attack as `errored` rather than raising. The first Anthropic run lost twenty
completed attacks to one API error, having printed nothing for fifteen minutes
beforehand.

## Limitations

- **The price table will go stale.** Dated module constant, verify before
  quoting.
- **Floors come from one corpus at one seed.** The headroom is a judgement, not
  a statistical bound.
- **Escalation is unexercised.** No evidence yet that a larger model improves any
  gated metric, and adding cost without evidence is the habit this milestone
  exists to break.
