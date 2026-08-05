# Retrieval

BM25 + dense, fused by reciprocal rank, reranked, weighted by document
authority. Reasoning: [ADR 0007](decisions/0007-rank-fusion-and-authority-tiebreak.md).

## Running it

```bash
python scripts/eval_retrieval.py                    # default offline stack
python scripts/eval_retrieval.py --no-authority     # ablation
python scripts/eval_retrieval.py --embedder st      # sentence-transformers
python scripts/eval_retrieval.py --reranker cross   # cross-encoder
python scripts/eval_retrieval.py --out reports/retrieval.json
```

Index build is about 1.4s for the whole corpus; 76 queries take 0.2s.

## Results

7 deals, 221 chunks, 76 scored queries.

```
recall@1     50.0%
recall@3     81.6%
recall@5    100.0%
recall@10   100.0%
MRR          0.675
```

| by planted defect | R@1 | R@5 | MRR |
|---|---|---|---|
| near_duplicate_draft | 100% | 100% | 1.00 |
| restated_prior_year | 100% | 100% | 1.00 |
| rotated_scanned_page | 100% | 100% | 1.00 |
| table_only_fact | 100% | 100% | 1.00 |
| prompt_injection | 100% | 100% | 1.00 |
| units_in_thousands | 0% | 100% | 0.33 |

## Ablation

| configuration | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|
| BM25 only | 46.1% | 80.3% | 97.4% | 0.669 |
| Dense (LSA) only | 15.8% | 38.2% | 72.4% | 0.366 |
| BM25 + dense | 46.1% | 80.3% | 97.4% | 0.667 |
| BM25 + rerank | 50.0% | 81.6% | 100% | **0.687** |
| Dense + rerank | 34.2% | 72.4% | 100% | 0.567 |
| Full stack | 50.0% | 81.6% | 100% | 0.675 |

**Dense retrieval is not currently earning its place, and that is stated rather
than hidden.** A sweep of the fusion weight found 0.0 optimal on this eval set:

| dense weight | 0.0 | 0.15 | 0.25 | 0.35 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| MRR | 0.679 | 0.666 | 0.648 | 0.620 | 0.620 | 0.599 | 0.593 |

The default is 0.15 — within noise of the best, and enough to keep the hybrid
genuinely operative. The reason for keeping it: the gold questions are generated
from templates and share vocabulary with the documents they target, which
structurally favours lexical matching. Real analyst questions paraphrase far
more, so the eval set is biased against dense retrieval rather than dense
retrieval being useless. `--embedder st` re-measures with a trained encoder.

## What authority weighting buys

Nothing on the gold set — the near-duplicate gold question says "final", which
does the work lexically. Its value shows on the question an analyst would
actually ask.

*"What was EBITDA in FY2025?"* on the near-duplicate deal:

| | top 3 |
|---|---|
| authority **off** | final (1.0000), **draft (0.9840)**, final |
| authority **on** | final (1.0000), final (0.8148), broker note |

A 1.6% margin is noise; a near-duplicate pair separated by noise will flip on
any small change. What the weighting buys is not a recall point — it is keeping a
contradictory document out of the agent's context at M7.

## Design notes

**The tokeniser preserves whole figures.** `\w+` turns `32,041,248` into three
tokens and every large figure collides with every other sharing a fragment.
Numeric tokens are also emitted as bare digits, so `32041248` and `32,041,248`
match either way.

**Query stopwords are removed, because IDF inverts on this corpus.** A loan
package is mostly tables, so ordinary English words are *rare*. Measured: `what`
scored IDF 3.12 against `ebitda` at 2.61. The symptom was the broker's cover
note — the only chunk in flowing prose, containing no figures — ranking first for
almost every financial question. Removing stopwords took recall@1 from 13.2% to
39.5%.

**Ranks count distinct pages, not chunks.** A hit is a page, and a page
contributing both a table and a prose chunk would otherwise burn two slots. This
alone took recall@5 from 98.7% to 100%.

**Indexes are per deal.** A lending question is always about one borrower.
Scoping at index time also means IDF reflects one package's vocabulary.

**Every result keeps its per-stage scores** — BM25 rank, dense rank, fused score,
rerank score, authority weight. That is what answers "why did this rank here",
which is the question that comes up every time retrieval underperforms.

## Limitations

- **Remaining rank-1 failures are balance-sheet questions losing to the cash
  flow statement**, which legitimately mentions cash, debt and year-end. A
  lexical reranker cannot resolve this; it is what a cross-encoder is for, and it
  is unmeasured here.
- **Derived metrics are scored against the page you would compute from.** Other
  statement pages are defensible, so their recall@1 understates real performance.
- **Indexes rebuild in memory each run.** 1.4s; not worth persisting yet.
- **The cross-encoder and sentence-transformer paths are wired but unmeasured**,
  because model downloads were unavailable in the build environment.
