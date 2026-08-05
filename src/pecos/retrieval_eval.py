"""Score retrieval against the ground-truth manifest (M5).

WHAT COUNTS AS A HIT
--------------------
A retrieved chunk hits when its **(document, page)** matches the page the
manifest cites -- not when its text happens to contain the answer.

That distinction is deliberate. Scoring on text containment would reward a
retriever that surfaced the right number from the wrong document, which is
exactly the failure the near-duplicate defect is built to catch: the DRAFT
statements contain figures that look right and are not. A citation to the wrong
page is a wrong answer in a credit memo, however plausible the number.

Page-level scoring also matches how M4 anchors chunks and how the manifest
records ground truth, so all three layers agree on what "the answer's location"
means, with no translation step where a bug could hide.

WHY THE BREAKDOWN MATTERS MORE THAN THE HEADLINE
------------------------------------------------
"Recall@5 is 0.88" is not actionable. "Recall@5 is 0.94 overall but 0.31 on
`table_only_fact`" names the next piece of work. Every result here is broken
down by defect tag and by fact type for that reason -- it is the payoff for
planting the defects in the first place, and it is what the CI eval gate at M9
will fail on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pecos.retrieval import HybridRetriever

DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class QueryResult:
    fact_id: str
    deal_id: str
    question: str
    fact_type: str
    defect_tag: str | None
    gold_document: str
    gold_page: int
    retrieved: list[tuple[str, int]]  # (document, page) in rank order
    rank: int | None  # 1-based rank of the first hit, None if never found

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


@dataclass
class RetrievalReport:
    ks: tuple[int, ...] = DEFAULT_KS
    results: list[QueryResult] = field(default_factory=list)
    skipped_unanswerable: int = 0
    skipped_out_of_scope: int = 0

    @property
    def n(self) -> int:
        return len(self.results)

    def recall_at(self, k: int, subset: list[QueryResult] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.hit_at(k)) / len(rows)

    def mrr(self, subset: list[QueryResult] | None = None) -> float:
        """Mean reciprocal rank.

        Reported alongside recall because they answer different questions.
        Recall@5 says whether the answer made the shortlist; MRR says how far
        down it sat. A system that always ranks the answer fifth and one that
        always ranks it first have identical recall@5 and very different value
        to whatever reads the results -- and at M7 that is an LLM with a limited
        context budget and a documented bias toward what it sees first.
        """
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return sum(1.0 / r.rank if r.rank else 0.0 for r in rows) / len(rows)

    def by_defect(self) -> dict[str, list[QueryResult]]:
        grouped: dict[str, list[QueryResult]] = defaultdict(list)
        for result in self.results:
            if result.defect_tag:
                grouped[result.defect_tag].append(result)
        return dict(grouped)

    def by_fact_type(self) -> dict[str, list[QueryResult]]:
        grouped: dict[str, list[QueryResult]] = defaultdict(list)
        for result in self.results:
            grouped[result.fact_type].append(result)
        return dict(grouped)

    def misses(self, k: int) -> list[QueryResult]:
        return [r for r in self.results if not r.hit_at(k)]


def evaluate_retrieval(
    retriever: HybridRetriever,
    facts: list[dict],
    ks: tuple[int, ...] = DEFAULT_KS,
    deal_ids: set[str] | None = None,
) -> RetrievalReport:
    """Run every gold question through the retriever and score the results.

    Retrieval depth is the largest k requested, so one pass produces every
    recall figure. The unanswerable question is excluded and counted: it has no
    gold page by construction, and scoring it as a miss would penalise the
    system for correctly having nothing to find. It gets its own treatment at
    M6, where refusal is the behaviour being measured.
    """
    report = RetrievalReport(ks=ks)
    depth = max(ks)
    available = set(retriever.indexes)

    for fact in facts:
        if deal_ids is not None and fact["deal_id"] not in deal_ids:
            report.skipped_out_of_scope += 1
            continue
        if fact["deal_id"] not in available:
            report.skipped_out_of_scope += 1
            continue
        if not fact.get("answerable", True) or not fact.get("source_document"):
            report.skipped_unanswerable += 1
            continue

        # Retrieve extra chunks, because several may come from the same page and
        # ranks are counted over DISTINCT pages.
        #
        # That definition follows from what a hit is. A hit is a page, so
        # recall@5 has to mean "the answer's page is among the first five pages
        # retrieved", not "among the first five chunks". Counting chunks
        # measures something no one wants to know: a page that contributes both
        # a table chunk and a prose chunk would burn two slots and make a
        # perfectly good retriever look like it ranked the answer lower.
        #
        # It is also the quantity that matters downstream. At M7 the question is
        # how many distinct pages the agent must read before it can cite the
        # answer, because pages are what consume its context budget.
        hits = retriever.retrieve(fact["question"], fact["deal_id"], k=depth * 3)
        retrieved: list[tuple[str, int]] = []
        for hit in hits:
            page = (hit.chunk["document"], hit.chunk["page_number"])
            if page not in retrieved:
                retrieved.append(page)
        retrieved = retrieved[:depth]

        rank = None
        target = (fact["source_document"], fact["source_page"])
        for position, page in enumerate(retrieved, start=1):
            if page == target:
                rank = position
                break

        report.results.append(
            QueryResult(
                fact_id=fact["fact_id"],
                deal_id=fact["deal_id"],
                question=fact["question"],
                fact_type=fact["fact_type"],
                defect_tag=fact.get("defect_tag"),
                gold_document=fact["source_document"],
                gold_page=fact["source_page"],
                retrieved=retrieved,
                rank=rank,
            )
        )

    return report


def format_report(report: RetrievalReport) -> str:
    """Render a report for a terminal. Used by the eval script and by CI logs."""
    lines: list[str] = []
    lines.append(f"queries scored        {report.n}")
    lines.append(f"skipped, unanswerable {report.skipped_unanswerable}")
    lines.append(f"skipped, out of scope {report.skipped_out_of_scope}")
    lines.append("")
    for k in report.ks:
        lines.append(f"  recall@{k:<3}           {report.recall_at(k):.1%}")
    lines.append(f"  MRR                  {report.mrr():.3f}")

    by_defect = report.by_defect()
    if by_defect:
        lines.append("")
        lines.append("by planted defect")
        for defect, rows in sorted(by_defect.items()):
            lines.append(
                f"  {defect:<24} n={len(rows):<3} "
                f"recall@1={report.recall_at(1, rows):.0%} "
                f"recall@5={report.recall_at(5, rows):.0%} "
                f"mrr={report.mrr(rows):.2f}"
            )

    lines.append("")
    lines.append("by fact type")
    for fact_type, rows in sorted(report.by_fact_type().items()):
        lines.append(
            f"  {fact_type:<24} n={len(rows):<3} "
            f"recall@1={report.recall_at(1, rows):.0%} "
            f"recall@5={report.recall_at(5, rows):.0%} "
            f"mrr={report.mrr(rows):.2f}"
        )

    misses = report.misses(max(report.ks))
    if misses:
        lines.append("")
        lines.append(f"never retrieved (k={max(report.ks)})")
        for miss in misses[:15]:
            lines.append(
                f"  {miss.fact_id}  {miss.gold_document}#{miss.gold_page}"
                f"  {miss.question[:60]}"
            )
    return "\n".join(lines)
