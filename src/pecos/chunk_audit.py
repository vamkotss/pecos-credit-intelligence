"""Audit chunking against ground truth (M4).

WHY THIS IS A SEPARATE MODULE
-----------------------------
`chunking.py` never reads the manifest. A chunker that consulted the answer key
would measure nothing, so the design rule is strict: chunking sees page records
and nothing else.

But chunking still has to be *scored*, because it sets a hard ceiling on
everything downstream. If a figure does not survive into any chunk, no retriever
can find it, no reranker can rescue it, and no amount of prompt engineering will
make the agent cite it. Retrieval work spent chasing a fact that chunking
already destroyed is wasted, and the failure looks identical to a retrieval bug.

This module is that measurement, kept deliberately outside the thing it
measures. It is the same shape as the leakage audit in the Lonestar fraud
project: a separate, honest check on whether the pipeline quietly threw
something away.

WHAT COUNTS AS A MISS
---------------------
Not every gold fact should be found in a chunk, and treating them all as
extractive would report a false failure rate. Three categories are excluded, and
each exclusion is recorded rather than silently dropped:

**Derived metrics.** Leverage and debt service coverage are never printed on any
page. They are computed from figures that are, which is M7's job. Expecting them
in a chunk would be expecting the corpus to contain something it does not.

**Behavioural facts.** The prompt-injection and unanswerable cases are scored on
what the model refuses to do, not on a string appearing anywhere.

**Rescaled pages.** The tax return prints 32,041 where the true figure is
32,041,248. The preparer rounded to thousands, so the exact figure is not on the
page and never could be. The audit compares against the printed form and records
the page's scale factor, so the magnitude is still checked without pretending
lost precision is recoverable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Fact types that are computed rather than printed. See the module docstring.
DERIVED_TYPES = frozenset({"derived_metric"})
BEHAVIOURAL_TYPES = frozenset({"behavioural"})


@dataclass
class ContainmentMiss:
    fact_id: str
    deal_id: str
    fact_type: str
    needle: str
    document: str
    page_number: int
    chunks_on_page: int
    defect_tag: str | None


@dataclass
class ContainmentResult:
    """How much of the ground truth survived chunking."""

    extractive: int = 0
    found: int = 0
    excluded_derived: int = 0
    excluded_behavioural: int = 0
    excluded_out_of_scope: int = 0
    misses: list[ContainmentMiss] = field(default_factory=list)
    by_fact_type: Counter = field(default_factory=Counter)
    by_defect: Counter = field(default_factory=Counter)

    @property
    def rate(self) -> float:
        """The retrieval ceiling. Nothing downstream can score above this."""
        if self.extractive == 0:
            return 0.0
        return self.found / self.extractive


def _needle_for(fact: dict, scale_factor: int) -> str | None:
    """The string that should appear on the page for this fact.

    Returns None when the fact has no printable form.
    """
    value = fact.get("answer_value")
    if value is None:
        return None

    if isinstance(value, int):
        if scale_factor != 1:
            # The page prints the rescaled figure, so that is what to look for.
            # Rounding here mirrors what the document preparer did.
            return f"{round(value / scale_factor):,}"
        return f"{value:,}"

    if isinstance(value, float):
        # Percentages print as given, e.g. "34.2".
        return f"{value:g}"

    return str(value)


def audit_containment(
    chunks: list[dict], facts: list[dict], deal_ids: set[str] | None = None
) -> ContainmentResult:
    """Check that every extractive gold fact survives into a chunk.

    A fact counts as found when its printed form appears in the text of at least
    one chunk anchored to the page the manifest cites. Anchoring matters as much
    as containment: a figure that survives into a chunk attributed to the wrong
    page would still be uncitable, and would score as a retrieval failure later
    for reasons no retrieval change could fix.
    """
    result = ContainmentResult()

    scoped = [c for c in chunks if deal_ids is None or c["deal_id"] in deal_ids]
    index: dict[tuple[str, str, int], list[dict]] = {}
    scales: dict[tuple[str, str, int], int] = {}
    for chunk in scoped:
        key = (chunk["deal_id"], chunk["document"], chunk["page_number"])
        index.setdefault(key, []).append(chunk)
        scales[key] = chunk.get("scale_factor", 1)

    available = {c["deal_id"] for c in scoped}

    for fact in facts:
        if fact["deal_id"] not in available:
            result.excluded_out_of_scope += 1
            continue
        if not fact.get("answerable", True) or fact["fact_type"] in BEHAVIOURAL_TYPES:
            result.excluded_behavioural += 1
            continue
        if fact["fact_type"] in DERIVED_TYPES:
            result.excluded_derived += 1
            continue

        key = (fact["deal_id"], fact["source_document"], fact["source_page"])
        page_chunks = index.get(key, [])
        needle = _needle_for(fact, scales.get(key, 1))
        if needle is None:
            result.excluded_behavioural += 1
            continue

        result.extractive += 1
        result.by_fact_type[fact["fact_type"]] += 1
        if fact.get("defect_tag"):
            result.by_defect[fact["defect_tag"]] += 1

        if any(needle in chunk["text"] for chunk in page_chunks):
            result.found += 1
        else:
            result.misses.append(
                ContainmentMiss(
                    fact_id=fact["fact_id"],
                    deal_id=fact["deal_id"],
                    fact_type=fact["fact_type"],
                    needle=needle,
                    document=fact["source_document"],
                    page_number=fact["source_page"],
                    chunks_on_page=len(page_chunks),
                    defect_tag=fact.get("defect_tag"),
                )
            )

    return result


def audit_near_duplicates(chunks: list[dict]) -> dict[str, dict]:
    """Summarise how separable near-duplicate documents are.

    Reports, per deal that has a non-final document, how many chunks carry each
    status and what authority ranks are available. The point is not the counts
    themselves but that the distinction exists in the metadata at all: for the
    draft statements the text is roughly 94% identical to the final, so if
    status had been dropped at the chunk boundary there would be nothing left to
    separate them by.
    """
    out: dict[str, dict] = {}
    for chunk in chunks:
        if chunk["doc_status"] == "final":
            continue
        entry = out.setdefault(
            chunk["deal_id"],
            {"statuses": Counter(), "documents": set(), "authorities": set()},
        )
        entry["statuses"][chunk["doc_status"]] += 1
        entry["documents"].add(chunk["document"])
        entry["authorities"].add(chunk["authority"])

    for entry in out.values():
        entry["documents"] = sorted(entry["documents"])
        entry["authorities"] = sorted(entry["authorities"])
        entry["statuses"] = dict(entry["statuses"])
    return out
