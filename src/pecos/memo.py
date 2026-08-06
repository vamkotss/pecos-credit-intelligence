"""The credit memo agent (M7).

WHY A GRAPH AND NOT A PROMPT
----------------------------
The obvious way to write a credit memo with an LLM is one large prompt: here are
twenty pages, write the memo. It works often enough to demo and fails in ways
that cannot be diagnosed, because there is no point in the process where you can
ask "was the leverage input right" separately from "was the conclusion right".

Splitting it into a graph makes each failure locatable:

    plan -> gather -> compute -> draft -> verify -> (revise) -> done

**plan** decides which questions the memo needs answered. Fixed, not
model-chosen: a credit memo has a required shape, and a lender does not want the
sections to vary with sampling.

**gather** retrieves for each question and pulls labelled figures out of the
statements, each tagged with the page it came from.

**compute** runs the calculator. No arithmetic happens anywhere else.

**draft** writes prose from figures that already exist. The model's job is
narrative and judgement, not extraction and not arithmetic -- the two things it
is worst at and the two things that matter most here.

**verify** checks every figure in the draft against cited pages and the
calculation log. This is a real gate, not a log line: an ungrounded figure sends
the draft back.

**revise** gets one attempt, with the specific ungrounded figures named. One,
not unlimited -- a model that cannot fix a grounding failure when told exactly
which number is wrong will not fix it on the fourth pass either, and an
unbounded loop turns a bad answer into a bad answer that costs ten times more.

WHAT THE VERIFY GATE IS FOR
---------------------------
M6 found that Claude produces figures it computed rather than quoted -- leverage
ratios, debt totals, unit conversions. Those were correct and unverifiable, which
in a credit memo is its own kind of failure.

Here they are verifiable, because `compute` produced them from cited inputs and
recorded the derivation. The gate can therefore distinguish three things that
look identical in raw text: a quoted figure, a computed figure, and an invented
one. Only the third fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from pecos.answering import contexts_from_hits
from pecos.policy import POLICY_CONSTANTS
from pecos.tools import (
    CalculationError,
    CalculationLog,
    Input,
    change,
    concentration,
    current_ratio,
    dscr,
    interest_coverage,
    leverage,
    margin,
    share_of,
    total,
    total_debt,
)

# The questions a Pecos credit memo has to answer. Fixed rather than
# model-chosen: a memo has a required shape, and a credit committee does not
# want the sections to vary between runs.
MEMO_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("request", "How much is the borrower requesting and for what purpose?"),
    ("business", "Describe the business, its industry and its ownership."),
    ("performance", "What were revenue, EBITDA and net income in the latest year?"),
    ("balance_sheet", "What were cash, total assets and total debt at year end?"),
    ("debt", "What existing debt facilities does the borrower have?"),
    ("receivables", "What is the receivables ageing and customer concentration?"),
    ("risks", "What legal, related-party or other risks were disclosed?"),
)

# Statement line labels the extractor looks for, mapped to the key it stores.
# Matched against the first cell of a table row, which is why chunking had to
# keep tables as tables -- flattened into prose, none of this is recoverable.
FIGURE_LABELS: tuple[tuple[str, str], ...] = (
    ("revenue", "revenue"),
    ("cost of goods sold", "cogs"),
    ("gross profit", "gross_profit"),
    ("ebitda", "ebitda"),
    ("net income", "net_income"),
    ("interest expense", "interest"),
    ("income tax provision", "taxes"),
    ("depreciation and amortisation", "depreciation"),
    ("cash and cash equivalents", "cash"),
    ("accounts receivable, net", "accounts_receivable"),
    ("total current assets", "current_assets"),
    ("total current liabilities", "current_liabilities"),
    ("current portion of long-term debt", "current_portion_ltd"),
    ("long-term debt, net of current portion", "ltd_noncurrent"),
    ("total assets", "total_assets"),
    ("total equity", "total_equity"),
    ("capital expenditures", "capex"),
    ("repayment of long-term debt", "principal"),
    ("total accounts receivable", "total_receivables"),
    # From the loan application, not the statements. Pro forma leverage cannot
    # be reconstructed without it, and pro forma leverage is what flips a
    # recommendation.
    ("facility requested", "facility_requested"),
)

_MONEY = re.compile(r"\(?\$?-?\d[\d,]*\)?")


def parse_money(cell: str) -> float | None:
    """Read a statement cell as a number.

    Parentheses mean negative. That is the accounting convention and it is a
    real parsing hazard: reading `(412,300)` as positive silently inverts a loss
    into a profit, and nothing downstream would notice.
    """
    text = cell.strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    value = float(digits)
    return -value if negative else value


@dataclass
class ExtractedFigure:
    key: str
    label: str
    value: float
    document: str
    page: int
    # Every year printed on the row, oldest first. Kept because growth and
    # change are among the metrics a memo needs, and computing them requires the
    # prior period -- which the first version discarded, so the drafter computed
    # them itself and the figures had no provenance.
    series: tuple[float, ...] = ()

    @property
    def prior(self) -> float | None:
        return self.series[-2] if len(self.series) >= 2 else None

    def as_input(self) -> Input:
        return Input(
            label=self.label, value=self.value, document=self.document, page=self.page
        )


def extract_figures(contexts: list[dict]) -> dict[str, ExtractedFigure]:
    """Pull labelled statement figures out of retrieved pages.

    Takes the **last** numeric cell in a row, because the statements are
    comparative and print oldest year first -- so the rightmost column is the
    most recent period. A memo that quoted the leftmost column would be
    describing a business as it was three years ago, and every figure would be
    individually correct.

    Only chunks from authoritative documents are read. A draft or superseded
    page can still be retrieved and cited when someone asks about it, but it
    must never silently supply the figures a memo is built on.
    """
    figures: dict[str, ExtractedFigure] = {}
    for context in contexts:
        if context.get("doc_status", "final") != "final":
            continue
        scale = context.get("scale_factor", 1)
        for line in context["text"].splitlines():
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) < 2:
                continue
            head = cells[0].lower().strip()
            for label, key in FIGURE_LABELS:
                if head != label:
                    continue
                series = tuple(
                    v * scale
                    for v in (parse_money(cell) for cell in cells[1:])
                    if v is not None
                )
                if series and key not in figures:
                    figures[key] = ExtractedFigure(
                        key=key,
                        label=cells[0].strip(),
                        value=series[-1],
                        document=context["document"],
                        page=context["page_number"],
                        series=series,
                    )
    return figures


@dataclass
class Receivables:
    """The ageing and customer table, which lives only in a table.

    Parsed separately from the statements because its rows are named after
    customers rather than accounting lines, so no label list can find them. The
    concentration figure the `table_only_fact` defect hides appears in no
    sentence anywhere in the corpus -- it is reachable only because chunking
    kept the table as a table.
    """

    document: str
    page: int
    total: float | None = None
    over_90: float | None = None
    customers: tuple[tuple[str, float], ...] = ()

    def input_for(self, label: str, value: float) -> Input:
        return Input(label=label, value=value, document=self.document, page=self.page)


def extract_receivables(contexts: list[dict]) -> Receivables | None:
    """Read the ageing buckets and customer balances.

    Accumulates across every receivables chunk rather than returning on the
    first match. The page holds two separate tables -- an ageing summary and a
    customer detail table -- and chunking correctly emits them as two chunks, so
    a parser that stopped at the first one got the totals without the customers
    or the customers without the totals, and computed neither concentration
    figure.
    """
    found: Receivables | None = None
    customers: list[tuple[str, float]] = []

    for context in contexts:
        document = context["document"].lower()
        if "aging" not in document and "ageing" not in document:
            continue
        if context.get("doc_status", "final") != "final":
            continue
        if found is None:
            found = Receivables(
                document=context["document"], page=context["page_number"]
            )
        for line in context["text"].splitlines():
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) < 2:
                continue
            head = cells[0].lower()
            value = parse_money(cells[1])
            if value is None:
                continue
            if head.startswith("total accounts receivable"):
                found.total = value
            elif head.startswith("over 90"):
                found.over_90 = value
            elif head in ("current", "bucket", "customer") or head.startswith(
                ("31 -", "61 -", "31-", "61-")
            ):
                continue
            elif len(cells) >= 3 and cells[2].endswith("%"):
                # A customer row is name, balance, share. The trailing percent is
                # what separates it from a stray two-column line.
                customers.append((cells[0], value))

    if found is None:
        return None
    found.customers = tuple(customers)
    return found if (found.total or customers) else None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class MemoState(TypedDict, total=False):
    deal_id: str
    questions: list[tuple[str, str]]
    contexts: dict[str, list[dict]]
    all_contexts: list[dict]
    figures: dict[str, ExtractedFigure]
    receivables: Any
    log: CalculationLog
    draft: str
    issues: list[str]
    reconstructions: dict[str, str]
    revisions: int
    done: bool


@dataclass
class MemoResult:
    deal_id: str
    text: str
    computations: CalculationLog
    citations: tuple[str, ...]
    ungrounded: tuple[str, ...]
    reconstructions: dict[str, str]
    revisions: int
    figures_extracted: int

    @property
    def verified(self) -> bool:
        return not self.ungrounded

    def audit_trail(self) -> str:
        return self.computations.audit_trail()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_gather_node(retriever, k: int = 6):
    def gather(state: MemoState) -> dict[str, Any]:
        contexts: dict[str, list[dict]] = {}
        pooled: list[dict] = []
        seen: set[str] = set()

        # A supplementary, deliberately keyword-shaped query alongside the
        # natural-language sections. The receivables page holds two tables and
        # the analyst-phrased question reliably surfaces only one of them --
        # this reaches the other. Retrieval is cheap; a missing concentration
        # figure is not.
        questions = list(state["questions"]) + [
            (
                "receivables_detail",
                "customer balance % of AR ageing bucket over 90 days "
                "total accounts receivable",
            )
        ]
        for section, question in questions:
            hits = retriever.retrieve(question, state["deal_id"], k=k)
            section_contexts = contexts_from_hits(hits)
            contexts[section] = section_contexts
            for context in section_contexts:
                if context["chunk_id"] not in seen:
                    seen.add(context["chunk_id"])
                    pooled.append(context)
        return {
            "contexts": contexts,
            "all_contexts": pooled,
            "figures": extract_figures(pooled),
            "receivables": extract_receivables(pooled),
        }

    return gather


def compute_node(state: MemoState) -> dict[str, Any]:
    """Run every calculation the extracted figures support.

    Calculations that cannot be performed honestly are skipped rather than
    filled with a placeholder. A memo missing a DSCR line is obviously
    incomplete; a memo showing DSCR as 0.00 because a divisor was missing is
    obviously wrong in a way nobody notices.
    """
    figures = state.get("figures", {})
    log = CalculationLog()

    def get(key: str) -> Input | None:
        figure = figures.get(key)
        return figure.as_input() if figure else None

    current_portion = get("current_portion_ltd")
    noncurrent = get("ltd_noncurrent")
    debt: Input | None = None
    if current_portion and noncurrent:
        try:
            computed = log.record(total_debt([current_portion, noncurrent]))
            debt = Input(
                label="total debt",
                value=computed.result,
                document=current_portion.document,
                page=current_portion.page,
            )
        except CalculationError:
            debt = None

    ebitda = get("ebitda")
    if debt and ebitda:
        try:
            log.record(leverage(debt, ebitda))
        except CalculationError:
            pass

    capex, taxes = get("capex"), get("taxes")
    interest, principal = get("interest"), get("principal")
    if ebitda and capex and taxes and interest and principal:
        try:
            log.record(
                dscr(
                    ebitda,
                    Input("capex", abs(capex.value), capex.document, capex.page),
                    Input("taxes", abs(taxes.value), taxes.document, taxes.page),
                    Input(
                        "interest",
                        abs(interest.value),
                        interest.document,
                        interest.page,
                    ),
                    Input(
                        "principal",
                        abs(principal.value),
                        principal.document,
                        principal.page,
                    ),
                )
            )
        except CalculationError:
            pass

    assets, liabilities = get("current_assets"), get("current_liabilities")
    if assets and liabilities:
        try:
            log.record(current_ratio(assets, liabilities))
        except CalculationError:
            pass

    # --- Margins, growth and coverage -------------------------------------
    # Everything below was added after the first full agent run failed
    # verification on twelve memos out of twelve. Nearly every offending figure
    # was a percentage the calculator did not offer, so the drafter computed it
    # -- despite an explicit instruction not to. Supplying the metric is the fix;
    # forbidding it harder was not going to work, and the model was right that a
    # credit memo needs them.
    revenue = get("revenue")
    if revenue:
        for key, name in (
            ("gross_profit", "Gross margin"),
            ("ebitda", "EBITDA margin"),
            ("net_income", "Net margin"),
        ):
            line = get(key)
            if line:
                try:
                    log.record(margin(line, revenue, name))
                except CalculationError:
                    pass

    for key, name in (("revenue", "Revenue growth"), ("ebitda", "EBITDA growth")):
        figure = figures.get(key)
        if figure and figure.prior is not None:
            try:
                log.record(
                    change(
                        figure.as_input(),
                        Input(
                            f"{figure.label} prior year",
                            figure.prior,
                            figure.document,
                            figure.page,
                        ),
                        name=name,
                    )
                )
            except CalculationError:
                pass

    cash = figures.get("cash")
    if cash and cash.prior is not None and len(cash.series) >= 2:
        try:
            log.record(
                change(
                    cash.as_input(),
                    Input(
                        "cash, earliest year",
                        cash.series[0],
                        cash.document,
                        cash.page,
                    ),
                    name="Cash change over the period",
                )
            )
        except CalculationError:
            pass

    interest = get("interest")
    if ebitda and interest:
        try:
            log.record(interest_coverage(ebitda, interest))
        except CalculationError:
            pass

    # --- Receivables -------------------------------------------------------
    book = state.get("receivables")
    if book and book.total:
        total_ar = book.input_for("total receivables", book.total)
        if book.over_90 is not None:
            try:
                log.record(
                    share_of(
                        book.input_for("over 90 days", book.over_90),
                        total_ar,
                        name="Receivables over 90 days",
                    )
                )
            except CalculationError:
                pass
        if book.customers:
            ranked = sorted(book.customers, key=lambda c: -c[1])
            try:
                log.record(
                    concentration(book.input_for(ranked[0][0], ranked[0][1]), total_ar)
                )
            except CalculationError:
                pass
            try:
                # Recorded as its own computation before being used, so the
                # intermediate is traceable rather than appearing in the audit
                # trail as a number from nowhere.
                top_five = log.record(
                    total(
                        [book.input_for(name, value) for name, value in ranked[:5]],
                        "Top five customer balances",
                    )
                )
                log.record(
                    share_of(
                        book.input_for("top five customers", top_five.result),
                        total_ar,
                        name="Top five customer concentration",
                    )
                )
            except CalculationError:
                pass

    return {"log": log}


def make_draft_node(drafter):
    def draft(state: MemoState) -> dict[str, Any]:
        return {
            "draft": drafter.draft(
                deal_id=state["deal_id"],
                contexts=state.get("all_contexts", []),
                figures=state.get("figures", {}),
                log=state.get("log", CalculationLog()),
                issues=state.get("issues", []),
            )
        }

    return draft


def verify_node(state: MemoState) -> dict[str, Any]:
    """The gate. Every figure in the draft must be quoted, computed, or shown to
    follow arithmetically from figures that are.

    Four categories, not three:

    - **quoted** -- printed on a page the memo cited
    - **computed** -- a result in the calculation log, which recorded its inputs
    - **reconstructible** -- reachable in one or two operations from grounded
      figures, with the derivation reported
    - **invented** -- none of the above

    The third category was added after two rounds of enumerating metrics failed
    to reach zero. The figures that finally made the problem clear were `21.3`
    and `3.95x` in the sentence *"the requested facility of $12,800,000 would
    increase total debt to approximately $21.3 million, raising pro forma
    leverage to approximately 3.95x EBITDA"*. Every input is on a cited page,
    and that sentence is what flips the recommendation from PROCEED to DEFER.

    No list of metrics would have contained it. Enumerating what an analyst
    might compute is not a solvable problem -- the point of employing an analyst
    is that they compute things nobody listed in advance.

    Reconstructions are **reported, not assumed**. `reconstructions` carries the
    derivation for each one so a reviewer can see the claimed arithmetic and
    reject it if the match is coincidental, which with three-significant-figure
    numbers it sometimes will be. That is the honest position: the gate has
    turned "unverifiable" into "here is the arithmetic, check it", which is a
    review task rather than a mystery.
    """
    from pecos.evaluation import extract_figures as figures_in_text
    from pecos.evaluation import figure_in
    from pecos.reconstruct import explain, values_from_figures

    draft = state.get("draft", "")
    contexts = state.get("all_contexts", [])
    log = state.get("log", CalculationLog())

    page_text = "\n".join(c["text"] for c in contexts)
    derived = log.derived_values()

    # Operands for reconstruction: labelled statement lines and calculation
    # results only.
    #
    # An earlier version also scraped every number off the retrieved pages. That
    # admitted NAICS codes, founding years and employee counts as arithmetic
    # inputs, and with forty operands available something matched almost any
    # figure -- so all twelve memos "verified" on derivations like
    # `423,628: NAICS code`. A gate that approves everything is indistinguishable
    # from no gate.
    values = values_from_figures(state.get("figures", {}), log)

    issues: list[str] = []
    reconstructions: dict[str, str] = {}

    for figure in figures_in_text(draft):
        if figure in derived or figure.lstrip("$") in derived:
            continue
        if figure in POLICY_CONSTANTS:
            continue
        if figure_in(figure, page_text):
            continue
        derivation = explain(figure, values)
        if derivation is not None:
            reconstructions[figure] = str(derivation)
            continue
        issues.append(figure)

    return {
        "issues": issues,
        "reconstructions": reconstructions,
        "revisions": state.get("revisions", 0),
    }


def should_revise(state: MemoState) -> str:
    """One revision, then stop.

    A model that cannot fix a grounding failure when told exactly which figure
    is wrong will not fix it on the fourth attempt either, and an unbounded loop
    turns a bad memo into a bad memo that costs ten times more to produce.
    """
    if state.get("issues") and state.get("revisions", 0) < 1:
        return "revise"
    return "done"


def revise_node(state: MemoState) -> dict[str, Any]:
    return {"revisions": state.get("revisions", 0) + 1}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_memo_graph(retriever, drafter, k: int = 6):
    """Compile the memo graph.

    LangGraph rather than a hand-rolled loop because the retry edge is
    conditional on state that several nodes contribute to, and expressing that
    as a graph keeps the control flow visible instead of buried in a while loop
    with three flags.
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(MemoState)
    graph.add_node("gather", make_gather_node(retriever, k=k))
    graph.add_node("compute", compute_node)
    graph.add_node("draft", make_draft_node(drafter))
    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)

    graph.set_entry_point("gather")
    graph.add_edge("gather", "compute")
    graph.add_edge("compute", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges(
        "verify", should_revise, {"revise": "revise", "done": END}
    )
    graph.add_edge("revise", "draft")
    return graph.compile()


@dataclass
class MemoWriter:
    """Runs the graph and packages the result."""

    retriever: Any
    drafter: Any
    k: int = 6
    _graph: Any = field(default=None, init=False, repr=False)

    def write(self, deal_id: str) -> MemoResult:
        if self._graph is None:
            self._graph = build_memo_graph(self.retriever, self.drafter, k=self.k)

        final: MemoState = self._graph.invoke(
            {
                "deal_id": deal_id,
                "questions": list(MEMO_QUESTIONS),
                "revisions": 0,
                "issues": [],
            }
        )
        log = final.get("log", CalculationLog())
        draft = final.get("draft", "")

        from pecos.answering import parse_citations

        citations = tuple(c.marker() for c in parse_citations(draft))
        return MemoResult(
            deal_id=deal_id,
            text=draft,
            computations=log,
            citations=citations,
            ungrounded=tuple(final.get("issues", [])),
            reconstructions=dict(final.get("reconstructions", {})),
            revisions=final.get("revisions", 0),
            figures_extracted=len(final.get("figures", {})),
        )
