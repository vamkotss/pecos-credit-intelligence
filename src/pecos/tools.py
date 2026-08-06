"""Credit calculations with recorded inputs (M7).

WHY A TOOL RATHER THAN LETTING THE MODEL DO ARITHMETIC
------------------------------------------------------
M6 measured Claude answering the gold questions and flagged seventeen
"hallucinated" figures. Almost none were hallucinations. They were leverage
ratios, debt totals and a units conversion -- numbers the model computed
correctly from figures that were on cited pages.

That exposed a real gap, and the gap is not in the metric alone. A figure a
model produced by doing arithmetic in its head has no provenance. It might be
right; nothing about the output says so, and nothing can check it. In a credit
memo that is the same problem as an invented figure wearing better clothes.

So arithmetic moves out of the model and into this module. Every calculation
returns a `Computation` carrying its **inputs, each tagged with the page it came
from**, the formula, and the result. Grounding then checks the inputs -- which
are quoted figures on cited pages -- rather than guessing at the output. A
derived figure becomes as traceable as a quoted one.

That is also why leverage is not simply looked up. It is printed on no page in
the corpus, by design: a lender's most-cited number is one nobody writes down.

THE UNITS TRAP IS HANDLED HERE TOO
----------------------------------
`convert_units` exists because M6 flagged `32,041,000` as invented when the
model correctly multiplied a figure printed in thousands. Scaling is arithmetic,
so it belongs with the arithmetic, and it produces the same audit trail: printed
value, multiplier, evidence for the multiplier, result.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Input:
    """One figure fed into a calculation, and where it came from.

    `document` and `page` are the same anchor the manifest and every chunk use,
    so an input can be verified by opening one page.
    """

    label: str
    value: float
    document: str | None = None
    page: int | None = None

    @property
    def citation(self) -> str | None:
        if self.document is None or self.page is None:
            return None
        return f"[{self.document}#p{self.page}]"


@dataclass(frozen=True)
class Computation:
    """A calculation, its inputs, and its result.

    The formula is a human-readable string rather than an expression object.
    Its job is to be read by a credit analyst reviewing the memo, not to be
    re-evaluated -- the result is already computed and re-evaluating it would
    just be a second chance to disagree.
    """

    name: str
    formula: str
    inputs: tuple[Input, ...]
    result: float
    unit: str
    note: str = ""

    @property
    def citations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for item in self.inputs:
            marker = item.citation
            if marker and marker not in seen:
                seen.append(marker)
        return tuple(seen)

    def formatted(self) -> str:
        if self.unit == "USD":
            return f"${self.result:,.0f}"
        if self.unit == "x":
            return f"{self.result:.2f}x"
        if self.unit == "percent":
            return f"{self.result:.1f}%"
        return f"{self.result:,.2f}"

    def audit_line(self) -> str:
        """One line an analyst can check without opening the code."""
        parts = ", ".join(f"{i.label}={i.value:,.0f}" for i in self.inputs)
        cites = " ".join(self.citations)
        return f"{self.name} = {self.formatted()}  [{self.formula}: {parts}] {cites}"


class CalculationError(ValueError):
    """Raised when a calculation cannot be performed honestly.

    Deliberately an exception rather than a sentinel value. A division by zero
    EBITDA is not "infinite leverage" -- it is a business with no operating
    profit, which is a finding the memo must state in words rather than a number
    it can quietly print.
    """


def _require(inputs: list[Input]) -> None:
    for item in inputs:
        if item.value is None:
            raise CalculationError(f"missing input: {item.label}")


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------


def total(
    components: list[Input], name: str, formula: str | None = None
) -> Computation:
    """Sum a set of figures, recording each one.

    Any intermediate a memo will print has to be a recorded computation in its
    own right, not an anonymous sum passed into the next step. The top-five
    customer total was originally computed inline and handed to `share_of`; it
    then appeared in the audit trail as an input value that existed on no page
    and in no calculation, and the verify gate correctly flagged it. An
    intermediate that cannot be traced is an unverifiable figure regardless of
    how briefly it lives.
    """
    _require(components)
    if not components:
        raise CalculationError(f"no components supplied for {name}")
    return Computation(
        name=name,
        formula=formula or " + ".join(item.label for item in components),
        inputs=tuple(components),
        result=sum(item.value for item in components),
        unit="USD",
    )


def total_debt(components: list[Input]) -> Computation:
    """Sum the interest-bearing debt lines.

    A separate calculation rather than a step inside leverage, because total
    debt is itself a figure the memo states and a reviewer checks. Burying it
    would mean the memo asserts a number whose derivation is invisible.
    """
    if not components:
        raise CalculationError("no debt components supplied")
    return total(components, "Total interest-bearing debt")


def leverage(debt: Input, ebitda: Input) -> Computation:
    """Total debt divided by EBITDA.

    How many years of cash profit would be needed to repay everything. Pecos
    treats above roughly 3.5x as needing a structural mitigant.
    """
    _require([debt, ebitda])
    if ebitda.value <= 0:
        raise CalculationError(
            "EBITDA is zero or negative, so leverage is undefined. State this "
            "in words rather than reporting a ratio."
        )
    return Computation(
        name="Total debt / EBITDA",
        formula="total debt / EBITDA",
        inputs=(debt, ebitda),
        result=debt.value / ebitda.value,
        unit="x",
    )


def dscr(
    ebitda: Input, capex: Input, taxes: Input, interest: Input, principal: Input
) -> Computation:
    """Debt service coverage ratio.

    Cash available after capital spending and tax, over what the bank is owed
    this year. Capex is subtracted because a business that stops replacing its
    equipment to make loan payments is not covering its debt service, it is
    liquidating slowly. Pecos requires 1.25x.
    """
    _require([ebitda, capex, taxes, interest, principal])
    service = interest.value + principal.value
    if service <= 0:
        raise CalculationError("no debt service in the period, so DSCR is undefined")
    available = ebitda.value - capex.value - taxes.value
    return Computation(
        name="DSCR",
        formula="(EBITDA - capex - taxes) / (interest + principal)",
        inputs=(ebitda, capex, taxes, interest, principal),
        result=available / service,
        unit="x",
    )


def current_ratio(current_assets: Input, current_liabilities: Input) -> Computation:
    _require([current_assets, current_liabilities])
    if current_liabilities.value <= 0:
        raise CalculationError(
            "current liabilities are zero, so the ratio is undefined"
        )
    return Computation(
        name="Current ratio",
        formula="current assets / current liabilities",
        inputs=(current_assets, current_liabilities),
        result=current_assets.value / current_liabilities.value,
        unit="x",
    )


def concentration(largest: Input, total: Input) -> Computation:
    _require([largest, total])
    if total.value <= 0:
        raise CalculationError("total receivables are zero")
    return Computation(
        name="Largest customer concentration",
        formula="largest customer balance / total receivables",
        inputs=(largest, total),
        result=largest.value / total.value * 100.0,
        unit="percent",
    )


def convert_units(printed: Input, scale_factor: int, evidence: str = "") -> Computation:
    """Restate a figure printed in thousands or millions as whole dollars.

    This exists because M6 flagged a correct conversion as an invented figure.
    Scaling is arithmetic and belongs with the arithmetic, and routing it here
    produces the same audit trail as any other calculation: the printed value,
    the multiplier, the evidence for the multiplier, and the result.

    Getting this wrong is the quietest serious error in the corpus. No
    confidence score flags it, the figure does not look wrong, and it changes a
    lending decision by three orders of magnitude.
    """
    _require([printed])
    if scale_factor < 1:
        raise CalculationError(f"invalid scale factor: {scale_factor}")
    return Computation(
        name=f"{printed.label} restated in whole dollars",
        formula=f"printed value x {scale_factor:,}",
        inputs=(printed,),
        result=printed.value * scale_factor,
        unit="USD",
        note=evidence or f"page states figures in units of {scale_factor:,}",
    )


def growth(current: Input, prior: Input) -> Computation:
    _require([current, prior])
    if prior.value == 0:
        raise CalculationError("prior period is zero, so growth is undefined")
    return Computation(
        name=f"{current.label} growth",
        formula="(current - prior) / prior",
        inputs=(current, prior),
        result=(current.value - prior.value) / abs(prior.value) * 100.0,
        unit="percent",
    )


def share_of(part: Input, whole: Input, name: str = "") -> Computation:
    """One figure as a percentage of another.

    The workhorse the first version lacked, and its absence is what broke the
    first full agent run. Twelve of twelve memos failed verification, and almost
    every offending figure was a percentage: margins, an over-90 receivables
    share, a cash decline, a top-five customer concentration. The prompt told
    the model not to compute anything the calculator had not provided; it
    computed them anyway.

    That is the useful lesson. A prompt-level prohibition does not reliably stop
    a model doing something the task obviously requires -- and the model was
    right that a credit memo needs those percentages. The fix is to supply them,
    not to forbid them harder.
    """
    _require([part, whole])
    if whole.value == 0:
        raise CalculationError(f"{whole.label} is zero, so a share of it is undefined")
    return Computation(
        name=name or f"{part.label} as % of {whole.label}",
        formula=f"{part.label} / {whole.label}",
        inputs=(part, whole),
        result=abs(part.value) / abs(whole.value) * 100.0,
        unit="percent",
    )


def margin(numerator: Input, revenue: Input, name: str) -> Computation:
    """A margin: some profit line over revenue."""
    return share_of(numerator, revenue, name=name)


def interest_coverage(ebitda: Input, interest: Input) -> Computation:
    """EBITDA over interest expense.

    Read alongside DSCR rather than instead of it. Interest coverage asks
    whether the borrower can pay the rent on its debt; DSCR asks whether it can
    also repay principal and keep replacing its equipment. A business can pass
    the first and fail the second, and that gap is usually the finding.
    """
    _require([ebitda, interest])
    if abs(interest.value) == 0:
        raise CalculationError("no interest expense, so coverage is undefined")
    return Computation(
        name="Interest coverage",
        formula="EBITDA / interest expense",
        inputs=(ebitda, interest),
        result=ebitda.value / abs(interest.value),
        unit="x",
    )


def change(current: Input, prior: Input, name: str = "") -> Computation:
    """Percentage change between two periods, signed.

    Separate from `growth` only in that it keeps the sign, so a decline reads
    as negative rather than as a positive "reduction" a reader has to interpret.
    """
    _require([current, prior])
    if prior.value == 0:
        raise CalculationError("prior period is zero, so the change is undefined")
    return Computation(
        name=name or f"{current.label} change",
        formula="(current - prior) / prior",
        inputs=(current, prior),
        result=(current.value - prior.value) / abs(prior.value) * 100.0,
        unit="percent",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALCULATIONS = {
    "total": total,
    "total_debt": total_debt,
    "leverage": leverage,
    "dscr": dscr,
    "current_ratio": current_ratio,
    "concentration": concentration,
    "convert_units": convert_units,
    "growth": growth,
    "share_of": share_of,
    "margin": margin,
    "interest_coverage": interest_coverage,
    "change": change,
}


@dataclass
class CalculationLog:
    """Every calculation performed while writing one memo.

    Kept as an ordered log rather than a dict, because the memo is reviewed as a
    narrative and an analyst checking a figure wants the derivations in the
    order the reasoning went, not alphabetically.
    """

    entries: list[Computation] = field(default_factory=list)

    def record(self, computation: Computation) -> Computation:
        self.entries.append(computation)
        return computation

    def derived_values(self) -> set[str]:
        """Formatted results of every calculation, for the grounding check.

        This is the set that lets `evaluation.py` distinguish a computed figure
        from an invented one. A number in this set was produced by a recorded
        calculation from cited inputs; a number outside it, and not on a cited
        page, came from nowhere.
        """
        values: set[str] = set()
        for entry in self.entries:
            values.add(entry.formatted())
            values.add(entry.formatted().lstrip("$"))
            if entry.unit == "USD":
                values.add(f"{entry.result:,.0f}")
                values.add(f"{int(entry.result)}")
            elif entry.unit == "percent":
                # A memo may print a percentage rounded either way, and a
                # decline may be written as a positive "reduction". All the
                # forms a drafter might reasonably choose count as the same
                # computed figure.
                magnitude = abs(entry.result)
                for rendered in (
                    f"{entry.result:.1f}%",
                    f"{entry.result:.0f}%",
                    f"{magnitude:.1f}%",
                    f"{magnitude:.0f}%",
                ):
                    values.add(rendered)
                    values.add(rendered.rstrip("%"))
            elif entry.unit == "x":
                values.add(f"{entry.result:.1f}x")
                values.add(f"{entry.result:.2f}")
                values.add(f"{entry.result:.1f}")
        return values

    def citations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for entry in self.entries:
            for marker in entry.citations:
                if marker not in seen:
                    seen.append(marker)
        return tuple(seen)

    def audit_trail(self) -> str:
        return "\n".join(entry.audit_line() for entry in self.entries)
