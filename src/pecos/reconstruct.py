"""Prove a printed figure follows from figures that are on the page (M7.3).

WHAT WENT WRONG IN THE FIRST VERSION
------------------------------------
The first reconstruction pass verified 12 memos out of 12, and the approvals
were worthless. The derivations it reported included:

    reconstructed 423,628: NAICS code
    reconstructed 60.3%: 1999 and employs 153 people. / Total debt / EBITDA
    reconstructed 62.6%: Capital expenditures / The company operates in industrial staff

A NAICS code is not a financial input. "1999 and employs 153 people" is a
sentence fragment being used as a numerator. **A gate that approves everything
is indistinguishable from no gate**, and how it got there is worth recording,
because it is a specific and repeatable mistake.

**Junk operands.** Values were harvested from any line carrying three or more
digits, so industry codes, founding years, employee counts and page furniture
all became legitimate arithmetic inputs. Labels came from the first forty
characters of the line, which is why prose fragments appear as operand names.

**A search space wide enough to hit anything.** Forty values give roughly 1,600
ordered pairs, each yielding several operations, and the two-step pass
multiplied that by forty again. At three significant figures with a half-percent
tolerance, something matches almost always. That is a birthday-problem result,
not a verification.

THE NARROWING
-------------
Three changes, every one a restriction:

1. **Operands are labelled financial figures only** -- statement lines the
   extractor recognised, results from the calculation log, and the requested
   facility. Nothing scraped from prose.
2. **Matching is exact at the printed precision.** The relative tolerance is
   gone. `21.3` must round-trip to 21.3, not to something within half a percent.
3. **The two-step pass is restricted to the pro forma shape** -- a debt-like
   figure plus the requested facility, over an earnings-like figure -- rather
   than any three values in any arrangement.

This verifies fewer memos, and that is the point. A number that fails here is a
real finding rather than a gap in a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

# Scales a memo may write a figure in. "$21.3 million" prints the token 21.3.
SCALES: tuple[float, ...] = (1.0, 1_000.0, 1_000_000.0)

# Hard cap on operands. Not a performance guard but a correctness one. Every
# additional value widens the space of coincidental matches, and the first
# version's failure was almost entirely a consequence of having too many.
MAX_VALUES = 24

# Labels that may take part in the pro forma shape, matched case-insensitively
# as substrings. Restricting the two-step pass to figures that could plausibly
# appear in that calculation is what stops it explaining arbitrary numbers.
_DEBT_LIKE = ("debt", "borrowing", "loan", "note")
_REQUEST_LIKE = ("requested", "request", "proposed", "facility")
_EARNINGS_LIKE = ("ebitda", "net income", "operating income", "cash flow")

# A figure with fewer significant digits than this cannot be reconstructed.
#
# Reconstruction is evidence, and two significant figures is not enough of it:
# `6.2` matched "total debt - net income" and `11` matched "total debt + net
# income", both plainly coincidental. With so few digits the buckets are wide
# enough that some combination lands in one almost regardless.
#
# Low-precision figures must therefore be quoted on a page or produced by a
# recorded calculation. That is not a gap -- the calculator already covers the
# common metrics, and reconstruction exists for the unanticipated ones, which
# are the figures an analyst writes precisely.
MIN_SIGNIFICANT_DIGITS = 3


def _significant_digits(text: str) -> int:
    digits = text.replace(",", "").replace(".", "").lstrip("-0")
    return len(digits.rstrip("0")) or len(digits)


@dataclass(frozen=True)
class Derivation:
    """How a figure was reconstructed, in terms a reader can check."""

    expression: str
    value: float
    operation: str

    def __str__(self) -> str:
        return f"{self.expression} = {self.value:,.4g}"


# Quantity kinds. Arithmetic that ignores these produces derivations like
# `Capital expenditures / Current ratio as %` -- dividing dollars by a ratio --
# and matches `4.97x` against a dollar amount, or a percentage of 1,830 against
# a printed 1.83 by rescaling it. A percentage is already normalised; rescaling
# one by a thousand is meaningless, and so is using a ratio as an operand.
USD = "usd"
RATIO = "ratio"
PERCENT = "percent"

_RATIO_LABELS = ("ratio", "/ ebitda", "coverage", "dscr", "leverage")
_PERCENT_LABELS = ("margin", "growth", "concentration", "%", "share", "change")


def infer_kind(label: str) -> str:
    lowered = label.lower()
    if any(marker in lowered for marker in _PERCENT_LABELS):
        return PERCENT
    if any(marker in lowered for marker in _RATIO_LABELS):
        return RATIO
    return USD


@dataclass(frozen=True)
class Value:
    """A grounded number and the statement line it was printed under.

    `label` is a real accounting line, never a slice of prose. A derivation
    reading `(total debt + facility requested) / EBITDA` is checkable; one
    reading `Capital expenditures / The company operates in industrial staff` is
    self-evidently nonsense, and that the first version could produce the second
    is what exposed the bug.
    """

    label: str
    value: float
    kind: str = USD

    def __hash__(self) -> int:
        return hash((self.label, self.value, self.kind))

    def is_a(self, kinds: tuple[str, ...]) -> bool:
        lowered = self.label.lower()
        return any(kind in lowered for kind in kinds)


def _printed_kind(printed: str) -> str:
    """What kind of quantity the memo says it is printing."""
    text = printed.strip()
    if text.endswith("%"):
        return PERCENT
    if text.endswith("x"):
        return RATIO
    return USD


def _matches(printed: str, candidate: float, kind: str = USD) -> bool:
    """Does a candidate equal the printed figure, exactly, at some scale?

    Comparison is at the precision printed and nowhere looser. A memo writing
    "approximately $21.3 million" has rounded to one decimal, so the candidate
    must round to 21.3 at some scale. The half-percent tolerance the first
    version also allowed is gone: at three significant figures it was wide
    enough that unrelated combinations matched routinely.
    """
    text = printed.strip().lstrip("$").rstrip("x%").replace(",", "")
    try:
        target = float(text)
    except ValueError:
        return False
    if target == 0:
        return abs(candidate) < 1e-9
    if _significant_digits(text) < MIN_SIGNIFICANT_DIGITS:
        return False

    # A figure printed as a percentage must be reconstructed by a percentage
    # calculation, and one printed as a multiple by a ratio. Without this a
    # dollar amount satisfies "4.97x" and a percentage of 1,830 satisfies a
    # printed 1.83 after being divided by a thousand.
    if _printed_kind(printed) != kind:
        return False

    decimals = len(text.split(".")[1]) if "." in text else 0
    factor = 10.0**decimals

    # Only dollar amounts are written at different scales. "$21.3 million" is a
    # real way to print 21,308,027; "3.9x" is not a way to print 3,900.
    scales = SCALES if kind == USD else (1.0,)
    for scale in scales:
        scaled = candidate / scale
        if round(scaled, decimals) == round(target, decimals):
            return True
        # Truncation as well as rounding. The memo that motivated all of this
        # wrote "approximately 3.95x" for 3.9552 -- truncated, not rounded -- and
        # a rounding-only check rejected the one derivation the whole feature
        # exists to accept. Truncation adds a single extra bucket at the printed
        # precision, which is a far narrower allowance than the half-percent
        # relative tolerance it replaces.
        if int(scaled * factor) / factor == round(target, decimals):
            return True
    return False


def _one_step(values: list[Value]) -> list[tuple[float, str, str, str]]:
    """Everything reachable in a single operation over labelled figures."""
    out: list[tuple[float, str, str, str]] = []

    for item in values:
        out.append((item.value, item.label, "quoted", item.kind))
        if item.value < 0:
            out.append((abs(item.value), f"|{item.label}|", "magnitude", item.kind))

    # Only dollar amounts are operands. A ratio or a percentage is a result, not
    # an input -- dividing capital expenditure by the current ratio is not an
    # operation any analyst performs, and allowing it was producing derivations
    # that were arithmetically true and financially meaningless.
    amounts = [v for v in values if v.kind == USD]
    for a, b in combinations(amounts, 2):
        if b.value != 0:
            out.append((a.value / b.value, f"{a.label} / {b.label}", "ratio", RATIO))
            out.append(
                (
                    abs(a.value) / abs(b.value) * 100.0,
                    f"{a.label} / {b.label} as %",
                    "percentage",
                    PERCENT,
                )
            )
            out.append(
                (
                    (a.value - b.value) / abs(b.value) * 100.0,
                    f"change from {b.label} to {a.label}",
                    "change",
                    PERCENT,
                )
            )
        if a.value != 0:
            out.append((b.value / a.value, f"{b.label} / {a.label}", "ratio", RATIO))
            out.append(
                (
                    abs(b.value) / abs(a.value) * 100.0,
                    f"{b.label} / {a.label} as %",
                    "percentage",
                    PERCENT,
                )
            )
        out.append((a.value + b.value, f"{a.label} + {b.label}", "sum", USD))
        out.append((a.value - b.value, f"{a.label} - {b.label}", "difference", USD))
        out.append((b.value - a.value, f"{b.label} - {a.label}", "difference", USD))

    return out


def _pro_forma(values: list[Value]) -> list[tuple[float, str, str, str]]:
    """The one two-step shape worth allowing.

    Existing debt plus the requested facility, over earnings. Included because
    it is the most common thing a credit analyst writes that is not a plain
    metric, and because it is what flips a recommendation:

        "the requested facility of $12,800,000 would increase total debt to
        approximately $21.3 million, raising pro forma leverage to
        approximately 3.95x EBITDA"

    Restricted by label rather than open to any three values. An unrestricted
    two-step pass multiplies the candidate space by the operand count and was
    the largest single contributor to the first version's false approvals.
    """
    out: list[tuple[float, str, str, str]] = []
    debts = [v for v in values if v.is_a(_DEBT_LIKE)]
    requests = [v for v in values if v.is_a(_REQUEST_LIKE)]
    earnings = [v for v in values if v.is_a(_EARNINGS_LIKE)]

    for debt in debts:
        for request in requests:
            if debt is request or debt.value == request.value:
                continue
            combined = debt.value + request.value
            out.append(
                (combined, f"{debt.label} + {request.label}", "pro_forma_total", USD)
            )
            for base in earnings:
                if base.value == 0:
                    continue
                out.append(
                    (
                        combined / base.value,
                        f"({debt.label} + {request.label}) / {base.label}",
                        "pro_forma_ratio",
                        RATIO,
                    )
                )
    return out


def explain(printed: str, values: list[Value]) -> Derivation | None:
    """Find an arithmetic path from labelled figures to a printed figure.

    Returns the derivation, or None when the figure cannot be reached -- in
    which case it is unaccounted for and fails the gate.

    A returned derivation is a **candidate for review, not a proof**. Even
    narrowed, a three-significant-figure number can coincide with an unrelated
    combination, which is exactly why the expression is reported rather than
    swallowed.
    """
    if not printed:
        return None
    values = values[:MAX_VALUES]

    for candidate, expression, operation, kind in _one_step(values):
        if _matches(printed, candidate, kind):
            return Derivation(expression, candidate, operation)

    for candidate, expression, operation, kind in _pro_forma(values):
        if _matches(printed, candidate, kind):
            return Derivation(expression, candidate, operation)

    return None


def values_from_figures(
    figures: dict, log, extra: list[Value] | None = None
) -> list[Value]:
    """Build the operand set from labelled sources only.

    Deliberately does **not** scrape numbers off the page. That is what let a
    NAICS code and an employee count become arithmetic inputs, and no amount of
    tightening elsewhere compensates for admitting operands that were never
    financial figures to begin with.
    """
    values: list[Value] = [
        Value(label=figure.label, value=figure.value, kind=USD)
        for figure in figures.values()
    ]
    values += [
        Value(
            label=entry.name,
            value=entry.result,
            kind={"USD": USD, "x": RATIO, "percent": PERCENT}.get(entry.unit, USD),
        )
        for entry in log.entries
    ]
    values += extra or []

    seen: set[tuple[str, float]] = set()
    unique: list[Value] = []
    for item in values:
        key = (item.label, item.value)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:MAX_VALUES]
