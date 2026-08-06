"""Cost accounting and budget enforcement (M9).

WHY THIS EXISTS, CONCRETELY
---------------------------
The red-team suite was run against Claude and crashed partway through with:

    anthropic.BadRequestError: Your credit balance is too low to access the
    Anthropic API.

Two things were wrong there, and neither was the API's fault.

**The pipeline had no idea what it was spending.** It ran until the provider
refused. `max_cost_usd_per_memo` had been sitting in the config since M1 and
nothing read it. A setting nothing enforces is a comment.

**A long job lost all its work on the first failure.** Twenty-odd completed
attacks vanished because the twenty-first raised. That is a resilience problem,
fixed in `redteam.py`, but the underlying cause was the same: nothing was
tracking how close the run was to the edge.

So: every model call is metered, every memo carries a ledger, and a budget is a
hard stop rather than a hope. `BudgetExceeded` is raised **before** a call that
would breach the limit, not after — a cap that only notices overspend once it has
happened is an audit trail, not a control.

PRICES ARE DATA, NOT ESTIMATES
------------------------------
The table below is list pricing per million tokens. It will go stale, so it is a
module constant with a date attached rather than a number buried in a function,
and `estimate_cost` on an unknown model raises rather than guessing. A silent
zero for an unpriced model is how a cost report ends up confidently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# List prices in USD per million tokens, as published on 2026-08-06. Verify
# against https://www.anthropic.com/pricing before quoting these anywhere.
PRICING: dict[str, tuple[float, float]] = {
    # model: (input per Mtok, output per Mtok)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-1": (15.00, 75.00),
}

# Rough characters per token for English prose. Only used when a call has not
# happened yet and a pre-flight estimate is needed; actual usage always comes
# from the API response.
CHARS_PER_TOKEN = 4.0


class BudgetExceeded(RuntimeError):
    """Raised before a call that would take the run over its budget."""


class UnknownModel(KeyError):
    """Raised rather than assuming a price.

    A model missing from the table is a model whose cost is unknown. Defaulting
    to zero would make the ledger silently understate spend, which is worse than
    failing loudly the first time someone points the pipeline at a new model.
    """


def price_for(model: str) -> tuple[float, float]:
    if model not in PRICING:
        raise UnknownModel(
            f"no price recorded for {model!r}. Add it to PRICING in cost.py "
            f"rather than letting the ledger guess."
        )
    return PRICING[model]


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call in USD."""
    input_price, output_price = price_for(model)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token count for a pre-flight estimate.

    Deliberately approximate and deliberately not a tokeniser: it is used only
    to decide whether a call is affordable before making it, where being roughly
    right in advance beats being exactly right afterwards.
    """
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class CallRecord:
    """One model call, priced."""

    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float = 0.0

    def line(self) -> str:
        return (
            f"{self.label:<24} {self.model:<28} "
            f"in={self.input_tokens:>6} out={self.output_tokens:>5} "
            f"${self.cost_usd:.4f}"
        )


@dataclass
class CostLedger:
    """Every call made while producing one artefact, with a hard ceiling.

    The budget is checked *before* a call, using an estimate of its input size.
    Checking afterwards would produce a ledger that accurately records having
    blown the budget, which is not a control.
    """

    budget_usd: float | None = None
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def remaining_usd(self) -> float | None:
        if self.budget_usd is None:
            return None
        return max(0.0, self.budget_usd - self.total_usd)

    def check_affordable(
        self, model: str, prompt: str, expected_output_tokens: int = 800
    ) -> None:
        """Raise if this call would breach the budget.

        The output estimate is generous on purpose. Underestimating it would let
        a call through that then overshoots, and the whole point of a pre-flight
        check is that the breach never happens.
        """
        if self.budget_usd is None:
            return
        projected = cost_of(model, estimate_tokens(prompt), expected_output_tokens)
        if self.total_usd + projected > self.budget_usd:
            raise BudgetExceeded(
                f"call would cost about ${projected:.4f}, taking the total to "
                f"${self.total_usd + projected:.4f} against a budget of "
                f"${self.budget_usd:.2f}. Raise max_cost_usd_per_memo or reduce "
                f"the work."
            )

    def record(
        self,
        label: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        seconds: float = 0.0,
    ) -> CallRecord:
        call = CallRecord(
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_of(model, input_tokens, output_tokens),
            seconds=seconds,
        )
        self.calls.append(call)
        return call

    def record_response(self, label: str, model: str, message, seconds: float = 0.0):
        """Record from an Anthropic response object.

        Actual usage from the response, never an estimate. The estimate exists
        to decide whether to make the call; once it has been made, the provider
        knows exactly what it cost and there is no reason to guess.
        """
        usage = getattr(message, "usage", None)
        return self.record(
            label=label,
            model=model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            seconds=seconds,
        )

    def summary(self) -> str:
        if not self.calls:
            return "no model calls (offline path)"
        lines = [call.line() for call in self.calls]
        lines.append(
            f"{'TOTAL':<24} {len(self.calls)} call(s)  "
            f"in={self.total_input_tokens} out={self.total_output_tokens}  "
            f"${self.total_usd:.4f}"
        )
        if self.budget_usd is not None:
            lines.append(
                f"{'budget':<24} ${self.budget_usd:.2f}  "
                f"remaining ${self.remaining_usd:.4f}"
            )
        return "\n".join(lines)
