"""Model routing (M9).

THE ARGUMENT
------------
Using one model for everything is a decision, not a default, and it is usually
the wrong one. The tasks in this pipeline are not alike:

**Extraction and classification** — is this text an instruction, does this answer
refuse, which excerpt is relevant. Narrow, verifiable, and a small model does
them as well as a large one. Paying Opus prices for "does this sentence contain
an instruction" is paying for judgement on a task with no judgement in it.

**Judgement** — writing a credit memo, weighing conflicting evidence, deciding
what a committee needs told. This is where a stronger model earns its price
difference, and where the difference shows up in output a human will read.

Fifteen times the cost is worth paying on the second and wasted on the first.

WHY ROUTING IS SAFE HERE, AND WOULD NOT BE ELSEWHERE
----------------------------------------------------
Routing usually trades quality for cost and you find out later which you lost.
This pipeline has an unusual property that removes most of that risk: **the
things that must be correct are not decided by a model at all.**

Figures are extracted mechanically and arithmetic is done by the calculator
(M7). Grounding is checked by string comparison (M6). The recommendation must
follow computed metrics (M8). A weaker model on the drafting step produces worse
prose; it cannot produce an ungrounded figure or an unsupported recommendation,
because those paths do not run through it.

That is what makes the cheap tier safe to use. Without those checks, routing
would be a straightforward gamble.

DEFAULT IS CHEAP, AND THE ESCALATION IS EXPLICIT
------------------------------------------------
Every task defaults to the small tier. Escalation is per-task and recorded, so
"we use a bigger model for the memo" is a line in a config rather than folklore.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Tier(StrEnum):
    """Capability tiers, cheapest first."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# Models per tier. Kept separate from `cost.PRICING` because the two answer
# different questions: this one is "what should we use", that one is "what does
# it cost". Conflating them makes it awkward to price a model you no longer use.
TIER_MODELS: dict[Tier, str] = {
    Tier.SMALL: "claude-haiku-4-5-20251001",
    Tier.MEDIUM: "claude-sonnet-4-5",
    Tier.LARGE: "claude-opus-4-1",
}


class Task(StrEnum):
    """Every place this pipeline can call a model."""

    ANSWER = "answer"  # answer one gold question from retrieved pages
    JUDGE = "judge"  # score faithfulness and relevance
    MEMO = "memo"  # write the credit memorandum
    CLASSIFY = "classify"  # short structured judgements


# Task to tier. The interesting entries are the ones that stay SMALL.
#
# JUDGE stays small deliberately, and that deserves saying out loud: a judge is
# graded on agreement with human review, and there is no evidence here that a
# larger judge agrees better. Spending more on the grader than on the thing
# being graded is a common and expensive reflex.
#
# MEMO is the one task where a stronger model has visibly earned it. The Claude
# memo for PCP-0011 noticed that four of five customers shared a name stem and
# raised related-party concentration -- a finding no template produces and no
# extraction step could have found.
DEFAULT_ROUTES: dict[Task, Tier] = {
    Task.ANSWER: Tier.SMALL,
    Task.JUDGE: Tier.SMALL,
    Task.CLASSIFY: Tier.SMALL,
    Task.MEMO: Tier.SMALL,
}


@dataclass(frozen=True)
class Route:
    task: Task
    tier: Tier
    model: str
    reason: str

    def line(self) -> str:
        return (
            f"{self.task.value:<10} {self.tier.value:<7} {self.model:<28} {self.reason}"
        )


@dataclass
class Router:
    """Chooses a model per task, with escalation recorded rather than implied."""

    routes: dict[Task, Tier] | None = None
    overrides: dict[Task, str] | None = None

    def __post_init__(self) -> None:
        self.routes = {**DEFAULT_ROUTES, **(self.routes or {})}
        self.overrides = self.overrides or {}

    def route(self, task: Task, escalate: bool = False) -> Route:
        """Pick a model.

        `escalate` moves one tier up and is intended for a retry after a
        verification failure, not as a general quality dial. Retrying a failed
        grounding check with a stronger model is a reasonable second attempt;
        starting there because the output "feels better" is how a bill grows
        without anyone being able to say what it bought.
        """
        if task in self.overrides:
            model = self.overrides[task]
            return Route(task, Tier.MEDIUM, model, "explicit override")

        tier = self.routes[task]
        reason = "default route"
        if escalate:
            order = [Tier.SMALL, Tier.MEDIUM, Tier.LARGE]
            index = min(order.index(tier) + 1, len(order) - 1)
            if order[index] != tier:
                tier = order[index]
                reason = "escalated after a verification failure"

        return Route(task, tier, TIER_MODELS[tier], reason)

    def table(self) -> str:
        return "\n".join(self.route(task).line() for task in Task)


def relative_cost(cheap: str, expensive: str, input_tokens: int, output_tokens: int):
    """What escalating one call would cost, as a multiple and a difference.

    Exists so a routing decision can be argued with a number instead of an
    intuition.
    """
    from pecos.cost import cost_of

    low = cost_of(cheap, input_tokens, output_tokens)
    high = cost_of(expensive, input_tokens, output_tokens)
    multiple = high / low if low else float("inf")
    return {"cheap_usd": low, "expensive_usd": high, "multiple": round(multiple, 1)}
