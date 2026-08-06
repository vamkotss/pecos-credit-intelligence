"""Pecos credit policy thresholds (M7).

Constants in one module rather than numbers inside a prompt or a drafter.

Three things follow from that. The thresholds can be tested. They can be changed
in one place when the credit committee changes them. And the memo verifier can
recognise them: "leverage of 3.12x is within the 3.5x policy limit" states one
measured figure and one rule, and the rule has no page to cite -- without this
list the verifier reports the lender's own policy as an ungrounded claim.
"""

from __future__ import annotations

# Total debt / EBITDA above this needs a structural mitigant.
MAX_LEVERAGE = 3.5

# Cash available after capex and tax, over debt service. Below this the borrower
# does not cover its obligations from operations.
MIN_DSCR = 1.25

# Current assets over current liabilities.
MIN_CURRENT_RATIO = 1.1

# Share of receivables owed by a single customer that warrants comment.
CONCENTRATION_CONCERN = 25.0

# Facility size band Pecos writes.
MIN_FACILITY_USD = 3_000_000
MAX_FACILITY_USD = 40_000_000

# Rendered forms a memo might print, so the verifier can tell a policy threshold
# from a borrower figure.
POLICY_CONSTANTS: frozenset[str] = frozenset(
    {
        f"{MAX_LEVERAGE:.1f}x",
        f"{MAX_LEVERAGE:.2f}x",
        f"{MIN_DSCR:.2f}x",
        f"{MIN_DSCR:.1f}x",
        f"{MIN_CURRENT_RATIO:.1f}x",
        f"{MIN_CURRENT_RATIO:.2f}x",
        f"{CONCENTRATION_CONCERN:.0f}%",
        f"{CONCENTRATION_CONCERN:.1f}%",
        f"{MIN_FACILITY_USD:,}",
        f"{MAX_FACILITY_USD:,}",
    }
)
