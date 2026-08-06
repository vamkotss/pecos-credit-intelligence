"""Fast unit tests for the calculator (M7).

Arithmetic moved out of the model and into `tools.py` so that every derived
figure carries its inputs and their pages. These tests check the arithmetic and,
more importantly, that the provenance survives -- a correct ratio with no
recorded inputs is exactly the unverifiable figure the module exists to prevent.
"""

from __future__ import annotations

import pytest

from pecos.policy import MAX_LEVERAGE, MIN_DSCR
from pecos.tools import (
    CalculationError,
    CalculationLog,
    Input,
    concentration,
    convert_units,
    current_ratio,
    dscr,
    growth,
    leverage,
    total_debt,
)

DOC = "02_financial_statements_comparative.pdf"


def _input(label: str, value: float, page: int = 1) -> Input:
    return Input(label=label, value=value, document=DOC, page=page)


def test_leverage_is_debt_over_ebitda():
    result = leverage(_input("total debt", 8_025_829, 2), _input("EBITDA", 2_575_918))
    assert round(result.result, 2) == 3.12
    assert result.formatted() == "3.12x"


def test_every_calculation_records_where_its_inputs_came_from():
    """The whole point of the module. A ratio with no recorded inputs is right
    or wrong with no way to tell which, which in a credit memo is the same
    problem as an invented figure wearing better clothes."""
    result = leverage(_input("total debt", 8_000_000, 2), _input("EBITDA", 2_000_000))
    assert result.citations == (f"[{DOC}#p2]", f"[{DOC}#p1]")
    assert all(i.document and i.page for i in result.inputs)


def test_the_audit_line_is_readable_without_the_code():
    line = leverage(
        _input("total debt", 8_000_000, 2), _input("EBITDA", 2_000_000)
    ).audit_line()
    assert "4.00x" in line
    assert "total debt=8,000,000" in line
    assert f"[{DOC}#p2]" in line


def test_zero_ebitda_raises_rather_than_reporting_infinite_leverage():
    """A business with no operating profit is a finding to state in words, not
    a number to print. A sentinel value here would produce a memo that looks
    complete and says something false."""
    with pytest.raises(CalculationError, match="undefined"):
        leverage(_input("total debt", 1_000_000), _input("EBITDA", 0))


def test_negative_ebitda_also_raises():
    with pytest.raises(CalculationError):
        leverage(_input("total debt", 1_000_000), _input("EBITDA", -500_000))


def test_total_debt_sums_its_components():
    result = total_debt(
        [_input("current portion", 615_848, 2), _input("LTD", 7_409_981, 2)]
    )
    assert result.result == 8_025_829
    assert result.formatted() == "$8,025,829"


def test_summing_nothing_raises():
    with pytest.raises(CalculationError):
        total_debt([])


def test_dscr_subtracts_capex_before_covering_debt_service():
    """A business that stops replacing equipment to make loan payments is not
    covering its debt service, it is liquidating slowly."""
    result = dscr(
        _input("EBITDA", 2_575_918),
        _input("capex", 469_828, 3),
        _input("taxes", 198_223),
        _input("interest", 697_988),
        _input("principal", 615_848, 3),
    )
    assert round(result.result, 2) == 1.45
    assert "capex" in result.formula


def test_dscr_without_debt_service_raises():
    with pytest.raises(CalculationError):
        dscr(
            _input("EBITDA", 1_000_000),
            _input("capex", 0),
            _input("taxes", 0),
            _input("interest", 0),
            _input("principal", 0),
        )


def test_current_ratio_and_its_zero_case():
    assert (
        round(
            current_ratio(
                _input("ca", 8_147_101, 2), _input("cl", 1_981_656, 2)
            ).result,
            2,
        )
        == 4.11
    )
    with pytest.raises(CalculationError):
        current_ratio(_input("ca", 100), _input("cl", 0))


def test_concentration_is_a_percentage():
    result = concentration(_input("largest", 1_803_387), _input("total", 5_275_014))
    assert round(result.result, 1) == 34.2
    assert result.formatted() == "34.2%"


def test_unit_conversion_is_arithmetic_with_an_audit_trail():
    """M6 flagged a correct thousands conversion as an invented figure. Scaling
    is arithmetic, so it belongs with the arithmetic and produces the same
    record: printed value, multiplier, evidence, result."""
    result = convert_units(
        _input("gross receipts", 32_041), 1_000, evidence="stated in thousands"
    )
    assert result.result == 32_041_000
    assert result.formatted() == "$32,041,000"
    assert "thousands" in result.note
    assert result.inputs[0].value == 32_041


def test_an_invalid_scale_factor_raises():
    with pytest.raises(CalculationError):
        convert_units(_input("x", 100), 0)


def test_growth_handles_a_zero_prior_period():
    assert round(growth(_input("rev", 110), _input("rev prior", 100)).result, 1) == 10.0
    with pytest.raises(CalculationError):
        growth(_input("rev", 110), _input("rev prior", 0))


def test_the_log_exposes_derived_values_for_the_grounding_check():
    """This set is what lets the verifier tell a computed figure from an
    invented one. Without it every ratio in the memo reads as a fabrication."""
    log = CalculationLog()
    log.record(total_debt([_input("a", 5_000_000, 2), _input("b", 3_025_829, 2)]))
    log.record(
        leverage(_input("total debt", 8_025_829, 2), _input("EBITDA", 2_575_918))
    )

    values = log.derived_values()
    assert "3.12x" in values
    assert "$8,025,829" in values
    assert "8,025,829" in values


def test_the_log_keeps_derivations_in_the_order_they_happened():
    """A memo is reviewed as a narrative. An analyst checking a figure wants the
    derivations in the order the reasoning went, not alphabetically."""
    log = CalculationLog()
    log.record(total_debt([_input("a", 1_000)]))
    log.record(current_ratio(_input("ca", 200), _input("cl", 100)))
    assert [e.name for e in log.entries] == [
        "Total interest-bearing debt",
        "Current ratio",
    ]


def test_policy_thresholds_are_constants_not_prompt_text():
    """A threshold buried in an instruction cannot be tested and drifts silently
    whenever the prompt is edited."""
    assert MAX_LEVERAGE == 3.5
    assert MIN_DSCR == 1.25


# ---------------------------------------------------------------------------
# Metrics added after the first full agent run failed verification
# ---------------------------------------------------------------------------


def test_share_of_expresses_one_figure_as_a_percentage_of_another():
    """The workhorse the first version lacked.

    Twelve of twelve memos failed verification on the first full agent run, and
    almost every offending figure was a percentage: margins, an over-90
    receivables share, a cash decline, a top-five concentration. The prompt told
    the model not to compute anything the calculator had not provided; it
    computed them anyway, and it was right that a credit memo needs them.
    """
    from pecos.tools import share_of

    result = share_of(_input("over 90 days", 233_919), _input("total AR", 5_889_736))
    assert round(result.result, 1) == 4.0
    assert result.formatted() == "4.0%"
    assert result.citations


def test_change_keeps_the_sign_so_a_decline_reads_as_a_decline():
    from pecos.tools import change

    result = change(_input("cash", 518_376, 2), _input("cash prior", 2_490_977, 2))
    assert result.result < 0
    assert round(result.result, 1) == -79.2


def test_interest_coverage_uses_the_magnitude_of_a_negative_expense():
    """Interest prints as (697,988) on an income statement. Coverage of a
    negative number would come out negative and look like a catastrophe."""
    from pecos.tools import interest_coverage

    result = interest_coverage(
        _input("EBITDA", 2_575_918), _input("interest", -697_988)
    )
    assert round(result.result, 2) == 3.69


def test_a_recorded_sum_is_traceable_where_an_inline_one_is_not():
    """The top-five customer total was originally summed inline and handed
    straight to the concentration calculation. It then appeared in the audit
    trail as an input that existed on no page and in no calculation, and the
    verify gate correctly flagged it. An intermediate that cannot be traced is
    an unverifiable figure however briefly it lives."""
    from pecos.tools import total

    result = total(
        [_input("A", 915_268), _input("B", 669_470)], "Top five customer balances"
    )
    assert result.result == 1_584_738
    log = CalculationLog()
    log.record(result)
    assert "$1,584,738" in log.derived_values()


def test_percentage_results_are_recognised_however_they_are_rounded():
    """A memo may print 4.0% or 4%, and a decline may be written as a positive
    reduction. All are the same computed figure."""
    from pecos.tools import change, share_of

    log = CalculationLog()
    log.record(share_of(_input("part", 233_919), _input("whole", 5_889_736)))
    log.record(change(_input("cash", 518_376), _input("prior", 2_490_977)))
    values = log.derived_values()
    assert "4.0%" in values
    assert "4%" in values
    assert "79%" in values, "a decline written as a positive reduction must match"
