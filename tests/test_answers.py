"""Answer validation: every rule, the tolerance formulas, and the map-level rules."""

from __future__ import annotations

import datetime
import math
from types import MappingProxyType
from typing import cast

import pytest

from idia import (
    WARNING_EXTRA_ANSWER_KEY,
    Box,
    BoxAnswer,
    ExtractionAnswer,
    Fault,
    Label,
    LabelAnswer,
    NoulAnswer,
    Question,
    Refusal,
    Rounding,
    Rule,
    RuleFailure,
    validate,
    validate_answers,
)
from tests._support import TWO_DECIMALS, UNROUNDED, box, choice, extraction, noul, score


def _rules(result: object) -> list[Rule]:
    assert isinstance(result, Fault)
    return [failure.rule for failure in result.failures]


def _failure(result: object, rule: Rule) -> RuleFailure:
    assert isinstance(result, Fault)
    return next(failure for failure in result.failures if failure.rule == rule)


# ----------------------------------------------------------------------------------------
# noul
# ----------------------------------------------------------------------------------------


def test_noul_answer_keeps_the_provider_float_unchanged() -> None:
    answer = NoulAnswer(probability=0.73, confidence=0.9)
    assert validate(noul(), answer, TWO_DECIMALS) is answer
    assert answer.probability == 0.73
    assert not hasattr(answer, "probabilities")


def test_noul_probability_outside_unit_interval_is_a_fault() -> None:
    result = validate(noul(), NoulAnswer(probability=1.2), UNROUNDED)
    assert _rules(result) == [Rule.probability_range]
    assert _failure(result, Rule.probability_range).bound == 1.0
    assert isinstance(result, Fault)
    assert result.key == "q"
    assert isinstance(result.mapped, NoulAnswer)
    assert result.mapped.probability == 1.2
    assert _rules(validate(noul(), NoulAnswer(probability=-0.1), UNROUNDED)) == [
        Rule.probability_range
    ]


def test_noul_probability_must_be_finite() -> None:
    assert _rules(validate(noul(), NoulAnswer(probability=math.nan), UNROUNDED)) == [
        Rule.not_finite
    ]


def _not_finite_cases(bad: float) -> list[tuple[Question, object]]:
    """One answer per field validate reads as a unit fraction, ``bad`` in that field."""
    return [
        (noul(), NoulAnswer(probability=bad)),
        (noul(), NoulAnswer(probability=0.5, confidence=bad)),
        (choice(), LabelAnswer(probabilities=(("red", bad), ("green", 0.0), ("blue", 0.0)))),
        (
            score(),
            LabelAnswer(probabilities=(("level 0", 0.0), ("level 1", bad), ("level 2", 0.0))),
        ),
        (
            choice(),
            LabelAnswer(
                probabilities=(("red", 1.0), ("green", 0.0), ("blue", 0.0)), confidence=bad
            ),
        ),
        (extraction(), ExtractionAnswer(value=None, p_null=bad)),
        (extraction(), ExtractionAnswer(value="x", confidence=bad)),
        (box(), BoxAnswer(region=Box(top=bad, left=0.0, bottom=1.0, right=1.0))),
        (box(), BoxAnswer(region=Box(top=0.0, left=0.0, bottom=1.0, right=bad))),
        (box(), BoxAnswer(region=Box(top=0.0, left=0.0, bottom=1.0, right=1.0), confidence=bad)),
    ]


@pytest.mark.parametrize(
    ("bad", "observed"),
    [
        (True, "bool"),
        (False, "bool"),
        (cast("float", "0.5"), "str"),
        (cast("float", [0.5]), "list"),
    ],
)
@pytest.mark.parametrize("case", range(10))
def test_a_bool_or_non_number_probability_faults_not_finite(
    bad: float, observed: str, case: int
) -> None:
    """A bool or any value that is not a number, in any probability, confidence, ``p_null``
    or box edge, is a ``not_finite`` fault observed as its type's name, and the sum, order
    and level checks are not judged over it."""
    question, answer = _not_finite_cases(bad)[case]
    result = validate(question, answer, UNROUNDED)  # pyright: ignore[reportArgumentType]
    assert _rules(result) == [Rule.not_finite]
    assert _failure(result, Rule.not_finite).observed == observed


def test_wrong_shape_for_the_kind_is_a_fault() -> None:
    label = LabelAnswer(probabilities=(("red", 1.0),))
    assert _rules(validate(noul(), label, UNROUNDED)) == [Rule.kind_shape]
    assert _rules(validate(choice(), NoulAnswer(probability=0.5), UNROUNDED)) == [Rule.kind_shape]
    assert _rules(validate(extraction(), label, UNROUNDED)) == [Rule.kind_shape]
    assert _rules(validate(box(), label, UNROUNDED)) == [Rule.kind_shape]
    assert _failure(validate(box(), label, UNROUNDED), Rule.kind_shape).observed == "LabelAnswer"


# ----------------------------------------------------------------------------------------
# choice and score
# ----------------------------------------------------------------------------------------


def test_sum_within_rounding_tolerance_passes_unchanged() -> None:
    answer = LabelAnswer(probabilities=(("red", 0.33), ("green", 0.33), ("blue", 0.33)))
    assert validate(choice(), answer, TWO_DECIMALS) is answer


def test_sum_outside_rounding_tolerance_is_a_fault_and_not_renormalized() -> None:
    answer = LabelAnswer(probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.3)))
    result = validate(choice(), answer, TWO_DECIMALS)
    assert _rules(result) == [Rule.sum_tolerance]
    failure = _failure(result, Rule.sum_tolerance)
    assert failure.observed == pytest.approx(0.1)
    assert failure.bound == pytest.approx(1e-6 + 3 * 0.005)
    assert isinstance(result, Fault)
    assert result.mapped is answer
    assert answer.probabilities == (("red", 0.5), ("green", 0.3), ("blue", 0.3))


def test_unrounded_provider_uses_1e_6_tolerance() -> None:
    answer = LabelAnswer(probabilities=(("red", 0.33), ("green", 0.33), ("blue", 0.33)))
    result = validate(choice(), answer, UNROUNDED)
    assert _rules(result) == [Rule.sum_tolerance]
    assert _failure(result, Rule.sum_tolerance).bound == 1e-6
    exact = LabelAnswer(probabilities=(("red", 0.25), ("green", 0.25), ("blue", 0.5)))
    assert validate(choice(), exact, UNROUNDED) is exact


def test_chosen_label_tied_at_max_passes() -> None:
    answer = LabelAnswer(
        probabilities=(("red", 0.4), ("green", 0.4), ("blue", 0.2)), chosen="green"
    )
    assert validate(choice(), answer, TWO_DECIMALS) is answer


def test_chosen_label_that_is_not_the_max_is_a_fault() -> None:
    answer = LabelAnswer(probabilities=(("red", 0.6), ("green", 0.3), ("blue", 0.1)), chosen="blue")
    result = validate(choice(), answer, TWO_DECIMALS)
    assert _rules(result) == [Rule.chosen_not_max]
    failure = _failure(result, Rule.chosen_not_max)
    assert failure.observed == 0.1
    assert failure.bound == 0.6
    unknown = LabelAnswer(probabilities=(("red", 0.6), ("green", 0.3), ("blue", 0.1)), chosen="x")
    assert _failure(validate(choice(), unknown, TWO_DECIMALS), Rule.chosen_not_max).observed == "x"


def test_label_answer_with_labels_out_of_order_is_a_fault() -> None:
    swapped = LabelAnswer(probabilities=(("green", 0.5), ("red", 0.3), ("blue", 0.2)))
    result = validate(choice(), swapped, TWO_DECIMALS)
    assert _rules(result) == [Rule.labels_mismatch]
    assert _failure(result, Rule.labels_mismatch).observed == "green,red,blue"
    missing = LabelAnswer(probabilities=(("red", 0.5), ("green", 0.5)))
    assert Rule.labels_mismatch in _rules(validate(choice(), missing, TWO_DECIMALS))
    repeated = LabelAnswer(probabilities=(("red", 0.5), ("red", 0.3), ("blue", 0.2)))
    assert Rule.labels_mismatch in _rules(validate(choice(), repeated, TWO_DECIMALS))


def test_echo_mismatch_is_a_fault() -> None:
    names = ("red", "green", "blue")
    probabilities = (("red", 0.5), ("green", 0.3), ("blue", 0.2))
    echoed = LabelAnswer(probabilities=probabilities, echo=names, echo_of="names")
    assert validate(choice(), echoed, TWO_DECIMALS) is echoed
    legend = tuple(f"the colour {name}" for name in names)
    described = LabelAnswer(probabilities=probabilities, echo=legend, echo_of="descriptions")
    assert validate(choice(), described, TWO_DECIMALS) is described
    wrong = LabelAnswer(probabilities=probabilities, echo=("red", "blue", "green"), echo_of="names")
    result = validate(choice(), wrong, TWO_DECIMALS)
    assert _rules(result) == [Rule.echo_mismatch]
    assert _failure(result, Rule.echo_mismatch).observed == "red,blue,green"
    wrong_legend = LabelAnswer(probabilities=probabilities, echo=names, echo_of="descriptions")
    assert _rules(validate(choice(), wrong_legend, TWO_DECIMALS)) == [Rule.echo_mismatch]


def test_expected_level_off_by_rounded_mean_passes() -> None:
    # Probabilities rounded to two decimals give a mean of 0.97; the provider's own
    # unrounded expected level was 0.98. The bound allows the difference.
    answer = LabelAnswer(
        probabilities=(("level 0", 0.33), ("level 1", 0.37), ("level 2", 0.30)),
        expected_level=0.98,
    )
    assert validate(score(), answer, TWO_DECIMALS) is answer


@pytest.mark.parametrize(("level", "observed"), [("1", "str"), (True, "bool")])
def test_an_expected_level_that_is_not_a_number_is_not_finite(level: object, observed: str) -> None:
    answer = LabelAnswer(
        probabilities=(("level 0", 0.33), ("level 1", 0.37), ("level 2", 0.30)),
        expected_level=cast(float, level),
    )
    result = validate(score(), answer, TWO_DECIMALS)
    assert _rules(result) == [Rule.not_finite]
    assert _failure(result, Rule.not_finite).observed == observed


def test_expected_level_outside_its_bound_is_a_fault() -> None:
    answer = LabelAnswer(
        probabilities=(("level 0", 0.33), ("level 1", 0.37), ("level 2", 0.30)),
        expected_level=1.5,
    )
    result = validate(score(), answer, TWO_DECIMALS)
    assert _rules(result) == [Rule.expected_level]
    failure = _failure(result, Rule.expected_level)
    assert failure.observed == pytest.approx(0.53)
    assert failure.bound == pytest.approx(1e-6 + 0.005 + 0.005 * 3)
    unrounded = LabelAnswer(
        probabilities=(("level 0", 0.5), ("level 1", 0.25), ("level 2", 0.25)),
        expected_level=0.75,
    )
    assert validate(score(), unrounded, UNROUNDED) is unrounded
    only_scores_rounded = Rounding(probability_decimals=None, score_decimals=1)
    off_by_rounding = LabelAnswer(
        probabilities=(("level 0", 0.5), ("level 1", 0.25), ("level 2", 0.25)),
        expected_level=0.8,
    )
    assert validate(score(), off_by_rounding, only_scores_rounded) is off_by_rounding


def test_chat_derived_expected_level_is_validated_and_marked_derived() -> None:
    derived = LabelAnswer(
        probabilities=(("level 0", 0.2), ("level 1", 0.3), ("level 2", 0.5)),
        expected_level=1.3,
        chosen="level 2",
        off_label_mass=0.04,
        derived=frozenset({"expected_level", "chosen"}),
    )
    result = validate(score(), derived, UNROUNDED)
    assert result is derived
    assert derived.derived == {"expected_level", "chosen"}
    wrong = LabelAnswer(
        probabilities=(("level 0", 0.2), ("level 1", 0.3), ("level 2", 0.5)),
        expected_level=2.0,
        off_label_mass=0.9,
        derived=frozenset({"expected_level"}),
    )
    assert _rules(validate(score(), wrong, UNROUNDED)) == [Rule.expected_level]


def test_every_label_rule_is_run_and_each_failure_is_recorded() -> None:
    answer = LabelAnswer(
        probabilities=(("red", 1.4), ("blue", -0.2), ("green", math.inf)),
        chosen="red",
        confidence=2.0,
    )
    rules = _rules(validate(choice(), answer, TWO_DECIMALS))
    assert rules.count(Rule.probability_range) == 2
    assert rules.count(Rule.not_finite) == 1
    assert Rule.labels_mismatch in rules
    assert Rule.sum_tolerance in rules
    assert Rule.chosen_not_max in rules
    assert Rule.confidence_range in rules


# ----------------------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------------------


def test_null_extraction_value_keeps_the_provider_guess() -> None:
    answer = ExtractionAnswer(value=None, guess="Ada", confidence=0.4, p_null=0.8)
    result = validate(extraction(), answer, UNROUNDED)
    assert result is answer
    assert isinstance(result, ExtractionAnswer)
    assert result.value is None
    assert result.guess == "Ada"


def test_extraction_value_of_the_wrong_type_is_a_fault() -> None:
    result = validate(
        extraction("q", "number"), ExtractionAnswer(value="12", p_null=0.0), UNROUNDED
    )
    assert _rules(result) == [Rule.value_type]
    assert _failure(result, Rule.value_type).observed == "str"
    guessed = ExtractionAnswer(value=12, guess=True, p_null=0.0)
    assert _rules(validate(extraction("q", "number"), guessed, UNROUNDED)) == [Rule.value_type]


@pytest.mark.parametrize(
    ("value_type", "good", "bad"),
    [
        ("text", "Ada", 1),
        ("number", 1.5, "1.5"),
        ("date", datetime.date(2026, 1, 1), datetime.datetime(2026, 1, 1)),
        ("time", datetime.time(9, 30), "09:30"),
        ("datetime", datetime.datetime(2026, 1, 1, 9), datetime.date(2026, 1, 1)),
        ("duration", datetime.timedelta(hours=1), 3600),
    ],
)
def test_each_value_type_is_checked(value_type: str, good: object, bad: object) -> None:
    question = extraction("q", value_type)
    passing = ExtractionAnswer(value=good, p_null=0.0)  # type: ignore[arg-type]
    assert validate(question, passing, UNROUNDED) is passing
    failing = ExtractionAnswer(value=bad, p_null=0.0)  # type: ignore[arg-type]
    assert _rules(validate(question, failing, UNROUNDED)) == [Rule.value_type]


def test_list_extraction_checks_every_item() -> None:
    question = extraction("q", "text", is_list=True)
    good = ExtractionAnswer(value=("Ada", "Grace"), p_null=0.0)
    assert validate(question, good, UNROUNDED) is good
    not_a_list = ExtractionAnswer(value="Ada", p_null=0.0)
    assert _rules(validate(question, not_a_list, UNROUNDED)) == [Rule.value_type]
    bad_item = ExtractionAnswer(value=("Ada", 3), p_null=0.0)
    assert _failure(validate(question, bad_item, UNROUNDED), Rule.value_type).observed == "int"
    scalar_question = extraction("q", "text")
    listed = ExtractionAnswer(value=("Ada",), p_null=0.0)
    assert _rules(validate(scalar_question, listed, UNROUNDED)) == [Rule.value_type]


def test_extraction_p_null_and_confidence_are_checked() -> None:
    answer = ExtractionAnswer(value="Ada", confidence=1.5, p_null=-0.5)
    rules = _rules(validate(extraction(), answer, UNROUNDED))
    assert rules == [Rule.p_null_range, Rule.confidence_range]
    assert _rules(
        validate(extraction(), ExtractionAnswer(value=None, p_null=math.nan), UNROUNDED)
    ) == [Rule.not_finite]


def test_an_extraction_without_p_null_passes_unchecked() -> None:
    answer = ExtractionAnswer(value="Ada", confidence=0.9)
    assert answer.p_null is None
    assert validate(extraction(), answer, UNROUNDED) is answer


# ----------------------------------------------------------------------------------------
# box
# ----------------------------------------------------------------------------------------


def test_box_with_top_below_bottom_is_a_fault() -> None:
    answer = BoxAnswer(region=Box(top=0.8, left=0.1, bottom=0.2, right=0.9))
    result = validate(box(), answer, UNROUNDED)
    assert _rules(result) == [Rule.box_order]
    failure = _failure(result, Rule.box_order)
    assert failure.observed == 0.8
    assert failure.bound == 0.2
    sideways = BoxAnswer(region=Box(top=0.1, left=0.9, bottom=0.2, right=0.1))
    assert _rules(validate(box(), sideways, UNROUNDED)) == [Rule.box_order]


def test_box_fraction_outside_unit_interval_is_a_fault() -> None:
    answer = BoxAnswer(region=Box(top=0.0, left=-0.1, bottom=1.5, right=1.0), confidence=0.5)
    rules = _rules(validate(box(), answer, UNROUNDED))
    assert rules == [Rule.box_range, Rule.box_range]
    good = BoxAnswer(region=Box(top=0.0, left=0.0, bottom=1.0, right=1.0), confidence=0.5)
    assert validate(box(), good, UNROUNDED) is good
    assert _rules(validate(box(), BoxAnswer(region=good.region, confidence=-1), UNROUNDED)) == [
        Rule.confidence_range
    ]


# ----------------------------------------------------------------------------------------
# faults and refusals pass through; the map-level rules
# ----------------------------------------------------------------------------------------


def test_candidate_shape_for_an_admitted_kind_is_a_fault() -> None:
    fault = Fault(
        key="a",
        failures=(RuleFailure(rule=Rule.candidate_shape, observed="estimate", bound=None),),
        mapped=None,
    )
    sibling = NoulAnswer(probability=0.5)
    answers, warnings = validate_answers(
        [extraction("a", "number"), noul("b")], {"a": fault, "b": sibling}, UNROUNDED
    )
    assert answers["a"] is fault
    assert answers["b"] is sibling
    assert warnings == ()


def test_one_unmappable_answer_is_a_fault_and_siblings_stand() -> None:
    unmappable = Fault(
        key="a",
        failures=(RuleFailure(rule=Rule.kind_shape, observed="<html>", bound=None),),
        mapped=None,
    )
    refused = Refusal()
    good = LabelAnswer(probabilities=(("red", 0.5), ("green", 0.25), ("blue", 0.25)))
    bad = NoulAnswer(probability=7.0)
    answers, _ = validate_answers(
        [noul("a"), noul("b"), choice("c"), noul("d")],
        {"a": unmappable, "b": refused, "c": good, "d": bad},
        UNROUNDED,
    )
    assert answers["a"] is unmappable
    assert answers["b"] is refused
    assert answers["c"] is good
    assert _rules(answers["d"]) == [Rule.probability_range]
    assert list(answers) == ["a", "b", "c", "d"]


def test_leaf_built_logprob_faults_pass_through_unchanged() -> None:
    for rule in (
        Rule.empty_logprobs,
        Rule.label_absent_from_logprobs,
        Rule.off_label_mass_above_bound,
    ):
        fault = Fault(
            key="q", failures=(RuleFailure(rule=rule, observed=0.3, bound=0.2),), mapped=None
        )
        assert validate(choice(), fault, UNROUNDED) is fault


def test_off_label_mass_is_not_checked_by_validate() -> None:
    answer = NoulAnswer(probability=0.5, off_label_mass=0.99, derived=frozenset({"probability"}))
    assert validate(noul(), answer, UNROUNDED) is answer


def test_missing_answer_key_is_a_fault() -> None:
    answers, warnings = validate_answers(
        [noul("a"), noul("b")], {"a": NoulAnswer(probability=0.5)}, UNROUNDED
    )
    assert warnings == ()
    missing = answers["b"]
    assert isinstance(missing, Fault)
    assert missing.key == "b"
    assert missing.failures == (RuleFailure(rule=Rule.missing_key, observed=None, bound=None),)
    assert missing.mapped is None
    assert isinstance(answers, MappingProxyType)


def test_extra_answer_key_is_dropped_with_a_warning() -> None:
    answers, warnings = validate_answers(
        [noul("a")],
        {"a": NoulAnswer(probability=0.5), "zz": NoulAnswer(probability=0.1)},
        UNROUNDED,
    )
    assert set(answers) == {"a"}
    assert len(warnings) == 1
    assert warnings[0].code == "extra-answer-key"
    assert warnings[0].field == "zz"
    assert "zz" in warnings[0].message


def test_echo_of_descriptions_falls_back_to_the_name_for_an_undescribed_level() -> None:
    question = Question(
        key="q",
        kind="score",
        instruction="How much?",
        labels=(Label(name="low", description="hardly any"), Label(name="high")),
    )
    probabilities = (("low", 0.4), ("high", 0.6))
    echoed = LabelAnswer(
        probabilities=probabilities, echo=("hardly any", "high"), echo_of="descriptions"
    )
    assert validate(question, echoed, UNROUNDED) is echoed
    wrong = LabelAnswer(probabilities=probabilities, echo=("low", "high"), echo_of="descriptions")
    assert _rules(validate(question, wrong, UNROUNDED)) == [Rule.echo_mismatch]


def test_fault_keyed_to_another_question_is_a_leaf_bug() -> None:
    fault = Fault(
        key="other",
        failures=(RuleFailure(rule=Rule.kind_shape, observed=None, bound=None),),
        mapped=None,
    )
    with pytest.raises(ValueError, match="'other'"):
        validate(noul("q"), fault, UNROUNDED)


def test_extra_answer_key_warning_uses_the_published_code() -> None:
    _, warnings = validate_answers([noul("a")], {"a": Refusal(), "b": Refusal()}, UNROUNDED)
    assert warnings[0].code == WARNING_EXTRA_ANSWER_KEY
