"""Answer validation: the rules a mapped answer passes before it enters a wire response.

A leaf calls :func:`validate_answers` after mapping the wire's output to each kind's shape,
with the rounding it declares. Every rule for the kind is run; an answer that passes is
returned unchanged, and one that fails becomes a :class:`~idia.core.Fault` naming every
rule it broke. Nothing is renormalized, rounded or corrected.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from idia.core import (
    WARNING_CHOSEN_BELOW_MAX,
    WARNING_EXTRA_ANSWER_KEY,
    Answer,
    BoxAnswer,
    ExtractionAnswer,
    Fault,
    LabelAnswer,
    MappedAnswer,
    NoulAnswer,
    Question,
    Refusal,
    Rounding,
    Rule,
    RuleFailure,
    Value,
    Warning,
)


def validate(question: Question, mapped: MappedAnswer, rounding: Rounding) -> Answer | Fault:
    """Run every rule for the question's kind over one mapped answer.

    A fault or refusal the leaf built passes through as it is; a fault keyed to another
    question is a leaf bug and raises ``ValueError``. Any other answer comes back
    unchanged when every rule passes, or as a fault recording the key, every failed rule with
    its observed value and bound, and the answer as mapped.
    """
    if isinstance(mapped, Fault):
        if mapped.key != question.key:
            raise ValueError(
                f"fault for key {mapped.key!r} was handed to question {question.key!r}"
            )
        return mapped
    if isinstance(mapped, Refusal):
        return mapped
    failures = _failures(question, mapped, rounding)
    if not failures:
        return mapped
    return Fault(key=question.key, failures=tuple(failures), mapped=mapped)


def validate_answers(
    questions: Sequence[Question],
    mapped: Mapping[str, MappedAnswer],
    rounding: Rounding,
) -> tuple[Mapping[str, Answer | Fault], tuple[Warning, ...]]:
    """Validate a map of answers against the questions asked.

    A question whose key is absent is a fault; a key with no question is dropped with an
    ``extra-answer-key`` warning; every other entry goes through :func:`validate` once. A
    label answer that stands with its chosen label below the highest probability, by at most
    one rounding step, adds a ``chosen-below-max`` warning naming the key. The answers come
    back keyed in question order.
    """
    answers: dict[str, Answer | Fault] = {}
    for question in questions:
        answer = mapped.get(question.key)
        if answer is None:
            failure = RuleFailure(rule=Rule.missing_key, observed=None, bound=None)
            answers[question.key] = Fault(key=question.key, failures=(failure,), mapped=None)
        else:
            answers[question.key] = validate(question, answer, rounding)
    asked = {question.key for question in questions}
    warnings = tuple(
        Warning(
            code=WARNING_EXTRA_ANSWER_KEY,
            message=f"answer key {key!r} matches no question and was dropped",
            field=key,
        )
        for key in mapped
        if key not in asked
    )
    below = tuple(
        warning
        for key, answer in answers.items()
        if (warning := _chosen_below_max(key, answer)) is not None
    )
    return MappingProxyType(answers), below + warnings


def _chosen_below_max(key: str, answer: Answer | Fault) -> Warning | None:
    """The warning for a label answer that stands with its chosen label below the highest
    probability, which the chosen-label rule tolerates by one rounding step."""
    if not isinstance(answer, LabelAnswer) or answer.chosen is None:
        return None
    by_name = dict(answer.probabilities)
    highest = max(by_name.values(), default=0.0)
    chosen = by_name.get(answer.chosen)
    if chosen is None or chosen >= highest:
        return None
    top = next(name for name, probability in answer.probabilities if probability == highest)
    return Warning(
        code=WARNING_CHOSEN_BELOW_MAX,
        message=(
            f"question {key!r}: the chosen label {answer.chosen!r} at {chosen} is below "
            f"{top!r} at {highest}, "
            "within one rounding step; the chosen label stands as given"
        ),
        field=key,
    )


# ----------------------------------------------------------------------------------------
# The rules
# ----------------------------------------------------------------------------------------


def _failures(question: Question, mapped: Answer, rounding: Rounding) -> list[RuleFailure]:
    kind = question.kind
    if kind == "noul":
        if not isinstance(mapped, NoulAnswer):
            return [_wrong_shape(mapped)]
        return _noul_failures(mapped)
    if kind in ("choice", "score"):
        if not isinstance(mapped, LabelAnswer):
            return [_wrong_shape(mapped)]
        return _label_failures(question, mapped, rounding)
    if kind == "extraction":
        if not isinstance(mapped, ExtractionAnswer):
            return [_wrong_shape(mapped)]
        return _extraction_failures(question, mapped)
    if not isinstance(mapped, BoxAnswer):
        return [_wrong_shape(mapped)]
    return _box_failures(mapped)


def _wrong_shape(mapped: Answer) -> RuleFailure:
    return RuleFailure(rule=Rule.kind_shape, observed=type(mapped).__name__, bound=None)


def _not_a_number(value: object) -> str | None:
    """What a value that cannot be judged as a number is observed as: its type's name when
    it is not an int or a float, or is a bool; ``int beyond float`` for an int too large
    for a float; ``None`` for a number."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return type(value).__name__
    try:
        float(value)
    except OverflowError:
        return "int beyond float"
    return None


def _is_number(value: object) -> bool:
    """Whether ``value`` is an int or a float, not a bool, that a float can hold."""
    return _not_a_number(value) is None


def _unit_interval(value: float, rule: Rule) -> list[RuleFailure]:
    """``not_finite`` when the value is not a number (a bool is not one, nor an int too large
    for a float) or not finite, else ``rule`` when it is outside [0, 1]. A value that is not
    a number is observed as its type's name, or as ``int beyond float``."""
    observed = _not_a_number(value)
    if observed is not None:
        return [RuleFailure(rule=Rule.not_finite, observed=observed, bound=None)]
    if not math.isfinite(value):
        return [RuleFailure(rule=Rule.not_finite, observed=value, bound=None)]
    if value < 0.0:
        return [RuleFailure(rule=rule, observed=value, bound=0.0)]
    if value > 1.0:
        return [RuleFailure(rule=rule, observed=value, bound=1.0)]
    return []


def _confidence(confidence: float | None) -> list[RuleFailure]:
    if confidence is None:
        return []
    return _unit_interval(confidence, Rule.confidence_range)


def _noul_failures(answer: NoulAnswer) -> list[RuleFailure]:
    return _unit_interval(answer.probability, Rule.probability_range) + _confidence(
        answer.confidence
    )


def _step(decimals: int | None) -> float:
    """How far below the highest probability a chosen label may lie: one rounding step,
    since a provider may round after it chooses, plus the float slack the sum allows."""
    return 1e-6 + (10.0**-decimals if decimals is not None else 0.0)


def _sum_tolerance(n: int, decimals: int | None) -> float:
    return 1e-6 + (n * 0.5 * 10.0**-decimals if decimals is not None else 0.0)


def _expected_level_bound(n: int, rounding: Rounding) -> float:
    s = rounding.score_decimals
    d = rounding.probability_decimals
    return (
        1e-6
        + (0.5 * 10.0**-s if s is not None else 0.0)
        + (0.5 * 10.0**-d * n * (n - 1) / 2 if d is not None else 0.0)
    )


def _label_failures(
    question: Question, answer: LabelAnswer, rounding: Rounding
) -> list[RuleFailure]:
    failures: list[RuleFailure] = []
    asked = tuple(label.name for label in question.labels)
    names = tuple(name for name, _ in answer.probabilities)
    if names != asked:
        failures.append(
            RuleFailure(rule=Rule.labels_mismatch, observed=",".join(names), bound=None)
        )
    if answer.echo is not None:
        # A level with no description is sent as its name, and the wire echoes that back.
        expected = (
            tuple(
                label.description if label.description is not None else label.name
                for label in question.labels
            )
            if answer.echo_of == "descriptions"
            else asked
        )
        if answer.echo != expected:
            failures.append(
                RuleFailure(rule=Rule.echo_mismatch, observed=",".join(answer.echo), bound=None)
            )
    probabilities = [probability for _, probability in answer.probabilities]
    for probability in probabilities:
        failures.extend(_unit_interval(probability, Rule.probability_range))
    if not all(_is_number(probability) for probability in probabilities):
        # The sum, the chosen label and the expected level are not judged over a value that
        # is not a number.
        failures.extend(_confidence(answer.confidence))
        return failures
    # Each is summed as a float, so ints a float can each hold never overflow together.
    total = sum(float(probability) for probability in probabilities)
    tolerance = _sum_tolerance(len(asked), rounding.probability_decimals)
    deviation = abs(total - 1.0)
    if not deviation <= tolerance:
        failures.append(RuleFailure(rule=Rule.sum_tolerance, observed=deviation, bound=tolerance))
    if answer.chosen is not None:
        highest = max(probabilities, default=0.0)
        by_name = dict(answer.probabilities)
        if answer.chosen not in by_name:
            failures.append(
                RuleFailure(rule=Rule.chosen_not_max, observed=answer.chosen, bound=highest)
            )
        elif highest - by_name[answer.chosen] > _step(rounding.probability_decimals):
            failures.append(
                RuleFailure(
                    rule=Rule.chosen_not_max, observed=by_name[answer.chosen], bound=highest
                )
            )
    if question.kind == "score" and answer.expected_level is not None:
        observed = _not_a_number(answer.expected_level)
        if observed is not None:
            failures.append(RuleFailure(rule=Rule.not_finite, observed=observed, bound=None))
        else:
            mean = sum(index * float(p) for index, p in enumerate(probabilities))
            bound = _expected_level_bound(len(asked), rounding)
            deviation = abs(answer.expected_level - mean)
            if not deviation <= bound:
                failures.append(
                    RuleFailure(rule=Rule.expected_level, observed=deviation, bound=bound)
                )
    failures.extend(_confidence(answer.confidence))
    return failures


def _has_type(value: object, value_type: str) -> bool:
    if value_type == "text":
        return isinstance(value, str)
    if value_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if value_type == "date":
        return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)
    if value_type == "time":
        return isinstance(value, datetime.time)
    if value_type == "datetime":
        return isinstance(value, datetime.datetime)
    return isinstance(value, datetime.timedelta)


def _value_failure(value: Value | None, question: Question) -> RuleFailure | None:
    """``value_type`` when the value is not of the declared type; ``None`` passes."""
    if value is None:
        return None
    value_type = question.value_type or "text"
    if question.is_list:
        if not isinstance(value, tuple):
            return RuleFailure(rule=Rule.value_type, observed=type(value).__name__, bound=None)
        for item in value:
            if not _has_type(item, value_type):
                return RuleFailure(rule=Rule.value_type, observed=type(item).__name__, bound=None)
        return None
    if isinstance(value, tuple) or not _has_type(value, value_type):
        return RuleFailure(rule=Rule.value_type, observed=type(value).__name__, bound=None)
    return None


def _extraction_failures(question: Question, answer: ExtractionAnswer) -> list[RuleFailure]:
    failures: list[RuleFailure] = []
    for held in (answer.value, answer.guess):
        failure = _value_failure(held, question)
        if failure is not None:
            failures.append(failure)
    if answer.p_null is not None:
        failures.extend(_unit_interval(answer.p_null, Rule.p_null_range))
    failures.extend(_confidence(answer.confidence))
    return failures


def _box_failures(answer: BoxAnswer) -> list[RuleFailure]:
    region = answer.region
    failures: list[RuleFailure] = []
    for edge in (region.top, region.left, region.bottom, region.right):
        failures.extend(_unit_interval(edge, Rule.box_range))
    if not all(_is_number(edge) for edge in (region.top, region.left, region.bottom, region.right)):
        failures.extend(_confidence(answer.confidence))
        return failures
    if region.top > region.bottom:
        failures.append(RuleFailure(rule=Rule.box_order, observed=region.top, bound=region.bottom))
    if region.left > region.right:
        failures.append(RuleFailure(rule=Rule.box_order, observed=region.left, bound=region.right))
    failures.extend(_confidence(answer.confidence))
    return failures
