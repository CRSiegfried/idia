"""Fit: does a request fit a provider's declared capabilities, and if so, what is sent.

Preflight checks a request against a :class:`~idia.core.Capabilities` record before
any call and either raises :class:`~idia.core.DoesNotFit` or returns a fit plan: the
wire requests the request becomes, in plan order. When the questions overflow and the state
fits, the plan packs the questions greedily, in order, into several wire requests that each
repeat the state, every key in exactly one. When the state alone overflows a token limit,
the overflow policy decides: :class:`Fail`, the default, raises; a named policy returns a
smaller state, which is checked again, and the plan carries a warning naming the policy.
Nothing is truncated silently.

A limit of ``None`` is unknown: its check is skipped and each affected wire request carries
an ``unchecked-limit`` warning naming the field. :data:`~idia.core.UNBOUNDED` is
neither checked nor warned.

Fit's own body measure covers the state and the questions, not the wire envelope a leaf
wraps them in, so a body limit is checked approximately.

The merge turns the plan's outcomes and their answers back into one answers map and the
outcomes in plan order. Only :class:`OverflowPolicy`, :class:`Fail` and :class:`FitPlan` are
public; the dispatcher calls the rest.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from idia.core import (
    WARNING_OVERFLOW_POLICY,
    WARNING_UNCHECKED_LIMIT,
    Answer,
    Capabilities,
    DoesNotFit,
    ImagePart,
    JsonPart,
    JsonValue,
    Limit,
    Modality,
    Notice,
    Outcome,
    Question,
    Request,
    State,
    TextPart,
    TokenEstimate,
    TokenEstimator,
    WireRequest,
    _declared_media_type,  # pyright: ignore[reportPrivateUsage]
    _image_media_type,  # pyright: ignore[reportPrivateUsage]
    _image_size,  # pyright: ignore[reportPrivateUsage]
    _signature_media_type,  # pyright: ignore[reportPrivateUsage]
)

# ----------------------------------------------------------------------------------------
# The overflow policy
# ----------------------------------------------------------------------------------------


@runtime_checkable
class OverflowPolicy(Protocol):
    """What preflight applies when the state alone is over a token limit.

    ``limit`` is the tighter of ``max_input_tokens`` and ``max_question_tokens`` the state
    is over. ``apply`` returns a state that fits it by ``estimator``, or raises
    :class:`~idia.core.DoesNotFit`. Preflight checks the returned state again, so a
    policy whose state still overflows raises all the same, and a response whose state a
    policy changed carries an ``overflow-policy`` warning naming it. A body-size overflow is
    never handed to a policy: it always raises.
    """

    @property
    def name(self) -> str: ...

    def apply(self, state: State, limit: int, estimator: TokenEstimator) -> State: ...


@dataclass(frozen=True, slots=True)
class Fail:
    """The default overflow policy: a state that overflows raises before any call.

    ``apply`` cannot know which limit the state was over, so the ``check`` it raises with
    names ``max_input_tokens``; preflight re-raises it with the limit the state was over,
    the record's off features and the estimate it measured.
    """

    @property
    def name(self) -> str:
        return "fail"

    def apply(self, state: State, limit: int, estimator: TokenEstimator) -> State:
        observed = _with_margin(estimator.estimate(state, ()))
        raise DoesNotFit(
            f"the state alone is estimated at {observed} tokens, over the limit of {limit}",
            check="max_input_tokens",
            observed=observed,
            limit=limit,
            state_overflow=True,
        )


# ----------------------------------------------------------------------------------------
# The fit plan
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class FitPlan:
    """The wire requests one request becomes, in plan order, and the warnings preflight
    raised.

    ``outcome_warnings`` runs parallel to ``wire_requests``: entry ``i`` belongs on the
    outcome of wire request ``i``. ``response_warnings`` belong on the response.
    """

    wire_requests: tuple[WireRequest, ...]
    outcome_warnings: tuple[tuple[Notice, ...], ...]
    response_warnings: tuple[Notice, ...] = ()

    def __post_init__(self) -> None:
        if len(self.outcome_warnings) != len(self.wire_requests):
            raise ValueError("outcome_warnings runs parallel to wire_requests")


# The limits a wire request's size is held to, which packing respects, cheapest first.
_PACKABLE: Final = ("max_questions", "max_total_options", "max_body_bytes", "max_input_tokens")

# The token limits the state alone is held to; over one, the overflow policy applies.
_STATE_TOKEN_LIMITS: Final = ("max_input_tokens", "max_question_tokens")

# Every limit an outcome can be warned about, in the order its warnings are listed.
_WARNED: Final = (
    "max_input_tokens",
    "max_question_tokens",
    "max_body_bytes",
    "max_questions",
    "max_images",
    "max_images_per_question",
    "max_image_bytes",
    "max_total_image_bytes",
    "min_image_side",
    "image_media_types",
    "max_labels",
    "max_levels",
    "min_levels",
    "max_total_options",
)


def _preflight(  # pyright: ignore[reportUnusedFunction]
    request: Request,
    capabilities: Capabilities,
    estimator: TokenEstimator,
    overflow: OverflowPolicy,
    *,
    model: str | None = None,
) -> FitPlan:
    """Check ``request`` against a leaf's record and plan its wire requests.

    Raises :class:`~idia.core.DoesNotFit` naming the failing field when the request
    cannot be sent; the estimator is called only for a known token limit. ``model`` is the
    leaf's model name, used only in the message for a kind or modality no feature adds.
    """
    checker = _Checker(capabilities, model)
    questions = request.questions
    checker.kinds(questions)
    checker.state(request.state)
    checker.keys(questions)
    checker.options(questions)
    checker.wording(questions)

    state = request.state
    response_warnings: tuple[Notice, ...] = ()
    over = checker.state_overflow(state, estimator)
    if over is not None:
        # A policy that returns at all is a named one: Fail always raises.
        state = checker.apply(overflow, state, over, estimator)
        response_warnings = (
            Notice(
                code=WARNING_OVERFLOW_POLICY,
                message=f"the state overflowed {over[0]}; policy {overflow.name!r} applied",
                field=over[0],
            ),
        )
        checker.state(state)
    checker.state_bytes(state)
    checker.question_tokens(state, questions, estimator)

    chunks = checker.pack(state, questions, estimator)
    unread = _unreadable_sizes(capabilities, state)
    return FitPlan(
        wire_requests=tuple(
            WireRequest(state=state, questions=chunk, options=request.options) for chunk in chunks
        ),
        outcome_warnings=tuple(_unchecked(capabilities, state, chunk, unread) for chunk in chunks),
        response_warnings=response_warnings,
    )


def _check_composite(  # pyright: ignore[reportUnusedFunction]
    request: Request, capabilities: Capabilities, *, routed_kinds: bool = False
) -> None:
    """Check ``request`` against a composite's derived record: kinds and modalities only.

    Every other field, the question count among them, is a member's: each member's share is
    checked against the member's own record. The derived record raises no warnings.
    ``routed_kinds`` is true for a router, whose kinds are those it routes, so a missing
    kind is one it routes to no member."""
    checker = _Checker(capabilities, routed_kinds=routed_kinds)
    checker.kinds(request.questions)
    checker.modalities(request.state)


def _merge(  # pyright: ignore[reportUnusedFunction]
    plan: Sequence[tuple[Outcome, Mapping[str, Answer] | None]],
) -> tuple[Mapping[str, Answer], tuple[Outcome, ...]]:
    """Merge the plan's outcomes, each with its answers or ``None`` when its wire request
    failed, into the answers map and the outcomes in plan order.

    A failed wire request's keys are absent from the answers.
    """
    answers: dict[str, Answer] = {}
    for _, received in plan:
        if received is not None:
            answers.update(received)
    return MappingProxyType(answers), tuple(outcome for outcome, _ in plan)


# ----------------------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------------------


class _Checker:
    """The checks against one record, each raising ``DoesNotFit`` with its off features.

    A missing kind or modality names only the off features the record says add it; every
    other check names every off feature. ``model`` is the leaf's model name, used only in
    the message that says no feature of the model adds a missing value."""

    def __init__(
        self, capabilities: Capabilities, model: str | None = None, *, routed_kinds: bool = False
    ) -> None:
        self.capabilities = capabilities
        self.model = model
        self.routed_kinds = routed_kinds
        self.off_features = capabilities.opt_in_features - capabilities.enabled_features
        self._state_tokens: int | None = None

    def fail(
        self, message: str, check: str, observed: object, limit: object, state: bool = False
    ) -> DoesNotFit:
        return DoesNotFit(
            message,
            check=check,
            observed=observed,
            limit=limit,
            state_overflow=state,
            off_features=self.off_features,
        )

    def limit(self, name: str, observed: int, what: str, state: bool = False) -> None:
        limit = _known(getattr(self.capabilities, name))
        if limit is not None and observed > limit:
            raise self.fail(f"{what}: {name} is {limit}", name, observed, limit, state)

    def kinds(self, questions: Sequence[Question]) -> None:
        asked = frozenset(question.kind for question in questions)
        self.declared("kinds", asked, self.capabilities.kinds, self.capabilities.feature_kinds)

    def modalities(self, state: State) -> None:
        used = frozenset(_modality(part) for part in state.parts)
        self.declared(
            "modalities", used, self.capabilities.modalities, self.capabilities.feature_modalities
        )

    def declared(
        self,
        check: str,
        asked: frozenset[str],
        declared: frozenset[str],
        adds: Mapping[str, frozenset[str]],
    ) -> None:
        """Raise when ``asked`` holds a value ``declared`` lacks, naming as off features only
        those the record says add a missing one.

        On a leaf, each value some feature adds has its own clause, and the values none adds
        share one, placed where the first of them falls. A composite has one clause."""
        missing = sorted(asked - declared)
        if not missing:
            return
        subject = "it" if len(missing) == 1 else "them"
        named: set[str] = set()
        clauses: list[str] = []
        if self.capabilities.composite and check == "kinds" and self.routed_kinds:
            clauses.append(f"this router routes {subject} to no member")
        elif self.capabilities.composite:
            clauses.append(
                f"no feature of this composite adds {subject}; its members' own records may"
            )
        else:
            unadded = [
                v for v in missing if not any(v in adds.get(f, ()) for f in self.off_features)
            ]
            for value in missing:
                features = sorted(f for f in self.off_features if value in adds.get(f, ()))
                named.update(features)
                which = "it" if len(missing) == 1 else value
                if len(features) == 1:
                    clauses.append(f"the {features[0]} feature adds {which}")
                elif features:
                    clauses.append(f"the {_joined(features, 'and')} features add {which}")
                elif value == unadded[0]:
                    which = "it" if len(missing) == 1 else _joined(unadded, "or")
                    model = "this model" if self.model is None else f"model {self.model}"
                    clauses.append(f"no feature of {model} adds {which}")
        raise DoesNotFit(
            f"{check} not supported: {', '.join(missing)}; {'; '.join(clauses)}",
            check=check,
            observed=frozenset(missing),
            limit=declared,
            off_features=frozenset(named),
        )

    def state(self, state: State) -> None:
        self.modalities(state)
        images = _image_count(state)
        if images and not self.capabilities.image_carriers:
            raise self.fail(
                f"the state has {images} images and the provider declares no image carrier",
                "image_carriers",
                images,
                self.capabilities.image_carriers,
            )
        self.limit("max_images", images, f"the state has {images} images")
        self.limit(
            "max_images_per_question", images, f"every question sees the state's {images} images"
        )
        if images:
            self.images(_images(state))

    def images(self, images: Sequence[ImagePart]) -> None:
        """Each image against the record's media types, byte caps and minimum side, the
        images counted from 0 among the state's images; an unknown field is not checked,
        and an image whose size cannot be read is not checked against the minimum side."""
        record = self.capabilities
        accepted = record.image_media_types
        if accepted is not None:
            for index, image in enumerate(images):
                if _image_media_type(image, accepted) is None:
                    read = (
                        _declared_media_type(image)
                        or _signature_media_type(image.data)
                        or "unknown"
                    )
                    allowed = ", ".join(sorted(accepted)) or "none"
                    raise self.fail(
                        f"image {index} is {read}: image_media_types is {allowed}",
                        "image_media_types",
                        read,
                        accepted,
                    )
        for index, image in enumerate(images):
            size = len(image.data)
            self.limit("max_image_bytes", size, f"image {index} is {size} bytes")
        total = sum(len(image.data) for image in images)
        self.limit("max_total_image_bytes", total, f"the state's images are {total} bytes")
        least = record.min_image_side
        if least:
            for index, image in enumerate(images):
                side = _image_size(image.data)
                if side is not None and min(side) < least:
                    width, height = side
                    raise self.fail(
                        f"image {index} is {width} by {height} px: min_image_side is {least}",
                        "min_image_side",
                        min(side),
                        least,
                    )

    def keys(self, questions: Sequence[Question]) -> None:
        patterns = [re.compile(pattern) for pattern in self.capabilities.key_patterns]
        failures: list[tuple[str, str]] = []
        for question in questions:
            for pattern in patterns:
                if pattern.fullmatch(question.key) is None:
                    failures.append((question.key, pattern.pattern))
                    break
        if failures:
            named = ", ".join(f"{key!r} does not match {pattern!r}" for key, pattern in failures)
            raise self.fail(
                f"keys do not match key_patterns: {named}",
                "key_patterns",
                tuple(key for key, _ in failures),
                self.capabilities.key_patterns,
            )

    def wording(self, questions: Sequence[Question]) -> None:
        """A ``noul`` with labels needs ``carries_noul_descriptions``, and an instruction
        that is not a string ``carries_structured_instructions``; every failing key is
        named."""
        record = self.capabilities
        if not record.carries_noul_descriptions:
            self._wording(
                questions,
                _describes_a_noul,
                "carries_noul_descriptions",
                "a noul's yes and no descriptions",
            )
        if not record.carries_structured_instructions:
            self._wording(
                questions,
                _is_structured,
                "carries_structured_instructions",
                "a structured instruction",
            )

    def _wording(
        self,
        questions: Sequence[Question],
        uses: Callable[[Question], bool],
        field: str,
        what: str,
    ) -> None:
        failing = tuple(question.key for question in questions if uses(question))
        if failing:
            named = ", ".join(repr(key) for key in failing)
            which, verb = ("question", "has") if len(failing) == 1 else ("questions", "have")
            raise self.fail(
                f"{which} {named} {verb} {what}: {field} is False", field, failing, False
            )

    def options(self, questions: Sequence[Question]) -> None:
        for question in questions:
            count = len(question.labels)
            if question.kind == "choice":
                self.limit("max_labels", count, f"question {question.key!r} has {count} labels")
            elif question.kind == "score":
                what = f"question {question.key!r} has {count} levels"
                self.limit("max_levels", count, what)
                least = self.capabilities.min_levels
                if least is not None and count < least:
                    raise self.fail(f"{what}: min_levels is {least}", "min_levels", count, least)

    def state_overflow(
        self, state: State, estimator: TokenEstimator
    ) -> tuple[str, int, int] | None:
        """The token limit the state alone is over, with the estimate and the limit, or
        ``None``. The tighter of the input and per-question limits is named first."""
        known = sorted(
            (limit, name)
            for name in _STATE_TOKEN_LIMITS
            if (limit := _known(getattr(self.capabilities, name))) is not None
        )
        if not known:
            return None
        self._state_tokens = _with_margin(estimator.estimate(state, ()))
        for limit, name in known:
            if self._state_tokens > limit:
                return name, self._state_tokens, limit
        return None

    def apply(
        self,
        overflow: OverflowPolicy,
        state: State,
        over: tuple[str, int, int],
        estimator: TokenEstimator,
    ) -> State:
        """Apply the policy once to a state over a token limit and check what it returns."""
        name, observed, limit = over
        try:
            reduced = overflow.apply(state, limit, estimator)
        except DoesNotFit as error:
            raise self.fail(error.message, name, observed, limit, state=True) from error
        still = self.state_overflow(reduced, estimator)
        if still is not None:
            name, observed, limit = still
            raise self.fail(
                f"the state the {overflow.name!r} policy returned is estimated at {observed} "
                f"tokens, over {name} {limit}",
                name,
                observed,
                limit,
                state=True,
            )
        return reduced

    def state_bytes(self, state: State) -> None:
        size = _state_bytes(state)
        self.limit("max_body_bytes", size, f"the state alone is {size} bytes", state=True)

    def question_tokens(
        self, state: State, questions: Sequence[Question], estimator: TokenEstimator
    ) -> None:
        if _known(self.capabilities.max_question_tokens) is None:
            return
        for question in questions:
            tokens = _with_margin(estimator.estimate(state, (question,)))
            self.limit(
                "max_question_tokens",
                tokens,
                f"question {question.key!r} with the state is estimated at {tokens} tokens",
            )

    def input_tokens(
        self, state: State, chunk: tuple[Question, ...], estimator: TokenEstimator
    ) -> int:
        tokens = _with_margin(estimator.estimate(state, chunk))
        if self.capabilities.recounts_state_per_question and len(chunk) > 1:
            assert self._state_tokens is not None, "state_overflow ran on this state"
            tokens += (len(chunk) - 1) * self._state_tokens
        return tokens

    def packable_failure(
        self, state: State, chunk: tuple[Question, ...], estimator: TokenEstimator
    ) -> DoesNotFit | None:
        """The first packable limit ``chunk`` with the state breaks, or ``None``."""
        for name in _PACKABLE:
            limit = _known(getattr(self.capabilities, name))
            if limit is None:
                continue
            if name == "max_questions":
                observed = len(chunk)
            elif name == "max_total_options":
                observed = sum(_options(question) for question in chunk)
            elif name == "max_body_bytes":
                observed = _state_bytes(state) + sum(_question_bytes(q) for q in chunk)
            else:
                observed = self.input_tokens(state, chunk, estimator)
            if observed > limit:
                keys = ", ".join(repr(question.key) for question in chunk)
                return self.fail(
                    f"question {keys} does not fit alone with the state: "
                    f"{name} {observed} is over {limit}",
                    name,
                    observed,
                    limit,
                )
        return None

    def pack(
        self, state: State, questions: Sequence[Question], estimator: TokenEstimator
    ) -> list[tuple[Question, ...]]:
        """One chunk of every question when the whole request fits with the state;
        otherwise the questions partitioned greedily, in order, into chunks that each fit.

        The whole request is estimated first, so one wire request is planned whenever it
        fits, however the estimator behaves on the prefixes."""
        whole = tuple(questions)
        if self.packable_failure(state, whole, estimator) is None:
            return [whole]
        chunks: list[tuple[Question, ...]] = []
        current: tuple[Question, ...] = ()
        for question in questions:
            if current and self.packable_failure(state, (*current, question), estimator) is None:
                current = (*current, question)
                continue
            if current:
                chunks.append(current)
            failure = self.packable_failure(state, (question,), estimator)
            if failure is not None:
                raise failure
            current = (question,)
        chunks.append(current)
        return chunks


def _unchecked(
    capabilities: Capabilities,
    state: State,
    chunk: tuple[Question, ...],
    unread: list[int] | None = None,
) -> tuple[Notice, ...]:
    """One ``unchecked-limit`` warning per unknown limit that bears on this wire request,
    and one naming the images whose size could not be read against a known minimum side.
    ``unread`` is those images' indexes when the caller has read them already."""
    kinds = {question.kind for question in chunk}
    images = _image_count(state) > 0
    bears = {
        "max_image_bytes": images,
        "max_total_image_bytes": images,
        "min_image_side": images,
        "image_media_types": images,
        "max_input_tokens": True,
        "max_question_tokens": True,
        "max_body_bytes": True,
        "max_questions": True,
        "max_images": images,
        "max_images_per_question": images,
        "max_labels": "choice" in kinds,
        "max_levels": "score" in kinds,
        "min_levels": "score" in kinds,
        "max_total_options": any(_options(question) for question in chunk),
    }
    warnings = [
        Notice(
            code=WARNING_UNCHECKED_LIMIT,
            message=f"{name} is unknown, so it was not checked",
            field=name,
        )
        for name in _WARNED
        if bears[name] and getattr(capabilities, name) is None
    ]
    if unread is None:
        unread = _unreadable_sizes(capabilities, state)
    if unread:
        warnings.append(
            Notice(
                code=WARNING_UNCHECKED_LIMIT,
                message=(
                    f"the {'size' if len(unread) == 1 else 'sizes'} of "
                    f"{'image' if len(unread) == 1 else 'images'} "
                    f"{_joined([str(i) for i in unread], 'and')} could not be read, "
                    "so min_image_side was not checked"
                ),
                field="min_image_side",
            )
        )
    return tuple(warnings)


def _unreadable_sizes(capabilities: Capabilities, state: State) -> list[int]:
    """The indexes, among the state's images, of those whose size cannot be read, when
    the record knows a minimum side above 0; none otherwise."""
    if not capabilities.min_image_side:
        return []
    return [i for i, image in enumerate(_images(state)) if _image_size(image.data) is None]


# ----------------------------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------------------------


def _joined(words: Sequence[str], conjunction: str) -> str:
    """``a``, ``a and b``, ``a, b and c``: no comma before the conjunction."""
    if len(words) == 1:
        return words[0]
    return f"{', '.join(words[:-1])} {conjunction} {words[-1]}"


def _known(limit: Limit) -> int | None:
    """The limit as a number to check against, or ``None`` when it is unknown or unbounded."""
    return limit if isinstance(limit, int) else None


def _with_margin(estimate: TokenEstimate) -> int:
    """``ceil(tokens * (1 + margin))``, the margin read as the decimal it prints as, so a
    margin of 0.1 on 100 tokens is 110, not 111."""
    return math.ceil(estimate.tokens * (1 + Fraction(repr(estimate.margin))))


def _modality(part: TextPart | JsonPart | ImagePart) -> Modality:
    if isinstance(part, TextPart):
        return "text"
    if isinstance(part, JsonPart):
        return "json"
    return "image"


def _image_count(state: State) -> int:
    return sum(isinstance(part, ImagePart) for part in state.parts)


def _images(state: State) -> list[ImagePart]:
    return [part for part in state.parts if isinstance(part, ImagePart)]


def _options(question: Question) -> int:
    """The options a question adds to a wire request's total: its labels or levels, two
    for a noul, none for an extraction or a box."""
    if question.kind in ("choice", "score"):
        return len(question.labels)
    return 2 if question.kind == "noul" else 0


def _json_default(value: object) -> dict[str, JsonValue]:
    if isinstance(value, Mapping):
        return dict(value)  # pyright: ignore[reportUnknownArgumentType]
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def _part_bytes(part: TextPart | JsonPart | ImagePart) -> int:
    """A part's size in Fit's measure: text and JSON as compact UTF-8, an image as the
    length of its base64 encoding."""
    if isinstance(part, TextPart):
        return len(part.text.encode())
    if isinstance(part, JsonPart):
        text = json.dumps(
            part.value, separators=(",", ":"), ensure_ascii=False, default=_json_default
        )
        return len(text.encode())
    return 4 * math.ceil(len(part.data) / 3)


def _state_bytes(state: State) -> int:
    return sum(_part_bytes(part) for part in state.parts)


def _describes_a_noul(question: Question) -> bool:
    return question.kind == "noul" and bool(question.labels)


def _is_structured(question: Question) -> bool:
    return not isinstance(question.instruction, str)


def _question_bytes(question: Question) -> int:
    """A question's size in Fit's measure: its key, instruction, label names and
    descriptions, in UTF-8; a structured instruction as its compact JSON."""
    instruction = question.instruction
    if not isinstance(instruction, str):
        instruction = json.dumps(
            instruction, separators=(",", ":"), ensure_ascii=False, default=_json_default
        )
    texts = [question.key, instruction]
    for label in question.labels:
        texts.append(label.name)
        if label.description is not None:
            texts.append(label.description)
    return sum(len(text.encode()) for text in texts)
