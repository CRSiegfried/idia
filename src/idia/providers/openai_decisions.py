"""The OpenAI decisions leaf: OpenAI's Decisions API through the ``openai`` SDK.

:class:`OpenAIDecisions` sends one wire request as one ``POST /v1/decisions``, read raw
through the SDK's ``with_raw_response`` with its retries at zero, and maps each answer to
its kind's shape before :func:`~idia.answers.validate_answers` runs over them with the
record's rounding. It is registered as ``openai-decisions``, installed by the ``openai``
extra.

A ``noul`` is sent as a ``predicate``, a ``choice`` with its label names as string values
and a ``score`` with its level names as labels. Each question is named ``q<i>`` in the
order asked, and the echoed name maps an answer back to its key. A ``refusal`` is that
question's :class:`~idia.core.Refusal`.
"""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from openai import AsyncOpenAI

from idia.answers import validate_answers
from idia.core import (
    WARNING_EXTRA_ANSWER_KEY,
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    Kind,
    LabelAnswer,
    MappedAnswer,
    Notice,
    NoulAnswer,
    ProviderError,
    Question,
    Refusal,
    Rounding,
    Rule,
    RuleFailure,
    State,
    TextPart,
    WireRequest,
    WireResponse,
    _declared_media_type,  # pyright: ignore[reportPrivateUsage]
    _image_media_type,  # pyright: ignore[reportPrivateUsage]
    unknown_options,
)
from idia.providers._openai import (
    OpenAIEstimator,
    OpenAILeafBase,
    compact,
    read_body,
    read_request_id,
    unreadable,
    usage,
)
from idia.providers.http import client_versions, wire_unreported

__all__ = [
    "OpenAIDecisions",
]

DEFAULT_MODEL: Final = "gpt-6-luna"

_WIRE_TYPE: Final[Mapping[Kind, str]] = {"noul": "predicate", "choice": "choice", "score": "score"}

# The estimator, set from the recorded usage: a request costs about 80 tokens, the state
# is counted once, and each question about 60 to 110 tokens beyond its text, a predicate
# the most. The figures cover every recorded request within the margin and overestimate a
# long request of predicates, many refused, about twofold. The image figure is unmeasured.
_OVERHEAD_TOKENS: Final = 100
_PER_QUESTION_TOKENS: Final = 90
_STATE_CHARS_PER_TOKEN: Final = 6
_QUESTION_CHARS_PER_TOKEN: Final = 4
_IMAGE_TOKENS: Final = 1_000
_MARGIN: Final = 0.1


def _record() -> Capabilities:
    """The record the decisions leaf declares."""
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score"}),
        modalities=frozenset({"text", "json", "image"}),
        image_carriers=frozenset({"parts"}),
        max_input_tokens=None,
        max_question_tokens=None,
        max_body_bytes=None,
        max_questions=200,
        max_images=128,
        max_images_per_question=128,
        max_image_bytes=None,
        max_total_image_bytes=None,
        max_labels=255,
        max_levels=None,
        max_total_options=None,
        min_levels=None,
        min_image_side=None,
        image_media_types=frozenset({"image/png", "image/jpeg"}),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=True,
        calibrated=False,
        snapshot_pinnable=False,
        carries_noul_descriptions=False,
        carries_structured_instructions=False,
        truncates_silently=None,
        confidence={"choice": "undocumented", "score": "undocumented"},
        rounding=Rounding(probability_decimals=2, score_decimals=2),
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
    )


class OpenAIDecisions(OpenAILeafBase):
    """A leaf for one model on OpenAI's Decisions API.

    ``api_key`` defaults to ``OPENAI_API_KEY``; with neither and no ``client`` building
    raises ``ValueError``. ``client`` is an injected ``AsyncOpenAI``, used with its retries
    at zero and never closed by the leaf. ``max_questions`` replaces the record's question
    cap, 200, for a caller who knows the wire takes more; ``capabilities`` replaces the
    whole record, and ``max_questions`` is then not read. Any model name is sent as given,
    but the record was taken from ``gpt-6-luna``'s documentation; a caller of another model
    passes its own.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        max_questions: int = 200,
        capabilities: Capabilities | None = None,
    ) -> None:
        if not model:
            raise ValueError("the decisions leaf needs a model")
        super().__init__(api_key=api_key, client=client, who="the decisions leaf")
        self._model = model
        if capabilities is None:
            record = _record()
            capabilities = (
                record
                if max_questions == record.max_questions
                else dataclasses.replace(record, max_questions=max_questions)
            )
        self._capabilities = capabilities
        self._estimator = OpenAIEstimator(
            overhead=_OVERHEAD_TOKENS,
            per_question=_PER_QUESTION_TOKENS,
            state_chars=_STATE_CHARS_PER_TOKEN,
            question_chars=_QUESTION_CHARS_PER_TOKEN,
            image_tokens=_IMAGE_TOKENS,
            margin=_MARGIN,
        )

    @property
    def name(self) -> str:
        return "openai-decisions"

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def estimator(self) -> OpenAIEstimator:
        return self._estimator

    def __repr__(self) -> str:
        return f"OpenAIDecisions(model={self._model!r})"

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Send the wire request once and return its answers, validated, with provenance."""
        body, sizes = _body(self._model, wire_request, self._capabilities.image_media_types)
        response = await self._send(
            deadline,
            lambda: self._client.decisions.with_raw_response.create(
                model=self._model,
                input=cast(Any, body["input"]),
                questions=cast(Any, body["questions"]),
            ),
        )
        read = read_body(response)
        answers_json = read.get("answers")
        if not isinstance(answers_json, tuple):
            raise unreadable(response, "the body holds no answers list")
        model_reported = read.get("model")
        raw_usage = read.get("usage")
        if not isinstance(model_reported, str) or not model_reported:
            raise unreadable(response, "the body fails the top-level schema")
        if not isinstance(raw_usage, Mapping):
            raise unreadable(response, "the body fails the top-level schema")
        mapped, dropped = _map_answers(wire_request.questions, answers_json)
        answers, warnings = validate_answers(
            wire_request.questions, mapped, self._capabilities.rounding
        )
        warnings = (
            *dropped,
            *warnings,
            Notice(
                code=WARNING_NO_SNAPSHOT,
                message=f"the wire reported {model_reported!r}, an alias with no snapshot; "
                "drift behind it cannot be detected from the response",
                field="snapshot",
            ),
        )
        normalized = usage(
            cast(Mapping[str, JsonValue], raw_usage), "input_tokens", "output_tokens"
        )
        request_id = read_request_id(response)
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self._capabilities.calibrated,
            model_reported=model_reported,
            snapshot=None,
            prompt_sent=_prompt_sent(body, sizes),
            usage=normalized,
            request_id=request_id,
            raw_body=read,
            warnings=warnings + unknown_options(wire_request.options, frozenset()),
            client_versions=client_versions("openai", "httpx2"),
            unreported=wire_unreported(snapshot=None, usage=normalized, request_id=request_id),
        )


# ----------------------------------------------------------------------------------------
# The wire body
# ----------------------------------------------------------------------------------------


def _name(index: int) -> str:
    return f"q{index}"


def _media_type(image: ImagePart, accepted: frozenset[str] | None) -> str:
    """The type Fit judged, PNG or JPEG. An image Fit would refuse, in a call made past
    Fit, goes with its declared type read as Fit reads it, else ``application/octet-stream``, for
    the wire to refuse."""
    judged = _image_media_type(image, accepted)
    if judged is not None:
        return judged
    return _declared_media_type(image) or "application/octet-stream"


def _parts(
    state: State, accepted: frozenset[str] | None
) -> tuple[list[dict[str, JsonValue]], dict[int, int]]:
    """The state's parts as content parts, and each image part's size by its position."""
    parts: list[dict[str, JsonValue]] = []
    sizes: dict[int, int] = {}
    for part in state.parts:
        if isinstance(part, TextPart):
            parts.append({"type": "input_text", "text": part.text})
        elif isinstance(part, JsonPart):
            parts.append({"type": "input_text", "text": compact(part.value)})
        else:
            data = base64.b64encode(part.data).decode("ascii")
            sizes[len(parts)] = len(part.data)
            image_url = f"data:{_media_type(part, accepted)};base64,{data}"
            parts.append({"type": "input_image", "image_url": image_url})
    return parts, sizes


def _question(index: int, question: Question) -> dict[str, JsonValue]:
    if question.kind not in _WIRE_TYPE:
        raise ProviderError(f"the decisions leaf does not answer {question.kind!r} questions")
    if not isinstance(question.instruction, str):
        raise ProviderError(
            f"the decisions leaf sends an instruction as text only: question {question.key!r} "
            "has a structured instruction"
        )
    if question.kind == "noul" and question.labels:
        raise ProviderError(
            f"the decisions leaf cannot send a noul's yes and no descriptions, which a "
            f"predicate has no field for: question {question.key!r}"
        )
    wire: dict[str, JsonValue] = {
        "type": _WIRE_TYPE[question.kind],
        "name": _name(index),
        "instructions": question.instruction,
    }
    if question.kind == "choice":
        wire["choices"] = tuple(
            {"value": label.name}
            if label.description is None
            else {"value": label.name, "description": label.description}
            for label in question.labels
        )
    elif question.kind == "score":
        wire["levels"] = tuple(
            {"label": label.name}
            if label.description is None
            else {"label": label.name, "description": label.description}
            for label in question.labels
        )
    return wire


def _body(
    model: str, wire_request: WireRequest, accepted: frozenset[str] | None
) -> tuple[dict[str, JsonValue], dict[int, int]]:
    """The request body, keys in a fixed order, and each image part's size in bytes."""
    parts, sizes = _parts(wire_request.state, accepted)
    body: dict[str, JsonValue] = {
        "model": model,
        "input": ({"role": "user", "content": tuple(parts)},),
        "questions": tuple(_question(i, q) for i, q in enumerate(wire_request.questions)),
    }
    return body, sizes


def _prompt_sent(body: Mapping[str, JsonValue], sizes: Mapping[int, int]) -> dict[str, JsonValue]:
    """The body sent, each image's data URL replaced by its media type and size."""
    if not sizes:
        return dict(body)
    [message] = cast(tuple[Mapping[str, JsonValue]], body["input"])
    content = list(cast(tuple[Mapping[str, JsonValue], ...], message["content"]))
    for position, size in sizes.items():
        url = cast(str, content[position]["image_url"])
        media_type = url[len("data:") : url.index(";")]
        content[position] = {"type": "input_image", "media_type": media_type, "bytes": size}
    return {**body, "input": ({"role": "user", "content": tuple(content)},)}


# ----------------------------------------------------------------------------------------
# Reading the answers
# ----------------------------------------------------------------------------------------


class _Shape(Exception):
    """An answer that is not its kind's wire shape; becomes that question's fault."""

    def __init__(self, observed: str) -> None:
        super().__init__(observed)
        self.observed = observed


def _number(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Shape(type(value).__name__)
    try:
        return float(value)
    except OverflowError:
        raise _Shape("int beyond float") from None


def _optional_number(value: JsonValue) -> float | None:
    return None if value is None else _number(value)


def _object(value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise _Shape(type(value).__name__)
    return cast(Mapping[str, JsonValue], value)


def _list(value: JsonValue) -> tuple[JsonValue, ...]:
    if not isinstance(value, tuple):
        raise _Shape(type(value).__name__)
    return value


def _map_answers(
    questions: Sequence[Question], answers: Sequence[JsonValue]
) -> tuple[dict[str, MappedAnswer], tuple[Notice, ...]]:
    """Each wire answer mapped to its kind's shape, keyed by the question its echoed name
    was assigned to.

    An answer whose name was not assigned, or that has none, is dropped with an
    ``extra-answer-key`` warning, and the question it may have been meant for faults as
    missing in validation: nothing is matched by position. A second answer echoing a name
    already seen is dropped with the same warning; the first stands.
    """
    by_name = {_name(i): question for i, question in enumerate(questions)}
    mapped: dict[str, MappedAnswer] = {}
    dropped: list[Notice] = []
    for position, answer in enumerate(answers):
        name = (
            cast(Mapping[str, JsonValue], answer).get("name")
            if isinstance(answer, Mapping)
            else None
        )
        question = by_name.get(name) if isinstance(name, str) else None
        if question is not None and question.key in mapped:
            dropped.append(
                Notice(
                    code=WARNING_EXTRA_ANSWER_KEY,
                    message=f"answer {position} repeats the name {name!r}; dropped",
                    field=question.key,
                )
            )
            continue
        if question is None:
            dropped.append(
                Notice(
                    code=WARNING_EXTRA_ANSWER_KEY,
                    message=f"answer {position} is named {name!r}, no question's name; dropped",
                    field=name if isinstance(name, str) else None,
                )
            )
            continue
        try:
            mapped[question.key] = _map(question, _object(answer))
        except _Shape as shape:
            failure = RuleFailure(rule=Rule.kind_shape, observed=shape.observed, bound=None)
            mapped[question.key] = Fault(key=question.key, failures=(failure,), mapped=None)
    return mapped, tuple(dropped)


def _map(question: Question, answer: Mapping[str, JsonValue]) -> MappedAnswer:
    wire_type = answer.get("type")
    if wire_type == "refusal":
        return Refusal()
    if wire_type != _WIRE_TYPE[question.kind]:
        raise _Shape(str(wire_type))
    if question.kind == "noul":
        return NoulAnswer(probability=_number(answer.get("probability")))
    confidence = _optional_number(answer.get("confidence"))
    entries = [_object(entry) for entry in _list(answer.get("probabilities"))]
    if question.kind == "choice":
        chosen = answer.get("choice")
        values = [chosen, *(entry.get("value") for entry in entries)]
        odd = [value for value in values if not isinstance(value, str)]
        if odd:
            raise _Shape(type(odd[0]).__name__)
        probabilities = {
            cast(str, entry.get("value")): _number(entry.get("probability")) for entry in entries
        }
        return LabelAnswer(
            probabilities=_in_label_order(question, probabilities),
            confidence=confidence,
            chosen=cast(str, chosen),
        )
    levels: list[tuple[int, str, float]] = []
    for entry in entries:
        index = entry.get("value")
        label = entry.get("label")
        if isinstance(index, bool) or not isinstance(index, int):
            raise _Shape(type(index).__name__)
        if not isinstance(label, str):
            raise _Shape(type(label).__name__)
        levels.append((index, label, _number(entry.get("probability"))))
    levels.sort(key=lambda level: level[0])
    names = [label.name for label in question.labels]
    return LabelAnswer(
        probabilities=tuple(
            (names[index] if 0 <= index < len(names) else str(index), probability)
            for index, _, probability in levels
        ),
        confidence=confidence,
        expected_level=_number(answer.get("score")),
        echo=tuple(label for _, label, _ in levels),
        echo_of="names",
    )


def _in_label_order(
    question: Question, probabilities: Mapping[str, float]
) -> tuple[tuple[str, float], ...]:
    """The wire's probabilities in the order asked, then any the wire added; a label the
    wire left out stays out, for validation to fault."""
    names = [label.name for label in question.labels]
    asked = [(name, probabilities[name]) for name in names if name in probabilities]
    added = [(name, p) for name, p in probabilities.items() if name not in names]
    return tuple(asked + added)
