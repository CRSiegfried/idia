"""The System One leaf: TypeSafe's decision API for one Jev model, over raw HTTP.

:class:`SystemOne` sends one wire request as one ``POST /v1/systemone`` and maps each
answer to its kind's shape before :func:`~idia.answers.validate_answers` runs over
them with the record's rounding. It is registered as ``system-one``, installed by the
``system-one`` extra.

The record depends on the model and on the two opt-in features. Every model takes
``noul``, ``choice`` and ``score`` over text and JSON. ``images`` adds the image modality,
and on ``jev-1.14.0`` and ``jev-preview`` the ``box`` kind with it; ``long_context`` raises
the per-request token limit.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import httpx2

from idia.answers import validate_answers
from idia.core import (
    UNBOUNDED,
    WARNING_NO_SNAPSHOT,
    Box,
    BoxAnswer,
    Capabilities,
    Deadline,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    Kind,
    LabelAnswer,
    MappedAnswer,
    NoulAnswer,
    Part,
    ProviderError,
    Question,
    Rounding,
    Rule,
    RuleFailure,
    State,
    TextPart,
    TokenEstimate,
    Usage,
    Warning,
    WireRequest,
    WireResponse,
    _image_media_type,  # pyright: ignore[reportPrivateUsage]
    unknown_options,
)
from idia.providers.http import HttpLeaf

DEFAULT_BASE_URL: Final = "https://api.typesafe.ai"
_PATH: Final = "/v1/systemone"
_REQUEST_ID_HEADER: Final = "x-typesafe-request-id"
_IMAGES_HEADER: Final = "TypeSafe-Danger-Zone-Images"
_LONG_CONTEXT_HEADER: Final = "TypeSafe-Danger-Zone-Long-Context"
_API_KEY_ENV: Final = "TYPESAFE_API_KEY"
_SNAPSHOT: Final = re.compile(r"jev-[0-9]+\.[0-9]+\.[0-9]+")

_BASE_KINDS: Final[frozenset[Kind]] = frozenset({"noul", "choice", "score"})


@dataclass(frozen=True, slots=True, kw_only=True)
class _Model:
    """What one model alias or version offers."""

    box: bool


SystemOneModel: TypeAlias = Literal["jev-1.13.0", "jev-latest", "jev-1.14.0", "jev-preview"]
"""The models System One knows, as ``provider("system-one", model=...)`` types them."""

_MODELS: Final[Mapping[str, _Model]] = {
    "jev-1.13.0": _Model(box=False),
    "jev-latest": _Model(box=False),
    "jev-1.14.0": _Model(box=True),
    "jev-preview": _Model(box=True),
}

# The estimator: Pydantic AI's fit for Jev, characters of JSON per token. On the recorded
# text requests usage ran 3% under to 8% over it, within the margin. A 320 by 320 PNG cost
# 302 tokens against the same request without it; how the cost grows with size is unknown.
_STATE_CHARS_PER_TOKEN: Final = 6
_QUESTION_CHARS_PER_TOKEN: Final = 4
_OVERHEAD_TOKENS: Final = 260
_IMAGE_TOKENS: Final = 320
_MARGIN: Final = 0.1

# The smallest side the wire takes, measured on jev-1.14.0 with PNGs: 280 by 280 and 280 by
# 320 either way were answered, 279 by 279 and 279 by 320 either way refused with a 400.
# The rule is per side, not by area. JPEG was not measured.
# The record declares it, and Fit refuses a smaller image before any call.
_MIN_IMAGE_SIDE: Final = 280


def _record(model: str, *, long_context: bool, images: bool) -> Capabilities:
    """The record a System One leaf declares for ``model`` with the features given.

    What each feature adds is the same whatever is enabled: ``images`` adds the image
    modality on every model and ``box`` on a model that offers it; ``long_context`` adds no
    kind or modality."""
    box = images and _MODELS[model].box
    kinds = _BASE_KINDS | frozenset[Kind]({"box"}) if box else _BASE_KINDS
    confidence: dict[Kind, str] = {
        "choice": "normalized_top_probability",
        "score": "mad_concentration",
    }
    if box:
        confidence["box"] = "undocumented"
    enabled = {"long_context"} if long_context else set[str]()
    if images:
        enabled.add("images")
    return Capabilities(
        kinds=kinds,
        modalities=frozenset({"text", "json", "image"} if images else {"text", "json"}),
        image_carriers=frozenset({"placeholder"} if images else ()),
        max_input_tokens=262_000 if long_context else 64_000,
        max_question_tokens=32_000,
        max_body_bytes=33_554_432,
        max_questions=UNBOUNDED,
        max_images=UNBOUNDED if images else 0,
        max_images_per_question=UNBOUNDED if images else 0,
        max_image_bytes=16 * 1024 * 1024,
        max_total_image_bytes=32 * 1024 * 1024,
        max_labels=255,
        max_levels=10,
        max_total_options=UNBOUNDED,
        min_levels=1,
        min_image_side=_MIN_IMAGE_SIDE,
        image_media_types=frozenset({"image/png", "image/jpeg"}),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=False,
        calibrated=True,
        snapshot_pinnable=True,
        truncates_silently=False,
        confidence=confidence,
        rounding=Rounding(probability_decimals=2, score_decimals=2),
        opt_in_features=frozenset({"long_context", "images"}),
        enabled_features=frozenset(enabled),
        feature_kinds={"images": frozenset[Kind]({"box"})} if _MODELS[model].box else {},
        feature_modalities={"images": frozenset({"image"})},
    )


class SystemOneEstimator:
    """Pydantic AI's heuristic for Jev: the state's JSON at 6 characters a token, the
    questions' at 4, 260 tokens of overhead, and a fixed figure per image, with a margin.

    ``image_tokens`` is the figure per image; ``margin`` the fraction preflight adds.
    The JSON is measured as ``json.dumps`` writes it, escaping non-ASCII, while the body
    sends it raw, so text outside ASCII is over-counted, on the safe side. The image figure
    was measured at one size, 320 by 320 pixels.
    """

    def __init__(self, *, image_tokens: int = _IMAGE_TOKENS, margin: float = _MARGIN) -> None:
        self.image_tokens = image_tokens
        self.margin = margin

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        wire_state, assets = _state(state)
        tokens = _OVERHEAD_TOKENS + math.ceil(
            len(json.dumps(wire_state, default=_plain)) / _STATE_CHARS_PER_TOKEN
        )
        if questions:
            tokens += math.ceil(
                len(json.dumps(_questions(questions), default=_plain)) / _QUESTION_CHARS_PER_TOKEN
            )
        tokens += len(assets) * self.image_tokens
        return TokenEstimate(tokens=tokens, margin=self.margin)


class SystemOne(HttpLeaf):
    """A leaf for one Jev model on TypeSafe's System One API.

    ``model`` is a pinned version or an alias the leaf knows; any other is refused here.
    ``api_key`` defaults to the ``TYPESAFE_API_KEY`` environment variable, and building
    without either raises ``ValueError``. ``long_context`` and ``images`` turn on the opt-in
    features, each sent as its header. ``capabilities`` replaces the declared record, for a
    caller who knows better; it must agree with the feature flags, which alone decide the
    headers sent.
    ``client`` is an injected ``httpx2.AsyncClient``, which the leaf never closes.
    """

    def __init__(
        self,
        *,
        model: str = "jev-1.13.0",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx2.AsyncClient | None = None,
        long_context: bool = False,
        images: bool = False,
        capabilities: Capabilities | None = None,
    ) -> None:
        if model not in _MODELS:
            known = ", ".join(sorted(_MODELS))
            raise ValueError(f"System One does not know the model {model!r}; known: {known}")
        key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV)
        if not key:
            raise ValueError(f"System One needs an api_key or the {_API_KEY_ENV} variable")
        super().__init__(base_url=base_url, request_id_header=_REQUEST_ID_HEADER, client=client)
        self._model = model
        self._key = key
        self._capabilities = (
            capabilities
            if capabilities is not None
            else _record(model, long_context=long_context, images=images)
        )
        self._estimator = SystemOneEstimator()
        self._headers = {"Authorization": f"Bearer {key}"}
        if long_context:
            self._headers[_LONG_CONTEXT_HEADER] = "true"
        if images:
            self._headers[_IMAGES_HEADER] = "true"

    @property
    def name(self) -> str:
        return "system-one"

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def estimator(self) -> SystemOneEstimator:
        return self._estimator

    def __repr__(self) -> str:
        return f"SystemOne(model={self._model!r})"

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Send the wire request once and return its answers, validated, with provenance."""
        _one_image_for_box(wire_request)
        body, sizes = _body(self._model, wire_request, self._capabilities.image_media_types)
        try:
            content = json.dumps(
                body, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_plain
            ).encode()
        except (ValueError, TypeError) as error:
            raise ProviderError(f"the state cannot be sent as JSON: {error}") from None
        response, read = await self._post(_PATH, content, deadline, headers=self._headers)
        answers_json, model_reported, usage = self._top_level(response, read)
        mapped = _map_answers(wire_request.questions, answers_json)
        answers, warnings = validate_answers(
            wire_request.questions, mapped, self._capabilities.rounding
        )
        snapshot = model_reported if _SNAPSHOT.fullmatch(model_reported) else None
        if snapshot is None:
            warnings += (
                Warning(
                    code=WARNING_NO_SNAPSHOT,
                    message=f"the wire reported {model_reported!r}, not a pinned version",
                    field="snapshot",
                ),
            )
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self._capabilities.calibrated,
            model_reported=model_reported,
            snapshot=snapshot,
            prompt_sent=_prompt_sent(body, sizes),
            usage=usage,
            request_id=self.request_id(response),
            raw_body=read,
            warnings=warnings + unknown_options(wire_request.options, frozenset()),
        )

    def _top_level(
        self, response: httpx2.Response, read: Mapping[str, JsonValue]
    ) -> tuple[Mapping[str, JsonValue], str, Usage]:
        """The answers, the reported model and the usage; a body without an answers object
        holds no answers, and one without the other two fails the top-level schema: either
        cannot be read."""
        answers = read.get("answers")
        if not isinstance(answers, Mapping):
            raise self._unreadable(response, "the body holds no answers map")
        model = read.get("model")
        usage = read.get("usage")
        if not isinstance(model, str) or not model or not isinstance(usage, Mapping):
            raise self._unreadable(response, "the body fails the top-level schema")
        raw_usage = cast(Mapping[str, JsonValue], usage)
        return (
            cast(Mapping[str, JsonValue], answers),
            model,
            Usage(
                input_tokens=_count(raw_usage.get("input_tokens")),
                output_tokens=_count(raw_usage.get("output_tokens")),
                raw_usage=raw_usage,
            ),
        )


# ----------------------------------------------------------------------------------------
# The wire body
# ----------------------------------------------------------------------------------------


def _plain(value: object) -> dict[str, JsonValue]:
    """A read-only mapping in a caller's JSON part, as a dict ``json`` can encode."""
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, JsonValue], value))
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def _placeholder(n: int) -> str:
    return f"<image-{n}>"


def _state(state: State) -> tuple[JsonValue, dict[str, ImagePart]]:
    """The state as the wire takes it, each image replaced by its placeholder, and the
    images by placeholder. One part is its own value; several are an array in order."""
    assets: dict[str, ImagePart] = {}
    values: list[JsonValue] = []
    for part in state.parts:
        values.append(_part(part, assets))
    return (values[0] if len(values) == 1 else tuple(values)), assets


def _part(part: Part, assets: dict[str, ImagePart]) -> JsonValue:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, JsonPart):
        return part.value
    placeholder = _placeholder(len(assets))
    assets[placeholder] = part
    return placeholder


def _question(question: Question) -> dict[str, JsonValue]:
    wire: dict[str, JsonValue] = {"type": question.kind, "instructions": question.instruction}
    if question.kind == "choice":
        wire["criteria"] = {label.name: label.description for label in question.labels}
    elif question.kind == "score":
        wire["criteria"] = tuple(
            label.description if label.description is not None else label.name
            for label in question.labels
        )
    return wire


def _questions(questions: Sequence[Question]) -> dict[str, JsonValue]:
    return {question.key: _question(question) for question in questions}


def _one_image_for_box(wire_request: WireRequest) -> None:
    """A ``box`` answers about one image, so its state may hold no more than one."""
    images = sum(isinstance(part, ImagePart) for part in wire_request.state.parts)
    if images > 1 and any(q.kind == "box" for q in wire_request.questions):
        raise ProviderError(
            f"a box question needs a state with at most one image; this one has {images}"
        )


def _image_format(image: ImagePart, accepted: frozenset[str] | None) -> str:
    """The wire's ``format``: the subtype of the type Fit judged, ``png`` or ``jpeg``. An
    image Fit would refuse, in a call made past Fit, goes with the subtype of the type it
    reads as, or none, for the wire to refuse."""
    media_type = _image_media_type(image, accepted) or _image_media_type(image, None) or ""
    return media_type.removeprefix("image/")


def _body(
    model: str, wire_request: WireRequest, accepted: frozenset[str] | None
) -> tuple[dict[str, JsonValue], dict[str, int]]:
    """The request body, keys in a fixed order, and each asset's size in bytes."""
    state, images = _state(wire_request.state)
    body: dict[str, JsonValue] = {
        "model": model,
        "state": state,
        "questions": _questions(wire_request.questions),
    }
    if images:
        body["assets"] = {
            placeholder: {
                "type": "image",
                "format": _image_format(image, accepted),
                "data_base64": base64.b64encode(image.data).decode("ascii"),
            }
            for placeholder, image in images.items()
        }
    return body, {placeholder: len(image.data) for placeholder, image in images.items()}


def _prompt_sent(body: Mapping[str, JsonValue], sizes: Mapping[str, int]) -> dict[str, JsonValue]:
    """The body sent, each asset's base64 replaced by its size in bytes."""
    sent = dict(body)
    assets = cast(Mapping[str, Mapping[str, JsonValue]], body.get("assets", {}))
    if assets:
        sent["assets"] = {
            placeholder: {
                **{k: v for k, v in asset.items() if k != "data_base64"},
                "bytes": sizes[placeholder],
            }
            for placeholder, asset in assets.items()
        }
    return sent


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


def _count(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _map_answers(
    questions: Sequence[Question], answers: Mapping[str, JsonValue]
) -> dict[str, MappedAnswer]:
    """Each wire answer mapped to its kind's shape, keyed as the wire keyed it.

    A key with no question is kept so that validation drops it with a warning; an answer
    whose wire type or shape is not its question's becomes that question's fault.
    """
    by_key = {question.key: question for question in questions}
    mapped: dict[str, MappedAnswer] = {}
    for key, answer in answers.items():
        question = by_key.get(key)
        try:
            if question is None:
                raise _Shape("no question")
            mapped[key] = _map(question, _object(answer))
        except _Shape as shape:
            failure = RuleFailure(rule=Rule.kind_shape, observed=shape.observed, bound=None)
            mapped[key] = Fault(key=key, failures=(failure,), mapped=None)
    return mapped


def _map(question: Question, answer: Mapping[str, JsonValue]) -> MappedAnswer:
    wire_type = answer.get("type")
    if wire_type != question.kind:
        raise _Shape(str(wire_type))
    if question.kind == "noul":
        return NoulAnswer(probability=_number(answer.get("noul")))
    if question.kind == "box":
        box = _object(answer.get("box"))
        region = Box(
            top=_number(box.get("top")),
            left=_number(box.get("left")),
            bottom=_number(box.get("bottom")),
            right=_number(box.get("right")),
        )
        return BoxAnswer(region=region, confidence=_optional_number(answer.get("confidence")))
    probabilities = _object(answer.get("probabilities"))
    confidence = _optional_number(answer.get("confidence"))
    if question.kind == "choice":
        chosen = answer.get("choice")
        if not isinstance(chosen, str):
            raise _Shape(type(chosen).__name__)
        return LabelAnswer(
            probabilities=_in_label_order(question, probabilities, by_index=False),
            confidence=confidence,
            chosen=chosen,
        )
    legend = _object(answer.get("legend"))
    echo: list[str] = []
    for index in sorted(legend, key=_level_order):
        description = legend[index]
        if not isinstance(description, str):
            raise _Shape(type(description).__name__)
        echo.append(description)
    return LabelAnswer(
        probabilities=_in_label_order(question, probabilities, by_index=True),
        confidence=confidence,
        expected_level=_number(answer.get("score")),
        echo=tuple(echo),
        echo_of="descriptions",
    )


def _level_order(index: str) -> tuple[int, str]:
    """A legend's levels by index; a key that is no index, or too long to be one, sorts
    last, by its text."""
    return (
        (int(index), "")
        if index.isascii() and index.isdecimal() and len(index) <= 9
        else (1 << 30, index)
    )


def _in_label_order(
    question: Question, probabilities: Mapping[str, JsonValue], *, by_index: bool
) -> tuple[tuple[str, float], ...]:
    """The wire's probabilities named by label, in the order asked, then any the wire
    added under their own key. A label the wire left out stays out, for validation to
    fault; a score's level is keyed by its index as a string."""
    names = {
        (str(index) if by_index else label.name): label.name
        for index, label in enumerate(question.labels)
    }
    asked = [(names[key], _number(probabilities[key])) for key in names if key in probabilities]
    added = [(key, _number(value)) for key, value in probabilities.items() if key not in names]
    return tuple(asked + added)
