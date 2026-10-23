"""The Cloudflare leaves: Clef for the label kinds and Clairic for extraction, over raw HTTP.

Both models run on Workers AI's REST route,
``POST /client/v4/accounts/<account>/ai/run/@cf/cloudflare/<model>``, with a Bearer token,
which wraps every body in Cloudflare's envelope ``{success, errors, messages, result}``.
:class:`Clef` sends ``noul``, ``choice`` and ``score`` in the System One field names with
images in a top-level array; :class:`Clairic` sends ``extraction`` questions as typed
``fields``. Each maps the wire's answers to their kinds' shapes before
:func:`~idia.answers.validate_answers` runs over them with the record's rounding. They are
registered as ``cloudflare-clef`` and ``cloudflare-clairic``, installed by the
``cloudflare`` extra; ``idia.cloudflare.cloudflare`` routes between them over one client.

Neither model has a pinned version: the wire reports the short name, so every response
carries a ``no-snapshot`` warning saying that alias drift cannot be detected from it.
"""

from __future__ import annotations

import base64
import datetime
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, cast

import httpx2

from idia.answers import validate_answers
from idia.core import (
    UNBOUNDED,
    WARNING_EXTRA_ANSWER_KEY,
    WARNING_NO_SNAPSHOT,
    WARNING_PROVIDER_TRUNCATED,
    Capabilities,
    Deadline,
    ExtractionAnswer,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    LabelAnswer,
    MappedAnswer,
    NoulAnswer,
    ProviderError,
    Question,
    Rounding,
    Rule,
    RuleFailure,
    Span,
    State,
    TextPart,
    TokenEstimate,
    Usage,
    Value,
    Warning,
    WireRequest,
    WireResponse,
    _image_media_type,  # pyright: ignore[reportPrivateUsage]
    _without_keys,  # pyright: ignore[reportPrivateUsage]
    unknown_options,
)
from idia.providers.http import HttpLeaf

_BASE_URL: Final = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run"
_CLEF_PATH: Final = "/@cf/cloudflare/{model}"
_CLAIRIC_PATH: Final = "/@cf/cloudflare/clairic"
_REQUEST_ID_HEADER: Final = "cf-ai-req-id"
_ACCOUNT_ENV: Final = "CLOUDFLARE_ACCOUNT_ID"
_TOKEN_ENV: Final = "CLOUDFLARE_API_TOKEN"

_QUOTA_CODES: Final = frozenset({3036, 4006})
"""Workers AI error codes that mark a quota spent, not a rate, which waiting will not cure:
the errors page names 3036 for the daily neuron allowance, and the wire answered 4006 when
the free allocation ran out."""
_MESSAGE_CHARS: Final = 300

_CLEF_MODELS: Final[Mapping[str, int]] = {"clef": 65_536, "clef-flash": 24_000}
"""Each Clef model and its hosted context in tokens; clef-flash's was cut on 2026-10-09."""
_CATALOG: Final = "@cf/cloudflare/"
"""The catalog prefix of each model's full name, ``@cf/cloudflare/<model>``."""
ClefModel = Literal["clef", "clef-flash", "@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"]
"""Each name a Clef leaf takes: a model's short name or its full name."""
_CLEF_NAMES: Final[Mapping[str, str]] = {
    **{model: model for model in _CLEF_MODELS},
    **{f"{_CATALOG}{model}": model for model in _CLEF_MODELS},
}
"""Each name a Clef leaf takes, short or full, and the short name it reports."""
_CLAIRIC_MODEL: Final = "clairic"
_CLAIRIC_NAMES: Final = frozenset({_CLAIRIC_MODEL, f"{_CATALOG}{_CLAIRIC_MODEL}"})
"""The two names Clairic takes, short and full; it reports the short one."""
_REFERENCE_DATE: Final = "reference_date"
_CLEF_OPTIONS: Final[frozenset[str]] = frozenset({_REFERENCE_DATE})
"""Clef reads no option. It knows Clairic's ``reference_date`` and neither sends it nor
warns of it, since the ``cloudflare`` router hands both leaves one request's options whole."""
_CLAIRIC_OPTIONS: Final = frozenset({_REFERENCE_DATE})
"""The one request option Clairic reads. Either leaf warns of any option outside its set
and ignores it."""

_CLEF_ID: Final = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_CLAIRIC_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{0,63}")
"""The wire's documented constraints on a Clef question id and a Clairic field name, which
the wire enforces: Clef (``clef``) refused the question id ``match:intent`` with 422 and
code 3030 (the cassette ``question-id-probe``), and Clairic the field name ``1bad`` with
422 and code 3030 (the cassette ``field-name-probe``).
A wire request whose keys all match is sent under them; one with any key that does not is
sent under ``q0..qN`` or ``f0..fN`` for every question, by position, and its answers are
mapped back to the caller's keys. The records declare no key pattern."""

_MAX_IMAGES: Final = 4
_IMAGE_BYTES: Final = 4 * 1024 * 1024
_IMAGES_BYTES: Final = 8 * 1024 * 1024
_LIST_ITEMS: Final = frozenset({"text", "number", "date"})
"""The item types a Clairic list takes."""
_CANDIDATE_TYPES: Final = frozenset({"estimate", "forecast"})
_DISTRIBUTION_KEYS: Final = frozenset(
    {"mean", "median", "p10", "p90", "p_below", "p_above", "quantiles", "steps"}
)

# The estimators, fitted to the recorded usage. Clef counted a 5,941-character state at
# about 4 characters a token and the state once however many questions; an image costs
# (w/32)*(h/32)+3 tokens capped at 1,024, and an ImagePart carries no size, so every image
# is costed at the cap. Clairic counted 133 to 148 schema tokens for one field and 76 state
# tokens for 296 characters (43 for 79 digit-heavy ones); it may split the fields into
# passes that each see the whole input, so the estimate is of one pass holding every field,
# which is what the context limit bounds.
_CLEF_OVERHEAD: Final = 260
_CLEF_STATE_CHARS: Final = 4
_CLEF_QUESTION_CHARS: Final = 4
_CLAIRIC_OVERHEAD: Final = 0
_CLAIRIC_STATE_CHARS: Final = 3
_CLAIRIC_FIELD_TOKENS: Final = 150
_IMAGE_TOKENS: Final = 1_024
_MARGIN: Final = 0.1


def _clef_record(model: str) -> Capabilities:
    """The record a Clef leaf declares for ``model``."""
    tokens = _CLEF_MODELS[model]
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score"}),
        modalities=frozenset({"text", "json", "image"}),
        image_carriers=frozenset({"array"}),
        max_input_tokens=tokens,
        max_question_tokens=tokens,
        max_body_bytes=13_631_488,
        max_questions=64,
        max_images=_MAX_IMAGES,
        max_images_per_question=_MAX_IMAGES,
        max_image_bytes=_IMAGE_BYTES,
        max_total_image_bytes=_IMAGES_BYTES,
        max_labels=255,
        max_levels=10,
        max_total_options=UNBOUNDED,
        min_levels=2,
        min_image_side=None,
        image_media_types=frozenset({"image/png", "image/jpeg", "image/webp"}),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=False,
        calibrated=False,
        snapshot_pinnable=False,
        truncates_silently=True,
        confidence={"choice": "undocumented", "score": "undocumented"},
        rounding=Rounding(probability_decimals=4, score_decimals=4),
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
    )


def _clairic_record() -> Capabilities:
    """The record a Clairic leaf declares."""
    return Capabilities(
        kinds=frozenset({"extraction"}),
        modalities=frozenset({"text", "image"}),
        image_carriers=frozenset({"array"}),
        max_input_tokens=65_536,
        max_question_tokens=65_536,
        max_body_bytes=16_777_216,
        max_questions=UNBOUNDED,
        max_images=_MAX_IMAGES,
        max_images_per_question=_MAX_IMAGES,
        max_image_bytes=_IMAGE_BYTES,
        max_total_image_bytes=_IMAGES_BYTES,
        max_labels=None,
        max_levels=None,
        max_total_options=None,
        min_levels=None,
        min_image_side=None,
        image_media_types=frozenset({"image/png", "image/jpeg", "image/webp"}),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=False,
        calibrated=False,
        snapshot_pinnable=False,
        truncates_silently=False,
        confidence={"extraction": "undocumented"},
        rounding=Rounding(probability_decimals=None, score_decimals=None),
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
    )


class _ClefEstimator:
    """Clef's estimate: 260 tokens of overhead, the state's JSON at 4 characters a token,
    the questions' at 4, and 1,024 tokens per image, the image formula's cap."""

    def __init__(self) -> None:
        self.margin = _MARGIN

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        wire_state, images = _clef_state(state)
        tokens = _CLEF_OVERHEAD + math.ceil(
            len(json.dumps(wire_state, default=_plain)) / _CLEF_STATE_CHARS
        )
        if questions:
            tokens += math.ceil(len(json.dumps(_clef_questions(questions))) / _CLEF_QUESTION_CHARS)
        tokens += len(images) * _IMAGE_TOKENS
        return TokenEstimate(tokens=tokens, margin=self.margin)


class _ClairicEstimator:
    """Clairic's estimate of one pass: the state's text at 3 characters a token, 150
    tokens of schema per field, and 1,024 tokens per image."""

    def __init__(self) -> None:
        self.margin = _MARGIN

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        text = "".join(part.text for part in state.parts if isinstance(part, TextPart))
        images = sum(isinstance(part, ImagePart) for part in state.parts)
        tokens = (
            _CLAIRIC_OVERHEAD
            + math.ceil(len(text) / _CLAIRIC_STATE_CHARS)
            + len(questions) * _CLAIRIC_FIELD_TOKENS
            + images * _IMAGE_TOKENS
        )
        return TokenEstimate(tokens=tokens, margin=self.margin)


class _CloudflareLeaf(HttpLeaf):
    """What the two leaves share: the account and token, the envelope, the status mapper
    and the provenance."""

    def __init__(
        self,
        *,
        account_id: str | None,
        api_token: str | None,
        client: httpx2.AsyncClient | None,
        who: str,
    ) -> None:
        account = account_id if account_id is not None else os.environ.get(_ACCOUNT_ENV)
        token = api_token if api_token is not None else os.environ.get(_TOKEN_ENV)
        if not account:
            raise ValueError(f"{who} needs an account_id or the {_ACCOUNT_ENV} variable")
        if not token:
            raise ValueError(f"{who} needs an api_token or the {_TOKEN_ENV} variable")
        super().__init__(
            base_url=_BASE_URL.format(account=account),
            request_id_header=_REQUEST_ID_HEADER,
            client=client,
        )
        self._headers = {"Authorization": f"Bearer {token}"}
        self._lender: _CloudflareLeaf | None = None

    @property
    def client(self) -> httpx2.AsyncClient:
        """The HTTP client the leaf sends with: its own, or, for a leaf sharing another's,
        that leaf's current one."""
        return self._lender.client if self._lender is not None else super().client

    def _client_for_this_loop(self) -> httpx2.AsyncClient:
        """The client for the running event loop: a sharing leaf asks the leaf whose client
        it shares, so both follow one loop and one client, which only that leaf closes. A
        sharing leaf closed by its own ``aclose`` is refused as any closed leaf is, before it
        asks."""
        if self._lender is None or self._closed:
            return super()._client_for_this_loop()
        return self._lender._client_for_this_loop()

    def map_status(self, response: httpx2.Response) -> ProviderError:
        """The status table, with the envelope's first error code and message in the
        message; codes 3036 and 4006, the daily quota spent, are not retryable on any status."""
        base = super().map_status(response)
        code, message = _envelope_error(response.content)
        if code is None and message is None:
            return base
        # Key-shaped text is filtered before the cut, so no piece of a key survives it.
        detail = _without_keys(": ".join(str(part) for part in (code, message) if part is not None))
        if len(detail) > _MESSAGE_CHARS:
            detail = detail[:_MESSAGE_CHARS] + "..."
        return ProviderError(
            f"HTTP {response.status_code}: {detail}",
            is_retryable=base.is_retryable and code not in _QUOTA_CODES,
            retry_after=base.retry_after,
            status_code=base.status_code,
            request_id=base.request_id,
        )

    async def _exchange(
        self, path: str, body: Mapping[str, JsonValue], deadline: Deadline
    ) -> tuple[httpx2.Response, Mapping[str, JsonValue], Mapping[str, JsonValue]]:
        """Send the body once; return the response, the envelope and its result."""
        try:
            content = json.dumps(
                body, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_plain
            )
        except (ValueError, TypeError) as error:
            raise ProviderError(f"the state cannot be sent as JSON: {error}") from None
        response, envelope = await self._post(
            path, content.encode(), deadline, headers=self._headers
        )
        if envelope.get("success") is not True:
            raise self._unreadable(response, "the envelope does not report success")
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            raise self._unreadable(response, "the envelope holds no result object")
        return response, envelope, cast(Mapping[str, JsonValue], result)

    def _top_level(
        self, response: httpx2.Response, result: Mapping[str, JsonValue]
    ) -> tuple[Mapping[str, JsonValue], str, Usage]:
        """The answers, the reported model and the usage, or a non-retryable error."""
        answers = result.get("answers")
        if not isinstance(answers, Mapping):
            raise self._unreadable(response, "the result holds no answers map")
        model = result.get("model")
        usage = result.get("usage")
        if not isinstance(model, str) or not model or not isinstance(usage, Mapping):
            raise self._unreadable(response, "the result fails the top-level schema")
        raw_usage = cast(Mapping[str, JsonValue], usage)
        return (
            cast(Mapping[str, JsonValue], answers),
            model,
            Usage(
                input_tokens=_count(raw_usage.get("input_tokens")),
                output_tokens=_count(raw_usage.get("output_tokens")),
                passes=_count(raw_usage.get("passes")),
                raw_usage=raw_usage,
            ),
        )

    def _response(
        self,
        *,
        capabilities: Capabilities,
        wire_request: WireRequest,
        mapped: Mapping[str, MappedAnswer],
        response: httpx2.Response,
        envelope: Mapping[str, JsonValue],
        model_reported: str,
        usage: Usage,
        prompt_sent: Mapping[str, JsonValue],
        known: frozenset[str],
        warnings: tuple[Warning, ...] = (),
    ) -> WireResponse:
        answers, validated = validate_answers(wire_request.questions, mapped, capabilities.rounding)
        snapshot = Warning(
            code=WARNING_NO_SNAPSHOT,
            message=f"the wire reported the short name {model_reported!r}, with no version: "
            "alias drift cannot be detected from the response",
            field="snapshot",
        )
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=capabilities.calibrated,
            model_reported=model_reported,
            snapshot=None,
            prompt_sent=prompt_sent,
            usage=usage,
            request_id=self.request_id(response),
            raw_body=envelope,
            warnings=(
                *validated,
                *warnings,
                snapshot,
                *unknown_options(wire_request.options, known),
            ),
        )


class Clef(_CloudflareLeaf):
    """A leaf for Cloudflare's Clef, or clef-flash, on Workers AI.

    ``model`` is ``clef`` or ``clef-flash``, each on its own route, ``@cf/cloudflare/<model>``,
    since Clef's route refuses ``clef-flash``; each is also taken by that full name, and the
    leaf reports the short name either way. It reads no request option, and takes Clairic's
    ``reference_date`` without a warning. ``account_id`` and ``api_token`` default to the
    ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN`` environment variables, and
    building without either raises ``ValueError``.
    ``capabilities`` replaces the declared record, for a caller who knows better.
    ``client`` is an injected ``httpx2.AsyncClient``, which the leaf never closes.
    """

    def __init__(
        self,
        *,
        model: ClefModel = "clef",
        account_id: str | None = None,
        api_token: str | None = None,
        client: httpx2.AsyncClient | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        short = _CLEF_NAMES.get(model)
        if short is None:
            raise ValueError(
                f"Clef does not know the model {model!r}; known: {', '.join(_CLEF_NAMES)}"
            )
        super().__init__(account_id=account_id, api_token=api_token, client=client, who="Clef")
        self._model: str = short
        self._capabilities = capabilities if capabilities is not None else _clef_record(short)
        self._estimator = _ClefEstimator()

    @property
    def name(self) -> str:
        return "cloudflare-clef"

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def estimator(self) -> _ClefEstimator:
        return self._estimator

    def __repr__(self) -> str:
        return f"Clef(model={self._model!r})"

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Send the wire request once and return its answers, validated, with provenance."""
        body, sizes = _clef_body(self._model, wire_request, self._capabilities.image_media_types)
        response, envelope, result = await self._exchange(
            _CLEF_PATH.format(model=self._model), body, deadline
        )
        answers, model_reported, usage = self._top_level(response, result)
        sent = _sent(body["questions"], wire_request.questions)
        mapped, dropped = _map_answers(sent, answers, _clef_answer)
        return self._response(
            capabilities=self._capabilities,
            wire_request=wire_request,
            mapped=mapped,
            response=response,
            envelope=envelope,
            model_reported=model_reported,
            usage=usage,
            prompt_sent=_prompt_sent(body, sizes),
            known=_CLEF_OPTIONS,
            warnings=dropped,
        )


class Clairic(_CloudflareLeaf):
    """A leaf for Cloudflare's Clairic, its extraction model, on Workers AI.

    It reads one request option, ``reference_date``, an ISO date that resolves dates
    written without a year, and sends it when given. ``model`` is ``clairic`` or its full
    name ``@cf/cloudflare/clairic``, its only model; the leaf reports ``clairic`` either
    way. Credentials, ``capabilities`` and ``client`` are as for :class:`Clef`.
    """

    def __init__(
        self,
        *,
        model: Literal["clairic", "@cf/cloudflare/clairic"] = "clairic",
        account_id: str | None = None,
        api_token: str | None = None,
        client: httpx2.AsyncClient | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        if model not in _CLAIRIC_NAMES:
            raise ValueError(
                f"Clairic does not know the model {model!r}; known: clairic, @cf/cloudflare/clairic"
            )
        super().__init__(account_id=account_id, api_token=api_token, client=client, who="Clairic")
        self._capabilities = capabilities if capabilities is not None else _clairic_record()
        self._estimator = _ClairicEstimator()

    @property
    def name(self) -> str:
        return "cloudflare-clairic"

    @property
    def model(self) -> str:
        return _CLAIRIC_MODEL

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def estimator(self) -> _ClairicEstimator:
        return self._estimator

    def __repr__(self) -> str:
        return "Clairic()"

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Send the wire request once and return its answers, validated, with provenance."""
        body, sizes, text_part = _clairic_body(
            wire_request, _reference_date(wire_request), self._capabilities.image_media_types
        )
        response, envelope, result = await self._exchange(_CLAIRIC_PATH, body, deadline)
        answers, model_reported, usage = self._top_level(response, result)

        def read(question: Question, answer: Mapping[str, JsonValue]) -> MappedAnswer:
            return _clairic_answer(question, answer, text_part)

        sent = _sent(body["fields"], wire_request.questions)
        mapped, warnings = _map_answers(sent, answers, read)
        if result.get("truncated") is True:
            warnings += (
                Warning(
                    code=WARNING_PROVIDER_TRUNCATED,
                    message="Clairic reported that it truncated the state; the answers are "
                    "read from what it kept",
                    field="state",
                ),
            )
        return self._response(
            capabilities=self._capabilities,
            wire_request=wire_request,
            mapped=mapped,
            response=response,
            envelope=envelope,
            model_reported=model_reported,
            usage=usage,
            prompt_sent=_prompt_sent(body, sizes),
            known=_CLAIRIC_OPTIONS,
            warnings=warnings,
        )


# ----------------------------------------------------------------------------------------
# The wire bodies
# ----------------------------------------------------------------------------------------


def _plain(value: object) -> dict[str, JsonValue]:
    """A read-only mapping in a caller's JSON part, as a dict ``json`` can encode."""
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, JsonValue], value))
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def _content_type(image: ImagePart, accepted: frozenset[str] | None) -> str:
    """The type Fit judged, PNG, JPEG or WebP. An image Fit would refuse, in a call made
    past Fit, goes with its declared type as given, else ``application/octet-stream``, for
    the wire to refuse."""
    judged = _image_media_type(image, accepted)
    if judged is not None:
        return judged
    return image.media_type or "application/octet-stream"


def _images(
    images: Sequence[ImagePart], accepted: frozenset[str] | None
) -> tuple[tuple[JsonValue, ...], dict[str, int]]:
    """The ``images`` array and each image's size in bytes, by index. The record declares
    the count, the byte caps and the types, which Fit checks before any call."""
    wire = tuple(
        {
            "content_type": _content_type(image, accepted),
            "base64": base64.b64encode(image.data).decode("ascii"),
        }
        for image in images
    )
    return wire, {str(index): len(image.data) for index, image in enumerate(images)}


def _prompt_sent(body: Mapping[str, JsonValue], sizes: Mapping[str, int]) -> dict[str, JsonValue]:
    """The body sent, each image's base64 replaced by its size in bytes."""
    sent = dict(body)
    images = cast(tuple[Mapping[str, JsonValue], ...], body.get("images", ()))
    if images:
        sent["images"] = tuple(
            {"content_type": image["content_type"], "bytes": sizes[str(index)]}
            for index, image in enumerate(images)
        )
    return sent


def _clef_state(state: State) -> tuple[JsonValue, list[ImagePart]]:
    """The state as Clef takes it, the images taken out in order: one remaining part is its
    own value, several an array, none the empty string."""
    values: list[JsonValue] = []
    images: list[ImagePart] = []
    for part in state.parts:
        if isinstance(part, ImagePart):
            images.append(part)
        elif isinstance(part, TextPart):
            values.append(part.text)
        else:
            values.append(part.value)
    if not values:
        return "", images
    return (values[0] if len(values) == 1 else tuple(values)), images


def _clef_question(question: Question) -> dict[str, JsonValue]:
    if question.kind not in ("noul", "choice", "score"):
        raise ProviderError(f"Clef does not answer {question.kind!r} questions")
    wire: dict[str, JsonValue] = {"type": question.kind, "instructions": question.instruction}
    if question.kind == "choice":
        wire["criteria"] = {label.name: label.description for label in question.labels}
    elif question.kind == "score":
        wire["criteria"] = tuple(
            label.description if label.description is not None else label.name
            for label in question.labels
        )
    return wire


def _wire_ids(questions: Sequence[Question], pattern: re.Pattern[str], prefix: str) -> list[str]:
    """The ids a wire request is sent under: the caller's keys when every one matches the
    wire's pattern, else ``<prefix>0..<prefix>N`` for every question, by position."""
    keys = [question.key for question in questions]
    if all(pattern.fullmatch(key) for key in keys):
        return keys
    return [f"{prefix}{index}" for index in range(len(keys))]


def _sent(wire: JsonValue, questions: Sequence[Question]) -> dict[str, Question]:
    """Each id the body was sent under, in order, with the question it stands for."""
    return dict(zip(cast(Mapping[str, JsonValue], wire), questions, strict=True))


def _clef_questions(questions: Sequence[Question]) -> dict[str, JsonValue]:
    """The ``questions`` object, under the ids :func:`_wire_ids` gives; the estimator
    counts the same object, so it counts the ids as sent."""
    ids = _wire_ids(questions, _CLEF_ID, "q")
    return {id_: _clef_question(question) for id_, question in zip(ids, questions, strict=True)}


def _clef_body(
    model: str, wire_request: WireRequest, accepted: frozenset[str] | None
) -> tuple[dict[str, JsonValue], dict[str, int]]:
    """Clef's body, keys ``model, state, questions[, images]`` in that order."""
    state, images = _clef_state(wire_request.state)
    body: dict[str, JsonValue] = {
        "model": model,
        "state": state,
        "questions": _clef_questions(wire_request.questions),
    }
    sizes: dict[str, int] = {}
    if images:
        body["images"], sizes = _images(images, accepted)
    return body, sizes


def _clairic_field(question: Question) -> dict[str, JsonValue]:
    """One field spec from the declared type and the instruction alone, a list's item type
    derived; nothing else is sent, so Clairic's defaults stand and any unit or precision is
    whatever the instruction says."""
    if question.kind != "extraction" or question.value_type is None:
        raise ProviderError(f"Clairic answers only extraction questions, not {question.kind!r}")
    if question.is_list:
        if question.value_type not in _LIST_ITEMS:
            raise ProviderError(
                f"Clairic lists hold text, number or date items, not {question.value_type!r}"
            )
        return {
            "type": "list",
            "instructions": question.instruction,
            "items": question.value_type,
        }
    return {"type": question.value_type, "instructions": question.instruction}


def _reference_date(wire_request: WireRequest) -> datetime.date | None:
    """The ``reference_date`` option as a date; one that is not an ISO date string is
    refused before sending."""
    held = wire_request.options.get(_REFERENCE_DATE)
    if held is None:
        return None
    try:
        if not isinstance(held, str):
            raise ValueError
        return datetime.date.fromisoformat(held)
    except ValueError:
        raise ProviderError(
            f"the reference_date option must be an ISO date string, not {held!r}"
        ) from None


def _clairic_body(
    wire_request: WireRequest,
    reference_date: datetime.date | None,
    accepted: frozenset[str] | None,
) -> tuple[dict[str, JsonValue], dict[str, int], int | None]:
    """Clairic's body, keys ``model[, images][, state], fields[, reference_date]`` in that
    order, the sizes of its images, and the index of its one text part, if any.

    The state is one text part at most, and no JSON part, since the spans index the text
    sent; a state of images alone sends no ``state``."""
    texts = [i for i, part in enumerate(wire_request.state.parts) if isinstance(part, TextPart)]
    if len(texts) > 1:
        raise ProviderError(f"Clairic takes one text part; this state has {len(texts)}")
    if any(isinstance(part, JsonPart) for part in wire_request.state.parts):
        raise ProviderError("Clairic takes text and images, not a JSON part")
    images = [part for part in wire_request.state.parts if isinstance(part, ImagePart)]
    questions = wire_request.questions
    ids = _wire_ids(questions, _CLAIRIC_NAME, "f")
    fields = {id_: _clairic_field(question) for id_, question in zip(ids, questions, strict=True)}
    body: dict[str, JsonValue] = {"model": _CLAIRIC_MODEL}
    sizes: dict[str, int] = {}
    if images:
        body["images"], sizes = _images(images, accepted)
    text_part = texts[0] if texts else None
    if text_part is not None:
        body["state"] = cast(TextPart, wire_request.state.parts[text_part]).text
    elif not images:
        body["state"] = ""
    body["fields"] = fields
    if reference_date is not None:
        body["reference_date"] = reference_date.isoformat()
    return body, sizes, text_part


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


def _envelope_error(content: bytes) -> tuple[int | None, str | None]:
    """The first error's code and message from a Cloudflare envelope, where there is one."""
    try:
        body: object = json.loads(content)
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    errors = cast(dict[str, object], body).get("errors")
    if not isinstance(errors, list) or not errors:
        return None, None
    first = cast(list[object], errors)[0]
    if not isinstance(first, dict):
        return None, None
    held = cast(dict[str, object], first)
    code = held.get("code")
    message = held.get("message")
    return (
        code if isinstance(code, int) and not isinstance(code, bool) else None,
        message if isinstance(message, str) else None,
    )


def _map_answers(
    sent: Mapping[str, Question],
    answers: Mapping[str, JsonValue],
    read: Callable[[Question, Mapping[str, JsonValue]], MappedAnswer],
) -> tuple[dict[str, MappedAnswer], tuple[Warning, ...]]:
    """Each wire answer mapped to its kind's shape, keyed by the caller's key of the
    question its id was sent for.

    An answer under an id not sent is dropped with an ``extra-answer-key`` warning; it is
    looked up among the ids sent, never the caller's keys, which an aliased request does
    not send. An answer whose wire type or shape is not its question's becomes that
    question's fault."""
    mapped: dict[str, MappedAnswer] = {}
    dropped: list[Warning] = []
    for wire_id, answer in answers.items():
        question = sent.get(wire_id)
        if question is None:
            dropped.append(
                Warning(
                    code=WARNING_EXTRA_ANSWER_KEY,
                    message=f"answer key {wire_id!r} matches no question and was dropped",
                    field=wire_id,
                )
            )
            continue
        try:
            mapped[question.key] = read(question, _object(answer))
        except _Shape as shape:
            failure = RuleFailure(rule=Rule.kind_shape, observed=shape.observed, bound=None)
            mapped[question.key] = Fault(key=question.key, failures=(failure,), mapped=None)
    return mapped, tuple(dropped)


def _clef_answer(question: Question, answer: Mapping[str, JsonValue]) -> MappedAnswer:
    wire_type = answer.get("type")
    if wire_type != question.kind:
        raise _Shape(str(wire_type))
    if question.kind == "noul":
        return NoulAnswer(probability=_number(answer.get("noul")))
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
    """A legend's levels by index; a key that is no index sorts last, by its text."""
    return (int(index), "") if index.isdecimal() else (1 << 30, index)


def _in_label_order(
    question: Question, probabilities: Mapping[str, JsonValue], *, by_index: bool
) -> tuple[tuple[str, float], ...]:
    """The wire's probabilities named by label, in the order asked, then any the wire
    added under their own key; a label left out stays out, for validation to fault."""
    names = {
        (str(index) if by_index else label.name): label.name
        for index, label in enumerate(question.labels)
    }
    asked = [(names[key], _number(probabilities[key])) for key in names if key in probabilities]
    added = [(key, _number(value)) for key, value in probabilities.items() if key not in names]
    return tuple(asked + added)


def _candidate(answer: Mapping[str, JsonValue]) -> str | None:
    """The candidate kind an answer holds, by its type or its distribution's keys."""
    wire_type = answer.get("type")
    if isinstance(wire_type, str) and wire_type in _CANDIDATE_TYPES:
        return wire_type
    held = _DISTRIBUTION_KEYS & set(answer)
    if not held:
        return None
    return "forecast" if "steps" in held else "estimate"


def _scalar(value: JsonValue, value_type: str, unit: JsonValue = "seconds") -> Value:
    """One value read as its declared type; a value that does not read is left as given,
    for validation to fault. A duration is a number of seconds, the unit Clairic reports
    beside it; one in any other unit is left as given."""
    try:
        if value_type == "date" and isinstance(value, str):
            return datetime.date.fromisoformat(value)
        if value_type == "time" and isinstance(value, str):
            return datetime.time.fromisoformat(value)
        if value_type == "datetime" and isinstance(value, str):
            return datetime.datetime.fromisoformat(value)
        if (
            value_type == "duration"
            and unit == "seconds"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            return datetime.timedelta(seconds=value)
    except (ValueError, OverflowError):
        pass
    return cast(Value, value)


def _value(value: JsonValue, question: Question, unit: JsonValue) -> Value | None:
    value_type = cast(str, question.value_type)
    if value is None:
        return None
    if question.is_list and isinstance(value, tuple):
        return cast(Value, tuple(_scalar(item, value_type) for item in value))
    return _scalar(value, value_type, unit)


def _span(pair: JsonValue, part: int) -> Span:
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in pair)
    ):
        raise _Shape(f"span {pair!r}")
    start, end = cast(tuple[int, int], pair)
    return Span(part=part, start=start, end=end)


def _spans(
    answer: Mapping[str, JsonValue], question: Question, part: int | None
) -> Span | tuple[Span, ...] | None:
    """The span, or a list's spans, into the state's one text part; none without one."""
    if part is None:
        return None
    if question.is_list:
        spans = answer.get("spans")
        if spans is None:
            return None
        if not isinstance(spans, tuple):
            raise _Shape(f"spans {spans!r}")
        return tuple(_span(pair, part) for pair in spans)
    pair = answer.get("span")
    return None if pair is None else _span(pair, part)


def _clairic_answer(
    question: Question, answer: Mapping[str, JsonValue], part: int | None
) -> MappedAnswer:
    candidate = _candidate(answer)
    if candidate is not None:
        failure = RuleFailure(rule=Rule.candidate_shape, observed=candidate, bound=None)
        return Fault(key=question.key, failures=(failure,), mapped=None)
    wanted = "list" if question.is_list else question.value_type
    wire_type = answer.get("type")
    if wire_type != wanted:
        raise _Shape(str(wire_type))
    if "value" not in answer:
        raise _Shape("no value")
    return ExtractionAnswer(
        value=_value(answer.get("value"), question, answer.get("unit", "seconds")),
        guess=_value(answer.get("guess"), question, answer.get("unit", "seconds")),
        confidence=_optional_number(answer.get("confidence")),
        p_null=_optional_number(answer.get("p_null")),
        span=_spans(answer, question, part),
    )


def _share_client(borrower: _CloudflareLeaf, lender: _CloudflareLeaf) -> None:  # pyright: ignore[reportUnusedFunction]
    """Have ``borrower`` send with ``lender``'s client, whichever one ``lender`` holds for
    the running loop; ``borrower`` never closes it. The ``cloudflare`` router's one
    client."""
    borrower._lender = lender  # pyright: ignore[reportPrivateUsage]
