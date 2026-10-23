"""The vocabulary every part of idia speaks.

This module holds every shape a request, an answer and a response take, the two protocols a
leaf and a provider satisfy, the errors, and the closed sets of kinds and elicitations. Every
shape is a frozen, slotted, keyword-only dataclass: once built, nothing on it changes, and a
map-valued field is a read-only view.

It imports only the standard library. Nothing here knows a wire API, a provider SDK or the
call loop; those live beside it and import it, never the other way.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias, TypeVar, cast, get_args, runtime_checkable

# ----------------------------------------------------------------------------------------
# Closed sets
# ----------------------------------------------------------------------------------------

Kind: TypeAlias = Literal["noul", "choice", "score", "extraction", "box"]
"""The answer shapes a question may ask for. The set is closed for this release."""

CANDIDATE_KINDS: Final[frozenset[str]] = frozenset({"estimate", "forecast", "multi_label"})
"""Shapes some provider returns that no kind in this release holds.

A question cannot be built with one; building it raises :class:`KindNotInRelease` naming the
candidate. A candidate enters the kind set only by a minor release.
"""

ValueType: TypeAlias = Literal["text", "number", "date", "time", "datetime", "duration"]
"""The type an extraction question declares for its value."""

Elicitation: TypeAlias = Literal["passthrough", "logprobs"]
"""How a provider's raw output became probabilities.

``passthrough`` is what every decision model reports: the probabilities are the wire's own.
``logprobs`` is the chat leaf's read of a token distribution by its fixed protocol.
"""

CANDIDATE_ELICITATIONS: Final[frozenset[str]] = frozenset(
    {"structured_output", "verbalised", "sampled_n"}
)
"""Elicitations named but not admitted in this release; a chat leaf cannot be built with one."""

Modality: TypeAlias = Literal["text", "json", "image"]
"""The medium of one part of a state."""

ImageCarrier: TypeAlias = Literal["placeholder", "array", "parts"]
"""How a wire API carries an image: a placeholder in the text, a top-level array, or a
content part beside the text."""

JsonValue: TypeAlias = (
    "bool | int | float | str | tuple[JsonValue, ...] | Mapping[str, JsonValue] | None"
)
"""A JSON value as Python data, with arrays as tuples, so it can be hashed and logged."""

WARNING_EXTRA_ANSWER_KEY: Final = "extra-answer-key"
"""Warning code: an answer key matched no question and was dropped."""

WARNING_UNCHECKED_LIMIT: Final = "unchecked-limit"
"""Warning code: preflight skipped a limit nobody knows; the field is named."""

WARNING_NO_SNAPSHOT: Final = "no-snapshot"
"""Warning code: the wire reported an alias or short name, so no snapshot is recorded."""

WARNING_LABEL_ABSENT_FROM_LOGPROBS: Final = "label-absent-from-logprobs"
"""Warning code: a chat leaf's read found some labels' letters outside the returned top-k
and read them as zero mass; the message names the letters and the ceiling on their mass,
the smallest probability returned, and the field the question's key."""

WARNING_PROVIDER_TRUNCATED: Final = "provider-truncated"
"""Warning code: the wire reported that it truncated the state before answering; the field
is ``state``. The answers stand, read from what the wire kept."""

WARNING_UNKNOWN_OPTION: Final = "unknown-option"
"""Warning code: the request carried options the leaf does not know; it ignored them, and
the message names them. The field is ``options``."""

WARNING_OVERFLOW_POLICY: Final = "overflow-policy"
"""Warning code: the state overflowed and a named overflow policy, named in the message,
changed it before any call."""

_KINDS: Final[frozenset[str]] = frozenset(get_args(Kind))
_VALUE_TYPES: Final[frozenset[str]] = frozenset(get_args(ValueType))
_CANDIDATE_VALUE_TYPES: Final[frozenset[str]] = frozenset({"estimate", "forecast"})
_ELICITATIONS: Final[frozenset[str]] = frozenset(get_args(Elicitation))

# ----------------------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------------------


class IdiaError(Exception):
    """The base of every error idia raises."""


_OPENAI_KEY_SHAPE: Final = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}")
"""An OpenAI-style API key: ``sk-`` and at least twenty key characters, not inside a longer
token."""

_BEARER_SHAPE: Final = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}=*")
"""A bearer credential written out: ``Bearer`` and a token of sixteen characters or more.
It can match prose, such as "bearer responsibilities", and then hides it too."""

_KEY_SHAPES: Final[tuple[re.Pattern[str], ...]] = (_OPENAI_KEY_SHAPE, _BEARER_SHAPE)
"""The key-shaped substrings no provider error's text carries."""

_FILTERED: Final = "<filtered>"
"""What replaces a key-shaped substring; it matches no shape, so filtering again changes
nothing."""


def _without_keys(text: str) -> str:
    """``text`` with every key-shaped substring replaced by ``<filtered>``."""
    for shape in _KEY_SHAPES:
        text = shape.sub(_FILTERED, text)
    return text


class ProviderError(IdiaError):
    """The one error a leaf raises when a wire request fails.

    The leaf's status mapper sets ``is_retryable`` and ``retry_after``; the core decides
    whether to try again. A fault in one answer is never a provider error. The message has
    every key-shaped substring, an ``sk-`` key or a bearer token written out, replaced by
    ``<filtered>`` when the error is built, so neither it nor a ``RequestFailed`` quoting it
    carries such a key; the other fields are kept as given.
    """

    message: str
    is_retryable: bool
    retry_after: float | None
    status_code: int | None
    request_id: str | None

    def __init__(
        self,
        message: str,
        *,
        is_retryable: bool = False,
        retry_after: float | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        filtered = _without_keys(message)
        super().__init__(filtered)
        self.message = filtered
        self.is_retryable = is_retryable
        self.retry_after = retry_after
        self.status_code = status_code
        self.request_id = request_id


class DeadlineExceeded(ProviderError):
    """The error on the outcome of a wire request the deadline ended: one never sent because
    the deadline passed, or one whose next retry it could not fit, its failed attempts
    listed with their own errors.

    The dispatcher or a limiter raises it, never a leaf, and it is never retryable.
    """

    def __init__(
        self, message: str = "the deadline passed before the wire request was sent"
    ) -> None:
        super().__init__(message, is_retryable=False)


class KindNotInRelease(IdiaError):
    """A question was built with a kind or value type this release does not admit.

    ``candidate`` is true when the name is a known candidate kind or value type waiting on
    a release, false when it is no kind at all; a value type that is unknown altogether
    raises ``ValueError`` instead.
    """

    kind: str
    candidate: bool

    def __init__(self, kind: str, candidate: bool) -> None:
        what = (
            "a candidate kind or value type not admitted in this release"
            if candidate
            else "not a kind"
        )
        super().__init__(f"{kind!r} is {what}")
        self.kind = kind
        self.candidate = candidate


class ElicitationNotInRelease(IdiaError):
    """A chat leaf was built with an elicitation this release does not admit."""

    elicitation: str
    candidate: bool

    def __init__(self, elicitation: str, candidate: bool) -> None:
        what = (
            "a candidate elicitation not admitted in this release"
            if candidate
            else "not an elicitation"
        )
        super().__init__(f"{elicitation!r} is {what}")
        self.elicitation = elicitation
        self.candidate = candidate


class DoesNotFit(IdiaError):
    """A request does not fit a provider's declared capabilities; raised before any call.

    ``check`` names the capabilities field the request failed, such as ``max_input_tokens``
    or ``kinds``; ``observed`` is what the request had and ``limit`` what the field allows.
    ``state_overflow`` is true when the state alone does not fit, whatever the questions.
    ``off_features`` lists opt-in features the provider offers and this instance did not
    turn on. For a missing kind or modality, ``observed`` is the missing values and
    ``off_features`` only the features the record says add one; for any other check, it is
    every feature not turned on, since no record says what a feature does to a limit.
    """

    message: str
    check: str
    observed: object
    limit: object
    state_overflow: bool
    off_features: frozenset[str]

    def __init__(
        self,
        message: str,
        *,
        check: str,
        observed: object,
        limit: object,
        state_overflow: bool = False,
        off_features: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(message)
        self.message = message
        self.check = check
        self.observed = observed
        self.limit = limit
        self.state_overflow = state_overflow
        self.off_features = off_features


def check_elicitation(name: str) -> Elicitation:
    """Return ``name`` as an admitted elicitation or raise :class:`ElicitationNotInRelease`.

    The chat leaf calls this at construction.
    """
    if name in CANDIDATE_ELICITATIONS:
        raise ElicitationNotInRelease(name, candidate=True)
    if name not in _ELICITATIONS:
        raise ElicitationNotInRelease(name, candidate=False)
    return cast(Elicitation, name)


# ----------------------------------------------------------------------------------------
# The constructors' refusals
# ----------------------------------------------------------------------------------------


def _check_type(
    owner: str, name: str, value: object, types: type | tuple[type, ...], what: str
) -> None:
    """Refuse a field of the wrong type with ``TypeError``, naming the field."""
    if not isinstance(value, types):
        raise TypeError(f"{owner}.{name} must be {what}, got {type(value).__name__}")


def _check_text(owner: str, name: str, value: object, hint: str = "") -> None:
    """Refuse a field that is not a ``str``, then an empty one with ``ValueError``."""
    _check_type(owner, name, value, str, "a str")
    if value == "":
        raise ValueError(f"{owner}.{name} must not be empty{hint}")


def _check_items(
    owner: str, name: str, value: object, types: type | tuple[type, ...], what: str
) -> None:
    """Refuse a field that is not a tuple, then any item of the wrong type, by its index."""
    _check_type(owner, name, value, tuple, "a tuple")
    for index, item in enumerate(cast(tuple[object, ...], value)):
        _check_type(owner, f"{name}[{index}]", item, types, what)


_DEADLINE_HORIZON: Final = 2_592_000.0
"""Thirty days, in seconds: a ``Deadline.at`` further ahead than this on the monotonic
clock was almost surely a wall-clock time."""

_CLOCK_TICK: Final[float] = time.get_clock_info("monotonic").resolution
"""The monotonic clock's resolution, which asyncio's event loop reads too: the loop fires a
timer up to this much (or one float ulp of the clock, if larger) before its time."""


# ----------------------------------------------------------------------------------------
# The deadline and the state
# ----------------------------------------------------------------------------------------


def _finite(number: float) -> bool:
    """Whether a number is finite as a float: an int too large for one is not."""
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True, kw_only=True)
class Deadline:
    """The instant after which the caller no longer wants an answer.

    ``at`` is on the monotonic clock (``time.monotonic()``), not the wall clock. Every wait,
    retry and call for one request counts against the same instant.
    """

    at: float

    def __post_init__(self) -> None:
        at = cast(object, self.at)
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            raise TypeError(f"Deadline.at must be a number, got {type(at).__name__}")
        if not _finite(at):
            raise ValueError(f"Deadline.at must be finite, got {at!r}")
        if at - time.monotonic() > _DEADLINE_HORIZON:
            raise ValueError(
                "Deadline.at is more than 30 days ahead on the monotonic clock; a wall-clock "
                "time was likely given: use Deadline.after(seconds)"
            )

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        """The deadline ``seconds`` from now; more than 30 days, or not finite, is refused.

        Zero or negative seconds give a deadline already passed, which is allowed."""
        if not _finite(seconds) or seconds > _DEADLINE_HORIZON:
            raise ValueError(
                "Deadline.after takes a finite number of seconds, at most 30 days "
                f"({_DEADLINE_HORIZON:.0f} seconds), got {seconds!r}"
            )
        return cls(at=time.monotonic() + seconds)

    def remaining(self) -> float:
        """Seconds left; negative once the deadline has passed."""
        return self.at - time.monotonic()

    def passed(self) -> bool:
        """Whether the deadline has passed or is nearer than the event loop can time.

        asyncio fires a timer up to one clock tick early, so a call cut at the deadline may
        end with that much left; this reads it as passed, as every caller should.
        """
        now = time.monotonic()
        return self.at - now <= max(_CLOCK_TICK, math.ulp(now))


@dataclass(frozen=True, slots=True, kw_only=True)
class TextPart:
    """A text part of a state. The text may be empty."""

    text: str

    def __post_init__(self) -> None:
        _check_type("TextPart", "text", self.text, str, "a str")


@dataclass(frozen=True, slots=True, kw_only=True)
class JsonPart:
    """A structured part of a state: a JSON value as Python data."""

    value: JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class ImagePart:
    """An image part of a state: its bytes, with a media type where known."""

    data: bytes
    media_type: str | None = None

    def __post_init__(self) -> None:
        _check_type("ImagePart", "data", self.data, bytes, "bytes")
        if self.media_type is not None:
            _check_text("ImagePart", "media_type", self.media_type, "; use None when unknown")


Part: TypeAlias = TextPart | JsonPart | ImagePart


# ----------------------------------------------------------------------------------------
# Reading an image: the one reader of its type and of its size, for Fit and every leaf
# ----------------------------------------------------------------------------------------

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE: Final = b"\xff\xd8\xff"
# The JPEG start-of-frame markers, which hold the size: C0 to CF but for C4 (Huffman
# tables), C8 (reserved) and CC (arithmetic coding).
_JPEG_FRAMES: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _declared_media_type(image: ImagePart) -> str | None:  # pyright: ignore[reportUnusedFunction]
    """The image's declared media type lower-cased, ``image/jpg`` read as ``image/jpeg``,
    or ``None`` when it declares none."""
    if image.media_type is None:
        return None
    declared = image.media_type.lower()
    return "image/jpeg" if declared == "image/jpg" else declared


def _signature_media_type(data: bytes) -> str | None:  # pyright: ignore[reportUnusedFunction]
    """``image/png``, ``image/jpeg`` or ``image/webp`` from the bytes' signature, else
    ``None``."""
    if data.startswith(_PNG_SIGNATURE):
        return "image/png"
    if data.startswith(_JPEG_SIGNATURE):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_media_type(  # pyright: ignore[reportUnusedFunction]
    image: ImagePart, accepted: frozenset[str] | None
) -> str | None:
    """The image's media type as a record with ``accepted`` reads it: the declared type
    when it is in the set, else the signature's when that is, else ``None``. With
    ``accepted`` ``None``, the declared type, else the signature's."""
    declared = _declared_media_type(image)
    signed = _signature_media_type(image.data)
    if accepted is None:
        return declared or signed
    if declared in accepted:
        return declared
    if signed in accepted:
        return signed
    return None


def _image_size(data: bytes) -> tuple[int, int] | None:  # pyright: ignore[reportUnusedFunction]
    """Width and height from a PNG's IHDR chunk or a JPEG's first start-of-frame segment,
    or ``None`` when the bytes are neither or end before the size."""
    if data.startswith(_PNG_SIGNATURE) and data[12:16] == b"IHDR" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(_JPEG_SIGNATURE):
        return _jpeg_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Walk a JPEG's segments to its first start-of-frame; ``None`` at the scan, the end
    or a byte where a marker should be."""
    at = 2
    while at + 4 <= len(data):
        if data[at] != 0xFF:
            return None
        marker = data[at + 1]
        if marker == 0xFF:
            at += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            at += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        length = int.from_bytes(data[at + 2 : at + 4], "big")
        if marker in _JPEG_FRAMES:
            if at + 9 > len(data):
                return None
            height = int.from_bytes(data[at + 5 : at + 7], "big")
            width = int.from_bytes(data[at + 7 : at + 9], "big")
            return width, height
        at += 2 + length
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class State:
    """The input one request is about: its parts in order."""

    parts: tuple[Part, ...]

    def __post_init__(self) -> None:
        _check_items(
            "State",
            "parts",
            self.parts,
            (TextPart, JsonPart, ImagePart),
            "a TextPart, JsonPart or ImagePart",
        )


# ----------------------------------------------------------------------------------------
# Questions and requests
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Label:
    """One allowed answer of a ``choice`` question, one level of a ``score``, or the ``yes``
    or ``no`` of a ``noul`` that describes them.

    A level's position in the question's labels, from 0, is its value.
    """

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        _check_text("Label", "name", self.name)
        if self.description is not None:
            _check_text("Label", "description", self.description, "; use None for no description")


@dataclass(frozen=True, slots=True, kw_only=True)
class Question:
    """One question put to a provider: a key, a kind, its instruction and its labels in order.

    The instruction is a non-empty string, or a non-empty JSON object or array for a wire API
    that takes structure; an object is held read-only, as a request's options are, its
    values as given, and the instruction is left out of the question's hash. A leaf
    sends it as it is or refuses it before sending, and never drops it.

    A ``choice`` takes two or more labels and a ``score`` one or more levels, both in
    ``labels``. A ``noul`` takes none, or exactly ``yes`` then ``no`` with at least one
    described, the wording of what counts as each. An ``extraction`` declares its
    ``value_type`` and whether it asks for a list. A ``box`` takes neither. A question of a
    candidate kind or candidate value type cannot be built: :class:`KindNotInRelease` is
    raised before any request exists.
    """

    key: str
    kind: Kind
    instruction: str | JsonValue = field(hash=False)
    labels: tuple[Label, ...] = ()
    value_type: ValueType | None = None
    is_list: bool = False

    def __post_init__(self) -> None:
        _check_text("Question", "key", self.key)
        _check_type("Question", "kind", self.kind, str, "a str")
        _check_instruction(self.instruction)
        if isinstance(self.instruction, Mapping):
            frozen = MappingProxyType(dict(cast(Mapping[str, JsonValue], self.instruction)))
            object.__setattr__(self, "instruction", frozen)
        _check_items("Question", "labels", self.labels, Label, "a Label")
        _check_type("Question", "is_list", self.is_list, bool, "a bool")
        if self.value_type is not None:
            _check_type("Question", "value_type", self.value_type, str, "a str")
        kind: str = self.kind
        if kind in CANDIDATE_KINDS:
            raise KindNotInRelease(kind, candidate=True)
        if kind not in _KINDS:
            raise KindNotInRelease(kind, candidate=False)
        value_type: str | None = self.value_type
        if value_type in _CANDIDATE_VALUE_TYPES:
            raise KindNotInRelease(value_type, candidate=True)
        if value_type is not None and value_type not in _VALUE_TYPES:
            raise ValueError(f"{value_type!r} is not a value type")
        if kind in ("choice", "score"):
            if not self.labels:
                raise ValueError(f"a {kind} question needs labels")
            if kind == "choice" and len(self.labels) < 2:
                raise ValueError("a choice question needs at least two labels")
            names = [label.name for label in self.labels]
            if len(set(names)) != len(names):
                raise ValueError("label names must be unique within a question")
        elif kind == "noul" and self.labels:
            names = tuple(label.name for label in self.labels)
            if names != _NOUL_LABELS:
                raise ValueError(
                    f"a noul question's labels are none, or yes then no, not {list(names)}"
                )
            if all(label.description is None for label in self.labels):
                raise ValueError(
                    "a noul question's yes and no labels need at least one description; "
                    "use no labels for none"
                )
        elif self.labels:
            raise ValueError(f"a {kind} question takes no labels")
        if kind == "extraction":
            if value_type is None:
                raise ValueError("an extraction question declares its value type")
        else:
            if value_type is not None:
                raise ValueError(f"a {kind} question declares no value type")
            if self.is_list:
                raise ValueError(f"a {kind} question cannot ask for a list")


_NOUL_LABELS: Final = ("yes", "no")
"""The names a ``noul``'s labels take, in order, when it describes them."""


def _check_instruction(instruction: object) -> None:
    """A non-empty string, or a non-empty JSON object (a mapping with string keys) or array
    (a tuple); the values inside are not checked, as a ``JsonPart``'s are not."""
    if isinstance(instruction, str):
        _check_text("Question", "instruction", instruction)
        return
    if isinstance(instruction, Mapping):
        keys = cast(Mapping[object, object], instruction)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("Question.instruction must have str keys")
    elif not isinstance(instruction, tuple):
        raise TypeError(
            "Question.instruction must be a str, a JSON object or a JSON array (a tuple), "
            f"not {type(instruction).__name__}"
        )
    if not instruction:
        raise ValueError("Question.instruction must not be empty")


def _no_options() -> Mapping[str, JsonValue]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class Request:
    """One state, the questions put to one provider, the caller's deadline, and options.

    Keys are unique within a request. ``options`` are per-request settings a leaf may read
    by key, such as Clairic's ``reference_date``; a leaf ignores a key it does not know and
    warns ``unknown-option`` naming it. Every wire request planned from the request carries
    the same options. They are held as a read-only mapping, left out of the object's
    ``hash()``, and part of the canonical wire request when non-empty.
    """

    state: State
    questions: tuple[Question, ...]
    deadline: Deadline
    options: Mapping[str, JsonValue] = field(default_factory=_no_options, hash=False)

    def __post_init__(self) -> None:
        _check_shape("Request", self.state, self.questions, self.options)
        _check_type("Request", "deadline", self.deadline, Deadline, "a Deadline")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if not self.questions:
            raise ValueError("a request asks at least one question")
        keys = [question.key for question in self.questions]
        if len(set(keys)) != len(keys):
            raise ValueError("question keys must be unique within a request")


@dataclass(frozen=True, slots=True, kw_only=True)
class WireRequest:
    """One entry of a fit plan: the state and the questions a leaf sends in one call, and
    the request's options, unchanged."""

    state: State
    questions: tuple[Question, ...]
    options: Mapping[str, JsonValue] = field(default_factory=_no_options, hash=False)

    def __post_init__(self) -> None:
        _check_shape("WireRequest", self.state, self.questions, self.options)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


def _check_shape(owner: str, state: object, questions: object, options: object) -> None:
    """The fields a request and a wire request share: a state, a tuple of questions and a
    mapping of options."""
    _check_type(owner, "state", state, State, "a State")
    _check_items(owner, "questions", questions, Question, "a Question")
    _check_type(owner, "options", options, Mapping, "a mapping")


# ----------------------------------------------------------------------------------------
# Capabilities
# ----------------------------------------------------------------------------------------


class Unbounded(enum.Enum):
    """The type of :data:`UNBOUNDED`."""

    UNBOUNDED = "unbounded"


UNBOUNDED: Final = Unbounded.UNBOUNDED
"""A limit the wire API documents as absent. Contrast ``None``, a limit nobody knows."""

Limit: TypeAlias = int | Unbounded | None
"""A capability limit: a number that is checked, :data:`UNBOUNDED`, or ``None`` for unknown."""

_LIMIT_FIELDS: Final = (
    "max_input_tokens",
    "max_question_tokens",
    "max_body_bytes",
    "max_questions",
    "max_images",
    "max_images_per_question",
    "max_image_bytes",
    "max_total_image_bytes",
    "max_labels",
    "max_levels",
    "max_total_options",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Rounding:
    """The decimals a provider rounds its probabilities and its expected level to.

    ``None`` means unrounded. The sum tolerance in answer validation follows from it.
    """

    probability_decimals: int | None
    score_decimals: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Capabilities:
    """What one provider accepts and returns, per model, with every field stated.

    Each limit is a number, :data:`UNBOUNDED` or ``None``; no limit has a default, so a
    leaf that forgets one does not type-check. A leaf's record may not declare
    ``truncates_silently=True`` with an unknown ``max_input_tokens``, since only a known
    limit lets preflight stop the state before the wire cuts it. A composite's record,
    marked ``composite=True``, is derived from its members' and skips that refusal: its
    members' records guard.

    ``feature_kinds`` and ``feature_modalities`` map an opt-in feature to the kinds, or the
    modalities, it adds on this model, whether or not it is on, so they are the same for a
    model whatever is enabled. A feature absent from a map adds none. The constructor
    refuses a key outside ``opt_in_features``, a feature mapped to the empty set, and an
    enabled feature mapped to a kind not in ``kinds`` or a modality not in ``modalities``.

    ``max_image_bytes``, ``max_total_image_bytes``, ``min_image_side`` and
    ``image_media_types`` bear only on a state with an image. The media types are written
    lower-case and canonical, and a leaf's record may not leave them empty beside the image
    modality.

    ``carries_noul_descriptions`` says whether the leaf sends a ``noul``'s ``yes`` and ``no``
    label descriptions, and ``carries_structured_instructions`` whether it sends an
    instruction that is a JSON object or array rather than a string. False means the leaf
    refuses that part, so Fit refuses a request that uses it before any call.
    """

    kinds: frozenset[Kind]
    modalities: frozenset[Modality]
    image_carriers: frozenset[ImageCarrier]
    max_input_tokens: Limit
    max_question_tokens: Limit
    max_body_bytes: Limit
    max_questions: Limit
    max_images: Limit
    max_images_per_question: Limit
    max_image_bytes: Limit
    max_total_image_bytes: Limit
    max_labels: Limit
    max_levels: Limit
    max_total_options: Limit
    min_levels: int | None
    min_image_side: int | None
    image_media_types: frozenset[str] | None
    key_patterns: tuple[str, ...]
    recounts_state_per_question: bool
    returns_refusal: bool
    calibrated: bool
    snapshot_pinnable: bool
    carries_noul_descriptions: bool
    carries_structured_instructions: bool
    truncates_silently: bool | None
    confidence: Mapping[Kind, str]
    rounding: Rounding
    opt_in_features: frozenset[str]
    enabled_features: frozenset[str]
    feature_kinds: Mapping[str, frozenset[Kind]]
    feature_modalities: Mapping[str, frozenset[Modality]]
    composite: bool = False

    def __post_init__(self) -> None:
        for name in _LIMIT_FIELDS:
            limit: object = getattr(self, name)
            if limit is not None and not isinstance(limit, Unbounded):
                if not isinstance(limit, int):
                    raise TypeError(
                        f"Capabilities.{name} must be an int, UNBOUNDED or None, "
                        f"got {type(limit).__name__}"
                    )
                if isinstance(limit, bool) or limit < 0:
                    raise ValueError(
                        f"Capabilities.{name} must be a non-negative int, UNBOUNDED or None"
                    )
        for name in ("min_levels", "min_image_side"):
            least: object = getattr(self, name)
            if least is not None:
                if not isinstance(least, int):
                    raise TypeError(
                        f"Capabilities.{name} must be an int or None, got {type(least).__name__}"
                    )
                if isinstance(least, bool) or least < 0:
                    raise ValueError(f"Capabilities.{name} must be a non-negative int or None")
        _check_media_types(self)
        _check_image_caps(self)
        if not self.composite and self.truncates_silently is True and self.max_input_tokens is None:
            raise ValueError(
                "a leaf that truncates silently must declare max_input_tokens; "
                "nothing else can stop the state before the wire cuts it"
            )
        if not self.enabled_features <= self.opt_in_features:
            raise ValueError("enabled_features must be a subset of opt_in_features")
        _check_feature_map(self, "feature_kinds", self.feature_kinds, self.kinds)
        _check_feature_map(self, "feature_modalities", self.feature_modalities, self.modalities)
        object.__setattr__(self, "confidence", MappingProxyType(dict(self.confidence)))
        object.__setattr__(self, "feature_kinds", _frozen_map(self.feature_kinds))
        object.__setattr__(self, "feature_modalities", _frozen_map(self.feature_modalities))


def _check_media_types(record: Capabilities) -> None:
    """Refuse an ``image_media_types`` that is not a set of strings or holds a type not in
    its lower-case canonical form, ``image/jpg`` among them, and freeze it. A leaf's record
    may not leave it empty beside the image modality, which would refuse every image; a
    composite's derived set may be empty, since its own check never reads it."""
    types = record.image_media_types
    if types is None:
        return
    _check_type("Capabilities", "image_media_types", types, (set, frozenset), "a set or None")
    for media_type in sorted(types, key=str):
        _check_type("Capabilities", "image_media_types[]", media_type, str, "a str")
        if media_type != media_type.lower() or media_type == "image/jpg":
            canonical = "image/jpeg" if media_type.lower() == "image/jpg" else media_type.lower()
            raise ValueError(
                f"image_media_types holds {media_type!r}; write it as {canonical!r}, the form "
                "an image's type is read in"
            )
    object.__setattr__(record, "image_media_types", frozenset(types))
    if not record.composite and not types and "image" in record.modalities:
        raise ValueError(
            "image_media_types is empty beside the image modality; an empty set refuses every image"
        )


def _check_image_caps(record: Capabilities) -> None:
    """Refuse, on a leaf's record, an image count or byte cap of 0 beside the image
    modality, which would refuse every image; a composite's derived cap may be 0, since its
    own check never reads the image fields and a router's count is its members'."""
    if record.composite or "image" not in record.modalities:
        return
    for name in (
        "max_images",
        "max_images_per_question",
        "max_image_bytes",
        "max_total_image_bytes",
    ):
        if getattr(record, name) == 0:
            raise ValueError(
                f"{name} is 0 beside the image modality; a cap of 0 refuses every image"
            )


def _check_feature_map(
    record: Capabilities, name: str, mapping: Mapping[str, frozenset[str]], declared: frozenset[str]
) -> None:
    """Refuse with ``TypeError`` a feature map that is not a mapping or has a value that is not
    a set; then refuse a map whose key is not offered, whose value is empty, or which says
    an enabled feature adds a value the record does not declare."""
    _check_type("Capabilities", name, mapping, Mapping, "a mapping")
    for feature, values in mapping.items():
        _check_type("Capabilities", f"{name}[{feature!r}]", values, (set, frozenset), "a set")
    field_name = name.removeprefix("feature_")
    for feature, values in sorted(mapping.items()):
        if feature not in record.opt_in_features:
            raise ValueError(f"{name} names {feature!r}, which is not in opt_in_features")
        if not values:
            raise ValueError(
                f"{name} maps {feature!r} to the empty set; leave a feature out to say it adds none"
            )
        undeclared = frozenset(values) - declared
        if feature in record.enabled_features and undeclared:
            raise ValueError(
                f"{name} says the enabled feature {feature!r} adds "
                f"{', '.join(sorted(undeclared))}, which {field_name} does not declare"
            )


_T = TypeVar("_T")


def _frozen_map(mapping: Mapping[str, frozenset[_T]]) -> Mapping[str, frozenset[_T]]:
    return MappingProxyType({key: frozenset(values) for key, values in mapping.items()})


# ----------------------------------------------------------------------------------------
# Token estimation
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenEstimate:
    """An estimator's figure for a state and questions, with its stated margin.

    ``margin`` is a fraction of ``tokens``, at least 0: preflight checks
    ``ceil(tokens * (1 + margin))`` against the limit. ``tokens`` is an int and ``margin``
    a finite number; neither may be negative.
    """

    tokens: int
    margin: float

    def __post_init__(self) -> None:
        tokens = cast(object, self.tokens)
        margin = cast(object, self.margin)
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise TypeError(f"TokenEstimate.tokens must be an int, got {type(tokens).__name__}")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise TypeError(f"TokenEstimate.margin must be a number, got {type(margin).__name__}")
        if not _finite(margin):
            raise ValueError(f"TokenEstimate.margin must be finite, got {margin!r}")
        if tokens < 0 or margin < 0:
            raise ValueError("tokens and margin are at least 0")


@runtime_checkable
class TokenEstimator(Protocol):
    """What every leaf names beside its capabilities and preflight calls.

    ``estimate`` returns the tokens the state and the questions will cost together, images
    costed by the estimator itself, with its stated margin. ``questions`` may be empty,
    meaning the state alone; preflight asks for that to check the state for overflow.
    """

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate: ...


# ----------------------------------------------------------------------------------------
# Answers
# ----------------------------------------------------------------------------------------


class Rule(enum.StrEnum):
    """Every rule answer validation runs, plus those a leaf sets itself.

    ``candidate_shape`` marks a candidate shape returned for an admitted kind; the three
    logprob rules mark what the chat leaf's read protocol could not do. A leaf builds those
    faults itself; ``validate`` never sets them.
    """

    missing_key = "missing_key"
    kind_shape = "kind_shape"
    labels_mismatch = "labels_mismatch"
    echo_mismatch = "echo_mismatch"
    not_finite = "not_finite"
    probability_range = "probability_range"
    sum_tolerance = "sum_tolerance"
    chosen_not_max = "chosen_not_max"
    expected_level = "expected_level"
    confidence_range = "confidence_range"
    p_null_range = "p_null_range"
    value_type = "value_type"
    box_range = "box_range"
    box_order = "box_order"
    candidate_shape = "candidate_shape"
    empty_logprobs = "empty_logprobs"
    label_absent_from_logprobs = "label_absent_from_logprobs"
    off_label_mass_above_bound = "off_label_mass_above_bound"


@dataclass(frozen=True, slots=True, kw_only=True)
class RuleFailure:
    """One rule an answer broke, with what was observed and the bound it had to respect.

    For a range rule ``observed`` is the value and ``bound`` the limit it crossed; for a
    tolerance rule ``observed`` is the deviation and ``bound`` the tolerance; for an order
    rule ``observed`` and ``bound`` are the two values out of order.
    """

    rule: Rule
    observed: float | str | None
    bound: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelAnswer:
    """The answer to a ``choice`` or ``score``: one probability per label, in the order asked.

    ``probabilities`` pairs each label name with the provider's probability, as given.
    ``chosen``, ``expected_level`` and an echo of the labels are kept where the provider
    reports them. ``off_label_mass`` and ``derived`` are filled by the chat leaf only: a
    decision model's answer has ``None`` and an empty set.
    """

    probabilities: tuple[tuple[str, float], ...]
    confidence: float | None = None
    chosen: str | None = None
    expected_level: float | None = None
    echo: tuple[str, ...] | None = None
    echo_of: Literal["names", "descriptions"] | None = None
    off_label_mass: float | None = None
    derived: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if (self.echo is None) != (self.echo_of is None):
            raise ValueError("an echo names what it echoes: echo and echo_of go together")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoulAnswer:
    """The answer to a ``noul``: the provider's probability of yes, as given.

    There is no second value and no sum check. ``off_label_mass`` and ``derived`` are the
    chat leaf's; a decision model's answer has ``None`` and an empty set.
    """

    probability: float
    confidence: float | None = None
    off_label_mass: float | None = None
    derived: frozenset[str] = frozenset()


_COMPLEMENT_PRECISION: Final = 400
"""Decimal digits enough to hold ``1 - p`` exactly for every float ``p`` in [0, 1]: a
float's repr writes at most 17 significant digits, the last no smaller than ``1e-324``, so
the difference needs at most 325; the default context's 28 would round it."""


def noul_complement(probability: float) -> float:
    """The probability of no for a ``noul``'s probability of yes: the exact decimal
    difference ``1 - probability``, of the probability as its repr writes it, converted to
    the nearest float, with no decimal rounding, so ``0.7`` gives ``0.3``.

    A bool or a value that is not a number raises ``TypeError``; one that is not finite or
    lies outside 0 to 1 raises ``ValueError``. Nothing is stored on the answer.
    """
    held = cast(object, probability)
    if isinstance(held, bool) or not isinstance(held, (int, float)):
        raise TypeError(f"noul_complement takes a number, got {type(held).__name__}")
    if not _finite(held) or not 0 <= held <= 1:
        raise ValueError(f"noul_complement takes a probability from 0 to 1, got {held!r}")
    value = float(held)
    with decimal.localcontext() as context:
        context.prec = _COMPLEMENT_PRECISION
        return float(decimal.Decimal(1) - decimal.Decimal(repr(value)))


_Scalar: TypeAlias = (
    str | int | float | datetime.date | datetime.time | datetime.datetime | datetime.timedelta
)
Value: TypeAlias = _Scalar | tuple[_Scalar, ...]
"""An extracted value: one scalar of the declared type, or a tuple of them for a list."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Span:
    """Where in the state an extracted value was found.

    ``part`` is the index, among the state's parts, of the text part the value was found in;
    ``start`` and ``end`` are 0-based, end-exclusive offsets in code points, so
    ``text[start:end]`` is the span's text.
    """

    part: int
    start: int
    end: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractionAnswer:
    """The answer to an ``extraction``.

    ``value`` is the provider's value of the declared type, or ``None`` when the state does
    not determine one; ``guess`` is the provider's own guess beside it, which idia never
    fills and never moves into ``value``. ``p_null`` is the provider's probability that no
    value exists, or ``None`` where it gives none, as for a list; idia never computes
    one. A list carries one span per item.
    """

    value: Value | None
    guess: Value | None = None
    confidence: float | None = None
    p_null: float | None = None
    span: Span | tuple[Span, ...] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Box:
    """A region over an image part, each edge a fraction of the image's extent."""

    top: float
    left: float
    bottom: float
    right: float


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxAnswer:
    """The answer to a ``box``: the region and an optional confidence."""

    region: Box
    confidence: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Refusal:
    """The provider declined to answer this question. An answer, never an exception."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Fault:
    """The provider's output for this question broke a rule.

    It records every rule that failed with what was observed and the bound, and the answer
    as mapped (``None`` when nothing could be mapped), so it can be inspected without the
    wire body. It is scoped to one question; the siblings stand.
    """

    key: str
    failures: tuple[RuleFailure, ...]
    mapped: Answer | None


Answer: TypeAlias = LabelAnswer | NoulAnswer | ExtractionAnswer | BoxAnswer | Refusal | Fault
"""Every shape an answer takes."""

MappedAnswer: TypeAlias = Answer
"""What a leaf hands to validation: the wire's output mapped to its kind's shape, or a
fault or refusal the leaf built itself."""

# ----------------------------------------------------------------------------------------
# Responses and their outcomes
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Warning:
    """Something the caller should know that is not a fault."""

    code: str
    message: str
    field: str | None = None


def unknown_options(options: Mapping[str, JsonValue], known: frozenset[str]) -> tuple[Warning, ...]:
    """The one ``unknown-option`` warning naming every key of ``options`` outside
    ``known``, or none. Every leaf puts it on its wire response."""
    unknown = sorted(set(options) - known)
    if not unknown:
        return ()
    return (
        Warning(
            code=WARNING_UNKNOWN_OPTION,
            message=f"options this leaf does not know were ignored: {', '.join(unknown)}",
            field="options",
        ),
    )


def _no_raw_usage() -> dict[str, JsonValue]:
    return {}


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Token counts and passes as the leaf normalizes them from the wire; each 0 is a
    reported 0 and None unreported; the wire's raw figures are beside them. ``passes`` is
    the number of passes the wire reports for the call."""

    input_tokens: int | None
    output_tokens: int | None
    passes: int | None = None
    raw_usage: Mapping[str, JsonValue] = field(default_factory=_no_raw_usage)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_usage", MappingProxyType(dict(self.raw_usage)))


@dataclass(frozen=True, slots=True, kw_only=True)
class FailedAttempt:
    """One call that ended in a provider error, as the dispatcher lists it on an outcome.

    ``started_at`` is wall-clock seconds since the epoch; ``duration`` is a monotonic
    duration in seconds.
    """

    provider: str
    model_requested: str
    error: ProviderError
    request_id: str | None
    started_at: float
    duration: float
    usage: Usage | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WireResponse:
    """What a leaf returns from one call.

    The answers are mapped and validated; the provenance is what only the leaf knows, the
    leaf's own name and model being read off the leaf itself; the raw body travels here and
    no further, since the dispatcher drops it when it builds the outcome.
    """

    answers: Mapping[str, Answer]
    elicitation: Elicitation
    calibrated: bool
    model_reported: str | None = None
    snapshot: str | None = None
    prompt_sent: str | Mapping[str, JsonValue] | None = None
    usage: Usage | None = None
    request_id: str | None = None
    raw_body: JsonValue | bytes | None = None
    warnings: tuple[Warning, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(self.answers)))
        if isinstance(self.prompt_sent, Mapping):
            object.__setattr__(self, "prompt_sent", MappingProxyType(dict(self.prompt_sent)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """What became of one wire request: the keys it carried, its error if it failed, and its
    provenance.

    The leaf's name and model come from the leaf, the rest of its provenance from its wire
    response; latency, start time and the failed attempts are the dispatcher's. A wire
    request that failed has no wire response, so its elicitation and calibration claim are
    unknown. ``started_at`` is wall-clock seconds since the epoch; ``latency`` is a monotonic
    duration in seconds covering the whole wire request, waits and retries included.
    """

    keys: tuple[str, ...]
    error: ProviderError | None
    provider: str
    model_requested: str
    model_reported: str | None
    snapshot: str | None
    elicitation: Elicitation | None
    calibrated: bool | None
    prompt_sent: str | Mapping[str, JsonValue] | None
    usage: Usage | None
    latency: float
    started_at: float
    request_id: str | None
    failed_attempts: tuple[FailedAttempt, ...]
    warnings: tuple[Warning, ...]

    def __post_init__(self) -> None:
        if isinstance(self.prompt_sent, Mapping):
            object.__setattr__(self, "prompt_sent", MappingProxyType(dict(self.prompt_sent)))

    @property
    def attempts(self) -> int:
        """How many calls were made: every failed attempt, plus one when the request was
        answered. The dispatcher lists every failed attempt, so nothing is stored. It is 0
        for a wire request never sent: past the deadline, or after a credential error."""
        return len(self.failed_attempts) + (1 if self.error is None else 0)


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    """The answers to a request, one outcome per wire request in plan order, and the
    warnings the core raised itself. Everything else is read off the outcomes."""

    answers: Mapping[str, Answer]
    outcomes: tuple[Outcome, ...]
    warnings: tuple[Warning, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(self.answers)))


class RequestFailed(IdiaError):
    """Raised when preflight passed and no wire request answered, carrying the full response.

    The message names the first failed outcome in plan order, its provider, model and
    error, and how many outcomes failed; a response with no failed outcome, which only a
    hand-built one has, gives the count of its outcomes alone.
    """

    response: Response

    def __init__(self, response: Response) -> None:
        super().__init__(_request_failed_message(response))
        self.response = response


def _request_failed_message(response: Response) -> str:
    failed = [outcome for outcome in response.outcomes if outcome.error is not None]
    if not failed:
        count = len(response.outcomes)
        return f"no wire request answered ({count} {'outcome' if count == 1 else 'outcomes'})"
    first = failed[0]
    error = first.error
    text = str(error) or type(error).__name__
    origin = f"{first.provider} {first.model_requested}: "
    if len(failed) == 1:
        return f"no wire request answered: {origin}{text} (1 outcome failed)"
    return (
        f"no wire request answered: {origin}{text} ({len(failed)} outcomes failed; the first shown)"
    )


# ----------------------------------------------------------------------------------------
# The two protocols
# ----------------------------------------------------------------------------------------


@runtime_checkable
class Leaf(Protocol):
    """What answers one wire request in one call.

    A leaf speaks one wire API for one model, ``model``, under its ``name``. It sends the
    wire request once, with the deadline as its own timeout, and returns a wire response or
    raises :class:`ProviderError`. It never retries. It is not a provider until the core
    wraps it. The members are read-only: a plain attribute, a property or a frozen
    dataclass field all satisfy them.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    @property
    def estimator(self) -> TokenEstimator: ...

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse: ...


@runtime_checkable
class Provider(Protocol):
    """What takes a request and returns a response, declaring its capabilities.

    The members are read-only: a plain attribute, a property or a frozen dataclass field
    all satisfy them.
    """

    @property
    def capabilities(self) -> Capabilities: ...

    async def answer(self, request: Request) -> Response: ...
