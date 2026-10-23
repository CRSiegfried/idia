"""What the System One tests share: the wire request per configuration and kind, the
test image, the cassette paths, and a reader for a cassette's encoded values.

A configuration is a model and the opt-in features it was built with; each has its own
cassette set under ``tests/cassettes/system-one/<model>/<features or base>/``.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import struct
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import idia
from idia import (
    UNBOUNDED,
    Box,
    BoxAnswer,
    Capabilities,
    ImagePart,
    Kind,
    Label,
    Question,
    Response,
    Rounding,
    State,
    TextPart,
    WireRequest,
)
from tests._http import features_dir

CASSETTES: Final = Path(__file__).parent / "cassettes" / "system-one"

TEXT: Final = State(
    parts=(
        TextPart(
            text="The match was called off after an hour of heavy rain; "
            "the sun came out at six, when the ground was already under water."
        ),
    )
)


def png(width: int = 320, height: int = 320) -> bytes:
    """A white PNG with a red square in its top-left quarter's lower right and a blue
    square in its bottom-right quarter, the same bytes on every run."""

    def colour(x: int, y: int) -> bytes:
        if width // 4 <= x < width // 2 and height // 4 <= y < height // 2:
            return b"\xff\x00\x00"
        if 5 * width // 8 <= x < 7 * width // 8 and 5 * height // 8 <= y < 7 * height // 8:
            return b"\x00\x00\xff"
        return b"\xff\xff\xff"

    rows = b"".join(b"\x00" + b"".join(colour(x, y) for x in range(width)) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


PICTURE: Final = State(
    parts=(
        TextPart(text="A test card: a white field with two coloured squares."),
        ImagePart(data=png(), media_type="image/png"),
    )
)

_WEATHER: Final = (
    Label(name="rain", description="It rained."),
    Label(name="sun"),
    Label(name="snow", description="It snowed."),
)
_OUTCOME: Final = (Label(name="played"), Label(name="called_off"))
_SEVERITY: Final = (
    Label(name="none", description="No rain"),
    Label(name="light", description="Light rain"),
    Label(name="heavy", description="Heavy rain"),
)
_MOOD: Final = (Label(name="low"), Label(name="mid"), Label(name="high"))
_SHAPES: Final = (Label(name="squares"), Label(name="circles"))
_COLOURS: Final = (
    Label(name="red", description="Mostly red"),
    Label(name="blue", description="Mostly blue"),
    Label(name="white", description="Mostly white"),
)

TEXT_QUESTIONS: Final[Mapping[Kind, tuple[Question, ...]]] = {
    "noul": (
        Question(key="noul_rain", kind="noul", instruction="Did it rain?"),
        Question(key="noul_sun", kind="noul", instruction="Did the sun come out?"),
    ),
    "choice": (
        Question(key="choice_weather", kind="choice", instruction="The weather?", labels=_WEATHER),
        Question(key="choice_outcome", kind="choice", instruction="The match?", labels=_OUTCOME),
    ),
    "score": (
        Question(key="score_rain", kind="score", instruction="How hard?", labels=_SEVERITY),
        Question(key="score_mood", kind="score", instruction="The mood?", labels=_MOOD),
    ),
}

PICTURE_QUESTIONS: Final[Mapping[Kind, tuple[Question, ...]]] = {
    "noul": (
        Question(key="noul_red", kind="noul", instruction="Is there a red square?"),
        Question(key="noul_green", kind="noul", instruction="Is there a green square?"),
    ),
    "choice": (
        Question(
            key="choice_colour", kind="choice", instruction="The main colour?", labels=_COLOURS
        ),
        Question(key="choice_shapes", kind="choice", instruction="The shapes?", labels=_SHAPES),
    ),
    "score": (
        Question(key="score_red", kind="score", instruction="How much red?", labels=_MOOD),
        Question(key="score_rain", kind="score", instruction="How hard?", labels=_SEVERITY),
    ),
    "box": (
        Question(key="box_red", kind="box", instruction="The red square."),
        Question(key="box_blue", kind="box", instruction="The blue square."),
    ),
}

IMAGE_SIDE_PROBES: Final[tuple[tuple[int, int], ...]] = (
    (280, 280),
    (279, 279),
    (279, 320),
    (320, 279),
    (280, 320),
    (320, 280),
)
"""The minimum-side probe's sizes, width by height, in the order sent to ``jev-1.14.0``."""


def side_probe(width: int, height: int) -> WireRequest:
    """The minimum-side probe at one size: the test card at ``width`` by ``height`` and one
    noul question."""
    picture = State(
        parts=(PICTURE.parts[0], ImagePart(data=png(width, height), media_type="image/png"))
    )
    return WireRequest(state=picture, questions=PICTURE_QUESTIONS["noul"][:1])


BOX_PROOF: Final = "provider-box_red"
"""The cassette of the box proof: the test card and one box question, asked through
``idia.provider("system-one", model="jev-1.14.0", images=True)``."""
BOX_PROOF_REQUEST: Final = WireRequest(state=PICTURE, questions=PICTURE_QUESTIONS["box"][:1])


def _holds(box: Box, at: float) -> bool:
    """Whether ``box`` holds the point ``(at, at)``, as fractions of each side."""
    return box.top <= at <= box.bottom and box.left <= at <= box.right


def check_red_square(response: Response) -> None:
    """The box proof's answer: a box on the red square, which spans 0.25 to 0.5 of each
    side, that holds its centre and not the blue square's, from a pinned snapshot."""
    [answer] = response.answers.values()
    assert isinstance(answer, BoxAnswer), answer
    box = answer.region
    assert 0 <= box.top < box.bottom <= 1, box
    assert 0 <= box.left < box.right <= 1, box
    assert _holds(box, 0.375), box
    assert not _holds(box, 0.75), box
    assert answer.confidence is None or 0 <= answer.confidence <= 1, answer.confidence
    [outcome] = response.outcomes
    assert outcome.error is None, outcome.error
    assert outcome.snapshot == "jev-1.14.0"
    assert outcome.request_id


ELEVEN_LEVELS: Final = WireRequest(
    state=TEXT,
    questions=(
        Question(
            key="score_eleven",
            kind="score",
            instruction="How much rain, from 0 to 10?",
            labels=tuple(Label(name=str(level)) for level in range(11)),
        ),
    ),
)
"""The failure every configuration records: eleven levels, one past the wire's cap."""


@dataclasses.dataclass(frozen=True, slots=True)
class Configuration:
    """One model built with one set of features, and its cassette set."""

    model: str
    features: frozenset[str]

    @property
    def directory(self) -> Path:
        return CASSETTES / self.model / features_dir(self.features)

    @property
    def kinds(self) -> tuple[Kind, ...]:
        return tuple(sorted(self._questions))

    @property
    def _questions(self) -> Mapping[Kind, tuple[Question, ...]]:
        return PICTURE_QUESTIONS if "images" in self.features else TEXT_QUESTIONS

    def wire_request(self, kind: Kind) -> WireRequest:
        state = PICTURE if "images" in self.features else TEXT
        return WireRequest(state=state, questions=self._questions[kind])

    def singles(self) -> dict[str, WireRequest]:
        """The provider half's request packed one question per wire request: a cassette
        name for each question of the ``choice`` request, and its wire request."""
        whole = self.wire_request(PROVIDER_KIND)
        return {
            f"single-{question.key}": WireRequest(state=whole.state, questions=(question,))
            for question in whole.questions
        }

    def options(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "images": "images" in self.features,
            "long_context": "long_context" in self.features,
        }


TEXT_ONLY: Final = Configuration("jev-1.13.0", frozenset())
WITH_IMAGES: Final = Configuration("jev-1.14.0", frozenset({"images"}))
CONFIGURATIONS: Final = (TEXT_ONLY, WITH_IMAGES)
FAILURE: Final = "failure"
SIDE_PROBE: Final = "probe-image-side"
"""The subdirectory of ``jev-1.14.0``'s image cassettes holding the minimum-side probe,
one cassette per size, kept out of the leaf's ``*.json`` glob."""


def side_probe_path(width: int, height: int) -> Path:
    return WITH_IMAGES.directory / SIDE_PROBE / f"{width}x{height}.json"


PROVIDER_KIND: Final[Kind] = "choice"
"""The kind of the request the provider half sends."""


def record(model: str, features: frozenset[str]) -> Capabilities:
    """The record a System One leaf must declare, written out field by field."""
    images = "images" in features
    long_context = "long_context" in features
    kinds: set[Kind] = {"noul", "choice", "score"}
    confidence: dict[Kind, str] = {
        "choice": "normalized_top_probability",
        "score": "mad_concentration",
    }
    boxes = model in ("jev-1.14.0", "jev-preview")
    if images and boxes:
        kinds.add("box")
        confidence["box"] = "undocumented"
    return Capabilities(
        kinds=frozenset(kinds),
        modalities=frozenset({"text", "json", "image"} if images else {"text", "json"}),
        image_carriers=frozenset({"placeholder"}) if images else frozenset(),
        max_input_tokens=262_000 if long_context else 64_000,
        max_question_tokens=32_000,
        max_body_bytes=32 * 1024 * 1024,
        max_questions=UNBOUNDED,
        max_images=UNBOUNDED if images else 0,
        max_images_per_question=UNBOUNDED if images else 0,
        max_image_bytes=16 * 1024 * 1024,
        max_total_image_bytes=32 * 1024 * 1024,
        max_labels=255,
        max_levels=10,
        max_total_options=UNBOUNDED,
        min_levels=1,
        min_image_side=280,
        image_media_types=frozenset({"image/png", "image/jpeg"}),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=False,
        calibrated=True,
        snapshot_pinnable=True,
        carries_noul_descriptions=True,
        carries_structured_instructions=True,
        truncates_silently=False,
        confidence=confidence,
        rounding=ROUNDING,
        opt_in_features=frozenset({"long_context", "images"}),
        enabled_features=features,
        feature_kinds={"images": frozenset[Kind]({"box"})} if boxes else {},
        feature_modalities={"images": frozenset({"image"})},
    )


ROUNDING: Final = Rounding(probability_decimals=2, score_decimals=2)


# ----------------------------------------------------------------------------------------
# Cassettes
# ----------------------------------------------------------------------------------------


def load(configuration: Configuration, name: str) -> dict[str, Any]:
    """One cassette of a configuration, as stored."""
    return cast(
        dict[str, Any],
        json.loads((configuration.directory / f"{name}.json").read_text(encoding="utf-8")),
    )


def decode(value: object) -> Any:
    """A value the cassette format encoded back into idia's types: an object tagged
    with a type name becomes that dataclass, an array a tuple, and an image's data bytes."""
    if isinstance(value, list):
        return tuple(decode(item) for item in cast(list[object], value))
    if not isinstance(value, dict):
        return value
    fields = {key: decode(item) for key, item in cast(dict[str, object], value).items()}
    name = fields.pop("type", None)
    cls = getattr(idia, name, None) if isinstance(name, str) else None
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        if name is not None:
            fields["type"] = name
        return fields
    if cls is ImagePart:
        fields["data"] = base64.b64decode(fields["data"])
    if "derived" in fields:
        fields["derived"] = frozenset(fields["derived"])
    return cls(**fields)


WORDING: Final = "wording"
"""The cassette, per model, of a question's wording as the wire takes it: a ``noul`` with
``yes`` and ``no`` descriptions, sent as ``criteria``, and a ``noul`` whose instruction is
a JSON object, sent as it is, in one request."""


def wording_request() -> WireRequest:
    """The request :data:`WORDING` records, over the text state."""
    return WireRequest(
        state=TEXT,
        questions=(
            Question(
                key="described",
                kind="noul",
                instruction="Was the match played to the end?",
                labels=(
                    Label(name="yes", description="The match ran its full time."),
                    Label(name="no", description="The match was stopped or called off early."),
                ),
            ),
            Question(
                key="structured",
                kind="noul",
                instruction={
                    "question": "Did the weather change what happened at the `event`?",
                    "event": "the match",
                },
            ),
        ),
    )
