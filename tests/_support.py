"""Builders the core tests share."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

from idia import (
    UNBOUNDED,
    Capabilities,
    Deadline,
    Label,
    Question,
    Request,
    Rounding,
    State,
    TextPart,
    TokenEstimate,
    TokenEstimator,
)

SEPARATE_EXTRAS: frozenset[str] = frozenset({"gliner"})
"""The extras CI installs in a job of their own and leaves out of the jobs that install every
other one: a test may skip for a missing dependency of one of these, and of no other extra."""

# One tick of ``time.monotonic()`` on Windows before Python 3.13 (15.6 ms, rounded up), where
# asyncio may also fire a timer one clock resolution early: a test's lower bound on a
# measured time allows this much slack.
TICK = 0.02

TWO_DECIMALS = Rounding(probability_decimals=2, score_decimals=2)
UNROUNDED = Rounding(probability_decimals=None, score_decimals=None)


def capabilities(**overrides: Any) -> Capabilities:
    """A full leaf record, every field stated, with ``overrides`` applied."""
    fields: dict[str, Any] = {
        "kinds": frozenset({"noul", "choice", "score"}),
        "modalities": frozenset({"text"}),
        "image_carriers": frozenset(),
        "max_input_tokens": 65_536,
        "max_question_tokens": None,
        "max_body_bytes": UNBOUNDED,
        "max_questions": UNBOUNDED,
        "max_images": 0,
        "max_images_per_question": 0,
        "max_image_bytes": UNBOUNDED,
        "max_total_image_bytes": UNBOUNDED,
        "max_labels": 20,
        "max_levels": 20,
        "max_total_options": UNBOUNDED,
        "min_levels": 2,
        "min_image_side": 0,
        "image_media_types": None,
        "key_patterns": (),
        "recounts_state_per_question": False,
        "returns_refusal": False,
        "calibrated": True,
        "snapshot_pinnable": True,
        "carries_noul_descriptions": True,
        "carries_structured_instructions": True,
        "truncates_silently": False,
        "confidence": {"choice": "top-probability"},
        "rounding": TWO_DECIMALS,
        "opt_in_features": frozenset({"long-context"}),
        "enabled_features": frozenset(),
        "feature_kinds": {},
        "feature_modalities": {},
    }
    fields.update(overrides)
    return Capabilities(**fields)


def noul(key: str = "q") -> Question:
    return Question(key=key, kind="noul", instruction="Is it so?")


def choice(key: str = "q", names: tuple[str, ...] = ("red", "green", "blue")) -> Question:
    labels = tuple(Label(name=name, description=f"the colour {name}") for name in names)
    return Question(key=key, kind="choice", instruction="Which colour?", labels=labels)


def score(key: str = "q", levels: int = 3) -> Question:
    labels = tuple(Label(name=f"level {index}") for index in range(levels))
    return Question(key=key, kind="score", instruction="How much?", labels=labels)


def extraction(key: str = "q", value_type: str = "text", is_list: bool = False) -> Question:
    return Question(
        key=key,
        kind="extraction",
        instruction="The name?",
        value_type=value_type,  # type: ignore[arg-type]
        is_list=is_list,
    )


def box(key: str = "q") -> Question:
    return Question(key=key, kind="box", instruction="Where is the cat?")


def text_state(text: str = "some text") -> State:
    return State(parts=(TextPart(text=text),))


class FixedEstimator:
    """An estimator that always answers the same figure."""

    def __init__(self, tokens: int = 10) -> None:
        self.tokens = tokens

    def estimate(self, state: State, questions: object) -> TokenEstimate:
        return TokenEstimate(tokens=self.tokens, margin=0.1)


def request(*questions: Question, state: State | None = None, seconds: float = 60.0) -> Request:
    """A request over a text state, one noul by default, due ``seconds`` from now."""
    return Request(
        state=state if state is not None else text_state(),
        questions=questions or (noul(),),
        deadline=Deadline.after(seconds),
    )


class TextChars:
    """An estimator that counts one token per character of the state's text, nothing for
    the questions, with no margin, so a policy's shortened state can be seen to fit."""

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        text = sum(len(part.text) for part in state.parts if isinstance(part, TextPart))
        return TokenEstimate(tokens=text, margin=0.0)


class Shorten:
    """A named overflow policy that keeps the first ``keep`` characters of each text part."""

    def __init__(self, keep: int) -> None:
        self.keep = keep

    @property
    def name(self) -> str:
        return "shorten"

    def apply(self, state: State, limit: int, estimator: TokenEstimator) -> State:
        parts = tuple(
            TextPart(text=part.text[: self.keep]) if isinstance(part, TextPart) else part
            for part in state.parts
        )
        return State(parts=parts)


def png_header(width: int = 320, height: int = 320, size: int | None = None) -> bytes:
    """A PNG's signature and IHDR chunk for an image of ``width`` by ``height`` pixels,
    padded with zeros to ``size`` bytes when given: enough for a reader of the type and
    the size, not an image a decoder could draw."""
    fields = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    crc = zlib.crc32(b"IHDR" + fields).to_bytes(4, "big")
    data = b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + fields + crc
    if size is not None:
        assert size >= len(data), f"a PNG header is {len(data)} bytes"
        data += bytes(size - len(data))
    return data
