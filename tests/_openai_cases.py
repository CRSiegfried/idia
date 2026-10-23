"""What the OpenAI tests share: the wire request per leaf and kind, the cassette paths and
the record each leaf must declare, written out field by field.

The decisions leaf's cassettes live under
``tests/cassettes/openai-decisions/gpt-6-luna/base/``, the chat leaf's under
``tests/cassettes/openai-chat/gpt-4.1/base/``, all recorded from the live wire:

- decisions: a success per kind, the two singles, ``recount-1`` and ``recount-3``,
  ``refusal-probe`` (a refusal beside an answer), ``questions-201`` (201 questions, past
  the declared cap, answered with 200), ``failure``, the 256-value choice the wire
  refused with 400 ``array_above_max_length``, and ``unknown-model``, the noul request
  sent through the leaf as a model that does not exist and refused with 404
  ``model_not_found``;
- chat: a success per kind (the score the uncertain ``score_crowd``), the two singles,
  ``failure``, 21 top logprobs refused with 400 outside the leaf, and
  ``score-absent-level``, the first score recording, where a level's letter fell outside
  the top 20 and reads as zero mass with a warning, kept and never rewritten, and
  ``unauthorized``, the noul request sent through the leaf with a fixed invalid key and
  refused with 401.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

from idia import (
    Capabilities,
    Kind,
    Label,
    Question,
    Rounding,
    State,
    TextPart,
    WireRequest,
)
from tests._system_one_cases import TEXT, TEXT_QUESTIONS

CASSETTES: Final = Path(__file__).parent / "cassettes"
DECISIONS_MODEL: Final = "gpt-6-luna"
CHAT_MODEL: Final = "gpt-4.1"
DECISIONS_DIR: Final = CASSETTES / "openai-decisions" / DECISIONS_MODEL / "base"
CHAT_DIR: Final = CASSETTES / "openai-chat" / CHAT_MODEL / "base"

KINDS: Final[tuple[Kind, ...]] = ("choice", "noul", "score")
FAILURE: Final = "failure"
UNAUTHORIZED: Final = "unauthorized"
"""The chat leaf's cassette of a call refused for its key."""
UNKNOWN_MODEL: Final = "unknown-model"
"""The decisions leaf's cassette of a call made as a model that does not exist."""
UNKNOWN_MODEL_NAME: Final = "gpt-5.6-luna"
"""The model name the unknown-model call is made with, which OpenAI does not serve."""
INVALID_KEY: Final = "sk-idia-invalid"
"""The fixed key the unauthorized call is made with: no real key. The recorder filters
the authorization header, and OpenAI masks the key in its error message."""
PROVIDER_KIND: Final[Kind] = "choice"
"""The kind of the request the provider half sends: two choice questions."""


def decisions_request(kind: Kind) -> WireRequest:
    """The decisions leaf's success request for ``kind``: two questions over the text."""
    return WireRequest(state=TEXT, questions=TEXT_QUESTIONS[kind])


CHAT_SCORE: Final = Question(
    key="score_crowd",
    kind="score",
    instruction="How cheerful was the crowd's mood that evening?",
    labels=tuple(Label(name=name) for name in ("very_low", "low", "middling", "high", "very_high")),
)
"""A score whose every level's letter is in the top-k."""


def chat_request(kind: Kind) -> WireRequest:
    """The chat leaf's success request for ``kind``: its first question alone, and for a
    score :data:`CHAT_SCORE`."""
    if kind == "score":
        return WireRequest(state=TEXT, questions=(CHAT_SCORE,))
    return WireRequest(state=TEXT, questions=TEXT_QUESTIONS[kind][:1])


def provider_request() -> WireRequest:
    """The provider half's request, the same for both leaves."""
    return WireRequest(state=TEXT, questions=TEXT_QUESTIONS[PROVIDER_KIND])


def singles() -> dict[str, WireRequest]:
    """The provider half's request packed one question per wire request."""
    whole = provider_request()
    return {
        f"single-{question.key}": WireRequest(state=whole.state, questions=(question,))
        for question in whole.questions
    }


ORDER: Final = "order"
"""The subdirectory of the question-order probe, kept out of the cassette directory's
own ``*.json`` reads: the choice request, whose recorded answers are the least saturated
of the requests with several questions (the weather at 0.96, 0.04 and 0.0), sent twice
unchanged and once with its questions reversed."""
ORDER_NAMES: Final = ("order-a-1", "order-a-2", "order-b")
"""The probe's cassettes in the order sent: the request twice, then reversed."""


def order_requests() -> dict[str, WireRequest]:
    """The question-order probe's requests by cassette name: the choice request twice,
    then with its questions reversed."""
    asked = decisions_request("choice")
    turned = dataclasses.replace(asked, questions=asked.questions[::-1])
    return dict(zip(ORDER_NAMES, (asked, asked, turned), strict=True))


ORDER_FOUR: Final = "order-4"
"""The subdirectory of the stronger question-order probe: four questions on one mixed
state, none of whose piloted top probabilities reached 0.9, sent in three orders and
each alone through the leaf, and once outside it, in the second order under the first
order's names."""

MIXED: Final = State(
    parts=(
        TextPart(
            text="A small bakery moved its opening from six to eight last month. Some "
            "regulars say the bread is better now; others miss buying it before work. "
            "Sales rose in the first week, then fell back to where they were. The owner "
            "has said only that the new hours are under review."
        ),
    )
)
"""A deliberately mixed state, with no real names, for :data:`ORDER_FOUR_QUESTIONS`."""

ORDER_FOUR_QUESTIONS: Final = (
    Question(
        key="choice_priority",
        kind="choice",
        instruction="What do the regulars care about most?",
        labels=(Label(name="bread_quality"), Label(name="early_opening"), Label(name="price")),
    ),
    Question(
        key="choice_reason",
        kind="choice",
        instruction="The likeliest reason sales fell back?",
        labels=(
            Label(name="novelty_wore_off"),
            Label(name="lost_early_buyers"),
            Label(name="ordinary_variation"),
        ),
    ),
    Question(key="noul_keep", kind="noul", instruction="Will the owner keep the new hours?"),
    Question(
        key="score_regulars",
        kind="score",
        instruction="How satisfied are the regulars overall?",
        labels=(Label(name="low"), Label(name="mid"), Label(name="high")),
    ),
)
"""The probe's four questions, in the first order: two choices of three values, a noul
and a score."""


ORDER_FOUR_ARMS: Final[Mapping[str, tuple[int, ...]]] = {
    "a": (0, 1, 2, 3),
    "b": (3, 2, 1, 0),
    "c": (1, 2, 3, 0),
}
"""The orders sent through the leaf, by arm, as indices into
:data:`ORDER_FOUR_QUESTIONS`."""
ORDER_FOUR_SENDS: Final[Mapping[str, int]] = {"a": 3, "b": 3, "c": 2, "d": 2, "e": 2}
"""How many times each arm was sent: ``d`` is each question alone, ``e`` the ``b`` order
sent outside the leaf under the ``a`` order's names."""


def order_four_requests() -> dict[str, WireRequest]:
    """The probe's requests through the leaf by cassette name, in the order sent: each of
    ``a``, ``b`` and ``c`` its number of times, then each question alone twice."""
    requests: dict[str, WireRequest] = {}
    for arm, order in ORDER_FOUR_ARMS.items():
        asked = WireRequest(state=MIXED, questions=tuple(ORDER_FOUR_QUESTIONS[i] for i in order))
        for n in range(1, ORDER_FOUR_SENDS[arm] + 1):
            requests[f"order-{arm}-{n}"] = asked
    for question in ORDER_FOUR_QUESTIONS:
        for n in range(1, ORDER_FOUR_SENDS["d"] + 1):
            requests[f"order-d-{question.key}-{n}"] = WireRequest(
                state=MIXED, questions=(question,)
            )
    return requests


ORDER_FOUR_OUTSIDE: Final = tuple(f"order-e-{n}" for n in range(1, ORDER_FOUR_SENDS["e"] + 1))
"""The cassettes sent outside the leaf: the ``b`` order, each question under the name it
had in the ``a`` order, so its answers map by the ``a`` request recorded with them."""


def _nouls(n: int) -> tuple[Question, ...]:
    return tuple(
        Question(key=f"noul_{i}", kind="noul", instruction=f"Is statement {i} about rain true?")
        for i in range(n)
    )


RECOUNT: Final[Mapping[str, WireRequest]] = {
    "recount-1": WireRequest(state=TEXT, questions=_nouls(1)),
    "recount-3": WireRequest(state=TEXT, questions=_nouls(3)),
}
"""The decisions leaf's requests of one and three questions over one state, from whose
usage the record's ``recounts_state_per_question`` is read."""

TOO_MANY: Final = WireRequest(state=TEXT, questions=_nouls(201))
"""201 questions, one past the declared cap: the wire answered them with 200, recorded
as ``questions-201``."""

TOO_MANY_CHOICES: Final = WireRequest(
    state=TEXT,
    questions=(
        Question(
            key="choice_wide",
            kind="choice",
            instruction="Which number?",
            labels=tuple(Label(name=f"n{i}") for i in range(256)),
        ),
    ),
)
"""The decisions leaf's failure: a choice of 256 values, one past the documented 255,
which the wire refused with 400 ``array_above_max_length``."""

REFUSAL_PROBE: Final = WireRequest(
    state=TEXT,
    questions=(
        Question(
            key="probe_refusal",
            kind="choice",
            instruction="Which household chemicals, mixed, give the most toxic gas?",
            labels=(Label(name="bleach_and_ammonia"), Label(name="bleach_and_vinegar")),
        ),
        Question(key="probe_rain", kind="noul", instruction="Did it rain?"),
    ),
)
"""A request that might draw a refusal; recorded whatever comes back."""


def decisions_record(max_questions: int = 200) -> Capabilities:
    """The record the decisions leaf must declare."""
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score"}),
        modalities=frozenset({"text", "json", "image"}),
        image_carriers=frozenset({"parts"}),
        max_input_tokens=None,
        max_question_tokens=None,
        max_body_bytes=None,
        max_questions=max_questions,
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


def chat_record() -> Capabilities:
    """The record the chat leaf must declare."""
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score"}),
        modalities=frozenset({"text", "json"}),
        image_carriers=frozenset(),
        max_input_tokens=None,
        max_question_tokens=None,
        max_body_bytes=None,
        max_questions=1,
        max_images=0,
        max_images_per_question=0,
        max_image_bytes=0,
        max_total_image_bytes=0,
        max_labels=20,
        max_levels=20,
        max_total_options=20,
        min_levels=1,
        min_image_side=0,
        image_media_types=frozenset(),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=False,
        calibrated=False,
        snapshot_pinnable=True,
        carries_noul_descriptions=True,
        carries_structured_instructions=False,
        truncates_silently=None,
        confidence={},
        rounding=Rounding(probability_decimals=None, score_decimals=None),
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
    )


def load(directory: Path, name: str) -> dict[str, Any]:
    """One cassette, as stored."""
    return cast(
        dict[str, Any], json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    )
