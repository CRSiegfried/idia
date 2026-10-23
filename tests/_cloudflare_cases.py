"""What the Cloudflare tests share: the states and questions each leaf is asked, the
records the Cloudflare leaves declare, written out field by field, and the cassette paths.

Clef's cassettes live under ``tests/cassettes/cloudflare-clef/<clef|clef-flash>/base/`` and
Clairic's under ``tests/cassettes/cloudflare-clairic/clairic/base/``. Every cassette was
recorded with the account id replaced by :data:`ACCOUNT`, which replay builds the leaves
with, so the URL sent matches the one stored.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from idia import (
    UNBOUNDED,
    Capabilities,
    ImagePart,
    JsonValue,
    Kind,
    Label,
    Question,
    Rounding,
    State,
    TextPart,
    WireRequest,
)
from tests._system_one_cases import png

CASSETTES: Final = Path(__file__).parent / "cassettes"
ACCOUNT: Final = "conformance-account"
"""The account id every cassette URL holds in place of the real one."""
TOKEN: Final = "conformance-token"
ACCOUNT_ENV: Final = "CLOUDFLARE_ACCOUNT_ID"
TOKEN_ENV: Final = "CLOUDFLARE_API_TOKEN"
REQUEST_ID_HEADER: Final = "cf-ai-req-id"
FAILURE: Final = "failure"
"""Clef's failure cassette, per model; Clairic's is :data:`FIELD_NAME_PROBE`."""
GONE: Final = "gone"
"""Where a live run keeps a model-gone reply: ``<dir>/gone/gone.json`` under a model's
cassette directory, a subdirectory no ``*.json`` glob of that directory reads."""
REFERENCE_DATE: Final = datetime.date(2026, 10, 21)
CLAIRIC_OPTIONS: Final[Mapping[str, JsonValue]] = {"reference_date": REFERENCE_DATE.isoformat()}
"""The options every Clairic request was recorded with and is replayed with."""


def clef_dir(model: str) -> Path:
    return CASSETTES / "cloudflare-clef" / model / "base"


CLAIRIC_DIR: Final = CASSETTES / "cloudflare-clairic" / "clairic" / "base"

_TEXT: Final = (
    "Riverside Rovers v Harbour Town, Saturday 17 October: the match was called "
    "off after an hour of heavy rain; the sun came out at six, when the ground was "
    "already under water. Ticket refunds come to 4,250.50 pounds. The referees were "
    "Ana Ruiz and Tom Bell, and the pitch inspection took 45 minutes."
)
TEXT: Final = State(parts=(TextPart(text=_TEXT),))
"""The one state both leaves are asked about, so the router's request reaches both."""

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
_COLOURS: Final = (
    Label(name="red", description="Mostly red"),
    Label(name="blue", description="Mostly blue"),
    Label(name="white", description="Mostly white"),
)

CLEF_QUESTIONS: Final[Mapping[Kind, tuple[Question, ...]]] = {
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

IMAGE_REQUEST: Final = WireRequest(
    state=PICTURE,
    questions=(
        Question(key="noul_red", kind="noul", instruction="Is there a red square?"),
        Question(
            key="choice_colour", kind="choice", instruction="The main colour?", labels=_COLOURS
        ),
    ),
)

RECOUNT_STATE: Final = State(parts=(TextPart(text=" ".join([_TEXT] * 20)),))
"""A long state, for reading from usage whether Clef counts it once per request."""
RECOUNT_ONE: Final = WireRequest(state=RECOUNT_STATE, questions=CLEF_QUESTIONS["noul"][:1])
RECOUNT_THREE: Final = WireRequest(
    state=RECOUNT_STATE,
    questions=(
        *CLEF_QUESTIONS["noul"],
        Question(key="noul_played", kind="noul", instruction="Was the match played?"),
    ),
)

CLEF_FAILURE: Final = WireRequest(
    state=TEXT,
    questions=tuple(
        Question(key=f"q{n}", kind="noul", instruction="Did it rain?") for n in range(65)
    ),
)
"""Clef's failure: 65 questions, one past the wire's cap."""

ALIAS: Final = "alias"
"""The cassette, per model, of a request whose keys no wire pattern takes, so the leaf
sends ``q0..qN`` or ``f0..fN``: Clef's choice request and Clairic's provider request,
each key written ``q:match/<key>`` by :func:`aliased`, which no wire pattern takes."""


def aliased(wire_request: WireRequest) -> WireRequest:
    """``wire_request`` with each key written ``q:match/<key>``, which neither the Clef id
    pattern nor the Clairic field-name pattern takes."""
    return dataclasses.replace(
        wire_request,
        questions=tuple(
            dataclasses.replace(question, key=f"q:match/{question.key}")
            for question in wire_request.questions
        ),
    )


ORDER: Final = "order"
"""The subdirectory, per model, of the question-order probe: the same request sent twice
unchanged and once with its questions reversed, kept out of the leaf's ``*.json`` glob."""
ORDER_NAMES: Final = ("order-a-1", "order-a-2", "order-b")
"""The probe's cassettes in the order sent: the request twice, then reversed."""

CLEF_ORDER: Final = WireRequest(
    state=TEXT,
    questions=(
        CLEF_QUESTIONS["choice"][0],
        CLEF_QUESTIONS["score"][1],
        CLEF_QUESTIONS["noul"][1],
        CLEF_QUESTIONS["choice"][1],
    ),
)
"""Clef's question-order request: the weather and the mood, whose recorded answers were
the least saturated (the mood's low level at 0.73 on clef and 0.89 on clef-flash), with
the sun and the outcome beside them, four questions of three kinds over one state."""


def order_requests(wire_request: WireRequest) -> dict[str, WireRequest]:
    """The question-order probe's requests by cassette name: ``wire_request`` twice, then
    with its questions reversed."""
    turned = dataclasses.replace(wire_request, questions=wire_request.questions[::-1])
    return dict(zip(ORDER_NAMES, (wire_request, wire_request, turned), strict=True))


PROVIDER_KIND: Final[Kind] = "choice"
"""The kind of Clef's share of the provider half's request."""


def clef_request(kind: Kind) -> WireRequest:
    return WireRequest(state=TEXT, questions=CLEF_QUESTIONS[kind])


def singles(wire_request: WireRequest) -> dict[str, WireRequest]:
    """A request packed one question per wire request: a cassette name per question."""
    return {
        f"single-{question.key}": WireRequest(
            state=wire_request.state, questions=(question,), options=wire_request.options
        )
        for question in wire_request.questions
    }


EXTRACTION: Final = WireRequest(
    state=TEXT,
    questions=(
        Question(
            key="home_team", kind="extraction", instruction="The home team.", value_type="text"
        ),
        Question(
            key="refunds",
            kind="extraction",
            instruction="The ticket refunds, in pounds, to two decimals.",
            value_type="number",
        ),
        Question(
            key="match_date",
            kind="extraction",
            instruction="The date the match was due to be played.",
            value_type="date",
        ),
        Question(
            key="referees",
            kind="extraction",
            instruction="The referees' names.",
            value_type="text",
            is_list=True,
        ),
        Question(
            key="inspection",
            kind="extraction",
            instruction="How long the pitch inspection took.",
            value_type="duration",
        ),
        Question(
            key="attendance",
            kind="extraction",
            instruction="The number of spectators at the ground.",
            value_type="number",
        ),
    ),
    options=CLAIRIC_OPTIONS,
)
"""Clairic's success: text, a number with decimals and a unit, a date written without a
year, a list, a duration, and a field the state does not determine."""

CLAIRIC_PROVIDER: Final = WireRequest(
    state=TEXT, questions=EXTRACTION.questions[:2], options=CLAIRIC_OPTIONS
)
"""Clairic's share of the provider half's request, and its kind cassette: the live run
answered its text field with a null value and a guess."""
CLAIRIC_KIND: Final = "provider"

EMOJI: Final = WireRequest(
    state=State(
        parts=(
            TextPart(
                text="🌧️ Storm warning 🌧️: the final was moved to Northgate Park, 🏟️ gates at noon."
            ),
        )
    ),
    questions=(
        Question(key="venue", kind="extraction", instruction="The new venue.", value_type="text"),
    ),
    options=CLAIRIC_OPTIONS,
)
"""A target after characters outside the BMP, so code points and UTF-16 units differ."""

TIMES: Final = WireRequest(
    state=State(
        parts=(
            TextPart(
                text="Kick-off was due at 15:00 on 17 October 2026 and was moved to "
                "2026-10-18T14:30."
            ),
        )
    ),
    questions=(
        Question(
            key="kickoff", kind="extraction", instruction="The kick-off time.", value_type="time"
        ),
        Question(
            key="moved_to",
            kind="extraction",
            instruction="When the match was moved to.",
            value_type="datetime",
        ),
    ),
    options=CLAIRIC_OPTIONS,
)
"""The two value types the model page does not show."""

_HOME_TEAM: Final = Question(
    key="home_team", kind="extraction", instruction="The home team.", value_type="text"
)
# Clairic counts about 4.01 characters per token on this text: its 239,381-character
# state (806 copies) cost 59,646 state tokens on the wire (2026-10-22). The estimator keeps
# text/3, which over-counts: the safe direction. 1,128 copies, about 1.4 times as many,
# are 335,015 characters, about 83,500 wire tokens, past the 65,536-token context.
_LONG_COPIES: Final = 1128

CLAIRIC_FAILURE: Final = WireRequest(
    state=State(parts=(TextPart(text=" ".join([_TEXT] * _LONG_COPIES)),)),
    questions=(_HOME_TEAM,),
    options=CLAIRIC_OPTIONS,
)
"""A request Fit refuses, sent to the leaf directly: a text state of 335,015 characters,
about 83,500 tokens at the wire's rate, past its 65,536-token context for one pass. It was
drawn as Clairic's failure, but the wire answered it (U7): its answer is
:data:`PAST_CONTEXT_TRUNCATED`. The 239,381-character state before it was answered whole,
at 59,781 input tokens, under the context."""

PAST_CONTEXT_DIR: Final = CLAIRIC_DIR / "past-context"
"""Evidence whose request Fit refuses, kept out of the leaf's own ``*.json`` glob."""
PAST_CONTEXT_TRUNCATED: Final = PAST_CONTEXT_DIR / "truncated.json"
"""Evidence kept on purpose: :data:`CLAIRIC_FAILURE` sent to the leaf directly on
2026-10-22. The wire truncated the state to fit its context and answered, HTTP 200 with
``truncated: true``: one pass, 65,523 input tokens, 65,388 of them state, and
``home_team`` read as "Riverside Rovers" at the state's start. Fit refuses the request,
so no dispatch ever sends it; a test that names this directory replays it to the leaf.
Recorded before ``write_cassette`` kept the order sent, its request body is stored with
sorted keys and matches no request the leaf sends, so the test replays it by name over a
single-response wire and compares the body sent by value: its sent order was never kept,
and a recorded body is never reordered by hand."""

FIELD_NAME_PROBE: Final = "field-name-probe"
"""A failure probe sent outside the leaf: a field named ``1bad``, which the leaf would
send as ``f0``, to show the wire refusing a name its pattern does not take (422, code
3030). It is also Clairic's failure cassette, the one its harness replays for a failure,
since the wire answers :data:`CLAIRIC_FAILURE` rather than refusing it."""
FIELD_NAME_REQUEST: Final = WireRequest(
    state=TEXT,
    questions=(dataclasses.replace(_HOME_TEAM, key="1bad"),),
    options=CLAIRIC_OPTIONS,
)
FIELD_NAME_BODY: Final[dict[str, object]] = {
    "model": "clairic",
    "state": _TEXT,
    "fields": {"1bad": {"type": "text", "instructions": "The home team."}},
    "reference_date": REFERENCE_DATE.isoformat(),
}
"""The raw request of :data:`FIELD_NAME_PROBE`, the body the leaf sent for it before it
aliased keys."""

QUESTION_ID_PROBE: Final = "question-id-probe"
"""A failure probe sent to ``clef`` outside the leaf: one question under the id
``match:intent``, whose ``:`` is the one character outside the wire's id pattern, to show
the wire refusing an id the leaf would send as ``q0``. Its body is the control's, a
recorded success, with only the id changed."""
QUESTION_ID_CONTROL: Final = "single-choice_weather"
"""The recorded success whose body the question-id probe repeats under another id."""
QUESTION_ID_KEY: Final = "match:intent"
QUESTION_ID_REQUEST: Final = WireRequest(
    state=TEXT,
    questions=(
        dataclasses.replace(
            singles(clef_request("choice"))[QUESTION_ID_CONTROL].questions[0], key=QUESTION_ID_KEY
        ),
    ),
)
QUESTION_ID_BODY: Final[dict[str, object]] = {
    "model": "clef",
    "state": _TEXT,
    "questions": {
        QUESTION_ID_KEY: {
            "type": "choice",
            "instructions": "The weather?",
            "criteria": {"rain": "It rained.", "sun": None, "snow": "It snowed."},
        }
    },
}
"""The raw request of :data:`QUESTION_ID_PROBE`: the control's body with its one id
replaced by :data:`QUESTION_ID_KEY`, keys in the order the leaf sends them."""

ESTIMATE_BODY: Final[dict[str, object]] = {
    "model": "clairic",
    "state": _TEXT,
    "fields": {
        "crowd": {
            "type": "estimate",
            "instructions": "The number of spectators at the ground.",
            "min": 0,
            "max": 100000,
        }
    },
}
"""The raw request, sent outside the leaf, whose reply is the candidate-shape cassette."""
CANDIDATE: Final = "candidate-estimate"
CANDIDATE_TARGET: Final = "refunds"
"""The answer of Clairic's kind cassette, the provider half's request, that the
candidate shape replaces."""


# ----------------------------------------------------------------------------------------
# The records
# ----------------------------------------------------------------------------------------

CLEF_ROUNDING: Final = Rounding(probability_decimals=4, score_decimals=4)
"""Four decimals, as every recorded probability, confidence and expected level shows."""
_CLEF_TOKENS: Final = {"clef": 65_536, "clef-flash": 24_000}


def clef_record(model: str) -> Capabilities:
    tokens = _CLEF_TOKENS[model]
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score"}),
        modalities=frozenset({"text", "json", "image"}),
        image_carriers=frozenset({"array"}),
        max_input_tokens=tokens,
        max_question_tokens=tokens,
        max_body_bytes=13 * 1024 * 1024,
        max_questions=64,
        max_images=4,
        max_images_per_question=4,
        max_image_bytes=4 * 1024 * 1024,
        max_total_image_bytes=8 * 1024 * 1024,
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
        rounding=CLEF_ROUNDING,
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
    )


def clairic_record() -> Capabilities:
    return Capabilities(
        kinds=frozenset({"extraction"}),
        modalities=frozenset({"text", "image"}),
        image_carriers=frozenset({"array"}),
        max_input_tokens=65_536,
        max_question_tokens=65_536,
        max_body_bytes=16 * 1024 * 1024,
        max_questions=UNBOUNDED,
        max_images=4,
        max_images_per_question=4,
        max_image_bytes=4 * 1024 * 1024,
        max_total_image_bytes=8 * 1024 * 1024,
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


def router_record(model: str) -> Capabilities:
    """The record the router must derive over the two leaves: the most restrictive limit
    of each, the label caps from Clef alone, no key pattern, the modalities' union."""
    clef = clef_record(model)
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score", "extraction"}),
        modalities=frozenset({"text", "json", "image"}),
        image_carriers=frozenset({"array"}),
        max_input_tokens=clef.max_input_tokens,
        max_question_tokens=clef.max_question_tokens,
        max_body_bytes=13 * 1024 * 1024,
        max_questions=64,
        max_images=4,
        max_images_per_question=4,
        max_image_bytes=4 * 1024 * 1024,
        max_total_image_bytes=8 * 1024 * 1024,
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
        confidence={
            "choice": "undocumented",
            "score": "undocumented",
            "extraction": "undocumented",
        },
        rounding=CLEF_ROUNDING,
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
        composite=True,
    )
