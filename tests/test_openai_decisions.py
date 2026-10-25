"""The OpenAI decisions leaf: its record, the wire body, the reading of each kind and of a
refusal, its provenance, its cassettes, and its place in the registry."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx2
import pytest
from openai import AsyncOpenAI

import idia
from idia import (
    WARNING_EXTRA_ANSWER_KEY,
    WARNING_NO_SNAPSHOT,
    Answer,
    Deadline,
    Dispatcher,
    DoesNotFit,
    Fail,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    Label,
    LabelAnswer,
    Leaf,
    NoulAnswer,
    Provider,
    ProviderError,
    Question,
    Refusal,
    Request,
    Rule,
    RuleFailure,
    State,
    TextPart,
    WireRequest,
    WireResponse,
    providers,
)
from idia.fit import _preflight
from idia.providers.openai_decisions import OpenAIDecisions
from tests._http import body_key, replay_headers
from tests._openai_cases import (
    DECISIONS_DIR,
    FAILURE,
    KINDS,
    ORDER,
    ORDER_FOUR,
    ORDER_FOUR_ARMS,
    ORDER_FOUR_OUTSIDE,
    ORDER_FOUR_QUESTIONS,
    ORDER_NAMES,
    RECOUNT,
    REFUSAL_PROBE,
    TOO_MANY,
    TOO_MANY_CHOICES,
    UNKNOWN_MODEL,
    UNKNOWN_MODEL_NAME,
    decisions_record,
    decisions_request,
    load,
    order_four_requests,
    order_requests,
    singles,
)
from tests._system_one_cases import TEXT, TEXT_QUESTIONS, decode, png

_USAGE = {"input_tokens": 40, "output_tokens": 0, "total_tokens": 40}


class Wire:
    """An SDK client over a mock transport that answers every request with one response
    and keeps what it was sent."""

    def __init__(self, body: object, status: int = 200, headers: Mapping[str, str] = {}) -> None:
        self.body = body
        self.status = status
        self.headers = {"x-request-id": "req-1", **headers}
        self.sent: list[httpx2.Request] = []
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))
        self.client = AsyncOpenAI(api_key="test-key", http_client=http)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(request)
        if isinstance(self.body, bytes):
            return httpx2.Response(self.status, content=self.body, headers=self.headers)
        return httpx2.Response(self.status, json=self.body, headers=self.headers)

    def leaf(self, **options: Any) -> OpenAIDecisions:
        return OpenAIDecisions(client=self.client, **options)

    @property
    def body_sent(self) -> Any:
        [request] = self.sent
        return json.loads(request.content)


def _answering(answers: Sequence[object], model: str = "gpt-6-luna") -> Wire:
    return Wire({"answers": answers, "model": model, "usage": _USAGE})


def _call(leaf: OpenAIDecisions, wire_request: WireRequest) -> WireResponse:
    return asyncio.run(leaf.call(wire_request, Deadline.after(5)))


def _raised(leaf: OpenAIDecisions, wire_request: WireRequest) -> ProviderError:
    with pytest.raises(ProviderError) as raised:
        _call(leaf, wire_request)
    return raised.value


def _one(question: Question, state: State = TEXT) -> WireRequest:
    return WireRequest(state=state, questions=(question,))


_NOUL = Question(key="n", kind="noul", instruction="Did it rain?")


def _deep(levels: int) -> JsonValue:
    value: JsonValue = 1
    for _ in range(levels):
        value = (value,)
    return value


def test_a_kind_the_leaf_does_not_answer_is_a_provider_error_not_a_key_error() -> None:
    extraction = Question(key="x", kind="extraction", instruction="Who?", value_type="text")
    error = _raised(_answering([]).leaf(), _one(extraction))
    assert "does not answer 'extraction'" in error.message


def test_a_json_part_nested_too_deeply_never_reaches_the_wire() -> None:
    # A value nested too deep never reaches the wire: it is refused when the part is built.
    with pytest.raises(ValueError, match=r"^JsonPart\.value nests too deep$"):
        JsonPart(value=_deep(100_000))


_CHOICE = TEXT_QUESTIONS["choice"][0]  # rain (described), sun, snow (described)
_SCORE = TEXT_QUESTIONS["score"][1]  # low, mid, high


def _choice_answer(**changes: object) -> dict[str, object]:
    answer: dict[str, object] = {
        "type": "choice",
        "name": "q0",
        "choice": "rain",
        "confidence": 0.8,
        "probabilities": [
            {"value": "sun", "probability": 0.1},
            {"value": "rain", "probability": 0.85},
            {"value": "snow", "probability": 0.05},
        ],
    }
    return {**answer, **changes}


def _score_answer(**changes: object) -> dict[str, object]:
    answer: dict[str, object] = {
        "type": "score",
        "name": "q0",
        "score": 0.4,
        "confidence": 0.5,
        "probabilities": [
            {"value": 1, "label": "mid", "probability": 0.2},
            {"value": 0, "label": "low", "probability": 0.7},
            {"value": 2, "label": "high", "probability": 0.1},
        ],
    }
    return {**answer, **changes}


# ----------------------------------------------------------------------------------------
# The record and construction
# ----------------------------------------------------------------------------------------


def test_the_record_states_every_field() -> None:
    assert Wire({}).leaf().capabilities == decisions_record()


def test_a_caller_may_raise_the_question_cap_or_replace_the_record() -> None:
    assert Wire({}).leaf(max_questions=400).capabilities == decisions_record(400)
    changed = dataclasses.replace(decisions_record(), max_levels=10)
    assert Wire({}).leaf(capabilities=changed).capabilities is changed


def test_name_model_and_its_place_as_a_leaf() -> None:
    leaf = Wire({}).leaf()
    assert (leaf.name, leaf.model) == ("openai-decisions", "gpt-6-luna")
    assert repr(leaf) == "OpenAIDecisions(model='gpt-6-luna')"
    assert isinstance(leaf, Leaf)
    assert not isinstance(leaf, Provider)
    assert Wire({}).leaf(model="gpt-6-luna-decisions").model == "gpt-6-luna-decisions"


def test_an_empty_model_is_refused() -> None:
    with pytest.raises(ValueError, match="model"):
        Wire({}).leaf(model="")


# ----------------------------------------------------------------------------------------
# The body
# ----------------------------------------------------------------------------------------


def test_the_body_per_kind() -> None:
    wire = _answering([])
    _call(wire.leaf(), WireRequest(state=TEXT, questions=(_NOUL, _CHOICE, _SCORE)))
    [request] = wire.sent
    assert str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert wire.body_sent == {
        "model": "gpt-6-luna",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": TEXT.parts[0].text}]}  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]
        ],
        "questions": [
            {"type": "predicate", "name": "q0", "instructions": "Did it rain?"},
            {
                "type": "choice",
                "name": "q1",
                "instructions": "The weather?",
                "choices": [
                    {"value": "rain", "description": "It rained."},
                    {"value": "sun"},
                    {"value": "snow", "description": "It snowed."},
                ],
            },
            {
                "type": "score",
                "name": "q2",
                "instructions": "The mood?",
                "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}],
            },
        ],
    }


def test_a_json_part_is_compact_text_and_an_image_a_data_url() -> None:
    picture = png(8, 8)
    state = State(
        parts=(
            TextPart(text="Look:"),
            JsonPart(value={"é": (1, 2)}),
            ImagePart(data=picture, media_type="image/PNG"),
        )
    )
    wire = _answering([])
    response = _call(wire.leaf(), _one(_NOUL, state))
    content = wire.body_sent["input"][0]["content"]
    assert content[:2] == [
        {"type": "input_text", "text": "Look:"},
        {"type": "input_text", "text": '{"é":[1,2]}'},
    ]
    encoded = base64.b64encode(picture).decode()
    assert content[2] == {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}
    assert response.prompt_sent is not None
    [message] = response.prompt_sent["input"]  # type: ignore[index]  # pyright: ignore[reportIndexIssue]
    assert message["content"][2] == {  # type: ignore[index]  # pyright: ignore[reportIndexIssue]
        "type": "input_image",
        "media_type": "image/png",
        "bytes": len(picture),
    }


@pytest.mark.parametrize(
    ("data", "media_type"),
    [(png(2, 2), "image/png"), (b"\xff\xd8\xff\xe0rest", "image/jpeg")],
)
def test_an_image_without_a_media_type_is_read_from_its_bytes(data: bytes, media_type: str) -> None:
    wire = _answering([])
    _call(wire.leaf(), _one(_NOUL, State(parts=(ImagePart(data=data),))))
    url = wire.body_sent["input"][0]["content"][0]["image_url"]
    assert url.startswith(f"data:{media_type};base64,")


def test_an_untyped_image_neither_png_nor_jpeg_does_not_fit_and_makes_no_call() -> None:
    wire = _answering([])
    wire_request = _one(_NOUL, State(parts=(ImagePart(data=b"GIF89a"),)))
    request = Request(
        state=wire_request.state, questions=wire_request.questions, deadline=Deadline.after(5)
    )
    with pytest.raises(DoesNotFit) as raised:
        asyncio.run(Dispatcher(wire.leaf()).answer(request))
    assert raised.value.check == "image_media_types"
    assert wire.sent == []


def test_a_leaf_called_past_fit_sends_an_untyped_image_for_the_wire_to_judge() -> None:
    wire = _answering([])
    _call(wire.leaf(), _one(_NOUL, State(parts=(ImagePart(data=b"GIF89a"),))))
    url = wire.body_sent["input"][0]["content"][0]["image_url"]
    assert url.startswith("data:application/octet-stream;base64,")


def test_the_leaf_sends_the_media_type_fit_judged() -> None:
    wire = _answering([])
    image = ImagePart(data=png(2, 2), media_type="image/gif")
    _call(wire.leaf(), _one(_NOUL, State(parts=(image,))))
    url = wire.body_sent["input"][0]["content"][0]["image_url"]
    assert url.startswith("data:image/png;base64,")


def test_a_state_that_is_not_json_is_refused_before_sending() -> None:
    wire = _answering([])
    error = _raised(wire.leaf(), _one(_NOUL, State(parts=(JsonPart(value=float("inf")),))))
    assert not error.is_retryable
    assert wire.sent == []


def test_the_prompt_sent_is_the_body_when_there_is_no_image() -> None:
    wire = _answering([])
    response = _call(wire.leaf(), _one(_NOUL))
    assert isinstance(response.prompt_sent, Mapping)
    assert json.loads(json.dumps(dict(response.prompt_sent))) == wire.body_sent


# ----------------------------------------------------------------------------------------
# Reading the answers
# ----------------------------------------------------------------------------------------


def test_a_noul_is_its_predicates_probability() -> None:
    wire = _answering([{"type": "predicate", "name": "q0", "probability": 0.64}])
    assert _call(wire.leaf(), _one(_NOUL)).answers == {"n": NoulAnswer(probability=0.64)}


def test_a_choice_in_label_order_with_its_choice_and_confidence() -> None:
    response = _call(_answering([_choice_answer()]).leaf(), _one(_CHOICE))
    assert response.answers[_CHOICE.key] == LabelAnswer(
        probabilities=(("rain", 0.85), ("sun", 0.1), ("snow", 0.05)),
        confidence=0.8,
        chosen="rain",
    )


def test_a_score_by_level_index_with_its_labels_echoed_by_name() -> None:
    response = _call(_answering([_score_answer()]).leaf(), _one(_SCORE))
    assert response.answers[_SCORE.key] == LabelAnswer(
        probabilities=(("low", 0.7), ("mid", 0.2), ("high", 0.1)),
        confidence=0.5,
        expected_level=0.4,
        echo=("low", "mid", "high"),
        echo_of="names",
    )


def test_a_level_label_echoed_differently_is_a_fault() -> None:
    answer = _score_answer()
    answer["probabilities"] = [
        {"value": 0, "label": "Low", "probability": 0.7},
        {"value": 1, "label": "mid", "probability": 0.2},
        {"value": 2, "label": "high", "probability": 0.1},
    ]
    fault = _call(_answering([answer]).leaf(), _one(_SCORE)).answers[_SCORE.key]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.echo_mismatch]


def test_a_level_index_out_of_range_keeps_its_number_for_validation_to_fault() -> None:
    answer = _score_answer()
    answer["probabilities"] = [
        {"value": 0, "label": "low", "probability": 0.7},
        {"value": 1, "label": "mid", "probability": 0.2},
        {"value": 7, "label": "high", "probability": 0.1},
    ]
    fault = _call(_answering([answer]).leaf(), _one(_SCORE)).answers[_SCORE.key]
    assert isinstance(fault, Fault)
    assert fault.failures[0] == RuleFailure(
        rule=Rule.labels_mismatch, observed="low,mid,7", bound=None
    )


@pytest.mark.parametrize(
    ("changes", "observed"),
    [
        ({"choice": True}, "bool"),
        ({"probabilities": [{"value": False, "probability": 1.0}]}, "bool"),
        ({"probabilities": [{"value": 3, "probability": 1.0}]}, "int"),
    ],
    ids=["choice", "probability", "number"],
)
def test_a_choice_value_that_is_not_a_string_is_a_kind_shape_fault(
    changes: dict[str, object], observed: str
) -> None:
    fault = _call(_answering([_choice_answer(**changes)]).leaf(), _one(_CHOICE)).answers[
        _CHOICE.key
    ]
    assert fault == Fault(
        key=_CHOICE.key,
        failures=(RuleFailure(rule=Rule.kind_shape, observed=observed, bound=None),),
        mapped=None,
    )


def test_a_repeated_name_keeps_the_first_answer_and_warns() -> None:
    wire = _answering(
        [
            {"type": "predicate", "name": "q0", "probability": 0.2},
            {"type": "predicate", "name": "q0", "probability": 0.9},
        ]
    )
    response = _call(wire.leaf(), _one(_NOUL))
    assert response.answers == {"n": NoulAnswer(probability=0.2)}
    [warning] = [w for w in response.warnings if w.code == WARNING_EXTRA_ANSWER_KEY]
    assert warning.field == "n"
    assert "repeats" in warning.message


def test_a_label_the_wire_added_comes_after_the_asked_ones() -> None:
    answer = _choice_answer(
        probabilities=[
            {"value": "hail", "probability": 0.1},
            {"value": "rain", "probability": 0.9},
        ]
    )
    fault = _call(_answering([answer]).leaf(), _one(_CHOICE)).answers[_CHOICE.key]
    assert isinstance(fault, Fault)
    assert isinstance(fault.mapped, LabelAnswer)
    assert fault.mapped.probabilities == (("rain", 0.9), ("hail", 0.1))


def test_a_refusal_is_an_answer_and_its_siblings_stand() -> None:
    wire = _answering(
        [{"type": "refusal", "name": "q0"}, {"type": "predicate", "name": "q1", "probability": 0.3}]
    )
    other = Question(key="m", kind="noul", instruction="Sun?")
    response = _call(wire.leaf(), WireRequest(state=TEXT, questions=(_CHOICE, other)))
    assert response.answers == {_CHOICE.key: Refusal(), "m": NoulAnswer(probability=0.3)}


def test_answers_are_matched_by_name_not_position() -> None:
    other = Question(key="m", kind="noul", instruction="Sun?")
    wire = _answering(
        [
            {"type": "predicate", "name": "q1", "probability": 0.3},
            {"type": "predicate", "name": "q0", "probability": 0.9},
        ]
    )
    response = _call(wire.leaf(), WireRequest(state=TEXT, questions=(_NOUL, other)))
    assert response.answers == {"n": NoulAnswer(probability=0.9), "m": NoulAnswer(probability=0.3)}


@pytest.mark.parametrize(
    "stray",
    [
        {"type": "predicate", "name": "q9", "probability": 0.5},
        {"type": "predicate", "name": None, "probability": 0.5},
        "not an answer",
    ],
    ids=["unassigned", "unnamed", "not-an-object"],
)
def test_an_answer_with_no_assigned_name_is_dropped_with_a_warning(stray: object) -> None:
    response = _call(_answering([stray]).leaf(), _one(_NOUL))
    fault = response.answers["n"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.missing_key]
    assert any(w.code == WARNING_EXTRA_ANSWER_KEY for w in response.warnings)


@pytest.mark.parametrize(
    ("question", "answer", "observed"),
    [
        (_NOUL, {"type": "choice", "name": "q0"}, "choice"),
        (_NOUL, {"type": "predicate", "name": "q0", "probability": "high"}, "str"),
        (_NOUL, {"type": "predicate", "name": "q0", "probability": True}, "bool"),
        (_NOUL, {"type": "predicate", "name": "q0", "probability": 10**400}, "int beyond float"),
        (_CHOICE, _choice_answer(probabilities={"rain": 1.0}), "dict"),
        (_CHOICE, _choice_answer(probabilities=[1.0]), "float"),
        (_CHOICE, _choice_answer(confidence="x"), "str"),
        (_SCORE, _score_answer(probabilities=[{"value": "0", "label": "low"}]), "str"),
        (_SCORE, _score_answer(probabilities=[{"value": True, "label": "low"}]), "bool"),
        (_SCORE, _score_answer(probabilities=[{"value": 0, "label": 0}]), "int"),
        (_SCORE, _score_answer(score=None), "NoneType"),
    ],
)
def test_a_wrong_shape_is_that_questions_kind_shape_fault(
    question: Question, answer: dict[str, object], observed: str
) -> None:
    wire = Wire(json.dumps({"answers": [answer], "model": "gpt-6-luna", "usage": _USAGE}).encode())
    fault = _call(wire.leaf(), _one(question)).answers[question.key]
    assert isinstance(fault, Fault)
    assert fault.failures == (RuleFailure(rule=Rule.kind_shape, observed=observed, bound=None),)


# ----------------------------------------------------------------------------------------
# Provenance and the top level
# ----------------------------------------------------------------------------------------


def test_the_provenance_only_the_leaf_knows() -> None:
    body = {"answers": [{"type": "predicate", "name": "q0", "probability": 0.5}]}
    wire = Wire({**body, "model": "gpt-6-luna", "usage": _USAGE})
    response = _call(wire.leaf(), _one(_NOUL))
    assert response.elicitation == "passthrough"
    assert response.calibrated is False
    assert (response.model_reported, response.snapshot) == ("gpt-6-luna", None)
    assert [w.code for w in response.warnings] == [WARNING_NO_SNAPSHOT]
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (40, 0)
    assert dict(response.usage.raw_usage) == _USAGE
    assert response.request_id == "req-1"
    assert isinstance(response.raw_body, Mapping)


@pytest.mark.parametrize(
    "body",
    [
        {"model": "gpt-6-luna", "usage": {}},
        {"model": "gpt-6-luna", "usage": {}, "answers": {}},
        {"answers": [], "usage": {}},
        {"answers": [], "model": "", "usage": {}},
        {"answers": [], "model": "gpt-6-luna", "usage": []},
    ],
)
def test_a_body_failing_the_top_level_schema_is_not_retryable(body: object) -> None:
    error = _raised(Wire(body).leaf(), _one(_NOUL))
    assert not error.is_retryable
    assert error.request_id == "req-1"


# ----------------------------------------------------------------------------------------
# The cassettes
# ----------------------------------------------------------------------------------------


def _replay(name: str) -> tuple[Wire, OpenAIDecisions, WireRequest]:
    cassette = load(DECISIONS_DIR, name)
    response = cassette["response"]
    headers = replay_headers(response["headers"])
    wire = Wire(response["body"], status=response["status"], headers=headers)
    wire_request = decode(cassette["wire_request"])
    assert isinstance(wire_request, WireRequest)
    return wire, wire.leaf(), wire_request


_CASES = [
    *((kind, decisions_request(kind)) for kind in KINDS),
    *singles().items(),
    *RECOUNT.items(),
    ("refusal-probe", REFUSAL_PROBE),
]
"""The cassettes of the live run that fit one wire request."""


@pytest.mark.parametrize(("name", "asked"), _CASES, ids=[name for name, _ in _CASES])
def test_each_cassette_is_the_body_sent_and_its_answers_as_recorded(
    name: str, asked: WireRequest
) -> None:
    cassette = load(DECISIONS_DIR, name)
    wire, leaf, wire_request = _replay(name)
    assert wire_request == asked
    response = _call(leaf, wire_request)
    assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
    assert response.answers == decode(cassette["expected"])
    assert not any(isinstance(answer, Fault) for answer in response.answers.values())
    assert cassette["request"]["headers"]["authorization"] == "<filtered>"


def test_the_question_order_probe_replays_as_recorded() -> None:
    """The question-order probe, recorded 2026-10-22: the choice request twice, then with
    its two questions reversed, so each question went under the other's name. All three
    answered identically: the weather ``rain`` 0.96, ``sun`` 0.04, ``snow`` 0.0 at
    confidence 0.94, the outcome ``called_off`` 1.0; the largest move of any probability
    or confidence, unchanged or reversed, is 0, at the wire's two decimals. The moves are
    evidence, not a contract. Each cassette is replayed alone, since the first two share a
    body."""
    assert sorted(path.name for path in (DECISIONS_DIR / ORDER).iterdir()) == [
        f"{name}.json" for name in ORDER_NAMES
    ]
    sent: list[list[str]] = []
    for name, asked in order_requests().items():
        cassette = load(DECISIONS_DIR, f"{ORDER}/{name}")
        wire, leaf, wire_request = _replay(f"{ORDER}/{name}")
        assert wire_request == asked
        response = _call(leaf, wire_request)
        assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
        assert response.answers == decode(cassette["expected"])
        sent.append([question["instructions"] for question in wire.body_sent["questions"]])
    assert sent[0] == sent[1] == sent[2][::-1] != sent[2]


_ORDER_FOUR_ANSWERS: Final[Mapping[str, Answer]] = {
    "choice_priority": LabelAnswer(
        probabilities=(("bread_quality", 0.26), ("early_opening", 0.73), ("price", 0.01)),
        confidence=0.6,
        chosen="early_opening",
    ),
    "choice_reason": LabelAnswer(
        probabilities=(
            ("novelty_wore_off", 0.15),
            ("lost_early_buyers", 0.26),
            ("ordinary_variation", 0.59),
        ),
        confidence=0.38,
        chosen="ordinary_variation",
    ),
    "noul_keep": NoulAnswer(probability=0.5),
    "score_regulars": LabelAnswer(
        probabilities=(("low", 0.25), ("mid", 0.75), ("high", 0.0)),
        confidence=0.63,
        expected_level=0.75,
        echo=("low", "mid", "high"),
        echo_of="names",
    ),
}
"""What every call of the four-question order probe answered, in every arm."""


def test_the_four_question_order_probe_replays_as_recorded() -> None:
    """The stronger question-order probe, recorded 2026-10-22 on a mixed state: four
    questions, none piloted at a top probability of 0.9 or more, in the orders ``a``
    (0 1 2 3, three times), ``b`` (3 2 1 0, three times), ``c`` (1 2 3 0, twice), each
    alone (``d``, twice each), and ``e``, the ``b`` order sent outside the leaf under the
    ``a`` order's names (twice), which separates the name from the position.

    Every one of the 18 calls answered alike: ``choice_priority`` bread_quality 0.26,
    early_opening 0.73, price 0.01 at confidence 0.6; ``choice_reason`` novelty_wore_off
    0.15, lost_early_buyers 0.26, ordinary_variation 0.59 at 0.38; ``noul_keep`` 0.5;
    ``score_regulars`` low 0.25, mid 0.75, high 0.0 at 0.63. The largest move of any
    probability or confidence, within ``a`` and from ``a`` to ``b``, ``c`` and ``d`` and
    from ``b`` to ``e``, is 0 for every question at the wire's two decimals: no dependence
    on order, name or company passes the 0.02 threshold. Usage 555 input tokens for each
    four-question call; alone 175, 181, 214 and 171. The moves are evidence, not a
    contract. Each cassette is replayed alone, since the repeats share a body."""
    directory = DECISIONS_DIR / ORDER_FOUR
    through = order_four_requests()
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        f"{name}.json" for name in [*through, *ORDER_FOUR_OUTSIDE]
    )
    for name, asked in through.items():
        cassette = load(DECISIONS_DIR, f"{ORDER_FOUR}/{name}")
        wire, leaf, wire_request = _replay(f"{ORDER_FOUR}/{name}")
        assert wire_request == asked
        response = _call(leaf, wire_request)
        assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
        assert response.answers == decode(cassette["expected"])
        assert response.answers == {q.key: _ORDER_FOUR_ANSWERS[q.key] for q in asked.questions}
    first = through["order-a-1"]
    for name in ORDER_FOUR_OUTSIDE:
        cassette = load(DECISIONS_DIR, f"{ORDER_FOUR}/{name}")
        wire, leaf, wire_request = _replay(f"{ORDER_FOUR}/{name}")
        assert wire_request == first
        response = _call(leaf, wire_request)
        outside = cassette["request"]["body"]["questions"]
        assert outside == [wire.body_sent["questions"][i] for i in ORDER_FOUR_ARMS["b"]]
        assert [q["name"] for q in outside] == ["q3", "q2", "q1", "q0"]
        assert response.answers == decode(cassette["expected"]) == _ORDER_FOUR_ANSWERS
        assert cassette["request"]["headers"]["authorization"] == "<filtered>"
    assert [q.key for q in ORDER_FOUR_QUESTIONS] == list(_ORDER_FOUR_ANSWERS)


def test_201_questions_were_answered_and_preflight_still_packs_at_the_declared_cap() -> None:
    _, leaf, wire_request = _replay("questions-201")
    assert wire_request == TOO_MANY
    response = _call(leaf, wire_request)
    assert len(response.answers) == 201
    assert not any(isinstance(answer, Fault) for answer in response.answers.values())
    assert any(isinstance(answer, Refusal) for answer in response.answers.values())
    request = Request(
        state=TOO_MANY.state, questions=TOO_MANY.questions, deadline=Deadline.after(5)
    )
    plan = _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert [len(w.questions) for w in plan.wire_requests] == [200, 1]


def test_256_values_are_refused_by_the_wire_and_by_preflight_first() -> None:
    _, leaf, wire_request = _replay(FAILURE)
    assert wire_request == TOO_MANY_CHOICES
    error = _raised(leaf, wire_request)
    assert error.status_code == 400
    assert not error.is_retryable
    assert error.request_id
    assert "array_above_max_length" in error.message
    request = Request(
        state=wire_request.state, questions=wire_request.questions, deadline=Deadline.after(5)
    )
    with pytest.raises(DoesNotFit) as refused:
        _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert refused.value.check == "max_labels"


def test_an_unknown_model_is_sent_as_given_and_its_404_is_not_retried() -> None:
    cassette = load(DECISIONS_DIR, UNKNOWN_MODEL)
    response = cassette["response"]
    wire = Wire(
        response["body"], status=response["status"], headers=replay_headers(response["headers"])
    )
    wire_request = decode(cassette["wire_request"])
    assert wire_request == decisions_request("noul")
    error = _raised(wire.leaf(model=UNKNOWN_MODEL_NAME), wire_request)
    assert (error.status_code, error.is_retryable, error.retry_after) == (404, False, None)
    assert error.request_id == response["headers"]["x-request-id"]
    assert error.message.startswith("HTTP 404: ")
    assert "model_not_found" in error.message
    assert wire.body_sent == cassette["request"]["body"]
    assert wire.body_sent["model"] == UNKNOWN_MODEL_NAME
    assert cassette["expected"] is None
    assert cassette["request"]["headers"]["authorization"] == "<filtered>"


def test_a_refusal_was_recorded_beside_an_answer() -> None:
    _, leaf, wire_request = _replay("refusal-probe")
    response = _call(leaf, wire_request)
    assert response.answers["probe_refusal"] == Refusal()
    assert isinstance(response.answers["probe_rain"], NoulAnswer)


def test_the_state_is_counted_once_per_request() -> None:
    def used(name: str) -> int:
        return load(DECISIONS_DIR, name)["response"]["body"]["usage"]["input_tokens"]

    # Two questions together cost their two singles less about the state's own tokens.
    together = used("choice")
    apart = used("single-choice_weather") + used("single-choice_outcome")
    state = Wire({}).leaf().estimator.estimate(TEXT, ()).tokens - 100
    assert apart - together >= state
    assert not Wire({}).leaf().capabilities.recounts_state_per_question


def test_the_recorded_probabilities_are_at_two_decimals() -> None:
    for name, _ in _CASES:
        body = load(DECISIONS_DIR, name)["response"]["body"]
        for answer in body["answers"]:
            figures = [answer.get("probability"), answer.get("score"), answer.get("confidence")]
            figures += [entry["probability"] for entry in answer.get("probabilities", [])]
            for figure in figures:
                assert figure is None or round(figure, 2) == figure, (name, figure)


def test_the_estimate_covers_the_recorded_usage() -> None:
    estimator = Wire({}).leaf().estimator
    probe = [f"{ORDER_FOUR}/{name}" for name in [*order_four_requests(), *ORDER_FOUR_OUTSIDE]]
    for name in [*(name for name, _ in _CASES), "questions-201", *probe]:
        cassette = load(DECISIONS_DIR, name)
        wire_request = decode(cassette["wire_request"])
        estimate = estimator.estimate(wire_request.state, wire_request.questions)
        used = cassette["response"]["body"]["usage"]["input_tokens"]
        assert used <= estimate.tokens * (1 + estimate.margin), name


@pytest.mark.parametrize(("name", "asked"), _CASES, ids=[name for name, _ in _CASES])
def test_every_recorded_request_plans_one_wire_request(name: str, asked: WireRequest) -> None:
    leaf = Wire({}).leaf()
    request = Request(state=asked.state, questions=asked.questions, deadline=Deadline.after(5))
    plan = _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert len(plan.wire_requests) == 1


# ----------------------------------------------------------------------------------------
# The registry and the extra
# ----------------------------------------------------------------------------------------


def test_the_registry_names_the_leaf_and_its_extra() -> None:
    entry = providers._REGISTRY["openai-decisions"]
    assert entry.target == "idia.providers.openai_decisions:OpenAIDecisions"
    assert entry.extra == "openai"
    assert providers._EXTRAS["openai"] == ("openai", "httpx2")


def test_provider_builds_and_wraps_it() -> None:
    wire = _answering([{"type": "predicate", "name": "q0", "probability": 0.6}])
    provider = idia.provider("openai-decisions", client=wire.client)
    assert isinstance(provider, Dispatcher)
    request = Request(state=TEXT, questions=(_NOUL,), deadline=Deadline.after(5))
    response = asyncio.run(provider.answer(request))
    assert response.answers == {"n": NoulAnswer(probability=0.6)}
    [outcome] = response.outcomes
    assert (outcome.provider, outcome.model_requested, outcome.snapshot) == (
        "openai-decisions",
        "gpt-6-luna",
        None,
    )


_WITHOUT_OPENAI = """
import sys
sys.modules["openai"] = None  # a blocked import raises ModuleNotFoundError
import idia
for name in ("openai-decisions", "openai-chat"):
    try:
        idia.provider(name, api_key="k", model="m")
    except ModuleNotFoundError as error:
        assert error.name == "openai", error.name
        print(error)
"""


def test_without_openai_idia_imports_and_each_leaf_asks_for_its_extra() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _WITHOUT_OPENAI], capture_output=True, text=True, check=True
    )
    assert done.stdout.count('pip install "idia[openai]"') == 2


def test_a_label_with_no_description_is_sent_by_name_alone() -> None:
    question = Question(
        key="c", kind="choice", instruction="?", labels=(Label(name="a"), Label(name="b"))
    )
    wire = _answering([])
    _call(wire.leaf(), _one(question))
    assert wire.body_sent["questions"][0]["choices"] == [{"value": "a"}, {"value": "b"}]
