"""The OpenAI chat leaf: the ``logprobs`` read protocol over synthetic logprob payloads,
its faults, its record and provenance, its cassettes, and its place in the registry."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import math
import operator
from collections.abc import Iterable, Mapping
from typing import Any, Final

import httpx2
import pytest
from openai import AsyncOpenAI

import idia
from idia import (
    WARNING_LABEL_ABSENT_FROM_LOGPROBS,
    WARNING_NO_SNAPSHOT,
    Deadline,
    Dispatcher,
    ElicitationNotInRelease,
    Fail,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    Label,
    LabelAnswer,
    NoulAnswer,
    ProviderError,
    Question,
    Request,
    Rounding,
    Rule,
    RuleFailure,
    State,
    TextPart,
    WireRequest,
    WireResponse,
    providers,
)
from idia.fit import _preflight
from idia.providers.chat import OpenAIChat
from idia.providers.chat import openai as chat_module
from tests._http import body_key, replay_headers
from tests._openai_cases import (
    CHAT_DIR,
    CHAT_SCORE,
    FAILURE,
    KINDS,
    UNAUTHORIZED,
    chat_record,
    chat_request,
    load,
    singles,
)
from tests._system_one_cases import TEXT, TEXT_QUESTIONS, decode

_MODEL = "gpt-4.1"
_SNAPSHOT = "gpt-4.1-2025-04-14"
_USAGE = {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}

_NOUL = Question(key="n", kind="noul", instruction="Did it rain?")
_CHOICE = TEXT_QUESTIONS["choice"][0]  # rain (described), sun, snow (described)
_SCORE = TEXT_QUESTIONS["score"][1]  # low, mid, high


def _top(masses: Mapping[str, float]) -> list[dict[str, object]]:
    return [
        {"token": t, "logprob": math.log(m), "bytes": list(t.encode())} for t, m in masses.items()
    ]


def _body(
    top: object, *, model: str = _SNAPSHOT, logprobs: object = None, content: object = None
) -> dict[str, object]:
    if content is None:
        content = [{"token": "A", "logprob": -0.1, "top_logprobs": top}]
    if logprobs is None:
        logprobs = {"content": content}
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{"index": 0, "message": {"content": "A"}, "logprobs": logprobs}],
        "usage": _USAGE,
    }


class Wire:
    """An SDK client over a mock transport answering every request with one response."""

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

    def leaf(self, **options: Any) -> OpenAIChat:
        return OpenAIChat(model=_MODEL, client=self.client, **options)

    @property
    def body_sent(self) -> Any:
        [request] = self.sent
        return json.loads(request.content)


def _answering(masses: Mapping[str, float], **body: Any) -> Wire:
    return Wire(_body(_top(masses), **body))


def _call(leaf: OpenAIChat, question: Question, state: State = TEXT) -> WireResponse:
    wire_request = WireRequest(state=state, questions=(question,))
    return asyncio.run(leaf.call(wire_request, Deadline.after(5)))


def _sent(response: WireResponse) -> dict[str, Any]:
    assert isinstance(response.prompt_sent, Mapping)
    return dict(response.prompt_sent)


def _answer(masses: Mapping[str, float], question: Question, **options: Any) -> object:
    return _call(_answering(masses).leaf(**options), question).answers[question.key]


# ----------------------------------------------------------------------------------------
# The read protocol: tests that confirm it
# ----------------------------------------------------------------------------------------


def test_chat_leaf_reads_the_first_token_over_letter_coded_labels() -> None:
    wire = Wire(
        _body(
            None,
            content=[
                {
                    "token": "A",
                    "logprob": -0.1,
                    "top_logprobs": _top({"A": 0.6, "B": 0.3, "C": 0.1}),
                },
                {"token": "C", "logprob": -0.1, "top_logprobs": _top({"C": 0.99, "A": 0.01})},
            ],
        )
    )
    response = _call(wire.leaf(), _CHOICE)
    answer = response.answers[_CHOICE.key]
    assert isinstance(answer, LabelAnswer)
    assert [name for name, _ in answer.probabilities] == ["rain", "sun", "snow"]
    assert [p for _, p in answer.probabilities] == pytest.approx([0.6, 0.3, 0.1])
    user = wire.body_sent["messages"][1]["content"]
    assert user.endswith("A. rain: It rained.\nB. sun\nC. snow: It snowed.")
    assert _sent(response)["letters"] == {"A": "rain", "B": "sun", "C": "snow"}


def test_chat_leaf_sums_surface_forms_of_a_letter() -> None:
    answer = _answer({"A": 0.4, " A": 0.2, "a": 0.1, " a": 0.1, "B": 0.1, " b": 0.1}, _NOUL)
    assert answer == NoulAnswer(probability=pytest.approx(0.8), off_label_mass=pytest.approx(0.0))  # type: ignore[arg-type]


def test_chat_leaf_renormalizes_over_labels_and_records_off_label_mass() -> None:
    answer = _answer({"A": 0.5, "B": 0.2, "C": 0.1, "The": 0.15, "I": 0.05}, _CHOICE)
    assert isinstance(answer, LabelAnswer)
    names = [name for name, _ in answer.probabilities]
    shares = [p for _, p in answer.probabilities]
    assert names == ["rain", "sun", "snow"]
    assert shares == pytest.approx([0.625, 0.25, 0.125])
    assert sum(shares) == pytest.approx(1.0)
    assert answer.off_label_mass == pytest.approx(0.2)
    assert answer.chosen == "rain"
    assert answer.derived == {"chosen"}
    assert answer.confidence is None


def test_chat_noul_is_reduced_to_one_probability_of_yes() -> None:
    wire = _answering({"A": 0.3, "B": 0.6, "Yes": 0.1})
    answer = _call(wire.leaf(), _NOUL).answers["n"]
    assert isinstance(answer, NoulAnswer)
    assert answer.probability == pytest.approx(1 / 3)
    assert answer.off_label_mass == pytest.approx(0.1)
    assert answer.derived == frozenset()
    assert wire.body_sent["messages"][1]["content"].endswith("Did it rain?\n\nA. Yes\nB. No")


@pytest.mark.parametrize(
    "body",
    [
        _body(None, logprobs={"content": None}),
        _body(None, content=[]),
        _body([]),
        _body(None, content=[{"token": "A", "logprob": -0.1}]),
        {**_body([]), "choices": [{"index": 0, "message": {"content": "A"}, "logprobs": None}]},
        {**_body([]), "choices": [{"index": 0, "message": {"content": "A"}}]},
    ],
    ids=["content-null", "content-empty", "top-empty", "top-missing", "logprobs-null", "absent"],
)
def test_empty_logprobs_is_a_fault(body: dict[str, object]) -> None:
    response = _call(Wire(body).leaf(), _CHOICE)
    assert response.answers[_CHOICE.key] == Fault(
        key=_CHOICE.key,
        failures=(RuleFailure(rule=Rule.empty_logprobs, observed=None, bound=None),),
        mapped=None,
    )
    assert _sent(response)["top_logprobs_returned"] == 0


def test_label_absent_from_top_k_reads_as_zero_mass_with_a_warning() -> None:
    response = _call(_answering({"A": 0.9, "The": 0.1, "I": 0.001}).leaf(), _CHOICE)
    answer = response.answers[_CHOICE.key]
    assert isinstance(answer, LabelAnswer)
    assert answer.probabilities == (("rain", 1.0), ("sun", 0.0), ("snow", 0.0))
    assert answer.chosen == "rain"
    assert answer.off_label_mass == pytest.approx(0.1)
    [warning] = response.warnings
    assert (warning.code, warning.field) == (WARNING_LABEL_ABSENT_FROM_LOGPROBS, _CHOICE.key)
    assert "sun (B), snow (C)" in warning.message
    assert "0.001" in warning.message


def test_no_label_letter_in_top_k_is_a_fault() -> None:
    response = _call(_answering({"The": 0.9, "I": 0.1}).leaf(), _CHOICE)
    assert response.answers[_CHOICE.key] == Fault(
        key=_CHOICE.key,
        failures=tuple(
            RuleFailure(rule=Rule.label_absent_from_logprobs, observed=letter, bound=None)
            for letter in "ABC"
        ),
        mapped=None,
    )
    assert not any(w.code == WARNING_LABEL_ABSENT_FROM_LOGPROBS for w in response.warnings)


def test_off_label_mass_above_bound_is_a_fault() -> None:
    fault = _answer({"A": 0.5, "B": 0.2, "The": 0.3}, _NOUL, max_off_label_mass=0.25)
    assert isinstance(fault, Fault)
    [failure] = fault.failures
    assert failure.rule is Rule.off_label_mass_above_bound
    assert failure.observed == pytest.approx(0.3)
    assert failure.bound == 0.25
    assert isinstance(fault.mapped, NoulAnswer)
    assert fault.mapped.probability == pytest.approx(5 / 7)
    within = _answer({"A": 0.5, "B": 0.3, "The": 0.2}, _NOUL, max_off_label_mass=0.25)
    assert isinstance(within, NoulAnswer)


def test_the_off_label_mass_is_unbounded_by_default_and_never_below_zero() -> None:
    answer = _answer({"A": 0.1, "B": 0.05, "The": 0.85}, _NOUL)
    assert isinstance(answer, NoulAnswer)
    assert answer.off_label_mass == pytest.approx(0.85)
    # Rounding in the wire's logprobs can put the label masses a hair above one.
    over = _answer({"A": 0.7, "B": 0.300001}, _NOUL)
    assert isinstance(over, NoulAnswer)
    assert over.off_label_mass == 0.0


def test_chat_leaf_declares_calibrated_false_no_confidence_and_max_questions_one() -> None:
    record = Wire({}).leaf().capabilities
    assert record == chat_record()
    assert record.calibrated is False
    assert dict(record.confidence) == {}
    assert record.max_questions == 1
    assert (record.max_labels, record.max_levels) == (20, 20)
    assert record.rounding == Rounding(probability_decimals=None, score_decimals=None)
    response = _call(_answering({"A": 0.7, "B": 0.3}).leaf(), _NOUL)
    assert (response.elicitation, response.calibrated) == ("logprobs", False)


@pytest.mark.parametrize("candidate", ["structured_output", "verbalised", "sampled_n"])
def test_candidate_elicitation_raises_at_construction(candidate: str) -> None:
    with pytest.raises(ElicitationNotInRelease) as raised:
        Wire({}).leaf(elicitation=candidate)
    assert (raised.value.elicitation, raised.value.candidate) == (candidate, True)


@pytest.mark.parametrize(
    ("name", "error"), [("passthrough", ValueError), ("guess", ElicitationNotInRelease)]
)
def test_only_logprobs_builds_a_chat_leaf(name: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        Wire({}).leaf(elicitation=name)


def test_chat_derived_expected_level_is_validated_and_marked_derived() -> None:
    answer = _answer({"A": 0.2, "B": 0.5, "C": 0.3}, _SCORE)
    assert isinstance(answer, LabelAnswer)
    assert answer.expected_level == pytest.approx(0.5 + 0.6)
    assert answer.chosen == "mid"
    assert answer.derived == {"chosen", "expected_level"}
    assert answer.echo is None
    # The same rules run over the derived figures: a wrong one would be a fault.
    from idia.answers import validate

    wrong = dataclasses.replace(answer, expected_level=2.0)
    fault = validate(_SCORE, wrong, Rounding(probability_decimals=None, score_decimals=None))
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.expected_level]


def test_a_tie_chooses_the_first_label_asked() -> None:
    answer = _answer({"A": 0.4, "B": 0.4, "C": 0.2}, _CHOICE)
    assert isinstance(answer, LabelAnswer)
    assert answer.chosen == "rain"


def test_leaf_sdk_retries_are_zero() -> None:
    created = OpenAIChat(model=_MODEL, api_key="k")
    assert created.client.max_retries == 0
    transport = created.client._client._transport
    assert isinstance(transport, httpx2.AsyncHTTPTransport)
    assert transport._pool._retries == 0
    injected = Wire({})
    assert injected.client.max_retries == 2
    assert injected.leaf().client.max_retries == 0


# ----------------------------------------------------------------------------------------
# Wrong shapes and refusals before sending
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "observed"),
    [
        (_body(None, logprobs="none"), "str"),
        (_body(None, content={"token": "A"}), "dict"),
        (_body(None, content=["A"]), "str"),
        (_body({"A": 0.1}), "dict"),
        (_body(["A"]), "str"),
        (_body([{"token": 1, "logprob": -1.0}]), "int"),
        (_body([{"token": "A", "logprob": "low"}]), "str"),
        (_body([{"token": "A", "logprob": True}]), "bool"),
        ({**_body([]), "choices": ["A"]}, "str"),
    ],
)
def test_logprobs_of_the_wrong_shape_are_a_kind_shape_fault(
    body: dict[str, object], observed: str
) -> None:
    fault = _call(Wire(body).leaf(), _NOUL).answers["n"]
    assert fault == Fault(
        key="n",
        failures=(RuleFailure(rule=Rule.kind_shape, observed=observed, bound=None),),
        mapped=None,
    )


def test_a_letter_far_down_the_top_k_alone_takes_all_the_share() -> None:
    top = [{"token": "A", "logprob": -9999.0}, {"token": "The", "logprob": -0.01}]
    response = _call(Wire(_body(top)).leaf(), _CHOICE)
    answer = response.answers[_CHOICE.key]
    assert isinstance(answer, LabelAnswer)
    assert answer.probabilities == (("rain", 1.0), ("sun", 0.0), ("snow", 0.0))
    assert [w.code for w in response.warnings] == [WARNING_LABEL_ABSENT_FROM_LOGPROBS]


@pytest.mark.parametrize(
    ("logprob", "observed"),
    [
        (800.0, "logprob 800.0 above 0"),
        (float("nan"), "logprob nan"),
        (float("-inf"), "logprob -inf"),
        (-(10**400), "int beyond float"),
    ],
    ids=["above-zero", "nan", "minus-inf", "int-beyond-float"],
)
def test_a_logprob_that_is_no_logprob_is_a_kind_shape_fault(logprob: float, observed: str) -> None:
    top = ({"token": "A", "logprob": logprob}, {"token": "B", "logprob": -1.0})
    mapped, warnings = chat_module._read(_NOUL, chat_module._letters(_NOUL), top, None)
    assert mapped == Fault(
        key="n",
        failures=(RuleFailure(rule=Rule.kind_shape, observed=observed, bound=None),),
        mapped=None,
    )
    assert warnings == ()


def test_a_logprob_above_zero_on_the_wire_is_that_questions_fault() -> None:
    fault = _call(Wire(_body([{"token": "A", "logprob": 800}])).leaf(), _NOUL).answers["n"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.kind_shape]


@pytest.mark.parametrize(
    "changes",
    [
        {"calibrated": True},
        {"confidence": {"choice": "undocumented"}},
        {"rounding": Rounding(probability_decimals=2, score_decimals=2)},
        {"max_questions": 2},
        {"max_labels": 21},
        {"max_levels": None},
        {
            "modalities": frozenset({"text", "json", "image"}),
            "image_carriers": frozenset({"parts"}),
            "image_media_types": frozenset({"image/png"}),
            "max_image_bytes": 1024,
            "max_total_image_bytes": 1024,
            "max_images": 1,
            "max_images_per_question": 1,
        },
        {"max_image_bytes": 1024},
        {"image_media_types": None},
    ],
    ids=[
        "calibrated",
        "confidence",
        "rounding",
        "max-questions",
        "max-labels",
        "max-levels",
        "image-modality",
        "image-bytes",
        "image-types",
    ],
)
def test_a_record_that_contradicts_the_protocol_is_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="fixed by its protocol"):
        Wire({}).leaf(capabilities=dataclasses.replace(chat_record(), **changes))


def test_a_record_within_the_protocol_may_be_passed() -> None:
    narrower = dataclasses.replace(chat_record(), max_labels=5, max_body_bytes=1_000_000)
    assert Wire({}).leaf(capabilities=narrower).capabilities is narrower


def test_a_token_listed_twice_sums() -> None:
    top = [*_top({"A": 0.3, "B": 0.4}), *_top({"A": 0.3})]
    answer = _call(Wire(_body(top)).leaf(), _NOUL).answers["n"]
    assert isinstance(answer, NoulAnswer)
    assert answer.probability == pytest.approx(0.6)


def test_a_wire_request_of_several_questions_is_refused_before_sending() -> None:
    wire = _answering({"A": 1.0})
    wire_request = WireRequest(state=TEXT, questions=TEXT_QUESTIONS["noul"])
    with pytest.raises(ProviderError, match="one question") as raised:
        asyncio.run(wire.leaf().call(wire_request, Deadline.after(5)))
    assert not raised.value.is_retryable
    assert wire.sent == []


def test_more_labels_than_letters_are_refused_before_sending() -> None:
    wide = Question(
        key="w",
        kind="choice",
        instruction="?",
        labels=tuple(Label(name=f"l{i}") for i in range(21)),
    )
    wire = _answering({"A": 1.0})
    with pytest.raises(ProviderError, match="20"):
        _call(wire.leaf(), wide)
    assert wire.sent == []


def test_an_image_is_refused_before_sending() -> None:
    wire = _answering({"A": 1.0})
    state = State(parts=(ImagePart(data=b"\x89PNG", media_type="image/png"),))
    with pytest.raises(ProviderError, match="image"):
        _call(wire.leaf(), _NOUL, state)
    assert wire.sent == []


@pytest.mark.parametrize("bound", [-0.1, 1.5])
def test_a_bound_outside_zero_to_one_is_refused(bound: float) -> None:
    with pytest.raises(ValueError, match="max_off_label_mass"):
        Wire({}).leaf(max_off_label_mass=bound)


def test_a_model_is_required() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAIChat(model="", api_key="k")


# ----------------------------------------------------------------------------------------
# The prompt, the call and provenance
# ----------------------------------------------------------------------------------------


def test_the_call_and_the_prompt_sent() -> None:
    state = State(parts=(TextPart(text="It rained."), JsonPart(value={"mm": 12})))
    wire = _answering({"A": 0.7, "B": 0.3})
    response = _call(wire.leaf(max_off_label_mass=0.5), _NOUL, state)
    sent = wire.body_sent
    [request] = wire.sent
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    messages = [
        {"role": "system", "content": "Answer with the letter of one option only."},
        {"role": "user", "content": 'It rained.\n\n{"mm":12}\n\nDid it rain?\n\nA. Yes\nB. No'},
    ]
    assert sent == {
        "model": _MODEL,
        "messages": messages,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0.0,
        "max_completion_tokens": 1,
    }
    assert json.loads(json.dumps(_sent(response))) == {
        "messages": messages,
        "model": _MODEL,
        "temperature": 0.0,
        "max_completion_tokens": 1,
        "logprobs": True,
        "top_logprobs_requested": 20,
        "top_logprobs_returned": 2,
        "letters": {"A": "Yes", "B": "No"},
        "max_off_label_mass": 0.5,
    }


def test_a_dated_model_reported_is_the_snapshot() -> None:
    response = _call(_answering({"A": 0.7, "B": 0.3}).leaf(), _NOUL)
    assert (response.model_reported, response.snapshot) == (_SNAPSHOT, _SNAPSHOT)
    assert response.warnings == ()
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (50, 1)
    assert response.request_id == "req-1"
    assert isinstance(response.raw_body, Mapping)


@pytest.mark.parametrize(
    "model", ["gpt-4.1", "gpt-4.1-2025-04-14\n", "gpt-4.1-\u0662\u0660\u0662\u0665-04-14"]
)
def test_an_alias_reported_is_no_snapshot_and_a_warning(model: str) -> None:
    response = _call(_answering({"A": 0.7, "B": 0.3}, model=model).leaf(), _NOUL)
    assert response.snapshot is None
    assert [w.code for w in response.warnings] == [WARNING_NO_SNAPSHOT]


_UNREADABLE: list[object] = [
    b"<html/>",
    {"model": _SNAPSHOT, "usage": {}},
    {"model": _SNAPSHOT, "usage": {}, "choices": []},
    {"model": 4.1, "usage": {}, "choices": [{}]},
    {"model": _SNAPSHOT, "usage": [], "choices": [{}]},
]


@pytest.mark.parametrize("body", _UNREADABLE)
def test_a_body_failing_the_top_level_schema_is_not_retryable(body: object) -> None:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    with pytest.raises(ProviderError) as raised:
        _call(Wire(content).leaf(), _NOUL)
    assert not raised.value.is_retryable


# ----------------------------------------------------------------------------------------
# The cassettes
# ----------------------------------------------------------------------------------------


def _replay(name: str) -> tuple[Wire, OpenAIChat, WireRequest]:
    cassette = load(CHAT_DIR, name)
    response = cassette["response"]
    headers = replay_headers(response["headers"])
    wire = Wire(response["body"], status=response["status"], headers=headers)
    wire_request = decode(cassette["wire_request"])
    assert isinstance(wire_request, WireRequest)
    return wire, wire.leaf(), wire_request


_CASES = [*((kind, chat_request(kind)) for kind in KINDS), *singles().items()]


@pytest.mark.parametrize(("name", "asked"), _CASES, ids=[name for name, _ in _CASES])
def test_each_cassette_is_the_body_sent_and_its_answers_as_recorded(
    name: str, asked: WireRequest
) -> None:
    cassette = load(CHAT_DIR, name)
    wire, leaf, wire_request = _replay(name)
    assert wire_request == asked
    response = asyncio.run(leaf.call(wire_request, Deadline.after(5)))
    assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
    assert response.answers == decode(cassette["expected"])
    assert not any(isinstance(answer, Fault) for answer in response.answers.values())
    assert cassette["request"]["headers"]["authorization"] == "<filtered>"


def test_the_recorded_failure_is_not_retryable_and_names_the_top_k_cap() -> None:
    _, leaf, wire_request = _replay(FAILURE)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(wire_request, Deadline.after(5)))
    assert raised.value.status_code == 400
    assert not raised.value.is_retryable
    assert raised.value.request_id
    assert "less than or equal to 20" in raised.value.message


def test_the_recorded_unauthorized_call_is_a_401_that_is_not_retried() -> None:
    cassette = load(CHAT_DIR, UNAUTHORIZED)
    _, leaf, wire_request = _replay(UNAUTHORIZED)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(wire_request, Deadline.after(5)))
    error = raised.value
    assert (error.status_code, error.is_retryable, error.retry_after) == (401, False, None)
    assert error.request_id == cassette["response"]["headers"]["x-request-id"]
    assert "Incorrect API key provided" in error.message
    assert cassette["expected"] is None
    assert cassette["request"]["headers"]["authorization"] == "<filtered>"


def test_the_wire_reported_a_dated_snapshot_and_took_a_one_token_answer() -> None:
    for name, _ in _CASES:
        cassette = load(CHAT_DIR, name)
        body = cassette["response"]["body"]
        assert body["model"] == _SNAPSHOT
        assert cassette["request"]["body"]["max_completion_tokens"] == 1
        assert body["usage"]["completion_tokens"] == 1
        [choice] = body["choices"]
        assert len(choice["logprobs"]["content"][0]["top_logprobs"]) == 20


def test_a_confident_score_left_a_level_outside_the_top_20_read_as_zero() -> None:
    # gpt-4.1 put C at a probability of about one; look-alikes of C and other formatting
    # tokens filled the top 20, and A fell outside it: zero mass, with a warning.
    cassette = load(CHAT_DIR, "score-absent-level")
    _, leaf, wire_request = _replay("score-absent-level")
    [question] = wire_request.questions
    response = asyncio.run(leaf.call(wire_request, Deadline.after(5)))
    answer = response.answers[question.key]
    assert isinstance(answer, LabelAnswer)
    assert [name for name, _ in answer.probabilities] == ["none", "light", "heavy"]
    assert [p for _, p in answer.probabilities] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert answer.probabilities[0][1] == 0.0
    assert answer.chosen == "heavy"
    [warning] = response.warnings
    assert (warning.code, warning.field) == (WARNING_LABEL_ABSENT_FROM_LOGPROBS, question.key)
    assert warning.message.startswith("none (A) not in the top 20")
    assert response.answers == decode(cassette["expected"])


_SCORE_TOP: Final = (
    ("A", -1.9361264946837764e-07),
    ("B", -16.1875),
    ("a", -31.296875),
    (" A", -32.859375),
    ("\u0410", -34.078125),
    ("\uff21", -34.203125),
    ("C", -34.671875),
    ("Answer", -35.25),
    ("_A", -36.578125),
    ("\tA", -36.59375),
    ("(A", -37.015625),
    ("D", -37.046875),
    ("The", -38.1875),
    ("*A", -38.25),
    ("A\u7ea7", -38.640625),
    ("\u0391", -38.984375),
    (".A", -39.140625),
    ("E", -39.390625),
    (":A", -39.671875),
    ("**", -40.109375),
)
"""The first token's top 20 from ``openai-chat/gpt-4.1/base/score.json`` at 76450d0, copied
so the test holds whatever the cassette becomes."""


def _naive_sum(values: Iterable[float]) -> float:
    return functools.reduce(operator.add, values, 0.0)  # not sum(): 3.12+ compensates


def _read_score(top: tuple[JsonValue, ...]) -> LabelAnswer:
    answer, warnings = chat_module._read(CHAT_SCORE, chat_module._letters(CHAT_SCORE), top, None)
    assert isinstance(answer, LabelAnswer)
    assert warnings == ()
    return answer


def test_the_read_arithmetic_is_bit_identical_across_python_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Letters far apart in mass: a plain float sum over them rounds differently on
    # CPython 3.11 and 3.12+; the read sums exactly.
    top = tuple({"token": token, "logprob": logprob} for token, logprob in _SCORE_TOP)
    answer = _read_score(top)
    assert answer.off_label_mass == 1.0031766217721838e-07
    assert answer.expected_level == 9.329494804719768e-08
    assert [share for _, share in answer.probabilities] == [
        0.9999999067050529,
        9.329494602091553e-08,
        8.753788588670532e-16,
        8.142291742333244e-17,
        7.813920122399748e-18,
    ]
    # Guard: the input still separates naive from exact summation in the read.
    monkeypatch.setattr(math, "fsum", _naive_sum)
    assert _read_score(top) != answer


def test_the_estimate_covers_the_recorded_usage() -> None:
    estimator = Wire({}).leaf().estimator
    for name, _ in _CASES:
        cassette = load(CHAT_DIR, name)
        wire_request = decode(cassette["wire_request"])
        estimate = estimator.estimate(wire_request.state, wire_request.questions)
        used = cassette["response"]["body"]["usage"]["prompt_tokens"]
        assert used <= estimate.tokens * (1 + estimate.margin), name


def test_a_request_of_two_questions_plans_two_wire_requests() -> None:
    leaf = Wire({}).leaf()
    request = Request(state=TEXT, questions=TEXT_QUESTIONS["choice"], deadline=Deadline.after(5))
    plan = _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert [len(w.questions) for w in plan.wire_requests] == [1, 1]


# ----------------------------------------------------------------------------------------
# The registry, the allowlist, and the elicitation review
# ----------------------------------------------------------------------------------------


def test_the_registry_names_the_leaf_and_its_extra() -> None:
    entry = providers._REGISTRY["openai-chat"]
    assert entry.target == "idia.providers.chat.openai:OpenAIChat"
    assert entry.extra == "openai"


def test_provider_builds_and_wraps_it_with_its_options() -> None:
    wire = _answering({"A": 0.5, "B": 0.2, "The": 0.3})
    provider = idia.provider(
        "openai-chat", model=_MODEL, client=wire.client, max_off_label_mass=0.1
    )
    assert isinstance(provider, Dispatcher)
    request = Request(state=TEXT, questions=(_NOUL,), deadline=Deadline.after(5))
    response = asyncio.run(provider.answer(request))
    fault = response.answers["n"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.off_label_mass_above_bound]
    [outcome] = response.outcomes
    assert (outcome.provider, outcome.model_requested, outcome.elicitation) == (
        "openai-chat",
        _MODEL,
        "logprobs",
    )


def test_only_the_off_label_mass_may_compute_a_complement() -> None:
    from idia.conformance import _review

    module = chat_module.__name__
    assert f"{module}:_off_label_mass" in _review._COMPLEMENT_ALLOWED
    assert _review.complements(_review.parse(module), module) == []


def test_name_model_bound_and_repr() -> None:
    leaf = Wire({}).leaf(max_off_label_mass=0.2)
    assert (leaf.name, leaf.model, leaf.max_off_label_mass) == ("openai-chat", _MODEL, 0.2)
    assert repr(leaf) == "OpenAIChat(model='gpt-4.1')"
    assert Wire({}).leaf().max_off_label_mass is None
