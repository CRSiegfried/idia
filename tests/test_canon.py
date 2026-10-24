"""The test doubles' renderings: every shape a recording holds comes back as it went in."""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, cast

import pytest

from idia import (
    Answer,
    Box,
    BoxAnswer,
    ExtractionAnswer,
    Fault,
    ImagePart,
    JsonPart,
    Label,
    LabelAnswer,
    Notice,
    NoulAnswer,
    Question,
    Refusal,
    Rule,
    RuleFailure,
    Span,
    State,
    TextPart,
    Usage,
    WireRequest,
    WireResponse,
)
from idia.core import UNBOUNDED, Capabilities, Deadline
from idia.providers._canon import (
    Json,
    canonical_hash,
    decode,
    encode,
    json_bytes,
    plain,
    thaw,
)


def _through_json(value: object, hint: object) -> Any:
    """Encode, write as JSON text, read back and decode: what a log on disk does."""
    text = json.dumps(plain(encode(value)), allow_nan=False)
    return decode(json.loads(text), hint)


ANSWERS: Mapping[str, Answer] = {
    "noul": NoulAnswer(probability=0.25, confidence=0.5, derived=frozenset({"b", "a"})),
    "label": LabelAnswer(
        probabilities=(("red", 0.5), ("green", 0.5)),
        confidence=0.5,
        chosen="red",
        expected_level=0.5,
        echo=("red", "green"),
        echo_of="names",
        off_label_mass=0.0,
    ),
    "text": ExtractionAnswer(
        value="Ada", guess=None, p_null=0.1, span=Span(part=0, start=1, end=4)
    ),
    "list": ExtractionAnswer(
        value=(1, 2.5, "x"),
        p_null=None,
        span=(Span(part=0, start=0, end=1), Span(part=1, start=2, end=5)),
    ),
    "date": ExtractionAnswer(value=datetime.date(2026, 1, 2), p_null=0.0),
    "time": ExtractionAnswer(value=datetime.time(12, 30), p_null=0.0),
    "datetime": ExtractionAnswer(
        value=None,
        guess=datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
        p_null=0.9,
    ),
    "duration": ExtractionAnswer(value=datetime.timedelta(minutes=1, seconds=1.5), p_null=0.0),
    "box": BoxAnswer(region=Box(top=0.1, left=0.2, bottom=0.5, right=0.6), confidence=None),
    "refusal": Refusal(),
    "fault": Fault(
        key="fault",
        failures=(
            RuleFailure(rule=Rule.sum_tolerance, observed=0.25, bound=0.01),
            RuleFailure(rule=Rule.candidate_shape, observed="estimate", bound=None),
            RuleFailure(rule=Rule.missing_key, observed=None, bound=None),
        ),
        mapped=LabelAnswer(probabilities=(("red", 0.75), ("green", 0.5))),
    ),
    "empty_fault": Fault(key="empty_fault", failures=(), mapped=None),
}


@pytest.mark.parametrize("key", list(ANSWERS))
def test_every_answer_kind_round_trips(key: str) -> None:
    answer = ANSWERS[key]
    assert _through_json(answer, Answer) == answer


def test_an_answer_map_round_trips() -> None:
    assert _through_json(ANSWERS, Mapping[str, Answer]) == ANSWERS


def test_non_finite_floats_come_back_as_floats() -> None:
    fault = Fault(
        key="k",
        failures=(
            RuleFailure(rule=Rule.not_finite, observed=math.inf, bound=-math.inf),
            RuleFailure(rule=Rule.not_finite, observed=math.nan, bound=None),
        ),
        mapped=NoulAnswer(probability=math.inf),
    )
    back = _through_json(fault, Answer)
    assert isinstance(back, Fault)
    assert back.failures[0].observed == math.inf
    assert back.failures[0].bound == -math.inf
    observed = back.failures[1].observed
    assert isinstance(observed, float)
    assert math.isnan(observed)
    assert isinstance(back.mapped, NoulAnswer)
    assert back.mapped.probability == math.inf


def test_strings_that_look_like_floats_stay_strings() -> None:
    extraction = ExtractionAnswer(value="nan", guess="inf", p_null=0.0)
    assert _through_json(extraction, Answer) == extraction
    failure = RuleFailure(rule=Rule.candidate_shape, observed="inf", bound=None)
    back = _through_json(failure, RuleFailure)
    assert back == failure
    assert isinstance(back.observed, str)


def test_a_tagged_float_must_be_a_non_finite_one() -> None:
    with pytest.raises(TypeError, match="not a non-finite float"):
        decode({"type": "float", "value": "1.5"}, float)
    with pytest.raises(TypeError, match="not a non-finite float"):
        decode({"type": "date", "value": "inf"}, float)


def test_raw_usage_comes_back_thawed_as_a_raw_body_does() -> None:
    raw: dict[str, Any] = {"n": [1, [2]], "o": {"p": [3]}}
    usage = Usage(input_tokens=1, output_tokens=1, raw_usage=raw)
    back = _through_json(usage, Usage)
    assert dict(back.raw_usage) == {"n": (1, (2,)), "o": {"p": (3,)}}
    assert dict(back.raw_usage) == thaw(json.loads(json.dumps({"n": [1, [2]], "o": {"p": [3]}})))


# A change to the rendering bumps the format of each that holds what changed, the recorded
# leaf's _FORMAT or the cassettes' FORMAT in tests/_http.py, and these two literals with it.
GOLDEN_HASH = "a1f1018fb225317e97aafb2340362885def243874265f4a63d19ea1bb80b0a25"
GOLDEN_RESPONSE = (
    '{"answers":{"f":{"failures":[{"bound":null,"observed":{"type":"float","value":"inf"},'
    '"rule":"not_finite","type":"RuleFailure"}],"key":"f","mapped":null,"type":"Fault"},'
    '"q":{"confidence":null,"derived":[],"off_label_mass":null,"probability":0.25,'
    '"type":"NoulAnswer"}},"calibrated":true,"elicitation":"passthrough",'
    '"model_reported":"m-1","prompt_sent":null,"raw_body":null,"request_id":"r-1",'
    '"snapshot":null,"type":"WireResponse","usage":{"input_tokens":1,"output_tokens":2,'
    '"raw_usage":{"n":[1]},"type":"Usage"},"warnings":[{"code":"no-snapshot","field":null,'
    '"message":"none","type":"Notice"}]}'
)


def test_the_rendering_is_pinned() -> None:
    wire_request = WireRequest(
        state=State(parts=(TextPart(text="hello"),)),
        questions=(Question(key="q", kind="noul", instruction="Is it so?"),),
    )
    assert canonical_hash("leaf", "model-1", wire_request) == GOLDEN_HASH
    response = WireResponse(
        answers={
            "q": NoulAnswer(probability=0.25),
            "f": Fault(
                key="f",
                failures=(RuleFailure(rule=Rule.not_finite, observed=math.inf, bound=None),),
                mapped=None,
            ),
        },
        elicitation="passthrough",
        calibrated=True,
        model_reported="m-1",
        usage=Usage(input_tokens=1, output_tokens=2, raw_usage={"n": (1,)}),
        request_id="r-1",
        warnings=(Notice(code="no-snapshot", message="none"),),
    )
    assert json_bytes(encode(response)).decode() == GOLDEN_RESPONSE
    assert decode(json.loads(GOLDEN_RESPONSE), WireResponse) == response


def test_a_wire_response_round_trips() -> None:
    response = WireResponse(
        answers=dict(ANSWERS),
        elicitation="passthrough",
        calibrated=True,
        model_reported="m-1",
        snapshot="m-1",
        prompt_sent={"messages": ({"role": "user", "content": "hi"},)},
        usage=Usage(
            input_tokens=3, output_tokens=None, passes=2, raw_usage={"in": 3, "detail": (1, 2)}
        ),
        request_id="req-1",
        raw_body=None,
        warnings=(Notice(code="no-snapshot", message="none", field="snapshot"),),
    )
    assert _through_json(response, WireResponse) == response
    texted = WireResponse(answers={}, elicitation="logprobs", calibrated=False, prompt_sent="hi")
    assert _through_json(texted, WireResponse) == texted


def test_a_wire_request_round_trips_and_hashes_stably() -> None:
    wire_request = WireRequest(
        state=State(
            parts=(
                TextPart(text="hello"),
                JsonPart(value={"type": "not a tag", "list": (1, None, True)}),
                ImagePart(data=b"\x00\x01", media_type="image/png"),
            )
        ),
        questions=(
            Question(
                key="c",
                kind="choice",
                instruction="Which?",
                labels=(Label(name="a", description="the a"), Label(name="b")),
            ),
            Question(key="e", kind="extraction", instruction="What?", value_type="date"),
        ),
    )
    back = _through_json(wire_request, WireRequest)
    assert back == wire_request
    key = canonical_hash("leaf", "model-1", wire_request)
    assert key == canonical_hash("leaf", "model-1", back)
    assert len(key) == 64
    assert key != canonical_hash("leaf", "model-2", wire_request)
    assert key != canonical_hash("other", "model-1", wire_request)
    reordered = WireRequest(state=wire_request.state, questions=wire_request.questions[::-1])
    assert key != canonical_hash("leaf", "model-1", reordered)


def test_a_json_value_is_thawed_and_never_read_for_tags() -> None:
    assert thaw({"type": "Refusal", "a": [1, [2]]}) == {"type": "Refusal", "a": (1, (2,))}
    assert decode({"type": "NoulAnswer"}, Json) == {"type": "NoulAnswer"}
    with pytest.raises(TypeError, match="not JSON"):
        thaw(object())


def test_decoding_takes_only_what_the_declared_type_admits() -> None:
    with pytest.raises(TypeError, match="no member"):
        decode({"type": "Capabilities"}, Answer)
    with pytest.raises(TypeError, match="not a type the test doubles rebuild"):
        decode({"type": "Capabilities"}, Capabilities)
    with pytest.raises(TypeError, match="not a type the test doubles rebuild"):
        decode({"type": "Deadline", "at": 1.0}, Deadline)
    with pytest.raises(TypeError, match="is not a NoulAnswer"):
        decode({"type": "Refusal"}, NoulAnswer)
    with pytest.raises(TypeError, match="cannot decode"):
        decode(1, list[int])


@pytest.mark.parametrize(
    ("value", "hint", "match"),
    [
        pytest.param(None, int, "not a int", id="none-for-int"),
        pytest.param(1, type(None), "is not None", id="int-for-none"),
        pytest.param("x", float, "not a float", id="str-for-float"),
        pytest.param(True, float, "not a float", id="bool-for-float"),
        pytest.param(1, bytes, "not base64", id="int-for-bytes"),
        pytest.param("x", int, "not a int", id="str-for-int"),
        pytest.param("x", Literal["a", "b"], "is not one of", id="literal"),
        pytest.param([1, 2], tuple[int], "2 items for a tuple of 1", id="tuple-length"),
        pytest.param(1, tuple[int, ...], "not an array", id="int-for-tuple"),
        pytest.param(1, Mapping[str, int], "not an object", id="int-for-mapping"),
        pytest.param({"type": "date", "value": 1}, datetime.date, "not an ISO", id="date"),
        pytest.param({"type": "time", "value": "x"}, datetime.timedelta, "not a timedelta", id="t"),
        pytest.param(
            {"type": "timedelta", "value": "x"}, datetime.timedelta, "seconds", id="duration"
        ),
        pytest.param(None, int | str, "None is not admitted", id="none-in-union"),
        pytest.param([1], int | None, "is an array", id="array-in-union"),
        pytest.param("x", int | None, "is a string", id="str-in-union"),
        pytest.param(1.5, int | None, "fits no member", id="float-in-union"),
        pytest.param({"type": "x"}, int | None, "is tagged", id="object-in-union"),
    ],
)
def test_decoding_refuses_a_value_its_type_does_not_admit(
    value: object, hint: object, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        decode(value, hint)


def test_decoding_unions_and_plain_types() -> None:
    assert decode("YWI=", bytes | None) == b"ab"
    assert decode("a", Literal["a"] | None) == "a"
    assert decode("sum_tolerance", Rule | None) is Rule.sum_tolerance
    assert decode(2, float | None) == 2
    assert decode(True, bool | None) is True
    assert decode(True, int) is True
    assert decode(3, Any) == 3
    assert decode([1, 2], frozenset[int]) == frozenset({1, 2})
    assert decode({"a": 1}, dict[str, int]) == {"a": 1}
    assert decode({"a": 1}, Mapping[str, int] | None) == {"a": 1}
    assert decode([1, 2], tuple[int, int] | None) == (1, 2)
    assert decode({"type": "float", "value": "inf"}, float) == math.inf
    assert decode(1, Json | None) == 1


def test_encoding_is_deterministic_and_refuses_what_it_cannot_render() -> None:
    assert json_bytes(encode({"b": frozenset({2, 1}), "a": Rule.box_order})) == (
        b'{"a":"box_order","b":[1,2]}'
    )
    with pytest.raises(TypeError, match="cannot render a object"):
        encode(object())
    assert plain({"a": (1, {"b": (2,)})}) == {"a": [1, {"b": [2]}]}
    assert encode(UNBOUNDED) == "unbounded"
    assert decode(None, type(None)) is None


def test_the_canonical_hash_covers_options_and_leaves_empty_ones_out() -> None:
    wire_request = WireRequest(
        state=State(parts=(TextPart(text="Ada"),)),
        questions=(Question(key="q", kind="noul", instruction="?"),),
    )
    key = canonical_hash("leaf", "model-1", wire_request)
    assert key == canonical_hash("leaf", "model-1", replace(wire_request, options={}))
    optioned = replace(wire_request, options={"reference_date": "2026-10-21"})
    other = replace(wire_request, options={"reference_date": "2026-10-22"})
    assert (
        len(
            {
                key,
                canonical_hash("leaf", "model-1", optioned),
                canonical_hash("leaf", "model-1", other),
            }
        )
        == 3
    )
    assert "options" not in cast(dict[str, object], encode(wire_request))
    assert decode(json.loads(json.dumps(plain(encode(optioned)))), WireRequest) == optioned


def test_decode_refuses_a_key_the_type_lacks() -> None:
    """A rendering with a field the type no longer has fails instead of replaying."""
    with pytest.raises(TypeError, match="Span does not admit"):
        decode({"type": "Span", "part": 0, "start": 0, "end": 1, "path": []}, Span)
