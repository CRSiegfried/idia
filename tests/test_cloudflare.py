"""The Cloudflare leaves over a mock transport: the bodies, the envelope, the reading per
kind, the records, the status mapper, the refusals before sending, the key aliases, the
registry, and the ``cloudflare`` router function."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import datetime
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import httpx2
import pytest

import idia
from idia import (
    WARNING_EXTRA_ANSWER_KEY,
    WARNING_NO_SNAPSHOT,
    WARNING_PROVIDER_TRUNCATED,
    WARNING_UNKNOWN_OPTION,
    ConcurrencyLimit,
    Deadline,
    Dispatcher,
    DoesNotFit,
    ExtractionAnswer,
    Fault,
    ImagePart,
    JsonPart,
    LabelAnswer,
    NoulAnswer,
    ProviderError,
    Question,
    RateLimiter,
    Request,
    RetryPolicy,
    Router,
    Rule,
    Span,
    State,
    TextPart,
    WireRequest,
    WireResponse,
)
from idia import providers as registry
from idia.cloudflare import _router, cloudflare
from idia.conformance import Success
from idia.providers.cloudflare import (
    _CLEF_ID,  # pyright: ignore[reportPrivateUsage]
    Clairic,
    Clef,
    _share_client,
)
from idia.providers.http import OTHER_LOOP
from tests._cloudflare_cases import (
    ACCOUNT,
    ACCOUNT_ENV,
    ALIAS,
    CLAIRIC_DIR,
    CLAIRIC_FAILURE,
    CLAIRIC_OPTIONS,
    CLAIRIC_PROVIDER,
    CLEF_ORDER,
    CLEF_QUESTIONS,
    EXTRACTION,
    GONE,
    IMAGE_REQUEST,
    ORDER,
    ORDER_NAMES,
    PAST_CONTEXT_TRUNCATED,
    QUESTION_ID_BODY,
    QUESTION_ID_CONTROL,
    QUESTION_ID_KEY,
    QUESTION_ID_PROBE,
    QUESTION_ID_REQUEST,
    TEXT,
    TOKEN,
    TOKEN_ENV,
    WORDING,
    clairic_record,
    clef_dir,
    clef_record,
    clef_request,
    order_requests,
    router_record,
    wording_request,
)
from tests._cloudflare_harness import ClairicHarness, ClefFlashHarness, ClefHarness
from tests._http import (
    CassetteTransport,
    NoExchange,
    body_key,
    load_cassette,
    parse_body,
    record_cassette,
    replay_headers,
)
from tests._system_one_cases import png
from tests.test_cloudflare_live import _check, _keep_gone

_URL: Final = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run"
_RAY: Final = "f2051537-ecae-418a-af28-05dbca3805d3"
_USAGE: Final = {"input_tokens": 0, "output_tokens": 0}

Handler = Callable[[httpx2.Request], httpx2.Response]


def _envelope(result: object, *, success: bool = True) -> dict[str, object]:
    return {"success": success, "errors": [], "messages": [], "result": result}


def _ok(result: object) -> httpx2.Response:
    return httpx2.Response(200, headers={"cf-ai-req-id": _RAY}, json=_envelope(result))


class _Wire:
    """A mock wire that answers every request with the next response and keeps each request."""

    def __init__(self, *responses: httpx2.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._responses.pop(0)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))

    def body(self, n: int = 0) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.requests[n].content))


def _clef(wire: _Wire, **options: Any) -> Clef:
    return Clef(account_id=ACCOUNT, api_token=TOKEN, client=wire.client(), **options)


def _clairic(wire: _Wire, **options: Any) -> Clairic:
    return Clairic(account_id=ACCOUNT, api_token=TOKEN, client=wire.client(), **options)


_RUNNER: list[asyncio.Runner] = []


@pytest.fixture(autouse=True)
def _one_loop() -> Iterator[None]:
    """Each test's calls run on one event loop, as an injected client's must."""
    with asyncio.Runner() as runner:
        _RUNNER.append(runner)
        try:
            yield
        finally:
            _RUNNER.pop()


def _call(leaf: Clef | Clairic, wire_request: WireRequest) -> WireResponse:
    """One call on the test's event loop."""
    return _RUNNER[-1].run(leaf.call(wire_request, Deadline.after(5.0)))


def _raises(leaf: Clef | Clairic, wire_request: WireRequest) -> ProviderError:
    with pytest.raises(ProviderError) as raised:
        _call(leaf, wire_request)
    return raised.value


def _clef_result(answers: Mapping[str, object], **extra: object) -> dict[str, object]:
    return {"model": "clef", "answers": answers, "usage": _USAGE, **extra}


def _clairic_result(answers: Mapping[str, object], **extra: object) -> dict[str, object]:
    usage = {
        "state_tokens": 60,
        "image_tokens": 0,
        "schema_tokens": 160,
        "input_tokens": 250,
        "output_tokens": 30,
        "passes": 1,
    }
    return {"model": "clairic", "answers": answers, "usage": usage, **extra}


_NOUL = {
    "noul_rain": {"type": "noul", "noul": 0.97},
    "noul_sun": {"type": "noul", "noul": 0.81},
}
_CHOICE = {
    "choice_weather": {
        "type": "choice",
        "choice": "rain",
        "probabilities": {"snow": 0.01, "rain": 0.95, "sun": 0.04},
        "confidence": 0.9,
    },
    "choice_outcome": {
        "type": "choice",
        "choice": "called_off",
        "probabilities": {"played": 0.02, "called_off": 0.98},
        "confidence": 0.96,
    },
}
_SCORE = {
    "score_rain": {
        "type": "score",
        "score": 1.9,
        "legend": {"0": "No rain", "1": "Light rain", "2": "Heavy rain"},
        "probabilities": {"0": 0.01, "1": 0.08, "2": 0.91},
        "confidence": 0.85,
    },
    "score_mood": {
        "type": "score",
        "score": 0.4,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.6, "1": 0.4, "2": 0.0},
        "confidence": 0.3,
    },
}


# ----------------------------------------------------------------------------------------
# Clef: the body
# ----------------------------------------------------------------------------------------


def test_clef_posts_each_kind_to_its_route_with_the_token() -> None:
    wire = _Wire(_ok(_clef_result(_NOUL)))
    _call(_clef(wire), clef_request("noul"))
    [request] = wire.requests
    assert str(request.url) == f"{_URL}/@cf/cloudflare/clef"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["user-agent"].startswith("idia/")


def test_clef_bodies_per_kind_are_compact_and_byte_stable() -> None:
    wire = _Wire(*(_ok(_clef_result(_NOUL)) for _ in range(2)))
    leaf = _clef(wire, model="clef-flash")
    for _ in range(2):
        _call(leaf, clef_request("noul"))
    first, second = (request.content for request in wire.requests)
    assert str(wire.requests[0].url) == f"{_URL}/@cf/cloudflare/clef-flash"
    assert first == second
    assert (
        first.decode()
        == (
            '{"model":"clef-flash","state":"' + TEXT.parts[0].text + '",'  # type: ignore[union-attr]
            '"questions":{"noul_rain":{"type":"noul","instructions":"Did it rain?"},'
            '"noul_sun":{"type":"noul","instructions":"Did the sun come out?"}}}'
        )
    )


def test_clef_choice_and_score_criteria() -> None:
    wire = _Wire(_ok(_clef_result(_CHOICE)), _ok(_clef_result(_SCORE)))
    leaf = _clef(wire)
    _call(leaf, clef_request("choice"))
    _call(leaf, clef_request("score"))
    choice = wire.body(0)["questions"]["choice_weather"]
    assert choice == {
        "type": "choice",
        "instructions": "The weather?",
        "criteria": {"rain": "It rained.", "sun": None, "snow": "It snowed."},
    }
    score = wire.body(1)["questions"]
    assert score["score_rain"]["criteria"] == ["No rain", "Light rain", "Heavy rain"]
    assert score["score_mood"]["criteria"] == ["low", "mid", "high"]
    assert list(wire.body(1)) == ["model", "state", "questions"]


def test_clef_sends_the_questions_in_request_order_and_the_criteria_in_label_order() -> None:
    """The order Clef sends, read from the body's own key order, which a cassette's match
    compares as well."""
    wire = _Wire(_ok(_clef_result(_CHOICE)))
    _call(_clef(wire), clef_request("choice"))
    body = wire.body(0)
    assert list(body) == ["model", "state", "questions"]
    assert list(body["questions"]) == ["choice_weather", "choice_outcome"]
    assert list(body["questions"]["choice_weather"]["criteria"]) == ["rain", "sun", "snow"]
    assert list(body["questions"]["choice_outcome"]["criteria"]) == ["played", "called_off"]


def test_clef_state_forms_and_the_images_array() -> None:
    wire = _Wire(*(_ok(_clef_result({})) for _ in range(3)))
    leaf = _clef(wire)
    noul = CLEF_QUESTIONS["noul"][:1]
    mixed = State(parts=(TextPart(text="a"), JsonPart(value={"b": (1, 2)})))
    _call(leaf, WireRequest(state=mixed, questions=noul))
    assert wire.body(0)["state"] == ["a", {"b": [1, 2]}]
    response = _call(leaf, IMAGE_REQUEST)
    body = wire.body(1)
    assert list(body) == ["model", "state", "questions", "images"]
    assert body["state"] == IMAGE_REQUEST.state.parts[0].text  # type: ignore[union-attr]
    [image] = body["images"]
    assert image["content_type"] == "image/png"
    assert isinstance(response.prompt_sent, Mapping)
    assert response.prompt_sent["images"] == ({"content_type": "image/png", "bytes": len(png())},)
    only = State(parts=(ImagePart(data=png()),))
    _call(leaf, WireRequest(state=only, questions=noul))
    assert wire.body(2)["state"] == ""


@pytest.mark.parametrize(
    ("data", "media_type", "expected"),
    [
        (b"\xff\xd8\xff\xe0rest", None, "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", None, "image/webp"),
        (b"anything", "image/jpg", "image/jpeg"),
        (b"anything", "IMAGE/WEBP", "image/webp"),
    ],
)
def test_image_content_types(data: bytes, media_type: str | None, expected: str) -> None:
    wire = _Wire(_ok(_clef_result({})))
    state = State(parts=(ImagePart(data=data, media_type=media_type),))
    _call(_clef(wire), WireRequest(state=state, questions=CLEF_QUESTIONS["noul"][:1]))
    assert wire.body()["images"][0]["content_type"] == expected


_PAST_THE_RECORD = [
    pytest.param(
        (ImagePart(data=b"GIF89a", media_type="image/gif"),), "image_media_types", id="gif"
    ),
    pytest.param((ImagePart(data=png()),) * 5, "max_images", id="five-images"),
    pytest.param(
        (ImagePart(data=b"\x89PNG\r\n\x1a\n" + bytes(4 * 1024 * 1024)),),
        "max_image_bytes",
        id="over-4-mib",
    ),
    pytest.param(
        (ImagePart(data=b"\x89PNG\r\n\x1a\n" + bytes(3 * 1024 * 1024)),) * 3,
        "max_total_image_bytes",
        id="over-8-mib-in-all",
    ),
]


def _refused_by_preflight(leaf: Clef | Clairic, wire_request: WireRequest) -> DoesNotFit:
    request = Request(
        state=wire_request.state, questions=wire_request.questions, deadline=Deadline.after(5)
    )
    with pytest.raises(DoesNotFit) as raised:
        _RUNNER[-1].run(Dispatcher(leaf).answer(request))
    return raised.value


@pytest.mark.parametrize(("images", "check"), _PAST_THE_RECORD)
def test_clef_images_past_the_record_do_not_fit_and_make_no_call(
    images: tuple[ImagePart, ...], check: str
) -> None:
    wire = _Wire()
    request = WireRequest(state=State(parts=images), questions=CLEF_QUESTIONS["noul"][:1])
    assert _refused_by_preflight(_clef(wire), request).check == check
    assert wire.requests == []


@pytest.mark.parametrize(("images", "check"), _PAST_THE_RECORD)
def test_clairic_images_past_the_record_do_not_fit_and_make_no_call(
    images: tuple[ImagePart, ...], check: str
) -> None:
    wire = _Wire()
    request = WireRequest(state=State(parts=images), questions=EXTRACTION.questions[:1])
    assert _refused_by_preflight(_clairic(wire), request).check == check
    assert wire.requests == []


@pytest.mark.parametrize(
    ("image", "sent"),
    [
        pytest.param(ImagePart(data=b"GIF89a", media_type="image/gif"), "image/gif", id="declared"),
        pytest.param(ImagePart(data=b"GIF89a"), "application/octet-stream", id="untyped"),
    ],
)
def test_a_leaf_called_past_fit_sends_an_image_it_does_not_check(
    image: ImagePart, sent: str
) -> None:
    state = State(parts=(image,))
    wire = _Wire(_ok(_clef_result({})))
    _call(_clef(wire), WireRequest(state=state, questions=CLEF_QUESTIONS["noul"][:1]))
    assert wire.body()["images"][0]["content_type"] == sent
    wire = _Wire(_ok(_clairic_result({})))
    _call(_clairic(wire), WireRequest(state=state, questions=EXTRACTION.questions[:1]))
    assert wire.body()["images"][0]["content_type"] == sent


def test_clef_sends_a_read_only_mapping_in_a_json_part() -> None:
    wire = _Wire(_ok(_clef_result({})))
    state = State(parts=(JsonPart(value=MappingProxyType({"a": 1})),))
    _call(_clef(wire), WireRequest(state=state, questions=CLEF_QUESTIONS["noul"][:1]))
    assert wire.body()["state"] == {"a": 1}


def test_clef_refuses_a_state_json_cannot_hold_and_a_kind_it_does_not_answer() -> None:
    wire = _Wire()
    odd = State(parts=(JsonPart(value=cast(Any, {1, 2})),))
    error = _raises(_clef(wire), WireRequest(state=odd, questions=CLEF_QUESTIONS["noul"]))
    assert "set is not a JSON value" in error.message
    bad = State(parts=(JsonPart(value=float("nan")),))
    error = _raises(_clef(wire), WireRequest(state=bad, questions=CLEF_QUESTIONS["noul"]))
    assert "cannot be sent as JSON" in error.message
    surrogate = State(parts=(TextPart(text="a\ud800"),))
    error = _raises(_clef(wire), WireRequest(state=surrogate, questions=CLEF_QUESTIONS["noul"]))
    assert "cannot be sent as JSON" in error.message
    deep: Any = 1
    for _ in range(100_000):
        deep = (deep,)
    # A value nested too deep never reaches the wire: it is refused when the part is built.
    with pytest.raises(ValueError, match=r"^JsonPart\.value nests too deep$"):
        JsonPart(value=deep)
    error = _raises(_clef(wire), EXTRACTION)
    assert "does not answer 'extraction'" in error.message
    assert wire.requests == []


# ----------------------------------------------------------------------------------------
# Clef: reading
# ----------------------------------------------------------------------------------------


def test_clef_reads_each_kind() -> None:
    wire = _Wire(_ok(_clef_result(_NOUL)), _ok(_clef_result(_CHOICE)), _ok(_clef_result(_SCORE)))
    leaf = _clef(wire)
    noul = _call(leaf, clef_request("noul")).answers
    assert noul["noul_rain"] == NoulAnswer(probability=0.97)
    choice = _call(leaf, clef_request("choice")).answers["choice_weather"]
    assert choice == LabelAnswer(
        probabilities=(("rain", 0.95), ("sun", 0.04), ("snow", 0.01)),
        confidence=0.9,
        chosen="rain",
    )
    score = _call(leaf, clef_request("score")).answers["score_rain"]
    assert score == LabelAnswer(
        probabilities=(("none", 0.01), ("light", 0.08), ("heavy", 0.91)),
        confidence=0.85,
        expected_level=1.9,
        echo=("No rain", "Light rain", "Heavy rain"),
        echo_of="descriptions",
    )


def test_clef_wrong_shapes_are_faults_and_an_extra_key_is_dropped() -> None:
    answers = {
        "noul_rain": {"type": "choice", "noul": 0.5},
        "noul_sun": {"type": "noul", "noul": "high"},
        "stray": {"type": "noul", "noul": 0.5},
    }
    wire = _Wire(_ok(_clef_result(answers)))
    response = _call(_clef(wire), clef_request("noul"))
    for key, observed in (("noul_rain", "choice"), ("noul_sun", "str")):
        fault = response.answers[key]
        assert isinstance(fault, Fault)
        assert [(f.rule, f.observed) for f in fault.failures] == [(Rule.kind_shape, observed)]
    assert [w.code for w in response.warnings] == [WARNING_EXTRA_ANSWER_KEY, WARNING_NO_SNAPSHOT]


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": 3, "probabilities": {}},
        {"type": "choice", "choice": "rain", "probabilities": []},
        {"type": "choice", "choice": "rain", "probabilities": {"rain": True}},
        {"type": "choice", "choice": "rain", "probabilities": {"rain": 10**400}},
        "not an object",
    ],
)
def test_clef_malformed_choice_answers_are_kind_shape_faults(answer: object) -> None:
    wire = _Wire(_ok(_clef_result({"choice_weather": answer})))
    request = WireRequest(state=TEXT, questions=CLEF_QUESTIONS["choice"][:1])
    fault = _call(_clef(wire), request).answers["choice_weather"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].rule is Rule.kind_shape


def test_clef_score_legend_odd_keys_and_values() -> None:
    odd = {**_SCORE["score_mood"], "legend": {"x": "low", "0": "mid"}, "probabilities": {}}
    bad = {**_SCORE["score_mood"], "legend": {"0": 1}}
    wire = _Wire(_ok(_clef_result({"score_mood": odd})), _ok(_clef_result({"score_mood": bad})))
    request = WireRequest(state=TEXT, questions=CLEF_QUESTIONS["score"][1:])
    leaf = _clef(wire)
    first = _call(leaf, request).answers["score_mood"]
    assert isinstance(first, Fault)
    assert isinstance(first.mapped, LabelAnswer)
    assert first.mapped.echo == ("mid", "low")
    second = _call(leaf, request).answers["score_mood"]
    assert isinstance(second, Fault)
    assert second.failures[0].rule is Rule.kind_shape


def test_clef_provenance() -> None:
    wire = _Wire(_ok(_clef_result(_NOUL)))
    response = _call(_clef(wire), clef_request("noul"))
    assert response.elicitation == "passthrough"
    assert response.calibrated is False
    assert (response.model_reported, response.snapshot) == ("clef", None)
    assert response.request_id == _RAY
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (0, 0)
    assert dict(response.usage.raw_usage) == _USAGE
    assert isinstance(response.raw_body, Mapping)
    assert set(response.raw_body) == {"success", "errors", "messages", "result"}
    [warning] = response.warnings
    assert (warning.code, warning.field) == (WARNING_NO_SNAPSHOT, "snapshot")
    assert "alias drift cannot be detected" in warning.message
    assert response.prompt_sent == {
        "model": "clef",
        "state": TEXT.parts[0].text,  # type: ignore[union-attr]
        "questions": {
            "noul_rain": {"type": "noul", "instructions": "Did it rain?"},
            "noul_sun": {"type": "noul", "instructions": "Did the sun come out?"},
        },
    }


def test_usage_counts_that_are_not_integers_are_none() -> None:
    result = {**_clef_result(_NOUL), "usage": {"input_tokens": "12", "output_tokens": True}}
    usage = _call(_clef(_Wire(_ok(result))), clef_request("noul")).usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (None, None)


# ----------------------------------------------------------------------------------------
# The envelope and the status mapper
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (_envelope(_clef_result(_NOUL), success=False), "does not report success"),
        ({"success": True, "errors": []}, "no result object"),
        (_envelope([1, 2]), "no result object"),
        (_envelope({}), "no answers map"),
        (_envelope({"model": 1, "answers": {}, "usage": {}}), "top-level schema"),
        (_envelope({"model": "clef", "answers": {}, "usage": []}), "top-level schema"),
    ],
)
def test_an_envelope_that_cannot_be_read_is_not_retryable(body: object, why: str) -> None:
    wire = _Wire(httpx2.Response(200, headers={"cf-ai-req-id": _RAY}, json=body))
    error = _raises(_clef(wire), clef_request("noul"))
    assert why in error.message
    assert (error.is_retryable, error.status_code, error.request_id) == (False, 200, _RAY)


def _failing(status: int, body: object, **headers: str) -> httpx2.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx2.Response(status, headers={"cf-ai-req-id": _RAY, **headers}, content=content)


_ALLOCATION: Final = "daily free allocation of 10,000 neurons"
"""The wording the wire gave with code 4006, which the errors page does not list."""


def _errors(code: int, message: str) -> dict[str, object]:
    return {
        "success": False,
        "errors": [{"code": code, "message": message}],
        "messages": [],
        "result": None,
    }


@pytest.mark.parametrize(
    ("status", "body", "retryable", "message"),
    [
        (429, _errors(3036, "daily neurons used"), False, "HTTP 429: 3036: daily neurons used"),
        (429, _errors(3040, "out of capacity"), True, "HTTP 429: 3040: out of capacity"),
        (503, _errors(3036, "daily neurons used"), False, "HTTP 503: 3036: daily neurons used"),
        (429, _errors(4006, _ALLOCATION), False, f"HTTP 429: 4006: {_ALLOCATION}"),
        (503, _errors(4006, _ALLOCATION), False, f"HTTP 503: 4006: {_ALLOCATION}"),
        (404, _errors(3042, "no such model"), False, "HTTP 404: 3042: no such model"),
        (400, _errors(5007, "x" * 400), False, "HTTP 400: 5007: " + "x" * 294 + "..."),
        (503, b"<html>bad gateway</html>", True, "HTTP 503: <html>bad gateway</html>"),
        (500, {"errors": "none"}, True, "HTTP 500"),
        (500, {"errors": []}, True, "HTTP 500"),
        (500, {"errors": ["text"]}, True, "HTTP 500"),
        (500, [1], True, "HTTP 500"),
        (502, {"errors": [{"code": "x", "message": "upstream"}]}, True, "HTTP 502: upstream"),
        (422, {"errors": [{"code": 3030}]}, False, "HTTP 422: 3030"),
    ],
)
def test_the_status_mapper_reads_the_envelopes_error(
    status: int, body: object, retryable: bool, message: str
) -> None:
    wire = _Wire(_failing(status, body, **{"retry-after": "2"}))
    error = _raises(_clef(wire), clef_request("noul"))
    assert error.message == message
    assert (error.is_retryable, error.status_code, error.request_id) == (retryable, status, _RAY)
    assert error.retry_after == 2.0


def test_an_error_body_nested_too_deeply_still_maps_to_a_provider_error() -> None:
    wire = _Wire(_failing(500, b"[" * 100_000))
    error = _raises(_clef(wire), clef_request("noul"))
    assert error.message.startswith("HTTP 500")
    assert error.is_retryable


def test_a_legend_key_too_long_to_be_an_index_sorts_last() -> None:
    long = {**_SCORE["score_mood"], "legend": {"9" * 5000: "low", "0": "mid"}, "probabilities": {}}
    wire = _Wire(_ok(_clef_result({"score_mood": long})))
    request = WireRequest(state=TEXT, questions=CLEF_QUESTIONS["score"][1:])
    answer = _call(_clef(wire), request).answers["score_mood"]
    assert isinstance(answer, Fault)
    assert isinstance(answer.mapped, LabelAnswer)
    assert answer.mapped.echo == ("mid", "low")


def test_an_envelope_error_quoting_a_bearer_is_filtered_before_it_is_cut() -> None:
    token = "Bearer " + "a" * 40
    # Cut first, only six of the token's characters would remain, which no shape matches.
    wire = _Wire(_failing(400, _errors(5007, "x" * 280 + " " + token)))
    error = _raises(_clef(wire), clef_request("noul"))
    assert "aaaa" not in error.message
    assert "<filtered>" in error.message


def test_a_passed_deadline_is_a_retryable_error_and_sends_nothing() -> None:
    wire = _Wire()
    leaf = _clef(wire)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(clef_request("noul"), Deadline(at=0.0)))
    assert raised.value.is_retryable
    assert wire.requests == []


# ----------------------------------------------------------------------------------------
# Construction, records, estimators, the registry
# ----------------------------------------------------------------------------------------


def test_the_records_are_as_declared() -> None:
    wire = _Wire()
    assert _clef(wire).capabilities == clef_record("clef")
    assert _clef(wire, model="clef-flash").capabilities == clef_record("clef-flash")
    assert _clairic(wire).capabilities == clairic_record()
    custom = clef_record("clef")
    assert _clef(wire, capabilities=custom).capabilities is custom
    assert _clairic(wire, capabilities=custom).capabilities is custom


def test_names_models_and_repr() -> None:
    wire = _Wire()
    clef = _clef(wire, model="clef-flash")
    clairic = _clairic(wire)
    assert (clef.name, clef.model, repr(clef)) == (
        "cloudflare-clef",
        "clef-flash",
        "Clef(model='clef-flash')",
    )
    assert (clairic.name, clairic.model, repr(clairic)) == (
        "cloudflare-clairic",
        "clairic",
        "Clairic()",
    )
    assert TOKEN not in repr(clef)
    assert ACCOUNT not in repr(clairic)


def test_construction_refuses_an_unknown_model_and_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="does not know the model 'clef-omni'"):
        Clef(model="clef-omni", account_id=ACCOUNT, api_token=TOKEN)  # type: ignore[arg-type]
    monkeypatch.delenv(ACCOUNT_ENV, raising=False)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match=ACCOUNT_ENV):
        Clef(api_token=TOKEN)
    with pytest.raises(ValueError, match=TOKEN_ENV):
        Clairic(account_id=ACCOUNT)
    monkeypatch.setenv(ACCOUNT_ENV, "from-env")
    monkeypatch.setenv(TOKEN_ENV, "env-token")
    wire = _Wire(_ok(_clef_result(_NOUL)))
    leaf = Clef(client=wire.client())
    _call(leaf, clef_request("noul"))
    assert "/accounts/from-env/" in str(wire.requests[0].url)
    assert wire.requests[0].headers["authorization"] == "Bearer env-token"


@pytest.mark.parametrize("model", ["clairic", "@cf/cloudflare/clairic"])
def test_clairic_takes_its_one_model_by_either_name(model: Any) -> None:
    wire = _Wire(_ok(_clairic_result({})))
    leaf = _clairic(wire, model=model)
    assert (leaf.model, repr(leaf)) == ("clairic", "Clairic()")
    _call(leaf, EXTRACTION)
    assert str(wire.requests[0].url) == f"{_URL}/@cf/cloudflare/clairic"
    assert wire.body()["model"] == "clairic"
    provider = idia.provider("cloudflare-clairic", model=model, account_id=ACCOUNT, api_token=TOKEN)
    assert provider.capabilities == clairic_record()
    asyncio.run(provider.aclose())


@pytest.mark.parametrize("model", ["clef", "@cf/cloudflare/clef", "Clairic", ""])
def test_clairic_refuses_any_other_model(model: Any) -> None:
    known = "known: clairic, @cf/cloudflare/clairic"
    with pytest.raises(ValueError, match=re.escape(f"model {model!r}; {known}")):
        Clairic(model=model, account_id=ACCOUNT, api_token=TOKEN)


@pytest.mark.parametrize(
    ("model", "short"),
    [
        ("clef", "clef"),
        ("@cf/cloudflare/clef", "clef"),
        ("clef-flash", "clef-flash"),
        ("@cf/cloudflare/clef-flash", "clef-flash"),
    ],
)
def test_each_clef_model_is_taken_by_either_name(model: Any, short: str) -> None:
    wire = _Wire(_ok(_clef_result(_NOUL)))
    leaf = _clef(wire, model=model)
    assert (leaf.model, repr(leaf)) == (short, f"Clef(model={short!r})")
    assert leaf.capabilities == clef_record(short)
    _call(leaf, clef_request("noul"))
    assert str(wire.requests[0].url) == f"{_URL}/@cf/cloudflare/{short}"
    assert wire.body()["model"] == short
    provider = idia.provider("cloudflare-clef", model=model, account_id=ACCOUNT, api_token=TOKEN)
    assert provider.capabilities == clef_record(short)
    asyncio.run(provider.aclose())


@pytest.mark.parametrize("model", ["clairic", "@cf/cloudflare/clairic", "Clef", "cf/clef", ""])
def test_clef_refuses_any_other_model(model: Any) -> None:
    known = "known: clef, clef-flash, @cf/cloudflare/clef, @cf/cloudflare/clef-flash"
    with pytest.raises(ValueError, match=re.escape(f"model {model!r}; {known}")):
        Clef(model=model, account_id=ACCOUNT, api_token=TOKEN)


def test_aclose_closes_only_a_client_the_leaf_created() -> None:
    created = Clef(account_id=ACCOUNT, api_token=TOKEN)
    asyncio.run(created.aclose())
    assert created.client.is_closed
    asyncio.run(created.aclose())
    assert created.client.is_closed
    wire = _Wire()
    injected = _clairic(wire)
    asyncio.run(injected.aclose())
    asyncio.run(injected.aclose())
    assert not injected.client.is_closed


def test_the_estimators() -> None:
    clef = _clef(_Wire()).estimator
    state_only = clef.estimate(TEXT, ())
    text = json.dumps(TEXT.parts[0].text)  # type: ignore[union-attr]
    assert state_only.tokens == 260 + -(-len(text) // 4)
    assert state_only.margin == 0.1
    with_image = clef.estimate(IMAGE_REQUEST.state, IMAGE_REQUEST.questions)
    assert with_image.tokens > 1_024 + 260
    clairic = _clairic(_Wire()).estimator
    estimate = clairic.estimate(IMAGE_REQUEST.state, CLAIRIC_PROVIDER.questions)
    text = cast(TextPart, IMAGE_REQUEST.state.parts[0]).text
    assert estimate.tokens == -(-len(text) // 3) + 2 * 150 + 1_024
    assert estimate.margin == 0.1


def test_the_registry_builds_each_leaf_under_its_name() -> None:
    assert registry._EXTRAS["cloudflare"] == ("httpx2",)
    for name, record in (
        ("cloudflare-clef", clef_record("clef")),
        ("cloudflare-clairic", clairic_record()),
    ):
        entry = registry._REGISTRY[name]
        assert entry.extra == "cloudflare"
        provider = idia.provider(name, account_id=ACCOUNT, api_token=TOKEN)
        assert isinstance(provider, Dispatcher)
        assert provider.capabilities == record
        asyncio.run(provider.aclose())


# ----------------------------------------------------------------------------------------
# Clairic: the body
# ----------------------------------------------------------------------------------------


def _extraction(
    key: str, value_type: str, *, is_list: bool = False, instruction: str = "What?"
) -> Question:
    return Question(
        key=key,
        kind="extraction",
        instruction=instruction,
        value_type=value_type,  # type: ignore[arg-type]
        is_list=is_list,
    )


def test_clairic_fields_carry_the_type_and_instruction_only() -> None:
    wire = _Wire(_ok(_clairic_result({})))
    response = _call(_clairic(wire), EXTRACTION)
    request = wire.requests[0]
    assert str(request.url) == f"{_URL}/@cf/cloudflare/clairic"
    body = wire.body()
    assert list(body) == ["model", "state", "fields", "reference_date"]
    assert body["model"] == "clairic"
    assert body["reference_date"] == "2026-10-21"
    assert body["fields"] == {
        "home_team": {"type": "text", "instructions": "The home team."},
        "refunds": {
            "type": "number",
            "instructions": "The ticket refunds, in pounds, to two decimals.",
        },
        "match_date": {"type": "date", "instructions": "The date the match was due to be played."},
        "referees": {
            "type": "list",
            "instructions": "The referees' names.",
            "items": "text",
        },
        "inspection": {"type": "duration", "instructions": "How long the pitch inspection took."},
        "attendance": {
            "type": "number",
            "instructions": "The number of spectators at the ground.",
        },
    }
    assert isinstance(response.prompt_sent, Mapping)
    assert response.prompt_sent["reference_date"] == "2026-10-21"
    assert b'"reference_date"' in request.content
    assert b", " not in request.content[:40]
    assert not any(w.code == WARNING_UNKNOWN_OPTION for w in response.warnings)


def test_clairic_sends_no_reference_date_without_the_option() -> None:
    wire = _Wire(_ok(_clairic_result({})))
    response = _call(_clairic(wire), replace(EXTRACTION, options={}))
    assert "reference_date" not in wire.body()
    assert isinstance(response.prompt_sent, Mapping)
    assert "reference_date" not in response.prompt_sent
    wire = _Wire(_ok(_clairic_result({})))
    _call(_clairic(wire), replace(EXTRACTION, options={"reference_date": None}))
    assert "reference_date" not in wire.body()


@pytest.mark.parametrize("held", ["21 October 2026", "2026-13-01", 20261021, ["2026-10-21"]])
def test_clairic_refuses_a_reference_date_that_is_not_an_iso_date(held: object) -> None:
    wire = _Wire()
    request = replace(EXTRACTION, options={"reference_date": held})
    error = _raises(_clairic(wire), request)
    assert "reference_date option must be an ISO date string" in error.message
    assert not error.is_retryable
    assert wire.requests == []


def test_clairic_refuses_a_reference_date_too_long_to_print() -> None:
    wire = _Wire()
    request = replace(EXTRACTION, options={"reference_date": 10**5000})
    error = _raises(_clairic(wire), request)
    assert error.message.endswith("not int")
    assert wire.requests == []


def test_clef_takes_clairics_reference_date_without_a_warning() -> None:
    wire = _Wire(_ok(_clef_result(_NOUL)), _ok(_clef_result(_NOUL)))
    leaf = _clef(wire)
    plain = _call(leaf, clef_request("noul"))
    dated = _call(leaf, replace(clef_request("noul"), options=CLAIRIC_OPTIONS))
    assert wire.requests[0].content == wire.requests[1].content
    assert "reference_date" not in wire.body(1)
    assert dated.answers == plain.answers
    assert dated.warnings == plain.warnings
    assert not any(w.code == WARNING_UNKNOWN_OPTION for w in dated.warnings)
    assert isinstance(dated.prompt_sent, Mapping)
    assert "reference_date" not in dated.prompt_sent


@pytest.mark.parametrize("build", ["clef", "clairic"])
def test_an_unknown_option_is_ignored_with_one_warning(build: str) -> None:
    if build == "clef":
        wire = _Wire(_ok(_clef_result(_NOUL)), _ok(_clef_result(_NOUL)))
        leaf: Clef | Clairic = _clef(wire)
        request = clef_request("noul")
    else:
        wire = _Wire(_ok(_clairic_result({})), _ok(_clairic_result({})))
        leaf = _clairic(wire)
        request = EXTRACTION
    plain = _call(leaf, request)
    options = {**request.options, "zeta": 1, "alpha": True}
    warned = _call(leaf, replace(request, options=options))
    assert wire.requests[0].content == wire.requests[1].content
    assert warned.answers == plain.answers
    [extra] = [w for w in warned.warnings if w not in plain.warnings]
    assert (extra.code, extra.field) == (WARNING_UNKNOWN_OPTION, "options")
    assert extra.message.endswith("alpha, zeta")
    assert len(warned.warnings) == len(plain.warnings) + 1


def test_clairic_images_come_before_the_state_and_an_image_only_state_sends_none() -> None:
    wire = _Wire(_ok(_clairic_result({})), _ok(_clairic_result({})), _ok(_clairic_result({})))
    leaf = _clairic(wire)
    _call(leaf, WireRequest(state=IMAGE_REQUEST.state, questions=CLAIRIC_PROVIDER.questions))
    assert list(wire.body(0)) == ["model", "images", "state", "fields"]
    only = State(parts=(ImagePart(data=png()),))
    _call(leaf, WireRequest(state=only, questions=CLAIRIC_PROVIDER.questions))
    assert list(wire.body(1)) == ["model", "images", "fields"]
    _call(leaf, WireRequest(state=State(parts=()), questions=CLAIRIC_PROVIDER.questions))
    assert wire.body(2)["state"] == ""


@pytest.mark.parametrize(
    ("state", "questions", "message"),
    [
        (
            State(parts=(TextPart(text="a"), TextPart(text="b"))),
            CLAIRIC_PROVIDER.questions,
            "one text part",
        ),
        (
            State(parts=(TextPart(text="a"), JsonPart(value=1))),
            CLAIRIC_PROVIDER.questions,
            "not a JSON part",
        ),
        (TEXT, (_extraction("times", "time", is_list=True),), "not 'time'"),
        (TEXT, (_extraction("when", "datetime", is_list=True),), "not 'datetime'"),
        (TEXT, (_extraction("took", "duration", is_list=True),), "not 'duration'"),
        (TEXT, CLEF_QUESTIONS["noul"], "not 'noul'"),
    ],
)
def test_clairic_refuses_before_sending(
    state: State, questions: tuple[Question, ...], message: str
) -> None:
    wire = _Wire()
    error = _raises(_clairic(wire), WireRequest(state=state, questions=questions))
    assert message in error.message
    assert not error.is_retryable
    assert wire.requests == []


# ----------------------------------------------------------------------------------------
# Clairic: reading
# ----------------------------------------------------------------------------------------


def _read(
    questions: tuple[Question, ...],
    answers: Mapping[str, object],
    *,
    state: State = TEXT,
    **extra: object,
) -> WireResponse:
    wire = _Wire(_ok(_clairic_result(answers, **extra)))
    return _call(_clairic(wire), WireRequest(state=state, questions=questions))


def test_clairic_reads_each_value_type_as_given() -> None:
    questions = (
        _extraction("name", "text"),
        _extraction("amount", "number"),
        _extraction("day", "date"),
        _extraction("at", "time"),
        _extraction("when", "datetime"),
        _extraction("took", "duration"),
    )
    answers = {
        "name": {"type": "text", "value": "Riverside Rovers", "p_null": 0.01, "span": [0, 16]},
        "amount": {"type": "number", "value": 1080, "confidence": 0.9, "p_null": 0.0},
        "day": {"type": "date", "value": "2026-10-17", "p_null": 0.02},
        "at": {"type": "time", "value": "15:00", "p_null": 0.1},
        "when": {"type": "datetime", "value": "2026-10-18T14:30", "p_null": 0.1},
        "took": {"type": "duration", "value": 2700, "unit": "seconds", "p_null": 0.1},
    }
    read = _read(questions, answers).answers
    assert read["name"] == ExtractionAnswer(
        value="Riverside Rovers", p_null=0.01, span=Span(part=0, start=0, end=16)
    )
    assert read["amount"] == ExtractionAnswer(value=1080, confidence=0.9, p_null=0.0)
    assert read["day"] == ExtractionAnswer(value=datetime.date(2026, 10, 17), p_null=0.02)
    assert read["at"] == ExtractionAnswer(value=datetime.time(15, 0), p_null=0.1)
    assert read["when"] == ExtractionAnswer(
        value=datetime.datetime(2026, 10, 18, 14, 30), p_null=0.1
    )
    assert read["took"] == ExtractionAnswer(value=datetime.timedelta(seconds=2700), p_null=0.1)


def test_clairic_a_null_value_keeps_its_guess_and_span() -> None:
    answer = {"type": "number", "value": None, "guess": 4000, "p_null": 1, "span": [3, 7]}
    read = _read((_extraction("crowd", "number"),), {"crowd": answer}).answers["crowd"]
    assert read == ExtractionAnswer(
        value=None, guess=4000, p_null=1.0, span=Span(part=0, start=3, end=7)
    )


def test_clairic_a_list_has_spans_and_no_p_null() -> None:
    question = _extraction("refs", "date", is_list=True)
    answer = {
        "type": "list",
        "value": ["2026-10-17", "2026-10-18"],
        "spans": [[1, 5], [9, 12]],
        "confidence": 0.8,
    }
    state = State(parts=(ImagePart(data=png()), TextPart(text="x" * 20)))
    read = _read((question,), {"refs": answer}, state=state).answers["refs"]
    assert read == ExtractionAnswer(
        value=(datetime.date(2026, 10, 17), datetime.date(2026, 10, 18)),
        confidence=0.8,
        span=(Span(part=1, start=1, end=5), Span(part=1, start=9, end=12)),
    )
    bare = _read((question,), {"refs": {"type": "list", "value": None}}).answers["refs"]
    assert bare == ExtractionAnswer(value=None)


def test_clairic_with_no_text_part_carries_no_span() -> None:
    only = State(parts=(ImagePart(data=png()),))
    answer = {"type": "text", "value": "red", "p_null": 0.0, "span": [0, 3]}
    read = _read((_extraction("colour", "text"),), {"colour": answer}, state=only)
    assert read.answers["colour"] == ExtractionAnswer(value="red", p_null=0.0)


def test_clairic_a_value_that_does_not_read_is_left_for_validation() -> None:
    answers = {
        "day": {"type": "date", "value": "17 October", "p_null": 0.0},
        "took": {"type": "duration", "value": "PT45M", "p_null": 0.0},
        "far": {"type": "duration", "value": 1e300, "p_null": 0.0},
        "hours": {"type": "duration", "value": 2, "unit": "hours", "p_null": 0.0},
        "names": {"type": "list", "value": "Ana", "spans": None},
    }
    questions = (
        _extraction("day", "date"),
        _extraction("took", "duration"),
        _extraction("far", "duration"),
        _extraction("hours", "duration"),
        _extraction("names", "text", is_list=True),
    )
    read = _read(questions, answers).answers
    for key in answers:
        fault = read[key]
        assert isinstance(fault, Fault), key
        assert [f.rule for f in fault.failures] == [Rule.value_type]


@pytest.mark.parametrize(
    ("answer", "candidate"),
    [
        ({"type": "estimate", "value": 4000, "mean": 4100.0, "p10": 3000}, "estimate"),
        ({"type": "forecast", "value": None, "steps": []}, "forecast"),
        ({"type": "number", "value": 4000, "median": 4000}, "estimate"),
        ({"type": "number", "value": 4000, "steps": [1]}, "forecast"),
    ],
)
def test_clairic_a_distribution_is_a_candidate_shape_fault(
    answer: Mapping[str, object], candidate: str
) -> None:
    fault = _read((_extraction("crowd", "number"),), {"crowd": answer}).answers["crowd"]
    assert fault == Fault(
        key="crowd",
        failures=(idia.RuleFailure(rule=Rule.candidate_shape, observed=candidate, bound=None),),
        mapped=None,
    )


@pytest.mark.parametrize(
    ("answer", "observed"),
    [
        ({"type": "text", "value": "a", "p_null": 0}, "text"),
        ({"type": "number", "p_null": 0}, "no value"),
        ({"type": "number", "value": 1, "span": [1]}, "span (1,)"),
        ({"type": "number", "value": 1, "span": [True, 2]}, "span (True, 2)"),
        ({"type": "number", "value": 1, "p_null": "low"}, "str"),
    ],
)
def test_clairic_wrong_shapes_are_kind_shape_faults(
    answer: Mapping[str, object], observed: str
) -> None:
    fault = _read((_extraction("n", "number"),), {"n": answer}).answers["n"]
    assert isinstance(fault, Fault)
    assert [(f.rule, f.observed) for f in fault.failures] == [(Rule.kind_shape, observed)]


def test_clairic_list_spans_must_be_a_list() -> None:
    answer = {"type": "list", "value": ["a"], "spans": "0-1"}
    question = _extraction("names", "text", is_list=True)
    fault = _read((question,), {"names": answer}).answers["names"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].rule is Rule.kind_shape


def test_clairic_truncation_is_a_warning_and_the_usage_is_kept_raw() -> None:
    answers = {"home_team": {"type": "text", "value": "Riverside Rovers", "p_null": 0.0}}
    response = _read(EXTRACTION.questions[:1], answers, truncated=True, state_text="ignored")
    assert response.answers["home_team"] == ExtractionAnswer(value="Riverside Rovers", p_null=0.0)
    assert [(w.code, w.field) for w in response.warnings] == [
        (WARNING_PROVIDER_TRUNCATED, "state"),
        (WARNING_NO_SNAPSHOT, "snapshot"),
    ]
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (250, 30)
    assert response.usage.passes == 1
    assert response.model_reported == "clairic"
    untruncated = _read(EXTRACTION.questions[:1], answers, truncated=False)
    assert [w.code for w in untruncated.warnings] == [WARNING_NO_SNAPSHOT]


def test_the_past_context_cassette_replays_truncated_with_its_usage() -> None:
    """The wire's answer to a state past its context, sent to the leaf directly since Fit
    refuses the request: one request, read with the truncation warning and the usage
    recorded. The cassette holds its body with sorted keys, so it is replayed by name and
    the body sent compared by value (:data:`PAST_CONTEXT_TRUNCATED`)."""
    cassette = load_cassette(PAST_CONTEXT_TRUNCATED)
    assert cassette.wire_request == CLAIRIC_FAILURE
    stored = cassette.response
    headers = replay_headers(stored["headers"])
    wire = _Wire(httpx2.Response(stored["status"], headers=headers, json=stored["body"]))
    response = _call(_clairic(wire), CLAIRIC_FAILURE)
    [request] = wire.requests
    assert (request.method, str(request.url), wire.body()) == (
        cassette.request["method"],
        cassette.request["url"],
        cassette.request["body"],
    )
    assert WARNING_PROVIDER_TRUNCATED in [w.code for w in response.warnings]
    assert response.answers == cassette.expected
    assert response.usage is not None
    assert response.usage.input_tokens == 65_523
    assert response.usage.raw_usage["state_tokens"] == 65_388
    assert response.usage.passes == 1
    assert response.usage.raw_usage["passes"] == 1


def test_usage_passes_is_normalized_from_the_wire_and_none_when_unreported() -> None:
    answers = {"home_team": {"type": "text", "value": "Riverside Rovers", "p_null": 0.0}}
    usage = {"input_tokens": 250, "output_tokens": 0, "passes": 2}
    reported = _read(EXTRACTION.questions[:1], answers, usage=usage).usage
    assert reported is not None
    assert (reported.passes, reported.raw_usage["passes"]) == (2, 2)
    for odd in ("2", True, 2.0):
        held = _read(EXTRACTION.questions[:1], answers, usage={**usage, "passes": odd}).usage
        assert held is not None
        assert held.passes is None
        assert held.raw_usage["passes"] == odd
    missing = _read(EXTRACTION.questions[:1], answers, usage={"input_tokens": 250}).usage
    assert missing is not None
    assert missing.passes is None
    clef = _call(_clef(_Wire(_ok(_clef_result(_NOUL)))), clef_request("noul")).usage
    assert clef is not None
    assert clef.passes is None
    assert "passes" not in clef.raw_usage


# ----------------------------------------------------------------------------------------
# The question-order probe: one request twice, then reversed
# ----------------------------------------------------------------------------------------


def _answering_with(
    response: Mapping[str, Any],
) -> tuple[list[httpx2.Request], httpx2.MockTransport]:
    """A mock transport answering every request with one stored response, and the list
    it keeps each request in."""
    sent: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        headers = replay_headers(response["headers"])
        return httpx2.Response(response["status"], headers=headers, json=response["body"])

    return sent, httpx2.MockTransport(answer)


def _order_replay(directory: Path, build: Callable[[httpx2.AsyncClient], Clef | Clairic]) -> str:
    """Replay each of the probe's three cassettes alone, since ``a-1`` and ``a-2`` share
    a body, which :class:`CassetteTransport` refuses. Each replay sends the request its
    cassette holds, the body the cassette stored, in its order, and reads the answers
    stored; returns the body's member of questions, ``questions`` or ``fields``."""
    order = directory / ORDER
    assert sorted(path.name for path in order.iterdir()) == [f"{n}.json" for n in ORDER_NAMES]
    sent_orders: list[list[str]] = []
    member = ""
    for path, asked in zip(
        (order / f"{name}.json" for name in ORDER_NAMES),
        order_requests(load_cassette(order / f"{ORDER_NAMES[0]}.json").wire_request).values(),
        strict=True,
    ):
        cassette = load_cassette(path)
        assert cassette.wire_request == asked
        stored = cast(dict[str, Any], cassette.request["body"])
        member = "questions" if "questions" in stored else "fields"
        sent, transport = _answering_with(cassette.response)
        leaf = build(httpx2.AsyncClient(transport=transport))
        response = _call(leaf, cassette.wire_request)
        [request] = sent
        body = cast(dict[str, Any], parse_body(request.content))
        assert (request.method, str(request.url), body_key(body)) == (
            cassette.request["method"],
            cassette.request["url"],
            body_key(stored),
        )
        assert response.answers == cassette.expected
        sent_orders.append(list(stored[member]))
    assert sent_orders[0] == sent_orders[1] == sent_orders[2][::-1] != sent_orders[2]
    return member


def test_the_question_id_probe_differs_from_its_control_only_in_the_id() -> None:
    """The probe's body, its one id renamed back, is the body of a recorded success, key
    for key and in order; so the wire's answer to the probe is its answer to the id."""
    control = load_cassette(clef_dir("clef") / f"{QUESTION_ID_CONTROL}.json")
    assert control.response["status"] == 200
    [(key, question)] = cast(dict[str, object], QUESTION_ID_BODY["questions"]).items()
    assert key == QUESTION_ID_KEY
    assert _CLEF_ID.fullmatch(key) is None
    assert _CLEF_ID.fullmatch(key.replace(":", "_")) is not None
    [control_key] = control.request["body"]["questions"]
    renamed = {**QUESTION_ID_BODY, "questions": {control_key: question}}
    assert renamed == control.request["body"]
    assert list(renamed) == list(control.request["body"])


def test_the_question_id_probe_records_the_wire_refusing_the_id() -> None:
    """``clef`` refused the probe's id, outside its pattern, with 422 and code 3030, and the
    leaf reads that answer as a non-retryable error naming both."""
    cassette = load_cassette(clef_dir("clef") / f"{QUESTION_ID_PROBE}.json")
    assert cassette.request["body"] == QUESTION_ID_BODY
    assert cassette.wire_request == QUESTION_ID_REQUEST
    assert cassette.expected is None
    status = cassette.response["status"]
    body = cassette.response["body"]
    assert status == 422
    assert body["errors"][0]["code"] == 3030
    response = httpx2.Response(status, content=json.dumps(body).encode())
    error = Clef(account_id=ACCOUNT, api_token=TOKEN).map_status(response)
    assert not error.is_retryable
    assert error.message.startswith("HTTP 422: 3030: ")


def test_the_leaf_sends_the_question_id_probe_under_q0() -> None:
    """What the probe sends raw, the leaf sends under ``q0``."""
    wire = _Wire(_failing(400, _errors(5006, "refused")))
    with pytest.raises(ProviderError):
        asyncio.run(_clef(wire).call(QUESTION_ID_REQUEST, Deadline.after(5)))
    sent = wire.body()
    assert sent["questions"] == {
        "q0": cast(dict[str, object], QUESTION_ID_BODY["questions"])[QUESTION_ID_KEY]
    }
    assert {k: v for k, v in sent.items() if k != "questions"} == {
        k: v for k, v in QUESTION_ID_BODY.items() if k != "questions"
    }


@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_the_clef_question_order_probe_replays_as_recorded(model: str) -> None:
    """Clef's question-order probe, recorded 2026-10-22: :data:`CLEF_ORDER` twice, then
    reversed. The two unchanged sends answered identically on both models (largest move
    0). Reversed, clef moved the weather's ``rain`` by 0.0528 (0.9731 to 0.9203) and its
    confidence by 0.1432, the mood's expected level by 0.0129; clef-flash moved the mood's
    ``low`` by 0.0320 (0.9466 to 0.9146), its confidence by 0.0850 and its expected level
    by 0.0380. Usage was 433 input tokens for all six. The moves are evidence, not a
    contract: the test asserts only that each cassette replays to its stored answers."""

    def build(client: httpx2.AsyncClient) -> Clef:
        return Clef(model=model, account_id=ACCOUNT, api_token=TOKEN, client=client)  # type: ignore[arg-type]

    assert load_cassette(clef_dir(model) / ORDER / "order-a-1.json").wire_request == CLEF_ORDER
    assert _order_replay(clef_dir(model), build) == "questions"


def test_the_clairic_field_order_probe_replays_as_recorded() -> None:
    """Clairic's field-order probe, recorded 2026-10-22: :data:`CLAIRIC_PROVIDER` twice,
    then with its two fields reversed. All three answered identically, to the last digit:
    ``home_team`` null at ``p_null`` 0.6884 with confidence 0.6442, ``refunds`` 4250.5;
    the largest move of any ``p_null`` or confidence, unchanged or reversed, is 0. The
    moves are evidence, not a contract."""

    def build(client: httpx2.AsyncClient) -> Clairic:
        return Clairic(account_id=ACCOUNT, api_token=TOKEN, client=client)

    first = load_cassette(CLAIRIC_DIR / ORDER / "order-a-1.json")
    assert first.wire_request == CLAIRIC_PROVIDER
    assert _order_replay(CLAIRIC_DIR, build) == "fields"


def test_a_reversed_clef_request_misses_the_cassette_of_its_first_order(tmp_path: Path) -> None:
    """The match compares key order: Clef's questions are a JSON object, so the probe's
    reversed request, equal in value to the first order's, is not answered by the first
    order's cassette, which answers its own request."""
    first = clef_dir("clef") / ORDER / "order-a-1.json"
    (tmp_path / first.name).write_bytes(first.read_bytes())
    transport = CassetteTransport(tmp_path)
    leaf = Clef(account_id=ACCOUNT, api_token=TOKEN, client=httpx2.AsyncClient(transport=transport))
    asked = order_requests(CLEF_ORDER)
    with pytest.raises(AssertionError, match="no cassette matches"):
        _call(leaf, asked["order-b"])
    assert transport.served == []
    assert len(transport.misses) == 1
    response = _call(leaf, asked["order-a-1"])
    assert transport.served == [tmp_path / first.name]
    assert response.answers == load_cassette(first).expected


# ----------------------------------------------------------------------------------------
# Key aliases: the caller's keys when the wire takes them all, else q0..qN or f0..fN
# ----------------------------------------------------------------------------------------

_REF: Final = "q:match/intent"


def _noul(key: str) -> Question:
    return Question(key=key, kind="noul", instruction="Did it rain?")


@pytest.mark.parametrize(
    ("keys", "sent"),
    [
        (("noul_rain", "a" * 100), ["noul_rain", "a" * 100]),
        ((_REF,), ["q0"]),
        (("a" * 101,), ["q0"]),
        (("noul_rain", _REF, "noul_sun"), ["q0", "q1", "q2"]),
    ],
)
def test_clef_aliases_every_key_when_any_is_one_the_wire_refuses(
    keys: tuple[str, ...], sent: list[str]
) -> None:
    wire = _Wire(_ok(_clef_result({id_: {"type": "noul", "noul": 0.5} for id_ in sent})))
    request = WireRequest(state=TEXT, questions=tuple(_noul(key) for key in keys))
    response = _call(_clef(wire), request)
    assert list(wire.body()["questions"]) == sent
    assert list(cast(Mapping[str, Any], response.prompt_sent)["questions"]) == sent
    assert response.answers == dict.fromkeys(keys, NoulAnswer(probability=0.5))
    assert [w.code for w in response.warnings] == [WARNING_NO_SNAPSHOT]


def test_clef_maps_aliased_answers_back_by_position() -> None:
    wire = _Wire(_ok(_clef_result({"q1": _NOUL["noul_sun"], "q0": _NOUL["noul_rain"]})))
    request = WireRequest(state=TEXT, questions=(_noul("q:rain"), _noul("q:sun")))
    answers = _call(_clef(wire), request).answers
    assert list(answers) == ["q:rain", "q:sun"]
    assert answers["q:rain"] == NoulAnswer(probability=0.97)
    assert answers["q:sun"] == NoulAnswer(probability=0.81)


def test_clef_an_aliased_fault_and_a_stray_id_name_the_callers_keys() -> None:
    """A stray wire id that equals a caller's key is matched to no question: answers are
    looked up among the ids sent."""
    answers = {
        "q0": {"type": "choice", "noul": 0.5},
        "q1": {"type": "noul", "noul": 0.2},
        "noul_rain": {"type": "noul", "noul": 0.9},
    }
    wire = _Wire(_ok(_clef_result(answers)))
    request = WireRequest(state=TEXT, questions=(_noul("noul_rain"), _noul(_REF)))
    response = _call(_clef(wire), request)
    fault = response.answers["noul_rain"]
    assert isinstance(fault, Fault)
    assert fault.key == "noul_rain"
    assert [(f.rule, f.observed) for f in fault.failures] == [(Rule.kind_shape, "choice")]
    assert response.answers[_REF] == NoulAnswer(probability=0.2)
    assert [(w.code, w.field) for w in response.warnings] == [
        (WARNING_EXTRA_ANSWER_KEY, "noul_rain"),
        (WARNING_NO_SNAPSHOT, "snapshot"),
    ]


def test_clef_estimator_counts_the_ids_as_sent() -> None:
    questions = (_noul("a" * 101), _noul(_REF))
    wire = _Wire(_ok(_clef_result({})))
    _call(_clef(wire), WireRequest(state=TEXT, questions=questions))
    body = wire.body()
    expected = (
        260
        + math.ceil(len(json.dumps(body["state"])) / 4)
        + math.ceil(len(json.dumps(body["questions"])) / 4)
    )
    assert _clef(_Wire()).estimator.estimate(TEXT, questions).tokens == expected


@pytest.mark.parametrize(
    ("keys", "sent"),
    [
        (("home_team", "a" * 64), ["home_team", "a" * 64]),
        ((_REF,), ["f0"]),
        (("a" * 65,), ["f0"]),
        (("1bad",), ["f0"]),
        (("home_team", _REF), ["f0", "f1"]),
    ],
)
def test_clairic_aliases_every_key_when_any_is_one_the_wire_refuses(
    keys: tuple[str, ...], sent: list[str]
) -> None:
    answer = {"type": "text", "value": "Riverside Rovers", "p_null": 0.0}
    wire = _Wire(_ok(_clairic_result(dict.fromkeys(sent, answer))))
    request = WireRequest(state=TEXT, questions=tuple(_extraction(key, "text") for key in keys))
    response = _call(_clairic(wire), request)
    assert list(wire.body()["fields"]) == sent
    assert list(cast(Mapping[str, Any], response.prompt_sent)["fields"]) == sent
    assert response.answers == dict.fromkeys(
        keys, ExtractionAnswer(value="Riverside Rovers", p_null=0.0)
    )


def test_clairic_an_aliased_fault_and_a_stray_id_name_the_callers_keys() -> None:
    answers = {
        "f1": {"type": "text", "value": "Riverside Rovers", "p_null": 0.0},
        "f0": {"type": "number", "value": 4000, "p10": 1},
        "crowd": {"type": "number", "value": 1, "p_null": 0.0},
    }
    questions = (_extraction("crowd", "number"), _extraction(_REF, "text"))
    response = _read(questions, answers, truncated=True)
    assert list(response.answers) == ["crowd", _REF]
    fault = response.answers["crowd"]
    assert isinstance(fault, Fault)
    assert fault.key == "crowd"
    assert [(f.rule, f.observed) for f in fault.failures] == [(Rule.candidate_shape, "estimate")]
    assert response.answers[_REF] == ExtractionAnswer(value="Riverside Rovers", p_null=0.0)
    assert [(w.code, w.field) for w in response.warnings] == [
        (WARNING_EXTRA_ANSWER_KEY, "crowd"),
        (WARNING_PROVIDER_TRUNCATED, "state"),
        (WARNING_NO_SNAPSHOT, "snapshot"),
    ]


def test_clairic_estimator_is_per_field_whatever_the_ids() -> None:
    estimator = _clairic(_Wire()).estimator
    short = estimator.estimate(TEXT, (_extraction("a", "text"),))
    aliased = estimator.estimate(TEXT, (_extraction("a" * 65, "text"),))
    assert short == aliased


def test_a_wire_error_names_the_aliased_id_unchanged() -> None:
    wire = _Wire(_failing(422, _errors(3030, "AiError: invalid field name 'f0'")))
    error = _raises(_clairic(wire), WireRequest(state=TEXT, questions=(_extraction(_REF, "text"),)))
    assert error.message == "HTTP 422: 3030: AiError: invalid field name 'f0'"
    assert error.request_id == _RAY


def test_the_records_declare_no_key_pattern() -> None:
    wire = _Wire()
    for leaf in (_clef(wire), _clef(wire, model="clef-flash"), _clairic(wire)):
        assert leaf.capabilities.key_patterns == ()


@pytest.mark.parametrize(
    ("harness", "prefix"),
    [(ClefHarness(), "q"), (ClefFlashHarness(), "q"), (ClairicHarness(), "f")],
    ids=["clef", "clef-flash", "clairic"],
)
def test_an_alias_cassette_replays_under_the_callers_keys(
    harness: ClefHarness | ClairicHarness, prefix: str
) -> None:
    """The recorded request whose keys no wire pattern takes, through the harness's own replay,
    which asserts the method, URL and body sent against the cassette: the leaf sends the
    aliases, and the wire's answers come back under the caller's keys."""
    cassette = load_cassette(harness.cassette_dir / f"{ALIAS}.json")
    leaf = harness.make([Success()])
    result = asyncio.run(leaf.call(cassette.wire_request, Deadline.after(5.0)))
    assert harness.requests_sent() == 1
    keys = [question.key for question in cassette.wire_request.questions]
    sent = cassette.request["body"]["questions" if prefix == "q" else "fields"]
    assert list(sent) == [f"{prefix}{index}" for index in range(len(keys))]
    assert all(key.startswith("q:match/") for key in keys)
    assert list(result.answers) == keys
    assert result.answers == cassette.expected


@pytest.mark.parametrize("harness", [ClefHarness(), ClefFlashHarness()], ids=["clef", "clef-flash"])
def test_the_clef_wording_cassette_sends_criteria_and_the_structured_instruction(
    harness: ClefHarness,
) -> None:
    """The recorded described ``noul`` and structured instruction, through the harness's
    replay, which asserts the body sent against the cassette."""
    cassette = load_cassette(harness.cassette_dir / f"{WORDING}.json")
    # The cassette's wire request is stored with sorted keys, as every canonical form is,
    # so the call is the request as asked, its instruction's keys in the order sent.
    wire_request = wording_request()
    assert cassette.wire_request == wire_request
    leaf = harness.make([Success()])
    result = asyncio.run(leaf.call(wire_request, Deadline.after(5.0)))
    assert harness.requests_sent() == 1
    assert result.answers == cassette.expected
    assert all(isinstance(answer, NoulAnswer) for answer in result.answers.values())
    described, structured = wire_request.questions
    sent = cassette.request["body"]["questions"]
    assert sent["described"]["criteria"] == {
        "true": described.labels[0].description,
        "false": described.labels[1].description,
    }
    assert sent["structured"]["instructions"] == structured.instruction
    estimate = leaf.estimator.estimate(wire_request.state, wire_request.questions)
    used = cassette.response["body"]["result"]["usage"]["input_tokens"]
    assert used <= estimate.tokens * (1 + estimate.margin)


# ----------------------------------------------------------------------------------------
# The router function
# ----------------------------------------------------------------------------------------


def _members(router: Router) -> list[Dispatcher]:
    members = cast(dict[str, Dispatcher], router._members)
    return list(members.values())


def test_cloudflare_routes_the_label_kinds_to_clef_and_extraction_to_clairic() -> None:
    router = cloudflare(account_id=ACCOUNT, api_token=TOKEN, clef_model="clef-flash")
    members = cast(dict[str, Dispatcher], router._members)
    assert list(members) == ["noul", "choice", "score", "extraction"]
    clef = members["noul"]._leaf
    assert members["choice"] is members["noul"] is members["score"]
    clairic = members["extraction"]._leaf
    assert isinstance(clef, Clef)
    assert isinstance(clairic, Clairic)
    assert clef.model == "clef-flash"
    assert clairic.client is clef.client
    assert router.capabilities == router_record("clef-flash")
    asyncio.run(router.aclose())
    assert clef.client.is_closed


def test_cloudflare_takes_a_clef_models_full_name() -> None:
    router = cloudflare(account_id=ACCOUNT, api_token=TOKEN, clef_model="@cf/cloudflare/clef-flash")
    clef = cast(dict[str, Dispatcher], router._members)["noul"]._leaf
    assert isinstance(clef, Clef)
    assert clef.model == "clef-flash"
    assert router.capabilities == router_record("clef-flash")
    asyncio.run(router.aclose())


def test_cloudflare_leaves_an_injected_client_open_and_shares_the_limiters() -> None:
    client = httpx2.AsyncClient()
    concurrency = ConcurrencyLimit(2)
    rate = RateLimiter(rate=5.0)
    retry = RetryPolicy(attempts=2)
    router = cloudflare(
        account_id=ACCOUNT,
        api_token=TOKEN,
        client=client,
        retry=retry,
        concurrency=concurrency,
        rate=rate,
    )
    label, extraction = dict.fromkeys(_members(router))
    for member in (label, extraction):
        assert member._concurrency is concurrency
        assert member._rate is rate
        assert member._retry is retry
        assert cast(Clef, member._leaf).client is client
    assert router.capabilities == router_record("clef")
    asyncio.run(router.aclose())
    assert not client.is_closed
    asyncio.run(client.aclose())


def test_the_router_answers_both_shares_over_one_connection() -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/clef"):
            return _ok(_clef_result(_CHOICE))
        return _ok(
            _clairic_result(
                {
                    "home_team": {"type": "text", "value": "Riverside Rovers", "p_null": 0.0},
                    "refunds": {"type": "number", "value": 4250.5, "p_null": 0.0},
                }
            )
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
    leaves = (
        Clef(account_id=ACCOUNT, api_token=TOKEN, client=client),
        Clairic(account_id=ACCOUNT, api_token=TOKEN, client=client),
    )
    router = _router(*leaves, retry=RetryPolicy(), concurrency=None, rate=None)
    request = Request(
        state=TEXT,
        questions=(*CLEF_QUESTIONS["choice"], *CLAIRIC_PROVIDER.questions),
        deadline=Deadline.after(5.0),
        options=CLAIRIC_OPTIONS,
    )
    response = asyncio.run(router.answer(request))
    assert sorted(response.answers) == sorted(q.key for q in request.questions)
    assert [o.provider for o in response.outcomes] == ["cloudflare-clef", "cloudflare-clairic"]
    warnings = [*response.warnings, *(w for o in response.outcomes for w in o.warnings)]
    assert not any(w.code == WARNING_UNKNOWN_OPTION for w in warnings)


def test_the_router_refuses_a_json_state_clairic_cannot_take_before_any_call() -> None:
    wire = _Wire()
    client = wire.client()
    router = _router(
        Clef(account_id=ACCOUNT, api_token=TOKEN, client=client),
        Clairic(account_id=ACCOUNT, api_token=TOKEN, client=client),
        retry=RetryPolicy(),
        concurrency=None,
        rate=None,
    )
    request = Request(
        state=State(parts=(JsonPart(value={"score": 3}),)),
        questions=(*CLEF_QUESTIONS["noul"], *CLAIRIC_PROVIDER.questions),
        deadline=Deadline.after(5.0),
    )
    with pytest.raises(DoesNotFit) as refused:
        asyncio.run(router.answer(request))
    assert refused.value.check == "modalities"
    assert wire.requests == []


def test_the_warning_codes_are_pinned() -> None:
    assert WARNING_PROVIDER_TRUNCATED == "provider-truncated"
    assert WARNING_UNKNOWN_OPTION == "unknown-option"


def test_record_cassette_raises_no_exchange_when_the_transport_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clairic maps a transport error to ``ProviderError``, so the recorder keeps no
    exchange: the live module's no-evidence case."""
    monkeypatch.setenv("IDIA_LIVE", "1")

    def refusing(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(NoExchange):
        asyncio.run(
            record_cassette(
                tmp_path / "unsent.json",
                lambda client: Clairic(account_id="account", api_token="token", client=client),
                CLAIRIC_FAILURE,
                Deadline.after(5.0),
                transport=httpx2.MockTransport(refusing),
            )
        )


def test_a_model_gone_is_kept_apart_and_never_overwritten(tmp_path: Path) -> None:
    """The model-gone reply, a side file, moves to ``gone/gone.json``; the case's
    committed cassette is never touched, and a second gone reply is refused, kept where it
    was."""
    failure = tmp_path / "failure.json"
    failure.write_text("committed failure")
    noul = tmp_path / "noul.json"
    noul.write_text("committed noul")
    side = tmp_path / "side" / "noul.json"
    side.parent.mkdir()
    side.write_text("gone reply")
    gone = tmp_path / GONE / f"{GONE}.json"
    assert _keep_gone(noul, side) == gone
    assert gone.read_text() == "gone reply"
    assert not side.parent.exists()
    assert failure.read_text() == "committed failure"
    assert noul.read_text() == "committed noul"
    side.parent.mkdir()
    side.write_text("a second gone reply")
    with pytest.raises(pytest.fail.Exception, match=r"gone\.json already holds"):
        _keep_gone(noul, side)
    assert gone.read_text() == "gone reply"
    assert side.read_text() == "a second gone reply"
    assert noul.read_text() == "committed noul"


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("not found", status_code=404),
        ProviderError("code 5007: no such model", status_code=400),
    ],
)
def test_check_routes_a_model_gone_to_keep_gone(tmp_path: Path, error: ProviderError) -> None:
    """No wire: a model-gone error through the live module's check, on a side file; the
    committed cassette is left as it was."""
    path = tmp_path / "choice.json"
    path.write_text("committed choice")
    side = tmp_path / "side" / "choice.json"
    side.parent.mkdir()
    side.write_text("gone reply")
    with pytest.raises(pytest.fail.Exception, match="the model is gone"):
        _check(path, side, error)
    assert (tmp_path / GONE / f"{GONE}.json").read_text() == "gone reply"
    assert path.read_text() == "committed choice"
    assert not side.parent.exists()


@pytest.mark.parametrize("status", [401, 403])
def test_check_deletes_the_side_file_on_a_credentials_failure(tmp_path: Path, status: int) -> None:
    """No wire: a refused credential leaves nothing to inspect, and the committed
    cassette as it was."""
    path = tmp_path / "choice.json"
    path.write_text("committed choice")
    side = tmp_path / "side" / "choice.json"
    side.parent.mkdir()
    side.write_text("refused reply")
    with pytest.raises(pytest.fail.Exception, match="credentials refused"):
        _check(path, side, ProviderError("no", status_code=status))
    assert not side.exists()
    assert path.read_text() == "committed choice"


def test_check_keeps_the_side_file_on_an_outcome_of_the_wrong_kind(tmp_path: Path) -> None:
    """No wire: a success's case answered with a failure keeps the side file as evidence,
    and the committed cassette as it was."""
    path = tmp_path / "choice.json"
    path.write_text("committed choice")
    side = tmp_path / "side" / "choice.json"
    side.parent.mkdir()
    side.write_text("a failure")
    with pytest.raises(pytest.fail.Exception, match="meant to record a success"):
        _check(path, side, ProviderError("bad", status_code=400))
    assert side.read_text() == "a failure"
    assert path.read_text() == "committed choice"


def _shared_answer(request: httpx2.Request) -> httpx2.Response:
    if request.url.path.endswith("/clef"):
        return _ok(_clef_result(_CHOICE))
    return _ok(
        _clairic_result(
            {
                "home_team": {"type": "text", "value": "Riverside Rovers", "p_null": 0.0},
                "refunds": {"type": "number", "value": 4250.5, "p_null": 0.0},
            }
        )
    )


def _created_over_the_mock(monkeypatch: pytest.MonkeyPatch) -> list[httpx2.AsyncClient]:
    """Every client a leaf creates, each over a mock transport, kept in order."""
    made: list[httpx2.AsyncClient] = []

    def over_the_mock() -> httpx2.AsyncClient:
        made.append(httpx2.AsyncClient(transport=httpx2.MockTransport(_shared_answer)))
        return made[-1]

    monkeypatch.setattr("idia.providers.http.created_client", over_the_mock)
    return made


def _both_shares() -> Request:
    return Request(
        state=TEXT,
        questions=(*CLEF_QUESTIONS["choice"], *CLAIRIC_PROVIDER.questions),
        deadline=Deadline.after(5.0),
        options=CLAIRIC_OPTIONS,
    )


def test_the_router_shares_its_created_client_on_each_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = _created_over_the_mock(monkeypatch)
    router = cloudflare(account_id=ACCOUNT, api_token=TOKEN)
    clef, clairic = (member._leaf for member in dict.fromkeys(_members(router)))
    for _ in range(2):
        response = asyncio.run(router.answer(_both_shares()))
        assert [o.provider for o in response.outcomes] == ["cloudflare-clef", "cloudflare-clairic"]
        assert all(o.error is None for o in response.outcomes)
        assert cast(Clairic, clairic).client is cast(Clef, clef).client is made[-1]
    assert len(made) == 2


def test_the_routers_aclose_closes_the_one_current_client(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _created_over_the_mock(monkeypatch)
    router = cloudflare(account_id=ACCOUNT, api_token=TOKEN)

    async def answer_and_close() -> None:
        await router.answer(_both_shares())
        await router.aclose()
        await router.aclose()

    asyncio.run(router.answer(_both_shares()))
    asyncio.run(answer_and_close())
    assert [client.is_closed for client in made] == [False, True]


def _clairic_of(router: Router) -> Clairic:
    *_, clairic = dict.fromkeys(member._leaf for member in _members(router))
    return cast(Clairic, clairic)


def _clairic_wire_request() -> WireRequest:
    return WireRequest(state=TEXT, questions=CLAIRIC_PROVIDER.questions, options=CLAIRIC_OPTIONS)


def _sharing_pair(wire: _Wire) -> tuple[Clef, Clairic, httpx2.AsyncClient]:
    """Clef and a Clairic sharing its injected client, as the router builds them."""
    client = wire.client()
    clef = Clef(account_id=ACCOUNT, api_token=TOKEN, client=client)
    clairic = Clairic(account_id=ACCOUNT, api_token=TOKEN, client=client)
    _share_client(clairic, clef)
    return clef, clairic, client


def test_a_clairic_closed_by_its_own_aclose_is_refused_while_clef_still_sends() -> None:
    wire = _Wire(_ok(_clef_result(_NOUL)))
    clef, clairic, client = _sharing_pair(wire)
    _RUNNER[-1].run(clairic.aclose())
    error = _raises(clairic, WireRequest(state=TEXT, questions=EXTRACTION.questions[:1]))
    assert (error.message, error.is_retryable) == ("the leaf is closed", False)
    assert wire.requests == []
    _call(clef, clef_request("noul"))
    assert len(wire.requests) == 1
    assert not client.is_closed


def test_after_clefs_aclose_a_clairic_sharing_its_client_is_refused_before_sending() -> None:
    wire = _Wire()
    clef, clairic, client = _sharing_pair(wire)
    _RUNNER[-1].run(clef.aclose())
    error = _raises(clairic, WireRequest(state=TEXT, questions=EXTRACTION.questions[:1]))
    assert (error.message, error.is_retryable) == ("the leaf is closed", False)
    assert wire.requests == []
    assert not client.is_closed


def test_a_clairic_sharing_a_client_the_caller_closed_is_refused_before_sending() -> None:
    wire = _Wire()
    _, clairic, client = _sharing_pair(wire)
    _RUNNER[-1].run(client.aclose())
    error = _raises(clairic, WireRequest(state=TEXT, questions=EXTRACTION.questions[:1]))
    assert (error.message, error.is_retryable) == ("the client is closed", False)
    assert wire.requests == []


def test_after_the_routers_aclose_a_clairic_call_is_refused_and_creates_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = _created_over_the_mock(monkeypatch)
    router = cloudflare(account_id=ACCOUNT, api_token=TOKEN)

    async def answer_close_and_call() -> None:
        await router.answer(_both_shares())
        await router.aclose()
        await _clairic_of(router).call(_clairic_wire_request(), Deadline.after(5.0))

    with pytest.raises(ProviderError) as raised:
        asyncio.run(answer_close_and_call())
    assert raised.value.message == "the leaf is closed"
    assert not raised.value.is_retryable
    assert len(made) == 1


def test_with_an_injected_client_a_clairic_call_on_a_second_loop_is_refused() -> None:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_shared_answer))
    router = cloudflare(account_id=ACCOUNT, api_token=TOKEN, client=client)
    response = asyncio.run(router.answer(_both_shares()))
    assert all(o.error is None for o in response.outcomes)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(_clairic_of(router).call(_clairic_wire_request(), Deadline.after(5.0)))
    assert raised.value.message == OTHER_LOOP
    assert not raised.value.is_retryable
    assert not client.is_closed


def test_each_leaf_states_its_profile_with_these_reasons_and_claims_nothing() -> None:
    answers = {"home_team": {"type": "text", "value": "Riverside Rovers", "p_null": 0.0}}
    usage = {"input_tokens": 250, "output_tokens": 0, "passes": 2}
    clairic = _read(EXTRACTION.questions[:1], answers, usage=usage)
    clef = _call(_clef(_Wire(_ok(_clef_result(_NOUL)))), clef_request("noul"))
    for response in (clairic, clef):
        assert set(response.client_versions) == {"httpx2"}
        assert response.claimed == frozenset()
        assert dict(response.versions) == {}
    assert dict(clairic.unreported) == {"versions": "not-applicable", "snapshot": "not-pinned"}
    assert dict(clef.unreported) == {
        "versions": "not-applicable",
        "snapshot": "not-pinned",
        "usage.passes": "not-reported",
    }
