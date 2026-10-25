"""The HTTP harness, run against a stub leaf over hand-made cassettes, so every exchange the
suite names is played through an ``httpx2.MockTransport``."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx2
import pytest

from idia import Deadline, Dispatcher, ProviderError, Request, RetryPolicy, WireResponse, core
from idia.conformance import (
    CheckNotApplicable,
    Error,
    Failure,
    LeafHarness,
    RefusalOnWire,
    Stall,
    Success,
    leaf_checks,
    run_checks,
)
from idia.conformance._scripted import CAPABILITIES
from idia.core import (
    Capabilities,
    JsonPart,
    JsonValue,
    Kind,
    Leaf,
    NoulAnswer,
    State,
    WireRequest,
)
from idia.providers._canon import encode, plain
from idia.providers.recorded import Recorded
from tests import _http
from tests._dishonest_http import ClosingLeaf, LeakingLeaf, ReopeningLeaf, UnclosableLeaf
from tests._http import (
    ACCOUNT_ID_SHAPE,
    BEARER_SHAPE,
    DROPPED_HEADERS,
    DROPPED_PREFIXES,
    FILTERED,
    KEY_SHAPES,
    OPENAI_KEY_SHAPE,
    SECRET_HEADERS,
    CassetteTransport,
    HttpLeafHarness,
    LeafSpy,
    NoExchange,
    RecordingTransport,
    body_key,
    cassette_path,
    features_dir,
    filter_headers,
    load_cassette,
    parse_body,
    record_cassette,
    write_cassette,
    write_run_cassettes,
)
from tests._stub_http_leaf import SECRET, URL, StubHttpLeaf

CASSETTES = Path(__file__).parent / "cassettes"
RECORD = replace(CAPABILITIES, returns_refusal=True)


def _first_key(harness: HttpLeafHarness, kind: Kind) -> str:
    return harness.wire_request(kind).questions[0].key


def _answers(body: JsonValue) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], cast(dict[str, JsonValue], body)["answers"])


class StubHarness(HttpLeafHarness):
    """The stub leaf's harness: every hook filled."""

    cassette_dir = cassette_path(CASSETTES, "stub", "stub-1", frozenset(), "x").parent
    failure_cassette = "failure"

    @property
    def capabilities(self) -> Capabilities:
        return RECORD

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return StubHttpLeaf(capabilities=replace(RECORD, enabled_features=features), client=client)

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        return replace(RECORD, enabled_features=features)

    def cassette_for(self, kind: Kind) -> str:
        return kind

    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue:
        changed = dict(cast(Mapping[str, JsonValue], body))
        answers = dict(_answers(changed))
        key = _first_key(self, kind)
        answer = dict(cast(Mapping[str, JsonValue], answers[key]))
        if how == "keys":
            answers["conformance_extra"] = answers.pop(key)
        elif how == "noul_range":
            answers[key] = {**answer, "probability": 1.5}
        else:
            pairs = [list(cast(tuple[str, float], pair)) for pair in answer["probabilities"]]  # type: ignore[union-attr]
            pairs[-1][1] = cast(float, pairs[-1][1]) + 0.25
            answers[key] = {**answer, "probabilities": cast(JsonValue, pairs)}
        changed["answers"] = answers
        return changed

    def unreadable(self, which: Literal["not_json", "schema", "no_answers"]) -> bytes:
        return {
            "not_json": b"<html>bad gateway</html>",
            "schema": b'["not", "an object"]',
            "no_answers": b'{"model": "stub-1"}',
        }[which]

    def refusal_body(self, kind: Kind) -> JsonValue | None:
        return self._replace_first(kind, {"type": "Refusal"})

    def candidate_body(self, kind: Kind) -> JsonValue | None:
        return self._replace_first(kind, {"type": "estimate", "value": 3})

    def _replace_first(self, kind: Kind, answer: JsonValue) -> JsonValue:
        body = dict(cast(Mapping[str, JsonValue], self._cassette(kind).response["body"]))
        body["answers"] = {**_answers(body), _first_key(self, kind): answer}
        return body


def _verdicts(make: Callable[[], LeafHarness]) -> dict[str, str]:
    return {v.id: v.result for v in asyncio.run(run_checks(leaf_checks(make)))}


def _with(result: str, verdicts: Mapping[str, str]) -> set[str]:
    return {id for id, held in verdicts.items() if held == result}


# ----------------------------------------------------------------------------------------
# The leaf half over the mock wire
# ----------------------------------------------------------------------------------------


def test_the_stub_passes_every_check_of_the_leaf_half() -> None:
    harness: LeafHarness = StubHarness()
    assert harness.prompt_hidden is False
    verdicts = _verdicts(StubHarness)
    assert list(verdicts) == [f"L{n}" for n in range(1, 33)]
    assert _with("fail", verdicts) == set()
    assert _with("not_applicable", verdicts) == set()


class Plain(StubHarness):
    """A wire with no refusal and no candidate shape."""

    def refusal_body(self, kind: Kind) -> JsonValue | None:
        return None

    def candidate_body(self, kind: Kind) -> JsonValue | None:
        return None


class Retrying(StubHarness):
    """A leaf that makes its own client with transport retries on."""

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        record = replace(RECORD, enabled_features=features)
        return StubHttpLeaf(capabilities=record, client=client, retries=2)


class Closing(StubHarness):
    """A leaf that closes an injected client too."""

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return ClosingLeaf(capabilities=replace(RECORD, enabled_features=features), client=client)


class Unclosable(StubHarness):
    """A leaf with no way to close the client it made."""

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return UnclosableLeaf(
            capabilities=replace(RECORD, enabled_features=features), client=client
        )


class Reopening(StubHarness):
    """A leaf that still sends after aclose."""

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return ReopeningLeaf(capabilities=replace(RECORD, enabled_features=features), client=client)


class Leaking(StubHarness):
    """A leaf that copies an unknown option into its body."""

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return LeakingLeaf(capabilities=replace(RECORD, enabled_features=features), client=client)


@pytest.mark.parametrize(
    ("make", "failing", "not_applicable"),
    [
        pytest.param(Plain, set[str](), {"L23", "L24"}, id="no-refusal-no-candidate"),
        pytest.param(Retrying, {"L9"}, set[str](), id="transport-retries"),
        pytest.param(Closing, {"L15"}, set[str](), id="closes-an-injected-client"),
        pytest.param(Unclosable, {"L15"}, {"L30"}, id="never-closes-its-own"),
        pytest.param(Leaking, {"L29"}, set[str](), id="leaks-an-option-into-its-body"),
        pytest.param(Reopening, {"L30"}, set[str](), id="sends-after-aclose"),
    ],
)
def test_the_harness_reaches_what_its_hooks_allow(
    make: Callable[[], LeafHarness], failing: set[str], not_applicable: set[str]
) -> None:
    verdicts = _verdicts(make)
    assert _with("fail", verdicts) == failing
    assert _with("not_applicable", verdicts) == not_applicable


def test_l29_fails_a_leaked_option_on_the_body_sent() -> None:
    l29 = [check for check in leaf_checks(Leaking) if check.id == "L29"]
    (verdict,) = asyncio.run(run_checks(l29))
    assert verdict.result == "fail"
    assert "does not match the request sent" in (verdict.reason or "")


# ----------------------------------------------------------------------------------------
# The mock wire itself
# ----------------------------------------------------------------------------------------


def _post(leaf: Leaf, wire_request: WireRequest) -> httpx2.Response:
    """Send what the stub sends, straight through the mock wire, to read its headers."""
    client = cast(StubHttpLeaf, leaf).client
    payload = plain({"model": "stub-1", "request": encode(wire_request)})
    return asyncio.run(client.post(URL, json=payload))


def test_a_failure_sends_retry_after_in_milliseconds_beside_a_conflicting_header() -> None:
    harness = StubHarness()
    leaf = harness.make([Failure(status=429, retry_after=1.5, request_id="r-9"), Failure()])
    response = _post(leaf, harness.wire_request("noul"))
    assert response.status_code == 429
    assert response.headers["retry-after-ms"] == "1500"
    assert response.headers["retry-after"] == "9"
    assert response.headers["x-request-id"] == "r-9"
    recorded = _post(leaf, harness.wire_request("noul"))
    assert recorded.status_code == 400
    assert recorded.headers["x-request-id"] == "stub-failure"
    assert harness.requests_sent() == 2


def test_a_success_must_match_the_body_recorded() -> None:
    harness = StubHarness()
    leaf = harness.make([Success("noul")])
    with pytest.raises(AssertionError, match="body in the match"):
        asyncio.run(leaf.call(harness.wire_request("choice"), Deadline.after(5.0)))


def test_a_success_with_no_kind_answers_whichever_request_it_matches() -> None:
    harness = StubHarness()
    leaf = harness.make([Success(), Success()])
    response = asyncio.run(leaf.call(harness.wire_request("choice"), Deadline.after(5.0)))
    assert dict(response.answers) == dict(harness.expected("choice"))
    other = WireRequest(state=harness.wire_request("noul").state, questions=())
    with pytest.raises(AssertionError, match="no cassette matches"):
        asyncio.run(leaf.call(other, Deadline.after(5.0)))


def test_a_success_with_no_kind_is_answered_by_a_probe_cassette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request packed one question per wire request matches no kind's cassette; a
    cassette written from a probe outside the leaf answers it."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    for path in StubHarness.cassette_dir.glob("*.json"):
        (tmp_path / path.name).write_text(path.read_text())
    noul = StubHarness().wire_request("noul")
    single = WireRequest(state=noul.state, questions=noul.questions[:1])
    payload = plain({"model": "stub-1", "request": encode(single)})
    probe = httpx2.Request("POST", URL, json=payload, headers={"authorization": SECRET})
    reply = _answering(probe)
    leaf = StubHttpLeaf(capabilities=RECORD)
    expected = StubHarness().expected("noul")
    answers = {key: answer for key, answer in expected.items() if key == "noul_a"}
    _http.write_cassette(
        tmp_path / "noul_a.json",
        leaf=leaf,
        request=probe,
        response=reply,
        wire_request=single,
        expected=answers,
    )
    assert (
        json.loads((tmp_path / "noul_a.json").read_text())["request"]["headers"]["authorization"]
        == FILTERED
    )

    class Probed(StubHarness):
        cassette_dir = tmp_path

    harness = Probed()
    played = harness.make([Success()])
    response = asyncio.run(played.call(single, Deadline.after(5.0)))
    assert set(response.answers) == {"noul_a"}
    monkeypatch.delenv("IDIA_LIVE")
    with pytest.raises(RuntimeError, match="IDIA_LIVE=1"):
        _http.write_cassette(
            tmp_path / "x.json",
            leaf=leaf,
            request=probe,
            response=reply,
            wire_request=single,
            expected=None,
        )


_UNSORTED: Final = {
    "model": "stub-1",
    "state": {"z": 1, "a": 2},
    "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}},
}
"""A body whose keys, at every depth, are out of sorted order."""


def _write_unsorted(path: Path) -> dict[str, Any]:
    """A cassette written from :data:`_UNSORTED` sent and received, its wire request holding
    a JSON part in that order too, read back in order."""
    request = httpx2.Request(
        "POST", URL, content=json.dumps(_UNSORTED).encode(), headers={"x-z": "1", "x-a": "2"}
    )
    response = httpx2.Response(
        200, content=json.dumps(_UNSORTED).encode(), headers={"x-z": "1", "x-a": "2"}
    )
    _http.write_cassette(
        path,
        leaf=StubHttpLeaf(capabilities=RECORD),
        request=request,
        response=response,
        wire_request=_with_json_part(StubHarness().wire_request("noul")),
        expected={"z": NoulAnswer(probability=0.5), "a": NoulAnswer(probability=0.25)},
    )
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _with_json_part(wire_request: WireRequest) -> WireRequest:
    parts = (*wire_request.state.parts, JsonPart(value={"z": 1, "a": {"y": 2, "b": 3}}))
    return replace(wire_request, state=State(parts=parts))


def _keys_sorted(value: object) -> bool:
    if isinstance(value, dict):
        held = cast(dict[str, object], value)
        return list(held) == sorted(held) and all(_keys_sorted(v) for v in held.values())
    if isinstance(value, list):
        return all(_keys_sorted(item) for item in cast(list[object], value))
    return True


def test_a_cassette_keeps_its_bodies_and_wire_request_in_order_and_sorts_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    document = _write_unsorted(tmp_path / "unsorted.json")
    for part in ("request", "response"):
        body = document[part]["body"]
        assert list(body) == ["model", "state", "questions"]
        assert list(body["questions"]) == ["z", "a"]
        assert list(body["state"]) == ["z", "a"]
        held = {key: value for key, value in document[part].items() if key != "body"}
        assert list(document[part]) == sorted(document[part])
        assert _keys_sorted(held)
    in_order = ("request", "response", "wire_request", "expected")
    rest = {key: value for key, value in document.items() if key not in in_order}
    assert list(document) == sorted(document)
    assert _keys_sorted(rest)
    part = document["wire_request"]["state"]["parts"][-1]
    assert list(document["wire_request"]) == ["type", "state", "questions"]
    assert list(part["value"]) == ["z", "a"]
    assert list(part["value"]["a"]) == ["y", "b"]
    assert list(document["expected"]) == ["z", "a"]
    assert list(document["expected"]["z"])[:2] == ["type", "probability"]


def test_a_cassette_misses_a_reordered_object_as_it_misses_a_reordered_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The match compares the body in the order parsed: an object's key order is compared
    as an array's is, and a body equal in value in another order misses alike."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    stored = {**_UNSORTED, "labels": ["b", "a"]}
    reply = httpx2.Response(200, json={"answers": {}})
    _http.write_cassette(
        tmp_path / "ordered.json",
        leaf=StubHttpLeaf(capabilities=RECORD),
        request=httpx2.Request("POST", URL, content=json.dumps(stored).encode()),
        response=reply,
        wire_request=StubHarness().wire_request("noul"),
        expected=None,
    )
    transport = CassetteTransport(tmp_path)
    reordered = {
        "labels": ["b", "a"],
        "questions": {"a": {"type": "noul"}, "z": {"type": "noul"}},
        "state": {"a": 2, "z": 1},
        "model": "stub-1",
    }
    swapped = {**stored, "labels": ["a", "b"]}
    assert reordered == stored != swapped
    with httpx2.Client(transport=transport) as client:
        for body in (reordered, swapped):
            with pytest.raises(AssertionError, match="no cassette matches"):
                client.post(URL, content=json.dumps(body).encode())
        assert client.post(URL, content=json.dumps(stored).encode()).status_code == 200
    assert transport.served == [tmp_path / "ordered.json"]
    assert [body for _, _, body in transport.misses] == [reordered, swapped]


def test_a_failed_write_leaves_an_earlier_cassette_unchanged_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    path = tmp_path / "noul.json"
    earlier = b'{"earlier": true}\n'
    path.write_bytes(earlier)

    def refused(source: object, target: object) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(_http.os, "replace", refused)
    with pytest.raises(OSError, match="replace refused"):
        _write_unsorted(path)
    assert path.read_bytes() == earlier
    assert list(tmp_path.iterdir()) == [path]
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(OSError, match="replace refused"):
        _write_unsorted(fresh / "noul.json")
    assert not (fresh / "noul.json").exists()
    assert list(fresh.iterdir()) == []
    assert list(tmp_path.glob(".noul.json.*")) == list(fresh.glob(".noul.json.*")) == []


def test_running_out_of_exchanges_fails_the_test() -> None:
    harness = StubHarness()
    leaf = harness.make([])
    with pytest.raises(AssertionError, match="exchanges ran out at request 1"):
        asyncio.run(leaf.call(harness.wire_request("noul"), Deadline.after(5.0)))


@pytest.mark.parametrize(
    ("exchange", "match"),
    [
        pytest.param(Error("connection"), "connection refused", id="connection"),
        pytest.param(Error("timeout"), "read timed out", id="timeout"),
        pytest.param(Stall(), "stalled", id="stall"),
    ],
)
def test_transport_failures_reach_the_leaf_as_its_errors(
    exchange: Error | Stall, match: str
) -> None:
    harness = StubHarness()
    leaf = harness.make([exchange])
    with pytest.raises(ProviderError, match=match) as raised:
        asyncio.run(leaf.call(harness.wire_request("noul"), Deadline.after(0.2)))
    assert raised.value.is_retryable


def test_a_refusal_the_wire_cannot_make_is_not_applicable_when_made() -> None:
    with pytest.raises(CheckNotApplicable, match="no refusal"):
        Plain().make([RefusalOnWire("noul")])


def test_the_harness_refuses_what_it_cannot_read() -> None:
    class Failed(StubHarness):
        def cassette_for(self, kind: Kind) -> str:
            return "failure"

    with pytest.raises(ValueError, match="holds no answers"):
        Failed().expected("noul")

    class NoClient(StubHarness):
        def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
            return Recorded(super().build(client, features), Path("unused.json"), "new")

    with pytest.raises(CheckNotApplicable, match="exposes no httpx2 client"):
        NoClient().sdk_retries()

    class MockOwn(StubHarness):
        def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
            mock = httpx2.AsyncClient(transport=httpx2.MockTransport(_answering))
            return super().build(client or mock, features)

    with pytest.raises(CheckNotApplicable, match="runs on a MockTransport"):
        MockOwn().sdk_retries()


# ----------------------------------------------------------------------------------------
# Cassettes and the secrets filter
# ----------------------------------------------------------------------------------------


def test_secrets_are_filtered_and_the_status_headers_kept() -> None:
    headers = {
        "Authorization": "Bearer x",
        "X-Api-Key": "k",
        "api-key": "k",
        "Cookie": "a=b",
        "Set-Cookie": "c=d",
        "X-Request-Id": "r-1",
        "Retry-After-Ms": "1500",
        "Retry-After": "2",
        "CF-Ray": "8f00-LHR",
        "X-RateLimit-Limit-Tokens": "2000000",
        "X-Stainless-OS": "MacOS",
    }
    kept = filter_headers(headers)
    for dropped in ("cf-ray", "x-ratelimit-limit-tokens", "x-stainless-os"):
        assert dropped not in kept
    assert kept == {
        "authorization": FILTERED,
        "x-api-key": FILTERED,
        "api-key": FILTERED,
        "cookie": FILTERED,
        "set-cookie": FILTERED,
        "x-request-id": "r-1",
        "retry-after-ms": "1500",
        "retry-after": "2",
    }


def test_a_cassette_is_kept_under_its_leaf_model_and_features() -> None:
    assert features_dir(frozenset()) == "base"
    assert features_dir(frozenset({"b", "a"})) == "a+b"
    assert cassette_path(Path("c"), "sys1", "m", frozenset({"x"}), "noul") == Path(
        "c/sys1/m/x/noul.json"
    )


def test_a_cassette_in_another_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    for format in (1, 3):
        path.write_text(json.dumps({"format": format}))
        with pytest.raises(ValueError, match=f"format {format}; this reads 2"):
            load_cassette(path)


def _answering(request: httpx2.Request) -> httpx2.Response:
    """The mock wire the committed cassettes were recorded from."""
    noul = load_cassette(StubHarness.cassette_dir / "noul.json")
    headers = {**noul.response["headers"], "set-cookie": "session=s3cret"}
    return httpx2.Response(200, headers=headers, json=noul.response["body"])


def test_record_cassette_refuses_outside_a_recording_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDIA_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="IDIA_LIVE=1"):
        asyncio.run(
            record_cassette(
                Path("never.json"),
                lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
                StubHarness().wire_request("noul"),
                Deadline.after(5.0),
            )
        )


def test_record_cassette_writes_the_cassette_with_secrets_filtered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    path = tmp_path / "noul.json"
    result = asyncio.run(
        record_cassette(
            path,
            lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
            StubHarness().wire_request("noul"),
            Deadline.after(5.0),
            transport=httpx2.MockTransport(_answering),
        )
    )
    assert isinstance(result, WireResponse)
    written = json.loads(path.read_text())
    committed = json.loads((StubHarness.cassette_dir / "noul.json").read_text())
    assert written["request"]["headers"]["authorization"] == FILTERED
    assert written["response"]["headers"]["set-cookie"] == FILTERED
    del written["request"]["headers"], committed["request"]["headers"]
    assert written == committed
    assert SECRET.split()[1] not in path.read_text()
    assert "s3cret" not in path.read_text()


def test_record_cassette_writes_a_failure_and_refuses_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")

    def failing(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": "bad"})

    path = tmp_path / "failure.json"
    result = asyncio.run(
        record_cassette(
            path,
            lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
            StubHarness().wire_request("noul"),
            Deadline.after(5.0),
            transport=httpx2.MockTransport(failing),
        )
    )
    assert isinstance(result, ProviderError)
    assert json.loads(path.read_text())["expected"] is None

    class Twice(StubHttpLeaf):
        async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
            await self.client.get("https://stub.invalid/ping")
            return await super().call(wire_request, deadline)

    with pytest.raises(AssertionError, match="made 2 requests"):
        asyncio.run(
            record_cassette(
                tmp_path / "twice.json",
                lambda client: Twice(capabilities=RECORD, client=client),
                StubHarness().wire_request("noul"),
                Deadline.after(5.0),
                transport=httpx2.MockTransport(_answering),
            )
        )


def test_record_cassette_raises_no_exchange_when_the_leaf_sent_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leaf that timed out or lost the connection raises ``ProviderError`` with no
    exchange recorded: a distinct error, not the count's ``AssertionError``, and no file."""
    monkeypatch.setenv("IDIA_LIVE", "1")

    class Unsent(StubHttpLeaf):
        async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
            raise ProviderError("the transport timed out: stalled", is_retryable=True)

    path = tmp_path / "unsent.json"
    with pytest.raises(NoExchange, match="made 0 requests") as raised:
        asyncio.run(
            record_cassette(
                path,
                lambda client: Unsent(capabilities=RECORD, client=client),
                StubHarness().wire_request("noul"),
                Deadline.after(5.0),
                transport=httpx2.MockTransport(_answering),
            )
        )
    assert raised.value.count == 0
    assert isinstance(raised.value, AssertionError)
    assert not path.exists()


def test_record_cassette_leaves_the_committed_cassette_when_the_check_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that raises leaves the cassette at its path byte for byte, writes nothing
    beside it, and keeps the side file it judged, outside the cassette tree."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    path = tmp_path / "noul.json"
    path.write_bytes((StubHarness.cassette_dir / "noul.json").read_bytes())
    before = path.read_bytes()

    def failing(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": "bad"})

    def refuse(side: Path, result: WireResponse | ProviderError) -> None:
        pytest.fail("refused")

    with pytest.raises(pytest.fail.Exception, match="refused") as raised:
        asyncio.run(
            record_cassette(
                path,
                lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
                StubHarness().wire_request("noul"),
                Deadline.after(5.0),
                transport=httpx2.MockTransport(failing),
                check=refuse,
            )
        )
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    [note] = raised.value.__notes__
    match = re.fullmatch(r"kept at (.+); (.+) untouched", note)
    assert match is not None
    assert match.group(2) == str(path)
    side = Path(match.group(1))
    try:
        assert side.is_file()
        assert side.name == path.name
        assert not side.is_relative_to(tmp_path)
        kept = json.loads(side.read_text())
        assert kept["response"]["status"] == 400
        assert kept["expected"] is None
    finally:
        side.unlink(missing_ok=True)
        side.parent.rmdir()


def test_record_cassette_names_only_the_path_when_the_check_removed_the_side_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that removes the side file and raises: the note names the cassette's path
    alone, the empty side directory is removed, and the cassette is left byte for byte."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    path = tmp_path / "noul.json"
    path.write_bytes((StubHarness.cassette_dir / "noul.json").read_bytes())
    before = path.read_bytes()
    sides: list[Path] = []

    def remove_and_refuse(side: Path, result: WireResponse | ProviderError) -> None:
        sides.append(side)
        side.unlink()
        pytest.fail("refused")

    with pytest.raises(pytest.fail.Exception, match="refused") as raised:
        asyncio.run(
            record_cassette(
                path,
                lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
                StubHarness().wire_request("noul"),
                Deadline.after(5.0),
                transport=httpx2.MockTransport(_answering),
                check=remove_and_refuse,
            )
        )
    assert raised.value.__notes__ == [f"{path} untouched"]
    [side] = sides
    assert not side.parent.exists()
    assert path.read_bytes() == before


def test_record_cassette_writes_at_its_path_what_the_check_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that returns sees the side file once; the cassette at its path is then the
    side file's text, and the side directory is gone."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    path = tmp_path / "noul.json"
    seen: list[tuple[Path, str]] = []

    def accept(side: Path, result: WireResponse | ProviderError) -> None:
        assert isinstance(result, WireResponse)
        seen.append((side, side.read_text()))

    asyncio.run(
        record_cassette(
            path,
            lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
            StubHarness().wire_request("noul"),
            Deadline.after(5.0),
            transport=httpx2.MockTransport(_answering),
            check=accept,
        )
    )
    [(side, text)] = seen
    assert path.read_text() == text
    assert not side.parent.exists()


def test_a_written_cassette_names_the_versions_that_shaped_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    recorder, spy = _recorded_run(httpx2.MockTransport(_answering))
    [(request, response, content)] = recorder.exchanges
    [(wire_request, answers)] = spy.calls
    path = tmp_path / "versions.json"
    write_cassette(
        path,
        leaf=spy.inner,
        request=request,
        response=response,
        content=content,
        wire_request=wire_request,
        expected=answers,
        recorded_with={"idia": "0.0.1", "httpx2": "2.16.0"},
    )
    document = json.loads(path.read_text())
    assert document["recorded_with"] == {"idia": "0.0.1", "httpx2": "2.16.0"}
    assert list(document["recorded_with"]) == ["httpx2", "idia"]
    assert list(document) == sorted(document)
    assert document["format"] == 2
    stored = load_cassette(path)
    assert stored.wire_request == wire_request
    assert stored.expected == answers
    written = write_run_cassettes(
        recorder.exchanges,
        spy.calls,
        leaf=spy.inner,
        path_for=lambda index, _: tmp_path / f"run-{index}.json",
        redact=None,
        secrets=[],
        recorded_with={"idia": "0.0.1"},
    )
    assert json.loads(written[0].read_text())["recorded_with"] == {"idia": "0.0.1"}
    assert load_cassette(written[0]).expected == answers
    write_cassette(
        path,
        leaf=spy.inner,
        request=request,
        response=response,
        content=content,
        wire_request=wire_request,
        expected=answers,
    )
    assert "recorded_with" not in json.loads(path.read_text())


def test_the_recording_transport_runs_on_a_real_transport_by_default() -> None:
    recorder = RecordingTransport()
    assert isinstance(recorder._inner, httpx2.AsyncHTTPTransport)
    asyncio.run(recorder.aclose())


def test_no_header_reaches_a_recorded_leafs_log(tmp_path: Path) -> None:
    """The log holds only wire responses and provider errors, which hold no header, so the
    cassette filter is the only secrets filter needed."""
    harness = StubHarness()
    log = tmp_path / "log.json"
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_answering))
    leaf = Recorded(StubHttpLeaf(capabilities=RECORD, client=client), log, "once")
    retry = RetryPolicy.without_delay()
    request = Request(
        state=harness.wire_request("noul").state,
        questions=harness.wire_request("noul").questions,
        deadline=Deadline.after(5.0),
    )
    asyncio.run(Dispatcher(leaf, retry=retry).answer(request))
    text = log.read_text()
    assert "s3cret" not in text
    assert SECRET.split()[1] not in text

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            held = cast(dict[str, object], value)
            return set(held) | {k for item in held.values() for k in keys(item)}
        if isinstance(value, list):
            return {k for item in cast(list[object], value) for k in keys(item)}
        return set()

    found = {key.lower() for key in keys(json.loads(text))}
    assert not found & (_http.SECRET_HEADERS | {"headers", "x-request-id"})


def test_no_committed_cassette_holds_a_secret_header_value() -> None:
    assert {"openai-organization", "openai-project"} <= SECRET_HEADERS
    held: list[str] = []
    for path in sorted(CASSETTES.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for side in ("request", "response"):
            headers = document.get(side, {}).get("headers", {})
            held += [
                f"{path.relative_to(CASSETTES)} {side} {name}"
                for name, value in headers.items()
                if name.lower() in SECRET_HEADERS and value != FILTERED
            ]
    assert held == []


def test_no_committed_cassette_holds_a_dropped_header() -> None:
    """No committed cassette stores a header :func:`filter_headers` leaves out, on either
    side, whatever the case of its name."""
    held: list[str] = []
    for path in sorted(CASSETTES.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for side in ("request", "response"):
            headers = document.get(side, {}).get("headers", {})
            held += [
                f"{path.relative_to(CASSETTES)} {side} {name}"
                for name in headers
                if name.lower() in DROPPED_HEADERS or name.lower().startswith(DROPPED_PREFIXES)
            ]
    assert held == []


def test_no_committed_cassette_holds_a_key_or_bearer_shaped_value() -> None:
    """The header filter covers headers only; no OpenAI key and no bearer credential
    appears anywhere in a committed cassette."""
    held = [
        f"{path.relative_to(CASSETTES)} {shape.pattern}"
        for path in sorted(CASSETTES.rglob("*.json"))
        for shape in KEY_SHAPES
        if shape.search(path.read_text(encoding="utf-8"))
    ]
    assert held == []


def test_the_cassette_filter_checks_the_key_shapes_core_filters_from_error_text() -> None:
    assert OPENAI_KEY_SHAPE is core._OPENAI_KEY_SHAPE
    assert BEARER_SHAPE is core._BEARER_SHAPE
    assert KEY_SHAPES is core._KEY_SHAPES
    assert FILTERED is core._FILTERED


@pytest.mark.parametrize(
    ("text", "shape"),
    [
        pytest.param("key sk-abcdefghijklmnopqrstuvwx1234", OPENAI_KEY_SHAPE, id="openai-key"),
        pytest.param("Authorization: Bearer abcdefghijklmnop0123", BEARER_SHAPE, id="bearer"),
        pytest.param("/accounts/0123456789abcdef0123456789abcdef/ai", ACCOUNT_ID_SHAPE, id="acct"),
    ],
)
def test_each_secret_shape_finds_its_secret(text: str, shape: re.Pattern[str]) -> None:
    assert shape.search(text)


@pytest.mark.parametrize(
    ("text", "shape"),
    [
        pytest.param("task-sk-abcdefghijklmnopqrstuvwx", OPENAI_KEY_SHAPE, id="inside-a-token"),
        pytest.param("sk-short", OPENAI_KEY_SHAPE, id="short-key"),
        pytest.param('"authorization": "<filtered>"', BEARER_SHAPE, id="filtered"),
        pytest.param("0123456789abcdef0123456789abcdef0", ACCOUNT_ID_SHAPE, id="33-hex"),
    ],
)
def test_a_secret_shape_passes_what_is_not_its_secret(text: str, shape: re.Pattern[str]) -> None:
    assert shape.search(text) is None


def test_record_cassette_redacts_a_secret_wherever_it_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    secret = "stub.invalid"
    path = tmp_path / "noul.json"
    asyncio.run(
        record_cassette(
            path,
            lambda client: StubHttpLeaf(capabilities=RECORD, client=client),
            StubHarness().wire_request("noul"),
            Deadline.after(5.0),
            transport=httpx2.MockTransport(_answering),
            redact={secret: "placeholder.invalid", "": "never"},
        )
    )
    text = path.read_text()
    assert secret not in text
    assert "placeholder.invalid" in json.loads(text)["request"]["url"]


def test_every_cloudflare_cassette_holds_the_placeholder_account_and_never_the_real_one() -> None:
    """The Cloudflare URL carries the account id, which the header filter cannot reach, so
    each cassette was written with it redacted; with the real id in the environment, no
    committed cassette holds it anywhere."""
    real = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or None
    for path in sorted(CASSETTES.glob("cloudflare-*/**/*.json")):
        text = path.read_text(encoding="utf-8")
        url = json.loads(text)["request"]["url"]
        assert "/accounts/conformance-account/" in url, path.relative_to(CASSETTES)
        assert real is None or real not in text, path.relative_to(CASSETTES)
    for path in sorted(CASSETTES.rglob("*.json")):
        assert real is None or real not in path.read_text(encoding="utf-8")


def test_no_cloudflare_cassette_holds_an_account_id_shaped_value() -> None:
    """Runs without the environment: a Cloudflare account id is 32 hex characters, and no
    such run appears in any Cloudflare cassette, the placeholder standing in its place."""
    paths = sorted(CASSETTES.glob("cloudflare-*/**/*.json"))
    assert paths
    for path in paths:
        found = ACCOUNT_ID_SHAPE.findall(path.read_text(encoding="utf-8"))
        assert found == [], path.relative_to(CASSETTES)


# ----------------------------------------------------------------------------------------
# A run of many wire requests: the spy, the run's cassettes and their replay
# ----------------------------------------------------------------------------------------


def _over(transport: httpx2.AsyncBaseTransport) -> StubHttpLeaf:
    return StubHttpLeaf(capabilities=RECORD, client=httpx2.AsyncClient(transport=transport))


def _copied(directory: Path, *names: str) -> set[Path]:
    """The stub's cassettes named, copied into ``directory``; the stub's own directory holds
    a failure recorded for the noul request, which the cassette transport refuses beside
    the noul success."""
    for name in names:
        source = StubHarness.cassette_dir / f"{name}.json"
        (directory / source.name).write_text(source.read_text())
    return {directory / f"{name}.json" for name in names}


def test_the_cassette_transport_serves_by_method_url_and_body(tmp_path: Path) -> None:
    harness = StubHarness()
    every = _copied(tmp_path, "noul", "choice", "score")
    transport = CassetteTransport(tmp_path)
    assert transport.unused() == every
    leaf = _over(transport)
    response = asyncio.run(leaf.call(harness.wire_request("noul"), Deadline.after(5.0)))
    assert dict(response.answers) == dict(harness.expected("noul"))
    assert transport.unused() == every - {tmp_path / "noul.json"}
    assert transport.misses == []


def test_the_cassette_transport_keeps_each_path_served_in_order_repeats_shown(
    tmp_path: Path,
) -> None:
    harness = StubHarness()
    _copied(tmp_path, "noul", "choice")
    transport = CassetteTransport(tmp_path)
    leaf = _over(transport)
    for kind in ("noul", "choice", "noul"):
        asyncio.run(leaf.call(harness.wire_request(cast(Kind, kind)), Deadline.after(5.0)))
    noul, choice = tmp_path / "noul.json", tmp_path / "choice.json"
    assert transport.served == [noul, choice, noul]
    assert transport.unused() == set()


def test_the_cassette_transport_replays_a_failure_as_recorded(tmp_path: Path) -> None:
    _copied(tmp_path, "failure", "choice")
    failure = load_cassette(tmp_path / "failure.json")
    transport = CassetteTransport(tmp_path)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(_over(transport).call(failure.wire_request, Deadline.after(5.0)))
    assert raised.value.status_code == failure.response["status"]
    assert transport.unused() == {tmp_path / "choice.json"}


def test_the_cassette_transport_keeps_a_miss_then_fails(tmp_path: Path) -> None:
    every = _copied(tmp_path, "noul")
    noul = StubHarness().wire_request("noul")
    other = WireRequest(state=noul.state, questions=())
    transport = CassetteTransport(tmp_path)
    with pytest.raises(AssertionError, match="no cassette matches"):
        asyncio.run(_over(transport).call(other, Deadline.after(5.0)))
    payload = plain({"model": "stub-1", "request": encode(other)})
    assert transport.misses == [("POST", URL, parse_body(json.dumps(payload).encode()))]
    assert transport.unused() == every


def test_the_cassette_transport_refuses_two_cassettes_for_one_request() -> None:
    """The stub's failure was recorded from the noul request, so the two cannot be served
    from one directory."""
    with pytest.raises(ValueError, match=r"failure\.json and noul\.json share a method"):
        CassetteTransport(StubHarness.cassette_dir)


def test_the_cassette_transport_reads_its_directory_only(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "noul.json").write_text(
        (StubHarness.cassette_dir / "noul.json").read_text()
    )
    assert CassetteTransport(tmp_path).unused() == set()


def test_parse_body_reads_json_where_it_parses_and_text_otherwise() -> None:
    assert parse_body(b'{"a": [1, 2]}') == {"a": [1, 2]}
    assert parse_body(b"<html>bad gateway</html>") == "<html>bad gateway</html>"


def test_the_leaf_spy_passes_the_leaf_through_and_keeps_each_call() -> None:
    harness = StubHarness()
    inner = _over(httpx2.MockTransport(_answering))
    spy = LeafSpy(inner)
    assert (spy.name, spy.model, spy.capabilities) == (inner.name, inner.model, inner.capabilities)
    assert spy.estimator is inner.estimator
    assert spy.client is inner.client
    noul = harness.wire_request("noul")
    response = asyncio.run(
        Dispatcher(spy, retry=RetryPolicy.without_delay()).answer(
            Request(state=noul.state, questions=noul.questions, deadline=Deadline.after(5.0))
        )
    )
    assert set(response.answers) == {question.key for question in noul.questions}
    assert spy.calls == [(noul, harness.expected("noul"))]
    assert spy.errors == []


def test_the_leaf_spy_keeps_a_call_that_raised() -> None:
    def failing(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": "bad"})

    spy = LeafSpy(_over(httpx2.MockTransport(failing)))
    noul = StubHarness().wire_request("noul")
    with pytest.raises(ProviderError) as raised:
        asyncio.run(spy.call(noul, Deadline.after(5.0)))
    assert spy.calls == [(noul, None)]
    assert spy.errors == [raised.value]


def test_the_leaf_spy_closes_the_leaf_where_it_can() -> None:
    owning = StubHttpLeaf(capabilities=RECORD)
    asyncio.run(LeafSpy(owning).aclose())
    assert owning.client.is_closed

    class Bare:
        name = "bare"
        model = "bare-1"
        capabilities = RECORD
        estimator = StubHttpLeaf(capabilities=RECORD).estimator

        async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
            raise AssertionError("never called")

    bare = LeafSpy(Bare())
    assert getattr(bare, "client", None) is None
    asyncio.run(bare.aclose())


def _recorded_run(
    transport: httpx2.AsyncBaseTransport,
) -> tuple[RecordingTransport, LeafSpy]:
    recorder = RecordingTransport(transport)
    spy = LeafSpy(_over(recorder))
    noul = StubHarness().wire_request("noul")
    asyncio.run(spy.call(noul, Deadline.after(5.0)))
    return recorder, spy


def test_write_run_cassettes_writes_each_exchange_with_its_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    recorder, spy = _recorded_run(httpx2.MockTransport(_answering))
    named: list[tuple[int, WireRequest]] = []

    def path_for(index: int, wire_request: WireRequest) -> Path:
        named.append((index, wire_request))
        return tmp_path / f"run-{index}.json"

    written = write_run_cassettes(
        recorder.exchanges,
        spy.calls,
        leaf=spy.inner,
        path_for=path_for,
        redact={"stub.invalid": "placeholder.invalid"},
        secrets=["stub-secret"],
    )
    assert written == [tmp_path / "run-0.json"]
    assert named == [(0, spy.calls[0][0])]
    stored = load_cassette(written[0])
    assert stored.request["url"] == URL.replace("stub.invalid", "placeholder.invalid")
    assert stored.request["headers"]["authorization"] == FILTERED
    assert stored.expected == spy.calls[0][1]


def test_write_run_cassettes_refuses_counts_that_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    recorder, spy = _recorded_run(httpx2.MockTransport(_answering))
    with pytest.raises(AssertionError, match="1 wire exchanges for 0 leaf calls"):
        write_run_cassettes(
            recorder.exchanges,
            [],
            leaf=spy.inner,
            path_for=lambda index, _: tmp_path / f"{index}.json",
            redact=None,
            secrets=[],
        )
    assert list(tmp_path.iterdir()) == []


def test_write_run_cassettes_removes_a_cassette_holding_a_secret_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact-value check: a secret with no shape of its own, here the stub's host,
    left in the URL by an empty ``redact``."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    recorder, spy = _recorded_run(httpx2.MockTransport(_answering))
    with pytest.raises(AssertionError, match="held a secret given; every file written removed"):
        write_run_cassettes(
            recorder.exchanges,
            spy.calls,
            leaf=spy.inner,
            path_for=lambda index, _: tmp_path / f"{index}.json",
            redact={},
            secrets=["", "stub.invalid"],
        )
    assert list(tmp_path.iterdir()) == []


def test_write_run_cassettes_removes_a_cassette_holding_a_secret_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bearer token in a header the filter does not name is caught by its shape."""
    monkeypatch.setenv("IDIA_LIVE", "1")

    def leaking(request: httpx2.Request) -> httpx2.Response:
        reply = _answering(request)
        headers = {**reply.headers, "x-echo": "Bearer abcdefghijklmnop0123"}
        return httpx2.Response(200, headers=headers, content=reply.content)

    recorder, spy = _recorded_run(httpx2.MockTransport(leaking))
    with pytest.raises(AssertionError, match="held a value shaped"):
        write_run_cassettes(
            recorder.exchanges,
            spy.calls,
            leaf=spy.inner,
            path_for=lambda index, _: tmp_path / f"{index}.json",
            redact=None,
            secrets=[],
        )
    assert list(tmp_path.iterdir()) == []


def test_write_run_cassettes_removes_every_file_it_wrote_when_a_later_one_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first exchange is clean and written; the second leaks a bearer token: both
    files go, and a file the call never wrote stays."""
    monkeypatch.setenv("IDIA_LIVE", "1")
    sent: list[int] = []

    def second_leaks(request: httpx2.Request) -> httpx2.Response:
        sent.append(1)
        reply = _answering(request)
        if len(sent) == 1:
            return reply
        headers = {**reply.headers, "x-echo": "Bearer abcdefghijklmnop0123"}
        return httpx2.Response(200, headers=headers, content=reply.content)

    recorder = RecordingTransport(httpx2.MockTransport(second_leaks))
    spy = LeafSpy(_over(recorder))
    noul = StubHarness().wire_request("noul")
    for _ in range(2):
        asyncio.run(spy.call(noul, Deadline.after(5.0)))
    kept = tmp_path / "kept.json"
    kept.write_text("{}")
    with pytest.raises(AssertionError, match=r"1\.json held a value shaped"):
        write_run_cassettes(
            recorder.exchanges,
            spy.calls,
            leaf=spy.inner,
            path_for=lambda index, _: tmp_path / f"{index}.json",
            redact=None,
            secrets=[],
        )
    assert list(tmp_path.iterdir()) == [kept]


def _harness_classes() -> list[type[HttpLeafHarness]]:
    """Every HTTP harness defined at module level in this package: each ``_*_harness.py``
    module is imported for the subclasses it defines."""
    for path in sorted(Path(__file__).parent.glob("_*_harness.py")):
        importlib.import_module(f"tests.{path.stem}")
    found: list[type[HttpLeafHarness]] = []
    pending: list[type[HttpLeafHarness]] = [HttpLeafHarness]
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if "<locals>" not in cls.__qualname__ and "failure_cassette" in dir(cls):
            found.append(cls)
    return found


def test_no_recorded_failure_is_a_credential_error_or_retryable() -> None:
    """P10 needs a partial failure, which a 401 or 403 cannot give: it leaves the rest of
    the plan unsent. And ``Failure()`` is a failure that is not retried. So no harness
    records its failure as a credential error or a retryable status."""
    classes = _harness_classes()
    concrete = [cls for cls in classes if hasattr(cls, "cassette_dir")]
    assert {cls.__name__ for cls in concrete} >= {
        "StubHarness",
        "SystemOneHarness",
        "WithImagesHarness",
        "DecisionsHarness",
        "ChatHarness",
        "ClefHarness",
        "ClefFlashHarness",
        "ClairicHarness",
    }
    for cls in concrete:
        cassette = load_cassette(cls.cassette_dir / f"{cls.failure_cassette}.json")
        status = cassette.response["status"]
        assert status not in (401, 403), f"{cls.__qualname__}'s recorded failure is a {status}"
        retryable = status in (408, 429) or status >= 500
        assert not retryable, f"{cls.__qualname__}'s recorded failure is a retryable {status}"


_LEAF_HARNESSES: Final = (
    "StubHarness",
    "SystemOneHarness",
    "WithImagesHarness",
    "DecisionsHarness",
    "ChatHarness",
    "ClefHarness",
    "ClefFlashHarness",
    "ClairicHarness",
)
"""The harness of each configuration whose leaf recorded its cassettes."""

_READ_BY_NAME: Final = frozenset(
    {
        # Probes sent outside the leaf: question order, an option or a key it never sends.
        "openai-decisions/gpt-6-luna/base/order-4/order-e-1.json",
        "openai-decisions/gpt-6-luna/base/order-4/order-e-2.json",
        # Asks for 21 logprobs, which the leaf never sends; written by write_cassette.
        "openai-chat/gpt-4.1/base/failure.json",
        "cloudflare-clef/clef/base/question-id-probe.json",
        # Recorded through the leaf under another model name, and read by name for it.
        "openai-decisions/gpt-6-luna/base/unknown-model.json",
        # Recorded before bodies kept their order: their bodies hold sorted keys.
        "openai-chat/gpt-4.1/base/score-absent-level.json",
        "cloudflare-clairic/clairic/base/candidate-estimate.json",
        "cloudflare-clairic/clairic/base/field-name-probe.json",
        "cloudflare-clairic/clairic/base/list-items-object.json",
        "cloudflare-clairic/clairic/base/past-context/truncated.json",
    }
)
"""Cassettes whose body the leaf does not send as stored, each read by name: a probe sent
outside the leaf, or one recorded before bodies kept their order."""


def _cassettes(*, read_by_name: bool) -> list[Any]:
    classes = {cls.__name__: cls for cls in _harness_classes()}
    return [
        pytest.param(classes[name], path, id=path.relative_to(CASSETTES).as_posix())
        for name in _LEAF_HARNESSES
        for path in sorted(classes[name].cassette_dir.rglob("*.json"))
        if (path.relative_to(CASSETTES).as_posix() in _READ_BY_NAME) == read_by_name
    ]


def _sent_and_stored(harness_class: type[HttpLeafHarness], path: Path) -> tuple[str, str]:
    """The body the leaf sends for a cassette's decoded wire request, and the body stored,
    each as the match compares it."""
    cassette = load_cassette(path)
    sent: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        response = cassette.response
        return _http._response(response["status"], response["headers"], response["body"])

    harness = harness_class()  # type: ignore[abstract]
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    leaf = harness.build(client, harness.capabilities.enabled_features)
    with contextlib.suppress(ProviderError):
        asyncio.run(leaf.call(cassette.wire_request, Deadline.after(30.0)))
    assert sent, "the leaf sent nothing"
    return body_key(parse_body(sent[0])), body_key(cassette.request["body"])


@pytest.mark.parametrize(("harness_class", "path"), _cassettes(read_by_name=False))
def test_a_cassettes_wire_request_gives_its_stored_body(
    harness_class: type[HttpLeafHarness], path: Path
) -> None:
    """Sending the decoded wire request of a cassette its leaf recorded, through that leaf,
    gives the body stored, every key in the order stored."""
    sent, stored = _sent_and_stored(harness_class, path)
    assert sent == stored


@pytest.mark.parametrize(("harness_class", "path"), _cassettes(read_by_name=True))
def test_each_cassette_read_by_name_does_not_give_its_stored_body(
    harness_class: type[HttpLeafHarness], path: Path
) -> None:
    """Each exemption is still needed: a cassette read by name whose body the leaf now sends
    as stored belongs with the rest."""
    sent, stored = _sent_and_stored(harness_class, path)
    assert sent != stored


def test_every_cassette_read_by_name_exists() -> None:
    assert len(_cassettes(read_by_name=True)) == len(_READ_BY_NAME)


class SortsTheBody(StubHarness):
    """A wire whose leaf sends a caller's JSON with every object's keys sorted: the body
    the harness reports is the one sent, sorted."""

    def body_sent(self) -> bytes:
        return json.dumps(json.loads(super().body_sent()), sort_keys=True).encode()


def test_a_body_sent_with_sorted_keys_fails_exactly_l32() -> None:
    verdicts = {v.id: v for v in asyncio.run(run_checks(leaf_checks(SortsTheBody)))}
    assert {id for id, v in verdicts.items() if v.result == "fail"} == {"L32"}
    assert "the body sent holds the structured instruction's keys" in (verdicts["L32"].reason or "")


def test_l32_checks_the_body_and_the_prompt_over_the_mock_wire() -> None:
    [l32] = [check for check in leaf_checks(StubHarness) if check.id == "L32"]
    assert asyncio.run(l32.run()) is None


class _NoBodySent:
    """The stub's harness with no ``body_sent``, as a harness written before it was."""

    def __init__(self) -> None:
        self._inner = StubHarness()

    def __getattr__(self, name: str) -> Any:
        if name == "body_sent":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_a_harness_without_body_sent_runs_l32_with_a_note() -> None:
    harness = cast(LeafHarness, _NoBodySent())
    assert not hasattr(harness, "body_sent")
    [l32] = [check for check in leaf_checks(lambda: harness) if check.id == "L32"]
    assert asyncio.run(l32.run()) == "the body was not checked: the harness reads no body"
