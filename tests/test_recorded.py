"""The recorded leaf: it records a leaf's calls once and replays them with no call."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import idia
from idia import (
    Deadline,
    Dispatcher,
    JsonPart,
    Leaf,
    NoulAnswer,
    Provider,
    ProviderError,
    Question,
    RequestFailed,
    RetryPolicy,
    State,
    Usage,
    WireRequest,
    WireResponse,
)
from idia.conformance import Check, CheckNotApplicable, leaf_checks, run_checks
from idia.conformance._scripted import ScriptedHarness
from idia.core import Capabilities, TokenEstimator
from idia.providers import _REGISTRY
from idia.providers._canon import CharacterEstimator, canonical_hash, canonical_request, plain
from idia.providers.recorded import Recorded, ReplayMiss
from idia.providers.scripted import Item, Scripted, ScriptFunction
from tests._support import capabilities, noul, text_state

DEADLINE = Deadline.after(60.0)
SCRIPTED_NOT_APPLICABLE = frozenset({"L9", "L12", "L13", "L15", "L24", "L30"})


def _wire(*keys: str) -> WireRequest:
    return WireRequest(state=text_state(), questions=tuple(noul(key) for key in keys))


def _scripted(*items: Item, name: str = "scripted", model: str = "scripted-1") -> Scripted:
    return Scripted(capabilities=capabilities(), script=list(items), name=name, model=model)


def _call(leaf: Leaf, wire_request: WireRequest) -> WireResponse:
    return asyncio.run(leaf.call(wire_request, DEADLINE))


def _answer(p: float) -> dict[str, NoulAnswer]:
    return {"a": NoulAnswer(probability=p)}


def _read(log: Path) -> dict[str, object]:
    return json.loads(log.read_text())


def _entries(log: Path) -> list[dict[str, Any]]:
    """The entries of a log's one key."""
    [call] = cast(dict[str, dict[str, Any]], _read(log)["calls"]).values()
    return call["entries"]


# ----------------------------------------------------------------------------------------
# Recording and replay
# ----------------------------------------------------------------------------------------


def test_once_records_then_none_replays_without_calling_the_leaf(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "noul.json"
    recorded = _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    inner = _scripted()
    replay = Recorded(inner, log, "none")
    replayed = _call(replay, _wire("a"))
    assert replayed == recorded
    assert inner.calls == []
    assert replayed.raw_body == recorded.raw_body
    document = _read(log)
    assert list(document) == ["format", "name", "model", "enabled_features", "calls"]
    assert (document["format"], document["name"], document["model"]) == (
        3,
        "scripted",
        "scripted-1",
    )
    key = canonical_hash("scripted", "scripted-1", _wire("a"))
    assert list(document["calls"]) == [key]  # type: ignore[arg-type]
    call = document["calls"][key]  # type: ignore[index]
    assert call["request"] == plain(canonical_request("scripted", "scripted-1", _wire("a")))


def test_a_miss_and_a_call_past_the_count_raise_replay_miss(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    replay = Recorded(_scripted(), log, "none")
    _call(replay, _wire("a"))
    key = canonical_hash("scripted", "scripted-1", _wire("a"))
    with pytest.raises(ReplayMiss, match=f"request {key}; call 2 has no recording"):
        _call(replay, _wire("a"))
    other = canonical_hash("scripted", "scripted-1", _wire("b"))
    with pytest.raises(ReplayMiss, match=f"0 calls for request {other}; call 1") as raised:
        _call(replay, _wire("b"))
    assert isinstance(raised.value, AssertionError)
    assert not isinstance(raised.value, ProviderError)
    assert len(replay.calls) == 3


def test_once_replays_when_the_log_exists(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    before = log.read_text()
    inner = _scripted(_answer(0.75))
    again = Recorded(inner, log, "once")
    assert _call(again, _wire("a")).answers["a"] == NoulAnswer(probability=0.25)
    with pytest.raises(ReplayMiss):
        _call(again, _wire("b"))
    assert inner.calls == []
    assert log.read_text() == before


def test_once_on_a_fresh_log_records_every_call_even_a_repeat(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    inner = _scripted(_answer(0.25), _answer(0.5))
    recorder = Recorded(inner, log, "once")
    _call(recorder, _wire("a"))
    _call(recorder, _wire("a"))
    assert len(inner.calls) == 2
    replay = Recorded(_scripted(), log, "none")
    assert _call(replay, _wire("a")).answers["a"] == NoulAnswer(probability=0.25)
    assert _call(replay, _wire("a")).answers["a"] == NoulAnswer(probability=0.5)


def test_new_replays_hits_records_misses_and_appends_past_the_count(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "new"), _wire("a"))
    inner = _scripted(_answer(0.5), _answer(0.75))
    adding = Recorded(inner, log, "new")
    assert _call(adding, _wire("a")).answers["a"] == NoulAnswer(probability=0.25)
    assert inner.calls == []
    assert _call(adding, _wire("b")).answers["b"].__class__.__name__ == "Fault"
    assert _call(adding, _wire("a")).answers["a"] == NoulAnswer(probability=0.75)
    assert len(inner.calls) == 2
    replay = Recorded(_scripted(), log, "none")
    assert _call(replay, _wire("a")).answers["a"] == NoulAnswer(probability=0.25)
    assert _call(replay, _wire("a")).answers["a"] == NoulAnswer(probability=0.75)
    _call(replay, _wire("b"))


class _Quota(ProviderError):
    """A leaf's own subclass, which a replay does not keep."""


def test_an_error_is_recorded_re_raised_and_replayed_as_the_base_class(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    error = _Quota(
        "over quota", is_retryable=True, retry_after=1.5, status_code=429, request_id="r-1"
    )
    with pytest.raises(_Quota) as raised:
        _call(Recorded(_scripted(error), log, "once"), _wire("a"))
    assert raised.value is error
    with pytest.raises(ProviderError) as replayed:
        _call(Recorded(_scripted(), log, "none"), _wire("a"))
    assert type(replayed.value) is ProviderError
    assert (
        replayed.value.message,
        replayed.value.is_retryable,
        replayed.value.retry_after,
        replayed.value.status_code,
        replayed.value.request_id,
    ) == ("over quota", True, 1.5, 429, "r-1")
    [entry] = _entries(log)
    assert set(entry["error"]) == {  # type: ignore[index]
        "message",
        "is_retryable",
        "retry_after",
        "status_code",
        "request_id",
    }


def _request() -> idia.Request:
    return idia.Request(state=text_state(), questions=(noul("a"),), deadline=Deadline.after(30.0))


def test_recorded_leaf_replays_calls_in_order_through_the_loop(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    retry = RetryPolicy.without_delay()
    busy = ProviderError("busy", is_retryable=True, status_code=503)
    recorder = Recorded(_scripted(busy, _answer(0.25)), log, "once")
    recorded = asyncio.run(Dispatcher(recorder, retry=retry).answer(_request()))
    inner = _scripted()
    replayed = asyncio.run(Dispatcher(Recorded(inner, log, "none"), retry=retry).answer(_request()))
    assert inner.calls == []
    assert replayed.answers == recorded.answers
    assert replayed.outcomes[0].attempts == recorded.outcomes[0].attempts == 2
    [attempt] = replayed.outcomes[0].failed_attempts
    assert (attempt.error.message, attempt.error.status_code) == ("busy", 503)


def test_recorded_provider_replays_request_failed(tmp_path: Path) -> None:
    log = tmp_path / "failing.json"
    retry = RetryPolicy.without_delay()
    final = ProviderError("bad request", status_code=400)
    with pytest.raises(RequestFailed):
        asyncio.run(
            Dispatcher(Recorded(_scripted(final), log, "once"), retry=retry).answer(_request())
        )
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(Dispatcher(Recorded(_scripted(), log, "none"), retry=retry).answer(_request()))
    error = raised.value.response.outcomes[0].error
    assert error is not None
    assert (error.message, error.status_code) == ("bad request", 400)


# ----------------------------------------------------------------------------------------
# With no leaf
# ----------------------------------------------------------------------------------------


def test_with_no_leaf_the_name_and_model_come_from_the_log(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    recorded = _call(
        Recorded(_scripted(_answer(0.25), name="sys1", model="m-7"), log, "once"), _wire("a")
    )
    record = capabilities()
    replay = Recorded(None, log, "none", capabilities=record)
    assert (replay.name, replay.model) == ("sys1", "m-7")
    assert replay.capabilities is record
    assert isinstance(replay.estimator, CharacterEstimator)
    assert isinstance(replay, Leaf)
    assert _call(replay, _wire("a")) == recorded


def test_with_no_leaf_capabilities_are_required_and_only_none_is_allowed(
    tmp_path: Path,
) -> None:
    log = tmp_path / "log.json"
    with pytest.raises(TypeError, match="needs capabilities="):
        Recorded(None, log, "none")
    with pytest.raises(ValueError, match="can only replay"):
        Recorded(None, log, "once", capabilities=capabilities())
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Recorded(None, log, "none", capabilities=capabilities())


def test_construction_refuses_what_cannot_be_honoured(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    with pytest.raises(TypeError, match="the wrapped leaf's"):
        Recorded(_scripted(), log, "once", capabilities=capabilities())
    with pytest.raises(ValueError, match="is not a mode"):
        Recorded(_scripted(), log, "all")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Recorded(_scripted(), log, "none")


def _featured(*items: Item, features: frozenset[str]) -> Scripted:
    return Scripted(capabilities=capabilities(enabled_features=features), script=list(items))


LONG = frozenset({"long-context"})


@pytest.mark.parametrize(
    ("recorded_with", "replayed_with"),
    [
        pytest.param(LONG, frozenset[str](), id="on-then-off"),
        pytest.param(frozenset[str](), LONG, id="off-then-on"),
    ],
)
def test_a_log_recorded_under_other_features_is_refused(
    tmp_path: Path, recorded_with: frozenset[str], replayed_with: frozenset[str]
) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_featured(_answer(0.25), features=recorded_with), log, "once"), _wire("a"))
    assert _read(log)["enabled_features"] == sorted(recorded_with)
    with pytest.raises(ValueError, match="with features"):
        Recorded(_featured(features=replayed_with), log, "none")
    with pytest.raises(ValueError, match="recorded with features"):
        Recorded(None, log, "none", capabilities=capabilities(enabled_features=replayed_with))
    replay = Recorded(_featured(features=recorded_with), log, "none")
    assert _call(replay, _wire("a")).answers["a"] == NoulAnswer(probability=0.25)
    bare = Recorded(None, log, "none", capabilities=capabilities(enabled_features=recorded_with))
    assert bare.capabilities.enabled_features == recorded_with


def test_a_log_recorded_from_another_leaf_or_format_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    with pytest.raises(ValueError, match="recorded from scripted/scripted-1 with features"):
        Recorded(_scripted(name="other"), log, "new")
    with pytest.raises(ValueError, match="not scripted/scripted-2 with"):
        Recorded(_scripted(model="scripted-2"), log, "none")
    log.write_text("[]")
    with pytest.raises(ValueError, match="not a recorded leaf's log"):
        Recorded(None, log, "none", capabilities=capabilities())


# ----------------------------------------------------------------------------------------
# The leaf's own surface
# ----------------------------------------------------------------------------------------


class _Raw:
    """A leaf whose raw body and usage are what a wire might send: bytes, or JSON holding
    a ``type`` key that must not be read as a tag."""

    name = "raw"
    model = "raw-1"

    def __init__(self, raw_body: object) -> None:
        self.capabilities: Capabilities = capabilities()
        self.estimator: TokenEstimator = CharacterEstimator()
        self._raw_body = raw_body

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        return WireResponse(
            answers={},
            elicitation="passthrough",
            calibrated=False,
            usage=Usage(input_tokens=1, output_tokens=2, raw_usage={"type": "Usage", "n": (1,)}),
            raw_body=self._raw_body,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "raw_body",
    [
        pytest.param(b"\x00not json", id="bytes"),
        pytest.param({"type": "Refusal", "items": (1, {"type": "x"})}, id="json-with-type"),
        pytest.param(None, id="none"),
    ],
)
def test_the_raw_body_and_raw_usage_are_kept_as_given(raw_body: object, tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    recorded = _call(Recorded(_Raw(raw_body), log, "once"), _wire("a"))
    replayed = _call(Recorded(_Raw(None), log, "none"), _wire("a"))
    assert replayed == recorded
    assert replayed.raw_body == raw_body
    [entry] = _entries(log)
    response = entry["response"]  # type: ignore[index]
    assert response["usage"]["raw_usage"] == {"type": "Usage", "n": [1]}
    if isinstance(raw_body, dict):
        assert response["raw_body"] == {"json": {"type": "Refusal", "items": [1, {"type": "x"}]}}


def test_a_response_with_no_usage_replays_with_none(tmp_path: Path) -> None:
    class NoUsage(_Raw):
        async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
            return WireResponse(answers={}, elicitation="logprobs", calibrated=True)

    log = tmp_path / "log.json"
    recorded = _call(Recorded(NoUsage(None), log, "once"), _wire("a"))
    replayed = _call(Recorded(NoUsage(None), log, "none"), _wire("a"))
    assert replayed == recorded
    assert replayed.usage is None


class _Passes(_Raw):
    """A leaf whose usage reports ``passes``, as a wire that splits a call into passes."""

    def __init__(self, passes: int | None) -> None:
        super().__init__(None)
        self._passes = passes

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        usage = Usage(input_tokens=1, output_tokens=0, passes=self._passes, raw_usage={"p": 1})
        return WireResponse(answers={}, elicitation="passthrough", calibrated=False, usage=usage)


def _logged_usage(log: Path) -> dict[str, object]:
    [entry] = _entries(log)
    return entry["response"]["usage"]  # type: ignore[index,no-any-return]


def test_usage_passes_is_recorded_and_replayed(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    recorded = _call(Recorded(_Passes(3), log, "once"), _wire("a"))
    replayed = _call(Recorded(_Passes(None), log, "none"), _wire("a"))
    assert replayed == recorded
    assert replayed.usage is not None
    assert replayed.usage.passes == 3
    assert _logged_usage(log)["passes"] == 3


def test_a_log_without_passes_replays_with_none(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_Passes(None), log, "once"), _wire("a"))
    assert "passes" not in _logged_usage(log)  # a log written before passes reads the same
    replayed = _call(Recorded(_Passes(7), log, "none"), _wire("a"))
    assert replayed.usage is not None
    assert replayed.usage.passes is None


def test_the_recorded_leaf_forwards_the_wrapped_leafs_surface(tmp_path: Path) -> None:
    inner = _scripted(name="sys1", model="m-2")
    leaf = Recorded(inner, tmp_path / "log.json", "once")
    assert isinstance(leaf, Leaf)
    assert not isinstance(leaf, Provider)
    assert (leaf.name, leaf.model) == ("sys1", "m-2")
    assert leaf.capabilities is inner.capabilities
    assert leaf.estimator is inner.estimator
    assert "Recorded" not in idia.__all__
    assert all("recorded" not in entry.target for entry in _REGISTRY.values())


def test_every_call_is_logged_before_it_is_answered(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    recorder = Recorded(_scripted(_answer(0.25), ProviderError("no")), log, "once")
    deadline = Deadline.after(5.0)
    asyncio.run(recorder.call(_wire("a"), deadline))
    with pytest.raises(ProviderError):
        asyncio.run(recorder.call(_wire("b"), deadline))
    assert [(c.wire_request, c.deadline) for c in recorder.calls] == [
        (_wire("a"), deadline),
        (_wire("b"), deadline),
    ]


def test_the_log_is_replaced_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "log.json"
    recorder = Recorded(_scripted(_answer(0.25), _answer(0.5)), log, "new")
    _call(recorder, _wire("a"))
    before = log.read_text()

    def broken(source: str, target: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        _call(recorder, _wire("b"))
    assert log.read_text() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.json"]


def test_the_log_bytes_are_the_same_on_every_platform(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    written = log.read_bytes()
    assert b"\r\n" not in written
    document = json.loads(written.decode("utf-8"))
    expected = json.dumps(document, indent=2, allow_nan=False) + "\n"
    assert written == expected.encode("utf-8")


def test_no_handle_on_the_log_survives_a_recording(tmp_path: Path) -> None:
    # On Windows a handle left open would refuse the unlink and the second write.
    log = tmp_path / "log.json"
    first = _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    log.unlink()
    second = _call(Recorded(_scripted(_answer(0.5)), log, "once"), _wire("a"))
    inner = _scripted()
    assert _call(Recorded(inner, log, "none"), _wire("a")) == second != first
    assert inner.calls == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.json"]


# ----------------------------------------------------------------------------------------
# The leaf half, against the scripted leaf under the recorded leaf, recording
# ----------------------------------------------------------------------------------------


class RecordedHarness(ScriptedHarness):
    """The scripted leaf wrapped in a recorded leaf that records to a fresh log."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self._directory = directory

    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        log = Path(tempfile.mkdtemp(dir=self._directory)) / "log.json"
        return Recorded(super()._build(capabilities, script), log, "once")


def _recorded_checks(directory: Path) -> Sequence[Check]:
    return leaf_checks(lambda: RecordedHarness(directory))


def test_the_recorded_leaf_passes_the_leaf_half_while_recording(tmp_path: Path) -> None:
    verdicts = {v.id: v for v in asyncio.run(run_checks(_recorded_checks(tmp_path)))}
    assert {id for id, v in verdicts.items() if v.result == "fail"} == set()
    assert {id for id, v in verdicts.items() if v.result == "not_applicable"} == (
        SCRIPTED_NOT_APPLICABLE
    )
    for id in ("L4", "L10", "L21"):
        reason = verdicts[id].reason
        assert reason is not None
        assert "idia.providers.recorded" in reason


def test_the_leaf_half_reaches_the_recorded_leafs_own_module(tmp_path: Path) -> None:
    [l10] = [check for check in _recorded_checks(tmp_path) if check.id == "L10"]
    assert asyncio.run(l10.run()) is not None
    [l9] = [check for check in _recorded_checks(tmp_path) if check.id == "L9"]
    with pytest.raises(CheckNotApplicable):
        asyncio.run(l9.run())


def test_options_are_part_of_the_key_and_the_warning_replays(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    optioned = WireRequest(state=text_state(), questions=(noul("a"),), options={"idia-test": 1})
    recorded = _call(Recorded(_scripted(_answer(0.25)), log, "once"), optioned)
    assert [w.code for w in recorded.warnings][-1] == idia.WARNING_UNKNOWN_OPTION
    replayed = _call(Recorded(_scripted(), log, "none"), optioned)
    assert replayed == recorded
    with pytest.raises(ReplayMiss):
        _call(Recorded(_scripted(), log, "none"), _wire("a"))


# ----------------------------------------------------------------------------------------
# Key order
# ----------------------------------------------------------------------------------------


def _json_wire(value: dict[str, Any]) -> WireRequest:
    return WireRequest(state=State(parts=(JsonPart(value=value),)), questions=(noul("a"),))


class _Ordered(_Raw):
    """A leaf whose ``prompt_sent`` and raw body hold mappings in an order that is not
    sorted, as a wire leaf sends a caller's JSON part."""

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        sent = {"state": {"zulu": 1, "alpha": {"yankee": 2, "bravo": 3}}, "model": "raw-1"}
        return WireResponse(
            answers={"a": NoulAnswer(probability=0.25)},
            elicitation="passthrough",
            calibrated=False,
            prompt_sent=sent,
            raw_body={"zulu": {"yankee": 1, "bravo": 2}, "alpha": 0},
            usage=Usage(input_tokens=1, output_tokens=2, raw_usage={"out": 2, "in": 1}),
        )


def test_a_replayed_prompt_sent_keeps_its_key_order(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    recorded = _call(Recorded(_Ordered(None), log, "once"), _wire("a"))
    replayed = _call(Recorded(_Ordered(None), log, "none"), _wire("a"))
    for held in (recorded, replayed):
        sent = cast(dict[str, Any], held.prompt_sent)
        assert list(sent) == ["state", "model"]
        assert list(sent["state"]) == ["zulu", "alpha"]
        assert list(sent["state"]["alpha"]) == ["yankee", "bravo"]
        raw = cast(dict[str, Any], held.raw_body)
        assert list(raw) == ["zulu", "alpha"]
        assert list(raw["zulu"]) == ["yankee", "bravo"]
        assert held.usage is not None
        assert list(held.usage.raw_usage) == ["out", "in"]


def test_two_key_orders_are_recorded_and_replayed_apart(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    first, second = _json_wire({"b": 1, "a": 2}), _json_wire({"a": 2, "b": 1})
    assert first == second
    recorder = Recorded(_scripted(_answer(0.25), _answer(0.75)), log, "once")
    _call(recorder, first)
    _call(recorder, second)
    assert len(cast(dict[str, object], _read(log)["calls"])) == 2
    replay = Recorded(_scripted(), log, "none")
    assert _call(replay, second).answers["a"] == NoulAnswer(probability=0.75)
    assert _call(replay, first).answers["a"] == NoulAnswer(probability=0.25)
    stored = [call["request"] for call in cast(dict[str, Any], _read(log)["calls"]).values()]
    values = sorted(list(r["state"]["parts"][0]["value"]) for r in stored)
    assert values == [["a", "b"], ["b", "a"]]


def test_the_log_bytes_are_the_same_whatever_order_its_keys_were_recorded_in(
    tmp_path: Path,
) -> None:
    logs = [tmp_path / "ab.json", tmp_path / "ba.json"]
    for log, keys in zip(logs, (("a", "b"), ("b", "a")), strict=True):
        recorder = Recorded(_Raw(None), log, "once")
        for key in keys:
            _call(recorder, _wire(key))
    assert logs[0].read_bytes() == logs[1].read_bytes()
    keys = list(cast(dict[str, object], _read(logs[0])["calls"]))
    assert keys == sorted(keys)


@pytest.mark.parametrize("format", [1, 2, 4, None])
def test_a_log_of_another_format_is_refused_at_construction(
    tmp_path: Path, format: int | None
) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    log.write_text(json.dumps({**_read(log), "format": format}))
    with pytest.raises(ValueError, match=rf"format {format}; this reads 3: .* upgrade_log"):
        Recorded(_scripted(), log, "none")
    with pytest.raises(ValueError, match="upgrade_log"):
        Recorded(None, log, "none", capabilities=capabilities())


def test_a_stored_request_that_does_not_hash_to_its_key_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), _wire("a"))
    document = _read(log)
    [call] = cast(dict[str, dict[str, Any]], document["calls"]).values()
    call["request"]["questions"][0]["key"] = "b"
    log.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not hash to its key"):
        Recorded(_scripted(), log, "none")
    call["request"] = None
    call["extra"] = 1
    log.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not a request and its entries"):
        Recorded(_scripted(), log, "none")


def test_a_structured_instruction_in_another_order_is_a_replay_miss(tmp_path: Path) -> None:
    def asked(instruction: dict[str, str]) -> WireRequest:
        question = Question(key="a", kind="noul", instruction=instruction)
        return WireRequest(state=text_state(), questions=(question,))

    log = tmp_path / "log.json"
    _call(Recorded(_scripted(_answer(0.25)), log, "once"), asked({"q": "?", "e": "x"}))
    with pytest.raises(ReplayMiss):
        _call(Recorded(_scripted(), log, "none"), asked({"e": "x", "q": "?"}))
