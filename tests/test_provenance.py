"""The provenance profile: each detail a value or a reason code, the details only claimed,
the runtime's and the client libraries' versions, and the outcome's rounding.

The shapes refuse a detail or reason outside the closed sets and an inconsistent pair; the
dispatcher gives every failed outcome ``not-sent`` and idia's version; the recorded leaf's
log leaves the new fields out when empty, so a log written before them replays with the
defaults, and one that fills them replays them unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import idia
from idia import (
    DETAILS,
    REASONS,
    Deadline,
    Dispatcher,
    JsonPart,
    LabelAnswer,
    NoulAnswer,
    Outcome,
    ProviderError,
    Question,
    RequestFailed,
    Response,
    RetryPolicy,
    Rounding,
    State,
    TextPart,
    Usage,
    WireRequest,
    WireResponse,
)
from idia.conformance._base import profile_problems
from idia.providers.recorded import Recorded
from idia.providers.scripted import Reply, Scripted
from tests._support import capabilities, choice, noul, request

_ROUNDING = Rounding(probability_decimals=2, score_decimals=2)


def _response(**fields: Any) -> WireResponse:
    base: dict[str, Any] = {
        "answers": {},
        "elicitation": "passthrough",
        "calibrated": False,
        "model_reported": "m",
    }
    return WireResponse(**{**base, **fields})


def _outcome(**fields: Any) -> Outcome:
    base: dict[str, Any] = {
        "keys": ("q",),
        "error": None,
        "provider": "p",
        "model_requested": "m",
        "model_reported": "m",
        "snapshot": None,
        "elicitation": "passthrough",
        "calibrated": False,
        "prompt_sent": None,
        "usage": None,
        "latency": 0.0,
        "started_at": 1.0,
        "request_id": None,
        "failed_attempts": (),
        "warnings": (),
        "rounding": _ROUNDING,
    }
    return Outcome(**{**base, **fields})


# ----------------------------------------------------------------------------------------
# The shapes
# ----------------------------------------------------------------------------------------


def test_the_closed_sets_are_the_profiles_details_and_reasons() -> None:
    assert {
        "model_reported",
        "snapshot",
        "versions",
        "elicitation",
        "calibrated",
        "prompt_sent",
        "usage.input_tokens",
        "usage.output_tokens",
        "request_id",
    } == DETAILS
    assert {
        "not-reported",
        "not-pinned",
        "not-sent",
        "hidden",
        "not-applicable",
        "not-exposed",
    } == REASONS


def test_the_new_fields_default_empty_and_are_read_only() -> None:
    response = _response(versions={"torch": "2.1"}, unreported={"snapshot": "not-pinned"})
    assert dict(response.client_versions) == {}
    assert response.claimed == frozenset()
    with pytest.raises(TypeError):
        response.versions["torch"] = "9"  # type: ignore[index]
    with pytest.raises(TypeError):
        response.unreported["snapshot"] = "hidden"  # type: ignore[index]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"unreported": {"provider": "not-reported"}}, "'provider', which is not a provenance"),
        ({"unreported": {"snapshot": "lost"}}, "the unknown reason 'lost'"),
        ({"claimed": frozenset({"cost"})}, "'cost', which is not a provenance detail"),
        (
            {"snapshot": "s", "unreported": {"snapshot": "not-pinned"}, "claimed": {"snapshot"}},
            "both unreported and claimed",
        ),
        ({"unreported": {"model_reported": "hidden"}}, "holds a value, so it has no reason"),
        ({"claimed": frozenset({"snapshot"})}, "claimed but holds no value"),
    ],
)
def test_a_wire_response_and_an_outcome_refuse_an_inconsistent_profile(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _response(**fields)
    with pytest.raises(ValueError, match=message):
        _outcome(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"versions": ["torch"]},
        {"versions": {"torch": 2}},
        {"client_versions": "httpx2"},
        {"unreported": ["snapshot"]},
        {"claimed": "snapshot"},
    ],
)
def test_a_wire_response_and_an_outcome_refuse_a_provenance_field_of_the_wrong_shape(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        _response(**fields)
    with pytest.raises(TypeError):
        _outcome(**fields)


def test_usage_figures_are_held_through_usage() -> None:
    usage = Usage(input_tokens=3, output_tokens=None)
    response = _response(usage=usage, unreported={"usage.output_tokens": "not-applicable"})
    assert response.unreported == {"usage.output_tokens": "not-applicable"}
    with pytest.raises(ValueError, match="holds a value"):
        _response(usage=usage, unreported={"usage.input_tokens": "not-reported"})


def test_the_public_names_are_listed() -> None:
    for name in ("ProvenanceDetail", "UnreportedReason", "DETAILS", "REASONS"):
        assert name in idia.__all__


# ----------------------------------------------------------------------------------------
# The dispatcher
# ----------------------------------------------------------------------------------------


def test_an_answered_outcome_carries_its_wire_responses_provenance_and_idias_version() -> None:
    reply = Reply(
        answers={"q": NoulAnswer(probability=0.25)},
        versions={"runtime": "1.0"},
        client_versions={"httpx2": "2.16"},
        claimed=frozenset({"model_reported"}),
    )
    record = capabilities(rounding=Rounding(probability_decimals=3, score_decimals=None))
    leaf = Scripted(capabilities=record, script=[reply])
    response = asyncio.run(Dispatcher(leaf).answer(request(noul())))
    (outcome,) = response.outcomes
    assert dict(outcome.versions) == {"runtime": "1.0"}
    assert dict(outcome.client_versions) == {"httpx2": "2.16", "idia": idia.__version__}
    assert outcome.claimed == {"model_reported"}
    assert dict(outcome.unreported) == {}
    assert outcome.rounding == record.rounding


# ----------------------------------------------------------------------------------------
# Extras: the figures a wire reports beyond the shared vocabulary
# ----------------------------------------------------------------------------------------


def test_extras_are_empty_by_default_and_held_read_only_in_the_order_built() -> None:
    assert dict(_response().extras) == {}
    assert dict(_outcome().extras) == {}
    built: dict[str, Any] = {"passes": 2, "detail": {"z": [1, 2.5], "a": None}}
    for held in (_response(extras=built), _outcome(extras=built)):
        built["detail"]["z"].append(3)
        assert list(held.extras) == ["passes", "detail"]
        detail = cast(Any, held.extras["detail"])
        assert list(detail) == ["z", "a"]
        assert detail["z"] == (1, 2.5)
        with pytest.raises(TypeError):
            detail["a"] = 1
        built["detail"]["z"].pop()


@pytest.mark.parametrize(
    ("extras", "message"),
    [
        pytest.param({1: "x"}, "the key 1, which is not a string", id="int-key"),
        pytest.param({"a": {"b": {2: 1}}}, "the key 2, which is not a string", id="deep-key"),
        pytest.param({"a": float("nan")}, "nan, which JSON cannot carry", id="nan"),
        pytest.param({"a": (1, float("inf"))}, r"\['a'\]\[1\] holds inf", id="deep-inf"),
        pytest.param({"a": b"x"}, "a bytes, which is not a JSON value", id="bytes"),
        pytest.param({"a": {1, 2}}, "a set, which is not a JSON value", id="set"),
    ],
)
def test_extras_outside_the_json_domain_are_refused(extras: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _response(extras=extras)
    with pytest.raises(ValueError, match=message):
        _outcome(extras=extras)


def test_extras_that_are_not_a_mapping_or_contain_themselves_are_refused() -> None:
    with pytest.raises(TypeError, match="extras must be a mapping"):
        _response(extras=[("passes", 2)])
    looped: dict[str, Any] = {}
    looped["self"] = looped
    with pytest.raises(ValueError, match="contains itself"):
        _outcome(extras=looped)
    deep: dict[str, Any] = {}
    for _ in range(sys.getrecursionlimit() + 10):
        deep = {"d": deep}
    with pytest.raises(ValueError, match="nests too deep"):
        _response(extras=deep)


def test_dispatch_carries_the_wire_responses_extras_onto_the_answered_outcome() -> None:
    reply = Reply(answers={"q": NoulAnswer(probability=0.25)}, extras={"passes": 2, "b": 0})
    leaf = Scripted(capabilities=capabilities(), script=[reply])
    (outcome,) = asyncio.run(Dispatcher(leaf).answer(request(noul()))).outcomes
    assert dict(outcome.extras) == {"passes": 2, "b": 0}
    assert list(outcome.extras) == ["passes", "b"]
    assert "extras" not in outcome.unreported  # not a provenance detail


def test_an_outcome_answered_on_a_retry_carries_the_answering_calls_extras() -> None:
    busy = ProviderError("busy", is_retryable=True, status_code=503)
    reply = Reply(answers={"q": NoulAnswer(probability=0.25)}, extras={"passes": 1})
    leaf = Scripted(capabilities=capabilities(), script=[busy, reply])
    retry = RetryPolicy(attempts=2, base_delay=0.0, jitter=0.0)
    (outcome,) = asyncio.run(Dispatcher(leaf, retry=retry).answer(request(noul()))).outcomes
    assert len(outcome.failed_attempts) == 1
    assert dict(outcome.extras) == {"passes": 1}


def test_a_failed_outcome_carries_no_extras() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[ProviderError("refused", status_code=400)])
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(Dispatcher(leaf).answer(request(noul())))
    (outcome,) = raised.value.response.outcomes
    assert dict(outcome.extras) == {}


def test_a_fallback_passes_the_answering_members_extras_through() -> None:
    from idia import Fallback

    first = Scripted(
        capabilities=capabilities(), script=[ProviderError("no", status_code=400)], name="first"
    )
    reply = Reply(answers={"q": NoulAnswer(probability=0.25)}, extras={"passes": 4})
    second = Scripted(capabilities=capabilities(), script=[reply], name="second")
    response = asyncio.run(Fallback([Dispatcher(first), Dispatcher(second)]).answer(request()))
    (outcome,) = response.outcomes
    assert dict(outcome.extras) == {"passes": 4}


def test_idias_own_version_wins_over_a_leafs() -> None:
    reply = Reply(answers={"q": NoulAnswer(probability=0.25)}, client_versions={"idia": "9"})
    leaf = Scripted(capabilities=capabilities(), script=[reply])
    (outcome,) = asyncio.run(Dispatcher(leaf).answer(request(noul()))).outcomes
    assert outcome.client_versions["idia"] == idia.__version__


def test_a_failed_outcome_gives_not_sent_to_every_detail_with_no_value() -> None:
    refused = ProviderError("refused", status_code=400, request_id="r-9")
    leaf = Scripted(capabilities=capabilities(), script=[refused])
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(Dispatcher(leaf).answer(request(noul())))
    (outcome,) = raised.value.response.outcomes
    assert outcome.request_id == "r-9"
    assert dict(outcome.unreported) == dict.fromkeys(DETAILS - {"request_id"}, "not-sent")
    assert dict(outcome.extras) == {}
    assert outcome.claimed == frozenset()
    assert dict(outcome.client_versions) == {"idia": idia.__version__}
    assert outcome.rounding == capabilities().rounding


def test_a_wire_request_never_sent_gives_not_sent_to_its_request_id_too() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[])
    past = request(noul(), seconds=-1.0)
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(Dispatcher(leaf, retry=RetryPolicy.without_delay()).answer(past))
    (outcome,) = raised.value.response.outcomes
    assert outcome.attempts == 0
    assert dict(outcome.unreported) == dict.fromkeys(DETAILS, "not-sent")
    assert dict(outcome.extras) == {}


def _two_noul_plan(script: list[Any], **dispatcher: Any) -> RequestFailed | Response:
    record = capabilities(max_questions=1)
    leaf = Scripted(capabilities=record, script=script)
    dispatch = Dispatcher(leaf, retry=RetryPolicy.without_delay(), **dispatcher)

    async def ask() -> RequestFailed | Response:
        try:
            return await dispatch.answer(request(noul("a"), noul("b")))
        except RequestFailed as failed:
            return failed

    return asyncio.run(ask())


def test_a_wire_request_left_unsent_by_a_credential_error_gives_not_sent_to_all() -> None:
    refused = ProviderError("unauthorized", status_code=401, request_id="r-1")
    ended = _two_noul_plan([refused])
    assert isinstance(ended, RequestFailed)
    first, unsent = ended.response.outcomes
    assert first.request_id == "r-1"
    assert unsent.request_id is None
    assert unsent.error is refused
    assert dict(unsent.unreported) == dict.fromkeys(DETAILS, "not-sent")
    assert dict(unsent.extras) == {}


def test_a_retry_the_deadline_cannot_fit_gives_not_sent_and_keeps_the_attempts_id() -> None:
    busy = ProviderError("busy", is_retryable=True, status_code=503, request_id="r-2")
    leaf = Scripted(capabilities=capabilities(), script=[busy])
    retry = RetryPolicy(attempts=3, base_delay=60.0, jitter=0.0)
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(Dispatcher(leaf, retry=retry).answer(request(noul(), seconds=5.0)))
    (outcome,) = raised.value.response.outcomes
    assert outcome.request_id is None
    assert [attempt.request_id for attempt in outcome.failed_attempts] == ["r-2"]
    assert dict(outcome.unreported) == dict.fromkeys(DETAILS, "not-sent")
    assert dict(outcome.extras) == {}


def test_a_routers_outcomes_carry_each_members_own_rounding() -> None:
    from idia import Router

    label = Rounding(probability_decimals=2, score_decimals=2)
    yes = Rounding(probability_decimals=4, score_decimals=None)
    choices = Scripted(
        capabilities=capabilities(kinds=frozenset({"choice"}), rounding=label),
        script=[{"c": LabelAnswer(probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.2)))}],
    )
    nouls = Scripted(
        capabilities=capabilities(kinds=frozenset({"noul"}), rounding=yes),
        script=[{"n": NoulAnswer(probability=0.25)}],
    )
    router = Router({"choice": Dispatcher(choices), "noul": Dispatcher(nouls)})
    response = asyncio.run(router.answer(request(choice("c"), noul("n"))))
    by_keys = {outcome.keys: outcome.rounding for outcome in response.outcomes}
    assert by_keys == {("c",): label, ("n",): yes}


def test_a_fallbacks_answering_outcome_keeps_its_own_provenance() -> None:
    from idia import Fallback

    refused = ProviderError("refused", status_code=400)
    first = Scripted(capabilities=capabilities(), script=[refused], name="first")
    reply = Reply(answers={"q": NoulAnswer(probability=0.25)}, versions={"runtime": "1"})
    second = Scripted(capabilities=capabilities(), script=[reply], name="second")
    response = asyncio.run(Fallback([Dispatcher(first), Dispatcher(second)]).answer(request()))
    (outcome,) = response.outcomes
    assert outcome.provider == "second"
    assert dict(outcome.versions) == {"runtime": "1"}
    assert len(outcome.failed_attempts) == 1
    assert dict(outcome.unreported) == {}


# ----------------------------------------------------------------------------------------
# The recorded leaf's log
# ----------------------------------------------------------------------------------------

_KEY = "8d292a40ec00d837b140c55c9a7547f84b48667cbacd4665472f5cf6bbf7fc49"
"""The key of :func:`_wire`'s request from the scripted leaf, in formats 3 and 4."""
_OLD_KEY = "b0a43f8b7956e51da539cb6b677dbfa06262286a65bd47ba6dc7f544cbdb6788"
"""The same request's key in format 2, which sorted the keys of the canonical request."""
_REQUEST = (
    '{"name":"scripted","model":"scripted-1","state":{"type":"State","parts":[{"type":'
    '"TextPart","text":"t"}]},"questions":[{"type":"Question","key":"q","kind":"noul",'
    '"instruction":"Is it so?","labels":[],"value_type":null,"is_list":false}]}'
)
_ENTRY = (
    '{"response":{"answers":{"q":{"confidence":null,"derived":[],'
    '"off_label_mass":null,"probability":0.25,"type":"NoulAnswer"}},"calibrated":true,'
    '"elicitation":"passthrough","model_reported":"scripted-1","prompt_sent":"x",'
    '"raw_body":{"json":{"answers":{}}},"request_id":"scripted-1","snapshot":"scripted-1",'
    '"type":"WireResponse","usage":{"input_tokens":1,"output_tokens":0,'
    '"raw_usage":{},"type":"Usage"},"warnings":[]}}'
)
"""A response recorded before the provenance fields existed: none of them is in it."""
_LOG = (
    '{"format":4,"name":"scripted","model":"scripted-1","enabled_features":[],"calls":{"'
    + _KEY
    + '":{"request":'
    + _REQUEST
    + ',"entries":['
    + _ENTRY
    + "]}}}"
)
"""A format-4 log holding that response."""
_OLD_LOG = (
    '{"calls":{"' + _OLD_KEY + '":[' + _ENTRY + ']},"enabled_features":[],"format":2,'
    '"model":"scripted-1","name":"scripted"}'
)
"""The same response in a format-2 log, keyed on the sorted request and storing none."""


def _wire() -> WireRequest:
    question = Question(key="q", kind="noul", instruction="Is it so?")
    return WireRequest(state=State(parts=(TextPart(text="t"),)), questions=(question,))


def _replay(log: Path) -> WireResponse:
    replayer = Recorded(None, log, "none", capabilities=capabilities())
    return asyncio.run(replayer.call(_wire(), Deadline.after(5)))


def test_a_log_written_before_the_provenance_fields_replays_with_their_defaults(
    tmp_path: Path,
) -> None:
    log = tmp_path / "old.json"
    log.write_text(_LOG, encoding="utf-8")
    replayed = _replay(log)
    assert replayed.answers == {"q": NoulAnswer(probability=0.25)}
    assert dict(replayed.versions) == {}
    assert dict(replayed.client_versions) == {}
    assert dict(replayed.unreported) == {}
    assert replayed.claimed == frozenset()


def _format_3(tmp_path: Path, **response: object) -> Path:
    """:data:`_LOG` as format 3 wrote it, its one response's fields replaced by
    ``response``: a usage holding ``passes``, say, as 0.1.0 recorded Clairic's."""
    document = json.loads(_LOG)
    document["format"] = 3
    (entry,) = document["calls"][_KEY]["entries"]
    entry["response"].update(response)
    log = tmp_path / "format-3.json"
    log.write_text(json.dumps(document), encoding="utf-8")
    return log


def test_a_format_3_log_is_refused_until_upgraded(tmp_path: Path) -> None:
    log = _format_3(tmp_path)
    with pytest.raises(ValueError, match=r"format 3; this reads 4: .*upgrade_log"):
        _replay(log)


def test_upgrading_a_format_3_log_moves_its_pass_count_into_extras(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    usage = {"input_tokens": 1, "output_tokens": 0, "passes": 2, "raw_usage": {"passes": 2}}
    log = _format_3(tmp_path, usage={**usage, "type": "Usage"})
    before = json.loads(log.read_text(encoding="utf-8"))
    assert upgrade_log(log) == 0
    after = json.loads(log.read_text(encoding="utf-8"))
    assert after["format"] == 4
    assert list(after["calls"]) == [_KEY]
    assert after["calls"][_KEY]["request"] == before["calls"][_KEY]["request"]
    (entry,) = after["calls"][_KEY]["entries"]
    assert "passes" not in entry["response"]["usage"]
    assert entry["response"]["extras"] == {"passes": 2}
    (was,) = before["calls"][_KEY]["entries"]
    kept = {k: v for k, v in entry["response"].items() if k not in {"usage", "extras"}}
    assert kept == {k: v for k, v in was["response"].items() if k != "usage"}
    replayed = _replay(log)
    assert dict(replayed.extras) == {"passes": 2}
    assert replayed.usage is not None
    assert replayed.usage.raw_usage["passes"] == 2


def test_upgrading_a_format_3_log_drops_the_passes_detail_and_a_null_count(
    tmp_path: Path,
) -> None:
    from idia.providers.recorded import upgrade_log

    log = _format_3(
        tmp_path,
        usage={"input_tokens": 1, "output_tokens": 0, "passes": None, "type": "Usage"},
        unreported={"usage.passes": "not-reported", "snapshot": "not-pinned"},
        claimed=["model_reported", "usage.passes"],
        snapshot=None,
        warnings=[{"code": "no-snapshot", "field": "snapshot", "message": "m", "type": "Notice"}],
    )
    assert upgrade_log(log, requests=[_wire()], reasons={"versions": "hidden"}) == 0
    (entry,) = json.loads(log.read_text(encoding="utf-8"))["calls"][_KEY]["entries"]
    assert "extras" not in entry["response"]
    assert entry["response"]["unreported"] == {"snapshot": "not-pinned"}
    assert entry["response"]["claimed"] == ["model_reported"]
    replayed = _replay(log)
    assert dict(replayed.extras) == {}
    assert replayed.claimed == {"model_reported"}


def test_a_format_3_response_holding_both_passes_and_extras_is_refused(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    usage = {"input_tokens": 1, "output_tokens": 0, "passes": 2, "type": "Usage"}
    log = _format_3(tmp_path, usage=usage, extras={"passes": 3})
    before = log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="both a pass count and extras"):
        upgrade_log(log)
    assert log.read_text(encoding="utf-8") == before


def test_upgrading_refuses_a_log_in_a_format_it_does_not_read(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = tmp_path / "log.json"
    log.write_text(_LOG.replace('"format":4', '"format":1'), encoding="utf-8")
    with pytest.raises(ValueError, match="format 1; upgrade_log reads only formats 2 and 3"):
        upgrade_log(log)
    log.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a recorded leaf's log"):
        upgrade_log(log)


def test_a_log_that_fills_the_provenance_fields_replays_them_unchanged(tmp_path: Path) -> None:
    reply = Reply(
        answers={"q": NoulAnswer(probability=0.25)},
        pinned=False,
        versions={"runtime": "1.0", "python": "3.11"},
        client_versions={"httpx2": "2.16"},
        claimed=frozenset({"model_reported"}),
    )
    log = tmp_path / "filled.json"
    leaf = Scripted(capabilities=capabilities(), script=[reply])
    recorded = asyncio.run(Recorded(leaf, log, "once").call(_wire(), Deadline.after(5)))
    stored = json.loads(log.read_text(encoding="utf-8"))
    ((entry,),) = (call["entries"] for call in stored["calls"].values())
    assert {"versions", "client_versions", "unreported", "claimed"} <= set(entry["response"])
    assert stored["format"] == 4
    replayed = _replay(log)
    for name in ("versions", "client_versions", "unreported", "claimed"):
        assert getattr(replayed, name) == getattr(recorded, name), name
    assert dict(replayed.unreported) == {"snapshot": "not-pinned"}


# ----------------------------------------------------------------------------------------
# The wire leaves' shared helpers
# ----------------------------------------------------------------------------------------


def test_client_versions_reads_each_installed_package_and_leaves_out_the_rest() -> None:
    pytest.importorskip("httpx2")
    from idia.providers.http import client_versions

    assert client_versions("httpx2", "idia-no-such-package") == {
        "httpx2": importlib.metadata.version("httpx2")
    }


def test_wire_unreported_gives_the_reasons_a_wire_leaf_states() -> None:
    pytest.importorskip("httpx2")
    from idia.providers.http import wire_unreported

    full = Usage(input_tokens=1, output_tokens=0)
    assert wire_unreported(snapshot="v1", usage=full, request_id="r") == {
        "versions": "not-applicable"
    }
    assert wire_unreported(snapshot=None, usage=None, request_id=None) == {
        "versions": "not-applicable",
        "snapshot": "not-pinned",
        "usage.input_tokens": "not-reported",
        "usage.output_tokens": "not-reported",
        "request_id": "not-reported",
    }


def test_client_versions_never_gives_idia_which_dispatch_adds() -> None:
    pytest.importorskip("httpx2")
    from idia.providers.http import client_versions

    assert "idia" not in client_versions("idia", "httpx2")


# ----------------------------------------------------------------------------------------
# Upgrading a log written before the profile
# ----------------------------------------------------------------------------------------


def _old_log(
    tmp_path: Path,
    name: str,
    *items: Any,
    wire: WireRequest | None = None,
    old: bool = True,
) -> Path:
    """A log as idia wrote it before the profile: recorded with every provenance field empty,
    which the log leaves out, every call of ``wire``, and with ``old`` in format 2, keyed
    on the sorted request, as idia wrote it before the key kept key order."""
    log = tmp_path / f"{name}.json"
    leaf = Scripted(capabilities=capabilities(), script=list(items), name=name, model="m-1")
    recorder = Recorded(leaf, log, "once")
    for _ in items:
        with contextlib.suppress(ProviderError):
            asyncio.run(recorder.call(wire or _wire(), Deadline.after(5)))
        recorder = Recorded(leaf, log, "new")
    assert all(k not in log.read_text() for k in ('"unreported"', '"versions"', '"claimed"'))
    if old:
        _as_format_2(log)
    return log


def _as_format_2(log: Path) -> None:
    """Rewrite a format-4 log as the format-2 writer wrote it: each key's calls under the
    hash of its request with sorted keys, no request stored, and every key sorted."""
    document = json.loads(log.read_text(encoding="utf-8"))
    calls: dict[str, Any] = {}
    for call in document["calls"].values():
        rendered = json.dumps(call["request"], sort_keys=True, separators=(",", ":"))
        calls.setdefault(hashlib.sha256(rendered.encode()).hexdigest(), []).extend(call["entries"])
    document.update(format=2, calls=calls)
    log.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calls(log: Path) -> dict[str, list[dict[str, Any]]]:
    """A format-4 log's entries by key."""
    document = json.loads(log.read_text(encoding="utf-8"))
    assert document["format"] == 4
    return {key: call["entries"] for key, call in document["calls"].items()}


def _replayed(log: Path) -> list[WireResponse]:
    return [
        _replay_entry(entry)
        for entries in _calls(log).values()
        for entry in entries
        if "response" in entry
    ]


def _replay_entry(entry: dict[str, Any]) -> WireResponse:
    from idia.providers.recorded import _replay  # pyright: ignore[reportPrivateUsage]

    return _replay(entry)


def _bare(**fields: Any) -> Reply:
    """A scripted reply that leaves the profile empty, as an old leaf's response was."""
    return Reply(answers={"q": NoulAnswer(probability=0.25)}, unreported={}, **fields)


def test_upgrading_a_wire_leafs_log_gives_the_reasons_the_leaf_gives_today(
    tmp_path: Path,
) -> None:
    from idia.providers.recorded import upgrade_log

    usage = Usage(input_tokens=3, output_tokens=0, raw_usage={"input_tokens": 3})
    log = _old_log(tmp_path, "cloudflare-clairic", _bare(pinned=False, usage=usage))
    assert upgrade_log(log, requests=[_wire()]) == 0
    (response,) = _replayed(log)
    assert dict(response.unreported) == {"versions": "not-applicable", "snapshot": "not-pinned"}
    assert dict(response.client_versions) == {}
    assert response.claimed == frozenset()
    assert profile_problems(response) == []
    assert response.answers == {"q": NoulAnswer(probability=0.25)}
    after = log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="format 4; upgrade_log reads only formats 2 and 3"):
        upgrade_log(log, requests=[_wire()])
    assert log.read_text(encoding="utf-8") == after


def test_upgrading_keeps_every_recorded_value_and_leaves_errors_alone(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    refused = ProviderError("refused", status_code=400, request_id="r-1")
    log = _old_log(tmp_path, "system-one", _bare(), refused)
    old = json.loads(log.read_text(encoding="utf-8"))
    assert upgrade_log(log, requests=[_wire()]) == 0
    new = json.loads(log.read_text(encoding="utf-8"))
    (key,) = new["calls"]
    ((old_key, entries),) = old["calls"].items()
    for was, now in zip(entries, new["calls"][key]["entries"], strict=True):
        if "error" in was:
            assert now == was
            continue
        added = {"versions", "client_versions", "unreported", "claimed"}
        assert {k: v for k, v in now["response"].items() if k not in added} == was["response"]
    assert old_key != key
    assert {k: v for k, v in new.items() if k != "calls"} == {
        **{k: v for k, v in old.items() if k != "calls"},
        "format": 4,
    }


def test_a_gliner_log_takes_its_versions_from_the_raw_body_and_claims(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "gliner", _bare(usage=Usage(input_tokens=9, output_tokens=None)))
    document = json.loads(log.read_text(encoding="utf-8"))
    (entries,) = document["calls"].values()
    entries[0]["response"]["raw_body"] = {
        "json": {
            "output": {"entities": {}},
            "versions": {
                "python": "3.11.13",
                "gliner2": "2.1.0",
                "torch": "2.14.1",
                "transformers": "5.20.2",
                "huggingface_hub": "2.3.0",
                "missing": None,
            },
        }
    }
    raw_body = entries[0]["response"]["raw_body"]
    log.write_text(json.dumps(document), encoding="utf-8")
    assert upgrade_log(log, requests=[_wire()]) == 0
    (response,) = _replayed(log)
    assert dict(response.versions) == {
        "python": "3.11.13",
        "gliner2": "2.1.0",
        "torch": "2.14.1",
        "transformers": "5.20.2",
    }
    assert dict(response.client_versions) == {"huggingface_hub": "2.3.0"}
    assert dict(response.unreported) == {"usage.output_tokens": "not-applicable"}
    assert response.claimed == {"model_reported", "snapshot"}
    assert profile_problems(response) == []
    (entries,) = _calls(log).values()
    assert entries[0]["response"]["raw_body"] == raw_body


@pytest.mark.parametrize(
    ("name", "reply", "message"),
    [
        pytest.param("someone-else", _bare(), "no upgrade rules for the leaf", id="unknown"),
        pytest.param("gliner", _bare(), "records no runtime version", id="gliner-no-versions"),
    ],
)
def test_upgrading_refuses_a_log_it_cannot_upgrade_honestly_and_writes_nothing(
    tmp_path: Path, name: str, reply: Reply, message: str
) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, name, reply)
    before = log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        upgrade_log(log, requests=[_wire()])
    assert log.read_text(encoding="utf-8") == before


def test_an_unknown_leaf_is_upgraded_with_the_reasons_its_caller_gives(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    usage = Usage(input_tokens=1, output_tokens=None)
    log = _old_log(tmp_path, "someone-else", _bare(usage=usage))
    with pytest.raises(ValueError, match=r"leave out usage\.output_tokens"):
        upgrade_log(log, requests=[_wire()], reasons={"versions": "not-applicable"})
    hidden: dict[Any, Any] = {"versions": "not-applicable", "usage.output_tokens": "hidden"}
    assert upgrade_log(log, requests=[_wire()], reasons=hidden) == 0
    (response,) = _replayed(log)
    assert dict(response.unreported) == hidden
    assert profile_problems(response) == []


def test_a_clairic_pass_count_only_the_raw_usage_holds_is_read_into_extras_as_today(
    tmp_path: Path,
) -> None:
    from idia.providers.recorded import upgrade_log

    usage = Usage(input_tokens=3, output_tokens=0, raw_usage={"input_tokens": 3, "passes": 2})
    log = _old_log(tmp_path, "cloudflare-clairic", _bare(pinned=False, usage=usage))
    assert upgrade_log(log, requests=[_wire()]) == 0
    (response,) = _replayed(log)
    assert dict(response.unreported) == {"versions": "not-applicable", "snapshot": "not-pinned"}
    assert response.usage is not None
    assert response.usage.raw_usage["passes"] == 2
    assert dict(response.extras) == {"passes": 2}


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        pytest.param("cloudflare-clef", {"passes": 2}, id="another-leaf"),
        pytest.param("cloudflare-clairic", {"passes": "2"}, id="not-an-integer"),
        pytest.param("cloudflare-clairic", {"passes": True}, id="a-bool"),
    ],
)
def test_an_upgrade_reads_no_extras_its_leaf_would_not_read_today(
    tmp_path: Path, name: str, raw: dict[str, Any]
) -> None:
    from idia.providers.recorded import upgrade_log

    usage = Usage(input_tokens=3, output_tokens=0, raw_usage=raw)
    log = _old_log(tmp_path, name, _bare(pinned=False, usage=usage))
    assert upgrade_log(log, requests=[_wire()]) == 0
    (response,) = _replayed(log)
    assert dict(response.extras) == {}


def test_a_callers_snapshot_reason_other_than_not_pinned_is_refused(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "someone-else", _bare(pinned=False))
    with pytest.raises(ValueError, match="reason is not-pinned"):
        upgrade_log(
            log, requests=[_wire()], reasons={"snapshot": "hidden", "versions": "not-applicable"}
        )


def test_a_response_recording_an_empty_request_id_is_refused(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "system-one", _bare())
    document = json.loads(log.read_text(encoding="utf-8"))
    (entries,) = document["calls"].values()
    entries[0]["response"]["request_id"] = ""
    log.write_text(json.dumps(document), encoding="utf-8")
    before = log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="empty request id"):
        upgrade_log(log, requests=[_wire()])
    assert log.read_text(encoding="utf-8") == before


def test_a_callers_not_sent_is_refused_since_only_the_dispatcher_gives_it(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "someone-else", _bare())
    with pytest.raises(ValueError, match="not-sent is the dispatcher's"):
        upgrade_log(
            log,
            requests=[_wire()],
            reasons={"usage.input_tokens": "not-sent", "versions": "not-applicable"},
        )


def test_a_response_whose_elicitation_has_a_reason_is_refused(tmp_path: Path) -> None:
    from idia.providers.recorded import _upgraded  # pyright: ignore[reportPrivateUsage]

    # A log cannot even decode one: this guards the rule behind that refusal.
    response = _response(elicitation=None, snapshot="v1", unreported={"elicitation": "hidden"})
    with pytest.raises(ValueError, match="always a value"):
        _upgraded(response, "system-one", None, tmp_path / "log.json")


def test_the_suite_reads_values_as_core_does_and_flags_an_empty_request_id() -> None:
    from idia.conformance import _base

    assert not hasattr(_base, "held")
    response = _response(
        request_id="",
        unreported={"snapshot": "not-pinned", "usage.input_tokens": "not-reported"}
        | {"usage.output_tokens": "not-reported"}
        | {"versions": "not-applicable", "prompt_sent": "not-reported"},
    )
    assert profile_problems(response) == ["the request id is empty; a leaf with none gives None"]


def _gliner_log(tmp_path: Path, versions: dict[str, object]) -> Path:
    log = _old_log(tmp_path, "gliner", _bare(usage=Usage(input_tokens=9, output_tokens=None)))
    document = json.loads(log.read_text(encoding="utf-8"))
    (entries,) = document["calls"].values()
    entries[0]["response"]["raw_body"] = {"json": {"output": {}, "versions": versions}}
    log.write_text(json.dumps(document), encoding="utf-8")
    return log


@pytest.mark.parametrize(
    "versions", [{}, {"huggingface_hub": "2.3.0"}], ids=["empty", "client-only"]
)
def test_a_gliner_log_recording_no_runtime_version_is_refused(
    tmp_path: Path, versions: dict[str, object]
) -> None:
    from idia.providers.recorded import upgrade_log

    log = _gliner_log(tmp_path, versions)
    before = log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="records no runtime version"):
        upgrade_log(log, requests=[_wire()])
    assert log.read_text(encoding="utf-8") == before


def test_a_leaf_with_rules_of_its_own_takes_no_reasons(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "system-one", _bare())
    with pytest.raises(ValueError, match="takes no reasons"):
        upgrade_log(log, requests=[_wire()], reasons={"snapshot": "not-pinned"})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_an_upgraded_log_keeps_its_mode_and_is_written_where_its_link_points(
    tmp_path: Path,
) -> None:
    from idia.providers.recorded import upgrade_log

    target = _old_log(tmp_path, "system-one", _bare())
    target.chmod(0o644)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert upgrade_log(link, requests=[_wire()]) == 0
    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert '"unreported"' in target.read_text(encoding="utf-8")


_ERROR: dict[str, Any] = {
    "message": "boom",
    "is_retryable": False,
    "retry_after": None,
    "status_code": 500,
    "request_id": None,
}


@pytest.mark.parametrize(
    "change",
    [
        "no-name",
        "calls-a-list",
        "features-unhashable",
        "raw-body-of-no-shape",
        "an-entry-not-a-mapping",
        "an-entry-of-both",
        "an-entry-of-neither",
        "retry-after-a-string",
        "retry-after-infinite",
        "retry-after-past-any-float",
        "is-retryable-a-string",
        "status-code-a-string",
        "base64-not-base64",
        "nan",
    ],
)
def test_a_log_that_is_not_a_recorded_leafs_is_refused_with_value_error(
    tmp_path: Path, change: str
) -> None:
    from idia.providers.recorded import upgrade_log

    for old in (True, False):
        directory = tmp_path / f"format-{2 if old else 3}"
        directory.mkdir()
        log = _old_log(directory, "system-one", _bare(), old=old)
        _broken(log, change, old=old)
        if old:
            with pytest.raises(ValueError, match="is not a recorded leaf's log"):
                upgrade_log(log, requests=[_wire()])
        else:
            with pytest.raises(ValueError, match="is not a recorded leaf's log"):
                Recorded(None, log, "none", capabilities=capabilities())


def _broken(log: Path, change: str, *, old: bool) -> None:
    """Break the log at ``log`` as ``change`` says."""
    document: dict[str, Any] = json.loads(log.read_text(encoding="utf-8"))
    (held,) = cast(dict[str, Any], document["calls"]).values()
    entries: list[Any] = held if old else held["entries"]
    if change == "no-name":
        del document["name"]
    elif change == "calls-a-list":
        document["calls"] = [1]
    elif change == "features-unhashable":
        document["enabled_features"] = [[1]]
    elif change == "raw-body-of-no-shape":
        entries[0]["response"]["raw_body"] = {"x": 1}
    elif change == "an-entry-not-a-mapping":
        entries.append(5)
    elif change == "an-entry-of-both":
        entries[0]["error"] = _ERROR
    elif change == "an-entry-of-neither":
        entries.append({})
    elif change == "base64-not-base64":
        entries[0]["response"]["raw_body"] = {"base64": "!!"}
    elif change == "nan":
        entries[0]["response"]["raw_body"] = {"json": {"score": float("nan")}}
    else:
        field, value = {
            "retry-after-a-string": ("retry_after", "x"),
            "retry-after-infinite": ("retry_after", float("inf")),
            "retry-after-past-any-float": ("retry_after", 10**400),
            "is-retryable-a-string": ("is_retryable", "yes"),
            "status-code-a-string": ("status_code", "a"),
        }[change]
        entries.append({"error": {**_ERROR, field: value}})
    log.write_text(json.dumps(document), encoding="utf-8")


def test_a_log_nested_past_the_parsers_depth_is_refused_with_value_error(
    tmp_path: Path,
) -> None:
    from idia.providers.recorded import upgrade_log

    log = tmp_path / "deep.json"
    log.write_text("[" * 100_000, encoding="utf-8")
    with pytest.raises(ValueError, match="nests too deep"):
        upgrade_log(log, requests=[_wire()])


def test_a_log_whose_errors_and_bodies_are_well_formed_loads(tmp_path: Path) -> None:
    log = _old_log(tmp_path, "system-one", _bare(), old=False)
    document: dict[str, Any] = json.loads(log.read_text(encoding="utf-8"))
    (call,) = cast(dict[str, Any], document["calls"]).values()
    entries: list[Any] = call["entries"]
    entries[0]["response"]["raw_body"] = {"base64": "aGk="}
    entries.append({"error": {**_ERROR, "retry_after": 1.5, "request_id": "r-1"}})
    log.write_text(json.dumps(document), encoding="utf-8")
    Recorded(None, log, "none", capabilities=capabilities())


def test_a_link_to_nothing_is_refused_and_nothing_is_created(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX links")
    log = tmp_path / "link.json"
    log.symlink_to(tmp_path / "elsewhere" / "log.json")
    leaf = Scripted(capabilities=capabilities(), script=[_bare()], name="system-one")
    with pytest.raises(ValueError, match="link to nothing"):
        asyncio.run(Recorded(leaf, log, "once").call(_wire(), Deadline.after(5)))
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_a_private_log_stays_private_through_a_rewrite(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "system-one", _bare())
    log.chmod(0o600)
    assert upgrade_log(log, requests=[_wire()]) == 0
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_a_call_whose_write_fails_is_not_kept_for_replay(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    nan = _bare(usage=Usage(input_tokens=1, output_tokens=1, raw_usage={"x": float("nan")}))
    leaf = Scripted(capabilities=capabilities(), script=[nan, _bare()], name="system-one")
    recorder = Recorded(leaf, log, "once")
    with pytest.raises(ValueError, match="Out of range float values"):
        asyncio.run(recorder.call(_wire(), Deadline.after(5)))
    asyncio.run(recorder.call(_wire(), Deadline.after(5)))
    assert len(leaf.calls) == 2  # the refused call was not replayed from memory


# ----------------------------------------------------------------------------------------
# Re-keying a format-2 log, keyed before the key kept key order
# ----------------------------------------------------------------------------------------


def _json_wire(value: Any, **options: Any) -> WireRequest:
    question = Question(key="q", kind="noul", instruction="Is it so?")
    return WireRequest(
        state=State(parts=(TextPart(text="t"), JsonPart(value=value))),
        questions=(question,),
        options=options,
    )


def test_an_old_log_is_re_keyed_for_a_request_with_one_order_only(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = tmp_path / "old.json"
    log.write_text(_OLD_LOG, encoding="utf-8")
    assert upgrade_log(log, requests=[_wire(), _wire()]) == 0
    document = json.loads(log.read_text(encoding="utf-8"))
    assert list(document) == ["format", "name", "model", "enabled_features", "calls"]
    assert document["format"] == 4
    assert list(document["calls"]) == [_KEY]
    assert document["calls"][_KEY]["request"] == json.loads(_REQUEST)
    replayed = _replay(log)
    assert replayed.answers == {"q": NoulAnswer(probability=0.25)}
    assert replayed.prompt_sent == "x"
    assert dict(replayed.unreported) == {"versions": "not-applicable"}

    # A nested JSON part of one key at every depth, and an option, re-key; errors move too.
    nested = _json_wire({"a": ({"b": {"c": 1}},)}, reference_date="2026-10-25")
    refused = ProviderError("refused", status_code=400, request_id="r-1")
    old = tmp_path / "nested.json"
    leaf = Scripted(
        capabilities=capabilities(), script=[_bare(), refused, _bare()], name="system-one"
    )
    recorder = Recorded(leaf, old, "once")
    for _ in range(3):
        with contextlib.suppress(ProviderError):
            asyncio.run(recorder.call(nested, Deadline.after(5)))
    _as_format_2(old)
    assert upgrade_log(old, requests=[nested]) == 0
    (entries,) = _calls(old).values()
    assert ["error" in entry for entry in entries] == [False, True, False]
    replayer = Recorded(None, old, "none", capabilities=capabilities())
    assert asyncio.run(replayer.call(nested, Deadline.after(5))).request_id == "system-one-1"
    with pytest.raises(ProviderError, match="refused"):
        asyncio.run(replayer.call(nested, Deadline.after(5)))
    assert asyncio.run(replayer.call(nested, Deadline.after(5))).request_id == "system-one-3"


@pytest.mark.parametrize(
    "wire",
    [
        pytest.param(_json_wire({"b": 1, "a": 2}), id="json-part"),
        pytest.param(_json_wire({"a": ({"y": 1, "x": 2},)}), id="nested-json-part"),
        pytest.param(_json_wire("v", one=1, two=2), id="options"),
        pytest.param(_json_wire("v", one={"y": 1, "x": 2}), id="nested-option"),
        pytest.param(
            WireRequest(
                state=State(parts=(TextPart(text="t"),)),
                questions=(Question(key="q", kind="noul", instruction={"q": "?", "e": "x"}),),
            ),
            id="structured-instruction",
        ),
    ],
)
def test_an_old_log_refuses_a_request_with_two_orders(tmp_path: Path, wire: WireRequest) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "system-one", _bare(), wire=wire)
    before = log.read_bytes()
    with pytest.raises(ValueError, match="cannot be re-keyed"):
        upgrade_log(log, requests=[_wire(), wire])
    assert log.read_bytes() == before
    assert upgrade_log(log, requests=[_wire()]) == 1


def test_an_entry_no_request_re_keys_is_a_replay_miss(tmp_path: Path) -> None:
    from idia.providers.recorded import ReplayMiss, upgrade_log

    log = tmp_path / "log.json"
    leaf = Scripted(capabilities=capabilities(), script=[_bare()] * 3, name="system-one")
    recorder = Recorded(leaf, log, "once")
    kept, dropped = _json_wire("kept"), _json_wire("dropped")
    for wire in (kept, dropped, dropped):
        asyncio.run(recorder.call(wire, Deadline.after(5)))
    _as_format_2(log)
    never = _json_wire("never recorded")
    assert upgrade_log(log, requests=[never, kept]) == 2
    replayer = Recorded(None, log, "none", capabilities=capabilities())
    asyncio.run(replayer.call(kept, Deadline.after(5)))
    with pytest.raises(ReplayMiss):
        asyncio.run(replayer.call(dropped, Deadline.after(5)))
    assert upgrade_log(_old_log(tmp_path, "scripted", _bare()), requests=[]) == 1


def test_an_upgrade_that_would_reorder_a_response_is_refused(tmp_path: Path) -> None:
    from idia.providers.recorded import (
        _expect_only_provenance_added,  # pyright: ignore[reportPrivateUsage]
    )

    stored: dict[str, Any] = {"response": {"raw_body": {"json": {"a": 1, "b": 2}}}}
    reordered: dict[str, Any] = {"response": {"raw_body": {"json": {"b": 2, "a": 1}}}}
    _expect_only_provenance_added(stored, stored, stored, tmp_path)
    with pytest.raises(ValueError, match="would change what it recorded"):
        _expect_only_provenance_added(stored, stored, reordered, tmp_path)
