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
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest

import idia
from idia import (
    DETAILS,
    REASONS,
    Deadline,
    Dispatcher,
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
        "usage.passes",
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
    assert dict(outcome.unreported) == {"usage.passes": "not-reported"}
    assert outcome.rounding == record.rounding


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
    assert dict(outcome.unreported) == {"usage.passes": "not-reported"}


# ----------------------------------------------------------------------------------------
# The recorded leaf's log
# ----------------------------------------------------------------------------------------

_KEY = "b0a43f8b7956e51da539cb6b677dbfa06262286a65bd47ba6dc7f544cbdb6788"
_OLD_LOG = (
    '{"calls":{"' + _KEY + '":[{"response":{"answers":{"q":{"confidence":null,"derived":[],'
    '"off_label_mass":null,"probability":0.25,"type":"NoulAnswer"}},"calibrated":true,'
    '"elicitation":"passthrough","model_reported":"scripted-1","prompt_sent":"x",'
    '"raw_body":{"json":{"answers":{}}},"request_id":"scripted-1","snapshot":"scripted-1",'
    '"type":"WireResponse","usage":{"input_tokens":1,"output_tokens":0,"passes":1,'
    '"raw_usage":{},"type":"Usage"},"warnings":[]}}]},"enabled_features":[],"format":2,'
    '"model":"scripted-1","name":"scripted"}'
)
"""A format-2 log written before the provenance fields existed: none of them is in it."""


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
    log.write_text(_OLD_LOG, encoding="utf-8")
    replayed = _replay(log)
    assert replayed.answers == {"q": NoulAnswer(probability=0.25)}
    assert dict(replayed.versions) == {}
    assert dict(replayed.client_versions) == {}
    assert dict(replayed.unreported) == {}
    assert replayed.claimed == frozenset()


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
    (entry,) = next(iter(stored["calls"].values()))
    assert {"versions", "client_versions", "unreported", "claimed"} <= set(entry["response"])
    assert stored["format"] == 2
    replayed = _replay(log)
    for name in ("versions", "client_versions", "unreported", "claimed"):
        assert getattr(replayed, name) == getattr(recorded, name), name
    assert dict(replayed.unreported) == {
        "snapshot": "not-pinned",
        "usage.passes": "not-reported",
    }


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

    full = Usage(input_tokens=1, output_tokens=0, passes=2)
    assert wire_unreported(snapshot="v1", usage=full, request_id="r") == {
        "versions": "not-applicable"
    }
    assert wire_unreported(snapshot=None, usage=None, request_id=None) == {
        "versions": "not-applicable",
        "snapshot": "not-pinned",
        "usage.input_tokens": "not-reported",
        "usage.output_tokens": "not-reported",
        "usage.passes": "not-reported",
        "request_id": "not-reported",
    }


def test_client_versions_never_gives_idia_which_dispatch_adds() -> None:
    pytest.importorskip("httpx2")
    from idia.providers.http import client_versions

    assert "idia" not in client_versions("idia", "httpx2")


# ----------------------------------------------------------------------------------------
# Upgrading a log written before the profile
# ----------------------------------------------------------------------------------------


def _old_log(tmp_path: Path, name: str, *items: Any) -> Path:
    """A log as idia wrote it before the profile: recorded with every provenance field empty,
    which the log leaves out."""
    log = tmp_path / f"{name}.json"
    leaf = Scripted(capabilities=capabilities(), script=list(items), name=name, model="m-1")
    recorder = Recorded(leaf, log, "once")
    for _ in items:
        with contextlib.suppress(ProviderError):
            asyncio.run(recorder.call(_wire(), Deadline.after(5)))
        recorder = Recorded(leaf, log, "new")
    assert all(k not in log.read_text() for k in ('"unreported"', '"versions"', '"claimed"'))
    return log


def _replayed(log: Path) -> list[WireResponse]:
    document = json.loads(log.read_text(encoding="utf-8"))
    return [
        _replay_entry(entry)
        for entries in document["calls"].values()
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
    from idia.conformance._base import profile_problems  # pyright: ignore[reportPrivateUsage]
    from idia.providers.recorded import upgrade_log

    usage = Usage(input_tokens=3, output_tokens=0, raw_usage={"input_tokens": 3})
    log = _old_log(tmp_path, "cloudflare-clairic", _bare(pinned=False, usage=usage))
    before = log.read_text(encoding="utf-8")
    assert upgrade_log(log) == 1
    (response,) = _replayed(log)
    assert dict(response.unreported) == {
        "versions": "not-applicable",
        "snapshot": "not-pinned",
        "usage.passes": "not-reported",
    }
    assert dict(response.client_versions) == {}
    assert response.claimed == frozenset()
    assert profile_problems(response) == []
    assert response.answers == {"q": NoulAnswer(probability=0.25)}
    assert json.loads(log.read_text())["format"] == 2
    after = log.read_text(encoding="utf-8")
    assert upgrade_log(log) == 0
    assert log.read_text(encoding="utf-8") == after != before


def test_upgrading_keeps_every_recorded_value_and_leaves_errors_alone(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    refused = ProviderError("refused", status_code=400, request_id="r-1")
    log = _old_log(tmp_path, "system-one", _bare(), refused)
    old = json.loads(log.read_text(encoding="utf-8"))
    upgrade_log(log)
    new = json.loads(log.read_text(encoding="utf-8"))
    for key, entries in old["calls"].items():
        for was, now in zip(entries, new["calls"][key], strict=True):
            if "error" in was:
                assert now == was
                continue
            added = {"versions", "client_versions", "unreported", "claimed"}
            assert {k: v for k, v in now["response"].items() if k not in added} == was["response"]
    assert {k: v for k, v in new.items() if k != "calls"} == {
        k: v for k, v in old.items() if k != "calls"
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
    assert upgrade_log(log) == 1
    (response,) = _replayed(log)
    assert dict(response.versions) == {
        "python": "3.11.13",
        "gliner2": "2.1.0",
        "torch": "2.14.1",
        "transformers": "5.20.2",
    }
    assert dict(response.client_versions) == {"huggingface_hub": "2.3.0"}
    assert dict(response.unreported) == {
        "usage.output_tokens": "not-applicable",
        "usage.passes": "not-reported",
    }
    assert response.claimed == {"model_reported", "snapshot"}
    (entries,) = json.loads(log.read_text(encoding="utf-8"))["calls"].values()
    assert entries[0]["response"]["raw_body"] == raw_body


@pytest.mark.parametrize(
    ("name", "reply", "message"),
    [
        pytest.param("someone-else", _bare(), "no upgrade rules for the leaf", id="unknown"),
        pytest.param("gliner", _bare(), "records no versions", id="gliner-no-versions"),
    ],
)
def test_upgrading_refuses_a_log_it_cannot_upgrade_honestly_and_writes_nothing(
    tmp_path: Path, name: str, reply: Reply, message: str
) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, name, reply)
    before = log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        upgrade_log(log)
    assert log.read_text(encoding="utf-8") == before


def test_an_unknown_leaf_is_upgraded_with_the_reasons_its_caller_gives(tmp_path: Path) -> None:
    from idia.providers.recorded import upgrade_log

    log = _old_log(tmp_path, "someone-else", _bare(usage=Usage(input_tokens=1, output_tokens=1)))
    with pytest.raises(ValueError, match=r"leave out usage\.passes"):
        upgrade_log(log, reasons={"versions": "not-applicable"})
    assert upgrade_log(log, reasons={"versions": "not-applicable", "usage.passes": "hidden"}) == 1
    (response,) = _replayed(log)
    assert dict(response.unreported) == {"versions": "not-applicable", "usage.passes": "hidden"}


def test_a_pass_count_only_the_raw_usage_holds_is_not_exposed_and_stays_as_recorded(
    tmp_path: Path,
) -> None:
    from idia.providers.recorded import upgrade_log

    usage = Usage(input_tokens=3, output_tokens=0, raw_usage={"input_tokens": 3, "passes": 2})
    log = _old_log(tmp_path, "cloudflare-clairic", _bare(pinned=False, usage=usage))
    assert upgrade_log(log) == 1
    (response,) = _replayed(log)
    assert response.unreported["usage.passes"] == "not-exposed"
    assert response.usage is not None
    assert response.usage.passes is None
    assert response.usage.raw_usage["passes"] == 2
