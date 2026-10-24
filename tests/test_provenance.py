"""The provenance profile: each detail a value or a reason code, the details only claimed,
the runtime's and the client libraries' versions, and the outcome's rounding.

The shapes refuse a detail or reason outside the closed sets and an inconsistent pair; the
dispatcher gives every failed outcome ``not-sent`` and idia's version; the recorded leaf's
log leaves the new fields out when empty, so a log written before them replays with the
defaults, and one that fills them replays them unchanged.
"""

from __future__ import annotations

import asyncio
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
    NoulAnswer,
    ProviderError,
    Question,
    RequestFailed,
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
from tests._support import capabilities, noul, request

_ROUNDING = Rounding(probability_decimals=2, score_decimals=2)


def _response(**fields: Any) -> WireResponse:
    base: dict[str, Any] = {
        "answers": {},
        "elicitation": "passthrough",
        "calibrated": False,
        "model_reported": "m",
    }
    return WireResponse(**{**base, **fields})


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
def test_a_wire_response_refuses_an_inconsistent_profile(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _response(**fields)


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
def test_a_wire_response_refuses_a_provenance_field_of_the_wrong_shape(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        _response(**fields)


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
