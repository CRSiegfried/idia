"""The scripted leaf: a leaf answered from a test's script, with a call log."""

from __future__ import annotations

import asyncio
import datetime
import json
from collections.abc import Mapping

import pytest

import idia
from idia import (
    Box,
    BoxAnswer,
    Deadline,
    ExtractionAnswer,
    Fault,
    ImagePart,
    JsonPart,
    LabelAnswer,
    Leaf,
    MappedAnswer,
    NoulAnswer,
    Provider,
    ProviderError,
    Refusal,
    Rule,
    RuleFailure,
    State,
    TextPart,
    Usage,
    Warning,
    WireRequest,
    WireResponse,
)
from idia.core import WARNING_NO_SNAPSHOT
from idia.providers import _REGISTRY  # pyright: ignore[reportPrivateUsage]
from idia.providers._canon import encode
from idia.providers.scripted import (
    Reply,
    Scripted,
    ScriptedCall,
    ScriptExhausted,
)
from tests._support import FixedEstimator, capabilities, choice, extraction, noul, text_state

DEADLINE = Deadline.after(60.0)


def _wire(*questions: idia.Question, state: State | None = None) -> WireRequest:
    return WireRequest(state=state or text_state(), questions=questions or (noul(),))


def _call(leaf: Scripted, wire_request: WireRequest) -> WireResponse:
    return asyncio.run(leaf.call(wire_request, DEADLINE))


def _yes(p: float = 0.7) -> dict[str, MappedAnswer]:
    return {"q": NoulAnswer(probability=p)}


def test_scripted_leaf_is_a_leaf_and_not_a_provider() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[])
    assert isinstance(leaf, Leaf)
    assert not isinstance(leaf, Provider)
    assert leaf.name == "scripted"
    assert leaf.model == "scripted-1"
    assert leaf.estimator is not None
    assert "scripted" not in _REGISTRY
    assert "Scripted" not in idia.__all__


def test_scripted_leaf_takes_its_name_model_and_estimator_from_the_test() -> None:
    estimator = FixedEstimator(tokens=3)
    leaf = Scripted(
        name="dm", model="dm-2", capabilities=capabilities(), estimator=estimator, script=[]
    )
    assert (leaf.name, leaf.model, leaf.estimator) == ("dm", "dm-2", estimator)


def test_scripted_leaf_logs_each_wire_request_in_order() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[_yes(0.1), _yes(0.2)])
    first, second = _wire(noul()), _wire(noul(), state=text_state("other"))
    _call(leaf, first)
    _call(leaf, second)
    assert leaf.calls == [
        ScriptedCall(wire_request=first, deadline=DEADLINE),
        ScriptedCall(wire_request=second, deadline=DEADLINE),
    ]


def test_scripted_leaf_logs_a_call_before_raising() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[ProviderError("boom")])
    wire_request = _wire()
    with pytest.raises(ProviderError):
        _call(leaf, wire_request)
    assert leaf.calls == [ScriptedCall(wire_request=wire_request, deadline=DEADLINE)]


def test_scripted_leaf_raises_a_scripted_provider_error() -> None:
    error = ProviderError("rate limited", is_retryable=True, retry_after=1.5, status_code=429)
    leaf = Scripted(capabilities=capabilities(), script=[error])
    with pytest.raises(ProviderError) as raised:
        _call(leaf, _wire())
    assert raised.value is error


def test_scripted_leaf_validates_scripted_answers_with_its_rounding() -> None:
    off = LabelAnswer(probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.1)))
    near = LabelAnswer(probabilities=(("red", 0.34), ("green", 0.33), ("blue", 0.32)))
    leaf = Scripted(capabilities=capabilities(), script=[{"a": off, "b": near}])
    response = _call(leaf, _wire(choice("a"), choice("b")))
    fault = response.answers["a"]
    assert isinstance(fault, Fault)
    assert [failure.rule for failure in fault.failures] == [Rule.sum_tolerance]
    assert fault.mapped == off
    # Within the two-decimal tolerance, so it stands as scripted.
    assert response.answers["b"] is near


def test_scripted_fault_stands_and_siblings_are_validated() -> None:
    scripted_fault = Fault(
        key="a",
        failures=(RuleFailure(rule=Rule.candidate_shape, observed="estimate", bound=None),),
        mapped=None,
    )
    leaf = Scripted(
        capabilities=capabilities(),
        script=[{"a": scripted_fault, "b": NoulAnswer(probability=1.5), "c": Refusal()}],
    )
    response = _call(leaf, _wire(noul("a"), noul("b"), noul("c"), noul("d")))
    assert response.answers["a"] is scripted_fault
    sibling = response.answers["b"]
    assert isinstance(sibling, Fault)
    assert sibling.failures == (RuleFailure(rule=Rule.probability_range, observed=1.5, bound=1.0),)
    assert response.answers["c"] == Refusal()
    missing = response.answers["d"]
    assert isinstance(missing, Fault)
    assert missing.failures[0].rule is Rule.missing_key


def test_scripted_leaf_fills_provenance() -> None:
    leaf = Scripted(capabilities=capabilities(calibrated=False), script=[_yes()])
    wire_request = _wire()
    response = _call(leaf, wire_request)
    assert response.answers == {"q": NoulAnswer(probability=0.7)}
    assert response.model_reported == "scripted-1"
    assert response.snapshot == "scripted-1"
    assert response.elicitation == "passthrough"
    assert response.calibrated is False
    assert isinstance(response.prompt_sent, str)
    assert json.loads(response.prompt_sent)["questions"][0]["key"] == "q"
    assert response.usage is not None
    assert (
        response.usage.input_tokens
        == leaf.estimator.estimate(wire_request.state, wire_request.questions).tokens
    )
    assert response.usage.output_tokens is not None
    assert response.usage.output_tokens > 0
    assert response.usage.raw_usage == {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    assert response.request_id == "scripted-1"
    assert response.warnings == ()
    assert isinstance(response.raw_body, Mapping)
    assert "answers" in response.raw_body


def test_scripted_reply_varies_the_provenance_and_warns_of_an_extra_key() -> None:
    usage = Usage(input_tokens=5, output_tokens=1)
    note = Warning(code="scripted", message="from the test")
    leaf = Scripted(
        capabilities=capabilities(),
        script=[
            Reply(answers={**_yes(), "extra": _yes()["q"]}, model_reported="dm-2025-01"),
            Reply(answers=_yes(), model_reported="dm-2025-02", usage=usage, warnings=(note,)),
        ],
    )
    first = _call(leaf, _wire())
    second = _call(leaf, _wire())
    assert (first.snapshot, second.snapshot) == ("dm-2025-01", "dm-2025-02")
    assert [w.code for w in first.warnings] == ["extra-answer-key"]
    assert second.usage is usage
    assert second.warnings == (note,)
    assert (first.request_id, second.request_id) == ("scripted-1", "scripted-2")


def test_scripted_snapshot_is_none_with_a_warning_when_not_pinnable() -> None:
    leaf = Scripted(capabilities=capabilities(snapshot_pinnable=False), script=[_yes()])
    response = _call(leaf, _wire())
    assert response.snapshot is None
    assert response.model_reported == "scripted-1"
    assert [(w.code, w.field) for w in response.warnings] == [(WARNING_NO_SNAPSHOT, "snapshot")]


def test_exhausted_script_raises_a_test_error_not_a_provider_error() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[_yes()])
    _call(leaf, _wire())
    with pytest.raises(ScriptExhausted) as raised:
        _call(leaf, _wire())
    assert isinstance(raised.value, AssertionError)
    assert not isinstance(raised.value, ProviderError)
    assert len(leaf.calls) == 2


def test_scripted_raw_body_is_deterministic_json() -> None:
    answers: dict[str, MappedAnswer] = {
        "b": LabelAnswer(
            probabilities=(("red", 0.2), ("green", 0.8)),
            chosen="green",
            derived=frozenset({"y", "x"}),
        ),
        "a": ExtractionAnswer(value=datetime.date(2026, 1, 2), p_null=0.1),
        "c": NoulAnswer(probability=float("nan")),
        "d": BoxAnswer(region=Box(top=0.1, left=0.1, bottom=0.5, right=0.5)),
        "e": ExtractionAnswer(value=datetime.timedelta(minutes=1), p_null=0.0),
        "f": Fault(
            key="f",
            failures=(RuleFailure(rule=Rule.candidate_shape, observed="forecast", bound=None),),
            mapped=None,
        ),
    }

    def body() -> str:
        leaf = Scripted(capabilities=capabilities(), script=[dict(answers)])
        return json.dumps(_call(leaf, _wire(noul("a"))).raw_body, sort_keys=True)

    first = body()
    assert first == body()
    decoded = json.loads(first)
    assert list(decoded["answers"]) == ["a", "b", "c", "d", "e", "f"]
    assert decoded["answers"]["a"] == {
        "type": "ExtractionAnswer",
        "value": {"type": "date", "value": "2026-01-02"},
        "guess": None,
        "confidence": None,
        "p_null": 0.1,
        "span": None,
    }
    assert decoded["answers"]["b"]["derived"] == ["x", "y"]
    assert decoded["answers"]["b"]["probabilities"] == [["red", 0.2], ["green", 0.8]]
    assert decoded["answers"]["c"]["probability"] == {"type": "float", "value": "nan"}
    assert decoded["answers"]["e"]["value"] == {"type": "timedelta", "value": 60.0}
    assert decoded["answers"]["f"]["failures"][0]["rule"] == "candidate_shape"


def test_prompt_sent_renders_every_part_deterministically() -> None:
    state = State(
        parts=(
            TextPart(text="hello"),
            JsonPart(value={"z": 1, "a": (True, None)}),
            ImagePart(data=b"\x00\x01", media_type="image/png"),
            ImagePart(data=b"cat"),
        )
    )
    wire_request = _wire(choice("a"), extraction("b", value_type="number"), state=state)

    def prompt() -> object:
        leaf = Scripted(capabilities=capabilities(), script=[{}])
        return _call(leaf, wire_request).prompt_sent

    sent = prompt()
    assert sent == prompt()
    assert isinstance(sent, str)
    parts = json.loads(sent)["state"]["parts"]
    assert parts[1]["value"] == {"a": [True, None], "z": 1}
    assert parts[2]["data"] == "AAE="
    assert parts[3] == {"type": "ImagePart", "data": "Y2F0", "media_type": None}


def test_rendering_refuses_a_type_it_cannot_render() -> None:
    with pytest.raises(TypeError, match="cannot render a object"):
        encode(object())


def test_default_estimator_counts_characters() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[])
    short = leaf.estimator.estimate(text_state("a"), (noul(),))
    long = leaf.estimator.estimate(text_state("a" * 400), (noul(),))
    assert long.tokens - short.tokens == 100
    assert short.margin == 0.0


def test_callable_script_receives_the_wire_request_and_deadline() -> None:
    seen: list[tuple[WireRequest, Deadline]] = []

    def answer(wire_request: WireRequest, deadline: Deadline) -> Mapping[str, MappedAnswer]:
        seen.append((wire_request, deadline))
        return {question.key: NoulAnswer(probability=0.5) for question in wire_request.questions}

    async def answer_async(wire_request: WireRequest, deadline: Deadline) -> Reply:
        seen.append((wire_request, deadline))
        return Reply(answers=answer(wire_request, deadline))

    wire_request = _wire(noul("x"), noul("y"))
    for script in (answer, answer_async):
        seen.clear()
        leaf = Scripted(capabilities=capabilities(), script=script)
        response = _call(leaf, wire_request)
        assert set(response.answers) == {"x", "y"}
        assert seen[0] == (wire_request, DEADLINE)
        assert len(leaf.calls) == 1

    def fail(wire_request: WireRequest, deadline: Deadline) -> ProviderError:
        return ProviderError("down", is_retryable=True, status_code=503)

    with pytest.raises(ProviderError):
        _call(Scripted(capabilities=capabilities(), script=fail), wire_request)


def test_scripted_leaf_sends_exactly_one_wire_request_per_call() -> None:
    rate_limited = ProviderError("slow down", is_retryable=True, retry_after=0.0, status_code=429)
    leaf = Scripted(capabilities=capabilities(), script=[rate_limited, _yes()])
    wire_request = _wire()
    with pytest.raises(ProviderError) as raised:
        _call(leaf, wire_request)
    assert raised.value.is_retryable
    assert len(leaf.calls) == 1
    response = _call(leaf, wire_request)
    assert len(leaf.calls) == 2
    assert response.answers["q"] == NoulAnswer(probability=0.7)


def test_pinnable_record_with_an_alias_reply_reports_no_snapshot_with_a_warning() -> None:
    leaf = Scripted(
        capabilities=capabilities(snapshot_pinnable=True),
        script=[Reply(answers=_yes(), model_reported="dm-latest", pinned=False)],
    )
    response = _call(leaf, _wire())
    assert response.snapshot is None
    assert response.model_reported == "dm-latest"
    assert [(w.code, w.field) for w in response.warnings] == [(WARNING_NO_SNAPSHOT, "snapshot")]


def test_async_script_that_stalls_past_the_deadline_raises_a_retryable_provider_error() -> None:
    async def stall(wire_request: WireRequest, deadline: Deadline) -> Reply:
        await asyncio.sleep(1)
        return Reply(answers=_yes())

    leaf = Scripted(capabilities=capabilities(), script=stall)
    wire_request = _wire()
    deadline = Deadline.after(0.01)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(wire_request, deadline))
    assert raised.value.is_retryable
    assert raised.value.retry_after is None
    assert raised.value.status_code is None
    assert raised.value.request_id == "scripted-1"
    assert leaf.calls == [ScriptedCall(wire_request=wire_request, deadline=deadline)]


def test_overlapping_calls_get_distinct_request_ids() -> None:
    async def answer(wire_request: WireRequest, deadline: Deadline) -> Reply:
        await asyncio.sleep(0)
        return Reply(answers=_yes())

    leaf = Scripted(capabilities=capabilities(), script=answer)

    async def both() -> tuple[WireResponse, WireResponse]:
        return await asyncio.gather(leaf.call(_wire(), DEADLINE), leaf.call(_wire(), DEADLINE))

    responses = asyncio.run(both())
    assert sorted(r.request_id or "" for r in responses) == ["scripted-1", "scripted-2"]


def test_a_script_item_of_the_wrong_type_raises_a_type_error() -> None:
    def wrong(wire_request: WireRequest, deadline: Deadline) -> object:
        return 0.7

    leaf = Scripted(capabilities=capabilities(), script=wrong)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="script gave a float"):
        _call(leaf, _wire())
