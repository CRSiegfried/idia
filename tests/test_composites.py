"""Router and Fallback: the derived record, the up-front checks, and how outcomes join."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, cast

import pytest

from idia import (
    UNBOUNDED,
    WARNING_OVERFLOW_POLICY,
    WARNING_UNCHECKED_LIMIT,
    Capabilities,
    Deadline,
    Dispatcher,
    DoesNotFit,
    Fallback,
    ImagePart,
    Kind,
    Limit,
    MappedAnswer,
    NoulAnswer,
    Provider,
    ProviderError,
    Request,
    RequestFailed,
    Response,
    RetryPolicy,
    Rounding,
    Router,
    State,
    WireRequest,
    WireResponse,
)
from idia.providers.scripted import Item, Script, Scripted
from tests._support import (
    TICK,
    Shorten,
    TextChars,
    box,
    capabilities,
    choice,
    noul,
    request,
    score,
    text_state,
)

QUICK = RetryPolicy(jitter=0.0)


def _yes(key: str = "q") -> dict[str, MappedAnswer]:
    return {key: NoulAnswer(probability=0.7)}


def _red(key: str = "q") -> dict[str, MappedAnswer]:
    from idia import LabelAnswer

    return {key: LabelAnswer(probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.2)))}


def _answering(*keys: str) -> Callable[[WireRequest, Deadline], Item]:
    """A script answering whatever noul or choice questions it is sent."""

    def script(wire_request: WireRequest, deadline: Deadline) -> Item:
        answers: dict[str, MappedAnswer] = {}
        for question in wire_request.questions:
            answers.update(_red(question.key) if question.kind == "choice" else _yes(question.key))
        return answers

    return script


def _slow(seconds: float) -> Callable[[WireRequest, Deadline], Awaitable[Item]]:
    async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
        await asyncio.sleep(seconds)
        return _answering()(wire_request, deadline)

    return script


def _member(
    name: str, script: Script | None = None, *, retry: RetryPolicy = QUICK, **overrides: Any
) -> tuple[Scripted, Dispatcher]:
    leaf = Scripted(
        name=name,
        model=f"{name}-1",
        capabilities=capabilities(**overrides),
        script=script if script is not None else _answering(),
    )
    return leaf, Dispatcher(leaf, retry=retry)


def _failing(name: str, **overrides: Any) -> tuple[Scripted, Dispatcher]:
    return _member(name, [ProviderError(f"{name} broke", request_id=f"{name}-err")], **overrides)


def _run(provider: Provider, request_: Request) -> Response:
    return asyncio.run(provider.answer(request_))


def _fail(provider: Provider, request_: Request) -> Response:
    with pytest.raises(RequestFailed) as raised:
        _run(provider, request_)
    return raised.value.response


class Foreign:
    """A provider not of idia's making."""

    def __init__(self, record: Capabilities) -> None:
        self.capabilities = record

    async def answer(self, request: Request) -> Response:
        return Response(answers={}, outcomes=())


# ----------------------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------------------


def test_router_and_fallback_satisfy_provider() -> None:
    _, a = _member("a")
    assert isinstance(Router({"noul": a}), Provider)
    assert isinstance(Fallback([a]), Provider)


def test_router_member_plans_its_own_fit() -> None:
    leaf_a, a = _member("a", max_questions=1)
    leaf_b, b = _member("b")
    router = Router({"noul": a, "choice": b})
    response = _run(router, request(noul("x"), noul("y"), choice("z")))
    assert [call.wire_request.questions for call in leaf_a.calls] == [(noul("x"),), (noul("y"),)]
    assert [call.wire_request.questions for call in leaf_b.calls] == [(choice("z"),)]
    assert [outcome.keys for outcome in response.outcomes] == [("x",), ("y",), ("z",)]
    assert set(response.answers) == {"x", "y", "z"}


def test_router_response_has_one_outcome_per_member_call() -> None:
    _, a = _member("a", max_questions=1)
    _, b = _member("b")
    response = _run(Router({"noul": a, "choice": b}), request(noul("x"), choice("z"), noul("y")))
    assert [outcome.provider for outcome in response.outcomes] == ["a", "a", "b"]
    assert [outcome.model_requested for outcome in response.outcomes] == ["a-1", "a-1", "b-1"]


def test_router_member_order_is_the_mappings_not_the_requests() -> None:
    _, a = _member("a")
    _, b = _member("b")
    response = _run(Router({"noul": a, "choice": b}), request(choice("z"), noul("x")))
    assert [outcome.provider for outcome in response.outcomes] == ["a", "b"]
    assert [outcome.keys for outcome in response.outcomes] == [("x",), ("z",)]


class Recording(Dispatcher):
    """A dispatcher that keeps each request it is handed before answering it."""

    def __init__(self, leaf: Scripted) -> None:
        super().__init__(leaf, retry=QUICK)
        self.received: list[Request] = []

    async def answer(self, request: Request) -> Response:
        self.received.append(request)
        return await super().answer(request)


def test_router_member_gets_the_requests_state_deadline_and_options_unchanged() -> None:
    leaf_a, _ = _member("a")
    leaf_b, _ = _member("b")
    labels, nouls = Recording(leaf_a), Recording(leaf_b)
    parent = Request(
        state=text_state("the state"),
        questions=(noul("x"), choice("z"), noul("y")),
        deadline=Deadline.after(30),
        options={"z": 1, "a": {"y": ({"q": 2, "b": 3},), "c": 4}},
    )
    _run(Router({"choice": labels, "noul": nouls}), parent)
    ([to_labels], [to_nouls]) = (labels.received, nouls.received)
    assert [q.key for q in to_labels.questions] == ["z"]
    assert [q.key for q in to_nouls.questions] == ["x", "y"]
    for received in (to_labels, to_nouls):
        assert received.state is parent.state
        assert received.deadline is parent.deadline
        assert _ordered(received.options) == [
            ("z", 1),
            ("a", [("y", [[("q", 2), ("b", 3)]]), ("c", 4)]),
        ]


def test_router_member_exception_other_than_request_failed_propagates_bare() -> None:
    def buggy(wire_request: WireRequest, deadline: Deadline) -> Item:
        raise TypeError("a bug in the member")

    _, a = _member("a", buggy)
    _, b = _member("b")
    with pytest.raises(TypeError, match="a bug in the member"):
        _run(Router({"noul": a, "choice": b}), request(noul("x"), choice("z")))


def test_router_member_routing_two_kinds_gets_one_request() -> None:
    leaf_a, a = _member("a")
    _, b = _member("b")
    response = _run(Router({"noul": a, "choice": a, "score": b}), request(choice("c"), noul("n")))
    assert [call.wire_request.questions for call in leaf_a.calls] == [(choice("c"), noul("n"))]
    assert len(response.outcomes) == 1


def test_router_member_with_no_questions_is_not_called() -> None:
    _, a = _member("a")
    leaf_b, b = _member("b")
    _run(Router({"noul": a, "choice": b}), request(noul("x")))
    assert leaf_b.calls == []


def test_router_refuses_a_kind_its_member_lacks() -> None:
    _, a = _member("a", kinds=frozenset({"noul"}))
    with pytest.raises(ValueError, match="routed to") as raised:
        Router({"noul": a, "choice": a})
    assert str(raised.value) == (
        "kind 'choice' is routed to the Dispatcher over a (a-1), whose record lacks it "
        "(its kinds: noul)"
    )
    with pytest.raises(ValueError, match="routed to") as raised:
        Router({"choice": Router({"noul": a})})
    assert str(raised.value) == (
        "kind 'choice' is routed to a Router, whose record lacks it (its kinds: noul)"
    )


def test_router_unrouted_kind_fails_preflight_before_any_call() -> None:
    leaf_a, a = _member("a")
    with pytest.raises(DoesNotFit) as raised:
        _run(Router({"noul": a}), request(noul("x"), choice("z")))
    assert raised.value.check == "kinds"
    assert leaf_a.calls == []


def test_router_says_it_routes_a_missing_kind_to_no_member() -> None:
    _, a = _member("a")
    router = Router({"noul": a})
    with pytest.raises(DoesNotFit) as raised:
        _run(router, request(noul("x"), choice("z")))
    assert raised.value.observed == frozenset({"choice"})
    assert raised.value.off_features == frozenset()
    assert str(raised.value) == "kinds not supported: choice; this router routes it to no member"
    with pytest.raises(DoesNotFit) as raised:
        _run(router, request(choice("y"), score("z")))
    assert str(raised.value) == (
        "kinds not supported: choice, score; this router routes them to no member"
    )


def test_router_missing_modality_keeps_the_composite_clause() -> None:
    _, a = _member("a")
    image = State(parts=(ImagePart(data=b"x"),))
    with pytest.raises(DoesNotFit) as raised:
        _run(Router({"noul": a}), request(noul("x"), state=image))
    assert str(raised.value) == (
        "modalities not supported: image; no feature of this composite adds it; "
        "its members' own records may"
    )


def test_fallback_missing_kind_and_modality_keep_the_composite_clause() -> None:
    _, a = _member("a")
    _, b = _member("b")
    fallback = Fallback([a, b])
    with pytest.raises(DoesNotFit) as raised:
        _run(fallback, request(noul("x"), box("z")))
    assert str(raised.value) == (
        "kinds not supported: box; no feature of this composite adds it; "
        "its members' own records may"
    )
    image = State(parts=(ImagePart(data=b"x"),))
    with pytest.raises(DoesNotFit) as raised:
        _run(fallback, request(noul("x"), state=image))
    assert str(raised.value) == (
        "modalities not supported: image; no feature of this composite adds it; "
        "its members' own records may"
    )


def test_router_runs_its_members_concurrently() -> None:
    # Each member waits, up to a second, for the other to enter; run one after the other,
    # the first would leave before the second entered.
    events: list[tuple[str, str]] = []
    both_in = asyncio.Event()

    def meeting(name: str) -> Callable[[WireRequest, Deadline], Awaitable[Item]]:
        async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
            events.append(("in", name))
            if len(events) == 2:
                both_in.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(both_in.wait(), 1.0)
            events.append(("out", name))
            return _answering()(wire_request, deadline)

        return script

    _, a = _member("a", meeting("a"))
    _, b = _member("b", meeting("b"))
    _run(Router({"noul": a, "choice": b}), request(noul("x"), choice("z")))
    assert sorted(events[:2]) == [("in", "a"), ("in", "b")]
    assert sorted(events[2:]) == [("out", "a"), ("out", "b")]


def test_router_some_members_failing_returns_a_response() -> None:
    _, a = _failing("a")
    _, b = _member("b")
    response = _run(Router({"noul": a, "choice": b}), request(noul("x"), choice("z")))
    assert [outcome.error is None for outcome in response.outcomes] == [False, True]
    assert set(response.answers) == {"z"}


def test_router_all_members_failing_raises_request_failed() -> None:
    _, a = _failing("a")
    _, b = _failing("b")
    response = _fail(Router({"noul": a, "choice": b}), request(noul("x"), choice("z")))
    assert [outcome.provider for outcome in response.outcomes] == ["a", "b"]
    assert dict(response.answers) == {}


def test_router_response_warnings_are_the_members_concatenated() -> None:
    def shortening(name: str) -> Dispatcher:
        leaf = Scripted(
            name=name,
            capabilities=capabilities(max_input_tokens=50, max_question_tokens=1000),
            script=_answering(),
            estimator=TextChars(),
        )
        return Dispatcher(leaf, overflow=Shorten(keep=20))

    router = Router({"noul": shortening("a"), "choice": shortening("b")})
    response = _run(router, request(noul("x"), choice("z"), state=text_state("x" * 100)))
    assert [w.code for w in response.warnings] == [WARNING_OVERFLOW_POLICY] * 2


# ----------------------------------------------------------------------------------------
# Fallback
# ----------------------------------------------------------------------------------------


def test_fallback_member_gets_the_remaining_deadline_not_a_fresh_one() -> None:
    leaf_a, a = _failing("a")
    leaf_b, b = _member("b")
    request_ = request()
    response = _run(Fallback([a, b]), request_)
    assert leaf_a.calls[0].deadline == request_.deadline
    assert leaf_b.calls[0].deadline == request_.deadline
    assert leaf_b.calls[0].wire_request.questions == request_.questions
    assert response.outcomes[0].provider == "b"


def test_fallback_gives_every_member_the_same_request() -> None:
    leaf_a, _ = _failing("a")
    leaf_b, _ = _failing("b")
    leaf_c, _ = _member("c")
    members = [Recording(leaf_a), Recording(leaf_b), Recording(leaf_c)]
    parent = Request(
        state=text_state("the state"),
        questions=(noul("x"), noul("y")),
        deadline=Deadline.after(30),
        options={"z": 1, "a": 2},
    )
    _run(Fallback(members), parent)
    assert [len(member.received) for member in members] == [1, 1, 1]
    assert all(member.received[0] is parent for member in members)


# A prompt as a leaf might record it: keys out of sorted order at every depth.
PROMPT: Final = {"zeta": {"y": 1, "b": {"q": 2, "c": 3}}, "alpha": "first", "mid": {"z": 0, "a": 1}}


def _ordered(value: object) -> object:
    """``value``'s keys in their order at every depth, as nested items lists."""
    if isinstance(value, Mapping):
        items = cast(Mapping[str, object], value).items()
        return [(key, _ordered(item)) for key, item in items]
    if isinstance(value, list | tuple):
        return [_ordered(item) for item in cast(Sequence[object], value)]
    return value


class Prompting(Scripted):
    """A scripted leaf whose ``prompt_sent`` is ``PROMPT``, in its own order."""

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        response = await super().call(wire_request, deadline)
        return dataclasses.replace(response, prompt_sent=PROMPT)


def test_an_outcome_keeps_the_leafs_prompt_sent_order() -> None:
    leaf = Prompting(name="p", model="p-1", capabilities=capabilities(), script=_answering())
    (direct,) = _run(Dispatcher(leaf, retry=QUICK), request()).outcomes
    _, failing = _failing("a")
    (carried,) = _run(Fallback([failing, Dispatcher(leaf, retry=QUICK)]), request()).outcomes
    assert carried.failed_attempts
    for outcome in (direct, carried):
        assert _ordered(outcome.prompt_sent) == _ordered(PROMPT)


def test_fallback_failed_member_is_listed_on_the_answering_outcome() -> None:
    errors = [
        ProviderError("busy", is_retryable=True, retry_after=0.0, request_id=f"a-{n}")
        for n in range(2)
    ]
    _, a = _member("a", errors, retry=RetryPolicy(attempts=2, jitter=0.0))
    _, b = _member("b")
    response = _run(Fallback([a, b]), request())
    (outcome,) = response.outcomes
    assert outcome.provider == "b"
    assert outcome.error is None
    assert [attempt.error for attempt in outcome.failed_attempts] == errors
    assert {attempt.provider for attempt in outcome.failed_attempts} == {"a"}
    assert outcome.attempts == 3


def test_fallback_latency_includes_earlier_members() -> None:
    a_wait = 0.25
    _, a = _member(
        "a",
        [ProviderError("busy", is_retryable=True, retry_after=a_wait), ProviderError("broke")],
        retry=RetryPolicy(attempts=2, jitter=0.0),
    )
    _, b = _member("b")
    before = time.time()
    response = _run(Fallback([a, b]), request())
    (outcome,) = response.outcomes
    assert outcome.latency >= a_wait - TICK
    assert before <= outcome.started_at < before + a_wait / 2


def test_fallback_first_member_answering_returns_its_response_as_is() -> None:
    _, a = _member("a")
    _, b = _member("b")
    response = _run(Fallback([a, b]), request())
    (outcome,) = response.outcomes
    assert outcome.failed_attempts == ()
    assert outcome.provider == "a"


def test_fallback_all_members_failing_carries_every_members_outcomes() -> None:
    _, a = _failing("a")
    _, b = _failing("b")
    response = _fail(Fallback([a, b]), request())
    assert [outcome.provider for outcome in response.outcomes] == ["a", "b"]
    assert [outcome.error is not None for outcome in response.outcomes] == [True, True]
    assert dict(response.answers) == {}


def test_a_composite_has_no_public_plan() -> None:
    _, a = _member("a")
    _, b = _member("b")
    assert callable(a.plan)
    assert not hasattr(Router({"noul": a, "choice": b}), "plan")
    assert not hasattr(Fallback([a, b]), "plan")


def test_composite_refuses_a_foreign_provider_at_build() -> None:
    foreign = Foreign(capabilities())
    assert isinstance(foreign, Provider)
    _, a = _member("a")
    with pytest.raises(TypeError, match="Foreign"):
        Router({"noul": a, "choice": foreign})
    with pytest.raises(TypeError, match="Foreign"):
        Fallback([a, foreign])


def test_fallback_latency_covers_its_own_wire_request_plus_the_earlier_members() -> None:
    a_wait, b_call = 0.25, 0.02
    _, a = _member(
        "a",
        [ProviderError("busy", is_retryable=True, retry_after=a_wait), ProviderError("broke")],
        retry=RetryPolicy(attempts=2, jitter=0.0),
    )
    _, b = _member("b", _slow(b_call), max_questions=1)
    before = time.time()
    start = time.monotonic()
    response = _run(Fallback([a, b]), request(noul("x"), noul("y")))
    elapsed = time.monotonic() - start
    first, second = response.outcomes
    assert elapsed >= a_wait + 2 * b_call - 3 * TICK
    assert first.latency >= a_wait + b_call - 2 * TICK
    assert second.latency >= b_call - TICK
    assert second.latency < a_wait
    assert first.latency > second.latency
    assert before <= first.started_at <= second.started_at
    assert second.failed_attempts == ()


def test_fallback_carries_onto_the_first_of_two_outcomes_after_its_retry() -> None:
    leaf_a, a = _failing("a")
    busy = ProviderError("busy", is_retryable=True, retry_after=0.0, request_id="b-busy")
    leaf_b, b = _member("b", [busy, _yes("x"), _yes("y")], max_questions=1)
    before = time.time()
    response = _run(Fallback([a, b]), request(noul("x"), noul("y")))
    first, second = response.outcomes
    assert (first.keys, second.keys) == (("x",), ("y",))
    assert (first.error, second.error) == (None, None)
    assert [attempt.request_id for attempt in first.failed_attempts] == ["a-err", "b-busy"]
    assert [attempt.provider for attempt in first.failed_attempts] == ["a", "b"]
    assert first.attempts == 3
    assert before <= first.started_at <= first.failed_attempts[0].started_at
    assert second.failed_attempts == ()
    assert second.attempts == 1
    assert first.failed_attempts[-1].started_at <= second.started_at
    assert (len(leaf_a.calls), len(leaf_b.calls)) == (1, 3)


def test_fallback_carries_onto_the_answering_members_first_outcome_when_it_failed() -> None:
    _, a = _failing("a")
    broke = ProviderError("broke", request_id="b-broke")
    _, b = _member("b", [broke, _yes("y")], max_questions=1)
    before = time.time()
    response = _run(Fallback([a, b]), request(noul("x"), noul("y")))
    first, second = response.outcomes
    assert first.error is broke
    assert second.error is None
    assert set(response.answers) == {"y"}
    assert [attempt.request_id for attempt in first.failed_attempts] == ["a-err", "b-broke"]
    assert first.attempts == 2
    assert before <= first.started_at <= first.failed_attempts[0].started_at
    assert second.failed_attempts == ()


def test_router_with_no_members_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        Router({})


def test_fallback_with_no_members_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        Fallback([])


def test_fallback_member_with_a_question_cap_packs_its_own_plan() -> None:
    leaf_a, a = _member("a", max_questions=1)
    response = _run(Fallback([a]), request(noul("x"), noul("y")))
    assert len(leaf_a.calls) == 2
    assert len(response.outcomes) == 2


# ----------------------------------------------------------------------------------------
# The up-front checks
# ----------------------------------------------------------------------------------------


def test_composite_refuses_before_any_member_is_called() -> None:
    leaf_a, a = _member("a")
    leaf_b, b = _member("b")
    image = State(parts=(ImagePart(data=b"x"),))
    with pytest.raises(DoesNotFit) as raised:
        _run(Router({"noul": a, "choice": b}), request(noul("x"), choice("z"), state=image))
    assert raised.value.check == "modalities"
    with pytest.raises(DoesNotFit):
        _run(Fallback([a, b]), request(state=image))
    assert leaf_a.calls == []
    assert leaf_b.calls == []


def test_composite_checks_every_member_share_before_any_call() -> None:
    leaf_a, a = _member("a")
    leaf_b, b = _member("b", max_labels=2)
    with pytest.raises(DoesNotFit) as raised:
        _run(Router({"noul": a, "choice": b}), request(noul("x"), choice("z")))
    assert raised.value.check == "max_labels"
    assert leaf_a.calls == []
    assert leaf_b.calls == []
    with pytest.raises(DoesNotFit) as raised:
        _run(Fallback([a, b]), request(choice("z")))
    assert raised.value.check == "max_labels"
    assert leaf_a.calls == []
    assert leaf_b.calls == []


def test_composite_unchecked_limit_warning_comes_from_the_member_check() -> None:
    _, a = _member("a", max_question_tokens=None)
    _, b = _member("b", max_question_tokens=1000)
    router = Router({"noul": a, "choice": b})
    assert router.capabilities.max_question_tokens == 1000
    response = _run(router, request(noul("x"), choice("z")))
    first, second = response.outcomes
    assert [w.field for w in first.warnings if w.code == WARNING_UNCHECKED_LIMIT] == [
        "max_question_tokens"
    ]
    assert [w for w in second.warnings if w.code == WARNING_UNCHECKED_LIMIT] == []


# ----------------------------------------------------------------------------------------
# The derived record
# ----------------------------------------------------------------------------------------


def test_router_kinds_are_the_union_and_label_caps_come_from_the_routing_member() -> None:
    _, a = _member(
        "a",
        kinds=frozenset({"noul", "choice"}),
        max_labels=5,
        modalities=frozenset({"text"}),
        confidence={"choice": "top-probability", "noul": "margin"},
        feature_kinds={"long-context": frozenset({"box"})},
        feature_modalities={"long-context": frozenset({"image"})},
    )
    _, b = _member(
        "b",
        kinds=frozenset({"score", "choice"}),
        max_labels=50,
        max_levels=7,
        min_levels=3,
        modalities=frozenset({"text", "json"}),
        confidence={"choice": "entropy"},
    )
    record = Router({"noul": a, "score": b, "choice": b}).capabilities
    assert record.composite
    assert record.kinds == frozenset({"noul", "score", "choice"})
    assert record.modalities == frozenset({"text", "json"})
    assert (record.max_labels, record.max_levels, record.min_levels) == (50, 7, 3)
    assert dict(record.confidence) == {"choice": "entropy", "noul": "margin"}
    assert record.opt_in_features == frozenset() == record.enabled_features
    assert record.feature_kinds == {} == record.feature_modalities
    assert record.image_carriers == frozenset()


def test_fallback_capabilities_are_the_most_restrictive_of_its_members() -> None:
    _, a = _member(
        "a",
        kinds=frozenset({"noul", "choice", "score"}),
        key_patterns=("a.*",),
        max_labels=20,
        max_levels=9,
        min_levels=2,
        confidence={"choice": "top-probability", "noul": "margin"},
    )
    _, b = _member(
        "b",
        kinds=frozenset({"noul", "choice"}),
        modalities=frozenset({"text", "json"}),
        key_patterns=("b.*",),
        max_labels=10,
        max_levels=UNBOUNDED,
        min_levels=None,
        calibrated=False,
        snapshot_pinnable=False,
        returns_refusal=True,
        recounts_state_per_question=True,
        rounding=Rounding(probability_decimals=None, score_decimals=1),
        confidence={"choice": "entropy"},
    )
    record = Fallback([a, b]).capabilities
    assert record.composite
    assert record.kinds == frozenset({"noul", "choice"})
    assert record.modalities == frozenset({"text"})
    assert (record.max_labels, record.max_levels, record.min_levels) == (10, 9, 2)
    assert record.key_patterns == ("a.*", "b.*")
    assert (record.calibrated, record.snapshot_pinnable) == (False, False)
    assert (record.returns_refusal, record.recounts_state_per_question) == (True, True)
    assert record.rounding == Rounding(probability_decimals=2, score_decimals=1)
    assert dict(record.confidence) == {"choice": "mixed"}
    assert record.opt_in_features == frozenset() == record.enabled_features
    assert record.feature_kinds == {} == record.feature_modalities


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ((1000, None), 1000),
        ((None, None), None),
        ((UNBOUNDED, None), None),
        ((UNBOUNDED, UNBOUNDED), UNBOUNDED),
        ((UNBOUNDED, 5), 5),
        ((7, 5, 9), 5),
    ],
)
def test_composite_limit_is_the_minimum_of_known_member_limits_or_unknown_when_none_is_known(
    limits: tuple[Limit, ...], expected: Limit
) -> None:
    members = [_member(f"m{n}", max_input_tokens=limit)[1] for n, limit in enumerate(limits)]
    assert Fallback(members).capabilities.max_input_tokens == expected
    routed: dict[Kind, Provider] = dict(zip(("noul", "choice", "score"), members, strict=False))
    assert Router(routed).capabilities.max_input_tokens == expected


def test_composite_with_a_silently_truncating_member_and_an_unknown_limit_is_constructible() -> (
    None
):
    _, truncating = _member("a", truncates_silently=True, max_input_tokens=100)
    _, unknown = _member("b", truncates_silently=None, max_input_tokens=None)
    record = Fallback([truncating, unknown]).capabilities
    assert record.truncates_silently is True
    assert record.max_input_tokens == 100
    _, unknown_only = _member("c", truncates_silently=None, max_input_tokens=None)
    _, honest = _member("d", truncates_silently=False, max_input_tokens=None)
    assert Fallback([unknown_only, honest]).capabilities.truncates_silently is None
    assert Fallback([honest, honest]).capabilities.truncates_silently is False
    assert Fallback([honest, honest]).capabilities.max_input_tokens is None


def _imaging(name: str, **fields: Any) -> tuple[Scripted, Dispatcher]:
    return _member(
        name,
        modalities=frozenset({"text", "image"}),
        image_carriers=frozenset({"parts"}),
        max_images=UNBOUNDED,
        max_images_per_question=UNBOUNDED,
        **fields,
    )


def test_composite_image_fields_are_the_most_restrictive_known_member_values() -> None:
    _, a = _imaging(
        "a",
        max_image_bytes=4_000,
        max_total_image_bytes=None,
        min_image_side=280,
        image_media_types=frozenset({"image/png", "image/jpeg"}),
    )
    _, b = _imaging(
        "b",
        max_image_bytes=UNBOUNDED,
        max_total_image_bytes=8_000,
        min_image_side=None,
        image_media_types=frozenset({"image/png", "image/webp"}),
    )
    _, unknowing = _imaging(
        "c",
        max_image_bytes=None,
        max_total_image_bytes=None,
        min_image_side=None,
        image_media_types=None,
    )
    # A member without images, its 0, 0, 0 and empty set, bears on none of the four.
    _, text = _member(
        "t", max_image_bytes=0, max_total_image_bytes=0, image_media_types=frozenset()
    )
    for composite in (
        Router({"noul": a, "choice": b, "score": text}),
        Fallback([a, b, unknowing, a]),
    ):
        record = composite.capabilities
        assert record.max_image_bytes == 4_000
        assert record.max_total_image_bytes == 8_000
        assert record.min_image_side == 280
        assert record.image_media_types == frozenset({"image/png"})
    none_known = Router({"noul": unknowing, "choice": text}).capabilities
    assert (none_known.max_image_bytes, none_known.max_total_image_bytes) == (None, None)
    assert (none_known.min_image_side, none_known.image_media_types) == (None, None)
    # A fallback whose intersected modalities lack image still shows its image member's
    # values, beside no carrier.
    mixed = Fallback([a, text]).capabilities
    assert "image" not in mixed.modalities
    assert (mixed.max_image_bytes, mixed.image_carriers) == (4_000, frozenset())
    no_images = Fallback([text, text]).capabilities
    assert (no_images.max_image_bytes, no_images.image_media_types) == (None, None)
    # Disjoint known sets derive the empty set, which the composite's own check never reads.
    _, webp = _imaging("w", image_media_types=frozenset({"image/webp"}))
    disjoint = Router({"noul": a, "choice": webp}).capabilities
    assert disjoint.image_media_types == frozenset()


def test_a_request_fit_refuses_makes_no_call_and_counts_no_attempt() -> None:
    leaf_a, a = _imaging("a", max_image_bytes=100)
    leaf_b, b = _imaging("b")
    big = State(parts=(ImagePart(data=b"\x89PNG\r\n\x1a\n" + bytes(200)),))
    for provider in (a, Router({"noul": a, "choice": b}), Fallback([a, b])):
        with pytest.raises(DoesNotFit) as raised:
            asyncio.run(provider.answer(request(noul("n"), choice("c"), state=big)))
        assert raised.value.check == "max_image_bytes"
        assert leaf_a.calls == []
        assert leaf_b.calls == []


def test_composite_image_carriers_follow_the_derived_modalities() -> None:
    _, a = _member(
        "a",
        modalities=frozenset({"text", "image"}),
        image_carriers=frozenset({"parts"}),
        max_images=UNBOUNDED,
        max_images_per_question=UNBOUNDED,
    )
    _, b = _member("b")
    assert Router({"noul": a, "choice": b}).capabilities.image_carriers == frozenset({"parts"})
    assert Fallback([a, b]).capabilities.image_carriers == frozenset()
    assert Fallback([a, a]).capabilities.image_carriers == frozenset({"parts"})


# ----------------------------------------------------------------------------------------
# Closing
# ----------------------------------------------------------------------------------------


class _Closable(Scripted):
    """A scripted leaf whose ``aclose`` appends its name to a shared log."""

    def __init__(self, name: str, log: list[str], *, broken: bool = False) -> None:
        super().__init__(name=name, capabilities=capabilities(), script=_answering())
        self._log = log
        self._broken = broken

    async def aclose(self) -> None:
        self._log.append(self.name)
        if self._broken:
            raise RuntimeError(f"{self.name} would not close")


def test_router_aclose_closes_each_member_once() -> None:
    log: list[str] = []
    a = Dispatcher(_Closable("a", log))
    b = Dispatcher(_Closable("b", log))
    router = Router({"noul": a, "choice": b, "score": a})

    async def use() -> None:
        async with router:
            pass

    asyncio.run(use())
    assert log == ["a", "b"]
    asyncio.run(router.aclose())
    assert log == ["a", "b", "a", "b"]


def test_fallback_aclose_closes_every_member_even_if_one_raises() -> None:
    log: list[str] = []
    members = [
        Dispatcher(_Closable("a", log, broken=True)),
        Dispatcher(_Closable("b", log, broken=True)),
        Dispatcher(_Closable("c", log)),
    ]
    with pytest.raises(RuntimeError, match="a would not close"):
        asyncio.run(Fallback(members).aclose())
    assert log == ["a", "b", "c"]

    async def nested() -> None:
        async with Fallback([Router({"noul": members[2]}), members[2]]):
            pass

    asyncio.run(nested())
    assert log == ["a", "b", "c", "c", "c"]


def test_a_members_aclose_raising_cancelled_error_stops_the_loop_at_once() -> None:
    log: list[str] = []

    class _Cancelling(_Closable):
        async def aclose(self) -> None:
            log.append(self.name)
            raise asyncio.CancelledError

    members = [Dispatcher(_Cancelling("a", log)), Dispatcher(_Closable("b", log))]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(Fallback(members).aclose())
    assert log == ["a"]
