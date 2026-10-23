"""The dispatcher: the loop, the deadline scope, the outcomes, the endings and the re-ask."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path

import pytest

import idia.core
import idia.providers
from idia import (
    WARNING_NO_SNAPSHOT,
    WARNING_OVERFLOW_POLICY,
    WARNING_UNCHECKED_LIMIT,
    ConcurrencyLimit,
    Deadline,
    DeadlineExceeded,
    Dispatcher,
    DoesNotFit,
    Fail,
    Fault,
    LabelAnswer,
    Leaf,
    MappedAnswer,
    NoulAnswer,
    Outcome,
    Provider,
    ProviderError,
    RateLimiter,
    Refusal,
    Request,
    RequestFailed,
    Response,
    RetryPolicy,
    Router,
    Rule,
    WireRequest,
)
from idia.providers.scripted import Item, Reply, Script, Scripted
from tests._support import (
    TICK,
    FixedEstimator,
    Shorten,
    TextChars,
    capabilities,
    choice,
    extraction,
    noul,
    request,
    text_state,
)

QUICK = RetryPolicy(jitter=0.0)


def _yes(key: str = "q", p: float = 0.7) -> dict[str, MappedAnswer]:
    return {key: NoulAnswer(probability=p)}


def _off_sum() -> LabelAnswer:
    return LabelAnswer(probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.1)))


def _good() -> LabelAnswer:
    return LabelAnswer(probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.2)))


def _retryable(after: float | None = 0.01, request_id: str | None = "r-1") -> ProviderError:
    return ProviderError(
        "busy", is_retryable=True, retry_after=after, status_code=503, request_id=request_id
    )


def _leaf(script: Script, **overrides: object) -> Scripted:
    return Scripted(capabilities=capabilities(**overrides), script=script)


def _answer(dispatcher: Dispatcher, request_: Request) -> Response:
    return asyncio.run(dispatcher.answer(request_))


def _failure(dispatcher: Dispatcher, request_: Request) -> Response:
    with pytest.raises(RequestFailed) as raised:
        _answer(dispatcher, request_)
    return raised.value.response


def test_dispatcher_refuses_a_provider_as_its_leaf() -> None:
    dispatcher = Dispatcher(_leaf([]))
    with pytest.raises(TypeError, match=r"^a Dispatcher wraps a leaf; Dispatcher is already a"):
        Dispatcher(dispatcher)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"^a Dispatcher wraps a leaf; Router is already a"):
        Dispatcher(Router({"noul": dispatcher}))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"^a Dispatcher wraps a leaf; object is not a Leaf"):
        Dispatcher(object())  # pyright: ignore[reportArgumentType]


def test_scripted_leaf_is_not_a_provider_until_dispatched() -> None:
    leaf = _leaf([])
    assert isinstance(leaf, Leaf)
    assert not isinstance(leaf, Provider)
    dispatcher = Dispatcher(leaf)
    assert isinstance(dispatcher, Provider)
    assert not isinstance(dispatcher, Leaf)
    assert dispatcher.capabilities is leaf.capabilities


def test_dispatcher_counts_attempts_once_when_the_leaf_raises_then_answers() -> None:
    error = _retryable()
    leaf = _leaf([error, _yes()])
    before = time.time()
    response = _answer(Dispatcher(leaf, retry=QUICK), request())
    (outcome,) = response.outcomes
    assert outcome.attempts == 2
    assert len(leaf.calls) == 2
    (attempt,) = outcome.failed_attempts
    assert attempt.error is error
    assert (attempt.provider, attempt.model_requested) == (leaf.name, leaf.model)
    assert attempt.request_id == "r-1"
    assert attempt.usage is None
    assert attempt.started_at >= before
    assert attempt.duration >= 0
    assert outcome.error is None
    assert response.answers["q"] == NoulAnswer(probability=0.7)


def test_each_call_gets_the_callers_deadline_instant() -> None:
    leaf = _leaf([_retryable(), _yes()])
    request_ = request()
    _answer(Dispatcher(leaf, retry=QUICK), request_)
    assert [call.deadline for call in leaf.calls] == [request_.deadline, request_.deadline]
    assert [call.wire_request.questions for call in leaf.calls] == [request_.questions] * 2


def test_outcome_carries_the_leafs_provenance_and_no_raw_body() -> None:
    leaf = _leaf([Reply(answers=_yes(), model_reported="scripted-1-2026")])
    before = time.time()
    response = _answer(Dispatcher(leaf), request())
    (outcome,) = response.outcomes
    assert (outcome.provider, outcome.model_requested) == ("scripted", "scripted-1")
    assert outcome.model_reported == "scripted-1-2026"
    assert outcome.snapshot == "scripted-1-2026"
    assert (outcome.elicitation, outcome.calibrated) == ("passthrough", True)
    assert outcome.prompt_sent is not None
    assert outcome.usage is not None
    assert outcome.request_id == "scripted-1"
    assert outcome.started_at >= before
    assert outcome.latency >= 0
    assert outcome.keys == ("q",)


def test_outcome_carries_no_raw_body() -> None:
    response = _answer(Dispatcher(_leaf([_yes()])), request())
    (outcome,) = response.outcomes
    assert not hasattr(outcome, "raw_body")
    assert "raw_body" not in Outcome.__slots__
    assert not hasattr(response, "provenance")


def test_capability_failure_raises_before_any_call() -> None:
    leaf = _leaf([_yes("a")])
    with pytest.raises(DoesNotFit) as raised:
        _answer(Dispatcher(leaf), request(noul("a"), extraction("b")))
    assert raised.value.check == "kinds"
    assert leaf.calls == []


def test_the_reason_for_a_missing_kind_names_the_leafs_model() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[], model="jev-1.13.0")
    with pytest.raises(DoesNotFit) as raised:
        _answer(Dispatcher(leaf), request(noul("a"), extraction("b")))
    assert str(raised.value) == (
        "kinds not supported: extraction; no feature of model jev-1.13.0 adds it"
    )


def test_dispatcher_estimator_argument_overrides_the_leafs() -> None:
    leaf = _leaf([_yes()])
    with pytest.raises(DoesNotFit) as raised:
        _answer(Dispatcher(leaf, estimator=FixedEstimator(tokens=10**6)), request())
    assert raised.value.state_overflow
    assert leaf.calls == []


def test_partial_failure_returns_a_response() -> None:
    leaf = _leaf(
        [_yes("a"), ProviderError("broken", is_retryable=False), _yes("c")], max_questions=1
    )
    response = _answer(Dispatcher(leaf), request(noul("a"), noul("b"), noul("c")))
    assert [outcome.keys for outcome in response.outcomes] == [("a",), ("b",), ("c",)]
    assert set(response.answers) == {"a", "c"}
    assert response.outcomes[1].error is not None
    assert response.outcomes[1].attempts == 1
    assert [outcome.error is None for outcome in response.outcomes] == [True, False, True]


def test_failed_wire_request_keys_are_absent_from_answers() -> None:
    leaf = _leaf([ProviderError("broken"), _yes("b")], max_questions=1)
    response = _answer(Dispatcher(leaf), request(noul("a"), noul("b")))
    assert "a" not in response.answers
    assert response.outcomes[0].keys == ("a",)
    assert response.outcomes[0].prompt_sent is None
    assert response.outcomes[0].usage is None


def test_total_failure_raises_request_failed_with_the_response() -> None:
    error = ProviderError("broken", is_retryable=False, request_id="x-9")
    response = _failure(Dispatcher(_leaf([error])), request())
    (outcome,) = response.outcomes
    assert outcome.error is error
    assert outcome.request_id == "x-9"
    assert outcome.attempts == 1
    assert outcome.failed_attempts[0].error is error
    assert dict(response.answers) == {}
    assert (outcome.elicitation, outcome.calibrated, outcome.model_reported) == (None, None, None)


def test_request_with_passed_deadline_raises_request_failed_with_no_attempts() -> None:
    leaf = _leaf([_yes()])
    passed = Request(
        state=text_state(), questions=(noul(),), deadline=Deadline(at=time.monotonic() - 1)
    )
    response = _failure(Dispatcher(leaf), passed)
    (outcome,) = response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert not outcome.error.is_retryable
    assert outcome.attempts == 0
    assert outcome.failed_attempts == ()
    assert (outcome.provider, outcome.model_requested) == (leaf.name, leaf.model)
    assert leaf.calls == []


@pytest.mark.parametrize("passed", [True, False])
def test_a_deadline_within_one_clock_tick_is_not_sent(
    monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    # The tick is patched, not raced: five seconds left is within it, or not.
    monkeypatch.setattr(idia.core, "_CLOCK_TICK", 60.0 if passed else -60.0)
    leaf = _leaf([_yes()])
    dispatcher = Dispatcher(leaf)
    if not passed:
        assert _answer(dispatcher, request(seconds=5)).answers["q"] == NoulAnswer(probability=0.7)
        assert len(leaf.calls) == 1
        return
    response = _failure(dispatcher, request(seconds=5))
    (outcome,) = response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.attempts == 0
    assert leaf.calls == []


def test_a_cut_within_one_tick_of_the_deadline_is_not_retried_and_ends_with_deadline_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _retryable(None)
    calls = 0

    def cut_at_the_deadline(wire_request: WireRequest, deadline: Deadline) -> Item:
        # The first call ends as a call cut by its timeout one tick early would: retryable,
        # within one tick of the deadline. Any later call would answer.
        nonlocal calls
        calls += 1
        if calls == 1:
            monkeypatch.setattr(idia.core, "_CLOCK_TICK", 60.0)
            return error
        return _yes()

    leaf = Scripted(capabilities=capabilities(), script=cut_at_the_deadline)
    dispatcher = Dispatcher(leaf, retry=RetryPolicy.without_delay(attempts=3))
    response = _failure(dispatcher, request(seconds=5))
    assert calls == 1
    (outcome,) = response.outcomes
    assert outcome.attempts == 1
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.failed_attempts[0].error is error


def test_attempts_are_derived_from_failed_attempts() -> None:
    errors = [_retryable(0.0, f"r-{n}") for n in range(3)]
    leaf = _leaf(errors)
    response = _failure(Dispatcher(leaf, retry=RetryPolicy(attempts=3, jitter=0.0)), request())
    (outcome,) = response.outcomes
    assert outcome.attempts == 3 == len(outcome.failed_attempts) == len(leaf.calls)
    assert outcome.error is errors[-1]
    assert [attempt.request_id for attempt in outcome.failed_attempts] == ["r-0", "r-1", "r-2"]
    assert "attempts" not in Outcome.__slots__


def test_retries_stop_at_the_policy_attempts() -> None:
    leaf = _leaf([_retryable(0.0), _retryable(0.0), _yes()])
    _failure(Dispatcher(leaf, retry=RetryPolicy(attempts=2, jitter=0.0)), request())
    assert len(leaf.calls) == 2


def test_retry_after_is_honoured() -> None:
    leaf = _leaf([_retryable(0.05), _yes()])
    start = time.monotonic()
    response = _answer(Dispatcher(leaf, retry=RetryPolicy(base_delay=10.0, jitter=0.0)), request())
    elapsed = time.monotonic() - start
    assert 0.05 - TICK <= elapsed < 1.0
    assert response.outcomes[0].latency >= 0.05 - TICK

    ignoring = RetryPolicy(base_delay=0.0, jitter=0.0, honour_retry_after=False)
    leaf = _leaf([_retryable(5.0), _yes()])
    start = time.monotonic()
    _answer(Dispatcher(leaf, retry=ignoring), request())
    assert time.monotonic() - start < 0.5


@pytest.mark.parametrize("cut", ["the-retry-delay", "the-rate-limiter"])
def test_a_retry_the_deadline_cannot_fit_ends_the_wire_request_with_deadline_exceeded(
    cut: str,
) -> None:
    """The deadline ended it: the outcome's error is ``DeadlineExceeded``, the attempt made
    stays listed with the provider's error, and the outcome has no request id of its own."""
    error = _retryable(5.0 if cut == "the-retry-delay" else 0.0, request_id="r-1")
    leaf = _leaf([error, _yes()])

    async def scenario() -> Response:
        limiter = RateLimiter(rate=1, per=10.0) if cut == "the-rate-limiter" else None
        with pytest.raises(RequestFailed) as raised:
            await Dispatcher(leaf, retry=QUICK, rate=limiter).answer(request(seconds=0.5))
        return raised.value.response

    start = time.monotonic()
    response = asyncio.run(scenario())
    assert time.monotonic() - start < 0.3
    (outcome,) = response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.failed_attempts[0].error is error
    assert outcome.failed_attempts[0].request_id == "r-1"
    assert outcome.request_id is None
    assert outcome.attempts == 1
    assert len(leaf.calls) == 1
    if cut == "the-retry-delay":
        assert outcome.error.message == "the deadline left no time to try again"
    else:
        assert outcome.error.message.startswith("the rate limiter's next token is")


class _OvershootingRateLimiter(RateLimiter):
    """A limiter whose wait always ends past the deadline without raising: a late wake."""

    async def acquire(self, deadline: Deadline) -> None:
        await asyncio.sleep(max(deadline.remaining(), 0.0) + 0.05)


def test_limiter_wait_ending_past_the_deadline_leaves_the_wire_request_unsent() -> None:
    leaf = _leaf([_yes()])
    slot = ConcurrencyLimit(1)

    async def scenario() -> Response:
        dispatcher = Dispatcher(
            leaf, retry=QUICK, concurrency=slot, rate=_OvershootingRateLimiter(rate=1)
        )
        with pytest.raises(RequestFailed) as raised:
            await dispatcher.answer(request(seconds=0.05))
        # The slot was given back: it can be taken again at once.
        await slot.acquire(Deadline.after(0.01))
        slot.release()
        return raised.value.response

    response = asyncio.run(scenario())
    assert leaf.calls == []
    (outcome,) = response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.attempts == 0


def test_limiter_wait_ending_past_the_deadline_before_a_retry_ends_with_deadline_exceeded() -> None:
    error = _retryable(0.0)
    leaf = _leaf([error, _yes()])

    class _OvershootsSecond(RateLimiter):
        waits = 0

        async def acquire(self, deadline: Deadline) -> None:
            self.waits += 1
            if self.waits > 1:
                await asyncio.sleep(max(deadline.remaining(), 0.0) + 0.05)

    async def scenario() -> Response:
        dispatcher = Dispatcher(leaf, retry=QUICK, rate=_OvershootsSecond(rate=1))
        with pytest.raises(RequestFailed) as raised:
            await dispatcher.answer(request(seconds=0.2))
        return raised.value.response

    response = asyncio.run(scenario())
    assert len(leaf.calls) == 1
    (outcome,) = response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.error.message == "the deadline left no time to try again"
    assert outcome.failed_attempts[0].error is error
    assert outcome.attempts == 1


def test_concurrent_answers_on_one_dispatcher_keep_their_own_outcomes_and_share_the_limits() -> (
    None
):
    keys = ("q0", "q1", "q2", "q3")
    in_flight = 0
    peak = 0
    failed_once: set[str] = set()

    async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
        nonlocal in_flight, peak
        (question,) = wire_request.questions
        key = question.key
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            in_flight -= 1
        if key not in failed_once:
            failed_once.add(key)
            return ProviderError("busy", is_retryable=True, request_id=f"busy-{key}")
        return _yes(key)

    leaf = _leaf(script)
    # Distinct deadlines, so each call's log entries can be told to hold its own.
    requests = [request(noul(key), seconds=60.0 + index) for index, key in enumerate(keys)]

    async def scenario() -> tuple[list[Response], Response]:
        dispatcher = Dispatcher(
            leaf,
            retry=RetryPolicy.without_delay(attempts=2),
            concurrency=ConcurrencyLimit(2),
            rate=RateLimiter(rate=8, per=60.0),
        )
        responses = await asyncio.gather(*(dispatcher.answer(r) for r in requests))
        # The four calls spent the bucket's eight tokens; the next token is 7.5 s away.
        with pytest.raises(RequestFailed) as raised:
            await dispatcher.answer(request(noul("q4"), seconds=0.2))
        return list(responses), raised.value.response

    responses, refused = asyncio.run(scenario())
    assert peak == 2
    for index, (key, response) in enumerate(zip(keys, responses, strict=True)):
        assert set(response.answers) == {key}
        (outcome,) = response.outcomes
        assert outcome.keys == (key,)
        assert outcome.error is None
        assert outcome.attempts == 2
        (attempt,) = outcome.failed_attempts
        assert attempt.request_id == f"busy-{key}"
        assert outcome.request_id is not None
        n = int(outcome.request_id.removeprefix("scripted-"))
        assert outcome.request_id == f"scripted-{n}"
        assert leaf.calls[n - 1].wire_request.questions == requests[index].questions
    assert len(leaf.calls) == 8
    for index, key in enumerate(keys):
        own = [call for call in leaf.calls if call.wire_request.questions[0].key == key]
        assert len(own) == 2
        assert all(call.deadline == requests[index].deadline for call in own)
    (outcome,) = refused.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.attempts == 0
    assert outcome.failed_attempts == ()
    assert dict(refused.answers) == {}


def test_quota_as_429_is_not_retried() -> None:
    quota = ProviderError("quota exhausted", is_retryable=False, status_code=429, retry_after=1.0)
    leaf = _leaf([quota, _yes()])
    response = _failure(Dispatcher(leaf, retry=QUICK), request())
    assert response.outcomes[0].error is quota
    assert len(leaf.calls) == 1


def test_unreadable_body_fails_its_outcome_and_is_not_retried() -> None:
    unreadable = ProviderError("the body is not JSON", is_retryable=False, status_code=200)
    leaf = _leaf([unreadable, _yes()])
    response = _failure(Dispatcher(leaf, retry=QUICK), request())
    (outcome,) = response.outcomes
    assert outcome.error is unreadable
    assert outcome.attempts == 1
    assert len(leaf.calls) == 1


def test_validation_fault_is_not_retried() -> None:
    leaf = _leaf([{"q": _off_sum()}, {"q": _good()}])
    response = _answer(Dispatcher(leaf, retry=QUICK), request(choice()))
    fault = response.answers["q"]
    assert isinstance(fault, Fault)
    assert [failure.rule for failure in fault.failures] == [Rule.sum_tolerance]
    (outcome,) = response.outcomes
    assert outcome.error is None
    assert outcome.attempts == 1
    assert len(leaf.calls) == 1


def test_packed_calls_to_a_drifting_alias_report_each_snapshot() -> None:
    leaf = _leaf(
        [
            Reply(answers=_yes("a"), model_reported="scripted-1-2026"),
            Reply(answers=_yes("b"), pinned=False),
        ],
        max_questions=1,
    )
    response = _answer(Dispatcher(leaf), request(noul("a"), noul("b")))
    first, second = response.outcomes
    assert first.snapshot == "scripted-1-2026"
    assert second.snapshot is None
    assert WARNING_NO_SNAPSHOT in [warning.code for warning in second.warnings]
    assert {outcome.provider for outcome in response.outcomes} == {leaf.name}
    assert {outcome.model_requested for outcome in response.outcomes} == {leaf.model}


def test_unchecked_limit_warnings_land_on_failed_outcomes_too() -> None:
    # The shared record leaves max_question_tokens unknown, so every outcome warns.
    leaf = _leaf([ProviderError("broken"), Reply(answers=_yes("b"), warnings=())], max_questions=1)
    response = _answer(Dispatcher(leaf), request(noul("a"), noul("b")))
    failed, answered = response.outcomes
    assert [(w.code, w.field) for w in failed.warnings] == [
        (WARNING_UNCHECKED_LIMIT, "max_question_tokens")
    ]
    assert (WARNING_UNCHECKED_LIMIT, "max_question_tokens") in [
        (w.code, w.field) for w in answered.warnings
    ]
    # The leaf's own warnings come first on an answered outcome.
    assert answered.warnings[-1].field == "max_question_tokens"
    assert response.warnings == ()


def test_overflow_policy_warning_lands_on_the_response() -> None:
    leaf = Scripted(
        capabilities=capabilities(max_input_tokens=50, max_question_tokens=1000),
        script=[_yes()],
        estimator=TextChars(),
    )
    dispatcher = Dispatcher(leaf, overflow=Shorten(keep=20))
    response = _answer(dispatcher, request(state=text_state("x" * 100)))
    (warning,) = response.warnings
    assert (warning.code, warning.field) == (WARNING_OVERFLOW_POLICY, "max_input_tokens")
    assert warning.message == "the state overflowed max_input_tokens; policy 'shorten' applied"
    assert response.outcomes[0].warnings == ()
    assert leaf.calls[0].wire_request.state == text_state("x" * 20)


def _overflowing_leaf(script: Script, **overrides: object) -> Scripted:
    """A leaf whose 100-character state is over its 50-token input limit."""
    return Scripted(
        capabilities=capabilities(max_input_tokens=50, max_question_tokens=1000, **overrides),
        script=script,
        estimator=TextChars(),
    )


def test_dispatcher_plan_makes_no_call_and_is_the_plan_answer_runs() -> None:
    leaf = _overflowing_leaf([_yes("a"), _yes("b")], max_questions=1, max_body_bytes=None)
    dispatcher = Dispatcher(leaf, overflow=Shorten(keep=20))
    request_ = request(noul("a"), noul("b"), state=text_state("x" * 100))
    plan = dispatcher.plan(request_)
    assert leaf.calls == []
    assert len(plan.wire_requests) == 2
    assert [w.code for w in plan.response_warnings] == [WARNING_OVERFLOW_POLICY]
    assert all(
        [w.code for w in warnings] == [WARNING_UNCHECKED_LIMIT]
        for warnings in plan.outcome_warnings
    )
    response = _answer(dispatcher, request_)
    assert [call.wire_request for call in leaf.calls] == list(plan.wire_requests)
    assert response.warnings == plan.response_warnings
    for outcome, warnings in zip(response.outcomes, plan.outcome_warnings, strict=True):
        assert outcome.warnings[len(outcome.warnings) - len(warnings) :] == warnings


def test_dispatcher_plan_raises_does_not_fit_before_any_call() -> None:
    leaf = _overflowing_leaf([_yes()])
    with pytest.raises(DoesNotFit, match="over the limit of 50") as raised:
        Dispatcher(leaf).plan(request(state=text_state("x" * 100)))
    assert raised.value.state_overflow
    assert leaf.calls == []


def test_dispatcher_reask_and_overflow_are_read_only() -> None:
    plain = Dispatcher(_leaf([_yes()]))
    assert plain.reask is False
    assert plain.overflow == Fail()
    policy = Shorten(keep=20)
    built = Dispatcher(_leaf([_yes()]), reask=True, overflow=policy)
    assert built.reask is True
    assert built.overflow is policy
    for name, value in (("reask", False), ("overflow", Fail())):
        with pytest.raises(AttributeError):
            setattr(built, name, value)
    assert (built.reask, built.overflow) == (True, policy)


def test_reask_outcome_follows_the_plan_outcomes_and_its_answer_stands() -> None:
    leaf = _leaf([{"a": _yes("a")["a"], "b": _off_sum()}, {"b": _good()}])
    dispatcher = Dispatcher(leaf, reask=True)
    request_ = request(noul("a"), choice("b"))
    response = _answer(dispatcher, request_)
    assert [outcome.keys for outcome in response.outcomes] == [("a", "b"), ("b",)]
    assert response.answers["b"] == _good()
    assert response.answers["a"] == NoulAnswer(probability=0.7)
    assert len(leaf.calls) == 2
    again = leaf.calls[1]
    assert again.wire_request.questions == (choice("b"),)
    assert again.wire_request.state == request_.state
    assert again.deadline == request_.deadline
    assert response.outcomes[1].error is None


def test_failed_reask_leaves_the_fault_standing() -> None:
    error = ProviderError("broken")
    leaf = _leaf([{"q": _off_sum()}, error])
    response = _answer(Dispatcher(leaf, reask=True, retry=QUICK), request(choice()))
    assert isinstance(response.answers["q"], Fault)
    assert len(response.outcomes) == 2
    assert response.outcomes[1].error is error
    assert response.outcomes[1].keys == ("q",)


def test_credential_failure_leaves_the_rest_of_the_plan_unsent() -> None:
    # A Decisions-shaped leaf: 200 questions a wire request, so 201 make two.
    error = ProviderError("HTTP 401: invalid key", status_code=401, request_id="req-401")
    leaf = _leaf([error, _yes("again")], max_questions=200)
    dispatcher = Dispatcher(leaf)
    keys = tuple(f"q{index}" for index in range(201))
    request_ = request(*(noul(key) for key in keys))
    plan = dispatcher.plan(request_)
    response = _failure(dispatcher, request_)
    assert len(leaf.calls) == 1
    sent, unsent = response.outcomes
    assert sent.keys == keys[:200]
    assert sent.attempts == 1
    assert sent.error is error
    assert unsent.keys == keys[200:]
    assert unsent.attempts == 0
    assert unsent.error is error
    assert unsent.failed_attempts == ()
    assert unsent.request_id is None
    assert (unsent.provider, unsent.model_requested) == (leaf.name, leaf.model)
    assert unsent.prompt_sent is None
    assert unsent.usage is None
    assert unsent.latency < 0.01
    assert unsent.started_at >= sent.started_at
    assert unsent.warnings == plan.outcome_warnings[1] != ()
    # The stop is the call's own: the next call on the same dispatcher is sent.
    assert _answer(dispatcher, request(noul("again"))).answers == _yes("again")
    assert len(leaf.calls) == 2


@pytest.mark.parametrize(
    ("status", "retryable", "stops"),
    [(403, False, True), (401, True, False), (500, False, False), (None, False, False)],
)
def test_only_a_non_retryable_401_or_403_stops_the_plan(
    status: int | None, retryable: bool, stops: bool
) -> None:
    error = ProviderError("refused", status_code=status, is_retryable=retryable)
    leaf = _leaf([error, error], max_questions=1)
    response = _failure(
        Dispatcher(leaf, retry=RetryPolicy.without_delay(attempts=1)),
        request(noul("a"), noul("b")),
    )
    assert len(leaf.calls) == (1 if stops else 2)
    assert [outcome.attempts for outcome in response.outcomes] == ([1, 0] if stops else [1, 1])


def test_credential_failure_leaves_the_reask_unsent_and_the_fault_standing() -> None:
    error = ProviderError("HTTP 401: invalid key", status_code=401)
    leaf = _leaf([{"a": _off_sum()}, error, {"c": _good()}], max_questions=1)
    response = _answer(Dispatcher(leaf, reask=True), request(choice("a"), choice("b"), choice("c")))
    assert len(leaf.calls) == 2
    assert [outcome.keys for outcome in response.outcomes] == [("a",), ("b",), ("c",), ("a",)]
    assert [outcome.attempts for outcome in response.outcomes] == [1, 1, 0, 0]
    assert response.outcomes[2].error is error
    assert response.outcomes[3].error is error
    assert isinstance(response.answers["a"], Fault)
    assert set(response.answers) == {"a"}


def test_credential_failure_on_a_reask_leaves_the_later_reasks_unsent() -> None:
    error = ProviderError("HTTP 403: forbidden", status_code=403)
    leaf = _leaf([{"a": _off_sum()}, {"b": _off_sum()}, error], max_questions=1)
    response = _answer(Dispatcher(leaf, reask=True), request(choice("a"), choice("b")))
    assert len(leaf.calls) == 3
    assert [outcome.keys for outcome in response.outcomes] == [("a",), ("b",), ("a",), ("b",)]
    assert [outcome.attempts for outcome in response.outcomes] == [1, 1, 1, 0]
    assert response.outcomes[3].error is error
    assert isinstance(response.answers["b"], Fault)


def test_reask_is_off_by_default_and_fires_on_a_fault_only() -> None:
    leaf = _leaf([{"q": _off_sum()}, {"q": _good()}])
    response = _answer(Dispatcher(leaf), request(choice()))
    assert isinstance(response.answers["q"], Fault)
    assert len(response.outcomes) == 1
    assert len(leaf.calls) == 1

    refusing = _leaf([{"q": Refusal()}, {"q": _good()}])
    response = _answer(Dispatcher(refusing, reask=True), request(choice()))
    assert isinstance(response.answers["q"], Refusal)
    assert len(refusing.calls) == 1


def test_reask_happens_at_most_once_per_wire_request() -> None:
    leaf = _leaf([{"q": _off_sum()}, {"q": _off_sum()}, {"q": _good()}])
    response = _answer(Dispatcher(leaf, reask=True), request(choice()))
    assert isinstance(response.answers["q"], Fault)
    assert len(leaf.calls) == 2
    assert len(response.outcomes) == 2


def test_reask_skips_a_failed_plan_wire_request() -> None:
    leaf = _leaf([ProviderError("broken"), {"b": _off_sum()}, {"b": _good()}], max_questions=1)
    response = _answer(Dispatcher(leaf, reask=True), request(noul("a"), choice("b")))
    assert [outcome.keys for outcome in response.outcomes] == [("a",), ("b",), ("b",)]
    assert "a" not in response.answers
    assert response.answers["b"] == _good()


def test_reask_carries_the_unchecked_limit_warnings_that_bear_on_it() -> None:
    leaf = _leaf([{"q": _off_sum()}, {"q": _good()}])
    response = _answer(Dispatcher(leaf, reask=True), request(choice()))
    assert [w.field for w in response.outcomes[1].warnings] == ["max_question_tokens"]


def test_no_provider_module_imports_the_loop() -> None:
    package = Path(idia.providers.__file__).parent
    for module in package.glob("*.py"):
        for node in ast.walk(ast.parse(module.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            for name in names:
                assert name not in {
                    "idia.retry",
                    "idia.limits",
                    "idia.dispatch",
                }, f"{module.name} imports {name}"


# ----------------------------------------------------------------------------------------
# Closing
# ----------------------------------------------------------------------------------------


class _Closable(Scripted):
    """A scripted leaf with an ``aclose`` that counts how often it was closed."""

    closed: int = 0

    async def aclose(self) -> None:
        self.closed += 1


def test_dispatcher_aclose_closes_a_leaf_that_has_aclose() -> None:
    leaf = _Closable(capabilities=capabilities(), script=[])
    asyncio.run(Dispatcher(leaf).aclose())
    assert leaf.closed == 1


def test_dispatcher_aclose_ignores_a_leaf_without_aclose() -> None:
    leaf = _leaf([_yes()])
    assert not hasattr(leaf, "aclose")
    asyncio.run(Dispatcher(leaf).aclose())
    assert leaf.calls == []


def test_dispatcher_is_an_async_context_manager() -> None:
    leaf = _Closable(capabilities=capabilities(), script=[_yes()])

    async def use() -> Response:
        async with Dispatcher(leaf) as dispatcher:
            assert leaf.closed == 0
            return await dispatcher.answer(request(noul()))

    response = asyncio.run(use())
    assert isinstance(response.answers["q"], NoulAnswer)
    assert leaf.closed == 1


def test_closing_twice_passes_through_each_time() -> None:
    leaf = _Closable(capabilities=capabilities(), script=[])
    dispatcher = Dispatcher(leaf)

    async def twice() -> None:
        await dispatcher.aclose()
        await dispatcher.aclose()
        async with dispatcher:
            pass

    asyncio.run(twice())
    assert leaf.closed == 3
