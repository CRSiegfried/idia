"""The concurrency limit and the rate limiter: waits bounded by the deadline, shareable."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest

from idia import (
    ConcurrencyLimit,
    Deadline,
    DeadlineExceeded,
    Dispatcher,
    MappedAnswer,
    NoulAnswer,
    ProviderError,
    RateLimiter,
    RequestFailed,
    Response,
    RetryPolicy,
    WireRequest,
)
from idia.providers.scripted import Item, Scripted
from tests._support import TICK, capabilities, noul, request


def _yes() -> dict[str, MappedAnswer]:
    return {"q": NoulAnswer(probability=0.7)}


def _slow(item: Item, seconds: float) -> Callable[[WireRequest, Deadline], Awaitable[Item]]:
    async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
        await asyncio.sleep(seconds)
        return item

    return script


@pytest.mark.parametrize("limit", [0, -1, True])
def test_concurrency_limit_refuses_bad_values(limit: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ConcurrencyLimit(limit)


@pytest.mark.parametrize(("rate", "per"), [(0, 1.0), (-1, 1.0), (1, 0.0)])
def test_rate_limiter_refuses_bad_values(rate: float, per: float) -> None:
    with pytest.raises(ValueError, match="above 0"):
        RateLimiter(rate, per=per)


def test_concurrency_limit_holds_a_second_caller_until_the_first_releases() -> None:
    async def scenario() -> list[str]:
        limit = ConcurrencyLimit(1)
        events: list[str] = []
        await limit.acquire(Deadline.after(1.0))

        async def second() -> None:
            await limit.acquire(Deadline.after(1.0))
            events.append("second in")
            limit.release()

        task = asyncio.create_task(second())
        await asyncio.sleep(0.02)
        events.append("first out")
        limit.release()
        await task
        return events

    assert asyncio.run(scenario()) == ["first out", "second in"]


def test_concurrency_wait_past_the_deadline_raises_deadline_exceeded() -> None:
    async def scenario() -> None:
        limit = ConcurrencyLimit(1)
        await limit.acquire(Deadline.after(1.0))
        with pytest.raises(DeadlineExceeded) as raised:
            await limit.acquire(Deadline.after(0.02))
        assert not raised.value.is_retryable
        limit.release()
        # The slot given back is the only one: the refused wait took nothing.
        await limit.acquire(Deadline.after(0.0))
        limit.release()

    asyncio.run(scenario())


def test_rate_limiter_gives_a_burst_then_waits_for_the_next_token() -> None:
    async def scenario() -> float:
        limiter = RateLimiter(rate=2, per=0.4)
        start = time.monotonic()
        await limiter.acquire(Deadline.after(3.0))
        await limiter.acquire(Deadline.after(3.0))
        assert time.monotonic() - start < 0.1
        await limiter.acquire(Deadline.after(3.0))
        return time.monotonic() - start

    assert 0.2 - TICK <= asyncio.run(scenario()) < 2.0


def test_rate_limiter_refuses_without_waiting_or_consuming_when_the_wait_overruns() -> None:
    async def scenario() -> None:
        limiter = RateLimiter(rate=1, per=10.0)
        await limiter.acquire(Deadline.after(1.0))
        start = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            await limiter.acquire(Deadline.after(0.5))
        assert time.monotonic() - start < 0.25
        assert limiter._tokens == pytest.approx(0.0, abs=0.01)  # type: ignore[reportUnknownMemberType]

    asyncio.run(scenario())


def test_limiter_wait_counts_against_the_deadline() -> None:
    """A limiter wait before the first attempt that would end past the deadline makes a
    never-sent outcome: ``DeadlineExceeded``, no attempts, no call."""

    async def scenario() -> tuple[Scripted, RequestFailed]:
        limiter = RateLimiter(rate=1, per=10.0)
        await limiter.acquire(Deadline.after(1.0))
        leaf = Scripted(capabilities=capabilities(), script=[_yes()])
        dispatcher = Dispatcher(leaf, rate=limiter)
        with pytest.raises(RequestFailed) as raised:
            await dispatcher.answer(request(seconds=0.2))
        return leaf, raised.value

    leaf, failure = asyncio.run(scenario())
    (outcome,) = failure.response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.attempts == 0
    assert outcome.failed_attempts == ()
    assert outcome.prompt_sent is None
    assert leaf.calls == []


def test_shared_limiter_bounds_two_dispatchers() -> None:
    """One concurrency limit shared by two dispatchers keeps one call in flight."""
    in_flight = 0
    peak = 0

    def tracking() -> Callable[[WireRequest, Deadline], Awaitable[Item]]:
        async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return _yes()

        return script

    async def scenario() -> list[Response]:
        limit = ConcurrencyLimit(1)
        first = Dispatcher(
            Scripted(capabilities=capabilities(), script=tracking()), concurrency=limit
        )
        second = Dispatcher(
            Scripted(capabilities=capabilities(), script=tracking()), concurrency=limit
        )
        return list(await asyncio.gather(first.answer(request()), second.answer(request())))

    responses = asyncio.run(scenario())
    assert len(responses) == 2
    assert peak == 1


def test_shared_rate_limiter_is_one_bucket_for_two_dispatchers() -> None:
    async def scenario() -> RequestFailed:
        limiter = RateLimiter(rate=2, per=10.0)
        first = Dispatcher(Scripted(capabilities=capabilities(), script=[_yes()]), rate=limiter)
        second = Dispatcher(Scripted(capabilities=capabilities(), script=[_yes()]), rate=limiter)
        await first.answer(request())
        await second.answer(request())
        third = Dispatcher(Scripted(capabilities=capabilities(), script=[_yes()]), rate=limiter)
        with pytest.raises(RequestFailed) as raised:
            await third.answer(request(seconds=0.2))
        return raised.value

    failure = asyncio.run(scenario())
    assert isinstance(failure.response.outcomes[0].error, DeadlineExceeded)


def test_concurrency_slot_is_released_before_a_retry_sleep() -> None:
    async def scenario() -> tuple[Response, Response]:
        limit = ConcurrencyLimit(1)
        retrying = Dispatcher(
            Scripted(
                capabilities=capabilities(),
                script=[ProviderError("busy", is_retryable=True, retry_after=0.5), _yes()],
            ),
            retry=RetryPolicy(jitter=0.0),
            concurrency=limit,
        )
        other = Dispatcher(
            Scripted(capabilities=capabilities(), script=_slow(_yes(), 0.01)), concurrency=limit
        )

        async def late_other() -> Response:
            await asyncio.sleep(0.02)
            return await other.answer(request())

        return await asyncio.gather(retrying.answer(request()), late_other())

    retried, other = asyncio.run(scenario())
    assert retried.outcomes[0].attempts == 2
    assert retried.outcomes[0].latency >= 0.5 - TICK
    # The other dispatcher ran while the first slept, so it did not wait out the sleep.
    assert other.outcomes[0].latency < 0.3


def test_one_dispatcher_with_a_concurrency_limit_serves_two_event_loops_in_turn() -> None:
    log: list[str] = []
    started = gate = asyncio.Event()

    async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
        (question,) = wire_request.questions
        log.append(f"start {question.key}")
        if question.key == "a":
            started.set()
            await gate.wait()
        log.append(f"end {question.key}")
        return {question.key: NoulAnswer(probability=0.7)}

    dispatcher = Dispatcher(
        Scripted(capabilities=capabilities(), script=script),
        retry=RetryPolicy.without_delay(attempts=1),
        concurrency=ConcurrencyLimit(1),
    )

    async def scenario() -> None:
        nonlocal started, gate
        started, gate = asyncio.Event(), asyncio.Event()

        async def opener() -> None:
            await started.wait()
            log.append("open")
            gate.set()

        await asyncio.gather(
            dispatcher.answer(request(noul("a"))),
            dispatcher.answer(request(noul("b"))),
            opener(),
        )

    for _ in range(2):
        log.clear()
        asyncio.run(scenario())
        # "b" was gathered before the opener yet starts after it: it waited for the slot.
        assert log == ["start a", "open", "end a", "start b", "end b"]
