"""``idia.leaf`` and ``idia.provider_names``: the leaf ``idia.provider`` wraps, unwrapped,
and the names the registry holds; and a leaf of the caller's own wrapped around one, counting
every call."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio

import pytest

from idia import (
    Deadline,
    Dispatcher,
    Leaf,
    NoulAnswer,
    Provider,
    ProviderError,
    RequestFailed,
    RetryPolicy,
    TokenEstimator,
    WireRequest,
    WireResponse,
    leaf,
    provider,
    provider_names,
    providers,
)
from idia.core import Capabilities
from idia.providers import ProviderNotFound, _Entry
from idia.providers.scripted import Scripted
from tests._closable_scripted import ClosableScripted
from tests._support import capabilities, noul, request


class Counting:
    """A leaf around another, counting every call it passes on: the wrapper a caller with a
    call budget builds around what ``idia.leaf`` returns."""

    def __init__(self, inner: Leaf) -> None:
        self._inner = inner
        self.name: str = inner.name
        self.model: str = inner.model
        self.capabilities: Capabilities = inner.capabilities
        self.estimator: TokenEstimator = inner.estimator
        self.calls = 0

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        self.calls += 1
        return await self._inner.call(wire_request, deadline)

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(providers._REGISTRY, "fake", _Entry(target="tests._fake_leaf:FakeLeaf"))


def test_leaf_builds_what_provider_wraps(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch)
    built = leaf("fake", model="fake-3")
    assert isinstance(built, Leaf)
    assert not isinstance(built, Provider)
    wrapped = provider("fake", model="fake-3")
    assert type(built) is type(wrapped._leaf)
    assert (built.name, built.model) == (wrapped._leaf.name, wrapped._leaf.model)


def test_leaf_unknown_name_raises_the_registry_error() -> None:
    with pytest.raises(ProviderNotFound) as raised:
        leaf("nowhere")
    assert raised.value.registered == provider_names()


def test_a_keyword_the_leaf_does_not_take_names_the_leaf_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake(monkeypatch)
    with pytest.raises(TypeError) as plain:
        leaf("fake", modle="x")
    assert str(plain.value).startswith("leaf('fake'): ")
    assert "Dispatcher" not in str(plain.value)


def test_each_dispatcher_keyword_given_to_leaf_points_to_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake(monkeypatch)
    with pytest.raises(TypeError) as raised:
        leaf("fake", retry=None, rate=None, reask=True)
    assert str(raised.value).endswith(
        "; retry, rate, reask are the Dispatcher's: wrap the leaf in Dispatcher(leaf, ...)"
    )


def test_provider_names_are_what_provider_not_found_lists() -> None:
    names = provider_names()
    assert names == tuple(sorted(providers._REGISTRY))
    assert names == (
        "cloudflare-clairic",
        "cloudflare-clef",
        "gliner",
        "openai-chat",
        "openai-decisions",
        "system-one",
    )
    with pytest.raises(ProviderNotFound) as raised:
        provider("nowhere")
    assert raised.value.registered == names


def test_a_counting_leaf_sees_every_attempt_retries_included() -> None:
    busy = ProviderError("busy", is_retryable=True, status_code=503)
    inner = Scripted(capabilities=capabilities(), script=[busy, {"q": NoulAnswer(probability=0.9)}])
    counting = Counting(inner)
    dispatcher = Dispatcher(counting, retry=RetryPolicy.without_delay(attempts=3))
    response = asyncio.run(dispatcher.answer(request(noul())))
    (outcome,) = response.outcomes
    assert counting.calls == outcome.attempts == len(inner.calls) == 2
    assert len(outcome.failed_attempts) == 1
    assert outcome.provider == "scripted"


def test_a_counting_leaf_counts_a_call_that_is_not_retried() -> None:
    refused = ProviderError("unauthorized", is_retryable=False, status_code=401)
    inner = Scripted(capabilities=capabilities(), script=[refused])
    counting = Counting(inner)
    dispatcher = Dispatcher(counting, retry=RetryPolicy.without_delay(attempts=3))
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(dispatcher.answer(request(noul())))
    (outcome,) = raised.value.response.outcomes
    assert counting.calls == outcome.attempts == 1


def test_closing_the_dispatcher_closes_the_leaf_inside_the_counting_leaf() -> None:
    inner = ClosableScripted(capabilities=capabilities(), script=[])

    async def scenario() -> None:
        async with Dispatcher(Counting(inner)):
            pass

    asyncio.run(scenario())
    assert inner.closed
