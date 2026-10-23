"""A client the leaf created follows the running event loop; an injected one is refused on
another loop; a closed leaf, and a client that reports itself closed, are refused before
sending; a ``RuntimeError`` the SDK lets through mid-call is mapped."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import math

import httpx2
import pytest
from openai import AsyncOpenAI

from idia import Deadline, ProviderError, Question, WireRequest
from idia.providers import _openai
from idia.providers.chat import OpenAIChat
from idia.providers.http import OTHER_LOOP
from idia.providers.openai_decisions import OpenAIDecisions
from tests._system_one_cases import TEXT

_REQUEST = WireRequest(
    state=TEXT, questions=(Question(key="n", kind="noul", instruction="Did it rain?"),)
)
_BODY = {
    "answers": [{"type": "predicate", "name": "q0", "probability": 0.75}],
    "model": "gpt-6-luna",
    "usage": {"input_tokens": 40, "output_tokens": 0, "total_tokens": 40},
}


class LoopBound:
    """A mock transport bound, as a pooled connection is, to the first loop that sends on
    it: on any other loop it raises what httpx2 raises over a closed loop's socket."""

    def __init__(self, body: object = _BODY) -> None:
        self.body = body
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sent = 0

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
        elif loop is not self.loop:
            raise RuntimeError("Event loop is closed")
        self.sent += 1
        return httpx2.Response(200, json=self.body, headers={"x-request-id": "req-1"})

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))


def test_the_loop_bound_transport_fails_on_a_second_loop_as_a_pooled_connection_does() -> None:
    """The double reproduces the fault: its client used on two loops raises."""
    client = LoopBound().client()
    asyncio.run(client.post("https://x.test/"))
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        asyncio.run(client.post("https://x.test/"))


def test_a_created_client_follows_the_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two ``asyncio.run`` calls on one leaf, both answered: the second runs on a fresh
    client, the first dropped unclosed."""
    transports: list[LoopBound] = []

    def created() -> httpx2.AsyncClient:
        transports.append(LoopBound())
        return transports[-1].client()

    monkeypatch.setattr(_openai, "created_client", created)
    leaf = OpenAIDecisions(api_key="k")
    first = leaf.client
    for _ in range(2):
        response = asyncio.run(leaf.call(_REQUEST, Deadline.after(5)))
        assert response.answers["n"].probability == 0.75  # type: ignore[union-attr]
    assert [t.sent for t in transports] == [1, 1]
    assert leaf.client is not first
    assert leaf.client.max_retries == 0
    assert not first.is_closed()


def _chat_body(yes: float) -> dict[str, object]:
    top = [
        {"token": letter, "logprob": math.log(mass), "bytes": list(letter.encode())}
        for letter, mass in (("A", yes), ("B", 1 - yes))
    ]
    return {
        "id": "chatcmpl-1",
        "model": "gpt-4.1-2025-04-14",
        "choices": [
            {
                "index": 0,
                "message": {"content": "A"},
                "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": top}]},
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51},
    }


def test_the_chat_leafs_created_client_follows_the_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chat leaf sends through the base's loop check too."""
    transports: list[LoopBound] = []

    def created() -> httpx2.AsyncClient:
        transports.append(LoopBound(_chat_body(0.75)))
        return transports[-1].client()

    monkeypatch.setattr(_openai, "created_client", created)
    leaf = OpenAIChat(model="gpt-4.1", api_key="k")
    for _ in range(2):
        response = asyncio.run(leaf.call(_REQUEST, Deadline.after(5)))
        assert response.answers["n"].probability == pytest.approx(0.75)  # type: ignore[union-attr]
    assert [t.sent for t in transports] == [1, 1]


def test_a_created_client_on_one_loop_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    transports: list[LoopBound] = []

    def created() -> httpx2.AsyncClient:
        transports.append(LoopBound())
        return transports[-1].client()

    monkeypatch.setattr(_openai, "created_client", created)
    leaf = OpenAIDecisions(api_key="k")

    async def twice() -> None:
        await leaf.call(_REQUEST, Deadline.after(5))
        await leaf.call(_REQUEST, Deadline.after(5))
        await leaf.aclose()

    asyncio.run(twice())
    assert [t.sent for t in transports] == [2]
    assert leaf.client.is_closed()


def test_aclose_on_another_loop_drops_the_created_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``asyncio.run(leaf.aclose())`` after a call on another loop does not raise; the
    client is dropped unclosed, and the closed leaf makes no other."""
    monkeypatch.setattr(_openai, "created_client", lambda: LoopBound().client())
    leaf = OpenAIDecisions(api_key="k")
    asyncio.run(leaf.call(_REQUEST, Deadline.after(5)))
    first = leaf.client
    asyncio.run(leaf.aclose())
    asyncio.run(leaf.aclose())
    assert leaf.client is first
    assert not first.is_closed()


@pytest.mark.parametrize("loops", [1, 2])
def test_a_closed_leaf_is_refused_and_creates_no_client(
    monkeypatch: pytest.MonkeyPatch, loops: int
) -> None:
    transports: list[LoopBound] = []

    def created() -> httpx2.AsyncClient:
        transports.append(LoopBound())
        return transports[-1].client()

    monkeypatch.setattr(_openai, "created_client", created)
    leaf = OpenAIDecisions(api_key="k")

    async def call_close_and_call(calls_after: int) -> None:
        await leaf.call(_REQUEST, Deadline.after(5))
        await leaf.aclose()
        for _ in range(calls_after):
            await leaf.call(_REQUEST, Deadline.after(5))

    if loops == 2:
        asyncio.run(call_close_and_call(0))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            call_close_and_call(1) if loops == 1 else leaf.call(_REQUEST, Deadline.after(5))
        )
    assert raised.value.message == "the leaf is closed"
    assert not raised.value.is_retryable
    assert len(transports) == 1


def test_an_injected_client_on_another_loop_is_refused() -> None:
    """Not retryable, no status, before anything is sent; the client is left open."""
    transport = LoopBound()
    injected = AsyncOpenAI(api_key="k", http_client=transport.client())
    leaf = OpenAIDecisions(client=injected)
    asyncio.run(leaf.call(_REQUEST, Deadline.after(5)))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(_REQUEST, Deadline.after(5)))
    error = raised.value
    assert error.message == OTHER_LOOP
    assert not error.is_retryable
    assert error.status_code is None
    assert transport.sent == 1
    assert not injected.is_closed()


def test_a_runtime_error_from_the_sdk_mid_call_is_a_retryable_provider_error() -> None:
    """The backstop, for a connection that died under the client mid-call: the leaf cannot
    see a loop it never ran on; the closed loop's ``RuntimeError``, which the SDK does not
    wrap, is mapped, retryable and naming its class."""
    transport = LoopBound()
    injected = AsyncOpenAI(api_key="k", http_client=transport.client())
    asyncio.run(injected.get("/models", cast_to=object))  # the caller used it on its own loop
    with pytest.raises(ProviderError) as raised:
        asyncio.run(OpenAIDecisions(client=injected).call(_REQUEST, Deadline.after(5)))
    assert raised.value.is_retryable
    assert raised.value.message == "the client failed: RuntimeError: Event loop is closed"


def test_a_runtime_error_subclass_from_the_sdk_escapes_unchanged() -> None:
    """Only a bare ``RuntimeError`` is mapped; a subclass is a bug, not a closed loop."""
    error = NotImplementedError("not built")

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error

    injected = AsyncOpenAI(
        api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    with pytest.raises(NotImplementedError) as raised:
        asyncio.run(OpenAIDecisions(client=injected).call(_REQUEST, Deadline.after(5)))
    assert raised.value is error


@pytest.mark.parametrize("chat", [False, True], ids=["decisions", "chat"])
def test_a_closed_sdk_client_is_refused_before_sending(chat: bool) -> None:
    transport = LoopBound()
    injected = AsyncOpenAI(api_key="k", http_client=transport.client())
    leaf = (
        OpenAIChat(model="gpt-4.1", client=injected) if chat else OpenAIDecisions(client=injected)
    )

    async def close_and_call() -> None:
        await injected.close()
        await leaf.call(_REQUEST, Deadline.after(5))

    with pytest.raises(ProviderError) as raised:
        asyncio.run(close_and_call())
    error = raised.value
    assert error.message == "the client is closed"
    assert not error.is_retryable
    assert error.status_code is None
    assert transport.sent == 0


def test_a_created_sdk_client_the_caller_closes_is_refused_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transports: list[LoopBound] = []

    def created() -> httpx2.AsyncClient:
        transports.append(LoopBound())
        return transports[-1].client()

    monkeypatch.setattr(_openai, "created_client", created)
    leaf = OpenAIDecisions(api_key="k")

    async def close_and_call() -> None:
        await leaf.client.close()
        await leaf.call(_REQUEST, Deadline.after(5))

    with pytest.raises(ProviderError, match=r"^the client is closed$"):
        asyncio.run(close_and_call())
    assert [t.sent for t in transports] == [0]


def test_an_injected_leaf_after_aclose_is_refused_and_its_client_stays_open() -> None:
    transport = LoopBound()
    injected = AsyncOpenAI(api_key="k", http_client=transport.client())
    leaf = OpenAIDecisions(client=injected)

    async def call_close_and_call() -> None:
        await leaf.call(_REQUEST, Deadline.after(5))
        await leaf.aclose()
        await leaf.call(_REQUEST, Deadline.after(5))

    with pytest.raises(ProviderError) as raised:
        asyncio.run(call_close_and_call())
    assert raised.value.message == "the leaf is closed"
    assert not raised.value.is_retryable
    assert transport.sent == 1
    assert not injected.is_closed()
