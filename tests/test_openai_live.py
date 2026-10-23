"""The one live run per OpenAI model, which records the cassettes the other tests replay.
Skipped unless ``IDIA_LIVE=1``; the key comes from ``OPENAI_API_KEY``.

The decisions run records a success per kind, the provider half's single-question
requests, one and three questions over one state (whose usage says whether the state is
counted again per question), a request that might draw a refusal, 201 questions, one past
the declared cap, and the failure: a choice of 256 values, one past the documented 255.
A test selected by ``-k order`` records the question-order probe once, the choice request
twice unchanged and once with its questions reversed, into ``order/``; it skips when the
three cassettes exist. One selected by ``-k order_four`` records the stronger probe once
into ``order-4/``: four unsaturated questions on one mixed state in three orders and each
alone through the leaf, and the second order sent outside it under the first order's
names; it skips when every cassette exists. One selected by ``-k unknown_model`` records
the noul request through the leaf as ``gpt-5.6-luna``, a model that does not exist, and
keeps the cassette only when the wire refuses it with 404 ``model_not_found``.

The chat run records a success per kind, one question each, the score being the uncertain
``score_crowd``, and the provider half's singles, and, outside the leaf, the
noul request asking for 21 top logprobs, one past the documented cap, as its failure. A
test selected by ``-k unauthorized`` records the noul request through the chat leaf with
the fixed invalid key, which the wire must refuse with 401.

``score-absent-level``, the chat leaf's first score recording, in which a level's letter
fell outside the top 20, is kept as it was recorded: no live test rewrites it. A
credentials failure, an unknown model, an empty read or a failure that comes back a
success stops the run.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import httpx2
import pytest
from openai import AsyncOpenAI

from idia import Deadline, Fault, Leaf, ProviderError, Rule, WireRequest, WireResponse
from idia.providers import openai_decisions as decisions_module
from idia.providers.chat import openai as chat_module
from idia.providers.chat.openai import OpenAIChat
from idia.providers.openai_decisions import OpenAIDecisions
from tests._http import RecordingTransport, record_cassette, write_cassette
from tests._openai_cases import (
    CHAT_DIR,
    CHAT_MODEL,
    DECISIONS_DIR,
    DECISIONS_MODEL,
    FAILURE,
    INVALID_KEY,
    KINDS,
    ORDER,
    ORDER_FOUR,
    ORDER_FOUR_ARMS,
    ORDER_FOUR_OUTSIDE,
    ORDER_FOUR_QUESTIONS,
    ORDER_NAMES,
    RECOUNT,
    REFUSAL_PROBE,
    TOO_MANY,
    TOO_MANY_CHOICES,
    UNAUTHORIZED,
    UNKNOWN_MODEL,
    UNKNOWN_MODEL_NAME,
    chat_request,
    decisions_request,
    order_four_requests,
    order_requests,
    singles,
)

pytestmark = pytest.mark.live

_SECONDS: Final = 60.0
_CHAT_URL: Final = "https://api.openai.com/v1/chat/completions"
_DECISIONS_URL: Final = "https://api.openai.com/v1/decisions"


def _key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.fail("OPENAI_API_KEY is not set")
    return key


def _sdk(client: httpx2.AsyncClient) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_key(), http_client=client, max_retries=0)


async def _record(
    path: Path, factory: Callable[[httpx2.AsyncClient], Leaf], wire_request: WireRequest
) -> None:
    """One live call through the leaf, written as the cassette at ``path`` once
    :func:`_check` passes it; a call that stops the run leaves ``path`` as it was."""
    await record_cassette(path, factory, wire_request, Deadline.after(_SECONDS), check=_check)


def _check(side: Path, result: WireResponse | ProviderError) -> None:
    """Stop the run on a credentials failure, an outcome of the wrong kind or an empty
    read, before the cassette, at ``side`` for now, reaches its path."""
    if isinstance(result, ProviderError):
        if result.status_code in (401, 403):
            pytest.fail(f"credentials refused: HTTP {result.status_code}; stopping")
        if side.stem != FAILURE:
            pytest.fail(f"{side.name}: HTTP {result.status_code}: {result.message}; stopping")
        return
    if side.stem == FAILURE:
        raise AssertionError(f"{side.name}: the request meant to fail was answered")
    empty = [
        key
        for key, answer in result.answers.items()
        if isinstance(answer, Fault) and any(f.rule is Rule.empty_logprobs for f in answer.failures)
    ]
    if empty:
        pytest.fail(f"{side.name}: empty logprobs for {empty}; stopping")


async def _decisions() -> None:
    def factory(client: httpx2.AsyncClient) -> Leaf:
        return OpenAIDecisions(model=DECISIONS_MODEL, client=_sdk(client))

    for kind in KINDS:
        await _record(DECISIONS_DIR / f"{kind}.json", factory, decisions_request(kind))
    for name, wire_request in singles().items():
        await _record(DECISIONS_DIR / f"{name}.json", factory, wire_request)
    for name, wire_request in RECOUNT.items():
        await _record(DECISIONS_DIR / f"{name}.json", factory, wire_request)
    await _record(DECISIONS_DIR / "refusal-probe.json", factory, REFUSAL_PROBE)
    await _record(DECISIONS_DIR / "questions-201.json", factory, TOO_MANY)
    await _record(DECISIONS_DIR / f"{FAILURE}.json", factory, TOO_MANY_CHOICES)


async def _chat_failure() -> None:
    """The noul request, sent outside the leaf with 21 top logprobs asked for."""
    wire_request = chat_request("noul")
    [question] = wire_request.questions
    letters = chat_module._letters(question)  # pyright: ignore[reportPrivateUsage]
    body = {
        "model": CHAT_MODEL,
        "messages": chat_module._messages(wire_request.state, question, letters),  # pyright: ignore[reportPrivateUsage]
        "logprobs": True,
        "top_logprobs": chat_module.TOP_LOGPROBS + 1,
        "temperature": chat_module.TEMPERATURE,
        "max_completion_tokens": 1,
    }
    recorder = RecordingTransport()
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    async with httpx2.AsyncClient(transport=recorder, timeout=_SECONDS) as client:
        await client.post(_CHAT_URL, content=json.dumps(body).encode(), headers=headers)
    [(request, response, content)] = recorder.exchanges
    named = OpenAIChat(model=CHAT_MODEL, api_key=_key())
    write_cassette(
        CHAT_DIR / f"{FAILURE}.json",
        leaf=named,
        request=request,
        response=response,
        content=content,
        wire_request=wire_request,
        expected=None,
    )
    await named.aclose()


async def _chat() -> None:
    def factory(client: httpx2.AsyncClient) -> Leaf:
        return OpenAIChat(model=CHAT_MODEL, client=_sdk(client))

    for kind in KINDS:
        await _record(CHAT_DIR / f"{kind}.json", factory, chat_request(kind))
    for name, wire_request in singles().items():
        await _record(CHAT_DIR / f"{name}.json", factory, wire_request)
    await _chat_failure()


def test_live_decisions() -> None:
    asyncio.run(_decisions())


async def _decisions_order() -> None:
    """The question-order probe, recorded once into ``order/``: the choice request twice,
    then reversed. Skipped when its three cassettes exist; a call that stops the run
    removes every one this run wrote, so the three are only ever kept together."""
    paths = [DECISIONS_DIR / ORDER / f"{name}.json" for name in ORDER_NAMES]
    if all(path.exists() for path in paths):
        pytest.skip("the question-order probe is recorded once; its three cassettes exist")

    def factory(client: httpx2.AsyncClient) -> Leaf:
        return OpenAIDecisions(model=DECISIONS_MODEL, client=_sdk(client))

    sent: list[Path] = []
    try:
        for path, wire_request in zip(paths, order_requests().values(), strict=True):
            await _record(path, factory, wire_request)
            sent.append(path)
    except BaseException:
        for path in sent:
            path.unlink(missing_ok=True)
        raise


def test_live_decisions_order() -> None:
    asyncio.run(_decisions_order())


async def _decisions_outside(path: Path) -> None:
    """The ``b`` order sent outside the leaf, each question under its ``a``-order name,
    written with the ``a`` request and the answers the leaf maps from the response."""
    first = order_four_requests()["order-a-1"]
    accepted = frozenset({"image/png", "image/jpeg"})
    body, _ = decisions_module._body(DECISIONS_MODEL, first, accepted)  # pyright: ignore[reportPrivateUsage]
    questions = cast(tuple[object, ...], body["questions"])
    order = ORDER_FOUR_ARMS["b"]
    body = {**body, "questions": tuple(questions[i] for i in order)}
    recorder = RecordingTransport()
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    async with httpx2.AsyncClient(transport=recorder, timeout=_SECONDS) as client:
        await client.post(_DECISIONS_URL, content=json.dumps(body).encode(), headers=headers)
    [(request, response, content)] = recorder.exchanges
    if response.status_code != 200:
        pytest.fail(f"{path.name}: HTTP {response.status_code}; stopping")

    def answer(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=content, headers={"content-type": "application/json"})

    mapping = httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
    named = OpenAIDecisions(model=DECISIONS_MODEL, client=_sdk(mapping))
    mapped = await named.call(first, Deadline.after(_SECONDS))
    await mapping.aclose()
    if any(isinstance(a, Fault) for a in mapped.answers.values()) or len(mapped.answers) != len(
        ORDER_FOUR_QUESTIONS
    ):
        pytest.fail(f"{path.name}: the answers did not map by the a-order names; stopping")
    write_cassette(
        path,
        leaf=named,
        request=request,
        response=response,
        content=content,
        wire_request=first,
        expected=mapped.answers,
    )


async def _decisions_order_four() -> None:
    """The stronger question-order probe, recorded once into ``order-4/``: ``a``, ``b``
    and ``c`` and each question alone through the leaf, then ``e`` outside it. Skipped
    when every cassette exists; a call that stops the run removes every one this run
    wrote, so they are only ever kept together."""
    through = order_four_requests()
    paths = [DECISIONS_DIR / ORDER_FOUR / f"{name}.json" for name in through]
    outside = [DECISIONS_DIR / ORDER_FOUR / f"{name}.json" for name in ORDER_FOUR_OUTSIDE]
    if all(path.exists() for path in [*paths, *outside]):
        pytest.skip("the four-question order probe is recorded once; its cassettes exist")

    def factory(client: httpx2.AsyncClient) -> Leaf:
        return OpenAIDecisions(model=DECISIONS_MODEL, client=_sdk(client))

    sent: list[Path] = []
    try:
        for path, wire_request in zip(paths, through.values(), strict=True):
            await _record(path, factory, wire_request)
            sent.append(path)
        for path in outside:
            await _decisions_outside(path)
            sent.append(path)
    except BaseException:
        for path in sent:
            path.unlink(missing_ok=True)
        raise


def test_live_decisions_order_four() -> None:
    asyncio.run(_decisions_order_four())


def _unknown_model_check(side: Path, result: WireResponse | ProviderError) -> None:
    """Keep the cassette only when the wire refused the unknown model with 404
    ``model_not_found``; stop on a credentials failure, an answer or any other status."""
    if not isinstance(result, ProviderError):
        pytest.fail(f"{side.name}: `{UNKNOWN_MODEL_NAME}` now exists; choose another name")
    if result.status_code in (401, 403):
        pytest.fail(f"credentials refused: HTTP {result.status_code}; stopping")
    if result.status_code != 404 or "model_not_found" not in result.message:
        pytest.fail(f"{side.name}: expected HTTP 404 model_not_found, got {result.message}")


async def _decisions_unknown_model() -> None:
    def factory(client: httpx2.AsyncClient) -> Leaf:
        return OpenAIDecisions(model=UNKNOWN_MODEL_NAME, client=_sdk(client))

    await record_cassette(
        DECISIONS_DIR / f"{UNKNOWN_MODEL}.json",
        factory,
        decisions_request("noul"),
        Deadline.after(_SECONDS),
        check=_unknown_model_check,
    )


def test_live_decisions_unknown_model() -> None:
    asyncio.run(_decisions_unknown_model())


def test_live_chat() -> None:
    asyncio.run(_chat())


def _unauthorized_check(side: Path, result: WireResponse | ProviderError) -> None:
    """Keep the cassette only when the wire refused the invalid key with 401."""
    if not isinstance(result, ProviderError) or result.status_code != 401:
        shown = result.status_code if isinstance(result, ProviderError) else "an answer"
        pytest.fail(f"{side.name}: expected HTTP 401, got {shown}; stopping")


async def _chat_unauthorized() -> None:
    def factory(client: httpx2.AsyncClient) -> Leaf:
        sdk = AsyncOpenAI(api_key=INVALID_KEY, http_client=client, max_retries=0)
        return OpenAIChat(model=CHAT_MODEL, client=sdk)

    await record_cassette(
        CHAT_DIR / f"{UNAUTHORIZED}.json",
        factory,
        chat_request("noul"),
        Deadline.after(_SECONDS),
        check=_unauthorized_check,
    )


def test_live_chat_unauthorized() -> None:
    asyncio.run(_chat_unauthorized())
