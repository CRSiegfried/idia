"""What the two OpenAI leaves share: the client's lifecycle, the mapper from the SDK's
errors to ``ProviderError``, the reading of a raw body, and the estimator."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from idia import (
    Deadline,
    ImagePart,
    JsonPart,
    Label,
    ProviderError,
    Question,
    State,
    TextPart,
)
from idia.providers import _openai
from idia.providers._openai import OpenAIEstimator, send
from idia.providers.chat.openai import OpenAIChat
from idia.providers.openai_decisions import OpenAIDecisions
from tests._openai_cases import decisions_request

_REQUEST = decisions_request("noul")


def _wire(status: int, body: object, headers: Mapping[str, str] | None = None) -> AsyncOpenAI:
    """An SDK client at its default retries over a transport that always answers so."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx2.Response(
            status, content=content, headers={"x-request-id": "req-9", **(headers or {})}
        )

    return AsyncOpenAI(
        api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    )


def _raised(client: AsyncOpenAI) -> ProviderError:
    leaf = OpenAIDecisions(client=client)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(_REQUEST, Deadline.after(5)))
    return raised.value


def _error(code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": "nope", "type": "invalid_request_error", "code": code}}


# ----------------------------------------------------------------------------------------
# The mapper
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (409, True),
        (413, False),
        (422, False),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_the_status_table_over_the_sdks_error(status: int, retryable: bool) -> None:
    error = _raised(_wire(status, _error("rate_limit_exceeded")))
    assert (error.status_code, error.is_retryable) == (status, retryable)
    assert error.request_id == "req-9"
    assert "nope" in error.message


def test_a_401_body_echoing_a_key_shaped_string_yields_the_placeholder() -> None:
    key = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"
    body = {"error": {"message": f"Incorrect API key provided: {key}.", "code": "invalid_api_key"}}
    error = _raised(_wire(401, body))
    assert key not in error.message
    assert key not in str(error)
    assert "<filtered>" in error.message
    assert error.request_id == "req-9"


def test_a_request_id_quoted_in_an_error_body_leaves_the_request_id_field_intact() -> None:
    quoted = "req_76f189c39eda412dbdf7edef202ea06b"
    error = _raised(_wire(500, {"error": {"message": f"retry later, quote {quoted}"}}))
    assert quoted in error.message
    assert error.request_id == "req-9"


def test_a_key_the_wire_masked_itself_passes_as_sent() -> None:
    masked = "sk-apost********alid"
    error = _raised(_wire(401, {"error": {"message": f"Incorrect API key provided: {masked}."}}))
    assert masked in error.message


def test_a_quota_exhausted_arriving_as_429_is_not_retryable() -> None:
    error = _raised(_wire(429, _error("insufficient_quota"), {"retry-after": "3"}))
    assert error.status_code == 429
    assert not error.is_retryable
    assert error.retry_after == 3.0
    assert error.request_id == "req-9"
    assert "quota" in error.message


def test_retry_after_ms_is_read_before_retry_after() -> None:
    error = _raised(_wire(429, _error(), {"retry-after-ms": "1500", "retry-after": "9"}))
    assert error.retry_after == 1.5
    assert error.is_retryable


def test_an_error_body_that_is_not_json_still_maps_by_status() -> None:
    error = _raised(_wire(502, b"<html>bad gateway</html>"))
    assert (error.status_code, error.is_retryable) == (502, True)


@pytest.mark.parametrize(
    "raised",
    [
        lambda: httpx2.ConnectError("refused"),
        lambda: httpx2.ReadTimeout("slow"),
    ],
    ids=["connection", "timeout"],
)
def test_a_connection_error_or_timeout_is_retryable(raised: Callable[[], Exception]) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise raised()

    client = AsyncOpenAI(
        api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    )
    error = _raised(client)
    assert error.is_retryable
    assert error.status_code is None


def _over(handle: Callable[[httpx2.Request], httpx2.Response], **options: Any) -> AsyncOpenAI:
    return AsyncOpenAI(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)), **options
    )


def _never(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
    raise AssertionError("nothing is sent")


def test_a_malformed_request_the_sdk_refuses_is_not_retried() -> None:
    """The SDK wraps the client's refusal of an illegal header in its connection error; the
    leaf reads its cause and names the class alone, quoting no header."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.LocalProtocolError("Illegal header value b'Bearer sk-abc\\nXYZ'")

    error = _raised(_over(handle, api_key="k"))
    assert (error.is_retryable, error.status_code) == (False, None)
    assert error.message == "the request is malformed: LocalProtocolError"


def test_a_key_the_sdk_cannot_encode_is_a_malformed_request_quoting_nothing() -> None:
    error = _raised(_over(_never, api_key="sk-t\u00e9"))
    assert (error.is_retryable, error.message) == (
        False,
        "the request is malformed: UnicodeEncodeError",
    )


def test_a_base_url_with_no_scheme_is_a_malformed_request_keeping_its_reason() -> None:
    # The real transport refuses the scheme before connecting; a mock transport would not.
    error = _raised(AsyncOpenAI(api_key="k", base_url="wire.test/"))
    assert (error.is_retryable, error.status_code) == (False, None)
    assert error.message.startswith("the request is malformed: UnsupportedProtocol: ")


def test_a_plain_connection_error_from_the_sdk_stays_retryable() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    assert _raised(_over(handle, api_key="k")).is_retryable


def test_another_sdk_error_is_not_retryable() -> None:
    async def failing() -> object:
        raise openai.OpenAIError("misconfigured")

    with pytest.raises(ProviderError) as raised:
        asyncio.run(send(Deadline.after(5), failing))
    assert not raised.value.is_retryable
    assert "misconfigured" in raised.value.message


def test_a_call_with_no_raw_response_is_refused() -> None:
    async def bare() -> object:
        return object()

    with pytest.raises(ProviderError, match="raw HTTP response"):
        asyncio.run(send(Deadline.after(5), bare))


def test_a_passed_deadline_is_retryable_and_sends_nothing() -> None:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200)

    client = AsyncOpenAI(
        api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    )
    leaf = OpenAIDecisions(client=client)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(leaf.call(_REQUEST, Deadline.after(-1)))
    assert raised.value.is_retryable
    assert sent == []


def test_a_provider_error_is_never_caught() -> None:
    mine = ProviderError("mine", is_retryable=True, status_code=418)

    async def raising() -> object:
        raise mine

    with pytest.raises(ProviderError) as raised:
        asyncio.run(send(Deadline.after(5), raising))
    assert raised.value is mine


# ----------------------------------------------------------------------------------------
# The body
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "why"),
    [(b"<html/>", "not JSON"), (b"[1, 2]", "not a JSON object"), (b'{"a": NaN}', "not JSON")],
)
def test_a_body_that_cannot_be_read_is_not_retryable(content: bytes, why: str) -> None:
    error = _raised(_wire(200, content))
    assert not error.is_retryable
    assert why in error.message
    assert (error.status_code, error.request_id) == (200, "req-9")


def test_compact_json_refuses_a_value_json_cannot_write() -> None:
    with pytest.raises(ProviderError, match="JSON"):
        _openai.compact(float("nan"))
    with pytest.raises(ProviderError, match="JSON"):
        _openai.compact({"x": object()})  # type: ignore[dict-item]  # pyright: ignore[reportArgumentType]
    assert _openai.compact({"é": (1, 2)}) == '{"é":[1,2]}'


def test_a_read_only_mapping_is_written_as_an_object() -> None:
    from types import MappingProxyType

    assert _openai.compact({"m": MappingProxyType({"a": 1})}) == '{"m":{"a":1}}'


# ----------------------------------------------------------------------------------------
# The client
# ----------------------------------------------------------------------------------------


def _chat(**options: Any) -> OpenAIChat:
    return OpenAIChat(model="m", **options)


@pytest.mark.parametrize("make", [OpenAIDecisions, _chat])
def test_a_created_client_has_no_retries_and_is_closed_once_safely(
    make: Callable[..., Any],
) -> None:
    leaf = make(api_key="k")
    assert leaf.client.max_retries == 0
    assert leaf.client._client._transport._pool._retries == 0

    async def close_twice() -> None:
        await leaf.aclose()
        await leaf.aclose()

    asyncio.run(close_twice())
    assert leaf.client.is_closed()


def test_an_injected_client_is_used_with_no_retries_and_left_open() -> None:
    injected = _wire(200, {})
    assert injected.max_retries == 2

    async def use() -> OpenAIDecisions:
        async with OpenAIDecisions(client=injected) as leaf:
            return leaf

    leaf = asyncio.run(use())
    assert leaf.client.max_retries == 0
    assert leaf.client._client is injected._client
    assert not injected.is_closed()


def test_the_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert OpenAIDecisions().client.api_key == "from-env"


def test_no_key_and_no_client_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIDecisions()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChat(model="gpt-4.1")


# ----------------------------------------------------------------------------------------
# The estimator
# ----------------------------------------------------------------------------------------


def test_the_estimate_counts_overhead_state_questions_and_images() -> None:
    estimator = OpenAIEstimator(
        overhead=100,
        per_question=7,
        state_chars=6,
        question_chars=4,
        image_tokens=500,
        margin=0.2,
    )
    state = State(
        parts=(
            TextPart(text="x" * 58),
            JsonPart(value={"a": 1}),
            ImagePart(data=b"\x89PNG", media_type="image/png"),
        )
    )
    alone = estimator.estimate(state, ())
    # 60 characters of quoted text and 8 of JSON: 68 / 6, rounded up.
    assert alone.tokens == 100 + 12 + 500
    assert alone.margin == 0.2
    question = Question(
        key="c", kind="choice", instruction="Which?", labels=(Label(name="a"), Label(name="b"))
    )
    asked = json.dumps([["choice", "Which?", [["a", None], ["b", None]]]])
    with_question = estimator.estimate(state, (question,))
    assert with_question.tokens == alone.tokens + 7 + -(-len(asked) // 4)
