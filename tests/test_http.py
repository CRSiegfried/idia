"""``HttpLeaf``: the client's lifecycle, the user agent, one POST under the deadline, the
body read as a whole, and the status table behind ``map_status``."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import datetime
import email.utils
import json
from collections.abc import Callable, Mapping
from typing import Final

import httpx2
import pytest

import idia
import idia.core
from idia import Deadline, ProviderError
from idia.providers import http
from idia.providers.http import (
    OTHER_LOOP,
    HttpLeaf,
    created_client,
    json_loads,
    retry_after,
    status_error,
    within,
)

_OK: dict[str, object] = {"model": "m", "answers": {}, "usage": {}}

Handler = Callable[[httpx2.Request], httpx2.Response]


def _leaf(handler: Handler) -> HttpLeaf:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return HttpLeaf(base_url="https://wire.test/", request_id_header="x-req", client=client)


def _respond(status: int, body: object = _OK, headers: Mapping[str, str] | None = None) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if isinstance(body, bytes):
            return httpx2.Response(status, content=body, headers=headers)
        return httpx2.Response(status, json=body, headers=headers)

    return handler


async def _post(leaf: HttpLeaf, deadline: float = 5.0) -> object:
    try:
        return await leaf._post("/v1/x", b"{}", Deadline.after(deadline))
    except ProviderError as error:
        return error


def _error(handler: Handler) -> ProviderError:
    raised = asyncio.run(_post(_leaf(handler)))
    assert isinstance(raised, ProviderError), raised
    return raised


# ----------------------------------------------------------------------------------------
# The status table
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529, 599])
def test_retryable_statuses(status: int) -> None:
    error = _error(_respond(status, {"error": "busy"}, {"x-req": "r-1"}))
    assert error.is_retryable
    assert error.status_code == status
    assert error.request_id == "r-1"
    assert str(status) in error.message


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 415, 422, 418, 451, 302])
def test_other_statuses_are_not_retryable(status: int) -> None:
    error = _error(_respond(status, {"detail": [{"msg": "bad"}]}))
    assert not error.is_retryable
    assert error.status_code == status
    assert error.request_id is None


def test_retry_after_ms_is_read_before_retry_after() -> None:
    error = _error(_respond(429, headers={"retry-after-ms": "1500", "retry-after": "7"}))
    assert error.retry_after == 1.5


def test_retry_after_in_seconds() -> None:
    assert _error(_respond(503, headers={"retry-after": "7"})).retry_after == 7.0


def test_retry_after_as_an_http_date() -> None:
    when = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=30)
    error = _error(_respond(429, headers={"retry-after": email.utils.format_datetime(when)}))
    assert error.retry_after is not None
    assert 25.0 < error.retry_after <= 30.0


def test_an_http_date_in_the_past_waits_nothing() -> None:
    error = _error(_respond(429, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}))
    assert error.retry_after == 0.0


def test_an_http_date_without_a_zone_is_read_as_utc() -> None:
    error = _error(_respond(429, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 -0000"}))
    assert error.retry_after == 0.0


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"retry-after": "soon"},
        {"retry-after-ms": "-5"},
        {"retry-after-ms": "nan"},
        {"retry-after": "Wed, 21 Oct 99999999999 07:28:00 GMT"},
    ],
)
def test_an_unreadable_retry_after_is_none(headers: dict[str, str]) -> None:
    assert _error(_respond(429, headers=headers)).retry_after is None


def test_a_bad_retry_after_ms_falls_back_to_retry_after() -> None:
    error = _error(_respond(429, headers={"retry-after-ms": "x", "retry-after": "2"}))
    assert error.retry_after == 2.0


def test_a_body_nested_too_deeply_is_unreadable_not_a_recursion_error() -> None:
    deep = b"[" * 100_000
    error = http.status_error(500, {}, deep, None)
    assert error.is_retryable
    with pytest.raises(ProviderError, match="nested too deeply") as raised:
        http.read_object(httpx2.Response(200, content=deep), "req-1")
    assert not raised.value.is_retryable


def test_the_message_names_the_wire_error_but_the_error_keeps_no_body() -> None:
    error = _error(_respond(400, {"error": {"code": "max_tokens_exceeded"}}))
    assert "max_tokens_exceeded" in error.message
    assert not any(isinstance(value, bytes) for value in vars(error).values())
    assert not {"body", "raw_body", "content", "text", "response"} & set(vars(error))


def test_a_401_body_echoing_a_bearer_token_yields_the_placeholder() -> None:
    token = "Bearer abcdefghijklmnop0123456789"
    error = _error(_respond(401, {"detail": f"invalid credentials: {token}"}))
    assert token not in error.message
    assert "<filtered>" in error.message
    assert error.status_code == 401


def test_a_key_that_straddles_the_cut_is_filtered_before_it() -> None:
    key = "sk-" + "a" * 40
    error = _error(_respond(400, {"message": "x" * 280 + " " + key}))
    # Cut first, the key would keep fewer than the twenty characters the sk- shape needs.
    assert "aaaa" not in error.message
    assert "<filtered>" in error.message


def test_a_key_that_straddles_the_cut_of_a_plain_text_body_is_filtered_before_it() -> None:
    body = ("x" * 290 + " sk-" + "a" * 40).encode()
    error = _error(_respond(502, body))
    assert "aaaa" not in error.message
    assert "<filtered" in error.message  # the placeholder itself may meet the cut


async def _post_with(leaf: HttpLeaf, headers: Mapping[str, str] | None = None) -> object:
    try:
        return await leaf._post("/v1/x", b"{}", Deadline.after(5), headers=headers)
    except ProviderError as error:
        return error


_ILLEGAL_HEADER: Final = "Illegal header value b'Bearer sk-abc\\nXYZsecret'"


@pytest.mark.parametrize(
    ("leaf", "headers", "message"),
    [
        pytest.param(
            lambda: _leaf(_raising(httpx2.LocalProtocolError(_ILLEGAL_HEADER))),
            None,
            "the request is malformed: LocalProtocolError",
            id="illegal-header",
        ),
        pytest.param(
            lambda: _leaf(_respond(200)),
            {"x-key": "sk-t\u00e9"},
            "the request is malformed: UnicodeEncodeError",
            id="unencodable-value",
        ),
        pytest.param(
            lambda: HttpLeaf(base_url="wire.test/", request_id_header="x-req"),
            None,
            "the request is malformed: UnsupportedProtocol: ",
            id="no-scheme",
        ),
        pytest.param(
            lambda: HttpLeaf(base_url="http://[::1/", request_id_header="x-req"),
            None,
            "the request is malformed: InvalidURL: ",
            id="unparsable-url",
        ),
    ],
)
def test_a_malformed_request_the_client_refuses_is_not_retried(
    leaf: Callable[[], HttpLeaf], headers: Mapping[str, str] | None, message: str
) -> None:
    """One case per cause; a header or a value is named by its class alone, since its text
    quotes it, and a URL keeps its reason."""
    raised = asyncio.run(_post_with(leaf(), headers))
    assert isinstance(raised, ProviderError)
    assert (raised.is_retryable, raised.status_code) == (False, None)
    if message.endswith(": "):
        assert raised.message.startswith(message)
    else:
        assert raised.message == message
    assert "XYZsecret" not in raised.message


def test_a_long_error_is_cut_short() -> None:
    error = _error(_respond(400, {"message": "x" * 1000}))
    assert len(error.message) < 400
    assert error.message.endswith("...")


@pytest.mark.parametrize(
    ("body", "detail"),
    [(b"Bad Gateway", "Bad Gateway"), (b"[1]", ""), (b"{}", ""), (b"", "")],
)
def test_an_error_body_without_a_message(body: bytes, detail: str) -> None:
    error = _error(_respond(502, body))
    assert error.message == "HTTP 502" + (f": {detail}" if detail else "")


def test_a_leaf_may_override_the_status_table() -> None:
    class Quota(HttpLeaf):
        def map_status(self, response: httpx2.Response) -> ProviderError:
            error = super().map_status(response)
            if b"quota" in response.content:
                error.is_retryable = False
            return error

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_respond(429, {"error": "quota"})))
    leaf = Quota(base_url="https://wire.test", request_id_header="x-req", client=client)
    raised = asyncio.run(_post(leaf))
    assert isinstance(raised, ProviderError)
    assert not raised.is_retryable


# ----------------------------------------------------------------------------------------
# Transport errors and the deadline
# ----------------------------------------------------------------------------------------


def _raising(error: Exception) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error

    return handler


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("refused"),
        httpx2.ReadError("reset"),
        httpx2.RemoteProtocolError("bad"),
        httpx2.ConnectTimeout("slow"),
        httpx2.ReadTimeout("slow"),
    ],
)
def test_connection_errors_and_timeouts_are_retryable(error: Exception) -> None:
    raised = _error(_raising(error))
    assert raised.is_retryable
    assert raised.status_code is None


def test_another_request_error_is_not_retryable() -> None:
    raised = _error(_raising(httpx2.DecodingError("garbled")))
    assert not raised.is_retryable


def test_a_runtime_error_from_the_client_mid_call_is_a_retryable_provider_error() -> None:
    """The backstop, for a connection that died under the client mid-call, as from a
    pooled connection whose event loop has closed."""
    raised = _error(_raising(RuntimeError("Event loop is closed")))
    assert raised.is_retryable
    assert raised.status_code is None
    assert raised.message == "the client failed: RuntimeError: Event loop is closed"


def test_a_runtime_error_subclass_propagates() -> None:
    """Only a bare ``RuntimeError`` is mapped; a subclass is a bug, not a closed loop."""
    error = NotImplementedError("not built")
    with pytest.raises(NotImplementedError) as raised:
        asyncio.run(_post(_leaf(_raising(error))))
    assert raised.value is error


def test_a_closed_injected_client_is_refused_before_sending() -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=_OK)

    leaf = _leaf(handler)

    async def close_then_post() -> object:
        await leaf.client.aclose()
        return await _post(leaf)

    raised = asyncio.run(close_then_post())
    assert isinstance(raised, ProviderError)
    assert raised.message == "the client is closed"
    assert not raised.is_retryable
    assert raised.status_code is None
    assert sent == []


def test_a_created_client_the_caller_closes_is_refused_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=_OK)

    monkeypatch.setattr(
        http, "created_client", lambda: httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    leaf = HttpLeaf(base_url="https://wire.test/", request_id_header="x-req")

    async def close_then_post() -> object:
        await leaf.client.aclose()
        return await _post(leaf)

    raised = asyncio.run(close_then_post())
    assert isinstance(raised, ProviderError)
    assert (raised.message, raised.is_retryable) == ("the client is closed", False)
    assert sent == []


def test_the_deadline_is_the_calls_timeout() -> None:
    async def stall(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(10)
        raise AssertionError("not reached")  # pragma: no cover

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(stall))
    leaf = HttpLeaf(base_url="https://wire.test", request_id_header="x-req", client=client)

    async def run() -> tuple[object, float]:
        start = asyncio.get_running_loop().time()
        raised = await _post(leaf, deadline=0.1)
        return raised, asyncio.get_running_loop().time() - start

    raised, elapsed = asyncio.run(run())
    assert isinstance(raised, ProviderError)
    assert raised.is_retryable
    assert "deadline" in raised.message
    assert elapsed < 1.0


def test_a_passed_deadline_ends_the_call_before_it_is_sent() -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        sent.append(request)
        return httpx2.Response(200, json=_OK)

    raised = asyncio.run(_post(_leaf(handler), deadline=-1.0))
    assert isinstance(raised, ProviderError)
    assert raised.is_retryable
    assert sent == []


# ----------------------------------------------------------------------------------------
# Reading a success
# ----------------------------------------------------------------------------------------


def test_a_success_is_read_as_a_json_object() -> None:
    response = asyncio.run(_post(_leaf(_respond(200, {"answers": {"a": [1, 2]}}))))
    assert isinstance(response, tuple)
    http, body = response  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(http, httpx2.Response)
    assert body == {"answers": {"a": (1, 2)}}


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[1, 2]", b'"answers"', b'{"answers": NaN}'],
)
def test_a_body_that_cannot_be_read_is_not_retryable(body: bytes) -> None:
    error = _error(_respond(200, body, {"x-req": "r-9"}))
    assert not error.is_retryable
    assert error.status_code == 200
    assert error.request_id == "r-9"


def test_json_loads_freezes_arrays() -> None:
    assert json_loads(b'{"a": [1, {"b": [2]}]}') == {"a": (1, {"b": (2,)})}


# ----------------------------------------------------------------------------------------
# What is sent, and the client's lifecycle
# ----------------------------------------------------------------------------------------


def test_each_request_carries_the_user_agent_and_the_leafs_headers() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_OK)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), headers={"User-Agent": "callers/1"}
    )
    leaf = HttpLeaf(base_url="https://wire.test/", request_id_header="x-req", client=client)
    asyncio.run(leaf._post("/v1/x", b'{"q": 1}', Deadline.after(5), headers={"X-Mine": "1"}))
    [request] = seen
    assert request.headers["user-agent"] == f"idia/{idia.__version__}"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-mine"] == "1"
    assert str(request.url) == "https://wire.test/v1/x"
    assert json.loads(request.content) == {"q": 1}
    assert client.headers["user-agent"] == "callers/1"


def test_a_created_client_has_no_transport_retries_and_is_closed() -> None:
    leaf = HttpLeaf(base_url="https://wire.test", request_id_header="x-req")
    assert leaf._created
    transport = leaf.client._transport  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    assert isinstance(transport, httpx2.AsyncHTTPTransport)
    assert transport._pool._retries == 0  # pyright: ignore[reportUnknownMemberType]
    asyncio.run(leaf.aclose())
    assert leaf.client.is_closed


def test_an_injected_client_is_left_open() -> None:
    leaf = _leaf(_respond(200))
    assert not leaf._created

    async def run() -> None:
        async with leaf as entered:
            assert entered is leaf

    asyncio.run(run())
    assert not leaf.client.is_closed


def test_async_with_closes_a_created_client() -> None:
    leaf = HttpLeaf(base_url="https://wire.test", request_id_header="x-req")

    async def run() -> None:
        async with leaf:
            pass

    asyncio.run(run())
    assert leaf.client.is_closed


# ----------------------------------------------------------------------------------------
# The client and the event loop
# ----------------------------------------------------------------------------------------


def _created_over(
    monkeypatch: pytest.MonkeyPatch, handler: Handler
) -> tuple[HttpLeaf, list[httpx2.AsyncClient]]:
    """A leaf that creates its own clients, each over a mock transport, kept in order."""
    made: list[httpx2.AsyncClient] = []

    def over_the_mock() -> httpx2.AsyncClient:
        made.append(httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
        return made[-1]

    monkeypatch.setattr("idia.providers.http.created_client", over_the_mock)
    return HttpLeaf(base_url="https://wire.test", request_id_header="x-req"), made


def test_a_created_client_follows_the_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    leaf, made = _created_over(monkeypatch, _respond(200))
    for _ in range(2):
        assert isinstance(asyncio.run(_post(leaf)), tuple)
    assert len(made) == 2
    assert leaf.client is made[1]
    assert not made[0].is_closed


def test_a_created_client_is_kept_across_calls_on_one_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf, made = _created_over(monkeypatch, _respond(200))

    async def run() -> None:
        await asyncio.gather(_post(leaf), _post(leaf))
        await _post(leaf)

    asyncio.run(run())
    assert made == [leaf.client]


def test_an_injected_client_on_another_loop_is_a_non_retryable_error() -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=_OK)

    leaf = _leaf(handler)
    assert isinstance(asyncio.run(_post(leaf)), tuple)
    raised = asyncio.run(_post(leaf))
    assert isinstance(raised, ProviderError), raised
    assert raised.message == OTHER_LOOP
    assert not raised.is_retryable
    assert raised.status_code is None
    assert len(sent) == 1
    assert not leaf.client.is_closed


def test_aclose_closes_only_the_current_client(monkeypatch: pytest.MonkeyPatch) -> None:
    leaf, made = _created_over(monkeypatch, _respond(200))
    asyncio.run(_post(leaf))

    async def run() -> None:
        await _post(leaf)
        await leaf.aclose()
        await leaf.aclose()

    asyncio.run(run())
    assert [client.is_closed for client in made] == [False, True]


def test_aclose_on_another_loop_drops_the_client_without_awaiting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf, made = _created_over(monkeypatch, _respond(200))
    asyncio.run(_post(leaf))
    asyncio.run(leaf.aclose())
    assert made == [leaf.client]
    assert not made[0].is_closed


@pytest.mark.parametrize("loops", [1, 2])
def test_a_call_after_aclose_is_a_non_retryable_error(
    monkeypatch: pytest.MonkeyPatch, loops: int
) -> None:
    leaf, made = _created_over(monkeypatch, _respond(200))

    async def call_and_close() -> object:
        await _post(leaf)
        await leaf.aclose()
        return await _post(leaf) if loops == 1 else None

    raised = asyncio.run(call_and_close())
    if loops == 2:
        raised = asyncio.run(_post(leaf))
    assert isinstance(raised, ProviderError), raised
    assert raised.message == "the leaf is closed"
    assert not raised.is_retryable
    assert len(made) == 1


def test_an_injected_leaf_after_aclose_is_a_non_retryable_error_and_its_client_stays_open() -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=_OK)

    leaf = _leaf(handler)

    async def call_close_and_call() -> object:
        await _post(leaf)
        await leaf.aclose()
        return await _post(leaf)

    raised = asyncio.run(call_close_and_call())
    assert isinstance(raised, ProviderError), raised
    assert raised.message == "the leaf is closed"
    assert not raised.is_retryable
    assert len(sent) == 1
    assert not leaf.client.is_closed


def test_client_is_read_only() -> None:
    leaf = _leaf(_respond(200))
    with pytest.raises(AttributeError):
        leaf.client = httpx2.AsyncClient()  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]


def test_the_base_reads_any_json_object_and_leaves_its_fields_to_the_leaf() -> None:
    response = asyncio.run(_post(_leaf(_respond(200, {"model": "m"}))))
    assert isinstance(response, tuple)
    assert response[1] == {"model": "m"}  # pyright: ignore[reportUnknownArgumentType]


# ----------------------------------------------------------------------------------------
# The functions a leaf that does not inherit the base reuses
# ----------------------------------------------------------------------------------------


def test_status_error_is_the_table_without_a_response() -> None:
    error = status_error(503, {"retry-after-ms": "200"}, b'{"error": "busy"}', "r-5")
    assert (error.status_code, error.is_retryable, error.retry_after) == (503, True, 0.2)
    assert error.request_id == "r-5"
    assert error.message == 'HTTP 503: "busy"'
    assert not status_error(422, {}, b"", None).is_retryable


def test_retry_after_reads_a_plain_mapping() -> None:
    assert retry_after({"retry-after-ms": "1500", "retry-after": "7"}) == 1.5
    assert retry_after({"retry-after": "7"}) == 7.0
    assert retry_after({}) is None


def test_created_client_has_no_retries_and_no_timeout() -> None:
    client = created_client()
    transport = client._transport  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    assert isinstance(transport, httpx2.AsyncHTTPTransport)
    assert transport._pool._retries == 0  # pyright: ignore[reportUnknownMemberType]
    assert client.timeout.read is None
    asyncio.run(client.aclose())


def test_within_returns_what_ends_in_time() -> None:
    async def answer() -> int:
        return 7

    assert asyncio.run(within(Deadline.after(5), answer())) == 7


def test_within_a_passed_deadline_awaits_nothing() -> None:
    started: list[bool] = []

    async def never() -> None:  # pragma: no cover
        started.append(True)

    async def run() -> None:
        await within(Deadline.after(-1), never())

    with pytest.raises(ProviderError) as raised:
        asyncio.run(run())
    assert raised.value.is_retryable
    assert started == []


def test_within_a_deadline_within_one_clock_tick_awaits_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idia.core, "_CLOCK_TICK", 60.0)

    async def run() -> None:
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        await within(Deadline.after(5), future)

    with pytest.raises(ProviderError, match="before the call was sent"):
        asyncio.run(run())


def test_within_a_deadline_reached_is_retryable() -> None:
    async def stall() -> None:
        await asyncio.sleep(10)

    with pytest.raises(ProviderError) as raised:
        asyncio.run(within(Deadline.after(0.05), stall()))
    assert raised.value.is_retryable
    assert "deadline" in raised.value.message


def test_within_lets_other_errors_through() -> None:
    async def fail() -> None:
        raise httpx2.ConnectError("refused")

    with pytest.raises(httpx2.ConnectError):
        asyncio.run(within(Deadline.after(5), fail()))


def test_within_a_passed_deadline_refuses_any_awaitable() -> None:
    async def run() -> None:
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        await within(Deadline.after(-1), future)

    with pytest.raises(ProviderError):
        asyncio.run(run())


def test_an_empty_request_id_header_is_no_request_id() -> None:
    leaf = HttpLeaf(base_url="https://wire.test/", request_id_header="x-req")
    assert leaf.request_id(httpx2.Response(200, headers={"x-req": ""})) is None
    assert leaf.request_id(httpx2.Response(200, headers={"x-req": "r-1"})) == "r-1"
    assert leaf.request_id(httpx2.Response(200)) is None
