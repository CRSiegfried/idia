"""``HttpLeaf``: what every leaf over a raw HTTP API shares.

A leaf may inherit it as a convenience; satisfying :class:`~idia.core.Leaf` is
structural and nothing depends on the base. It owns the HTTP client's lifecycle, the user
agent, the one-attempt POST under the caller's deadline, reading a body as a whole, and the
status table, :func:`status_error`, behind :meth:`HttpLeaf.map_status`, which a leaf may
override:

- 408, 409, 429 and every 5xx are retryable, as are connection errors and timeouts;
- every other status is not;
- ``retry_after`` is read from ``retry-after-ms`` first, then ``retry-after`` as seconds or
  an HTTP-date;
- the request id comes from the header the leaf names.

A client the leaf creates, :func:`created_client`, makes no transport-level retries and is
closed by :meth:`HttpLeaf.aclose`; an injected client is used as it is and never closed
here. A client is bound to the event loop it is first used on: a created one follows the
running loop and an injected one is refused on another, by :func:`follow_the_loop`; a
client that reports itself closed is refused before sending.
:func:`within`, :func:`status_error`, :func:`retry_after`, :func:`follow_the_loop`,
:func:`malformed_request` and :func:`client_failed` serve a leaf that does not inherit the
base. A request the client refuses as malformed before sending is not retried.
"""

from __future__ import annotations

import asyncio
import datetime
import email.utils
import inspect
import json
import math
from collections.abc import Awaitable, Mapping
from types import TracebackType
from typing import Final, Self, TypeVar, cast

import httpx2

from idia import __version__
from idia.core import (
    Deadline,
    JsonValue,
    ProviderError,
    _without_keys,  # pyright: ignore[reportPrivateUsage]
)

_USER_AGENT: Final = f"idia/{__version__}"
_RETRYABLE: Final = frozenset({408, 409, 429})
_MESSAGE_CHARS: Final = 300
_T = TypeVar("_T")
OTHER_LOOP: Final = (
    "this client was first used on another event loop; "
    "pass a client per loop or let the leaf create one"
)


def created_client() -> httpx2.AsyncClient:
    """A client as an HTTP leaf creates one: no transport-level retries and no timeout of
    its own, so the caller's deadline is the only timeout."""
    return httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(retries=0), timeout=None)


def follow_the_loop(
    bound: asyncio.AbstractEventLoop | None, *, created: bool, closed: bool
) -> tuple[asyncio.AbstractEventLoop, bool]:
    """The running event loop for a leaf's client, and whether to make the client afresh.

    ``bound`` is the loop the client was last used on, ``None`` before its first use. A
    client the leaf ``created`` follows the running loop: on another loop the answer is
    ``True``, and the leaf drops the old client unclosed, since its loop may be gone, and
    makes a fresh one. An injected client on another loop raises a non-retryable
    ``ProviderError`` before anything is sent, as does a ``closed`` leaf.
    """
    if closed:
        raise ProviderError("the leaf is closed")
    loop = asyncio.get_running_loop()
    if bound is None or bound is loop:
        return loop, False
    if not created:
        raise ProviderError(OTHER_LOOP)
    return loop, True


def on_this_loop(bound: asyncio.AbstractEventLoop | None) -> bool:
    """Whether a client last used on ``bound`` may be closed on the running loop: it was
    never used, or used on this one."""
    return bound is None or bound is asyncio.get_running_loop()


_UNQUOTED: Final = (httpx2.LocalProtocolError, UnicodeEncodeError)
"""Refusals whose text quotes the header or the value, so their message names the class."""

_MALFORMED: Final = (*_UNQUOTED, httpx2.UnsupportedProtocol, httpx2.InvalidURL)
"""Every refusal of a request as malformed before anything is sent: the one list
:func:`malformed_request` reads and :meth:`HttpLeaf._post` catches."""


def malformed_request(error: BaseException) -> ProviderError | None:
    """The non-retryable error for a request the client refuses before anything is sent
    because it is malformed, or ``None`` for any other error: a retry repeats the refusal,
    and it has no status for the credential stop to read. Each is named by its class. An
    illegal header value, or a value that cannot be encoded, keeps no text, since its text
    quotes the header or the string; a URL with no or another scheme, or one that cannot be
    parsed, keeps its text, which names the scheme or the reason."""
    if not isinstance(error, _MALFORMED):
        return None
    if isinstance(error, _UNQUOTED):
        return ProviderError(f"the request is malformed: {type(error).__name__}")
    return ProviderError(f"the request is malformed: {type(error).__name__}: {error}")


def client_failed(error: RuntimeError) -> ProviderError:
    """The retryable error for a bare ``RuntimeError`` the client let out mid-call, the
    backstop for a connection that died under the client, as from a pooled connection
    whose event loop has closed; a closed client is refused before sending instead, and a
    subclass, a bug that fails the same way each time, is not passed here. The call may
    have reached the wire; the next one, on a connection made afresh, may not fail."""
    return ProviderError(f"the client failed: {type(error).__name__}: {error}", is_retryable=True)


async def within(deadline: Deadline, awaitable: Awaitable[_T]) -> _T:
    """Await ``awaitable`` so that it ends by ``deadline``.

    A deadline already passed, as ``Deadline.passed()`` reads it, raises a retryable
    ``ProviderError`` before anything is awaited; one reached while waiting does the same,
    the ``TimeoutError`` caught outside the timeout block. Every other exception propagates.
    """
    if deadline.passed():
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise ProviderError("the deadline passed before the call was sent", is_retryable=True)
    try:
        async with asyncio.timeout(deadline.remaining()):
            return await awaitable
    except TimeoutError:
        raise ProviderError("the call did not end by its deadline", is_retryable=True) from None


def status_error(
    status: int, headers: Mapping[str, str], content: bytes, request_id: str | None
) -> ProviderError:
    """The status table: the error for a response whose status is not a success.

    408, 409, 429 and every 5xx are retryable and every other status is not;
    ``retry_after`` is read by :func:`retry_after`; the message names the body's
    ``error``, ``message`` or ``detail``, cut short, and the body itself is not kept.
    """
    detail = _detail(content)
    return ProviderError(
        f"HTTP {status}" + (f": {detail}" if detail else ""),
        is_retryable=status in _RETRYABLE or status >= 500,
        retry_after=retry_after(headers),
        status_code=status,
        request_id=request_id,
    )


class HttpLeaf:
    """The base an HTTP leaf may inherit: a client, a status table and one POST per call.

    ``client`` is an injected ``httpx2.AsyncClient``, used as it is and left open, or
    ``None``, in which case :func:`created_client` makes one, closed by :meth:`aclose`.
    A created client follows the running event loop; an injected one is used on the loop
    it was first used on (:func:`follow_the_loop`). ``request_id_header`` names the
    response header that carries the wire's request id.
    """

    _created: bool
    _loop: asyncio.AbstractEventLoop | None
    _closed: bool

    def __init__(
        self,
        *,
        base_url: str,
        request_id_header: str,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._request_id_header = request_id_header
        self._created = client is None
        self._client = client if client is not None else created_client()
        self._loop = None
        self._closed = False

    @property
    def client(self) -> httpx2.AsyncClient:
        """The HTTP client the leaf sends with, injected or created: for a created client,
        the one for the event loop the leaf last ran on."""
        return self._client

    def _client_for_this_loop(self) -> httpx2.AsyncClient:
        """The client to send with on the running event loop, by :func:`follow_the_loop`:
        a created client made afresh on another loop, the old one dropped unclosed; an
        injected client on another loop, a closed leaf, or a client that reports itself
        closed, a non-retryable ``ProviderError`` before anything is sent."""
        loop, fresh = follow_the_loop(self._loop, created=self._created, closed=self._closed)
        if fresh:
            self._client = created_client()
        if self._client.is_closed:
            raise ProviderError("the client is closed")
        self._loop = loop
        return self._client

    async def aclose(self) -> None:
        """Close the leaf, and its client if the leaf created it; an injected client is left
        open.

        Safe to call again; a call after it raises ``the leaf is closed``, and a leaf whose
        client it created creates no further client. A created client last used on another
        event loop is dropped, not awaited, since that loop may be closed; its connections
        close when it is collected.
        """
        if self._closed:
            return
        self._closed = True
        if self._created and on_this_loop(self._loop):
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def map_status(self, response: httpx2.Response) -> ProviderError:
        """The error for a response whose status is not a success: :func:`status_error`.

        A leaf overrides this to reclassify a status, such as a quota exhausted arriving
        as a 429, and may call it for the rest.
        """
        return status_error(
            response.status_code, response.headers, response.content, self.request_id(response)
        )

    def request_id(self, response: httpx2.Response) -> str | None:
        """The wire's request id, from the header the leaf named."""
        return response.headers.get(self._request_id_header)

    async def _post(
        self,
        path: str,
        json: bytes,
        deadline: Deadline,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[httpx2.Response, Mapping[str, JsonValue]]:
        """Send ``json``, already encoded, once, ending by the deadline; return the
        response and its body read as a JSON object.

        A status other than a success raises :meth:`map_status`'s error; a connection
        error or a timeout, the deadline's included, a retryable ``ProviderError``; a body
        that is not a JSON object a non-retryable one. The client is the one for the running
        event loop (:meth:`_client_for_this_loop`), refused before sending when closed; a
        bare ``RuntimeError`` it lets out mid-call is :func:`client_failed`'s retryable
        error, and a subclass propagates.
        """
        client = self._client_for_this_loop()
        sent = {
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/json",
            **(headers or {}),
        }
        try:
            response = await within(
                deadline, client.post(self._base_url + path, content=json, headers=sent)
            )
        except httpx2.TimeoutException as error:
            raise ProviderError(f"the transport timed out: {error}", is_retryable=True) from None
        except _MALFORMED as error:
            raise cast(ProviderError, malformed_request(error)) from None
        except httpx2.TransportError as error:
            raise ProviderError(
                f"the connection failed: {type(error).__name__}: {error}", is_retryable=True
            ) from None
        except httpx2.RequestError as error:
            raise ProviderError(f"the request failed: {type(error).__name__}: {error}") from None
        except RuntimeError as error:  # httpx2 lets a bare one out, as from a closed loop's socket
            if type(error) is not RuntimeError:
                raise
            raise client_failed(error) from None
        if not response.is_success:
            raise self.map_status(response)
        return response, self._read(response)

    def _read(self, response: httpx2.Response) -> Mapping[str, JsonValue]:
        """The body as a JSON object, or a non-retryable error."""
        return read_object(response, self.request_id(response))

    def _unreadable(self, response: httpx2.Response, why: str) -> ProviderError:
        """The non-retryable error for a success whose body cannot be read as a whole."""
        return unreadable(response, why, self.request_id(response))


def read_object(response: httpx2.Response, request_id: str | None) -> Mapping[str, JsonValue]:
    """A success's body as a JSON object, read by :func:`json_loads`, or the
    non-retryable :func:`unreadable` error carrying ``request_id``."""
    try:
        body = json_loads(response.content)
    except ValueError:
        raise unreadable(response, "the body is not JSON", request_id) from None
    if not isinstance(body, Mapping):
        raise unreadable(response, "the body is not a JSON object", request_id)
    return cast(Mapping[str, JsonValue], body)


def unreadable(response: httpx2.Response, why: str, request_id: str | None) -> ProviderError:
    """The non-retryable error for a success whose body cannot be read as a whole."""
    return ProviderError(
        f"HTTP {response.status_code}, but {why}",
        is_retryable=False,
        status_code=response.status_code,
        request_id=request_id,
    )


def json_loads(content: bytes) -> JsonValue:
    """JSON as :data:`~idia.core.JsonValue`: arrays become tuples, and a non-finite
    number is refused."""

    def refuse(constant: str) -> float:
        raise ValueError(f"{constant} is not a JSON number")

    def frozen(value: object) -> JsonValue:
        if isinstance(value, list):
            return tuple(frozen(item) for item in cast(list[object], value))
        if isinstance(value, dict):
            return {key: frozen(item) for key, item in cast(dict[str, object], value).items()}
        return cast(JsonValue, value)

    return frozen(json.loads(content, parse_constant=refuse))


def retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait: ``retry-after-ms`` first, then ``retry-after`` as seconds or an
    HTTP-date; ``None`` when neither header is present or readable."""
    milliseconds = _seconds(headers.get("retry-after-ms"))
    if milliseconds is not None:
        return milliseconds / 1000
    value = headers.get("retry-after")
    if value is None:
        return None
    seconds = _seconds(value)
    if seconds is not None:
        return seconds
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)
    return max(0.0, (when - datetime.datetime.now(datetime.UTC)).total_seconds())


def _seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _detail(content: bytes) -> str:
    """A short message from an error body: its ``error``, ``message`` or ``detail``, as the
    wire put it, cut to a few hundred characters; the body itself is not kept. Key-shaped
    text is filtered before the cut, so no piece of a key survives it."""
    try:
        body: object = json.loads(content)
    except ValueError:
        return _without_keys(content.decode(errors="replace"))[:_MESSAGE_CHARS].strip()
    if not isinstance(body, dict):
        return ""
    fields = cast(dict[str, object], body)
    for name in ("error", "message", "detail"):
        if name in fields:
            text = _without_keys(
                json.dumps(fields[name], ensure_ascii=False, separators=(",", ":"))
            )
            return text if len(text) <= _MESSAGE_CHARS else text[:_MESSAGE_CHARS] + "..."
    return ""
