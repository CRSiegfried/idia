"""What the two OpenAI leaves share: the ``openai`` client's lifecycle, the one mapper from
the SDK's errors to :class:`~idia.core.ProviderError`, the reading of a raw body, and
the token estimator.

The SDK's own retries are always zero: an injected ``AsyncOpenAI`` is used through
``with_options(max_retries=0)`` and never closed here; one the leaf creates rides on
:func:`~idia.providers.http.created_client`, which has no transport-level retries,
and is closed by ``aclose``. A created client follows the running event loop and an
injected one is refused on another, by :func:`~idia.providers.http.follow_the_loop`.
Each call is awaited under
:func:`~idia.providers.http.within`, so the caller's deadline is its timeout.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import TracebackType
from typing import Final, Self, cast

import httpx2
import openai
from openai import AsyncOpenAI

from idia.core import (
    Deadline,
    JsonPart,
    JsonValue,
    ProviderError,
    Question,
    State,
    TextPart,
    TokenEstimate,
    Usage,
)
from idia.providers import http
from idia.providers.http import (
    client_failed,
    created_client,
    follow_the_loop,
    on_this_loop,
    status_error,
    within,
)

API_KEY_ENV: Final = "OPENAI_API_KEY"
REQUEST_ID_HEADER: Final = "x-request-id"
_QUOTA_CODE: Final = "insufficient_quota"


class OpenAILeafBase:
    """The client lifecycle both OpenAI leaves inherit.

    ``client`` is an injected ``AsyncOpenAI``, used through a copy whose retries are zero
    and never closed by the leaf, or ``None``, in which case the leaf creates one with the
    key given or ``OPENAI_API_KEY``, closed by :meth:`aclose`. A created client follows the
    running event loop; an injected one is used on the loop it was first used on.
    """

    _loop: asyncio.AbstractEventLoop | None = None
    _closed: bool = False

    def __init__(self, *, api_key: str | None, client: AsyncOpenAI | None, who: str) -> None:
        if client is not None:
            self._created = False
            self._client = client.with_options(max_retries=0)
            return
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"{who} needs an api_key, an injected client or {API_KEY_ENV}")
        self._created = True
        self._key = key
        self._client = self._create()

    def _create(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self._key, http_client=created_client(), max_retries=0)

    @property
    def client(self) -> AsyncOpenAI:
        """The ``AsyncOpenAI`` client the leaf sends with, its retries at zero: for a created
        client, the one for the event loop the leaf last ran on."""
        return self._client

    async def _send(
        self, deadline: Deadline, make_call: Callable[[], Awaitable[object]]
    ) -> httpx2.Response:
        """:func:`send`, on the client for the running event loop: a created client made
        afresh on another loop, the old one dropped unclosed; an injected client on another
        loop, a leaf after :meth:`aclose`, or a client whose ``is_closed()`` is true, a
        non-retryable ``ProviderError`` before any SDK call."""
        loop, fresh = follow_the_loop(self._loop, created=self._created, closed=self._closed)
        if fresh:
            self._client = self._create()
        if self._client.is_closed():
            raise ProviderError("the client is closed")
        self._loop = loop
        return await send(deadline, make_call)

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
            await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


async def send(deadline: Deadline, make_call: Callable[[], Awaitable[object]]) -> httpx2.Response:
    """Make one SDK call under the deadline and return its raw HTTP response.

    ``make_call`` starts a ``with_raw_response`` call. A status that is not a success is
    read by :func:`status_error`, a 429 whose error code is ``insufficient_quota`` made not
    retryable; a connection error or timeout is retryable, but a request the client refuses
    before sending because it is malformed is not (:func:`http.malformed_request`); any
    other SDK error is not retryable. A bare ``RuntimeError`` the SDK lets through mid-call,
    the backstop for a connection that died under the client, is :func:`client_failed`'s
    retryable error, and a subclass propagates.
    """
    try:
        raw = await within(deadline, make_call())
    except openai.APIStatusError as error:
        raise map_status_error(error) from None
    except UnicodeEncodeError as error:  # a header value the SDK cannot encode, before sending
        raise cast(ProviderError, http.malformed_request(error)) from None
    except openai.APIConnectionError as error:  # APITimeoutError is a subclass
        malformed = http.malformed_request(error.__cause__) if error.__cause__ else None
        if malformed is not None:
            raise malformed from None
        raise ProviderError(f"the connection failed: {error}", is_retryable=True) from None
    except openai.OpenAIError as error:
        raise ProviderError(f"the SDK failed: {type(error).__name__}: {error}") from None
    except RuntimeError as error:  # the SDK lets a bare one through, as from a closed loop's socket
        if type(error) is not RuntimeError:
            raise
        raise client_failed(error) from None
    response = getattr(raw, "http_response", None)
    if not isinstance(response, httpx2.Response):
        raise ProviderError("the SDK returned no raw HTTP response")
    return response


def map_status_error(error: openai.APIStatusError) -> ProviderError:
    """The status table over the SDK's error, a quota exhausted arriving as 429 made not
    retryable."""
    response = error.response
    mapped = status_error(error.status_code, response.headers, response.content, error.request_id)
    if error.status_code == 429 and error.code == _QUOTA_CODE:
        return ProviderError(
            f"{mapped.message} (quota exhausted)",
            is_retryable=False,
            retry_after=mapped.retry_after,
            status_code=429,
            request_id=mapped.request_id,
        )
    return mapped


def unreadable(response: httpx2.Response, why: str) -> ProviderError:
    """:func:`~idia.providers.http.unreadable` with OpenAI's request id."""
    return http.unreadable(response, why, response.headers.get(REQUEST_ID_HEADER))


def read_body(response: httpx2.Response) -> Mapping[str, JsonValue]:
    """:func:`~idia.providers.http.read_object` with OpenAI's request id."""
    return http.read_object(response, response.headers.get(REQUEST_ID_HEADER))


def count(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def usage(raw: Mapping[str, JsonValue], input_name: str, output_name: str) -> Usage:
    """Usage normalized from the wire's names, the raw figures kept beside it."""
    return Usage(
        input_tokens=count(raw.get(input_name)),
        output_tokens=count(raw.get(output_name)),
        raw_usage=raw,
    )


def plain(value: object) -> dict[str, JsonValue]:
    """A read-only mapping in a caller's JSON part, as a dict ``json`` can encode."""
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, JsonValue], value))
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def compact(value: JsonValue) -> str:
    """A JSON part as the text sent: compact, raw Unicode, no non-finite number."""
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=plain
        )
    except (ValueError, TypeError, RecursionError) as error:
        raise ProviderError(f"the state cannot be sent as JSON: {error}") from None


def _measured(value: object) -> int:
    return len(json.dumps(value, default=lambda v: repr(v)))


class OpenAIEstimator:
    """Characters of JSON per token: ``overhead`` tokens, the state's text at
    ``state_chars`` characters a token, ``per_question`` tokens for each question and its
    JSON at ``question_chars``, and ``image_tokens`` per image, with ``margin`` the
    fraction preflight adds.

    A text part is measured as its text, a JSON part as its JSON, and each question as its
    kind, instruction and labels written as JSON, so text outside ASCII is over-counted, on
    the safe side.
    """

    def __init__(
        self,
        *,
        overhead: int,
        per_question: int,
        state_chars: int,
        question_chars: int,
        image_tokens: int,
        margin: float,
    ) -> None:
        self.overhead = overhead
        self.per_question = per_question
        self.state_chars = state_chars
        self.question_chars = question_chars
        self.image_tokens = image_tokens
        self.margin = margin

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        state_chars = 0
        images = 0
        for part in state.parts:
            if isinstance(part, TextPart):
                state_chars += _measured(part.text)
            elif isinstance(part, JsonPart):
                state_chars += _measured(part.value)
            else:
                images += 1
        tokens = (
            self.overhead + math.ceil(state_chars / self.state_chars) + images * self.image_tokens
        )
        if questions:
            asked = [
                [q.kind, q.instruction, [[label.name, label.description] for label in q.labels]]
                for q in questions
            ]
            tokens += len(questions) * self.per_question
            tokens += math.ceil(_measured(asked) / self.question_chars)
        return TokenEstimate(tokens=tokens, margin=self.margin)
