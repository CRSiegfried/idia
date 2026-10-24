"""A leaf on an ``httpx2`` client, for testing the HTTP harness and nothing else.

It POSTs the wire request as JSON and reads a trivial wire: a body holding the reported
model, the answers in the test doubles' encoding and a usage object. It maps statuses as
an HTTP leaf does, reads ``retry-after-ms`` before ``retry-after``, takes the request id
from ``x-request-id``, closes only a client it created, refuses a call once it or its
client is closed, and computes no complement.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from typing import Final, cast

import httpx2

from idia.answers import validate_answers
from idia.core import (
    CANDIDATE_KINDS,
    WARNING_NO_SNAPSHOT,
    Answer,
    Capabilities,
    Deadline,
    Fault,
    MappedAnswer,
    Notice,
    ProviderError,
    Rule,
    RuleFailure,
    TokenEstimator,
    Usage,
    WireRequest,
    WireResponse,
    unknown_options,
)
from idia.providers._canon import CharacterEstimator, decode, encode, plain, thaw

URL: Final = "https://stub.invalid/v1/answer"
SECRET: Final = "Bearer stub-secret"


class StubHttpLeaf:
    """The stub leaf over ``client``, or over a client it creates when given none."""

    def __init__(
        self,
        *,
        capabilities: Capabilities,
        client: httpx2.AsyncClient | None = None,
        model: str = "stub-1",
        retries: int = 0,
    ) -> None:
        self.name = "stub"
        self.model = model
        self.capabilities = capabilities
        self.estimator: TokenEstimator = CharacterEstimator()
        self._created = client is None
        self._closed = False
        self._client = (
            client
            if client is not None
            else httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(retries=retries))
        )

    @property
    def client(self) -> httpx2.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        """Close the client, if this leaf created it; every later call is refused."""
        self._closed = True
        if self._created:
            await self._client.aclose()

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        if self._closed:
            raise ProviderError("the leaf is closed")
        if self._client.is_closed:
            raise ProviderError("the client is closed")
        # The stub's wire knows no options, so none reach its body.
        sent = dataclasses.replace(wire_request, options={})
        payload = plain({"model": self.model, "request": encode(sent)})
        try:
            response = await asyncio.wait_for(
                self._client.post(URL, json=payload, headers={"authorization": SECRET}),
                max(deadline.remaining(), 0.0),
            )
        except TimeoutError:
            raise ProviderError(
                "the stub wire stalled past the deadline", is_retryable=True
            ) from None
        except httpx2.TransportError as error:
            raise ProviderError(f"the stub wire failed: {error}", is_retryable=True) from None
        request_id = response.headers.get("x-request-id")
        status = response.status_code
        if status >= 400:
            raise ProviderError(
                f"the stub wire answered {status}",
                is_retryable=status in (408, 409, 429) or status >= 500,
                retry_after=_retry_after(response.headers),
                status_code=status,
                request_id=request_id,
            )
        body = _read(response, request_id)
        answers_body = cast(Mapping[str, object], body["answers"])
        mapped = {key: _map(key, value) for key, value in answers_body.items()}
        rounding = self.capabilities.rounding
        answers, warnings = validate_answers(wire_request.questions, mapped, rounding)
        model_reported = cast(str, body["model"])
        snapshot = model_reported if self.capabilities.snapshot_pinnable else None
        if snapshot is None:
            warnings += (Notice(code=WARNING_NO_SNAPSHOT, message="no snapshot", field="snapshot"),)
        usage = cast(Mapping[str, object], body.get("usage", {}))
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self.capabilities.calibrated,
            model_reported=model_reported,
            snapshot=snapshot,
            prompt_sent=response.request.content.decode(),
            usage=Usage(
                input_tokens=cast(int | None, usage.get("input_tokens")),
                output_tokens=cast(int | None, usage.get("output_tokens")),
                raw_usage=cast(Mapping[str, object], thaw(usage)),  # type: ignore[arg-type]
            ),
            request_id=request_id,
            raw_body=thaw(body),
            warnings=warnings + unknown_options(wire_request.options, frozenset()),
        )


def _read(response: httpx2.Response, request_id: str | None) -> Mapping[str, object]:
    """The body as a whole, or a non-retryable error naming what could not be read."""
    try:
        body: object = response.json()
    except ValueError:
        body = None
        problem = "is not JSON"
    else:
        problem = None
    if problem is None and not (isinstance(body, dict) and "model" in body):
        problem = "fails the schema"
    if problem is None and not isinstance(cast(Mapping[str, object], body).get("answers"), dict):
        problem = "holds no answers"
    if problem is not None:
        raise ProviderError(
            f"the stub wire's body {problem}",
            status_code=response.status_code,
            request_id=request_id,
        )
    return cast(Mapping[str, object], body)


def _map(key: str, value: object) -> MappedAnswer:
    shape = cast(Mapping[str, object], value).get("type") if isinstance(value, dict) else None
    if shape in CANDIDATE_KINDS:
        failure = RuleFailure(rule=Rule.candidate_shape, observed=cast(str, shape), bound=None)
        return Fault(key=key, failures=(failure,), mapped=None)
    answer: Answer = decode(cast(object, value), Answer)
    return answer


def _retry_after(headers: httpx2.Headers) -> float | None:
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None:
        return float(milliseconds) / 1000
    seconds = headers.get("retry-after")
    return float(seconds) if seconds is not None else None
