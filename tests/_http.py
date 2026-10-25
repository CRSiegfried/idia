"""The HTTP harness: the leaf half for a leaf on an ``httpx2`` client, over stored exchanges.

A leaf's tests subclass :class:`HttpLeafHarness` and fill its hooks; the base turns each
exchange the suite names into a response from an ``httpx2.MockTransport``, read from the
leaf's cassettes. A cassette is one JSON file per case, at
``tests/cassettes/<leaf>/<model>/<features>/<name>.json``::

    {"format": 2, "leaf": ..., "model": ..., "features": [...],
     "request": {"method", "url", "headers", "body"},
     "wire_request": <the WireRequest, encoded>,
     "response": {"status", "headers", "body"},
     "expected": <the answers as mapped, encoded, or null for a failure>,
     "recorded_with": <optional: {package: version}, the versions that shaped it>}

:func:`write_cassette` stores the request and response bodies verbatim, keys in the order
sent and received, the wire request and the answers in the order held, every mapping a
caller built in the order built, so sending the ``wire_request`` of a cassette its leaf
recorded through that leaf gives its stored body; the rest of the document, the envelope
and the headers, is written with sorted keys, for stable diffs. A cassette written before
bodies kept their order, as a probe recorded once and never written again, holds its
bodies with sorted keys, matches no request a leaf sends and is replayed only by name; so
does a probe sent outside the leaf. A replayed success asserts that the leaf sent the
cassette's method, URL and JSON body: the body is in the match, so two requests to one URL
cannot answer for each other. The match compares :func:`body_key`, the parsed body dumped
in the order parsed, so an object's key order is compared as well as an array's: a wire's
answers can move with the order of its questions.

Except the stub leaf's cassettes, which no wire answers and which are written by hand in
the order the stub leaf sends, a
leaf's cassette is written only by :func:`write_cassette`, under ``IDIA_LIVE=1``: for
one call through the leaf by :func:`record_cassette` and its :class:`RecordingTransport`,
for a run by :func:`write_run_cassettes`, or for a probe sent outside the leaf. The header
filter removes secrets from both sets of headers and drops the user agent and the headers
that fingerprint the account or the recording machine, :data:`DROPPED_HEADERS` and every
name starting with one of :data:`DROPPED_PREFIXES`; nothing else of a header reaches a
recorded leaf's log, since a wire response holds none.

A run of several wire requests, such as an engine deciding through a dispatcher, is
recorded by spying on its leaf with :class:`LeafSpy`, built before the dispatcher, and
writing each exchange the recording transport kept with :func:`write_run_cassettes`; it
is replayed by :class:`CassetteTransport`, which serves every cassette of one directory by
its method, URL and body, keeps each path it served, in order, and each request no
cassette matches, and names the cassettes never served.

Secrets are kept out of a committed cassette in four layers. At write time, the header
filter replaces every value of :data:`SECRET_HEADERS` in both sets of headers; ``redact``
replaces a secret that is not a header, such as Cloudflare's account id in the URL, with
its placeholder wherever it appears; and :func:`write_run_cassettes` checks each written
file for the exact value of every secret the run was given, and for
:data:`OPENAI_KEY_SHAPE`, :data:`BEARER_SHAPE` and any further shape it is given, removing
the file and failing when one is held: the exact-value check is the only one that can
catch a Cloudflare API token outside a header, since the token has no shape of its own. In
the normal suite, ``tests/test_http_harness.py`` checks every committed cassette for the
key and bearer shapes and every Cloudflare cassette for :data:`ACCOUNT_ID_SHAPE`, with no
credentials in the environment.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import abc
import asyncio
import contextlib
import json
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx2

from idia import core
from idia.conformance import (
    AnyBody,
    CandidateShape,
    CheckNotApplicable,
    Error,
    Exchange,
    Failure,
    Perturbed,
    RefusalOnWire,
    Stall,
    Success,
    Unreadable,
)
from idia.core import (
    Answer,
    Capabilities,
    Deadline,
    JsonValue,
    Kind,
    Leaf,
    ProviderError,
    TokenEstimator,
    WireRequest,
    WireResponse,
)
from idia.providers._canon import decode, encode, plain

FORMAT: Final = 2
"""The cassette format this module reads and writes. A change to ``_canon``'s rendering,
which a cassette's wire request and answers are written in, bumps it: format 2 writes them
in the order held, where format 1 sorted their keys."""

SECRET_HEADERS: Final = frozenset(
    {
        "authorization",
        "x-api-key",
        "api-key",
        "cookie",
        "set-cookie",
        "openai-organization",
        "openai-project",
    }
)
"""Headers whose values never reach a cassette. The filter covers headers only: a secret
carried in a URL or its query string is not filtered."""

FILTERED: Final = core._FILTERED

DROPPED_HEADERS: Final = frozenset(
    {
        "user-agent",
        "cf-ray",
        "cf-auditlog-id",
        "x-stainless-os",
        "x-stainless-arch",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
    }
)
"""Headers a cassette never stores: the user agent, which names the client that sent the
request, and the headers that fingerprint the account or the recording machine; no leaf
reads them."""

DROPPED_PREFIXES: Final = ("x-ratelimit-",)
"""Prefixes of header names a cassette never stores: the rate-limit headers reveal the
account's tier, and no leaf reads them."""

OPENAI_KEY_SHAPE: Final = core._OPENAI_KEY_SHAPE
"""Core's ``sk-`` key shape: a cassette holding one fails its write."""

BEARER_SHAPE: Final = core._BEARER_SHAPE
"""Core's bearer shape: a cassette holding one fails its write loudly, prose that matches
it included."""

KEY_SHAPES: Final = core._KEY_SHAPES
"""The key shapes Core filters out of every provider error's text: the cassette checks use
the same tuple, so a shape Core adds reaches them too."""

ACCOUNT_ID_SHAPE: Final = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])")
"""A Cloudflare account id: a run of exactly 32 lower-case hex characters. An OpenAI or
System One request id has the same shape, so it is checked in Cloudflare cassettes only."""

_UNSENT_HEADERS: Final = frozenset({"content-length", "content-encoding", "transfer-encoding"})
"""Headers a replayed response drops: its body is written again, decoded."""

_STALL: Final = 3600.0


def filter_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """``headers`` with lower-case names, every secret's value replaced by ``FILTERED`` and
    every dropped header left out; the request id and the status and retry headers are
    kept."""
    kept: dict[str, str] = {}
    for name, value in headers.items():
        lower = name.lower()
        if lower in DROPPED_HEADERS or lower.startswith(DROPPED_PREFIXES):
            continue
        kept[lower] = FILTERED if lower in SECRET_HEADERS else value
    return kept


def features_dir(features: frozenset[str]) -> str:
    """The cassette directory for a configuration: ``base`` with no opt-in feature on,
    otherwise the features sorted and joined by ``+``."""
    return "+".join(sorted(features)) if features else "base"


def cassette_path(root: Path, leaf: str, model: str, features: frozenset[str], name: str) -> Path:
    """Where a cassette lives: ``<root>/<leaf>/<model>/<features>/<name>.json``."""
    return root / leaf / model / features_dir(features) / f"{name}.json"


@dataclass(frozen=True, slots=True)
class Cassette:
    """One stored exchange, as read."""

    request: Mapping[str, Any]
    response: Mapping[str, Any]
    wire_request: WireRequest
    expected: Mapping[str, Answer] | None


def load_cassette(path: Path) -> Cassette:
    """Read a cassette, refusing a format this module does not write."""
    document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if document.get("format") != FORMAT:
        raise ValueError(f"{path} is in format {document.get('format')!r}; this reads {FORMAT}")
    expected = document["expected"]
    return Cassette(
        request=document["request"],
        response=document["response"],
        wire_request=decode(document["wire_request"], WireRequest),
        expected=None if expected is None else decode(expected, Mapping[str, Answer]),
    )


def parse_body(content: bytes) -> object:
    """A body as JSON where it parses, otherwise as text: how a cassette stores a body and
    how a sent body is read to match one."""
    text = content.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def body_key(body: object) -> str:
    """A parsed body as the match compares it: dumped in the order parsed, so two bodies
    equal in value but with an object's keys or an array's items in another order differ;
    it also tells ``1`` from ``1.0``, since the body is stored verbatim."""
    return json.dumps(body)


def replay_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """A stored response's headers as a replay sends them: without the length and the
    encodings, since the body is written again, decoded."""
    return {name: value for name, value in headers.items() if name not in _UNSENT_HEADERS}


def _response(status: int, headers: Mapping[str, str], body: object) -> httpx2.Response:
    content = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    return httpx2.Response(status, headers=replay_headers(headers), content=content)


# ----------------------------------------------------------------------------------------
# The harness
# ----------------------------------------------------------------------------------------


class HttpLeafHarness(abc.ABC):
    """A :class:`~idia.conformance.LeafHarness` for a leaf on an ``httpx2`` client.

    A subclass sets ``cassette_dir``, the directory of one configuration's cassettes, and
    ``failure_cassette``, the name of its recorded failure, and fills the hooks:
    ``capabilities``, the record the leaf declares; ``capabilities_for``, the record it
    must declare with given features on; ``build``, the leaf over an injected
    client, or creating its own when given ``None``; ``cassette_for``, the cassette name
    of each claimed kind's success; ``perturb``, ``unreadable``, ``refusal_body`` and
    ``candidate_body``, the wire's own forms of what the suite asks for.

    The leaf exposes its client as a read-only ``client`` attribute and closes a client it
    created, and only that one, in ``async aclose()``; the request id travels in the
    ``request_id_header`` header.
    """

    prompt_hidden: bool = False
    request_id_header: str = "x-request-id"
    cassette_dir: Path
    failure_cassette: str

    def __init__(self) -> None:
        self._sent = 0
        self._body: bytes | None = None
        self._cassettes: dict[str, Cassette] = {}

    # The hooks a leaf's harness fills ------------------------------------------------

    @property
    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """The record the leaf declares, built with its default features."""

    @abc.abstractmethod
    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        """The leaf, with ``features`` on, over ``client``; with ``None`` it makes its own."""

    @abc.abstractmethod
    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        """The record the leaf must declare with ``features`` on, stated by the test,
        not read back from the built leaf, so L7 can fail."""

    @abc.abstractmethod
    def cassette_for(self, kind: Kind) -> str:
        """The name of the success cassette for ``kind``, under ``cassette_dir``."""

    @abc.abstractmethod
    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue | None:
        """``kind``'s recorded success body with one change: one answer's sum moved outside
        tolerance, one key removed and one added, or a noul outside [0, 1]; ``None`` where
        the wire cannot be perturbed that way, which makes the check not applicable."""

    @abc.abstractmethod
    def unreadable(self, which: Literal["not_json", "schema", "no_answers"]) -> bytes:
        """A body that is not JSON, fails the top-level schema, or holds no answers."""

    def refusal_body(self, kind: Kind) -> JsonValue | None:
        """A success body for ``kind`` with a refusal in it; ``None`` where the wire has
        no refusal."""
        return None

    def candidate_body(self, kind: Kind) -> JsonValue | None:
        """A success body for ``kind`` holding a candidate shape; ``None`` where the wire
        returns none."""
        return None

    # The LeafHarness protocol ----------------------------------------------------------

    def make(self, exchanges: Sequence[Exchange], features: frozenset[str] | None = None) -> Leaf:
        transport = httpx2.MockTransport(self.handler(exchanges))
        return self.build(httpx2.AsyncClient(transport=transport), self._features(features))

    def handler(
        self, exchanges: Sequence[Exchange]
    ) -> Callable[[httpx2.Request], Coroutine[Any, Any, httpx2.Response]]:
        """The mock wire's handler playing ``exchanges`` in order, its count reset, for a
        harness that puts several leaves' handlers behind one transport, as a router over
        one shared connection does; :meth:`make` builds its transport from it."""
        bodies = [self._special(exchange) for exchange in exchanges]
        self._sent = 0
        self._body = None
        return self._handler(iter(zip(exchanges, bodies, strict=True)))

    def wire_request(self, kind: Kind) -> WireRequest:
        return self._cassette(self.cassette_for(kind)).wire_request

    def expected(self, kind: Kind) -> Mapping[str, Answer]:
        expected = self._cassette(self.cassette_for(kind)).expected
        if expected is None:
            raise ValueError(f"the {kind} cassette holds no answers")
        return expected

    def requests_sent(self) -> int:
        return self._sent

    def body_sent(self) -> bytes:
        if self._body is None:
            raise AssertionError("the leaf sent no request")
        return self._body

    def sdk_retries(self) -> int:
        """The retries of the transport under the client the leaf makes itself. It reads
        httpx2's private ``_transport`` and ``_pool._retries``, which a release of httpx2
        may rename: fragile, and the first place to look when L9 breaks on an upgrade."""
        client = _client(self.build(None, self._features(None)))
        transport = client._transport
        if not isinstance(transport, httpx2.AsyncHTTPTransport):
            raise CheckNotApplicable(f"the leaf's own client runs on a {type(transport).__name__}")
        return transport._pool._retries

    async def client_closed(self, *, injected: bool) -> bool:
        features = self._features(None)
        if injected:
            kind = sorted(self.capabilities.kinds)[0]
            leaf = self.make([Success(kind)], features)
            await leaf.call(self.wire_request(kind), Deadline.after(30.0))
        else:
            leaf = self.build(None, features)
        close = cast(Callable[[], Awaitable[None]] | None, getattr(leaf, "aclose", None))
        if close is not None:
            await close()
        return _client(leaf).is_closed

    # The mock wire ------------------------------------------------------------------

    def _features(self, features: frozenset[str] | None) -> frozenset[str]:
        return self.capabilities.enabled_features if features is None else features

    def _cassette(self, name: str) -> Cassette:
        if name not in self._cassettes:
            self._cassettes[name] = load_cassette(self.cassette_dir / f"{name}.json")
        return self._cassettes[name]

    def _special(self, exchange: Exchange) -> JsonValue | None:
        """The hook's body for a perturbed success, a refusal or a candidate shape; not
        applicable where the wire has none."""
        if isinstance(exchange, Perturbed):
            stored = self._cassette(self.cassette_for(exchange.kind)).response["body"]
            body = self.perturb(_thawed(stored), exchange.how, exchange.kind)
            if body is None:
                raise CheckNotApplicable(f"the wire cannot be perturbed by {exchange.how}")
            return body
        if isinstance(exchange, RefusalOnWire):
            body = self.refusal_body(exchange.kind)
            if body is None:
                raise CheckNotApplicable("the wire has no refusal")
            return body
        if isinstance(exchange, CandidateShape):
            body = self.candidate_body(exchange.kind)
            if body is None:
                raise CheckNotApplicable("the wire returns no candidate shape")
            return body
        return None

    def _handler(
        self, plays: Iterator[tuple[Exchange, JsonValue | None]]
    ) -> Callable[[httpx2.Request], Coroutine[Any, Any, httpx2.Response]]:
        async def handle(request: httpx2.Request) -> httpx2.Response:
            self._sent += 1
            self._body = request.content
            play = next(plays, None)
            if play is None:
                raise AssertionError(f"the harness's exchanges ran out at request {self._sent}")
            exchange, special = play
            if isinstance(exchange, Error):
                if exchange.which == "connection":
                    raise httpx2.ConnectError("conformance: connection refused", request=request)
                raise httpx2.ReadTimeout("conformance: read timed out", request=request)
            if isinstance(exchange, Stall):
                await asyncio.sleep(_STALL)
                raise AssertionError("a stalled request outlived the call's deadline")
            if isinstance(exchange, Failure):
                return self._failure(exchange)
            if isinstance(exchange, Unreadable):
                content = self.unreadable(exchange.which)
                headers = {"content-type": "application/json"}
                return httpx2.Response(200, headers=headers, content=content)
            if isinstance(exchange, AnyBody):
                cassette = self._cassette(self.cassette_for(exchange.kind))
            else:
                cassette = self._matching(exchange.kind, request)
            response = cassette.response
            body = response["body"]
            if special is not None:
                body = special
            return _response(response["status"], response["headers"], plain(body))

        return handle

    def _matching(self, kind: Kind | None, request: httpx2.Request) -> Cassette:
        """The cassette that answers ``request``: ``kind``'s, which must match it, or with
        no kind any success cassette in ``cassette_dir`` whose method, URL and body it
        matches, so a request packed differently from the claimed kinds' cassettes, one
        question per wire request, is answered by a cassette recorded for it."""
        sent = (request.method, str(request.url), parse_body(request.content))
        if kind is not None:
            cassette = self._cassette(self.cassette_for(kind))
            stored = cassette.request
            recorded = (stored["method"], stored["url"], stored["body"])
            if _key(sent) != _key(recorded):
                raise AssertionError(
                    f"the {kind} cassette does not match the request sent: body in the match, "
                    f"sent {sent!r}, recorded {recorded!r}"
                )
            return cassette
        for path in sorted(self.cassette_dir.glob("*.json")):
            cassette = self._cassette(path.stem)
            stored = cassette.request
            matched = _key((stored["method"], stored["url"], stored["body"])) == _key(sent)
            if matched and cassette.expected is not None:
                return cassette
        raise AssertionError(f"no cassette matches the request sent: {sent!r}")

    def _failure(self, exchange: Failure) -> httpx2.Response:
        response = self._cassette(self.failure_cassette).response
        if exchange == Failure():
            return _response(response["status"], response["headers"], response["body"])
        dropped = {"retry-after", "retry-after-ms", self.request_id_header}
        headers = {k: v for k, v in response["headers"].items() if k not in dropped}
        if exchange.retry_after is not None:
            headers["retry-after-ms"] = str(round(exchange.retry_after * 1000))
            headers["retry-after"] = str(round(exchange.retry_after) + 7)
        if exchange.request_id is not None:
            headers[self.request_id_header] = exchange.request_id
        status = response["status"] if exchange.status is None else exchange.status
        return _response(status, headers, response["body"])


def _key(exchange: tuple[str, str, object]) -> tuple[str, str, str]:
    """A method, URL and parsed body as the match compares them."""
    method, url, body = exchange
    return (method, url, body_key(body))


def _thawed(body: object) -> JsonValue:
    """A stored body as a ``JsonValue``, so a hook can edit a copy of it."""
    return cast(JsonValue, json.loads(json.dumps(body)))


def _client(leaf: Leaf) -> httpx2.AsyncClient:
    client = getattr(leaf, "client", None)
    if not isinstance(client, httpx2.AsyncClient):
        raise CheckNotApplicable("the leaf exposes no httpx2 client as client")
    return client


# ----------------------------------------------------------------------------------------
# Recording, under IDIA_LIVE=1 only
# ----------------------------------------------------------------------------------------


class NoExchange(AssertionError):
    """A recorded call that reached the wire not once, as when the leaf timed out or the
    connection failed and it raised :class:`~idia.core.ProviderError` with nothing
    sent or nothing read: no evidence, and no cassette written. ``count`` is the number
    of exchanges recorded, always 0."""

    count: Final = 0

    def __init__(self) -> None:
        super().__init__("one call made 0 requests: no exchange, nothing written")


class RecordingTransport(httpx2.AsyncBaseTransport):
    """Passes each request to the real transport and keeps the request and the response
    read whole, for :func:`record_cassette` to write with secrets filtered."""

    def __init__(self, inner: httpx2.AsyncBaseTransport | None = None) -> None:
        self._inner = inner if inner is not None else httpx2.AsyncHTTPTransport(retries=0)
        self.exchanges: list[tuple[httpx2.Request, httpx2.Response, bytes]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self._inner.handle_async_request(request)
        stream = cast(httpx2.AsyncByteStream, response.stream)
        raw = b"".join([chunk async for chunk in stream])
        await response.aclose()
        replay = httpx2.Response(response.status_code, headers=response.headers, content=raw)
        decoded = httpx2.Response(response.status_code, headers=response.headers, content=raw)
        self.exchanges.append((request, response, decoded.read()))
        return replay

    async def aclose(self) -> None:
        await self._inner.aclose()


async def record_cassette(
    path: Path,
    leaf_factory: Callable[[httpx2.AsyncClient], Leaf],
    wire_request: WireRequest,
    deadline: Deadline,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    redact: Mapping[str, str] | None = None,
    check: Callable[[Path, WireResponse | ProviderError], None] | None = None,
) -> WireResponse | ProviderError:
    """Make one call through a leaf built over a recording client and write its cassette
    to ``path``, with ``expected`` the answers the leaf returned, or ``null`` when it
    raised. Refuses unless ``IDIA_LIVE=1``; ``transport`` replaces the real one;
    ``redact`` is passed to :func:`write_cassette`. A call that made no exchange raises
    :class:`NoExchange`; one that made more than one, ``AssertionError``; neither writes.

    ``check(side, result)``, when given, runs before anything reaches ``path``: the
    cassette is first written to ``side``, a file of the same name in a fresh temporary
    directory outside the cassette tree, and only when ``check`` returns is it written at
    ``path`` and ``side`` removed. When ``check`` raises, ``path`` is left as it was, or
    absent, ``side`` is kept for inspection unless the check removed or moved it, and the
    error is raised again with a note naming ``path``, and ``side`` when it is kept."""
    if os.environ.get("IDIA_LIVE") != "1":
        raise RuntimeError("a cassette is recorded only in a live run: set IDIA_LIVE=1")
    recorder = RecordingTransport(transport)
    async with httpx2.AsyncClient(transport=recorder, timeout=None) as client:
        leaf = leaf_factory(client)
        result: WireResponse | ProviderError
        try:
            result = await leaf.call(wire_request, deadline)
        except ProviderError as error:
            result = error
    if not recorder.exchanges:
        raise NoExchange
    if len(recorder.exchanges) != 1:
        raise AssertionError(f"one call made {len(recorder.exchanges)} requests, not one")
    [(request, response, content)] = recorder.exchanges
    answers = None if isinstance(result, ProviderError) else result.answers
    text = _render_cassette(
        leaf=leaf,
        request=request,
        response=response,
        content=content,
        wire_request=wire_request,
        expected=answers,
        redact=redact,
    )
    if check is None:
        await asyncio.to_thread(_write_atomic, path, text)
        return result
    side = Path(tempfile.mkdtemp(prefix=f"idia-cassette-{path.stem}-")) / path.name
    await asyncio.to_thread(_write_atomic, side, text)
    try:
        check(side, result)
    except BaseException as error:
        if side.exists():
            error.add_note(f"kept at {side}; {path} untouched")
        else:
            error.add_note(f"{path} untouched")
            with contextlib.suppress(OSError):
                side.parent.rmdir()
        raise
    await asyncio.to_thread(_write_atomic, path, text)
    side.unlink(missing_ok=True)
    side.parent.rmdir()
    return result


def write_cassette(
    path: Path,
    *,
    leaf: Leaf,
    request: httpx2.Request,
    response: httpx2.Response,
    wire_request: WireRequest,
    expected: Mapping[str, Answer] | None,
    content: bytes | None = None,
    redact: Mapping[str, str] | None = None,
    recorded_with: Mapping[str, str] | None = None,
) -> None:
    """Write one exchange as a cassette, the secrets filtered from both sets of headers:
    for a probe sent outside the leaf as well as for :func:`record_cassette`. ``leaf``
    names the leaf, model and features; ``content`` is the response body decoded, by
    default the read response's own; ``expected`` the answers as mapped, or ``None`` for
    a failure. ``redact`` maps each secret that is not a header, such as an account id in
    the URL, to the placeholder written in its place wherever it appears in the cassette.
    ``recorded_with``, when given, names the versions that shaped the cassette, such as
    idia's, written as the member ``recorded_with``, which no reader
    needs. The two bodies are written verbatim, keys in the order sent and received, the wire
    request and the answers in the order held, and the rest with sorted keys. The file is
    written through a temporary file beside it, replaced in
    one step, so a failed write leaves an earlier cassette as it was, or none. Refuses
    unless ``IDIA_LIVE=1``."""
    if os.environ.get("IDIA_LIVE") != "1":
        raise RuntimeError("a cassette is recorded only in a live run: set IDIA_LIVE=1")
    text = _render_cassette(
        leaf=leaf,
        request=request,
        response=response,
        wire_request=wire_request,
        expected=expected,
        content=content,
        redact=redact,
        recorded_with=recorded_with,
    )
    _write_atomic(path, text)


def _render_cassette(
    *,
    leaf: Leaf,
    request: httpx2.Request,
    response: httpx2.Response,
    wire_request: WireRequest,
    expected: Mapping[str, Answer] | None,
    content: bytes | None = None,
    redact: Mapping[str, str] | None = None,
    recorded_with: Mapping[str, str] | None = None,
) -> str:
    """The text :func:`write_cassette` writes for one exchange."""
    body = response.content if content is None else content
    versions = {} if recorded_with is None else {"recorded_with": dict(recorded_with)}
    document = _sorted(
        {
            **versions,
            "format": FORMAT,
            "leaf": leaf.name,
            "model": leaf.model,
            "features": sorted(leaf.capabilities.enabled_features),
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": filter_headers(request.headers),
                "body": None,
            },
            "wire_request": None,
            "response": {
                "status": response.status_code,
                "headers": filter_headers(response.headers),
                "body": None,
            },
            "expected": None,
        }
    )
    document["wire_request"] = plain(encode(wire_request))
    document["expected"] = None if expected is None else plain(encode(expected))
    document["request"]["body"] = parse_body(request.content)
    document["response"]["body"] = parse_body(body)
    text = json.dumps(document, indent=2) + "\n"
    for secret, placeholder in (redact or {}).items():
        if secret:
            text = text.replace(secret, placeholder)
    return text


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` at ``path`` through a temporary file beside it, replaced in one
    step, so a failed write leaves an earlier file as it was, or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _sorted(value: Any) -> Any:
    """``value`` with every object's keys in sorted order, at every depth."""
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(cast(dict[str, Any], value))}
    if isinstance(value, list):
        return [_sorted(item) for item in cast(list[Any], value)]
    return value


# ----------------------------------------------------------------------------------------
# A run of many wire requests: spied, written, replayed
# ----------------------------------------------------------------------------------------


class LeafSpy:
    """A leaf that delegates everything to ``inner`` and keeps each call's wire request
    with the answers it returned, ``None`` for a call that raised, in ``calls``, and each
    error raised in ``errors``, in order. ``name``, ``model``, ``capabilities`` and
    ``estimator`` are the inner leaf's, and so are ``client`` and ``aclose`` where it has
    them. Wrap it before building the dispatcher, which reads the leaf's estimator once."""

    def __init__(self, inner: Leaf) -> None:
        self.inner = inner
        self.calls: list[tuple[WireRequest, Mapping[str, Answer] | None]] = []
        self.errors: list[ProviderError] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def capabilities(self) -> Capabilities:
        return self.inner.capabilities

    @property
    def estimator(self) -> TokenEstimator:
        return self.inner.estimator

    @property
    def client(self) -> object:
        """The inner leaf's client; ``AttributeError`` where it exposes none."""
        return cast(object, getattr(self.inner, "client"))  # noqa: B009

    async def aclose(self) -> None:
        """Close the inner leaf where it can be closed."""
        close = cast(Callable[[], Awaitable[None]] | None, getattr(self.inner, "aclose", None))
        if close is not None:
            await close()

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        try:
            result = await self.inner.call(wire_request, deadline)
        except ProviderError as error:
            self.calls.append((wire_request, None))
            self.errors.append(error)
            raise
        self.calls.append((wire_request, result.answers))
        return result


def write_run_cassettes(
    exchanges: Sequence[tuple[httpx2.Request, httpx2.Response, bytes]],
    calls: Sequence[tuple[WireRequest, Mapping[str, Answer] | None]],
    *,
    leaf: Leaf,
    path_for: Callable[[int, WireRequest], Path],
    redact: Mapping[str, str] | None,
    secrets: Sequence[str],
    shapes: Sequence[re.Pattern[str]] = KEY_SHAPES,
    recorded_with: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write a run's cassettes: each of a :class:`RecordingTransport`'s ``exchanges``
    paired with the :class:`LeafSpy` call it answered, one to one, at
    ``path_for(index, wire_request)``, by :func:`write_cassette` with ``redact`` and
    ``recorded_with``. Each
    written file is then read back: one that holds any of ``secrets`` verbatim, or a match
    of any of ``shapes``, fails the run. When any write or check fails, every file this
    call wrote is removed; a file it never reached, or failed to write, is left alone. A
    count that differs fails before anything is written. Returns the paths written, in
    order."""
    if len(exchanges) != len(calls):
        raise AssertionError(
            f"{len(exchanges)} wire exchanges for {len(calls)} leaf calls; nothing written"
        )
    written: list[Path] = []
    try:
        for index, ((request, response, content), (wire_request, answers)) in enumerate(
            zip(exchanges, calls, strict=True)
        ):
            path = path_for(index, wire_request)
            write_cassette(
                path,
                leaf=leaf,
                request=request,
                response=response,
                content=content,
                wire_request=wire_request,
                expected=answers,
                redact=redact,
                recorded_with=recorded_with,
            )
            written.append(path)
            text = path.read_text(encoding="utf-8")
            held = any(secret and secret in text for secret in secrets)
            shaped = [shape.pattern for shape in shapes if shape.search(text)]
            if held or shaped:
                what = "a secret given" if held else f"a value shaped {shaped[0]}"
                raise AssertionError(f"{path.name} held {what}; every file written removed")
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return written


class CassetteTransport(httpx2.MockTransport):
    """The mock wire over every cassette directly in ``directory``, not its
    subdirectories: a request is answered by the cassette whose method, URL and body it
    matches, the body read by :func:`parse_body` and compared by :func:`body_key`, so
    an object's key order is in the match, with the status, headers and body
    stored; a failure's cassette replays its failure. Two cassettes with one method, URL
    and body in one order are refused at construction; two equal in value but in
    another order are held apart. A request no cassette matches is appended to
    ``misses`` and then fails with ``AssertionError``, so a client that turns the error
    into its own still leaves the miss to be read. ``served`` keeps each path it served,
    in order, one entry per request answered, so a repeat shows; :meth:`unused` names the
    cassettes never served."""

    def __init__(self, directory: Path) -> None:
        self._cassettes: list[tuple[Path, tuple[str, str, str], Cassette]] = []
        for path in sorted(directory.glob("*.json")):
            cassette = load_cassette(path)
            stored = cassette.request
            key = _key((cast(str, stored["method"]), cast(str, stored["url"]), stored["body"]))
            for other, held, _ in self._cassettes:
                if held == key:
                    raise ValueError(f"{other.name} and {path.name} share a method, URL and body")
            self._cassettes.append((path, key, cassette))
        self.served: list[Path] = []
        self.misses: list[tuple[str, str, object]] = []
        super().__init__(self._handle)

    def unused(self) -> set[Path]:
        """The cassettes no request has been answered by."""
        return {path for path, _, _ in self._cassettes} - set(self.served)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        sent = (request.method, str(request.url), parse_body(request.content))
        for path, key, cassette in self._cassettes:
            if key == _key(sent):
                self.served.append(path)
                response = cassette.response
                return _response(response["status"], response["headers"], response["body"])
        self.misses.append(sent)
        raise AssertionError(f"no cassette matches the request sent: {sent!r}")
