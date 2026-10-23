"""The Cloudflare leaves under the conformance suite: a leaf harness per model over the
generic HTTP harness, each leaf wrapped for the provider half, and the ``cloudflare`` router
over one shared mock connection.

Every cassette stores the raw envelope, so a perturbation edits inside its ``result``.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any, ClassVar, Final, Literal, cast

import httpx2

from idia import (
    Capabilities,
    Deadline,
    Dispatcher,
    JsonValue,
    Kind,
    Leaf,
    Provider,
    Request,
    RetryPolicy,
    WireRequest,
)
from idia.cloudflare import _router
from idia.conformance import CheckNotApplicable, Exchange, Perturbed, Success
from idia.providers.cloudflare import Clairic, Clef, ClefModel
from tests._cloudflare_cases import (
    ACCOUNT,
    CANDIDATE,
    CANDIDATE_TARGET,
    CLAIRIC_DIR,
    CLAIRIC_KIND,
    CLAIRIC_OPTIONS,
    CLAIRIC_PROVIDER,
    FAILURE,
    FIELD_NAME_PROBE,
    REQUEST_ID_HEADER,
    TOKEN,
    clairic_record,
    clef_dir,
    clef_record,
    clef_request,
    router_record,
)
from tests._http import HttpLeafHarness

_EXTRA_KEY: Final = "conformance_extra"
_ZERO_DELAY: Final = RetryPolicy.without_delay()

Handler = Callable[[httpx2.Request], Coroutine[Any, Any, httpx2.Response]]


def _answers(body: JsonValue) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """A copy of a stored envelope and the answers map inside its result."""
    changed = cast(dict[str, Any], copy.deepcopy(body))
    return changed, cast(dict[str, dict[str, Any]], changed["result"]["answers"])


def _unreadable(which: Literal["not_json", "schema", "no_answers"]) -> bytes:
    """Workers AI's forms of a body that cannot be read: an HTML gateway page, an envelope
    whose result fails the top-level schema, and one whose result holds no answers."""
    if which == "not_json":
        return b"<html><body>502 Bad Gateway</body></html>"
    if which == "schema":
        return (
            b'{"success": true, "errors": [], "messages": [], '
            b'"result": {"model": 7, "answers": {}, "usage": []}}'
        )
    return b'{"success": true, "errors": [], "messages": [], "result": {}}'


class _CloudflareHarness(HttpLeafHarness):
    """What both leaf harnesses share."""

    model: ClassVar[str]
    request_id_header = REQUEST_ID_HEADER
    failure_cassette = FAILURE
    overrides: Mapping[str, object] = {}
    """Fields the provider half changes on the record, passed as the caller's copy."""

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        return self.capabilities

    def _changed(self) -> Capabilities | None:
        if not self.overrides:
            return None
        return dataclasses.replace(self.capabilities, **self.overrides)

    def unreadable(self, which: Literal["not_json", "schema", "no_answers"]) -> bytes:
        return _unreadable(which)


class ClefHarness(_CloudflareHarness):
    """The leaf half's harness for Clef and its cassette set."""

    model: ClassVar[str] = "clef"
    cassette_dir = clef_dir("clef")

    @property
    def capabilities(self) -> Capabilities:
        return clef_record(self.model)

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Clef:
        return Clef(
            model=cast(ClefModel, self.model),
            account_id=ACCOUNT,
            api_token=TOKEN,
            client=client,
            capabilities=self._changed(),
        )

    def cassette_for(self, kind: Kind) -> str:
        return kind

    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue:
        """The recorded envelope with one change to its first answer of ``kind``."""
        changed, answers = _answers(body)
        key = next(k for k, answer in answers.items() if answer.get("type") == kind)
        if how == "keys":
            answers[_EXTRA_KEY] = answers.pop(key)
        elif how == "noul_range":
            answers[key]["noul"] = 1.5
        else:
            probabilities = cast(dict[str, float], answers[key]["probabilities"])
            lowest = min(probabilities, key=probabilities.__getitem__)
            probabilities[lowest] += 0.25
        return changed


class ClefFlashHarness(ClefHarness):
    model: ClassVar[str] = "clef-flash"
    cassette_dir = clef_dir("clef-flash")


class ClairicHarness(_CloudflareHarness):
    """The leaf half's harness for Clairic and its cassette set. Every recording was made
    with the ``reference_date`` option, which a cassette's stored wire request, written
    before requests carried options, leaves out; the harness puts it back."""

    model: ClassVar[str] = "clairic"
    cassette_dir = CLAIRIC_DIR
    failure_cassette = FIELD_NAME_PROBE
    """The wire's one recorded refusal; it answers a state past its context truncated rather
    than refusing it."""

    @property
    def capabilities(self) -> Capabilities:
        return clairic_record()

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Clairic:
        return Clairic(
            account_id=ACCOUNT,
            api_token=TOKEN,
            client=client,
            capabilities=self._changed(),
        )

    def cassette_for(self, kind: Kind) -> str:
        return CLAIRIC_KIND

    def wire_request(self, kind: Kind) -> WireRequest:
        return dataclasses.replace(super().wire_request(kind), options=CLAIRIC_OPTIONS)

    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue | None:
        """Only a key can be moved: an extraction has no sum and no noul."""
        if how != "keys":
            return None
        changed, answers = _answers(body)
        key = next(iter(answers))
        answers[_EXTRA_KEY] = answers.pop(key)
        return changed

    def candidate_body(self, kind: Kind) -> JsonValue | None:
        """The recorded extraction with one answer replaced by what Clairic returned, in a
        separate recorded call outside the leaf, for a field asked as an ``estimate``."""
        stored = self._cassette(CANDIDATE).response["body"]
        [estimate] = cast(dict[str, Any], stored)["result"]["answers"].values()
        changed, answers = _answers(self._cassette(self.cassette_for(kind)).response["body"])
        answers[CANDIDATE_TARGET] = copy.deepcopy(estimate)
        return changed


# ----------------------------------------------------------------------------------------
# The provider half
# ----------------------------------------------------------------------------------------


def _logged(
    leaf: Clef | Clairic, calls: list[tuple[WireRequest, Deadline | None]]
) -> Clef | Clairic:
    call = leaf.call

    async def logged(wire_request: WireRequest, deadline: Deadline) -> Any:
        calls.append((wire_request, deadline))
        return await call(wire_request, deadline)

    setattr(leaf, "call", logged)  # noqa: B010
    return leaf


class WrappedCloudflare:
    """The provider half's harness for one Cloudflare leaf wrapped in a zero-delay
    dispatcher, its record changed by ``overrides``."""

    shape: Literal["leaf"] = "leaf"
    members: int = 1

    def __init__(self, harness: type[_CloudflareHarness], request: WireRequest, name: str) -> None:
        self._harness = harness()
        self._name = name
        self._request = request
        self._calls: list[tuple[WireRequest, Deadline | None]] = []

    @property
    def expected_capabilities(self) -> Capabilities:
        return self._harness.capabilities

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        [plays] = exchanges
        self._harness.overrides = dict(overrides or {})
        self._calls = []
        leaf = cast(Clef | Clairic, self._harness.make(list(plays)))
        return Dispatcher(_logged(leaf, self._calls), retry=_ZERO_DELAY)

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        return self._harness.make(list(exchanges))

    def request(self) -> Request:
        return Request(
            state=self._request.state,
            questions=self._request.questions,
            deadline=Deadline.after(30.0),
            options=self._request.options,
        )

    def requests_sent(self) -> int:
        return self._harness.requests_sent()

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
        return (tuple(self._calls),)

    def member_names(self) -> tuple[tuple[str, str], ...]:
        return ((self._name, self._harness.model),)


def wrapped_clef(harness: type[ClefHarness] = ClefHarness) -> WrappedCloudflare:
    return WrappedCloudflare(harness, clef_request("choice"), "cloudflare-clef")


def wrapped_clairic() -> WrappedCloudflare:
    return WrappedCloudflare(ClairicHarness, CLAIRIC_PROVIDER, "cloudflare-clairic")


class CloudflareRouter:
    """The provider half's harness for the ``cloudflare`` router: Clef and Clairic over one
    shared mock connection, which hands each request to the member whose route it names,
    each member's record changed by ``overrides``.

    A perturbed body is meaningful only to a member that claims its kind; any other member
    plays a recorded success in its place."""

    shape: Literal["router"] = "router"
    members: int = 2

    def __init__(self, clef: type[ClefHarness] = ClefHarness) -> None:
        self._harnesses: tuple[_CloudflareHarness, _CloudflareHarness] = (
            clef(),
            ClairicHarness(),
        )
        self._calls: tuple[list[tuple[WireRequest, Deadline | None]], ...] = ([], [])

    @property
    def expected_capabilities(self) -> Capabilities:
        return router_record(self._harnesses[0].model)

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        handlers: dict[str, Handler] = {}
        for harness, plays in zip(self._harnesses, exchanges, strict=True):
            segment = harness.model
            harness.overrides = dict(overrides or {})
            kinds = harness.capabilities.kinds
            handlers[segment] = harness.handler(
                [
                    Success() if isinstance(play, Perturbed) and play.kind not in kinds else play
                    for play in plays
                ]
            )

        async def route(request: httpx2.Request) -> httpx2.Response:
            return await handlers[request.url.path.rsplit("/", 1)[-1]](request)

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(route))
        self._calls = ([], [])
        clef, clairic = (
            _logged(cast(Clef | Clairic, harness.build(client, frozenset())), calls)
            for harness, calls in zip(self._harnesses, self._calls, strict=True)
        )
        return _router(
            cast(Clef, clef), cast(Clairic, clairic), retry=_ZERO_DELAY, concurrency=None, rate=None
        )

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        raise CheckNotApplicable("a router has no one leaf to record")

    def request(self) -> Request:
        clef = clef_request("choice")
        return Request(
            state=clef.state,
            questions=(*clef.questions, *CLAIRIC_PROVIDER.questions),
            deadline=Deadline.after(30.0),
            options=CLAIRIC_OPTIONS,
        )

    def requests_sent(self) -> int:
        return sum(harness.requests_sent() for harness in self._harnesses)

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
        return tuple(tuple(calls) for calls in self._calls)

    def member_names(self) -> tuple[tuple[str, str], ...]:
        return (
            ("cloudflare-clef", self._harnesses[0].model),
            ("cloudflare-clairic", "clairic"),
        )
