"""The GLiNER2 leaf under the conformance suite: the leaf half over an injected model that
plays the stored outputs, and the leaf wrapped in a dispatcher for the provider half.

A model in process has no statuses, no transport, no HTTP client and no SDK: the harness
raises ``CheckNotApplicable`` for each, and the tests assert exactly which checks that leaves
out. It cannot make a key with no question either: the runtime answers by label and the leaf
maps by question key, so no key without a question can reach validation. Each leaf is given
the model with ``revision=_REVISION``, the commit the outputs were recorded at, as the
caller's claim.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from idia import (
    Answer,
    Capabilities,
    Deadline,
    Dispatcher,
    ExtractionAnswer,
    Kind,
    Leaf,
    Provider,
    Request,
    RetryPolicy,
    Span,
    WireRequest,
    WireResponse,
)
from idia.conformance import CheckNotApplicable, Exchange
from idia.providers._gliner import _RECORD, _REVISION, Gliner
from tests._gliner_cases import ASKED, OUTPUTS, SENDER, STATE, TOPICS, Playback

EXPECTED: Final[Mapping[str, Answer]] = {
    "sender": ExtractionAnswer(
        value="Dana Whitfield",
        confidence=OUTPUTS[ASKED[0]][0]["entities"]["the sender"][0]["confidence"],
        span=Span(part=0, start=21, end=35),
    ),
    "topics": ExtractionAnswer(
        value=("Q3 budget", "hiring"),
        span=(Span(part=0, start=46, end=55), Span(part=0, start=60, end=66)),
    ),
}
"""The answers the leaf maps the stored output of the full request to."""


class GlinerHarness:
    """The leaf half: a fresh leaf over a fresh :class:`Playback` per check."""

    capabilities: Capabilities = _RECORD
    prompt_hidden: bool = False

    def __init__(self) -> None:
        self._model = Playback()

    def make(self, exchanges: Sequence[Exchange], features: frozenset[str] | None = None) -> Leaf:
        self._model = Playback(exchanges)
        return Gliner(runtime=self._model, revision=_REVISION)

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        return _RECORD

    def wire_request(self, kind: Kind) -> WireRequest:
        return WireRequest(state=STATE, questions=(SENDER, TOPICS))

    def expected(self, kind: Kind) -> Mapping[str, Answer]:
        return EXPECTED

    def requests_sent(self) -> int:
        return self._model.calls

    def sdk_retries(self) -> int:
        raise CheckNotApplicable("the leaf is built on no SDK")

    async def client_closed(self, *, injected: bool) -> bool:
        raise CheckNotApplicable("the leaf has no HTTP client")


class _LoggedGliner(Gliner):
    """The leaf with its record replaceable and a log of what it was called with."""

    def __init__(self, runtime: Playback, capabilities: Capabilities = _RECORD) -> None:
        super().__init__(runtime=runtime, revision=_REVISION)
        self._replaced = capabilities
        self.log: list[tuple[WireRequest, Deadline | None]] = []

    @property
    def capabilities(self) -> Capabilities:
        return self._replaced

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        self.log.append((wire_request, deadline))
        return await super().call(wire_request, deadline)


class GlinerProviderHarness:
    """The provider half: the leaf wrapped in a dispatcher with no retry delay."""

    shape: Literal["leaf"] = "leaf"
    members: int = 1
    expected_capabilities: Capabilities = _RECORD

    def __init__(self) -> None:
        self._model = Playback()
        self._leaf = _LoggedGliner(self._model)

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        self._model = Playback(exchanges)
        self._leaf = _LoggedGliner(self._model)
        return self._leaf

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        self._model = Playback(exchanges[0])
        record = dataclasses.replace(_RECORD, **dict(overrides or {}))  # pyright: ignore[reportArgumentType]
        self._leaf = _LoggedGliner(self._model, record)
        return Dispatcher(self._leaf, retry=RetryPolicy.without_delay())

    def request(self) -> Request:
        return Request(state=STATE, questions=(SENDER, TOPICS), deadline=Deadline.after(30.0))

    def requests_sent(self) -> int:
        return self._model.calls

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
        return (tuple(self._leaf.log),)

    def member_names(self) -> tuple[tuple[str, str], ...]:
        return ((self._leaf.name, self._leaf.model),)
