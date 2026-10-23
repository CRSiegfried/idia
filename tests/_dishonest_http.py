"""Stub HTTP leaves that each break one rule, the client rule of L15, the option rule of
L29 or the closed rule of L30, in a module that imports no call loop, so the review checks
that read it pass."""

from __future__ import annotations

import dataclasses
import json

from idia.core import Deadline, State, TextPart, WireRequest, WireResponse
from tests._stub_http_leaf import StubHttpLeaf


class ClosingLeaf(StubHttpLeaf):
    """Closes its client even when the client was injected."""

    async def aclose(self) -> None:
        self._closed = True
        await self.client.aclose()


class UnclosableLeaf(StubHttpLeaf):
    """Has no way to close the client it made."""

    aclose = None  # pyright: ignore[reportAssignmentType]


class ReopeningLeaf(StubHttpLeaf):
    """Closes its own client but goes on sending after aclose."""

    async def aclose(self) -> None:
        await super().aclose()
        self._closed = False


class LeakingLeaf(StubHttpLeaf):
    """Copies the options it does not know into its body, as a last text part of the state;
    the answers and the warnings stay as an honest leaf's, so only the body differs."""

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        if wire_request.options:
            leaked = TextPart(text=json.dumps(dict(wire_request.options), sort_keys=True))
            wire_request = dataclasses.replace(
                wire_request, state=State(parts=(*wire_request.state.parts, leaked))
            )
        return await super().call(wire_request, deadline)
