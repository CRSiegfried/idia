"""A scripted leaf with an ``aclose``, so a harness can drive the check of a call after
closing; in a module of its own, so the checks that read the leaf's source read only it."""

from __future__ import annotations

from idia.core import Deadline, ProviderError, WireRequest, WireResponse
from idia.providers.scripted import Scripted


class ClosableScripted(Scripted):
    """The scripted leaf, refusing every call after :meth:`aclose` before the script runs."""

    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        if self.closed:
            raise ProviderError("the leaf is closed")
        return await super().call(wire_request, deadline)
