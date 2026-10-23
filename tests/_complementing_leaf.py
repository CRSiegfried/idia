"""A scripted leaf that fills a noul's complement itself, in a module of its own, so L21
reads only it: the double L21 must fail."""

from __future__ import annotations

from idia import noul_complement
from idia.core import Deadline, WireRequest, WireResponse
from idia.providers.scripted import Scripted


class ComplementingScripted(Scripted):
    """The scripted leaf, working out the probability of no it has no use for."""

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        _ = noul_complement(0.5)
        return await super().call(wire_request, deadline)
