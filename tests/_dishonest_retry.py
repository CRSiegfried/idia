"""A leaf that retries inside ``call``, in a module of its own: its ``except Exception``
is what L10 reads, so it must not share a module with the other dishonest leaves."""

from __future__ import annotations

from idia.conformance._scripted import ScriptedHarness
from idia.core import Capabilities, Deadline, Leaf, TokenEstimator, WireRequest, WireResponse
from idia.providers.scripted import ScriptFunction


class Retrying:
    """Wraps a leaf and calls it a second time when the first call raises anything."""

    def __init__(self, inner: Leaf) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def capabilities(self) -> Capabilities:
        return self._inner.capabilities

    @property
    def estimator(self) -> TokenEstimator:
        return self._inner.estimator

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        try:
            return await self._inner.call(wire_request, deadline)
        except Exception as first:
            try:
                return await self._inner.call(wire_request, deadline)
            except Exception:
                raise first from None


class RetryingHarness(ScriptedHarness):
    """The scripted leaf behind a wrapper that retries inside ``call``."""

    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Retrying:
        return Retrying(super()._build(capabilities, script))
