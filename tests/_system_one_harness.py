"""The System One leaf under the conformance suite: a leaf harness per configuration over
the generic HTTP harness, and the wrapped leaf for the provider half."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
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
from idia.conformance import Exchange
from idia.providers.system_one import SystemOne
from tests._http import HttpLeafHarness
from tests._system_one_cases import (
    FAILURE,
    PROVIDER_KIND,
    TEXT_ONLY,
    WITH_IMAGES,
    Configuration,
    record,
)

_EXTRA_KEY: Final = "conformance_extra"
_ZERO_DELAY: Final = RetryPolicy.without_delay()


class SystemOneHarness(HttpLeafHarness):
    """The leaf half's harness for one System One configuration and its cassette set."""

    configuration: ClassVar[Configuration] = TEXT_ONLY
    cassette_dir = TEXT_ONLY.directory
    failure_cassette = FAILURE
    request_id_header = "x-typesafe-request-id"

    @property
    def capabilities(self) -> Capabilities:
        return record(self.configuration.model, self.configuration.features)

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        return record(self.configuration.model, features)

    overrides: Mapping[str, object] = {}
    """Fields the provider half changes on the record, passed as the caller's copy."""

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> SystemOne:
        changed = (
            dataclasses.replace(self.capabilities_for(features), **self.overrides)
            if self.overrides
            else None
        )
        return SystemOne(
            model=self.configuration.model,
            api_key="conformance-key",
            client=client,
            images="images" in features,
            long_context="long_context" in features,
            capabilities=changed,
        )

    def cassette_for(self, kind: Kind) -> str:
        return kind

    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue:
        """The recorded success with one change to its first answer of ``kind``."""
        changed = cast(dict[str, Any], copy.deepcopy(body))
        answers = cast(dict[str, dict[str, Any]], changed["answers"])
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

    def unreadable(self, which: Literal["not_json", "schema", "no_answers"]) -> bytes:
        if which == "not_json":
            return b"<html><body>502 Bad Gateway</body></html>"
        if which == "schema":
            return b'{"model": 1.13, "answers": {}, "usage": []}'
        return b'{"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 0}}'


class WithImagesHarness(SystemOneHarness):
    configuration: ClassVar[Configuration] = WITH_IMAGES
    cassette_dir = WITH_IMAGES.directory


class WrappedSystemOne:
    """The provider half's harness: one System One leaf over a configuration's cassettes,
    wrapped in a zero-delay dispatcher, its record changed by ``overrides``."""

    shape: Literal["leaf"] = "leaf"
    members: int = 1

    def __init__(self, harness: type[SystemOneHarness] = SystemOneHarness) -> None:
        self._harness = harness()
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
        leaf = self._harness.make(list(plays))
        assert isinstance(leaf, SystemOne)
        self._calls = []
        call = leaf.call

        async def logged(wire_request: WireRequest, deadline: Deadline) -> Any:
            self._calls.append((wire_request, deadline))
            return await call(wire_request, deadline)

        setattr(leaf, "call", logged)  # noqa: B010
        return Dispatcher(leaf, retry=_ZERO_DELAY)

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        return self._harness.make(list(exchanges))

    def request(self) -> Request:
        wire_request = self._harness.wire_request(PROVIDER_KIND)
        return Request(
            state=wire_request.state,
            questions=wire_request.questions,
            deadline=Deadline.after(30.0),
        )

    def requests_sent(self) -> int:
        return self._harness.requests_sent()

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
        return (tuple(self._calls),)

    def member_names(self) -> tuple[tuple[str, str], ...]:
        return (("system-one", self._harness.configuration.model),)
