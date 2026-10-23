"""The OpenAI leaves under the conformance suite: a leaf harness per leaf over the generic
HTTP harness, the mock ``httpx2`` client wrapped in an ``AsyncOpenAI``, and the wrapped
leaf for the provider half."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, cast

import httpx2
from openai import AsyncOpenAI

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
from idia.conformance import CheckNotApplicable, Exchange, Success
from idia.providers._openai import OpenAILeafBase
from idia.providers.chat.openai import OpenAIChat
from idia.providers.openai_decisions import OpenAIDecisions
from tests._http import HttpLeafHarness
from tests._openai_cases import (
    CHAT_DIR,
    CHAT_MODEL,
    DECISIONS_DIR,
    DECISIONS_MODEL,
    FAILURE,
    chat_record,
    decisions_record,
    provider_request,
)

_EXTRA_NAME: Final = "conformance_extra"
_ZERO_DELAY: Final = RetryPolicy.without_delay()
_WIRE_TYPE: Final[Mapping[Kind, str]] = {"noul": "predicate", "choice": "choice", "score": "score"}
_KEY: Final = "conformance-key"


class _OpenAIHarness(HttpLeafHarness):
    """What both OpenAI leaf harnesses share: the injected client is an ``AsyncOpenAI``
    over the mock ``httpx2`` client, left at the SDK's default retries so that a leaf that
    did not zero them would send twice."""

    leaf_name: ClassVar[str]
    request_id_header = "x-request-id"
    failure_cassette = FAILURE
    overrides: Mapping[str, object] = {}
    """Fields the provider half changes on the record, passed as the caller's copy."""

    def __init__(self) -> None:
        super().__init__()
        self.injected: AsyncOpenAI | None = None

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        return self.capabilities

    def _changed(self) -> Capabilities | None:
        if not self.overrides:
            return None
        return dataclasses.replace(self.capabilities, **self.overrides)

    def _sdk(self, client: httpx2.AsyncClient | None) -> AsyncOpenAI | None:
        self.injected = None if client is None else AsyncOpenAI(api_key=_KEY, http_client=client)
        return self.injected

    def cassette_for(self, kind: Kind) -> str:
        return kind

    def sdk_retries(self) -> int:
        """The SDK's retries on the client the leaf creates and on an injected one, plus
        the created client's transport-level retries, read from httpx2's private pool."""
        created = cast(OpenAILeafBase, self.build(None, frozenset()))
        transport = created.client._client._transport
        if not isinstance(transport, httpx2.AsyncHTTPTransport):
            raise CheckNotApplicable(f"the leaf's own client runs on a {type(transport).__name__}")
        injected = cast(OpenAILeafBase, self.make([]))
        return created.client.max_retries + transport._pool._retries + injected.client.max_retries

    async def client_closed(self, *, injected: bool) -> bool:
        if injected:
            kind = sorted(self.capabilities.kinds)[0]
            leaf = cast(OpenAILeafBase, self.make([Success(kind)]))
            await cast(Leaf, leaf).call(self.wire_request(kind), Deadline.after(30.0))
            await leaf.aclose()
            assert self.injected is not None
            return self.injected.is_closed()
        leaf = cast(OpenAILeafBase, self.build(None, frozenset()))
        await leaf.aclose()
        return leaf.client.is_closed()


class DecisionsHarness(_OpenAIHarness):
    """The leaf half's harness for the decisions leaf and its cassettes."""

    leaf_name = "openai-decisions"
    cassette_dir: Path = DECISIONS_DIR

    @property
    def capabilities(self) -> Capabilities:
        return decisions_record()

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return OpenAIDecisions(
            model=DECISIONS_MODEL,
            api_key=_KEY,
            client=self._sdk(client),
            capabilities=self._changed(),
        )

    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue:
        """The recorded success with one change to its first answer of ``kind``."""
        changed = cast(dict[str, Any], copy.deepcopy(body))
        answers = cast(list[dict[str, Any]], changed["answers"])
        answer = next(a for a in answers if a.get("type") == _WIRE_TYPE[kind])
        if how == "keys":
            answer["name"] = _EXTRA_NAME
        elif how == "noul_range":
            answer["probability"] = 1.5
        else:
            entries = cast(list[dict[str, Any]], answer["probabilities"])
            min(entries, key=lambda entry: entry["probability"])["probability"] += 0.25
        return changed

    def unreadable(self, which: Literal["not_json", "schema", "no_answers"]) -> bytes:
        if which == "not_json":
            return b"<html><body>502 Bad Gateway</body></html>"
        if which == "schema":
            return b'{"model": 6, "answers": [], "usage": []}'
        return b'{"model": "gpt-6-luna", "usage": {"input_tokens": 1, "output_tokens": 0}}'

    def refusal_body(self, kind: Kind) -> JsonValue:
        """The recorded success with its first answer a refusal under the same name."""
        changed = cast(dict[str, Any], copy.deepcopy(self._cassette(kind).response["body"]))
        answers = cast(list[dict[str, Any]], changed["answers"])
        answers[0] = {"type": "refusal", "name": answers[0]["name"]}
        return changed


class ChatHarness(_OpenAIHarness):
    """The leaf half's harness for the chat leaf and its cassettes. The read renormalizes
    over the labels and the wire carries no keys, so no perturbation can break a sum, a
    key or a noul's range: ``perturb`` makes those checks not applicable."""

    leaf_name = "openai-chat"
    cassette_dir: Path = CHAT_DIR

    @property
    def capabilities(self) -> Capabilities:
        return chat_record()

    def build(self, client: httpx2.AsyncClient | None, features: frozenset[str]) -> Leaf:
        return OpenAIChat(
            model=CHAT_MODEL,
            api_key=_KEY,
            client=self._sdk(client),
            capabilities=self._changed(),
        )

    def perturb(
        self, body: JsonValue, how: Literal["sum", "keys", "noul_range"], kind: Kind
    ) -> JsonValue | None:
        return None

    def unreadable(self, which: Literal["not_json", "schema", "no_answers"]) -> bytes:
        if which == "not_json":
            return b"<html><body>502 Bad Gateway</body></html>"
        if which == "schema":
            return b'{"model": 4.1, "choices": [{"index": 0}], "usage": []}'
        return b'{"model": "gpt-4.1-2025-04-14", "usage": {"prompt_tokens": 1}}'


class WrappedOpenAI:
    """The provider half's harness: one OpenAI leaf over its cassettes, wrapped in a
    zero-delay dispatcher, its record changed by ``overrides``."""

    shape: Literal["leaf"] = "leaf"
    members: int = 1

    def __init__(self, harness: type[_OpenAIHarness]) -> None:
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
        wire_request = provider_request()
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
        model = DECISIONS_MODEL if self._harness.leaf_name == "openai-decisions" else CHAT_MODEL
        return ((self._harness.leaf_name, model),)
