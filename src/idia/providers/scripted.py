"""The scripted leaf: a leaf whose every call is answered from a test's script.

:class:`Scripted` satisfies :class:`~idia.core.Leaf` structurally, with no base class.
Its name, model and capabilities record are the test's. Each call is logged, in order,
before it is answered, and then answered from the next item of a script or from a function
the test passes:

- a :class:`Reply`, or a bare mapping of question key to mapped answer as shorthand for
  one, is handed to :func:`~idia.answers.validate_answers` with the declared rounding,
  as a wire leaf hands over what it mapped; a scripted ``Fault`` or ``Refusal`` passes
  through as it stands and its siblings are validated;
- a :class:`~idia.core.ProviderError` is raised as it is.

An async script function is awaited under the call's deadline, as a wire leaf's transport
is: one that stalls past it ends in a retryable ``ProviderError``, which is how a test
simulates a timeout. A sync script function is not timed and must not sleep.

The leaf fills the provenance a wire leaf fills, deterministically, so a test can assert on
it. A script that runs out raises :class:`ScriptExhausted`, a test failure and never a
provider error, so the call loop cannot mistake it for a wire failure. The scripted leaf
is not in the provider registry: a test builds it and wraps it itself.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from idia.answers import validate_answers
from idia.core import (
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    JsonValue,
    MappedAnswer,
    ProviderError,
    TokenEstimator,
    Usage,
    Warning,
    WireRequest,
    WireResponse,
    unknown_options,
)
from idia.providers._canon import (
    CharacterEstimator,
    encode,
    json_bytes,
    render_wire_request,
    tokens,
)

__all__ = [
    "Reply",
    "ScriptExhausted",
    "Scripted",
    "ScriptedCall",
]


class ScriptExhausted(AssertionError):
    """The leaf was called more times than its script has items.

    It is a test failure, not a :class:`~idia.core.ProviderError`: the call loop
    never retries it and it never becomes an outcome's error.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class Reply:
    """One scripted answer to a call: the mapped answers and the provenance a test varies.

    ``model_reported`` defaults to the leaf's model and ``usage`` to the leaf's estimate;
    ``warnings`` are added to the wire response's own. ``pinned`` false says the reported
    model is an alias, not a pinned version, so no snapshot exists even on a record whose
    ``snapshot_pinnable`` is true.
    """

    answers: Mapping[str, MappedAnswer]
    model_reported: str | None = None
    pinned: bool = True
    usage: Usage | None = None
    warnings: tuple[Warning, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedCall:
    """One entry of the call log: the wire request the leaf received and its deadline."""

    wire_request: WireRequest
    deadline: Deadline


Item: TypeAlias = Reply | Mapping[str, MappedAnswer] | ProviderError
"""What answers one call: a reply, a bare answer map, or a provider error to raise."""

ScriptFunction: TypeAlias = Callable[[WireRequest, Deadline], Item | Awaitable[Item]]
"""A function that answers each call; it may be sync or async."""

Script: TypeAlias = Sequence[Item] | ScriptFunction
"""Items consumed one per call, in order, or a function called once per call."""


class Scripted:
    """A leaf that answers each call from a test's script and logs every call it receives.

    ``calls`` is the log, one :class:`ScriptedCall` per call in the order received, written
    before the call is answered or raises. ``estimator`` defaults to a fixed-rate character
    count and is never ``None`` once built.
    """

    name: str
    model: str
    capabilities: Capabilities
    estimator: TokenEstimator
    calls: list[ScriptedCall]

    def __init__(
        self,
        *,
        capabilities: Capabilities,
        script: Script,
        name: str = "scripted",
        model: str = "scripted-1",
        estimator: TokenEstimator | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.capabilities = capabilities
        self.estimator = estimator if estimator is not None else CharacterEstimator()
        self.calls = []
        self._function: ScriptFunction | None = None
        self._items: Iterator[Item] = iter(())
        if callable(script):
            self._function = script
        else:
            self._items = iter(tuple(script))

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Log the call, then answer it from the script or raise the scripted error."""
        self.calls.append(ScriptedCall(wire_request=wire_request, deadline=deadline))
        n = len(self.calls)
        item = await self._next(wire_request, deadline, n)
        if isinstance(item, ProviderError):
            raise item
        if isinstance(item, Reply):
            reply = item
        elif isinstance(item, Mapping):
            reply = Reply(answers=cast(Mapping[str, MappedAnswer], item))
        else:
            raise TypeError(
                f"{self.name}'s script gave a {type(item).__name__}; "
                "a script item is a Reply, a mapping of key to answer, or a ProviderError"
            )
        return self._respond(wire_request, reply, n)

    async def _next(self, wire_request: WireRequest, deadline: Deadline, n: int) -> object:
        """The next item, typed as what a test may really pass; ``call`` checks it."""
        if self._function is not None:
            result = self._function(wire_request, deadline)
            if not inspect.isawaitable(result):
                return result
            try:
                return await asyncio.wait_for(result, max(deadline.remaining(), 0.0))
            except TimeoutError:
                raise ProviderError(
                    message=f"{self.name}'s script stalled past the deadline",
                    is_retryable=True,
                    retry_after=None,
                    status_code=None,
                    request_id=f"{self.name}-{n}",
                ) from None
        try:
            return next(self._items)
        except StopIteration:
            raise ScriptExhausted(f"{self.name} was called {n} times; its script ran out") from None

    def _respond(self, wire_request: WireRequest, reply: Reply, n: int) -> WireResponse:
        rounding = self.capabilities.rounding
        answers, warnings = validate_answers(wire_request.questions, reply.answers, rounding)
        model_reported = reply.model_reported if reply.model_reported is not None else self.model
        snapshot: str | None = None
        if self.capabilities.snapshot_pinnable and reply.pinned:
            snapshot = model_reported
        else:
            warnings += (
                Warning(
                    code=WARNING_NO_SNAPSHOT,
                    message=f"{model_reported} is not a pinned version; no snapshot exists",
                    field="snapshot",
                ),
            )
        raw_body: JsonValue = {"answers": encode(reply.answers)}
        usage = reply.usage
        if usage is None:
            estimate = self.estimator.estimate(wire_request.state, wire_request.questions)
            output_tokens = tokens(json_bytes(raw_body).decode())
            usage = Usage(
                input_tokens=estimate.tokens,
                output_tokens=output_tokens,
                raw_usage={"input_tokens": estimate.tokens, "output_tokens": output_tokens},
            )
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self.capabilities.calibrated,
            model_reported=model_reported,
            snapshot=snapshot,
            prompt_sent=render_wire_request(wire_request),
            usage=usage,
            request_id=f"{self.name}-{n}",
            raw_body=raw_body,
            warnings=warnings + reply.warnings + unknown_options(wire_request.options, frozenset()),
        )
