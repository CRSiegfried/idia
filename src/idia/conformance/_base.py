"""The suite's shared shapes: checks, verdicts, exchanges and the two harness protocols."""

from __future__ import annotations

import sys
from collections.abc import Callable, Coroutine, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from idia import answers
from idia.core import (
    Answer,
    Capabilities,
    Deadline,
    Fault,
    Kind,
    Leaf,
    MappedAnswer,
    Provider,
    Question,
    Request,
    Rounding,
    Rule,
    WireRequest,
)

# ----------------------------------------------------------------------------------------
# Checks and verdicts
# ----------------------------------------------------------------------------------------


class CheckNotApplicable(Exception):
    """Raised by a check, or by the harness it drives, when the check cannot be reached for
    the provider under test: a leaf with no SDK has no SDK retries to inspect."""

    reason: str

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Check:
    """One conformance check: its ID from the suite's tables, what it checks, and a
    coroutine function that runs it against a fresh harness.

    ``run`` returns when the check passes, with a note on what the pass did not cover or
    ``None``; it raises :class:`AssertionError` when the check fails and
    :class:`CheckNotApplicable` when it cannot be reached.
    """

    id: str
    description: str
    run: Callable[[], Coroutine[Any, Any, str | None]]


@dataclass(frozen=True, slots=True)
class Verdict:
    """What :func:`run_checks` reports for one check. ``reason`` is the failure, the
    reason it was not applicable, or on a pass what the pass did not cover, if anything."""

    id: str
    result: Literal["pass", "fail", "not_applicable"]
    reason: str | None = None


async def run_checks(checks: Iterable[Check]) -> tuple[Verdict, ...]:
    """Run each check in turn and return one verdict per check, in order.

    Any exception other than :class:`CheckNotApplicable` is a failure; a
    ``BaseException`` such as cancellation propagates. The checks run one at a time and
    must not run concurrently with each other: the ``validate`` spy patches modules
    process-wide while a check runs.
    """
    verdicts: list[Verdict] = []
    for check in checks:
        try:
            note = await check.run()
        except CheckNotApplicable as not_applicable:
            verdicts.append(Verdict(check.id, "not_applicable", not_applicable.reason))
        except Exception as error:
            verdicts.append(Verdict(check.id, "fail", f"{type(error).__name__}: {error}"))
        else:
            verdicts.append(Verdict(check.id, "pass", note))
    return tuple(verdicts)


def missing(answer: Answer | None) -> bool:
    """A fault for a key the wire left out, which ``validate_answers`` builds itself."""
    return isinstance(answer, Fault) and [f.rule for f in answer.failures] == [Rule.missing_key]


def is_a(value: object, kind: type | tuple[type, ...]) -> bool:
    """``isinstance``, for a value whose declared type already claims the answer."""
    return isinstance(value, kind)


def expect(condition: bool, message: str) -> None:
    """Fail the check with ``message`` unless ``condition`` holds; unlike ``assert``, it
    survives ``python -O``."""
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------------------
# Exchanges: what the wire does on one call
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Success:
    """The recorded success for ``harness.wire_request(kind)``; with no kind, the recorded
    success that answers whatever the provider harness's ``request()`` sends."""

    kind: Kind | None = None


@dataclass(frozen=True, slots=True)
class Failure:
    """A failure the wire or the runtime reports. With no status, the recorded failure the
    harness serves, which is not retryable and is not a credential error (401 or 403); a
    wire API's carries a status, and a runtime with no statuses gives none. With a status,
    a response of that status carrying the request id given; a harness whose transport has
    no statuses raises :class:`CheckNotApplicable` for it.

    An HTTP harness sends ``retry_after`` as ``retry-after-ms`` and a different,
    conflicting value as ``retry-after``, so L12's equality also checks that the
    millisecond header takes precedence."""

    status: int | None = None
    retry_after: float | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class Unreadable:
    """A body that cannot be read as a whole: not JSON, failing the top-level schema, or
    holding no answers map."""

    which: Literal["not_json", "schema", "no_answers"]


@dataclass(frozen=True, slots=True)
class Perturbed:
    """The recorded success for ``kind`` with one change: one answer's sum moved outside
    tolerance (``sum``), one key removed and one added (``keys``), or a ``noul``'s
    probability set outside [0, 1] (``noul_range``)."""

    kind: Kind
    how: Literal["sum", "keys", "noul_range"]


@dataclass(frozen=True, slots=True)
class Error:
    """A transport error instead of a response."""

    which: Literal["connection", "timeout"]


@dataclass(frozen=True, slots=True)
class Stall:
    """A wire or a runtime that never answers, so the call must end by its deadline."""


@dataclass(frozen=True, slots=True)
class RefusalOnWire:
    """The recorded refusal for ``harness.wire_request(kind)``."""

    kind: Kind


@dataclass(frozen=True, slots=True)
class CandidateShape:
    """The recorded candidate shape returned for a question of the admitted ``kind``."""

    kind: Kind


Exchange: TypeAlias = (
    Success | Failure | Unreadable | Perturbed | Error | Stall | RefusalOnWire | CandidateShape
)
"""What the wire does on one call; a harness turns each into its own transport's terms."""

# ----------------------------------------------------------------------------------------
# The harnesses a provider's author supplies
# ----------------------------------------------------------------------------------------


class LeafHarness(Protocol):
    """What a leaf's author supplies to run the leaf half against its leaf.

    ``make`` builds a fresh leaf whose wire plays ``exchanges`` in order, one per wire
    request, with the opt-in ``features`` on, by default the record's own
    ``enabled_features``, so every check builds the leaf as its record declares; a leaf
    with several configurations runs the suite once per configuration. ``make`` raises
    :class:`CheckNotApplicable` for an exchange its transport cannot make.
    ``capabilities_for(features)`` is the record the leaf must declare when built with
    ``features`` on, every field the feature changes included. ``requests_sent`` counts
    the wire requests the leaf ``make`` built last has sent. ``wire_request(kind)`` is the request
    each recorded exchange for ``kind`` answers, and ``expected(kind)`` its answers as
    mapped. An HTTP harness sends ``Failure.retry_after`` as ``retry-after-ms`` beside a
    conflicting ``retry-after``. ``sdk_retries`` is the retries below the leaf: an SDK's,
    or for a leaf on a raw HTTP client the client's transport-level retries, which count
    as SDK retries; ``client_closed`` reports whether the leaf's HTTP client is closed
    after use. Each raises :class:`CheckNotApplicable` for a leaf with no SDK or client.
    ``prompt_hidden`` is true when the wire API hides the prompt, so L26 accepts a
    response whose ``prompt_sent`` is ``None``.
    """

    @property
    def capabilities(self) -> Capabilities: ...

    @property
    def prompt_hidden(self) -> bool: ...

    def make(
        self, exchanges: Sequence[Exchange], features: frozenset[str] | None = None
    ) -> Leaf: ...

    def capabilities_for(self, features: frozenset[str]) -> Capabilities: ...

    def wire_request(self, kind: Kind) -> WireRequest: ...

    def expected(self, kind: Kind) -> Mapping[str, Answer]: ...

    def requests_sent(self) -> int: ...

    def sdk_retries(self) -> int: ...

    async def client_closed(self, *, injected: bool) -> bool: ...


class ProviderHarness(Protocol):
    """What a provider's author supplies to run the provider half.

    ``shape`` says what is under test: a wrapped leaf, a router or a fallback provider;
    ``members`` how many members it has, one for a wrapped leaf. ``make`` builds a fresh
    provider whose member ``i`` plays ``exchanges[i]``, every member leaf's record changed
    by ``overrides`` through ``dataclasses.replace``; each dispatcher has a zero-delay
    retry policy. ``request`` is a fresh request that fits, with at least two questions,
    reaching every member of a router and the first member of a fallback. ``member_calls``
    is each member's log of the wire requests its leaf received since the last ``make``,
    in member order, each with
    the deadline the leaf was given, or ``None`` where the harness cannot see it;
    ``member_names`` each member leaf's ``(name, model)``. Member order is a router's
    mapping order and a fallback's sequence order.
    ``expected_capabilities`` is the record the provider declares, for a composite the one
    it is expected to derive; a composite's members include a pair with a known and an
    unknown token limit, one of them truncating silently, so the derivation is exercised.
    ``leaf`` builds the wrapped leaf alone, unwrapped, its wire playing ``exchanges`` in
    order, so the suite can put it under the recorded leaf; ``requests_sent`` then counts
    its wire requests. A composite has no one leaf and raises :class:`CheckNotApplicable`.
    """

    @property
    def shape(self) -> Literal["leaf", "router", "fallback"]: ...

    @property
    def members(self) -> int: ...

    @property
    def expected_capabilities(self) -> Capabilities: ...

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider: ...

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf: ...

    def request(self) -> Request: ...

    def requests_sent(self) -> int: ...

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]: ...

    def member_names(self) -> tuple[tuple[str, str], ...]: ...


# ----------------------------------------------------------------------------------------
# The validate spy
# ----------------------------------------------------------------------------------------


@contextmanager
def validate_spy() -> Generator[list[tuple[str, Rounding]], None, None]:
    """Patch every loaded module's binding of ``idia.answers.validate`` with a spy
    that records each question key it validates with the rounding it was given, then
    restore them.

    Patching by identity catches ``idia.answers`` itself, so calls made through
    ``validate_answers``, and a leaf's module that imported ``validate`` by name.
    """
    real = answers.validate
    keys: list[tuple[str, Rounding]] = []

    def spy(question: Question, mapped: MappedAnswer, rounding: Rounding) -> Answer:
        keys.append((question.key, rounding))
        return real(question, mapped, rounding)

    bindings: list[tuple[dict[str, object], str]] = []
    for module in list(sys.modules.values()):
        namespace: dict[str, object] = getattr(module, "__dict__", {})
        bindings.extend(
            (namespace, name) for name, value in list(namespace.items()) if value is real
        )
    for namespace, name in bindings:
        namespace[name] = spy
    try:
        yield keys
    finally:
        for namespace, name in bindings:
            namespace[name] = real
