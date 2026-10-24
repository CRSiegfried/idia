"""Dispatch: the call loop over one leaf, and the composites built from providers.

:class:`Dispatcher` wraps a :class:`~idia.core.Leaf` into a
:class:`~idia.core.Provider`. On each ``answer`` it runs Fit's preflight and plan,
then, inside the one deadline scope the request names, sends each wire request in plan
order: it waits on the concurrency limit and the rate limiter, calls the leaf, judges a
:class:`~idia.core.ProviderError` by the retry policy, refuses any sleep that would
overrun the deadline, and builds one outcome per wire request. The response is Fit's merge
of those outcomes, or :class:`~idia.core.RequestFailed` carrying it when none answered.

:class:`Router` and :class:`Fallback` are built from providers, never from leaves. Each
derives its capabilities record from its members' when it is built, and preflights the
derived record and every member's share before any member is called. A member is one of
idia's own providers, a :class:`Dispatcher`, :class:`Router` or :class:`Fallback`, so
that every member can preflight without calling; any other provider is refused at build.

Each of the three has ``aclose()`` and is an async context manager. Closing is structural
and optional: a dispatcher closes its leaf when the leaf has an ``aclose``, and a composite
closes each of its members, so a leaf that created an HTTP client of its own, as one built
through :func:`idia.provider` does, can be closed through the provider that wraps it.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from types import TracebackType
from typing import Final, Self, TypeAlias

from idia.core import (
    UNBOUNDED,
    Answer,
    Capabilities,
    Deadline,
    DeadlineExceeded,
    FailedAttempt,
    Fault,
    ImageCarrier,
    Kind,
    Leaf,
    Limit,
    Modality,
    Notice,
    Outcome,
    Provider,
    ProviderError,
    Question,
    Request,
    RequestFailed,
    Response,
    Rounding,
    TokenEstimator,
    WireRequest,
    WireResponse,
)
from idia.fit import (
    Fail,
    FitPlan,
    OverflowPolicy,
    _check_composite,
    _merge,
    _preflight,
    _unchecked,
)
from idia.limits import ConcurrencyLimit, RateLimiter
from idia.retry import RetryPolicy

_Result: TypeAlias = tuple[Outcome, Mapping[str, Answer] | None]
"""One wire request's outcome, with its answers or ``None`` when it failed."""

_DEFAULT_RETRY = RetryPolicy()


def _no_time_to_retry() -> DeadlineExceeded:
    """The error of a wire request whose next retry the deadline cannot fit."""
    return DeadlineExceeded("the deadline left no time to try again")


_FAIL = Fail()

_CREDENTIAL_STATUSES: Final = frozenset({401, 403})
"""The statuses of a credential error: a non-retryable one with either leaves the rest of
the call's wire requests unsent."""


def _credential_error(outcome: Outcome) -> ProviderError | None:
    """The outcome's error when it is a credential error, else ``None``."""
    error = outcome.error
    if error is None or error.is_retryable or error.status_code not in _CREDENTIAL_STATUSES:
        return None
    return error


# ----------------------------------------------------------------------------------------
# Closing
# ----------------------------------------------------------------------------------------


async def _close(target: object) -> None:
    """Await ``target.aclose()`` when it has one; a target without one is left alone."""
    aclose: Callable[[], Awaitable[None]] | None = getattr(target, "aclose", None)
    if aclose is not None:
        await aclose()


async def _close_all(members: Iterable[_Member]) -> None:
    """Close every member in order, each tried whether or not an earlier one raised; the
    first error is re-raised once all were tried."""
    first: Exception | None = None
    for member in members:
        try:
            await member.aclose()
        except Exception as error:
            if first is None:
                first = error
    if first is not None:
        raise first


# ----------------------------------------------------------------------------------------
# The dispatcher
# ----------------------------------------------------------------------------------------


class Dispatcher:
    """A leaf wrapped into a provider: the one place every attempt, wait and outcome is made.

    ``estimator=None`` means the leaf's own. ``reask`` turns on the re-ask after a fault: a
    second wire request for the faulted keys of each plan wire request, at most once each,
    under the same deadline, its outcome appended after the plan's and its answers standing.
    """

    def __init__(
        self,
        leaf: Leaf,
        *,
        retry: RetryPolicy = _DEFAULT_RETRY,
        concurrency: ConcurrencyLimit | None = None,
        rate: RateLimiter | None = None,
        estimator: TokenEstimator | None = None,
        overflow: OverflowPolicy = _FAIL,
        reask: bool = False,
    ) -> None:
        if isinstance(leaf, Dispatcher | Router | Fallback):
            raise TypeError(
                f"a Dispatcher wraps a leaf; {type(leaf).__name__} is already a provider, "
                "so put it in a Router or Fallback instead"
            )
        if not isinstance(leaf, Leaf):  # pyright: ignore[reportUnnecessaryIsInstance]  # for untyped callers
            raise TypeError(
                f"a Dispatcher wraps a leaf; {type(leaf).__name__} is not a Leaf "
                "(name, model, capabilities, estimator and call)"
            )
        self._leaf = leaf
        self._retry = retry
        self._concurrency = concurrency
        self._rate = rate
        self._estimator = estimator if estimator is not None else leaf.estimator
        self._overflow = overflow
        self._reask = reask

    @property
    def capabilities(self) -> Capabilities:
        return self._leaf.capabilities

    async def aclose(self) -> None:
        """Await the leaf's ``aclose`` when it has one; a leaf without one is left alone.

        Closing again calls it again: no closed flag is kept here."""
        await _close(self._leaf)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @property
    def reask(self) -> bool:
        """Whether the re-ask after a fault is on, as built."""
        return self._reask

    @property
    def overflow(self) -> OverflowPolicy:
        """The overflow policy ``plan`` and ``answer`` pass to Fit, as built."""
        return self._overflow

    def plan(self, request: Request) -> FitPlan:
        """Preflight and plan with no call to the leaf, using the leaf's record and
        ``model``, the dispatcher's estimator and its overflow policy; returns the
        :class:`~idia.FitPlan` that :meth:`answer` runs, or raises ``DoesNotFit``."""
        return _preflight(
            request,
            self._leaf.capabilities,
            self._estimator,
            self._overflow,
            model=self._leaf.model,
        )

    def _plan(self, request: Request) -> FitPlan:
        """The check a composite runs on this member before calling any: :meth:`plan`."""
        return self.plan(request)

    async def answer(self, request: Request) -> Response:
        plan = self.plan(request)
        deadline = request.deadline
        results: list[_Result] = []
        # A credential error stops this call only: the wire requests after it go unsent.
        stop: ProviderError | None = None
        for wire_request, warnings in zip(plan.wire_requests, plan.outcome_warnings, strict=True):
            if stop is not None:
                results.append(self._unsent(wire_request, stop, warnings))
                continue
            result = await self._send(wire_request, deadline, warnings)
            results.append(result)
            stop = _credential_error(result[0])
        answers, outcomes = _merge(results)
        if all(received is None for _, received in results):
            raise RequestFailed(
                Response(answers=answers, outcomes=outcomes, warnings=plan.response_warnings)
            )
        if self._reask:
            again = await self._ask_again(plan, results, deadline, stop)
            merged = dict(answers)
            for outcome, received in again:
                outcomes += (outcome,)
                if received is not None:
                    merged.update(received)
            answers = merged
        return Response(answers=answers, outcomes=outcomes, warnings=plan.response_warnings)

    async def _ask_again(
        self,
        plan: FitPlan,
        results: Sequence[_Result],
        deadline: Deadline,
        stop: ProviderError | None,
    ) -> list[_Result]:
        """One more wire request per plan wire request whose answers hold a fault; after a
        credential error, ``stop``, each is left unsent."""
        again: list[_Result] = []
        for wire_request, (_, received) in zip(plan.wire_requests, results, strict=True):
            if received is None:
                continue
            faulted = tuple(
                question
                for question in wire_request.questions
                if isinstance(received.get(question.key), Fault)
            )
            if not faulted:
                continue
            retry_request = WireRequest(
                state=wire_request.state, questions=faulted, options=wire_request.options
            )
            warnings = _unchecked(self._leaf.capabilities, wire_request.state, faulted)
            if stop is not None:
                again.append(self._unsent(retry_request, stop, warnings))
                continue
            result = await self._send(retry_request, deadline, warnings)
            again.append(result)
            stop = _credential_error(result[0])
        return again

    def _unsent(
        self, wire_request: WireRequest, error: ProviderError, warnings: tuple[Notice, ...]
    ) -> _Result:
        """The outcome of a wire request left unsent after a credential error: that error,
        no attempts and no request id, as for one never sent because the deadline passed."""
        started_at = time.time()
        start = time.monotonic()
        keys = tuple(question.key for question in wire_request.questions)
        outcome = self._failed(
            keys,
            error,
            started_at=started_at,
            latency=time.monotonic() - start,
            request_id=None,
            failed=(),
            warnings=warnings,
        )
        return outcome, None

    def _failed(
        self,
        keys: tuple[str, ...],
        error: ProviderError,
        *,
        started_at: float,
        latency: float,
        request_id: str | None,
        failed: tuple[FailedAttempt, ...],
        warnings: tuple[Notice, ...],
    ) -> Outcome:
        """A failed wire request's outcome: no wire response, so no provenance from one."""
        return Outcome(
            keys=keys,
            error=error,
            provider=self._leaf.name,
            model_requested=self._leaf.model,
            model_reported=None,
            snapshot=None,
            elicitation=None,
            calibrated=None,
            prompt_sent=None,
            usage=None,
            latency=latency,
            started_at=started_at,
            request_id=request_id,
            failed_attempts=failed,
            warnings=warnings,
        )

    async def _send(
        self, wire_request: WireRequest, deadline: Deadline, warnings: tuple[Notice, ...]
    ) -> _Result:
        """Send one wire request with its retries and build its outcome."""
        started_at = time.time()
        start = time.monotonic()
        failed: list[FailedAttempt] = []
        result: WireResponse | ProviderError
        if deadline.passed():
            result = DeadlineExceeded()
        else:
            result = await self._attempts(wire_request, deadline, failed)
        leaf = self._leaf
        keys = tuple(question.key for question in wire_request.questions)
        latency = time.monotonic() - start
        if isinstance(result, ProviderError):
            outcome = self._failed(
                keys,
                result,
                started_at=started_at,
                latency=latency,
                request_id=result.request_id,
                failed=tuple(failed),
                warnings=warnings,
            )
            return outcome, None
        outcome = Outcome(
            keys=keys,
            error=None,
            provider=leaf.name,
            model_requested=leaf.model,
            model_reported=result.model_reported,
            snapshot=result.snapshot,
            elicitation=result.elicitation,
            calibrated=result.calibrated,
            prompt_sent=result.prompt_sent,
            usage=result.usage,
            latency=latency,
            started_at=started_at,
            request_id=result.request_id,
            failed_attempts=tuple(failed),
            warnings=result.warnings + warnings,
        )
        return outcome, result.answers

    async def _attempts(
        self, wire_request: WireRequest, deadline: Deadline, failed: list[FailedAttempt]
    ) -> WireResponse | ProviderError:
        """The attempt loop: each call after the limiter waits, retried by the policy.

        Returns the wire response, or the error the wire request ends with: the last
        ``ProviderError`` when it is not retryable or the policy's attempts are spent;
        otherwise ``DeadlineExceeded``, before any call or after failed attempts, when a
        limiter refused, the deadline passed, or the next retry's delay would end past it.
        The failed attempts stay listed with their own errors.
        """
        while True:
            try:
                await self._acquire(deadline)
            except DeadlineExceeded as exceeded:
                return exceeded
            if deadline.passed():
                self._release()
                return _no_time_to_retry() if failed else DeadlineExceeded()
            attempt_started_at = time.time()
            attempt_start = time.monotonic()
            try:
                return await self._leaf.call(wire_request, deadline)
            except ProviderError as error:
                failed.append(
                    FailedAttempt(
                        provider=self._leaf.name,
                        model_requested=self._leaf.model,
                        error=error,
                        request_id=error.request_id,
                        started_at=attempt_started_at,
                        duration=time.monotonic() - attempt_start,
                        usage=None,
                    )
                )
                last = error
            finally:
                self._release()
            if not last.is_retryable or len(failed) >= self._retry.attempts:
                return last
            delay = self._retry.delay(len(failed), last.retry_after)
            if delay > deadline.remaining():
                return _no_time_to_retry()
            await asyncio.sleep(delay)

    async def _acquire(self, deadline: Deadline) -> None:
        """Take the concurrency slot, then a rate token; the slot is held while waiting for
        the token, and given back if the wait is refused."""
        if self._concurrency is not None:
            await self._concurrency.acquire(deadline)
        if self._rate is not None:
            try:
                await self._rate.acquire(deadline)
            except BaseException:
                self._release()
                raise

    def _release(self) -> None:
        if self._concurrency is not None:
            self._concurrency.release()


# ----------------------------------------------------------------------------------------
# The composites
# ----------------------------------------------------------------------------------------


def _label(member: _Member) -> str:
    """A member as a build error names it: a dispatcher by its leaf's name and model."""
    if isinstance(member, Dispatcher):
        return f"the Dispatcher over {member._leaf.name} ({member._leaf.model})"
    return f"a {type(member).__name__}"


def _own(members: Sequence[Provider]) -> tuple[_Member, ...]:
    """The members as idia's own providers, or ``TypeError`` for any other."""
    for member in members:
        if not isinstance(member, Dispatcher | Router | Fallback):
            raise TypeError(
                f"a composite is built from idia's own providers; "
                f"{type(member).__name__} is not a Dispatcher, Router or Fallback"
            )
    return tuple(member for member in members if isinstance(member, Dispatcher | Router | Fallback))


async def _answer_or_failure(member: Provider, request: Request) -> tuple[Response, bool]:
    """A member's response and whether it answered; a ``RequestFailed`` is its response."""
    try:
        return await member.answer(request), True
    except RequestFailed as failure:
        return failure.response, False


class Router:
    """One member per kind: each member gets the questions of its kinds as a request of
    its own, and the members' outcomes are concatenated in member order.

    A member routing two kinds gets one request. Members are called concurrently. Member
    order is the mapping's: each member's first appearance in it. A kind the mapping lacks
    fails preflight; a kind routed to a member whose record lacks it is refused at build.
    A member's ``RequestFailed`` is collected beside the others' responses; any other
    exception propagates as itself.
    """

    capabilities: Capabilities

    def __init__(self, members: Mapping[Kind, Provider]) -> None:
        if not members:
            raise ValueError("a router needs at least one member")
        own = dict(zip(members, _own(list(members.values())), strict=True))
        for kind, member in own.items():
            if kind not in member.capabilities.kinds:
                kinds = ", ".join(sorted(member.capabilities.kinds))
                raise ValueError(
                    f"kind {kind!r} is routed to {_label(member)}, whose record lacks it "
                    f"(its kinds: {kinds})"
                )
        self._members: dict[Kind, _Member] = own
        self.capabilities = _derive_router(self._members)

    async def aclose(self) -> None:
        """Close each distinct member once, in member order, every member tried."""
        await _close_all({id(member): member for member in self._members.values()}.values())

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _shares(self, request: Request) -> tuple[tuple[_Member, Request], ...]:
        """One request per member, in the mapping's order of first appearance; a member
        with no questions gets none."""
        groups: dict[int, tuple[_Member, list[Question]]] = {
            id(member): (member, []) for member in self._members.values()
        }
        for question in request.questions:
            groups[id(self._members[question.kind])][1].append(question)
        return tuple(
            (
                member,
                Request(
                    state=request.state,
                    questions=tuple(qs),
                    deadline=request.deadline,
                    options=request.options,
                ),
            )
            for member, qs in groups.values()
            if qs
        )

    def _plan(self, request: Request) -> tuple[tuple[_Member, Request], ...]:
        """Check the derived record, then every member's share, before any call."""
        _check_composite(request, self.capabilities, routed_kinds=True)
        shares = self._shares(request)
        for member, share in shares:
            member._plan(share)
        return shares

    async def answer(self, request: Request) -> Response:
        shares = self._plan(request)
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(_answer_or_failure(m, share)) for m, share in shares]
        except BaseExceptionGroup as failures:
            # A member's bug is the caller's to see as itself, not wrapped by the group.
            raise failures.exceptions[0] from None
        answers: dict[str, Answer] = {}
        outcomes: tuple[Outcome, ...] = ()
        warnings: tuple[Notice, ...] = ()
        answered = False
        for task in tasks:
            response, ok = task.result()
            answers.update(response.answers)
            outcomes += response.outcomes
            warnings += response.warnings
            answered = answered or ok
        response = Response(answers=answers, outcomes=outcomes, warnings=warnings)
        if not answered:
            raise RequestFailed(response)
        return response


class Fallback:
    """Members tried in order with the same request and the same deadline.

    It falls through when a member raises ``RequestFailed`` and on nothing else: a
    ``DoesNotFit`` propagates. The failed members' attempts are listed on the first outcome
    of the member that answered, whose latency then covers its own wire request and the
    time the earlier members took. When every member fails, ``RequestFailed`` carries
    every member's outcomes in member order.
    """

    capabilities: Capabilities

    def __init__(self, members: Sequence[Provider]) -> None:
        if not members:
            raise ValueError("a fallback needs at least one member")
        self._members = _own(members)
        self.capabilities = _derive_fallback(self._members)

    async def aclose(self) -> None:
        """Close each distinct member once, in member order, every member tried."""
        await _close_all({id(member): member for member in self._members}.values())

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _plan(self, request: Request) -> None:
        """Check the derived record, then every member against the whole request."""
        _check_composite(request, self.capabilities)
        for member in self._members:
            member._plan(request)

    async def answer(self, request: Request) -> Response:
        self._plan(request)
        started_at = time.time()
        start = time.monotonic()
        failures: list[Response] = []
        for member in self._members:
            member_start = time.monotonic()
            try:
                response = await member.answer(request)
            except RequestFailed as failure:
                failures.append(failure.response)
                continue
            if not failures or not response.outcomes:
                return response
            carried = tuple(
                attempt
                for failed in failures
                for outcome in failed.outcomes
                for attempt in outcome.failed_attempts
            )
            first, *rest = response.outcomes
            first = replace(
                first,
                failed_attempts=carried + first.failed_attempts,
                latency=first.latency + (member_start - start),
                started_at=started_at,
            )
            return Response(
                answers=response.answers, outcomes=(first, *rest), warnings=response.warnings
            )
        raise RequestFailed(
            Response(
                answers={},
                outcomes=tuple(outcome for failed in failures for outcome in failed.outcomes),
                warnings=tuple(warning for failed in failures for warning in failed.warnings),
            )
        )


_Member: TypeAlias = Dispatcher | Router | Fallback
"""What a composite is built from: idia's own providers, each able to preflight."""


# ----------------------------------------------------------------------------------------
# Deriving a composite's capabilities
# ----------------------------------------------------------------------------------------

_OPTION_KINDS: tuple[Kind, ...] = ("choice", "score", "noul")


def _min_limit(values: Sequence[Limit]) -> Limit:
    """The most restrictive of several limits: the smallest known number; else ``None``
    when any is unknown; else ``UNBOUNDED``, every member documenting no cap."""
    known = [value for value in values if isinstance(value, int)]
    if known:
        return min(known)
    if not values or any(value is None for value in values):
        return None
    return UNBOUNDED


def _fewest(values: Sequence[int | None]) -> int | None:
    """The fewest decimals, a known number beating ``None``."""
    known = [value for value in values if value is not None]
    return min(known) if known else None


def _truncates(values: Sequence[bool | None]) -> bool | None:
    """True if any member truncates silently, else ``None`` if any is undocumented."""
    if any(value is True for value in values):
        return True
    if any(value is None for value in values):
        return None
    return False


def _confidence(
    kinds: frozenset[Kind], sources: Mapping[Kind, Sequence[Capabilities]]
) -> dict[Kind, str]:
    """The formula per kind from the records that answer it: absent when any record lacks
    the kind, ``mixed`` when they differ."""
    result: dict[Kind, str] = {}
    for kind in sorted(kinds):
        formulas = {record.confidence.get(kind) for record in sources[kind]}
        if not formulas or None in formulas:
            continue
        formula = formulas.pop()
        result[kind] = formula if not formulas and formula is not None else "mixed"
    return result


def _derived(
    records: Sequence[Capabilities],
    *,
    kinds: frozenset[Kind],
    modalities: frozenset[Modality],
    labels_from: Sequence[Capabilities],
    levels_from: Sequence[Capabilities],
    options_from: Sequence[Capabilities],
    nouls_from: Sequence[Capabilities],
    confidence: Mapping[Kind, str],
) -> Capabilities:
    """The record every field of which takes the most restrictive value over ``records``,
    the kind-scoped caps over the records that answer those kinds."""
    carriers: frozenset[ImageCarrier] = frozenset()
    if "image" in modalities:
        carriers = frozenset(c for record in records for c in record.image_carriers)
    min_levels = [record.min_levels for record in levels_from if record.min_levels is not None]
    # The image fields come from the members that take images alone, each None when none
    # of them knows it; the composite's own check never reads them.
    imaging = [record for record in records if "image" in record.modalities]
    sides = [record.min_image_side for record in imaging if record.min_image_side is not None]
    known_types = [
        record.image_media_types for record in imaging if record.image_media_types is not None
    ]
    return Capabilities(
        kinds=kinds,
        modalities=modalities,
        image_carriers=carriers,
        max_input_tokens=_min_limit([record.max_input_tokens for record in records]),
        max_question_tokens=_min_limit([record.max_question_tokens for record in records]),
        max_body_bytes=_min_limit([record.max_body_bytes for record in records]),
        max_questions=_min_limit([record.max_questions for record in records]),
        max_images=_min_limit([record.max_images for record in records]),
        max_images_per_question=_min_limit([record.max_images_per_question for record in records]),
        max_image_bytes=_min_limit([record.max_image_bytes for record in imaging]),
        max_total_image_bytes=_min_limit([record.max_total_image_bytes for record in imaging]),
        max_labels=_min_limit([record.max_labels for record in labels_from]),
        max_levels=_min_limit([record.max_levels for record in levels_from]),
        max_total_options=_min_limit([record.max_total_options for record in options_from]),
        min_levels=max(min_levels) if min_levels else None,
        min_image_side=max(sides) if sides else None,
        image_media_types=frozenset[str].intersection(*known_types) if known_types else None,
        key_patterns=tuple(pattern for record in records for pattern in record.key_patterns),
        recounts_state_per_question=any(r.recounts_state_per_question for r in records),
        returns_refusal=any(record.returns_refusal for record in records),
        calibrated=all(record.calibrated for record in records),
        snapshot_pinnable=all(record.snapshot_pinnable for record in records),
        # Asks more of the caller when false; false too when no member answers a noul.
        carries_noul_descriptions=bool(nouls_from)
        and all(record.carries_noul_descriptions for record in nouls_from),
        carries_structured_instructions=all(
            record.carries_structured_instructions for record in records
        ),
        truncates_silently=_truncates([record.truncates_silently for record in records]),
        confidence=confidence,
        rounding=Rounding(
            probability_decimals=_fewest([r.rounding.probability_decimals for r in records]),
            score_decimals=_fewest([r.rounding.score_decimals for r in records]),
        ),
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
        composite=True,
    )


def _derive_router(members: Mapping[Kind, _Member]) -> Capabilities:
    """Kinds are the mapping's keys, modalities the union, the kind-scoped fields from the
    members that route those kinds, everything else the most restrictive over the distinct
    members in first-appearance order."""
    distinct = list({id(member): member for member in members.values()}.values())
    records = [member.capabilities for member in distinct]

    def routing(kinds: Sequence[Kind]) -> list[Capabilities]:
        seen: dict[int, Capabilities] = {}
        for kind in kinds:
            member = members.get(kind)
            if member is not None:
                seen.setdefault(id(member), member.capabilities)
        return list(seen.values())

    kinds = frozenset(members)
    return _derived(
        records,
        kinds=kinds,
        modalities=frozenset(m for record in records for m in record.modalities),
        labels_from=routing(("choice",)),
        levels_from=routing(("score",)),
        options_from=routing(_OPTION_KINDS),
        nouls_from=routing(("noul",)),
        confidence=_confidence(kinds, {kind: [members[kind].capabilities] for kind in kinds}),
    )


def _derive_fallback(members: Sequence[_Member]) -> Capabilities:
    """Kinds and modalities are the intersection and every field the most restrictive
    over all members; a kind absent from any member's confidence map is absent."""
    records = [member.capabilities for member in members]
    kinds = frozenset[Kind].intersection(*(record.kinds for record in records))
    modalities = frozenset[Modality].intersection(*(record.modalities for record in records))
    return _derived(
        records,
        kinds=kinds,
        modalities=modalities,
        labels_from=records,
        levels_from=records,
        options_from=records,
        nouls_from=records,
        confidence=_confidence(kinds, dict.fromkeys(kinds, records)),
    )
