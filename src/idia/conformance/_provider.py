"""The provider half: P1 to P23, run against a wrapped leaf or a composite."""

from __future__ import annotations

import dataclasses
import inspect
import tempfile
import time
import zlib
from collections import Counter
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

from idia.conformance._base import (
    Check,
    CheckNotApplicable,
    Exchange,
    Failure,
    Perturbed,
    ProviderHarness,
    Success,
    Unreadable,
    expect,
    is_a,
    missing,
    validate_spy,
)
from idia.core import (
    UNBOUNDED,
    WARNING_UNCHECKED_LIMIT,
    Capabilities,
    Deadline,
    DeadlineExceeded,
    DoesNotFit,
    Fault,
    ImagePart,
    JsonPart,
    Kind,
    Label,
    Outcome,
    Provider,
    ProviderError,
    Question,
    Request,
    RequestFailed,
    Response,
    Rule,
    State,
)
from idia.dispatch import Dispatcher
from idia.providers.recorded import Recorded
from idia.retry import RetryPolicy

_KINDS: Final[tuple[Kind, ...]] = ("noul", "choice", "score", "extraction", "box")
_MISFIT_KEY: Final = "conformance_misfit"
_SIZE_LIMITS: Final = ("max_input_tokens", "max_question_tokens", "max_body_bytes")
_T = TypeVar("_T")


async def _outcome(call: Awaitable[_T]) -> _T | Exception:
    try:
        return await call
    except Exception as error:
        return error


def _answering(harness: ProviderHarness, request: Request) -> list[list[Exchange]]:
    """Enough successes for every member to answer every wire request it could be sent."""
    plenty = len(request.questions) + 2
    return [[Success()] * plenty for _ in range(harness.members)]


def _first(
    first: Sequence[Exchange], harness: ProviderHarness, request: Request
) -> list[list[Exchange]]:
    """``first`` then successes for member 0; successes for the others."""
    exchanges = _answering(harness, request)
    exchanges[0] = [*first, *exchanges[0]]
    return exchanges


def _asked(request: Request) -> list[str]:
    return [question.key for question in request.questions]


def _question(kind: Kind, labels: int = 2) -> Question:
    names = tuple(Label(name=f"option {i}") for i in range(labels))
    if kind in ("choice", "score"):
        return Question(key=_MISFIT_KEY, kind=kind, instruction="Which?", labels=names)
    if kind == "extraction":
        return Question(key=_MISFIT_KEY, kind=kind, instruction="What?", value_type="text")
    return Question(key=_MISFIT_KEY, kind=kind, instruction="Is it so?")


def _misfits(request: Request, record: Capabilities) -> list[tuple[str, Request]]:
    """Requests that do not fit ``record``: an undeclared kind, an undeclared image, a
    choice one label past a known ``max_labels`` and a score one level past a known
    ``max_levels``."""
    found: list[tuple[str, Request]] = []
    undeclared: list[Kind] = [kind for kind in _KINDS if kind not in record.kinds]
    if undeclared:
        bad = (*request.questions, _question(undeclared[0]))
        found.append(("an undeclared kind", dataclasses.replace(request, questions=bad)))
    if "image" not in record.modalities:
        image = ImagePart(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
        state = State(parts=(*request.state.parts, image))
        found.append(("an undeclared modality", dataclasses.replace(request, state=state)))
    if isinstance(record.max_labels, int) and "choice" in record.kinds:
        bad = (*request.questions, _question("choice", record.max_labels + 1))
        found.append(("a choice past max_labels", dataclasses.replace(request, questions=bad)))
    if isinstance(record.max_levels, int) and "score" in record.kinds:
        bad = (*request.questions, _question("score", record.max_levels + 1))
        found.append(("a score past max_levels", dataclasses.replace(request, questions=bad)))
    return found


def _composite(harness: ProviderHarness) -> None:
    if harness.shape == "leaf":
        raise CheckNotApplicable("a wrapped leaf is not a composite")


def _sent_keys(harness: ProviderHarness) -> list[tuple[str, ...]]:
    return [
        tuple(question.key for question in wire_request.questions)
        for log in harness.member_calls()
        for wire_request, _ in log
    ]


def _resent(harness: ProviderHarness) -> list[str]:
    """Keys of any wire request a member received twice: a retry."""
    return [
        ",".join(question.key for question in wire_request.questions)
        for log in harness.member_calls()
        for wire_request, count in Counter(wire_request for wire_request, _ in log).items()
        if count > 1
    ]


async def _answer(
    harness: ProviderHarness,
    exchanges: Sequence[Sequence[Exchange]],
    request: Request,
    **overrides: object,
) -> Response:
    provider = harness.make(exchanges, overrides=overrides or None)
    return await provider.answer(request)


# ----------------------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------------------


async def _p1(harness: ProviderHarness) -> str | None:
    provider = harness.make(_answering(harness, harness.request()))
    expect(is_a(provider, Provider), "the provider does not satisfy Provider")
    expect(is_a(provider.capabilities, Capabilities), "its capabilities is not a Capabilities")
    return None


async def _p2(harness: ProviderHarness) -> str | None:
    request = harness.request()
    response = await _answer(harness, _answering(harness, request), request)
    expect(is_a(response, Response), f"answer returned a {type(response).__name__}")
    asked = _asked(request)
    expect(
        sorted(response.answers) == sorted(asked),
        f"answers keyed {sorted(response.answers)} for questions {sorted(asked)}",
    )
    return None


async def _refused(
    harness: ProviderHarness, label: str, request: Request, **overrides: object
) -> DoesNotFit:
    provider = harness.make(_answering(harness, request), overrides=overrides or None)
    raised = await _outcome(provider.answer(request))
    expect(isinstance(raised, DoesNotFit), f"{label} ended in {raised!r}, not DoesNotFit")
    assert isinstance(raised, DoesNotFit)
    sent = harness.requests_sent()
    expect(sent == 0, f"{label} sent {sent} wire requests before it was refused")
    return raised


async def _p3(harness: ProviderHarness) -> str | None:
    record = harness.expected_capabilities
    misfits = _misfits(harness.request(), record)
    limits = [name for name in _SIZE_LIMITS if isinstance(getattr(record, name), int)]
    if not misfits and not limits:
        raise CheckNotApplicable("the record leaves no kind, modality, cap or limit to exceed")
    for label, request in misfits:
        await _refused(harness, label, request)
    for name in limits:
        await _refused(harness, f"{name} set to 1", harness.request(), **{name: 1})
    return None


async def _p4(harness: ProviderHarness) -> str | None:
    request = harness.request()
    response = await _answer(harness, _answering(harness, request), request, max_body_bytes=None)
    for outcome in response.outcomes:
        expect(
            any(
                w.code == WARNING_UNCHECKED_LIMIT and w.field == "max_body_bytes"
                for w in outcome.warnings
            ),
            f"the outcome for {outcome.keys} does not warn that max_body_bytes went unchecked",
        )
    return None


async def _p5(harness: ProviderHarness) -> str | None:
    request = harness.request()
    response = await _answer(
        harness, _answering(harness, request), request, max_body_bytes=UNBOUNDED
    )
    warnings = [*response.warnings, *(w for o in response.outcomes for w in o.warnings)]
    warned = [w for w in warnings if w.field == "max_body_bytes"]
    expect(not warned, f"an UNBOUNDED max_body_bytes was warned: {warned}")
    return None


async def _p6(harness: ProviderHarness) -> str | None:
    record = harness.expected_capabilities
    off = record.opt_in_features - record.enabled_features
    if not off:
        raise CheckNotApplicable("the provider declares no opt-in feature left off")
    misfits = _misfits(harness.request(), record)
    if not misfits:
        raise CheckNotApplicable("the record leaves no kind, modality or label cap to exceed")
    for label, request in misfits:
        refused = await _refused(harness, label, request)
        expected = off
        if refused.check in ("kinds", "modalities"):
            absent = _missing(request, record, refused.check)
            expect(
                refused.observed == absent,
                f"{label} observes {refused.observed!r}, not the missing {sorted(absent)}",
            )
            adds = record.feature_kinds if refused.check == "kinds" else record.feature_modalities
            expected = frozenset(f for f in off if absent & adds.get(f, frozenset()))
        expect(
            refused.off_features == expected,
            f"{label} names off features {sorted(refused.off_features)}, not {sorted(expected)}",
        )
    return None


def _missing(request: Request, record: Capabilities, check: str) -> frozenset[str]:
    """The kinds, or the modalities, ``request`` uses and ``record`` lacks."""
    if check == "kinds":
        return frozenset(q.kind for q in request.questions) - record.kinds
    used = {
        "image" if isinstance(part, ImagePart) else "json" if isinstance(part, JsonPart) else "text"
        for part in request.state.parts
    }
    return frozenset(used) - record.modalities


async def _p7(harness: ProviderHarness) -> str | None:
    request = harness.request()
    response = await _answer(harness, _answering(harness, request), request, max_questions=1)
    carried = [outcome.keys for outcome in response.outcomes]
    expect(
        carried == _sent_keys(harness),
        f"outcomes carry {carried}, the wire requests sent carried {_sent_keys(harness)}",
    )
    flat = sorted(key for keys in carried for key in keys)
    expect(flat == sorted(_asked(request)), f"the outcomes name {flat}, not each key once")
    return None


async def _p8(harness: ProviderHarness) -> str | None:
    request = harness.request()
    response = await _answer(harness, _answering(harness, request), request)
    expect(not hasattr(response, "provenance"), "the Response has a provenance attribute")
    members = set(harness.member_names())
    for outcome in response.outcomes:
        absent = [
            name
            for name, held in (
                ("provider", bool(outcome.provider)),
                ("model_requested", bool(outcome.model_requested)),
                ("model_reported", outcome.model_reported is not None),
                ("elicitation", outcome.elicitation is not None),
                ("calibrated", outcome.calibrated is not None),
                ("usage", outcome.usage is not None),
                ("latency", outcome.latency >= 0),
                ("started_at", outcome.started_at > 0),
                ("request_id", outcome.request_id is not None),
            )
            if not held
        ]
        expect(not absent, f"the outcome for {outcome.keys} leaves out {absent}")
        named = (outcome.provider, outcome.model_requested)
        expect(named in members, f"the outcome names {named}, which is no member's leaf")
        held = [f.name for f in dataclasses.fields(outcome)]
        raw = [name for name in held if isinstance(getattr(outcome, name), bytes | bytearray)]
        expect("raw_body" not in held and not raw, f"the outcome carries a raw body: {raw}")
    return None


async def _p9(harness: ProviderHarness) -> str | None:
    request = harness.request()
    exchanges = _first([Failure(status=429)], harness, request)
    response = await _answer(harness, exchanges, request)
    retried = [outcome for outcome in response.outcomes if outcome.failed_attempts]
    expect(
        len(retried) == 1 and len(retried[0].failed_attempts) == 1,
        f"a 429 then an answer gave {[len(o.failed_attempts) for o in response.outcomes]} "
        "failed attempts per outcome, not one",
    )
    expect(all(o.error is None for o in response.outcomes), "an outcome did not answer")
    expect(retried[0].attempts == 2, f"the retried outcome counts {retried[0].attempts} attempts")
    stored = {field.name for field in dataclasses.fields(Outcome)}
    derived = isinstance(inspect.getattr_static(Outcome, "attempts", None), property)
    expect("attempts" not in stored and derived, "the attempt count is stored, not derived")
    return None


async def _p10(harness: ProviderHarness) -> str | None:
    request = harness.request()
    expect(len(request.questions) >= 2, "the harness's request has fewer than two questions")
    exchanges = _first([Failure()], harness, request)
    response = await _answer(harness, exchanges, request, max_questions=1)
    failed = [o for o in response.outcomes if o.error is not None]
    answered = [o for o in response.outcomes if o.error is None]
    expect(bool(failed) and bool(answered), "the request did not partly fail")
    for outcome in failed:
        leaked = sorted(set(outcome.keys) & set(response.answers))
        expect(not leaked, f"failed keys {leaked} are in the answers")
    for outcome in answered:
        expect(set(outcome.keys) <= set(response.answers), f"{outcome.keys} lost their answers")
    return None


async def _p11(harness: ProviderHarness) -> str | None:
    request = harness.request()
    failing: list[list[Exchange]] = [
        [Failure()] * (len(request.questions) + 2) for _ in range(harness.members)
    ]
    provider = harness.make(failing)
    raised = await _outcome(provider.answer(request))
    expect(isinstance(raised, RequestFailed), f"no answer ended in {raised!r}, not RequestFailed")
    assert isinstance(raised, RequestFailed)
    response = raised.response
    expect(bool(response.outcomes) and not response.answers, "RequestFailed's response is empty")
    for index, outcome in enumerate(response.outcomes):
        expect(outcome.error is not None, f"the outcome for {outcome.keys} has no error")
        expect(
            isinstance(outcome.error, DeadlineExceeded)
            or bool(outcome.failed_attempts)
            or _unsent_after_credential_error(outcome, response.outcomes[:index]),
            f"the outcome for {outcome.keys} lists no failed attempts",
        )
    return None


def _unsent_after_credential_error(outcome: Outcome, earlier: Sequence[Outcome]) -> bool:
    """An outcome left unsent by an earlier credential error: no attempts, no request id,
    and the very error object of an earlier outcome, not retryable, status 401 or 403."""
    error = outcome.error
    return (
        outcome.attempts == 0
        and outcome.request_id is None
        and error is not None
        and not error.is_retryable
        and error.status_code in (401, 403)
        and any(other.error is error for other in earlier)
    )


async def _p12(harness: ProviderHarness) -> str | None:
    request = harness.request()
    kinds: list[Kind] = [q.kind for q in request.questions if q.kind in ("choice", "score")]
    plenty = len(request.questions) + 2
    skipped: list[str] = []
    if not kinds:
        skipped.append("the fault half was not run: the request has no question whose sum can move")
    else:
        moved: list[list[Exchange]] = [
            [Perturbed(kinds[0], "sum"), *[Success()] * plenty] for _ in range(harness.members)
        ]
        try:
            provider = harness.make(moved)
        except CheckNotApplicable as not_applicable:
            skipped.append(f"the fault half was not run: {not_applicable.reason}")
        else:
            response = await provider.answer(request)
            faulted = [
                key
                for key, answer in response.answers.items()
                if isinstance(answer, Fault)
                and any(f.rule is Rule.sum_tolerance for f in answer.failures)
            ]
            expect(bool(faulted), "a moved sum did not come back as a fault")
            expect(not _resent(harness), f"a validation fault was retried: {_resent(harness)}")
            expect(
                all(not o.failed_attempts for o in response.outcomes),
                "a fault became a failed attempt",
            )
    unreadable: list[list[Exchange]] = [
        [Unreadable("not_json"), *[Success()] * plenty] for _ in range(harness.members)
    ]
    try:
        provider = harness.make(unreadable)
    except CheckNotApplicable as not_applicable:
        skipped.append(f"the unreadable-body half was not run: {not_applicable.reason}")
    else:
        await _outcome(provider.answer(request))
        expect(not _resent(harness), f"an unreadable body was retried: {_resent(harness)}")
    if len(skipped) == 2:
        raise CheckNotApplicable("; ".join(skipped))
    return "; ".join(skipped) or None


async def _p13(harness: ProviderHarness) -> str | None:
    request = harness.request()
    provider = harness.make(_answering(harness, request))
    with validate_spy() as validated:
        response = await provider.answer(request)
    counts = Counter(key for key, _ in validated)
    twice = sorted(key for key, n in counts.items() if n != 1)
    expect(not twice, f"validate ran more than once for {twice}")
    answered = {key for key, answer in response.answers.items() if not missing(answer)}
    never = sorted(answered - set(counts))
    expect(not never, f"answers returned without passing validate: {never}")
    return None


async def _p14(harness: ProviderHarness) -> str | None:
    _composite(harness)
    request = harness.request()
    provider = harness.make(_answering(harness, request))
    record = provider.capabilities
    expected = harness.expected_capabilities
    expect(record.composite, "the composite's record is not marked composite")
    differ = [
        field.name
        for field in dataclasses.fields(Capabilities)
        if field.name not in ("kinds", "confidence")
        and getattr(record, field.name) != getattr(expected, field.name)
    ]
    expect(not differ, f"the derived record differs from the expected one in {differ}")
    await provider.answer(request)
    expect(provider.capabilities is record, "the record was derived again after building")
    return None


async def _p15(harness: ProviderHarness) -> str | None:
    _composite(harness)
    record = harness.make(_answering(harness, harness.request())).capabilities
    expected = harness.expected_capabilities
    expect(
        record.kinds == expected.kinds,
        f"kinds {sorted(record.kinds)}, not {sorted(expected.kinds)}",
    )
    expect(
        dict(record.confidence) == dict(expected.confidence),
        f"confidence {dict(record.confidence)}, not {dict(expected.confidence)}",
    )
    return None


async def _p16(harness: ProviderHarness) -> str | None:
    _composite(harness)
    misfits = _misfits(harness.request(), harness.expected_capabilities)
    if not misfits:
        raise CheckNotApplicable("the derived record leaves nothing to exceed")
    for label, request in misfits:
        await _refused(harness, label, request)
        called = [i for i, log in enumerate(harness.member_calls()) if log]
        expect(not called, f"{label} reached members {called} before it was refused")
    return None


async def _p17(harness: ProviderHarness) -> str | None:
    if harness.shape != "router":
        raise CheckNotApplicable("only a router splits a request among its members")
    request = harness.request()
    response = await _answer(harness, _answering(harness, request), request, max_questions=1)
    logs = harness.member_calls()
    owners: dict[Kind, set[int]] = {}
    for i, log in enumerate(logs):
        for wire_request, _ in log:
            for question in wire_request.questions:
                owners.setdefault(question.kind, set()).add(i)
    shared = sorted(kind for kind, members in owners.items() if len(members) > 1)
    expect(not shared, f"the questions of {shared} went to more than one member")
    expect(sum(1 for log in logs if log) >= 2, "the request reached fewer than two members")
    packed = [len(w.questions) for log in logs for w, _ in log if len(w.questions) != 1]
    expect(not packed, "a member did not plan its own fit under its question cap")
    carried = [outcome.keys for outcome in response.outcomes]
    expect(carried == _sent_keys(harness), f"outcomes {carried} are not one per member call")
    first = harness.member_names()[0][0]
    expect(
        response.outcomes[0].provider == first,
        f"the first outcome is {response.outcomes[0].provider}'s, not the first member's",
    )
    return None


async def _p18(harness: ProviderHarness) -> str | None:
    if harness.shape != "fallback":
        raise CheckNotApplicable("only a fallback provider falls through")
    for first in (Failure(), Failure(status=401)):
        await _falls_through(harness, first)
    return None


async def _falls_through(harness: ProviderHarness, first: Failure) -> None:
    """Member 0 fails every call with ``first``; the next member answers, given the same
    request and the same deadline, and its first outcome carries member 0's attempts."""
    request = harness.request()
    exchanges = _answering(harness, request)
    exchanges[0] = [first] * len(exchanges[0])
    response = await _answer(harness, exchanges, request)
    named = "a failure" if first.status is None else f"a {first.status}"
    logs = [list(log) for log in harness.member_calls()]
    expect(len(logs) >= 2 and all(logs[:2]), f"the fallback did not fall through after {named}")
    failed, answering = logs[0], logs[1]
    expect(failed[0][0] == answering[0][0], "the second member was not passed the same request")
    deadlines = [deadline for _, deadline in (*failed, *answering)]
    if any(deadline is None for deadline in deadlines):
        raise CheckNotApplicable("the harness cannot see the deadline a member was given")
    expect(
        all(deadline == request.deadline for deadline in deadlines),
        f"after {named}, a member was given a fresh deadline, not what remained of the request's",
    )
    carried = response.outcomes[0].failed_attempts
    expect(
        len(carried) == len(failed)
        and all(a.provider != response.outcomes[0].provider for a in carried),
        f"after {named}, the answering outcome lists {len(carried)} failed attempts; the failed "
        f"member made {len(failed)}",
    )


def _replayable(result: Response | Exception) -> object:
    """What a replay must reproduce of a ``Response`` or ``RequestFailed``: everything but
    the latency and the start times the replay measures again, errors by their fields."""
    if isinstance(result, RequestFailed):
        return ("RequestFailed", _replayable(result.response))
    if isinstance(result, Exception):
        return (type(result).__name__, str(result))
    return (
        dict(result.answers),
        result.warnings,
        tuple(
            (
                outcome.keys,
                _error_fields(outcome.error),
                outcome.provider,
                outcome.model_requested,
                outcome.model_reported,
                outcome.snapshot,
                outcome.elicitation,
                outcome.calibrated,
                outcome.prompt_sent,
                outcome.usage,
                outcome.request_id,
                outcome.warnings,
                tuple(
                    (
                        attempt.provider,
                        attempt.model_requested,
                        _error_fields(attempt.error),
                        attempt.request_id,
                        attempt.usage,
                    )
                    for attempt in outcome.failed_attempts
                ),
            )
            for outcome in result.outcomes
        ),
    )


def _error_fields(error: ProviderError | None) -> object:
    if error is None:
        return None
    return (
        error.message,
        error.is_retryable,
        error.retry_after,
        error.status_code,
        error.request_id,
    )


async def _p19(harness: ProviderHarness) -> str | None:
    request = harness.request()
    plenty = len(request.questions) + 2
    runs: dict[str, list[Exchange]] = {
        "an answer": [Success()] * plenty,
        "a 503 then an answer": [Failure(status=503), *[Success()] * plenty],
        "a failure": [Failure()] * plenty,
    }
    retry = RetryPolicy.without_delay()
    with tempfile.TemporaryDirectory() as directory:
        for n, (label, exchanges) in enumerate(runs.items()):
            log = Path(directory) / f"p19-{n}.json"
            recorder = Recorded(harness.leaf(exchanges), log, "once")
            recorded = await _outcome(Dispatcher(recorder, retry=retry).answer(request))
            expect(
                isinstance(recorded, Response | RequestFailed),
                f"recording {label} ended in {recorded!r}",
            )
            expect(harness.requests_sent() > 0, f"recording {label} sent no wire request")
            if label == "a 503 then an answer":
                expect(
                    isinstance(recorded, Response)
                    and any(outcome.failed_attempts for outcome in recorded.outcomes),
                    f"recording {label} retried nothing: {recorded!r}",
                )
            replayer = Recorded(harness.leaf([]), log, "none")
            replayed = await _outcome(Dispatcher(replayer, retry=retry).answer(request))
            sent = harness.requests_sent()
            expect(sent == 0, f"the replay of {label} sent {sent} wire requests")
            expect(
                _replayable(replayed) == _replayable(recorded),
                f"the replay of {label} gave {replayed!r}, not what was recorded",
            )
    return None


async def _credential_stop(harness: ProviderHarness, status: int) -> str | None:
    """Every member fails with a non-retryable ``status``. The stop ends at the member:
    each member makes its first call and leaves the rest of its plan unsent."""
    request = harness.request()
    plenty = len(request.questions) + 2
    refused: list[list[Exchange]] = [
        [Failure(status=status)] * plenty for _ in range(harness.members)
    ]
    provider = harness.make(refused, overrides={"max_questions": 1})
    raised = await _outcome(provider.answer(request))
    expect(isinstance(raised, RequestFailed), f"a {status} on every call ended in {raised!r}")
    assert isinstance(raised, RequestFailed)
    called = [len(log) for log in harness.member_calls()]
    expect(
        all(n == 1 for n in called),
        f"after a {status}, the members made {called} calls; each makes exactly its first",
    )
    names = harness.member_names()
    if len(set(names)) != len(names):
        raise CheckNotApplicable("two members share a name and model, so their outcomes mix")
    outcomes = raised.response.outcomes
    unsent = 0
    for member, name in enumerate(names):
        own = [(i, o) for i, o in enumerate(outcomes) if (o.provider, o.model_requested) == name]
        expect(bool(own), f"member {member} has no outcome")
        (first_index, first), rest = own[0], own[1:]
        tried = [a for a in first.failed_attempts if (a.provider, a.model_requested) == name]
        error = first.error
        expect(
            error is None or not error.is_retryable,
            f"member {member}'s leaf mapped a {status} as retryable: {error!r}",
        )
        expect(
            len(tried) == 1 and error is not None and error.status_code == status,
            f"member {member}'s first outcome carries {error!r} after {len(tried)} "
            f"attempts, not one non-retryable {status}",
        )
        for index, outcome in rest:
            expect(
                outcome.error is first.error
                and _unsent_after_credential_error(outcome, outcomes[first_index:index]),
                f"member {member}'s outcome for {outcome.keys} was sent, or lost its {status}",
            )
        unsent += len(rest)
    if not unsent:
        expect(harness.shape == "router", "the plan left nothing unsent; one question per call")
        return "no member's share held a second wire request, so nothing was left unsent"
    return None


async def _p20(harness: ProviderHarness) -> str | None:
    return await _credential_stop(harness, 401)


async def _p21(harness: ProviderHarness) -> str | None:
    return await _credential_stop(harness, 403)


async def _p22(harness: ProviderHarness) -> str | None:
    fresh = harness.request()
    request = dataclasses.replace(fresh, deadline=Deadline(at=time.monotonic() - 1.0))
    provider = harness.make(_answering(harness, request))
    raised = await _outcome(provider.answer(request))
    expect(
        isinstance(raised, RequestFailed),
        f"a request whose deadline had passed ended in {raised!r}, not RequestFailed",
    )
    assert isinstance(raised, RequestFailed)
    called = [len(log) for log in harness.member_calls()]
    sent = harness.requests_sent()
    expect(
        not any(called) and sent == 0,
        f"a request whose deadline had passed sent {sent} wire requests, members {called}",
    )
    outcomes = raised.response.outcomes
    expect(bool(outcomes), "RequestFailed's response has no outcome")
    for outcome in outcomes:
        expect(
            isinstance(outcome.error, DeadlineExceeded)
            and outcome.attempts == 0
            and not outcome.failed_attempts,
            f"the outcome for {outcome.keys} is {outcome.error!r} with {outcome.attempts} "
            "attempts, not DeadlineExceeded with none",
        )
    return None


def _png(side: int, other: int | None = None, size: int = 0) -> bytes:
    """A PNG's signature and IHDR for an image ``side`` wide and ``other`` (else ``side``)
    high, padded with zeros to ``size`` bytes: what the readers of a type and a size read."""
    high = side if other is None else other
    fields = side.to_bytes(4, "big") + high.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    crc = zlib.crc32(b"IHDR" + fields).to_bytes(4, "big")
    data = b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + fields + crc
    return data + bytes(max(size - len(data), 0))


def _jpeg(side: int, other: int | None = None, size: int = 0) -> bytes:
    """A JPEG's start and one start-of-frame segment for an image ``side`` wide and
    ``other`` (else ``side``) high, padded with zeros to ``size`` bytes."""
    high = side if other is None else other
    frame = b"\x08" + high.to_bytes(2, "big") + side.to_bytes(2, "big") + b"\x03"
    data = b"\xff\xd8\xff\xc0" + (len(frame) + 2).to_bytes(2, "big") + frame
    return data + bytes(max(size - len(data), 0))


_GIF: Final = b"GIF89a" + bytes(32)
_WEBP: Final = b"RIFF\x00\x00\x00\x00WEBPVP8 " + bytes(32)


def _image_drivers(
    record: Capabilities, held: int = 0
) -> list[tuple[str, tuple[ImagePart, ...], dict[str, object]]]:
    """For each known image field, images that break it and fit every other field, with
    the overrides that keep the other limits out of the way; ``held`` is the images the
    request already carries."""
    accepted = record.image_media_types
    build: Callable[..., bytes] | None = None
    if accepted is None or "image/png" in accepted:
        build = _png
    elif "image/jpeg" in accepted:
        build = _jpeg
    least = record.min_image_side or 1
    room: dict[str, object] = {
        "max_body_bytes": UNBOUNDED,
        "max_input_tokens": UNBOUNDED,
        "max_question_tokens": UNBOUNDED,
    }
    drivers: list[tuple[str, tuple[ImagePart, ...], dict[str, object]]] = []
    if build is not None and isinstance(record.max_image_bytes, int):
        caps = {**room, "max_image_bytes": 4096, "max_total_image_bytes": UNBOUNDED}
        drivers.append(("max_image_bytes", (ImagePart(data=build(least, size=4097)),), caps))
    counts = (record.max_images, record.max_images_per_question)
    two = all(not isinstance(n, int) or n >= held + 2 for n in counts)
    if build is not None and isinstance(record.max_total_image_bytes, int) and two:
        caps = {**room, "max_image_bytes": 4096, "max_total_image_bytes": 4096}
        pair = (ImagePart(data=build(least, size=2100)),) * 2
        drivers.append(("max_total_image_bytes", pair, caps))
    if build is not None and record.min_image_side:
        drivers.append(("min_image_side", (ImagePart(data=build(least, least - 1)),), room))
    if accepted is not None:
        if "image/gif" not in accepted:
            gif = (ImagePart(data=_GIF, media_type="image/gif"),)
            drivers.append(("image_media_types", gif, room))
        if "image/webp" not in accepted:
            drivers.append(("image_media_types", (ImagePart(data=_WEBP),), room))
    return drivers


async def _p23(harness: ProviderHarness) -> str | None:
    record = harness.expected_capabilities
    if "image" not in record.modalities:
        raise CheckNotApplicable("the provider takes no images")
    held = sum(isinstance(part, ImagePart) for part in harness.request().state.parts)
    drivers = _image_drivers(record, held)
    if not drivers:
        raise CheckNotApplicable("nothing in the record's image limits can be exceeded")
    # A router hands every member the whole state, so a member that takes no images may
    # refuse it on its modalities first: still before any call, but not on the field.
    allowed = {"modalities"} if harness.shape == "router" else set[str]()
    for check, images, overrides in drivers:
        request = harness.request()
        state = State(parts=(*request.state.parts, *images))
        past = dataclasses.replace(request, state=state)
        refused = await _refused(harness, f"an image past {check}", past, **overrides)
        expect(
            refused.check in {check, *allowed},
            f"an image past {check} was refused on {refused.check}",
        )
        called = [i for i, log in enumerate(harness.member_calls()) if log]
        expect(not called, f"an image past {check} reached members {called}")
    return None


_ProviderCheck = Callable[[ProviderHarness], Awaitable[str | None]]

_CHECKS: Final[tuple[tuple[str, str, _ProviderCheck], ...]] = (
    ("P1", "The provider satisfies Provider.", _p1),
    ("P2", "A request that fits gets one answer per question key.", _p2),
    ("P3", "A request that does not fit raises before any wire request.", _p3),
    ("P4", "An unknown limit passes with an unchecked-limit warning on each outcome.", _p4),
    ("P5", "An UNBOUNDED limit passes with no warning naming it.", _p5),
    (
        "P6",
        "The reason a request does not fit names the off opt-in features that may make it "
        "fit: for a missing kind or modality, those the record says add it; for a cap, every "
        "one.",
        _p6,
    ),
    ("P7", "One outcome per wire request, in plan order, naming its keys.", _p7),
    ("P8", "Each outcome carries its provenance and no raw body.", _p8),
    ("P9", "A retryable failure then an answer is one outcome with one failed attempt.", _p9),
    ("P10", "Partly failed: a Response with the failed keys absent.", _p10),
    ("P11", "No wire request answered: RequestFailed carrying the full response.", _p11),
    ("P12", "A fault or an unreadable body takes exactly one attempt.", _p12),
    ("P13", "Every answer passes validate exactly once, never again in the provider.", _p13),
    ("P14", "A composite's record is derived at build, the most restrictive per field.", _p14),
    ("P15", "A composite's kinds and confidence map are derived as expected.", _p15),
    ("P16", "A composite preflights its derived record before calling any member.", _p16),
    ("P17", "A router gives each member its kinds' questions; one outcome per call.", _p17),
    (
        "P18",
        "A fallback passes the same request and what remains of its deadline, after a failure "
        "or a credential error, and carries the attempts.",
        _p18,
    ),
    ("P19", "A recorded leaf replays its calls through the loop with no wire call.", _p19),
    ("P20", "A non-retryable 401 stops each member's plan after its first call.", _p20),
    ("P21", "A non-retryable 403 stops each member's plan after its first call.", _p21),
    (
        "P22",
        "A request whose deadline has passed sends nothing: DeadlineExceeded, no attempts.",
        _p22,
    ),
    (
        "P23",
        "An image over a known cap, under a known minimum side or of a type outside a known "
        "set is refused before any wire request, on the field it breaks.",
        _p23,
    ),
)


def provider_checks(make_provider: Callable[[], ProviderHarness]) -> tuple[Check, ...]:
    """The provider half, P1 to P23, each run against a fresh harness."""

    def bind(check: _ProviderCheck) -> Callable[[], Coroutine[Any, Any, str | None]]:
        async def run() -> str | None:
            return await check(make_provider())

        return run

    return tuple(Check(id, description, bind(check)) for id, description, check in _CHECKS)
