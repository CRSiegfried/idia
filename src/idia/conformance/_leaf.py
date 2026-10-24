"""The leaf half: L1 to L31, run against a leaf a harness builds."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Final, TypeVar

from idia.conformance import _review
from idia.conformance._base import (
    CandidateShape,
    Check,
    CheckNotApplicable,
    Error,
    Exchange,
    Failure,
    LeafHarness,
    Perturbed,
    RefusalOnWire,
    Stall,
    Success,
    Unreadable,
    expect,
    is_a,
    missing,
    profile_problems,
    validate_spy,
)
from idia.core import (
    CANDIDATE_KINDS,
    WARNING_EXTRA_ANSWER_KEY,
    WARNING_NO_SNAPSHOT,
    WARNING_UNKNOWN_OPTION,
    Answer,
    BoxAnswer,
    Capabilities,
    Deadline,
    ExtractionAnswer,
    Fault,
    Kind,
    LabelAnswer,
    Leaf,
    NoulAnswer,
    Provider,
    ProviderError,
    Refusal,
    Rule,
    TokenEstimator,
    WireResponse,
)
from idia.dispatch import Dispatcher

_SECONDS: Final = 30.0
"""The deadline every call but the stall gets."""

_STALL: Final = 0.2
"""The deadline a stalled call gets."""
_GRACE: Final = 2.0
"""How long past the stalled call's deadline the check waits before it gives up."""

_SHAPES: Final[dict[Kind, type]] = {
    "noul": NoulAnswer,
    "choice": LabelAnswer,
    "score": LabelAnswer,
    "extraction": ExtractionAnswer,
    "box": BoxAnswer,
}

_T = TypeVar("_T")


async def _outcome(call: Awaitable[_T]) -> _T | Exception:
    """What the call returned, or the exception it raised."""
    try:
        return await call
    except Exception as error:
        return error


def _claimed(harness: LeafHarness) -> list[Kind]:
    return sorted(harness.capabilities.kinds)


async def _send(harness: LeafHarness, kind: Kind, exchange: Exchange) -> WireResponse:
    """Build a leaf over one exchange and call it with the wire request for ``kind``."""
    leaf = harness.make([exchange])
    return await leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS))


def _changed(response: WireResponse, baseline: WireResponse) -> dict[str, Answer]:
    return {
        key: answer
        for key, answer in response.answers.items()
        if answer != baseline.answers.get(key)
    }


def _confidence(answer: Answer | None) -> float | None:
    if isinstance(answer, LabelAnswer | NoulAnswer | ExtractionAnswer | BoxAnswer):
        return answer.confidence
    return None


def _source_note(leaf: Leaf) -> str:
    return (
        f"read {type(leaf).__module__}, the module that defines {type(leaf).__name__}; "
        "code the leaf keeps in other modules was not read"
    )


# ----------------------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------------------


async def _l1(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    leaf = harness.make([Success(kind)])
    expect(is_a(leaf, Leaf), "the leaf does not satisfy the Leaf protocol")
    name = leaf.name
    model = leaf.model
    record = leaf.capabilities
    estimator = leaf.estimator
    expect(is_a(name, str) and bool(name), f"the leaf's name is {name!r}")
    expect(is_a(model, str) and bool(model), f"the leaf's model is {model!r}")
    expect(is_a(record, Capabilities), "the leaf's capabilities is not a Capabilities")
    expect(is_a(estimator, TokenEstimator), "the leaf's estimator is not an estimator")
    expect(inspect.iscoroutinefunction(leaf.call), "the leaf's call is not a coroutine")
    response = await leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS))
    expect(is_a(response, WireResponse), f"call returned a {type(response).__name__}")
    return None


async def _l2(harness: LeafHarness) -> str | None:
    leaf = harness.make([])
    expect(not isinstance(leaf, Provider), "the leaf is a Provider before it is wrapped")
    wrapped = Dispatcher(harness.make([]))
    expect(is_a(wrapped, Provider), "the leaf wrapped in a Dispatcher is not a Provider")
    return None


async def _l3(harness: LeafHarness) -> str | None:
    declared = harness.capabilities
    record = harness.make([]).capabilities
    expect(is_a(record, Capabilities), "the leaf's capabilities is not a Capabilities")
    expect(record == declared, "the leaf's record is not the one its harness declares")
    for field in (f.name for f in dataclasses.fields(Capabilities)):
        try:
            setattr(record, field, getattr(record, field))
        except AttributeError:
            continue
        raise AssertionError(f"assigning to the record's {field} did not raise")
    return None


async def _l4(harness: LeafHarness) -> str | None:
    leaf = harness.make([])
    module = type(leaf).__module__
    own = {cls.__name__ for cls in type(leaf).__mro__ if cls.__module__ == module}
    found = _review.class_limits(_review.parse(module), own)
    expect(not found, f"a capabilities field is declared as a class attribute: {'; '.join(found)}")
    return _source_note(leaf)


async def _l5(harness: LeafHarness) -> str | None:
    for kind in _claimed(harness):
        response = await _send(harness, kind, Success(kind))
        shape = _SHAPES[kind]
        values = list(response.answers.values())
        wrong = [type(a).__name__ for a in values if not isinstance(a, shape | Refusal)]
        expect(
            any(isinstance(a, shape) for a in values) and not wrong,
            f"the recorded {kind} response mapped to {[type(a).__name__ for a in values]}, "
            f"not {shape.__name__}",
        )
    return None


async def _l6(harness: LeafHarness) -> str | None:
    rules = {Rule.sum_tolerance, Rule.expected_level}
    for kind in _claimed(harness):
        response = await _send(harness, kind, Success(kind))
        faulted = [
            key
            for key, answer in response.answers.items()
            if isinstance(answer, Fault) and any(f.rule in rules for f in answer.failures)
        ]
        expect(not faulted, f"the recorded {kind} response has sum or level faults: {faulted}")
    return None


async def _l7(harness: LeafHarness) -> str | None:
    record = harness.capabilities
    configurations = [record.enabled_features]
    for feature in sorted(record.opt_in_features):
        configurations += [frozenset({feature}), frozenset()]
    for features in configurations:
        built = harness.make([], features).capabilities
        expect(
            features <= built.enabled_features <= built.opt_in_features,
            f"built with {sorted(features)} on, the leaf lists {sorted(built.enabled_features)}",
        )
        wanted = harness.capabilities_for(features)
        differ = [
            f.name
            for f in dataclasses.fields(Capabilities)
            if getattr(built, f.name) != getattr(wanted, f.name)
        ]
        expect(not differ, f"built with {sorted(features)} on, the record differs in {differ}")
        for name in ("feature_kinds", "feature_modalities"):
            expect(
                getattr(built, name) == getattr(record, name),
                f"built with {sorted(features)} on, {name} changes; it holds whatever is enabled",
            )
    return None


async def _l8(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    wire_request = harness.wire_request(kind)
    skipped: list[str] = []
    notes: list[str] = []
    try:
        leaf = harness.make([Failure(status=429), Success(kind)])
    except CheckNotApplicable as no_status:
        try:
            leaf = harness.make([Stall(), Success(kind)])
        except CheckNotApplicable as no_stall:
            skipped.append(
                f"the retryable half was not run: a 429: {no_status.reason}; "
                f"a stall: {no_stall.reason}"
            )
        else:
            call = leaf.call(wire_request, Deadline.after(_STALL))
            cut = await _outcome(asyncio.wait_for(call, _STALL + _GRACE))
            sent = harness.requests_sent()
            expect(sent == 1, f"one call cut by its deadline sent {sent} wire requests")
            notes.append(f"the retryable half ran over a stall, not a 429: {no_status.reason}")
            if not (isinstance(cut, ProviderError) and cut.is_retryable):
                notes.append(
                    f"the stall ended in {cut!r}, not a retryable ProviderError, so the "
                    "retryable half was not exercised"
                )
    else:
        await _outcome(leaf.call(wire_request, Deadline.after(_SECONDS)))
        sent = harness.requests_sent()
        expect(sent == 1, f"one call over a retryable failure sent {sent} wire requests")
        await _outcome(leaf.call(wire_request, Deadline.after(_SECONDS)))
        sent = harness.requests_sent()
        expect(sent == 2, f"the call after a retryable failure sent {sent - 1} wire requests")
    try:
        leaf = harness.make([Success(kind), Success(kind)])
    except CheckNotApplicable as no_success:
        skipped.append(f"the success half was not run: {no_success.reason}")
    else:
        await _outcome(leaf.call(wire_request, Deadline.after(_SECONDS)))
        sent = harness.requests_sent()
        expect(sent == 1, f"one call over a success sent {sent} wire requests")
    if len(skipped) == 2:
        raise CheckNotApplicable("; ".join(skipped))
    return "; ".join(skipped + notes) or None


async def _l9(harness: LeafHarness) -> str | None:
    retries = harness.sdk_retries()
    expect(retries == 0, f"the leaf's SDK retries {retries} times")
    return None


async def _l10(harness: LeafHarness) -> str | None:
    leaf = harness.make([])
    module = type(leaf).__module__
    tree = _review.parse(module)
    found = _review.loop_imports(tree, module) + _review.provider_error_catches(tree, module)
    expect(not found, f"the leaf's module reaches into the call loop: {'; '.join(found)}")
    return _source_note(leaf)


async def _l11(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    drivers: list[Exchange] = [
        Failure(),
        Failure(status=429),
        Error("connection"),
        Error("timeout"),
        Unreadable("not_json"),
        Unreadable("schema"),
        Unreadable("no_answers"),
    ]
    skipped: list[str] = []
    for exchange in drivers:
        try:
            leaf = harness.make([exchange])
        except CheckNotApplicable as not_applicable:
            skipped.append(f"{exchange}: {not_applicable.reason}")
            continue
        raised = await _outcome(leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS)))
        expect(
            isinstance(raised, ProviderError),
            f"over {exchange} the call ended in {type(raised).__name__}, not a ProviderError",
        )
    try:
        leaf = harness.make([Success(kind), Success(kind)])
    except CheckNotApplicable as not_applicable:
        skipped.append(f"a second event loop: {not_applicable.reason}")
    else:
        await leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS))

        async def again() -> WireResponse | Exception:
            return await _outcome(leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS)))

        later = await asyncio.to_thread(asyncio.run, again())
        expect(
            isinstance(later, WireResponse | ProviderError),
            f"on a second event loop the call ended in {type(later).__name__}",
        )
    if len(skipped) == len(drivers) + 1:
        raise CheckNotApplicable(f"the harness can make no failure: {'; '.join(skipped)}")
    return f"not driven: {'; '.join(skipped)}" if skipped else None


def _no_body(error: ProviderError) -> None:
    held = vars(error)
    named = sorted(set(held) & {"body", "raw_body", "content", "text", "response"})
    raw = sorted(name for name, value in held.items() if isinstance(value, bytes | bytearray))
    expect(not named and not raw, f"the ProviderError carries a body: {named + raw}")


async def _l12(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    wire_request = harness.wire_request(kind)
    leaf = harness.make([Failure(status=429, retry_after=1.5, request_id="conformance-429")])
    error = await _outcome(leaf.call(wire_request, Deadline.after(_SECONDS)))
    expect(isinstance(error, ProviderError), f"a 429 ended in {type(error).__name__}")
    assert isinstance(error, ProviderError)
    expect(error.status_code == 429, f"a 429 carries status {error.status_code}")
    expect(error.is_retryable, "a 429 is not retryable")
    expect(error.retry_after == 1.5, f"a 429 asking 1.5 s carries {error.retry_after}")
    expect(error.request_id == "conformance-429", f"the request id is {error.request_id!r}")
    expect(bool(error.message), "the ProviderError has no message")
    _no_body(error)
    leaf = harness.make([Failure()])
    error = await _outcome(leaf.call(wire_request, Deadline.after(_SECONDS)))
    expect(isinstance(error, ProviderError), f"the failure ended in {type(error).__name__}")
    assert isinstance(error, ProviderError)
    expect(error.status_code is not None, "the recorded failure carries no status")
    _no_body(error)
    return None


async def _l13(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    for which in ("not_json", "schema", "no_answers"):
        leaf = harness.make([Unreadable(which)])
        error = await _outcome(leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS)))
        expect(
            isinstance(error, ProviderError) and not error.is_retryable,
            f"an unreadable body ({which}) ended in {error!r}, not a non-retryable ProviderError",
        )
    return None


async def _l14(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    leaf = harness.make([Stall()])
    start = time.monotonic()
    call = leaf.call(harness.wire_request(kind), Deadline.after(_STALL))
    try:
        error = await _outcome(asyncio.wait_for(call, _STALL + _GRACE))
    finally:
        elapsed = time.monotonic() - start
    expect(
        not isinstance(error, TimeoutError),
        f"a stalled call was still running after {elapsed:.1f} s against a {_STALL} s deadline",
    )
    expect(isinstance(error, ProviderError), f"a stalled call ended in {error!r}")
    return None


async def _l15(harness: LeafHarness) -> str | None:
    expect(not await harness.client_closed(injected=True), "the leaf closed an injected client")
    expect(await harness.client_closed(injected=False), "the leaf left its own client open")
    return None


async def _l16(harness: LeafHarness) -> str | None:
    for kind in _claimed(harness):
        leaf = harness.make([Success(kind)])
        wire_request = harness.wire_request(kind)
        with validate_spy() as validated:
            response = await leaf.call(wire_request, Deadline.after(_SECONDS))
        mapped = {k for k, answer in response.answers.items() if not missing(answer)}
        skipped = sorted(mapped - {key for key, _ in validated})
        expect(not skipped, f"answers returned without passing validate: {skipped}")
        declared = harness.capabilities.rounding
        wrong = sorted({key for key, rounding in validated if rounding != declared})
        expect(not wrong, f"validate got a rounding other than the declared one for {wrong}")
    return None


async def _l17(harness: LeafHarness) -> str | None:
    for kind in _claimed(harness):
        response = await _send(harness, kind, Success(kind))
        expected = harness.expected(kind)
        expect(
            set(response.answers) == set(expected),
            f"the {kind} success returned keys {sorted(response.answers)}, "
            f"the wire gave {sorted(expected)}",
        )
        for key, answer in response.answers.items():
            expect(not isinstance(answer, Fault), f"{key}: the wire's answer became {answer!r}")
            expect(
                answer == expected[key],
                f"{key}: returned {answer!r}, the wire gave {expected[key]!r}",
            )
    return None


async def _l18(harness: LeafHarness) -> str | None:
    kinds: list[Kind] = [k for k in _claimed(harness) if k in ("choice", "score")]
    if not kinds:
        raise CheckNotApplicable("the leaf claims no kind whose probabilities sum")
    kind = kinds[0]
    baseline = await _send(harness, kind, Success(kind))
    response = await _send(harness, kind, Perturbed(kind, "sum"))
    changed = _changed(response, baseline)
    expect(len(changed) == 1, f"one moved sum changed {sorted(changed)}")
    [(key, fault)] = changed.items()
    expect(isinstance(fault, Fault) and fault.key == key, f"{key} became {fault!r}, not a Fault")
    assert isinstance(fault, Fault)
    failures = [f for f in fault.failures if f.rule is Rule.sum_tolerance]
    expect(
        bool(failures) and failures[0].observed is not None and failures[0].bound is not None,
        f"the fault for {key} records no sum_tolerance failure with its observed value and bound",
    )
    expect(fault.mapped is not None, f"the fault for {key} does not keep the answer as mapped")
    expect(set(response.answers) == set(baseline.answers), "the siblings did not stand")
    return None


async def _l19(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    asked = {question.key for question in harness.wire_request(kind).questions}
    baseline = await _send(harness, kind, Success(kind))
    response = await _send(harness, kind, Perturbed(kind, "keys"))
    expect(set(response.answers) == asked, f"answers {sorted(response.answers)} for {asked}")
    changed = _changed(response, baseline)
    expect(
        len(changed) == 1 and all(missing(answer) for answer in changed.values()),
        f"one removed key changed {changed!r}, not one missing_key fault",
    )
    [(key, fault)] = changed.items()
    expect(
        isinstance(fault, Fault) and fault.key == key and fault.mapped is None,
        f"the fault for the removed {key} is {fault!r}: wrong key or holds a mapped answer",
    )
    expect(
        any(w.code == WARNING_EXTRA_ANSWER_KEY for w in response.warnings),
        "a key with no question was dropped without a warning",
    )
    return None


async def _l20(harness: LeafHarness) -> str | None:
    if "noul" not in harness.capabilities.kinds:
        raise CheckNotApplicable("the leaf does not claim noul")
    baseline = await _send(harness, "noul", Success("noul"))
    response = await _send(harness, "noul", Perturbed("noul", "noul_range"))
    changed = _changed(response, baseline)
    expect(
        len(changed) == 1
        and all(
            isinstance(a, Fault) and any(f.rule is Rule.probability_range for f in a.failures)
            for a in changed.values()
        ),
        f"a noul outside [0, 1] gave {changed!r}, not one probability_range fault",
    )
    return None


async def _l21(harness: LeafHarness) -> str | None:
    leaf = harness.make([])
    module = type(leaf).__module__
    tree = _review.parse(module)
    found = _review.complements(tree, module)
    expect(not found, f"the leaf's module computes 1 - p: {'; '.join(found)}")
    calls = _review.complement_calls(tree, module)
    expect(not calls, f"the leaf's module uses noul_complement: {'; '.join(calls)}")
    return _source_note(leaf)


async def _l22(harness: LeafHarness) -> str | None:
    declared = harness.capabilities.confidence
    for kind in _claimed(harness):
        response = await _send(harness, kind, Success(kind))
        expected = harness.expected(kind)
        for key, answer in response.answers.items():
            confidence = _confidence(answer)
            if kind not in declared:
                expect(confidence is None, f"{key} carries a confidence {kind} does not declare")
            elif confidence is not None:
                wire = _confidence(expected.get(key))
                expect(confidence == wire, f"{key}: confidence {confidence}, the wire gave {wire}")
    return None


async def _l23(harness: LeafHarness) -> str | None:
    kinds = _claimed(harness)
    if harness.capabilities.returns_refusal:
        response = await _send(harness, kinds[0], RefusalOnWire(kinds[0]))
        values = list(response.answers.values())
        expect(any(isinstance(a, Refusal) for a in values), "a refusal on the wire was lost")
        expect(not any(isinstance(a, Fault) for a in values), "a refusal's siblings faulted")
        return None
    for kind in kinds:
        response = await _send(harness, kind, Success(kind))
        refused = [k for k, a in response.answers.items() if isinstance(a, Refusal)]
        expect(not refused, f"returns_refusal is false, yet {refused} came back as Refusal")
    return None


async def _l24(harness: LeafHarness) -> str | None:
    reasons: list[str] = []
    for kind in _claimed(harness):
        try:
            leaf = harness.make([CandidateShape(kind)])
        except CheckNotApplicable as not_applicable:
            reasons.append(not_applicable.reason)
            continue
        response = await leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS))
        named = [
            failure.observed
            for answer in response.answers.values()
            if isinstance(answer, Fault)
            for failure in answer.failures
            if failure.rule is Rule.candidate_shape
        ]
        expect(
            bool(named) and all(observed in CANDIDATE_KINDS for observed in named),
            f"a candidate shape for {kind} gave {named}, not a candidate_shape fault naming it",
        )
        return None
    raise CheckNotApplicable("; ".join(sorted(set(reasons))))


async def _l25(harness: LeafHarness) -> str | None:
    if "extraction" not in harness.capabilities.kinds:
        raise CheckNotApplicable("the leaf does not claim extraction")
    expected = {
        key: answer
        for key, answer in harness.expected("extraction").items()
        if isinstance(answer, ExtractionAnswer)
    }
    if not expected:
        raise CheckNotApplicable("the harness expects no extraction answer")
    guessed = {
        key: answer
        for key, answer in expected.items()
        if answer.value is None and answer.guess is not None
    }
    response = await _send(harness, "extraction", Success("extraction"))
    for key, wire in expected.items():
        answer = response.answers.get(key)
        expect(
            isinstance(answer, ExtractionAnswer)
            and (answer.value, answer.guess) == (wire.value, wire.guess),
            f"{key}: the wire's value {wire.value!r} and guess {wire.guess!r} became {answer!r}",
        )
    if not guessed:
        return (
            "no expected extraction has a null value and a guess, so only that the leaf fills "
            "no guess was checked, not that a guess is kept and never moved into value"
        )
    return None


async def _l26(harness: LeafHarness) -> str | None:
    for kind in _claimed(harness):
        response = await _send(harness, kind, Success(kind))
        problems = profile_problems(response)
        always = [d for d in ("elicitation", "calibrated") if d in response.unreported]
        if always:
            problems.append(f"{always} must hold a value, never a reason")
        sent = sorted(d for d, reason in response.unreported.items() if reason == "not-sent")
        if sent:
            problems.append(f"{sent} are not-sent, which only Dispatch gives a failed outcome")
        hidden = response.unreported.get("prompt_sent") == "hidden"
        if hidden != harness.prompt_hidden:
            problems.append(
                f"the prompt sent is hidden: {hidden}; the harness says hidden: "
                f"{harness.prompt_hidden}"
            )
        if (response.prompt_sent is None) != harness.prompt_hidden:
            problems.append(
                f"the prompt sent is None: {response.prompt_sent is None}; only a hidden one is"
            )
        if response.raw_body is None:
            problems.append("the raw body is missing")
        expect(not problems, f"the {kind} wire response's provenance: {'; '.join(problems)}")
    return None


async def _l27(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    response = await _send(harness, kind, Success(kind))
    declared = harness.capabilities.calibrated
    expect(
        response.calibrated == declared, f"calibrated {response.calibrated}, declared {declared}"
    )
    return None


async def _l28(harness: LeafHarness) -> str | None:
    pinnable = harness.capabilities.snapshot_pinnable
    for kind in _claimed(harness):
        response = await _send(harness, kind, Success(kind))
        warned = any(w.code == WARNING_NO_SNAPSHOT for w in response.warnings)
        if not pinnable:
            expect(response.snapshot is None, f"a snapshot {response.snapshot!r} is not pinnable")
        expect(
            (response.snapshot is None) == warned,
            f"snapshot {response.snapshot!r} with a no-snapshot warning: {warned}",
        )
        not_pinned = response.unreported.get("snapshot") == "not-pinned"
        expect(
            (response.snapshot is None) == not_pinned,
            f"snapshot {response.snapshot!r} with the reason not-pinned: {not_pinned}",
        )
    return None


_UNKNOWN_OPTION: Final = "idia-conformance-unknown"
"""An option no leaf knows."""


async def _l29(harness: LeafHarness) -> str | None:
    """The answers and the warnings are compared here; the body sent is compared through
    the ``Success`` contract, a harness serving the recorded success only to the request
    recorded for it."""
    kind = _claimed(harness)[0]
    wire_request = harness.wire_request(kind)
    baseline = await _send(harness, kind, Success(kind))
    optioned = dataclasses.replace(
        wire_request, options={**wire_request.options, _UNKNOWN_OPTION: 1}
    )
    leaf = harness.make([Success(kind)])
    response = await leaf.call(optioned, Deadline.after(_SECONDS))
    expect(
        dict(response.answers) == dict(baseline.answers),
        "an unknown option changed the answers",
    )
    added = list((Counter(response.warnings) - Counter(baseline.warnings)).elements())
    expect(
        len(response.warnings) == len(baseline.warnings) + 1
        and len(added) == 1
        and added[0].code == WARNING_UNKNOWN_OPTION
        and _UNKNOWN_OPTION in added[0].message,
        f"an unknown option gave the warnings {added}, not one unknown-option naming it",
    )
    return None


async def _l30(harness: LeafHarness) -> str | None:
    kind = _claimed(harness)[0]
    leaf = harness.make([Success(kind)])
    aclose: Callable[[], Awaitable[None]] | None = getattr(leaf, "aclose", None)
    if aclose is None:
        raise CheckNotApplicable("the leaf has no aclose")
    await aclose()
    raised = await _outcome(leaf.call(harness.wire_request(kind), Deadline.after(_SECONDS)))
    expect(
        isinstance(raised, ProviderError) and not raised.is_retryable,
        f"a call after aclose ended in {raised!r}, not a non-retryable ProviderError",
    )
    sent = harness.requests_sent()
    expect(sent == 0, f"a call after aclose sent {sent} wire requests")
    return None


_IMAGE_FIELDS: Final = (
    "max_image_bytes",
    "max_total_image_bytes",
    "min_image_side",
    "image_media_types",
)


async def _l31(harness: LeafHarness) -> str | None:
    record = harness.capabilities
    configurations: list[frozenset[str]] = [record.enabled_features, frozenset()]
    configurations += [frozenset({feature}) for feature in sorted(record.opt_in_features)]
    for features in dict.fromkeys(configurations):
        built = harness.make([], features).capabilities
        on = f"built with {sorted(features)} on"
        if "image" in built.modalities:
            expect(
                len(built.image_carriers) == 1,
                f"{on}, it takes images by {len(built.image_carriers)} carriers, not one",
            )
            for name in ("max_image_bytes", "max_total_image_bytes"):
                expect(getattr(built, name) != 0, f"{on}, it takes images but {name} is 0")
            continue
        expect(not built.image_carriers, f"{on}, it has an image carrier but no image modality")
        declared = tuple(getattr(built, name) for name in _IMAGE_FIELDS)
        # Each field is 0, or the empty set, or None where the leaf's author does not know it.
        if all(value in (None, 0) for value in declared[:3]) and declared[3] in (
            None,
            frozenset(),
        ):
            continue
        shown = dict(zip(_IMAGE_FIELDS, declared, strict=True))
        adding = sorted(f for f, adds in built.feature_modalities.items() if "image" in adds)
        expect(
            bool(adding),
            f"{on}, it takes no images yet declares {shown}, not 0, 0, 0 and no types or unknown",
        )
        wanted = harness.capabilities_for(features | {adding[0]})
        expect(
            declared == tuple(getattr(wanted, name) for name in _IMAGE_FIELDS),
            f"{on}, its image fields are not those the {adding[0]} feature would bring",
        )
    return None


_LeafCheck = Callable[[LeafHarness], Awaitable[str | None]]

_CHECKS: Final[tuple[tuple[str, str, _LeafCheck], ...]] = (
    ("L1", "The leaf satisfies Leaf, and call returns a WireResponse.", _l1),
    ("L2", "The leaf is not a Provider until wrapped in a Dispatcher.", _l2),
    ("L3", "Its capabilities are the declared, frozen Capabilities.", _l3),
    ("L4", "The leaf's class declares no capabilities field as a class attribute.", _l4),
    ("L5", "Each claimed kind's recorded success maps to that kind's shape.", _l5),
    ("L6", "Every recorded success validates with no sum or expected-level fault.", _l6),
    (
        "L7",
        "Built with each opt-in feature on and off, the record lists what is on and equals "
        "capabilities_for; its feature maps do not change.",
        _l7,
    ),
    ("L8", "One call sends one wire request, over a success or a retryable failure.", _l8),
    ("L9", "A leaf built on an SDK sets its retries to zero.", _l9),
    (
        "L10",
        "The leaf's module imports nothing of the call loop and keeps no ProviderError from it.",
        _l10,
    ),
    ("L11", "call raises ProviderError and nothing else for every failure.", _l11),
    (
        "L12",
        "A ProviderError carries status, request id, retryability, retry_after, no body.",
        _l12,
    ),
    ("L13", "A body that cannot be read raises a non-retryable ProviderError.", _l13),
    ("L14", "call ends by its deadline when the transport stalls.", _l14),
    ("L15", "An injected client is left open; a client the leaf made is closed.", _l15),
    ("L16", "Every mapped answer passes through validate before it is returned.", _l16),
    ("L17", "A recorded success comes back whole, with no fault, as the wire gave it.", _l17),
    ("L18", "One answer breaking a rule is a Fault and its siblings stand.", _l18),
    ("L19", "A missing key is a Fault; a key with no question is dropped with a warning.", _l19),
    ("L20", "A noul outside [0, 1] is a Fault.", _l20),
    ("L21", "The leaf's module computes no 1 - p and uses no noul_complement.", _l21),
    ("L22", "Confidence only for declared kinds, and only the wire's own.", _l22),
    (
        "L23",
        "A refusal is a Refusal where returns_refusal holds; otherwise no success holds one.",
        _l23,
    ),
    ("L24", "A candidate shape is a Fault the leaf builds, naming a candidate kind.", _l24),
    ("L25", "An extraction keeps the wire's guess and never moves it into value.", _l25),
    ("L26", "The wire response fills the provenance only the leaf knows.", _l26),
    ("L27", "calibrated on the wire response equals the declared calibrated.", _l27),
    ("L28", "A snapshot only where pinnable, otherwise a no-snapshot warning.", _l28),
    ("L29", "An unknown option yields the unknown-option warning and nothing else.", _l29),
    ("L30", "A call after aclose raises a non-retryable ProviderError and sends nothing.", _l30),
    (
        "L31",
        "The image fields agree with the record: one carrier and caps not 0 where it takes "
        "images; 0, 0, 0 and no types, or unknown, or its image feature's values, where not.",
        _l31,
    ),
)


def leaf_checks(make_harness: Callable[[], LeafHarness]) -> tuple[Check, ...]:
    """The leaf half, L1 to L31, each run against a fresh harness from ``make_harness``."""

    def bind(check: _LeafCheck) -> Callable[[], Coroutine[Any, Any, str | None]]:
        async def run() -> str | None:
            return await check(make_harness())

        return run

    return tuple(Check(id, description, bind(check)) for id, description, check in _CHECKS)
