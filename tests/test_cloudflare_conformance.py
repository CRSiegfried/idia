"""The conformance suite against the Cloudflare leaves, over their cassettes: the leaf half
per model, the provider half over each wrapped leaf and over the ``cloudflare`` router on
one shared connection; and what the record and the estimators were set from."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from idia import (
    Deadline,
    DoesNotFit,
    ExtractionAnswer,
    ImagePart,
    Request,
    Span,
    TextPart,
)
from idia.conformance import (
    Check,
    LeafHarness,
    ProviderHarness,
    leaf_checks,
    provider_checks,
    run_checks,
)
from idia.dispatch import Dispatcher
from idia.providers.cloudflare import Clairic, Clef
from tests._cloudflare_cases import (
    ACCOUNT,
    CLAIRIC_DIR,
    CLAIRIC_FAILURE,
    CLEF_FAILURE,
    TOKEN,
    clef_dir,
)
from tests._cloudflare_harness import (
    ClairicHarness,
    ClefFlashHarness,
    ClefHarness,
    CloudflareRouter,
    wrapped_clairic,
    wrapped_clef,
)
from tests._conformance import params, run
from tests._http import Cassette, load_cassette

_CLEF_NOT_APPLICABLE = {"L24", "L25"}
"""Clef returns no candidate shape and has no extraction kind."""
_CLAIRIC_NOT_APPLICABLE = {"L18", "L20"}
"""Clairic claims no kind whose probabilities sum and no noul."""
_WRAPPED_NOT_APPLICABLE = {"P6", "P14", "P15", "P16", "P17", "P18"}
"""A wrapped leaf with no opt-in feature is no composite."""
_ROUTER_NOT_APPLICABLE = {"P6", "P18", "P19"}
"""No opt-in feature; a router is no fallback and has no one leaf to record."""


def _not_applicable(checks: Iterable[Check]) -> set[str]:
    verdicts = asyncio.run(run_checks(checks))
    failed = {v.id: v.reason for v in verdicts if v.result == "fail"}
    assert not failed, failed
    return {v.id for v in verdicts if v.result == "not_applicable"}


@pytest.mark.parametrize("check", params(leaf_checks(ClefHarness)))
def test_leaf_half_clef(check: Check) -> None:
    run(check)


@pytest.mark.parametrize("check", params(leaf_checks(ClefFlashHarness)))
def test_leaf_half_clef_flash(check: Check) -> None:
    run(check)


@pytest.mark.parametrize("check", params(leaf_checks(ClairicHarness)))
def test_leaf_half_clairic(check: Check) -> None:
    run(check)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (ClefHarness, _CLEF_NOT_APPLICABLE),
        (ClefFlashHarness, _CLEF_NOT_APPLICABLE),
        (ClairicHarness, _CLAIRIC_NOT_APPLICABLE),
    ],
    ids=["clef", "clef-flash", "clairic"],
)
def test_the_leaf_halfs_not_applicable_set(
    make: Callable[[], LeafHarness], expected: set[str]
) -> None:
    assert _not_applicable(leaf_checks(make)) == expected


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (wrapped_clef, _WRAPPED_NOT_APPLICABLE),
        (wrapped_clairic, _WRAPPED_NOT_APPLICABLE),
        (CloudflareRouter, _ROUTER_NOT_APPLICABLE),
        (lambda: CloudflareRouter(ClefFlashHarness), _ROUTER_NOT_APPLICABLE),
    ],
    ids=["clef", "clairic", "router", "router-clef-flash"],
)
def test_provider_half(make: Callable[[], ProviderHarness], expected: set[str]) -> None:
    assert _not_applicable(provider_checks(make)) == expected


# ----------------------------------------------------------------------------------------
# What the record and the estimators were set from
# ----------------------------------------------------------------------------------------


def _successes(directory: Path) -> list[Cassette]:
    found = [load_cassette(path) for path in sorted(directory.glob("*.json"))]
    return [cassette for cassette in found if cassette.expected is not None]


def _usage(cassette: Cassette) -> dict[str, int]:
    return cassette.response["body"]["result"]["usage"]


def _clef_successes() -> list[Cassette]:
    directories = (clef_dir("clef"), clef_dir("clef-flash"))
    return [cassette for directory in directories for cassette in _successes(directory)]


def test_the_clef_estimate_covers_the_recorded_usage() -> None:
    estimator = Clef(account_id=ACCOUNT, api_token=TOKEN).estimator
    for cassette in _clef_successes():
        request = cassette.wire_request
        estimate = estimator.estimate(request.state, request.questions)
        covered = math.ceil(estimate.tokens * (1 + estimate.margin))
        assert covered >= _usage(cassette)["input_tokens"], request.questions[0].key


def test_the_clairic_estimate_covers_one_pass_of_the_recorded_usage() -> None:
    estimator = Clairic(account_id=ACCOUNT, api_token=TOKEN).estimator
    for cassette in _successes(CLAIRIC_DIR):
        request = cassette.wire_request
        estimate = estimator.estimate(request.state, request.questions)
        usage = _usage(cassette)
        covered = math.ceil(estimate.tokens * (1 + estimate.margin))
        assert covered >= usage["input_tokens"] / usage["passes"], request.questions[0].key


@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_clef_counts_the_state_once_per_request(model: str) -> None:
    """Three questions over a long state cost little more than one: not the state again."""
    one, three = (
        _usage(load_cassette(clef_dir(model) / f"{name}.json"))["input_tokens"]
        for name in ("recount-one", "recount-three")
    )
    assert three - one < one / 2


def test_every_recorded_text_span_slices_the_state_to_its_value() -> None:
    """Clairic's offsets are code points: the emoji cassette's target follows characters
    that are two UTF-16 units each, and still slices exactly."""
    sliced = 0
    for cassette in _successes(CLAIRIC_DIR):
        assert cassette.expected is not None
        kinds = {q.key: q.value_type for q in cassette.wire_request.questions}
        for key, answer in cassette.expected.items():
            if not isinstance(answer, ExtractionAnswer) or kinds[key] != "text":
                continue
            text = answer.value if answer.value is not None else answer.guess
            texts = text if isinstance(text, tuple) else (text,)
            spans = answer.span if isinstance(answer.span, tuple) else (answer.span,)
            assert len(spans) == len(texts)
            for value, span in zip(texts, spans, strict=True):
                assert isinstance(span, Span)
                part = cassette.wire_request.state.parts[span.part]
                assert isinstance(part, TextPart)
                assert part.text[span.start : span.end] == value
                sliced += 1
    assert sliced >= 5


def test_each_recorded_success_plans_one_wire_request_warned_only_of_min_side() -> None:
    """One wire request each, and no warning but the one a state with an image carries:
    ``min_image_side`` is unknown on both leaves."""
    for leaf, cassettes in (
        (Clef(account_id=ACCOUNT, api_token=TOKEN), _successes(clef_dir("clef"))),
        (
            Clef(model="clef-flash", account_id=ACCOUNT, api_token=TOKEN),
            _successes(clef_dir("clef-flash")),
        ),
        (Clairic(account_id=ACCOUNT, api_token=TOKEN), _successes(CLAIRIC_DIR)),
    ):
        dispatcher = Dispatcher(leaf)
        for cassette in cassettes:
            request = Request(
                state=cassette.wire_request.state,
                questions=cassette.wire_request.questions,
                deadline=Deadline.after(30.0),
            )
            plan = dispatcher.plan(request)
            assert len(plan.wire_requests) == 1
            has_image = any(isinstance(part, ImagePart) for part in request.state.parts)
            expected = ("min_image_side",) if has_image else ()
            [warnings] = plan.outcome_warnings
            assert tuple(w.field for w in warnings) == expected, warnings
            assert not plan.response_warnings


def test_fit_never_sends_what_the_wire_refused() -> None:
    """Clef refused 65 questions with 422 and code 3030; Fit packs them in two."""
    dispatcher = Dispatcher(Clef(account_id=ACCOUNT, api_token=TOKEN))
    request = Request(
        state=CLEF_FAILURE.state, questions=CLEF_FAILURE.questions, deadline=Deadline.after(30.0)
    )
    plan = dispatcher.plan(request)
    assert [len(w.questions) for w in plan.wire_requests] == [64, 1]


def test_fit_refuses_the_state_the_wire_truncated() -> None:
    """Clairic answered a state past its context truncated, not refused (the past-context
    cassette); Fit refuses that request before any call."""
    dispatcher = Dispatcher(Clairic(account_id=ACCOUNT, api_token=TOKEN))
    request = Request(
        state=CLAIRIC_FAILURE.state,
        questions=CLAIRIC_FAILURE.questions,
        deadline=Deadline.after(30.0),
        options=CLAIRIC_FAILURE.options,
    )
    with pytest.raises(DoesNotFit) as refused:
        dispatcher.plan(request)
    assert refused.value.check == "max_input_tokens"
    assert refused.value.state_overflow
    assert refused.value.limit == 65_536
