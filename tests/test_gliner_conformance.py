"""The conformance suite against the GLiNER2 leaf, over its stored outputs: the leaf half,
and the provider half over the wrapped leaf, with exactly the checks a model in process
cannot reach left out."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from idia.conformance import Check, leaf_checks, provider_checks, run_checks
from tests._conformance import params, run
from tests._gliner_harness import GlinerHarness, GlinerProviderHarness

_LEAF_NOT_APPLICABLE = {"L9", "L12", "L15", "L18", "L19", "L20", "L24", "L32"}
"""No SDK (L9), no statuses (L12), no HTTP client (L15), no kind whose probabilities sum
(L18), no noul (L20) and no candidate kind (L24); and no key without a question (L19): the
runtime answers by label and the leaf maps by question key, so no key without a question can
reach validation."""
_WRAPPED_NOT_APPLICABLE = {
    "P6",
    "P9",
    "P14",
    "P15",
    "P16",
    "P17",
    "P18",
    "P20",
    "P21",
    "P23",
}
"""No opt-in feature (P6), no statuses (P9, P20, P21), no composite, router or fallback (P14
to P18), and no image (P23)."""


def _verdicts(checks: Iterable[Check]) -> tuple[set[str], dict[str, str]]:
    """The checks not applicable, and the notes on the checks that passed."""
    verdicts = asyncio.run(run_checks(checks))
    failed = {v.id: v.reason for v in verdicts if v.result == "fail"}
    assert not failed, failed
    skipped = {v.id for v in verdicts if v.result == "not_applicable"}
    notes = {v.id: v.reason for v in verdicts if v.result == "pass" and v.reason}
    return skipped, notes


@pytest.mark.parametrize("check", params(leaf_checks(GlinerHarness)))
def test_leaf_half(check: Check) -> None:
    run(check)


@pytest.mark.parametrize("check", params(provider_checks(GlinerProviderHarness)))
def test_provider_half(check: Check) -> None:
    run(check)


def test_the_leaf_half_leaves_out_exactly_what_a_model_in_process_cannot_make() -> None:
    skipped, notes = _verdicts(leaf_checks(GlinerHarness))
    assert skipped == _LEAF_NOT_APPLICABLE
    read = (
        "read idia.providers._gliner, the module that defines Gliner; code the leaf keeps in "
        "other modules was not read"
    )
    assert notes == {
        "L4": read,
        "L8": "the retryable half ran over a stall, not a 429: the model has no statuses",
        "L10": read,
        "L11": "not driven: Failure(status=429, retry_after=None, request_id=None): the model "
        "has no statuses; Error(which='connection'): a model in process cannot make "
        "Error(which='connection'); Error(which='timeout'): a model in process cannot make "
        "Error(which='timeout')",
        "L21": read,
        "L25": "no expected extraction has a null value and a guess, so only that the leaf "
        "fills no guess was checked, not that a guess is kept and never moved into value",
    }


def test_the_provider_half_leaves_out_exactly_what_a_wrapped_leaf_cannot_reach() -> None:
    skipped, notes = _verdicts(provider_checks(GlinerProviderHarness))
    assert skipped == _WRAPPED_NOT_APPLICABLE
    assert notes == {
        "P12": "the fault half was not run: the request has no question whose sum can move",
        "P19": "a 503 then an answer was not recorded: the model has no statuses",
    }
