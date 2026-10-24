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

_LEAF_NOT_APPLICABLE = {"L9", "L12", "L13", "L15", "L18", "L19", "L20", "L24"}
"""No statuses, no transport error, no unreadable body, no SDK or client, no label kind,
and no candidate shape."""
_WRAPPED_NOT_APPLICABLE = {
    "P6",
    "P9",
    "P12",
    "P14",
    "P15",
    "P16",
    "P17",
    "P18",
    "P20",
    "P21",
    "P23",
}
"""No opt-in feature, no statuses, no transport error, and no composite."""


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
    # The runtime gives no request id and no output count, so L26 passes with its note.
    assert notes["L26"] == (
        "the wire responses carry no output token count and no request id, right only "
        "for a leaf whose wire or runtime gives none"
    )
    assert notes["L8"] == (
        "the retryable half ran over a stall, not a 429: the model has no statuses"
    )


def test_the_provider_half_leaves_out_exactly_what_a_wrapped_leaf_cannot_reach() -> None:
    skipped, notes = _verdicts(provider_checks(GlinerProviderHarness))
    assert skipped == _WRAPPED_NOT_APPLICABLE
    assert notes == {
        "P8": "an outcome carries no request id, right only for a leaf whose wire or runtime "
        "gives none",
        "P19": "a 503 then an answer was not recorded: the model has no statuses",
    }
