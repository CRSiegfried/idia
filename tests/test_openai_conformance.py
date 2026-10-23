"""The conformance suite against the two OpenAI leaves: the leaf half per leaf, over its
cassettes and a mock transport under an ``AsyncOpenAI`` client, and the provider half once
wrapped."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable

import pytest

from idia.conformance import (
    Check,
    LeafHarness,
    ProviderHarness,
    leaf_checks,
    provider_checks,
    run_checks,
)
from tests._conformance import params, run
from tests._openai_harness import ChatHarness, DecisionsHarness, WrappedOpenAI

_DECISIONS_NOT_APPLICABLE = {"L24", "L25"}
"""The Decisions API returns no candidate shape and has no extraction kind."""

_CHAT_NOT_APPLICABLE = {"L18", "L19", "L20", "L24", "L25"}
"""Besides those two: the chat read renormalizes over the labels and its wire carries no
keys, so no perturbation can move a sum, a key or a noul's range."""

_PROVIDER_NOT_APPLICABLE = {"P6", "P14", "P15", "P16", "P17", "P18"}
"""Neither leaf has an opt-in feature, and a wrapped leaf is no composite."""


def _not_applicable(checks: Iterable[Check]) -> set[str]:
    verdicts = asyncio.run(run_checks(checks))
    failed = {v.id: v.reason for v in verdicts if v.result == "fail"}
    assert not failed, failed
    return {v.id for v in verdicts if v.result == "not_applicable"}


@pytest.mark.parametrize("check", params(leaf_checks(DecisionsHarness)))
def test_leaf_half_decisions(check: Check) -> None:
    run(check)


@pytest.mark.parametrize("check", params(leaf_checks(ChatHarness)))
def test_leaf_half_chat(check: Check) -> None:
    run(check)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        pytest.param(DecisionsHarness, _DECISIONS_NOT_APPLICABLE, id="decisions"),
        pytest.param(ChatHarness, _CHAT_NOT_APPLICABLE, id="chat"),
    ],
)
def test_the_leaf_halfs_not_applicable_set(
    make: Callable[[], LeafHarness], expected: set[str]
) -> None:
    assert _not_applicable(leaf_checks(make)) == expected


@pytest.mark.parametrize(
    ("leaf", "extra"),
    [
        pytest.param(DecisionsHarness, set[str](), id="decisions"),
        pytest.param(ChatHarness, {"P23"}, id="chat"),
    ],
)
def test_provider_half(leaf: type[DecisionsHarness] | type[ChatHarness], extra: set[str]) -> None:
    """The chat leaf takes no images, so P23 has nothing to send."""

    def make() -> ProviderHarness:
        return WrappedOpenAI(leaf)

    assert _not_applicable(provider_checks(make)) == _PROVIDER_NOT_APPLICABLE | extra
