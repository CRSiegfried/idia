"""The conformance suite against the System One leaf: the leaf half once per configuration,
over its cassettes and a mock transport, and the provider half once wrapped."""

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
from tests._system_one_harness import (
    SystemOneHarness,
    WithImagesHarness,
    WrappedSystemOne,
)

_HARNESSES: list[Callable[[], LeafHarness]] = [SystemOneHarness, WithImagesHarness]
_PROVIDERS: list[Callable[[], ProviderHarness]] = [
    WrappedSystemOne,
    lambda: WrappedSystemOne(WithImagesHarness),
]

_LEAF_NOT_APPLICABLE = {"L24", "L25"}
"""System One returns no candidate shape and has no extraction kind."""

_PROVIDER_NOT_APPLICABLE = {"P14", "P15", "P16", "P17", "P18"}
"""A wrapped leaf is no composite."""


def _not_applicable(checks: Iterable[Check]) -> set[str]:
    verdicts = asyncio.run(run_checks(checks))
    failed = {v.id: v.reason for v in verdicts if v.result == "fail"}
    assert not failed, failed
    return {v.id for v in verdicts if v.result == "not_applicable"}


@pytest.mark.parametrize("check", params(leaf_checks(SystemOneHarness)))
def test_leaf_half_text_only(check: Check) -> None:
    run(check)


@pytest.mark.parametrize("check", params(leaf_checks(WithImagesHarness)))
def test_leaf_half_with_images(check: Check) -> None:
    run(check)


@pytest.mark.parametrize(
    ("make", "extra"),
    [
        pytest.param(_PROVIDERS[0], {"P23"}, id="text-only"),
        pytest.param(_PROVIDERS[1], set[str](), id="with-images"),
    ],
)
def test_provider_half(make: Callable[[], ProviderHarness], extra: set[str]) -> None:
    """Without the images feature the leaf takes no images, so P23 has nothing to send."""
    assert _not_applicable(provider_checks(make)) == _PROVIDER_NOT_APPLICABLE | extra


@pytest.mark.parametrize("make", _HARNESSES, ids=["text-only", "with-images"])
def test_the_leaf_halfs_not_applicable_set(make: Callable[[], LeafHarness]) -> None:
    assert _not_applicable(leaf_checks(make)) == _LEAF_NOT_APPLICABLE
