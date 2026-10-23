"""Run the conformance suite under pytest.

A leaf's or a composite's test module invokes the suite with one parametrized test per
half, each check its own test case named by its ID::

    @pytest.mark.parametrize("check", params(leaf_checks(MyLeafHarness)))
    def test_leaf_half(check: Check) -> None:
        run(check)

A check that is not applicable is skipped with its reason.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from idia.conformance import Check, CheckNotApplicable


def params(checks: Iterable[Check]) -> list[object]:
    """One ``pytest.param`` per check, its ID the check's."""
    return [pytest.param(check, id=check.id) for check in checks]


def run(check: Check) -> str | None:
    """Run one check; a check that is not applicable is skipped with its reason."""
    try:
        return asyncio.run(check.run())
    except CheckNotApplicable as not_applicable:
        pytest.skip(not_applicable.reason)
