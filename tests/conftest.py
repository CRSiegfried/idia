"""Shared fixtures and the live-call gate."""

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test marked ``live`` unless ``IDIA_LIVE=1`` is set."""
    if item.get_closest_marker("live") is not None and os.environ.get("IDIA_LIVE") != "1":
        pytest.skip("live call; set IDIA_LIVE=1 to run")
