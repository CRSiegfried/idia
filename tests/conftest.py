"""Shared fixtures and the gates on real calls: a wire API, and a model run in process."""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """With ``IDIA_LOCAL_MODEL=1``, set the Hugging Face hub offline before any test module
    is imported, so a run against a model in process reads only cached weights."""
    if os.environ.get("IDIA_LOCAL_MODEL") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test marked ``live`` unless ``IDIA_LIVE=1`` is set, and one marked
    ``local_model`` unless ``IDIA_LOCAL_MODEL=1`` is set."""
    if item.get_closest_marker("live") is not None and os.environ.get("IDIA_LIVE") != "1":
        pytest.skip("live call; set IDIA_LIVE=1 to run")
    if (
        item.get_closest_marker("local_model") is not None
        and os.environ.get("IDIA_LOCAL_MODEL") != "1"
    ):
        pytest.skip("loads a model in process; set IDIA_LOCAL_MODEL=1 to run, its weights cached")
