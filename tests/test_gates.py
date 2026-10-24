"""The gates on real calls: a test marked ``live`` runs only with ``IDIA_LIVE=1``, one marked
``local_model`` only with ``IDIA_LOCAL_MODEL=1``, which also sets the hub offline."""

from __future__ import annotations

import os

import pytest

from tests import conftest


class Item:
    """The one part of a pytest item the gate reads: its markers."""

    def __init__(self, *markers: str) -> None:
        self._markers = markers

    def get_closest_marker(self, name: str) -> object | None:
        return name if name in self._markers else None


def _setup(item: Item) -> None:
    conftest.pytest_runtest_setup(item)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("marker", "switch"), [("live", "IDIA_LIVE"), ("local_model", "IDIA_LOCAL_MODEL")]
)
def test_a_marked_test_is_skipped_unless_its_switch_is_on(
    marker: str, switch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(switch, raising=False)
    with pytest.raises(pytest.skip.Exception, match=switch):
        _setup(Item(marker))
    monkeypatch.setenv(switch, "1")
    _setup(Item(marker))


def test_one_switch_does_not_open_the_other_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIA_LIVE", "1")
    monkeypatch.delenv("IDIA_LOCAL_MODEL", raising=False)
    with pytest.raises(pytest.skip.Exception, match="IDIA_LOCAL_MODEL"):
        _setup(Item("local_model"))


def test_an_unmarked_test_runs_with_neither_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDIA_LIVE", raising=False)
    monkeypatch.delenv("IDIA_LOCAL_MODEL", raising=False)
    _setup(Item())


def test_the_local_model_switch_sets_the_hub_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIA_LOCAL_MODEL", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    conftest.pytest_configure(None)  # pyright: ignore[reportArgumentType]
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_without_the_switch_the_hub_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDIA_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    conftest.pytest_configure(None)  # pyright: ignore[reportArgumentType]
    assert "HF_HUB_OFFLINE" not in os.environ
