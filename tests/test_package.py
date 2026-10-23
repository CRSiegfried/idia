"""The package exposes its version and its public API: what `idia.__all__` lists and what
the `__all__` of each listed module lists."""

import importlib
import importlib.metadata

import pytest

import idia

LISTED: tuple[tuple[str, str | None], ...] = (
    ("idia.cloudflare", "httpx2"),
    ("idia.conformance", None),
    ("idia.providers.http", "httpx2"),
    ("idia.providers.system_one", "httpx2"),
    ("idia.providers.openai_decisions", "openai"),
    ("idia.providers.chat", "openai"),
    ("idia.providers.cloudflare", "httpx2"),
    ("idia.providers.scripted", None),
    ("idia.providers.recorded", None),
)
"""Each module whose `__all__` is public, with the dependency its extra installs."""


def test_version_matches_the_distribution() -> None:
    assert idia.__version__ == importlib.metadata.version("idia")


def test_public_api_is_what_all_lists() -> None:
    assert idia.__all__
    for name in idia.__all__:
        assert not name.startswith("_")
        assert hasattr(idia, name)


@pytest.mark.parametrize(("name", "dependency"), LISTED, ids=[name for name, _ in LISTED])
def test_each_listed_module_publishes_its_all(name: str, dependency: str | None) -> None:
    if dependency is not None:
        pytest.importorskip(dependency)
    module = importlib.import_module(name)
    public: list[str] = list(getattr(module, "__all__", []))
    assert public, f"{name} defines no __all__"
    assert len(public) == len(set(public)), f"{name}.__all__ repeats a name"
    for item in public:
        assert not item.startswith("_"), f"{name}.__all__ lists {item}"
        assert hasattr(module, item), f"{name}.__all__ lists {item}, which it does not hold"
