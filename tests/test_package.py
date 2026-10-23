"""The package exposes its version and what `__all__` lists."""

import importlib.metadata

import idia


def test_version_matches_the_distribution() -> None:
    assert idia.__version__ == importlib.metadata.version("idia")


def test_public_api_is_what_all_lists() -> None:
    assert idia.__all__
    for name in idia.__all__:
        assert not name.startswith("_")
        assert hasattr(idia, name)
