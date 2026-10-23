"""A leaf module whose dependency is not installed, for the import-guard test."""

import idia_test_missing_sdk  # type: ignore[import-not-found]  # noqa: F401  # pyright: ignore[reportMissingImports]
