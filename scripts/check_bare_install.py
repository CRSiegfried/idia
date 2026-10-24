"""Check an idia installed with no extras: it imports, and each extra's hint is right.

CI runs it after ``uv build``, with the interpreter of a fresh venv that holds the built wheel
and nothing else: ``<venv>/bin/python scripts/check_bare_install.py``. It exits 0 when the
install passes and 1 with one line per problem otherwise. Standard library and idia only.

- No extra's dependency is installed, and importing ``idia``, its test doubles and its
  conformance suite imports none of them.
- Every registered provider with an extra raises ``ModuleNotFoundError`` with the
  ``pip install "idia[<extra>]"`` hint.

DEPENDENCIES is built from the registry's extras.

The blocked-import tests prove the guard's logic in the dev environment, which has the
extras installed, though not always every one: CI's test jobs leave out the extras run
in a job of their own. This proves the package as a user installs it.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import idia
from idia.providers import _EXTRAS, _REGISTRY  # pyright: ignore[reportPrivateUsage]

DEPENDENCIES = tuple(sorted({module for modules in _EXTRAS.values() for module in modules}))
"""The top-level modules the extras install."""

CORE_MODULES = (
    "idia.providers.scripted",
    "idia.providers.recorded",
    "idia.conformance",
)
"""Modules that ship in the core with no extra, beside ``idia`` itself."""


def _hint(extra: str) -> str:
    return f'pip install "idia[{extra}]"'


def _expect_hint(what: str, extra: str, load: Callable[[], object]) -> str | None:
    """The problem with calling ``load``, or ``None`` when it raises the extra's hint."""
    try:
        load()
    except ModuleNotFoundError as error:
        if _hint(extra) not in str(error):
            return f"{what}: no {_hint(extra)!r} in {error}"
        return None
    return f"{what}: built with the {extra!r} extra not installed"


def main() -> int:
    problems: list[str] = []
    source = Path(idia.__file__).resolve()
    if "site-packages" not in source.parts:
        problems.append(f"idia is imported from {source}, not an installed wheel")
    for module in CORE_MODULES:
        importlib.import_module(module)
    for name in DEPENDENCIES:
        if importlib.util.find_spec(name) is not None:
            problems.append(f"{name} is installed; the check needs a venv with no extras")
        if name in sys.modules:
            problems.append(f"importing idia or a core module imported {name}")
    if problems:
        print("\n".join(problems))
        return 1
    for name, entry in _REGISTRY.items():
        if entry.extra is None:
            continue
        if entry.extra not in _EXTRAS:
            problems.append(f"{name}: its extra {entry.extra!r} has no _EXTRAS entry")
            continue
        problem = _expect_hint(
            f"provider({name!r})", entry.extra, lambda name=name: idia.provider(name)
        )
        if problem is not None:
            problems.append(problem)
    for problem in problems:
        print(problem)
    if not problems:
        checked = sum(entry.extra is not None for entry in _REGISTRY.values())
        print(f"check_bare_install: idia imports bare; {checked} extras give the hint")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
