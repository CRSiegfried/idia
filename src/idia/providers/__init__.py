"""The leaves idia ships, and the registry that finds one by name.

Each wire leaf is an optional extra. The registry maps a provider name to the module and
class of its leaf, and :func:`_load_leaf` imports that module behind a guard: when the
import fails because the extra's dependency is missing, the error is re-raised with the
``pip install`` hint. Nothing under this package imports the call loop.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from idia.core import IdiaError, Leaf


class ProviderNotFound(IdiaError, LookupError):
    """No leaf is registered under this name; ``registered`` holds the names that are."""

    name: str
    registered: tuple[str, ...]

    def __init__(self, name: str, registered: tuple[str, ...] = ()) -> None:
        message = f"no provider is registered as {name!r}"
        if registered:
            message += f"; registered: {', '.join(registered)}"
        super().__init__(message)
        self.name = name
        self.registered = registered


@dataclass(frozen=True, slots=True, kw_only=True)
class _Entry:
    """One registry entry: ``module:Class`` naming the leaf, and the extra that installs it."""

    target: str
    extra: str | None = None


_REGISTRY: Final[dict[str, _Entry]] = {
    "system-one": _Entry(target="idia.providers.system_one:SystemOne", extra="system-one"),
    "openai-decisions": _Entry(
        target="idia.providers.openai_decisions:OpenAIDecisions", extra="openai"
    ),
    "openai-chat": _Entry(target="idia.providers.chat.openai:OpenAIChat", extra="openai"),
    "cloudflare-clef": _Entry(target="idia.providers.cloudflare:Clef", extra="cloudflare"),
    "cloudflare-clairic": _Entry(target="idia.providers.cloudflare:Clairic", extra="cloudflare"),
}
"""Provider name to its leaf. Each wire leaf registers here."""

_EXTRAS: Final[dict[str, str]] = {
    "system-one": "httpx2",
    "openai": "openai",
    "cloudflare": "httpx2",
}
"""Extra name to the top-level module its dependency installs, for the import guard."""


def _load_leaf(name: str, **options: object) -> Leaf:  # pyright: ignore[reportUnusedFunction]
    """Import the leaf registered under ``name`` and build it from ``options``.

    ``idia.provider`` calls this and wraps what it returns.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise ProviderNotFound(name, tuple(sorted(_REGISTRY)))
    module_name, _, class_name = entry.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        dependency = _EXTRAS.get(entry.extra) if entry.extra is not None else None
        missing = error.name or ""
        if dependency is not None and (
            missing == dependency or missing.startswith(dependency + ".")
        ):
            raise ModuleNotFoundError(
                f"provider {name!r} needs the {entry.extra!r} extra: "
                f'pip install "idia[{entry.extra}]"',
                name=error.name,
            ) from error
        raise
    leaf_class = getattr(module, class_name)
    try:
        leaf = leaf_class(**options)
    except TypeError as error:
        if _binds(leaf_class, options):
            raise
        raise TypeError(f"provider({name!r}): {error}{_hand_built(leaf_class, options)}") from error
    if not isinstance(leaf, Leaf):
        raise TypeError(f"{entry.target} does not satisfy Leaf")
    return leaf


def _binds(leaf_class: Callable[..., object], options: dict[str, object]) -> bool:
    """Whether ``options`` bind to the leaf class's constructor: a ``TypeError`` raised
    when they do is the leaf's own, about a value, and passes through unchanged."""
    try:
        inspect.signature(leaf_class).bind(**options)
    except TypeError:
        return False
    return True


_DISPATCHER_ONLY: Final = ("estimator", "overflow", "reask")
"""The ``Dispatcher`` keywords ``idia.provider`` does not pass on: a leaf gets them."""


def _hand_built(leaf_class: Callable[..., object], options: dict[str, object]) -> str:
    """For a ``Dispatcher`` keyword the leaf does not take, a pointer to building the
    ``Dispatcher`` by hand; else nothing."""
    taken = inspect.signature(leaf_class).parameters
    named = [keyword for keyword in _DISPATCHER_ONLY if keyword in options and keyword not in taken]
    if not named:
        return ""
    return (
        f"; {', '.join(named)} {'is' if len(named) == 1 else 'are'} the Dispatcher's: "
        "build the leaf and wrap it in Dispatcher(leaf, ...) by hand"
    )
