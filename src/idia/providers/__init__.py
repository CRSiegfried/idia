"""The leaves idia ships, and the registry that finds one by name.

Each leaf is an optional extra. The registry maps a provider name to the module and
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
    "gliner": _Entry(target="idia.providers.gliner:Gliner", extra="gliner"),
}
"""Provider name to its leaf. Each leaf idia ships registers here."""

_EXTRAS: Final[dict[str, tuple[str, ...]]] = {
    "system-one": ("httpx2",),
    "openai": ("openai", "httpx2"),
    "cloudflare": ("httpx2",),
    "gliner": ("gliner2", "torch"),
}
"""Extra name to the top-level modules its dependencies install, for the import guard."""


def _registered() -> tuple[str, ...]:  # pyright: ignore[reportUnusedFunction]
    """The registered names, sorted: what ``idia.provider_names`` returns and
    ``ProviderNotFound`` lists."""
    return tuple(sorted(_REGISTRY))


def _load_leaf(  # pyright: ignore[reportUnusedFunction]
    name: str, caller: str = "provider", /, **options: object
) -> Leaf:
    """Import the leaf registered under ``name`` and build it from ``options``.

    ``idia.leaf`` returns what this builds and ``idia.provider`` wraps it; ``caller`` names
    which of the two asked, for the message of a ``TypeError``.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise ProviderNotFound(name, _registered())
    module_name, _, class_name = entry.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        dependencies = _EXTRAS.get(entry.extra, ()) if entry.extra is not None else ()
        missing = error.name or ""
        if any(
            missing == dependency or missing.startswith(dependency + ".")
            for dependency in dependencies
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
        hint = _hand_built(leaf_class, options, caller)
        raise TypeError(f"{caller}({name!r}): {error}{hint}") from error
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

_DISPATCHER_KEYWORDS: Final = ("retry", "concurrency", "rate", *_DISPATCHER_ONLY)
"""Every ``Dispatcher`` keyword: ``idia.leaf`` passes none of them on."""


def _hand_built(leaf_class: Callable[..., object], options: dict[str, object], caller: str) -> str:
    """For a ``Dispatcher`` keyword the leaf does not take, a pointer to the ``Dispatcher``
    built by hand; else nothing."""
    taken = inspect.signature(leaf_class).parameters
    keywords = _DISPATCHER_KEYWORDS if caller == "leaf" else _DISPATCHER_ONLY
    named = [keyword for keyword in keywords if keyword in options and keyword not in taken]
    if not named:
        return ""
    where = (
        "wrap the leaf in Dispatcher(leaf, ...)"
        if caller == "leaf"
        else "build the leaf and wrap it in Dispatcher(leaf, ...) by hand"
    )
    return f"; {', '.join(named)} {'is' if len(named) == 1 else 'are'} the Dispatcher's: {where}"
