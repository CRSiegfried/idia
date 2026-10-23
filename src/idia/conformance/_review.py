"""The source checks of the leaf half: walks over a module's source.

Each finder takes a parsed module, and its dotted name where it resolves imports or an
allowlist, and returns what it found, one string per finding, so a check can name every
offending line.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import sys
from collections.abc import Collection
from typing import Final

from idia.conformance._base import CheckNotApplicable
from idia.core import Capabilities, ProviderError, noul_complement

_LOOP_MODULES: Final = frozenset({"idia.retry", "idia.limits", "idia.dispatch", "idia._provider"})
"""The call loop's modules, which no leaf imports."""

_LOOP_NAMES: Final = frozenset(
    {"Dispatcher", "Router", "Fallback", "RetryPolicy", "ConcurrencyLimit", "RateLimiter"}
    | {"provider", "retry", "limits", "dispatch", "_provider"}
)
"""What the root package re-exports from the call loop, and the loop's module names."""


_BROAD: Final = frozenset({"bare except", "BaseException", "Exception"})

_RECORD_FIELDS: Final = frozenset(
    {field.name for field in dataclasses.fields(Capabilities)} | {"capabilities"}
)

_COMPLEMENT_ALLOWED: Final[frozenset[str]] = frozenset(
    # The chat leaf's off-label mass, one less the summed label masses: its read protocol
    # records the mass outside the labels, which is no answer's probability.
    {"idia.providers.chat.openai:_off_label_mass"}
)
"""``module:function`` names allowed to compute ``1 - x``. The chat leaf's noul is A's
share of A and B and needs none."""


def parse(module_name: str) -> ast.Module:
    """The module's source, parsed; not applicable when it has none to read."""
    module = sys.modules.get(module_name)
    try:
        source = inspect.getsource(module) if module is not None else None
    except (OSError, TypeError):
        source = None
    if source is None:
        raise CheckNotApplicable(f"the source of {module_name} cannot be read")
    return ast.parse(source)


def _resolve(node: ast.ImportFrom, module_name: str) -> str:
    """The absolute module an ``import from`` names, relative imports resolved against the
    module's package: itself for a package's ``__init__``, its parent otherwise."""
    if node.level == 0:
        return node.module or ""
    module = sys.modules.get(module_name)
    package = getattr(module, "__package__", None)
    if package is None:
        package = module_name.rpartition(".")[0]
    try:
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    except ImportError:
        return ""


def loop_imports(tree: ast.Module, module_name: str) -> list[str]:
    """Each import of the call loop: its modules, or its names from the root package."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name in _LOOP_MODULES
            )
        elif isinstance(node, ast.ImportFrom):
            source = _resolve(node, module_name)
            if source in _LOOP_MODULES:
                found.append(f"line {node.lineno}: from {source} import ...")
            elif source == "idia":
                found.extend(
                    f"line {node.lineno}: from idia import {alias.name}"
                    for alias in node.names
                    if alias.name in _LOOP_NAMES
                )
    return found


def _names(node: ast.expr | None) -> list[str]:
    if isinstance(node, ast.Tuple):
        return [name for element in node.elts for name in _names(element)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def _subclasses(base: type) -> set[type]:
    found = {base}
    for subclass in base.__subclasses__():
        found |= _subclasses(subclass)
    return found


def _reraises(handler: ast.ExceptHandler) -> bool:
    return bool(handler.body) and isinstance(handler.body[-1], ast.Raise)


def _passes_through(handler: ast.ExceptHandler) -> bool:
    """The handler's body ends in a bare ``raise`` and returns nowhere: the caught error
    always goes on unchanged."""
    return bool(handler.body) and _is_bare_raise(handler.body[-1]) and not _returns(handler)


def _returns(handler: ast.ExceptHandler) -> bool:
    """A ``return`` anywhere in the handler's body, functions defined inside it aside."""

    def visit(node: ast.AST) -> bool:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            return False
        if isinstance(node, ast.Return):
            return True
        return any(visit(child) for child in ast.iter_child_nodes(node))

    return any(visit(statement) for statement in handler.body)


def _is_bare_raise(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Raise) and statement.exc is None


def _handlers(tree: ast.Module) -> list[tuple[ast.ExceptHandler, bool]]:
    """Each ``except`` clause, with whether a loop in its own function encloses it."""
    found: list[tuple[ast.ExceptHandler, bool]] = []

    def visit(node: ast.AST, looped: bool) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            looped = False
        if isinstance(node, ast.ExceptHandler):
            found.append((node, looped))
        inner = looped or isinstance(node, ast.For | ast.AsyncFor | ast.While)
        for child in ast.iter_child_nodes(node):
            visit(child, inner)

    visit(tree, False)
    return found


def provider_error_catches(tree: ast.Module, module_name: str) -> list[str]:
    """Each ``except`` clause that catches a ``ProviderError`` and could keep it from the
    loop: a name that is ``ProviderError`` or one of its subclasses, by its name in the
    module or as loaded, unless the clause ends in a bare ``raise``, returns nowhere in
    its body and no loop in its function encloses it, as a recorder that stores the error
    and lets it go on does; and
    a bare ``except``, ``except BaseException`` or ``except Exception`` whose body does not
    end by raising, as a leaf turning a transport's error into a ``ProviderError`` does."""
    caught = {cls.__name__ for cls in _subclasses(ProviderError)}
    namespace: dict[str, object] = getattr(sys.modules.get(module_name), "__dict__", {})
    found: list[str] = []
    for node, looped in _handlers(tree):
        names = _names(node.type) if node.type is not None else ["bare except"]
        passes_on = _passes_through(node) and not looped
        for name in names:
            bound = namespace.get(name)
            named = name in caught or (isinstance(bound, type) and issubclass(bound, ProviderError))
            broad = name in _BROAD and not _reraises(node)
            if (named and not passes_on) or broad:
                found.append(f"line {node.lineno}: except {name}")
    return found


def class_limits(tree: ast.Module, classes: Collection[str] | None = None) -> list[str]:
    """Each class attribute given a value under a capabilities field's name, in any case,
    or under ``capabilities`` itself: in the classes named, or in every class when none
    are."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if classes is not None and node.name not in classes:
            continue
        for statement in node.body:
            targets: list[ast.expr] = []
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets = [statement.target]
            found.extend(
                f"line {statement.lineno}: {node.name}.{target.id}"
                for target in targets
                if isinstance(target, ast.Name) and target.id.lower() in _RECORD_FIELDS
            )
    return found


def _is_one(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, int | float)
        and node.value == 1
    )


def complements(tree: ast.Module, module_name: str) -> list[str]:
    """Each ``1 - x`` outside the allowed functions."""
    found: list[str] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = (*scope, node.name)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and _is_one(node.left)
            and f"{module_name}:{'.'.join(scope)}" not in _COMPLEMENT_ALLOWED
        ):
            found.append(f"line {node.lineno}: {ast.unparse(node)}")
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return found


def complement_calls(tree: ast.Module, module_name: str) -> list[str]:
    """Each reference to ``noul_complement``, and each import of it from idia, under
    whatever name: a leaf hands over the probability of yes alone, so it has no use for
    the complement. A value in the module bound to the function, under any name, too."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _resolve(node, module_name) in (
            "idia",
            "idia.core",
        ):
            found += [
                f"line {node.lineno}: imports noul_complement"
                for alias in node.names
                if alias.name == "noul_complement"
            ]
        elif (isinstance(node, ast.Name) and node.id == "noul_complement") or (
            isinstance(node, ast.Attribute) and node.attr == "noul_complement"
        ):
            found.append(f"line {node.lineno}: {ast.unparse(node)}")
    namespace: dict[str, object] = getattr(sys.modules.get(module_name), "__dict__", {})
    found += [
        f"binds noul_complement as {name}"
        for name, value in namespace.items()
        if value is noul_complement and name != "noul_complement"
    ]
    return found
