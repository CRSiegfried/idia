"""``idia.provider``: a registered leaf wrapped in a dispatcher."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import cast, get_args

import pytest

from idia import Dispatcher, Provider, RetryPolicy, _provider, provider, providers
from idia.providers import ProviderNotFound, _Entry


def test_provider_wraps_the_registry_leaf_in_a_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(providers._REGISTRY, "fake", _Entry(target="tests._fake_leaf:FakeLeaf"))
    policy = RetryPolicy(attempts=1)
    wrapped = provider("fake", retry=policy, model="fake-3")
    assert isinstance(wrapped, Dispatcher)
    assert isinstance(wrapped, Provider)
    assert wrapped._leaf.model == "fake-3"
    assert wrapped.capabilities is wrapped._leaf.capabilities
    assert wrapped._retry is policy
    assert wrapped.reask is False


def test_provider_cannot_set_reask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(providers._REGISTRY, "fake", _Entry(target="tests._fake_leaf:FakeLeaf"))
    assert "reask" not in inspect.signature(provider).parameters
    # Any other keyword goes to the leaf, which does not take it.
    with pytest.raises(TypeError) as raised:
        provider("fake", reask=True)
    assert str(raised.value).endswith(
        "got an unexpected keyword argument 'reask'; reask is the Dispatcher's: "
        "build the leaf and wrap it in Dispatcher(leaf, ...) by hand"
    )


def test_each_dispatcher_keyword_given_to_provider_points_to_a_hand_built_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(providers._REGISTRY, "fake", _Entry(target="tests._fake_leaf:FakeLeaf"))
    with pytest.raises(TypeError, match="unexpected keyword argument") as raised:
        provider("fake", overflow=None, estimator=None, reask=True)
    assert str(raised.value).endswith(
        "; estimator, overflow, reask are the Dispatcher's: "
        "build the leaf and wrap it in Dispatcher(leaf, ...) by hand"
    )
    with pytest.raises(TypeError) as plain:
        provider("fake", modle="x")
    assert "Dispatcher" not in str(plain.value)


def test_provider_unknown_name_raises_the_registry_error() -> None:
    with pytest.raises(ProviderNotFound):
        provider("nowhere")


def test_a_keyword_the_leaf_does_not_take_names_the_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(providers._REGISTRY, "fake", _Entry(target="tests._fake_leaf:FakeLeaf"))
    with pytest.raises(TypeError) as raised:
        provider("fake", modle="fake-2")
    message = str(raised.value)
    assert message.startswith("provider('fake'): ")
    assert "got an unexpected keyword argument 'modle'" in message
    cause = raised.value.__cause__
    assert isinstance(cause, TypeError)
    assert message == f"provider('fake'): {cause}"


def test_a_type_error_the_leaf_raises_about_a_value_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        providers._REGISTRY, "checking", _Entry(target="tests._fake_leaf:CheckingLeaf")
    )
    with pytest.raises(TypeError) as raised:
        provider("checking", model=3)
    assert str(raised.value) == "the model is a str, not int"
    assert raised.value.__cause__ is None


# ----------------------------------------------------------------------------------------
# The overloads, read from the source: they exist for the type checker alone


_LIMITS = {"retry", "concurrency", "rate"}


def _overloads(function_name: str = "provider") -> list[ast.FunctionDef]:
    tree = ast.parse(Path(_provider.__file__).read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == function_name
        and any(ast.unparse(d) == "overload" for d in node.decorator_list)
    ]


def _literal_names(function: ast.FunctionDef) -> set[str] | None:
    """The names a ``Literal`` annotation on ``name`` admits, or None for a plain str."""
    annotation = function.args.args[0].annotation
    assert annotation is not None
    if not isinstance(annotation, ast.Subscript):
        return None
    value = annotation.slice
    elements = value.elts if isinstance(value, ast.Tuple) else [value]
    return {str(cast(ast.Constant, element).value) for element in elements}


def _typed(function_name: str = "provider") -> dict[str, ast.FunctionDef]:
    """Each registered name's own overload: the one whose Literal admits that name alone."""
    typed: dict[str, ast.FunctionDef] = {}
    for function in _overloads(function_name):
        names = _literal_names(function)
        if names is not None and len(names) == 1:
            typed[names.pop()] = function
    return typed


def test_the_overloads_name_exactly_the_registered_providers() -> None:
    overloads = _overloads()
    assert set(_typed()) == set(providers._REGISTRY)
    catches = [f for f in overloads if (_literal_names(f) or set()) == set(providers._REGISTRY)]
    assert len(catches) == 1
    assert any("deprecated" in ast.unparse(d) for d in catches[0].decorator_list)
    # The last overload takes any name, for a name known only at run time.
    assert _literal_names(overloads[-1]) is None
    assert overloads[-1].args.kwarg is not None


@pytest.mark.parametrize("name", sorted(providers._REGISTRY))
def test_each_overload_takes_its_leafs_keywords_as_the_leaf_types_them(name: str) -> None:
    function = _typed()[name]
    entry = providers._REGISTRY[name]
    # An extra the jobs that sync every extra leave out, a model run in process, is
    # checked here by the job that installs it.
    for dependency in providers._EXTRAS.get(entry.extra or "", ()):
        pytest.importorskip(dependency)
    module_name, _, class_name = entry.target.partition(":")
    leaf_class = getattr(importlib.import_module(module_name), class_name)
    parameters = dict(inspect.signature(leaf_class).parameters)
    keywords = {arg.arg: arg for arg in function.args.kwonlyargs}
    defaults = dict(zip(keywords, function.args.kw_defaults, strict=True))
    assert set(keywords) == set(parameters) | _LIMITS
    for keyword, parameter in parameters.items():
        annotation = keywords[keyword].annotation
        assert annotation is not None
        expected = parameter.annotation
        if (name, keyword) == ("system-one", "model"):
            expected = "SystemOneModel"
        if keyword == "client":
            assert "httpx2" in expected or "AsyncOpenAI" in expected, expected
            expected = "object | None"  # no SDK type in an overload
        assert ast.unparse(annotation) == expected, keyword
        has_default = parameter.default is not inspect.Parameter.empty
        assert (defaults[keyword] is not None) == has_default, keyword


def test_the_system_one_model_literal_is_the_models_the_leaf_knows() -> None:
    from idia.providers import system_one

    assert set(get_args(system_one.SystemOneModel)) == set(system_one._MODELS)


_BUILTINS = {"str", "int", "float", "bool", "object", "None"}


def test_no_overload_annotation_names_a_module_outside_idia_and_typing() -> None:
    """A program without an extra must type-check: an overload names no SDK type."""
    tree = ast.parse(Path(_provider.__file__).read_text(encoding="utf-8"))
    sources: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                sources[alias.asname or alias.name] = node.module or ""
        elif isinstance(node, ast.Import):
            for alias in node.names:
                sources[(alias.asname or alias.name).split(".")[0]] = alias.name
    allowed = ("idia", "typing", "typing_extensions")
    named: set[str] = set()
    for function in [*_overloads(), *_overloads("leaf")]:
        arguments = function.args
        annotations = [
            arg.annotation
            for arg in (*arguments.args, *arguments.kwonlyargs, arguments.kwarg)
            if arg is not None and arg.annotation is not None
        ]
        for annotation in [*annotations, function.returns]:
            assert annotation is not None
            for node in ast.walk(annotation):
                if isinstance(node, ast.Name) and node.id not in _BUILTINS:
                    named.add(node.id)
    assert named
    for name in named:
        module = sources.get(name)
        assert module is not None, f"{name} is not imported"
        assert module.split(".")[0] in allowed, f"{name} comes from {module}"


def test_the_leaf_overloads_are_the_provider_overloads_without_the_dispatcher_keywords() -> None:
    provider_overloads, leaf_overloads = _overloads(), _overloads("leaf")
    assert len(leaf_overloads) == len(provider_overloads)
    for given, built in zip(provider_overloads, leaf_overloads, strict=True):
        assert _literal_names(built) == _literal_names(given)
        assert built.returns is not None
        assert ast.unparse(built.returns) == "Leaf"
        expected = [a for a in given.args.kwonlyargs if a.arg not in _LIMITS]
        assert [ast.dump(a) for a in built.args.kwonlyargs] == [ast.dump(a) for a in expected]
        assert (built.args.kwarg is None) == (given.args.kwarg is None)
