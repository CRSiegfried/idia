"""The registry and the import guard."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from idia import IdiaError, Leaf, providers
from idia.providers import ProviderNotFound, _Entry, _load_leaf


def test_registry_ships_these_leaves() -> None:
    assert set(providers._REGISTRY) == {
        "system-one",
        "openai-decisions",
        "openai-chat",
        "cloudflare-clef",
        "cloudflare-clairic",
    }
    assert set(providers._EXTRAS) == {"system-one", "openai", "cloudflare"}


def test_unknown_name_raises_a_typed_error() -> None:
    with pytest.raises(ProviderNotFound) as raised:
        _load_leaf("nowhere")
    assert raised.value.name == "nowhere"
    assert raised.value.registered == tuple(sorted(providers._REGISTRY))
    assert str(raised.value).startswith("no provider is registered as 'nowhere'; registered: ")
    for name in providers._REGISTRY:
        assert name in str(raised.value)
    assert str(ProviderNotFound("nowhere")) == "no provider is registered as 'nowhere'"
    assert ProviderNotFound("nowhere").registered == ()
    assert isinstance(raised.value, IdiaError)
    assert isinstance(raised.value, LookupError)


def test_a_registered_leaf_is_loaded_with_its_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(providers._REGISTRY, "fake", _Entry(target="tests._fake_leaf:FakeLeaf"))
    leaf = _load_leaf("fake", model="fake-2")
    assert isinstance(leaf, Leaf)
    assert leaf.model == "fake-2"


def test_a_registered_class_that_is_not_a_leaf_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(providers._REGISTRY, "bad", _Entry(target="tests._fake_leaf:NotALeaf"))
    with pytest.raises(TypeError, match="Leaf"):
        _load_leaf("bad")


def test_missing_extra_dependency_is_reraised_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        providers._REGISTRY,
        "guarded",
        _Entry(target="tests._fake_leaf_missing_dependency:Guarded", extra="guarded"),
    )
    monkeypatch.setitem(providers._EXTRAS, "guarded", ("idia_test_missing_sdk",))
    with pytest.raises(ModuleNotFoundError) as raised:
        _load_leaf("guarded")
    assert 'pip install "idia[guarded]"' in str(raised.value)
    assert raised.value.name == "idia_test_missing_sdk"
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_any_module_an_extra_installs_gets_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        providers._REGISTRY,
        "guarded",
        _Entry(target="tests._fake_leaf_missing_dependency:Guarded", extra="guarded"),
    )
    monkeypatch.setitem(providers._EXTRAS, "guarded", ("some_sdk", "idia_test_missing_sdk"))
    with pytest.raises(ModuleNotFoundError) as raised:
        _load_leaf("guarded")
    assert 'pip install "idia[guarded]"' in str(raised.value)


def test_an_unrelated_import_failure_is_not_disguised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        providers._REGISTRY,
        "other",
        _Entry(target="tests._fake_leaf_missing_dependency:Guarded", extra="other"),
    )
    monkeypatch.setitem(providers._EXTRAS, "other", ("some_other_sdk",))
    with pytest.raises(ModuleNotFoundError) as raised:
        _load_leaf("other")
    assert "pip install" not in str(raised.value)

    monkeypatch.setitem(providers._REGISTRY, "typo", _Entry(target="idia.providers.nope:X"))
    with pytest.raises(ModuleNotFoundError) as raised:
        _load_leaf("typo")
    assert "pip install" not in str(raised.value)


def test_every_extra_a_registry_entry_names_is_known_to_the_guard() -> None:
    for name, entry in providers._REGISTRY.items():
        assert entry.extra is None or entry.extra in providers._EXTRAS, name


def _elicitations(path: Path) -> list[tuple[int, object]]:
    """Each ``elicitation=`` passed in a call in the module at ``path``: a constant's value,
    or the expression's source where it is not a constant."""
    found: list[tuple[int, object]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "elicitation":
                    value = keyword.value
                    held = value.value if isinstance(value, ast.Constant) else ast.unparse(value)
                    found.append((node.lineno, held))
    return found


def test_only_the_chat_leaf_reports_an_elicitation_other_than_passthrough() -> None:
    root = Path(providers.__file__).parent
    reported = {
        str(path.relative_to(root)): _elicitations(path) for path in sorted(root.rglob("*.py"))
    }
    chat = str(Path("chat") / "openai.py")
    assert [value for _, value in reported.pop(chat)] == ["logprobs"]
    others = {
        module: found
        for module, found in reported.items()
        if any(value != "passthrough" for _, value in found)
    }
    assert others == {}
    assert reported["system_one.py"]
    assert reported["openai_decisions.py"]
