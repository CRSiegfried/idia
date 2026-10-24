# Contributing

Thanks for your interest in idia. The design and the spec are still young, so contributions start with an issue.

## Issues first

- **Bugs:** open an issue with a minimal reproduction, the idia and Python versions, the provider, and what you expected.
- **Changes and features:** open a proposal issue before writing code. Say what problem it solves and which provider it bears on.
- **Pull requests** are welcome once an issue is agreed. Link it in the PR.

## Development

```sh
uv sync --all-extras
uv run ruff check
uv run ruff format --check
uv run pyright
uv run python scripts/check_spec.py
uv run python scripts/check_terms.py
uv run pytest --cov
```

All six must pass, the coverage floor included: the spec check for the links, headings and index of `docs/spec`, and the terms check that `src/`, `docs/`, `tests/`, `scripts/`, `skills/`, the workflows, the top-level Markdown and `pyproject.toml` read in plain library terms, its patterns tested against planted terms on every run. On a push to main, CI also checks that commit subjects describe library changes, with the same patterns; a pull request's subjects are not checked, and `uv run python scripts/check_terms.py --subjects main..HEAD` shows what it would find. Plain `uv sync` leaves the extras out, and the provider tests then skip. CI runs the same six on every push to main and every pull request, on Python 3.11 to 3.14 on Linux, macOS and Windows, after `uv sync --all-extras --no-extra gliner`, and fails if an extra is missing rather than letting its tests skip. The `gliner` extra, a model run in process on torch, has a job of its own that installs it with torch from the CPU-only index and runs the tests with the same coverage floor, those that import its module among them; no job downloads model weights. Its build job also installs the built wheel alone in a fresh venv and runs `scripts/check_bare_install.py`, which checks that idia imports with no extra and that each provider names its extra in the `pip install` hint. Tests never make a live call: a leaf's wire tests replay recorded cassettes, and the core's tests answer from the scripted leaf. A test that must reach a wire API carries the `live` marker, as a module-level `pytestmark` in its own `*_live.py` module, and runs only with `IDIA_LIVE=1`, once per model, recording the cassette it leaves behind with secrets filtered out. Select the live tests with `-m live`, the marker, not `-k live`, which matches names and so picks up other tests too: `IDIA_LIVE=1 uv run pytest -m live -s`. A test that loads a model in this process carries the `local_model` marker instead and runs only with `IDIA_LOCAL_MODEL=1`, which also sets `HF_HUB_OFFLINE=1`, so it reads only cached weights and never downloads: `IDIA_LOCAL_MODEL=1 uv run pytest -m local_model -s`.

A script run outside pytest that imports `tests`, such as a one-off live recording, needs the repository root on the path: `PYTHONPATH=. uv run python …`.

## Ground rules for changes

- **The core has no provider and no engine dependency.** Each leaf or bridge with a dependency, a wire SDK or a local runtime such as torch, is an optional extra behind an import guard; the scripted and recorded leaves ship in the core with none. `core.py` imports nothing from `providers/`.
- **A leaf never retries.** It sends one wire request in one call, sets any SDK's retries to zero, raises `ProviderError` and nothing else, and leaves every retry, wait and attempt count to the core's dispatcher.
- **Probabilities are what the wire returned.** Nothing renormalizes, rounds or corrects one; a distribution that breaks a rule becomes a typed fault for that question, and its siblings stand.
- **The spec is normative.** A change that alters behavior the spec describes changes the spec in the same PR, with tests.
- **`docs/spec` is written one sentence per line**, so a diff shows only the sentences changed.
- **Every provider passes the conformance suite** before it is used, a leaf half for the leaf and a provider half once wrapped.
- **The public API is what `idia.__all__` lists, and what the `__all__` of each of these modules lists:** `idia.cloudflare`, `idia.conformance`, `idia.providers.http`, `idia.providers.system_one`, `idia.providers.openai_decisions`, `idia.providers.chat`, `idia.providers.cloudflare`, `idia.providers.scripted` and `idia.providers.recorded`. The leaves live in their own modules rather than in `idia`, since each needs its extra and `import idia` needs none. Every other name is internal: modules and names starting with an underscore, a name a listed module leaves out of its `__all__`, and `idia.providers` itself. Removing or changing a public name needs a deprecation first: one minor release with a `DeprecationWarning`. A new `Capabilities` field is required, so adding one breaks a record written by hand; list each such field under **Changed**.
- **Record user-visible changes** under `Unreleased` in [CHANGELOG.md](CHANGELOG.md). A new kind, elicitation or validation rule is a minor release.
