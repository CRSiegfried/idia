# Add a provider to idia

This guide is for a contributor moving a leaf into idia itself, so that `idia.provider(name)`
and `idia.leaf(name)` build it and an extra installs what it needs. Write the leaf first, as
[Write a leaf](write-a-leaf.md) describes: that guide's rules hold for every leaf idia ships,
and this one adds what shipping it takes. The examples below quote Clairic, an extraction
leaf over HTTP, where idia already does the thing described; a new leaf goes beside it. A
model run in your own process differs in a few places, each noted.

Read [CONTRIBUTING](../../CONTRIBUTING.md) first: its ground rules hold for every change.

## The module

A leaf lives in its own module under `idia.providers`, named for its provider:
`src/idia/providers/<name>.py`, or a package when it has several modules. The module's
`__all__` lists the names a caller may use: the leaf class and anything its constructor
takes. A leaf over a raw HTTP API may inherit `HttpLeaf` from `idia.providers.http`, which
owns the client, the deadline and the status table; see
[Providers](../spec/providers.md#the-http-base). A leaf over an SDK sets the SDK's own
retries to zero.

The module imports its extra's dependency at its top, so that importing it without the
extra fails with `ModuleNotFoundError`, which the import guard turns into the install hint.
Nothing under `idia.providers` imports the call loop, and `idia.core` imports nothing from
`idia.providers`.

A model run in your own process needs its tests to run against stored outputs with no
runtime installed. Under the guard as it is, that takes two modules: the leaf's logic in an
unlisted module that does not import the runtime, such as `idia/providers/_<name>.py`, which
the tests import with an injected model, and the registered module importing the runtime at
its top and re-exporting the class. Moving the runtime's import into `load()` instead is not
supported yet: it would let `idia.leaf(name)` succeed with no extra installed, so the guard
and the bare-install check would have to change.

## The registry and the import guard

`idia.providers` holds the registry: each name maps to the leaf's `module:Class` and the
extra that installs it. Add the leaf's entry beside the others:

```python
# src/idia/providers/__init__.py
    "cloudflare-clairic": _Entry(target="idia.providers.cloudflare:Clairic", extra="cloudflare"),
```

The guard needs the top-level modules each extra installs, so that a missing one is
reported with the extra to install. Add the extra's entry when it is new:

```python
# src/idia/providers/__init__.py
_EXTRAS: Final[dict[str, tuple[str, ...]]] = {
    "system-one": ("httpx2",),
    "openai": ("openai", "httpx2"),
    "cloudflare": ("httpx2",),
    "gliner": ("gliner2", "torch"),
}
```

`test_registry_ships_these_leaves`, in `tests/test_providers.py`, lists every registered name
and extra; add yours.

## The typed overloads

`idia.provider` and `idia.leaf` have one overload per registered name, in
`src/idia/_provider.py`, typed with the leaf's constructor keywords so that a type checker
refuses an option the leaf does not take. Add one of each for the new name, the leaf's
keywords first; the `provider` overload adds `retry`, `concurrency` and `rate`. An
annotation names no type from the extra's dependency, so that a program without the extra
still type-checks: a client is `object | None`. Clairic's `leaf` overload:

```python
# src/idia/_provider.py
    @overload
    def leaf(
        name: Literal["cloudflare-clairic"],
        *,
        model: Literal["clairic", "@cf/cloudflare/clairic"] = ...,
        account_id: str | None = ...,
        api_token: str | None = ...,
        client: object | None = ...,
        capabilities: Capabilities | None = ...,
    ) -> Leaf: ...
```

Add the name, too, to the `Literal` of each deprecated overload, the one that catches a
registered name given a keyword its leaf does not take. `tests/test_provider.py` reads the
overloads from the source and checks each against the leaf's constructor.

A leaf with a `load()` step is not ready after `idia.provider(name)` unless it was given a
loaded model: a `Dispatcher` has no load step, and every call through it fails with a
non-retryable `ProviderError`. A caller builds it with `idia.leaf(name)`, awaits `load()`, then wraps it
in a `Dispatcher`. Whether `idia.provider` should load such a leaf, refuse it, or take one
already loaded is still open.

## The extra

Each provider's dependencies are an extra in `pyproject.toml`, so that the core installs
with none:

```toml
# pyproject.toml
[project.optional-dependencies]
# One extra per wire provider; the core needs none of them.
system-one = ["httpx2>=2.16"]
openai = ["openai>=3.35.1", "httpx2>=2.16"]
cloudflare = ["httpx2>=2.16"]
```

Then run `uv lock` and commit the lock file. CI syncs every extra, so that the provider
tests run rather than skip; add the new dependency's import to its "Every extra imports"
step in `.github/workflows/ci.yml`. The build job installs the wheel alone and runs
`scripts/check_bare_install.py`, which checks that idia imports with no extra and that each
registered provider names its extra in the install hint.

A local runtime such as torch is large, and CI downloads no model weights: such a leaf's
tests replay stored outputs through an injected model, needing neither. So its extra is left
out of the jobs that sync every extra, and checked by a job of its own:

- In `.github/workflows/ci.yml`, the jobs that run `uv sync --locked --all-extras` add
  `--no-extra <extra>`.
- A job of its own runs `uv sync --locked --extra <extra>` with torch taken from the
  CPU-only wheel index, then the "Every extra imports" check for that extra alone, as
  `uv run python -c "import <runtime>, torch"`.
- The CPU-only index is named in `pyproject.toml`, as uv's documentation for PyTorch shows:
  a `[[tool.uv.index]]` entry for `https://download.pytorch.org/whl/cpu` with
  `explicit = true`, and `torch` taken from it under `[tool.uv.sources]`. Run `uv lock` and
  check that the lock's torch wheels are the CPU ones before you commit it.

## The public API

A module's names are public only once the module is listed: in the ground rules of
[CONTRIBUTING](../../CONTRIBUTING.md#ground-rules-for-changes), and in `LISTED` in
`tests/test_package.py`, which checks that each listed module's `__all__` names what it
holds. Listing a module promises its names under the deprecation rule, so the maintainer
lists it: the change that adds the leaf writes the module's `__all__` and leaves the module
unlisted unless asked. Until it is listed, the leaf is reached through `idia.provider` and
`idia.leaf`.

## The spec

`docs/spec/providers.md` is normative and is changed in the same pull request as the code.
Add the leaf's row to the table at its top (name, leaf, extra, dependency) and a section of
its own covering:

- its constructor and options, and the option keys it reads per request;
- what it sends, and the label or question wording it chooses;
- how it reads the answers, kind by kind, including what it leaves `None`;
- what it refuses before sending, and each error it raises, retryable or not;
- its provenance: `elicitation`, `calibrated`, `model_reported`, `snapshot`, `prompt_sent`,
  `usage`, `request_id` and `raw_body`, and for a model in your own process the runtime's
  versions the raw body records;
- its capabilities record, field by field, each value as measured or documented, `None`
  where unknown;
- its estimator, and what it counts.

`docs/spec` is written one sentence per line. Update, too, the README's table of providers,
and the providers the agent skill in `skills/idia/SKILL.md` names in its description.

## Tests: the conformance suite and its recordings

Every leaf passes the conformance suite: the leaf half against the leaf, the provider half
once wrapped, each through a harness of the leaf's own, as [Write a leaf](write-a-leaf.md#test-it-with-the-conformance-suite)
shows. Assert the set of checks that are not applicable exactly.

A wire leaf's tests replay **cassettes**: one JSON file per recorded exchange, under
`tests/cassettes/<leaf>/<model>/<features>/<name>.json`, replayed through
`HttpLeafHarness` in `tests/_http.py`, which a leaf's harness subclasses. The
[test-doubles spec](../spec/test-doubles.md#the-http-harness-in-idias-tests) describes the
harness and the cassette format. A cassette is recorded once by a live run and must store no
secret: the recording filters secret headers and replaces the values it is told are secret,
a run's recording removes any file still holding a known secret, and the tests fail any
committed cassette that holds a key's shape.

A model run in your own process has no exchanges to record. Its tests replay **stored
outputs**: what the runtime returned for each input, recorded once by a gated run against
the real model, with the runtime's versions noted beside them, and played by a model the
harness injects in place of the real one.

## Gate every real call

No test in CI calls a wire API or loads a model.

A test that calls a wire API carries the `live` marker, as a module-level `pytestmark` in its
own `*_live.py` module, and runs only with `IDIA_LIVE=1`. Run it once per model, recording
the cassettes it leaves behind:

```sh
IDIA_LIVE=1 uv run pytest -m live -s
```

A test that loads a model in your process carries the `local_model` marker and runs only
with `IDIA_LOCAL_MODEL=1`, with the runtime installed and the weights already cached. That
switch also sets `HF_HUB_OFFLINE=1` before any test module is imported, so the run reads
nothing from the network: it never downloads weights. Run it once per model, recording the
stored outputs it checks:

```sh
IDIA_LOCAL_MODEL=1 uv run pytest -m local_model -s
```

Both gates are in `tests/conftest.py`:

```python
# tests/conftest.py
def pytest_configure(config: pytest.Config) -> None:
    """With ``IDIA_LOCAL_MODEL=1``, set the Hugging Face hub offline before any test module
    is imported, so a run against a model in process reads only cached weights."""
    if os.environ.get("IDIA_LOCAL_MODEL") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test marked ``live`` unless ``IDIA_LIVE=1`` is set, and one marked
    ``local_model`` unless ``IDIA_LOCAL_MODEL=1`` is set."""
    if item.get_closest_marker("live") is not None and os.environ.get("IDIA_LIVE") != "1":
        pytest.skip("live call; set IDIA_LIVE=1 to run")
    if (
        item.get_closest_marker("local_model") is not None
        and os.environ.get("IDIA_LOCAL_MODEL") != "1"
    ):
        pytest.skip("loads a model in process; set IDIA_LOCAL_MODEL=1 to run, its weights cached")
```

Each marker is declared under `markers` in `pyproject.toml`'s pytest options, since the
tests run with `--strict-markers`.

## The changelog

Record the new provider under `Unreleased` in `CHANGELOG.md`, under **Added**: its name, its
extra, and what it answers.

## The checks

Run every check before you open the pull request; CI runs the same:

```sh
uv sync --all-extras
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest --cov
uv run python scripts/check_spec.py
uv run python scripts/check_terms.py
uv run python scripts/check_terms.py --subjects main..HEAD
```

`check_terms.py` checks that the library, its docs and its commit subjects describe the
library; with `--subjects` it reads the subjects of your commits, which CI checks on a push
to main.
