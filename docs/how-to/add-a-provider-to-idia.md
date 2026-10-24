# Add a provider to idia

This guide is for a contributor moving a leaf into idia itself, so that `idia.provider(name)`
and `idia.leaf(name)` build it and an extra installs what it needs. Write the leaf first, as
[Write a leaf](write-a-leaf.md) describes: that guide's rules hold for every leaf idia ships,
and this one adds what shipping it takes.

The worked example is `gliner`, the GLiNER2 leaf: a span model run in your own process, whose
extra pulls in torch. A leaf over a wire API takes the same steps with fewer of them, and
the places it differs are noted; Clairic, in `idia.providers.cloudflare`, is one to read
beside it. The blocks headed by a file name quote that file as it is.

Read [CONTRIBUTING](../../CONTRIBUTING.md) first: its ground rules hold for every change.

## Inside idia

A leaf inside idia follows the outside guide, with four differences:

- **Import from `idia.core`**, and `validate_answers` from `idia.answers`, not from `idia`,
  as idia's own leaves do: a leaf's imports then name only the modules that define the
  vocabulary, never the package root, which also re-exports the call loop.
- **`name` is the registered name**: the leaf's `name` is the string its registry entry is
  keyed by, `"gliner"` for the GLiNER2 leaf, so every outcome names the provider a caller
  asked for.
- **A JSON value's arrays are tuples.** `JsonValue` is typed with tuples, so the
  `prompt_sent`, `raw_body` and `raw_usage` you build hold tuples, not lists; strict pyright
  refuses a list there.
- **pyright runs in strict mode** over `src`, `tests` and `scripts` alike.

## The module

A leaf lives under `idia.providers`, named for its provider: `src/idia/providers/<name>.py`,
or a package when it has several modules. The module's `__all__` lists the names a caller
may use: the leaf class and anything its constructor takes. A leaf over a raw HTTP API may
inherit `HttpLeaf` from `idia.providers.http`, which owns the client, the deadline and the
status table; see [Providers](../spec/providers.md#the-http-base). A leaf over an SDK sets
the SDK's own retries to zero.

The registered module imports at its top every package the leaf needs from its extra,
including those its runtime imports lazily, so that importing it without the extra, or with
one of those packages missing, fails with `ModuleNotFoundError`, which the import guard
turns into the install hint. Import each one, not only the runtime: some runtimes import
their heavy dependencies lazily, so `import gliner2` succeeds without torch, and a missing
torch would surface only at the first call. Nothing under `idia.providers` imports the call
loop, and `idia.core` imports nothing from `idia.providers`.

A model run in your own process needs its tests to run against stored outputs with no
runtime installed, so it takes two modules:

- **The leaf's logic** in an unlisted module that imports none of the runtime,
  `idia/providers/_<name>.py`. The tests import it and inject a model. The real loader lives
  here too, as a function the leaf's `load()` runs on its worker. It imports the runtime by
  name, into `Any`, so that strict pyright passes with the runtime not installed:

```python
# src/idia/providers/_gliner.py
    # Imported by name, so that this module type-checks with the runtime not installed.
    hub: Any = importlib.import_module("huggingface_hub")
    gliner2: Any = importlib.import_module("gliner2")
```

- **The registered module**, which imports every package of the extra and re-exports the
  leaf and its runtime protocol. Each import carries the pyright and ruff ignores a missing,
  unused import needs:

```python
# src/idia/providers/gliner.py
import gliner2  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401
import google.protobuf  # pyright: ignore[reportMissingImports, reportMissingModuleSource, reportUnusedImport] # noqa: F401
import huggingface_hub  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401
import sentencepiece  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401
import torch  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401
import transformers  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401

from idia.providers._gliner import Gliner, GlinerRuntime
```

Moving the runtime's import into `load()` instead is not supported: it would let
`idia.leaf(name)` succeed with no extra installed, so the guard and the bare-install check
would have to change.

## The registry and the import guard

`idia.providers` holds the registry: each name maps to the leaf's `module:Class` and the
extra that installs it. Add the leaf's entry beside the others:

```python
# src/idia/providers/__init__.py
    "gliner": _Entry(target="idia.providers.gliner:Gliner", extra="gliner"),
```

The guard needs every top-level module each extra installs, so that a missing one is
reported with the extra to install. List each package the registered module imports, by its
top-level name (`google` for protobuf):

```python
# src/idia/providers/__init__.py
_EXTRAS: Final[dict[str, tuple[str, ...]]] = {
    "system-one": ("httpx2",),
    "openai": ("openai", "httpx2"),
    "cloudflare": ("httpx2",),
    "gliner": ("gliner2", "torch", "transformers", "huggingface_hub", "google", "sentencepiece"),
}
```

These two blocks are quoted from the source, and `tests/test_docs.py` checks each quote in
this guide verbatim: when you change a quoted part, change its quote here too.

Three tests list every registered name, and each needs yours:
`test_registry_ships_these_leaves` and its extras in `tests/test_providers.py`, and
`test_provider_names_are_what_provider_not_found_lists` in `tests/test_leaf.py`.

The docstring of `idia/providers/__init__.py`, the introduction of `docs/spec/providers.md`
and the ground rule in CONTRIBUTING each say what the registered leaves are, and are written
for every kind of leaf: check that each still holds for yours.

## The typed overloads

`idia.provider` and `idia.leaf` have one overload per registered name, in
`src/idia/_provider.py`, typed with the leaf's constructor keywords so that a type checker
refuses an option the leaf does not take. Add one of each for the new name, the leaf's
keywords first; the `provider` overload adds `retry`, `concurrency` and `rate`. An
annotation names no type from the extra's dependencies, so that a program without the extra
still type-checks: a client is `object | None`. A runtime a caller injects is typed by the
protocol the private module defines, imported under `TYPE_CHECKING`:

```python
# src/idia/_provider.py
    from idia.providers._gliner import GlinerRuntime
```

```python
# src/idia/_provider.py
    @overload
    def leaf(
        name: Literal["gliner"],
        *,
        model: Literal["fastino/gliner2.5-small-v1"] = ...,
        revision: str | None = ...,
        threshold: float = ...,
        runtime: GlinerRuntime | None = ...,
    ) -> Leaf: ...
```

Add the name, too, to the `Literal` of each deprecated overload, the one that catches a
registered name given a keyword its leaf does not take. `tests/test_provider.py` reads the
overloads from the source and checks each against the leaf's constructor.

A leaf with a `load()` step is not ready after `idia.provider(name)` unless it was given a
loaded model: a `Dispatcher` has no load step, and every call through it fails with a
non-retryable `ProviderError`. A caller builds it with `idia.leaf(name)`, awaits `load()`,
then wraps it in a `Dispatcher`. Whether `idia.provider` should load such a leaf, refuse
it, or take one already loaded is still open.

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
# One extra per model run in process. torch is named so that it comes from the CPU-only index.
gliner = ["gliner2[local]>=2.0.0", "torch>=2.1", "protobuf>=4", "sentencepiece>=0.2"]
```

Give each dependency a floor, not a pin: the stored outputs record the versions they were
made with, and the gated run against the real model tells you when a newer runtime moves
them. Then run `uv lock` and commit the lock file. The build job installs the wheel alone
and runs `scripts/check_bare_install.py`, which checks that idia imports with no extra and
that each registered provider names its extra in the install hint.

A wire leaf's extra is small, and CI installs it with every other: add its import to the
"Every extra imports" step of the test job in `.github/workflows/ci.yml`.

A local runtime such as torch is large, and CI downloads no model weights, so such an extra
is left out of the jobs that install every extra and installed by a job of its own:

- The lint and test jobs run `uv sync --locked --all-extras --no-extra <extra>`.
- The job of its own installs the extra alone, with torch from the CPU-only index, on one
  Python version, and runs the import check and then the whole suite with its coverage
  floor, since the tests that import the registered module run only there:

```yaml
# .github/workflows/ci.yml
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
        with:
          python-version: "3.13"
          working-directory: idia
      # torch comes from the CPU-only index that pyproject.toml names; no weights are fetched.
      - run: uv sync --locked --extra gliner
      - name: Every extra imports
        run: uv run python -c "import gliner2, torch, transformers, huggingface_hub, google.protobuf, sentencepiece"
      # The tests that import the registered module, which the other jobs skip, run here, with
      # the coverage floor the test jobs hold.
      - run: uv run pytest --cov
```

- The CPU-only index is named in `pyproject.toml`. `[tool.uv.sources]` reaches a package
  only when it is a direct dependency, so the extra names `torch` itself, even though the
  runtime already depends on it. Run `uv lock` and check that the lock's torch wheels are
  the CPU ones before you commit it:

```toml
# pyproject.toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

- Name the extra in `SEPARATE_EXTRAS`, in `tests/_support.py`. Two existing tests import
  every registered module, the one that checks the overloads and the one that checks no
  provider module reaches the call loop; they skip only for an extra listed there, so
  without it both fail in the jobs that leave the extra out:

```python
# tests/_support.py
SEPARATE_EXTRAS: frozenset[str] = frozenset({"gliner"})
```

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
  `usage`, `request_id`, `versions`, `client_versions` and `raw_body`, each detail's reason
  in `unreported` where it holds no value, and what it `claimed`;
- its capabilities record, field by field, each value as measured or documented, `None`
  where unknown;
- its estimator, and what it counts;
- for a model in your own process, its load step: what `load()` fetches, how it resolves the
  revision, what it loads from that commit, and what it raises when the runtime or the
  weights are missing;
- anything the runtime does that a caller would not guess, such as an undocumented method
  the leaf relies on.

Measure the record rather than assume it, and say in the spec what you measured. GLiNER2
scores one shared pool of candidate spans across all the labels of a pass, so its
`max_questions` of 16 is measured, not documented: passes of up to 16 labels found every
label, 32 lost 2 and 64 found 29.

`docs/spec` is written one sentence per line. Update, too, the README's table of providers,
and the providers the agent skill in `skills/idia/SKILL.md` names in its description.

## Tests: the conformance suite and its recordings

Every leaf passes the conformance suite: the leaf half against the leaf, the provider half
once wrapped, each through a harness of the leaf's own, as
[Write a leaf](write-a-leaf.md#test-it-with-the-conformance-suite) shows. Assert the set of
checks that are not applicable exactly, and every note a passing check gives. A harness
satisfies the suite's protocols by shape: declare its attributes as plain class attributes
or properties, not `Final`, which a protocol's attribute does not accept.

A wire leaf's tests replay **cassettes**: one JSON file per recorded exchange, under
`tests/cassettes/<leaf>/<model>/<features>/<name>.json`, replayed through
`HttpLeafHarness` in `tests/_http.py`, which a leaf's harness subclasses. The
[test-doubles spec](../spec/test-doubles.md#the-http-harness-in-idias-tests) describes the
harness and the cassette format. A cassette is recorded once by a live run and must store no
secret: the recording filters secret headers and replaces the values it is told are secret,
a run's recording removes any file still holding a known secret, and the tests fail any
committed cassette that holds a key's shape.

A model run in your own process has no exchanges to record. Its tests replay **stored
outputs**, one JSON file per model at `tests/stored/<name>/<model>/outputs.json`, a model id
with a slash taking two directories: `{"recorded_with": {...}, "cases": [...]}`. The
`recorded_with` object holds the commit the pinned revision resolved to, the Python version
and each runtime package's version; each case holds every argument the leaf passed the runtime (the text, the labels
and any threshold), the runtime's raw output and its token count. A model the harness
injects in place of the real one plays them back, keyed by those arguments.

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
nothing from the network: it never downloads weights. Run it once per model, and add
`IDIA_RECORD=1`, set by hand, to record the stored outputs again:

```sh
IDIA_LOCAL_MODEL=1 uv run pytest -m local_model -s
IDIA_LOCAL_MODEL=1 IDIA_RECORD=1 uv run pytest -m local_model -s
```

The gated run checks each recorded case against the real model: its spans against the
texts the case expects, and where the case is stored, the spans and the token count exactly
and each score within a tolerance. It writes nothing unless `IDIA_RECORD=1` is set as well,
so a run that passes leaves the stored outputs as they were reviewed, and a score cannot
drift a little further on each run. With the switch, a run that passes writes back the
outputs of every recorded case, each stored one compared first, with the versions it ran
under; a failed comparison writes nothing either way. When a case's outputs have moved past
the tolerance for a reason you accept, its comparison fails and the switch writes nothing:
remove that case from the stored file, run with `IDIA_RECORD=1`, and review the diff of
the stored file. The switch, at the end of GLiNER2's gated test:

```python
# tests/test_gliner_local_model.py
    if os.environ.get("IDIA_RECORD") == "1":
        versions = {package: importlib.metadata.version(package) for package in _VERSIONED}
        recorded_with = {"revision": _REVISION, "python": platform.python_version(), **versions}
        record_outputs(cases, STORED, recorded_with)
```

The writer writes the file whole or not at all, through a file beside the stored one
replaced in one step, in UTF-8 with LF line endings. GLiNER2's writer:

```python
# tests/_gliner_cases.py
    document = {"recorded_with": dict(recorded_with), "cases": list(cases)}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with open(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise
```

The tests run with every warning an error, so a warning the runtime gives while it loads
fails the run. Allow that one warning by its message, in the gated module, never a whole
category:

```python
# tests/test_gliner_local_model.py
pytestmark = [
    pytest.mark.local_model,
    # Loading the checkpoint, torch warns that torch.jit.script is deprecated.
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated:FutureWarning"),
]
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

Record the new provider in `CHANGELOG.md` under the release being prepared, under **Added**:
its name, its extra, and what it answers. That is the top section until it is tagged, then a
new `Unreleased` section above it.

## The checks

Run every check before you open the pull request, twice: once with every extra except
those in `SEPARATE_EXTRAS`, as the lint and test jobs run, and once with every extra, as the
separate extra's own job runs. Some errors show in only one of them: pyright's on a missing runtime
show only without the extra, and the registered module's tests run only with it.

```sh
uv sync --all-extras --no-extra gliner
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest --cov
uv run python scripts/check_spec.py
uv run python scripts/check_terms.py
uv run python scripts/check_terms.py --subjects <base>..HEAD

uv sync --all-extras
uv run pyright
uv run pytest --cov

uv build
uvx twine check --strict dist/*
```

Then check the wheel alone, as the build job does: in a fresh virtual environment with only
the built wheel installed, `python scripts/check_bare_install.py`.

`check_terms.py` checks that the library, its docs and its commit subjects describe the
library. With `--subjects` it reads the subjects of the commits in a range, which CI checks
on a push to main: give it the commit your branch started from as `<base>`, since on a
branch stacked on another, `main..HEAD` also checks the commits of the branch beneath.
