# Write a leaf

A leaf is the part of idia that talks to one model: it takes one wire request, asks the
model, and returns the answers with what it knows of how they were made. This guide is for
writing a leaf outside idia, in your own package, against idia's public names only. You
wrap the finished leaf in a `Dispatcher` and pass idia's conformance suite.

The example is `SpanLeaf`, a leaf over a model that runs in your own process and finds spans
of text: an `extraction` of `text`, a single value or a list. Its "model" is a small Python
function, so every example here runs as written. A leaf over a wire API follows the same
steps, with an HTTP client where `SpanLeaf` has a worker thread; the section
[A model in your own process](#a-model-in-your-own-process) is the part only an in-process
leaf needs.

The example is six files in one directory, each code block headed by the file it belongs
to: `spanleaf.py`, the leaf; `main.py`, which uses it; `stored.py`, the model's stored
outputs; `test_spanleaf.py`, its tests; and `conftest.py` with `test_real_model.py`, the gate
for a run against the real model. You need
Python 3.11 or later, idia, and pytest for the tests.

## Use public names only

Import only what `idia.__all__` lists, and what the `__all__` of these modules lists:
`idia.conformance`, `idia.providers.scripted`, `idia.providers.recorded`,
`idia.providers.http`, `idia.cloudflare`, `idia.providers.system_one`,
`idia.providers.openai_decisions`, `idia.providers.chat` and `idia.providers.cloudflare`.
Every other name may change without notice: a name starting with an underscore, a module
starting with one, a name a listed module leaves out of its `__all__`, and `idia.providers`
itself. The list is kept in the
[contributing notes](https://github.com/CRSiegfried/idia/blob/main/CONTRIBUTING.md#ground-rules-for-changes).

`idia.provider(name)` builds a leaf idia ships and wraps it in a `Dispatcher`;
`idia.leaf(name)` builds the same leaf unwrapped, so you can wrap a leaf of your own around
it, and `idia.provider_names()` lists the names. A leaf of your own is never registered:
you build it and wrap it yourself.

## What a leaf is

`Leaf` is a protocol, satisfied by shape: nothing is subclassed. A leaf has five members:

- `name`, a short name for the leaf, and `model`, the model it asks. Every outcome names
  both.
- `capabilities`, the record of what it takes and returns ([below](#state-every-field-of-the-record)).
- `estimator`, which counts the tokens of a request ([below](#count-what-the-model-reads)).
- `async call(wire_request, deadline) -> WireResponse`, which answers one wire request.

A data member may be a plain attribute, a property or a dataclass field. `aclose()` is not
part of `Leaf`, but a leaf that holds anything to release has one: the dispatcher awaits it
when it is closed, so it is `async`, and it may be called again, so closing twice must be
harmless.

Five rules hold for every leaf:

- **One call sends one wire request**, or runs the model once. A leaf never retries: retries,
  waits and attempt counts are the dispatcher's. A leaf built on an SDK sets the SDK's own
  retries to zero.
- **It raises `ProviderError` and nothing else.**
- **It returns what the model gave.** Nothing is renormalized, rounded, corrected or
  computed: no confidence the model did not give, no `p_null` it did not give, no second
  probability for a `noul`, no `1 - p`.
- **It is not a provider until wrapped.** `Dispatcher(leaf)` makes it one.
- **It may be called concurrently.** One dispatcher may run several requests at once, each
  calling the leaf.

The conformance suite reads the module that defines your leaf's class, so keep that module
to the leaf. It must import nothing of the call loop: not `idia.dispatch`, `idia.retry` or
`idia.limits`, and not `Dispatcher`, `Router`, `Fallback`, `RetryPolicy`,
`ConcurrencyLimit`, `RateLimiter` or `provider` from `idia`. An `except` that names
`ProviderError` must end in a bare `raise`; a broad `except Exception` must end by raising.
The module must compute no `1 - x` and call no `noul_complement`. And the class must not
declare a `Capabilities` field, or `capabilities` itself, as a class attribute: set
`capabilities` on the instance.

`SpanLeaf`'s module starts with its imports and its model. The model takes the text and the
labels to look for, and returns the spans it found under each label, each a text, a start
and end offset, and a score:

```python
# spanleaf.py
"""A leaf that runs a span-finding model in this process."""

from __future__ import annotations

import asyncio
import platform
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, cast

from idia import (
    UNBOUNDED,
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    ExtractionAnswer,
    Fault,
    JsonValue,
    MappedAnswer,
    Notice,
    ProvenanceDetail,
    ProviderError,
    Question,
    Rounding,
    Rule,
    RuleFailure,
    Span,
    State,
    TextPart,
    TokenEstimate,
    UnreportedReason,
    Usage,
    WireRequest,
    WireResponse,
    unknown_options,
    validate_answers,
)

Found = tuple[str, int, int, float]
"""One span the model found: its text, its start and end offsets, and its score."""


class SpanModel(Protocol):
    """What the leaf needs of a loaded model: one pass over a text, and the model's own
    count of the tokens that pass encodes."""

    def extract(self, text: str, labels: Sequence[str]) -> Mapping[str, Sequence[Found]]: ...

    def count_tokens(self, text: str, labels: Sequence[str]) -> int: ...


class ToyModel:
    """A stand-in for a real span model: it finds the words it knows, scored by length."""

    known = {"a city": ("Paris", "Lyon", "Oslo"), "a person": ("Ada", "Grace")}

    def extract(self, text: str, labels: Sequence[str]) -> dict[str, list[Found]]:
        found: dict[str, list[Found]] = {}
        for label in labels:
            spans: list[Found] = []
            for word in self.known.get(label, ()):
                start = text.find(word)
                if start >= 0:
                    spans.append((word, start, start + len(word), round(0.5 + len(word) / 20, 2)))
            found[label] = spans
        return found

    def count_tokens(self, text: str, labels: Sequence[str]) -> int:
        return len(text.split()) + sum(len(label.split()) for label in labels)


def load_toy_model(revision: str | None) -> tuple[SpanModel, str | None]:
    """Load the model at the revision given, and return it with the commit that revision
    resolved to. A real loader resolves the revision first, then loads everything, the
    weights, the tokenizer and the configuration, from that commit's own files. The toy has
    no files, so its commit is the revision as given."""
    return ToyModel(), revision


def _run(model: SpanModel, text: str, labels: Sequence[str]) -> tuple[object, int]:
    """One pass on the worker: what the model returned, and its count of what it encoded."""
    return model.extract(text, labels), model.count_tokens(text, labels)
```

## State every field of the record

A `Capabilities` record says what the leaf takes and returns. idia's preflight, Fit, checks
every request against it before any call and refuses one that does not fit with
`DoesNotFit`, so a request that passes must be one your leaf takes. Every field must be
stated; none has a default.

A limit is a non-negative `int`, `UNBOUNDED` when nothing caps it, or `None` when you do not
know it. Fit checks a number, skips `UNBOUNDED` silently, and skips `None` with an
`unchecked-limit` warning on each wire request it bears on. Take each limit from the model's
documentation or measure it against the real model: `UNBOUNDED` when the documentation
states that there is no cap, `None` when neither tells you. Do not guess. `min_levels` and
`min_image_side` are an `int` or `None`, never `UNBOUNDED`, and 0 there means no minimum.

What Fit does with each field:

- `kinds` and `modalities`: a question of a kind not listed, or a state part (text, JSON or
  image) of a modality not listed, does not fit.
- `image_carriers`, `max_images`, `max_images_per_question`, `max_image_bytes`,
  `max_total_image_bytes`, `min_image_side` and `image_media_types`: the images of a state.
  A leaf without the image modality usually has no carrier, 0 for each limit and an empty
  set of media types; it may state `None` where it does not know one, or the values a
  feature that adds images brings. Fit reads none of them without an image in the state.
- `max_input_tokens`: each wire request's token estimate. `max_question_tokens`: the state
  with each one question.
- `max_body_bytes`: Fit's own measure of the body: the state's text and JSON as compact
  UTF-8, each image as its base64 length, and each question's key, instruction, label names
  and descriptions. The envelope your leaf adds is not counted, so the cap holds only
  approximately. A model with no body to cap has `UNBOUNDED`.
- `truncates_silently`: whether the model cuts an over-long input without saying so. `True`
  needs a known `max_input_tokens`, since only that lets Fit stop the state first, and the
  constructor refuses it with `None`. `None` means undocumented.
- `recounts_state_per_question`: the model reads the state once per question, so a wire
  request of `n` questions counts the state `n` times.
- `max_questions` and `max_total_options`: how many questions, and how many options in all,
  one wire request may carry, a `noul` counting 2. When the questions overflow these,
  `max_body_bytes` or `max_input_tokens`, and the state alone fits, Fit packs them in order
  into several wire requests, each repeating the state; a question that does not fit alone
  with the state raises `DoesNotFit`.
- `max_labels`, `max_levels` and `min_levels`: the labels of a `choice` and the levels of a
  `score`. Fit reads them only for those kinds, so for a leaf that answers neither, 0, as
  `SpanLeaf` gives, or `None` is neither checked nor warned.
- `key_patterns`: patterns every question key must match in full, for a model that
  restricts its ids.
- `carries_noul_descriptions` and `carries_structured_instructions`: whether the leaf sends
  a `noul`'s yes and no descriptions, and an instruction that is a JSON object or array.
  When false, Fit refuses a request that uses it.
- `returns_refusal`, `calibrated` and `snapshot_pinnable`: whether the model can decline a
  question, whether its probabilities are calibrated, and whether a response can name the
  exact model that answered. The conformance suite checks each against the responses.
- `confidence`: for each kind whose answers carry a confidence, what that confidence is.
  The string is free-form and idia never reads it: name what the number is, as the model's
  documentation names it, such as `"top-span score"`, or `"undocumented"` where the model
  publishes no formula. A kind not listed carries none, and a fallback over leaves whose
  strings differ reports `"mixed"`.
- `rounding`: the decimals the model rounds its probabilities and scores to, `None` for
  unrounded. Validation takes its tolerances from it.
- `opt_in_features`, `enabled_features`, `feature_kinds` and `feature_modalities`: the
  features a caller may turn on, those on, and the kinds or modalities each adds. A leaf
  with none has empty sets and empty maps.

`SpanLeaf`'s record. Its window of 512 tokens and its cap of 16 questions to a call stand in
for what a run against a real model measures ([below](#measure-the-record)):

```python
# spanleaf.py
CAPABILITIES = Capabilities(
    kinds=frozenset({"extraction"}),
    modalities=frozenset({"text"}),
    image_carriers=frozenset(),
    max_input_tokens=512,
    max_question_tokens=UNBOUNDED,
    max_body_bytes=UNBOUNDED,
    max_questions=16,
    max_images=0,
    max_images_per_question=0,
    max_image_bytes=0,
    max_total_image_bytes=0,
    max_labels=0,
    max_levels=0,
    max_total_options=UNBOUNDED,
    min_levels=0,
    min_image_side=0,
    image_media_types=frozenset(),
    key_patterns=(),
    recounts_state_per_question=False,
    returns_refusal=False,
    calibrated=False,
    snapshot_pinnable=True,
    carries_noul_descriptions=False,
    carries_structured_instructions=False,
    truncates_silently=False,
    confidence={"extraction": "top-span score"},
    rounding=Rounding(probability_decimals=None, score_decimals=None),
    opt_in_features=frozenset(),
    enabled_features=frozenset(),
    feature_kinds={},
    feature_modalities={},
)
```

## Count what the model reads

The estimator counts a request's tokens for Fit. It satisfies `TokenEstimator`:
`estimate(state, questions)` returns `TokenEstimate(tokens, margin)`, and Fit checks
`ceil(tokens * (1 + margin))` against the limits, the margin read as the decimal it prints
as. `questions` may be empty, meaning the state alone, which is how Fit checks the state for
overflow. A caller may give the dispatcher another estimator, which then replaces yours for
Fit's estimate only.

Count the state once in `estimate(state, questions)`. Where the record has
`recounts_state_per_question`, Fit itself adds the state's own estimate `n - 1` more times
for a wire request of `n` questions, each estimate taking its margin before the sum. If your
leaf takes images, the estimator prices them too, by your model's image formula: no field of
the record does. Where `truncates_silently` holds, the margin is all that stands between an
undercount and a silent cut, so measure it.

Count everything the model reads, not only the state: the labels, the instructions and any
schema text you send with them share the model's window. The margin covers how far your
count may fall short of the model's own.

```python
# spanleaf.py
class WordEstimator:
    """Counts a word as a token, each label once: the labels share the model's window, and
    two questions with one instruction share one label."""

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        words = sum(len(part.text.split()) for part in state.parts if isinstance(part, TextPart))
        labels = dict.fromkeys(str(question.instruction) for question in questions)
        words += sum(len(label.split()) for label in labels)
        return TokenEstimate(tokens=words, margin=0.2)
```

## Map the output, then validate it

Map what the model returned to the kind's answer shape and nothing more: a `NoulAnswer`, a
`LabelAnswer`, an `ExtractionAnswer` or a `BoxAnswer`, each keyed by its question's key; a
`Refusal` where the model declined the question. Then call
`validate_answers(questions, mapped, capabilities.rounding)`. It returns the answers and the
warnings: an answer that breaks a rule becomes a `Fault` for that question alone, a missing
key a `Fault`, and a key with no question is dropped with a warning.

When one question's answer cannot be mapped, build a `Fault` for it yourself:
`Fault(key, failures=(RuleFailure(rule=Rule.kind_shape, observed=<the type it came back
as>, bound=None),), mapped=None)`, or `Rule.candidate_shape` for an answer in the shape of a
kind idia has not admitted yet. Its siblings stand, and a `Fault` or `Refusal` you build
passes through `validate_answers` unchanged. Only an output that cannot be read at all fails
the whole call, with a non-retryable `ProviderError`. `SpanLeaf` does both, in `call` and
`_mapped` below. Parse a `date`, `time`, `datetime` or
`duration` value to its standard-library type before validating, and leave one you cannot
parse as given, for validation to fault. A list answer has one span per item.

Never compute a confidence. Pass the model's own score as `confidence` where it gives one,
for a kind your record lists, and leave it `None` where it gives none, as for a list of
spans: an average of the spans' scores is a number the model never gave. A score outside
[0, 1] becomes a `Fault`: do not rescale it. Leave `guess` and `p_null` as the model gave
them, `None` when it gives none.

A span is `Span(part, start, end)`: the index of the text part in the state, and 0-based,
end-exclusive offsets in code points, so `text[start:end]` is the span's text. Convert a
model's token or byte offsets before you build one.

Send each question's wording as the model takes it, or refuse the question: never drop part
of it. `SpanLeaf` sends each question's instruction as the label to look for, and records the
labels it sent.

## Raise ProviderError, retryable or not

A leaf raises `ProviderError(message, is_retryable=..., status_code=..., request_id=...)`
and nothing else. `is_retryable` tells the dispatcher whether another attempt could succeed:

`is_retryable` is false unless you say otherwise.

| What happened | `is_retryable` |
|---|---|
| The deadline passed before the call, or cut it | yes |
| A connection error or a transport timeout | yes |
| A wire status 408, 409, 429 (not a quota) or 5xx | yes |
| A quota the body names, whatever the status | no |
| Any other wire status that is not a success | no |
| An output the leaf cannot read | no |
| A question or state the leaf cannot send, refused before anything is sent | no |
| The model failed while running | no |
| The model is not loaded, or the leaf is closed | no |

Set `status_code` only to a status the wire returned, `retry_after` to the seconds the wire
asks you to wait (its `retry-after-ms` or `retry-after`), `None` otherwise, and `request_id`
when you have one. A refused credential is not retryable and carries its 401 or 403: the
dispatcher reads exactly that to leave the rest of the request unsent.
Never raise `DeadlineExceeded`: the dispatcher puts that on an outcome. Let
`asyncio.CancelledError` propagate; it is not a failure of the model. A model that declines
a question gives a `Refusal` answer, not an error.

## Treat the deadline as your own timeout

`call` gets the caller's deadline, the one instant every attempt, wait and retry shares. It
is the leaf's own timeout:

- Check `deadline.passed()` before sending anything, and raise a retryable `ProviderError` if
  it has: nothing is sent.
- Use `deadline.remaining()` only to size a wait, such as a timeout on the response, never
  below 0.
- When the wait runs out, raise a retryable `ProviderError`. If attempts remain, the
  dispatcher ends the wire request with `DeadlineExceeded`, your error kept on its failed
  attempt; on the last attempt the outcome carries your error.

## Warn, and record where the answers came from

A `WireResponse` carries the answers and the provenance only the leaf knows:

- `elicitation`: `"passthrough"`, the model's own answers as given.
- `calibrated`: the record's `calibrated`.
- `model_reported`: the model the wire, or the runtime, says answered.
- `snapshot`: the exact model version that answered, only when the record's
  `snapshot_pinnable` holds and the caller pinned one; otherwise `None`, with a
  `WARNING_NO_SNAPSHOT` warning.
- `prompt_sent`: what was sent, as JSON values; `None` only where the wire hides it.
  `SpanLeaf` leaves the text itself out on purpose and records its length and the labels.
- `usage`: `Usage(input_tokens, output_tokens, passes=None, raw_usage=...)`, each count as
  the wire or the runtime reports it and `None` where it reports none, never an estimate;
  the raw figures go in `raw_usage`. A model in your own process reports its input as its
  own tokenizer counts what it encoded, as `SpanLeaf` does.
- `request_id`: the wire's id for the call, `None` where nothing assigns one, as in process.
- `versions`: the versions of the runtime that ran the model, package to version, with
  `python` for the interpreter, read from the installed packages; empty for a wire API,
  whose runtime is the service's own. A per-call fingerprint a wire returns is not a version:
  leave it in the raw body.
- `client_versions`: the versions of the client libraries that made the call, such as your
  HTTP client or SDK, never `idia`, which the dispatcher adds.
- `raw_body`: what the model returned, whole. The dispatcher drops it, but a recording keeps
  it.

Every detail of the provenance profile is a value or has a reason in `unreported`, a map
from the detail's name to one of `UnreportedReason`'s codes; `DETAILS` and `REASONS` name
them all. The profile's details are `model_reported`, `snapshot`, `versions`,
`elicitation`, `calibrated`, `prompt_sent`, `usage.input_tokens`, `usage.output_tokens`,
`usage.passes` and `request_id`. The codes a leaf gives:

- `not-reported`: the wire or the runtime gives none, such as a missing request id;
- `not-pinned`: the snapshot, exactly when you give the `no-snapshot` warning;
- `hidden`: the provider's service withholds it;
- `not-applicable`: the detail has no meaning for your source, such as `versions` for a wire
  API, or for your kind of model, such as the output tokens of an encoder that writes none;
- `not-exposed`: a library between you and the provider has it but does not make it public.

`not-sent` is the dispatcher's alone, for a wire request that failed. A detail never has
both a value and a reason, and the constructor refuses one that does. Name in `claimed` the
details whose value you assert but cannot verify: `SpanLeaf` claims its `model_reported`,
which is its own constant, not something its model reports, and the revision a caller gives
for a model it loaded itself. The conformance suite checks the whole profile.
- `warnings`: `Notice` values. Put the warnings `validate_answers` returned first, then your
  own, such as `WARNING_NO_SNAPSHOT` or `WARNING_PROVIDER_TRUNCATED` when the model reports
  that it cut the input, and last `unknown_options(wire_request.options, known)`, which
  warns once naming every request option your leaf does not know.

A request's `options` are settings a caller passes per request. A key your leaf reads, it
records in `prompt_sent`. A key it knows but does not read, such as one meant for another
model behind the same router, has no effect, no warning and no record. A key it does not
know goes into `unknown_options`, and changes nothing else.

## A model in your own process

A model that runs in your process, on a local runtime such as torch, follows six more
practices, those idia sets for a leaf that runs its model in process.

- **One worker of its own.** Loading and every forward pass block, so run them on a
  `ThreadPoolExecutor(max_workers=1)` the leaf owns, never on the event loop. Calls to one
  leaf then run one at a time, and a `ConcurrencyLimit` above 1 on its dispatcher gives no
  parallelism.
- **Loading is a step of its own.** Give the leaf an `async load()`, the one place it may
  reach the network, to download the weights, and have the caller await it before the first
  call. Nothing inside `call` touches the network: the model is already in memory. A warm-up,
  one small forward pass at the end of `load()`, keeps its cost out of the first call. A
  call before the model is loaded raises a non-retryable `ProviderError` and runs nothing. A
  leaf given a loaded model needs no load.
- **A deadline cut leaves the worker busy.** Wait for the worker with
  `asyncio.wait_for(future, max(deadline.remaining(), 0))`, and turn a timeout into a
  retryable `ProviderError`. The forward pass cannot be stopped: it runs to its end, and a
  call from another request, made before the pass ends, waits behind it inside its own
  deadline. A call still queued behind it when its own deadline passes never runs.
- **`aclose`.** Set a closed flag that `call` checks before anything is submitted, so a call
  after `aclose` raises a non-retryable `ProviderError`, `the leaf is closed`, and runs
  nothing. Drop a model the leaf loaded, never one it was given, and shut the worker down
  without waiting for a pass still running. Closing twice is harmless.
- **The snapshot is the pinned revision.** In `load()`, resolve the pinned revision to the
  commit it names, then load everything, the weights, the tokenizer and the configuration,
  from that commit's own files: some runtimes pin the weights but read the tokenizer from a
  path that ignores the revision. Report that commit as `snapshot`, so a branch or tag is
  reported as its commit, and `None` with `WARNING_NO_SNAPSHOT` when none was pinned. A leaf
  given a loaded model resolves nothing: its caller gives the commit the model was loaded
  at, and the leaf reports it as given, unresolved, or `None` with the warning when the
  caller gives none.
- **The runtime's versions go in `versions`**, read from the installed packages whether the
  leaf loaded its model or was given one: for a torch model, its runtime package and torch,
  and `python`. A library that only fetches the weights goes in `client_versions`.

What `load()` raises when the runtime or the weights are missing is not settled; `SpanLeaf`
lets its loader's own error propagate, since `load()` is not a call.

A text longer than the model's window is not cut into windows by the leaf. With a known
`max_input_tokens`, Fit's estimate of the state alone exceeds it, and under the default
overflow policy, `Fail`, Fit raises `DoesNotFit` with `state_overflow` true before any call;
the caller splits the text.

### Measure the record

Several of the record's values are the model's, and only a run against it tells you them:

- **The window.** Read the runtime's maximum input length, the encoder's and the runtime's
  own setting, and confirm it by tokenizing a text with the labels as the runtime builds
  them. That is `max_input_tokens`.
- **Silent truncation.** Put a known entity just past the window and see whether it is lost
  with no warning. That is `truncates_silently`, which, true, needs the window known.
- **The labels' share.** Count the same text with 1, 4 and 16 labels and fit what each label
  adds; the estimator counts it.
- **The estimator.** Fit calls the estimator on the event loop's thread while the worker may
  be running the model, and a fast tokenizer is not safe to use from two threads at once;
  a fake model in the tests has no tokenizer at all. So an estimator usually counts by a
  rule of its own, such as words, with a margin that covers its worst undercount measured
  against the real tokenizer over texts like yours: if it falls short by 9% at worst, a
  margin of at least 0.09. Use the real tokenizer only if it is safe across threads and
  your tests can supply one.
- **The question cap.** How many labels one pass carries before the window or the answers
  give out. That is `max_questions`. Where you cannot measure it, declare `None`: Fit then
  warns on each request and never splits one on that account.
- **The wording.** Send each question's instruction as it is, as the label; never reword it.
  Two questions with the same instruction share one label in the model's call, and each
  still gets its own answer by key.
- **Settings.** Any threshold or other setting the leaf passes the runtime is a choice of
  yours: record it in `prompt_sent` and say how you chose it.
- **Offsets.** Check that the runtime's offsets count code points in the original text, and
  convert them if not.

The leaf:

```python
# spanleaf.py
class SpanLeaf:
    """Finds spans of text in this process, one call at a time on a worker of its own.

    Give it ``revision`` to pin the model; give it ``runtime``, a model already loaded, and
    ``revision`` then names the revision that model was loaded at."""

    name = "span-leaf"

    def __init__(
        self,
        *,
        model: str = "toy-spans-1",
        revision: str | None = None,
        runtime: SpanModel | None = None,
    ) -> None:
        self.model = model
        self.capabilities = CAPABILITIES
        self.estimator = WordEstimator()
        self._revision = revision
        self._snapshot = revision if runtime is not None else None
        self._claimed: frozenset[ProvenanceDetail] = frozenset({"model_reported"})
        if self._snapshot is not None:
            self._claimed = frozenset({"model_reported", "snapshot"})
        self._runtime = runtime
        self._injected = runtime is not None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="span-leaf")
        self._closed = False

    async def load(self) -> None:
        """Load the model on the worker: the one step that may fetch or read weights."""
        if self._closed:
            raise ProviderError("the leaf is closed")
        if self._runtime is None:
            loop = asyncio.get_running_loop()
            loaded = await loop.run_in_executor(self._worker, load_toy_model, self._revision)
            self._runtime, self._snapshot = loaded

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        if self._closed:
            raise ProviderError("the leaf is closed")
        runtime = self._runtime
        if runtime is None:
            raise ProviderError("the model is not loaded: await leaf.load() first")
        text = _one_text(wire_request)
        labels = {question.key: _label(question) for question in wire_request.questions}
        sent = tuple(dict.fromkeys(labels.values()))
        if deadline.passed():
            raise ProviderError("the deadline passed before the call", is_retryable=True)
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(self._worker, _run, runtime, text, sent)
        try:
            output, tokens = await asyncio.wait_for(work, max(deadline.remaining(), 0.0))
        except TimeoutError:
            raise ProviderError(
                "the deadline cut the call; the worker finishes the pass before the next",
                is_retryable=True,
            ) from None
        except Exception as error:
            raise ProviderError(f"the model failed: {type(error).__name__}: {error}") from error
        if not isinstance(output, Mapping):
            raise ProviderError(f"the model's output cannot be read: {type(output).__name__}")
        found = cast(Mapping[str, Sequence[Found]], output)
        mapped = _mapped(wire_request.questions, labels, found)
        answers, warnings = validate_answers(
            wire_request.questions, mapped, self.capabilities.rounding
        )
        if self._snapshot is None:
            warnings += (
                Notice(
                    code=WARNING_NO_SNAPSHOT,
                    message="no revision was pinned, so the weights that answered are unknown",
                    field="snapshot",
                ),
            )
        warnings += unknown_options(wire_request.options, frozenset())
        spans: dict[str, JsonValue] = {label: tuple(found[label]) for label in found}
        unreported: dict[ProvenanceDetail, UnreportedReason] = {
            "usage.output_tokens": "not-applicable",
            "usage.passes": "not-reported",
            "request_id": "not-reported",
        }
        if self._snapshot is None:
            unreported["snapshot"] = "not-pinned"
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self.capabilities.calibrated,
            model_reported=self.model,
            snapshot=self._snapshot,
            prompt_sent={"text_chars": len(text), "labels": labels},
            usage=Usage(
                input_tokens=tokens,
                output_tokens=None,
                raw_usage={"input_tokens": tokens, "counted_by": "tokenizer"},
            ),
            raw_body={"spans": spans},
            warnings=warnings,
            # A real leaf names its runtime package and torch beside Python.
            versions={"python": platform.python_version()},
            unreported=unreported,
            claimed=self._claimed,
        )

    async def aclose(self) -> None:
        """Refuse every later call; drop a model this leaf loaded, never one it was given."""
        self._closed = True
        if not self._injected:
            self._runtime = None
        self._worker.shutdown(wait=False)
```

And the helpers it calls. `_one_text` refuses, before anything runs, what the record cannot
say it refuses: an extraction of another value type, and a state that is not one text part.
`_mapped` maps each question's spans, and builds a `Fault` for a question whose spans it
cannot read, its siblings standing. `_answer` reads a single value as the highest-scoring
span, with that span's score as its confidence, whatever order the model returned them in,
and a list as every span, with no confidence:

```python
# spanleaf.py
def _one_text(wire_request: WireRequest) -> str:
    parts = wire_request.state.parts
    if len(parts) != 1 or not isinstance(parts[0], TextPart):
        raise ProviderError("the state must be one text part")
    for question in wire_request.questions:
        if question.kind != "extraction" or question.value_type != "text":
            raise ProviderError(f"question {question.key!r} is not an extraction of text")
    return parts[0].text


def _label(question: Question) -> str:
    if not isinstance(question.instruction, str):
        raise ProviderError(f"question {question.key!r} has a structured instruction")
    return question.instruction


def _mapped(
    questions: Sequence[Question], labels: Mapping[str, str], found: Mapping[str, Sequence[Found]]
) -> dict[str, MappedAnswer]:
    mapped: dict[str, MappedAnswer] = {}
    for question in questions:
        spans = found.get(labels[question.key], ())
        try:
            mapped[question.key] = _answer(question, spans)
        except (IndexError, TypeError, ValueError):
            failure = RuleFailure(rule=Rule.kind_shape, observed=repr(spans)[:80], bound=None)
            mapped[question.key] = Fault(key=question.key, failures=(failure,), mapped=None)
    return mapped


def _answer(question: Question, spans: Sequence[Found]) -> MappedAnswer:
    if question.is_list:
        return ExtractionAnswer(
            value=tuple(text for text, _, _, _ in spans),
            span=tuple(Span(part=0, start=start, end=end) for _, start, end, _ in spans),
        )
    if not spans:
        return ExtractionAnswer(value=None)
    text, start, end, score = max(spans, key=lambda span: span[3])
    return ExtractionAnswer(value=text, confidence=score, span=Span(part=0, start=start, end=end))
```

## Wrap it in a Dispatcher

Build the leaf, wrap it, and await `load()`: `Dispatcher(leaf, retry=..., concurrency=...,
rate=..., estimator=..., overflow=..., reask=...)` is a provider. `async with` closes the leaf
when the block ends, also when loading fails.
`idia.provider(name)` builds only the leaves idia registers, so a leaf of your own is
always wrapped by hand.

```python
# main.py
import asyncio

from idia import Deadline, Dispatcher, ExtractionAnswer, Question, Request, State, TextPart
from spanleaf import SpanLeaf


async def main() -> None:
    leaf = SpanLeaf(revision="rev-1")
    request = Request(
        state=State(parts=(TextPart(text="Grace met Ada in Lyon."),)),
        questions=(
            Question(key="who", kind="extraction", instruction="a person", value_type="text"),
            Question(key="where", kind="extraction", instruction="a city", value_type="text"),
        ),
        deadline=Deadline.after(10),
    )
    async with Dispatcher(leaf) as spans:
        await leaf.load()
        response = await spans.answer(request)
    for key, answer in response.answers.items():
        assert isinstance(answer, ExtractionAnswer)
        print(key, answer.value, answer.confidence, answer.span)
    outcome = response.outcomes[0]
    assert outcome.usage is not None
    print(outcome.snapshot, outcome.usage.input_tokens)


asyncio.run(main())
```

`python main.py` prints:

```text
who Grace 0.75 Span(part=0, start=0, end=5)
where Lyon 0.7 Span(part=0, start=17, end=21)
rev-1 9
```

## Test it with the conformance suite

`idia.conformance` holds the checks every leaf passes: `leaf_checks(make_harness)` against
the leaf alone and `provider_checks(make_provider_harness)` against it wrapped. Each takes a
factory of a harness you write, the part that knows your leaf's transport. The suite speaks
to it in exchanges, what the wire or the model does on one call: `Success(kind)`,
`Failure(status=None)`, `Stall()`, `Error(...)`, `Unreadable(...)`, `Perturbed(...)`,
`RefusalOnWire(kind)` and `CandidateShape(kind)`; the
[conformance spec](../spec/conformance.md#the-harness) gives each.

Your harness turns each exchange into what your transport does. For an in-process leaf
that is a model you inject in place of the real one, through the interface the leaf calls, here
`SpanModel`, playing stored outputs. A real runtime's token count comes from the loaded
model, so the fake supplies it too, stored beside each output:

- **`Success(kind)`** plays the output the real model gave for `wire_request(kind)`,
  recorded by a run against it, and only for that request: a call whose inputs differ from
  the stored ones must fail, so the check fails. In process, the model raising is how a call
  fails, and the leaf turns that into a non-retryable `ProviderError`. `Success()` with no
  kind answers whatever the provider harness's `request()` sends, matched by its inputs:
  the provider half also splits that request one question to a call, so store an output
  for each one-question split too.
- **`Failure()`** with no status is a failure that is not retryable; for a model, the
  runtime raising.
- **`Stall()`** never answers in time, so the call must end by its deadline.
- An exchange your transport cannot make honestly, such as a status from a model that has
  none, makes your harness raise `CheckNotApplicable` from `make`. Never fake a status, a
  body or an SDK the leaf does not have.

A check whose drivers your harness cannot make is not applicable, or passes with a note
saying what it did not run. Assert the set of checks that are not applicable exactly, and
the notes of the checks that pass with one, so a check that stops running is noticed.

```python
# stored.py
"""The model's stored outputs: written by a run against it, read back by the tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from spanleaf import Found, SpanModel


@dataclass(frozen=True)
class Pass:
    """What one pass of the model returned, and its count of the tokens it encoded."""

    spans: Mapping[str, Sequence[Found]]
    input_tokens: int


Outputs = dict[tuple[str, tuple[str, ...]], Pass]
"""The model's pass for each text and set of labels."""


def record_outputs(
    model: SpanModel,
    asked: Iterable[tuple[str, tuple[str, ...]]],
    path: Path,
    recorded_with: Mapping[str, str | None],
) -> None:
    """Run the model on each text and set of labels, and write what it returned to path,
    with the versions and the revision it ran under."""
    cases = [
        {
            "text": text,
            "labels": list(labels),
            "spans": model.extract(text, labels),
            "input_tokens": model.count_tokens(text, labels),
        }
        for text, labels in asked
    ]
    document = {"recorded_with": dict(recorded_with), "cases": cases}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_outputs(path: Path) -> Outputs:
    """The passes record_outputs wrote, each span a tuple again."""
    document = json.loads(path.read_text(encoding="utf-8"))
    outputs: Outputs = {}
    for case in document["cases"]:
        spans = {
            label: tuple((s[0], s[1], s[2], s[3]) for s in found)
            for label, found in case["spans"].items()
        }
        outputs[(case["text"], tuple(case["labels"]))] = Pass(spans, case["input_tokens"])
    return outputs
```

The tests, the stored outputs, and the model that plays them:

```python
# test_spanleaf.py
"""SpanLeaf against idia's conformance suite, over stored model outputs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from idia import (
    Answer,
    Capabilities,
    Deadline,
    Dispatcher,
    ExtractionAnswer,
    Fault,
    Kind,
    Leaf,
    Provider,
    ProviderError,
    Question,
    Request,
    RetryPolicy,
    Span,
    State,
    TextPart,
    WireRequest,
    WireResponse,
)
from idia.conformance import (
    Check,
    CheckNotApplicable,
    Exchange,
    Failure,
    Stall,
    Success,
    leaf_checks,
    provider_checks,
    run_checks,
)
from idia.providers.recorded import Recorded
from idia.providers.scripted import Scripted
from spanleaf import CAPABILITIES, Found, SpanLeaf, ToyModel
from stored import Outputs, Pass, load_outputs, record_outputs

TEXT = "Ada flew from Paris to Oslo."
STATE = State(parts=(TextPart(text=TEXT),))
WHO = Question(key="who", kind="extraction", instruction="a person", value_type="text")
CITIES = Question(
    key="cities", kind="extraction", instruction="a city", value_type="text", is_list=True
)

ADA = ("Ada", 0, 3, 0.65)
CITIES_FOUND = (("Paris", 14, 19, 0.75), ("Oslo", 23, 27, 0.7))
STORED: Outputs = {
    (TEXT, ("a person", "a city")): Pass({"a person": (ADA,), "a city": CITIES_FOUND}, 10),
    (TEXT, ("a person",)): Pass({"a person": (ADA,)}, 8),
    (TEXT, ("a city",)): Pass({"a city": CITIES_FOUND}, 8),
}
"""What the model returned for each text and set of labels: the request the leaf half sends,
and each one-question split of it, which the provider half sends. A real leaf keeps these
in the file a run against the model wrote, and reads them with load_outputs."""

REQUESTED = ("a person", "a city")
"""The labels of the wire request every Success(kind) answers."""

EXPECTED: Mapping[str, Answer] = {
    "who": ExtractionAnswer(value="Ada", confidence=0.65, span=Span(part=0, start=0, end=3)),
    "cities": ExtractionAnswer(
        value=("Paris", "Oslo"),
        span=(Span(part=0, start=14, end=19), Span(part=0, start=23, end=27)),
    ),
}
"""The answers SpanLeaf maps those outputs to."""


class Playback:
    """The model the harness injects in place of the real one: it plays the exchanges, one per
    pass, and counts tokens as the stored pass says."""

    def __init__(self, exchanges: Sequence[Exchange]) -> None:
        for exchange in exchanges:
            if isinstance(exchange, Failure) and exchange.status is not None:
                raise CheckNotApplicable("the model has no statuses")
            if not isinstance(exchange, Success | Failure | Stall):
                raise CheckNotApplicable(f"a model in process cannot make {exchange}")
        self._exchanges = list(exchanges)
        self.calls = 0

    def extract(self, text: str, labels: Sequence[str]) -> Mapping[str, Sequence[Found]]:
        self.calls += 1
        if not self._exchanges:
            raise RuntimeError(f"the exchanges ran out at call {self.calls}")
        exchange = self._exchanges.pop(0)
        if isinstance(exchange, Failure):
            raise RuntimeError("the model failed")
        if not isinstance(exchange, Success):
            time.sleep(1.0)
            raise RuntimeError("stalled: no answer")
        if exchange.kind is not None and (text, tuple(labels)) != (TEXT, REQUESTED):
            raise RuntimeError("harness: the leaf sent something other than the recorded request")
        return self._stored(text, labels).spans

    def count_tokens(self, text: str, labels: Sequence[str]) -> int:
        return self._stored(text, labels).input_tokens

    def _stored(self, text: str, labels: Sequence[str]) -> Pass:
        stored = STORED.get((text, tuple(labels)))
        if stored is None:
            raise RuntimeError("harness: no stored output for these inputs")
        return stored
```

The leaf half's harness. A leaf with opt-in features is built with the features `make` is
given, and its suite is run once per configuration you ship; `SpanLeaf` has none:

```python
# test_spanleaf.py
class SpanLeafHarness:
    """The leaf half's harness: SpanLeaf over a Playback model."""

    capabilities = CAPABILITIES
    prompt_hidden = False

    def __init__(self) -> None:
        self._model = Playback([])

    def make(self, exchanges: Sequence[Exchange], features: frozenset[str] | None = None) -> Leaf:
        self._model = Playback(exchanges)
        return SpanLeaf(revision="rev-1", runtime=self._model)

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        return CAPABILITIES

    def wire_request(self, kind: Kind) -> WireRequest:
        return WireRequest(state=STATE, questions=(WHO, CITIES))

    def expected(self, kind: Kind) -> Mapping[str, Answer]:
        return EXPECTED

    def requests_sent(self) -> int:
        return self._model.calls

    def sdk_retries(self) -> int:
        raise CheckNotApplicable("the leaf is built on no SDK")

    async def client_closed(self, *, injected: bool) -> bool:
        raise CheckNotApplicable("the leaf has no HTTP client")
```

The provider half's harness wraps the leaf in a dispatcher with no retry delay, and keeps
each wire request the leaf received with its deadline:

```python
# test_spanleaf.py
class LoggedSpanLeaf(SpanLeaf):
    """SpanLeaf keeping each wire request it receives, with its deadline."""

    def __init__(self, *, revision: str | None, runtime: Playback) -> None:
        super().__init__(revision=revision, runtime=runtime)
        self.log: list[tuple[WireRequest, Deadline | None]] = []

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        self.log.append((wire_request, deadline))
        return await super().call(wire_request, deadline)


class SpanProviderHarness:
    """The provider half's harness: SpanLeaf wrapped in a Dispatcher."""

    shape: Literal["leaf"] = "leaf"
    members = 1
    expected_capabilities = CAPABILITIES

    def __init__(self) -> None:
        self._model = Playback([])
        self._leaf = LoggedSpanLeaf(revision="rev-1", runtime=self._model)

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        self._model = Playback(exchanges)
        self._leaf = LoggedSpanLeaf(revision="rev-1", runtime=self._model)
        return self._leaf

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        self.leaf(exchanges[0])
        self._leaf.capabilities = replace(CAPABILITIES, **(overrides or {}))
        return Dispatcher(self._leaf, retry=RetryPolicy.without_delay())

    def request(self) -> Request:
        return Request(state=STATE, questions=(WHO, CITIES), deadline=Deadline.after(30.0))

    def requests_sent(self) -> int:
        return self._model.calls

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
        return (tuple(self._leaf.log),)

    def member_names(self) -> tuple[tuple[str, str], ...]:
        return ((self._leaf.name, self._leaf.model),)
```

Run each check as its own test, one at a time: the checks must never run concurrently,
since one of them patches idia's validation while it runs, so do not spread them over
processes or threads. A check that is not applicable is skipped with its reason:

```python
# test_spanleaf.py
def params(checks: Sequence[Check]) -> list[object]:
    return [pytest.param(check, id=check.id) for check in checks]


def run(check: Check) -> None:
    try:
        asyncio.run(check.run())
    except CheckNotApplicable as not_applicable:
        pytest.skip(not_applicable.reason)


@pytest.mark.parametrize("check", params(leaf_checks(SpanLeafHarness)))
def test_leaf_half(check: Check) -> None:
    run(check)


@pytest.mark.parametrize("check", params(provider_checks(SpanProviderHarness)))
def test_provider_half(check: Check) -> None:
    run(check)
```

Then assert which checks did not apply, and why that is right for your leaf, and every note
a passing check gives, so a check that stops running part of itself is noticed too. These
tests pin idia's own wording: when an upgrade changes a note, reread that check before you
update the test. For `SpanLeaf`: no SDK (L9), no status (L12, P9, P20, P21), no body to read
(L13, P12), no HTTP client (L15), no answer key a model can drop or add, since the leaf maps
every label it asked (L19), and no candidate shape (L24), no kind whose probabilities sum
(L18), no `noul` (L20), no opt-in feature (P6), no image (P23), and not a composite (P14 to
P18):

```python
# test_spanleaf.py
def not_applicable(checks: Sequence[Check]) -> set[str]:
    verdicts = asyncio.run(run_checks(checks))
    return {verdict.id for verdict in verdicts if verdict.result == "not_applicable"}


def test_the_leaf_half_skips_exactly_what_a_model_in_process_cannot_make() -> None:
    expected = {"L9", "L12", "L13", "L15", "L18", "L19", "L20", "L24"}
    assert not_applicable(leaf_checks(SpanLeafHarness)) == expected


def test_the_provider_half_skips_exactly_what_a_wrapped_leaf_cannot_reach() -> None:
    expected = {"P6", "P9", "P12", "P14", "P15", "P16", "P17", "P18", "P20", "P21", "P23"}
    assert not_applicable(provider_checks(SpanProviderHarness)) == expected


def notes(checks: Sequence[Check]) -> dict[str, str]:
    verdicts = asyncio.run(run_checks(checks))
    return {v.id: v.reason for v in verdicts if v.result == "pass" and v.reason is not None}


READ = (
    "read spanleaf, the module that defines SpanLeaf; "
    "code the leaf keeps in other modules was not read"
)
CANNOT = "a model in process cannot make"


def test_the_leaf_half_notes_what_it_did_not_run() -> None:
    assert notes(leaf_checks(SpanLeafHarness)) == {
        "L4": READ,
        "L8": "the retryable half ran over a stall, not a 429: the model has no statuses",
        "L10": READ,
        "L11": "not driven: Failure(status=429, retry_after=None, request_id=None): the model "
        f"has no statuses; Error(which='connection'): {CANNOT} Error(which='connection'); "
        f"Error(which='timeout'): {CANNOT} Error(which='timeout'); "
        f"Unreadable(which='not_json'): {CANNOT} Unreadable(which='not_json'); "
        f"Unreadable(which='schema'): {CANNOT} Unreadable(which='schema'); "
        f"Unreadable(which='no_answers'): {CANNOT} Unreadable(which='no_answers')",
        "L21": READ,
        "L25": "no expected extraction has a null value and a guess, so only that the leaf "
        "fills no guess was checked, not that a guess is kept and never moved into value",
    }


def test_the_provider_half_notes_what_it_did_not_run() -> None:
    assert notes(provider_checks(SpanProviderHarness)) == {
        "P19": "a 503 then an answer was not recorded: the model has no statuses",
    }
```

L11's note says the suite never gives the leaf an output it cannot read: a model in
process returns Python objects, not a body. So test that path yourself, with a model that
returns one:

```python
# test_spanleaf.py
class Garbled:
    """A model whose spans for one label are not spans, or whose output is not a mapping."""

    def __init__(self, output: object) -> None:
        self._output = output

    def extract(self, text: str, labels: Sequence[str]) -> Mapping[str, Sequence[Found]]:
        return cast(Mapping[str, Sequence[Found]], self._output)

    def count_tokens(self, text: str, labels: Sequence[str]) -> int:
        return 10


def ask_garbled(output: object) -> Mapping[str, Answer]:
    async def ask() -> Mapping[str, Answer]:
        leaf = SpanLeaf(revision="rev-1", runtime=Garbled(output))
        try:
            wire_request = WireRequest(state=STATE, questions=(WHO, CITIES))
            return (await leaf.call(wire_request, Deadline.after(5))).answers
        finally:
            await leaf.aclose()

    return asyncio.run(ask())


def test_spans_the_leaf_cannot_read_are_a_fault_and_their_sibling_stands() -> None:
    answers = ask_garbled({"a person": [("Ada", 0)], "a city": CITIES_FOUND})
    assert isinstance(answers["who"], Fault)
    assert answers["cities"] == EXPECTED["cities"]


def test_an_output_that_is_not_a_mapping_fails_the_call_for_good() -> None:
    with pytest.raises(ProviderError, match="cannot be read") as raised:
        ask_garbled(["Ada"])
    assert not raised.value.is_retryable
```

## Let code that uses the leaf test without it

Code that consumes your leaf can be tested with no model at all. `Scripted`, in
`idia.providers.scripted`, answers from a script under your leaf's name, model and record:

```python
# test_spanleaf.py
def test_code_that_reads_the_cities_with_a_scripted_leaf() -> None:
    cities = ExtractionAnswer(value=("Oslo",), span=(Span(part=0, start=23, end=27),))
    leaf = Scripted(
        capabilities=CAPABILITIES,
        script=[{"cities": cities}],
        name="span-leaf",
        model="toy-spans-1",
    )
    request = Request(state=STATE, questions=(CITIES,), deadline=Deadline.after(5))
    response = asyncio.run(Dispatcher(leaf).answer(request))
    assert response.answers["cities"] == cities
```

`Recorded`, in `idia.providers.recorded`, records the calls of a real leaf to a log once and
replays them after, with no model loaded. A consumer replays a log with
`Recorded(None, log, "none", capabilities=...)`; keep one log per leaf configuration, since a
log does not know the revision it was made under. With no leaf, the replay plans with its
own character count, so a request near a limit may split differently and miss the log;
when the plan matters, replay with the real leaf, not loaded, in mode `none`, which never
calls it:

```python
# test_spanleaf.py
def test_the_stored_outputs_round_trip_through_their_file(tmp_path: Path) -> None:
    path = tmp_path / "outputs.json"
    record_outputs(ToyModel(), STORED, path, {"revision": "rev-1"})
    assert load_outputs(path) == STORED


def test_a_recording_replays_with_no_model(tmp_path: Path) -> None:
    log = tmp_path / "span-leaf.json"
    request = Request(state=STATE, questions=(WHO, CITIES), deadline=Deadline.after(5))

    async def ask(leaf: Leaf) -> Mapping[str, Answer]:
        async with Dispatcher(leaf, retry=RetryPolicy.without_delay()) as provider:
            return (await provider.answer(request)).answers

    live = SpanLeaf(revision="rev-1", runtime=ToyModel())
    recorded = asyncio.run(ask(Recorded(live, log, "once")))
    replayed = asyncio.run(ask(Recorded(None, log, "none", capabilities=CAPABILITIES)))
    assert replayed == recorded == EXPECTED
```

## Gate a run against the real model

The tests above never load the real model. A test that does, to check the stored outputs
against it or to record them again, runs only when asked: behind a marker, and an
environment variable set by hand, which also sets the model hub offline before any test
module is imported: the hub reads that setting when it is first imported, so set it before
anything imports the hub. The run then reads only the files already cached and never
downloads.

Inject the fake at the runtime's own call, as `Playback` stands in for `SpanModel`, the object whose methods the leaf calls, so
that the leaf's reading of the runtime's output stays under test. Key each stored output on
every argument the leaf passes the runtime: here the text and the labels; for a real span
model, also any threshold or mode it is given. Store the outputs with the revision and the
runtime's versions. A real model's scores can move a little between builds of its runtime,
so compare the spans exactly and the scores within a tolerance, and record again when they
move further:

```python
# conftest.py
import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_model: loads the real model")
    if os.environ.get("SPANLEAF_REAL_MODEL") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("real_model") and os.environ.get("SPANLEAF_REAL_MODEL") != "1":
        pytest.skip("loads the real model; set SPANLEAF_REAL_MODEL=1 to run")
```

```python
# test_real_model.py
import platform
from pathlib import Path

import pytest

from spanleaf import load_toy_model
from stored import load_outputs, record_outputs
from test_spanleaf import STORED


@pytest.mark.real_model
def test_the_real_model_still_gives_the_stored_outputs(tmp_path: Path) -> None:
    path = tmp_path / "outputs.json"
    model, commit = load_toy_model("rev-1")
    versions = {"revision": commit, "python": platform.python_version()}
    record_outputs(model, STORED, path, versions)
    for asked, now in load_outputs(path).items():
        then = STORED[asked]
        assert now.input_tokens == then.input_tokens
        assert now.spans.keys() == then.spans.keys()
        for label, spans in now.spans.items():
            assert [span[:3] for span in spans] == [span[:3] for span in then.spans[label]]
            for (*_, score), (*_, was) in zip(spans, then.spans[label], strict=True):
                assert abs(score - was) <= 0.01
```

Run the tests with `pytest`; the real-model test runs with `SPANLEAF_REAL_MODEL=1 pytest -m
real_model`. To record the stored outputs again, write them with `record_outputs` to the
file your tests read, and commit it.

## Where next

- [The conformance suite](../spec/conformance.md): every check, the harness protocols and
  the exchanges.
- [Vocabulary](../spec/vocabulary.md): the shapes, the answers, validation and the errors.
- [Fit](../spec/fit.md): how a request is checked against the record and split.
- [Dispatch](../spec/dispatch.md): the deadline, retries and the outcomes a leaf's errors
  become.
