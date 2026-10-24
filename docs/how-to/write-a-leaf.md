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

The example is four files in one directory, each code block headed by the file it belongs
to: `spanleaf.py`, the leaf; `main.py`, which uses it; `test_spanleaf.py`, its tests; and
`conftest.py` with `test_real_model.py`, the gate for a run against the real model. You need
Python 3.11 or later, idia, and pytest for the tests.

## Use public names only

Import only what `idia.__all__` lists and what the `__all__` of the modules the
[contributing notes](../../CONTRIBUTING.md#ground-rules-for-changes) list: among them
`idia.conformance`, `idia.providers.scripted` and `idia.providers.recorded`. Every other name
may change without notice: a name starting with an underscore, a module starting with one,
and `idia.providers` itself.

## What a leaf is

`Leaf` is a protocol, satisfied by shape: nothing is subclassed. A leaf has five members:

- `name`, a short name for the leaf, and `model`, the model it asks. Every outcome names
  both.
- `capabilities`, the record of what it takes and returns ([below](#state-every-field-of-the-record)).
- `estimator`, which counts the tokens of a request ([below](#count-what-the-model-reads)).
- `async call(wire_request, deadline) -> WireResponse`, which answers one wire request.

A data member may be a plain attribute, a property or a dataclass field. `aclose()` is not
part of `Leaf`, but a leaf that holds anything to release has one, and the dispatcher calls
it when it is closed.

Four rules hold for every leaf:

- **One call sends one wire request**, or runs the model once. A leaf never retries: retries,
  waits and attempt counts are the dispatcher's.
- **It raises `ProviderError` and nothing else.**
- **It returns what the model gave.** Nothing is renormalized, rounded, corrected or
  computed: no confidence, no probability of no value, no `1 - p`.
- **It is not a provider until wrapped.** `Dispatcher(leaf)` makes it one.

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
import importlib.metadata
import platform
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from idia import (
    UNBOUNDED,
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    ExtractionAnswer,
    JsonValue,
    MappedAnswer,
    Notice,
    ProviderError,
    Question,
    Rounding,
    Span,
    State,
    TextPart,
    TokenEstimate,
    Usage,
    WireRequest,
    WireResponse,
    unknown_options,
    validate_answers,
)

Found = tuple[str, int, int, float]
"""One span the model found: its text, its start and end offsets, and its score."""

Extractor = Callable[[str, Sequence[str]], Mapping[str, Sequence[Found]]]
"""The model: a text and the labels to look for, to the spans found under each label."""


def toy_model(text: str, labels: Sequence[str]) -> dict[str, list[Found]]:
    """A stand-in for a real span model: it finds the words it knows, scored by length."""
    known = {"a city": ("Paris", "Lyon", "Oslo"), "a person": ("Ada", "Grace")}
    found: dict[str, list[Found]] = {}
    for label in labels:
        spans: list[Found] = []
        for word in known.get(label, ()):
            start = text.find(word)
            if start >= 0:
                spans.append((word, start, start + len(word), round(0.5 + len(word) / 20, 2)))
        found[label] = sorted(spans, key=lambda span: -span[3])
    return found


def load_toy_model(revision: str | None) -> Extractor:
    """Load the model from local files only. A real loader reads its cached weights here,
    at the revision given."""
    return toy_model
```

## State every field of the record

A `Capabilities` record says what the leaf takes and returns. idia's preflight, Fit, checks
every request against it before any call and refuses one that does not fit with
`DoesNotFit`, so a request that passes must be one your leaf takes. Every field must be
stated; none has a default.

A limit is a non-negative `int`, `UNBOUNDED` when nothing caps it, or `None` when you do not
know it. Fit checks a number, skips `UNBOUNDED` silently, and skips `None` with an
`unchecked-limit` warning on each wire request it bears on. Do not guess a limit: measure it
against the real model, or leave it `None`.

What Fit does with each field:

- `kinds` and `modalities`: a question of a kind not listed, or a state part (text, JSON or
  image) of a modality not listed, does not fit.
- `image_carriers`, `max_images`, `max_images_per_question`, `max_image_bytes`,
  `max_total_image_bytes`, `min_image_side` and `image_media_types`: the images of a state.
  A leaf without the image modality has no carrier and 0 for each of the others, and an
  empty set of media types.
- `max_input_tokens`: each wire request's token estimate. `max_question_tokens`: the state
  with each one question. `max_body_bytes`: Fit's measure of the body, the state's text and
  JSON as UTF-8 and each question's key, instruction and labels. A model with no body to
  cap has `UNBOUNDED`.
- `truncates_silently`: whether the model cuts an over-long input without saying so. `True`
  needs a known `max_input_tokens`, since only that lets Fit stop the state first, and the
  constructor refuses it with `None`. `None` means undocumented.
- `recounts_state_per_question`: the model reads the state once per question, so a wire
  request of `n` questions counts the state `n` times.
- `max_questions` and `max_total_options`: how many questions, and how many labels and
  levels in all, one wire request may carry. A request over them is split into several wire
  requests, each repeating the state.
- `max_labels`, `max_levels` and `min_levels`: the labels of a `choice` and the levels of a
  `score`. A leaf that answers neither has 0 for each.
- `key_patterns`: patterns every question key must match in full, for a model that
  restricts its ids.
- `carries_noul_descriptions` and `carries_structured_instructions`: whether the leaf sends
  a `noul`'s yes and no descriptions, and an instruction that is a JSON object or array.
  When false, Fit refuses a request that uses it.
- `returns_refusal`, `calibrated` and `snapshot_pinnable`: whether the model can decline a
  question, whether its probabilities are calibrated, and whether a response can name the
  exact model that answered. The conformance suite checks each against the responses.
- `confidence`: for each kind whose answers carry a confidence, what that confidence is,
  `"undocumented"` where the model publishes no formula. A kind not listed carries none.
- `rounding`: the decimals the model rounds its probabilities and scores to, `None` for
  unrounded. Validation takes its tolerances from it.
- `opt_in_features`, `enabled_features`, `feature_kinds` and `feature_modalities`: the
  features a caller may turn on, those on, and the kinds or modalities each adds. A leaf
  with none has empty sets and empty maps.

`SpanLeaf`'s record. Its window of 512 tokens stands in for the measured window of a real
model:

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
`ceil(tokens * (1 + margin))` against the limits. `questions` may be empty, meaning the state
alone, which is how Fit checks the state for overflow. A caller may give the dispatcher
another estimator, which then replaces yours.

Count everything the model reads, not only the state: the labels, the instructions and any
schema text you send with them share the model's window. The margin covers how far your
count may fall short of the model's own.

```python
# spanleaf.py
class WordEstimator:
    """Counts a word as a token, labels included: they share the model's window."""

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        words = sum(len(part.text.split()) for part in state.parts if isinstance(part, TextPart))
        words += sum(len(str(question.instruction).split()) for question in questions)
        return TokenEstimate(tokens=words, margin=0.2)
```

## Map the output, then validate it

Map what the model returned to the kind's answer shape and nothing more: a `NoulAnswer`, a
`LabelAnswer`, an `ExtractionAnswer` or a `BoxAnswer`, each keyed by its question's key; a
`Refusal` where the model declined the question. Then call
`validate_answers(questions, mapped, capabilities.rounding)`. It returns the answers and the
warnings: an answer that breaks a rule becomes a `Fault` for that question alone, a missing
key a `Fault`, and a key with no question is dropped with a warning.

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

| What happened | `is_retryable` |
|---|---|
| The deadline passed before the call, or cut it | yes |
| A connection error or a transport timeout | yes |
| A wire status 408, 409, 429 or 5xx | yes |
| Any other wire status that is not a success | no |
| An output the leaf cannot read | no |
| A question or state the leaf cannot send, refused before anything is sent | no |
| The model failed while running | no |
| The model is not loaded, or the leaf is closed | no |

Set `status_code` only to a status the wire returned, and `request_id` when you have one.
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
- When the wait runs out, raise a retryable `ProviderError`. The dispatcher decides whether
  to retry; once the deadline has passed it ends the wire request with `DeadlineExceeded`.

## Warn, and record where the answers came from

A `WireResponse` carries the answers and the provenance only the leaf knows:

- `elicitation`: `"passthrough"`, the model's own answers as given.
- `calibrated`: the record's `calibrated`.
- `model_reported`: the model the wire, or the runtime, says answered.
- `snapshot`: the exact model version that answered, only when the record's
  `snapshot_pinnable` holds and the caller pinned one; otherwise `None`, with a
  `WARNING_NO_SNAPSHOT` warning.
- `prompt_sent`: what was sent, as JSON values; `None` only where the wire hides it.
- `usage`: `Usage(input_tokens, output_tokens, raw_usage=...)`. A model that returns values
  and no text reports 0 output tokens. Put the figures and how they were counted in
  `raw_usage`.
- `request_id`: the wire's id for the call. A model with none gets one from the leaf, such as
  `<name>-<n>`, numbered as each call starts.
- `raw_body`: what the model returned, whole. The dispatcher drops it, but a recording keeps
  it, so put the runtime's versions here.
- `warnings`: `Notice` values. Put the warnings `validate_answers` returned first, then your
  own, such as `WARNING_NO_SNAPSHOT` or `WARNING_PROVIDER_TRUNCATED` when the model reports
  that it cut the input, and last `unknown_options(wire_request.options, known)`, which
  warns once naming every request option your leaf does not know. An option you do not know
  changes nothing else.

## A model in your own process

A model that runs in your process, on a local runtime such as torch, needs six more rules.

- **One worker of its own.** Loading and every forward pass block, so run them on a
  `ThreadPoolExecutor(max_workers=1)` the leaf owns, never on the event loop. Calls to one
  leaf then run one at a time, and a `ConcurrencyLimit` above 1 on its dispatcher gives no
  parallelism.
- **Loading is a step of its own.** Give the leaf an `async load()`, the one place it may
  fetch or read weights, and have the caller await it before the first call. Nothing inside
  `call` touches the network: load from local files, with any hub set offline. A warm-up,
  one small forward pass at the end of `load()`, keeps its cost out of the first call. A
  call before the model is loaded raises a non-retryable `ProviderError` and runs nothing. A
  leaf given a loaded model needs no load.
- **A deadline cut leaves the worker busy.** Wait for the worker with
  `asyncio.wait_for(future, max(deadline.remaining(), 0))`, and turn a timeout into a
  retryable `ProviderError`. The forward pass cannot be stopped: it runs to its end, and the
  next call, a retry included, waits behind it inside its own deadline. A call still queued
  behind it when its own deadline passes never runs.
- **`aclose`.** Set a closed flag that `call` checks before anything is submitted, so a call
  after `aclose` raises a non-retryable `ProviderError`, `the leaf is closed`, and runs
  nothing. Drop a model the leaf loaded, never one it was given, and shut the worker down
  without waiting for a pass still running. Closing twice is harmless.
- **The snapshot is the pinned revision.** Report as `snapshot` the revision of the weights
  the caller pinned, and `None` with `WARNING_NO_SNAPSHOT` when none was pinned.
- **The runtime's versions go in the raw body**, so a recording keeps what answered.

A text longer than the model's window is not cut into windows by the leaf: with a known
`max_input_tokens`, Fit stops it before any call, and the caller splits it.

The leaf:

```python
# spanleaf.py
class SpanLeaf:
    """Finds spans of text in this process, one call at a time on a worker of its own."""

    name = "span-leaf"

    def __init__(
        self,
        *,
        model: str = "toy-spans-1",
        revision: str | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.model = model
        self.capabilities = CAPABILITIES
        self.estimator = WordEstimator()
        self._revision = revision
        self._extractor = extractor
        self._injected = extractor is not None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="span-leaf")
        self._closed = False
        self._calls = 0

    async def load(self) -> None:
        """Load the model on the worker: the one step that may fetch or read weights."""
        if self._closed:
            raise ProviderError("the leaf is closed")
        if self._extractor is None:
            loop = asyncio.get_running_loop()
            self._extractor = await loop.run_in_executor(
                self._worker, load_toy_model, self._revision
            )

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        if self._closed:
            raise ProviderError("the leaf is closed")
        extractor = self._extractor
        if extractor is None:
            raise ProviderError("the model is not loaded: await leaf.load() first")
        text = _one_text(wire_request)
        labels = {question.key: _label(question) for question in wire_request.questions}
        if deadline.passed():
            raise ProviderError("the deadline passed before the call", is_retryable=True)
        self._calls += 1
        request_id = f"{self.name}-{self._calls}"
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(self._worker, extractor, text, tuple(labels.values()))
        try:
            found = await asyncio.wait_for(work, max(deadline.remaining(), 0.0))
        except TimeoutError:
            raise ProviderError(
                "the deadline cut the call; the worker finishes the pass before the next",
                is_retryable=True,
                request_id=request_id,
            ) from None
        except Exception as error:
            raise ProviderError(
                f"the model failed: {type(error).__name__}: {error}", request_id=request_id
            ) from error
        mapped = {
            question.key: _answer(question, found.get(labels[question.key], ()))
            for question in wire_request.questions
        }
        answers, warnings = validate_answers(
            wire_request.questions, mapped, self.capabilities.rounding
        )
        if self._revision is None:
            warnings += (
                Notice(
                    code=WARNING_NO_SNAPSHOT,
                    message="no revision was pinned, so the weights that answered are unknown",
                    field="snapshot",
                ),
            )
        warnings += unknown_options(wire_request.options, frozenset())
        tokens = self.estimator.estimate(wire_request.state, wire_request.questions).tokens
        spans: dict[str, JsonValue] = {label: tuple(found[label]) for label in found}
        versions: dict[str, JsonValue] = {
            "python": platform.python_version(),
            "idia": _version("idia"),
        }
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self.capabilities.calibrated,
            model_reported=self.model,
            snapshot=self._revision,
            prompt_sent={"text_chars": len(text), "labels": labels},
            usage=Usage(
                input_tokens=tokens,
                output_tokens=0,
                raw_usage={"input_tokens": tokens, "output_tokens": 0, "counted_by": "estimator"},
            ),
            request_id=request_id,
            raw_body={"spans": spans, "versions": versions},
            warnings=warnings,
        )

    async def aclose(self) -> None:
        """Refuse every later call; drop a model this leaf loaded, never one it was given."""
        self._closed = True
        if not self._injected:
            self._extractor = None
        self._worker.shutdown(wait=False)
```

And the helpers it calls. `_one_text` refuses, before anything runs, what the record cannot
say it refuses: an extraction of another value type, and a state that is not one text part.
`_answer` reads a single value as the top-scoring span, with that span's score as its
confidence, and a list as every span, with no confidence:

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


def _answer(question: Question, spans: Sequence[Found]) -> MappedAnswer:
    if question.is_list:
        return ExtractionAnswer(
            value=tuple(text for text, _, _, _ in spans),
            span=tuple(Span(part=0, start=start, end=end) for _, start, end, _ in spans),
        )
    if not spans:
        return ExtractionAnswer(value=None)
    text, start, end, score = spans[0]
    return ExtractionAnswer(value=text, confidence=score, span=Span(part=0, start=start, end=end))


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
```

## Wrap it in a Dispatcher

Build the leaf, await `load()`, and wrap it: `Dispatcher(leaf, retry=..., concurrency=...,
rate=...)` is a provider. `async with` closes the leaf when the block ends.
`idia.provider(name)` builds only the leaves idia registers, so a leaf of your own is
always wrapped by hand.

```python
# main.py
import asyncio

from idia import Deadline, Dispatcher, ExtractionAnswer, Question, Request, State, TextPart
from spanleaf import SpanLeaf


async def main() -> None:
    leaf = SpanLeaf(revision="rev-1")
    await leaf.load()
    request = Request(
        state=State(parts=(TextPart(text="Grace met Ada in Lyon."),)),
        questions=(
            Question(key="who", kind="extraction", instruction="a person", value_type="text"),
            Question(key="where", kind="extraction", instruction="a city", value_type="text"),
        ),
        deadline=Deadline.after(10),
    )
    async with Dispatcher(leaf) as spans:
        response = await spans.answer(request)
    for key, answer in response.answers.items():
        assert isinstance(answer, ExtractionAnswer)
        print(key, answer.value, answer.confidence, answer.span)
    print(response.outcomes[0].snapshot, response.outcomes[0].request_id)


asyncio.run(main())
```

`python main.py` prints:

```text
who Grace 0.75 Span(part=0, start=0, end=5)
where Lyon 0.7 Span(part=0, start=17, end=21)
rev-1 span-leaf-1
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
that is a model you inject in place of the real one, playing stored outputs:

- **`Success(kind)`** plays the output the real model gave for `wire_request(kind)`,
  recorded by a run against it, and only for that request: a call whose inputs differ from
  the stored ones must fail, so the check fails.
- **`Failure()`** with no status is a failure that is not retryable; for a model, the
  runtime raising.
- **`Stall()`** never answers in time, so the call must end by its deadline.
- An exchange your transport cannot make honestly, such as a status from a model that has
  none, makes your harness raise `CheckNotApplicable` from `make`. Never fake a status, a
  body or an SDK the leaf does not have.

A check whose drivers your harness cannot make is not applicable, or passes with a note
saying what it did not run. Assert the set of checks that are not applicable exactly, so a
check that stops running is noticed.

The stored outputs, and the model that plays them:

```python
# test_spanleaf.py
"""SpanLeaf against idia's conformance suite, over stored model outputs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from idia import (
    Answer,
    Capabilities,
    Deadline,
    Dispatcher,
    ExtractionAnswer,
    Kind,
    Leaf,
    Provider,
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
from spanleaf import CAPABILITIES, Found, SpanLeaf, toy_model

TEXT = "Ada flew from Paris to Oslo."
STATE = State(parts=(TextPart(text=TEXT),))
WHO = Question(key="who", kind="extraction", instruction="a person", value_type="text")
CITIES = Question(
    key="cities", kind="extraction", instruction="a city", value_type="text", is_list=True
)

STORED: Mapping[tuple[str, ...], Mapping[str, Sequence[Found]]] = {
    ("a person", "a city"): {
        "a person": [("Ada", 0, 3, 0.65)],
        "a city": [("Paris", 14, 19, 0.75), ("Oslo", 23, 27, 0.7)],
    },
    ("a person",): {"a person": [("Ada", 0, 3, 0.65)]},
    ("a city",): {"a city": [("Paris", 14, 19, 0.75), ("Oslo", 23, 27, 0.7)]},
}
"""What the model returned for TEXT and each set of labels, stored by a run against it."""

EXPECTED: Mapping[str, Answer] = {
    "who": ExtractionAnswer(value="Ada", confidence=0.65, span=Span(part=0, start=0, end=3)),
    "cities": ExtractionAnswer(
        value=("Paris", "Oslo"),
        span=(Span(part=0, start=14, end=19), Span(part=0, start=23, end=27)),
    ),
}
"""The answers SpanLeaf maps those outputs to."""


class Playback:
    """The model the harness injects: it plays the exchanges, one per call."""

    def __init__(self, exchanges: Sequence[Exchange]) -> None:
        for exchange in exchanges:
            if isinstance(exchange, Failure) and exchange.status is not None:
                raise CheckNotApplicable("the model has no statuses")
            if not isinstance(exchange, Success | Failure | Stall):
                raise CheckNotApplicable(f"a model in process cannot make {exchange}")
        self._exchanges = list(exchanges)
        self.calls = 0

    def __call__(self, text: str, labels: Sequence[str]) -> Mapping[str, Sequence[Found]]:
        self.calls += 1
        if not self._exchanges:
            raise RuntimeError(f"the exchanges ran out at call {self.calls}")
        exchange = self._exchanges.pop(0)
        if isinstance(exchange, Failure):
            raise RuntimeError("the model failed")
        if isinstance(exchange, Stall):
            time.sleep(1.0)
        stored = STORED.get(tuple(labels))
        if text != TEXT or stored is None:
            raise RuntimeError(f"no stored output for {labels!r} over this text")
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
        return SpanLeaf(revision="rev-1", extractor=self._model)

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

    def __init__(self, *, revision: str | None, extractor: Playback) -> None:
        super().__init__(revision=revision, extractor=extractor)
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
        self._leaf = LoggedSpanLeaf(revision="rev-1", extractor=self._model)

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        self._model = Playback(exchanges)
        self._leaf = LoggedSpanLeaf(revision="rev-1", extractor=self._model)
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

Then assert which checks did not apply, and why that is right for your leaf. For
`SpanLeaf`: no SDK (L9), no status (L12, P9, P20, P21), no body to read (L13, P12), no HTTP
client (L15), outputs that cannot be bent into broken ones (L19, L24), no kind whose
probabilities sum (L18), no `noul` (L20), no opt-in feature (P6), no image (P23), and not
a composite (P14 to P18):

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
log does not know the revision it was made under:

```python
# test_spanleaf.py
def test_a_recording_replays_with_no_model(tmp_path: Path) -> None:
    log = tmp_path / "span-leaf.json"
    request = Request(state=STATE, questions=(WHO, CITIES), deadline=Deadline.after(5))

    async def ask(leaf: Leaf) -> Mapping[str, Answer]:
        async with Dispatcher(leaf, retry=RetryPolicy.without_delay()) as provider:
            return (await provider.answer(request)).answers

    live = SpanLeaf(revision="rev-1", extractor=toy_model)
    recorded = asyncio.run(ask(Recorded(live, log, "once")))
    replayed = asyncio.run(ask(Recorded(None, log, "none", capabilities=CAPABILITIES)))
    assert replayed == recorded == EXPECTED
```

## Gate a run against the real model

The tests above never load the real model. A test that does, to check the leaf against it
or to record the stored outputs again, runs only when asked: behind a marker, and an
environment variable set by hand. Run it with the model's files already cached and any hub
set offline, so the run reads nothing from the network:

```python
# conftest.py
import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_model: loads the real model")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("real_model") and os.environ.get("SPANLEAF_REAL_MODEL") != "1":
        pytest.skip("loads the real model; set SPANLEAF_REAL_MODEL=1 to run")
```

```python
# test_real_model.py
import asyncio

import pytest

from idia import Deadline, WireRequest
from spanleaf import SpanLeaf
from test_spanleaf import CITIES, EXPECTED, STATE, WHO


@pytest.mark.real_model
def test_the_real_model_gives_the_stored_answers() -> None:
    async def ask() -> object:
        leaf = SpanLeaf(revision="rev-1")
        await leaf.load()
        try:
            wire_request = WireRequest(state=STATE, questions=(WHO, CITIES))
            return (await leaf.call(wire_request, Deadline.after(30))).answers
        finally:
            await leaf.aclose()

    assert asyncio.run(ask()) == EXPECTED
```

Run the tests with `pytest`; the real-model test runs with `SPANLEAF_REAL_MODEL=1 pytest -m
real_model`.

## Where next

- [The conformance suite](../spec/conformance.md): every check, the harness protocols and
  the exchanges.
- [Vocabulary](../spec/vocabulary.md): the shapes, the answers, validation and the errors.
- [Fit](../spec/fit.md): how a request is checked against the record and split.
- [Dispatch](../spec/dispatch.md): the deadline, retries and the outcomes a leaf's errors
  become.
