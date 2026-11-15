"""The GLiNER2 leaf's logic: a span extractor run in this process, answering ``extraction``
of text.

This module imports neither gliner2 nor torch, so the tests build the leaf over an injected
model and replay stored outputs with no runtime installed. ``idia.providers.gliner``, the
registered module, imports the runtime at its top, so that a missing ``gliner`` extra is
reported by the import guard, and re-exports the leaf. Only :func:`_load_model`, which
:meth:`Gliner.load` runs on the leaf's worker, imports the runtime here.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import platform
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from idia.answers import validate_answers
from idia.core import (
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
    TokenEstimator,
    UnreportedReason,
    Usage,
    WireRequest,
    WireResponse,
    unknown_options,
)

_MODEL: Final = "fastino/gliner2.5-small-v1"
_REVISION: Final = "7132dc4561c3f94563c6147e75ffa8ef34c4964a"
"""The commit of ``fastino/gliner2.5-small-v1`` the record was measured on and the stored
outputs were recorded from."""
_THRESHOLD: Final = 0.5
"""gliner2's own default: a span scored below it is not returned."""
_RUNTIME: Final = ("gliner2", "torch", "transformers")
"""The packages that run the model, whose versions every wire response's ``versions`` holds,
beside Python's."""
_CLIENT: Final = ("huggingface_hub",)
"""The package that fetches and resolves the weights, in ``client_versions``."""
_VERSIONED: Final = (*_RUNTIME, *_CLIENT)
"""Every package whose version the stored outputs record."""


class _Pinned(str):
    """The default revision: the commit the record was measured on. A class of its own, so
    that the leaf tells the default from a revision the caller passed, even the same one."""

    __slots__ = ()


_PINNED: Final = _Pinned(_REVISION)


class GlinerRuntime(Protocol):
    """What the leaf asks of a loaded model: one pass, and its tokenizer's count.

    :meth:`Gliner.load` builds one over gliner2; a caller or a test may inject another.
    """

    def extract_entities(
        self, text: str, labels: Sequence[str], threshold: float
    ) -> Mapping[str, Any]:
        """One forward pass over ``text`` with ``labels``: gliner2's
        ``{"entities": {label: [{"text", "confidence", "start", "end"}, ...]}}``."""
        ...

    def input_tokens(self, text: str, labels: Sequence[str]) -> int:
        """The number of tokens the model's own tokenizer encodes for ``text`` and
        ``labels``, as the pass builds them."""
        ...


class _LoadedGliner:
    """A gliner2 extractor behind :class:`GlinerRuntime`."""

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    def extract_entities(
        self, text: str, labels: Sequence[str], threshold: float
    ) -> Mapping[str, Any]:
        return cast(
            "Mapping[str, Any]",
            self._extractor.extract_entities(
                text, list(labels), threshold=threshold, include_confidence=True, include_spans=True
            ),
        )

    def input_tokens(self, text: str, labels: Sequence[str]) -> int:
        # gliner2 adds "." to a text not ending in ".", "!" or "?" before it encodes it.
        if not text.endswith((".", "!", "?")):
            text += "."
        schema = self._extractor.create_schema().entities(list(labels)).build()
        return len(self._extractor.processor.transform_and_format(text, schema).input_ids)


def _load_model(model: str, revision: str | None) -> tuple[GlinerRuntime, str]:
    """Resolve ``revision`` to its commit, from the cache or the hub, and load the weights
    from that snapshot; the one function that may reach the network.

    The checkpoint's architecture is ``boundary``, which ``AutoExtractor`` dispatches on;
    ``GLiNER2.from_pretrained`` builds the span architecture and fails on it.
    """
    # Imported by name, so that this module type-checks with the runtime not installed.
    hub: Any = importlib.import_module("huggingface_hub")
    gliner2: Any = importlib.import_module("gliner2")
    path = str(hub.snapshot_download(model, revision=revision))
    commit = Path(path).name
    extractor: Any = gliner2.AutoExtractor.from_pretrained(path)
    extractor.eval()
    return _LoadedGliner(extractor), commit


_RECORD: Final = Capabilities(
    kinds=frozenset({"extraction"}),
    modalities=frozenset({"text"}),
    image_carriers=frozenset(),
    max_input_tokens=2048,
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
    confidence={"extraction": "top_span_score"},
    rounding=Rounding(probability_decimals=None, score_decimals=None),
    opt_in_features=frozenset(),
    enabled_features=frozenset(),
    feature_kinds={},
    feature_modalities={},
)

_PIECE: Final = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


def _pieces(text: str) -> int:
    """An ASCII word costs 1 + len // 5, every digit and every other character 1."""
    return sum(
        1 + len(piece) // 5 if piece[0].isascii() and piece[0].isalpha() else 1
        for piece in _PIECE.findall(text)
    )


class _PieceEstimator:
    """Estimates gliner2's token count with no tokenizer: 8 for the frame, the text's
    pieces, and 2 plus its pieces for each distinct label. Against the real tokenizer it
    overcounts English by about 30% and undercounts Latin by up to 9%, hence the margin."""

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        tokens = 8 + sum(_pieces(part.text) for part in state.parts if isinstance(part, TextPart))
        labels = {str(question.instruction) for question in questions}
        tokens += sum(2 + _pieces(label) for label in labels)
        return TokenEstimate(tokens=tokens, margin=0.1)


class Gliner:
    """A leaf for ``fastino/gliner2.5-small-v1``, run in this process with gliner2.

    It answers ``extraction`` of text, a value or a list, from one text part: each
    question's instruction is sent as its label, as it is. ``revision`` pins the weights,
    by default the commit the record was measured on; ``None`` loads the hub's ``main``
    and every response warns ``no-snapshot``. ``threshold`` is gliner2's: a span scored
    below it is not returned.

    Await :meth:`load` before the first call: it resolves the revision to its commit and
    loads the weights from that snapshot, the only step that may reach the network. A call
    before it raises a non-retryable ``ProviderError``. ``runtime`` is a model already
    loaded, which needs no load. Its weights are the caller's to name: the leaf reports a
    ``revision`` the caller passed as the snapshot, as the caller's claim, and with none
    passed reports no snapshot and warns ``no-snapshot``. Loading and every pass run on
    one worker the leaf owns, one at a time.
    """

    def __init__(
        self,
        *,
        model: Literal["fastino/gliner2.5-small-v1"] = "fastino/gliner2.5-small-v1",
        revision: str | None = _PINNED,
        threshold: float = _THRESHOLD,
        runtime: GlinerRuntime | None = None,
    ) -> None:
        if model != _MODEL:
            raise ValueError(f"Gliner does not know the model {model!r}; known: {_MODEL}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"the threshold is in (0, 1], not {threshold!r}")
        if revision is not None and not str(revision).strip():
            raise ValueError("a revision names a commit, branch or tag; pass None for none")
        self._revision = revision
        self._threshold = threshold
        self._runtime = runtime
        self._given = runtime is not None
        self._snapshot: str | None = None
        self._claimed: frozenset[ProvenanceDetail] = frozenset({"model_reported"})
        if runtime is not None and revision is not None and not isinstance(revision, _Pinned):
            self._snapshot = str(revision)
            self._claimed = frozenset({"model_reported", "snapshot"})
        self._versions = {"python": platform.python_version(), **_installed(_RUNTIME)}
        self._client_versions = _installed(_CLIENT)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="idia-gliner")
        self._estimator = _PieceEstimator()
        self._closed = False
        self._guard = threading.Lock()
        self._pending: Future[tuple[GlinerRuntime, str] | None] | None = None

    @property
    def name(self) -> str:
        return "gliner"

    @property
    def model(self) -> str:
        return _MODEL

    @property
    def capabilities(self) -> Capabilities:
        return _RECORD

    @property
    def estimator(self) -> TokenEstimator:
        return self._estimator

    def __repr__(self) -> str:
        revision = None if self._revision is None else str(self._revision)
        return f"Gliner(revision={revision!r})"

    async def load(self) -> None:
        """Resolve the revision and load the weights on the worker, then run one small pass
        so that its cost stays out of the first call. Loading again, or a leaf given a
        runtime, does nothing. A load made while another runs, on the same loop or a later
        one, waits for it and shares its result, and a caller cancelled while it waits
        leaves the load running for the next. What the loader raises propagates; a leaf
        closed while it loads drops the model and raises the closed leaf's
        ``ProviderError`` instead."""
        revision = None if self._revision is None else str(self._revision)
        with self._guard:
            self._refuse_if_closed()
            if self._runtime is not None:
                return
            pending = self._pending
            if pending is None or (
                pending.done() and (pending.cancelled() or pending.exception() is not None)
            ):
                pending = self._worker.submit(self._load_and_warm, revision)
                self._pending = pending
        try:
            loaded = await asyncio.shield(asyncio.wrap_future(pending))
        except asyncio.CancelledError:
            self._closed_cancellation()
            raise
        except Exception as error:
            if self._closed:
                raise ProviderError("the leaf is closed") from error
            raise
        with self._guard:
            if self._closed or loaded is None:
                raise ProviderError("the leaf is closed")
            if self._runtime is None:
                self._runtime, commit = loaded
                self._snapshot = commit if revision is not None else None

    def _load_and_warm(self, revision: str | None) -> tuple[GlinerRuntime, str] | None:
        """On the worker: load the model, then warm it up unless the leaf closed
        meanwhile, which gives ``None``."""
        runtime, commit = _load_model(_MODEL, revision)
        if self._closed:
            return None
        runtime.extract_entities("Warm up.", ("x",), self._threshold)
        return runtime, commit

    def _refuse_if_closed(self) -> None:
        if self._closed:
            raise ProviderError("the leaf is closed")

    def _closed_cancellation(self) -> None:
        """Turn the cancellation of a pass that :meth:`aclose` cancelled in the queue into
        the closed leaf's ``ProviderError``; a cancellation of the awaiting task itself
        is left to propagate."""
        task = asyncio.current_task()
        if self._closed and (task is None or task.cancelling() == 0):
            raise ProviderError("the leaf is closed") from None

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Run one pass over the wire request's text with every question's label."""
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
        work = loop.run_in_executor(self._worker, _run, runtime, text, sent, self._threshold)
        try:
            raw, tokens = await asyncio.wait_for(work, max(deadline.remaining(), 0.0))
        except asyncio.CancelledError:
            self._closed_cancellation()
            raise
        except TimeoutError:
            error = work.exception() if work.done() and not work.cancelled() else None
            if error is not None:
                # The runtime raised it: a model failure, not the deadline's cut.
                raise ProviderError(f"the model failed: {type(error).__name__}: {error}") from error
            raise ProviderError(
                "the deadline passed during the pass, which runs to its end on the worker",
                is_retryable=True,
            ) from None
        except Exception as error:
            raise ProviderError(f"the model failed: {type(error).__name__}: {error}") from error
        entities: object = raw.get("entities")
        if not isinstance(entities, Mapping):
            raise ProviderError("the model's output cannot be read: no entities map")
        found = cast("Mapping[str, object]", entities)
        # gliner2 returns every label asked, an empty list where it found nothing, so a
        # label missing from the output is a key the output left out: validation faults it.
        mapped = {
            question.key: _answer(question, found[labels[question.key]])
            for question in wire_request.questions
            if labels[question.key] in found
        }
        answers, warnings = validate_answers(wire_request.questions, mapped, _RECORD.rounding)
        if self._snapshot is None:
            warnings += (
                Notice(
                    code=WARNING_NO_SNAPSHOT,
                    message="no revision was pinned, so the weights that answered are unknown",
                    field="snapshot",
                ),
            )
        warnings += unknown_options(wire_request.options, frozenset())
        unreported: dict[ProvenanceDetail, UnreportedReason] = {
            "usage.output_tokens": "not-applicable",
            "request_id": "not-reported",
        }
        if self._snapshot is None:
            unreported["snapshot"] = "not-pinned"
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=_RECORD.calibrated,
            model_reported=_MODEL,
            snapshot=self._snapshot,
            prompt_sent={
                "text_chars": len(text),
                "labels": dict(labels),
                "threshold": self._threshold,
            },
            usage=Usage(
                input_tokens=tokens,
                output_tokens=None,
                raw_usage={"input_tokens": tokens, "counted_by": "gliner2 tokenizer"},
            ),
            raw_body={"output": _json(raw)},
            warnings=warnings,
            versions=self._versions,
            client_versions=self._client_versions,
            unreported=unreported,
            claimed=self._claimed,
        )

    async def aclose(self) -> None:
        """Refuse every later call, drop a model this leaf loaded, never one it was given,
        and shut the worker down: a pass still queued never runs, and its call raises the
        closed leaf's ``ProviderError``; a pass already running is not waited for."""
        with self._guard:
            self._closed = True
            self._pending = None
            if not self._given:
                self._runtime = None
            self._worker.shutdown(wait=False, cancel_futures=True)


def _run(
    runtime: GlinerRuntime, text: str, labels: tuple[str, ...], threshold: float
) -> tuple[Mapping[str, Any], int]:
    raw = runtime.extract_entities(text, labels, threshold)
    tokens = runtime.input_tokens(text, labels)
    if not isinstance(raw, Mapping) or type(tokens) is not int:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("the runtime returned no mapping or no token count")
    return raw, tokens


def _one_text(wire_request: WireRequest) -> str:
    """The state's one text part; refused before the pass otherwise, as is a question
    that is not an extraction of text."""
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


def _found(spans: object) -> list[tuple[str, int, int, float]] | None:
    """The spans as (text, start, end, score), or ``None`` where they cannot be read."""
    if not isinstance(spans, list):
        return None
    found: list[tuple[str, int, int, float]] = []
    for span in cast("list[object]", spans):
        if not isinstance(span, Mapping):
            return None
        fields = cast("Mapping[str, object]", span)
        text, start, end, score = (fields.get(k) for k in ("text", "start", "end", "confidence"))
        if not (isinstance(text, str) and type(start) is int and type(end) is int):
            return None
        if not isinstance(score, int | float) or isinstance(score, bool):
            return None
        found.append((text, start, end, float(score)))
    return found


def _answer(question: Question, spans: object) -> MappedAnswer:
    """A list: every span, its ``confidence`` ``None``; a value: the top span by score,
    with the score as ``confidence``, or ``None`` when no span was found."""
    found = _found(spans)
    if found is None:
        return Fault(
            key=question.key,
            failures=(
                RuleFailure(rule=Rule.kind_shape, observed=type(spans).__name__, bound=None),
            ),
            mapped=None,
        )
    if question.is_list:
        return ExtractionAnswer(
            value=tuple(text for text, _, _, _ in found),
            span=tuple(Span(part=0, start=start, end=end) for _, start, end, _ in found),
        )
    if not found:
        return ExtractionAnswer(value=None)
    text, start, end, score = max(found, key=lambda span: span[3])
    return ExtractionAnswer(value=text, confidence=score, span=Span(part=0, start=start, end=end))


def _json(value: object) -> JsonValue:
    """The runtime's output as JSON values, arrays as tuples; anything else as its repr."""
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in cast("Mapping[object, object]", value).items()}
    if isinstance(value, list | tuple):
        return tuple(_json(v) for v in cast("Sequence[object]", value))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def _installed(packages: Sequence[str]) -> dict[str, str]:
    """The installed version of each of ``packages``, read from its metadata; a package with
    no metadata is left out."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions
