"""The GLiNER2 leaf over an injected model: what it refuses before the pass, what it reads,
its provenance, its lifecycle and its estimator. No test here needs gliner2 or torch."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from idia import (
    WARNING_NO_SNAPSHOT,
    WARNING_UNKNOWN_OPTION,
    Answer,
    Deadline,
    DeadlineExceeded,
    Dispatcher,
    DoesNotFit,
    ExtractionAnswer,
    Fault,
    Leaf,
    ProviderError,
    Question,
    Request,
    RequestFailed,
    RetryPolicy,
    Rule,
    Span,
    State,
    TextPart,
    ValueType,
    WireRequest,
)
from idia.conformance import Stall, Success
from idia.providers import _gliner
from idia.providers._gliner import _RECORD, Gliner
from idia.providers.recorded import Recorded
from tests._gliner_cases import (
    ASKED,
    NON_ASCII,
    NON_ASCII_LABELS,
    NOTHING,
    OUTPUTS,
    SENDER,
    STATE,
    TEXT,
    TOPICS,
    Answering,
    Playback,
    recorded_with,
)
from tests._gliner_harness import EXPECTED

REVISION = "7132dc4561c3f94563c6147e75ffa8ef34c4964a"


def _wire(*questions: Question, state: State = STATE, **options: Any) -> WireRequest:
    return WireRequest(state=state, questions=questions, options=options)


def _call(leaf: Gliner, wire_request: WireRequest, seconds: float = 5.0) -> Any:
    return asyncio.run(leaf.call(wire_request, Deadline.after(seconds)))


# ----------------------------------------------------------------------------------------
# Construction and lifecycle


def test_the_leaf_is_named_and_pinned_by_default() -> None:
    leaf = Gliner()
    assert isinstance(leaf, Leaf)
    assert (leaf.name, leaf.model) == ("gliner", "fastino/gliner2.5-small-v1")
    assert leaf.capabilities is _RECORD
    assert leaf._revision == REVISION == _gliner._REVISION
    assert repr(leaf) == f"Gliner(revision={REVISION!r})"
    asyncio.run(leaf.aclose())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model": "fastino/gliner2-base-v1"}, "does not know the model"),
        ({"threshold": 0.0}, "the threshold is in (0, 1]"),
        ({"threshold": 1.5}, "the threshold is in (0, 1]"),
    ],
)
def test_construction_refuses_another_model_or_a_threshold_outside_the_unit_interval(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace("]", r"\]")):
        Gliner(**options)


def test_a_call_before_load_is_refused_and_not_retryable() -> None:
    leaf = Gliner()
    with pytest.raises(ProviderError, match=r"await leaf\.load") as raised:
        _call(leaf, _wire(SENDER))
    assert not raised.value.is_retryable
    asyncio.run(leaf.aclose())


def test_load_resolves_the_revision_and_loads_from_that_snapshot_on_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Playback([Success(), Success("extraction")])
    loads: list[tuple[str, str | None, str]] = []

    def load(name: str, revision: str | None) -> tuple[Playback, str]:
        import threading

        loads.append((name, revision, threading.current_thread().name))
        return model, "c0ffee"

    monkeypatch.setattr(_gliner, "_load_model", load)

    async def run() -> None:
        leaf = Gliner(revision="main")
        with pytest.raises(RuntimeError):  # the warm-up's pass has no stored output
            await leaf.load()

    asyncio.run(run())
    assert loads[0][:2] == ("fastino/gliner2.5-small-v1", "main")
    assert loads[0][2].startswith("idia-gliner")
    assert model.calls == 1


def test_load_reports_the_commit_the_revision_resolved_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = OUTPUTS[ASKED[0]][0]
    model = Answering(output, tokens=33)

    def load(name: str, revision: str | None) -> tuple[Answering, str]:
        return model, "c0ffee"

    monkeypatch.setattr(_gliner, "_load_model", load)

    async def run(revision: str | None) -> Any:
        leaf = Gliner(revision=revision)
        await leaf.load()
        await leaf.load()  # loading again does nothing
        response = await leaf.call(_wire(SENDER, TOPICS), Deadline.after(5))
        await leaf.aclose()
        return response

    pinned = asyncio.run(run("main"))
    assert pinned.snapshot == "c0ffee"
    assert not [w for w in pinned.warnings if w.code == WARNING_NO_SNAPSHOT]
    unpinned = asyncio.run(run(None))
    assert unpinned.snapshot is None
    assert [w.field for w in unpinned.warnings if w.code == WARNING_NO_SNAPSHOT] == ["snapshot"]
    assert model.calls == 4  # per leaf, one warm-up and one call


def test_after_aclose_a_call_and_a_load_are_refused_and_closing_twice_is_harmless() -> None:
    model = Playback([Success("extraction")])
    leaf = Gliner(revision=REVISION, runtime=model)

    async def run() -> None:
        await leaf.aclose()
        await leaf.aclose()
        with pytest.raises(ProviderError, match="closed") as called:
            await leaf.call(_wire(SENDER), Deadline.after(5))
        assert not called.value.is_retryable
        with pytest.raises(ProviderError, match="closed"):
            await leaf.load()

    asyncio.run(run())
    assert model.calls == 0
    assert leaf._runtime is model  # a model it was given is never dropped


def test_aclose_drops_a_model_the_leaf_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Answering({"entities": {"x": []}})

    def load(name: str, revision: str | None) -> tuple[Answering, str]:
        return model, REVISION

    monkeypatch.setattr(_gliner, "_load_model", load)

    async def run() -> Gliner:
        leaf = Gliner()
        await leaf.load()
        await leaf.aclose()
        return leaf

    assert asyncio.run(run())._runtime is None


def test_a_given_model_needs_no_load_and_is_reported_at_the_revision_given() -> None:
    model = Playback([Success("extraction")])
    leaf = Gliner(runtime=model, revision="v1")
    asyncio.run(leaf.load())
    assert model.calls == 0
    response = _call(leaf, _wire(SENDER, TOPICS))
    assert response.snapshot == "v1"
    assert not [w for w in response.warnings if w.code == WARNING_NO_SNAPSHOT]


@pytest.mark.parametrize("options", [{}, {"revision": None}], ids=["no-revision", "none"])
def test_a_given_model_with_no_revision_passed_has_no_snapshot_and_warns(
    options: dict[str, Any],
) -> None:
    # The default revision names the weights the leaf loads, not those of a model it is given.
    leaf = Gliner(runtime=Playback([Success("extraction")]), **options)
    response = _call(leaf, _wire(SENDER, TOPICS))
    assert response.snapshot is None
    assert [w.field for w in response.warnings if w.code == WARNING_NO_SNAPSHOT] == ["snapshot"]
    assert response.answers == EXPECTED


def test_a_given_model_at_the_default_commit_passed_by_the_caller_is_reported() -> None:
    leaf = Gliner(runtime=Playback([Success("extraction")]), revision=str(_gliner._REVISION))
    assert _call(leaf, _wire(SENDER, TOPICS)).snapshot == REVISION


def test_aclose_while_loading_drops_the_model_and_refuses_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Answering({"entities": {"x": []}})
    loading = threading.Event()
    release = threading.Event()

    def load(name: str, revision: str | None) -> tuple[Answering, str]:
        loading.set()
        release.wait(5)
        return model, REVISION

    monkeypatch.setattr(_gliner, "_load_model", load)

    async def run() -> Gliner:
        leaf = Gliner()
        loaded = asyncio.ensure_future(leaf.load())
        await asyncio.to_thread(loading.wait, 5)
        await leaf.aclose()
        release.set()
        with pytest.raises(ProviderError, match="the leaf is closed") as raised:
            await loaded
        assert not raised.value.is_retryable
        return leaf

    leaf = asyncio.run(run())
    assert leaf._runtime is None
    assert model.calls == 0  # the warm-up never ran


def test_a_load_queued_behind_another_never_runs_after_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Answering({"entities": {"x": []}})
    loading = threading.Event()
    release = threading.Event()
    loads: list[str | None] = []

    def load(name: str, revision: str | None) -> tuple[Answering, str]:
        loads.append(revision)
        loading.set()
        release.wait(5)
        return model, REVISION

    monkeypatch.setattr(_gliner, "_load_model", load)

    async def run() -> None:
        leaf = Gliner()
        first = asyncio.ensure_future(leaf.load())
        await asyncio.to_thread(loading.wait, 5)
        queued = asyncio.ensure_future(leaf.load())
        await asyncio.sleep(0.05)  # the second load waits in the worker's queue
        await leaf.aclose()
        release.set()
        for loaded in (queued, first):
            with pytest.raises(ProviderError, match="the leaf is closed"):
                await loaded

    asyncio.run(run())
    assert len(loads) == 1


def test_a_cancelled_load_is_cancelled_and_loads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loading = threading.Event()
    release = threading.Event()

    def load(name: str, revision: str | None) -> tuple[Answering, str]:
        loading.set()
        release.wait(5)
        return Answering({"entities": {"x": []}}), REVISION

    monkeypatch.setattr(_gliner, "_load_model", load)

    async def run() -> Gliner:
        leaf = Gliner()
        loaded = asyncio.ensure_future(leaf.load())
        await asyncio.to_thread(loading.wait, 5)
        loaded.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loaded
        release.set()
        await leaf.aclose()
        return leaf

    assert asyncio.run(run())._runtime is None


def test_a_pass_queued_behind_another_never_runs_after_aclose() -> None:
    running = threading.Event()
    release = threading.Event()

    class Held(Answering):
        def extract_entities(
            self, text: str, labels: Sequence[str], threshold: float
        ) -> Mapping[str, Any]:
            running.set()
            release.wait(5)
            return super().extract_entities(text, labels, threshold)

    model = Held({"entities": {"the sender": []}})
    leaf = Gliner(runtime=model, revision=REVISION)

    async def run() -> None:
        first = asyncio.ensure_future(leaf.call(_wire(SENDER), Deadline.after(5)))
        await asyncio.to_thread(running.wait, 5)
        queued = asyncio.ensure_future(leaf.call(_wire(SENDER), Deadline.after(5)))
        await asyncio.sleep(0.05)  # the second pass waits in the worker's queue
        await leaf.aclose()
        with pytest.raises(ProviderError, match="the leaf is closed") as raised:
            await queued
        assert not raised.value.is_retryable
        release.set()
        await first  # the pass already running ends and is answered

    asyncio.run(run())
    assert model.calls == 1


def test_a_cancelled_call_is_cancelled_and_not_a_closed_leaf() -> None:
    model = Playback([Stall()])
    leaf = Gliner(runtime=model, revision=REVISION)

    async def run() -> None:
        call = asyncio.ensure_future(leaf.call(_wire(SENDER), Deadline.after(5)))
        await asyncio.sleep(0.05)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(run())
    asyncio.run(leaf.aclose())


# ----------------------------------------------------------------------------------------
# Refused before the pass


@pytest.mark.parametrize("value_type", ["number", "date", "time", "datetime", "duration"])
@pytest.mark.parametrize("is_list", [False, True])
def test_another_value_type_is_refused_before_the_pass(
    value_type: ValueType, is_list: bool
) -> None:
    model = Playback([Success()])
    question = Question(
        key="n", kind="extraction", instruction="a figure", value_type=value_type, is_list=is_list
    )
    with pytest.raises(ProviderError, match="not an extraction of text") as raised:
        _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, question))
    assert not raised.value.is_retryable
    assert model.calls == 0


@pytest.mark.parametrize(
    "state",
    [
        State(parts=(TextPart(text=TEXT), TextPart(text="more"))),
        State(parts=()),
    ],
    ids=["two-text-parts", "no-part"],
)
def test_a_state_that_is_not_one_text_part_is_refused_before_the_pass(state: State) -> None:
    model = Playback([Success()])
    with pytest.raises(ProviderError, match="one text part") as raised:
        _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, state=state))
    assert not raised.value.is_retryable
    assert model.calls == 0


def test_a_structured_instruction_is_refused_before_the_pass() -> None:
    model = Playback([Success()])
    question = Question(
        key="sender", kind="extraction", instruction={"field": "sender"}, value_type="text"
    )
    with pytest.raises(ProviderError, match="structured instruction"):
        _call(Gliner(revision=REVISION, runtime=model), _wire(question))
    assert model.calls == 0


def test_a_text_past_the_window_is_refused_whole_and_never_split() -> None:
    model = Playback([Success()])
    long_text = "The budget review covered revenue and churn. " * 400
    request = Request(
        state=State(parts=(TextPart(text=long_text),)),
        questions=(SENDER,),
        deadline=Deadline.after(5),
    )
    with pytest.raises(DoesNotFit) as raised:
        asyncio.run(Dispatcher(Gliner(revision=REVISION, runtime=model)).answer(request))
    assert raised.value.state_overflow
    assert raised.value.check == "max_input_tokens"
    assert model.calls == 0


def test_a_deadline_already_passed_is_retryable_and_runs_nothing() -> None:
    model = Playback([Success("extraction")])
    with pytest.raises(ProviderError, match="deadline") as raised:
        _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER), seconds=0.0)
    assert raised.value.is_retryable
    assert model.calls == 0


def test_a_deadline_cut_is_retryable() -> None:
    model = Playback([Stall()])
    with pytest.raises(ProviderError, match="deadline passed during the pass") as raised:
        _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER), seconds=0.05)
    assert raised.value.is_retryable


def test_a_cut_through_a_dispatcher_is_one_failed_attempt_then_the_deadline() -> None:
    model = Playback([Stall(), Stall()])
    request = Request(state=STATE, questions=(SENDER,), deadline=Deadline.after(0.2))
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(Dispatcher(Gliner(runtime=model, revision=REVISION)).answer(request))
    [outcome] = raised.value.response.outcomes
    assert isinstance(outcome.error, DeadlineExceeded)
    assert outcome.request_id is None
    [attempt] = outcome.failed_attempts
    assert attempt.error.is_retryable
    assert "deadline passed during the pass" in str(attempt.error)
    assert model.calls == 1


def test_a_model_failure_is_not_retryable() -> None:
    from idia.conformance import Failure

    with pytest.raises(ProviderError, match="the model failed: RuntimeError") as raised:
        _call(Gliner(revision=REVISION, runtime=Playback([Failure()])), _wire(SENDER))
    assert not raised.value.is_retryable


# ----------------------------------------------------------------------------------------
# Reading and provenance


def test_the_response_carries_the_snapshot_usage_versions_and_no_request_id() -> None:
    response = _call(
        Gliner(revision=REVISION, runtime=Playback([Success("extraction")])), _wire(SENDER, TOPICS)
    )
    assert response.answers == EXPECTED
    assert response.elicitation == "passthrough"
    assert response.calibrated is False
    assert response.model_reported == "fastino/gliner2.5-small-v1"
    assert response.snapshot == REVISION
    assert response.request_id is None
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (33, None)
    assert response.usage.raw_usage == {"input_tokens": 33, "counted_by": "gliner2 tokenizer"}
    assert response.prompt_sent == {
        "text_chars": len(TEXT),
        "labels": {"sender": "the sender", "topics": "the topics"},
        "threshold": 0.5,
    }
    raw_body = cast("dict[str, Any]", response.raw_body)
    assert set(raw_body["versions"]) == {
        "python",
        "gliner2",
        "torch",
        "transformers",
        "huggingface_hub",
    }
    assert raw_body["output"] == _gliner._json(OUTPUTS[ASKED[0]][0])
    assert response.warnings == ()


def test_a_scalar_is_the_top_span_by_score_and_a_list_has_no_confidence() -> None:
    spans = [
        {"text": "Dana", "confidence": 0.6, "start": 21, "end": 25},
        {"text": "Dana Whitfield", "confidence": 0.9, "start": 21, "end": 35},
    ]
    model = Answering({"entities": {"the sender": spans, "the topics": spans}})
    answers = _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, TOPICS)).answers
    assert answers["sender"] == ExtractionAnswer(
        value="Dana Whitfield", confidence=0.9, span=Span(part=0, start=21, end=35)
    )
    topics = answers["topics"]
    assert isinstance(topics, ExtractionAnswer)
    assert topics.value == ("Dana", "Dana Whitfield")
    assert topics.confidence is None


def test_no_span_is_a_null_value_and_an_empty_list() -> None:
    model = Answering({"entities": {"the sender": [], "the topics": []}})
    answers = _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, TOPICS)).answers
    assert answers["sender"] == ExtractionAnswer(value=None)
    assert answers["topics"] == ExtractionAnswer(value=(), span=())


def test_one_label_asked_by_two_questions_is_sent_once() -> None:
    model = Answering({"entities": {"the sender": []}})
    again = Question(key="again", kind="extraction", instruction="the sender", value_type="text")
    response = _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, again))
    assert set(response.answers) == {"sender", "again"}
    assert response.prompt_sent["labels"] == {"sender": "the sender", "again": "the sender"}


def test_a_label_missing_from_the_output_is_a_missing_key_and_its_siblings_stand() -> None:
    # gliner2 returns every label asked, so a missing one is a key the output left out.
    model = Answering({"entities": {"the topics": []}})
    answers = _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, TOPICS)).answers
    fault = answers["sender"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.missing_key]
    assert fault.mapped is None
    assert answers["topics"] == ExtractionAnswer(value=(), span=())


def test_a_label_that_found_nothing_is_a_null_value_from_the_stored_output() -> None:
    nothing = Question(key="phone", kind="extraction", instruction=NOTHING, value_type="text")
    leaf = Gliner(revision=REVISION, runtime=Playback([Success()]))
    answers = _call(leaf, _wire(SENDER, nothing)).answers
    assert answers["phone"] == ExtractionAnswer(value=None)
    sender = answers["sender"]
    assert isinstance(sender, ExtractionAnswer)
    assert sender.value == "Dana Whitfield"


def test_offsets_into_a_text_not_ascii_count_characters() -> None:
    questions = tuple(
        Question(key=f"q{i}", kind="extraction", instruction=label, value_type="text")
        for i, label in enumerate(NON_ASCII_LABELS)
    )
    state = State(parts=(TextPart(text=NON_ASCII),))
    leaf = Gliner(revision=REVISION, runtime=Playback([Success()]))
    answers = _call(leaf, _wire(*questions, state=state)).answers
    values: list[object] = []
    for answer in answers.values():
        assert isinstance(answer, ExtractionAnswer)
        assert isinstance(answer.span, Span)
        assert NON_ASCII[answer.span.start : answer.span.end] == answer.value
        # A byte offset would differ: the text starts with a character of two bytes.
        assert len(NON_ASCII[: answer.span.start].encode()) != answer.span.start
        values.append(answer.value)
    assert values == ["Zoë Müller", "Zürich", "Straße des 17. Juni"]


@pytest.mark.parametrize(
    "spans",
    [
        None,
        "Dana",
        ["Dana"],
        [{"text": "Dana", "start": "21", "end": 25, "confidence": 0.9}],
        [{"text": "Dana", "start": 21, "end": 25, "confidence": True}],
    ],
    ids=["null", "not-a-list", "not-a-mapping", "bad-offset", "bad-score"],
)
def test_spans_that_cannot_be_read_fault_that_question_alone(spans: object) -> None:
    model = Answering({"entities": {"the sender": spans, "the topics": []}})
    answers = _call(Gliner(revision=REVISION, runtime=model), _wire(SENDER, TOPICS)).answers
    fault = answers["sender"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].rule is Rule.kind_shape
    assert answers["topics"] == ExtractionAnswer(value=(), span=())


@pytest.mark.parametrize("output", [{"spans": []}, {"entities": []}])
def test_an_output_with_no_entities_map_is_not_retryable(output: Mapping[str, Any]) -> None:
    with pytest.raises(ProviderError, match="no entities map") as raised:
        _call(Gliner(revision=REVISION, runtime=Answering(output)), _wire(SENDER))
    assert not raised.value.is_retryable


def test_a_runtime_returning_no_count_is_a_model_failure() -> None:
    class NoCount(Answering):
        def input_tokens(self, text: str, labels: Sequence[str]) -> int:
            return "33"  # type: ignore[return-value]

    with pytest.raises(ProviderError, match="no token count"):
        _call(Gliner(revision=REVISION, runtime=NoCount({"entities": {}})), _wire(SENDER))


def test_an_unknown_option_is_warned_and_changes_nothing() -> None:
    response = _call(
        Gliner(revision=REVISION, runtime=Playback([Success("extraction")])),
        _wire(SENDER, TOPICS, reference_date="2026-01-01"),
    )
    assert response.answers == EXPECTED
    assert [w.code for w in response.warnings] == [WARNING_UNKNOWN_OPTION]


def test_raw_output_is_kept_as_json() -> None:
    assert _gliner._json({"a": [1, None, object]}) == {"a": (1, None, repr(object))}


def test_a_recording_replays_with_no_model(tmp_path: Path) -> None:
    log = tmp_path / "gliner.json"
    request = Request(state=STATE, questions=(SENDER, TOPICS), deadline=Deadline.after(5))

    async def ask(leaf: Leaf) -> Mapping[str, Answer]:
        async with Dispatcher(leaf, retry=RetryPolicy.without_delay()) as provider:
            return (await provider.answer(request)).answers

    live = Gliner(revision=REVISION, runtime=Playback([Success("extraction")]))
    recorded = asyncio.run(ask(Recorded(live, log, "once")))
    replayed = asyncio.run(ask(Recorded(None, log, "none", capabilities=_RECORD)))
    assert replayed == recorded == EXPECTED


# ----------------------------------------------------------------------------------------
# The estimator


def test_the_estimate_counts_the_frame_the_text_and_each_distinct_label_once() -> None:
    again = Question(key="again", kind="extraction", instruction="the sender", value_type="text")
    state = State(parts=(TextPart(text="Dana Whitfield, 2026."),))
    estimate = Gliner().estimator.estimate(state, (SENDER, again))
    # Frame 8; "Dana" 1, "Whitfield" 2, "," 1, digits 4, "." 1; "the sender": 2 + 1 + 2.
    assert (estimate.tokens, estimate.margin) == (8 + 9 + 5, 0.1)


def test_the_estimate_is_at_least_the_stored_count() -> None:
    estimate = Gliner().estimator.estimate(STATE, (SENDER, TOPICS))
    assert estimate.tokens >= OUTPUTS[ASKED[0]][1]


def test_the_estimate_with_its_margin_covers_the_stored_count_of_a_text_not_english() -> None:
    questions = tuple(
        Question(key=f"q{i}", kind="extraction", instruction=label, value_type="text")
        for i, label in enumerate(NON_ASCII_LABELS)
    )
    state = State(parts=(TextPart(text=NON_ASCII),))
    estimate = Gliner().estimator.estimate(state, questions)
    tokens = OUTPUTS[(NON_ASCII, NON_ASCII_LABELS, 0.5)][1]
    assert estimate.tokens * (1 + estimate.margin) >= tokens


def test_the_stored_outputs_name_the_revision_and_every_versioned_package() -> None:
    recorded = recorded_with()
    assert recorded["revision"] == REVISION
    for package in (*_gliner._VERSIONED, "python"):
        assert recorded.get(package), package


# ----------------------------------------------------------------------------------------
# The wrapper over a gliner2 extractor


class _Built:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    def build(self) -> dict[str, Any]:
        return {"entities": self.labels}


class _Schema:
    def entities(self, labels: list[str]) -> _Built:
        return _Built(labels)


class _Encoded:
    def __init__(self, text: str) -> None:
        self.input_ids = text.split()


class _Processor:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, Any]]] = []

    def transform_and_format(self, text: str, schema: dict[str, Any]) -> _Encoded:
        self.seen.append((text, schema))
        return _Encoded(text)


class _Extractor:
    """What the wrapper asks of a gliner2 extractor, recorded."""

    def __init__(self) -> None:
        self.processor = _Processor()
        self.asked: list[tuple[str, list[str], dict[str, Any]]] = []

    def create_schema(self) -> _Schema:
        return _Schema()

    def extract_entities(self, text: str, labels: list[str], **options: Any) -> dict[str, Any]:
        self.asked.append((text, labels, options))
        return {"entities": {}}


def test_the_wrapper_asks_for_scores_and_spans_and_counts_the_text_as_gliner2_ends_it() -> None:
    extractor = _Extractor()
    runtime = _gliner._LoadedGliner(extractor)
    assert runtime.extract_entities("a b", ("x",), 0.4) == {"entities": {}}
    assert extractor.asked == [
        ("a b", ["x"], {"threshold": 0.4, "include_confidence": True, "include_spans": True})
    ]
    assert runtime.input_tokens("a b", ("x", "y")) == 2
    assert runtime.input_tokens("a b!", ("x",)) == 2
    assert extractor.processor.seen == [
        ("a b.", {"entities": ["x", "y"]}),
        ("a b!", {"entities": ["x"]}),
    ]


def test_a_runtime_package_not_installed_has_no_version() -> None:
    assert _gliner._version("idia-no-such-package") is None
