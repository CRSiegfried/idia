"""The runs against the real GLiNER2 model, one of which records the stored outputs the other
tests replay. Skipped unless ``IDIA_LOCAL_MODEL=1``, which also sets ``HF_HUB_OFFLINE=1``:
the weights at the pinned revision must already be cached, and nothing is downloaded.

The model is loaded once, through the leaf's own loader. Each recorded case runs once: its
spans must be the texts the case expects, at offsets that count characters, and where the
case is stored, the same spans, each score within :data:`TOLERANCE` of the stored one and the
same token count. The outputs that passed are written back, with the runtime's versions, and
nothing else is. Then the estimator is measured against the tokenizer, 64 labels run in one
pass, and the loaded leaf answers through a dispatcher. Run them once per model:
``IDIA_LOCAL_MODEL=1 uv run pytest -m local_model -s``.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import importlib.metadata
import platform
import time
from collections.abc import Mapping
from typing import Any, Final, cast

import pytest

from idia import Deadline, Dispatcher, ExtractionAnswer, Question, Request, State, TextPart
from idia.providers._gliner import (
    _MODEL,
    _RECORD,
    _REVISION,
    _VERSIONED,
    GlinerRuntime,
    _load_model,
    _PieceEstimator,
)
from tests._gliner_cases import (
    ESTIMATED,
    RECORDED,
    SENDER,
    STATE,
    STORED,
    TOPICS,
    load_outputs,
    record_outputs,
)

pytestmark = [
    pytest.mark.local_model,
    # Loading the checkpoint, torch warns that torch.jit.script is deprecated.
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated:FutureWarning"),
]

TOLERANCE: Final = 0.01
"""How far a span's score may move between runtime versions before the run fails."""

PROBE: Final = (
    ("the city", "Lisbon"),
    ("the country", "Norway"),
    ("the company", "Northwind Traders"),
    ("the date", "March 3, 2026"),
    ("the phone number", "+1 555 0142"),
    ("the email address", "dana@example.com"),
    ("the amount of money", "$4,200"),
    ("the fruit", "mango"),
    ("the vehicle", "bicycle"),
    ("the language", "Portuguese"),
    ("the sport", "volleyball"),
    ("the musical instrument", "cello"),
    ("the planet", "Jupiter"),
    ("the river", "Danube"),
    ("the mountain", "Kilimanjaro"),
    ("the ocean", "Pacific Ocean"),
    ("the university", "Stanford University"),
    ("the airline", "Lufthansa"),
    ("the flight number", "LH 452"),
    ("the street", "Baker Street"),
    ("the postal code", "SW1A 1AA"),
    ("the weekday", "Thursday"),
    ("the month", "November"),
    ("the percentage", "17%"),
    ("the weight", "80 kilograms"),
    ("the temperature", "21 degrees Celsius"),
    ("the book title", "Moby-Dick"),
    ("the movie", "Casablanca"),
    ("the band", "Radiohead"),
    ("the sports team", "Real Madrid"),
    ("the job title", "chief financial officer"),
    ("the department", "human resources"),
    ("the programming language", "Rust"),
    ("the operating system", "Linux"),
    ("the web browser", "Firefox"),
    ("the website", "wikipedia.org"),
    ("the drink", "espresso"),
    ("the dish", "lasagna"),
    ("the flower", "tulip"),
    ("the tree", "oak"),
    ("the gemstone", "emerald"),
    ("the metal", "copper"),
    ("the chemical element", "helium"),
    ("the disease", "influenza"),
    ("the medicine", "ibuprofen"),
    ("the body part", "elbow"),
    ("the holiday", "Thanksgiving"),
    ("the currency", "Japanese yen"),
    ("the music genre", "jazz"),
    ("the animal", "giraffe"),
    ("the color", "turquoise"),
    ("the bird", "penguin"),
    ("the insect", "butterfly"),
    ("the fabric", "linen"),
    ("the board game", "chess"),
    ("the painter", "Rembrandt"),
    ("the composer", "Mozart"),
    ("the scientist", "Marie Curie"),
    ("the ship", "Titanic"),
    ("the car brand", "Toyota"),
    ("the hotel", "Hotel Adlon"),
    ("the museum", "Louvre"),
    ("the newspaper", "The Guardian"),
    ("the dance", "tango"),
)
"""64 labels, each with a span of its own in the probe's text."""

WINDOW: Final = 2048
"""The record's ``max_input_tokens``, which the probe fills."""

_FILLER: Final = " The rest of this note repeats what was said before, with nothing new in it."


@pytest.fixture(scope="module")
def runtime() -> GlinerRuntime:
    pytest.importorskip("gliner2")
    loaded, commit = _load_model(_MODEL, _REVISION)
    assert commit == _REVISION
    return loaded


def _question(index: int, label: str) -> Question:
    return Question(key=f"q{index}", kind="extraction", instruction=label, value_type="text")


def test_the_real_model_gives_the_stored_outputs_and_records_them(runtime: GlinerRuntime) -> None:
    stored = load_outputs()
    cases: list[dict[str, Any]] = []
    for asked, expected in RECORDED.items():
        text, labels, threshold = asked
        output = runtime.extract_entities(text, labels, threshold)
        tokens = runtime.input_tokens(text, labels)
        entities = output["entities"]
        # Every label asked comes back, an empty list where nothing was found.
        assert list(entities) == list(labels)
        for label in labels:
            spans = entities[label]
            assert tuple(span["text"] for span in spans) == expected[label], label
            for span in spans:
                assert text[span["start"] : span["end"]] == span["text"], label
        if asked in stored:
            then, then_tokens = stored[asked]
            assert tokens == then_tokens
            assert entities.keys() == then["entities"].keys()
            for label, spans in entities.items():
                was = then["entities"][label]
                assert [(s["text"], s["start"], s["end"]) for s in spans] == [
                    (s["text"], s["start"], s["end"]) for s in was
                ], label
                for new, old in zip(spans, was, strict=True):
                    assert abs(new["confidence"] - old["confidence"]) <= TOLERANCE, label
        cases.append(
            {
                "text": text,
                "labels": list(labels),
                "threshold": threshold,
                "output": output,
                "input_tokens": tokens,
            }
        )
    versions = {package: importlib.metadata.version(package) for package in _VERSIONED}
    record_outputs(
        cases, STORED, {"revision": _REVISION, "python": platform.python_version(), **versions}
    )


def test_the_estimate_with_its_margin_covers_the_tokenizers_count(runtime: GlinerRuntime) -> None:
    estimator = _PieceEstimator()
    for text, labels in ESTIMATED:
        tokens = runtime.input_tokens(text, labels)
        estimate = estimator.estimate(
            State(parts=(TextPart(text=text),)),
            tuple(_question(i, label) for i, label in enumerate(labels)),
        )
        assert estimate.tokens * (1 + estimate.margin) >= tokens, (text, estimate, tokens)


def test_64_labels_within_the_window_run_in_one_pass(
    runtime: GlinerRuntime, record_property: Any
) -> None:
    labels = [label for label, _ in PROBE]
    text = " ".join(f"{label[4:].capitalize()}: {value}." for label, value in PROBE)
    assert _RECORD.max_input_tokens == WINDOW
    while runtime.input_tokens(text + _FILLER, labels) <= WINDOW:
        text += _FILLER
    tokens = runtime.input_tokens(text, labels)
    assert tokens <= WINDOW
    runtime.extract_entities(text, labels, 0.5)  # the first pass at this size is not timed
    passes = 3
    outputs: list[Mapping[str, Any]] = []
    start = time.perf_counter()
    for _ in range(passes):
        outputs.append(runtime.extract_entities(text, labels, 0.5))
    seconds = (time.perf_counter() - start) / passes
    entities = cast("Mapping[str, object]", outputs[-1]["entities"])
    assert list(entities) == labels
    assert all(isinstance(spans, list) for spans in entities.values())
    found = sum(1 for spans in entities.values() if spans)
    record_property("seconds_per_pass", seconds)
    record_property("labels_with_spans", found)
    print(f"64 labels, {tokens} tokens: {found} with spans, {seconds:.3f} s per pass")


def test_the_loaded_leaf_answers_through_a_dispatcher() -> None:
    gliner = pytest.importorskip("idia.providers.gliner")

    async def ask() -> None:
        leaf = gliner.Gliner()
        await leaf.load()
        async with Dispatcher(leaf) as provider:
            request = Request(state=STATE, questions=(SENDER, TOPICS), deadline=Deadline.after(60))
            response = await provider.answer(request)
        sender = response.answers["sender"]
        assert isinstance(sender, ExtractionAnswer)
        assert sender.value == "Dana Whitfield"
        topics = response.answers["topics"]
        assert isinstance(topics, ExtractionAnswer)
        assert topics.value == ("Q3 budget", "hiring")
        assert topics.confidence is None
        outcome = response.outcomes[0]
        assert outcome.snapshot == _REVISION
        assert outcome.request_id is None
        assert outcome.usage is not None
        assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (33, None)

    asyncio.run(ask())
